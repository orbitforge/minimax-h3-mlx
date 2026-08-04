"""Checkpoint loading for the MiniMax-H3 MLX port.

The MLX module tree reproduces the original checkpoint names exactly, so loading is a 1:1 key
match — the only tensor the checkpoint carries that the port does not hold is ``rope.inv_freq``,
which is recomputed bit-identically from the config.

MiniMax-H3 ships a **mixed-precision** transformer: the two input patch projections, the timestep
MLP and the two output heads are float32 while everything else (including the AdaLN projections)
is bfloat16. That split is preserved on load — it is not incidental. The timestep MLP feeds every
block's modulation, so rounding it biases all 50 blocks identically at every sampling step and the
error accumulates coherently along the denoising trajectory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Callable

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from .config import DiTConfig
from .checkpoint_forge.topology import BLOCK_COUNT, FORMAT_IDENTIFIER
from .dit import CACHE_ONLY_CONSTRUCTION, MiniMaxH3DiT, RESIDENT_CONSTRUCTION

# Substring matches, mirroring the reference's `_keep_in_fp32_modules`.
FP32_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)

# Carried by the checkpoint but recomputed by the port.
SKIP_KEYS = ("rope.inv_freq",)
SUPPORTED_DERIVED_SCHEMA_VERSION = 1
DERIVED_BASE_TENSOR_COUNT = 850
DERIVED_SIDECAR_COUNT = 50
DERIVED_SIDECAR_TENSOR_COUNT = 200
DERIVED_BASE_SHARD_COUNT = 5


@dataclass(frozen=True)
class CheckpointFormatInfo:
    """Validated, lightweight format routing information for a transformer checkpoint."""

    checkpoint_format: str
    derived_root: Path | None
    base_root: Path
    conversion_manifest_path: Path | None
    adaln_manifest_path: Path | None
    construction_mode: str
    base_shards: tuple[str, ...]


@dataclass(frozen=True)
class LoadStats:
    loaded_tensor_count: int
    loaded_logical_bytes: int
    expected_tensor_count: int


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validated_base_shards(weight_map: Mapping[str, object]) -> tuple[str, ...]:
    values = list(weight_map.values())
    _require(
        all(isinstance(name, str) for name in values),
        "derived base index shard names must be strings",
    )
    names = tuple(sorted(set(values)))
    _require(
        len(names) == DERIVED_BASE_SHARD_COUNT,
        f"derived base index must reference exactly {DERIVED_BASE_SHARD_COUNT} shards",
    )
    for name in names:
        _require(
            Path(name).name == name
            and name.startswith("model-")
            and name.endswith("-of-00005.safetensors"),
            f"unsafe derived base shard filename: {name!r}",
        )
    return names


def inspect_checkpoint_format(model_dir: str | Path) -> CheckpointFormatInfo:
    """Route a transformer directory using explicit derived-format metadata.

    This is a lightweight structural/runtime check. It validates manifests, indexes, filenames, and
    required files, but deliberately does not hash the 30 GB source or open any AdaLN sidecar
    payload. Complete forensic verification remains the forge verifier's separate operation.
    """
    root = Path(model_dir)
    conversion_path = root / "conversion_manifest.json"
    if not conversion_path.exists():
        return CheckpointFormatInfo(
            checkpoint_format="original",
            derived_root=None,
            base_root=root,
            conversion_manifest_path=None,
            adaln_manifest_path=None,
            construction_mode=RESIDENT_CONSTRUCTION,
            base_shards=(),
        )

    manifest = _read_json(conversion_path, "derived conversion manifest")
    _require(
        manifest.get("format_identifier") == FORMAT_IDENTIFIER,
        f"unsupported derived checkpoint format identifier: {manifest.get('format_identifier')!r}",
    )
    _require(
        manifest.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION,
        f"unsupported derived checkpoint schema version: {manifest.get('schema_version')!r}",
    )
    _require(manifest.get("bounded") is False, "bounded derived checkpoints cannot load a transformer base")
    _require(
        manifest.get("verification_status") == "verified",
        "derived checkpoint verification_status must be 'verified'",
    )
    _require(manifest.get("derived_base_tensor_count") == DERIVED_BASE_TENSOR_COUNT, "derived base tensor count is not complete")
    _require(manifest.get("total_logical_tensor_count") == 1050, "derived checkpoint tensor count is incomplete")
    _require(manifest.get("sidecar_count") == DERIVED_SIDECAR_COUNT, "derived sidecar count is incomplete")
    _require(manifest.get("sidecar_tensor_count") == DERIVED_SIDECAR_TENSOR_COUNT, "derived sidecar tensor count is incomplete")
    _require(manifest.get("selected_blocks") == list(range(BLOCK_COUNT)), "derived checkpoint does not contain all 50 blocks")

    base_root = root / "base"
    base_index_path = base_root / "model.safetensors.index.json"
    base_index = _read_json(base_index_path, "derived base index")
    weight_map = base_index.get("weight_map")
    _require(isinstance(weight_map, dict) and len(weight_map) == DERIVED_BASE_TENSOR_COUNT, "derived base index must contain exactly 850 tensors")
    base_shards = _validated_base_shards(weight_map)
    _require(not any(key.startswith("blocks.") and ".adaln_proj." in key for key in weight_map), "derived base index contains block-level AdaLN tensors")
    _require("final_layer.adaln_proj.linear.weight" in weight_map, "derived base index is missing final-layer AdaLN weight")
    _require("final_layer.adaln_proj.linear.bias" in weight_map, "derived base index is missing final-layer AdaLN bias")
    actual_base_shards = {
        path.name for path in base_root.glob("*.safetensors") if path.is_file()
    }
    missing_base_shards = sorted(set(base_shards) - actual_base_shards)
    unexpected_base_shards = sorted(actual_base_shards - set(base_shards))
    _require(
        not missing_base_shards and not unexpected_base_shards,
        "derived base payload file set mismatch: "
        f"missing={missing_base_shards}, unexpected={unexpected_base_shards}",
    )

    adaln_path = root / "adaln" / "manifest.json"
    adaln_manifest = _read_json(adaln_path, "AdaLN sidecar manifest")
    _require(adaln_manifest.get("format_identifier") == FORMAT_IDENTIFIER, "invalid AdaLN sidecar-manifest format identifier")
    _require(adaln_manifest.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION, "unsupported AdaLN sidecar-manifest schema version")
    _require(adaln_manifest.get("bounded") is False, "bounded AdaLN sidecar manifest is not loadable")
    blocks = adaln_manifest.get("blocks")
    _require(isinstance(blocks, dict) and set(blocks) == {str(i) for i in range(BLOCK_COUNT)}, "AdaLN sidecar manifest must describe all 50 blocks")
    expected_sidecars = {f"block-{index:03d}.safetensors" for index in range(BLOCK_COUNT)}
    actual_sidecars = {
        path.name for path in (root / "adaln").glob("*.safetensors") if path.is_file()
    }
    missing_sidecars = sorted(expected_sidecars - actual_sidecars)
    unexpected_sidecars = sorted(actual_sidecars - expected_sidecars)
    _require(
        not missing_sidecars and not unexpected_sidecars,
        "AdaLN sidecar payload file set mismatch: "
        f"missing={missing_sidecars}, unexpected={unexpected_sidecars}",
    )
    for block_index in range(BLOCK_COUNT):
        entry = blocks[str(block_index)]
        expected_name = f"block-{block_index:03d}.safetensors"
        _require(isinstance(entry, dict) and entry.get("sidecar_filename") == expected_name, f"invalid AdaLN sidecar entry for block {block_index}")
        _require((root / "adaln" / expected_name).is_file(), f"missing AdaLN sidecar: {expected_name}")

    return CheckpointFormatInfo(
        checkpoint_format="derived",
        derived_root=root,
        base_root=base_root,
        conversion_manifest_path=conversion_path,
        adaln_manifest_path=adaln_path,
        construction_mode=CACHE_ONLY_CONSTRUCTION,
        base_shards=base_shards,
    )


def validate_derived_base_index(info: CheckpointFormatInfo, expected_keys: set[str]) -> dict[str, str]:
    """Require an exact model-tree/index key match before any derived base payload is loaded."""
    index = _read_json(info.base_root / "model.safetensors.index.json", "derived base index")
    actual = index.get("weight_map")
    _require(isinstance(actual, dict), "derived base index has no weight_map")
    _require(all(isinstance(key, str) for key in actual), "derived base index keys must be strings")
    _require(all(isinstance(value, str) for value in actual.values()), "derived base index values must be strings")
    _require(set(_validated_base_shards(actual)) == set(info.base_shards), "derived base index shard set changed after format inspection")
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise KeyError(
            f"Derived base/module mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    _require(
        not any(key.startswith("blocks.") and ".adaln_proj." in key for key in actual_keys),
        "derived base index unexpectedly contains block-level AdaLN tensors",
    )
    _require("final_layer.adaln_proj.linear.weight" in actual_keys, "derived base is missing final-layer AdaLN weight")
    _require("final_layer.adaln_proj.linear.bias" in actual_keys, "derived base is missing final-layer AdaLN bias")
    return dict(actual)


def _is_block_adaln_key(key: str) -> bool:
    return key.startswith("blocks.") and ".adaln_proj." in key


def collect_weight_payloads(
    shard_payloads: Iterable[tuple[str, Mapping[str, mx.array]]],
    expected_keys: set[str],
    *,
    strict: bool = True,
    derived: bool = False,
    index_weight_map: Mapping[str, str] | None = None,
) -> tuple[dict[str, mx.array], int]:
    """Collect physical shard tensors and enforce the requested validation policy.

    The iterable is consumed one shard at a time, so this seam is unit-testable with tiny
    dictionaries without changing the real loader's peak memory behavior.
    """
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    duplicates: list[str] = []
    disagreements: list[str] = []
    loaded_logical_bytes = 0

    for shard_name, loaded in shard_payloads:
        for key, tensor in loaded.items():
            if key in SKIP_KEYS:
                if derived:
                    unexpected.append(key)
                continue
            indexed_shard = index_weight_map.get(key) if index_weight_map is not None else None
            if key not in expected_keys:
                unexpected.append(key)
                continue
            if index_weight_map is not None and indexed_shard != shard_name:
                disagreements.append(f"{key} (index={indexed_shard!r}, payload={shard_name!r})")
                continue
            if key in weights:
                duplicates.append(key)
                continue
            loaded_logical_bytes += tensor.nbytes
            weights[key] = tensor

    missing = sorted(expected_keys - weights.keys())
    enforce = strict or derived
    if enforce and (missing or unexpected or duplicates or disagreements):
        details = [
            f"{len(missing)} missing (e.g. {missing[:4]})",
            f"{len(unexpected)} unexpected physical (e.g. {unexpected[:4]})",
            f"{len(duplicates)} duplicate physical (e.g. {duplicates[:4]})",
            f"{len(disagreements)} payload/index disagreements (e.g. {disagreements[:4]})",
        ]
        if any(_is_block_adaln_key(key) for key in unexpected):
            details.append("block-level AdaLN tensors are not valid in the derived base")
        raise KeyError("Checkpoint/module mismatch: " + ", ".join(details))
    return weights, loaded_logical_bytes


def is_fp32_key(key: str) -> bool:
    return key.startswith(FP32_PREFIXES)


def shard_paths(model_dir: str | Path) -> list[Path]:
    """Resolve the safetensors shards of a transformer directory, in index order."""
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        names = sorted(set(weight_map.values()))
        return [model_dir / name for name in names]
    shards = sorted(model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No safetensors found in {model_dir}.")
    return shards


def load_dit(
    model_dir: str | Path,
    dtype: mx.Dtype | None = None,
    strict: bool = True,
    verbose: bool = False,
    keep_adaln: bool = False,
    telemetry: Callable[[str, MiniMaxH3DiT | None, CheckpointFormatInfo], None] | None = None,
    tensor_loader: Callable[[str], dict[str, mx.array]] | None = None,
) -> MiniMaxH3DiT:
    """Load the 33B DiT from a released ``FL2VA/transformer`` (or ``Ref2VA/transformer``) directory.

    Args:
        model_dir: the transformer directory holding ``config.json`` and the shards.
        dtype: cast every tensor to this dtype. ``None`` (default) preserves the checkpoint's
            mixed float32/bfloat16 split, which is what the reference runs.
        strict: raise if the checkpoint and the module tree disagree on any key.
        verbose: print per-shard progress.

    Returns:
        A parameter-loaded :class:`MiniMaxH3DiT`.
    """
    model_dir = Path(model_dir)
    format_info = inspect_checkpoint_format(model_dir)
    if telemetry is not None:
        telemetry("format_inspected", None, format_info)
    if format_info.checkpoint_format == "derived" and keep_adaln:
        raise ValueError(
            "--keep-adaln is not supported for derived checkpoints: block AdaLN weights live in "
            "sidecars and resident derived loading is not implemented"
        )
    config = DiTConfig.from_json(model_dir / "config.json")
    if telemetry is not None:
        telemetry("before_model_construction", None, format_info)
    model = MiniMaxH3DiT(config, construction_mode=format_info.construction_mode)
    model.checkpoint_format_info = format_info
    if telemetry is not None:
        telemetry("model_constructed", model, format_info)

    # A quantized build carries `quant_config.json`. Quantized layers hold packed weights plus
    # scales and biases, so the module tree has to be quantized *before* loading or the keys will
    # not line up — the same recipe is replayed from the file rather than guessed.
    quant_path = model_dir / "quant_config.json"
    if quant_path.exists():
        from .quantize import QuantConfig, apply_quantization_structure

        with open(quant_path) as fh:
            recipe = json.load(fh)
        apply_quantization_structure(
            model,
            QuantConfig(
                bits=recipe["bits"],
                group_size=recipe["group_size"],
                quantize_adaln=recipe.get("quantize_adaln", False),
                adaln_bits=recipe.get("adaln_bits") or 8,
            ),
        )
        if verbose:
            print(f"  quantized structure: {recipe['bits']}-bit, group {recipe['group_size']}")

    if telemetry is not None:
        telemetry("quantized_structure_ready", model, format_info)

    expected = {key for key, _ in tree_flatten(model.parameters())}
    index_weight_map = None
    if format_info.checkpoint_format == "derived":
        index_weight_map = validate_derived_base_index(format_info, expected)

    load_tensor_file = tensor_loader or mx.load
    if format_info.checkpoint_format == "derived":
        shards = [format_info.base_root / name for name in format_info.base_shards]
    else:
        shards = shard_paths(format_info.base_root)

    def payloads():
        for shard in shards:
            started = time.perf_counter()
            loaded = load_tensor_file(str(shard))
            transformed: dict[str, mx.array] = {}
            for key, tensor in loaded.items():
                if key in expected:
                    if dtype is not None:
                        # Bulk conversion of the whole 33B stack. Done on the CPU stream and materialized
                        # per tensor: casting ~130 GB through Metal is enough submissions to trip the
                        # command-buffer limits when anything else is using the device, and this path is
                        # I/O-dominated anyway.
                        with mx.stream(mx.cpu):
                            tensor = tensor.astype(dtype)
                            mx.eval(tensor)
                    elif is_fp32_key(key) and tensor.dtype != mx.float32:
                        tensor = tensor.astype(mx.float32)
                transformed[key] = tensor
            if verbose:
                gb = sum(t.nbytes for t in loaded.values()) / 1e9
                print(f"  {shard.name}: {len(loaded)} tensors, {gb:.2f} GB, "
                      f"{time.perf_counter() - started:.1f}s")
            yield shard.name, transformed

    weights, loaded_logical_bytes = collect_weight_payloads(
        payloads(),
        expected,
        strict=strict,
        derived=format_info.checkpoint_format == "derived",
        index_weight_map=index_weight_map,
    )

    if format_info.checkpoint_format == "derived" and len(weights) != DERIVED_BASE_TENSOR_COUNT:
        raise KeyError(
            f"derived base loaded {len(weights)} tensors; expected {DERIVED_BASE_TENSOR_COUNT}"
        )
    model.update(tree_unflatten(list(weights.items())))
    model.load_stats = LoadStats(
        loaded_tensor_count=len(weights),
        loaded_logical_bytes=loaded_logical_bytes,
        expected_tensor_count=len(expected),
    )
    if telemetry is not None:
        telemetry("base_weights_attached", model, format_info)
    mx.eval(model.parameters())
    if telemetry is not None:
        telemetry("base_parameters_evaluated", model, format_info)
    return model


def load_video_vae(model_dir: str | Path, strict: bool = True):
    """Load the video VAE from a released ``video_vae/`` directory.

    The weights live in ``source/model.safetensors`` under the original CompVis-style names, which
    the port reproduces. Only the convolution weights move: torch stores
    ``(C_out, C_in, kD, kH, kW)`` and MLX wants ``(C_out, kD, kH, kW, C_in)``.
    """
    from .video_vae import VideoVAE, VideoVAEConfig

    model_dir = Path(model_dir)
    config = load_video_vae_config(model_dir)
    model = VideoVAE(config)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []
    for key, tensor in mx.load(str(model_dir / "source" / "model.safetensors")).items():
        # An all-zero buffer of the masked-autoencoding objective; the decoder never reads it.
        if key == "decoder.mask_token":
            continue
        if key not in expected:
            unexpected.append(key)
            continue
        if tensor.ndim == 5:
            # Channels-last conv weights, materialized one at a time **on the CPU stream**.
            #
            # Deferring 10 GB of transposes into a single graph overruns the Metal command-buffer
            # deadline outright. Doing them individually on the GPU is enough on an idle machine but
            # still fails when something else is competing for the device — which is exactly when a
            # user is most likely to be loading a model. The CPU stream has no such deadline. It
            # trades some load time for not failing — a one-time cost on a path that otherwise
            # aborts a multi-hour run at the last component.
            with mx.stream(mx.cpu):
                tensor = mx.contiguous(tensor.transpose(0, 2, 3, 4, 1))
                mx.eval(tensor)
        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Video VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def load_video_vae_config(model_dir: str | Path):
    """Read video-VAE geometry and normalization metadata without loading weights."""
    from .video_vae import VideoVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)
    with open(model_dir / "source" / "config.json") as fh:
        source = json.load(fh)

    ch = source["ch"]
    return VideoVAEConfig(
        in_channels=source["in_channels"],
        out_channels=source["out_ch"],
        latent_channels=source["z_channels"],
        block_out_channels=tuple(ch * m for m in source["ch_mult"]),
        layers_per_block=source["num_res_blocks"],
        spatial_downsample_factors=tuple(source["space_down"]),
        temporal_downsample_factors=tuple(source["time_down"]),
        decoder_num_layers=source["vit_decoder_kwargs"]["num_layers"],
        decoder_num_attention_heads=source["vit_decoder_kwargs"]["heads"],
        decoder_attention_head_dim=source["vit_decoder_kwargs"]["dim_head"],
        decoder_rope_theta=source["vit_decoder_kwargs"]["rope_theta"],
        decoder_rope_dim_ratio=source["vit_decoder_kwargs"]["rope_dim_ratio"],
        clip_length=wrapper.get("vae_clip_length", 17),
        token_drop=wrapper.get("vae_token_drop", 3),
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def load_audio_vae(model_dir: str | Path, strict: bool = True):
    """Load the audio VAE from a released ``audio_vae/`` directory.

    Weight norm is **folded** here: the checkpoint stores ``weight_g`` / ``weight_v`` and the
    effective weight is ``g * v / ||v||`` with the norm taken over every axis but the first. Folding
    once at load is exactly equivalent to recomputing it on every forward, and it lets the port hold
    a plain weight (proven equivalent by the parity test, which reconstructs the pair).
    """
    from .audio_vae import AudioVAE, AudioVAEConfig

    model_dir = Path(model_dir)
    config = load_audio_vae_config(model_dir)
    model = AudioVAE(config)
    expected = {key for key, _ in tree_flatten(model.parameters())}

    raw = dict(mx.load(str(model_dir / "model.safetensors")))
    weights: dict[str, mx.array] = {}
    unexpected: list[str] = []

    for key, tensor in raw.items():
        if key.endswith(".filter"):
            continue  # recomputed by kaiser_sinc_filter1d
        if key.endswith(".weight_v"):
            base = key[: -len("_v")]
            g = raw[f"{base}_g"]
            v = tensor
            norm = mx.sqrt(mx.sum(mx.square(v.reshape(v.shape[0], -1)), axis=1)).reshape(-1, 1, 1)
            tensor = g * v / norm
            key = base
        elif key.endswith(".weight_g"):
            continue

        if key not in expected:
            unexpected.append(key)
            continue

        if key.endswith(".weight") and tensor.ndim == 3:
            # Transposed convs are stored (C_in, C_out, kL); plain convs (C_out, C_in, kL).
            tensor = tensor.transpose(1, 2, 0) if ".ups." in key else tensor.transpose(0, 2, 1)
        elif key.endswith(".alpha") and tensor.ndim == 3:
            tensor = tensor.transpose(0, 2, 1)  # (1, C, 1) -> (1, 1, C)

        weights[key] = tensor

    missing = sorted(expected - weights.keys())
    if strict and (missing or unexpected):
        raise KeyError(
            f"Audio VAE mismatch: {len(missing)} missing (e.g. {missing[:4]}), "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:4]})."
        )
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model


def load_audio_vae_config(model_dir: str | Path):
    """Read audio-VAE geometry and normalization metadata without loading weights."""
    from .audio_vae import AudioVAEConfig

    model_dir = Path(model_dir)
    with open(model_dir / "metadata.json") as fh:
        kwargs = json.load(fh)["metadata"]["kwargs"]
    with open(model_dir / "config.json") as fh:
        wrapper = json.load(fh)

    return AudioVAEConfig(
        encoder_dim=kwargs["encoder_dim"],
        encoder_rates=tuple(kwargs["encoder_rates"]),
        latent_dim=kwargs["latent_dim"],
        latent_channels=kwargs["vae_latent_channels"],
        decoder_dim=kwargs["decoder_dim"],
        decoder_rates=tuple(kwargs["decoder_rates"]),
        sampling_rate=kwargs["sample_rate"],
        latents_mean=tuple(wrapper.get("latents_mean", ())),
        latents_std=tuple(wrapper.get("latents_std", ())),
    )


def parameter_summary(model: MiniMaxH3DiT) -> dict[str, object]:
    """Parameter counts and footprint, split by the AdaLN projections that can be dropped."""
    total = adaln = 0
    nbytes = adaln_bytes = 0
    for key, value in tree_flatten(model.parameters()):
        total += value.size
        nbytes += value.nbytes
        if ".adaln_proj." in key and key.startswith("blocks."):
            adaln += value.size
            adaln_bytes += value.nbytes
    return {
        "total_params": total,
        "adaln_params": adaln,
        "core_params": total - adaln,
        "total_gb": nbytes / 1e9,
        "adaln_gb": adaln_bytes / 1e9,
        "core_gb": (nbytes - adaln_bytes) / 1e9,
    }

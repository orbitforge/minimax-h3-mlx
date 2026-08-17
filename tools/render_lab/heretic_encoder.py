"""Host-only Heretic state-28 -> H3 conditioning encoder.

This file is intentionally a child-process boundary.  The Render Lab starts
it, waits for its artifact and release evidence, and only then starts the H3
generator in a new process.  MLX and the Heretic model are imported lazily so
the browser/controller remains MLX-free and fail-closed asset checks happen
before any expensive model load.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from minimax_h3_mlx.conditioning_artifact import (  # noqa: E402
    CONDITIONING_DTYPE,
    CONDITIONING_WIDTH,
    bfloat16_bits_from_float32,
    build_encoder_provenance,
    create_conditioning_artifact_from_bits,
    validate_conditioning_artifact,
)
from tools.render_lab.encoder_catalog import (  # noqa: E402
    HERETIC_BRIDGE_KEYS,
    HERETIC_BRIDGE_SHA256,
    HERETIC_ENCODER_ID,
    HERETIC_FULL_LAYERS,
    HERETIC_SOURCE_WIDTH,
    HERETIC_STATE,
    HERETIC_TARGET_WIDTH,
    probe_heretic_assets,
)


class HereticEncodeError(RuntimeError):
    """A host-only experimental conditioning encode failed closed."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _token_ids_and_pieces(tokenizer: Any, prompt: str) -> tuple[list[int], list[str]]:
    encoded = tokenizer(prompt, add_special_tokens=False)
    raw_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    while isinstance(raw_ids, (list, tuple)) and raw_ids and isinstance(raw_ids[0], (list, tuple)):
        raw_ids = raw_ids[0]
    ids = [int(value) for value in raw_ids]
    pieces = [str(value) for value in tokenizer.convert_ids_to_tokens(ids)]
    if len(ids) != len(pieces):
        raise HereticEncodeError("Tokenizer returned a token/piece count mismatch")
    return ids, pieces


def require_exact_token_piece_alignment(
    canonical_ids: Sequence[int],
    canonical_pieces: Sequence[str],
    heretic_ids: Sequence[int],
    heretic_pieces: Sequence[str],
) -> None:
    """Fail closed unless both tokenizers produce identical token pieces."""
    if tuple(canonical_pieces) != tuple(heretic_pieces):
        raise HereticEncodeError(
            "Canonical Qwen3-VL and Heretic token-piece alignment mismatch; refusing experimental encode"
        )
    if len(canonical_ids) != len(canonical_pieces) or len(heretic_ids) != len(heretic_pieces):
        raise HereticEncodeError("Tokenizer token IDs and pieces are not aligned")


def _load_tokenizer(path: Path) -> Any:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise HereticEncodeError(f"Could not load local tokenizer at {path}: {exc}") from exc


def _load_bridge(path: Path) -> dict[str, np.ndarray]:
    if _sha256_file(path) != HERETIC_BRIDGE_SHA256:
        raise HereticEncodeError(
            f"Heretic state-28 bridge SHA-256 mismatch; expected {HERETIC_BRIDGE_SHA256}"
        )
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if frozenset(loaded.files) != HERETIC_BRIDGE_KEYS:
                raise HereticEncodeError("Heretic bridge key set mismatch")
            bridge = {name: np.array(loaded[name], dtype=np.float32, copy=True) for name in loaded.files}
    except HereticEncodeError:
        raise
    except Exception as exc:
        raise HereticEncodeError(f"Could not load Heretic bridge {path}: {exc}") from exc
    expected_shapes = {
        "input_mean": (HERETIC_SOURCE_WIDTH,),
        "input_scale": (HERETIC_SOURCE_WIDTH,),
        "target_mean": (HERETIC_TARGET_WIDTH,),
        "weights": (HERETIC_SOURCE_WIDTH, HERETIC_TARGET_WIDTH),
    }
    for name, shape in expected_shapes.items():
        if bridge[name].shape != shape or not np.all(np.isfinite(bridge[name])):
            raise HereticEncodeError(f"Heretic bridge {name} has invalid shape or non-finite values")
    if np.any(bridge["input_scale"] == 0):
        raise HereticEncodeError("Heretic bridge input_scale contains zero")
    return bridge


def _manual_state28_forward(model: Any, input_ids: Any, mx: Any) -> Any:
    """Run exactly layers 0..27 and return the pre-final-norm state 28."""
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask

    core = model.language_model.model
    layers = core.layers
    if len(layers) != HERETIC_FULL_LAYERS:
        raise HereticEncodeError(f"Heretic decoder layer count changed: {len(layers)}")
    hidden_states = core.embed_tokens(input_ids)
    prefix_layers = layers[:HERETIC_STATE]
    cache = [None] * HERETIC_STATE
    fa_mask = create_attention_mask(hidden_states, cache[core.fa_idx])
    ssm_mask = create_ssm_mask(hidden_states, cache[core.ssm_idx])
    for layer, layer_cache in zip(prefix_layers, cache):
        mask = ssm_mask if layer.is_linear else fa_mask
        hidden_states = layer(hidden_states, mask=mask, cache=layer_cache)
    return hidden_states


def _timed_state28_forward(
    model: Any,
    input_ids: Any,
    mx: Any,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> tuple[Any, float]:
    """Return state 28 and time through its materialized MLX evaluation."""
    state28_started = clock()
    source_state = _manual_state28_forward(model, input_ids, mx)
    mx.eval(source_state)
    return source_state, clock() - state28_started


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for key, name in (
        ("active_bytes", "get_active_memory"),
        ("peak_bytes", "get_peak_memory"),
        ("cache_bytes", "get_cache_memory"),
    ):
        function = getattr(mx, name, None)
        try:
            value = function() if callable(function) else None
            result[key] = int(value) if value is not None else None
        except Exception:
            result[key] = None
    return result


def _release_mlx(mx: Any) -> dict[str, Any]:
    gc.collect()
    try:
        mx.synchronize()
    except Exception:
        pass
    try:
        mx.clear_cache()
    except Exception:
        pass
    gc.collect()
    after = _memory_snapshot(mx)
    active = after.get("active_bytes")
    cache = after.get("cache_bytes")
    return {
        "active_bytes_after_release": active,
        "cache_bytes_after_release": cache,
        "clean": active is not None and cache is not None and active <= 1024 and cache == 0,
        "status": "released" if active is not None and cache is not None and active <= 1024 and cache == 0 else "release-gate-failed",
    }


def _provenance(
    *,
    checkpoint_root: Path,
    model_path: Path,
    bridge_path: Path,
    bridge: dict[str, np.ndarray],
    canonical_ids: Sequence[int],
    heretic_ids: Sequence[int],
    pieces: Sequence[str],
    model_config_sha256: str | None,
) -> dict[str, Any]:
    return {
        "encoder_id": HERETIC_ENCODER_ID,
        "experimental": True,
        "family": "qwen3_5_moe",
        "config": {
            "sha256": model_config_sha256,
            "model_type": "qwen3_5_moe",
            "hidden_size": HERETIC_SOURCE_WIDTH,
            "full_decoder_layers": HERETIC_FULL_LAYERS,
        },
        "selected_state": {
            "hidden_state": f"hidden_states[{HERETIC_STATE}]",
            "normalization": "unnormalized-pre-final-norm",
            "selected_decoder_layer": HERETIC_STATE,
            "logical_dtype": CONDITIONING_DTYPE,
        },
        "source_model": {
            "path": str(model_path),
            "config_sha256": model_config_sha256,
            "maximum_executed_state": HERETIC_STATE,
            "layers_29_through_40_executed": False,
        },
        "weights": {
            "source": "heretic-source-model",
            "manifest": {
                "present": True,
                "config_sha256": model_config_sha256,
                "model_index_sha256": _sha256_file(model_path / "model.safetensors.index.json")
                if (model_path / "model.safetensors.index.json").is_file()
                else None,
            },
        },
        "tokenizer": {
            "source": "heretic-source-model",
            "manifest": {"present": (model_path / "tokenizer.json").is_file()},
        },
        "processor": {
            "source": "not-used-for-text-only-encode",
            "manifest": {"present": False},
        },
        "canonical_h3_encoder": build_encoder_provenance(checkpoint_root, selected_layer=50),
        "bridge": {
            "path": str(bridge_path),
            "sha256": _sha256_file(bridge_path),
            "input_width": HERETIC_SOURCE_WIDTH,
            "target_width": HERETIC_TARGET_WIDTH,
            "keys": sorted(bridge),
            "shapes": {name: [int(item) for item in value.shape] for name, value in bridge.items()},
            "operation": "(state28 - input_mean) / input_scale @ weights + target_mean",
        },
        "token_alignment": {
            "exact_token_piece_alignment": True,
            "canonical_h3_token_ids": [int(value) for value in canonical_ids],
            "heretic_token_ids": [int(value) for value in heretic_ids],
            "token_pieces": [str(value) for value in pieces],
        },
    }


def encode(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_root = Path(args.checkpoint).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    bridge_path = Path(args.bridge).expanduser().resolve()
    assets = probe_heretic_assets(REPO_ROOT)
    if not assets.available or assets.model_path != model_path or assets.bridge_path != bridge_path:
        raise HereticEncodeError(assets.reason or "Heretic assets failed admission")
    bridge = _load_bridge(bridge_path)

    started = time.perf_counter()
    canonical_tokenizer_path = checkpoint_root / "tokenizer"
    if not canonical_tokenizer_path.is_dir():
        canonical_tokenizer_path = checkpoint_root / "text_encoder"
    canonical_tokenizer = _load_tokenizer(canonical_tokenizer_path)
    heretic_tokenizer = _load_tokenizer(model_path)
    canonical_ids, canonical_pieces = _token_ids_and_pieces(canonical_tokenizer, args.prompt)
    heretic_ids, heretic_pieces = _token_ids_and_pieces(heretic_tokenizer, args.prompt)
    require_exact_token_piece_alignment(canonical_ids, canonical_pieces, heretic_ids, heretic_pieces)

    try:
        import mlx.core as mx
        from mlx_lm import load

        model_load_started = time.perf_counter()
        model, _mlx_tokenizer, _config = load(str(model_path), lazy=True, return_config=True)
        heretic_load_seconds = time.perf_counter() - model_load_started
        input_ids = mx.array([heretic_ids], dtype=mx.int32)
        source_state, state28_forward_seconds = _timed_state28_forward(model, input_ids, mx)
        if tuple(source_state.shape) != (1, len(heretic_ids), HERETIC_SOURCE_WIDTH):
            raise HereticEncodeError(f"Heretic state-28 shape mismatch: {source_state.shape}")
        source_state = source_state.astype(mx.float32)
        input_mean = mx.array(bridge["input_mean"], dtype=mx.float32)
        input_scale = mx.array(bridge["input_scale"], dtype=mx.float32)
        weights = mx.array(bridge["weights"], dtype=mx.float32)
        target_mean = mx.array(bridge["target_mean"], dtype=mx.float32)
        projected = ((source_state - input_mean) / input_scale) @ weights + target_mean
        conditioning = projected.astype(mx.bfloat16)
        mx.eval(conditioning)
        conditioning_float32 = np.array(conditioning.astype(mx.float32), dtype=np.float32, copy=True)
        conditioning_bits = bfloat16_bits_from_float32(conditioning_float32)
        encode_seconds = time.perf_counter() - started
        peak_memory = _memory_snapshot(mx)
        provenance = _provenance(
            checkpoint_root=checkpoint_root,
            model_path=model_path,
            bridge_path=bridge_path,
            bridge=bridge,
            canonical_ids=canonical_ids,
            heretic_ids=heretic_ids,
            pieces=canonical_pieces,
            model_config_sha256=assets.model_config_sha256,
        )
        artifact = create_conditioning_artifact_from_bits(
            args.artifact,
            prompt=args.prompt,
            conditioning_bits=conditioning_bits,
            text_token_tags=np.ones((len(canonical_ids),), dtype=np.int32),
            token_ids=np.asarray([canonical_ids], dtype=np.int32),
            encoder_provenance=provenance,
        )
        validate_conditioning_artifact(artifact, checkpoint_root=checkpoint_root, prompt=args.prompt)
        model = None
        _mlx_tokenizer = None
        source_state = None
        input_ids = None
        input_mean = None
        input_scale = None
        weights = None
        target_mean = None
        projected = None
        conditioning = None
        canonical_tokenizer = None
        heretic_tokenizer = None
        release = _release_mlx(mx)
        evidence = {
            "schema_version": 1,
            "encoder_id": HERETIC_ENCODER_ID,
            "experimental": True,
            "prompt": args.prompt,
            "source_model": {
                "path": str(model_path),
                "config_sha256": assets.model_config_sha256,
                "model_index_sha256": _sha256_file(model_path / "model.safetensors.index.json")
                if (model_path / "model.safetensors.index.json").is_file()
                else None,
                "source_state": HERETIC_STATE,
                "source_width": HERETIC_SOURCE_WIDTH,
                "full_decoder_layers": HERETIC_FULL_LAYERS,
                "maximum_executed_state": HERETIC_STATE,
                "layers_29_through_40_executed": False,
            },
            "bridge": {
                "path": str(bridge_path),
                "sha256": HERETIC_BRIDGE_SHA256,
                "input_width": HERETIC_SOURCE_WIDTH,
                "target_width": HERETIC_TARGET_WIDTH,
                "keys": sorted(bridge),
                "shapes": {name: [int(item) for item in value.shape] for name, value in bridge.items()},
            },
            "token_alignment": provenance["token_alignment"],
            "canonical_h3_token_count": len(canonical_ids),
            "heretic_token_count": len(heretic_ids),
            "conditioning_artifact": {
                "path": str(artifact.path),
                "identity": artifact.artifact_identity,
                "tensor_checksum": artifact.tensor_checksum,
                "shape": list(artifact.conditioning_shape),
                "dtype": CONDITIONING_DTYPE,
            },
            "encode_seconds": encode_seconds,
            "heretic_load_seconds": heretic_load_seconds,
            "state28_forward_seconds": state28_forward_seconds,
            "source_shape": [1, len(heretic_ids), HERETIC_SOURCE_WIDTH],
            "target_shape": [1, len(canonical_ids), HERETIC_TARGET_WIDTH],
            "peak_memory": peak_memory,
            "peak_memory_bytes": peak_memory.get("peak_bytes"),
            "release": release,
            "active_memory_after_release_bytes": release.get("active_bytes_after_release"),
            "cache_memory_after_release_bytes": release.get("cache_bytes_after_release"),
            "canonical_teacher_qwen_loaded": False,
            "h3_transformer_loaded": False,
            "h3_launched_before_encoder_exit": False,
            "release_gate": bool(release["clean"]),
            "status": "complete" if release["clean"] else "release-gate-failed",
        }
        _write_json(Path(args.evidence), evidence)
        _write_json(Path(args.release_evidence), release)
        if not release["clean"]:
            raise HereticEncodeError("Heretic release gate failed; H3 must not be launched")
        return evidence
    except Exception:
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--release-evidence", required=True)
    args = parser.parse_args(argv)
    try:
        encode(args)
    except Exception as exc:
        print(f"Heretic conditioning encode failed: {exc}", file=sys.stderr)
        return 2
    print("Heretic state28 conditioning encode complete; clean process exit gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

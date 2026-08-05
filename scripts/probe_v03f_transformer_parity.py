"""v0.3f full resident-versus-derived transformer numerical parity probe."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ORIGINAL = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit"
DEFAULT_DERIVED = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_ARTIFACT = ROOT / "out" / "v0.3f" / "transformer-reference.safetensors"
DEFAULT_REPORT = ROOT / "out" / "v0.3f" / "transformer-parity-report.json"

ARTIFACT_FORMAT = "minimax-h3-mlx-v03f-transformer-reference"
ARTIFACT_SCHEMA_VERSION = 2
CANONICAL_TIMESTEP_DTYPE = "float32"
REFERENCE_CHECKPOINT_FINGERPRINT_METHOD = (
    "sha256(index-json)+sha256(full-content-of-every-indexed-resident-shard)-v1"
)
DETERMINISTIC_INPUT_METHOD = "tensor-index-pattern-v1"
TIMESTEP_VALUE = 0.5
RELATIVE_DENOMINATOR_FLOOR = 1.0e-6
ALLCLOSE_ATOL = 1.0e-5
ALLCLOSE_RTOL = 1.0e-5
EXPECTED_BLOCK_COUNT = 50
ARTIFACT_KEYS = (
    "video_input",
    "audio_input",
    "text_input",
    "timestep",
    "timestep_indices",
    "token_tags",
    "position_ids",
    "video_indices",
    "audio_indices",
    "text_indices",
    "resident_video_output",
    "resident_audio_output",
)
REFERENCE_METADATA_REQUIRED_KEYS = frozenset({
    "artifact_format", "artifact_schema_version", "artifact_file_format",
    "reference_checkpoint", "derived_checkpoint", "reference_checkpoint_fingerprint",
    "reference_checkpoint_fingerprint_method", "reference_config_sha256",
    "derived_config_sha256", "derived_conversion_manifest_sha256",
    "derived_sidecar_manifest_sha256", "artifact_sha256", "deterministic_input_method",
    "timestep_values", "timestep_dtype", "configured_resident_block_count",
    "observed_resident_block_count", "observed_resident_block_indices", "tensor_keys",
    "tensor_inventory", "packed_layout",
})

DETERMINISTIC_INPUT_SPEC = {
    "modulus": 23,
    "offset": 1.0,
    "scale_base": 0.001,
    "scale_step": 0.0001,
    "allowed_salts": (0, 1, 2),
}


def artifact_metadata_path(artifact: Path) -> Path:
    if artifact.suffix != ".safetensors":
        raise ValueError(f"v0.3f reference artifact must use .safetensors: {artifact}")
    return artifact.with_suffix(".json")


def existing_artifact_paths(*paths: Path | str) -> list[str]:
    """Return only paths that already exist, without changing the filesystem."""
    return [str(path) for path in paths if Path(path).is_file()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_input_specification() -> dict[str, Any]:
    return {
        "method": DETERMINISTIC_INPUT_METHOD,
        "modulus": DETERMINISTIC_INPUT_SPEC["modulus"],
        "offset": DETERMINISTIC_INPUT_SPEC["offset"],
        "scale_base": DETERMINISTIC_INPUT_SPEC["scale_base"],
        "scale_step": DETERMINISTIC_INPUT_SPEC["scale_step"],
        "allowed_salts": list(DETERMINISTIC_INPUT_SPEC["allowed_salts"]),
    }


def deterministic_input_pattern_parameters(salt: int) -> tuple[int, float, float]:
    if salt not in DETERMINISTIC_INPUT_SPEC["allowed_salts"]:
        raise ValueError(f"deterministic input salt must be one of {DETERMINISTIC_INPUT_SPEC['allowed_salts']}")
    return (
        DETERMINISTIC_INPUT_SPEC["modulus"],
        DETERMINISTIC_INPUT_SPEC["offset"],
        DETERMINISTIC_INPUT_SPEC["scale_base"] + salt * DETERMINISTIC_INPUT_SPEC["scale_step"],
    )


def normalize_dtype_name(dtype: Any) -> str:
    name = str(dtype)
    prefix = "mlx.core."
    return name[len(prefix):] if name.startswith(prefix) else name


def validate_canonical_timestep_dtype(dtype: Any) -> None:
    normalized = normalize_dtype_name(dtype)
    if normalized != CANONICAL_TIMESTEP_DTYPE:
        raise ValueError(f"minimal packed timestep dtype is not canonical: {dtype}")


def deterministic_input_values(element_count: int, salt: int) -> tuple[float, ...]:
    if not isinstance(element_count, int) or isinstance(element_count, bool) or element_count <= 0:
        raise ValueError("deterministic input element count must be a strictly positive integer")
    modulus, offset, scale = deterministic_input_pattern_parameters(salt)
    return tuple((((index + salt + 1) % modulus) + offset) * scale for index in range(element_count))


def validate_resident_shard_names(weight_map: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("resident checkpoint index has no weight_map")
    values = list(weight_map.values())
    if not all(isinstance(name, str) for name in values):
        raise ValueError("resident checkpoint index weight_map shard names must all be strings")
    names = sorted(set(values))
    if not all(name and name not in {".", ".."} and Path(name).name == name for name in names):
        raise ValueError("resident checkpoint index contains unsafe shard names; each must be a non-empty safe basename")
    return tuple(names)


def validate_artifact_tensor_keys(keys: Any) -> None:
    if not isinstance(keys, list):
        raise ValueError("reference tensor keys must be a list")
    if keys != list(ARTIFACT_KEYS):
        raise ValueError(f"reference tensor key order or membership mismatch: got {keys}, expected {list(ARTIFACT_KEYS)}")


def validate_reference_metadata(
    metadata: Mapping[str, Any], *, original: Path, derived: Path,
    expected_checksums: Mapping[str, str], expected_tensor_inventory: Mapping[str, Any],
    expected_packed_layout: Mapping[str, Any], expected_observed_block_indices: list[int] | tuple[int, ...],
    expected_configured_block_count: int = EXPECTED_BLOCK_COUNT,
    expected_timestep_values: list[float] | tuple[float, ...] = (TIMESTEP_VALUE,),
    expected_timestep_dtype: str = CANONICAL_TIMESTEP_DTYPE,
) -> None:
    missing = sorted(set(REFERENCE_METADATA_REQUIRED_KEYS) - set(metadata))
    unexpected = sorted(set(metadata) - set(REFERENCE_METADATA_REQUIRED_KEYS))
    if missing or unexpected:
        raise ValueError(f"reference metadata key contract mismatch: missing={missing}, unexpected={unexpected}")
    expected_scalars = {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors",
        "reference_checkpoint": str(original.resolve()),
        "derived_checkpoint": str(derived.resolve()),
        "reference_checkpoint_fingerprint_method": REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
        "deterministic_input_method": DETERMINISTIC_INPUT_METHOD,
        "timestep_values": list(expected_timestep_values),
        "timestep_dtype": expected_timestep_dtype,
        "configured_resident_block_count": expected_configured_block_count,
        "observed_resident_block_count": len(expected_observed_block_indices),
        "observed_resident_block_indices": list(expected_observed_block_indices),
        "packed_layout": dict(expected_packed_layout),
    }
    for key, expected in expected_scalars.items():
        if metadata[key] != expected:
            raise ValueError(f"reference metadata mismatch for {key}: got {metadata[key]!r}, expected {expected!r}")
    for key, expected in expected_checksums.items():
        if metadata.get(key) != expected:
            raise ValueError(f"reference metadata checksum mismatch for {key}")
    validate_artifact_tensor_keys(metadata["tensor_keys"])
    inventory = metadata["tensor_inventory"]
    if not isinstance(inventory, dict):
        raise ValueError("reference tensor inventory must be a dictionary")
    missing_inventory_keys = sorted(set(ARTIFACT_KEYS) - set(inventory))
    unexpected_inventory_keys = sorted(set(inventory) - set(ARTIFACT_KEYS))
    if missing_inventory_keys or unexpected_inventory_keys:
        raise ValueError(
            "reference tensor inventory key membership mismatch: "
            f"missing={missing_inventory_keys}, unexpected={unexpected_inventory_keys}"
        )
    for key in ARTIFACT_KEYS:
        if inventory[key] != expected_tensor_inventory[key]:
            raise ValueError(f"reference tensor inventory does not match the serialized tensors for {key}")


def _resident_shard_paths(checkpoint: Path) -> tuple[Path, ...]:
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"resident checkpoint index is missing: {index_path}")
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resident checkpoint index is not valid JSON: {index_path}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    try:
        names = validate_resident_shard_names(weight_map)
    except ValueError as exc:
        raise ValueError(f"{exc}: {index_path}") from exc
    paths = tuple(checkpoint / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"resident checkpoint index references missing shards: {missing}")
    return paths


def resident_checkpoint_fingerprint(checkpoint: Path) -> str:
    checkpoint = checkpoint.resolve()
    index_path = checkpoint / "model.safetensors.index.json"
    payload = {
        "method": REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
        "index_sha256": sha256_file(index_path),
        "shards": [{"filename": p.name, "sha256": sha256_file(p)}
                   for p in _resident_shard_paths(checkpoint)],
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


class _ObservedTransformerBlock:
    def __init__(self, block_index: int, block: Any, observed: list[int]) -> None:
        self.block_index, self.block, self.observed = block_index, block, observed

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.observed.append(self.block_index)
        return self.block(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.block, name)


@contextmanager
def observe_transformer_block_execution(dit):
    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, list):
        raise TypeError("transformer block observation requires a mutable list of blocks")
    original = list(blocks)
    observed: list[int] = []
    blocks[:] = [_ObservedTransformerBlock(i, block, observed) for i, block in enumerate(original)]
    try:
        yield observed
    finally:
        blocks[:] = original


def validate_observed_block_indices(configured_count: int, observed: list[int] | tuple[int, ...]) -> None:
    expected, actual = list(range(configured_count)), list(observed)
    if actual != expected:
        raise ValueError(
            "transformer block execution observation mismatch: "
            f"configured={configured_count}, observed_indices={actual}, expected_indices={expected}"
        )


def build_execution_receipt(configured_count: int, observed: list[int] | tuple[int, ...]) -> dict[str, Any]:
    validate_observed_block_indices(configured_count, observed)
    return {
        "configured_transformer_block_count": configured_count,
        "observed_transformer_block_indices": list(observed),
        "observed_transformer_block_count": len(observed),
    }


def validate_combined_exact_parity(video_exact: bool, audio_exact: bool) -> None:
    if not video_exact or not audio_exact:
        raise ValueError(
            "full transformer parity failed: "
            f"video_exact={video_exact}, audio_exact={audio_exact}"
        )


def clear_derived_comparison_arrays(resident_combined: Any, derived_combined: Any) -> tuple[None, None]:
    """Drop both concatenated MLX comparison arrays before allocator cleanup."""
    resident_combined = None
    derived_combined = None
    return resident_combined, derived_combined


def detach_exception(exc: BaseException) -> BaseException:
    """Remove model-work traceback and exception chaining before resource cleanup."""
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None
    exc.__suppress_context__ = True
    return exc


def write_diagnostic_report(report_path: Path, report: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def validate_parity_after_report(report_path: Path, video_exact: bool, audio_exact: bool) -> None:
    if not report_path.is_file():
        raise ValueError(f"diagnostic parity report must exist before parity validation: {report_path}")
    validate_combined_exact_parity(video_exact, audio_exact)


def emit_parity_success_message(parity_validated: bool) -> None:
    if not parity_validated:
        raise ValueError("cannot emit transformer parity success before parity validation")
    print("TRANSFORMER PARITY PASSED", flush=True)


def validate_complete_cache_stats(*, cache_table_count: int, configured_block_count: int,
                                 stats: Any, actual_sidecar_names: list[str] | tuple[str, ...]) -> None:
    expected = [f"block-{i:03d}.safetensors" for i in range(configured_block_count)]
    for label, actual in (
        ("cache.tables length", cache_table_count),
        ("stats.blocks_completed", stats.blocks_completed),
        ("stats.sidecar_files_opened", stats.sidecar_files_opened),
        ("stats.unique_sidecar_files_opened", stats.unique_sidecar_files_opened),
        ("stats.successful_payload_opens", stats.successful_payload_opens),
        ("stats.completed_payload_releases", stats.completed_payload_releases),
    ):
        if actual != configured_block_count:
            raise ValueError(f"complete cache contract violation: {label}={actual}, expected {configured_block_count}")
    if list(actual_sidecar_names) != expected:
        raise ValueError(f"complete cache contract violation: sidecar filename order is {list(actual_sidecar_names)}, expected {expected}")
    for label, actual, wanted in (
        ("stats.every_sidecar_released_before_next_opened", stats.every_sidecar_released_before_next_opened, True),
        ("stats.sidecar_overlap_observed", stats.sidecar_overlap_observed, False),
        ("stats.next_sidecar_opened_before_previous_release", stats.next_sidecar_opened_before_previous_release, False),
        ("stats.dense_temporary_projection_created", stats.dense_temporary_projection_created, False),
    ):
        if actual is not wanted:
            raise ValueError(f"complete cache contract violation: {label}={actual!r}, expected {wanted!r}")


def snapshot(mx) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, name in (("active_memory", "get_active_memory"),
                        ("allocator_cache", "get_cache_memory"),
                        ("peak_memory", "get_peak_memory")):
        getter = getattr(mx, name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except Exception:
            result[label] = None
    return result


def begin_phase(mx) -> tuple[float, dict[str, int | None]]:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()
    return time.perf_counter(), snapshot(mx)


def emit_phase(phase: str, started: float, before: dict[str, int | None],
               after: dict[str, int | None], *, inputs=None, outputs=None,
               artifacts=None, checkpoint_open_count=None, sidecar_open_count=None,
               block_execution=None, release_result=None) -> None:
    receipt = {
        "phase": phase, "wall_clock_seconds": round(time.perf_counter() - started, 6),
        "memory_before": before, "memory_after": after, "inputs": inputs or {},
        "outputs": outputs or {}, "artifacts": artifacts or [],
        "checkpoint_open_count": checkpoint_open_count, "sidecar_open_count": sidecar_open_count,
        "block_execution": block_execution or {}, "release_result": release_result,
    }
    print(f"phase_receipt={json.dumps(receipt, sort_keys=True)}", flush=True)


def _shape_dtype(array) -> dict[str, Any]:
    return {"shape": [int(v) for v in array.shape], "dtype": str(array.dtype)}


def _require_finite(mx, array, label: str) -> None:
    value = mx.all(mx.isfinite(array))
    mx.eval(value)
    if not bool(value.item()):
        raise ValueError(f"{label} contains non-finite values")


def _mx_bool(mx, value) -> bool:
    mx.eval(value)
    return bool(value.item())


def _exact_array_equal(mx, left, right) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and _mx_bool(mx, mx.all(left == right))


def _release(mx) -> dict[str, Any]:
    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    available = callable(clear)
    if available:
        clear()
    gc.collect()
    return {
        "memory": snapshot(mx),
        "allocator_cache_purge_available": available,
        "allocator_cache_purged": available,
    }


def _load_telemetry(mx, started: float, opened: list[str]):
    def record(stage: str, _model, _info) -> None:
        print("checkpoint_telemetry=" + json.dumps({
            "stage": stage,
            "wall_clock_seconds": round(time.perf_counter() - started, 6),
            "memory": snapshot(mx), "checkpoint_open_count": len(opened),
        }, sort_keys=True), flush=True)
    def load(path: str):
        opened.append(path)
        return mx.load(path)
    return record, load


def _sidecar_telemetry(mx, event: str, details: Mapping[str, Any]) -> None:
    if event in {"shared_timestep_embedding_materialized", "sidecar_opening",
                 "sidecar_released", "cache_completed"}:
        print("sidecar_telemetry=" + json.dumps({
            "event": event,
            "details": {k: str(v) if k == "stats" else v for k, v in details.items()},
            "memory": details.get("memory", snapshot(mx)),
        }, default=str, sort_keys=True), flush=True)


def _build_layout_and_inputs(mx, dit):
    import numpy as np
    from minimax_h3_mlx.config import TAG_TEXT
    from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps
    layout = build_packed_sequence(np.array([TAG_TEXT], dtype=np.int64),
                                   num_latent_frames=1, latent_height=2, latent_width=2,
                                   num_audio_latents=1, patch_size=tuple(dit.config.patch_size),
                                   keyframe_anchors=())
    timestep, timestep_indices = build_row_timesteps(layout, TIMESTEP_VALUE, TIMESTEP_VALUE,
                                                      TIMESTEP_VALUE, TIMESTEP_VALUE)
    if timestep.tolist() != [TIMESTEP_VALUE] or timestep_indices.tolist() != [0] * layout.sequence_length:
        raise ValueError(f"minimal packed timetable is not canonical: {timestep.tolist()}, {timestep_indices.tolist()}")
    validate_canonical_timestep_dtype(timestep.dtype)
    def pattern(shape: tuple[int, ...], salt: int):
        modulus, offset, scale = deterministic_input_pattern_parameters(salt)
        values = (mx.arange(math.prod(shape), dtype=mx.float32) + salt + 1) % modulus
        return ((values + offset) * scale).reshape(shape).astype(mx.bfloat16)
    video = pattern((1, len(layout.video_indices), int(dit.config.video_patch_dim)), 0)
    audio = pattern((1, len(layout.audio_indices), int(dit.config.audio_latents_dim)), 1)
    text = pattern((1, len(layout.text_indices), int(dit.config.text_dim)), 2)
    for label, array in (("video input", video), ("audio input", audio), ("text input", text)):
        _require_finite(mx, array, label)
        if not _mx_bool(mx, mx.all(array != 0)):
            raise ValueError(f"{label} violates the nonzero deterministic-input contract")
    return layout, timestep, timestep_indices, video, audio, text


def _layout_contract(layout, timestep, timestep_indices) -> dict[str, Any]:
    return {
        "sequence_length": int(layout.sequence_length),
        "text_token_count": int(layout.text_indices.shape[0]),
        "video_token_count": int(layout.video_indices.shape[0]),
        "audio_token_count": int(layout.audio_indices.shape[0]),
        "latent_frames": 1, "latent_height": 2, "latent_width": 2,
        "audio_latents": 1, "audio_channels": 2, "patch_size_source": "dit.config.patch_size",
        "timestep_values": [float(v) for v in timestep.tolist()],
        "timestep_indices": [int(v) for v in timestep_indices.tolist()],
        "token_tags": [int(v) for v in layout.token_tags.tolist()],
        "position_ids": [[float(v) for v in row] for row in layout.position_ids.tolist()],
        "video_indices": [int(v) for v in layout.video_indices.tolist()],
        "audio_indices": [int(v) for v in layout.audio_indices.tolist()],
        "text_indices": [int(v) for v in layout.text_indices.tolist()],
    }


def _build_metadata(*, original: Path, derived: Path, dit, layout, timestep,
                    timestep_indices, arrays: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors", "reference_checkpoint": str(original),
        "derived_checkpoint": str(derived), "reference_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
        "reference_checkpoint_fingerprint_method": REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
        "reference_config_sha256": sha256_file(original / "config.json"),
        "derived_config_sha256": sha256_file(derived / "config.json"),
        "derived_conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
        "derived_sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json"),
        "deterministic_input_method": DETERMINISTIC_INPUT_METHOD,
        "timestep_values": [float(v) for v in timestep.tolist()], "timestep_dtype": CANONICAL_TIMESTEP_DTYPE,
        "configured_resident_block_count": int(len(dit.blocks)), "tensor_keys": list(ARTIFACT_KEYS),
        "tensor_inventory": {k: _shape_dtype(v) for k, v in arrays.items()},
        "packed_layout": _layout_contract(layout, timestep, timestep_indices),
    }


def _write_reference_artifact(mx, artifact: Path, arrays: Mapping[str, Any],
                              metadata: Mapping[str, Any], *, overwrite: bool) -> Path:
    meta_path = artifact_metadata_path(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (artifact.exists() or meta_path.exists()):
        raise FileExistsError(f"reference artifact already exists; pass --overwrite to replace {artifact}")
    if set(arrays) != set(ARTIFACT_KEYS):
        raise ValueError(f"reference artifact arrays have wrong keys: {sorted(arrays)}")
    mx.save_safetensors(str(artifact), dict(arrays), metadata={
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": str(ARTIFACT_SCHEMA_VERSION),
        "deterministic_input_method": DETERMINISTIC_INPUT_METHOD,
        "tensor_keys": json.dumps(list(ARTIFACT_KEYS), separators=(",", ":")),
    })
    receipt = dict(metadata)
    receipt["artifact_sha256"] = sha256_file(artifact)
    meta_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return meta_path


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"reference metadata is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"reference metadata must be a JSON object: {path}")
    missing = sorted(REFERENCE_METADATA_REQUIRED_KEYS - set(value))
    unexpected = sorted(set(value) - REFERENCE_METADATA_REQUIRED_KEYS)
    if missing or unexpected:
        raise ValueError(f"reference metadata key contract mismatch: missing={missing}, unexpected={unexpected}")
    return value


def _validate_array(mx, array, expected: Mapping[str, Any], label: str) -> None:
    if list(array.shape) != list(expected["shape"]) or str(array.dtype) != expected["dtype"]:
        raise ValueError(f"{label} shape/dtype mismatch: got {_shape_dtype(array)}, expected {expected}")
    _require_finite(mx, array, label)


def _load_and_validate_reference(mx, artifact: Path, *, original: Path, derived: Path, dit):
    meta_path = artifact_metadata_path(artifact)
    if not artifact.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"reference artifact and metadata are both required: {artifact}, {meta_path}")
    metadata = _read_metadata(meta_path)
    if metadata["artifact_sha256"] != sha256_file(artifact):
        raise ValueError("reference artifact checksum does not match its metadata")
    arrays = mx.load(str(artifact))
    if set(arrays) != set(ARTIFACT_KEYS):
        raise ValueError(f"reference tensor key set mismatch: got {sorted(arrays)}, expected {list(ARTIFACT_KEYS)}")
    actual_inventory = {key: _shape_dtype(value) for key, value in arrays.items()}
    layout, timestep, timestep_indices, video, audio, text = _build_layout_and_inputs(mx, dit)
    expected_inputs = {
        "video_input": video, "audio_input": audio, "text_input": text, "timestep": timestep,
        "timestep_indices": timestep_indices, "token_tags": layout.token_tags,
        "position_ids": layout.position_ids, "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices, "text_indices": layout.text_indices,
    }
    expected_packed_layout = _layout_contract(layout, timestep, timestep_indices)
    expected_checksums = {
        "artifact_sha256": sha256_file(artifact),
        "reference_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
        "reference_config_sha256": sha256_file(original / "config.json"),
        "derived_config_sha256": sha256_file(derived / "config.json"),
        "derived_conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
        "derived_sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json"),
    }
    validate_reference_metadata(
        metadata,
        original=original,
        derived=derived,
        expected_checksums=expected_checksums,
        expected_tensor_inventory=actual_inventory,
        expected_packed_layout=expected_packed_layout,
        expected_observed_block_indices=list(range(EXPECTED_BLOCK_COUNT)),
    )
    for key in ARTIFACT_KEYS:
        expected = metadata["tensor_inventory"].get(key)
        if not isinstance(expected, dict):
            raise ValueError(f"reference tensor inventory is missing {key}")
        _validate_array(mx, arrays[key], expected, f"reference {key}")
    for key, expected in expected_inputs.items():
        if not _exact_array_equal(mx, arrays[key], expected):
            raise ValueError(f"reference serialized {key} violates deterministic input/layout contract")
    return arrays, {"layout": layout, "timestep": timestep, "timestep_indices": timestep_indices}


def _validate_reference_outputs(mx, arrays, layout, dit) -> None:
    from minimax_h3_mlx.dit import param_dtype
    expected_video = (1, len(layout.video_indices), int(dit.config.video_patch_dim))
    expected_audio = (1, len(layout.audio_indices), int(dit.config.audio_latents_dim))
    if tuple(arrays["resident_video_output"].shape) != expected_video or tuple(arrays["resident_audio_output"].shape) != expected_audio:
        raise ValueError("reference output shapes do not match the derived transformer config")
    if arrays["resident_video_output"].dtype != param_dtype(dit.final_layer.video_out) or arrays["resident_audio_output"].dtype != param_dtype(dit.final_layer.audio_out):
        raise ValueError("reference output dtypes do not match the derived transformer output heads")


def _build_complete_cache(mx, dit, timestep):
    from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache
    started, before = begin_phase(mx)
    sidecar_paths: list[str] = []
    def telemetry(event: str, details: Mapping[str, Any]) -> None:
        if event == "sidecar_opening":
            sidecar_paths.append(str(details["path"]))
        _sidecar_telemetry(mx, event, details)
    cache, stats = build_streamed_modulation_cache(dit, timestep, dtype=mx.bfloat16, telemetry=telemetry)
    validate_complete_cache_stats(cache_table_count=len(cache.tables), configured_block_count=len(dit.blocks),
                                  stats=stats, actual_sidecar_names=[Path(p).name for p in sidecar_paths])
    emit_phase("derived_complete_cache_construction", started, before, snapshot(mx),
               inputs={"timestep": _shape_dtype(timestep)},
               outputs={"cache_tables": len(cache.tables), "tensors_per_table": 6,
                        "storage_dtype": str(mx.bfloat16),
                        "cache_lifecycle": {k: v for k, v in stats.__dict__.items() if k != "per_block"}},
               sidecar_open_count=stats.sidecar_files_opened)
    return cache, stats


def _metric_report(mx, resident, derived) -> dict[str, Any]:
    if resident.shape != derived.shape or resident.dtype != derived.dtype:
        raise ValueError(f"parity metric shape/dtype mismatch: {_shape_dtype(resident)} vs {_shape_dtype(derived)}")
    left, right = resident.astype(mx.float32), derived.astype(mx.float32)
    difference = mx.abs(left - right)
    denominator = mx.maximum(mx.abs(left), mx.array(RELATIVE_DENOMINATOR_FLOOR, dtype=mx.float32))
    exact = _mx_bool(mx, mx.all(resident == derived))
    maximum_absolute, mean_absolute = mx.max(difference), mx.mean(difference)
    rms = mx.sqrt(mx.mean(mx.square(left - right)))
    maximum_relative = mx.max(difference / denominator)
    allclose = mx.allclose(left, right, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)
    mismatched = mx.sum((resident != derived).astype(mx.int32))
    mx.eval(maximum_absolute, mean_absolute, rms, maximum_relative, allclose, mismatched)
    return {
        "exact_equality": exact, "maximum_absolute_difference": float(maximum_absolute.item()),
        "mean_absolute_difference": float(mean_absolute.item()), "root_mean_square_difference": float(rms.item()),
        "maximum_relative_difference": float(maximum_relative.item()),
        "relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR, "allclose": bool(allclose.item()),
        "allclose_atol": ALLCLOSE_ATOL, "allclose_rtol": ALLCLOSE_RTOL,
        "element_count": int(math.prod(resident.shape)),
        "mismatched_element_count_exact": int(mismatched.item()),
        "resident": _shape_dtype(resident), "derived": _shape_dtype(derived),
    }


def _validate_transformer_config(dit, mode: str) -> None:
    if getattr(dit, "construction_mode", None) != mode:
        raise ValueError(f"transformer construction mode is {getattr(dit, 'construction_mode', None)!r}, expected {mode!r}")
    if int(dit.config.num_layers) != EXPECTED_BLOCK_COUNT or len(dit.blocks) != EXPECTED_BLOCK_COUNT:
        raise ValueError(f"transformer must have configured and observed-capable 50 blocks: config={dit.config.num_layers}, actual={len(dit.blocks)}")


def _forward_and_validate(mx, dit, layout, timestep, timestep_indices, video, audio, text, cache=None):
    from minimax_h3_mlx.dit import param_dtype
    with observe_transformer_block_execution(dit) as observed:
        kwargs = {"modulation_cache": cache} if cache is not None else {}
        video_out, audio_out = dit(video, audio, text, timestep, timestep_indices,
                                   layout.token_tags, layout.position_ids, layout.video_indices,
                                   layout.audio_indices, layout.text_indices, **kwargs)
    execution = build_execution_receipt(len(dit.blocks), observed)
    mx.eval(video_out, audio_out)
    expected_video = (1, len(layout.video_indices), int(dit.config.video_patch_dim))
    expected_audio = (1, len(layout.audio_indices), int(dit.config.audio_latents_dim))
    if tuple(video_out.shape) != expected_video or tuple(audio_out.shape) != expected_audio:
        raise ValueError(f"transformer output shapes are invalid: video={video_out.shape}, audio={audio_out.shape}")
    if video_out.dtype != param_dtype(dit.final_layer.video_out) or audio_out.dtype != param_dtype(dit.final_layer.audio_out):
        raise ValueError(f"transformer output dtypes are invalid: video={video_out.dtype}, audio={audio_out.dtype}")
    _require_finite(mx, video_out, "video transformer output")
    _require_finite(mx, audio_out, "audio transformer output")
    return video_out, audio_out, execution


def cmd_create_reference(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_dit
    original, derived, artifact = Path(args.original).resolve(), Path(args.derived).resolve(), Path(args.artifact).resolve()
    total_started, _ = begin_phase(mx)
    opened: list[str] = []
    dit = arrays = layout = timestep = timestep_indices = video = audio = text = video_out = audio_out = None
    release_receipt = None
    failure: BaseException | None = None
    try:
        load_started, load_before = begin_phase(mx)
        record, loader = _load_telemetry(mx, total_started, opened)
        dit = load_dit(original, verbose=True, telemetry=record, tensor_loader=loader)
        _validate_transformer_config(dit, "resident")
        emit_phase("resident_transformer_load", load_started, load_before, snapshot(mx),
                   outputs={"construction_mode": dit.construction_mode, "configured_block_count": len(dit.blocks)},
                   checkpoint_open_count=len(opened))
        layout, timestep, timestep_indices, video, audio, text = _build_layout_and_inputs(mx, dit)
        forward_started, forward_before = begin_phase(mx)
        video_out, audio_out, execution = _forward_and_validate(mx, dit, layout, timestep, timestep_indices, video, audio, text)
        emit_phase("resident_transformer_forward", forward_started, forward_before, snapshot(mx),
                   inputs={"video": _shape_dtype(video), "audio": _shape_dtype(audio), "text": _shape_dtype(text),
                           "timestep": _shape_dtype(timestep), "timestep_indices": _shape_dtype(timestep_indices),
                           "token_tags": _shape_dtype(layout.token_tags), "position_ids": _shape_dtype(layout.position_ids),
                           "external_conditioner": False, "modulation_cache": False},
                   outputs={"video": _shape_dtype(video_out), "audio": _shape_dtype(audio_out), "finite": True},
                   checkpoint_open_count=len(opened), block_execution=execution)
        arrays = {"video_input": video, "audio_input": audio, "text_input": text, "timestep": timestep,
                  "timestep_indices": timestep_indices, "token_tags": layout.token_tags,
                  "position_ids": layout.position_ids, "video_indices": layout.video_indices,
                  "audio_indices": layout.audio_indices, "text_indices": layout.text_indices,
                  "resident_video_output": video_out, "resident_audio_output": audio_out}
        metadata = _build_metadata(original=original, derived=derived, dit=dit, layout=layout,
                                   timestep=timestep, timestep_indices=timestep_indices, arrays=arrays)
        metadata["observed_resident_block_count"] = execution["observed_transformer_block_count"]
        metadata["observed_resident_block_indices"] = execution["observed_transformer_block_indices"]
        write_started, write_before = begin_phase(mx)
        metadata_path = _write_reference_artifact(mx, artifact, arrays, metadata, overwrite=args.overwrite)
        emit_phase("resident_reference_artifact_write", write_started, write_before, snapshot(mx),
                   outputs={"tensor_count": len(arrays), "tensor_inventory": metadata["tensor_inventory"]},
                   artifacts=existing_artifact_paths(artifact, metadata_path), checkpoint_open_count=len(opened),
                   block_execution=execution)
        print(f"reference_artifact={artifact}", flush=True)
        print(f"reference_metadata={metadata_path}", flush=True)
        print(f"reference_tensor_count={len(arrays)}", flush=True)
        print(f"reference_transformer_block_indices_observed={execution['observed_transformer_block_indices']}", flush=True)
    except BaseException as exc:
        failure = detach_exception(exc)
    arrays = layout = timestep = timestep_indices = video = audio = text = video_out = audio_out = None
    dit = None
    release_started, release_before = begin_phase(mx)
    release_receipt = _release(mx)
    emit_phase("resident_reference_release", release_started, release_before, release_receipt["memory"],
               artifacts=existing_artifact_paths(artifact, artifact_metadata_path(artifact)),
               checkpoint_open_count=len(opened),
               release_result=json.dumps(release_receipt, sort_keys=True))
    print(f"resident_release_final_memory={json.dumps(release_receipt, sort_keys=True)}", flush=True)
    if failure is not None:
        raise failure.with_traceback(None)
    print("RESIDENT TRANSFORMER REFERENCE CREATED", flush=True)
    return 0


def cmd_compare_derived(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.load import load_dit
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    artifact, report_path = Path(args.artifact).resolve(), Path(args.report).resolve()
    total_started, _ = begin_phase(mx)
    opened: list[str] = []
    dit = arrays = layout = timestep = timestep_indices = video = audio = text = video_out = audio_out = cache = saved = None
    resident_combined = derived_combined = None
    stats = report = release_receipt = None
    parity_validated = False
    failure: BaseException | None = None
    try:
        # Validate the serialized reference and all checkpoint/manifest bindings before opening
        # the derived model. The shell only supplies the derived config needed to reconstruct the
        # canonical packed inputs; it is not a transformer and holds no model parameters.
        derived_config = DiTConfig.from_json(derived / "config.json")
        arrays, saved = _load_and_validate_reference(
            mx, artifact, original=original, derived=derived,
            dit=SimpleNamespace(config=derived_config),
        )
        layout, timestep, timestep_indices = saved["layout"], saved["timestep"], saved["timestep_indices"]
        load_started, load_before = begin_phase(mx)
        dit = load_dit(derived, verbose=True,
                       telemetry=lambda stage, _model, _info: print(f"derived_checkpoint_stage={stage}", flush=True),
                       tensor_loader=lambda path: (opened.append(path) or mx.load(path)))
        _validate_transformer_config(dit, "cache_only")
        emit_phase("derived_transformer_load", load_started, load_before, snapshot(mx),
                   outputs={"construction_mode": dit.construction_mode, "configured_block_count": len(dit.blocks)},
                   checkpoint_open_count=len(opened))
        _validate_reference_outputs(mx, arrays, layout, dit)
        video, audio, text = arrays["video_input"], arrays["audio_input"], arrays["text_input"]
        cache, stats = _build_complete_cache(mx, dit, timestep)
        forward_started, forward_before = begin_phase(mx)
        video_out, audio_out, execution = _forward_and_validate(mx, dit, layout, timestep, timestep_indices, video, audio, text, cache)
        video_metrics = _metric_report(mx, arrays["resident_video_output"], video_out)
        audio_metrics = _metric_report(mx, arrays["resident_audio_output"], audio_out)
        resident_combined = mx.concatenate([arrays["resident_video_output"].reshape(-1), arrays["resident_audio_output"].reshape(-1)])
        derived_combined = mx.concatenate([video_out.reshape(-1), audio_out.reshape(-1)])
        combined_metrics = _metric_report(mx, resident_combined, derived_combined)
        report = {"artifact": str(artifact), "reference_metadata": str(artifact_metadata_path(artifact)),
                  "reference_checkpoint": str(original), "derived_checkpoint": str(derived),
                  "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
                  "video": video_metrics, "audio": audio_metrics, "combined": combined_metrics,
                  "execution_observation": execution,
                  "cache_lifecycle": {k: v for k, v in stats.__dict__.items() if k != "per_block"},
                  "checkpoint_open_count": len(opened), "sidecar_open_count": stats.sidecar_files_opened,
                  "memory_before_release": snapshot(mx)}
        write_diagnostic_report(report_path, report)
        emit_phase("derived_transformer_forward_parity", forward_started, forward_before, snapshot(mx),
                   inputs={"video": _shape_dtype(video), "audio": _shape_dtype(audio), "text": _shape_dtype(text), "timestep": _shape_dtype(timestep)},
                   outputs={"video": video_metrics, "audio": audio_metrics, "combined": combined_metrics},
                   artifacts=existing_artifact_paths(artifact, artifact_metadata_path(artifact), report_path),
                   checkpoint_open_count=len(opened), sidecar_open_count=stats.sidecar_files_opened,
                   block_execution=execution)
        print(f"video_parity={json.dumps(video_metrics, sort_keys=True)}", flush=True)
        print(f"audio_parity={json.dumps(audio_metrics, sort_keys=True)}", flush=True)
        print(f"combined_parity={json.dumps(combined_metrics, sort_keys=True)}", flush=True)
        print(f"execution_observation={json.dumps(execution, sort_keys=True)}", flush=True)
        print(f"cache_lifecycle={json.dumps(report['cache_lifecycle'], sort_keys=True, default=str)}", flush=True)
        print(f"parity_report={report_path}", flush=True)
        validate_parity_after_report(report_path, video_metrics["exact_equality"], audio_metrics["exact_equality"])
        parity_validated = True
    except BaseException as exc:
        failure = detach_exception(exc)
    resident_combined, derived_combined = clear_derived_comparison_arrays(resident_combined, derived_combined)
    arrays = layout = timestep = timestep_indices = video = audio = text = video_out = audio_out = cache = saved = None
    dit = None
    release_started, release_before = begin_phase(mx)
    release_receipt = _release(mx)
    if report is not None:
        report["memory_release"] = release_receipt
        write_diagnostic_report(report_path, report)
    emit_phase("derived_transformer_release", release_started, release_before, release_receipt["memory"],
               artifacts=existing_artifact_paths(artifact, artifact_metadata_path(artifact), report_path),
               checkpoint_open_count=len(opened), sidecar_open_count=stats.sidecar_files_opened if stats is not None else None,
               release_result=json.dumps(release_receipt, sort_keys=True))
    print(f"derived_release_final_memory={json.dumps(release_receipt, sort_keys=True)}", flush=True)
    if failure is not None:
        raise failure.with_traceback(None)
    emit_parity_success_message(parity_validated)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-reference", help="create the resident full-forward reference")
    create.add_argument("--original", default=DEFAULT_ORIGINAL)
    create.add_argument("--derived", default=DEFAULT_DERIVED)
    create.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=cmd_create_reference)
    compare = subparsers.add_parser("compare-derived", help="compare the cache-only derived full forward")
    compare.add_argument("--original", default=DEFAULT_ORIGINAL)
    compare.add_argument("--derived", default=DEFAULT_DERIVED)
    compare.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    compare.add_argument("--report", default=str(DEFAULT_REPORT))
    compare.set_defaults(func=cmd_compare_derived)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

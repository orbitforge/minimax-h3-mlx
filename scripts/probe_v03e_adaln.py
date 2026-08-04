"""v0.3e AdaLN parity and complete cache-only transformer probes.

Run this script from an external Terminal with the repository virtual environment.  The four
subcommands are deliberately separate so the resident checkpoint and the derived base never need
to coexist:

* ``create-reference`` loads the original resident transformer and writes selected BF16 tables;
* ``verify-resident-release`` loads and releases the original resident transformer;
* ``compare-sidecars`` loads the derived transformer, builds its complete streamed cache, and
  compares blocks 0, 5, and 49 against the saved tables;
* ``forward`` loads the derived transformer, builds the complete streamed cache, and executes the
  real cache-only transformer once on the smallest valid packed layout established by ``packing``.

No Qwen, VAE, scheduler, denoising, decoding, rendering, or generation path is entered.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ORIGINAL = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit"
DEFAULT_DERIVED = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_ARTIFACT = ROOT / "out" / "v0.3e" / "adaln-reference.safetensors"
DEFAULT_REPORT = ROOT / "out" / "v0.3e" / "adaln-parity-report.json"

SELECTED_BLOCKS = (0, 5, 49)
TIMESTEPS = (0.0, 0.5, 1.0)
TENSOR_ORDER = (
    "shift_msa",
    "scale_msa",
    "gate_msa",
    "shift_mlp",
    "scale_mlp",
    "gate_mlp",
)
ARTIFACT_FORMAT = "minimax-h3-mlx-v03e-adaln-reference"
ARTIFACT_SCHEMA_VERSION = 2
REFERENCE_CHECKPOINT_FINGERPRINT_METHOD = (
    "sha256(index-json)+sha256(full-content-of-every-indexed-resident-shard)-v1"
)
REFERENCE_METADATA_REQUIRED_KEYS = frozenset({
    "artifact_format", "artifact_schema_version", "reference_checkpoint", "derived_checkpoint",
    "reference_checkpoint_fingerprint", "reference_checkpoint_fingerprint_method",
    "reference_config_sha256", "derived_config_sha256", "derived_conversion_manifest_sha256",
    "derived_sidecar_manifest_sha256", "selected_blocks", "timestep_values", "timestep_dtype",
    "storage_dtype", "tensor_order", "tensor_keys", "num_layers", "hidden_size",
    "timestep_input_dim", "time_embed_dim", "adaln_out_features", "artifact_sha256",
})


def validate_selected_blocks(blocks: list[int] | tuple[int, ...]) -> None:
    """Reject missing, duplicate, or reordered parity blocks before any model work."""
    expected = list(SELECTED_BLOCKS)
    actual = list(blocks)
    if actual != expected:
        if len(set(actual)) != len(actual):
            reason = "duplicate blocks"
        elif set(actual) != set(expected):
            reason = "missing or unexpected blocks"
        else:
            reason = "reordered blocks"
        raise ValueError(f"selected AdaLN blocks have {reason}: got {actual}, expected {expected}")


def validate_exact_equality(exact_equality: bool) -> None:
    """Reject a reported parity mismatch before the command can report success."""
    if not exact_equality:
        raise ValueError("streamed AdaLN parity failed: exact_equality=False")


def artifact_metadata_path(artifact: Path) -> Path:
    if artifact.suffix != ".safetensors":
        raise ValueError(f"AdaLN reference artifact must use .safetensors: {artifact}")
    return artifact.with_suffix(".json")


def artifact_tensor_keys() -> list[str]:
    return [f"block_{block:03d}_{name}" for block in SELECTED_BLOCKS for name in TENSOR_ORDER]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resident_shard_paths(checkpoint: Path) -> tuple[Path, ...]:
    """Resolve the resident shard set from its canonical safetensors index."""
    index_path = checkpoint / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"resident checkpoint index is missing: {index_path}")
    try:
        index = json.loads(index_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"resident checkpoint index is not valid JSON: {index_path}") from exc
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"resident checkpoint index has no weight_map: {index_path}")
    raw_shard_names = list(weight_map.values())
    if not all(isinstance(name, str) for name in raw_shard_names):
        raise ValueError(f"resident checkpoint index contains non-string shard names: {index_path}")
    shard_names = sorted(set(raw_shard_names))
    if not all(Path(name).name == name for name in shard_names):
        raise ValueError(f"resident checkpoint index contains unsafe shard names: {index_path}")
    paths = tuple(checkpoint / name for name in shard_names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"resident checkpoint index references missing shards: {missing}")
    return paths


def resident_checkpoint_fingerprint(checkpoint: Path) -> str:
    """Fingerprint the indexed resident shard set and every shard's full contents.

    The index hash binds the canonical shard mapping and the ordered shard records bind both the
    indexed filenames and their complete bytes.  This is intentionally stronger than a manifest
    that records only shard names, sizes, or modification times.
    """
    checkpoint = checkpoint.resolve()
    index_path = checkpoint / "model.safetensors.index.json"
    shard_records = [
        {"filename": path.name, "sha256": sha256_file(path)}
        for path in _resident_shard_paths(checkpoint)
    ]
    payload = {
        "method": REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
        "index_sha256": sha256_file(index_path),
        "shards": shard_records,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class _ObservedTransformerBlock:
    """Probe-local callable that records real block invocation before delegation."""

    def __init__(self, block_index: int, block: Any, observed: list[int]) -> None:
        self.block_index = block_index
        self.block = block
        self.observed = observed

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.observed.append(self.block_index)
        return self.block(*args, **kwargs)


@contextmanager
def observe_transformer_block_execution(dit):
    """Observe one forward call and restore the original block list in all cases."""
    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, list):
        raise TypeError("transformer block observation requires a mutable list of blocks")
    original_blocks = list(blocks)
    observed: list[int] = []
    blocks[:] = [
        _ObservedTransformerBlock(index, block, observed)
        for index, block in enumerate(original_blocks)
    ]
    try:
        yield observed
    finally:
        blocks[:] = original_blocks


def validate_observed_block_indices(configured_count: int, observed: list[int] | tuple[int, ...]) -> None:
    expected = list(range(configured_count))
    actual = list(observed)
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


def validate_complete_cache_stats(
    *,
    cache_table_count: int,
    configured_block_count: int,
    stats: Any,
    actual_sidecar_names: list[str] | tuple[str, ...],
) -> None:
    """Fail independently and clearly for every complete-cache lifecycle contract."""
    expected_sidecar_names = [f"block-{index:03d}.safetensors" for index in range(configured_block_count)]
    count_checks = (
        ("cache.tables length", cache_table_count),
        ("stats.blocks_completed", stats.blocks_completed),
        ("stats.sidecar_files_opened", stats.sidecar_files_opened),
        ("stats.unique_sidecar_files_opened", stats.unique_sidecar_files_opened),
        ("stats.successful_payload_opens", stats.successful_payload_opens),
        ("stats.completed_payload_releases", stats.completed_payload_releases),
    )
    for label, actual in count_checks:
        if actual != configured_block_count:
            raise ValueError(
                f"complete cache contract violation: {label}={actual}, "
                f"expected {configured_block_count}"
            )
    if list(actual_sidecar_names) != expected_sidecar_names:
        raise ValueError(
            "complete cache contract violation: sidecar filename order is "
            f"{list(actual_sidecar_names)}, expected {expected_sidecar_names}"
        )
    flag_checks = (
        ("stats.every_sidecar_released_before_next_opened", stats.every_sidecar_released_before_next_opened, True),
        ("stats.sidecar_overlap_observed", stats.sidecar_overlap_observed, False),
        ("stats.next_sidecar_opened_before_previous_release", stats.next_sidecar_opened_before_previous_release, False),
        ("stats.dense_temporary_projection_created", stats.dense_temporary_projection_created, False),
    )
    for label, actual, expected in flag_checks:
        if actual is not expected:
            raise ValueError(f"complete cache contract violation: {label}={actual!r}, expected {expected!r}")


def snapshot(mx) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, name in (
        ("active_memory", "get_active_memory"),
        ("allocator_cache", "get_cache_memory"),
        ("peak_memory", "get_peak_memory"),
    ):
        getter = getattr(mx, name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except Exception:
            result[label] = None
    return result


def reset_peak(mx) -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def begin_phase(mx) -> tuple[float, dict[str, int | None]]:
    reset_peak(mx)
    return time.perf_counter(), snapshot(mx)


def emit_phase(
    phase: str,
    started: float,
    before: dict[str, int | None],
    after: dict[str, int | None],
    *,
    inputs: Mapping[str, Any] | None = None,
    outputs: Mapping[str, Any] | None = None,
    artifacts: list[str] | None = None,
    checkpoint_open_count: int | None = None,
    sidecar_open_count: int | None = None,
    release_result: str | None = None,
) -> None:
    receipt = {
        "phase": phase,
        "wall_clock_seconds": round(time.perf_counter() - started, 6),
        "memory_before": before,
        "memory_after": after,
        "inputs": inputs or {},
        "outputs": outputs or {},
        "artifacts": artifacts or [],
        "checkpoint_open_count": checkpoint_open_count,
        "sidecar_open_count": sidecar_open_count,
        "release_result": release_result,
    }
    print(f"phase_receipt={json.dumps(receipt, sort_keys=True)}", flush=True)


def _shape_dtype(array) -> dict[str, Any]:
    return {"shape": [int(value) for value in array.shape], "dtype": str(array.dtype)}


def _require_finite(mx, array, label: str) -> None:
    finite = mx.all(mx.isfinite(array))
    mx.eval(finite)
    if not bool(finite.item()):
        raise ValueError(f"{label} contains non-finite values")


def _validate_bf16_table(mx, table: Any, *, block_index: int, label: str, expected_shape: tuple[int, int]) -> None:
    if not isinstance(table, tuple) or len(table) != 6:
        raise ValueError(f"block {block_index} {label} must contain exactly six tensors")
    for tensor_index, tensor in enumerate(table):
        if not isinstance(tensor, mx.array):
            raise ValueError(f"block {block_index} {label} tensor {tensor_index} is not an MLX array")
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"block {block_index} {label} tensor {tensor_index} has shape {tensor.shape}; "
                f"expected {expected_shape}"
            )
        if tensor.dtype != mx.bfloat16:
            raise ValueError(
                f"block {block_index} {label} tensor {tensor_index} has dtype {tensor.dtype}; expected bfloat16"
            )
        _require_finite(mx, tensor, f"block {block_index} {label} tensor {tensor_index}")


def _timestep_array(mx):
    return mx.array(TIMESTEPS, dtype=mx.float32)


def _config_metadata(config) -> dict[str, str]:
    return {
        "num_layers": str(int(config.num_layers)),
        "hidden_size": str(int(config.hidden_size)),
        "timestep_input_dim": str(int(config.timestep_input_dim)),
        "time_embed_dim": str(int(config.time_embed_dim)),
        "adaln_out_features": str(int(config.adaln_out_features)),
    }


def _build_reference_metadata(
    *,
    original: Path,
    derived: Path,
    config,
    timesteps: list[float],
) -> dict[str, str]:
    metadata = {
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_schema_version": str(ARTIFACT_SCHEMA_VERSION),
        "reference_checkpoint": str(original),
        "derived_checkpoint": str(derived),
        "reference_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
        "reference_checkpoint_fingerprint_method": REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
        "reference_config_sha256": sha256_file(original / "config.json"),
        "derived_config_sha256": sha256_file(derived / "config.json"),
        "derived_conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
        "derived_sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json"),
        "selected_blocks": json.dumps(list(SELECTED_BLOCKS), separators=(",", ":")),
        "timestep_values": json.dumps(timesteps, separators=(",", ":")),
        "timestep_dtype": "float32",
        "storage_dtype": "bfloat16",
        "tensor_order": json.dumps(list(TENSOR_ORDER), separators=(",", ":")),
        "tensor_keys": json.dumps(artifact_tensor_keys(), separators=(",", ":")),
    }
    metadata.update(_config_metadata(config))
    return metadata


def _write_reference_artifact(mx, artifact: Path, arrays: Mapping[str, Any], metadata: Mapping[str, str], *, overwrite: bool) -> Path:
    metadata_path = artifact_metadata_path(artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (artifact.exists() or metadata_path.exists()):
        raise FileExistsError(
            f"reference artifact already exists; pass --overwrite to replace {artifact} and {metadata_path}"
        )
    mx.save_safetensors(str(artifact), dict(arrays), metadata=dict(metadata))
    sidecar_metadata = dict(metadata)
    sidecar_metadata["artifact_sha256"] = sha256_file(artifact)
    metadata_path.write_text(json.dumps(sidecar_metadata, indent=2, sort_keys=True) + "\n")
    return metadata_path


def _read_reference_artifact(mx, artifact: Path, *, original: Path, derived: Path, config) -> dict[str, Any]:
    metadata_path = artifact_metadata_path(artifact)
    if not artifact.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"reference artifact and metadata are both required: {artifact}, {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict):
        raise ValueError(f"reference metadata must be a JSON object: {metadata_path}")

    missing = sorted(REFERENCE_METADATA_REQUIRED_KEYS - set(metadata))
    if missing:
        raise ValueError(f"reference metadata is incomplete; missing {missing}")
    if metadata.get("artifact_sha256") != sha256_file(artifact):
        raise ValueError("reference artifact checksum does not match its metadata; the artifact is stale or mismatched")

    expected = _build_reference_metadata(
        original=original,
        derived=derived,
        config=config,
        timesteps=[float(value) for value in _timestep_array(mx).tolist()],
    )
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(
                f"reference metadata mismatch for {key}: got {metadata.get(key)!r}, expected {expected_value!r}; "
                "the artifact is stale or belongs to a different checkpoint/timetable"
            )

    arrays = mx.load(str(artifact))
    if set(arrays) != set(artifact_tensor_keys()):
        raise ValueError(
            f"reference tensor keys are incomplete, duplicated, or reordered: got {sorted(arrays)}; "
            f"expected {artifact_tensor_keys()}"
        )
    expected_shape = (len(TIMESTEPS) * 3, int(config.hidden_size))
    for block in SELECTED_BLOCKS:
        table = tuple(arrays[f"block_{block:03d}_{name}"] for name in TENSOR_ORDER)
        _validate_bf16_table(mx, table, block_index=block, label="reference artifact", expected_shape=expected_shape)
    return arrays


def _capture_resident_tables(mx, dit, timesteps) -> dict[str, Any]:
    from minimax_h3_mlx.dit import timestep_embedding

    if getattr(dit, "construction_mode", None) != "resident":
        raise ValueError(f"reference checkpoint did not construct a resident transformer: {dit.construction_mode!r}")
    validate_selected_blocks(list(SELECTED_BLOCKS))
    if len(dit.blocks) <= SELECTED_BLOCKS[-1]:
        raise ValueError(f"resident transformer is missing a selected block: count={len(dit.blocks)}")

    temb = dit.time_embedder(timestep_embedding(timesteps, dit.config.timestep_input_dim))
    mx.eval(temb)
    expected_shape = (len(TIMESTEPS) * 3, int(dit.config.hidden_size))
    arrays: dict[str, Any] = {}
    for block_index in SELECTED_BLOCKS:
        projected = dit.blocks[block_index].adaln_proj(temb)
        if not isinstance(projected, tuple) or len(projected) != 6:
            raise ValueError(f"resident block {block_index} returned an incomplete modulation tuple")
        table = tuple(tensor.astype(mx.bfloat16) for tensor in projected)
        mx.eval(*table)
        _validate_bf16_table(mx, table, block_index=block_index, label="resident reference", expected_shape=expected_shape)
        for name, tensor in zip(TENSOR_ORDER, table):
            arrays[f"block_{block_index:03d}_{name}"] = tensor
    return arrays


def _release(mx, *objects: Any) -> dict[str, int | None]:
    del objects
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    gc.collect()
    return snapshot(mx)


def _load_telemetry(mx, started: float, opened: list[str]):
    def record(stage: str, _model, _info) -> None:
        print(
            "checkpoint_telemetry="
            + json.dumps(
                {
                    "stage": stage,
                    "wall_clock_seconds": round(time.perf_counter() - started, 6),
                    "memory": snapshot(mx),
                    "checkpoint_open_count": len(opened),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def load_tensor_file(path: str):
        opened.append(path)
        return mx.load(path)

    return record, load_tensor_file


def _sidecar_telemetry(mx, event: str, details: Mapping[str, Any]) -> None:
    if event in {"shared_timestep_embedding_materialized", "sidecar_opening", "sidecar_released", "cache_completed"}:
        print(
            "sidecar_telemetry="
            + json.dumps(
                {
                    "event": event,
                    "details": {key: str(value) if key == "stats" else value for key, value in details.items()},
                    "memory": details.get("memory", snapshot(mx)),
                },
                default=str,
                sort_keys=True,
            ),
            flush=True,
        )


def cmd_create_reference(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_dit

    original = Path(args.original).resolve()
    derived = Path(args.derived).resolve()
    artifact = Path(args.artifact).resolve()
    timesteps = _timestep_array(mx)
    started, before = begin_phase(mx)
    opened: list[str] = []
    dit = None
    arrays = None
    metadata_path = None
    try:
        record, tensor_loader = _load_telemetry(mx, started, opened)
        dit = load_dit(original, verbose=True, telemetry=record, tensor_loader=tensor_loader)
        arrays = _capture_resident_tables(mx, dit, timesteps)
        metadata = _build_reference_metadata(
            original=original,
            derived=derived,
            config=dit.config,
            timesteps=[float(value) for value in timesteps.tolist()],
        )
        metadata_path = _write_reference_artifact(mx, artifact, arrays, metadata, overwrite=args.overwrite)
        print(f"reference_artifact={artifact}", flush=True)
        print(f"reference_metadata={metadata_path}", flush=True)
        print(f"selected_blocks={list(SELECTED_BLOCKS)}", flush=True)
        print(f"timestep_values={timesteps.tolist()}", flush=True)
        print(f"reference_tensor_count={len(arrays)}", flush=True)
        emit_phase(
            "resident_reference_creation",
            started,
            before,
            snapshot(mx),
            inputs={"timesteps": _shape_dtype(timesteps), "selected_blocks": list(SELECTED_BLOCKS)},
            outputs={
                "tensor_count": len(arrays),
                "tensor_shapes": [list(shape) for shape in sorted({tuple(value.shape) for value in arrays.values()})],
                "storage_dtype": "bfloat16",
            },
            artifacts=[str(artifact), str(metadata_path)],
            checkpoint_open_count=len(opened),
        )
    finally:
        arrays = None
        release_started, release_before = begin_phase(mx)
        dit = None
        after = _release(mx)
        emit_phase(
            "resident_reference_release",
            release_started,
            release_before,
            after,
            checkpoint_open_count=len(opened),
            release_result="completed",
        )
    print("RESIDENT REFERENCE CREATION PASSED", flush=True)
    return 0


def cmd_verify_resident_release(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_dit

    original = Path(args.original).resolve()
    started, before = begin_phase(mx)
    opened: list[str] = []
    dit = None
    try:
        record, tensor_loader = _load_telemetry(mx, started, opened)
        dit = load_dit(original, verbose=True, telemetry=record, tensor_loader=tensor_loader)
        if getattr(dit, "construction_mode", None) != "resident":
            raise ValueError("resident-release verification did not load a resident transformer")
        print(f"resident_release_loaded_checkpoint={original}", flush=True)
        print(f"resident_release_checkpoint_open_count={len(opened)}", flush=True)
        print(f"resident_release_loaded_memory={snapshot(mx)}", flush=True)
        emit_phase(
            "resident_release_load",
            started,
            before,
            snapshot(mx),
            checkpoint_open_count=len(opened),
        )
    finally:
        release_started, release_before = begin_phase(mx)
        dit = None
        after = _release(mx)
        emit_phase(
            "resident_release_verification",
            release_started,
            release_before,
            after,
            checkpoint_open_count=len(opened),
            release_result="completed",
        )
    print(f"resident_release_final_memory={after}", flush=True)
    print("RESIDENT RELEASE VERIFICATION PASSED", flush=True)
    return 0


def _build_complete_cache(mx, dit, timesteps=None):
    from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache

    if timesteps is None:
        timesteps = _timestep_array(mx)
    cache_started, cache_before = begin_phase(mx)
    sidecar_paths: list[str] = []

    def telemetry(event: str, details: Mapping[str, Any]) -> None:
        if event == "sidecar_opening":
            sidecar_paths.append(str(details["path"]))
        _sidecar_telemetry(mx, event, details)

    cache, stats = build_streamed_modulation_cache(
        dit,
        timesteps,
        dtype=mx.bfloat16,
        telemetry=telemetry,
    )
    expected_shape = (int(timesteps.shape[0]) * 3, int(dit.config.hidden_size))
    for block_index, table in enumerate(cache.tables):
        _validate_bf16_table(mx, table, block_index=block_index, label="streamed cache", expected_shape=expected_shape)
    actual_sidecar_names = [Path(path).name for path in sidecar_paths]
    validate_complete_cache_stats(
        cache_table_count=len(cache.tables),
        configured_block_count=len(dit.blocks),
        stats=stats,
        actual_sidecar_names=actual_sidecar_names,
    )
    emit_phase(
        "streamed_cache_construction",
        cache_started,
        cache_before,
        snapshot(mx),
        inputs={"timesteps": _shape_dtype(timesteps), "timetable_count": int(timesteps.shape[0])},
        outputs={"blocks": len(cache.tables), "tensors_per_block": 6, "table_shape": list(expected_shape), "storage_dtype": str(mx.bfloat16)},
        sidecar_open_count=stats.sidecar_files_opened,
    )
    print(f"streamed_cache_stats={json.dumps({key: value for key, value in stats.__dict__.items() if key != 'per_block'}, default=str, sort_keys=True)}", flush=True)
    return cache, stats, timesteps


def cmd_compare_sidecars(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_dit

    original = Path(args.original).resolve()
    derived = Path(args.derived).resolve()
    artifact = Path(args.artifact).resolve()
    report_path = Path(args.report).resolve()
    started, before = begin_phase(mx)
    arrays = None
    dit = None
    cache = None
    opened: list[str] = []
    stats = None
    try:
        # Read only the small BF16 reference before loading the derived model.  The original model
        # was released by the separate create-reference command.
        from minimax_h3_mlx.config import DiTConfig

        arrays = _read_reference_artifact(
            mx,
            artifact,
            original=original,
            derived=derived,
            config=DiTConfig.from_json(derived / "config.json"),
        )
        dit = load_dit(derived, verbose=True, telemetry=lambda stage, model, info: print(f"derived_checkpoint_stage={stage}", flush=True), tensor_loader=lambda path: (opened.append(path) or mx.load(path)))
        if getattr(dit, "construction_mode", None) != "cache_only":
            raise ValueError("derived parity checkpoint is not cache-only")
        cache, stats, timesteps = _build_complete_cache(mx, dit)
        expected_shape = (len(TIMESTEPS) * 3, int(dit.config.hidden_size))
        blocks_report = []
        exact = True
        compare_started, compare_before = begin_phase(mx)
        for block_index in SELECTED_BLOCKS:
            table = cache.get(block_index)
            _validate_bf16_table(mx, table, block_index=block_index, label="derived cache", expected_shape=expected_shape)
            tensor_reports = []
            block_max = 0.0
            for tensor_index, name in enumerate(TENSOR_ORDER):
                reference = arrays[f"block_{block_index:03d}_{name}"]
                streamed = table[tensor_index]
                difference = mx.max(mx.abs(reference.astype(mx.float32) - streamed.astype(mx.float32)))
                mx.eval(difference)
                maximum = float(difference.item())
                block_max = max(block_max, maximum)
                exact = exact and maximum == 0.0
                tensor_reports.append({
                    "name": name,
                    "reference": _shape_dtype(reference),
                    "streamed": _shape_dtype(streamed),
                    "max_absolute_difference": maximum,
                })
            blocks_report.append({"block_index": block_index, "tensors": tensor_reports, "overall_max_absolute_difference": block_max})
        report = {
            "artifact": str(artifact),
            "reference_checkpoint": str(original),
            "derived_checkpoint": str(derived),
            "selected_blocks": list(SELECTED_BLOCKS),
            "timestep_values": timesteps.tolist(),
            "tensor_order": list(TENSOR_ORDER),
            "storage_dtype": "bfloat16",
            "blocks": blocks_report,
            "exact_equality": exact,
            "checkpoint_open_count": len(opened),
            "sidecar_open_count": stats.sidecar_files_opened,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        for block in blocks_report:
            print(f"block_{block['block_index']:03d}_overall_max_absolute_difference={block['overall_max_absolute_difference']}", flush=True)
            for tensor in block["tensors"]:
                print(
                    f"block_{block['block_index']:03d}_{tensor['name']}_max_absolute_difference="
                    f"{tensor['max_absolute_difference']}",
                    flush=True,
                )
        emit_phase(
            "adaln_parity_comparison",
            compare_started,
            compare_before,
            snapshot(mx),
            inputs={"timesteps": _shape_dtype(timesteps), "selected_blocks": list(SELECTED_BLOCKS), "storage_dtype": "bfloat16"},
            outputs={"exact_equality": exact, "block_reports": len(blocks_report)},
            artifacts=[str(artifact), str(report_path)],
            checkpoint_open_count=len(opened),
            sidecar_open_count=stats.sidecar_files_opened,
        )
        print(f"parity_report={report_path}", flush=True)
        print(f"exact_equality={exact}", flush=True)
        validate_exact_equality(exact)
    finally:
        arrays = None
        cache = None
        table = reference = streamed = difference = None
        release_started, release_before = begin_phase(mx)
        dit = None
        after = _release(mx)
        emit_phase(
            "derived_parity_release",
            release_started,
            release_before,
            after,
            checkpoint_open_count=len(opened),
            sidecar_open_count=stats.sidecar_files_opened if stats is not None else None,
            release_result="completed",
        )
    print("STREAMED ADALN PARITY REPORT GENERATED", flush=True)
    return 0


def cmd_forward(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import numpy as np
    from minimax_h3_mlx.config import TAG_TEXT
    from minimax_h3_mlx.dit import param_dtype
    from minimax_h3_mlx.load import load_dit
    from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps

    derived = Path(args.derived).resolve()
    total_started, _ = begin_phase(mx)
    opened: list[str] = []
    dit = None
    cache = None
    stats = None
    try:
        load_started, load_before = begin_phase(mx)
        record, tensor_loader = _load_telemetry(mx, total_started, opened)
        dit = load_dit(derived, verbose=True, telemetry=record, tensor_loader=tensor_loader)
        if getattr(dit, "construction_mode", None) != "cache_only":
            raise ValueError("tiny-forward checkpoint is not cache-only")
        if len(dit.blocks) != 50:
            raise ValueError(f"tiny-forward requires the complete 50-block transformer, got {len(dit.blocks)} blocks")
        emit_phase("derived_transformer_load", load_started, load_before, snapshot(mx), checkpoint_open_count=len(opened))

        cache, stats, cache_timesteps = _build_complete_cache(mx, dit, mx.array([0.5], dtype=mx.float32))
        # Smallest layout supported by the actual packing contract: one text row, one video patch,
        # one latent audio timestep per stereo channel, and a 2x2 latent spatial grid for (2, 2)
        # patching.  No VAE or external text encoder is needed because the DiT API consumes embeds.
        layout = build_packed_sequence(
            np.array([TAG_TEXT], dtype=np.int64),
            num_latent_frames=1,
            latent_height=2,
            latent_width=2,
            num_audio_latents=1,
            patch_size=tuple(dit.config.patch_size),
            keyframe_anchors=(),
        )
        timestep, timestep_indices = build_row_timesteps(layout, 0.5, 0.5, 0.5, 0.5)
        if timestep.tolist() != [0.5]:
            raise ValueError(f"smallest forward did not produce the expected one-entry timetable: {timestep.tolist()}")
        if timestep.tolist() != cache_timesteps.tolist():
            raise ValueError(
                f"tiny-forward timetable does not match the streamed cache: forward={timestep.tolist()} "
                f"cache={cache_timesteps.tolist()}"
            )
        forward_started, forward_before = begin_phase(mx)
        video = mx.zeros((1, len(layout.video_indices), int(dit.config.video_patch_dim)), dtype=mx.bfloat16)
        audio = mx.zeros((1, len(layout.audio_indices), int(dit.config.audio_latents_dim)), dtype=mx.bfloat16)
        text = mx.zeros((1, len(layout.text_indices), int(dit.config.text_dim)), dtype=mx.bfloat16)
        with observe_transformer_block_execution(dit) as observed_block_indices:
            video_out, audio_out = dit(
                video,
                audio,
                text,
                timestep,
                timestep_indices,
                layout.token_tags,
                layout.position_ids,
                layout.video_indices,
                layout.audio_indices,
                layout.text_indices,
                modulation_cache=cache,
            )
        execution_receipt = build_execution_receipt(len(dit.blocks), observed_block_indices)
        mx.eval(video_out, audio_out)
        expected_video_shape = (1, len(layout.video_indices), int(dit.config.video_patch_dim))
        expected_audio_shape = (1, len(layout.audio_indices), int(dit.config.audio_latents_dim))
        if tuple(video_out.shape) != expected_video_shape or tuple(audio_out.shape) != expected_audio_shape:
            raise ValueError(
                f"tiny forward output shapes are invalid: video={video_out.shape} expected {expected_video_shape}; "
                f"audio={audio_out.shape} expected {expected_audio_shape}"
            )
        expected_video_dtype = param_dtype(dit.final_layer.video_out)
        expected_audio_dtype = param_dtype(dit.final_layer.audio_out)
        if video_out.dtype != expected_video_dtype or audio_out.dtype != expected_audio_dtype:
            raise ValueError(
                f"tiny forward output dtypes are invalid: video={video_out.dtype} expected {expected_video_dtype}; "
                f"audio={audio_out.dtype} expected {expected_audio_dtype}"
            )
        _require_finite(mx, video_out, "tiny forward video output")
        _require_finite(mx, audio_out, "tiny forward audio output")
        emit_phase(
            "tiny_full_cache_only_forward",
            forward_started,
            forward_before,
            snapshot(mx),
            inputs={
                "video": _shape_dtype(video),
                "audio": _shape_dtype(audio),
                "text": _shape_dtype(text),
                "timestep": _shape_dtype(timestep),
                "timestep_indices": _shape_dtype(timestep_indices),
                "token_tags": _shape_dtype(layout.token_tags),
                "position_ids": _shape_dtype(layout.position_ids),
                "sequence_length": layout.sequence_length,
                "configured_transformer_block_count": execution_receipt["configured_transformer_block_count"],
                "external_conditioner": False,
            },
            outputs={
                "video": _shape_dtype(video_out),
                "audio": _shape_dtype(audio_out),
                "finite": True,
                "observed_transformer_block_indices": execution_receipt["observed_transformer_block_indices"],
                "observed_transformer_block_count": execution_receipt["observed_transformer_block_count"],
            },
            checkpoint_open_count=len(opened),
            sidecar_open_count=stats.sidecar_files_opened,
        )
        print(f"tiny_forward_sequence_length={layout.sequence_length}", flush=True)
        print(f"tiny_forward_video_input={_shape_dtype(video)}", flush=True)
        print(f"tiny_forward_audio_input={_shape_dtype(audio)}", flush=True)
        print(f"tiny_forward_text_input={_shape_dtype(text)}", flush=True)
        print(f"tiny_forward_video_output={_shape_dtype(video_out)}", flush=True)
        print(f"tiny_forward_audio_output={_shape_dtype(audio_out)}", flush=True)
        print(f"tiny_forward_transformer_blocks_configured={execution_receipt['configured_transformer_block_count']}", flush=True)
        print(
            "tiny_forward_transformer_block_indices_observed="
            f"{execution_receipt['observed_transformer_block_indices']}",
            flush=True,
        )
        print(f"tiny_forward_transformer_blocks_observed_count={execution_receipt['observed_transformer_block_count']}", flush=True)
    finally:
        cache = None
        video = audio = text = video_out = audio_out = None
        release_started, release_before = begin_phase(mx)
        dit = None
        after = _release(mx)
        emit_phase(
            "tiny_forward_release",
            release_started,
            release_before,
            after,
            checkpoint_open_count=len(opened),
            sidecar_open_count=stats.sidecar_files_opened if stats is not None else None,
            release_result="completed",
        )
    print("TINY FULL CACHE-ONLY FORWARD PASSED", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-reference", help="create BF16 resident reference tables")
    create.add_argument("--original", default=DEFAULT_ORIGINAL)
    create.add_argument("--derived", default=DEFAULT_DERIVED)
    create.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=cmd_create_reference)

    release = subparsers.add_parser("verify-resident-release", help="load and release the resident checkpoint")
    release.add_argument("--original", default=DEFAULT_ORIGINAL)
    release.set_defaults(func=cmd_verify_resident_release)

    compare = subparsers.add_parser("compare-sidecars", help="compare streamed sidecar tables to the reference")
    compare.add_argument("--original", default=DEFAULT_ORIGINAL)
    compare.add_argument("--derived", default=DEFAULT_DERIVED)
    compare.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    compare.add_argument("--report", default=str(DEFAULT_REPORT))
    compare.set_defaults(func=cmd_compare_sidecars)

    forward = subparsers.add_parser("forward", help="run one complete cache-only transformer forward")
    forward.add_argument("--derived", default=DEFAULT_DERIVED)
    forward.set_defaults(func=cmd_forward)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

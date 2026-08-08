"""Sequential AdaLN sidecar execution for complete derived checkpoints.

The derived checkpoint keeps the 50 block AdaLN projections in one sidecar per block.  This module
executes those projections one at a time with MLX's packed quantized matmul primitive and retains
only the six materialized modulation tables from each block.
"""

from __future__ import annotations

import gc
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from .adaln import ModulationCache
from .checkpoint_forge.topology import BLOCK_COUNT, FORMAT_IDENTIFIER
from .config import MODALITY_NUM
from .dit import CACHE_ONLY_CONSTRUCTION, timestep_embedding
from .load import CheckpointFormatInfo, SUPPORTED_DERIVED_SCHEMA_VERSION


SIDECAR_ROLES = ("packed_weight", "scales", "quantization_biases", "learned_bias")
SIDECAR_SUFFIXES = {
    "packed_weight": "weight",
    "scales": "scales",
    "quantization_biases": "biases",
    "learned_bias": "bias",
}
SIDECAR_DTYPE_NAMES = {
    "packed_weight": "U32",
    "scales": "BF16",
    "quantization_biases": "BF16",
    "learned_bias": "BF16",
}


@dataclass(frozen=True)
class MemorySnapshot:
    active: int | None
    allocator_cache: int | None
    peak: int | None


@dataclass
class _SidecarLifecycle:
    """Builder-owned sidecar references; injected loaders may retain their own references."""

    current_payload_live: bool = False
    successful_payload_opens: int = 0
    completed_payload_releases: int = 0
    overlap_observed: bool = False
    next_open_before_previous_release: bool = False

    def require_clear_before_open(self, block_index: int, path: Path) -> None:
        if self.current_payload_live:
            self.overlap_observed = True
            self.next_open_before_previous_release = True
            raise RuntimeError(
                f"block {block_index} sidecar {path}: previous builder-owned sidecar payload is still live"
            )

    def mark_payload_opened(self) -> None:
        self.current_payload_live = True
        self.successful_payload_opens += 1

    def mark_payload_released(self) -> None:
        if self.current_payload_live:
            self.current_payload_live = False
            self.completed_payload_releases += 1


@dataclass(frozen=True)
class StreamedAdaLNBlockStats:
    block_index: int
    sidecar_filename: str
    sidecar_logical_bytes: int
    active_before_load: int | None
    allocator_cache_before_load: int | None
    active_after_sidecar_materialization: int | None
    allocator_cache_after_sidecar_materialization: int | None
    active_after_modulation_materialization: int | None
    allocator_cache_after_modulation_materialization: int | None
    cumulative_retained_cache_bytes: int
    active_after_release_and_purge: int | None
    allocator_cache_after_release_and_purge: int | None
    elapsed_sidecar_io_and_reconstruction_seconds: float
    elapsed_sidecar_materialization_seconds: float
    elapsed_projection_compute_seconds: float
    elapsed_projection_materialization_seconds: float
    elapsed_modulation_materialization_seconds: float
    elapsed_cache_entry_assembly_bookkeeping_seconds: float
    elapsed_release_purge_seconds: float
    elapsed_seconds: float


@dataclass(frozen=True)
class StreamedAdaLNBuildStats:
    """Immutable receipt for one sequential build.

    Lifecycle fields describe only references owned by this builder.  An injected sidecar loader may
    retain external references beyond the builder's control.  ``None`` for dense-temporary status
    means an injected projection executor was used and its allocation behavior is unknown.
    """

    timetable_count: int
    blocks_completed: int
    sidecar_files_opened: int
    unique_sidecar_files_opened: int
    sidecar_logical_bytes_processed: int
    final_cache_bytes: int
    storage_dtype: str
    elapsed_total_seconds: float
    elapsed_shared_timestep_embedding_seconds: float
    elapsed_sidecar_load_seconds: float
    elapsed_sidecar_io_and_reconstruction_seconds: float
    elapsed_sidecar_materialization_seconds: float
    elapsed_projection_seconds: float
    elapsed_projection_compute_seconds: float
    elapsed_projection_materialization_seconds: float
    elapsed_materialization_seconds: float
    elapsed_modulation_materialization_seconds: float
    elapsed_cache_entry_assembly_bookkeeping_seconds: float
    elapsed_release_purge_seconds: float
    elapsed_cache_finalize_materialization_seconds: float
    successful_payload_opens: int
    completed_payload_releases: int
    sidecar_overlap_observed: bool
    next_sidecar_opened_before_previous_release: bool
    peak_mlx_active_memory: int | None
    peak_allocator_cache: int | None
    maximum_one_block_active_memory_increase: int | None
    per_block: tuple[StreamedAdaLNBlockStats, ...]
    allocator_purge_available: bool
    every_sidecar_released_before_next_opened: bool
    dense_temporary_projection_created: bool | None


def _snapshot() -> MemorySnapshot:
    def read(name: str) -> int | None:
        getter = getattr(mx, name, None)
        if not callable(getter):
            return None
        try:
            return int(getter())
        except Exception:
            return None

    return MemorySnapshot(read("get_active_memory"), read("get_cache_memory"), read("get_peak_memory"))


def _dtype_name(dtype: Any) -> str:
    return str(dtype)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _projection_metadata(config, quant_config: Mapping[str, Any], entry: Mapping[str, Any]) -> dict[str, Any]:
    _require(quant_config.get("quantize_adaln") is True, "derived quant_config.json must set quantize_adaln=true")
    _require(quant_config.get("adaln_bits") == 8, "derived AdaLN quantization must use 8 bits")
    _require(quant_config.get("group_size") == 64, "derived AdaLN quantization must use group size 64")

    projection = entry.get("projection")
    _require(isinstance(projection, dict), "AdaLN sidecar entry is missing projection metadata")
    expected = {
        "quantization_bits": 8,
        "quantization_group_size": 64,
        "logical_input_features": int(config.time_embed_dim),
        "logical_output_features": int(config.adaln_out_features),
        "packed_weight_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 4],
        "scales_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 64],
        "quantization_biases_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 64],
        "learned_bias_shape": [int(config.adaln_out_features)],
    }
    _require(projection == expected, f"unsupported AdaLN projection metadata: {projection!r}")
    return expected


def _validate_format_and_manifest(
    dit,
    info: CheckpointFormatInfo,
    block_count: int,
) -> dict[int, tuple[Path, dict[str, Any], dict[str, Any]]]:
    _require(getattr(dit, "construction_mode", None) == CACHE_ONLY_CONSTRUCTION,
             "streamed AdaLN cache construction requires a cache-only transformer")
    _require(info is not None, "streamed AdaLN cache construction requires validated checkpoint format info")
    _require(getattr(info, "checkpoint_format", None) == "derived",
             "streamed AdaLN cache construction requires a derived checkpoint")
    _require(getattr(info, "construction_mode", None) == CACHE_ONLY_CONSTRUCTION,
             "validated checkpoint format info is not cache-only")
    derived_root = getattr(info, "derived_root", None)
    manifest_path = getattr(info, "adaln_manifest_path", None)
    conversion_path = getattr(info, "conversion_manifest_path", None)
    _require(isinstance(derived_root, Path) and derived_root.is_dir(), "derived checkpoint root is unavailable")
    _require(isinstance(conversion_path, Path) and conversion_path.is_file(), "derived conversion manifest is unavailable")
    _require(isinstance(manifest_path, Path) and manifest_path.is_file(), "AdaLN sidecar manifest is unavailable")

    conversion = _read_json(conversion_path, "derived conversion manifest")
    _require(conversion.get("format_identifier") == FORMAT_IDENTIFIER, "unsupported derived checkpoint format identifier")
    _require(conversion.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION,
             "unsupported derived checkpoint schema version")
    _require(conversion.get("bounded") is False, "bounded derived checkpoints cannot build a complete cache")
    _require(conversion.get("verification_status") == "verified", "derived checkpoint is not verified")
    _require(conversion.get("selected_blocks") == list(range(BLOCK_COUNT)),
             "derived checkpoint does not contain all 50 validated blocks")

    sidecar_manifest = _read_json(manifest_path, "AdaLN sidecar manifest")
    required_manifest_keys = {"format_identifier", "schema_version", "bounded", "blocks"}
    _require(set(sidecar_manifest) == required_manifest_keys, "unsupported AdaLN sidecar manifest schema")
    _require(sidecar_manifest.get("format_identifier") == FORMAT_IDENTIFIER,
             "unsupported AdaLN sidecar manifest format identifier")
    _require(sidecar_manifest.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION,
             "unsupported AdaLN sidecar manifest schema version")
    _require(sidecar_manifest.get("bounded") is False, "bounded AdaLN sidecar manifests cannot build a complete cache")

    quant_path = derived_root / "quant_config.json"
    _require(quant_path.is_file(), "derived quant_config.json is missing")
    quant_config = _read_json(quant_path, "derived quant_config.json")
    _require(quant_config.get("bits") is not None and quant_config.get("group_size") is not None,
             "derived quantization recipe is incomplete")
    _require(quant_config.get("adaln_bits") == 8, "derived AdaLN bit width must be 8")
    _require(quant_config.get("group_size") == 64, "derived AdaLN group size must be 64")
    _require(quant_config.get("quantize_adaln") is True, "derived quant_config.json must set quantize_adaln=true")

    blocks = sidecar_manifest.get("blocks")
    _require(isinstance(blocks, dict) and set(blocks) == {str(i) for i in range(BLOCK_COUNT)},
             "AdaLN sidecar manifest must describe all 50 blocks")
    _require(block_count > 0, "transformer has no blocks")
    resolved: dict[int, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for block_index in range(block_count):
        entry = blocks.get(str(block_index))
        _require(isinstance(entry, dict), f"malformed AdaLN sidecar manifest entry for block {block_index}")
        expected_filename = f"block-{block_index:03d}.safetensors"
        _require(entry.get("block_index") == block_index, f"reordered AdaLN sidecar manifest entry for block {block_index}")
        _require(entry.get("sidecar_filename") == expected_filename,
                 f"malformed AdaLN sidecar filename for block {block_index}")
        path = derived_root / "adaln" / expected_filename
        _require(path.is_file(), f"missing AdaLN sidecar for block {block_index}: {path}")
        projection = _projection_metadata(dit.config, quant_config, entry)
        tensors = entry.get("tensors")
        _require(isinstance(tensors, list), f"sidecar tensor manifest for block {block_index} must be a list")
        expected_keys = {
            f"blocks.{block_index}.adaln_proj.linear.{suffix}" for suffix in SIDECAR_SUFFIXES.values()
        }
        actual_keys = [item.get("tensor_key") for item in tensors if isinstance(item, dict)]
        _require(len(tensors) == 4 and len(actual_keys) == 4 and len(set(actual_keys)) == 4,
                 f"sidecar manifest block {block_index} must contain exactly four tensor entries")
        _require(set(actual_keys) == expected_keys, f"sidecar manifest block {block_index} has wrong tensor keys")
        _require(actual_keys == sorted(actual_keys),
                 f"sidecar manifest block {block_index} tensor entries are not in canonical order")
        for item in tensors:
            role = item["tensor_role"]
            role_to_suffix = {
                "packed_weight": "weight",
                "scales": "scales",
                "quantization_biases": "biases",
                "learned_bias": "bias",
            }
            _require(role in role_to_suffix, f"unsupported sidecar tensor role in block {block_index}: {role!r}")
            expected_key = f"blocks.{block_index}.adaln_proj.linear.{role_to_suffix[role]}"
            _require(item.get("tensor_key") == expected_key, f"sidecar tensor role/key mismatch in block {block_index}")
            shape_key = {
                "packed_weight": "packed_weight_shape",
                "scales": "scales_shape",
                "quantization_biases": "quantization_biases_shape",
                "learned_bias": "learned_bias_shape",
            }[role]
            _require(item.get("source_shape") == projection[shape_key], f"wrong {role} shape in block {block_index} manifest")
            _require(item.get("source_dtype") == SIDECAR_DTYPE_NAMES[role], f"wrong {role} dtype in block {block_index} manifest")
            _require(isinstance(item.get("byte_count"), int) and item["byte_count"] > 0,
                     f"invalid byte_count for {role} in block {block_index} manifest")
            if role == "learned_bias":
                _require(item.get("quantization_format") == "unquantized" and item.get("quantization_bits") is None and item.get("group_size") is None,
                         f"learned bias quantization metadata is invalid in block {block_index}")
            else:
                _require(item.get("quantization_format") == "affine" and item.get("quantization_bits") == 8 and item.get("group_size") == 64,
                         f"quantization metadata is invalid in block {block_index} for {role}")
        resolved[block_index] = (path, {item["tensor_key"]: item for item in tensors}, projection)
    return resolved


def _validate_sidecar_payload(
    block_index: int,
    path: Path,
    payload: Mapping[str, Any],
    tensor_manifest: Mapping[str, Mapping[str, Any]],
    projection: Mapping[str, Any],
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    expected_keys = set(tensor_manifest)
    if set(payload) != expected_keys:
        raise ValueError(
            f"block {block_index} sidecar {path} must contain exactly four tensors; "
            f"got {sorted(payload)}"
        )
    suffix_to_key = {item["tensor_role"]: key for key, item in tensor_manifest.items()}
    values: dict[str, mx.array] = {}
    expected_shapes = {
        "packed_weight": tuple(projection["packed_weight_shape"]),
        "scales": tuple(projection["scales_shape"]),
        "quantization_biases": tuple(projection["quantization_biases_shape"]),
        "learned_bias": tuple(projection["learned_bias_shape"]),
    }
    expected_dtypes = {
        "packed_weight": getattr(mx, "uint32"),
        "scales": getattr(mx, "bfloat16"),
        "quantization_biases": getattr(mx, "bfloat16"),
        "learned_bias": getattr(mx, "bfloat16"),
    }
    for role, key in suffix_to_key.items():
        tensor = payload[key]
        if not isinstance(tensor, mx.array):
            raise ValueError(f"block {block_index} sidecar {path} tensor {key} is not an MLX array")
        if tuple(tensor.shape) != expected_shapes[role]:
            raise ValueError(
                f"block {block_index} sidecar {path} tensor {key} has shape {tensor.shape}; "
                f"expected {expected_shapes[role]}"
            )
        if tensor.dtype != expected_dtypes[role]:
            raise ValueError(
                f"block {block_index} sidecar {path} tensor {key} has dtype {tensor.dtype}; "
                f"expected {expected_dtypes[role]}"
            )
        values[role] = tensor
    return values["packed_weight"], values["scales"], values["quantization_biases"], values["learned_bias"]


def _execute_packed_projection(
    activation: mx.array,
    packed_weight: mx.array,
    scales: mx.array,
    quantization_biases: mx.array,
    learned_bias: mx.array,
) -> mx.array:
    """Execute the official MLX affine quantized-linear primitive without a dense weight."""
    output = mx.quantized_matmul(
        activation,
        packed_weight,
        scales=scales,
        biases=quantization_biases,
        transpose=True,
        group_size=64,
        bits=8,
        mode="affine",
    )
    return output + learned_bias


def _materialize_finite(array: mx.array) -> None:
    finite = mx.all(mx.isfinite(array))
    mx.eval(finite)
    if not bool(finite.item()):
        raise ValueError("modulation output contains non-finite values")


def _purge_allocator_cache() -> bool:
    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        return False
    try:
        clear_cache()
    except Exception:
        return False
    return True


def build_streamed_modulation_cache(
    dit,
    timesteps: mx.array,
    *,
    dtype: mx.Dtype = mx.bfloat16,
    sidecar_loader: Callable[[str], Mapping[str, mx.array]] | None = None,
    projection_executor: Callable[[mx.array, mx.array, mx.array, mx.array, mx.array], mx.array] | None = None,
    telemetry: Callable[[str, Mapping[str, Any]], None] | None = None,
    allocator_purge: Callable[[], bool] | None = None,
) -> tuple[ModulationCache, StreamedAdaLNBuildStats]:
    """Build a complete modulation cache by opening exactly one AdaLN sidecar at a time."""
    started = time.perf_counter()
    if not isinstance(timesteps, mx.array) or timesteps.ndim != 1:
        raise ValueError("streamed AdaLN timetable must be a one-dimensional MLX array")
    if timesteps.shape[0] == 0:
        raise ValueError("streamed AdaLN timetable must contain at least one entry")
    values = [float(value) for value in timesteps.tolist()]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("streamed AdaLN timetable must contain finite values")
    if len(set(values)) != len(values):
        raise ValueError("streamed AdaLN timetable must not contain duplicate values")
    block_count = len(getattr(dit, "blocks", ()))
    _require(block_count > 0, "streamed AdaLN cache construction requires transformer blocks")
    _require(block_count <= BLOCK_COUNT, f"transformer exposes {block_count} blocks; expected at most {BLOCK_COUNT}")
    info = getattr(dit, "checkpoint_format_info", None)
    resolved = _validate_format_and_manifest(dit, info, block_count)

    loader = sidecar_loader or mx.load
    executor = projection_executor or _execute_packed_projection
    purge_cache = allocator_purge if allocator_purge is not None else _purge_allocator_cache
    tables: list[tuple[mx.array, ...]] = []
    cache = None
    per_block: list[StreamedAdaLNBlockStats] = []
    opened: list[str] = []
    sidecar_bytes = 0
    load_elapsed = projection_elapsed = materialization_elapsed = release_elapsed = 0.0
    sidecar_io_and_reconstruction_elapsed = 0.0
    sidecar_materialization_elapsed = 0.0
    projection_compute_elapsed = 0.0
    projection_materialization_elapsed = 0.0
    modulation_materialization_elapsed = 0.0
    assembly_bookkeeping_elapsed = 0.0
    cache_finalize_materialization_elapsed = 0.0
    purge_available = allocator_purge is not None or callable(getattr(mx, "clear_cache", None))
    lifecycle = _SidecarLifecycle()
    current_block_index: int | None = None
    current_sidecar_path: Path | None = None
    _emit = telemetry or (lambda _event, _details: None)
    shared_elapsed = 0.0

    try:
        shared_started = time.perf_counter()
        temb = dit.time_embedder(timestep_embedding(timesteps, dit.config.timestep_input_dim))
        mx.eval(temb)
        adaln_activation = nn.silu(temb)
        mx.eval(adaln_activation)
        shared_elapsed = time.perf_counter() - shared_started
        _emit("shared_timestep_embedding_materialized", {"elapsed_seconds": shared_elapsed, "memory": _snapshot()})

        for block_index in range(block_count):
            path, tensor_manifest, projection = resolved[block_index]
            current_block_index = block_index
            current_sidecar_path = path
            lifecycle.require_clear_before_open(block_index, path)
            block_started = time.perf_counter()
            before = _snapshot()
            opened.append(str(path))
            _emit("sidecar_opening", {
                "block_index": block_index,
                "path": str(path),
                "memory": before,
                "builder_payload_live": lifecycle.current_payload_live,
                "successful_payload_opens": lifecycle.successful_payload_opens,
                "completed_payload_releases": lifecycle.completed_payload_releases,
            })
            payload = None
            loaded = None
            packed_weight = scales = quantization_biases = learned_bias = None
            projected = None
            activation = reshaped = None
            table = None
            after_modulation = None
            block_failed = False
            block_sidecar_io_and_reconstruction_elapsed = 0.0
            block_sidecar_materialization_elapsed = 0.0
            block_projection_compute_elapsed = 0.0
            block_projection_materialization_elapsed = 0.0
            block_modulation_materialization_elapsed = 0.0
            block_assembly_bookkeeping_elapsed = 0.0
            block_release_purge_elapsed = 0.0
            try:
                load_started = time.perf_counter()
                loaded = loader(str(path))
                if not isinstance(loaded, Mapping):
                    raise ValueError(f"sidecar loader returned {type(loaded).__name__}, expected a mapping")
                payload = loaded
                lifecycle.mark_payload_opened()
                packed_weight, scales, quantization_biases, learned_bias = _validate_sidecar_payload(
                    block_index, path, payload, tensor_manifest, projection
                )
                block_sidecar_io_and_reconstruction_elapsed = time.perf_counter() - load_started
                sidecar_io_and_reconstruction_elapsed += block_sidecar_io_and_reconstruction_elapsed

                sidecar_materialization_started = time.perf_counter()
                mx.eval(packed_weight, scales, quantization_biases, learned_bias)
                block_sidecar_materialization_elapsed = time.perf_counter() - sidecar_materialization_started
                sidecar_materialization_elapsed += block_sidecar_materialization_elapsed
                load_elapsed += (
                    block_sidecar_io_and_reconstruction_elapsed
                    + block_sidecar_materialization_elapsed
                )
                after_sidecar = _snapshot()
                _emit("sidecar_materialized", {"block_index": block_index, "path": str(path), "memory": after_sidecar})
                sidecar_bytes += sum(int(tensor.nbytes) for tensor in payload.values())

                projection_compute_started = time.perf_counter()
                activation = adaln_activation.astype(scales.dtype)
                projected = executor(activation, packed_weight, scales, quantization_biases, learned_bias)
                expected_projected_shape = (len(values), int(dit.config.adaln_out_features))
                if tuple(projected.shape) != expected_projected_shape:
                    raise ValueError(
                        f"projection returned shape {projected.shape}; expected {expected_projected_shape}"
                    )
                block_projection_compute_elapsed = time.perf_counter() - projection_compute_started
                projection_compute_elapsed += block_projection_compute_elapsed

                projection_materialization_started = time.perf_counter()
                mx.eval(projected)
                block_projection_materialization_elapsed = time.perf_counter() - projection_materialization_started
                projection_materialization_elapsed += block_projection_materialization_elapsed
                projection_elapsed += (
                    block_projection_compute_elapsed
                    + block_projection_materialization_elapsed
                )

                assembly_started = time.perf_counter()
                reshaped = projected.reshape(len(values) * MODALITY_NUM, 6 * int(dit.config.hidden_size))
                table = tuple(
                    reshaped[..., offset * int(dit.config.hidden_size):(offset + 1) * int(dit.config.hidden_size)].astype(dtype)
                    for offset in range(6)
                )
                expected_table_shape = (len(values) * MODALITY_NUM, int(dit.config.hidden_size))
                if len(table) != 6 or any(tuple(array.shape) != expected_table_shape for array in table):
                    raise ValueError(f"block {block_index} produced an invalid six-array modulation table")
                if any(array.dtype not in {mx.float16, mx.bfloat16, mx.float32} for array in table):
                    raise ValueError(f"block {block_index} produced a non-floating modulation table")
                assembly_before_materialization_elapsed = time.perf_counter() - assembly_started

                modulation_materialization_started = time.perf_counter()
                mx.eval(*table)
                for array in table:
                    _materialize_finite(array)
                block_modulation_materialization_elapsed = time.perf_counter() - modulation_materialization_started
                modulation_materialization_elapsed += block_modulation_materialization_elapsed
                materialization_elapsed += block_modulation_materialization_elapsed
                after_modulation = _snapshot()
                tables_started = time.perf_counter()
                tables.append(table)
                table = None
                cumulative_bytes = sum(int(array.nbytes) for prior in tables for array in prior)
                block_assembly_bookkeeping_elapsed = (
                    assembly_before_materialization_elapsed
                    + time.perf_counter() - tables_started
                )
                assembly_bookkeeping_elapsed += block_assembly_bookkeeping_elapsed
            except Exception as exc:
                block_failed = True
                raise ValueError(f"block {block_index} sidecar {path}: {exc}") from exc
            finally:
                payload = None
                loaded = None
                packed_weight = scales = quantization_biases = learned_bias = None
                projected = None
                activation = reshaped = None
                table = None
                lifecycle.mark_payload_released()
                if block_failed:
                    try:
                        _emit("sidecar_cleanup_ready", {
                            "block_index": block_index,
                            "path": str(path),
                            "purge_kind": "failure-cleanup-deferred",
                            "builder_payload_live": lifecycle.current_payload_live,
                            "current_table_live": False,
                            "partial_table_count": len(tables),
                        })
                    except Exception:
                        pass
                else:
                    release_started = time.perf_counter()
                    gc.collect()
                    purge_succeeded = purge_cache()
                    block_release_purge_elapsed = time.perf_counter() - release_started
                    release_elapsed += block_release_purge_elapsed
                    after_release = _snapshot()
                    _emit("sidecar_released", {
                        "block_index": block_index,
                        "path": str(path),
                        "purge_succeeded": purge_succeeded,
                        "purge_kind": "normal-successful-block",
                        "memory": after_release,
                        "builder_payload_live": lifecycle.current_payload_live,
                        "successful_payload_opens": lifecycle.successful_payload_opens,
                        "completed_payload_releases": lifecycle.completed_payload_releases,
                    })
            _require(after_modulation is not None, f"block {block_index} modulation materialization did not complete")
            block_elapsed = time.perf_counter() - block_started
            _emit("cache_block_timing", {
                "block_index": block_index,
                "path": str(path),
                "attribution_schema_version": 1,
                "timings": {
                    "sidecar_io_and_reconstruction_seconds": block_sidecar_io_and_reconstruction_elapsed,
                    "sidecar_materialization_seconds": block_sidecar_materialization_elapsed,
                    "projection_compute_seconds": block_projection_compute_elapsed,
                    "projection_materialization_seconds": block_projection_materialization_elapsed,
                    "modulation_materialization_seconds": block_modulation_materialization_elapsed,
                    "cache_entry_assembly_bookkeeping_seconds": block_assembly_bookkeeping_elapsed,
                    "release_purge_seconds": block_release_purge_elapsed,
                    "total_block_cache_construction_seconds": block_elapsed,
                },
            })
            per_block.append(StreamedAdaLNBlockStats(
                block_index=block_index,
                sidecar_filename=path.name,
                sidecar_logical_bytes=sum(
                    int(tensor["byte_count"])
                    for tensor in tensor_manifest.values()
                    if isinstance(tensor.get("byte_count"), int)
                ),
                active_before_load=before.active,
                allocator_cache_before_load=before.allocator_cache,
                active_after_sidecar_materialization=after_sidecar.active,
                allocator_cache_after_sidecar_materialization=after_sidecar.allocator_cache,
                active_after_modulation_materialization=after_modulation.active,
                allocator_cache_after_modulation_materialization=after_modulation.allocator_cache,
                cumulative_retained_cache_bytes=cumulative_bytes,
                active_after_release_and_purge=after_release.active,
                allocator_cache_after_release_and_purge=after_release.allocator_cache,
                elapsed_sidecar_io_and_reconstruction_seconds=block_sidecar_io_and_reconstruction_elapsed,
                elapsed_sidecar_materialization_seconds=block_sidecar_materialization_elapsed,
                elapsed_projection_compute_seconds=block_projection_compute_elapsed,
                elapsed_projection_materialization_seconds=block_projection_materialization_elapsed,
                elapsed_modulation_materialization_seconds=block_modulation_materialization_elapsed,
                elapsed_cache_entry_assembly_bookkeeping_seconds=block_assembly_bookkeeping_elapsed,
                elapsed_release_purge_seconds=block_release_purge_elapsed,
                elapsed_seconds=block_elapsed,
            ))
            current_block_index = None
            current_sidecar_path = None

        cache_finalize_started = time.perf_counter()
        cache = ModulationCache(tables, timesteps)
        cache.materialize()
        final_cache_bytes = cache.nbytes()
        cache_finalize_materialization_elapsed = time.perf_counter() - cache_finalize_started
        _require(len(tables) == block_count, f"streamed AdaLN cache completed {len(tables)} of {block_count} blocks")
        increases = [
            max(0, block.active_after_modulation_materialization - block.active_before_load)
            for block in per_block
            if block.active_after_modulation_materialization is not None and block.active_before_load is not None
        ]
        stats = StreamedAdaLNBuildStats(
            timetable_count=len(values),
            blocks_completed=len(tables),
            sidecar_files_opened=len(opened),
            unique_sidecar_files_opened=len(set(opened)),
            sidecar_logical_bytes_processed=sidecar_bytes,
            final_cache_bytes=final_cache_bytes,
            storage_dtype=_dtype_name(dtype),
            elapsed_total_seconds=time.perf_counter() - started,
            elapsed_shared_timestep_embedding_seconds=shared_elapsed,
            elapsed_sidecar_load_seconds=load_elapsed,
            elapsed_sidecar_io_and_reconstruction_seconds=sidecar_io_and_reconstruction_elapsed,
            elapsed_sidecar_materialization_seconds=sidecar_materialization_elapsed,
            elapsed_projection_seconds=projection_elapsed,
            elapsed_projection_compute_seconds=projection_compute_elapsed,
            elapsed_projection_materialization_seconds=projection_materialization_elapsed,
            elapsed_materialization_seconds=materialization_elapsed,
            elapsed_modulation_materialization_seconds=modulation_materialization_elapsed,
            elapsed_cache_entry_assembly_bookkeeping_seconds=assembly_bookkeeping_elapsed,
            elapsed_release_purge_seconds=release_elapsed,
            elapsed_cache_finalize_materialization_seconds=cache_finalize_materialization_elapsed,
            successful_payload_opens=lifecycle.successful_payload_opens,
            completed_payload_releases=lifecycle.completed_payload_releases,
            sidecar_overlap_observed=lifecycle.overlap_observed,
            next_sidecar_opened_before_previous_release=lifecycle.next_open_before_previous_release,
            peak_mlx_active_memory=_snapshot().peak or max(
                (value for block in per_block for value in (
                    block.active_before_load,
                    block.active_after_sidecar_materialization,
                    block.active_after_modulation_materialization,
                    block.active_after_release_and_purge,
                ) if value is not None),
                default=None,
            ),
            peak_allocator_cache=max(
                (value for block in per_block for value in (
                    block.allocator_cache_before_load,
                    block.allocator_cache_after_sidecar_materialization,
                    block.allocator_cache_after_modulation_materialization,
                    block.allocator_cache_after_release_and_purge,
                ) if value is not None),
                default=None,
            ),
            maximum_one_block_active_memory_increase=max(increases, default=None),
            per_block=tuple(per_block),
            allocator_purge_available=purge_available,
            every_sidecar_released_before_next_opened=(
                lifecycle.successful_payload_opens == lifecycle.completed_payload_releases
                and not lifecycle.overlap_observed
                and not lifecycle.next_open_before_previous_release
            ),
            dense_temporary_projection_created=(False if projection_executor is None else None),
        )
        _emit("cache_completed", {"stats": stats, "attribution_schema_version": 1, "memory": _snapshot()})
        return cache, stats
    except Exception as exc:
        failure_block = current_block_index
        failure_path = str(current_sidecar_path) if current_sidecar_path is not None else None
        tables.clear()
        cache = None
        payload = loaded = None
        packed_weight = scales = quantization_biases = learned_bias = None
        projected = activation = reshaped = table = None
        temb = None
        adaln_activation = None
        lifecycle.current_payload_live = False
        try:
            _emit("failure_cleanup_ready", {
                "block_index": failure_block,
                "path": failure_path,
                "purge_kind": "failure-cleanup",
                "builder_payload_live": lifecycle.current_payload_live,
                "current_table_live": False,
                "partial_table_count": len(tables),
                "shared_embedding_live": False,
                "adaln_activation_live": False,
                "remaining_intermediates_live": False,
                "successful_payload_opens": lifecycle.successful_payload_opens,
                "completed_payload_releases": lifecycle.completed_payload_releases,
                "sidecar_overlap_observed": lifecycle.overlap_observed,
                "next_sidecar_opened_before_previous_release": lifecycle.next_open_before_previous_release,
                "exception_type": type(exc).__name__,
            })
        except Exception:
            pass
        gc.collect()
        try:
            purge_cache()
        except Exception:
            pass
        raise

"""MiniMax-H3 v0.5d derived full-schedule and standalone-media harness.

The parent process is intentionally MLX-free. It owns one fresh attempt namespace, starts one
conditioning worker and, only after that worker has produced a valid release receipt, starts one
derived cache-only denoising worker. The denoising worker has one independently created streamed
AdaLN session per adjacent scheduler transition. After all derived gates pass, the parent starts
exactly one child-only video decoder and, only after its termination and release/publication gates
pass, exactly one child-only audio decoder. The workers publish validated PNG and WAV artifacts;
the parent then owns one staged MP4 mux and inspection attempt through injectable subprocess seams.
The public-generation CLI remains out of scope.

The production imports are lazy and live only in the child workers. The public helpers in this
module are therefore safe for MLX-free contract tests and for ``--help`` checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Mapping, NoReturn, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.denoise import (
    CANONICAL_PREDICTION_DTYPE,
    copy_runtime_array,
    materialize_predictions,
    validate_updated_latents,
    validated_transformer_forward,
)


LOCKED_PROMPT = (
    "A single dodecahedron rotating slowly on one axis, centered in frame, with a fixed camera and no other objects. "
    "Clean black studio background. Each face is a different impossible material: stained glass, water, chrome, fabric, "
    "portal, soap bubble, crystal, lava, stone, smoke, moss, and a star-filled void. The shape stays perfectly rigid "
    "and symmetrical. Cinematic lighting, soft blue rim light, realistic reflections and refractions, surreal materials, "
    "high detail, smooth motion."
)
PROMPT_UTF8_BYTE_COUNT = len(LOCKED_PROMPT.encode("utf-8"))
PROMPT_SHA256 = hashlib.sha256(LOCKED_PROMPT.encode("utf-8")).hexdigest()
EXPECTED_TOKEN_COUNT = 103
EXPECTED_CONDITIONING_SHAPE = (1, EXPECTED_TOKEN_COUNT, 5120)
EXPECTED_CONDITIONING_DTYPE = "bfloat16"

CANONICAL_SEED = 0
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
REQUESTED_SIGMA_POINTS = 16
EXPECTED_VIDEO_SIGMA_POINTS = 16
EXPECTED_AUDIO_SIGMA_POINTS = 16
EXPECTED_DENOISING_TRANSITIONS = 15
EXPECTED_TRANSFORMER_FORWARDS = 15

VIDEO_NATIVE_SHAPE = (1, 24, 9, 8, 8)
AUDIO_NATIVE_SHAPE = (2, 32, 50)
EXPECTED_TARGET_AUDIO_ROWS = 100
EXPECTED_TARGET_VIDEO_ROWS = 144
EXPECTED_TOTAL_ROWS = EXPECTED_TOKEN_COUNT + EXPECTED_TARGET_AUDIO_ROWS + EXPECTED_TARGET_VIDEO_ROWS
EXPECTED_VIDEO_ROW_WIDTH = 96
EXPECTED_AUDIO_ROW_WIDTH = 32
EXPECTED_BLOCK_COUNT = 50

CONDITIONING_ARRAY_KEYS = frozenset(
    {
        "text_conditioning",
        "token_ids",
        "token_presence_mask",
        "text_token_tags",
        "initial_video_native",
        "initial_audio_native",
        "packed_position_ids",
        "packed_token_tags",
        "packed_video_indices",
        "packed_audio_indices",
        "packed_text_indices",
    }
)
CONDITIONING_LOGICAL_DTYPES = {
    "text_conditioning": EXPECTED_CONDITIONING_DTYPE,
    "token_ids": "int32",
    "token_presence_mask": "int32",
    "text_token_tags": "int32",
    "initial_video_native": "float32",
    "initial_audio_native": "float32",
    "packed_position_ids": "float32",
    "packed_token_tags": "int32",
    "packed_video_indices": "int32",
    "packed_audio_indices": "int32",
    "packed_text_indices": "int32",
}
CONDITIONING_STORAGE_DTYPES = {
    key: ("float32" if key in {"text_conditioning", "initial_video_native", "initial_audio_native", "packed_position_ids"} else "int32")
    for key in CONDITIONING_ARRAY_KEYS
}
CONDITIONING_ARRAY_SHAPES = {
    "text_conditioning": EXPECTED_CONDITIONING_SHAPE,
    "token_ids": (1, EXPECTED_TOKEN_COUNT),
    "token_presence_mask": (1, EXPECTED_TOKEN_COUNT),
    "text_token_tags": (EXPECTED_TOKEN_COUNT,),
    "initial_video_native": VIDEO_NATIVE_SHAPE,
    "initial_audio_native": AUDIO_NATIVE_SHAPE,
    "packed_position_ids": (EXPECTED_TOTAL_ROWS, 3),
    "packed_token_tags": (EXPECTED_TOTAL_ROWS,),
    "packed_video_indices": (EXPECTED_TARGET_VIDEO_ROWS,),
    "packed_audio_indices": (EXPECTED_TARGET_AUDIO_ROWS,),
    "packed_text_indices": (EXPECTED_TOKEN_COUNT,),
}
SESSION_STAT_FIELDS = {
    "blocks_completed": EXPECTED_BLOCK_COUNT,
    "sidecar_files_opened": EXPECTED_BLOCK_COUNT,
    "unique_sidecar_files_opened": EXPECTED_BLOCK_COUNT,
    "successful_payload_opens": EXPECTED_BLOCK_COUNT,
    "completed_payload_releases": EXPECTED_BLOCK_COUNT,
    "every_sidecar_released_before_next_opened": True,
    "sidecar_overlap_observed": False,
    "next_sidecar_opened_before_previous_release": False,
    "dense_temporary_projection_created": False,
}

EXPECTED_LIFECYCLE_TOTALS = {
    "cache_sessions": EXPECTED_DENOISING_TRANSITIONS,
    "cache_sessions_released": EXPECTED_DENOISING_TRANSITIONS,
    "blocks_completed": EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT,
    "sidecar_opens": EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT,
    "sidecar_releases": EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT,
    "maximum_simultaneous_sidecars": 1,
    "overlap_violations": 0,
    "dense_temporary_reconstructions": 0,
    "open_sidecars_after_cleanup": 0,
    "transformer_forwards": EXPECTED_TRANSFORMER_FORWARDS,
    "video_scheduler_updates": EXPECTED_DENOISING_TRANSITIONS,
    "audio_scheduler_updates": EXPECTED_DENOISING_TRANSITIONS,
}

PROBE_FORMAT = "minimax-h3-mlx-v05e-derived-full-schedule"
SCHEMA_VERSION = 2
ATTRIBUTION_SCHEMA_VERSION = 1
ATTRIBUTION_COMPONENT_FIELDS = (
    "sidecar_io_and_reconstruction_seconds",
    "projection_compute_seconds",
    "materialization_evaluation_seconds",
    "cache_entry_assembly_bookkeeping_seconds",
    "release_purge_seconds",
)
ATTRIBUTION_SESSION_OVERHEAD_FIELDS = (
    "shared_timestep_embedding_seconds",
    "cache_finalize_materialization_seconds",
)
ATTRIBUTION_REPORT_CATEGORY_FIELDS = (*ATTRIBUTION_COMPONENT_FIELDS, *ATTRIBUTION_SESSION_OVERHEAD_FIELDS)
ATTRIBUTION_TOTAL_FIELD = "total_block_cache_construction_seconds"
ATTRIBUTION_BLOCK_EVENT = "cache_block_timing"
ATTRIBUTION_SESSION_EVENT = "cache_attribution"
FINAL_ARTIFACT_SCHEMA_VERSION = 1
FINGERPRINT_METHOD = "sha256-logical-shape-dtype-plus-canonical-float32-values-v1"
RNG_METHOD = "mlx.core.random.seed(0)+mlx.core.random.normal-float32-video-then-audio-v1"
DERIVED_FORMAT_IDENTIFIER = "minimax-h3-mlx-streamed-adaln-v1"
DERIVED_SCHEMA_VERSION = 1
EXPECTED_DERIVED_BASE_TENSOR_COUNT = 850

HOST_PROCESS_SNAPSHOT_COMMAND = ("ps", "-axo", "pid=,ppid=,comm=,args=")
HOST_PROCESS_SNAPSHOT_TIMEOUT_SECONDS = 5.0
HOST_PROCESS_SNAPSHOT_MAX_PROCESSES = 4096
HOST_PROCESS_COMMAND_MAX_CHARACTERS = 4096
EXPECTED_HARNESS_CHILD_MARKERS = frozenset(
    {
        "__conditioning-worker",
        "__derived-worker",
        "__video-worker",
        "__audio-worker",
    }
)

DECODER_PHASE_ORDER = (
    "derived-finalization",
    "derived-release-gate",
    "video-worker-start",
    "video-worker-termination",
    "video-release-gate",
    "audio-worker-start",
    "audio-worker-termination",
    "audio-release-gate",
)
DECODER_STATUSES = frozenset({"not_started", "suppressed", "running", "failed", "completed"})
DECODER_WORKER_RECEIPT_KEYS = frozenset(
    {
        "worker_started",
        "worker_pid",
        "worker_exit_observed",
        "worker_exit_code",
        "worker_termination_confirmed",
        "worker_receipt_valid",
        "input_artifact_valid",
        "release_gate_passed",
        "allocator_cache_zero",
        "published_artifact_valid",
    }
)
DECODER_ARTIFACT_SCHEMA_VERSION = 1
DECODER_ARTIFACT_REQUIRED_KEYS = frozenset(
    {
        "artifact_identity",
        "schema_version",
        "attempt_identifier",
        "checkpoint_identity",
        "worker_identity",
        "array_keys",
        "arrays",
        "artifact_npz_sha256",
        "metadata_sha256",
    }
)
VIDEO_WORKER_IDENTITY = "video"
VIDEO_FRAME_MANIFEST_IDENTITY = "minimax-h3-mlx-v05d-video-frame-manifest"
VIDEO_FRAME_MANIFEST_SCHEMA_VERSION = 1
VIDEO_FRAME_COUNT = 30
VIDEO_FRAME_HEIGHT = 128
VIDEO_FRAME_WIDTH = 128
VIDEO_FRAME_FPS = 24
VIDEO_FRAME_DURATION_SECONDS = 1.25
VIDEO_RAW_SHAPE = (1, 3, VIDEO_FRAME_COUNT, VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH)
VIDEO_RGB_SHAPE = (VIDEO_FRAME_COUNT, VIDEO_FRAME_HEIGHT, VIDEO_FRAME_WIDTH, 3)
VIDEO_FRAME_MANIFEST_KEYS = frozenset(
    {
        "manifest_identity",
        "schema_version",
        "attempt_identifier",
        "worker_identity",
        "publication_state",
        "frame_count",
        "width",
        "height",
        "fps",
        "duration_seconds",
        "frames",
        "manifest_sha256",
    }
)
AUDIO_SAMPLE_RATE = 32000
AUDIO_SAMPLE_COUNT = 40000
AUDIO_SAMPLE_WIDTH_BYTES = 2
AUDIO_DURATION_SECONDS = 1.25
AUDIO_RAW_SHAPE = (2, 1, AUDIO_SAMPLE_COUNT)
AUDIO_WAVEFORM_SHAPE = (2, AUDIO_SAMPLE_COUNT)
AUDIO_WORKER_IDENTITY = "audio"
AUDIO_WAV_MANIFEST_IDENTITY = "minimax-h3-mlx-v05d-audio-wav-manifest"
AUDIO_WAV_MANIFEST_SCHEMA_VERSION = 1
AUDIO_WAV_MANIFEST_KEYS = frozenset(
    {
        "manifest_identity",
        "schema_version",
        "attempt_identifier",
        "worker_identity",
        "publication_state",
        "channels",
        "sample_rate",
        "sample_count",
        "sample_width_bytes",
        "duration_seconds",
        "size_bytes",
        "wav_sha256",
        "manifest_sha256",
    }
)

MP4_MUX_LAUNCH_GATE_KEYS = frozenset(
    {
        "derived_phase_status",
        "video_status",
        "audio_status",
        "standalone_media_status",
        "video_release_gate_passed",
        "audio_release_gate_passed",
        "video_worker_termination_confirmed",
        "audio_worker_termination_confirmed",
        "frame_manifest_valid",
        "wav_manifest_valid",
        "passed",
    }
)
MP4_MANIFEST_IDENTITY = "minimax-h3-mlx-v05d-mp4-manifest"
MP4_MANIFEST_SCHEMA_VERSION = 1
MP4_PARTIAL_FILENAME = "dodecahedron.partial.mp4"
MP4_FINAL_FILENAME = "dodecahedron.mp4"
MP4_MANIFEST_FILENAME = "mp4-manifest.json"
MP4_EXPECTED_DURATION_SECONDS = 1.25
# A 20 ms tolerance absorbs container timestamp rounding while still rejecting a materially
# different 30-frame/24-fps output.
MP4_DURATION_TOLERANCE_SECONDS = 0.02
MP4_EXPECTED_VIDEO_CODEC = "h264"
MP4_EXPECTED_AUDIO_CODEC = "aac"
MP4_EXPECTED_PIXEL_FORMAT = "yuv420p"
MP4_EXPECTED_AUDIO_BITRATE = "192k"
MP4_MANIFEST_KEYS = frozenset(
    {
        "manifest_identity",
        "schema_version",
        "attempt_identifier",
        "publication_state",
        "mp4_path",
        "size_bytes",
        "mp4_sha256",
        "video_frame_manifest_path",
        "video_frame_manifest_sha256",
        "video_frame_manifest_file_sha256",
        "audio_manifest_path",
        "audio_manifest_sha256",
        "audio_manifest_file_sha256",
        "video",
        "audio",
        "container",
        "ffprobe_json_sha256",
        "manifest_sha256",
    }
)
EXPECTED_GENERATION_EXCLUSIONS = {
    "resident_comparison_path": True,
    "decoder_phase": False,
    "video_decode": False,
    "png_output": False,
    "audio_decode": False,
    "wav_output": False,
    "ffmpeg_invoked": False,
    "ffprobe_invoked": False,
    "mp4_muxing": False,
    "public_cli_changed": True,
}
DERIVED_DECODER_REQUIRED_GATES = (
    "derived-worker-started-exactly-once",
    "derived-worker-exit-code-zero",
    "derived-worker-termination-confirmed",
    "final-native-latent-metadata-linkage",
    "final-latent-npz-sha256",
    "video-latent-shape-and-logical-dtype",
    "audio-latent-shape-and-logical-dtype",
    "video-latent-fingerprint",
    "audio-latent-fingerprint",
    "schedule-16-16-16-15-15",
    "streamed-adaln-totals",
    "transformer-release-receipt",
    "final-active-memory-gate",
    "final-allocator-cache-zero",
    "final-artifact-contract",
)

GENERATION_EXCLUSIONS = {
    **EXPECTED_GENERATION_EXCLUSIONS,
}

REPORT_KEYS = frozenset(
    {
        "status",
        "run_state",
        "functional_success",
        "schema_version",
        "probe_identity",
        "attempt",
        "invocation",
        "git_identity",
        "checkpoint_identity",
        "prompt",
        "geometry",
        "packing",
        "schedule_contract",
        "conditioning_worker",
        "derived_worker",
        "denoising",
        "streamed_adaln_lifecycle",
        "cache_attribution",
        "final_artifact",
        "event_file_path",
        "event_file_record_count",
        "total_event_records",
        "cache_session_count",
        "sidecar_open_event_count",
        "sidecar_release_event_count",
        "validated_block_pairs",
        "event_file_sha256",
        "memory_telemetry",
        "timing_telemetry",
        "host_contention",
        "phase_order",
        "decoder_phase",
        "video_decoder",
        "audio_decoder",
        "video_artifacts",
        "audio_artifacts",
        "decoder_memory",
        "decoder_timing",
        "decoder_phase_order",
        "standalone_media",
        "mp4_mux",
        "mp4_artifact",
        "mux_timing",
        "mux_failure",
        "decoder_failure",
        "output_paths",
        "generation_exclusions",
        "latent_generation_status",
        "video_status",
        "audio_status",
        "standalone_media_status",
        "mp4_mux_status",
        "failure",
    }
)

FINAL_ARTIFACT_KEYS = frozenset(
    {
        "artifact_identity",
        "schema_version",
        "attempt_identifier",
        "native_video",
        "native_audio",
        "packed_final_state_fingerprint",
        "schedule_contract",
        "completed_transition_count",
        "transformer_forward_count",
        "scheduler_update_counts",
        "streamed_adaln_lifecycle",
        "worker_identity",
        "worker_exit_receipt",
        "transformer_release_receipt",
        "final_active_memory",
        "final_allocator_cache",
        "final_allocator_cache_zero",
        "final_artifact_npz_sha256",
        "metadata_sha256",
        "memory_receipt",
        "git_identity",
        "checkpoint_identity",
    }
)


def _json_safe(value: Any) -> Any:
    """Convert dataclasses, NumPy values, and paths into deterministic JSON values."""
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except BaseException:
            pass
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    converted: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        is_mlx = bool(getattr(value, "__mlx_array__", False)) or value.__class__.__module__.startswith("mlx.")
        converted[key] = np.ascontiguousarray(_as_float32_numpy(value) if is_mlx else np.asarray(value))
    np.savez(path, **converted)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _mlx_core_for(value: Any) -> Any | None:
    """Return lazy MLX dispatch for a marked or genuinely MLX-owned array."""
    marker = getattr(value, "__mlx_array__", False)
    if marker:
        supplied = getattr(value, "__mlx_core__", None)
        if supplied is not None:
            return supplied
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            return None
    module = value.__class__.__module__
    if module == "mlx.core" or module.startswith("mlx."):
        import mlx.core as mx
        return mx
    return None


def _as_float32_numpy(value: Any, mx: Any | None = None) -> np.ndarray:
    """Materialize an array for a small receipt or artifact boundary.

    The import is guarded by the MLX-array marker, so ordinary contract-test paths never import
    MLX.  The production worker evaluates before crossing this boundary.
    """
    is_mlx = bool(getattr(value, "__mlx_array__", False)) or value.__class__.__module__.startswith("mlx.")
    if is_mlx:
        dispatch = mx or _mlx_core_for(value)
        if dispatch is None:
            raise RuntimeError("MLX serialization boundary could not resolve mlx.core dispatch")

        value = value.astype(dispatch.float32)
        dispatch.eval(value)
    array = np.array(value, dtype=np.float32, copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError("fingerprints and serialized arrays require finite values")
    return array


def array_fingerprint(value: Any, *, logical_dtype: str | None = None, mx: Any | None = None) -> str:
    """Hash logical shape/dtype and evaluated canonical float32 values."""
    array = np.ascontiguousarray(_as_float32_numpy(value, mx=mx), dtype=np.float32)
    descriptor = json.dumps(
        {"shape": list(array.shape), "dtype": logical_dtype or _dtype_name(getattr(value, "dtype", "float32"))},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_metadata_sha256(metadata: Mapping[str, Any]) -> str:
    """Hash final metadata without its self-referential linkage field."""
    canonical = {key: value for key, value in metadata.items() if key != "metadata_sha256"}
    payload = json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DecoderArtifactSpec:
    """Parent-side description of a decoder artifact a future worker may publish."""

    def __init__(
        self,
        artifact_path: Path,
        metadata_path: Path,
        expected_keys: frozenset[str],
        array_specs: Mapping[str, Mapping[str, Any]],
        attempt_identifier: str,
        checkpoint_identity: Mapping[str, Any],
        worker_identity: str,
        artifact_identity: str | None = None,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.metadata_path = Path(metadata_path)
        self.expected_keys = frozenset(expected_keys)
        self.array_specs = array_specs
        self.attempt_identifier = attempt_identifier
        self.checkpoint_identity = checkpoint_identity
        self.worker_identity = worker_identity
        self.artifact_identity = artifact_identity


def _decoder_array_spec(spec: Mapping[str, Any], key: str) -> tuple[tuple[int, ...], str, str | None]:
    if not isinstance(spec, Mapping):
        raise ValueError(f"decoder artifact array spec for {key} is not an object")
    shape = spec.get("shape")
    logical_dtype = spec.get("logical_dtype")
    storage_dtype = spec.get("storage_dtype")
    if not isinstance(shape, (list, tuple)) or not shape or any(isinstance(item, bool) for item in shape):
        raise ValueError(f"decoder artifact array spec for {key} has no valid shape")
    try:
        normalized_shape = tuple(int(item) for item in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"decoder artifact array spec for {key} has an invalid shape") from exc
    if any(item < 0 for item in normalized_shape) or not isinstance(logical_dtype, str) or not logical_dtype:
        raise ValueError(f"decoder artifact array spec for {key} has an invalid logical dtype or shape")
    if storage_dtype is not None and (not isinstance(storage_dtype, str) or not storage_dtype):
        raise ValueError(f"decoder artifact array spec for {key} has an invalid storage dtype")
    return normalized_shape, logical_dtype, storage_dtype


def build_decoder_artifact_metadata(
    artifact_path: Path,
    *,
    metadata_path: Path | None = None,
    expected_keys: Sequence[str],
    array_specs: Mapping[str, Mapping[str, Any]],
    attempt_identifier: str,
    checkpoint_identity: Mapping[str, Any],
    worker_identity: str,
    artifact_identity: str | None = None,
) -> dict[str, Any]:
    """Build the strict metadata record future decoder workers can publish.

    This seam accepts only materialized NumPy arrays and performs no MLX import.  A worker may
    materialize its runtime arrays before calling it, then write the returned metadata atomically.
    """
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(f"decoder artifact was not written: {path}")
    expected_key_set = set(expected_keys)
    if expected_key_set != set(array_specs):
        raise ValueError(
            f"decoder artifact array contract is not exact: missing={sorted(expected_key_set - set(array_specs))}, "
            f"unexpected={sorted(set(array_specs) - expected_key_set)}"
        )
    if not isinstance(attempt_identifier, str) or not attempt_identifier:
        raise ValueError("decoder artifact attempt identifier is required")
    if not isinstance(worker_identity, str) or not worker_identity:
        raise ValueError("decoder artifact worker identity is required")
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    if set(arrays) != expected_key_set:
        raise ValueError(
            f"decoder artifact NPZ key set is not exact: missing={sorted(expected_key_set - set(arrays))}, "
            f"unexpected={sorted(set(arrays) - expected_key_set)}"
        )
    recorded_arrays: dict[str, Any] = {}
    for key in sorted(expected_key_set):
        array = np.asarray(arrays[key])
        expected_shape, logical_dtype, expected_storage_dtype = _decoder_array_spec(array_specs[key], key)
        actual_storage_dtype = np.dtype(array.dtype).name
        if tuple(array.shape) != expected_shape:
            raise ValueError(f"decoder artifact array {key} has shape {tuple(array.shape)}, expected {expected_shape}")
        if expected_storage_dtype is not None and actual_storage_dtype != np.dtype(expected_storage_dtype).name:
            raise ValueError(
                f"decoder artifact array {key} has storage dtype {actual_storage_dtype}, expected {expected_storage_dtype}"
            )
        recorded_arrays[key] = {
            "shape": list(expected_shape),
            "storage_dtype": actual_storage_dtype,
            "logical_dtype": logical_dtype,
            "fingerprint": array_fingerprint(array, logical_dtype=logical_dtype),
        }
    metadata: dict[str, Any] = {
        "artifact_identity": artifact_identity or f"minimax-h3-mlx-v05d-{worker_identity}-decoder-artifact",
        "schema_version": DECODER_ARTIFACT_SCHEMA_VERSION,
        "attempt_identifier": attempt_identifier,
        "checkpoint_identity": _json_safe(dict(checkpoint_identity)),
        "worker_identity": worker_identity,
        "array_keys": sorted(expected_key_set),
        "arrays": recorded_arrays,
        "artifact_npz_sha256": sha256_file(path),
        "metadata_sha256": None,
    }
    metadata["metadata_sha256"] = stable_metadata_sha256(metadata)
    if metadata_path is not None:
        _write_json(Path(metadata_path), metadata)
    return metadata


def validate_decoder_artifact(
    artifact_path: Path,
    metadata_path: Path,
    *,
    expected_keys: Sequence[str],
    array_specs: Mapping[str, Mapping[str, Any]],
    attempt_identifier: str,
    checkpoint_identity: Mapping[str, Any],
    worker_identity: str,
    artifact_identity: str | None = None,
    arrays: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a decoder artifact without importing MLX or trusting worker claims.

    The NPZ and JSON files are both read and linked by checksum.  Logical dtype is metadata
    evidence (for example ``bfloat16`` stored as materialized ``float32``), while the on-disk
    storage dtype is checked independently.
    """
    path = Path(artifact_path)
    metadata_file = Path(metadata_path)
    if not path.is_file():
        raise FileNotFoundError(f"decoder artifact is missing: {path}")
    if not metadata_file.is_file():
        raise FileNotFoundError(f"decoder artifact metadata is missing: {metadata_file}")
    metadata = _read_json_object(metadata_file, "decoder artifact metadata")
    if set(metadata) != DECODER_ARTIFACT_REQUIRED_KEYS:
        raise ValueError(
            f"decoder artifact metadata schema mismatch: missing={sorted(DECODER_ARTIFACT_REQUIRED_KEYS - set(metadata))}, "
            f"unexpected={sorted(set(metadata) - DECODER_ARTIFACT_REQUIRED_KEYS)}"
        )
    expected_key_set = set(expected_keys)
    if expected_key_set != set(array_specs):
        raise ValueError("decoder artifact expected keys and array specs are not identical")
    if not isinstance(metadata.get("artifact_identity"), str) or not metadata.get("artifact_identity"):
        raise ValueError("decoder artifact identity is missing")
    if artifact_identity is not None and metadata.get("artifact_identity") != artifact_identity:
        raise ValueError("decoder artifact identity mismatch")
    if metadata.get("schema_version") != DECODER_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("decoder artifact schema version mismatch")
    if metadata.get("attempt_identifier") != attempt_identifier:
        raise ValueError("decoder artifact attempt identifier mismatch")
    if metadata.get("checkpoint_identity") != _json_safe(dict(checkpoint_identity)):
        raise ValueError("decoder artifact checkpoint identity mismatch")
    if metadata.get("worker_identity") != worker_identity:
        raise ValueError("decoder artifact worker identity mismatch")
    if metadata.get("array_keys") != sorted(expected_key_set):
        raise ValueError("decoder artifact metadata key set is not exact")
    if metadata.get("metadata_sha256") != stable_metadata_sha256(metadata):
        raise ValueError("decoder artifact metadata checksum/linkage is stale")
    actual_file_sha256 = sha256_file(path)
    if metadata.get("artifact_npz_sha256") != actual_file_sha256:
        raise ValueError("decoder artifact NPZ SHA-256 does not match its metadata")
    with np.load(path, allow_pickle=False) as loaded:
        loaded_arrays = {key: np.array(loaded[key], copy=True) for key in loaded.files}
    if set(loaded_arrays) != expected_key_set:
        raise ValueError(
            f"decoder artifact NPZ key set mismatch: missing={sorted(expected_key_set - set(loaded_arrays))}, "
            f"unexpected={sorted(set(loaded_arrays) - expected_key_set)}"
        )
    if arrays is not None:
        if set(arrays) != expected_key_set:
            raise ValueError("decoder artifact supplied array key set is not exact")
        for key in sorted(expected_key_set):
            if not np.array_equal(np.asarray(arrays[key]), loaded_arrays[key]):
                raise ValueError(f"decoder artifact supplied array {key} differs from its file")
    recorded_arrays = metadata.get("arrays")
    if not isinstance(recorded_arrays, Mapping) or set(recorded_arrays) != expected_key_set:
        raise ValueError("decoder artifact per-array metadata is not exact")
    for key in sorted(expected_key_set):
        expected_shape, expected_logical_dtype, expected_storage_dtype = _decoder_array_spec(array_specs[key], key)
        array = np.asarray(loaded_arrays[key])
        recorded = recorded_arrays[key]
        if not isinstance(recorded, Mapping) or set(recorded) != {"shape", "storage_dtype", "logical_dtype", "fingerprint"}:
            raise ValueError(f"decoder artifact array metadata for {key} is incomplete")
        actual_storage_dtype = np.dtype(array.dtype).name
        if tuple(array.shape) != expected_shape or recorded.get("shape") != list(expected_shape):
            raise ValueError(f"decoder artifact array {key} shape mismatch")
        if expected_storage_dtype is not None and actual_storage_dtype != np.dtype(expected_storage_dtype).name:
            raise ValueError(f"decoder artifact array {key} storage dtype mismatch")
        if recorded.get("storage_dtype") != actual_storage_dtype:
            raise ValueError(f"decoder artifact array {key} storage dtype differs from metadata")
        if recorded.get("logical_dtype") != expected_logical_dtype:
            raise ValueError(f"decoder artifact array {key} logical dtype mismatch")
        fingerprint = array_fingerprint(array, logical_dtype=expected_logical_dtype)
        if recorded.get("fingerprint") != fingerprint:
            raise ValueError(f"decoder artifact array {key} fingerprint mismatch")
    return {
        "passed": True,
        "artifact_identity": metadata["artifact_identity"],
        "schema_version": metadata["schema_version"],
        "attempt_identifier": metadata["attempt_identifier"],
        "checkpoint_identity": metadata["checkpoint_identity"],
        "worker_identity": metadata["worker_identity"],
        "array_keys": metadata["array_keys"],
        "artifact_npz_sha256": metadata["artifact_npz_sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
    }


validate_decoder_artifact_contract = validate_decoder_artifact


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def shape_dtype(value: Any, *, logical_dtype: str | None = None) -> dict[str, Any]:
    return {"shape": _shape(value), "dtype": logical_dtype or _dtype_name(value.dtype)}


def _serialized_array(value: Any) -> np.ndarray:
    is_mlx = bool(getattr(value, "__mlx_array__", False)) or value.__class__.__module__.startswith("mlx.")
    if is_mlx:
        return np.ascontiguousarray(_as_float32_numpy(value))
    return np.ascontiguousarray(np.asarray(value))


def conditioning_artifact_binding(path: Path, arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact byte and per-array receipt for the conditioning NPZ."""
    keys = set(arrays)
    if keys != set(CONDITIONING_ARRAY_KEYS):
        raise ValueError(
            f"conditioning artifact arrays must be exact: missing={sorted(CONDITIONING_ARRAY_KEYS - keys)}, "
            f"unexpected={sorted(keys - CONDITIONING_ARRAY_KEYS)}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"conditioning artifact was not written: {path}")
    serialized: dict[str, Any] = {
        "sha256": sha256_file(path),
        "array_keys": sorted(CONDITIONING_ARRAY_KEYS),
        "arrays": {},
    }
    for key in sorted(CONDITIONING_ARRAY_KEYS):
        array = _serialized_array(arrays[key])
        logical_dtype = CONDITIONING_LOGICAL_DTYPES[key]
        if tuple(array.shape) != CONDITIONING_ARRAY_SHAPES[key]:
            raise ValueError(f"conditioning array {key} has noncanonical shape {tuple(array.shape)}")
        if np.dtype(array.dtype).name != CONDITIONING_STORAGE_DTYPES[key]:
            raise ValueError(f"conditioning array {key} has noncanonical storage dtype {array.dtype}")
        serialized["arrays"][key] = {
            "shape": [int(item) for item in array.shape],
            "storage_dtype": np.dtype(array.dtype).name,
            "logical_dtype": logical_dtype,
            "fingerprint": array_fingerprint(array, logical_dtype=logical_dtype),
        }
    return serialized


def validate_conditioning_artifact_binding(
    receipt: Mapping[str, Any],
    path: Path,
    *,
    arrays: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Validate the NPZ bytes, schema, dtypes, fingerprints, and packed-layout binding."""
    binding = receipt.get("conditioning_artifact")
    if not isinstance(binding, Mapping):
        raise ValueError("conditioning receipt is missing its artifact binding")
    if set(binding) != {"sha256", "array_keys", "arrays"}:
        raise ValueError("conditioning artifact binding schema is incomplete")
    loaded = dict(arrays) if arrays is not None else _load_npz(path)
    if set(loaded) != set(CONDITIONING_ARRAY_KEYS):
        raise ValueError(
            f"conditioning artifact key set mismatch: missing={sorted(CONDITIONING_ARRAY_KEYS - set(loaded))}, "
            f"unexpected={sorted(set(loaded) - CONDITIONING_ARRAY_KEYS)}"
        )
    if binding.get("array_keys") != sorted(CONDITIONING_ARRAY_KEYS):
        raise ValueError("conditioning receipt array key set is not canonical")
    actual_sha256 = sha256_file(path)
    if binding.get("sha256") != actual_sha256:
        raise ValueError("conditioning artifact file SHA-256 does not match its receipt")
    recorded_arrays = binding.get("arrays")
    if not isinstance(recorded_arrays, Mapping) or set(recorded_arrays) != set(CONDITIONING_ARRAY_KEYS):
        raise ValueError("conditioning receipt per-array binding is incomplete")
    for key in sorted(CONDITIONING_ARRAY_KEYS):
        array = np.asarray(loaded[key])
        recorded = recorded_arrays[key]
        if not isinstance(recorded, Mapping):
            raise ValueError(f"conditioning receipt binding for {key} is invalid")
        expected_shape = list(CONDITIONING_ARRAY_SHAPES[key])
        if tuple(array.shape) != tuple(expected_shape):
            raise ValueError(f"conditioning array {key} has noncanonical shape {tuple(array.shape)}")
        if recorded.get("shape") != expected_shape:
            raise ValueError(f"conditioning array {key} shape differs from its receipt")
        storage_dtype = np.dtype(array.dtype).name
        if storage_dtype != CONDITIONING_STORAGE_DTYPES[key]:
            raise ValueError(f"conditioning array {key} has noncanonical storage dtype {storage_dtype}")
        if recorded.get("storage_dtype") != storage_dtype:
            raise ValueError(f"conditioning array {key} storage dtype differs from its receipt")
        logical_dtype = CONDITIONING_LOGICAL_DTYPES[key]
        if recorded.get("logical_dtype") != logical_dtype:
            raise ValueError(f"conditioning array {key} logical dtype differs from its receipt")
        fingerprint = array_fingerprint(array, logical_dtype=logical_dtype)
        if recorded.get("fingerprint") != fingerprint:
            raise ValueError(f"conditioning array {key} fingerprint differs from its receipt")

    conditioning_receipt = receipt.get("conditioning")
    if not isinstance(conditioning_receipt, Mapping) or conditioning_receipt.get("fingerprint") != recorded_arrays["text_conditioning"].get("fingerprint"):
        raise ValueError("text conditioning fingerprint differs from its receipt")

    tokenizer = receipt.get("tokenizer")
    receipt_token_ids = tokenizer.get("token_ids") if isinstance(tokenizer, Mapping) else None
    if receipt_token_ids is None or not np.array_equal(
        np.asarray(loaded["token_ids"], dtype=np.int32), np.asarray(receipt_token_ids, dtype=np.int32)
    ):
        raise ValueError("conditioning artifact token IDs differ from the conditioning receipt")
    receipt_presence = tokenizer.get("token_presence_mask") if isinstance(tokenizer, Mapping) else None
    if receipt_presence is not None and not np.array_equal(
        np.asarray(loaded["token_presence_mask"], dtype=np.int32), np.asarray(receipt_presence, dtype=np.int32)
    ):
        raise ValueError("conditioning artifact token-presence mask differs from the conditioning receipt")
    deterministic = receipt.get("deterministic_inputs")
    if not isinstance(deterministic, Mapping):
        raise ValueError("conditioning receipt is missing deterministic initial-latent evidence")
    for array_key, receipt_key in (("initial_video_native", "video"), ("initial_audio_native", "audio")):
        expected = deterministic.get(receipt_key)
        if not isinstance(expected, Mapping) or expected.get("fingerprint") != array_fingerprint(
            loaded[array_key], logical_dtype="float32"
        ):
            raise ValueError(f"conditioning artifact {array_key} differs from its receipt")

    packing = receipt.get("packing")
    if not isinstance(packing, Mapping) or packing.get("total_rows") != EXPECTED_TOTAL_ROWS:
        raise ValueError("conditioning artifact packed row count is not locked")
    row_count = int(packing["total_rows"])
    expected_shapes = {key: CONDITIONING_ARRAY_SHAPES[key] for key in (
        "packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices"
    )}
    for key, expected_shape in expected_shapes.items():
        if tuple(loaded[key].shape) != expected_shape or expected_shape[0] != row_count and key in {"packed_position_ids", "packed_token_tags"}:
            raise ValueError(f"conditioning artifact {key} row count or shape is invalid")
    recorded_packed_shapes = {
        "packed_position_ids": packing.get("position_ids_shape"),
        "packed_token_tags": packing.get("token_tags_shape"),
        "packed_video_indices": packing.get("video_indices_shape"),
        "packed_audio_indices": packing.get("audio_indices_shape"),
        "packed_text_indices": packing.get("text_indices_shape"),
    }
    for key, shape in recorded_packed_shapes.items():
        if shape is not None and list(expected_shapes[key]) != list(shape):
            raise ValueError(f"conditioning receipt packed shape for {key} differs from the artifact")
    expected_ranges = derive_row_ranges(EXPECTED_TOKEN_COUNT)
    expected_indices = {
        "packed_text_indices": np.arange(*expected_ranges["text"], dtype=np.int32),
        "packed_audio_indices": np.arange(*expected_ranges["target_audio"], dtype=np.int32),
        "packed_video_indices": np.arange(*expected_ranges["target_video"], dtype=np.int32),
    }
    for key, expected in expected_indices.items():
        if not np.array_equal(loaded[key], expected):
            raise ValueError(f"conditioning artifact {key} does not match the locked packed row ranges")
    expected_tags = np.concatenate(
        [
            np.asarray(loaded["text_token_tags"], dtype=np.int32),
            np.full(EXPECTED_TARGET_AUDIO_ROWS, 2, dtype=np.int32),
            np.full(EXPECTED_TARGET_VIDEO_ROWS, 0, dtype=np.int32),
        ]
    )
    if not np.array_equal(loaded["packed_token_tags"], expected_tags):
        raise ValueError("conditioning artifact packed token tags do not match the locked row layout")
    return loaded


def error_receipt(error: BaseException | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, Mapping):
        return {
            "type": str(error.get("type", "RuntimeError")),
            "message": str(error.get("message", error)),
            "traceback": str(error.get("traceback", "")),
        }
    formatted = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    return {"type": type(error).__name__, "message": str(error), "traceback": formatted}


def failure_fields(
    primary: BaseException | Mapping[str, Any] | None,
    cleanup: BaseException | Mapping[str, Any] | None,
    *,
    cleanup_attempted: bool,
    cleanup_succeeded: bool | None = None,
) -> dict[str, Any]:
    primary_receipt = error_receipt(primary)
    cleanup_receipt = error_receipt(cleanup)
    if cleanup_succeeded is None:
        cleanup_succeeded = bool(cleanup_attempted and cleanup_receipt is None)
    return {
        "primary_error_type": primary_receipt.get("type") if primary_receipt else None,
        "primary_error_message": primary_receipt.get("message") if primary_receipt else None,
        "primary_error_traceback": primary_receipt.get("traceback") if primary_receipt else None,
        "cleanup_attempted": bool(cleanup_attempted),
        "cleanup_succeeded": bool(cleanup_succeeded),
        "cleanup_error_type": cleanup_receipt.get("type") if cleanup_receipt else None,
        "cleanup_error_message": cleanup_receipt.get("message") if cleanup_receipt else None,
        "cleanup_error_traceback": cleanup_receipt.get("traceback") if cleanup_receipt else None,
    }


class PhaseFailure(RuntimeError):
    """A phase failure that retains primary and cleanup exceptions independently."""

    def __init__(
        self,
        phase: str,
        primary: BaseException | Mapping[str, Any],
        *,
        cleanup: BaseException | Mapping[str, Any] | None = None,
        details: Mapping[str, Any] | None = None,
        cleanup_attempted: bool = False,
    ) -> None:
        self.phase = phase
        self.primary_error = primary
        self.cleanup_error = cleanup
        self.details = dict(details or {})
        self.cleanup_attempted = cleanup_attempted
        super().__init__(error_receipt(primary)["message"])


class DenoisingFailure(PhaseFailure):
    """Failure evidence emitted by the full-schedule loop."""

    def __init__(self, primary: BaseException | Mapping[str, Any], *, cleanup=None, state=None, cleanup_attempted=True):
        self.state = dict(state or {})
        super().__init__(
            "derived-denoising",
            primary,
            cleanup=cleanup,
            details={"denoising": self.state},
            cleanup_attempted=cleanup_attempted,
        )


class DecoderGateFailure(PhaseFailure):
    """A derived gate failure that carries the exact gate and suppression evidence."""

    def __init__(
        self,
        failed_gate: str,
        primary: BaseException | Mapping[str, Any],
        *,
        gate_receipt: Mapping[str, Any],
        phase: str = "derived-finalization",
        cleanup: BaseException | Mapping[str, Any] | None = None,
        cleanup_attempted: bool = False,
    ) -> None:
        self.failed_gate = failed_gate
        self.gate_receipt = dict(gate_receipt)
        super().__init__(
            phase,
            primary,
            cleanup=cleanup,
            details={"decoder_gate": self.gate_receipt, "failed_gate": failed_gate},
            cleanup_attempted=cleanup_attempted,
        )


def decoder_worker_receipt(
    worker_identity: str,
    *,
    worker_pid: int = 123,
    worker_started: bool = True,
    worker_exit_observed: bool = True,
    worker_exit_code: int | None = 0,
    worker_termination_confirmed: bool = True,
    worker_receipt_valid: bool = True,
    input_artifact_valid: bool = True,
    release_gate_passed: bool = True,
    allocator_cache_zero: bool = True,
    published_artifact_valid: bool = True,
) -> dict[str, Any]:
    """Create the complete parent-side receipt shape used by fake/future workers."""
    receipt = {
        "worker_identity": worker_identity,
        "worker_started": worker_started,
        "worker_pid": worker_pid,
        "worker_exit_observed": worker_exit_observed,
        "worker_exit_code": worker_exit_code,
        "worker_termination_confirmed": worker_termination_confirmed,
        "worker_receipt_valid": worker_receipt_valid,
        "input_artifact_valid": input_artifact_valid,
        "release_gate_passed": release_gate_passed,
        "allocator_cache_zero": allocator_cache_zero,
        "published_artifact_valid": published_artifact_valid,
    }
    if worker_identity == AUDIO_WORKER_IDENTITY:
        receipt.update(
            {
                "wav_manifest_valid": published_artifact_valid,
                "audio_vae_load_count": 1 if published_artifact_valid else 0,
                "decode_count": 1 if published_artifact_valid else 0,
            }
        )
    return receipt


def validate_decoder_worker_termination(receipt: Mapping[str, Any], *, identity: str) -> None:
    """Validate the termination boundary while allowing a nonzero exit for failure reporting."""
    if receipt.get("worker_identity") != identity:
        raise ValueError(f"{identity} decoder worker identity mismatch")
    for key in ("worker_started", "worker_exit_observed", "worker_termination_confirmed"):
        if type(receipt.get(key)) is not bool or receipt.get(key) is not True:
            raise ValueError(f"{identity} decoder worker termination field {key} is not confirmed")
    pid = receipt.get("worker_pid")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise ValueError(f"{identity} decoder worker termination is missing a positive PID")
    exit_code = receipt.get("worker_exit_code")
    if not isinstance(exit_code, int) or isinstance(exit_code, bool):
        raise ValueError(f"{identity} decoder worker termination is missing an exit code")


def validate_decoder_worker_receipt(receipt: Mapping[str, Any], *, identity: str) -> None:
    """Require a successful, terminated worker and every downstream release/artifact gate."""
    missing = sorted(DECODER_WORKER_RECEIPT_KEYS - set(receipt))
    if missing:
        raise ValueError(f"{identity} decoder worker receipt is missing fields: {missing}")
    validate_decoder_worker_termination(receipt, identity=identity)
    if receipt.get("worker_exit_code") != 0:
        raise ValueError(f"{identity} decoder worker exited with code {receipt.get('worker_exit_code')}")
    for key in (
        "worker_receipt_valid",
        "input_artifact_valid",
        "release_gate_passed",
        "allocator_cache_zero",
        "published_artifact_valid",
    ):
        if type(receipt.get(key)) is not bool or receipt.get(key) is not True:
            raise ValueError(f"{identity} decoder worker receipt gate {key} did not pass")
    if identity == AUDIO_WORKER_IDENTITY:
        if receipt.get("wav_manifest_valid") is not True:
            raise ValueError("audio decoder worker receipt WAV manifest gate did not pass")
        if receipt.get("audio_vae_load_count") != 1 or receipt.get("decode_count") != 1:
            raise ValueError("audio decoder worker receipt does not prove exactly-once VAE load and decode")


def decoder_phase_order_receipt(
    observed: Sequence[str],
    *,
    phase_status: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    result = {
        "required": list(DECODER_PHASE_ORDER),
        "observed": list(observed),
        "phase_status": dict(phase_status or {}),
        "valid": False,
    }
    validate_decoder_phase_order(result)
    result["valid"] = True
    return result


def validate_decoder_phase_order(order: Mapping[str, Any] | Sequence[str]) -> None:
    """Validate the strict partial order without requiring excluded future phases to run."""
    if isinstance(order, Mapping):
        required = order.get("required")
        observed = order.get("observed")
        statuses = order.get("phase_status", {})
        if required != list(DECODER_PHASE_ORDER):
            raise ValueError("decoder phase order required sequence is not canonical")
    else:
        observed = order
        statuses = {}
    if not isinstance(observed, list) or any(phase not in DECODER_PHASE_ORDER for phase in observed):
        raise ValueError("decoder phase order contains an unknown phase")
    if len(observed) != len(set(observed)):
        raise ValueError("decoder phase order contains a duplicate phase")
    positions = {phase: index for index, phase in enumerate(DECODER_PHASE_ORDER)}
    if [positions[phase] for phase in observed] != sorted(positions[phase] for phase in observed):
        raise ValueError("decoder phase order is not strictly ordered")
    if "video-worker-start" in observed and not {
        "derived-finalization",
        "derived-release-gate",
    }.issubset(observed):
        raise ValueError("video worker starts before both derived gates")
    if "audio-worker-start" in observed and not {
        "derived-finalization",
        "derived-release-gate",
        "video-worker-start",
        "video-worker-termination",
        "video-release-gate",
    }.issubset(observed):
        raise ValueError("audio worker starts before video termination and release gates")
    if statuses is not None:
        if not isinstance(statuses, Mapping):
            raise ValueError("decoder phase status map is invalid")
        for phase, status in statuses.items():
            if phase not in DECODER_PHASE_ORDER or status not in DECODER_STATUSES:
                raise ValueError("decoder phase status map contains an invalid phase or status")
        for phase in observed:
            if statuses.get(phase) == "not_started":
                raise ValueError(f"observed decoder phase {phase} cannot be not_started")


def _decoder_section(identity: str) -> dict[str, Any]:
    return {
        "status": "not_started",
        "worker_identity": identity,
        "worker_started": False,
        "worker_launch_count": 0,
        "worker_pid": None,
        "worker_exit_code": None,
        "worker_termination_confirmed": False,
        "worker_receipt_path": None,
        "worker_log_path": None,
        "worker_receipt": None,
        "input_artifact_valid": False,
        "release_gate_passed": False,
        "allocator_cache_zero": False,
        "published_artifact_valid": False,
        "wav_manifest_valid": False,
        "decode_count": 0,
        "vae_load_count": 0,
        "artifact_validation": None,
    }


class DecoderPhaseOrchestrator:
    """Parent state machine for an explicitly scoped decoder phase."""

    def __init__(
        self,
        *,
        derived_gate: Mapping[str, Any],
        worker_launcher: Callable[[str], Mapping[str, Any]],
        video_artifact: DecoderArtifactSpec | None = None,
        audio_artifact: DecoderArtifactSpec | None = None,
        implemented_phase_scope: Mapping[str, bool] | None = None,
        implemented_scope: Mapping[str, bool] | None = None,
        artifact_validators: Mapping[str, Callable[[], Mapping[str, Any]]] | None = None,
    ) -> None:
        self.derived_gate = dict(derived_gate)
        self.worker_launcher = worker_launcher
        self.artifact_specs = {"video": video_artifact, "audio": audio_artifact}
        if implemented_phase_scope is not None and implemented_scope is not None:
            raise ValueError("provide only one explicit decoder phase scope")
        scope = dict(implemented_phase_scope or implemented_scope or {"video": True, "audio": True})
        if set(scope) != {"video", "audio"} or any(type(value) is not bool for value in scope.values()):
            raise ValueError("implemented decoder phase scope must contain boolean video and audio fields")
        if scope["audio"] and not scope["video"]:
            raise ValueError("audio decoder cannot be implemented without the video phase")
        self.implemented_phase_scope = scope
        self.artifact_validators = dict(artifact_validators or {})
        if scope == {"video": True, "audio": False}:
            scope_name = "video_only"
        elif scope == {"video": True, "audio": True}:
            scope_name = "video_then_audio"
        else:
            scope_name = "none"
        self.scope_name = scope_name
        self.phase_status = {phase: "not_started" for phase in DECODER_PHASE_ORDER}
        self.observed: list[str] = []
        self.result: dict[str, Any] = {
            "decoder_phase": {
                "status": "not_started",
                "implemented_scope": scope_name,
                "implemented_phase_scope": dict(scope),
                "derived_gate": self.derived_gate,
                "worker_launches": {"video": 0, "audio": 0},
                "retry_allowed": False,
                "replacement_worker_allowed": False,
            },
            "video_decoder": _decoder_section("video"),
            "audio_decoder": _decoder_section("audio"),
            "video_artifacts": {},
            "audio_artifacts": {},
            "decoder_phase_order": decoder_phase_order_receipt([], phase_status=self.phase_status),
            "decoder_failure": None,
        }

    def _phase(self, phase: str, status: str) -> None:
        if phase not in DECODER_PHASE_ORDER:
            raise ValueError(f"unknown decoder phase: {phase}")
        if phase not in self.observed:
            self.observed.append(phase)
        self.phase_status[phase] = status

    def _suppress_later_phases(self, failed_phase: str) -> None:
        start = DECODER_PHASE_ORDER.index(failed_phase) + 1 if failed_phase in DECODER_PHASE_ORDER else 0
        for phase in DECODER_PHASE_ORDER[start:]:
            if self.phase_status[phase] == "not_started":
                self.phase_status[phase] = "suppressed"

    def _failure(
        self,
        failed_gate: str,
        primary: BaseException | Mapping[str, Any],
        *,
        worker_identity: str | None = None,
        cleanup: BaseException | Mapping[str, Any] | None = None,
        cleanup_attempted: bool = False,
    ) -> None:
        self.result["decoder_phase"]["status"] = "failed"
        self.result["decoder_phase"]["failed_gate"] = failed_gate
        self._suppress_later_phases(failed_gate)
        for identity in ("video", "audio"):
            section = self.result[f"{identity}_decoder"]
            if not self.implemented_phase_scope[identity]:
                section["status"] = "not_started"
                section["suppression_reason"] = "phase_not_implemented"
                continue
            if section["status"] not in {"failed", "completed"}:
                section["status"] = "suppressed"
                section["suppression_reason"] = failed_gate
        self.result["decoder_failure"] = {
            "failed_gate": failed_gate,
            "worker_identity": worker_identity,
            "primary_error": error_receipt(primary),
            "cleanup_error": error_receipt(cleanup),
            **failure_fields(
                primary,
                cleanup,
                cleanup_attempted=cleanup_attempted,
                cleanup_succeeded=cleanup is None if cleanup_attempted else False,
            ),
            "suppressed_phases": [
                phase for phase, status in self.phase_status.items() if status == "suppressed"
            ],
            "retry_suppressed": True,
            "replacement_worker_suppressed": True,
        }
        self.result["decoder_phase_order"] = decoder_phase_order_receipt(
            self.observed,
            phase_status=self.phase_status,
        )

    def _validate_artifact(self, identity: str) -> dict[str, Any] | None:
        validator = self.artifact_validators.get(identity)
        if validator is not None:
            value = validator()
            if not isinstance(value, Mapping):
                raise ValueError(f"{identity} decoder artifact validator did not return an object")
            return dict(value)
        spec = self.artifact_specs[identity]
        if spec is None:
            return None
        return validate_decoder_artifact(
            spec.artifact_path,
            spec.metadata_path,
            expected_keys=spec.expected_keys,
            array_specs=spec.array_specs,
            attempt_identifier=spec.attempt_identifier,
            checkpoint_identity=spec.checkpoint_identity,
            worker_identity=spec.worker_identity,
            artifact_identity=spec.artifact_identity,
        )

    def _launch(self, identity: str) -> bool:
        if not self.implemented_phase_scope.get(identity, False):
            raise RuntimeError(f"{identity} decoder phase is not implemented in this scope")
        section = self.result[f"{identity}_decoder"]
        termination_phase = f"{identity}-worker-termination"
        release_phase = f"{identity}-release-gate"
        if section["worker_launch_count"] != 0:
            raise RuntimeError(f"duplicate {identity} decoder worker launch rejected")
        section["status"] = "running"
        section["worker_started"] = True
        section["worker_launch_count"] = 1
        self.result["decoder_phase"]["worker_launches"][identity] = 1
        self._phase(f"{identity}-worker-start", "completed")
        receipt: Mapping[str, Any] | None = None
        try:
            raw_receipt = self.worker_launcher(identity)
            if not isinstance(raw_receipt, Mapping):
                raise ValueError(f"{identity} decoder worker did not return a receipt object")
            receipt = dict(raw_receipt)
            section["worker_receipt"] = receipt
            section["worker_pid"] = receipt.get("worker_pid")
            section["worker_exit_code"] = receipt.get("worker_exit_code")
            section["worker_termination_confirmed"] = receipt.get("worker_termination_confirmed")
            section["worker_receipt_path"] = receipt.get("worker_receipt_path")
            section["worker_log_path"] = receipt.get("worker_log_path")
            self._phase(termination_phase, "running")
            validate_decoder_worker_termination(receipt, identity=identity)
            self._phase(termination_phase, "completed")
            self._phase(release_phase, "running")
            validate_decoder_worker_receipt(receipt, identity=identity)
            artifact_validation = self._validate_artifact(identity)
            section["artifact_validation"] = artifact_validation
            if artifact_validation is not None:
                self.result[f"{identity}_artifacts"] = artifact_validation
            section.update(
                {
                    "input_artifact_valid": receipt["input_artifact_valid"],
                    "release_gate_passed": receipt["release_gate_passed"],
                    "allocator_cache_zero": receipt["allocator_cache_zero"],
                    "published_artifact_valid": receipt["published_artifact_valid"],
                    "wav_manifest_valid": receipt.get("wav_manifest_valid", False),
                    "decode_count": receipt.get("decode_count", receipt.get("audio_decode_count", 0)),
                    "vae_load_count": receipt.get("vae_load_count", receipt.get("audio_vae_load_count", 0)),
                    "status": "completed",
                }
            )
            self._phase(release_phase, "completed")
            return True
        except BaseException as exc:
            section["status"] = "failed"
            if receipt is not None:
                section["worker_receipt"] = dict(receipt)
            if termination_phase not in self.observed:
                self._phase(termination_phase, "failed")
            elif self.phase_status[termination_phase] != "completed":
                self.phase_status[termination_phase] = "failed"
            else:
                self._phase(release_phase, "failed")
            cleanup = receipt.get("cleanup_error") if isinstance(receipt, Mapping) else None
            self._failure(
                termination_phase if self.phase_status[termination_phase] == "failed" else release_phase,
                exc,
                worker_identity=identity,
                cleanup=cleanup,
                cleanup_attempted=cleanup is not None,
            )
            return False

    def run(self) -> dict[str, Any]:
        observed_gates = self.derived_gate.get("gates")
        complete_gate_receipt = (
            isinstance(observed_gates, Mapping)
            and all(observed_gates.get(name) is True for name in DERIVED_DECODER_REQUIRED_GATES)
            and self.derived_gate.get("finalization_passed") is True
            and self.derived_gate.get("release_gate_passed") is True
            and self.derived_gate.get("passed") is True
        )
        finalization_passed = self.derived_gate.get("finalization_passed") is True
        if finalization_passed:
            self._phase("derived-finalization", "completed")
        else:
            self._phase("derived-finalization", "failed")
        if not complete_gate_receipt:
            failed_gate = str(
                self.derived_gate.get("failed_gate")
                or ("derived-decoder-gate-receipt" if self.derived_gate.get("passed") is True else "derived-finalization")
            )
            phase = "derived-release-gate" if self.derived_gate.get("finalization_passed") is True else "derived-finalization"
            if phase == "derived-release-gate":
                self._phase(phase, "failed")
            self._failure(
                phase,
                self.derived_gate.get("primary_error") or f"derived decoder gate failed: {failed_gate}",
                cleanup=self.derived_gate.get("cleanup_error"),
                cleanup_attempted=self.derived_gate.get("cleanup_attempted") is True,
            )
            self.result["decoder_phase"]["failed_gate"] = failed_gate
            self.result["decoder_failure"]["failed_gate"] = failed_gate
            return self.result
        self._phase("derived-release-gate", "completed")
        if self.implemented_phase_scope["video"] and not self._launch("video"):
            return self.result
        if self.implemented_phase_scope["audio"] and not self._launch("audio"):
            return self.result
        self.result["decoder_phase"]["status"] = "completed"
        self.result["decoder_phase_order"] = decoder_phase_order_receipt(
            self.observed,
            phase_status=self.phase_status,
        )
        return self.result


def validate_locked_prompt(prompt: str) -> None:
    if prompt != LOCKED_PROMPT:
        raise ValueError("prompt must equal the locked v0.5d UTF-8 prompt exactly")


def validate_seed(seed: int) -> None:
    if seed != CANONICAL_SEED:
        raise ValueError(f"canonical v0.5d proof accepts only seed {CANONICAL_SEED}, got {seed}")


def prompt_receipt(prompt: str, token_ids: Any | None = None) -> dict[str, Any]:
    validate_locked_prompt(prompt)
    result: dict[str, Any] = {
        "text": prompt,
        "utf8_byte_count": len(prompt.encode("utf-8")),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_is_literal": True,
        "negative_prompt": None,
        "image_conditioning": False,
        "chat_template": None,
        "special_tokens": False,
    }
    if token_ids is not None:
        ids = np.asarray(token_ids, dtype=np.int32)
        result["token_ids"] = ids.tolist()
        result["token_count"] = int(ids.shape[-1])
    return result


def canonical_geometry_contract() -> dict[str, Any]:
    return {
        "resolution": [128, 128],
        "frames": 30,
        "fps": 24,
        "duration_seconds": 1.25,
        "video_native_latent_shape": list(VIDEO_NATIVE_SHAPE),
        "audio_native_latent_shape": list(AUDIO_NATIVE_SHAPE),
        "audio_sample_rate": 32_000,
        "audio_samples_per_channel": 40_000,
        "text_rows": EXPECTED_TOKEN_COUNT,
        "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
        "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
        "total_rows": EXPECTED_TOTAL_ROWS,
    }


def derive_row_ranges(text_rows: int) -> dict[str, Any]:
    if not isinstance(text_rows, int) or isinstance(text_rows, bool) or text_rows <= 0:
        raise ValueError("text row count must be a positive integer")
    audio_start = text_rows
    video_start = audio_start + EXPECTED_TARGET_AUDIO_ROWS
    return {
        "text": [0, text_rows],
        "target_audio": [audio_start, video_start],
        "target_video": [video_start, video_start + EXPECTED_TARGET_VIDEO_ROWS],
        "text_rows": text_rows,
        "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
        "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
        "total_rows": text_rows + EXPECTED_TARGET_AUDIO_ROWS + EXPECTED_TARGET_VIDEO_ROWS,
    }


def packed_contract(text_rows: int = EXPECTED_TOKEN_COUNT) -> dict[str, Any]:
    ranges = derive_row_ranges(text_rows)
    return {
        "row_order": "[text | target audio | target video]",
        "row_ranges": ranges,
        "text_rows": text_rows,
        "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
        "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
        "total_rows": ranges["total_rows"],
        "feature_widths": {"text": 5120, "target_audio": EXPECTED_AUDIO_ROW_WIDTH, "target_video": EXPECTED_VIDEO_ROW_WIDTH},
        "attention_mask": None,
        "padding_rows": 0,
    }


def validate_packed_contract(contract: Mapping[str, Any], *, expected_text_rows: int = EXPECTED_TOKEN_COUNT) -> None:
    expected = packed_contract(expected_text_rows)
    if contract.get("row_order") != expected["row_order"]:
        raise ValueError("packed row order must be [text | target audio | target video]")
    if contract.get("row_ranges") != expected["row_ranges"]:
        raise ValueError("packed row ranges do not match locked geometry")
    for key in ("text_rows", "target_audio_rows", "target_video_rows", "total_rows"):
        if contract.get(key) != expected[key]:
            raise ValueError(f"packed {key} does not match locked geometry")
    if contract.get("attention_mask") is not None:
        raise ValueError("production packing must not emit an attention mask")
    if contract.get("padding_rows") != 0:
        raise ValueError("production packing must not emit padding rows")


def deterministic_input_receipt(video: Any, audio: Any) -> dict[str, Any]:
    return {
        "rng_implementation": RNG_METHOD,
        "seed": CANONICAL_SEED,
        "draw_order": ["video_native", "audio_native"],
        "fingerprint_method": FINGERPRINT_METHOD,
        "video": shape_dtype(video, logical_dtype="float32") | {"fingerprint": array_fingerprint(video, logical_dtype="float32")},
        "audio": shape_dtype(audio, logical_dtype="float32") | {"fingerprint": array_fingerprint(audio, logical_dtype="float32")},
    }


def packed_state_fingerprint(video: Any, audio: Any) -> str:
    descriptor = json.dumps(
        {
            "video": {"shape": _shape(video), "dtype": _dtype_name(getattr(video, "dtype", "float32"))},
            "audio": {"shape": _shape(audio), "dtype": _dtype_name(getattr(audio, "dtype", "float32"))},
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = descriptor + b"\0" + _as_float32_numpy(video).tobytes(order="C") + _as_float32_numpy(audio).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _linspace_1_to_0(num_points: int) -> np.ndarray:
    """The MLX-free mirror of the repository scheduler's float32-compatible base grid."""
    if num_points < 2:
        raise ValueError("a schedule requires at least two sigma points")
    start, end = 1.0, 0.0
    step = float(np.float32((end - start) / np.float32(num_points - 1)))
    half = num_points // 2
    indices = np.arange(num_points, dtype=np.float64)
    result = np.empty(num_points, dtype=np.float64)
    result[:half] = start + step * indices[:half]
    result[half:] = end - step * (num_points - 1 - indices[half:])
    return result.astype(np.float32)


def _sigma_to_timestep(sigma: float) -> float:
    return float(np.float32(1.0) - np.float32(sigma))


class _ContractScheduler:
    """Small NumPy scheduler used only by MLX-free tests and contract helpers."""

    def __init__(self, *, shift: float):
        self.shift = float(shift)
        self.sigmas: np.ndarray | None = None
        self.timesteps: np.ndarray | None = None
        self.num_inference_steps: int | None = None
        self.step_index: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> None:
        base = _linspace_1_to_0(num_inference_steps)
        shift = np.float32(self.shift)
        shifted = (shift * base) / (np.float32(1.0) + np.float32(self.shift - 1.0) * base)
        values: list[float] = []
        for value in shifted.tolist():
            if not values or value != values[-1]:
                values.append(float(value))
        self.sigmas = np.asarray(values, dtype=np.float32)
        self.timesteps = np.asarray([1.0 - value for value in values[:-1]], dtype=np.float32)
        self.num_inference_steps = len(values) - 1
        self.step_index = None

    def transition(self, step_index: int) -> dict[str, Any]:
        if self.sigmas is None or self.timesteps is None or self.num_inference_steps is None:
            raise ValueError("contract scheduler has no schedule")
        if step_index < 0 or step_index >= self.num_inference_steps:
            raise IndexError(step_index)
        return {
            "step_index": step_index,
            "current_timestep": float(self.timesteps[step_index]),
            "next_timestep": float(1.0 - self.sigmas[step_index + 1]),
            "current_sigma": float(self.sigmas[step_index]),
            "next_sigma": float(self.sigmas[step_index + 1]),
        }


class SchedulePlan:
    def __init__(
        self,
        requested_sigma_points: int,
        video_shift: float,
        audio_shift: float,
        base_sigma_grid: tuple[float, ...],
        video_sigmas: tuple[float, ...],
        audio_sigmas: tuple[float, ...],
        video_timesteps: tuple[float, ...],
        audio_timesteps: tuple[float, ...],
        transitions: tuple[dict[str, Any], ...],
        video_scheduler: Any = None,
        audio_scheduler: Any = None,
    ) -> None:
        self.requested_sigma_points = requested_sigma_points
        self.video_shift = video_shift
        self.audio_shift = audio_shift
        self.base_sigma_grid = base_sigma_grid
        self.video_sigmas = video_sigmas
        self.audio_sigmas = audio_sigmas
        self.video_timesteps = video_timesteps
        self.audio_timesteps = audio_timesteps
        self.transitions = transitions
        self.video_scheduler = video_scheduler
        self.audio_scheduler = audio_scheduler

    def receipt(self) -> dict[str, Any]:
        return {
            "requested_sigma_points": self.requested_sigma_points,
            "effective_video_sigma_points": len(self.video_sigmas),
            "effective_audio_sigma_points": len(self.audio_sigmas),
            "denoising_transitions": len(self.transitions),
            "transformer_forwards": EXPECTED_TRANSFORMER_FORWARDS,
            "video_shift": self.video_shift,
            "audio_shift": self.audio_shift,
            "base_sigma_grid": list(self.base_sigma_grid),
            "video_shifted_sigma_grid": list(self.video_sigmas),
            "audio_shifted_sigma_grid": list(self.audio_sigmas),
            "transitions": list(self.transitions),
            "scheduler_implementation": "minimax_h3_mlx.scheduler.MiniMaxH3Scheduler",
            "terminal_zero_required": True,
        }


def _schedule_values(scheduler: Any) -> tuple[list[float], list[float]]:
    sigmas = getattr(scheduler, "sigmas", None)
    timesteps = getattr(scheduler, "timesteps", None)
    if sigmas is None or timesteps is None:
        raise ValueError("scheduler must expose sigmas and timesteps after set_timesteps")
    sigma_values = [float(value) for value in (sigmas.tolist() if hasattr(sigmas, "tolist") else sigmas)]
    timestep_values = [float(value) for value in (timesteps.tolist() if hasattr(timesteps, "tolist") else timesteps)]
    return sigma_values, timestep_values


def schedule_plan_from_schedulers(video_scheduler: Any, audio_scheduler: Any, *, requested_points: int = REQUESTED_SIGMA_POINTS) -> SchedulePlan:
    video_sigmas, video_timesteps = _schedule_values(video_scheduler)
    audio_sigmas, audio_timesteps = _schedule_values(audio_scheduler)
    transitions: list[dict[str, Any]] = []
    if len(video_sigmas) != len(audio_sigmas):
        raise ValueError("video and audio effective schedule lengths differ")
    if len(video_timesteps) != len(video_sigmas) - 1 or len(audio_timesteps) != len(audio_sigmas) - 1:
        raise ValueError("scheduler timestep and sigma lengths do not form adjacent transitions")
    for index in range(len(video_sigmas) - 1):
        transitions.append(
            {
                "step_index": index,
                "video_current_sigma": video_sigmas[index],
                "video_next_sigma": video_sigmas[index + 1],
                "audio_current_sigma": audio_sigmas[index],
                "audio_next_sigma": audio_sigmas[index + 1],
                "video_current_timestep": video_timesteps[index],
                "video_next_timestep": _sigma_to_timestep(video_sigmas[index + 1]),
                "audio_current_timestep": audio_timesteps[index],
                "audio_next_timestep": _sigma_to_timestep(audio_sigmas[index + 1]),
            }
        )
    plan = SchedulePlan(
        requested_sigma_points=requested_points,
        video_shift=float(video_scheduler.shift),
        audio_shift=float(audio_scheduler.shift),
        base_sigma_grid=tuple(float(value) for value in _linspace_1_to_0(requested_points).tolist()),
        video_sigmas=tuple(video_sigmas),
        audio_sigmas=tuple(audio_sigmas),
        video_timesteps=tuple(video_timesteps),
        audio_timesteps=tuple(audio_timesteps),
        transitions=tuple(transitions),
        video_scheduler=video_scheduler,
        audio_scheduler=audio_scheduler,
    )
    validate_schedule_contract(plan)
    return plan


def build_full_schedule(
    *,
    scheduler_factory: Callable[..., Any] | None = None,
    requested_points: int = REQUESTED_SIGMA_POINTS,
) -> SchedulePlan:
    """Build both modality schedules through the supplied repository scheduler factory.

    With no factory this uses the NumPy contract scheduler so import-time and unit-test paths stay
    MLX-free.  The real derived worker passes ``minimax_h3_mlx.scheduler.MiniMaxH3Scheduler``.
    """
    factory = scheduler_factory or _ContractScheduler
    video_scheduler = factory(shift=VIDEO_SHIFT)
    audio_scheduler = factory(shift=AUDIO_SHIFT)
    video_scheduler.set_timesteps(requested_points)
    audio_scheduler.set_timesteps(requested_points)
    return schedule_plan_from_schedulers(video_scheduler, audio_scheduler, requested_points=requested_points)


def validate_schedule_contract(plan: SchedulePlan | Mapping[str, Any]) -> None:
    serialized = not isinstance(plan, SchedulePlan)
    if isinstance(plan, SchedulePlan):
        requested = plan.requested_sigma_points
        transformer_forwards = EXPECTED_TRANSFORMER_FORWARDS
        base_sigmas = list(plan.base_sigma_grid)
        video_sigmas = list(plan.video_sigmas)
        audio_sigmas = list(plan.audio_sigmas)
        video_timesteps = list(plan.video_timesteps)
        audio_timesteps = list(plan.audio_timesteps)
        transitions = list(plan.transitions)
        video_shift = plan.video_shift
        audio_shift = plan.audio_shift
    else:
        requested = plan.get("requested_sigma_points")
        video_shift = plan.get("video_shift")
        audio_shift = plan.get("audio_shift")
        video_count = plan.get("effective_video_sigma_points")
        audio_count = plan.get("effective_audio_sigma_points")
        transition_count = plan.get("denoising_transitions")
        transformer_forwards = plan.get("transformer_forwards")
        if (requested, video_count, audio_count, transition_count, transformer_forwards) != (
            REQUESTED_SIGMA_POINTS,
            EXPECTED_VIDEO_SIGMA_POINTS,
            EXPECTED_AUDIO_SIGMA_POINTS,
            EXPECTED_DENOISING_TRANSITIONS,
            EXPECTED_TRANSFORMER_FORWARDS,
            ):
            raise ValueError("schedule counts do not match the locked 16/16/16/15/15 contract")
        base_sigmas = list(plan.get("base_sigma_grid", ()))
        video_sigmas = list(plan.get("video_shifted_sigma_grid", ()))
        audio_sigmas = list(plan.get("audio_shifted_sigma_grid", ()))
        video_timesteps = []
        audio_timesteps = []
        transitions = list(plan.get("transitions", ()))
        if len(base_sigmas) != REQUESTED_SIGMA_POINTS:
            raise ValueError("serialized base schedule must contain exactly 16 points")
        if len(video_sigmas) != EXPECTED_VIDEO_SIGMA_POINTS or len(audio_sigmas) != EXPECTED_AUDIO_SIGMA_POINTS:
            raise ValueError("serialized effective schedules must each contain 16 points")
        if plan.get("video_shift") != VIDEO_SHIFT or plan.get("audio_shift") != AUDIO_SHIFT:
            raise ValueError("serialized scheduler shifts differ from the locked contract")
    if requested != REQUESTED_SIGMA_POINTS:
        raise ValueError(f"requested sigma point count must be {REQUESTED_SIGMA_POINTS}")
    if transformer_forwards != EXPECTED_TRANSFORMER_FORWARDS:
        raise ValueError(f"serialized schedule must require exactly {EXPECTED_TRANSFORMER_FORWARDS} transformer forwards")
    expected_base = _linspace_1_to_0(REQUESTED_SIGMA_POINTS).tolist()
    if len(base_sigmas) != REQUESTED_SIGMA_POINTS or not np.array_equal(
        np.asarray(base_sigmas, dtype=np.float32), np.asarray(expected_base, dtype=np.float32)
    ):
        raise ValueError("base sigma grid does not match the canonical 16-point grid")
    if len(video_sigmas) != EXPECTED_VIDEO_SIGMA_POINTS or len(audio_sigmas) != EXPECTED_AUDIO_SIGMA_POINTS:
        raise ValueError("effective video and audio schedules must each contain 16 points")
    if len(transitions) != EXPECTED_DENOISING_TRANSITIONS:
        raise ValueError("full schedule must contain exactly 15 transitions")
    for label, values in (("video", video_sigmas), ("audio", audio_sigmas)):
        if values[0] != 1.0:
            raise ValueError(f"{label} schedule must begin at exactly 1.0")
        if values[-1] != 0.0:
            raise ValueError(f"{label} schedule is missing terminal zero")
        if any(not (right < left) for left, right in zip(values, values[1:])):
            raise ValueError(f"{label} schedule is not strictly decreasing")
    if not serialized:
        if len(video_timesteps) != EXPECTED_DENOISING_TRANSITIONS or len(audio_timesteps) != EXPECTED_DENOISING_TRANSITIONS:
            raise ValueError("video and audio timestep arrays must each contain 15 transitions")
    if (float(video_shift), float(audio_shift)) != (VIDEO_SHIFT, AUDIO_SHIFT):
        raise ValueError("scheduler shifts differ from the locked video/audio shifts")
    expected_indices = list(range(EXPECTED_DENOISING_TRANSITIONS))
    observed_indices = [int(item.get("step_index", -1)) for item in transitions]
    if observed_indices != expected_indices:
        raise ValueError(f"transition ordering must be exactly ordered {expected_indices}, got {observed_indices}")
    required_transition_fields = {
        "step_index", "video_current_sigma", "video_next_sigma", "audio_current_sigma", "audio_next_sigma",
        "video_current_timestep", "video_next_timestep", "audio_current_timestep", "audio_next_timestep",
    }
    for index, transition in enumerate(transitions):
        if not isinstance(transition, Mapping) or not required_transition_fields.issubset(transition):
            raise ValueError(f"transition {index} is missing serialized scheduler fields")
        expected_values = {
            "video_current_sigma": video_sigmas[index],
            "video_next_sigma": video_sigmas[index + 1],
            "audio_current_sigma": audio_sigmas[index],
            "audio_next_sigma": audio_sigmas[index + 1],
            "video_current_timestep": _sigma_to_timestep(video_sigmas[index]),
            "video_next_timestep": _sigma_to_timestep(video_sigmas[index + 1]),
            "audio_current_timestep": _sigma_to_timestep(audio_sigmas[index]),
            "audio_next_timestep": _sigma_to_timestep(audio_sigmas[index + 1]),
        }
        for field, expected in expected_values.items():
            if float(transition[field]) != float(expected):
                raise ValueError(f"transition {index} {field} does not match its serialized sigma grid")
        if not serialized:
            if float(video_timesteps[index]) != float(transition["video_current_timestep"]):
                raise ValueError(f"video timestep {index} does not match the scheduler transition")
            if float(audio_timesteps[index]) != float(transition["audio_current_timestep"]):
                raise ValueError(f"audio timestep {index} does not match the scheduler transition")


class ForwardExecution:
    def __init__(
        self,
        video_prediction: Any,
        audio_prediction: Any,
        transformer_forwards: int = 1,
        timing_seconds: float | None = None,
        materialize_callback: Callable[[], None] | None = None,
    ) -> None:
        self.video_prediction = video_prediction
        self.audio_prediction = audio_prediction
        self.transformer_forwards = transformer_forwards
        self.timing_seconds = timing_seconds
        self._materialize_callback = materialize_callback
        self.materialized = False
        self.materialization_count = 0

    def materialize_predictions(self) -> None:
        if self.materialized:
            return
        if self._materialize_callback is not None:
            self._materialize_callback()
        else:
            materialize_predictions(self.video_prediction, self.audio_prediction)
        self.materialization_count += 1
        self.materialized = True


class FullScheduleResult:
    def __init__(
        self,
        final_video_latent: Any,
        final_audio_latent: Any,
        final_native_video: Any,
        final_native_audio: Any,
        transitions: tuple[dict[str, Any], ...],
        transformer_forwards: int,
        video_scheduler_updates: int,
        audio_scheduler_updates: int,
        lifecycle: dict[str, Any],
        cache_attribution: dict[str, Any],
        memory_telemetry: dict[str, Any],
        timing_telemetry: dict[str, Any],
    ) -> None:
        self.final_video_latent = final_video_latent
        self.final_audio_latent = final_audio_latent
        self.final_native_video = final_native_video
        self.final_native_audio = final_native_audio
        self.transitions = transitions
        self.transformer_forwards = transformer_forwards
        self.video_scheduler_updates = video_scheduler_updates
        self.audio_scheduler_updates = audio_scheduler_updates
        self.lifecycle = lifecycle
        self.cache_attribution = cache_attribution
        self.memory_telemetry = memory_telemetry
        self.timing_telemetry = timing_telemetry

    def receipt(self) -> dict[str, Any]:
        return {
            "completed_transition_count": len(self.transitions),
            "transformer_forward_count": self.transformer_forwards,
            "scheduler_update_counts": {
                "video": self.video_scheduler_updates,
                "audio": self.audio_scheduler_updates,
            },
            "transitions": list(self.transitions),
            "streamed_adaln_lifecycle": dict(self.lifecycle),
            "cache_attribution": dict(self.cache_attribution),
            "memory_telemetry": dict(self.memory_telemetry),
            "timing_telemetry": dict(self.timing_telemetry),
            "final_packed_video_fingerprint": array_fingerprint(self.final_video_latent, logical_dtype="bfloat16"),
            "final_packed_audio_fingerprint": array_fingerprint(self.final_audio_latent, logical_dtype="bfloat16"),
            "final_native_video_fingerprint": array_fingerprint(self.final_native_video, logical_dtype="bfloat16"),
            "final_native_audio_fingerprint": array_fingerprint(self.final_native_audio, logical_dtype="bfloat16"),
            "packed_final_state_fingerprint": packed_state_fingerprint(self.final_video_latent, self.final_audio_latent),
        }


def _copy(value: Any) -> Any:
    return copy_runtime_array(value)


def _call_provider(provider: Any, step_index: int, transition: Mapping[str, Any]) -> Any:
    if callable(provider):
        try:
            return provider(step_index, transition)
        except TypeError as exc:
            try:
                inspect.signature(provider).bind(step_index, transition)
            except (TypeError, ValueError):
                return provider(step_index)
            raise exc
    if isinstance(provider, Mapping):
        return provider[step_index]
    if isinstance(provider, Sequence) and not isinstance(provider, (str, bytes, bytearray)):
        return provider[step_index]
    raise TypeError("timestep provider must be callable, a mapping, or a sequence")


def _resolve_timestep(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        if "timestep" not in value or "timestep_indices" not in value:
            raise ValueError("timestep provider must include timestep and timestep_indices")
        return value["timestep"], value["timestep_indices"]
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    raise ValueError("timestep provider must return (timestep, timestep_indices)")


def _cache_for_transition(provider: Any, step_index: int, timestep: Any, transition: Mapping[str, Any]) -> Any:
    method = getattr(provider, "cache_for_step", None)
    if not callable(method):
        raise ValueError("streamed cache provider must expose cache_for_step")
    try:
        return method(step_index, timestep, transition)
    except TypeError as exc:
        try:
            inspect.signature(method).bind(step_index, timestep, transition)
        except (TypeError, ValueError):
            return method(step_index, timestep)
        raise exc


def _release_cache_for_transition(
    provider: Any,
    step_index: int,
    cache: Any,
    *,
    forward_materialized: bool,
) -> None:
    method = getattr(provider, "release_step", None)
    if not callable(method):
        raise ValueError("streamed cache provider must expose release_step")
    try:
        inspect.signature(method).bind(step_index, cache, forward_materialized=forward_materialized)
    except (TypeError, ValueError):
        method(step_index, cache)
    else:
        method(step_index, cache, forward_materialized=forward_materialized)


def _snapshot(snapshot: Callable[[], Mapping[str, Any]] | None) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    try:
        return dict(_json_safe(snapshot()))
    except BaseException as exc:
        return {"status": "unavailable", "error": error_receipt(exc)}


def _normalize_forward(value: Any) -> ForwardExecution:
    if isinstance(value, ForwardExecution):
        return value
    if isinstance(value, Mapping):
        if "video_prediction" not in value or "audio_prediction" not in value:
            raise ValueError("forward result is missing video_prediction or audio_prediction")
        return ForwardExecution(
            value["video_prediction"],
            value["audio_prediction"],
            int(value.get("transformer_forwards", 1)),
            value.get("timing_seconds"),
        )
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return ForwardExecution(value[0], value[1])
    raise ValueError("forward runner must return ForwardExecution or two predictions")


def _provider_lifecycle(provider: Any, transformer_forwards: int, video_updates: int, audio_updates: int) -> dict[str, Any]:
    if provider is None:
        result = {key: 0 for key in EXPECTED_LIFECYCLE_TOTALS}
        result.update(
            {
                "maximum_simultaneous_sidecars": 0,
                "transformer_forwards": transformer_forwards,
                "video_scheduler_updates": video_updates,
                "audio_scheduler_updates": audio_updates,
            }
        )
        return result
    aggregate = getattr(provider, "aggregate", None)
    if not callable(aggregate):
        raise ValueError("streamed cache provider must expose aggregate()")
    result = dict(aggregate())
    result["transformer_forwards"] = transformer_forwards
    result["video_scheduler_updates"] = video_updates
    result["audio_scheduler_updates"] = audio_updates
    return result


def _provider_cache_attribution(provider: Any) -> dict[str, Any]:
    if provider is None:
        return build_cache_attribution_aggregate([])
    method = getattr(provider, "cache_attribution", None)
    if not callable(method):
        raise ValueError("streamed cache provider must expose cache_attribution()")
    return dict(method())


def _state_for_failure(
    transitions: list[dict[str, Any]],
    *,
    transformer_forwards: int,
    video_updates: int,
    audio_updates: int,
    provider: Any,
    memory: Mapping[str, Any],
    timings: Mapping[str, Any],
    primary: BaseException,
    cleanup: BaseException | None,
    cleanup_attempted: bool,
    cleanup_succeeded: bool,
) -> dict[str, Any]:
    return {
        "completed_transition_count": sum(item.get("status") == "success" for item in transitions),
        "transformer_forward_count": transformer_forwards,
        "scheduler_update_counts": {"video": video_updates, "audio": audio_updates},
        "transitions": list(transitions),
        "streamed_adaln_lifecycle": _provider_lifecycle(provider, transformer_forwards, video_updates, audio_updates),
        "cache_attribution": _provider_cache_attribution(provider),
        "memory_telemetry": dict(memory),
        "timing_telemetry": dict(timings),
        "failure": failure_fields(
            primary,
            cleanup,
            cleanup_attempted=cleanup_attempted,
            cleanup_succeeded=cleanup_succeeded,
        ),
    }


def run_full_schedule(
    transformer: Any,
    scheduler: Any,
    schedule: SchedulePlan,
    *,
    initial_video_latent: Any,
    initial_audio_latent: Any,
    timestep_provider: Any,
    text_embedding: Any = None,
    packed_inputs: Mapping[str, Any] | None = None,
    cache_provider: Any = None,
    forward_runner: Callable[..., Any] | None = None,
    native_latent_provider: Callable[[Any, Any], tuple[Any, Any]] | None = None,
    memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    expected_video_shape: tuple[int, ...] | None = None,
    expected_audio_shape: tuple[int, ...] | None = None,
    expected_text_shape: tuple[int, ...] | None = None,
    expected_video_dtype: str | None = None,
    expected_audio_dtype: str | None = None,
    expected_text_dtype: str | None = None,
    expected_prediction_dtype: str = CANONICAL_PREDICTION_DTYPE,
) -> FullScheduleResult:
    """Run exactly the locked adjacent transitions with cache release before scheduler update."""
    validate_schedule_contract(schedule)
    packed = dict(packed_inputs or {})
    required_packed = {"token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"}
    if set(packed) != required_packed:
        raise ValueError(
            f"full-schedule forward requires the exact packed input set: missing={sorted(required_packed - set(packed))}, "
            f"unexpected={sorted(set(packed) - required_packed)}"
        )
    video_latent = _copy(initial_video_latent)
    audio_latent = _copy(initial_audio_latent)
    expected_video_shape = expected_video_shape or tuple(int(item) for item in video_latent.shape)
    expected_audio_shape = expected_audio_shape or tuple(int(item) for item in audio_latent.shape)
    if expected_text_shape is None and text_embedding is not None:
        expected_text_shape = tuple(int(item) for item in text_embedding.shape)
    expected_video_dtype = expected_video_dtype or _dtype_name(getattr(video_latent, "dtype", None))
    expected_audio_dtype = expected_audio_dtype or _dtype_name(getattr(audio_latent, "dtype", None))
    expected_text_dtype = expected_text_dtype or (
        _dtype_name(getattr(text_embedding, "dtype", None)) if text_embedding is not None else "bfloat16"
    )
    transitions: list[dict[str, Any]] = []
    transformer_forwards = 0
    video_updates = 0
    audio_updates = 0
    memory: dict[str, Any] = {"derived_worker": {"before_denoising": _snapshot(memory_snapshot)}}
    timings: dict[str, Any] = {"transitions": []}
    forward = forward_runner
    native = native_latent_provider or (lambda video, audio: (video, audio))

    for transition in schedule.transitions:
        step_index = int(transition["step_index"])
        if step_index != len(transitions):
            raise DenoisingFailure(
                ValueError("transition index was skipped, repeated, or reordered"),
                state={"transitions": transitions},
                cleanup_attempted=False,
            )
        record: dict[str, Any] = {
            "step_index": step_index,
            "status": "incomplete",
            "video_current_sigma": transition["video_current_sigma"],
            "video_next_sigma": transition["video_next_sigma"],
            "audio_current_sigma": transition["audio_current_sigma"],
            "audio_next_sigma": transition["audio_next_sigma"],
            "video_current_timestep": transition["video_current_timestep"],
            "video_next_timestep": transition["video_next_timestep"],
            "audio_current_timestep": transition["audio_current_timestep"],
            "audio_next_timestep": transition["audio_next_timestep"],
            "transformer_forwards": 0,
            "video_scheduler_updates": 0,
            "audio_scheduler_updates": 0,
            "memory": {},
            "timings": {},
        }
        started = time.perf_counter()
        record["memory"]["before_cache"] = _snapshot(memory_snapshot)
        cache = None
        cache_acquired = False
        forward_success = False
        forward_materialized = False
        primary: BaseException | None = None
        cleanup: BaseException | None = None
        cleanup_attempted = False
        cleanup_succeeded = False
        timestep = timestep_indices = None
        execution: ForwardExecution | None = None
        execution_holder: dict[str, ForwardExecution] = {}
        try:
            timestep, timestep_indices = _resolve_timestep(_call_provider(timestep_provider, step_index, transition))
            record["timestep"] = _json_safe(timestep)
            record["timestep_indices"] = _json_safe(timestep_indices)
            cache_started = time.perf_counter()
            if cache_provider is not None:
                cache = _cache_for_transition(cache_provider, step_index, timestep, transition)
                cache_acquired = True
            cache_wall_seconds = time.perf_counter() - cache_started
            record["timings"]["streamed_cache_construction_seconds"] = cache_wall_seconds
            if cache_provider is not None:
                note_wall = getattr(cache_provider, "note_cache_construction_wall", None)
                if not callable(note_wall):
                    raise ValueError("streamed cache provider must expose note_cache_construction_wall()")
                note_wall(step_index, cache_wall_seconds)
            record["memory"]["after_cache"] = _snapshot(memory_snapshot)

            def forward_adapter(
                prepared_video: Any,
                prepared_audio: Any,
                text: Any,
                current_timestep: Any,
                current_indices: Any,
                token_tags: Any,
                position_ids: Any,
                video_indices: Any,
                audio_indices: Any,
                text_indices: Any,
                **kwargs: Any,
            ) -> tuple[Any, Any]:
                if forward is None:
                    raw = transformer(
                        prepared_video,
                        prepared_audio,
                        text,
                        current_timestep,
                        current_indices,
                        token_tags,
                        position_ids,
                        video_indices,
                        audio_indices,
                        text_indices,
                        **kwargs,
                    )
                else:
                    raw = forward(
                        step_index,
                        transition,
                        prepared_video,
                        prepared_audio,
                        current_timestep,
                        current_indices,
                        kwargs.get("modulation_cache"),
                        packed,
                    )
                resolved = _normalize_forward(raw)
                execution_holder["execution"] = resolved
                return resolved.video_prediction, resolved.audio_prediction

            def materialize_adapter(_video_prediction: Any, _audio_prediction: Any) -> None:
                resolved = execution_holder.get("execution")
                if resolved is None:
                    raise ValueError("forward materialization was requested before the forward result existed")
                resolved.materialize_predictions()

            forward_started = time.perf_counter()
            validated = validated_transformer_forward(
                forward_adapter,
                scheduler,
                video_latent=video_latent,
                audio_latent=audio_latent,
                text_embedding=text_embedding,
                timestep=timestep,
                timestep_indices=timestep_indices,
                token_tags=packed["token_tags"],
                position_ids=packed["position_ids"],
                video_indices=packed["video_indices"],
                audio_indices=packed["audio_indices"],
                text_indices=packed["text_indices"],
                step_index=step_index,
                modulation_cache=cache,
                expected_video_shape=expected_video_shape,
                expected_audio_shape=expected_audio_shape,
                expected_text_shape=expected_text_shape,
                expected_video_dtype=expected_video_dtype,
                expected_audio_dtype=expected_audio_dtype,
                expected_text_dtype=expected_text_dtype,
                expected_prediction_dtype=expected_prediction_dtype,
                materialize=materialize_adapter,
            )
            execution = execution_holder.get("execution")
            if execution is None:
                raise ValueError("validated forward completed without a forward execution record")
            if execution.transformer_forwards != 1:
                raise ValueError("each transition must perform exactly one transformer forward")
            if not execution.materialized:
                raise ValueError("transformer predictions were not materialized before cache release")
            record["timings"]["transformer_forward_seconds"] = time.perf_counter() - forward_started
            record["timings"]["materialization_completed_before_release"] = True
            record["prediction_materialized"] = True
            record["prepared_video_shape"] = [int(item) for item in validated.prepared_video.shape]
            record["prepared_audio_shape"] = [int(item) for item in validated.prepared_audio.shape]
            transformer_forwards += 1
            record["transformer_forwards"] = 1
            forward_success = True
            forward_materialized = True
            record["memory"]["after_forward"] = _snapshot(memory_snapshot)
        except BaseException as exc:
            primary = exc
        finally:
            if cache_provider is not None and cache_acquired:
                cleanup_attempted = True
                release_started = time.perf_counter()
                try:
                    if forward_materialized:
                        _release_cache_for_transition(
                            cache_provider,
                            step_index,
                            cache,
                            forward_materialized=True,
                        )
                    else:
                        failure_cleanup = getattr(cache_provider, "cleanup_failed_step", None)
                        if not callable(failure_cleanup):
                            raise RuntimeError("cache release is not permitted before prediction materialization and failure cleanup is unavailable")
                        failure_cleanup(step_index, cache)
                    cleanup_succeeded = True
                    record["timings"]["streamed_cache_release_seconds"] = time.perf_counter() - release_started
                except BaseException as exc:
                    cleanup = exc
                    cleanup_succeeded = False
                    record["timings"]["streamed_cache_release_seconds"] = time.perf_counter() - release_started
                finally:
                    cache = None
                    note_drop = getattr(cache_provider, "note_local_cache_reference_dropped", None)
                    if callable(note_drop):
                        try:
                            note_drop(step_index, _snapshot(memory_snapshot), forward_materialized=forward_materialized)
                        except BaseException as exc:
                            if cleanup is None:
                                cleanup = exc
                                cleanup_succeeded = False
                            else:
                                record.setdefault("additional_cleanup_errors", []).append(error_receipt(exc))
                    record["cache_reference_dropped_before_scheduler"] = True
                    record["memory"]["after_cache_reference_drop"] = _snapshot(memory_snapshot)
            elif primary is not None:
                cleanup_attempted = bool(getattr(primary, "cache_cleanup_attempted", False))
                cleanup_succeeded = bool(getattr(primary, "cache_cleanup_succeeded", False))
                cleanup = getattr(primary, "cache_cleanup_error", None)

        if primary is not None or cleanup is not None:
            if primary is None:
                primary = cleanup
                cleanup = None
            record["failure"] = failure_fields(
                primary,
                cleanup,
                cleanup_attempted=cleanup_attempted,
                cleanup_succeeded=cleanup_succeeded,
            )
            transitions.append(record)
            state = _state_for_failure(
                transitions,
                transformer_forwards=transformer_forwards,
                video_updates=video_updates,
                audio_updates=audio_updates,
                provider=cache_provider,
                memory=memory,
                timings=timings,
                primary=primary,
                cleanup=cleanup,
                cleanup_attempted=cleanup_attempted,
                cleanup_succeeded=cleanup_succeeded,
            )
            raise DenoisingFailure(primary, cleanup=cleanup, state=state, cleanup_attempted=cleanup_attempted) from primary

        if not forward_success:
            raise AssertionError("successful transition did not produce one forward")
        scheduler_started = time.perf_counter()
        try:
            step = getattr(scheduler, "step", None)
            if not callable(step):
                raise ValueError("scheduler must expose step")
            updated = step(
                execution.video_prediction,
                execution.audio_prediction,
                _copy(video_latent),
                _copy(audio_latent),
                step_index,
            )
            if not isinstance(updated, (tuple, list)) or len(updated) != 2:
                raise ValueError("scheduler step must return packed video and audio rows")
            updated_video, updated_audio = updated
            validate_updated_latents(
                updated_video,
                updated_audio,
                expected_video_shape=expected_video_shape,
                expected_audio_shape=expected_audio_shape,
                expected_video_dtype=expected_video_dtype,
                expected_audio_dtype=expected_audio_dtype,
            )
            video_updates += 1
            audio_updates += 1
            record["video_scheduler_updates"] = 1
            record["audio_scheduler_updates"] = 1
            record["timings"]["scheduler_update_seconds"] = time.perf_counter() - scheduler_started
            record["memory"]["after_scheduler_update"] = _snapshot(memory_snapshot)
        except BaseException as exc:
            record["failure"] = failure_fields(exc, None, cleanup_attempted=True, cleanup_succeeded=True)
            transitions.append(record)
            state = _state_for_failure(
                transitions,
                transformer_forwards=transformer_forwards,
                video_updates=video_updates,
                audio_updates=audio_updates,
                provider=cache_provider,
                memory=memory,
                timings=timings,
                primary=exc,
                cleanup=None,
                cleanup_attempted=True,
                cleanup_succeeded=True,
            )
            raise DenoisingFailure(exc, state=state, cleanup_attempted=True) from exc

        native_video, native_audio = native(updated_video, updated_audio)
        record["updated_packed_video_fingerprint"] = array_fingerprint(updated_video, logical_dtype="bfloat16")
        record["updated_packed_audio_fingerprint"] = array_fingerprint(updated_audio, logical_dtype="bfloat16")
        record["updated_native_video_fingerprint"] = array_fingerprint(native_video, logical_dtype="bfloat16")
        record["updated_native_audio_fingerprint"] = array_fingerprint(native_audio, logical_dtype="bfloat16")
        record["packed_state_fingerprint"] = packed_state_fingerprint(updated_video, updated_audio)
        record["status"] = "success"
        record["timings"]["total_transition_seconds"] = time.perf_counter() - started
        transitions.append(record)
        timings["transitions"].append(dict(record["timings"]))
        video_latent, audio_latent = updated_video, updated_audio

    if len(transitions) != EXPECTED_DENOISING_TRANSITIONS:
        raise DenoisingFailure(
            ValueError("final success requires all fifteen transitions"),
            state={"transitions": transitions},
            cleanup_attempted=False,
        )
    lifecycle = _provider_lifecycle(cache_provider, transformer_forwards, video_updates, audio_updates)
    validate_lifecycle_totals(lifecycle)
    cache_attribution = _provider_cache_attribution(cache_provider)
    validate_cache_attribution(cache_attribution)
    final_native_video, final_native_audio = native(video_latent, audio_latent)
    memory["derived_worker"]["after_denoising"] = _snapshot(memory_snapshot)
    return FullScheduleResult(
        video_latent,
        audio_latent,
        final_native_video,
        final_native_audio,
        tuple(transitions),
        transformer_forwards,
        video_updates,
        audio_updates,
        lifecycle,
        cache_attribution,
        memory,
        timings,
    )


class JsonlEventWriter:
    """Append-only event sink used for the detailed per-block stream."""

    def __init__(self, path: Path):
        self.path = path
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, details: Mapping[str, Any] | None = None) -> None:
        record = {"event": event, **dict(details or {})}
        with self.path.open("a") as handle:
            handle.write(json.dumps(_json_safe(record), sort_keys=True) + "\n")
        self.count += 1

    def summary(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"path": str(self.path), "record_count": 0, "sha256": None}
        try:
            summary = validate_event_stream(self.path)
        except (OSError, ValueError):
            summary = event_file_summary(self.path)
        return {
            "path": summary["event_file_path"],
            "record_count": summary["event_file_record_count"],
            "sha256": summary["event_file_sha256"],
            "total_event_records": summary["total_event_records"],
            "cache_session_count": summary["cache_session_count"],
            "sidecar_open_event_count": summary["sidecar_open_event_count"],
            "sidecar_release_event_count": summary["sidecar_release_event_count"],
            "attribution_block_event_count": summary["attribution_block_event_count"],
            "attribution_session_event_count": summary["attribution_session_event_count"],
            "validated_block_pairs": summary["validated_block_pairs"],
        }


def _bounded_release_cache(cache: Any) -> str:
    """Release a cache through one observable bounded path without purging the allocator."""
    if cache is None:
        raise RuntimeError("cannot release a missing modulation cache")
    release = getattr(cache, "release", None)
    if callable(release):
        release()
        tables = getattr(cache, "tables", None)
        if isinstance(tables, list) and tables:
            raise RuntimeError("cache.release returned while retained modulation tables were still live")
        return "cache.release"
    tables = getattr(cache, "tables", None)
    if not isinstance(tables, list):
        raise RuntimeError("modulation cache has no bounded release path")
    tables.clear()
    try:
        setattr(cache, "tables", [])
        if hasattr(cache, "timesteps"):
            setattr(cache, "timesteps", None)
    except BaseException as exc:
        raise RuntimeError("modulation cache references could not be dropped") from exc
    if getattr(cache, "tables", None):
        raise RuntimeError("modulation cache tables remained live after bounded release")
    return "tables-cleared"


def _timing_seconds(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"cache attribution timing {label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"cache attribution timing {label} is not finite and nonnegative")
    return result


def _nearest_rank_percentile(values: Sequence[float], percentile: float) -> float | None:
    """Return a conservative nearest-rank percentile; omit p95 for fewer than five samples."""
    if not values or (percentile == 0.95 and len(values) < 5):
        return None
    ordered = sorted(float(value) for value in values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _timing_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"sample_count": 0, "mean_seconds": None, "median_seconds": None, "p95_seconds": None}
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median_value = ordered[midpoint]
    else:
        median_value = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "sample_count": len(ordered),
        "mean_seconds": math.fsum(ordered) / len(ordered),
        "median_seconds": median_value,
        "p95_seconds": _nearest_rank_percentile(ordered, 0.95),
    }


def calculate_unattributed_remainder(wall_total_seconds: float, measured_component_seconds: float) -> dict[str, Any]:
    """Preserve the raw wall-minus-components remainder, including negative timer noise."""
    wall = _timing_seconds(wall_total_seconds, "wall_total_seconds")
    components = _timing_seconds(measured_component_seconds, "measured_component_seconds")
    remainder = wall - components
    if remainder < 0.0:
        status = "negative-timer-noise-or-overlapping-boundaries"
        warning = (
            "measured components exceed wall time; the negative remainder is retained for review "
            "and is not clamped"
        )
    else:
        status = "nonnegative"
        warning = None
    return {
        "wall_total_seconds": wall,
        "measured_component_seconds": components,
        "unattributed_remainder_seconds": remainder,
        "unattributed_remainder_status": status,
        "unattributed_remainder_warning": warning,
    }


def _raw_attribution_block(raw: Mapping[str, Any], expected_index: int) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"cache attribution block {expected_index} is not an object")
    if "timings" in raw:
        timing_source = raw.get("timings")
        if not isinstance(timing_source, Mapping):
            raise ValueError(f"cache attribution block {expected_index} timings are not an object")
    else:
        timing_source = raw
    block_index = raw.get("block_index")
    if type(block_index) is not int or block_index != expected_index:
        raise ValueError(f"cache attribution block ordering is not exactly 0..49 at {expected_index}")
    path = raw.get("path")
    filename = raw.get("sidecar_filename") or (Path(str(path)).name if path is not None else None)
    expected_filename = f"block-{expected_index:03d}.safetensors"
    if filename is not None and Path(str(filename)).name != expected_filename:
        raise ValueError(f"cache attribution block {expected_index} sidecar identity is invalid")

    def direct_or_alias(field: str, *aliases: str) -> float | None:
        if field in timing_source:
            return _timing_seconds(timing_source[field], f"block-{expected_index}.{field}")
        for alias in aliases:
            if alias in timing_source:
                return _timing_seconds(timing_source[alias], f"block-{expected_index}.{alias}")
        return None

    sidecar_io = direct_or_alias(
        "sidecar_io_and_reconstruction_seconds",
        "elapsed_sidecar_io_and_reconstruction_seconds",
    )
    projection_compute = direct_or_alias(
        "projection_compute_seconds",
        "elapsed_projection_compute_seconds",
    )
    sidecar_materialization = direct_or_alias(
        "sidecar_materialization_seconds",
        "elapsed_sidecar_materialization_seconds",
    )
    projection_materialization = direct_or_alias(
        "projection_materialization_seconds",
        "elapsed_projection_materialization_seconds",
    )
    modulation_materialization = direct_or_alias(
        "modulation_materialization_seconds",
        "elapsed_modulation_materialization_seconds",
    )
    materialization = direct_or_alias("materialization_evaluation_seconds")
    if materialization is None:
        materialization_parts = (
            sidecar_materialization,
            projection_materialization,
            modulation_materialization,
        )
        if any(part is None for part in materialization_parts):
            raise ValueError(f"cache attribution block {expected_index} is missing materialization boundaries")
        materialization = math.fsum(part for part in materialization_parts if part is not None)
    if sidecar_io is None or projection_compute is None:
        raise ValueError(f"cache attribution block {expected_index} is missing sidecar or projection timing")
    assembly = direct_or_alias(
        "cache_entry_assembly_bookkeeping_seconds",
        "elapsed_cache_entry_assembly_bookkeeping_seconds",
    )
    release = direct_or_alias("release_purge_seconds", "elapsed_release_purge_seconds")
    total = direct_or_alias(
        ATTRIBUTION_TOTAL_FIELD,
        "elapsed_seconds",
    )
    if assembly is None or release is None or total is None:
        raise ValueError(f"cache attribution block {expected_index} is missing assembly, release, or total timing")
    return {
        "block_index": expected_index,
        "sidecar_filename": expected_filename,
        "sidecar_io_and_reconstruction_seconds": sidecar_io,
        "projection_compute_seconds": projection_compute,
        "materialization_evaluation_seconds": materialization,
        "cache_entry_assembly_bookkeeping_seconds": assembly,
        "release_purge_seconds": release,
        ATTRIBUTION_TOTAL_FIELD: total,
    }


def build_cache_session_attribution(
    stats: Any,
    *,
    session_index: int,
    wall_clock_seconds: float | None = None,
    sidecar_opens: int = EXPECTED_BLOCK_COUNT,
    sidecar_releases: int = EXPECTED_BLOCK_COUNT,
    block_events: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Normalize one builder receipt into the v0.5e per-session attribution contract."""
    if type(session_index) is not int or session_index < 0:
        raise ValueError("cache attribution session index must be a nonnegative integer")
    normalized_stats = _json_safe(stats)
    if not isinstance(normalized_stats, Mapping):
        raise ValueError("cache attribution builder statistics are not an object")
    raw_blocks = normalized_stats.get("per_block")
    if not isinstance(raw_blocks, list) or len(raw_blocks) != EXPECTED_BLOCK_COUNT:
        raise ValueError("cache attribution requires exactly 50 per-block timing records")
    blocks = [_raw_attribution_block(item, index) for index, item in enumerate(raw_blocks)]

    if block_events is not None:
        if len(block_events) != EXPECTED_BLOCK_COUNT:
            raise ValueError("cache attribution event evidence does not contain exactly 50 blocks")
        for index, event in enumerate(block_events):
            observed = _raw_attribution_block(event, index)
            expected = blocks[index]
            if observed["sidecar_filename"] != expected["sidecar_filename"]:
                raise ValueError(f"cache attribution event sidecar identity differs at block {index}")
            for field in (*ATTRIBUTION_COMPONENT_FIELDS, ATTRIBUTION_TOTAL_FIELD):
                if not math.isclose(observed[field], expected[field], rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"cache attribution event timing differs from stats at block {index}: {field}")

    if type(sidecar_opens) is not int or sidecar_opens != EXPECTED_BLOCK_COUNT:
        raise ValueError("cache attribution sidecar open count is not 50")
    if type(sidecar_releases) is not int or sidecar_releases != EXPECTED_BLOCK_COUNT:
        raise ValueError("cache attribution sidecar release count is not 50")
    component_totals = {
        field: math.fsum(block[field] for block in blocks)
        for field in ATTRIBUTION_COMPONENT_FIELDS
    }
    raw_overhead = {
        "shared_timestep_embedding_seconds": normalized_stats.get("elapsed_shared_timestep_embedding_seconds"),
        "cache_finalize_materialization_seconds": normalized_stats.get("elapsed_cache_finalize_materialization_seconds"),
    }
    overhead_totals = {
        field: _timing_seconds(value, f"session.{field}")
        for field, value in raw_overhead.items()
    }
    measured_components = math.fsum((*component_totals.values(), *overhead_totals.values()))
    if wall_clock_seconds is None:
        wall_clock_seconds = normalized_stats.get("elapsed_total_seconds")
    if wall_clock_seconds is None:
        raise ValueError("cache attribution wall-clock cache-session timing is unavailable")
    remainder = calculate_unattributed_remainder(wall_clock_seconds, measured_components)
    return {
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "session_index": session_index,
        "blocks_completed": len(blocks),
        "sidecar_opens": sidecar_opens,
        "sidecar_releases": sidecar_releases,
        "sidecar_io_measurement": "combined-with-deserialize-reconstruction",
        "sidecar_io_seconds": None,
        "deserialize_reconstruction_seconds": None,
        "sum_sidecar_io_seconds": None,
        "sum_deserialize_reconstruction_seconds": None,
        "blocks": blocks,
        "component_totals_seconds": component_totals,
        "session_overhead_seconds": overhead_totals,
        "sum_sidecar_io_and_reconstruction_seconds": component_totals["sidecar_io_and_reconstruction_seconds"],
        "sum_sidecar_io_and_deserialize_reconstruction_seconds": component_totals["sidecar_io_and_reconstruction_seconds"],
        "sum_projection_compute_seconds": component_totals["projection_compute_seconds"],
        "sum_materialization_evaluation_seconds": component_totals["materialization_evaluation_seconds"],
        "sum_cache_entry_assembly_bookkeeping_seconds": component_totals["cache_entry_assembly_bookkeeping_seconds"],
        "sum_bookkeeping_seconds": component_totals["cache_entry_assembly_bookkeeping_seconds"],
        "sum_release_purge_seconds": component_totals["release_purge_seconds"],
        "wall_clock_cache_session_total_seconds": remainder["wall_total_seconds"],
        "measured_component_sum_seconds": remainder["measured_component_seconds"],
        "unattributed_remainder_seconds": remainder["unattributed_remainder_seconds"],
        "unattributed_remainder_status": remainder["unattributed_remainder_status"],
        "unattributed_remainder_warning": remainder["unattributed_remainder_warning"],
        "category_percentages_of_cache_wall_time": {
            field: (
                (component_totals if field in component_totals else overhead_totals)[field]
                / remainder["wall_total_seconds"] * 100.0
                if remainder["wall_total_seconds"] > 0.0 else None
            )
            for field in ATTRIBUTION_REPORT_CATEGORY_FIELDS
        },
        "builder_elapsed_total_seconds": normalized_stats.get("elapsed_total_seconds"),
        "shared_timestep_embedding_seconds": normalized_stats.get("elapsed_shared_timestep_embedding_seconds"),
        "cache_finalize_materialization_seconds": normalized_stats.get("elapsed_cache_finalize_materialization_seconds"),
        "measurement_notes": {
            "sidecar_io_and_reconstruction_seconds": "loader and tensor validation are one combined boundary; no fabricated open/read split",
            "projection_compute_seconds": "projection graph/executor boundary before explicit projected-output evaluation",
            "materialization_evaluation_seconds": "existing per-block MLX evaluations for payload, projection, and modulation tables",
            "cache_entry_assembly_bookkeeping_seconds": "reshape, table slicing, validation, append, and retained-byte accounting",
            "release_purge_seconds": "existing per-block reference release and allocator-purge boundary",
            "shared_timestep_embedding_seconds": "existing shared timestep embedding evaluation before block iteration",
            "cache_finalize_materialization_seconds": "existing final ModulationCache construction/materialization boundary after block iteration",
            ATTRIBUTION_TOTAL_FIELD: "block start through existing release/purge completion; telemetry emission is outside the total",
        },
    }


def build_cache_attribution_aggregate(session_attributions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate ordered session attributions without clamping or inventing missing percentiles."""
    sessions = [dict(item) for item in session_attributions]
    if not sessions:
        return {
            "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
            "session_count": 0,
            "block_count": 0,
            "sessions": [],
            "component_totals_seconds": {field: 0.0 for field in ATTRIBUTION_COMPONENT_FIELDS},
            "session_overhead_seconds": {field: 0.0 for field in ATTRIBUTION_SESSION_OVERHEAD_FIELDS},
            "cache_wall_total_seconds": 0.0,
            "unattributed_remainder_seconds": 0.0,
            "unattributed_remainder_status": "nonnegative",
            "unattributed_remainder_warning": None,
        }
    for index, session in enumerate(sessions):
        if session.get("session_index") != index:
            raise ValueError(f"cache attribution session ordering is not exactly 0..{len(sessions) - 1}")
        if session.get("attribution_schema_version") != ATTRIBUTION_SCHEMA_VERSION:
            raise ValueError(f"cache attribution session {index} schema version is invalid")
        if session.get("blocks_completed") != EXPECTED_BLOCK_COUNT:
            raise ValueError(f"cache attribution session {index} does not contain 50 blocks")
        if session.get("sidecar_opens") != EXPECTED_BLOCK_COUNT or session.get("sidecar_releases") != EXPECTED_BLOCK_COUNT:
            raise ValueError(f"cache attribution session {index} sidecar counters changed")
        if not isinstance(session.get("blocks"), list) or len(session["blocks"]) != EXPECTED_BLOCK_COUNT:
            raise ValueError(f"cache attribution session {index} block list is incomplete")

    component_totals = {
        field: math.fsum(float(session["component_totals_seconds"][field]) for session in sessions)
        for field in ATTRIBUTION_COMPONENT_FIELDS
    }
    overhead_totals = {
        field: math.fsum(float(session["session_overhead_seconds"][field]) for session in sessions)
        for field in ATTRIBUTION_SESSION_OVERHEAD_FIELDS
    }
    cache_wall_total = math.fsum(float(session["wall_clock_cache_session_total_seconds"]) for session in sessions)
    measured_components = math.fsum((*component_totals.values(), *overhead_totals.values()))
    remainder = calculate_unattributed_remainder(cache_wall_total, measured_components)
    all_blocks = [
        {**block, "session_index": session["session_index"]}
        for session in sessions
        for block in session["blocks"]
    ]
    if [int(block["block_index"]) for block in all_blocks] != list(range(EXPECTED_BLOCK_COUNT)) * len(sessions):
        raise ValueError("cache attribution per-block ordering was not retained")
    per_block_statistics: dict[str, Any] = {}
    for block_index in range(EXPECTED_BLOCK_COUNT):
        observations = [block for block in all_blocks if block["block_index"] == block_index]
        per_block_statistics[str(block_index)] = {
            "block_index": block_index,
            "sample_count": len(observations),
            "total_block_cache_construction": _timing_statistics(
                [block[ATTRIBUTION_TOTAL_FIELD] for block in observations]
            ),
            "category_statistics": {
                field: _timing_statistics([block[field] for block in observations])
                for field in ATTRIBUTION_COMPONENT_FIELDS
            },
        }
    session_wall_values = [float(session["wall_clock_cache_session_total_seconds"]) for session in sessions]
    warm_values = session_wall_values[1:]
    first_vs_warm = {
        "first_session_seconds": session_wall_values[0],
        "warm_session_indices": list(range(1, len(sessions))),
        "warm_session_count": len(warm_values),
        "warm_sessions_1_14_mean_seconds": math.fsum(warm_values) / len(warm_values) if warm_values else None,
        "warm_sessions_1_14_total_seconds": math.fsum(warm_values),
        "delta_first_minus_warm_mean_seconds": (
            session_wall_values[0] - math.fsum(warm_values) / len(warm_values) if warm_values else None
        ),
    }

    def block_identity(block: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "session_index": block["session_index"],
            "block_index": block["block_index"],
            "sidecar_filename": block["sidecar_filename"],
            "seconds": block[ATTRIBUTION_TOTAL_FIELD],
        }

    slowest = max(all_blocks, key=lambda item: item[ATTRIBUTION_TOTAL_FIELD])
    fastest = min(all_blocks, key=lambda item: item[ATTRIBUTION_TOTAL_FIELD])
    slowest_category = max(ATTRIBUTION_COMPONENT_FIELDS, key=lambda field: slowest[field])
    fastest_category = min(ATTRIBUTION_COMPONENT_FIELDS, key=lambda field: fastest[field])
    return {
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "session_count": len(sessions),
        "block_count": len(all_blocks),
        "sessions": sessions,
        "component_totals_seconds": component_totals,
        "session_overhead_seconds": overhead_totals,
        "cache_wall_total_seconds": cache_wall_total,
        "total_cache_wall_time_seconds": cache_wall_total,
        "category_totals_seconds": {**component_totals, **overhead_totals},
        "total_block_cache_construction_seconds": math.fsum(block[ATTRIBUTION_TOTAL_FIELD] for block in all_blocks),
        "category_percentages_of_cache_wall_time": {
            field: (component_totals if field in component_totals else overhead_totals)[field] / cache_wall_total * 100.0
            if cache_wall_total > 0.0 else None
            for field in ATTRIBUTION_REPORT_CATEGORY_FIELDS
        },
        "per_block_statistics": per_block_statistics,
        "per_session_mean_seconds": math.fsum(session_wall_values) / len(session_wall_values),
        "per_session_wall_clock_statistics": _timing_statistics(session_wall_values),
        "first_session_vs_sessions_1_14": first_vs_warm,
        "slowest_block": block_identity(slowest),
        "slowest_block_category": {"session_index": slowest["session_index"], "block_index": slowest["block_index"], "category": slowest_category, "seconds": slowest[slowest_category]},
        "fastest_block": block_identity(fastest),
        "fastest_block_category": {"session_index": fastest["session_index"], "block_index": fastest["block_index"], "category": fastest_category, "seconds": fastest[fastest_category]},
        "sidecar_opens": sum(int(session["sidecar_opens"]) for session in sessions),
        "sidecar_releases": sum(int(session["sidecar_releases"]) for session in sessions),
        "unattributed_remainder_seconds": remainder["unattributed_remainder_seconds"],
        "unattributed_remainder_status": remainder["unattributed_remainder_status"],
        "unattributed_remainder_warning": remainder["unattributed_remainder_warning"],
        "measurement_notes": sessions[0].get("measurement_notes", {}),
    }


def validate_cache_attribution(observed: Mapping[str, Any], *, require_full_schedule: bool = True) -> None:
    if observed.get("attribution_schema_version") != ATTRIBUTION_SCHEMA_VERSION:
        raise ValueError("cache attribution schema version is invalid")
    session_count = observed.get("session_count")
    block_count = observed.get("block_count")
    expected_sessions = EXPECTED_DENOISING_TRANSITIONS if require_full_schedule else session_count
    if require_full_schedule and session_count != expected_sessions:
        raise ValueError("cache attribution requires exactly 15 sessions")
    if require_full_schedule and block_count != expected_sessions * EXPECTED_BLOCK_COUNT:
        raise ValueError("cache attribution requires exactly 750 blocks")
    sessions = observed.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != session_count:
        raise ValueError("cache attribution session list does not match its count")
    expected_component_totals = {field: 0.0 for field in ATTRIBUTION_COMPONENT_FIELDS}
    expected_overhead_totals = {field: 0.0 for field in ATTRIBUTION_SESSION_OVERHEAD_FIELDS}
    expected_wall_total = 0.0
    for index, session in enumerate(sessions):
        if session.get("session_index") != index:
            raise ValueError(f"cache attribution session ordering is invalid at {index}")
        blocks = session.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != EXPECTED_BLOCK_COUNT:
            raise ValueError(f"cache attribution session {index} does not retain 50 ordered blocks")
        for block_index, block in enumerate(blocks):
            _raw_attribution_block(block, block_index)
        for field in ATTRIBUTION_COMPONENT_FIELDS:
            expected = math.fsum(float(block[field]) for block in blocks)
            actual = float(session["component_totals_seconds"][field])
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"cache attribution session {index} component sum changed for {field}")
            expected_component_totals[field] += actual
        for field in ATTRIBUTION_SESSION_OVERHEAD_FIELDS:
            overhead = _timing_seconds(session.get("session_overhead_seconds", {}).get(field), f"session.{field}")
            expected_overhead_totals[field] += overhead
        expected_wall_total += float(session["wall_clock_cache_session_total_seconds"])
        expected_measured = math.fsum((
            *session["component_totals_seconds"].values(),
            *session["session_overhead_seconds"].values(),
        ))
        if not math.isclose(session["measured_component_sum_seconds"], expected_measured, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cache attribution session {index} measured component sum changed")
        remainder = calculate_unattributed_remainder(
            session["wall_clock_cache_session_total_seconds"],
            expected_measured,
        )
        if not math.isclose(session["unattributed_remainder_seconds"], remainder["unattributed_remainder_seconds"], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cache attribution session {index} hides its unattributed remainder")
    for field, expected in expected_component_totals.items():
        if not math.isclose(observed["component_totals_seconds"][field], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cache attribution aggregate sum changed for {field}")
    for field, expected in expected_overhead_totals.items():
        if not math.isclose(observed["session_overhead_seconds"][field], expected, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"cache attribution aggregate overhead sum changed for {field}")
    if not math.isclose(observed["cache_wall_total_seconds"], expected_wall_total, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("cache attribution aggregate wall-clock total changed")
    aggregate_remainder = calculate_unattributed_remainder(
        observed["cache_wall_total_seconds"],
        math.fsum((*expected_component_totals.values(), *expected_overhead_totals.values())),
    )
    if not math.isclose(observed["unattributed_remainder_seconds"], aggregate_remainder["unattributed_remainder_seconds"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("cache attribution aggregate hides its unattributed remainder")


class StreamedCacheSessionProvider:
    """One cache session per transition with observed sidecar lifecycle accounting."""

    def __init__(
        self,
        cache_builder: Callable[[int, Any, Callable[[str, Mapping[str, Any]], None]], Any],
        *,
        event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
        cleanup_hook: Callable[[int, Any], None] | None = None,
    ):
        self.cache_builder = cache_builder
        self.event_sink = event_sink
        self.cleanup_hook = cleanup_hook
        self.records: list[dict[str, Any]] = []
        self.active = False
        self.active_record: dict[str, Any] | None = None
        self.last_session_token: str | None = None
        self._session_number = 0
        self._open_sidecars = 0
        self._maximum_simultaneous_sidecars = 0
        self._overlap_violations = 0
        self._active_cache: Any = None
        self._last_released_record: dict[str, Any] | None = None
        self.telemetry_failures: list[dict[str, Any]] = []

    def _emit(self, event: str, details: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            try:
                self.event_sink(event, details)
            except BaseException as exc:
                failure = {"event": event, "error": error_receipt(exc)}
                if self.active_record is not None:
                    self.active_record.setdefault("telemetry_failures", []).append(failure)
                self.telemetry_failures.append(failure)
                raise

    def _violation(self, record: dict[str, Any], message: str, details: Mapping[str, Any] | None = None) -> None:
        record.setdefault("violations", []).append(message)
        record["state_machine_valid"] = False
        payload = dict(details or {})
        payload.update({
            "transition_index": record["step_index"],
            "cache_session_id": record["cache_session_id"],
            "violation": message,
        })
        self._emit("sidecar_lifecycle_violation", payload)
        raise ValueError(message)

    def _sidecar_telemetry(self, record: dict[str, Any], event: str, details: Mapping[str, Any]) -> None:
        payload = dict(details)
        payload.update({"transition_index": record["step_index"], "cache_session_id": record["cache_session_id"]})
        if event == "sidecar_opening":
            block_index = details.get("block_index")
            path = Path(str(details.get("path", ""))).name
            expected_path = f"block-{record['next_block_index']:03d}.safetensors"
            if not isinstance(block_index, int) or isinstance(block_index, bool):
                self._violation(record, "sidecar open block index is not an integer", payload)
            if self._open_sidecars:
                self._overlap_violations += 1
                record["overlap_violations"] += 1
                self._violation(record, "sidecar opened while another sidecar remained active", payload)
            if block_index != record["next_block_index"]:
                self._violation(record, f"sidecar block opening order violation: got {block_index}, expected {record['next_block_index']}", payload)
            if block_index in record["opened_blocks"]:
                self._violation(record, f"duplicate sidecar open for block {block_index}", payload)
            if path != expected_path:
                self._violation(record, f"sidecar path mismatch for block {block_index}: got {path!r}, expected {expected_path!r}", payload)
            self._open_sidecars += 1
            self._maximum_simultaneous_sidecars = max(self._maximum_simultaneous_sidecars, self._open_sidecars)
            record["sidecar_opens"] += 1
            record["open_sidecars"] = self._open_sidecars
            record["active_sidecar"] = {"block_index": block_index, "path": path}
            record["opened_blocks"].append(block_index)
        elif event == "sidecar_released":
            block_index = details.get("block_index")
            path = Path(str(details.get("path", ""))).name
            active = record.get("active_sidecar")
            if active is None:
                self._violation(record, "sidecar release occurred without an active open", payload)
            if active.get("block_index") != block_index or active.get("path") != path:
                self._violation(record, f"sidecar release does not match its active open: got block={block_index}, path={path!r}", payload)
            if block_index in record["released_blocks"]:
                self._violation(record, f"duplicate sidecar release for block {block_index}", payload)
            if self._open_sidecars <= 0:
                self._violation(record, "sidecar release underflow", payload)
            self._open_sidecars -= 1
            record["sidecar_releases"] += 1
            record["open_sidecars"] = self._open_sidecars
            record["released_blocks"].append(block_index)
            record["active_sidecar"] = None
            record["next_block_index"] += 1
        elif event == "cache_completed":
            stats = payload.get("stats")
            summary = self._stats_summary(stats)
            record["stats"] = summary
            payload["stats"] = summary
            payload["attribution_schema_version"] = ATTRIBUTION_SCHEMA_VERSION
        elif event == ATTRIBUTION_BLOCK_EVENT:
            block_index = details.get("block_index")
            if block_index != record["next_attribution_block_index"]:
                self._violation(
                    record,
                    f"cache attribution block ordering violation: got {block_index}, expected {record['next_attribution_block_index']}",
                    payload,
                )
            if record.get("active_sidecar") is not None:
                self._violation(record, "cache attribution was emitted before sidecar release", payload)
            normalized = _raw_attribution_block(payload, record["next_attribution_block_index"])
            record["attribution_blocks"].append(normalized)
            record["next_attribution_block_index"] += 1
            payload["attribution_schema_version"] = ATTRIBUTION_SCHEMA_VERSION
            payload["timings"] = {field: normalized[field] for field in (*ATTRIBUTION_COMPONENT_FIELDS, ATTRIBUTION_TOTAL_FIELD)}
        self._emit(event, payload)

    @staticmethod
    def _split_builder_result(value: Any) -> tuple[Any, Any]:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return value[0], value[1]
        return value, None

    @staticmethod
    def _stats_summary(stats: Any) -> dict[str, Any] | None:
        if stats is None:
            return None
        raw = _json_safe(stats)
        if not isinstance(raw, Mapping):
            return {"value": raw}
        return dict(raw)

    @staticmethod
    def _validate_stats(record: dict[str, Any], stats: Any) -> dict[str, Any]:
        summary = StreamedCacheSessionProvider._stats_summary(stats)
        if not isinstance(summary, Mapping):
            raise ValueError("streamed AdaLN builder statistics are unavailable")
        for field, expected in SESSION_STAT_FIELDS.items():
            if field not in summary:
                raise ValueError(f"streamed AdaLN builder statistics field {field} is unavailable")
            if summary[field] != expected:
                raise ValueError(f"streamed AdaLN session {field}={summary[field]!r}, expected {expected!r}")
        if record.get("active_sidecar") is not None or record.get("open_sidecars") != 0:
            raise ValueError("streamed AdaLN session completed with an open sidecar")
        if record.get("opened_blocks") != list(range(EXPECTED_BLOCK_COUNT)):
            raise ValueError("streamed AdaLN session did not open blocks exactly 0..49")
        if record.get("released_blocks") != list(range(EXPECTED_BLOCK_COUNT)):
            raise ValueError("streamed AdaLN session did not release blocks exactly 0..49")
        if record.get("attribution_blocks") is None or len(record["attribution_blocks"]) != EXPECTED_BLOCK_COUNT:
            raise ValueError("streamed AdaLN session did not emit exactly 50 attribution blocks")
        if record.get("sidecar_opens") != EXPECTED_BLOCK_COUNT or record.get("sidecar_releases") != EXPECTED_BLOCK_COUNT:
            raise ValueError("streamed AdaLN session sidecar event totals are not exactly 50")
        return dict(summary)

    def _mark_exception(self, error: BaseException, *, cleanup_error: BaseException | None, attempted: bool) -> None:
        setattr(error, "cache_cleanup_attempted", attempted)
        setattr(error, "cache_cleanup_succeeded", bool(attempted and cleanup_error is None))
        setattr(error, "cache_cleanup_error", cleanup_error)

    def cache_for_step(self, step_index: int, timestep: Any, transition: Mapping[str, Any] | None = None) -> Any:
        if self.active:
            self._overlap_violations += 1
            raise RuntimeError("streamed AdaLN cache sessions overlapped")
        self._session_number += 1
        token = f"cache-session-{self._session_number:02d}"
        record: dict[str, Any] = {
            "step_index": int(step_index),
            "cache_session_id": token,
            "status": "incomplete",
            "sidecar_opens": 0,
            "sidecar_releases": 0,
            "open_sidecars": 0,
            "overlap_violations": 0,
            "dense_temporary_reconstructions": 0,
            "cleanup_attempted": False,
            "cleanup_succeeded": False,
            "cleanup_error": None,
            "forward_materialized_before_release": False,
            "local_cache_reference_dropped": False,
            "provider_cache_reference_dropped": False,
            "memory_after_reference_release": None,
            "state_machine_valid": True,
            "next_block_index": 0,
            "next_attribution_block_index": 0,
            "active_sidecar": None,
            "opened_blocks": [],
            "released_blocks": [],
            "attribution_blocks": [],
            "telemetry_failures": [],
            "violations": [],
            "bounded_release_invocations": 0,
            "events": [],
        }
        self.records.append(record)
        self.active = True
        self.active_record = record
        self._active_cache = None
        self.last_session_token = token
        try:
            record["events"].append({"event": "session-acquire-start", "step_index": step_index, "cache_session_id": token})
            self._emit("session-acquire-start", {"step_index": step_index, "cache_session_id": token})
            cache_started = time.perf_counter()
            value = self.cache_builder(
                step_index,
                timestep,
                lambda event, details: self._sidecar_telemetry(record, event, details),
            )
            cache, stats = self._split_builder_result(value)
            self._active_cache = cache
            record["stats"] = self._validate_stats(record, stats)
            record.update({field: record["stats"][field] for field in SESSION_STAT_FIELDS})
            record["cache_attribution"] = build_cache_session_attribution(
                record["stats"],
                session_index=step_index,
                wall_clock_seconds=time.perf_counter() - cache_started,
                sidecar_opens=record["sidecar_opens"],
                sidecar_releases=record["sidecar_releases"],
                block_events=record["attribution_blocks"],
            )
            record["dense_temporary_reconstructions"] = int(record["dense_temporary_projection_created"])
            record["status"] = "acquired"
            record["events"].append({"event": "session-acquire-complete", "step_index": step_index, "cache_session_id": token})
            self._emit("session-acquire-complete", {"step_index": step_index, "cache_session_id": token})
            return cache
        except BaseException as exc:
            record["status"] = "failed"
            record["failure"] = error_receipt(exc)
            cleanup_error: BaseException | None = None
            record["cleanup_attempted"] = True
            try:
                if self.cleanup_hook is not None:
                    self.cleanup_hook(step_index, self._active_cache)
                elif self._active_cache is not None:
                    record["bounded_release_invocations"] += 1
                    record["bounded_release_path"] = _bounded_release_cache(self._active_cache)
            except BaseException as cleanup_exc:
                cleanup_error = cleanup_exc
            record["cleanup_error"] = error_receipt(cleanup_error)
            record["cleanup_succeeded"] = cleanup_error is None
            record["open_sidecars_after_cleanup"] = self._open_sidecars
            self.active = False
            self.active_record = None
            self._active_cache = None
            self._mark_exception(exc, cleanup_error=cleanup_error, attempted=True)
            try:
                self._emit(
                    "session-failure",
                    {
                        "step_index": step_index,
                        "cache_session_id": token,
                        "failure": error_receipt(exc),
                        "cleanup_error": error_receipt(cleanup_error),
                    },
                )
            except BaseException:
                pass
            raise

    def release_step(self, step_index: int, cache: Any, *, forward_materialized: bool = True) -> None:
        record = self.active_record
        if not self.active or record is None or record["step_index"] != step_index:
            raise RuntimeError("streamed AdaLN cache release does not match the active transition")
        if forward_materialized is not True:
            raise RuntimeError("cache release requires successfully materialized predictions")
        record["forward_materialized_before_release"] = True
        record["events"].append({"event": "session-release-start", "step_index": step_index, "cache_session_id": record["cache_session_id"]})
        self._emit("session-release-start", {"step_index": step_index, "cache_session_id": record["cache_session_id"]})
        cleanup_error: BaseException | None = None
        try:
            if self.cleanup_hook is not None:
                record["bounded_release_invocations"] += 1
                record["bounded_release_path"] = "cleanup_hook"
                self.cleanup_hook(step_index, cache)
            else:
                record["bounded_release_invocations"] += 1
                record["bounded_release_path"] = _bounded_release_cache(cache)
        except BaseException as exc:
            cleanup_error = exc
        record["cleanup_attempted"] = True
        record["cleanup_error"] = error_receipt(cleanup_error)
        record["cleanup_succeeded"] = cleanup_error is None
        record["open_sidecars_after_cleanup"] = self._open_sidecars
        record["status"] = "released" if cleanup_error is None else "failed"
        record["events"].append(
            {
                "event": "session-release-complete" if cleanup_error is None else "session-release-failure",
                "step_index": step_index,
                "cache_session_id": record["cache_session_id"],
                "cleanup_error": error_receipt(cleanup_error),
            }
        )
        self._emit(
            "session-release-complete" if cleanup_error is None else "session-release-failure",
            {"step_index": step_index, "cache_session_id": record["cache_session_id"], "cleanup_error": error_receipt(cleanup_error)},
        )
        self._active_cache = None
        self.active = False
        self.active_record = None
        self._last_released_record = record
        if cleanup_error is not None:
            raise cleanup_error

    def cleanup_failed_step(self, step_index: int, cache: Any) -> None:
        """Clean up a failed forward without recording a successful cache release."""
        record = self.active_record
        if not self.active or record is None or record["step_index"] != step_index:
            raise RuntimeError("failed cache cleanup does not match the active transition")
        record["cleanup_attempted"] = True
        record["forward_materialized_before_release"] = False
        cleanup_error: BaseException | None = None
        try:
            if self.cleanup_hook is not None:
                record["bounded_release_invocations"] += 1
                record["bounded_release_path"] = "cleanup_hook"
                self.cleanup_hook(step_index, cache)
            else:
                record["bounded_release_invocations"] += 1
                record["bounded_release_path"] = _bounded_release_cache(cache)
        except BaseException as exc:
            cleanup_error = exc
        record["cleanup_error"] = error_receipt(cleanup_error)
        record["cleanup_succeeded"] = cleanup_error is None
        record["status"] = "failed-cleanup"
        record["open_sidecars_after_cleanup"] = self._open_sidecars
        self._emit(
            "session-failure-cleanup",
            {
                "step_index": step_index,
                "cache_session_id": record["cache_session_id"],
                "cleanup_error": error_receipt(cleanup_error),
            },
        )
        self._active_cache = None
        self.active = False
        self.active_record = None
        self._last_released_record = record
        if cleanup_error is not None:
            raise cleanup_error

    def note_local_cache_reference_dropped(
        self,
        step_index: int,
        memory: Mapping[str, Any] | None,
        *,
        forward_materialized: bool,
    ) -> None:
        record = self._last_released_record
        if record is None or record.get("step_index") != step_index:
            raise RuntimeError("cache reference drop was not associated with the released session")
        record["local_cache_reference_dropped"] = True
        record["provider_cache_reference_dropped"] = self._active_cache is None
        record["memory_after_reference_release"] = dict(memory) if isinstance(memory, Mapping) else memory
        record["forward_materialized_before_release"] = bool(forward_materialized)

    def note_cache_construction_wall(self, step_index: int, wall_clock_seconds: float) -> None:
        record = self.active_record
        if not self.active or record is None or record.get("step_index") != step_index:
            raise RuntimeError("cache attribution wall time was not associated with the active session")
        attribution = record.get("cache_attribution")
        if not isinstance(attribution, Mapping):
            raise RuntimeError("cache attribution was unavailable at the cache boundary")
        updated = build_cache_session_attribution(
            record["stats"],
            session_index=step_index,
            wall_clock_seconds=wall_clock_seconds,
            sidecar_opens=record["sidecar_opens"],
            sidecar_releases=record["sidecar_releases"],
            block_events=record["attribution_blocks"],
        )
        record["cache_attribution"] = updated
        record["events"].append(
            {
                "event": ATTRIBUTION_SESSION_EVENT,
                "step_index": step_index,
                "cache_session_id": record["cache_session_id"],
            }
        )
        self._emit(
            ATTRIBUTION_SESSION_EVENT,
            {
                "step_index": step_index,
                "cache_session_id": record["cache_session_id"],
                "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
                "wall_clock_cache_session_total_seconds": updated["wall_clock_cache_session_total_seconds"],
                "component_totals_seconds": updated["component_totals_seconds"],
                "session_overhead_seconds": updated["session_overhead_seconds"],
                "measured_component_sum_seconds": updated["measured_component_sum_seconds"],
                "unattributed_remainder_seconds": updated["unattributed_remainder_seconds"],
                "unattributed_remainder_status": updated["unattributed_remainder_status"],
            },
        )

    def cache_attribution(self) -> dict[str, Any]:
        sessions = [
            record["cache_attribution"]
            for record in self.records
            if isinstance(record.get("cache_attribution"), Mapping)
        ]
        return build_cache_attribution_aggregate(sessions)

    def aggregate(self) -> dict[str, Any]:
        return {
            "cache_sessions": len(self.records),
            "cache_sessions_created": len(self.records),
            "cache_sessions_released": sum(record.get("status") == "released" for record in self.records),
            "blocks_completed": sum(record.get("blocks_completed", 0) for record in self.records),
            "sidecar_opens": sum(record.get("sidecar_opens", 0) for record in self.records),
            "sidecar_releases": sum(record.get("sidecar_releases", 0) for record in self.records),
            "sidecar_files_opened": sum(record.get("sidecar_files_opened", 0) for record in self.records),
            "unique_sidecar_files_opened": sum(record.get("unique_sidecar_files_opened", 0) for record in self.records),
            "successful_payload_opens": sum(record.get("successful_payload_opens", 0) for record in self.records),
            "completed_payload_releases": sum(record.get("completed_payload_releases", 0) for record in self.records),
            "every_sidecar_released_before_next_opened": bool(self.records) and all(
                record.get("every_sidecar_released_before_next_opened") is True for record in self.records
            ),
            "sidecar_overlap_observed": any(record.get("sidecar_overlap_observed") is True for record in self.records),
            "next_sidecar_opened_before_previous_release": any(
                record.get("next_sidecar_opened_before_previous_release") is True for record in self.records
            ),
            "dense_temporary_projection_created": any(
                record.get("dense_temporary_projection_created") is True for record in self.records
            ),
            "maximum_simultaneous_sidecars": self._maximum_simultaneous_sidecars,
            "overlap_violations": self._overlap_violations,
            "dense_temporary_reconstructions": sum(record.get("dense_temporary_reconstructions", 0) for record in self.records),
            "open_sidecars_after_cleanup": self._open_sidecars,
            "telemetry_failure_count": sum(len(record.get("telemetry_failures", [])) for record in self.records),
            "telemetry_failures": [failure for record in self.records for failure in record.get("telemetry_failures", [])],
            "transformer_forwards": 0,
            "video_scheduler_updates": 0,
            "audio_scheduler_updates": 0,
            "sessions": list(self.records),
        }


def expected_lifecycle_totals() -> dict[str, int]:
    return dict(EXPECTED_LIFECYCLE_TOTALS)


def expected_lifecycle_sessions() -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    for index in range(EXPECTED_DENOISING_TRANSITIONS):
        session = {
            "step_index": index,
            "cache_session_id": f"cache-session-{index + 1:02d}",
            "status": "released",
            "sidecar_opens": EXPECTED_BLOCK_COUNT,
            "sidecar_releases": EXPECTED_BLOCK_COUNT,
            "open_sidecars": 0,
            "opened_blocks": list(range(EXPECTED_BLOCK_COUNT)),
            "released_blocks": list(range(EXPECTED_BLOCK_COUNT)),
            "violations": [],
            "state_machine_valid": True,
            "forward_materialized_before_release": True,
            "local_cache_reference_dropped": True,
            "provider_cache_reference_dropped": True,
            "memory_after_reference_release": {"active": 0, "allocator_cache": 0},
            "bounded_release_invocations": 1,
        }
        session.update(SESSION_STAT_FIELDS)
        sessions.append(session)
    return sessions


def validate_lifecycle_totals(observed: Mapping[str, Any]) -> None:
    sessions = observed.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != EXPECTED_DENOISING_TRANSITIONS:
        raise ValueError("streamed AdaLN lifecycle requires exactly 15 individually valid sessions")
    for index, session in enumerate(sessions):
        if not isinstance(session, Mapping):
            raise ValueError(f"streamed AdaLN session {index} is unavailable")
        if session.get("step_index") != index:
            raise ValueError(f"streamed AdaLN session transition identity is not ordered at {index}")
        if session.get("status") != "released":
            raise ValueError(f"streamed AdaLN session {index} was not fully released")
        for field, expected in SESSION_STAT_FIELDS.items():
            actual = session.get(field)
            if actual is None and isinstance(session.get("stats"), Mapping):
                actual = session["stats"].get(field)
            if actual is None:
                raise ValueError(f"streamed AdaLN session {index} field {field} is unavailable")
            if actual != expected:
                raise ValueError(f"streamed AdaLN session {index} {field}={actual!r}, expected {expected!r}")
        if session.get("sidecar_opens") != EXPECTED_BLOCK_COUNT or session.get("sidecar_releases") != EXPECTED_BLOCK_COUNT:
            raise ValueError(f"streamed AdaLN session {index} sidecar event count is not 50")
        if session.get("opened_blocks") != list(range(EXPECTED_BLOCK_COUNT)):
            raise ValueError(f"streamed AdaLN session {index} opened blocks are not exactly 0..49")
        if session.get("released_blocks") != list(range(EXPECTED_BLOCK_COUNT)):
            raise ValueError(f"streamed AdaLN session {index} released blocks are not exactly 0..49")
        if session.get("state_machine_valid") is not True or session.get("open_sidecars") != 0 or session.get("violations"):
            raise ValueError(f"streamed AdaLN session {index} retained sidecar violations or open payloads")
        for field in ("forward_materialized_before_release", "local_cache_reference_dropped", "provider_cache_reference_dropped"):
            if session.get(field) is not True:
                raise ValueError(f"streamed AdaLN session {index} missing {field} evidence")
        if not isinstance(session.get("memory_after_reference_release"), Mapping):
            raise ValueError(f"streamed AdaLN session {index} is missing active memory after reference release")
        if session.get("bounded_release_invocations") != 1:
            raise ValueError(f"streamed AdaLN session {index} bounded release path was not invoked exactly once")
    for key, expected in EXPECTED_LIFECYCLE_TOTALS.items():
        if int(observed.get(key, -1)) != expected:
            raise ValueError(f"streamed AdaLN lifecycle total {key}={observed.get(key)!r}, expected {expected}")
    if int(observed.get("telemetry_failure_count", 0)) != 0:
        raise ValueError("streamed AdaLN telemetry failure evidence is present")


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, name in (("active", "get_active_memory"), ("allocator_cache", "get_cache_memory"), ("peak", "get_peak_memory")):
        getter = getattr(mx, name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except BaseException:
            result[label] = None
    return result


def _release_runtime(mx: Any, references: Mapping[str, Any], baseline: Mapping[str, Any] | None, tolerance: int) -> dict[str, Any]:
    if isinstance(references, dict):
        for key in list(references):
            references[key] = None
    gc.collect()
    before = _memory_snapshot(mx)
    purge_error: BaseException | None = None
    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        purge_error = RuntimeError("MLX allocator cache purge is unavailable")
    else:
        try:
            clear_cache()
        except BaseException as exc:
            purge_error = exc
    after = _memory_snapshot(mx)
    baseline_active = baseline.get("active") if isinstance(baseline, Mapping) else None
    after_active = after.get("active")
    active_available = baseline_active is not None and after_active is not None
    active_within = bool(active_available and after_active <= baseline_active + tolerance)
    cache_zero = after.get("allocator_cache") == 0
    passed = purge_error is None and active_within and cache_zero
    return {
        "passed": passed,
        "memory_before_allocator_purge": before,
        "memory_after_allocator_purge": after,
        "active_memory_baseline_bytes": baseline_active,
        "active_memory_tolerance_bytes": tolerance,
        "active_memory_gate_available": active_available,
        "active_memory_within_tolerance": active_within,
        "allocator_cache_after_bytes": after.get("allocator_cache"),
        "allocator_cache_zero": cache_zero,
        "allocator_purge_error": error_receipt(purge_error),
    }


def _materialize_value(value: Any, mx: Any, materialize: Callable[[Any], None] | None = None) -> None:
    if materialize is not None:
        materialize(value)
        return
    evaluate = getattr(mx, "eval", None)
    if not callable(evaluate):
        raise RuntimeError("MLX materialization callback is unavailable")
    evaluate(value)


_V05A_DECODER_HELPERS: Any | None = None


def _v05a_decoder_helpers() -> Any:
    """Load the proven v0.5a video conversion/layout helpers without importing MLX at import time."""
    global _V05A_DECODER_HELPERS
    if _V05A_DECODER_HELPERS is not None:
        return _V05A_DECODER_HELPERS
    import importlib.util

    helper_path = ROOT / "scripts" / "probe_v05a_decoders.py"
    spec = importlib.util.spec_from_file_location("minimax_h3_mlx_v05a_decoder_helpers", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to load the v0.5a decoder helpers: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _V05A_DECODER_HELPERS = module
    return module


def validate_final_video_input_artifact(
    artifact_path: Path,
    metadata_path: Path,
    *,
    expected_attempt_identifier: str,
    expected_checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate every final-latent gate needed by the video child."""
    artifact_file = Path(artifact_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    metadata = _read_json_object(metadata_file, "final native latent metadata")
    arrays = _load_npz(artifact_file)
    validate_final_artifact(
        metadata,
        arrays=arrays,
        artifact_path=artifact_file,
        metadata_path=metadata_file,
    )
    if metadata.get("attempt_identifier") != expected_attempt_identifier:
        raise ValueError("video worker final latent attempt identifier mismatch")
    if metadata.get("checkpoint_identity") != _json_safe(dict(expected_checkpoint_identity)):
        raise ValueError("video worker final latent checkpoint identity mismatch")
    if metadata.get("worker_identity") != "derived":
        raise ValueError("video worker final latent derived worker identity is invalid")
    exit_receipt = metadata.get("worker_exit_receipt")
    if (
        not isinstance(exit_receipt, Mapping)
        or exit_receipt.get("worker_started") is not True
        or exit_receipt.get("worker_exit_observed") is not True
        or exit_receipt.get("worker_exit_code") != 0
        or exit_receipt.get("worker_termination_confirmed") is not True
    ):
        raise ValueError("video worker final latent derived worker termination is not confirmed")
    release = metadata.get("transformer_release_receipt")
    if (
        not isinstance(release, Mapping)
        or release.get("passed") is not True
        or release.get("allocator_cache_zero") is not True
    ):
        raise ValueError("video worker final latent transformer release gate did not pass")
    if metadata.get("final_allocator_cache_zero") is not True or metadata.get("final_allocator_cache") != 0:
        raise ValueError("video worker final latent allocator cache is not zero")
    if set(arrays) != {"final_video_native", "final_audio_native"}:
        raise ValueError("video worker final latent NPZ key set is not exact")
    video = np.asarray(arrays["final_video_native"])
    if video.dtype != np.dtype(np.float32):
        raise ValueError(f"video worker final latent storage dtype must be float32, got {video.dtype}")
    if tuple(video.shape) != VIDEO_NATIVE_SHAPE:
        raise ValueError("video worker final latent shape is not (1,24,9,8,8)")
    descriptor = metadata.get("native_video")
    if not isinstance(descriptor, Mapping) or descriptor.get("dtype") != "bfloat16":
        raise ValueError("video worker final latent logical dtype is not bfloat16")
    expected_fingerprint = descriptor.get("fingerprint")
    if expected_fingerprint != array_fingerprint(video, logical_dtype="bfloat16"):
        raise ValueError("video worker final latent fingerprint is invalid")
    return {
        "passed": True,
        "artifact_path": str(artifact_file),
        "metadata_path": str(metadata_file),
        "artifact_npz_sha256": metadata["final_artifact_npz_sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
        "attempt_identifier": metadata["attempt_identifier"],
        "checkpoint_identity": metadata["checkpoint_identity"],
        "derived_worker_identity": metadata["worker_identity"],
        "derived_worker_termination_confirmed": True,
        "transformer_release_gate_passed": True,
        "allocator_cache_zero": True,
        "video_shape": list(video.shape),
        "video_storage_dtype": np.dtype(video.dtype).name,
        "video_logical_dtype": descriptor["dtype"],
        "video_fingerprint": expected_fingerprint,
        "arrays": arrays,
        "metadata": metadata,
    }


def restore_video_latent_logical_bfloat16(
    stored: Any,
    mx: Any,
    *,
    expected_fingerprint: str,
    materialize: Callable[[Any], None] | None = None,
) -> tuple[Any, str]:
    """Restore float32 NPZ storage to a materialized logical MLX bfloat16 latent."""
    stored_array = np.asarray(stored)
    if stored_array.dtype != np.dtype(np.float32):
        raise ValueError(f"video latent NPZ storage must be float32, got {stored_array.dtype}")
    if tuple(stored_array.shape) != VIDEO_NATIVE_SHAPE:
        raise ValueError(f"video latent shape mismatch: {tuple(stored_array.shape)} != {VIDEO_NATIVE_SHAPE}")
    float32_latent = mx.array(np.ascontiguousarray(stored_array, dtype=np.float32), dtype=mx.float32)
    logical_latent = float32_latent.astype(mx.bfloat16)
    _materialize_value(logical_latent, mx, materialize)
    fingerprint = array_fingerprint(logical_latent, logical_dtype="bfloat16", mx=mx)
    if fingerprint != expected_fingerprint:
        raise ValueError("video latent logical bfloat16 fingerprint mismatch")
    return logical_latent, fingerprint


def normalize_video_latent_for_decode(
    latent: Any,
    config: Any,
    mx: Any,
    *,
    materialize: Callable[[Any], None] | None = None,
) -> Any:
    """Apply the v0.5a video normalization and materialize an MLX float32 decoder input."""
    helpers = _v05a_decoder_helpers()
    mean, std = helpers._normalization_arrays(mx, config, audio=False)
    scaled = latent * std + mean
    decoder_input = scaled.astype(mx.float32)
    _materialize_value(decoder_input, mx, materialize)
    if _dtype_name(getattr(decoder_input, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError("video decoder input was not materialized as float32")
    return decoder_input


def validate_locked_video_config(config: Any, layout: Any) -> dict[str, Any]:
    """Validate the exact 30-frame, 128x128 geometry before loading video VAE weights."""
    expected_fields = {
        "latent_channels": 24,
        "out_channels": 3,
        "spatial_compression_ratio": 16,
        "temporal_compression_ratio": 4,
        "clip_length": 17,
        "token_drop": 3,
    }
    for field, expected in expected_fields.items():
        if int(getattr(config, field)) != expected:
            raise ValueError(f"video config {field}={getattr(config, field)!r}, expected {expected}")
    expected_layout = {
        "clip_length": 17,
        "temporal_compression_ratio": 4,
        "tokens_chunk_size": 5,
        "token_drop": 3,
        "token_overlap": 2,
        "frame_pre_padding": 3,
        "frame_overlap": 5,
        "chunk_num_frames": 20,
        "tail_trim_remainder": 1,
        "minimum_latent_frames": 7,
    }
    for field, expected in expected_layout.items():
        if int(getattr(layout, field)) != expected:
            raise ValueError(f"video decode layout {field}={getattr(layout, field)!r}, expected {expected}")
    decoded_frames = _v05a_decoder_helpers().video_decoded_frame_count(9, layout)
    if decoded_frames != VIDEO_FRAME_COUNT:
        raise ValueError(f"video decoder geometry produces {decoded_frames} frames, expected {VIDEO_FRAME_COUNT}")
    mean = np.asarray(getattr(config, "latents_mean"), dtype=np.float32)
    std = np.asarray(getattr(config, "latents_std"), dtype=np.float32)
    if mean.shape != (24,) or std.shape != (24,):
        raise ValueError("video latent normalization metadata must have shape [24]")
    return {
        "native_latent_shape": list(VIDEO_NATIVE_SHAPE),
        "raw_shape": list(VIDEO_RAW_SHAPE),
        "rgb_shape": list(VIDEO_RGB_SHAPE),
        "frames": VIDEO_FRAME_COUNT,
        "resolution": [VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT],
        "fps": VIDEO_FRAME_FPS,
        "duration_seconds": VIDEO_FRAME_DURATION_SECONDS,
        "layout": {field: int(getattr(layout, field)) for field in expected_layout},
    }


def materialize_and_validate_video_raw_output(
    raw: Any,
    mx: Any,
    *,
    materialize: Callable[[Any], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize once, then validate the locked raw decoder tensor and finite values."""
    if _dtype_name(getattr(raw, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError(f"raw video decoder output must be float32, got {getattr(raw, 'dtype', None)}")
    _materialize_value(raw, mx, materialize)
    raw_np = np.array(raw, dtype=np.float32, copy=True)
    if tuple(raw_np.shape) != VIDEO_RAW_SHAPE:
        raise ValueError(f"raw video shape mismatch: {tuple(raw_np.shape)} != {VIDEO_RAW_SHAPE}")
    if not np.isfinite(raw_np).all():
        raise ValueError("raw video decoder output contains non-finite values")
    return raw_np, {"shape": list(raw_np.shape), "dtype": str(raw_np.dtype)}


def convert_and_validate_video_rgb(raw_np: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Use the proven v0.5a inverse-normalization/RGB conversion and enforce the locked output."""
    frames = _v05a_decoder_helpers().video_frames_from_raw(raw_np)
    return frames, validate_video_rgb_output(frames)


def validate_video_rgb_output(frames: Any) -> dict[str, Any]:
    frames = np.asarray(frames)
    if tuple(frames.shape) != VIDEO_RGB_SHAPE:
        raise ValueError(f"RGB video shape mismatch: {tuple(frames.shape)} != {VIDEO_RGB_SHAPE}")
    if frames.dtype != np.dtype(np.uint8):
        raise ValueError(f"RGB video dtype must be uint8, got {frames.dtype}")
    if not np.isfinite(frames).all():
        raise ValueError("RGB video contains non-finite values")
    return {"shape": list(frames.shape), "dtype": str(frames.dtype)}


def _inspect_video_frame_set(frames_directory: Path) -> dict[str, Any]:
    """Validate every staged PNG, including signature, decoded mode, dimensions, and checksum."""
    directory = Path(frames_directory).resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"video frame directory is missing: {directory}")
    entries = sorted(path for path in directory.iterdir() if path.is_file())
    expected_names = [f"frame_{index:05d}.png" for index in range(VIDEO_FRAME_COUNT)]
    actual_names = [path.name for path in entries]
    if actual_names != expected_names:
        raise ValueError(
            f"video frame filenames are not exactly contiguous: expected={expected_names}, actual={actual_names}"
        )
    frame_records: list[dict[str, Any]] = []
    signature = b"\x89PNG\r\n\x1a\n"
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for PNG publication validation") from exc
    for index, path in enumerate(entries):
        if path.read_bytes()[:8] != signature:
            raise ValueError(f"video frame {path.name} has an invalid PNG signature")
        size_bytes = path.stat().st_size
        if size_bytes <= 0:
            raise ValueError(f"video frame {path.name} is empty")
        try:
            with Image.open(path) as image:
                image.load()
                mode = image.mode
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"video frame {path.name} is not a readable PNG") from exc
        if mode != "RGB":
            raise ValueError(f"video frame {path.name} mode is {mode}, expected RGB")
        if (width, height) != (VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT):
            raise ValueError(
                f"video frame {path.name} dimensions are {(width, height)}, expected {(VIDEO_FRAME_WIDTH, VIDEO_FRAME_HEIGHT)}"
            )
        frame_records.append(
            {
                "path": path.name,
                "index": index,
                "sha256": sha256_file(path),
                "size_bytes": size_bytes,
                "mode": mode,
                "width": width,
                "height": height,
            }
        )
    return {
        "frame_count": len(frame_records),
        "width": VIDEO_FRAME_WIDTH,
        "height": VIDEO_FRAME_HEIGHT,
        "frames": frame_records,
    }


def stable_video_frame_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    payload = json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_video_frame_manifest(
    frames_directory: Path,
    *,
    attempt_identifier: str,
    worker_identity: str = VIDEO_WORKER_IDENTITY,
    publication_state: str = "published",
) -> dict[str, Any]:
    inspected = _inspect_video_frame_set(Path(frames_directory))
    manifest = {
        "manifest_identity": VIDEO_FRAME_MANIFEST_IDENTITY,
        "schema_version": VIDEO_FRAME_MANIFEST_SCHEMA_VERSION,
        "attempt_identifier": attempt_identifier,
        "worker_identity": worker_identity,
        "publication_state": publication_state,
        "frame_count": VIDEO_FRAME_COUNT,
        "width": VIDEO_FRAME_WIDTH,
        "height": VIDEO_FRAME_HEIGHT,
        "fps": VIDEO_FRAME_FPS,
        "duration_seconds": VIDEO_FRAME_DURATION_SECONDS,
        "frames": inspected["frames"],
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = stable_video_frame_manifest_sha256(manifest)
    return manifest


def validate_video_frame_manifest(
    manifest_path: Path,
    frames_directory: Path,
    *,
    expected_attempt_identifier: str,
    expected_worker_identity: str = VIDEO_WORKER_IDENTITY,
) -> dict[str, Any]:
    """Independently validate the published manifest and all 30 files it links."""
    manifest = _read_json_object(Path(manifest_path).resolve(), "video frame manifest")
    if set(manifest) != VIDEO_FRAME_MANIFEST_KEYS:
        raise ValueError(
            f"video frame manifest schema mismatch: missing={sorted(VIDEO_FRAME_MANIFEST_KEYS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - VIDEO_FRAME_MANIFEST_KEYS)}"
        )
    if manifest.get("manifest_identity") != VIDEO_FRAME_MANIFEST_IDENTITY:
        raise ValueError("video frame manifest identity mismatch")
    if manifest.get("schema_version") != VIDEO_FRAME_MANIFEST_SCHEMA_VERSION:
        raise ValueError("video frame manifest schema version mismatch")
    if manifest.get("attempt_identifier") != expected_attempt_identifier:
        raise ValueError("video frame manifest attempt identifier mismatch")
    if manifest.get("worker_identity") != expected_worker_identity:
        raise ValueError("video frame manifest worker identity mismatch")
    if manifest.get("publication_state") != "published":
        raise ValueError("video frame manifest publication state is not published")
    if manifest.get("frame_count") != VIDEO_FRAME_COUNT or manifest.get("width") != VIDEO_FRAME_WIDTH or manifest.get("height") != VIDEO_FRAME_HEIGHT:
        raise ValueError("video frame manifest geometry is not 30 frames at 128x128")
    if manifest.get("fps") != VIDEO_FRAME_FPS or manifest.get("duration_seconds") != VIDEO_FRAME_DURATION_SECONDS:
        raise ValueError("video frame manifest timing metadata is invalid")
    if manifest.get("manifest_sha256") != stable_video_frame_manifest_sha256(manifest):
        raise ValueError("video frame manifest checksum linkage is stale")
    inspected = _inspect_video_frame_set(Path(frames_directory))
    if manifest.get("frames") != inspected["frames"]:
        raise ValueError("video frame manifest frame checksum linkage is stale")
    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "frames_path": str(Path(frames_directory).resolve()),
        "publication_state": manifest["publication_state"],
        "manifest_sha256": manifest["manifest_sha256"],
        "frame_count": manifest["frame_count"],
        "width": manifest["width"],
        "height": manifest["height"],
        "fps": manifest["fps"],
        "duration_seconds": manifest["duration_seconds"],
        "frames": manifest["frames"],
    }


def publish_video_frames_atomically(
    frames_partial: Path,
    frames_final: Path,
    manifest_path: Path,
    *,
    attempt_identifier: str,
    worker_identity: str = VIDEO_WORKER_IDENTITY,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Validate staged frames, then atomically rename the directory into its final name."""
    partial = Path(frames_partial).resolve()
    final = Path(frames_final).resolve()
    manifest_file = Path(manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final video frames directory: {final}")
    if not partial.is_dir():
        raise FileNotFoundError(f"staged video frames directory is missing: {partial}")
    if manifest_file.exists():
        raise FileExistsError(f"refusing existing video frame manifest: {manifest_file}")
    manifest = build_video_frame_manifest(
        partial,
        attempt_identifier=attempt_identifier,
        worker_identity=worker_identity,
        publication_state="published",
    )
    _write_json(manifest_file, manifest)
    validate_video_frame_manifest(
        manifest_file,
        partial,
        expected_attempt_identifier=attempt_identifier,
        expected_worker_identity=worker_identity,
    )
    if final.exists():
        raise FileExistsError(f"refusing existing final video frames directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    (rename or (lambda source, destination: source.rename(destination)))(partial, final)
    return validate_video_frame_manifest(
        manifest_file,
        final,
        expected_attempt_identifier=attempt_identifier,
        expected_worker_identity=worker_identity,
    )


def collect_video_frame_evidence(frames_directory: Path) -> list[dict[str, Any]]:
    directory = Path(frames_directory).resolve()
    if not directory.is_dir():
        return []
    evidence: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.png")):
        item: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size if path.is_file() else 0}
        if path.is_file():
            item["sha256"] = sha256_file(path)
        evidence.append(item)
    return evidence


def execute_video_decode_once(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected_attempt_identifier: str,
    expected_checkpoint_identity: Mapping[str, Any],
    video_root: Path,
    frames_partial: Path,
    frames_final: Path,
    manifest_path: Path,
    mx: Any,
    load_video_config: Callable[[Path], Any],
    load_video_vae: Callable[[Path], Any],
    save_frames: Callable[[Path, np.ndarray], Any] | None = None,
    materialize: Callable[[Any], None] | None = None,
    memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one injected video decode, with all MLX work supplied by the child."""
    refs = references if references is not None else {}
    input_validation = validate_final_video_input_artifact(
        artifact_path,
        metadata_path,
        expected_attempt_identifier=expected_attempt_identifier,
        expected_checkpoint_identity=expected_checkpoint_identity,
    )
    partial = Path(frames_partial).resolve()
    final = Path(frames_final).resolve()
    manifest_file = Path(manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final video frames directory: {final}")
    if partial.exists():
        raise FileExistsError(f"refusing existing staged video frames directory: {partial}")
    config = load_video_config(Path(video_root).resolve())
    refs["config"] = config
    layout = _v05a_decoder_helpers().resolve_video_decode_layout(config)
    geometry = validate_locked_video_config(config, layout)
    memory: dict[str, Any] = {}
    snapshot = memory_snapshot or (lambda: _memory_snapshot(mx))
    memory["before_load"] = dict(snapshot())
    stored_video = input_validation["arrays"]["final_video_native"]
    refs["stored_latent"] = stored_video
    latent, logical_fingerprint = restore_video_latent_logical_bfloat16(
        stored_video,
        mx,
        expected_fingerprint=input_validation["video_fingerprint"],
        materialize=materialize,
    )
    refs["latent"] = latent
    decoder_input = normalize_video_latent_for_decode(latent, config, mx, materialize=materialize)
    refs["decoder_input"] = decoder_input
    if _dtype_name(getattr(decoder_input, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError("video decoder input must be materialized as float32")
    decoder = load_video_vae(Path(video_root).resolve())
    refs["decoder"] = decoder
    memory["after_load"] = dict(snapshot())
    raw = decoder.decode(decoder_input)
    refs["raw"] = raw
    raw_np, raw_shape_dtype = materialize_and_validate_video_raw_output(raw, mx, materialize=materialize)
    refs["raw_np"] = raw_np
    frames, rgb_shape_dtype = convert_and_validate_video_rgb(raw_np)
    refs["frames"] = frames
    memory["peak"] = dict(snapshot())
    if save_frames is None:
        from minimax_h3_mlx.media import save_frames as save_frames_impl

        save_frames = save_frames_impl
    save_frames(partial, frames)
    manifest_validation = publish_video_frames_atomically(
        partial,
        final,
        manifest_file,
        attempt_identifier=expected_attempt_identifier,
    )
    return {
        "input_artifact": {key: value for key, value in input_validation.items() if key not in {"arrays", "metadata"}},
        "video_geometry": geometry,
        "logical_video_fingerprint": logical_fingerprint,
        "raw_shape_dtype": raw_shape_dtype,
        "rgb_shape_dtype": rgb_shape_dtype,
        "frame_manifest": manifest_validation,
        "memory": memory,
        "references": refs,
    }


def release_video_decoder(
    mx: Any,
    references: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    tolerance: int,
    *,
    memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drop every video reference, purge the MLX allocator, and return the strict release gate."""
    snapshot = memory_snapshot or (lambda: _memory_snapshot(mx))
    before_release = dict(snapshot())
    if isinstance(references, dict):
        for key in list(references):
            references[key] = None
    release = _release_runtime(mx, references, baseline, tolerance)
    release["memory_before_release"] = before_release
    release["memory_after_release"] = release.get("memory_after_allocator_purge")
    return release


def validate_final_audio_input_artifact(
    artifact_path: Path,
    metadata_path: Path,
    *,
    expected_attempt_identifier: str,
    expected_checkpoint_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently validate every final-latent gate required by the audio child.

    This function is deliberately MLX-free.  It is called before the child imports the audio VAE
    loader, so a stale checksum, identity, shape, logical dtype, or fingerprint cannot cause VAE
    allocation.
    """
    artifact_file = Path(artifact_path).resolve()
    metadata_file = Path(metadata_path).resolve()
    metadata = _read_json_object(metadata_file, "final native latent metadata")
    arrays = _load_npz(artifact_file)
    validate_final_artifact(
        metadata,
        arrays=arrays,
        artifact_path=artifact_file,
        metadata_path=metadata_file,
    )
    if metadata.get("attempt_identifier") != expected_attempt_identifier:
        raise ValueError("audio worker final latent attempt identifier mismatch")
    if metadata.get("checkpoint_identity") != _json_safe(dict(expected_checkpoint_identity)):
        raise ValueError("audio worker final latent checkpoint identity mismatch")
    if metadata.get("worker_identity") != "derived":
        raise ValueError("audio worker final latent derived worker identity is invalid")
    exit_receipt = metadata.get("worker_exit_receipt")
    if (
        not isinstance(exit_receipt, Mapping)
        or exit_receipt.get("worker_started") is not True
        or exit_receipt.get("worker_exit_observed") is not True
        or exit_receipt.get("worker_exit_code") != 0
        or exit_receipt.get("worker_termination_confirmed") is not True
    ):
        raise ValueError("audio worker final latent derived worker termination is not confirmed")
    release = metadata.get("transformer_release_receipt")
    if (
        not isinstance(release, Mapping)
        or release.get("passed") is not True
        or release.get("allocator_cache_zero") is not True
    ):
        raise ValueError("audio worker final latent transformer release gate did not pass")
    if metadata.get("final_allocator_cache_zero") is not True or metadata.get("final_allocator_cache") != 0:
        raise ValueError("audio worker final latent allocator cache is not zero")
    if set(arrays) != {"final_video_native", "final_audio_native"}:
        raise ValueError("audio worker final latent NPZ key set is not exact")
    audio = np.asarray(arrays["final_audio_native"])
    if audio.dtype != np.dtype(np.float32):
        raise ValueError(f"audio worker final latent storage dtype must be float32, got {audio.dtype}")
    if tuple(audio.shape) != AUDIO_NATIVE_SHAPE:
        raise ValueError("audio worker final latent shape is not (2,32,50)")
    descriptor = metadata.get("native_audio")
    if (
        not isinstance(descriptor, Mapping)
        or tuple(descriptor.get("shape", ())) != AUDIO_NATIVE_SHAPE
        or descriptor.get("dtype") != "bfloat16"
    ):
        raise ValueError("audio worker final latent logical dtype is not bfloat16")
    expected_fingerprint = descriptor.get("fingerprint")
    if expected_fingerprint != array_fingerprint(audio, logical_dtype="bfloat16"):
        raise ValueError("audio worker final latent fingerprint is invalid")
    return {
        "passed": True,
        "artifact_path": str(artifact_file),
        "metadata_path": str(metadata_file),
        "artifact_npz_sha256": metadata["final_artifact_npz_sha256"],
        "metadata_sha256": metadata["metadata_sha256"],
        "attempt_identifier": metadata["attempt_identifier"],
        "checkpoint_identity": metadata["checkpoint_identity"],
        "derived_worker_identity": metadata["worker_identity"],
        "derived_worker_termination_confirmed": True,
        "transformer_release_gate_passed": True,
        "allocator_cache_zero": True,
        "audio_shape": list(audio.shape),
        "audio_storage_dtype": np.dtype(audio.dtype).name,
        "audio_logical_dtype": descriptor["dtype"],
        "audio_fingerprint": expected_fingerprint,
        "arrays": arrays,
        "metadata": metadata,
    }


def restore_audio_latent_logical_bfloat16(
    stored: Any,
    mx: Any,
    *,
    expected_fingerprint: str,
    materialize: Callable[[Any], None] | None = None,
) -> tuple[Any, str]:
    """Restore float32 NPZ storage to a materialized logical MLX bfloat16 audio latent."""
    stored_array = np.asarray(stored)
    if stored_array.dtype != np.dtype(np.float32):
        raise ValueError(f"audio latent NPZ storage must be float32, got {stored_array.dtype}")
    if tuple(stored_array.shape) != AUDIO_NATIVE_SHAPE:
        raise ValueError(f"audio latent shape mismatch: {tuple(stored_array.shape)} != {AUDIO_NATIVE_SHAPE}")
    float32_latent = mx.array(np.ascontiguousarray(stored_array, dtype=np.float32), dtype=mx.float32)
    logical_latent = float32_latent.astype(mx.bfloat16)
    _materialize_value(logical_latent, mx, materialize)
    fingerprint = array_fingerprint(logical_latent, logical_dtype="bfloat16", mx=mx)
    if fingerprint != expected_fingerprint:
        raise ValueError("audio latent logical bfloat16 fingerprint mismatch")
    return logical_latent, fingerprint


def normalize_audio_latent_for_decode(
    latent: Any,
    config: Any,
    mx: Any,
    *,
    materialize: Callable[[Any], None] | None = None,
) -> Any:
    """Apply the established audio latent normalization and materialize float32 decoder input."""
    helpers = _v05a_decoder_helpers()
    mean, std = helpers._normalization_arrays(mx, config, audio=True)
    scaled = latent * std + mean
    decoder_input = scaled.astype(mx.float32)
    _materialize_value(decoder_input, mx, materialize)
    if _dtype_name(getattr(decoder_input, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError("audio decoder input was not materialized as float32")
    return decoder_input


def validate_locked_audio_config(config: Any) -> dict[str, Any]:
    """Validate the exact 32 kHz stereo audio geometry before loading audio VAE weights."""
    latent_channels = int(getattr(config, "latent_channels"))
    sample_rate = int(getattr(config, "sampling_rate"))
    decoder_rates = tuple(int(rate) for rate in getattr(config, "decoder_rates"))
    hop_length = int(getattr(config, "hop_length"))
    if latent_channels != AUDIO_NATIVE_SHAPE[1]:
        raise ValueError(f"audio config latent_channels={latent_channels}, expected {AUDIO_NATIVE_SHAPE[1]}")
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"audio config sampling_rate={sample_rate}, expected {AUDIO_SAMPLE_RATE}")
    if math.prod(decoder_rates) != hop_length or hop_length != 800:
        raise ValueError("audio decoder rates and hop length do not match the locked 800-sample hop")
    mean = np.asarray(getattr(config, "latents_mean"), dtype=np.float32)
    std = np.asarray(getattr(config, "latents_std"), dtype=np.float32)
    if mean.shape != (AUDIO_NATIVE_SHAPE[1],) or std.shape != (AUDIO_NATIVE_SHAPE[1],):
        raise ValueError("audio latent normalization metadata must have shape [32]")
    return {
        "native_latent_shape": list(AUDIO_NATIVE_SHAPE),
        "raw_shape": list(AUDIO_RAW_SHAPE),
        "waveform_shape": list(AUDIO_WAVEFORM_SHAPE),
        "channels": 2,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "samples_per_channel": AUDIO_SAMPLE_COUNT,
        "sample_width_bytes": AUDIO_SAMPLE_WIDTH_BYTES,
        "duration_seconds": AUDIO_DURATION_SECONDS,
        "decoder_rates": list(decoder_rates),
        "hop_length": hop_length,
    }


def materialize_and_validate_audio_raw_output(
    raw: Any,
    mx: Any,
    *,
    materialize: Callable[[Any], None] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Materialize once, then validate the locked raw audio tensor and finite values."""
    if _dtype_name(getattr(raw, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError(f"raw audio decoder output must be float32, got {getattr(raw, 'dtype', None)}")
    _materialize_value(raw, mx, materialize)
    raw_np = np.array(raw, dtype=np.float32, copy=True)
    if tuple(raw_np.shape) != AUDIO_RAW_SHAPE:
        raise ValueError(f"raw audio shape mismatch: {tuple(raw_np.shape)} != {AUDIO_RAW_SHAPE}")
    if not np.isfinite(raw_np).all():
        raise ValueError("raw audio decoder output contains non-finite values")
    return raw_np, {"shape": list(raw_np.shape), "dtype": str(raw_np.dtype)}


def convert_and_validate_audio_waveform(raw_np: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert ``(2,1,40000)`` raw audio to a finite stereo ``(2,40000)`` waveform."""
    raw_array = np.asarray(raw_np)
    if tuple(raw_array.shape) != AUDIO_RAW_SHAPE:
        raise ValueError(f"raw audio shape mismatch: {tuple(raw_array.shape)} != {AUDIO_RAW_SHAPE}")
    waveform = np.ascontiguousarray(raw_array[:, 0, :], dtype=np.float32)
    if tuple(waveform.shape) != AUDIO_WAVEFORM_SHAPE:
        raise ValueError(f"stereo waveform shape mismatch: {tuple(waveform.shape)} != {AUDIO_WAVEFORM_SHAPE}")
    if waveform.dtype != np.dtype(np.float32):
        raise ValueError(f"stereo waveform must be float32, got {waveform.dtype}")
    if not np.isfinite(waveform).all():
        raise ValueError("stereo waveform contains non-finite values")
    return waveform, {"shape": list(waveform.shape), "dtype": str(waveform.dtype)}


def _wav_metadata(path: Path) -> dict[str, Any]:
    """Read structural RIFF/WAVE metadata and prove that the data chunk is not truncated."""
    import wave

    wav_path = Path(path).resolve()
    if not wav_path.is_file() or wav_path.stat().st_size <= 0:
        raise ValueError(f"WAV is missing or empty: {wav_path}")
    header = wav_path.read_bytes()[:12]
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        raise ValueError("WAV is not a RIFF/WAVE file")
    try:
        with wave.open(str(wav_path), "rb") as handle:
            channels = handle.getnchannels()
            sample_rate = handle.getframerate()
            sample_count = handle.getnframes()
            sample_width = handle.getsampwidth()
            expected_data_bytes = channels * sample_count * sample_width
            actual_data = handle.readframes(sample_count)
    except (OSError, EOFError, wave.Error) as exc:
        raise ValueError(f"WAV structure is unreadable: {wav_path}") from exc
    if len(actual_data) != expected_data_bytes:
        raise ValueError("WAV data chunk is truncated")
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "sample_width_bytes": sample_width,
        "duration_seconds": sample_count / sample_rate if sample_rate else 0.0,
        "size_bytes": wav_path.stat().st_size,
        "wav_sha256": sha256_file(wav_path),
    }


def validate_audio_wav_metadata(metadata: Mapping[str, Any]) -> None:
    """Validate the exact stereo, 32 kHz, 40,000-sample structural WAV contract."""
    if metadata.get("channels") != 2:
        raise ValueError(f"WAV channel count must be 2, got {metadata.get('channels')}")
    if metadata.get("sample_rate") != AUDIO_SAMPLE_RATE:
        raise ValueError(f"WAV sample rate mismatch: {metadata.get('sample_rate')} != {AUDIO_SAMPLE_RATE}")
    if metadata.get("sample_count") != AUDIO_SAMPLE_COUNT:
        raise ValueError(f"WAV sample count mismatch: {metadata.get('sample_count')} != {AUDIO_SAMPLE_COUNT}")
    if metadata.get("sample_width_bytes") != AUDIO_SAMPLE_WIDTH_BYTES:
        raise ValueError(
            f"WAV sample width mismatch: {metadata.get('sample_width_bytes')} != {AUDIO_SAMPLE_WIDTH_BYTES}"
        )
    if metadata.get("duration_seconds") != AUDIO_DURATION_SECONDS:
        raise ValueError("WAV duration does not match the locked 1.25-second structural contract")
    if not isinstance(metadata.get("size_bytes"), int) or metadata.get("size_bytes", 0) <= 0:
        raise ValueError("WAV file size must be nonzero")
    checksum = metadata.get("wav_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("WAV checksum is missing or malformed")


def stable_audio_wav_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    payload = json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_audio_wav_manifest(
    wav_path: Path,
    *,
    attempt_identifier: str,
    worker_identity: str = AUDIO_WORKER_IDENTITY,
    publication_state: str = "published",
) -> dict[str, Any]:
    metadata = _wav_metadata(Path(wav_path))
    validate_audio_wav_metadata(metadata)
    manifest = {
        "manifest_identity": AUDIO_WAV_MANIFEST_IDENTITY,
        "schema_version": AUDIO_WAV_MANIFEST_SCHEMA_VERSION,
        "attempt_identifier": attempt_identifier,
        "worker_identity": worker_identity,
        "publication_state": publication_state,
        **metadata,
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = stable_audio_wav_manifest_sha256(manifest)
    return manifest


def validate_audio_wav_manifest(
    manifest_path: Path,
    wav_path: Path,
    *,
    expected_attempt_identifier: str,
    expected_worker_identity: str = AUDIO_WORKER_IDENTITY,
) -> dict[str, Any]:
    """Independently validate manifest linkage and the complete final/staged WAV structure."""
    manifest = _read_json_object(Path(manifest_path).resolve(), "audio WAV manifest")
    if set(manifest) != AUDIO_WAV_MANIFEST_KEYS:
        raise ValueError(
            f"audio WAV manifest schema mismatch: missing={sorted(AUDIO_WAV_MANIFEST_KEYS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - AUDIO_WAV_MANIFEST_KEYS)}"
        )
    if manifest.get("manifest_identity") != AUDIO_WAV_MANIFEST_IDENTITY:
        raise ValueError("audio WAV manifest identity mismatch")
    if manifest.get("schema_version") != AUDIO_WAV_MANIFEST_SCHEMA_VERSION:
        raise ValueError("audio WAV manifest schema version mismatch")
    if manifest.get("attempt_identifier") != expected_attempt_identifier:
        raise ValueError("audio WAV manifest attempt identifier mismatch")
    if manifest.get("worker_identity") != expected_worker_identity:
        raise ValueError("audio WAV manifest worker identity mismatch")
    if manifest.get("publication_state") != "published":
        raise ValueError("audio WAV manifest publication state is not published")
    if manifest.get("manifest_sha256") != stable_audio_wav_manifest_sha256(manifest):
        raise ValueError("audio WAV manifest checksum linkage is stale")
    actual = _wav_metadata(Path(wav_path))
    validate_audio_wav_metadata(actual)
    for field in (
        "channels",
        "sample_rate",
        "sample_count",
        "sample_width_bytes",
        "duration_seconds",
        "size_bytes",
        "wav_sha256",
    ):
        if manifest.get(field) != actual.get(field):
            raise ValueError(f"audio WAV manifest {field} linkage is stale")
    if manifest.get("size_bytes") != actual["size_bytes"] or manifest.get("wav_sha256") != actual["wav_sha256"]:
        raise ValueError("audio WAV manifest file checksum linkage is stale")
    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "wav_path": str(Path(wav_path).resolve()),
        "publication_state": manifest["publication_state"],
        "manifest_sha256": manifest["manifest_sha256"],
        "wav_sha256": manifest["wav_sha256"],
        "size_bytes": manifest["size_bytes"],
        "channels": manifest["channels"],
        "sample_rate": manifest["sample_rate"],
        "sample_count": manifest["sample_count"],
        "sample_width_bytes": manifest["sample_width_bytes"],
        "duration_seconds": manifest["duration_seconds"],
    }


def publish_audio_wav_atomically(
    wav_partial: Path,
    wav_final: Path,
    manifest_path: Path,
    *,
    attempt_identifier: str,
    worker_identity: str = AUDIO_WORKER_IDENTITY,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Validate staged WAV evidence, then atomically rename it into the final audio path."""
    partial = Path(wav_partial).resolve()
    final = Path(wav_final).resolve()
    manifest_file = Path(manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final audio WAV: {final}")
    if not partial.is_file():
        raise FileNotFoundError(f"staged audio WAV is missing: {partial}")
    if manifest_file.exists():
        raise FileExistsError(f"refusing existing audio WAV manifest: {manifest_file}")
    manifest = build_audio_wav_manifest(
        partial,
        attempt_identifier=attempt_identifier,
        worker_identity=worker_identity,
        publication_state="published",
    )
    _write_json(manifest_file, manifest)
    validate_audio_wav_manifest(
        manifest_file,
        partial,
        expected_attempt_identifier=attempt_identifier,
        expected_worker_identity=worker_identity,
    )
    if final.exists():
        raise FileExistsError(f"refusing existing final audio WAV: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    (rename or (lambda source, destination: source.rename(destination)))(partial, final)
    return validate_audio_wav_manifest(
        manifest_file,
        final,
        expected_attempt_identifier=attempt_identifier,
        expected_worker_identity=worker_identity,
    )


class MP4MuxFailure(RuntimeError):
    """A mux failure carrying primary, cleanup, subprocess, and preservation evidence."""

    def __init__(
        self,
        primary_error: BaseException,
        *,
        receipt: Mapping[str, Any],
        cleanup_error: BaseException | None = None,
    ) -> None:
        self.primary_error = primary_error
        self.cleanup_error = cleanup_error
        self.receipt = dict(receipt)
        super().__init__(str(primary_error))


class MP4MuxLaunchGateFailure(MP4MuxFailure):
    """A mux attempt suppressed before either subprocess can launch."""


def build_ffmpeg_command(
    frames_directory: Path,
    wav_path: Path,
    mp4_partial_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> list[str]:
    """Build the deterministic image-sequence plus stereo-WAV ffmpeg argv."""
    if not isinstance(ffmpeg_binary, str) or not ffmpeg_binary:
        raise ValueError("ffmpeg binary name is required")
    return [
        ffmpeg_binary,
        "-y",
        "-framerate",
        str(VIDEO_FRAME_FPS),
        "-start_number",
        "0",
        "-i",
        str(Path(frames_directory).resolve() / "frame_%05d.png"),
        "-i",
        str(Path(wav_path).resolve()),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-frames:v",
        str(VIDEO_FRAME_COUNT),
        "-c:v",
        "libx264",
        "-pix_fmt",
        MP4_EXPECTED_PIXEL_FORMAT,
        "-c:a",
        MP4_EXPECTED_AUDIO_CODEC,
        "-b:a",
        MP4_EXPECTED_AUDIO_BITRATE,
        "-shortest",
        str(Path(mp4_partial_path).resolve()),
    ]


def build_ffprobe_command(
    mp4_path: Path,
    *,
    ffprobe_binary: str = "ffprobe",
) -> list[str]:
    """Build the JSON ffprobe argv used before MP4 publication."""
    if not isinstance(ffprobe_binary, str) or not ffprobe_binary:
        raise ValueError("ffprobe binary name is required")
    return [
        ffprobe_binary,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-count_frames",
        str(Path(mp4_path).resolve()),
    ]


def _subprocess_stream_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _default_mux_subprocess_runner(
    argv: Sequence[str],
    *,
    capture_output: bool,
    text: bool,
    check: bool,
    timeout: float,
) -> Any:
    """Run one argv without a shell; tests inject a runner instead of reaching this seam."""
    return subprocess.run(
        list(argv),
        shell=False,
        capture_output=capture_output,
        text=text,
        check=check,
        timeout=timeout,
    )


def _empty_mux_subprocess_receipt(tool: str) -> dict[str, Any]:
    return {
        "tool": tool,
        "status": "not_performed",
        "invoked": False,
        "argv": None,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "timed_out": False,
        "timeout_seconds": None,
        "elapsed_seconds": 0.0,
        "runner_error": None,
    }


def _run_mux_subprocess(
    tool: str,
    argv: Sequence[str],
    *,
    runner: Callable[..., Any],
    timeout_seconds: float,
) -> tuple[dict[str, Any], BaseException | None]:
    if timeout_seconds <= 0:
        raise ValueError("mux subprocess timeout must be positive")
    receipt = _empty_mux_subprocess_receipt(tool)
    receipt.update(
        {
            "status": "running",
            "invoked": True,
            "argv": [str(argument) for argument in argv],
            "timeout_seconds": timeout_seconds,
        }
    )
    started = time.perf_counter()
    try:
        completed = runner(
            list(argv),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        receipt.update(
            {
                "status": "timeout",
                "timed_out": True,
                "stdout": _subprocess_stream_text(getattr(exc, "stdout", getattr(exc, "output", None))),
                "stderr": _subprocess_stream_text(getattr(exc, "stderr", None)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return receipt, TimeoutError(f"{tool} timed out after {timeout_seconds:g} seconds")
    except BaseException as exc:
        receipt.update(
            {
                "status": "failed",
                "runner_error": error_receipt(exc),
                "stdout": _subprocess_stream_text(getattr(exc, "stdout", getattr(exc, "output", None))),
                "stderr": _subprocess_stream_text(getattr(exc, "stderr", None)),
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        return receipt, RuntimeError(f"{tool} subprocess runner failed: {exc}")

    returncode = getattr(completed, "returncode", None)
    if isinstance(completed, Mapping):
        returncode = completed.get("returncode", completed.get("exit_code"))
        stdout = completed.get("stdout")
        stderr = completed.get("stderr")
    else:
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
    receipt.update(
        {
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "stdout": _subprocess_stream_text(stdout),
            "stderr": _subprocess_stream_text(stderr),
            "elapsed_seconds": time.perf_counter() - started,
        }
    )
    if returncode != 0:
        return receipt, RuntimeError(f"{tool} exited with status {returncode}")
    return receipt, None


def _numeric_probe_value(value: Any, field: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"ffprobe field {field} is missing")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ffprobe field {field} is not numeric: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"ffprobe field {field} is not finite")
    return result


def _integer_probe_value(value: Any, field: str) -> int:
    number = _numeric_probe_value(value, field)
    if not number.is_integer():
        raise ValueError(f"ffprobe field {field} is not an integer: {value!r}")
    return int(number)


def _probe_frame_rate(stream: Mapping[str, Any]) -> tuple[float, str]:
    for field in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(field)
        if value not in (None, "", "N/A"):
            if isinstance(value, str) and "/" in value:
                numerator_text, denominator_text = value.split("/", 1)
                numerator = _numeric_probe_value(numerator_text, field)
                denominator = _numeric_probe_value(denominator_text, field)
                if denominator == 0:
                    raise ValueError(f"ffprobe field {field} has a zero denominator")
                rate = numerator / denominator
            else:
                rate = _numeric_probe_value(value, field)
            if rate <= 0:
                raise ValueError(f"ffprobe field {field} must be positive")
            return rate, field
    raise ValueError("ffprobe video frame rate evidence is missing")


def _duration_within_tolerance(value: float, expected: float = MP4_EXPECTED_DURATION_SECONDS) -> bool:
    return abs(value - expected) <= MP4_DURATION_TOLERANCE_SECONDS


def validate_ffprobe_json(
    probe_json: Mapping[str, Any],
    *,
    mp4_path: Path | None = None,
) -> dict[str, Any]:
    """Validate exactly one H.264 video stream, one AAC stereo stream, and 1.25 s duration."""
    if not isinstance(probe_json, Mapping):
        raise ValueError("ffprobe JSON must be an object")
    streams = probe_json.get("streams")
    if not isinstance(streams, list):
        raise ValueError("ffprobe JSON is missing its streams array")
    video_streams = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if isinstance(stream, Mapping) and stream.get("codec_type") == "audio"]
    if len(video_streams) != 1:
        raise ValueError(f"ffprobe must contain exactly one usable video stream, found {len(video_streams)}")
    if len(audio_streams) != 1:
        raise ValueError(f"ffprobe must contain exactly one usable audio stream, found {len(audio_streams)}")
    video = video_streams[0]
    audio = audio_streams[0]

    codec_name = video.get("codec_name")
    if not isinstance(codec_name, str) or codec_name.lower() != MP4_EXPECTED_VIDEO_CODEC:
        raise ValueError(f"ffprobe video codec must be H.264, got {codec_name!r}")
    width = _integer_probe_value(video.get("width"), "video.width")
    height = _integer_probe_value(video.get("height"), "video.height")
    if width != VIDEO_FRAME_WIDTH or height != VIDEO_FRAME_HEIGHT:
        raise ValueError(f"ffprobe video dimensions must be 128x128, got {width}x{height}")
    frame_rate, frame_rate_field = _probe_frame_rate(video)
    if not math.isclose(frame_rate, VIDEO_FRAME_FPS, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"ffprobe video frame rate must be 24 fps, got {frame_rate:g}")
    pixel_format = video.get("pix_fmt")
    if not isinstance(pixel_format, str) or pixel_format != MP4_EXPECTED_PIXEL_FORMAT:
        raise ValueError(f"ffprobe pixel format must be yuv420p, got {pixel_format!r}")

    frame_count: int | None = None
    frame_count_field: str | None = None
    for field in ("nb_frames", "nb_read_frames"):
        value = video.get(field)
        if value not in (None, "", "N/A"):
            frame_count = _integer_probe_value(value, f"video.{field}")
            frame_count_field = field
            break
    format_info = probe_json.get("format")
    if not isinstance(format_info, Mapping):
        raise ValueError("ffprobe JSON is missing its format object")
    format_duration = _numeric_probe_value(format_info.get("duration"), "format.duration")
    if not _duration_within_tolerance(format_duration):
        raise ValueError(
            f"ffprobe container duration must be approximately {MP4_EXPECTED_DURATION_SECONDS:g} seconds, "
            f"got {format_duration:g}"
        )
    stream_duration: float | None = None
    if video.get("duration") not in (None, "", "N/A"):
        stream_duration = _numeric_probe_value(video.get("duration"), "video.duration")
        if not _duration_within_tolerance(stream_duration):
            raise ValueError(
                f"ffprobe video duration must be approximately {MP4_EXPECTED_DURATION_SECONDS:g} seconds, "
                f"got {stream_duration:g}"
            )
    duration_evidence = stream_duration if stream_duration is not None else format_duration
    if frame_count is None:
        if not math.isclose(duration_evidence * frame_rate, VIDEO_FRAME_COUNT, rel_tol=0.0, abs_tol=0.5):
            raise ValueError("ffprobe video duration/frame-rate evidence is inconsistent with 30 frames")
    elif frame_count != VIDEO_FRAME_COUNT:
        raise ValueError(f"ffprobe video frame count must be 30, got {frame_count}")

    audio_codec_name = audio.get("codec_name")
    if not isinstance(audio_codec_name, str) or audio_codec_name.lower() != MP4_EXPECTED_AUDIO_CODEC:
        raise ValueError(f"ffprobe audio codec must be AAC, got {audio_codec_name!r}")
    channels = _integer_probe_value(audio.get("channels"), "audio.channels")
    if channels != 2:
        raise ValueError(f"ffprobe audio channels must be 2, got {channels}")
    sample_rate = _integer_probe_value(audio.get("sample_rate"), "audio.sample_rate")
    if sample_rate != AUDIO_SAMPLE_RATE:
        raise ValueError(f"ffprobe audio sample rate must be 32000 Hz, got {sample_rate}")

    size_bytes: int | None = None
    if mp4_path is not None:
        artifact = Path(mp4_path).resolve()
        if not artifact.is_file():
            raise ValueError(f"staged MP4 is missing: {artifact}")
        size_bytes = artifact.stat().st_size
        if size_bytes <= 0:
            raise ValueError(f"staged MP4 is empty: {artifact}")
    else:
        if format_info.get("size") in (None, "", "N/A"):
            raise ValueError("ffprobe container size evidence is missing")
        size_bytes = _integer_probe_value(format_info.get("size"), "format.size")
        if size_bytes <= 0:
            raise ValueError("ffprobe container size must be nonzero")

    return {
        "passed": True,
        "video": {
            "codec_family": MP4_EXPECTED_VIDEO_CODEC,
            "codec_name": codec_name,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "frame_rate_field": frame_rate_field,
            "pixel_format": pixel_format,
            "frame_count": frame_count if frame_count is not None else VIDEO_FRAME_COUNT,
            "frame_count_field": frame_count_field,
            "duration_seconds": duration_evidence,
        },
        "audio": {
            "codec_family": MP4_EXPECTED_AUDIO_CODEC,
            "codec_name": audio_codec_name,
            "channels": channels,
            "sample_rate": sample_rate,
            "bitrate_requested": MP4_EXPECTED_AUDIO_BITRATE,
        },
        "container": {
            "duration_seconds": format_duration,
            "size_bytes": size_bytes,
        },
    }


def stable_mp4_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    canonical = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    payload = json.dumps(_json_safe(canonical), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_probe_json_sha256(probe_json: Mapping[str, Any]) -> str:
    payload = json.dumps(_json_safe(probe_json), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_mp4_manifest(
    artifact_path: Path,
    *,
    published_path: Path,
    attempt_identifier: str,
    video_manifest: Mapping[str, Any],
    audio_manifest: Mapping[str, Any],
    probe_metadata: Mapping[str, Any],
    probe_json: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the checksum-bound MP4 manifest before final publication."""
    artifact = Path(artifact_path).resolve()
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise ValueError("cannot manifest a missing or empty staged MP4")
    if not isinstance(attempt_identifier, str) or not attempt_identifier:
        raise ValueError("MP4 manifest attempt identifier is required")
    if video_manifest.get("publication_state") != "published" or audio_manifest.get("publication_state") != "published":
        raise ValueError("MP4 manifest requires published PNG and WAV manifests")
    manifest = {
        "manifest_identity": MP4_MANIFEST_IDENTITY,
        "schema_version": MP4_MANIFEST_SCHEMA_VERSION,
        "attempt_identifier": attempt_identifier,
        "publication_state": "published",
        "mp4_path": str(Path(published_path).resolve()),
        "size_bytes": artifact.stat().st_size,
        "mp4_sha256": sha256_file(artifact),
        "video_frame_manifest_path": str(Path(video_manifest["manifest_path"]).resolve()),
        "video_frame_manifest_sha256": video_manifest["manifest_sha256"],
        "video_frame_manifest_file_sha256": sha256_file(Path(video_manifest["manifest_path"])),
        "audio_manifest_path": str(Path(audio_manifest["manifest_path"]).resolve()),
        "audio_manifest_sha256": audio_manifest["manifest_sha256"],
        "audio_manifest_file_sha256": sha256_file(Path(audio_manifest["manifest_path"])),
        "video": dict(probe_metadata["video"]),
        "audio": dict(probe_metadata["audio"]),
        "container": dict(probe_metadata["container"]),
        "ffprobe_json_sha256": _stable_probe_json_sha256(probe_json),
        "manifest_sha256": None,
    }
    manifest["manifest_sha256"] = stable_mp4_manifest_sha256(manifest)
    return manifest


def validate_mp4_manifest(
    manifest_path: Path,
    mp4_path: Path,
    *,
    expected_attempt_identifier: str,
    expected_published_path: Path | None = None,
    expected_video_manifest_path: Path | None = None,
    expected_audio_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Validate MP4 manifest schema, source-manifest linkage, and artifact checksum linkage."""
    manifest_file = Path(manifest_path).resolve()
    manifest = _read_json_object(manifest_file, "MP4 manifest")
    if set(manifest) != MP4_MANIFEST_KEYS:
        raise ValueError(
            f"MP4 manifest schema mismatch: missing={sorted(MP4_MANIFEST_KEYS - set(manifest))}, "
            f"unexpected={sorted(set(manifest) - MP4_MANIFEST_KEYS)}"
        )
    if manifest.get("manifest_identity") != MP4_MANIFEST_IDENTITY:
        raise ValueError("MP4 manifest identity mismatch")
    if manifest.get("schema_version") != MP4_MANIFEST_SCHEMA_VERSION:
        raise ValueError("MP4 manifest schema version mismatch")
    if manifest.get("attempt_identifier") != expected_attempt_identifier:
        raise ValueError("MP4 manifest attempt identifier mismatch")
    if manifest.get("publication_state") != "published":
        raise ValueError("MP4 manifest publication state is not published")
    if manifest.get("manifest_sha256") != stable_mp4_manifest_sha256(manifest):
        raise ValueError("MP4 manifest checksum linkage is stale")
    artifact = Path(mp4_path).resolve()
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise ValueError(f"published or staged MP4 is missing or empty: {artifact}")
    if manifest.get("mp4_path") != str(Path(expected_published_path or manifest["mp4_path"]).resolve()):
        raise ValueError("MP4 manifest published path linkage is stale")
    actual_size = artifact.stat().st_size
    actual_sha = sha256_file(artifact)
    if manifest.get("size_bytes") != actual_size or manifest.get("mp4_sha256") != actual_sha:
        raise ValueError("MP4 manifest file checksum linkage is stale")

    for path_key, checksum_key, file_checksum_key, expected_path in (
        ("video_frame_manifest_path", "video_frame_manifest_sha256", "video_frame_manifest_file_sha256", expected_video_manifest_path),
        ("audio_manifest_path", "audio_manifest_sha256", "audio_manifest_file_sha256", expected_audio_manifest_path),
    ):
        linked_path = Path(manifest[path_key]).resolve()
        if expected_path is not None and linked_path != Path(expected_path).resolve():
            raise ValueError(f"MP4 manifest {path_key} linkage is stale")
        if not linked_path.is_file():
            raise ValueError(f"MP4 manifest {path_key} linkage is stale")
        linked_manifest = _read_json_object(linked_path, f"linked {path_key}")
        if linked_manifest.get("manifest_sha256") != manifest[checksum_key]:
            raise ValueError(f"MP4 manifest {checksum_key} linkage is stale")
        if sha256_file(linked_path) != manifest[file_checksum_key]:
            raise ValueError(f"MP4 manifest {checksum_key} linkage is stale")

    video = manifest.get("video")
    audio = manifest.get("audio")
    container = manifest.get("container")
    if not isinstance(video, Mapping) or not isinstance(audio, Mapping) or not isinstance(container, Mapping):
        raise ValueError("MP4 manifest media sections are incomplete")
    if (
        video.get("codec_family") != MP4_EXPECTED_VIDEO_CODEC
        or video.get("width") != VIDEO_FRAME_WIDTH
        or video.get("height") != VIDEO_FRAME_HEIGHT
        or not math.isclose(float(video.get("frame_rate")), VIDEO_FRAME_FPS, rel_tol=0.0, abs_tol=1e-6)
        or video.get("pixel_format") != MP4_EXPECTED_PIXEL_FORMAT
        or video.get("frame_count") != VIDEO_FRAME_COUNT
    ):
        raise ValueError("MP4 manifest video properties are invalid")
    if (
        audio.get("codec_family") != MP4_EXPECTED_AUDIO_CODEC
        or audio.get("channels") != 2
        or audio.get("sample_rate") != AUDIO_SAMPLE_RATE
        or audio.get("bitrate_requested") != MP4_EXPECTED_AUDIO_BITRATE
    ):
        raise ValueError("MP4 manifest audio properties are invalid")
    if not isinstance(manifest.get("ffprobe_json_sha256"), str) or len(manifest["ffprobe_json_sha256"]) != 64:
        raise ValueError("MP4 manifest ffprobe checksum is invalid")
    duration = _numeric_probe_value(container.get("duration_seconds"), "manifest.container.duration_seconds")
    if not _duration_within_tolerance(duration) or container.get("size_bytes") != actual_size:
        raise ValueError("MP4 manifest container linkage is invalid")
    return {
        "manifest_path": str(manifest_file),
        "mp4_path": str(artifact),
        "publication_state": manifest["publication_state"],
        "size_bytes": actual_size,
        "mp4_sha256": actual_sha,
        "video_frame_manifest_path": manifest["video_frame_manifest_path"],
        "video_frame_manifest_sha256": manifest["video_frame_manifest_sha256"],
        "video_frame_manifest_file_sha256": manifest["video_frame_manifest_file_sha256"],
        "audio_manifest_path": manifest["audio_manifest_path"],
        "audio_manifest_sha256": manifest["audio_manifest_sha256"],
        "audio_manifest_file_sha256": manifest["audio_manifest_file_sha256"],
        "video": dict(video),
        "audio": dict(audio),
        "container": dict(container),
        "ffprobe_json_sha256": manifest["ffprobe_json_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
    }


def publish_mp4_atomically(
    mp4_partial_path: Path,
    mp4_final_path: Path,
    manifest_path: Path,
    *,
    manifest: Mapping[str, Any],
    expected_attempt_identifier: str,
    expected_video_manifest_path: Path,
    expected_audio_manifest_path: Path,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Validate a staged MP4 manifest, then atomically rename without replacing a final file."""
    partial = Path(mp4_partial_path).resolve()
    final = Path(mp4_final_path).resolve()
    manifest_file = Path(manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final MP4: {final}")
    if not partial.is_file() or partial.stat().st_size <= 0:
        raise ValueError(f"staged MP4 is missing or empty: {partial}")
    if not manifest_file.is_file():
        raise FileNotFoundError(f"MP4 manifest is missing before publication: {manifest_file}")
    if dict(manifest) != _read_json_object(manifest_file, "MP4 manifest"):
        raise ValueError("MP4 manifest changed before publication")
    validate_mp4_manifest(
        manifest_file,
        partial,
        expected_attempt_identifier=expected_attempt_identifier,
        expected_published_path=final,
        expected_video_manifest_path=expected_video_manifest_path,
        expected_audio_manifest_path=expected_audio_manifest_path,
    )
    if final.exists():
        raise FileExistsError(f"refusing existing final MP4: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    (rename or (lambda source, destination: source.rename(destination)))(partial, final)
    if not final.is_file() or partial.exists():
        raise RuntimeError("MP4 atomic publication did not produce exactly one final file")
    return validate_mp4_manifest(
        manifest_file,
        final,
        expected_attempt_identifier=expected_attempt_identifier,
        expected_published_path=final,
        expected_video_manifest_path=expected_video_manifest_path,
        expected_audio_manifest_path=expected_audio_manifest_path,
    )


def build_mux_launch_gate(
    report: Mapping[str, Any],
    paths: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the fail-closed gate that must pass before either mux subprocess can run."""
    video_decoder = report.get("video_decoder") if isinstance(report.get("video_decoder"), Mapping) else {}
    audio_decoder = report.get("audio_decoder") if isinstance(report.get("audio_decoder"), Mapping) else {}
    video_receipt = video_decoder.get("worker_receipt") if isinstance(video_decoder.get("worker_receipt"), Mapping) else {}
    audio_receipt = audio_decoder.get("worker_receipt") if isinstance(audio_decoder.get("worker_receipt"), Mapping) else {}
    frame_manifest_valid = False
    wav_manifest_valid = False
    errors: list[str] = []
    path_map = paths or report.get("output_paths") or {}
    attempt_identifier = report.get("attempt", {}).get("attempt_identifier") if isinstance(report.get("attempt"), Mapping) else None
    try:
        if isinstance(path_map, Mapping) and all(key in path_map for key in ("video_frame_manifest", "frames")):
            validate_video_frame_manifest(
                Path(path_map["video_frame_manifest"]),
                Path(path_map["frames"]),
                expected_attempt_identifier=str(attempt_identifier),
            )
            frame_manifest_valid = True
        else:
            frame_manifest_valid = (
                isinstance(report.get("video_artifacts"), Mapping)
                and report["video_artifacts"].get("publication_state") == "published"
                and isinstance(report["video_artifacts"].get("manifest_sha256"), str)
            )
    except BaseException as exc:
        errors.append(f"frame manifest: {exc}")
    try:
        if isinstance(path_map, Mapping) and all(key in path_map for key in ("audio_manifest", "audio_wav")):
            validate_audio_wav_manifest(
                Path(path_map["audio_manifest"]),
                Path(path_map["audio_wav"]),
                expected_attempt_identifier=str(attempt_identifier),
            )
            wav_manifest_valid = True
        else:
            wav_manifest_valid = (
                isinstance(report.get("audio_artifacts"), Mapping)
                and report["audio_artifacts"].get("publication_state") == "published"
                and isinstance(report["audio_artifacts"].get("manifest_sha256"), str)
            )
    except BaseException as exc:
        errors.append(f"WAV manifest: {exc}")
    gate = {
        "derived_phase_status": report.get("latent_generation_status"),
        "video_status": report.get("video_status"),
        "audio_status": report.get("audio_status"),
        "standalone_media_status": report.get("standalone_media_status"),
        "video_release_gate_passed": video_decoder.get("release_gate_passed") is True,
        "audio_release_gate_passed": audio_decoder.get("release_gate_passed") is True,
        "video_worker_termination_confirmed": (
            video_decoder.get("worker_termination_confirmed") is True
            or video_receipt.get("worker_termination_confirmed") is True
        ),
        "audio_worker_termination_confirmed": (
            audio_decoder.get("worker_termination_confirmed") is True
            or audio_receipt.get("worker_termination_confirmed") is True
        ),
        "frame_manifest_valid": frame_manifest_valid,
        "wav_manifest_valid": wav_manifest_valid,
        "passed": False,
    }
    gate["passed"] = (
        gate["derived_phase_status"] == "completed"
        and gate["video_status"] == "completed"
        and gate["audio_status"] == "completed"
        and gate["standalone_media_status"] == "completed"
        and all(gate[key] is True for key in MP4_MUX_LAUNCH_GATE_KEYS - {"passed", "derived_phase_status", "video_status", "audio_status", "standalone_media_status"})
    )
    if errors:
        gate["errors"] = errors
    return gate


def validate_mux_launch_gate(gate: Mapping[str, Any]) -> None:
    """Fail closed on any missing, false, or uncertain upstream mux gate."""
    missing = sorted(MP4_MUX_LAUNCH_GATE_KEYS - set(gate))
    if missing:
        raise ValueError(f"MP4 mux launch gate is missing fields: {missing}")
    if gate.get("derived_phase_status") != "completed":
        raise ValueError("MP4 mux launch gate derived phase is not completed")
    for key in ("video_status", "audio_status", "standalone_media_status"):
        if gate.get(key) != "completed":
            raise ValueError(f"MP4 mux launch gate {key} is not completed")
    boolean_gates = MP4_MUX_LAUNCH_GATE_KEYS - {
        "derived_phase_status",
        "video_status",
        "audio_status",
        "standalone_media_status",
        "passed",
    }
    for key in sorted(boolean_gates):
        if type(gate.get(key)) is not bool or gate.get(key) is not True:
            raise ValueError(f"MP4 mux launch gate {key} did not pass")
    if gate.get("passed") is not True:
        raise ValueError("MP4 mux launch gate did not pass")


def _mp4_file_evidence(path: Path) -> dict[str, Any]:
    artifact = Path(path).resolve()
    evidence: dict[str, Any] = {"path": str(artifact), "exists": artifact.exists()}
    if artifact.is_file():
        evidence["size_bytes"] = artifact.stat().st_size
        if artifact.stat().st_size > 0:
            evidence["sha256"] = sha256_file(artifact)
    return evidence


def _raise_mp4_mux_failure(
    primary: BaseException,
    *,
    partial: Path,
    manifest_path: Path,
    manifest_created: bool,
    frames_directory: Path,
    wav_path: Path,
    gate: Mapping[str, Any],
    ffmpeg_receipt: Mapping[str, Any],
    ffprobe_receipt: Mapping[str, Any],
    source_validation: Mapping[str, Any],
    started: float,
) -> NoReturn:
    partial_before = _mp4_file_evidence(partial)
    cleanup_attempted = True
    cleanup_error: BaseException | None = None
    try:
        if partial.exists():
            if not partial.is_file():
                raise IsADirectoryError(f"staged MP4 path is not a file: {partial}")
            partial.unlink()
        if manifest_created and manifest_path.exists():
            manifest_path.unlink()
    except BaseException as exc:
        cleanup_error = exc
    receipt = {
        "status": "failed",
        "partial_cleanup_policy": "delete_partial_mp4_after_failure",
        "partial_path": str(partial),
        "partial_before_cleanup": partial_before,
        "partial_after_cleanup": _mp4_file_evidence(partial),
        "manifest_path": str(manifest_path),
        "manifest_created": manifest_created,
        "ffmpeg": dict(ffmpeg_receipt),
        "ffprobe": dict(ffprobe_receipt),
        "source_validation": _json_safe(dict(source_validation)),
        "launch_gate": dict(gate),
        "frames_preserved": Path(frames_directory).is_dir(),
        "wav_preserved": Path(wav_path).is_file(),
        "retry_suppressed": True,
        "invocation_counts": {
            "ffmpeg": 1 if ffmpeg_receipt.get("invoked") is True else 0,
            "ffprobe": 1 if ffprobe_receipt.get("invoked") is True else 0,
        },
        "primary_error": error_receipt(primary),
        "cleanup_error": error_receipt(cleanup_error),
        **failure_fields(
            primary,
            cleanup_error,
            cleanup_attempted=cleanup_attempted,
            cleanup_succeeded=cleanup_error is None,
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    raise MP4MuxFailure(primary, receipt=receipt, cleanup_error=cleanup_error) from primary


def execute_mp4_mux(
    *,
    frames_directory: Path,
    video_manifest_path: Path,
    wav_path: Path,
    audio_manifest_path: Path,
    mp4_partial_path: Path,
    mp4_final_path: Path,
    mp4_manifest_path: Path,
    attempt_identifier: str,
    launch_gate: Mapping[str, Any],
    subprocess_runner: Callable[..., Any] | None = None,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    timeout_seconds: float = 120.0,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Mux validated standalone media once, inspect it once, and publish only after all gates pass."""
    validate_mux_launch_gate(launch_gate)
    frames = Path(frames_directory).resolve()
    video_manifest_file = Path(video_manifest_path).resolve()
    wav = Path(wav_path).resolve()
    audio_manifest_file = Path(audio_manifest_path).resolve()
    partial = Path(mp4_partial_path).resolve()
    final = Path(mp4_final_path).resolve()
    manifest_file = Path(mp4_manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final MP4: {final}")
    if manifest_file.exists():
        raise FileExistsError(f"refusing existing MP4 manifest: {manifest_file}")
    if partial.exists():
        raise FileExistsError(f"refusing existing staged MP4: {partial}")
    runner = subprocess_runner or _default_mux_subprocess_runner
    started = time.perf_counter()
    ffmpeg_receipt = _empty_mux_subprocess_receipt("ffmpeg")
    ffprobe_receipt = _empty_mux_subprocess_receipt("ffprobe")
    source_validation: dict[str, Any] = {}
    manifest_created = False
    try:
        source_validation["video"] = validate_video_frame_manifest(
            video_manifest_file,
            frames,
            expected_attempt_identifier=attempt_identifier,
        )
        source_validation["audio"] = validate_audio_wav_manifest(
            audio_manifest_file,
            wav,
            expected_attempt_identifier=attempt_identifier,
        )
        ffmpeg_argv = build_ffmpeg_command(frames, wav, partial, ffmpeg_binary=ffmpeg_binary)
        ffmpeg_receipt, runner_error = _run_mux_subprocess(
            "ffmpeg",
            ffmpeg_argv,
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
        if runner_error is not None:
            raise runner_error
        if ffmpeg_receipt.get("returncode") != 0:
            raise RuntimeError(f"ffmpeg exited with status {ffmpeg_receipt.get('returncode')}")
        if not partial.is_file() or partial.stat().st_size <= 0:
            raise ValueError("ffmpeg completed successfully without a nonzero staged MP4")

        ffprobe_argv = build_ffprobe_command(partial, ffprobe_binary=ffprobe_binary)
        ffprobe_receipt, runner_error = _run_mux_subprocess(
            "ffprobe",
            ffprobe_argv,
            runner=runner,
            timeout_seconds=timeout_seconds,
        )
        if runner_error is not None:
            raise runner_error
        if ffprobe_receipt.get("returncode") != 0:
            raise RuntimeError(f"ffprobe exited with status {ffprobe_receipt.get('returncode')}")
        try:
            probe_json = json.loads(ffprobe_receipt.get("stdout", ""))
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("ffprobe did not return valid JSON") from exc
        probe_metadata = validate_ffprobe_json(probe_json, mp4_path=partial)
        manifest = build_mp4_manifest(
            partial,
            published_path=final,
            attempt_identifier=attempt_identifier,
            video_manifest=source_validation["video"],
            audio_manifest=source_validation["audio"],
            probe_metadata=probe_metadata,
            probe_json=probe_json,
        )
        _write_json(manifest_file, manifest)
        manifest_created = True
        validate_mp4_manifest(
            manifest_file,
            partial,
            expected_attempt_identifier=attempt_identifier,
            expected_published_path=final,
            expected_video_manifest_path=video_manifest_file,
            expected_audio_manifest_path=audio_manifest_file,
        )
        artifact = publish_mp4_atomically(
            partial,
            final,
            manifest_file,
            manifest=manifest,
            expected_attempt_identifier=attempt_identifier,
            expected_video_manifest_path=video_manifest_file,
            expected_audio_manifest_path=audio_manifest_file,
            rename=rename,
        )
        if sha256_file(final) != manifest["mp4_sha256"]:
            raise ValueError("final MP4 checksum changed across atomic publication")
        return {
            "status": "completed",
            "invoked": True,
            "launch_gate": dict(launch_gate),
            "partial_path": str(partial),
            "output_path": str(final),
            "manifest_path": str(manifest_file),
            "ffmpeg": dict(ffmpeg_receipt),
            "ffprobe": dict(ffprobe_receipt),
            "mp4_artifact": artifact,
            "mux_timing": {
                "total_seconds": time.perf_counter() - started,
                "ffmpeg_seconds": ffmpeg_receipt.get("elapsed_seconds"),
                "ffprobe_seconds": ffprobe_receipt.get("elapsed_seconds"),
                "executed": True,
            },
            "retry_suppressed": True,
            "invocation_counts": {"ffmpeg": 1, "ffprobe": 1},
        }
    except MP4MuxFailure:
        raise
    except BaseException as exc:
        _raise_mp4_mux_failure(
            exc,
            partial=partial,
            manifest_path=manifest_file,
            manifest_created=manifest_created,
            frames_directory=frames,
            wav_path=wav,
            gate=launch_gate,
            ffmpeg_receipt=ffmpeg_receipt,
            ffprobe_receipt=ffprobe_receipt,
            source_validation=source_validation,
            started=started,
        )
    raise AssertionError("unreachable MP4 mux path")


def _mux_failure_report(
    report: dict[str, Any],
    *,
    gate: Mapping[str, Any],
    primary: BaseException,
    cleanup: BaseException | None,
    receipt: Mapping[str, Any],
    timing: Mapping[str, Any],
) -> dict[str, Any]:
    report["status"] = "failed"
    report["run_state"] = "failed"
    report["functional_success"] = False
    report["mp4_mux_status"] = "failed"
    report["mp4_mux"] = {
        "status": "failed",
        "invoked": receipt.get("invocation_counts", {}).get("ffmpeg", 0) > 0,
        "output_path": report.get("output_paths", {}).get("mp4") if isinstance(report.get("output_paths"), Mapping) else None,
        "partial_path": receipt.get("partial_path"),
        "manifest_path": receipt.get("manifest_path"),
        "launch_gate": dict(gate),
        "ffmpeg": receipt.get("ffmpeg"),
        "ffprobe": receipt.get("ffprobe"),
        "retry_suppressed": True,
        "invocation_counts": receipt.get("invocation_counts", {}),
    }
    report["mp4_artifact"] = None
    report["mux_timing"] = dict(timing)
    report["mux_failure"] = dict(receipt)
    report["failure"] = {
        "active_phase": "mp4-mux",
        "worker_identity": "parent",
        "completed_stages": list(report.get("phase_order", [])),
        "primary_error": error_receipt(primary),
        "cleanup_error": error_receipt(cleanup),
        **failure_fields(
            primary,
            cleanup,
            cleanup_attempted=receipt.get("cleanup_attempted") is True,
            cleanup_succeeded=(cleanup is None if receipt.get("cleanup_attempted") is True else False),
        ),
        "later_phase_suppression": {
            "derived_worker_suppressed": False,
            "decoder_suppressed": False,
            "media_suppressed": True,
            "retry_suppressed": True,
        },
    }
    report["standalone_media_status"] = "completed"
    standalone = dict(report.get("standalone_media") or {})
    standalone.update(
        {
            "status": "completed",
            "latent_generation_status": report.get("latent_generation_status"),
            "video_status": report.get("video_status"),
            "audio_status": report.get("audio_status"),
            "standalone_media_status": report.get("standalone_media_status"),
            "mp4_mux_status": report.get("mp4_mux_status"),
        }
    )
    report["standalone_media"] = standalone
    refresh_canonical_timing_eligibility(report)
    return report


def apply_mp4_mux_report(
    report: dict[str, Any],
    paths: Mapping[str, Any],
    *,
    subprocess_runner: Callable[..., Any] | None = None,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    timeout_seconds: float = 120.0,
    rename: Callable[[Path, Path], None] | None = None,
) -> dict[str, Any]:
    """Apply one mux attempt to a report, preserving standalone evidence on every failure."""
    started = time.perf_counter()
    gate = build_mux_launch_gate(report, paths)
    report["mp4_mux"] = {
        "status": "suppressed",
        "invoked": False,
        "output_path": None,
        "launch_gate": gate,
        "retry_suppressed": True,
        "invocation_counts": {"ffmpeg": 0, "ffprobe": 0},
    }
    report["mp4_artifact"] = None
    report["mux_failure"] = None
    report["mux_timing"] = {"total_seconds": 0.0, "executed": False}
    report["mp4_mux_status"] = "suppressed"
    if gate.get("passed") is not True:
        primary = ValueError("MP4 mux launch gate did not pass")
        receipt = {
            "status": "suppressed",
            "launch_gate": gate,
            "partial_cleanup_policy": "delete_partial_mp4_after_failure",
            "retry_suppressed": True,
            "invocation_counts": {"ffmpeg": 0, "ffprobe": 0},
            "primary_error": error_receipt(primary),
            "cleanup_error": None,
            "cleanup_attempted": False,
            "cleanup_succeeded": True,
            "frames_preserved": True,
            "wav_preserved": True,
        }
        report["mux_failure"] = receipt
        report["failure"] = {
            "active_phase": "mp4-mux-launch-gate",
            "worker_identity": "parent",
            "completed_stages": list(report.get("phase_order", [])),
            "primary_error": error_receipt(primary),
            "cleanup_error": None,
            **failure_fields(primary, None, cleanup_attempted=False, cleanup_succeeded=False),
            "later_phase_suppression": {
                "derived_worker_suppressed": False,
                "decoder_suppressed": False,
                "media_suppressed": True,
                "retry_suppressed": True,
            },
        }
        report["status"] = "failed"
        report["run_state"] = "failed"
        standalone = dict(report.get("standalone_media") or {})
        standalone["mp4_mux_status"] = "suppressed"
        report["standalone_media"] = standalone
        report["mux_timing"] = {"total_seconds": time.perf_counter() - started, "executed": False}
        return report

    report.setdefault("invocation", {})["mp4_mux_attempts"] = 1
    try:
        result = execute_mp4_mux(
            frames_directory=Path(paths["frames"]),
            video_manifest_path=Path(paths["video_frame_manifest"]),
            wav_path=Path(paths["audio_wav"]),
            audio_manifest_path=Path(paths["audio_manifest"]),
            mp4_partial_path=Path(paths["mp4_partial"]),
            mp4_final_path=Path(paths["mp4"]),
            mp4_manifest_path=Path(paths["mp4_manifest"]),
            attempt_identifier=str(paths["attempt_identifier"]),
            launch_gate=gate,
            subprocess_runner=subprocess_runner,
            ffmpeg_binary=ffmpeg_binary,
            ffprobe_binary=ffprobe_binary,
            timeout_seconds=timeout_seconds,
            rename=rename,
        )
    except MP4MuxFailure as exc:
        timing = {
            "total_seconds": time.perf_counter() - started,
            "ffmpeg_seconds": exc.receipt.get("ffmpeg", {}).get("elapsed_seconds"),
            "ffprobe_seconds": exc.receipt.get("ffprobe", {}).get("elapsed_seconds"),
            "executed": any(exc.receipt.get(tool, {}).get("invoked") is True for tool in ("ffmpeg", "ffprobe")),
        }
        return _mux_failure_report(
            report,
            gate=gate,
            primary=exc.primary_error,
            cleanup=exc.cleanup_error,
            receipt=exc.receipt,
            timing=timing,
        )
    except BaseException as exc:
        receipt = {
            "status": "failed",
            "launch_gate": gate,
            "partial_cleanup_policy": "delete_partial_mp4_after_failure",
            "retry_suppressed": True,
            "invocation_counts": {"ffmpeg": 0, "ffprobe": 0},
            "primary_error": error_receipt(exc),
            "cleanup_error": None,
            "cleanup_attempted": False,
            "cleanup_succeeded": False,
            "frames_preserved": Path(paths["frames"]).is_dir(),
            "wav_preserved": Path(paths["audio_wav"]).is_file(),
            "partial_path": str(Path(paths["mp4_partial"]).resolve()),
            "manifest_path": str(Path(paths["mp4_manifest"]).resolve()),
        }
        return _mux_failure_report(
            report,
            gate=gate,
            primary=exc,
            cleanup=None,
            receipt=receipt,
            timing={"total_seconds": time.perf_counter() - started, "executed": False},
        )

    report["mp4_mux_status"] = "completed"
    report["mp4_mux"] = dict(result)
    report["mp4_artifact"] = result["mp4_artifact"]
    report["mux_failure"] = None
    report["mux_timing"] = dict(result["mux_timing"])
    standalone = dict(report.get("standalone_media") or {})
    standalone["mp4_mux_status"] = "completed"
    report["standalone_media"] = standalone
    return report


def collect_audio_wav_evidence(wav_path: Path) -> dict[str, Any] | None:
    path = Path(wav_path).resolve()
    if not path.is_file():
        return None
    evidence: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        evidence["wav_sha256"] = sha256_file(path)
    except OSError as exc:
        evidence["checksum_error"] = error_receipt(exc)
    return evidence


def execute_audio_decode_once(
    *,
    artifact_path: Path,
    metadata_path: Path,
    expected_attempt_identifier: str,
    expected_checkpoint_identity: Mapping[str, Any],
    audio_root: Path,
    wav_partial: Path,
    wav_final: Path,
    manifest_path: Path,
    mx: Any,
    load_audio_config: Callable[[Path], Any],
    load_audio_vae: Callable[[Path], Any],
    save_wav: Callable[[Path, np.ndarray, int], Any] | None = None,
    materialize: Callable[[Any], None] | None = None,
    memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
    references: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run exactly one injected audio decode with strict pre-load and staged-publication gates."""
    refs = references if references is not None else {}
    input_validation = validate_final_audio_input_artifact(
        artifact_path,
        metadata_path,
        expected_attempt_identifier=expected_attempt_identifier,
        expected_checkpoint_identity=expected_checkpoint_identity,
    )
    partial = Path(wav_partial).resolve()
    final = Path(wav_final).resolve()
    manifest_file = Path(manifest_path).resolve()
    if final.exists():
        raise FileExistsError(f"refusing existing final audio WAV: {final}")
    if partial.exists():
        raise FileExistsError(f"refusing existing staged audio WAV: {partial}")
    config = load_audio_config(Path(audio_root).resolve())
    refs["config"] = config
    geometry = validate_locked_audio_config(config)
    memory: dict[str, Any] = {}
    snapshot = memory_snapshot or (lambda: _memory_snapshot(mx))
    memory["before_load"] = dict(snapshot())
    stored_audio = input_validation["arrays"]["final_audio_native"]
    refs["stored_latent"] = stored_audio
    latent, logical_fingerprint = restore_audio_latent_logical_bfloat16(
        stored_audio,
        mx,
        expected_fingerprint=input_validation["audio_fingerprint"],
        materialize=materialize,
    )
    refs["latent"] = latent
    decoder_input = normalize_audio_latent_for_decode(latent, config, mx, materialize=materialize)
    refs["decoder_input"] = decoder_input
    if _dtype_name(getattr(decoder_input, "dtype", None)) != _dtype_name(mx.float32):
        raise ValueError("audio decoder input must be materialized as float32")
    decoder = load_audio_vae(Path(audio_root).resolve())
    refs["decoder"] = decoder
    memory["after_load"] = dict(snapshot())
    raw = decoder.decode(decoder_input)
    refs["raw"] = raw
    raw_np, raw_shape_dtype = materialize_and_validate_audio_raw_output(raw, mx, materialize=materialize)
    refs["raw_np"] = raw_np
    waveform, waveform_shape_dtype = convert_and_validate_audio_waveform(raw_np)
    refs["waveform"] = waveform
    memory["peak"] = dict(snapshot())
    if save_wav is None:
        from minimax_h3_mlx.media import save_wav as save_wav_impl

        save_wav = save_wav_impl
    save_wav(partial, waveform, AUDIO_SAMPLE_RATE)
    manifest_validation = publish_audio_wav_atomically(
        partial,
        final,
        manifest_file,
        attempt_identifier=expected_attempt_identifier,
    )
    return {
        "input_artifact": {key: value for key, value in input_validation.items() if key not in {"arrays", "metadata"}},
        "audio_geometry": geometry,
        "logical_audio_fingerprint": logical_fingerprint,
        "raw_shape_dtype": raw_shape_dtype,
        "waveform_shape_dtype": waveform_shape_dtype,
        "wav_manifest": manifest_validation,
        "memory": memory,
        "references": refs,
    }


def release_audio_decoder(
    mx: Any,
    references: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    tolerance: int,
    *,
    memory_snapshot: Callable[[], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Drop every audio reference, purge the MLX allocator, and return the strict release gate."""
    snapshot = memory_snapshot or (lambda: _memory_snapshot(mx))
    before_release = dict(snapshot())
    if isinstance(references, dict):
        for key in list(references):
            references[key] = None
    release = _release_runtime(mx, references, baseline, tolerance)
    release["memory_before_release"] = before_release
    release["memory_after_release"] = release.get("memory_after_allocator_purge")
    return release


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def classify_host_process(
    process: Mapping[str, Any],
    *,
    current_pid: int,
) -> dict[str, Any]:
    """Classify only explicit compute workloads; ordinary GPU-composited apps stay clear."""
    pid = process.get("pid")
    ppid = process.get("ppid")
    process_name = str(process.get("process_name", ""))
    command = str(process.get("command", ""))
    searchable = f"{process_name} {command}".lower()
    executable = Path(process_name).name.lower()

    if pid == current_pid:
        return {
            "outcome": "expected_harness_process",
            "known_conflict": False,
            "rule": "current-harness-pid",
            "reason": "current proof-harness process is not a pre-existing conflict",
        }
    if (
        ppid == current_pid
        and Path(__file__).name.lower() in searchable
        and any(marker in command for marker in EXPECTED_HARNESS_CHILD_MARKERS)
    ):
        return {
            "outcome": "expected_harness_process",
            "known_conflict": False,
            "rule": "expected-harness-child",
            "reason": "expected proof-harness worker is not a pre-existing conflict",
        }

    model_server_markers = (
        "llama-server",
        "text-generation-launcher",
        "vllm.entrypoints",
        "mlx_lm.server",
        "mlx-vlm.server",
    )
    if any(marker in searchable for marker in model_server_markers) or (
        executable == "ollama" and any(marker in searchable for marker in (" serve", " runner"))
    ):
        return {
            "outcome": "known_conflict",
            "known_conflict": True,
            "rule": "model-server",
            "reason": "known model-server process may contend for unified memory or compute",
        }

    generation_markers = (
        "comfyui",
        "stable-diffusion-webui",
        "automatic1111",
        "invokeai",
        "diffusionbee",
        "drawthings",
        "draw things",
        "mlx_lm.generate",
        "mlx-vlm.generate",
        "mlx_audio.tts.generate",
    )
    if any(marker in searchable for marker in generation_markers):
        return {
            "outcome": "known_conflict",
            "known_conflict": True,
            "rule": "known-generation-workload",
            "reason": "known image, video, audio, or language-model workload may be active",
        }

    python_process = executable.startswith("python") or " python" in searchable
    workload_markers = ("generate", "generation", "train", "training", "infer", "inference", "probe", "worker")
    if python_process and "mlx" in searchable and any(marker in searchable for marker in workload_markers):
        return {
            "outcome": "known_conflict",
            "known_conflict": True,
            "rule": "python-mlx-workload",
            "reason": "Python command identifies an MLX generation, training, inference, probe, or worker workload",
        }
    if python_process and any(marker in searchable for marker in ("--device mps", "device=mps", "metal compute")) and any(
        marker in searchable for marker in workload_markers
    ):
        return {
            "outcome": "known_conflict",
            "known_conflict": True,
            "rule": "known-metal-compute-workload",
            "reason": "Python command identifies an MPS or Metal compute workload",
        }

    return {
        "outcome": "no_known_conflict",
        "known_conflict": False,
        "rule": None,
        "reason": "no narrow known-conflict rule matched",
    }


def capture_host_process_snapshot(
    *,
    runner: Callable[..., Any] | None = None,
    current_pid: int | None = None,
    capture_timestamp: str | None = None,
    timeout_seconds: float = HOST_PROCESS_SNAPSHOT_TIMEOUT_SECONDS,
    max_processes: int = HOST_PROCESS_SNAPSHOT_MAX_PROCESSES,
) -> dict[str, Any]:
    """Capture one bounded, read-only process snapshot without turning failure into a run failure."""
    run = runner or subprocess.run
    harness_pid = os.getpid() if current_pid is None else current_pid
    command = list(HOST_PROCESS_SNAPSHOT_COMMAND)
    snapshot: dict[str, Any] = {
        "capture_timestamp": capture_timestamp or _utc_timestamp(),
        "tool": "ps",
        "command": command,
        "read_only": True,
        "timeout_seconds": timeout_seconds,
        "max_processes": max_processes,
        "capture_success": False,
        "failure": None,
        "current_process_pid": harness_pid,
        "process_count_scanned": 0,
        "processes": [],
        "known_conflicting_processes": [],
        "truncated": False,
        "automatic_scan_proves_absolute_idleness": False,
    }
    try:
        completed = run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        snapshot["failure"] = error_receipt(exc)
        return snapshot
    if completed.returncode != 0:
        snapshot["failure"] = {
            "type": "ProcessSnapshotCommandError",
            "message": f"ps exited with status {completed.returncode}: {str(completed.stderr).strip()}",
            "traceback": "",
        }
        return snapshot

    lines = [line for line in str(completed.stdout).splitlines() if line.strip()]
    if len(lines) > max_processes:
        snapshot["truncated"] = True
        lines = lines[:max_processes]
    malformed_lines = 0
    current_process_present = False
    processes: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for line in lines:
        columns = line.strip().split(None, 3)
        if len(columns) < 3:
            malformed_lines += 1
            continue
        try:
            pid = int(columns[0])
            ppid = int(columns[1])
        except ValueError:
            malformed_lines += 1
            continue
        process_name = columns[2]
        full_command = columns[3] if len(columns) == 4 else process_name
        command_truncated = len(full_command) > HOST_PROCESS_COMMAND_MAX_CHARACTERS
        recorded_command = full_command[:HOST_PROCESS_COMMAND_MAX_CHARACTERS]
        entry: dict[str, Any] = {
            "pid": pid,
            "ppid": ppid,
            "process_name": process_name,
            "command": recorded_command,
            "command_truncated": command_truncated,
        }
        classification = classify_host_process(
            {**entry, "command": full_command},
            current_pid=harness_pid,
        )
        entry["classification"] = classification
        processes.append(entry)
        current_process_present = current_process_present or pid == harness_pid
        if classification["known_conflict"] is True:
            conflicts.append(
                {
                    "pid": pid,
                    "ppid": ppid,
                    "process_name": process_name,
                    "command": recorded_command,
                    "command_truncated": command_truncated,
                    "classification_rule": classification["rule"],
                    "classification_reason": classification["reason"],
                }
            )
    snapshot["process_count_scanned"] = len(processes)
    snapshot["processes"] = processes
    snapshot["known_conflicting_processes"] = conflicts
    snapshot["malformed_line_count"] = malformed_lines
    snapshot["current_process_present"] = current_process_present
    if snapshot["truncated"]:
        snapshot["failure"] = {
            "type": "ProcessSnapshotTruncated",
            "message": f"process snapshot exceeded the bound of {max_processes} processes",
            "traceback": "",
        }
    elif malformed_lines:
        snapshot["failure"] = {
            "type": "ProcessSnapshotParseError",
            "message": f"process snapshot contained {malformed_lines} malformed rows",
            "traceback": "",
        }
    elif not current_process_present:
        snapshot["failure"] = {
            "type": "ProcessSnapshotIncomplete",
            "message": "process snapshot did not contain the current harness PID",
            "traceback": "",
        }
    else:
        snapshot["capture_success"] = True
    return snapshot


def canonical_timing_eligibility(
    *,
    functional_success: bool,
    operator_declared_uncontended: bool,
    process_snapshot_captured: bool,
    known_conflicting_processes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the exact four-gate canonical timing eligibility formula."""
    reasons: list[str] = []
    if functional_success is not True:
        reasons.append("functional_success is not true")
    if operator_declared_uncontended is not True:
        reasons.append("operator_declared_uncontended is not true")
    if process_snapshot_captured is not True:
        reasons.append("process_snapshot_captured is not true")
    if list(known_conflicting_processes):
        reasons.append("known_conflicting_processes is not empty")
    return {
        "canonical_timing_eligible": not reasons,
        "canonical_timing_ineligibility_reasons": reasons,
    }


def apply_host_process_snapshot(report: dict[str, Any], snapshot: Mapping[str, Any]) -> None:
    """Attach snapshot evidence and refresh eligibility without changing functional status."""
    host = report["host_contention"]
    host["process_snapshot"] = _json_safe(dict(snapshot))
    host["process_snapshot_captured"] = snapshot.get("capture_success") is True
    conflicts = snapshot.get("known_conflicting_processes")
    host["known_conflicting_processes"] = list(conflicts) if isinstance(conflicts, list) else []
    refresh_canonical_timing_eligibility(report)


def refresh_canonical_timing_eligibility(report: dict[str, Any]) -> None:
    host = report.get("host_contention")
    if not isinstance(host, dict):
        return
    host.update(
        canonical_timing_eligibility(
            functional_success=report.get("functional_success") is True,
            operator_declared_uncontended=host.get("operator_declared_uncontended") is True,
            process_snapshot_captured=host.get("process_snapshot_captured") is True,
            known_conflicting_processes=host.get("known_conflicting_processes", []),
        )
    )


def validate_derived_filesystem(derived_root: Path) -> dict[str, Any]:
    """Validate only small metadata and payload names; never opens a tensor payload."""
    root = derived_root.resolve()
    required = [
        root / "config.json",
        root / "conversion_manifest.json",
        root / "quant_config.json",
        root / "base" / "model.safetensors.index.json",
        root / "adaln" / "manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("derived cache-only checkpoint metadata is missing: " + "; ".join(missing))
    conversion = _read_json_object(root / "conversion_manifest.json", "derived conversion manifest")
    if conversion.get("format_identifier") != DERIVED_FORMAT_IDENTIFIER:
        raise ValueError("derived checkpoint format identifier is not the repository streamed-AdaLN format")
    if conversion.get("schema_version") != DERIVED_SCHEMA_VERSION or conversion.get("bounded") is not False:
        raise ValueError("derived checkpoint schema or bounded flag is not loadable")
    if conversion.get("verification_status") != "verified":
        raise ValueError("derived checkpoint is not verified")
    if conversion.get("selected_blocks") != list(range(EXPECTED_BLOCK_COUNT)):
        raise ValueError("derived checkpoint does not contain all 50 selected blocks")
    manifest_base_tensor_count = conversion.get("derived_base_tensor_count")
    if manifest_base_tensor_count != EXPECTED_DERIVED_BASE_TENSOR_COUNT:
        raise ValueError(
            "derived conversion manifest base tensor count is "
            f"{manifest_base_tensor_count!r}, expected {EXPECTED_DERIVED_BASE_TENSOR_COUNT}"
        )
    base_index = _read_json_object(root / "base" / "model.safetensors.index.json", "derived base tensor index")
    weight_map = base_index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()):
        raise ValueError("derived base tensor index weight_map is invalid")
    index_base_tensor_count = len(weight_map)
    if index_base_tensor_count != EXPECTED_DERIVED_BASE_TENSOR_COUNT:
        raise ValueError(
            "derived base tensor index count is "
            f"{index_base_tensor_count}, expected {EXPECTED_DERIVED_BASE_TENSOR_COUNT}"
        )
    if manifest_base_tensor_count != index_base_tensor_count:
        raise ValueError("derived base tensor count differs between conversion manifest and base index")
    sidecar_manifest = _read_json_object(root / "adaln" / "manifest.json", "AdaLN sidecar manifest")
    if sidecar_manifest.get("format_identifier") != DERIVED_FORMAT_IDENTIFIER or sidecar_manifest.get("schema_version") != DERIVED_SCHEMA_VERSION:
        raise ValueError("AdaLN sidecar manifest identity is invalid")
    if sidecar_manifest.get("bounded") is not False:
        raise ValueError("bounded AdaLN sidecar manifests cannot serve the full schedule")
    blocks = sidecar_manifest.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {str(index) for index in range(EXPECTED_BLOCK_COUNT)}:
        raise ValueError("AdaLN sidecar manifest must describe exactly blocks 0..49")
    missing_sidecars = [
        str(root / "adaln" / f"block-{index:03d}.safetensors")
        for index in range(EXPECTED_BLOCK_COUNT)
        if not (root / "adaln" / f"block-{index:03d}.safetensors").is_file()
    ]
    if missing_sidecars:
        raise FileNotFoundError("derived sidecar payload files are missing: " + "; ".join(missing_sidecars))
    small_files = required + [root / "adaln" / "manifest.json"]
    return {
        "checkpoint_format": "derived",
        "construction_mode": "cache_only",
        "format_identifier": DERIVED_FORMAT_IDENTIFIER,
        "schema_version": DERIVED_SCHEMA_VERSION,
        "metadata_files": {str(path.relative_to(root)): sha256_file(path) for path in small_files},
        "selected_blocks": list(range(EXPECTED_BLOCK_COUNT)),
        "derived_base_tensor_count": index_base_tensor_count,
        "payloads_opened": False,
    }


def capture_git_identity() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        completed = subprocess.run(["git", *arguments], cwd=str(ROOT), capture_output=True, text=True, check=False)
        return completed.stdout.strip() if completed.returncode == 0 else ""

    relation = run("rev-list", "--left-right", "--count", "@{upstream}...HEAD")
    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "upstream": run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}"),
        "upstream_relation": relation,
        "status_short": run("status", "--short"),
        "status_lines": [line for line in run("status", "--short").splitlines() if line],
    }


def _checkpoint_identity(root: Path, derived: Path, preflight: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_root": str(root.resolve()),
        "derived_transformer": str(derived.resolve()),
        "derived_filesystem": dict(preflight),
    }


def ensure_attempt_namespace(root: Path, attempt_identifier: str | None = None) -> dict[str, Any]:
    root = root.expanduser().resolve()
    newly_created = not root.exists()
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"attempt output path is not a directory: {root}")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"refusing to reuse nonempty v0.5d attempt namespace: {root}")
    root.mkdir(parents=True, exist_ok=True)
    identifier = attempt_identifier or root.name
    return {
        "root": str(root),
        "attempt_identifier": identifier,
        "namespace_newly_created": newly_created,
        "report": str(root / "derived-full-schedule-report.json"),
        "process_snapshot": str(root / "host-process-snapshot.json"),
        "conditioning_artifact": str(root / "conditioning-artifact.npz"),
        "conditioning_receipt": str(root / "conditioning-worker-receipt.json"),
        "conditioning_log": str(root / "conditioning-worker.log"),
        "final_artifact": str(root / "final-native-latent.npz"),
        "final_artifact_metadata": str(root / "final-native-latent.json"),
        "derived_receipt": str(root / "derived-worker-receipt.json"),
        "derived_log": str(root / "derived-worker.log"),
        "event_file": str(root / "streamed-adaln-events.jsonl"),
        "frames_partial": str(root / "frames.partial"),
        "frames": str(root / "frames"),
        "video_frame_manifest": str(root / "video-frame-manifest.json"),
        "video_worker_receipt": str(root / "video-worker-receipt.json"),
        "video_worker_log": str(root / "video-worker.log"),
        "audio_partial": str(root / "audio.partial.wav"),
        "audio_wav": str(root / "audio.wav"),
        "audio_manifest": str(root / "audio-manifest.json"),
        "audio_worker_receipt": str(root / "audio-worker-receipt.json"),
        "audio_worker_log": str(root / "audio-worker.log"),
        "mp4_partial": str(root / MP4_PARTIAL_FILENAME),
        "mp4": str(root / MP4_FINAL_FILENAME),
        "mp4_manifest": str(root / MP4_MANIFEST_FILENAME),
    }


def _base_report(args: argparse.Namespace, paths: Mapping[str, Any]) -> dict[str, Any]:
    operator_declared_uncontended = getattr(args, "operator_declared_uncontended", False)
    if type(operator_declared_uncontended) is not bool:
        raise TypeError("operator_declared_uncontended must be a literal boolean declaration")
    report = {
        "status": "incomplete",
        "run_state": "incomplete",
        "functional_success": False,
        "schema_version": SCHEMA_VERSION,
        "probe_identity": PROBE_FORMAT,
        "attempt": {
            "attempt_identifier": paths["attempt_identifier"],
            "namespace_newly_created": paths["namespace_newly_created"],
            "root": paths["root"],
        },
        "invocation": {
            "attempts": 1,
            "conditioning_worker_attempts": 0,
            "derived_worker_attempts": 0,
            "video_worker_attempts": 0,
            "audio_worker_attempts": 0,
            "mp4_mux_attempts": 0,
            "retry_policy": {
                "one_invocation_attempt": True,
                "one_conditioning_worker_attempt": True,
                "one_derived_worker_attempt": True,
                "internal_retry_loop": False,
                "automatic_worker_replacement": False,
                "recursive_restart": False,
                "second_transformer_load": False,
                "second_generation_attempt": False,
                "one_mp4_mux_attempt": True,
            },
            "host_command": shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        },
        "git_identity": {},
        "checkpoint_identity": {},
        "prompt": {
            "text": getattr(args, "prompt", ""),
            "utf8_byte_count": len(str(getattr(args, "prompt", "")).encode("utf-8")),
            "sha256": hashlib.sha256(str(getattr(args, "prompt", "")).encode("utf-8")).hexdigest(),
            "prompt_is_literal": getattr(args, "prompt", None) == LOCKED_PROMPT,
            "negative_prompt": None,
            "image_conditioning": False,
            "chat_template": None,
            "special_tokens": False,
        },
        "geometry": canonical_geometry_contract(),
        "packing": packed_contract(),
        "schedule_contract": {},
        "conditioning_worker": {},
        "derived_worker": {},
        "denoising": {},
        "streamed_adaln_lifecycle": {},
        "cache_attribution": {},
        "final_artifact": {},
        "decoder_phase": {
            "status": "not_started",
            "reason": "Slice 3C implements one video worker followed by one audio worker",
            "implemented_scope": "video_then_audio",
            "implemented_phase_scope": {"video": True, "audio": True},
            "worker_launches": {"video": 0, "audio": 0},
            "retry_allowed": False,
            "replacement_worker_allowed": False,
        },
        "video_decoder": _decoder_section("video"),
        "audio_decoder": _decoder_section("audio"),
        "video_artifacts": {},
        "audio_artifacts": {},
        "decoder_memory": {},
        "decoder_timing": {},
        "decoder_phase_order": decoder_phase_order_receipt(
            [],
            phase_status={phase: "not_started" for phase in DECODER_PHASE_ORDER},
        ),
        "decoder_failure": None,
        "standalone_media": {
            "status": "not_started",
            "latent_generation_status": "incomplete",
            "video_status": "not_started",
            "audio_status": "not_started",
            "standalone_media_status": "not_started",
            "mp4_mux_status": "not_performed",
        },
        "mp4_mux": {"status": "not_performed", "invoked": False, "output_path": None},
        "mp4_artifact": None,
        "mux_timing": {},
        "mux_failure": None,
        "latent_generation_status": "incomplete",
        "video_status": "not_started",
        "audio_status": "not_started",
        "standalone_media_status": "not_started",
        "mp4_mux_status": "not_performed",
        "event_file_path": paths["event_file"],
        "event_file_record_count": 0,
        "total_event_records": 0,
        "cache_session_count": 0,
        "sidecar_open_event_count": 0,
        "sidecar_release_event_count": 0,
        "validated_block_pairs": 0,
        "event_file_sha256": None,
        "memory_telemetry": {},
        "timing_telemetry": {
            "parent_orchestration_seconds": None,
            "schedule_construction_seconds": None,
            "conditioning": {},
            "derived": {},
            "final_serialization_seconds": None,
        },
        "host_contention": {
            "operator_declared_uncontended": operator_declared_uncontended,
            "process_snapshot_captured": False,
            "process_snapshot_path": paths["process_snapshot"],
            "process_snapshot": None,
            "known_conflicting_processes": [],
            "canonical_timing_eligible": False,
            "canonical_timing_ineligibility_reasons": [],
        },
        "phase_order": [],
        "output_paths": dict(paths),
        "generation_exclusions": dict(GENERATION_EXCLUSIONS),
        "failure": None,
    }
    refresh_canonical_timing_eligibility(report)
    return report


def _worker_base(identity: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "worker_identity": identity,
        "pid": os.getpid(),
        "worker_started": True,
        "worker_exit_observed": False,
        "worker_exit_code": None,
        "worker_termination_confirmed": False,
        "worker_receipt_valid": False,
        "completed_stages": [],
        "primary_error": None,
        "cleanup_error": None,
        "memory_telemetry": {},
        "timing_telemetry": {},
    }


def _write_worker_failure(
    path: Path,
    receipt: dict[str, Any],
    primary: BaseException | Mapping[str, Any],
    cleanup: BaseException | Mapping[str, Any] | None,
    *,
    cleanup_attempted: bool,
) -> None:
    receipt["status"] = "failed"
    receipt["primary_error"] = error_receipt(primary)
    receipt["cleanup_error"] = error_receipt(cleanup)
    receipt.update(failure_fields(primary, cleanup, cleanup_attempted=cleanup_attempted, cleanup_succeeded=cleanup is None if cleanup_attempted else False))
    _write_json(path, receipt)


def _clear_exception_frames(error: BaseException | None) -> None:
    """Drop traceback-held decoder locals before a release gate measures residency."""
    if error is None:
        return
    try:
        traceback.clear_frames(error.__traceback__)
    except BaseException:
        pass


def _detach_exception_graph(error: BaseException) -> None:
    """Clear traceback frames and exception links that can retain runtime objects."""
    pending = [error]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        cause = current.__cause__
        context = current.__context__
        _clear_exception_frames(current)
        try:
            current.__traceback__ = None
            current.__cause__ = None
            current.__context__ = None
            current.__suppress_context__ = True
        except BaseException:
            pass
        pending.extend(item for item in (cause, context) if isinstance(item, BaseException))


def serialize_and_detach_error(
    error: BaseException | Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Capture complete error evidence, then detach live exception traceback state."""
    if error is None:
        return None
    receipt = error_receipt(error)
    if isinstance(error, BaseException):
        _detach_exception_graph(error)
    return receipt


def serialize_and_detach_failure(
    primary: BaseException | Mapping[str, Any] | None,
    cleanup: BaseException | Mapping[str, Any] | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Serialize primary and cleanup independently before runtime release is measured."""
    return serialize_and_detach_error(primary), serialize_and_detach_error(cleanup)


def preserve_primary_on_release_failure(
    primary: BaseException | Mapping[str, Any] | None,
    cleanup: BaseException | Mapping[str, Any] | None,
    release_error: BaseException,
) -> tuple[BaseException | Mapping[str, Any] | None, BaseException | Mapping[str, Any] | None]:
    """Represent a release failure without replacing an already captured primary error."""
    release_receipt = serialize_and_detach_error(release_error)
    if primary is None:
        return release_receipt, cleanup
    if cleanup is None:
        return primary, release_receipt
    return primary, cleanup


def _conditioning_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--derived-transformer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _derived_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--derived-transformer", required=True)
    parser.add_argument("--attempt-identifier", required=True)
    parser.add_argument("--conditioning-artifact", required=True)
    parser.add_argument("--conditioning-receipt", required=True)
    parser.add_argument("--final-artifact", required=True)
    parser.add_argument("--final-artifact-metadata", required=True)
    parser.add_argument("--event-file", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _conditioning_worker_main(argv: Sequence[str]) -> int:
    args = _conditioning_worker_parser(argv)
    receipt = _worker_base("conditioning")
    receipt_path = Path(args.receipt).resolve()
    artifact_path = Path(args.artifact).resolve()
    mx = None
    references: dict[str, Any] = {}
    baseline = None
    encoder = input_ids = token_tags = conditioning = video_native = audio_native = None
    layout = geometry = schedule = None
    primary: BaseException | Mapping[str, Any] | None = None
    cleanup: BaseException | Mapping[str, Any] | None = None
    release: dict[str, Any] = {}
    started = time.perf_counter()
    try:
        validate_locked_prompt(args.prompt)
        validate_seed(args.seed)
        import mlx.core as mx

        from minimax_h3_mlx.config import DiTConfig
        from minimax_h3_mlx.geometry import ProductionMultimodalGeometry
        from minimax_h3_mlx.load import load_audio_vae_config, load_video_vae_config
        from minimax_h3_mlx.packing import build_packed_sequence
        from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
        from minimax_h3_mlx.video_decode_layout import resolve_video_decode_layout

        root = Path(args.checkpoint_root).resolve()
        derived = Path(args.derived_transformer).resolve()
        baseline = _memory_snapshot(mx)
        receipt["memory_telemetry"]["before_conditioning_load"] = baseline
        encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=False, verbose=args.verbose)
        references["encoder"] = encoder
        input_ids, token_tags, vision_inputs = encoder.build_request(args.prompt, None)
        if vision_inputs is not None:
            raise ValueError("locked conditioning must not contain image conditioning")
        conditioning, encoded_tags = encoder.encode(args.prompt, None)
        if list(np.asarray(token_tags)) != list(np.asarray(encoded_tags)):
            raise ValueError("token tags changed between request construction and conditioning")
        token_count = int(input_ids.shape[1])
        if token_count != EXPECTED_TOKEN_COUNT:
            raise ValueError(f"locked prompt token count is {token_count}, expected {EXPECTED_TOKEN_COUNT}")
        if tuple(conditioning.shape) != EXPECTED_CONDITIONING_SHAPE:
            raise ValueError(f"conditioning shape is {conditioning.shape}, expected {EXPECTED_CONDITIONING_SHAPE}")
        if _dtype_name(conditioning.dtype) != EXPECTED_CONDITIONING_DTYPE:
            raise ValueError("conditioning dtype is not bfloat16")
        receipt["completed_stages"].extend(["conditioning-worker-started", "text-encoder-loaded", "prompt-tokenized-and-conditioned"])

        mx.random.seed(args.seed)
        video_native = mx.random.normal(VIDEO_NATIVE_SHAPE).astype(mx.float32)
        audio_native = mx.random.normal(AUDIO_NATIVE_SHAPE).astype(mx.float32)
        mx.eval(conditioning, input_ids, video_native, audio_native)

        derived_config = DiTConfig.from_json(derived / "config.json")
        video_config = load_video_vae_config(root / "video_vae")
        audio_config = load_audio_vae_config(root / "audio_vae")
        layout_info = resolve_video_decode_layout(video_config)
        geometry = ProductionMultimodalGeometry.canonical(video_config, audio_config, derived_config, layout_info)
        if tuple(geometry.video_latent_shape) != VIDEO_NATIVE_SHAPE or tuple(geometry.audio_latent_shape) != AUDIO_NATIVE_SHAPE:
            raise ValueError("runtime geometry differs from locked 128x128/30-frame geometry")
        layout = build_packed_sequence(
            np.asarray(token_tags, dtype=np.int64),
            VIDEO_NATIVE_SHAPE[2],
            VIDEO_NATIVE_SHAPE[3],
            VIDEO_NATIVE_SHAPE[4],
            AUDIO_NATIVE_SHAPE[2],
            tuple(derived_config.patch_size),
            keyframe_anchors=(),
        )
        if int(layout.sequence_length) != EXPECTED_TOTAL_ROWS:
            raise ValueError(f"packed row count is {layout.sequence_length}, expected {EXPECTED_TOTAL_ROWS}")

        schedule_started = time.perf_counter()
        schedule_video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        schedule_audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        schedule_video.set_timesteps(REQUESTED_SIGMA_POINTS)
        schedule_audio.set_timesteps(REQUESTED_SIGMA_POINTS)
        schedule = schedule_plan_from_schedulers(schedule_video, schedule_audio)
        receipt["timing_telemetry"]["schedule_construction_seconds"] = time.perf_counter() - schedule_started

        token_ids_np = np.asarray(input_ids, dtype=np.int32)
        token_tags_np = np.asarray(token_tags, dtype=np.int32)
        video_np = _as_float32_numpy(video_native)
        audio_np = _as_float32_numpy(audio_native)
        token_presence_np = np.ones(token_ids_np.shape, dtype=np.int32)
        artifact_arrays = {
            "text_conditioning": _as_float32_numpy(conditioning),
            "token_ids": token_ids_np,
            "token_presence_mask": token_presence_np,
            "text_token_tags": token_tags_np,
            "initial_video_native": video_np,
            "initial_audio_native": audio_np,
            "packed_position_ids": np.asarray(layout.position_ids, dtype=np.float32),
            "packed_token_tags": np.asarray(layout.token_tags, dtype=np.int32),
            "packed_video_indices": np.asarray(layout.video_indices, dtype=np.int32),
            "packed_audio_indices": np.asarray(layout.audio_indices, dtype=np.int32),
            "packed_text_indices": np.asarray(layout.text_indices, dtype=np.int32),
        }
        _write_npz(artifact_path, artifact_arrays)
        packed = packed_contract(token_count)
        packed["position_ids_shape"] = list(layout.position_ids.shape)
        packed["token_tags_shape"] = list(layout.token_tags.shape)
        packed["video_indices_shape"] = list(layout.video_indices.shape)
        packed["audio_indices_shape"] = list(layout.audio_indices.shape)
        packed["text_indices_shape"] = list(layout.text_indices.shape)
        validate_packed_contract(packed)
        receipt["prompt"] = prompt_receipt(args.prompt, token_ids_np)
        receipt["tokenizer"] = {
            "entrypoint": "MiniMaxH3TextEncoder.tokenizer",
            "call": "tokenizer(prompt, add_special_tokens=False)",
            "add_special_tokens": False,
            "chat_template": None,
            "negative_prompt": None,
            "token_count": token_count,
            "token_ids": token_ids_np.tolist(),
            "token_presence_mask": token_presence_np.tolist(),
        }
        receipt["conditioning"] = {
            "shape": list(conditioning.shape),
            "dtype": EXPECTED_CONDITIONING_DTYPE,
            "fingerprint": array_fingerprint(conditioning, logical_dtype=EXPECTED_CONDITIONING_DTYPE),
            "expected_fingerprint": None,
            "token_count": token_count,
            "attention_mask_policy": "create_attention_mask(hidden_states, None)",
        }
        receipt["deterministic_inputs"] = deterministic_input_receipt(video_np, audio_np)
        receipt["geometry"] = canonical_geometry_contract()
        receipt["packing"] = packed
        receipt["schedule_contract"] = schedule.receipt()
        receipt["artifact_schema"] = {
            "identity": "minimax-h3-mlx-v05d-conditioning-artifact",
            "arrays": sorted(CONDITIONING_ARRAY_KEYS),
            "conditioning_released_before_derived_worker": True,
        }
        receipt["conditioning_artifact"] = conditioning_artifact_binding(artifact_path, artifact_arrays)
        validate_conditioning_artifact_binding(receipt, artifact_path)
        receipt["completed_stages"].extend(["metadata-and-input-artifact-written", "full-schedule-contract-constructed"])
    except BaseException as exc:
        primary = exc
    finally:
        if mx is not None:
            encoder = input_ids = token_tags = conditioning = video_native = audio_native = None
            layout = geometry = schedule = None
            release = _release_runtime(mx, references, baseline, args.tolerance)
            receipt["memory_telemetry"]["after_conditioning_release"] = release.get("memory_after_allocator_purge")
            receipt["conditioning_release"] = release
            if primary is None and release.get("passed") is not True:
                primary = RuntimeError("conditioning release and allocator-cache gate failed")
            elif primary is not None and release.get("passed") is not True:
                release_error = RuntimeError("conditioning cleanup/release gate failed")
                if cleanup is None:
                    cleanup = release_error
                else:
                    receipt.setdefault("additional_cleanup_errors", []).append(error_receipt(release_error))
    receipt["timing_telemetry"]["worker_total_seconds"] = time.perf_counter() - started
    if primary is not None:
        _write_worker_failure(receipt_path, receipt, primary, cleanup, cleanup_attempted=mx is not None)
        return 1
    receipt["status"] = "success"
    receipt["completed_stages"].append("conditioning-released-before-derived-worker")
    receipt["worker_receipt_valid"] = True
    _write_json(receipt_path, receipt)
    return 0


def _native_from_packed(video: Any, audio: Any, *, config: Any) -> tuple[Any, Any]:
    from minimax_h3_mlx.packing import unpatchify_video_tokens, unpack_audio_tokens

    native_video = unpatchify_video_tokens(
        video[0], VIDEO_NATIVE_SHAPE[2], VIDEO_NATIVE_SHAPE[3], VIDEO_NATIVE_SHAPE[4], VIDEO_NATIVE_SHAPE[1], tuple(config.patch_size)
    )
    native_audio = unpack_audio_tokens(audio[0], AUDIO_NATIVE_SHAPE[2])
    return native_video, native_audio


def _derived_worker_main(argv: Sequence[str]) -> int:
    args = _derived_worker_parser(argv)
    receipt = _worker_base("derived")
    receipt["cache_attribution"] = {}
    receipt_path = Path(args.receipt).resolve()
    final_artifact_path = Path(args.final_artifact).resolve()
    final_metadata_path = Path(args.final_artifact_metadata).resolve()
    mx = None
    references: dict[str, Any] = {}
    baseline = None
    dit = text = initial_video_native = initial_audio_native = video_rows = audio_rows = None
    packed = layout = schedule = scheduler = result = config = None
    primary: BaseException | None = None
    cleanup: BaseException | None = None
    provider: StreamedCacheSessionProvider | None = None
    event_writer = JsonlEventWriter(Path(args.event_file).resolve())
    started = time.perf_counter()
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        from minimax_h3_mlx.config import DiTConfig
        from minimax_h3_mlx.dit import CACHE_ONLY_CONSTRUCTION
        from minimax_h3_mlx.load import load_dit
        from minimax_h3_mlx.packing import build_row_timesteps, pack_audio_latents, patchify_video_latents
        from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler

        conditioning_receipt = _read_json_object(Path(args.conditioning_receipt), "conditioning worker receipt")
        validate_conditioning_receipt(conditioning_receipt)
        arrays = validate_conditioning_artifact_binding(conditioning_receipt, Path(args.conditioning_artifact))
        if tuple(arrays["initial_video_native"].shape) != VIDEO_NATIVE_SHAPE or tuple(arrays["initial_audio_native"].shape) != AUDIO_NATIVE_SHAPE:
            raise ValueError("conditioning artifact native latent shapes do not match the locked geometry")
        if tuple(arrays["text_conditioning"].shape) != EXPECTED_CONDITIONING_SHAPE:
            raise ValueError("conditioning artifact text shape is not (1,103,5120)")
        derived_root = Path(args.derived_transformer).resolve()
        config = DiTConfig.from_json(derived_root / "config.json")
        text = mx.array(arrays["text_conditioning"], dtype=mx.float32).astype(mx.bfloat16)
        initial_video_native = mx.array(arrays["initial_video_native"], dtype=mx.float32)
        initial_audio_native = mx.array(arrays["initial_audio_native"], dtype=mx.float32)
        video_rows = patchify_video_latents(initial_video_native, tuple(config.patch_size)).astype(mx.bfloat16)[None]
        audio_rows = pack_audio_latents(initial_audio_native).astype(mx.bfloat16)[None]
        packed = {
            "token_tags": mx.array(arrays["packed_token_tags"], dtype=mx.int32),
            "position_ids": mx.array(arrays["packed_position_ids"], dtype=mx.float32),
            "video_indices": mx.array(arrays["packed_video_indices"], dtype=mx.int32),
            "audio_indices": mx.array(arrays["packed_audio_indices"], dtype=mx.int32),
            "text_indices": mx.array(arrays["packed_text_indices"], dtype=mx.int32),
        }
        mx.eval(text, video_rows, audio_rows, *packed.values())
        layout = SimpleNamespace(
            sequence_length=int(packed["position_ids"].shape[0]),
            position_ids=packed["position_ids"],
            token_tags=packed["token_tags"],
            video_indices=packed["video_indices"],
            audio_indices=packed["audio_indices"],
            text_indices=packed["text_indices"],
            num_condition_video_rows=0,
            num_condition_audio_rows=0,
        )
        if layout.sequence_length != EXPECTED_TOTAL_ROWS:
            raise ValueError("derived worker packed sequence row count is not 347")

        schedule_started = time.perf_counter()
        scheduler_video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        scheduler_audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        scheduler_video.set_timesteps(REQUESTED_SIGMA_POINTS)
        scheduler_audio.set_timesteps(REQUESTED_SIGMA_POINTS)
        schedule = schedule_plan_from_schedulers(scheduler_video, scheduler_audio)
        receipt["timing_telemetry"]["schedule_construction_seconds"] = time.perf_counter() - schedule_started
        scheduler = MiniMaxH3MultimodalScheduler(scheduler_video, scheduler_audio)

        baseline = _memory_snapshot(mx)
        receipt["memory_telemetry"]["before_derived_load"] = baseline
        load_started = time.perf_counter()
        dit = load_dit(derived_root, verbose=args.verbose)
        references["transformer"] = dit
        receipt["timing_telemetry"]["derived_transformer_load_seconds"] = time.perf_counter() - load_started
        receipt["memory_telemetry"]["after_derived_load"] = _memory_snapshot(mx)
        if getattr(dit, "construction_mode", None) != CACHE_ONLY_CONSTRUCTION:
            raise ValueError("derived worker did not load a cache-only transformer")
        info = getattr(dit, "checkpoint_format_info", None)
        if getattr(info, "checkpoint_format", None) != "derived":
            raise ValueError("derived worker did not load the derived checkpoint format")
        parameter_keys = [key for key, _ in tree_flatten(dit.parameters())]
        dense_keys = sorted(key for key in parameter_keys if key.startswith("blocks.") and ".adaln_proj." in key)
        if dense_keys:
            raise ValueError("derived cache-only transformer unexpectedly contains dense block AdaLN parameters")
        receipt["residency"] = {
            "construction_mode": getattr(dit, "construction_mode", None),
            "checkpoint_format": getattr(info, "checkpoint_format", None),
            "dense_block_adaln_parameter_keys": dense_keys,
            "dense_temporary_projection_reconstructed": False,
            "sidecar_manifest_path": str(getattr(info, "adaln_manifest_path", "")),
        }
        receipt["completed_stages"].extend(["conditioning-artifact-validated", "derived-cache-only-transformer-loaded"])

        def build_cache(step_index: int, timestep: Any, telemetry: Callable[[str, Mapping[str, Any]], None]) -> Any:
            from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache

            return build_streamed_modulation_cache(dit, timestep, dtype=mx.bfloat16, telemetry=telemetry)

        provider = StreamedCacheSessionProvider(build_cache, event_sink=event_writer)

        def timetable(step_index: int, transition: Mapping[str, Any]) -> tuple[Any, Any]:
            timestep, indices = build_row_timesteps(
                layout,
                transition["video_current_timestep"],
                transition["audio_current_timestep"],
                0.999,
                1.0,
            )
            return timestep, indices

        result = run_full_schedule(
            dit,
            scheduler,
            schedule,
            initial_video_latent=video_rows,
            initial_audio_latent=audio_rows,
            timestep_provider=timetable,
            text_embedding=text,
            packed_inputs=packed,
            cache_provider=provider,
            native_latent_provider=lambda video, audio: _native_from_packed(video, audio, config=config),
            memory_snapshot=lambda: _memory_snapshot(mx),
            expected_video_shape=(1, EXPECTED_TARGET_VIDEO_ROWS, EXPECTED_VIDEO_ROW_WIDTH),
            expected_audio_shape=(1, EXPECTED_TARGET_AUDIO_ROWS, EXPECTED_AUDIO_ROW_WIDTH),
            expected_text_shape=EXPECTED_CONDITIONING_SHAPE,
            expected_video_dtype="bfloat16",
            expected_audio_dtype="bfloat16",
            expected_text_dtype="bfloat16",
            expected_prediction_dtype=CANONICAL_PREDICTION_DTYPE,
        )
        receipt["schedule_contract"] = schedule.receipt()
        receipt["denoising"] = result.receipt()
        receipt["streamed_adaln_lifecycle"] = result.lifecycle
        receipt["cache_attribution"] = result.cache_attribution
        receipt["memory_telemetry"].update(result.memory_telemetry)
        receipt["timing_telemetry"].update(result.timing_telemetry)
        receipt["completed_stages"].extend(["full-fifteen-transition-denoising-complete", "all-cache-sessions-released"])

        final_video_np = _as_float32_numpy(result.final_native_video)
        final_audio_np = _as_float32_numpy(result.final_native_audio)
        serialization_started = time.perf_counter()
        metadata = {
            "artifact_identity": "minimax-h3-mlx-v05d-final-native-latent",
            "schema_version": FINAL_ARTIFACT_SCHEMA_VERSION,
            "attempt_identifier": args.attempt_identifier,
            "native_video": shape_dtype(final_video_np, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(final_video_np, logical_dtype="bfloat16")},
            "native_audio": shape_dtype(final_audio_np, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(final_audio_np, logical_dtype="bfloat16")},
            "packed_final_state_fingerprint": packed_state_fingerprint(result.final_video_latent, result.final_audio_latent),
            "schedule_contract": schedule.receipt(),
            "completed_transition_count": result.transitions.__len__(),
            "transformer_forward_count": result.transformer_forwards,
            "scheduler_update_counts": {"video": result.video_scheduler_updates, "audio": result.audio_scheduler_updates},
            "streamed_adaln_lifecycle": result.lifecycle,
            "worker_identity": "derived",
            "worker_exit_receipt": {
                "worker_started": True,
                "worker_exit_observed": False,
                "worker_exit_code": None,
                "worker_pid": os.getpid(),
                "worker_termination_confirmed": False,
            },
            "transformer_release_receipt": {},
            "final_active_memory": None,
            "final_allocator_cache": None,
            "final_allocator_cache_zero": False,
            "final_artifact_npz_sha256": None,
            "metadata_sha256": None,
            "memory_receipt": receipt["memory_telemetry"],
            "git_identity": capture_git_identity(),
            "checkpoint_identity": _checkpoint_identity(
                Path(args.checkpoint_root),
                derived_root,
                validate_derived_filesystem(derived_root),
            ),
        }
        validate_final_artifact(
            metadata,
            arrays={"final_video_native": final_video_np, "final_audio_native": final_audio_np},
            require_worker_termination=False,
        )
        _write_npz(final_artifact_path, {"final_video_native": final_video_np, "final_audio_native": final_audio_np})
        _write_json(final_metadata_path, metadata)
        receipt["timing_telemetry"]["final_serialization_seconds"] = time.perf_counter() - serialization_started
        receipt["final_artifact"] = metadata
        receipt["event_file"] = event_writer.summary()
        receipt["completed_stages"].append("final-native-latent-artifact-written")
    except DenoisingFailure as exc:
        primary = exc.primary_error if isinstance(exc.primary_error, BaseException) else exc
        cleanup = exc.cleanup_error if isinstance(exc.cleanup_error, BaseException) else None
        receipt["denoising"] = exc.state
        receipt["streamed_adaln_lifecycle"] = exc.state.get("streamed_adaln_lifecycle", {})
        receipt["cache_attribution"] = exc.state.get("cache_attribution", {})
        receipt["memory_telemetry"].update(exc.state.get("memory_telemetry", {}))
        receipt["timing_telemetry"].update(exc.state.get("timing_telemetry", {}))
    except BaseException as exc:
        primary = exc
        if provider is not None:
            receipt["streamed_adaln_lifecycle"] = provider.aggregate()
            receipt["cache_attribution"] = provider.cache_attribution()
    finally:
        if mx is not None:
            # Capture evidence and sever the exception graph before the release gate measures
            # residency.  The serialized mappings retain the primary and cleanup diagnostics
            # without retaining traceback frames or runtime locals.
            primary, cleanup = serialize_and_detach_failure(primary, cleanup)
            if provider is not None and provider.active:
                try:
                    provider.cleanup_failed_step(provider.active_record["step_index"], provider._active_cache)
                except BaseException as exc:
                    if primary is None:
                        primary = serialize_and_detach_error(exc)
                    elif cleanup is None:
                        cleanup = serialize_and_detach_error(exc)
                    else:
                        receipt.setdefault("additional_cleanup_errors", []).append(serialize_and_detach_error(exc))
            if provider is not None:
                provider.cache_builder = None
                provider.event_sink = None
            dit = text = initial_video_native = initial_audio_native = video_rows = audio_rows = None
            packed = layout = schedule = scheduler = result = config = None
            release = _release_runtime(mx, references, baseline, args.tolerance)
            receipt["transformer_release"] = release
            receipt["memory_telemetry"]["before_transformer_release"] = release.get("memory_before_allocator_purge")
            receipt["memory_telemetry"]["after_transformer_release"] = release.get("memory_after_allocator_purge")
            receipt["memory_telemetry"]["final_active_memory"] = release.get("memory_after_allocator_purge", {}).get("active")
            receipt["memory_telemetry"]["final_allocator_cache"] = release.get("memory_after_allocator_purge", {}).get("allocator_cache")
            if primary is None and release.get("passed") is not True:
                primary = serialize_and_detach_error(RuntimeError("derived transformer release and allocator-cache gate failed"))
            elif primary is not None and release.get("passed") is not True:
                release_error = RuntimeError("derived transformer cleanup/release gate failed")
                if cleanup is None:
                    primary, cleanup = preserve_primary_on_release_failure(primary, cleanup, release_error)
                else:
                    receipt.setdefault("additional_cleanup_errors", []).append(serialize_and_detach_error(release_error))
    receipt["timing_telemetry"]["worker_total_seconds"] = time.perf_counter() - started
    receipt["event_file"] = event_writer.summary()
    if primary is not None:
        state = receipt.get("denoising", {})
        receipt["denoising"] = state
        _write_worker_failure(receipt_path, receipt, primary, cleanup, cleanup_attempted=mx is not None)
        return 1
    receipt["status"] = "success"
    receipt["worker_receipt_valid"] = True
    receipt["completed_stages"].append("derived-transformer-released")
    _write_json(receipt_path, receipt)
    return 0


def _video_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--derived-transformer", required=True)
    parser.add_argument("--attempt-identifier", required=True)
    parser.add_argument("--final-artifact", required=True)
    parser.add_argument("--final-artifact-metadata", required=True)
    parser.add_argument("--frames-partial", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--video-frame-manifest", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _video_worker_main(argv: Sequence[str]) -> int:
    args = _video_worker_parser(argv)
    receipt = _worker_base(VIDEO_WORKER_IDENTITY)
    receipt.update(
        {
            "input_artifact_valid": False,
            "release_gate_passed": False,
            "allocator_cache_zero": False,
            "published_artifact_valid": False,
            "frames_partial_path": str(Path(args.frames_partial).resolve()),
            "frames_path": str(Path(args.frames).resolve()),
            "video_frame_manifest_path": str(Path(args.video_frame_manifest).resolve()),
            "worker_receipt_path": str(Path(args.receipt).resolve()),
            "publication_state": "not_published",
            "staged_frame_evidence": [],
        }
    )
    receipt_path = Path(args.receipt).resolve()
    root = Path(args.checkpoint_root).resolve()
    derived = Path(args.derived_transformer).resolve()
    final_artifact = Path(args.final_artifact).resolve()
    final_metadata = Path(args.final_artifact_metadata).resolve()
    frames_partial = Path(args.frames_partial).resolve()
    frames_final = Path(args.frames).resolve()
    manifest_path = Path(args.video_frame_manifest).resolve()
    references: dict[str, Any] = {
        "config": None,
        "stored_latent": None,
        "latent": None,
        "decoder_input": None,
        "decoder": None,
        "raw": None,
        "raw_np": None,
        "frames": None,
    }
    mx = None
    baseline: Mapping[str, Any] | None = None
    primary: BaseException | None = None
    cleanup: BaseException | None = None
    started = time.perf_counter()
    try:
        preflight = validate_derived_filesystem(derived)
        checkpoint_identity = _checkpoint_identity(root, derived, preflight)
        input_validation = validate_final_video_input_artifact(
            final_artifact,
            final_metadata,
            expected_attempt_identifier=args.attempt_identifier,
            expected_checkpoint_identity=checkpoint_identity,
        )
        receipt["input_artifact"] = {
            key: value
            for key, value in input_validation.items()
            if key not in {"arrays", "metadata"}
        }
        receipt["input_artifact_valid"] = True
        receipt["completed_stages"].extend(
            ["video-worker-started", "final-native-latent-independently-validated"]
        )
        if frames_final.exists():
            raise FileExistsError(f"refusing existing final video frames directory: {frames_final}")
        if frames_partial.exists():
            raise FileExistsError(f"refusing existing staged video frames directory: {frames_partial}")

        # Runtime imports are deliberately confined to this child and are reached only after the
        # complete input gate above has passed.
        import mlx.core as mx

        from minimax_h3_mlx.load import load_video_vae, load_video_vae_config

        baseline = _memory_snapshot(mx)
        receipt["memory_telemetry"]["before_load"] = baseline
        result = execute_video_decode_once(
            artifact_path=final_artifact,
            metadata_path=final_metadata,
            expected_attempt_identifier=args.attempt_identifier,
            expected_checkpoint_identity=checkpoint_identity,
            video_root=root / "video_vae",
            frames_partial=frames_partial,
            frames_final=frames_final,
            manifest_path=manifest_path,
            mx=mx,
            load_video_config=load_video_vae_config,
            load_video_vae=load_video_vae,
            materialize=lambda value: mx.eval(value),
            memory_snapshot=lambda: _memory_snapshot(mx),
            references=references,
        )
        receipt["video_geometry"] = result["video_geometry"]
        receipt["logical_video_fingerprint"] = result["logical_video_fingerprint"]
        receipt["video_output"] = {
            "raw": result["raw_shape_dtype"],
            "rgb": result["rgb_shape_dtype"],
        }
        receipt["video_artifacts"] = result["frame_manifest"]
        receipt["memory_telemetry"].update(result["memory"])
        receipt["memory_telemetry"]["after_load"] = result["memory"].get("after_load")
        receipt["memory_telemetry"]["peak"] = result["memory"].get("peak")
        receipt["publication_state"] = "published"
        receipt["published_artifact_valid"] = True
        receipt["completed_stages"].extend(
            ["video-vae-loaded-once", "video-decoded-once", "png-set-staged-and-validated", "frames-published-atomically"]
        )
    except BaseException as exc:
        primary = exc
    finally:
        if mx is not None:
            try:
                release = release_video_decoder(
                    mx,
                    references,
                    baseline,
                    args.tolerance,
                    memory_snapshot=lambda: _memory_snapshot(mx),
                )
                receipt["video_release"] = release
                receipt["release_gate_passed"] = release.get("passed") is True
                receipt["allocator_cache_zero"] = release.get("allocator_cache_zero") is True
                receipt["memory_telemetry"]["before_release"] = release.get("memory_before_release")
                receipt["memory_telemetry"]["after_release"] = release.get("memory_after_release")
                receipt["memory_telemetry"]["before_allocator_purge"] = release.get("memory_before_allocator_purge")
                if primary is None and release.get("passed") is not True:
                    primary = RuntimeError("video VAE release and allocator-cache gate failed")
                elif primary is not None and release.get("passed") is not True:
                    cleanup = RuntimeError("video VAE cleanup/release gate failed")
            except BaseException as exc:
                if primary is None:
                    primary = exc
                elif cleanup is None:
                    cleanup = exc
        if primary is not None and frames_final.is_dir() and not frames_partial.exists():
            try:
                frames_final.rename(frames_partial)
                receipt["publication_state"] = "rolled_back_after_failure"
            except BaseException as exc:
                if cleanup is None:
                    cleanup = exc
                else:
                    receipt.setdefault("additional_cleanup_errors", []).append(error_receipt(exc))
        receipt["staged_frame_evidence"] = collect_video_frame_evidence(frames_partial)
    receipt["timing_telemetry"]["worker_total_seconds"] = time.perf_counter() - started
    if primary is not None:
        _write_worker_failure(receipt_path, receipt, primary, cleanup, cleanup_attempted=mx is not None)
        return 1
    receipt["status"] = "success"
    receipt["worker_receipt_valid"] = True
    receipt["completed_stages"].append("video-vae-released-with-zero-cache")
    _write_json(receipt_path, receipt)
    return 0


def _audio_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--derived-transformer", required=True)
    parser.add_argument("--attempt-identifier", required=True)
    parser.add_argument("--final-artifact", required=True)
    parser.add_argument("--final-artifact-metadata", required=True)
    parser.add_argument("--audio-partial", required=True)
    parser.add_argument("--audio-wav", required=True)
    parser.add_argument("--audio-manifest", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _audio_worker_main(argv: Sequence[str]) -> int:
    args = _audio_worker_parser(argv)
    receipt = _worker_base(AUDIO_WORKER_IDENTITY)
    receipt.update(
        {
            "input_artifact_valid": False,
            "release_gate_passed": False,
            "allocator_cache_zero": False,
            "published_artifact_valid": False,
            "wav_manifest_valid": False,
            "audio_config_load_count": 0,
            "audio_vae_load_count": 0,
            "decode_count": 0,
            "audio_partial_path": str(Path(args.audio_partial).resolve()),
            "audio_wav_path": str(Path(args.audio_wav).resolve()),
            "audio_manifest_path": str(Path(args.audio_manifest).resolve()),
            "worker_receipt_path": str(Path(args.receipt).resolve()),
            "publication_state": "not_published",
            "staged_audio_evidence": None,
        }
    )
    receipt_path = Path(args.receipt).resolve()
    root = Path(args.checkpoint_root).resolve()
    derived = Path(args.derived_transformer).resolve()
    final_artifact = Path(args.final_artifact).resolve()
    final_metadata = Path(args.final_artifact_metadata).resolve()
    audio_partial = Path(args.audio_partial).resolve()
    audio_final = Path(args.audio_wav).resolve()
    manifest_path = Path(args.audio_manifest).resolve()
    references: dict[str, Any] = {
        "config": None,
        "stored_latent": None,
        "latent": None,
        "decoder_input": None,
        "decoder": None,
        "raw": None,
        "raw_np": None,
        "waveform": None,
    }
    mx = None
    baseline: Mapping[str, Any] | None = None
    primary: BaseException | None = None
    cleanup: BaseException | None = None
    started = time.perf_counter()
    try:
        preflight = validate_derived_filesystem(derived)
        checkpoint_identity = _checkpoint_identity(root, derived, preflight)
        input_validation = validate_final_audio_input_artifact(
            final_artifact,
            final_metadata,
            expected_attempt_identifier=args.attempt_identifier,
            expected_checkpoint_identity=checkpoint_identity,
        )
        receipt["input_artifact"] = {
            key: value
            for key, value in input_validation.items()
            if key not in {"arrays", "metadata"}
        }
        receipt["input_artifact_valid"] = True
        receipt["completed_stages"].extend(
            ["audio-worker-started", "final-native-latent-independently-validated"]
        )
        if audio_final.exists():
            raise FileExistsError(f"refusing existing final audio WAV: {audio_final}")
        if audio_partial.exists():
            raise FileExistsError(f"refusing existing staged audio WAV: {audio_partial}")

        # Runtime imports are deliberately confined to this child and occur only after the complete
        # final-latent input gate above has passed.
        import mlx.core as mx

        from minimax_h3_mlx.load import load_audio_vae, load_audio_vae_config

        baseline = _memory_snapshot(mx)
        receipt["memory_telemetry"]["before_load"] = baseline

        def counted_config(path: Path) -> Any:
            receipt["audio_config_load_count"] += 1
            return load_audio_vae_config(path)

        def counted_vae(path: Path) -> Any:
            receipt["audio_vae_load_count"] += 1
            decoder = load_audio_vae(path)

            class CountedDecoder:
                def __init__(self, wrapped: Any) -> None:
                    self.wrapped = wrapped

                def decode(self, value: Any) -> Any:
                    receipt["decode_count"] += 1
                    return self.wrapped.decode(value)

            return CountedDecoder(decoder)

        result = execute_audio_decode_once(
            artifact_path=final_artifact,
            metadata_path=final_metadata,
            expected_attempt_identifier=args.attempt_identifier,
            expected_checkpoint_identity=checkpoint_identity,
            audio_root=root / "audio_vae",
            wav_partial=audio_partial,
            wav_final=audio_final,
            manifest_path=manifest_path,
            mx=mx,
            load_audio_config=counted_config,
            load_audio_vae=counted_vae,
            materialize=lambda value: mx.eval(value),
            memory_snapshot=lambda: _memory_snapshot(mx),
            references=references,
        )
        receipt["audio_geometry"] = result["audio_geometry"]
        receipt["logical_audio_fingerprint"] = result["logical_audio_fingerprint"]
        receipt["audio_output"] = {
            "raw": result["raw_shape_dtype"],
            "waveform": result["waveform_shape_dtype"],
        }
        receipt["audio_artifacts"] = result["wav_manifest"]
        receipt["memory_telemetry"].update(result["memory"])
        receipt["memory_telemetry"]["after_load"] = result["memory"].get("after_load")
        receipt["memory_telemetry"]["peak"] = result["memory"].get("peak")
        receipt["publication_state"] = "published"
        receipt["wav_manifest_valid"] = True
        receipt["published_artifact_valid"] = True
        receipt["completed_stages"].extend(
            ["audio-config-loaded-once", "audio-vae-loaded-once", "audio-decoded-once", "wav-staged-and-validated", "audio-published-atomically"]
        )
    except BaseException as exc:
        primary = exc
    finally:
        if mx is not None:
            _clear_exception_frames(primary)
            try:
                release = release_audio_decoder(
                    mx,
                    references,
                    baseline,
                    args.tolerance,
                    memory_snapshot=lambda: _memory_snapshot(mx),
                )
                receipt["audio_release"] = release
                receipt["release_gate_passed"] = release.get("passed") is True
                receipt["allocator_cache_zero"] = release.get("allocator_cache_zero") is True
                receipt["memory_telemetry"]["before_release"] = release.get("memory_before_release")
                receipt["memory_telemetry"]["after_release"] = release.get("memory_after_release")
                receipt["memory_telemetry"]["before_allocator_purge"] = release.get("memory_before_allocator_purge")
                if primary is None and release.get("passed") is not True:
                    primary = RuntimeError("audio VAE release and allocator-cache gate failed")
                elif primary is not None and release.get("passed") is not True:
                    cleanup = RuntimeError("audio VAE cleanup/release gate failed")
            except BaseException as exc:
                if primary is None:
                    primary = exc
                elif cleanup is None:
                    cleanup = exc
        if primary is not None and audio_final.is_file() and not audio_partial.exists():
            try:
                audio_final.rename(audio_partial)
                receipt["publication_state"] = "rolled_back_after_failure"
            except BaseException as exc:
                if cleanup is None:
                    cleanup = exc
                else:
                    receipt.setdefault("additional_cleanup_errors", []).append(error_receipt(exc))
        receipt["staged_audio_evidence"] = collect_audio_wav_evidence(audio_partial)
    receipt["timing_telemetry"]["worker_total_seconds"] = time.perf_counter() - started
    if primary is not None:
        _write_worker_failure(receipt_path, receipt, primary, cleanup, cleanup_attempted=mx is not None)
        return 1
    receipt["status"] = "success"
    receipt["worker_receipt_valid"] = True
    receipt["completed_stages"].append("audio-vae-released-with-zero-cache")
    _write_json(receipt_path, receipt)
    return 0


def _child_command(worker: str, arguments: Sequence[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), worker, *arguments]


def _run_child(worker: str, arguments: Sequence[str], log_path: Path, receipt_path: Path, expected_identity: str) -> dict[str, Any]:
    command = _child_command(worker, arguments)
    started = time.perf_counter()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(command, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    exit_code = process.returncode
    terminated = process.poll() is not None
    log_path.write_text(stdout + ("\n--- stderr ---\n" + stderr if stderr else ""))
    receipt: dict[str, Any]
    if receipt_path.is_file():
        try:
            loaded = json.loads(receipt_path.read_text())
            receipt = dict(loaded) if isinstance(loaded, dict) else {"status": "failed"}
        except (OSError, json.JSONDecodeError) as exc:
            receipt = {"status": "failed", "primary_error": error_receipt(exc)}
    else:
        receipt = {"status": "failed", "primary_error": {"type": "MissingReceipt", "message": "worker exited without a receipt", "traceback": ""}}
    receipt["worker_started"] = True
    receipt["worker_exit_observed"] = True
    receipt["worker_exit_code"] = exit_code
    receipt["worker_reported_pid"] = receipt.get("pid")
    receipt["worker_pid"] = process.pid
    receipt["worker_termination_confirmed"] = terminated
    receipt["worker_receipt_path"] = str(receipt_path.resolve())
    receipt["worker_log_path"] = str(log_path.resolve())
    receipt["worker_receipt_valid"] = bool(
        receipt.get("worker_identity") == expected_identity
        and receipt.get("status") == "success"
        and receipt.get("worker_started")
        and receipt.get("worker_exit_observed")
        and receipt.get("worker_termination_confirmed")
        and exit_code == 0
    )
    receipt["subprocess"] = {
        "command": shlex.join(command),
        "exit_status": exit_code,
        "pid": process.pid,
        "termination_confirmed": terminated,
        "elapsed_seconds": time.perf_counter() - started,
        "log_path": str(log_path),
        "receipt_path": str(receipt_path),
    }
    return receipt


def validate_worker_boundary(receipt: Mapping[str, Any], *, identity: str) -> None:
    if receipt.get("worker_identity") != identity:
        raise ValueError(f"worker identity mismatch: expected {identity}")
    for key in ("worker_started", "worker_exit_observed", "worker_termination_confirmed", "worker_receipt_valid"):
        if receipt.get(key) is not True:
            raise ValueError(f"worker boundary field {key} is not confirmed")
    if not isinstance(receipt.get("worker_pid"), int) or receipt.get("worker_pid") <= 0:
        raise ValueError("worker boundary is missing a positive worker PID")
    if receipt.get("worker_exit_code") != 0 or receipt.get("status") != "success":
        raise ValueError(f"{identity} worker did not exit successfully")


def validate_conditioning_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the conditioning boundary before a derived worker can be launched."""
    if receipt.get("status") != "success":
        raise ValueError("conditioning receipt is not successful")
    prompt = receipt.get("prompt")
    if not isinstance(prompt, Mapping) or prompt.get("text") != LOCKED_PROMPT or prompt.get("utf8_byte_count") != PROMPT_UTF8_BYTE_COUNT or prompt.get("sha256") != PROMPT_SHA256:
        raise ValueError("conditioning receipt does not preserve the locked prompt literal")
    if prompt.get("token_count") != EXPECTED_TOKEN_COUNT or prompt.get("negative_prompt") is not None or prompt.get("image_conditioning") is not False or prompt.get("special_tokens") is not False:
        raise ValueError("conditioning receipt token/prompt policy is not locked")
    conditioning = receipt.get("conditioning")
    if not isinstance(conditioning, Mapping) or tuple(conditioning.get("shape", ())) != EXPECTED_CONDITIONING_SHAPE or conditioning.get("dtype") != EXPECTED_CONDITIONING_DTYPE:
        raise ValueError("conditioning receipt shape or dtype is not locked")
    if not conditioning.get("fingerprint"):
        raise ValueError("conditioning receipt is missing its fingerprint")
    tokenizer = receipt.get("tokenizer")
    token_ids = tokenizer.get("token_ids") if isinstance(tokenizer, Mapping) else None
    if not isinstance(token_ids, list) or len(token_ids) != 1 or len(token_ids[0]) != EXPECTED_TOKEN_COUNT:
        raise ValueError("conditioning receipt token IDs are incomplete")
    validate_packed_contract(receipt.get("packing", {}))
    geometry = receipt.get("geometry")
    if not isinstance(geometry, Mapping) or geometry.get("total_rows") != EXPECTED_TOTAL_ROWS:
        raise ValueError("conditioning receipt geometry is incomplete")
    release = receipt.get("conditioning_release")
    if not isinstance(release, Mapping) or release.get("passed") is not True:
        raise ValueError("conditioning release gate did not pass")
    binding = receipt.get("conditioning_artifact")
    if (
        not isinstance(binding, Mapping)
        or set(binding) != {"sha256", "array_keys", "arrays"}
        or binding.get("array_keys") != sorted(CONDITIONING_ARRAY_KEYS)
    ):
        raise ValueError("conditioning receipt is missing an exact artifact binding")


def _artifact_required_arrays() -> set[str]:
    return {"final_video_native", "final_audio_native"}


def validate_final_artifact(
    artifact: Mapping[str, Any],
    *,
    arrays: Mapping[str, Any] | None = None,
    require_worker_termination: bool = True,
    artifact_path: Path | None = None,
    metadata_path: Path | None = None,
) -> None:
    if set(artifact) != FINAL_ARTIFACT_KEYS:
        raise ValueError(f"final artifact schema mismatch: missing={sorted(FINAL_ARTIFACT_KEYS - set(artifact))}, unexpected={sorted(set(artifact) - FINAL_ARTIFACT_KEYS)}")
    if artifact.get("artifact_identity") != "minimax-h3-mlx-v05d-final-native-latent" or artifact.get("schema_version") != FINAL_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("final native latent artifact identity mismatch")
    if not isinstance(artifact.get("attempt_identifier"), str) or not artifact.get("attempt_identifier"):
        raise ValueError("final native latent artifact attempt identifier is missing")
    if artifact.get("worker_identity") != "derived":
        raise ValueError("final artifact must identify the derived worker")
    if not isinstance(artifact.get("checkpoint_identity"), Mapping):
        raise ValueError("final artifact checkpoint identity is missing")
    exit_receipt = artifact.get("worker_exit_receipt")
    if require_worker_termination and (
        not isinstance(exit_receipt, Mapping)
        or exit_receipt.get("worker_exit_observed") is not True
        or exit_receipt.get("worker_exit_code") != 0
        or exit_receipt.get("worker_termination_confirmed") is not True
    ):
        raise ValueError("final artifact is missing confirmed worker termination")
    if artifact.get("completed_transition_count") != EXPECTED_DENOISING_TRANSITIONS or artifact.get("transformer_forward_count") != EXPECTED_TRANSFORMER_FORWARDS:
        raise ValueError("final artifact transition or forward count is incomplete")
    update_counts = artifact.get("scheduler_update_counts")
    if not isinstance(update_counts, Mapping) or update_counts.get("video") != EXPECTED_DENOISING_TRANSITIONS or update_counts.get("audio") != EXPECTED_DENOISING_TRANSITIONS:
        raise ValueError("final artifact scheduler-update counts are incomplete")
    validate_schedule_contract(artifact.get("schedule_contract", {}))
    validate_lifecycle_totals(artifact.get("streamed_adaln_lifecycle", {}))
    for label, expected_shape in (("native_video", VIDEO_NATIVE_SHAPE), ("native_audio", AUDIO_NATIVE_SHAPE)):
        value = artifact.get(label)
        if not isinstance(value, Mapping) or tuple(value.get("shape", ())) != expected_shape or value.get("dtype") != "bfloat16" or not value.get("fingerprint"):
            raise ValueError(f"final artifact {label} schema is invalid")
    if arrays is not None:
        required_arrays = _artifact_required_arrays()
        if set(arrays) != required_arrays:
            raise ValueError(
                f"final artifact arrays are not exact: missing={sorted(required_arrays - set(arrays))}, "
                f"unexpected={sorted(set(arrays) - required_arrays)}"
            )
        video_array = np.asarray(arrays["final_video_native"])
        audio_array = np.asarray(arrays["final_audio_native"])
        if tuple(video_array.shape) != VIDEO_NATIVE_SHAPE or tuple(audio_array.shape) != AUDIO_NATIVE_SHAPE:
            raise ValueError("final native latent arrays have the wrong shape")
        if video_array.dtype != np.dtype(np.float32) or audio_array.dtype != np.dtype(np.float32):
            raise ValueError("final native latent NPZ storage dtype must be float32")
        if artifact["native_video"]["fingerprint"] != array_fingerprint(arrays["final_video_native"], logical_dtype="bfloat16"):
            raise ValueError("final video latent fingerprint does not match its serialized array")
        if artifact["native_audio"]["fingerprint"] != array_fingerprint(arrays["final_audio_native"], logical_dtype="bfloat16"):
            raise ValueError("final audio latent fingerprint does not match its serialized array")
    if require_worker_termination:
        release = artifact.get("transformer_release_receipt")
        if not isinstance(release, Mapping) or release.get("passed") is not True:
            raise ValueError("final artifact is missing post-release transformer evidence")
        after_release = release.get("memory_after_allocator_purge")
        if not isinstance(after_release, Mapping):
            raise ValueError("final artifact is missing post-release memory evidence")
        if artifact.get("final_active_memory") != after_release.get("active"):
            raise ValueError("final artifact active memory is stale or inconsistent")
        if artifact.get("final_allocator_cache") != after_release.get("allocator_cache"):
            raise ValueError("final artifact allocator-cache evidence is stale or inconsistent")
        if artifact.get("final_allocator_cache_zero") is not True or release.get("allocator_cache_zero") is not True:
            raise ValueError("final artifact is missing the allocator-cache-zero gate")
        if not isinstance(artifact.get("final_artifact_npz_sha256"), str) or not artifact["final_artifact_npz_sha256"]:
            raise ValueError("final artifact is missing its NPZ SHA-256")
        if not isinstance(artifact.get("metadata_sha256"), str) or artifact.get("metadata_sha256") != stable_metadata_sha256(artifact):
            raise ValueError("final artifact metadata SHA-256 linkage is stale")
        if artifact_path is not None and artifact.get("final_artifact_npz_sha256") != sha256_file(artifact_path):
            raise ValueError("final artifact NPZ SHA-256 does not match its file")
        if metadata_path is not None and not metadata_path.is_file():
            raise ValueError("final artifact metadata file is missing")
        if metadata_path is not None:
            on_disk = _read_json_object(metadata_path, "final native latent metadata")
            if on_disk != dict(artifact) or stable_metadata_sha256(on_disk) != artifact.get("metadata_sha256"):
                raise ValueError("final artifact metadata SHA-256 linkage does not match the serialized metadata")


def validate_derived_decoder_gate(
    derived_receipt: Mapping[str, Any],
    artifact_path: Path,
    metadata_path: Path,
    *,
    derived_worker_attempts: int = 1,
    expected_attempt_identifier: str | None = None,
    expected_checkpoint_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every derived prerequisite before a decoder worker can be launched."""
    gates: dict[str, bool] = {}
    gate_receipt: dict[str, Any] = {
        "status": "failed",
        "passed": False,
        "finalization_passed": False,
        "release_gate_passed": False,
        "failed_gate": None,
        "gates": gates,
        "primary_error": None,
        "cleanup_error": None,
        "cleanup_attempted": False,
    }

    def check(name: str, callback: Callable[[], None], *, phase: str) -> None:
        try:
            callback()
        except BaseException as exc:
            gate_receipt["failed_gate"] = name
            gate_receipt["phase"] = phase
            gate_receipt["primary_error"] = error_receipt(exc)
            cleanup = derived_receipt.get("cleanup_error")
            raise DecoderGateFailure(
                name,
                exc,
                gate_receipt=gate_receipt,
                phase=phase,
                cleanup=cleanup,
                cleanup_attempted=cleanup is not None,
            ) from exc
        gates[name] = True

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(message)

    check(
        "derived-worker-started-exactly-once",
        lambda: require(
            derived_worker_attempts == 1 and derived_receipt.get("worker_started") is True,
            "derived worker must be started exactly once",
        ),
        phase="derived-finalization",
    )
    check(
        "derived-worker-exit-code-zero",
        lambda: require(
            derived_receipt.get("worker_exit_code") == 0,
            "derived worker did not exit with code zero",
        ),
        phase="derived-finalization",
    )
    check(
        "derived-worker-termination-confirmed",
        lambda: validate_worker_boundary(derived_receipt, identity="derived"),
        phase="derived-finalization",
    )

    metadata: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    def load_metadata() -> None:
        nonlocal metadata
        metadata = _read_json_object(Path(metadata_path), "final native latent metadata")
        if metadata.get("metadata_sha256") != stable_metadata_sha256(metadata):
            raise ValueError("final native latent metadata checksum/linkage is stale")
        if expected_attempt_identifier is not None and metadata.get("attempt_identifier") != expected_attempt_identifier:
            raise ValueError("final native latent metadata attempt identifier mismatch")
        if expected_checkpoint_identity is not None and metadata.get("checkpoint_identity") != _json_safe(dict(expected_checkpoint_identity)):
            raise ValueError("final native latent metadata checkpoint identity mismatch")

    check("final-native-latent-metadata-linkage", load_metadata, phase="derived-finalization")

    def load_arrays_and_checksum() -> None:
        nonlocal arrays
        if metadata.get("final_artifact_npz_sha256") != sha256_file(Path(artifact_path)):
            raise ValueError("final latent NPZ SHA-256 does not match final metadata")
        arrays = _load_npz(Path(artifact_path))
        if set(arrays) != {"final_video_native", "final_audio_native"}:
            raise ValueError("final latent NPZ key set is not exact")

    check("final-latent-npz-sha256", load_arrays_and_checksum, phase="derived-finalization")

    def video_shape_and_dtype() -> None:
        descriptor = metadata.get("native_video")
        if tuple(arrays["final_video_native"].shape) != VIDEO_NATIVE_SHAPE or not isinstance(descriptor, Mapping):
            raise ValueError("final video latent shape is not (1,24,9,8,8)")
        if tuple(descriptor.get("shape", ())) != VIDEO_NATIVE_SHAPE or descriptor.get("dtype") != "bfloat16":
            raise ValueError("final video latent logical shape or dtype is invalid")

    check("video-latent-shape-and-logical-dtype", video_shape_and_dtype, phase="derived-finalization")

    def audio_shape_and_dtype() -> None:
        descriptor = metadata.get("native_audio")
        if tuple(arrays["final_audio_native"].shape) != AUDIO_NATIVE_SHAPE or not isinstance(descriptor, Mapping):
            raise ValueError("final audio latent shape is not (2,32,50)")
        if tuple(descriptor.get("shape", ())) != AUDIO_NATIVE_SHAPE or descriptor.get("dtype") != "bfloat16":
            raise ValueError("final audio latent logical shape or dtype is invalid")

    check("audio-latent-shape-and-logical-dtype", audio_shape_and_dtype, phase="derived-finalization")

    def video_fingerprint() -> None:
        if metadata["native_video"].get("fingerprint") != array_fingerprint(arrays["final_video_native"], logical_dtype="bfloat16"):
            raise ValueError("final video latent fingerprint does not match its NPZ array")

    check("video-latent-fingerprint", video_fingerprint, phase="derived-finalization")

    def audio_fingerprint() -> None:
        if metadata["native_audio"].get("fingerprint") != array_fingerprint(arrays["final_audio_native"], logical_dtype="bfloat16"):
            raise ValueError("final audio latent fingerprint does not match its NPZ array")

    check("audio-latent-fingerprint", audio_fingerprint, phase="derived-finalization")

    def schedule() -> None:
        validate_schedule_contract(metadata.get("schedule_contract", {}))

    check("schedule-16-16-16-15-15", schedule, phase="derived-finalization")

    def lifecycle() -> None:
        validate_lifecycle_totals(metadata.get("streamed_adaln_lifecycle", {}))

    check("streamed-adaln-totals", lifecycle, phase="derived-finalization")
    gate_receipt["finalization_passed"] = True

    def transformer_release() -> None:
        release = metadata.get("transformer_release_receipt")
        if not isinstance(release, Mapping) or release.get("passed") is not True:
            raise ValueError("transformer release receipt did not pass")

    check("transformer-release-receipt", transformer_release, phase="derived-release-gate")

    def active_memory() -> None:
        release = metadata["transformer_release_receipt"]
        after = release.get("memory_after_allocator_purge")
        if (
            release.get("active_memory_within_tolerance") is not True
            or not isinstance(after, Mapping)
            or metadata.get("final_active_memory") != after.get("active")
        ):
            raise ValueError("final active-memory gate did not pass")

    check("final-active-memory-gate", active_memory, phase="derived-release-gate")

    def allocator_cache() -> None:
        release = metadata["transformer_release_receipt"]
        if metadata.get("final_allocator_cache_zero") is not True or release.get("allocator_cache_zero") is not True or metadata.get("final_allocator_cache") != 0:
            raise ValueError("final allocator cache is not zero")

    check("final-allocator-cache-zero", allocator_cache, phase="derived-release-gate")

    check(
        "final-artifact-contract",
        lambda: validate_final_artifact(
            metadata,
            arrays=arrays,
            artifact_path=Path(artifact_path),
            metadata_path=Path(metadata_path),
        ),
        phase="derived-release-gate",
    )
    gate_receipt.update(
        {
            "status": "passed",
            "passed": True,
            "release_gate_passed": True,
            "phase": "derived-release-gate",
            "artifact": {
                "npz_sha256": metadata["final_artifact_npz_sha256"],
                "metadata_sha256": metadata["metadata_sha256"],
                "video_fingerprint": metadata["native_video"]["fingerprint"],
                "audio_fingerprint": metadata["native_audio"]["fingerprint"],
            },
        }
    )
    return gate_receipt


def event_file_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "event_file_path": str(path),
            "event_file_record_count": 0,
            "total_event_records": 0,
            "cache_session_count": 0,
            "sidecar_open_event_count": 0,
            "sidecar_release_event_count": 0,
            "attribution_block_event_count": 0,
            "attribution_session_event_count": 0,
            "validated_block_pairs": 0,
            "event_file_sha256": None,
        }
    records: list[Mapping[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            records.append(value)
    total = len(records)
    return {
        "event_file_path": str(path),
        "event_file_record_count": total,
        "total_event_records": total,
        "cache_session_count": len({record.get("cache_session_id") for record in records if record.get("event") == "session-acquire-start"}),
        "sidecar_open_event_count": sum(record.get("event") == "sidecar_opening" for record in records),
        "sidecar_release_event_count": sum(record.get("event") == "sidecar_released" for record in records),
        "attribution_block_event_count": sum(record.get("event") == ATTRIBUTION_BLOCK_EVENT for record in records),
        "attribution_session_event_count": sum(record.get("event") == ATTRIBUTION_SESSION_EVENT for record in records),
        "validated_block_pairs": 0,
        "event_file_sha256": sha256_file(path),
    }


def validate_event_stream(path: Path) -> dict[str, Any]:
    """Validate transition/session/block identity in the detailed JSONL stream."""
    if not path.is_file():
        raise FileNotFoundError(f"streamed AdaLN event file is missing: {path}")
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"event record {line_number} is not valid JSON") from exc
        if not isinstance(value, Mapping) or not isinstance(value.get("event"), str):
            raise ValueError(f"event record {line_number} is not a JSON event object")
        records.append(value)
    sessions: dict[str, dict[str, Any]] = {}
    sidecar_opens = sidecar_releases = validated_pairs = 0
    attribution_block_events = attribution_session_events = 0
    for record in records:
        event = record["event"]
        session_id = record.get("cache_session_id")
        if event == "session-acquire-start":
            if not isinstance(session_id, str) or session_id in sessions:
                raise ValueError("event stream has a duplicate or missing cache-session identity")
            transition_index = record.get("transition_index", record.get("step_index"))
            if not isinstance(transition_index, int) or isinstance(transition_index, bool):
                raise ValueError("cache session event is missing its transition identity")
            if transition_index != len(sessions):
                raise ValueError("event stream transition identities are not exactly ordered 0..14")
            if session_id != f"cache-session-{transition_index + 1:02d}":
                raise ValueError("event stream cache-session identity does not match its transition")
            sessions[session_id] = {
                "transition_index": transition_index,
                "next_block": 0,
                "active": None,
                "opens": 0,
                "releases": 0,
                "acquire_complete": False,
                "next_attribution_block": 0,
                "attribution_blocks": 0,
                "attribution_component_totals": {field: 0.0 for field in ATTRIBUTION_COMPONENT_FIELDS},
                "attribution_complete": False,
                "release_start": False,
                "release_complete": False,
            }
            continue
        if event in {"session-acquire-complete", "session-release-start", "session-release-complete", "session-release-failure"}:
            if not isinstance(session_id, str) or session_id not in sessions:
                raise ValueError("session lifecycle event refers to an unknown cache session")
            state = sessions[session_id]
            transition_identity = record.get("transition_index", record.get("step_index"))
            if transition_identity != state["transition_index"]:
                raise ValueError("session lifecycle event transition identity does not match its cache session")
            if event == "session-acquire-complete":
                if state["acquire_complete"] or state["release_start"] or state["active"] is not None or state["next_block"] != EXPECTED_BLOCK_COUNT:
                    raise ValueError("event stream acquire completion is out of order or incomplete")
                state["acquire_complete"] = True
            elif event == "session-release-start":
                if not state["acquire_complete"] or state["release_start"] or state["active"] is not None or state["next_block"] != EXPECTED_BLOCK_COUNT:
                    raise ValueError("event stream release start is out of order or incomplete")
                state["release_start"] = True
            elif event == "session-release-complete":
                if not state["release_start"] or state["release_complete"]:
                    raise ValueError("event stream release completion is out of order or duplicated")
                state["release_complete"] = True
            else:
                raise ValueError("event stream contains a failed cache-session release")
            continue
        if event == ATTRIBUTION_BLOCK_EVENT:
            if not isinstance(session_id, str) or session_id not in sessions:
                raise ValueError("cache attribution event refers to an unknown cache session")
            state = sessions[session_id]
            if record.get("transition_index") != state["transition_index"]:
                raise ValueError("cache attribution event transition identity does not match its cache session")
            if state["acquire_complete"] or state["release_start"] or state["active"] is not None:
                raise ValueError("cache attribution event is outside the cache-acquire interval")
            if record.get("attribution_schema_version") != ATTRIBUTION_SCHEMA_VERSION:
                raise ValueError("cache attribution event schema version is invalid")
            normalized = _raw_attribution_block(record, state["next_attribution_block"])
            state["next_attribution_block"] += 1
            state["attribution_blocks"] += 1
            for field in ATTRIBUTION_COMPONENT_FIELDS:
                state["attribution_component_totals"][field] += normalized[field]
            attribution_block_events += 1
            continue
        if event == ATTRIBUTION_SESSION_EVENT:
            if not isinstance(session_id, str) or session_id not in sessions:
                raise ValueError("cache attribution session event refers to an unknown cache session")
            state = sessions[session_id]
            if record.get("transition_index") != state["transition_index"]:
                raise ValueError("cache attribution session transition identity does not match its cache session")
            if not state["acquire_complete"] or state["release_start"] or state["attribution_complete"]:
                raise ValueError("cache attribution session event is out of order or duplicated")
            if state["attribution_blocks"] != EXPECTED_BLOCK_COUNT:
                raise ValueError("cache attribution session event is missing block timing evidence")
            if record.get("attribution_schema_version") != ATTRIBUTION_SCHEMA_VERSION:
                raise ValueError("cache attribution session schema version is invalid")
            totals = record.get("component_totals_seconds")
            if not isinstance(totals, Mapping) or set(totals) != set(ATTRIBUTION_COMPONENT_FIELDS):
                raise ValueError("cache attribution session component totals are incomplete")
            for field in ATTRIBUTION_COMPONENT_FIELDS:
                observed_total = _timing_seconds(totals.get(field), f"session.{field}")
                if not math.isclose(observed_total, state["attribution_component_totals"][field], rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(f"cache attribution session total differs from block events for {field}")
            overhead = record.get("session_overhead_seconds")
            if not isinstance(overhead, Mapping) or set(overhead) != set(ATTRIBUTION_SESSION_OVERHEAD_FIELDS):
                raise ValueError("cache attribution session overhead totals are incomplete")
            overhead_total = math.fsum(
                _timing_seconds(overhead.get(field), f"session.{field}")
                for field in ATTRIBUTION_SESSION_OVERHEAD_FIELDS
            )
            wall = _timing_seconds(record.get("wall_clock_cache_session_total_seconds"), "session.wall_total")
            measured = _timing_seconds(record.get("measured_component_sum_seconds"), "session.measured_components")
            expected_measured = math.fsum((*state["attribution_component_totals"].values(), overhead_total))
            if not math.isclose(measured, expected_measured, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError("cache attribution measured component sum differs from block events")
            remainder = calculate_unattributed_remainder(wall, measured)
            observed_remainder = record.get("unattributed_remainder_seconds")
            if isinstance(observed_remainder, bool) or not isinstance(observed_remainder, (int, float)) or not math.isfinite(float(observed_remainder)):
                raise ValueError("cache attribution session remainder is not finite")
            if not math.isclose(
                float(observed_remainder),
                remainder["unattributed_remainder_seconds"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError("cache attribution session unattributed remainder is stale")
            if record.get("unattributed_remainder_status") != remainder["unattributed_remainder_status"]:
                raise ValueError("cache attribution session remainder status is stale")
            state["attribution_complete"] = True
            attribution_session_events += 1
            continue
        if event not in {"sidecar_opening", "sidecar_released"}:
            continue
        if not isinstance(session_id, str) or session_id not in sessions:
            raise ValueError("sidecar event refers to an unknown cache session")
        state = sessions[session_id]
        if record.get("transition_index") != state["transition_index"]:
            raise ValueError("sidecar event transition identity does not match its cache session")
        if state["acquire_complete"] or state["release_start"]:
            raise ValueError("event stream sidecar event is outside the cache-acquire interval")
        block_index = record.get("block_index")
        path_value = Path(str(record.get("path", ""))).name
        if not isinstance(block_index, int) or isinstance(block_index, bool):
            raise ValueError("sidecar event block identity is invalid")
        if event == "sidecar_opening":
            expected_path = f"block-{state['next_block']:03d}.safetensors"
            if state["active"] is not None:
                raise ValueError("event stream observed overlapping sidecar opens")
            if block_index != state["next_block"]:
                raise ValueError("event stream sidecar opens are out of order")
            if path_value != expected_path:
                raise ValueError("event stream sidecar path does not match its block identity")
            state["active"] = (block_index, path_value)
            state["opens"] += 1
            sidecar_opens += 1
        else:
            if state["active"] is None:
                raise ValueError("event stream observed a sidecar release without an open")
            if state["active"] != (block_index, path_value):
                raise ValueError("event stream sidecar release does not match its open")
            state["active"] = None
            state["releases"] += 1
            state["next_block"] += 1
            sidecar_releases += 1
            validated_pairs += 1
    if len(sessions) != EXPECTED_DENOISING_TRANSITIONS:
        raise ValueError("event stream cache session count is not exactly 15")
    observed_transitions = sorted(state["transition_index"] for state in sessions.values())
    if observed_transitions != list(range(EXPECTED_DENOISING_TRANSITIONS)):
        raise ValueError("event stream transition identities are not exactly 0..14")
    for state in sessions.values():
        if state["active"] is not None or state["next_block"] != EXPECTED_BLOCK_COUNT:
            raise ValueError("event stream contains an incomplete sidecar session")
        if state["opens"] != EXPECTED_BLOCK_COUNT or state["releases"] != EXPECTED_BLOCK_COUNT:
            raise ValueError("event stream session does not contain exactly 50 open/release pairs")
        if state["attribution_blocks"] != EXPECTED_BLOCK_COUNT or state["next_attribution_block"] != EXPECTED_BLOCK_COUNT:
            raise ValueError("event stream session does not contain exactly 50 attribution blocks")
        if not state["acquire_complete"] or not state["attribution_complete"] or not state["release_start"] or not state["release_complete"]:
            raise ValueError("event stream session lifecycle boundaries are incomplete")
    summary = event_file_summary(path)
    summary.update(
        {
            "cache_session_count": len(sessions),
            "sidecar_open_event_count": sidecar_opens,
            "sidecar_release_event_count": sidecar_releases,
            "attribution_block_event_count": attribution_block_events,
            "attribution_session_event_count": attribution_session_events,
            "validated_block_pairs": validated_pairs,
        }
    )
    if (
        summary["cache_session_count"] != EXPECTED_DENOISING_TRANSITIONS
        or sidecar_opens != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
        or sidecar_releases != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
        or validated_pairs != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
        or attribution_block_events != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
        or attribution_session_events != EXPECTED_DENOISING_TRANSITIONS
    ):
        raise ValueError("event stream does not contain exactly 15 sessions and 750 validated block pairs")
    return summary


def validate_event_file_linkage(report: Mapping[str, Any], path: Path) -> None:
    expected = event_file_summary(path)
    for key in ("event_file_path", "event_file_record_count", "event_file_sha256"):
        if report.get(key) != expected[key]:
            raise ValueError(f"event-file checksum linkage is stale for {key}")


def _finalize_artifact_worker_exit(
    metadata_path: Path,
    boundary: Mapping[str, Any],
    *,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    metadata = _read_json_object(metadata_path, "final native latent metadata")
    metadata["worker_exit_receipt"] = {
        "worker_started": boundary.get("worker_started"),
        "worker_exit_observed": boundary.get("worker_exit_observed"),
        "worker_exit_code": boundary.get("worker_exit_code"),
        "worker_pid": boundary.get("worker_pid"),
        "worker_termination_confirmed": boundary.get("worker_termination_confirmed"),
    }
    release = boundary.get("transformer_release") or boundary.get("transformer_release_receipt")
    if not isinstance(release, Mapping):
        raise ValueError("derived worker boundary is missing its transformer release receipt")
    after_release = release.get("memory_after_allocator_purge")
    if not isinstance(after_release, Mapping):
        raise ValueError("derived worker boundary is missing post-release memory evidence")
    if artifact_path is None or not artifact_path.is_file():
        raise ValueError("final artifact NPZ is missing before metadata finalization")
    metadata["transformer_release_receipt"] = dict(release)
    metadata["final_active_memory"] = after_release.get("active")
    metadata["final_allocator_cache"] = after_release.get("allocator_cache")
    metadata["final_allocator_cache_zero"] = release.get("allocator_cache_zero") is True
    metadata["final_artifact_npz_sha256"] = sha256_file(artifact_path)
    metadata["metadata_sha256"] = stable_metadata_sha256(metadata)
    _write_json(metadata_path, metadata)
    return metadata


def _failure_report(
    report: dict[str, Any],
    phase: str,
    worker: str,
    primary: BaseException | Mapping[str, Any],
    *,
    cleanup: BaseException | Mapping[str, Any] | None = None,
    cleanup_attempted: bool = False,
) -> dict[str, Any]:
    report["status"] = "failed"
    report["run_state"] = "failed"
    report["functional_success"] = False
    report["failure"] = {
        "active_phase": phase,
        "worker_identity": worker,
        "completed_stages": list(report.get("phase_order", [])),
        "primary_error": error_receipt(primary),
        "cleanup_error": error_receipt(cleanup),
        **failure_fields(primary, cleanup, cleanup_attempted=cleanup_attempted, cleanup_succeeded=cleanup is None if cleanup_attempted else False),
        "later_phase_suppression": {
            "derived_worker_suppressed": "derived-worker" not in report.get("phase_order", []),
            "decoder_suppressed": True,
            "media_suppressed": True,
            "retry_suppressed": True,
        },
    }
    decoder_phase = report.get("decoder_phase")
    if not isinstance(decoder_phase, dict):
        decoder_phase = {
            "status": "suppressed",
            "worker_launches": {"video": 0, "audio": 0},
            "retry_allowed": False,
            "replacement_worker_allowed": False,
        }
        report["decoder_phase"] = decoder_phase
    else:
        if decoder_phase.get("status") not in {"failed", "completed"}:
            decoder_phase["status"] = "suppressed"
        decoder_phase.setdefault("worker_launches", {"video": 0, "audio": 0})
        decoder_phase["retry_allowed"] = False
        decoder_phase["replacement_worker_allowed"] = False
    for identity in ("video", "audio"):
        section = report.get(f"{identity}_decoder")
        if not isinstance(section, dict):
            section = _decoder_section(identity)
            report[f"{identity}_decoder"] = section
        decoder_phase_value = report.get("decoder_phase", {})
        implemented_scope = decoder_phase_value.get("implemented_phase_scope", {}) if isinstance(decoder_phase_value, Mapping) else {}
        if isinstance(implemented_scope, Mapping) and implemented_scope.get(identity) is False:
            section["status"] = "not_started"
            section["suppression_reason"] = "phase_not_implemented"
            continue
        if section.get("status") not in {"failed", "completed"}:
            section["status"] = "suppressed"
            section["suppression_reason"] = phase
    if not isinstance(report.get("decoder_failure"), Mapping):
        existing_order = report.get("decoder_phase_order")
        observed_order = existing_order.get("observed", []) if isinstance(existing_order, Mapping) else []
        report["decoder_failure"] = {
            "failed_gate": phase,
            "worker_identity": worker,
            "primary_error": error_receipt(primary),
            "cleanup_error": error_receipt(cleanup),
            "suppressed_phases": [
                name for name in DECODER_PHASE_ORDER if name not in observed_order
            ],
            "retry_suppressed": True,
            "replacement_worker_suppressed": True,
        }
    order = report.get("decoder_phase_order")
    if not isinstance(order, Mapping) or not order.get("valid"):
        observed: list[str] = []
        if "final-native-latent-validation" in report.get("phase_order", []):
            observed = ["derived-finalization", "derived-release-gate"]
        report["decoder_phase_order"] = decoder_phase_order_receipt(
            observed,
            phase_status={
                phase_name: ("suppressed" if phase_name not in observed else "completed")
                for phase_name in DECODER_PHASE_ORDER
            },
        )
    report.setdefault("decoder_memory", {})
    report.setdefault("decoder_timing", {})
    report["latent_generation_status"] = "failed"
    report["video_status"] = report.get("video_decoder", {}).get("status", "suppressed")
    report["audio_status"] = report.get("audio_decoder", {}).get("status", "suppressed")
    if report["video_status"] == "not_started":
        report["video_status"] = "suppressed"
    if report["audio_status"] == "not_started":
        report["audio_status"] = "suppressed"
    report["standalone_media_status"] = "failed"
    report["mp4_mux_status"] = "not_performed"
    report["standalone_media"] = {
        "status": "failed",
        "latent_generation_status": report["latent_generation_status"],
        "video_status": report["video_status"],
        "audio_status": report["audio_status"],
        "standalone_media_status": report["standalone_media_status"],
        "mp4_mux_status": report["mp4_mux_status"],
    }
    report["mp4_mux"] = {"status": "not_performed", "invoked": False, "output_path": None}
    refresh_canonical_timing_eligibility(report)
    return report


def _parent_run(
    args: argparse.Namespace,
    paths: Mapping[str, Any],
    report: dict[str, Any],
    *,
    subprocess_runner: Callable[..., Any] | None = None,
    process_snapshot_runner: Callable[..., Any] | None = None,
    mux_timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    root = Path(args.checkpoint_root).expanduser().resolve()
    derived = Path(args.derived_transformer).expanduser().resolve()
    report["phase_order"].append("preflight")
    snapshot = capture_host_process_snapshot(runner=process_snapshot_runner)
    apply_host_process_snapshot(report, snapshot)
    _write_json(Path(paths["process_snapshot"]), snapshot)
    _write_json(Path(paths["report"]), report)
    preflight = validate_derived_filesystem(derived)
    report["git_identity"] = capture_git_identity()
    report["checkpoint_identity"] = _checkpoint_identity(root, derived, preflight)
    schedule_started = time.perf_counter()
    report["schedule_contract"] = build_full_schedule().receipt()
    report["timing_telemetry"]["schedule_construction_seconds"] = time.perf_counter() - schedule_started
    report["prompt"] = prompt_receipt(args.prompt)
    validate_packed_contract(report["packing"])
    _write_json(Path(paths["report"]), report)

    conditioning_arguments = [
        "--checkpoint-root", str(root),
        "--derived-transformer", str(derived),
        "--prompt", args.prompt,
        "--seed", str(args.seed),
        "--artifact", paths["conditioning_artifact"],
        "--receipt", paths["conditioning_receipt"],
        "--tolerance", str(args.active_memory_tolerance_bytes),
    ]
    if args.verbose:
        conditioning_arguments.append("--verbose")
    report["phase_order"].append("conditioning-worker")
    report["invocation"]["conditioning_worker_attempts"] = 1
    conditioning = _run_child("__conditioning-worker", conditioning_arguments, Path(paths["conditioning_log"]), Path(paths["conditioning_receipt"]), "conditioning")
    report["conditioning_worker"] = conditioning
    if not conditioning.get("worker_receipt_valid"):
        raise PhaseFailure("conditioning-worker", conditioning.get("primary_error") or "conditioning worker boundary failed", cleanup=conditioning.get("cleanup_error"), details={"child": conditioning}, cleanup_attempted=True)
    validate_worker_boundary(conditioning, identity="conditioning")
    try:
        validate_conditioning_receipt(conditioning)
        validate_conditioning_artifact_binding(conditioning, Path(paths["conditioning_artifact"]))
    except BaseException as exc:
        raise PhaseFailure("conditioning-release-gate", exc, details={"child": conditioning}) from exc
    report["prompt"] = conditioning.get("prompt", report["prompt"])
    report["geometry"] = conditioning.get("geometry", report["geometry"])
    report["packing"] = conditioning.get("packing", report["packing"])
    report["schedule_contract"] = conditioning.get("schedule_contract", report["schedule_contract"])
    report["memory_telemetry"]["conditioning"] = conditioning.get("memory_telemetry", {})
    report["timing_telemetry"]["conditioning"] = conditioning.get("timing_telemetry", {})
    _write_json(Path(paths["report"]), report)

    derived_arguments = [
        "--checkpoint-root", str(root),
        "--derived-transformer", str(derived),
        "--attempt-identifier", paths["attempt_identifier"],
        "--conditioning-artifact", paths["conditioning_artifact"],
        "--conditioning-receipt", paths["conditioning_receipt"],
        "--final-artifact", paths["final_artifact"],
        "--final-artifact-metadata", paths["final_artifact_metadata"],
        "--event-file", paths["event_file"],
        "--receipt", paths["derived_receipt"],
        "--tolerance", str(args.active_memory_tolerance_bytes),
    ]
    if args.verbose:
        derived_arguments.append("--verbose")
    report["phase_order"].append("derived-worker")
    report["invocation"]["derived_worker_attempts"] = 1
    derived_receipt = _run_child("__derived-worker", derived_arguments, Path(paths["derived_log"]), Path(paths["derived_receipt"]), "derived")
    report["derived_worker"] = derived_receipt
    report["memory_telemetry"]["derived"] = derived_receipt.get("memory_telemetry", {})
    report["timing_telemetry"]["derived"] = derived_receipt.get("timing_telemetry", {})
    report["denoising"] = derived_receipt.get("denoising", {})
    report["streamed_adaln_lifecycle"] = derived_receipt.get("streamed_adaln_lifecycle", {})
    report["cache_attribution"] = derived_receipt.get("cache_attribution", {})
    if not derived_receipt.get("worker_receipt_valid"):
        raise PhaseFailure("derived-worker", derived_receipt.get("primary_error") or "derived worker boundary failed", cleanup=derived_receipt.get("cleanup_error"), details={"child": derived_receipt}, cleanup_attempted=True)
    validate_worker_boundary(derived_receipt, identity="derived")
    if derived_receipt.get("transformer_release", {}).get("passed") is not True:
        raise PhaseFailure("derived-release-gate", "derived transformer release gate failed", details={"child": derived_receipt}, cleanup_attempted=True)
    metadata = _finalize_artifact_worker_exit(
        Path(paths["final_artifact_metadata"]),
        derived_receipt,
        artifact_path=Path(paths["final_artifact"]),
    )
    arrays = _load_npz(Path(paths["final_artifact"]))
    validate_final_artifact(
        metadata,
        arrays=arrays,
        artifact_path=Path(paths["final_artifact"]),
        metadata_path=Path(paths["final_artifact_metadata"]),
    )
    report["final_artifact"] = metadata
    event_summary = validate_event_stream(Path(paths["event_file"]))
    report["streamed_adaln_lifecycle"].update(event_summary)
    report["event_file_path"] = event_summary["event_file_path"]
    report["event_file_record_count"] = event_summary["event_file_record_count"]
    report["event_file_sha256"] = event_summary["event_file_sha256"]
    validate_event_file_linkage(report, Path(paths["event_file"]))
    report["total_event_records"] = event_summary["total_event_records"]
    report["cache_session_count"] = event_summary["cache_session_count"]
    report["sidecar_open_event_count"] = event_summary["sidecar_open_event_count"]
    report["sidecar_release_event_count"] = event_summary["sidecar_release_event_count"]
    report["validated_block_pairs"] = event_summary["validated_block_pairs"]
    validate_lifecycle_totals(report["streamed_adaln_lifecycle"])
    report["phase_order"].extend(["derived-release-gate", "final-native-latent-validation"])
    try:
        decoder_gate = validate_derived_decoder_gate(
            derived_receipt,
            Path(paths["final_artifact"]),
            Path(paths["final_artifact_metadata"]),
            derived_worker_attempts=report["invocation"]["derived_worker_attempts"],
            expected_attempt_identifier=paths["attempt_identifier"],
            expected_checkpoint_identity=report["checkpoint_identity"],
        )
    except DecoderGateFailure as exc:
        gate = dict(exc.gate_receipt)
        report["decoder_phase"] = {
            "status": "suppressed",
            "implemented_scope": "video_then_audio",
            "implemented_phase_scope": {"video": True, "audio": True},
            "derived_gate": gate,
            "worker_launches": {"video": 0, "audio": 0},
            "retry_allowed": False,
            "replacement_worker_allowed": False,
        }
        observed = ["derived-finalization"]
        phase_status = {phase: "suppressed" for phase in DECODER_PHASE_ORDER}
        phase_status["derived-finalization"] = "completed" if gate.get("finalization_passed") is True else "failed"
        if gate.get("finalization_passed") is True:
            observed.append("derived-release-gate")
            phase_status["derived-release-gate"] = "failed"
        report["decoder_phase_order"] = decoder_phase_order_receipt(observed, phase_status=phase_status)
        report["decoder_failure"] = {
            "failed_gate": exc.failed_gate,
            "worker_identity": "derived",
            "primary_error": error_receipt(exc.primary_error),
            "cleanup_error": error_receipt(exc.cleanup_error),
            "suppressed_phases": [phase for phase, status in phase_status.items() if status == "suppressed"],
            "retry_suppressed": True,
            "replacement_worker_suppressed": True,
        }
        report["latent_generation_status"] = "failed"
        report["video_status"] = "suppressed"
        report["audio_status"] = "suppressed"
        report["standalone_media_status"] = "suppressed"
        report["mp4_mux_status"] = "not_performed"
        report["standalone_media"] = {
            "status": "suppressed",
            "latent_generation_status": report["latent_generation_status"],
            "video_status": report["video_status"],
            "audio_status": report["audio_status"],
            "standalone_media_status": report["standalone_media_status"],
            "mp4_mux_status": report["mp4_mux_status"],
        }
        report["mp4_mux"] = {"status": "not_performed", "invoked": False, "output_path": None}
        raise
    video_arguments = [
        "--checkpoint-root", str(root),
        "--derived-transformer", str(derived),
        "--attempt-identifier", paths["attempt_identifier"],
        "--final-artifact", paths["final_artifact"],
        "--final-artifact-metadata", paths["final_artifact_metadata"],
        "--frames-partial", paths["frames_partial"],
        "--frames", paths["frames"],
        "--video-frame-manifest", paths["video_frame_manifest"],
        "--receipt", paths["video_worker_receipt"],
        "--tolerance", str(args.active_memory_tolerance_bytes),
    ]
    if args.verbose:
        video_arguments.append("--verbose")

    audio_arguments = [
        "--checkpoint-root", str(root),
        "--derived-transformer", str(derived),
        "--attempt-identifier", paths["attempt_identifier"],
        "--final-artifact", paths["final_artifact"],
        "--final-artifact-metadata", paths["final_artifact_metadata"],
        "--audio-partial", paths["audio_partial"],
        "--audio-wav", paths["audio_wav"],
        "--audio-manifest", paths["audio_manifest"],
        "--receipt", paths["audio_worker_receipt"],
        "--tolerance", str(args.active_memory_tolerance_bytes),
    ]
    if args.verbose:
        audio_arguments.append("--verbose")

    def launch_worker(identity: str) -> Mapping[str, Any]:
        if identity == VIDEO_WORKER_IDENTITY:
            report["phase_order"].append("video-worker")
            report["invocation"]["video_worker_attempts"] += 1
            return _run_child(
                "__video-worker",
                video_arguments,
                Path(paths["video_worker_log"]),
                Path(paths["video_worker_receipt"]),
                VIDEO_WORKER_IDENTITY,
            )
        if identity == AUDIO_WORKER_IDENTITY:
            report["phase_order"].append("audio-worker")
            report["invocation"]["audio_worker_attempts"] += 1
            return _run_child(
                "__audio-worker",
                audio_arguments,
                Path(paths["audio_worker_log"]),
                Path(paths["audio_worker_receipt"]),
                AUDIO_WORKER_IDENTITY,
            )
        raise RuntimeError(f"unknown decoder worker identity: {identity}")

    def validate_published_video() -> Mapping[str, Any]:
        if Path(paths["frames_partial"]).exists():
            raise ValueError("video staged directory remains after worker termination")
        return validate_video_frame_manifest(
            Path(paths["video_frame_manifest"]),
            Path(paths["frames"]),
            expected_attempt_identifier=paths["attempt_identifier"],
            expected_worker_identity=VIDEO_WORKER_IDENTITY,
        )

    def validate_published_audio() -> Mapping[str, Any]:
        if Path(paths["audio_partial"]).exists():
            raise ValueError("audio staged WAV remains after worker termination")
        return validate_audio_wav_manifest(
            Path(paths["audio_manifest"]),
            Path(paths["audio_wav"]),
            expected_attempt_identifier=paths["attempt_identifier"],
            expected_worker_identity=AUDIO_WORKER_IDENTITY,
        )

    decoder_orchestration = DecoderPhaseOrchestrator(
        derived_gate=decoder_gate,
        worker_launcher=launch_worker,
        implemented_phase_scope={"video": True, "audio": True},
        artifact_validators={"video": validate_published_video, "audio": validate_published_audio},
    ).run()
    report["decoder_phase"] = dict(decoder_orchestration["decoder_phase"])
    report["decoder_phase"]["reason"] = "Slice 3C implements video decode followed by audio decode and standalone publication"
    report["decoder_phase"]["implemented_scope"] = "video_then_audio"
    report["decoder_phase"]["implemented_phase_scope"] = {"video": True, "audio": True}
    report["video_decoder"] = dict(decoder_orchestration["video_decoder"])
    report["audio_decoder"] = dict(decoder_orchestration["audio_decoder"])
    report["video_artifacts"] = dict(decoder_orchestration["video_artifacts"])
    report["audio_artifacts"] = dict(decoder_orchestration["audio_artifacts"])
    report["decoder_memory"] = {
        "video": report["video_decoder"].get("worker_receipt", {}).get("memory_telemetry", {}),
        "audio": report["audio_decoder"].get("worker_receipt", {}).get("memory_telemetry", {}),
    }
    report["decoder_timing"] = {
        "video": report["video_decoder"].get("worker_receipt", {}).get("timing_telemetry", {}),
        "audio": report["audio_decoder"].get("worker_receipt", {}).get("timing_telemetry", {}),
    }
    report["decoder_phase_order"] = decoder_orchestration["decoder_phase_order"]
    report["decoder_failure"] = decoder_orchestration["decoder_failure"]
    if report["decoder_phase"].get("status") != "completed":
        failure = decoder_orchestration.get("decoder_failure") or {
            "primary_error": {"type": "DecoderPhaseFailure", "message": "video decoder phase failed"}
        }
        child = report["video_decoder"].get("worker_receipt") or {}
        raise PhaseFailure(
            "decoder-phase",
            failure.get("primary_error") or failure.get("primary_error_message") or "video decoder phase failed",
            cleanup=failure.get("cleanup_error"),
            details={"child": child, "decoder": decoder_orchestration},
            cleanup_attempted=failure.get("cleanup_attempted") is True,
        )
    report["latent_generation_status"] = "completed"
    report["video_status"] = "completed"
    report["audio_status"] = "completed"
    report["standalone_media_status"] = "completed"
    report["mp4_mux_status"] = "not_performed"
    report["standalone_media"] = {
        "status": "completed",
        "latent_generation_status": report["latent_generation_status"],
        "video_status": report["video_status"],
        "audio_status": report["audio_status"],
        "standalone_media_status": report["standalone_media_status"],
        "mp4_mux_status": report["mp4_mux_status"],
        "video_publication_state": report["video_artifacts"].get("publication_state"),
        "audio_publication_state": report["audio_artifacts"].get("publication_state"),
    }
    report["phase_order"].append("mp4-mux")
    report = apply_mp4_mux_report(
        report,
        paths,
        subprocess_runner=subprocess_runner,
        timeout_seconds=mux_timeout_seconds,
    )
    if report.get("status") == "failed":
        report["functional_success"] = False
        refresh_canonical_timing_eligibility(report)
        return report
    report["status"] = "success"
    report["run_state"] = "successful"
    report["functional_success"] = True
    report["failure"] = None
    refresh_canonical_timing_eligibility(report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run-derived-full-schedule", help="run the locked derived schedule and standalone PNG-plus-WAV proof")
    run.add_argument("--checkpoint-root", required=True)
    run.add_argument("--derived-transformer", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--active-memory-tolerance-bytes", required=True, type=int)
    run.add_argument("--operator-declared-uncontended", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=run_command)
    return parser


def validate_report(report: Mapping[str, Any]) -> None:
    if set(report) != REPORT_KEYS:
        raise ValueError(f"v0.5e report schema mismatch: missing={sorted(REPORT_KEYS - set(report))}, unexpected={sorted(set(report) - REPORT_KEYS)}")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("probe_identity") != PROBE_FORMAT:
        raise ValueError("v0.5e report identity mismatch")
    if type(report.get("functional_success")) is not bool:
        raise ValueError("v0.5e report functional_success must be a literal boolean")
    host = report.get("host_contention")
    if not isinstance(host, Mapping):
        raise ValueError("v0.5e report is missing host-contention evidence")
    for key in ("operator_declared_uncontended", "process_snapshot_captured", "canonical_timing_eligible"):
        if type(host.get(key)) is not bool:
            raise ValueError(f"v0.5e host-contention field {key} must be a literal boolean")
    conflicts = host.get("known_conflicting_processes")
    if not isinstance(conflicts, list) or not all(isinstance(item, Mapping) for item in conflicts):
        raise ValueError("v0.5e known_conflicting_processes must be a structured list")
    snapshot = host.get("process_snapshot")
    if host.get("process_snapshot_captured") is True and (
        not isinstance(snapshot, Mapping) or snapshot.get("capture_success") is not True
    ):
        raise ValueError("v0.5e process snapshot capture claim is not backed by snapshot evidence")
    if isinstance(snapshot, Mapping) and snapshot.get("known_conflicting_processes") != conflicts:
        raise ValueError("v0.5e known conflict list differs from its process snapshot")
    expected_eligibility = canonical_timing_eligibility(
        functional_success=report.get("functional_success") is True,
        operator_declared_uncontended=host.get("operator_declared_uncontended") is True,
        process_snapshot_captured=host.get("process_snapshot_captured") is True,
        known_conflicting_processes=conflicts,
    )
    if host.get("canonical_timing_eligible") is not expected_eligibility["canonical_timing_eligible"]:
        raise ValueError("v0.5e canonical timing eligibility does not match the exact four-gate formula")
    if host.get("canonical_timing_ineligibility_reasons") != expected_eligibility["canonical_timing_ineligibility_reasons"]:
        raise ValueError("v0.5e canonical timing ineligibility reasons are stale")
    decoder_order = report.get("decoder_phase_order")
    if not isinstance(decoder_order, Mapping) or decoder_order.get("valid") is not True:
        raise ValueError("v0.5e report is missing a valid decoder phase order receipt")
    validate_decoder_phase_order(decoder_order)
    for identity in ("video", "audio"):
        section = report.get(f"{identity}_decoder")
        if not isinstance(section, Mapping) or section.get("status") not in DECODER_STATUSES:
            raise ValueError(f"v0.5e report has an invalid {identity} decoder status")
    if report.get("status") == "success":
        if report.get("run_state") != "successful" or report.get("functional_success") is not True or report.get("failure") is not None:
            raise ValueError("successful report must have run_state=successful and failure=null")
        validate_schedule_contract(report.get("schedule_contract", {}))
        validate_lifecycle_totals(report.get("streamed_adaln_lifecycle", {}))
        validate_cache_attribution(report.get("cache_attribution", {}))
        if report.get("derived_worker", {}).get("cache_attribution") != report.get("cache_attribution"):
            raise ValueError("successful report cache attribution differs from the derived-worker receipt")
        validate_final_artifact(report.get("final_artifact", {}))
        validate_worker_boundary(report.get("conditioning_worker", {}), identity="conditioning")
        validate_worker_boundary(report.get("derived_worker", {}), identity="derived")
        if report["conditioning_worker"].get("conditioning_release", {}).get("passed") is not True or report["derived_worker"].get("transformer_release", {}).get("passed") is not True:
            raise ValueError("successful report must prove both worker release gates")
        required_statuses = {
            "latent_generation_status": "completed",
            "video_status": "completed",
            "audio_status": "completed",
            "standalone_media_status": "completed",
            "mp4_mux_status": "completed",
        }
        if any(report.get(key) != value for key, value in required_statuses.items()):
            raise ValueError("successful report has invalid standalone-media status fields")
        standalone_media = report.get("standalone_media")
        if (
            not isinstance(standalone_media, Mapping)
            or standalone_media.get("status") != "completed"
            or any(standalone_media.get(key) != value for key, value in required_statuses.items())
        ):
            raise ValueError("successful report must prove standalone media and MP4 completion")
        mp4_mux = report.get("mp4_mux")
        if (
            not isinstance(mp4_mux, Mapping)
            or mp4_mux.get("status") != "completed"
            or mp4_mux.get("invoked") is not True
            or not isinstance(mp4_mux.get("output_path"), str)
            or mp4_mux.get("retry_suppressed") is not True
            or mp4_mux.get("invocation_counts") != {"ffmpeg": 1, "ffprobe": 1}
        ):
            raise ValueError("successful report must prove exactly one completed MP4 mux attempt")
        launch_gate = mp4_mux.get("launch_gate")
        if not isinstance(launch_gate, Mapping):
            raise ValueError("successful report is missing the MP4 mux launch gate")
        validate_mux_launch_gate(launch_gate)
        mp4_artifact = report.get("mp4_artifact")
        if (
            not isinstance(mp4_artifact, Mapping)
            or mp4_artifact.get("publication_state") != "published"
            or not isinstance(mp4_artifact.get("mp4_path"), str)
            or not isinstance(mp4_artifact.get("manifest_path"), str)
            or not isinstance(mp4_artifact.get("mp4_sha256"), str)
            or len(mp4_artifact.get("mp4_sha256", "")) != 64
            or not isinstance(mp4_artifact.get("manifest_sha256"), str)
            or len(mp4_artifact.get("manifest_sha256", "")) != 64
            or not isinstance(mp4_artifact.get("size_bytes"), int)
            or mp4_artifact.get("size_bytes", 0) <= 0
        ):
            raise ValueError("successful report must prove the published MP4 artifact and manifest")
        if not isinstance(report.get("mux_timing"), Mapping) or report.get("mux_failure") is not None:
            raise ValueError("successful report must include mux timing and no mux failure")
        ffmpeg_receipt = mp4_mux.get("ffmpeg")
        ffprobe_receipt = mp4_mux.get("ffprobe")
        if (
            not isinstance(ffmpeg_receipt, Mapping)
            or not isinstance(ffprobe_receipt, Mapping)
            or ffmpeg_receipt.get("invoked") is not True
            or ffprobe_receipt.get("invoked") is not True
            or ffmpeg_receipt.get("returncode") != 0
            or ffprobe_receipt.get("returncode") != 0
            or not isinstance(ffmpeg_receipt.get("argv"), list)
            or not isinstance(ffprobe_receipt.get("argv"), list)
        ):
            raise ValueError("successful report must preserve successful ffmpeg and ffprobe receipts")
        if not isinstance(report.get("decoder_memory"), Mapping) or not isinstance(report.get("decoder_timing"), Mapping):
            raise ValueError("successful report must include decoder memory and timing sections")
        decoder_phase = report.get("decoder_phase")
        derived_gate = decoder_phase.get("derived_gate") if isinstance(decoder_phase, Mapping) else None
        if (
            not isinstance(decoder_phase, Mapping)
            or decoder_phase.get("status") != "completed"
            or decoder_phase.get("implemented_scope") != "video_then_audio"
            or decoder_phase.get("implemented_phase_scope") != {"video": True, "audio": True}
            or decoder_phase.get("worker_launches") != {"video": 1, "audio": 1}
            or not isinstance(derived_gate, Mapping)
            or derived_gate.get("passed") is not True
            or report.get("video_decoder", {}).get("status") != "completed"
            or report.get("video_decoder", {}).get("worker_launch_count") != 1
            or report.get("audio_decoder", {}).get("status") != "completed"
            or report.get("audio_decoder", {}).get("worker_launch_count") != 1
            or report.get("decoder_failure") is not None
            or report.get("decoder_phase_order", {}).get("observed") != list(DECODER_PHASE_ORDER)
        ):
            raise ValueError("successful report must prove the ordered video-then-audio decoder phase")
        video_artifacts = report.get("video_artifacts")
        if (
            not isinstance(video_artifacts, Mapping)
            or video_artifacts.get("publication_state") != "published"
            or video_artifacts.get("frame_count") != VIDEO_FRAME_COUNT
            or video_artifacts.get("width") != VIDEO_FRAME_WIDTH
            or video_artifacts.get("height") != VIDEO_FRAME_HEIGHT
            or video_artifacts.get("fps") != VIDEO_FRAME_FPS
            or video_artifacts.get("duration_seconds") != VIDEO_FRAME_DURATION_SECONDS
            or not isinstance(video_artifacts.get("manifest_sha256"), str)
        ):
            raise ValueError("successful report must prove the published 30-frame video manifest")
        validate_decoder_worker_receipt(report["video_decoder"].get("worker_receipt", {}), identity=VIDEO_WORKER_IDENTITY)
        for identity in (VIDEO_WORKER_IDENTITY, AUDIO_WORKER_IDENTITY):
            section = report[f"{identity}_decoder"]
            if section.get("release_gate_passed") is not True or section.get("allocator_cache_zero") is not True or section.get("published_artifact_valid") is not True:
                raise ValueError(f"successful report must prove the {identity} decoder release and publication gates")
        audio_artifacts = report.get("audio_artifacts")
        if (
            not isinstance(audio_artifacts, Mapping)
            or audio_artifacts.get("publication_state") != "published"
            or audio_artifacts.get("channels") != 2
            or audio_artifacts.get("sample_rate") != AUDIO_SAMPLE_RATE
            or audio_artifacts.get("sample_count") != AUDIO_SAMPLE_COUNT
            or audio_artifacts.get("sample_width_bytes") != AUDIO_SAMPLE_WIDTH_BYTES
            or audio_artifacts.get("duration_seconds") != AUDIO_DURATION_SECONDS
            or not isinstance(audio_artifacts.get("wav_sha256"), str)
            or not isinstance(audio_artifacts.get("manifest_sha256"), str)
        ):
            raise ValueError("successful report must prove the published stereo WAV manifest")
        validate_decoder_worker_receipt(report["audio_decoder"].get("worker_receipt", {}), identity=AUDIO_WORKER_IDENTITY)
        if (
            report["audio_decoder"].get("wav_manifest_valid") is not True
            or report["audio_decoder"].get("decode_count") != 1
            or report["audio_decoder"].get("vae_load_count") != 1
            or report["audio_decoder"].get("release_gate_passed") is not True
            or report["audio_decoder"].get("allocator_cache_zero") is not True
        ):
            raise ValueError("successful report must prove exactly-once audio decode and release")
        if (
            report.get("event_file_record_count", 0) != report.get("total_event_records")
            or report.get("cache_session_count") != EXPECTED_DENOISING_TRANSITIONS
            or report.get("sidecar_open_event_count") != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
            or report.get("sidecar_release_event_count") != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
            or report.get("streamed_adaln_lifecycle", {}).get("attribution_block_event_count") != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
            or report.get("streamed_adaln_lifecycle", {}).get("attribution_session_event_count") != EXPECTED_DENOISING_TRANSITIONS
            or report.get("validated_block_pairs") != EXPECTED_DENOISING_TRANSITIONS * EXPECTED_BLOCK_COUNT
            or not report.get("event_file_sha256")
        ):
            raise ValueError("successful report must link exactly 15 cache sessions and 750 valid event pairs")
        if report.get("generation_exclusions") != EXPECTED_GENERATION_EXCLUSIONS:
            raise ValueError("successful report generation exclusions do not match the exact standalone-media map")
    elif report.get("status") == "failed":
        if report.get("run_state") != "failed" or report.get("functional_success") is not False or not isinstance(report.get("failure"), Mapping):
            raise ValueError("failed report must preserve failed run state and failure evidence")
        required = {"primary_error_type", "primary_error_message", "primary_error_traceback", "cleanup_attempted", "cleanup_succeeded", "cleanup_error_type", "cleanup_error_message", "cleanup_error_traceback"}
        if not required.issubset(report["failure"]):
            raise ValueError("failed report is missing primary/cleanup error fields")
        mp4_status = report.get("mp4_mux_status")
        if mp4_status not in {"not_performed", "suppressed", "failed"}:
            raise ValueError("failed report has an invalid MP4 mux status")
        mp4_mux = report.get("mp4_mux")
        if not isinstance(mp4_mux, Mapping) or type(mp4_mux.get("invoked")) is not bool:
            raise ValueError("failed report must preserve MP4 invocation evidence")
        if mp4_status == "failed":
            if mp4_mux.get("status") != "failed" or not isinstance(report.get("mux_failure"), Mapping):
                raise ValueError("failed MP4 report must preserve mux failure evidence")
        elif mp4_mux.get("invoked") is not False:
            raise ValueError("suppressed or unperformed MP4 report cannot claim invocation")
        if report.get("standalone_media_status") == "completed" and mp4_status not in {"failed", "suppressed"}:
            raise ValueError("failed report has inconsistent standalone-media and MP4 statuses")
    elif report.get("status") == "incomplete":
        if report.get("run_state") != "incomplete" or report.get("functional_success") is not False:
            raise ValueError("incomplete report has an invalid run state")
    else:
        raise ValueError("report status must be incomplete, failed, or success")


def run_command(args: argparse.Namespace) -> int:
    orchestration_started = time.perf_counter()
    root = Path(args.output_root).expanduser().resolve()
    paths: dict[str, Any] | None = None
    report: dict[str, Any] | None = None
    try:
        validate_locked_prompt(args.prompt)
        validate_seed(args.seed)
        if args.active_memory_tolerance_bytes < 0:
            raise ValueError("active-memory-tolerance-bytes must be nonnegative")
        paths = ensure_attempt_namespace(root)
        report = _base_report(args, paths)
        _write_json(Path(paths["report"]), report)
        report = _parent_run(args, paths, report)
        report["timing_telemetry"]["parent_orchestration_seconds"] = time.perf_counter() - orchestration_started
        validate_report(report)
        _write_json(Path(paths["report"]), report)
        if report.get("status") == "failed":
            print(json.dumps({"status": "failed", "run_state": "failed", "report": paths["report"]}, indent=2))
            return 1
        print(json.dumps({"status": "success", "run_state": "successful", "report": paths["report"]}, indent=2))
        return 0
    except FileExistsError as exc:
        print(json.dumps({"status": "failed", "run_state": "failed", "error": error_receipt(exc)}, indent=2))
        return 1
    except PhaseFailure as exc:
        if report is None:
            raise
        child = exc.details.get("child", {}) if isinstance(exc.details, Mapping) else {}
        worker = child.get("worker_identity", "parent") if isinstance(child, Mapping) else "parent"
        _failure_report(report, exc.phase, worker, exc.primary_error, cleanup=exc.cleanup_error, cleanup_attempted=exc.cleanup_attempted)
    except BaseException as exc:
        if report is None:
            if paths is not None:
                report = _base_report(args, paths)
            else:
                print(json.dumps({"status": "failed", "error": error_receipt(exc)}, indent=2))
                return 1
        _failure_report(report, report.get("phase_order", ["preflight"])[-1] if report.get("phase_order") else "preflight", "parent", exc)
    if report is not None and paths is not None:
        report["timing_telemetry"]["parent_orchestration_seconds"] = time.perf_counter() - orchestration_started
        try:
            validate_report(report)
        except BaseException as validation_error:
            _failure_report(report, "report-validation", "parent", validation_error)
        _write_json(Path(paths["report"]), report)
        print(json.dumps({"status": "failed", "run_state": "failed", "report": paths["report"]}, indent=2))
        return 1
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "__conditioning-worker":
        return _conditioning_worker_main(raw[1:])
    if raw and raw[0] == "__derived-worker":
        return _derived_worker_main(raw[1:])
    if raw and raw[0] == "__video-worker":
        return _video_worker_main(raw[1:])
    if raw and raw[0] == "__audio-worker":
        return _audio_worker_main(raw[1:])
    parsed = build_parser().parse_args(raw)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())

"""MiniMax-H3 v0.5a decoder-only lifecycle and media proof.

Importing this module is deliberately MLX-free. The decode-deterministic-media
command imports MLX only after its output namespace has been checked and then loads
the video and audio VAEs directly, sequentially. It never constructs the pipeline,
text encoder, transformer, scheduler, or AdaLN cache.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import traceback
from typing import Any, Callable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_CHECKPOINT_ROOT = Path(
    "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va"
)
DEFAULT_OUTPUT_ROOT = ROOT / "out" / "v0.5a"
COMMITTED_BASELINE = "5c49be8 Add v0.4c real text conditioning parity"
ARTIFACT_FORMAT = "minimax-h3-mlx-v05a-decoder-only"
SCHEMA_VERSION = 1
RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES = 1 * 1024 * 1024
DETERMINISTIC_INPUT_METHOD = "flat-index-sine-pattern-float32-v1"
FINGERPRINT_METHOD = "sha256-shape-dtype-canonical-float32-values-v1"

VIDEO_PIXEL_MEAN = (0.485, 0.456, 0.406)
VIDEO_PIXEL_STD = (0.229, 0.224, 0.225)
VIDEO_FPS = 24

SOURCE_INSPECTION_FILES = (
    "minimax_h3_mlx/video_decode_layout.py",
    "minimax_h3_mlx/video_vae.py",
    "minimax_h3_mlx/audio_vae.py",
    "minimax_h3_mlx/load.py",
    "minimax_h3_mlx/packing.py",
    "minimax_h3_mlx/pipeline.py",
    "minimax_h3_mlx/media.py",
    "tests/test_video_vae_parity.py",
    "tests/test_audio_vae_parity.py",
    "tests/test_packing_parity.py",
)

from minimax_h3_mlx.video_decode_layout import (
    VideoDecodeLayout,
    resolve_video_decode_layout,
)

CONFIG_RELATIVE_PATHS = (
    "video_vae/config.json",
    "video_vae/source/config.json",
    "audio_vae/config.json",
    "audio_vae/metadata.json",
    "audio_vae/config.yaml",
)

RUNTIME_REFERENCE_KEYS = (
    "decoder",
    "config",
    "latent",
    "scaled_latent",
    "raw_decoded",
    "converted_output",
    "temporary_normalization",
)

SUCCESS_REPORT_KEYS = frozenset(
    {
        "status",
        "probe_format",
        "schema_version",
        "committed_baseline",
        "checkpoint_root",
        "component_paths",
        "config_file_checksums",
        "source_derived_contracts",
        "selected_minimum_geometries",
        "deterministic_inputs",
        "latent_scaling_formulas",
        "output_shapes_and_dtypes",
        "video_media",
        "audio_media",
        "phase_order",
        "generation_components",
        "video_vae_ever_loaded",
        "video_vae_currently_resident",
        "audio_vae_ever_loaded",
        "audio_vae_currently_resident",
        "video_memory",
        "audio_memory",
        "final_memory",
        "output_paths",
        "failure",
    }
)


class ProbeFailure(RuntimeError):
    """A runtime failure carrying phase and cleanup evidence for the report."""

    def __init__(
        self,
        phase: str,
        original_error: BaseException,
        *,
        cleanup_error: BaseException | None = None,
        completed_stages: Sequence[str] = (),
    ) -> None:
        self.phase = phase
        self.original_error = original_error
        self.cleanup_error = cleanup_error
        self.cleanup_result: Mapping[str, Any] | None = None
        self.completed_stages = list(completed_stages)
        suffix = f"; cleanup error: {cleanup_error}" if cleanup_error else ""
        super().__init__(f"{phase} failed: {original_error}{suffix}")


class ReleaseGateError(RuntimeError):
    """A decoder release could not prove the requested memory contract."""

    def __init__(self, message: str, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        super().__init__(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    decode = subparsers.add_parser(
        "decode-deterministic-media",
        help="decode the smallest deterministic native video and audio latents sequentially",
    )
    decode.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    decode.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    decode.add_argument(
        "--active-memory-tolerance-bytes",
        type=int,
        default=RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
    )
    decode.add_argument("--overwrite", action="store_true")
    decode.add_argument("--verbose", action="store_true")
    decode.set_defaults(func=cmd_decode_deterministic_media)
    return parser


def normalize_dtype_name(dtype: Any) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in value.shape)


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(_shape(value)), "dtype": normalize_dtype_name(value.dtype)}


def _mlx_core_for(value: Any, supplied: Any | None = None) -> Any | None:
    if supplied is not None:
        return supplied
    if getattr(value, "__mlx_array__", False):
        return getattr(value, "__mlx_core__", None)
    module = value.__class__.__module__
    if module == "mlx.core" or module.startswith("mlx."):
        try:
            import mlx.core as mx
        except ImportError:
            return None
        return mx
    return None


def _mlx_eval(mx: Any, *values: Any) -> None:
    evaluate = getattr(mx, "eval", None)
    if callable(evaluate):
        evaluate(*values)


def array_fingerprint(value: Any, mx: Any | None = None) -> str:
    """Fingerprint canonical float32 values without converting an MLX BF16 array directly."""
    dispatch = _mlx_core_for(value, mx)
    if dispatch is not None:
        canonical = value.astype(dispatch.float32)
        _mlx_eval(dispatch, canonical)
        array = np.array(canonical, dtype=np.float32, copy=True)
    else:
        array = np.array(value, dtype=np.float32, copy=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|float32|")
    digest.update(np.ascontiguousarray(array).tobytes(order="C"))
    return digest.hexdigest()


def deterministic_values(element_count: int, salt: int) -> tuple[float, ...]:
    if not isinstance(element_count, int) or isinstance(element_count, bool) or element_count <= 0:
        raise ValueError("deterministic latent element count must be a positive integer")
    if salt not in (0, 1):
        raise ValueError("deterministic latent salt must be 0 for video or 1 for audio")
    values = []
    for index in range(element_count):
        phase = float(index + 1 + salt * 17)
        values.append(float(0.125 * math.sin(phase * 0.173) + 0.03125 * math.cos(phase * 0.071)))
    return tuple(values)


def make_deterministic_latent(mx: Any, shape: Sequence[int], salt: int) -> Any:
    shape = tuple(int(item) for item in shape)
    if not shape or any(item <= 0 for item in shape):
        raise ValueError(f"latent shape must contain positive dimensions: {shape}")
    values = np.asarray(deterministic_values(math.prod(shape), salt), dtype=np.float32).reshape(shape)
    return mx.array(values, dtype=mx.float32)


def video_decoded_frame_count(latent_frames: int, layout: VideoDecodeLayout) -> int:
    """Mirror VideoVAE.decode chunk, overlap, padding, and tail-trim arithmetic."""
    if latent_frames <= 0:
        raise ValueError("video latent frame count must be positive")
    ratio_t = layout.temporal_compression_ratio
    chunk_tokens = layout.tokens_chunk_size
    token_drop = layout.token_drop
    num_tokens = latent_frames + token_drop
    pad_tokens = (-num_tokens) % chunk_tokens
    num_chunks = (num_tokens + pad_tokens) // chunk_tokens - int(token_drop > 0)
    if num_chunks < 1:
        raise ValueError(
            "video latent frame count is below the native decoder minimum "
            f"of {layout.minimum_latent_frames}"
        )
    output_frames = num_chunks * (
        layout.chunk_num_frames - layout.frame_pre_padding
    ) + layout.frame_overlap
    if pad_tokens > 0:
        before_pad = latent_frames
        output_frames -= sum(
            layout.tail_trim_remainder
            if layout.tail_trim_remainder and (before_pad + offset) % chunk_tokens == 0
            else ratio_t
            for offset in range(pad_tokens)
        )
    return int(output_frames)


def select_minimum_video_geometry(
    config: Any, layout: VideoDecodeLayout
) -> dict[str, int | str]:
    minimum_frames = layout.minimum_latent_frames
    geometry = {
        "batch": 1,
        "latent_channels": int(config.latent_channels),
        "latent_frames": minimum_frames,
        "latent_height": 1,
        "latent_width": 1,
        "reason": (
            "minimum native temporal floor from VideoVAE.decode; one positive latent voxel per "
            "spatial axis; the ViT decoder expands every voxel directly into a patch"
        ),
    }
    validate_video_geometry(config, geometry, layout)
    return geometry


def select_minimum_audio_geometry(config: Any) -> dict[str, int | str]:
    geometry = {
        "batch": 2,
        "latent_channels": int(config.latent_channels),
        "latent_length": 1,
        "reason": "one native latent per stereo batch item; the decoder has no positive-length floor above one",
    }
    validate_audio_geometry(config, geometry)
    return geometry


def validate_video_geometry(
    config: Any, geometry: Mapping[str, Any], layout: VideoDecodeLayout
) -> None:
    expected_channels = int(config.latent_channels)
    if int(geometry["batch"]) != 1 or int(geometry["latent_channels"]) != expected_channels:
        raise ValueError("video latent geometry does not match native batch or channel contract")
    minimum = layout.minimum_latent_frames
    if int(geometry["latent_frames"]) < minimum:
        raise ValueError(f"video latent frames must be at least {minimum}")
    if int(geometry["latent_height"]) < 1 or int(geometry["latent_width"]) < 1:
        raise ValueError("video latent height and width must be positive")


def validate_audio_geometry(config: Any, geometry: Mapping[str, Any]) -> None:
    if int(geometry["batch"]) != 2 or int(geometry["latent_channels"]) != int(config.latent_channels):
        raise ValueError("audio geometry must use two stereo batch items and native latent channels")
    if int(geometry["latent_length"]) < 1:
        raise ValueError("audio latent length must be positive")
    if math.prod(tuple(int(rate) for rate in config.decoder_rates)) != int(config.hop_length):
        raise ValueError("audio decoder rates do not equal the native encoder hop length")


def validate_native_video_latent(value: Any, config: Any, geometry: Mapping[str, Any], mx: Any) -> None:
    expected = (
        int(geometry["batch"]),
        int(geometry["latent_channels"]),
        int(geometry["latent_frames"]),
        int(geometry["latent_height"]),
        int(geometry["latent_width"]),
    )
    if _shape(value) != expected:
        raise ValueError(f"video native latent shape mismatch: {_shape(value)} != {expected}")
    if normalize_dtype_name(value.dtype) != normalize_dtype_name(mx.float32):
        raise ValueError(f"video native latent dtype must be float32, got {value.dtype}")


def validate_native_audio_latent(value: Any, config: Any, geometry: Mapping[str, Any], mx: Any) -> None:
    expected = (
        int(geometry["batch"]),
        int(geometry["latent_channels"]),
        int(geometry["latent_length"]),
    )
    if _shape(value) != expected:
        raise ValueError(f"audio native latent shape mismatch: {_shape(value)} != {expected}")
    if normalize_dtype_name(value.dtype) != normalize_dtype_name(mx.float32):
        raise ValueError(f"audio native latent dtype must be float32, got {value.dtype}")


def expected_video_output_shape(
    config: Any, geometry: Mapping[str, Any], layout: VideoDecodeLayout
) -> tuple[int, ...]:
    return (
        1,
        int(config.out_channels),
        video_decoded_frame_count(int(geometry["latent_frames"]), layout),
        int(geometry["latent_height"]) * int(config.spatial_compression_ratio),
        int(geometry["latent_width"]) * int(config.spatial_compression_ratio),
    )


def expected_audio_output_shape(config: Any, geometry: Mapping[str, Any]) -> tuple[int, ...]:
    return (
        int(geometry["batch"]),
        1,
        int(geometry["latent_length"]) * int(config.hop_length),
    )


def validate_video_output(
    raw: np.ndarray,
    frames: np.ndarray,
    config: Any,
    geometry: Mapping[str, Any],
    layout: VideoDecodeLayout,
) -> None:
    expected_raw = expected_video_output_shape(config, geometry, layout)
    if tuple(raw.shape) != expected_raw:
        raise ValueError(f"raw video shape mismatch: {raw.shape} != {expected_raw}")
    if raw.dtype != np.float32:
        raise ValueError(f"raw video must be materialized as float32, got {raw.dtype}")
    if not np.isfinite(raw).all():
        raise ValueError("raw video decoder output contains non-finite values")
    expected_frames = (expected_raw[2], expected_raw[3], expected_raw[4], 3)
    if tuple(frames.shape) != expected_frames:
        raise ValueError(f"final RGB frame shape mismatch: {frames.shape} != {expected_frames}")
    if frames.dtype != np.uint8:
        raise ValueError(f"final RGB frames must be uint8, got {frames.dtype}")
    if frames.size and (int(frames.min()) < 0 or int(frames.max()) > 255):
        raise ValueError("final RGB frame values are outside [0, 255]")


def validate_audio_output(raw: np.ndarray, waveform: np.ndarray, config: Any, geometry: Mapping[str, Any]) -> None:
    expected_raw = expected_audio_output_shape(config, geometry)
    if tuple(raw.shape) != expected_raw:
        raise ValueError(f"raw audio shape mismatch: {raw.shape} != {expected_raw}")
    if raw.dtype != np.float32:
        raise ValueError(f"raw audio must be materialized as float32, got {raw.dtype}")
    if not np.isfinite(raw).all():
        raise ValueError("raw audio decoder output contains non-finite values")
    expected_waveform = (2, expected_raw[2])
    if tuple(waveform.shape) != expected_waveform:
        raise ValueError(f"stereo waveform shape mismatch: {waveform.shape} != {expected_waveform}")
    if waveform.dtype != np.float32:
        raise ValueError(f"stereo waveform must be float32, got {waveform.dtype}")
    if waveform.shape[1] <= 0:
        raise ValueError("audio waveform must have nonzero duration")
    if not np.isfinite(waveform).all():
        raise ValueError("audio waveform contains non-finite values")


def video_frames_from_raw(raw: np.ndarray) -> np.ndarray:
    mean = np.asarray(VIDEO_PIXEL_MEAN, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    std = np.asarray(VIDEO_PIXEL_STD, dtype=np.float32).reshape(1, 3, 1, 1, 1)
    frames = np.clip(raw * std + mean, 0.0, 1.0)[0].transpose(1, 2, 3, 0)
    return (frames * 255.0 + 0.5).astype(np.uint8)


def _build_video_source_contract(
    video_config: Any, video_layout: VideoDecodeLayout
) -> dict[str, Any]:
    return {
        "decoder_signature": "VideoVAE.decode(self, z: mx.array) -> mx.array",
        "native_latent_axis_order": "(B, C, F, H, W)",
        "latent_channels": int(video_config.latent_channels),
        "spatial_compression_ratio": int(video_config.spatial_compression_ratio),
        "temporal_compression_ratio": video_layout.temporal_compression_ratio,
        "clip_length": video_layout.clip_length,
        "tokens_chunk_size": video_layout.tokens_chunk_size,
        "token_drop": video_layout.token_drop,
        "token_overlap": video_layout.token_overlap,
        "frame_pre_padding": video_layout.frame_pre_padding,
        "frame_overlap": video_layout.frame_overlap,
        "chunk_num_frames": video_layout.chunk_num_frames,
        "tail_trim_remainder": video_layout.tail_trim_remainder,
        "minimum_accepted_latent_frame_count": video_layout.minimum_latent_frames,
        "minimum_accepted_latent_height": 1,
        "minimum_accepted_latent_width": 1,
        "divisibility_or_padding_requirements": (
            "positive latent H/W; no decoder divisibility check; temporal decode repeats tail "
            "latents for chunk alignment and trims the repeated pixel tail"
        ),
        "latent_mean_shape": [len(video_config.latents_mean)],
        "latent_std_shape": [len(video_config.latents_std)],
        "expected_input_dtype": "float32 (production path casts normalized latents to mx.float32)",
        "internal_decode_dtype": (
            "production decode input is float32; ViT RMS/LN normalization uses float32 intermediates; "
            "no separate decoder-wide dtype conversion"
        ),
        "raw_decoded_axis_order": "(B, C, F, H, W)",
        "raw_decoded_value_normalization": "ImageNet-normalized RGB over a [0, 1] base range",
        "final_rgb_conversion": (
            "raw * PIXEL_STD + PIXEL_MEAN, clip to [0, 1], select batch 0, transpose to "
            "(F, H, W, 3), round via *255 + 0.5, cast uint8"
        ),
        "decoded_frame_count_formula": (
            "num_tokens=F+token_drop; pad_tokens=(-num_tokens)%tokens_chunk_size; "
            "num_chunks=(num_tokens+pad_tokens)//tokens_chunk_size-(token_drop>0); "
            "frames=num_chunks*(chunk_num_frames-frame_pre_padding)+frame_overlap "
            "minus the exact repeated-tail trim for pad_tokens; "
            "tail trim uses clip_length%ratio_t for chunk-boundary positions"
        ),
        "decoded_height_formula": "latent_height * spatial_compression_ratio",
        "decoded_width_formula": "latent_width * spatial_compression_ratio",
        "temporal_or_spatial_tiling": {
            "spatial": "enabled by default at 256 pixels with 64-pixel minimum overlap",
            "temporal": (
                "chunked decode: "
                f"clip_length={video_layout.clip_length}, "
                f"{video_layout.tokens_chunk_size} latent tokens/chunk, "
                f"token_drop={video_layout.token_drop}, "
                f"token_overlap={video_layout.token_overlap}, "
                f"frame_pre_padding={video_layout.frame_pre_padding}, "
                f"frame_overlap={video_layout.frame_overlap}"
            ),
        },
        "smallest_geometry_exercises_decoder": (
            "yes: the native ViT decoder runs all 36 transformer blocks and its direct "
            "temporal/spatial patch projection; 1x1 spatial geometry bypasses optional spatial tiling"
        ),
    }


def _build_audio_source_contract(audio_config: Any) -> dict[str, Any]:
    audio_hop = int(audio_config.hop_length)
    return {
        "decoder_signature": "AudioVAE.decode(self, latents: mx.array) -> mx.array",
        "native_latent_axis_order": "(B, C, L)",
        "latent_channels": int(audio_config.latent_channels),
        "stereo_representation": "two batch items, each mono (B, 1, samples); pipeline returns (2, samples)",
        "latent_rate": int(audio_config.sampling_rate // audio_hop),
        "sample_rate": int(audio_config.sampling_rate),
        "minimum_accepted_latent_length": 1,
        "stride_or_divisibility_requirements": (
            f"decoder rates {list(audio_config.decoder_rates)} multiply to hop {audio_hop}; "
            "no explicit positive-length divisibility guard"
        ),
        "latent_mean_shape": [len(audio_config.latents_mean)],
        "latent_std_shape": [len(audio_config.latents_std)],
        "expected_input_dtype": "float32 (production path casts normalized latents to mx.float32)",
        "internal_decode_dtype": "production input is float32; decoder operations retain the input/weight MLX dtypes",
        "raw_waveform_axis_order": "(B, 1, samples)",
        "exact_output_sample_count_formula": f"latent_length * {audio_hop}",
        "output_clipping_or_normalization": (
            "BigVGAN decoder applies mx.clip(output, -1, 1); save_wav also clips before int16 PCM; "
            "the probe does not normalize audio"
        ),
        "non_finite_behavior": "decoder output is not masked; the probe rejects any non-finite raw or stereo waveform value",
    }


def build_source_contracts(
    video_config: Any,
    audio_config: Any,
    video_layout: VideoDecodeLayout,
) -> dict[str, Any]:
    return {
        "video": _build_video_source_contract(video_config, video_layout),
        "audio": _build_audio_source_contract(audio_config),
        "inspected_files": list(SOURCE_INSPECTION_FILES),
    }


def build_video_source_contract(
    video_config: Any, video_layout: VideoDecodeLayout
) -> dict[str, Any]:
    return _build_video_source_contract(video_config, video_layout)


def build_audio_source_contract(audio_config: Any) -> dict[str, Any]:
    return _build_audio_source_contract(audio_config)


def validate_phase_order(events: Sequence[str]) -> None:
    required = (
        "video-baseline",
        "video-config-load",
        "video-vae-load",
        "video-decode",
        "video-runtime-clear",
        "video-release-gate",
        "audio-baseline",
        "audio-config-load",
        "audio-vae-load",
        "audio-decode",
        "audio-runtime-clear",
        "audio-release-gate",
    )
    positions = {}
    for name in required:
        if name not in events:
            raise ValueError(f"phase order is missing {name}")
        positions[name] = events.index(name)
    if any(positions[left] >= positions[right] for left, right in zip(required, required[1:])):
        raise ValueError(f"phase order violates sequential decoder lifecycle: {list(events)}")


def run_sequential_phases(
    video_phase: Callable[[], Mapping[str, Any]],
    audio_phase: Callable[[], Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Run a testable release-gated video-then-audio sequence."""
    try:
        video = video_phase()
    except ProbeFailure:
        raise
    except BaseException as exc:
        state = getattr(video_phase, "probe_state", {})
        failure = ProbeFailure(
            "video",
            exc,
            completed_stages=state.get("completed_stages", ()),
        )
        failure.residency = dict(state.get("residency", {}))
        failure.partial_output_paths = list(state.get("partial_output_paths", ()))
        failure.geometries = {
            key: dict(value) for key, value in state.get("geometries", {}).items()
        }
        failure.memory = {
            key: dict(value) for key, value in state.get("memory", {}).items()
        }
        failure.phase_order = list(state.get("phase_order", ()))
        raise failure from exc
    video_release = video.get("release_gate", {})
    if not bool(video_release.get("passed")):
        raise ReleaseGateError("video release gate failed; audio phase was suppressed", video_release)
    if video.get("video_vae_currently_resident") is True:
        raise ReleaseGateError(
            "video VAE is still resident; audio phase was suppressed",
            {**video_release, "video_vae_currently_resident": True},
        )
    try:
        audio = audio_phase()
    except ProbeFailure:
        raise
    except BaseException as exc:
        state = getattr(audio_phase, "probe_state", {})
        failure = ProbeFailure(
            "audio",
            exc,
            completed_stages=state.get("completed_stages", ()),
        )
        failure.residency = dict(state.get("residency", {}))
        failure.partial_output_paths = list(state.get("partial_output_paths", ()))
        failure.geometries = {
            key: dict(value) for key, value in state.get("geometries", {}).items()
        }
        failure.memory = {
            key: dict(value) for key, value in state.get("memory", {}).items()
        }
        failure.phase_order = list(state.get("phase_order", ()))
        raise failure from exc
    audio_release = audio.get("release_gate", {})
    if not bool(audio_release.get("passed")):
        raise ReleaseGateError("audio release gate failed; overall proof is not successful", audio_release)
    if audio.get("audio_vae_currently_resident") is True:
        raise ReleaseGateError(
            "audio VAE is still resident; overall proof is not successful",
            {**audio_release, "audio_vae_currently_resident": True},
        )
    return video, audio


def clear_runtime_references(references: dict[str, Any]) -> bool:
    for key in RUNTIME_REFERENCE_KEYS:
        references[key] = None
    return all(references.get(key) is None for key in RUNTIME_REFERENCE_KEYS)


def _phase_cleanup(
    mx: Any,
    references: dict[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerance_bytes: int,
    on_runtime_clear: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Clear a phase's registry and allocator even when the decoder never loaded."""
    callback_error: BaseException | None = None
    if on_runtime_clear is not None:
        try:
            on_runtime_clear()
        except BaseException as exc:
            callback_error = exc
    try:
        result = release_runtime(mx, references, baseline, tolerance_bytes=tolerance_bytes)
    except BaseException as release_error:
        if callback_error is not None:
            raise release_error from callback_error
        raise
    if callback_error is not None:
        raise ReleaseGateError(
            "runtime-clear bookkeeping failed",
            {"passed": False, "cleanup_error": _json_error(callback_error)},
        ) from callback_error
    return result


def execute_scoped_phase(
    worker: Callable[[], Mapping[str, Any]],
    *,
    phase: str,
    mx: Any,
    references: dict[str, Any],
    baseline: Mapping[str, Any],
    tolerance_bytes: int,
    on_runtime_clear: Callable[[], None] | None = None,
    on_release_success: Callable[[Mapping[str, Any]], None] | None = None,
) -> Mapping[str, Any]:
    """Run a decoder worker, then release it after the worker scope has unwound.

    The worker must return CPU-only metadata and paths. Its decoder, MLX arrays, and
    converted media therefore leave local scope before the release gate is evaluated.
    """
    try:
        worker_result = worker()
    except BaseException as original_error:
        _clear_exception_frames(original_error)
        cleanup_error: BaseException | None = None
        cleanup_result: Mapping[str, Any] | None = None
        try:
            cleanup_result = _phase_cleanup(
                mx,
                references,
                baseline,
                tolerance_bytes=tolerance_bytes,
                on_runtime_clear=on_runtime_clear,
            )
        except BaseException as cleanup_exc:
            _clear_exception_frames(cleanup_exc)
            if isinstance(cleanup_exc, ReleaseGateError):
                cleanup_result = cleanup_exc.result
            cleanup_error = (
                cleanup_exc.__cause__
                if isinstance(cleanup_exc, ReleaseGateError) and cleanup_exc.__cause__ is not None
                else cleanup_exc
            )
        failure = ProbeFailure(
            phase,
            original_error,
            cleanup_error=cleanup_error,
        )
        failure.cleanup_result = cleanup_result
        if cleanup_result is not None and bool(cleanup_result.get("passed")):
            if on_release_success is not None:
                on_release_success(cleanup_result)
        raise failure from original_error

    try:
        release_gate = _phase_cleanup(
            mx,
            references,
            baseline,
            tolerance_bytes=tolerance_bytes,
            on_runtime_clear=on_runtime_clear,
        )
    except BaseException as release_error:
        raise ProbeFailure(phase, release_error) from release_error
    if on_release_success is not None:
        on_release_success(release_gate)
    return {**dict(worker_result), "release_gate": release_gate}


def _process_memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        return {"process_rss_bytes": int(psutil.Process().memory_info().rss), "process_rss_kind": "current"}
    except Exception:
        try:
            return {
                "process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                "process_rss_kind": "peak",
            }
        except Exception:
            return {"process_rss_bytes": None, "process_rss_kind": "unavailable"}


def memory_snapshot(mx: Any) -> dict[str, Any]:
    result = _process_memory_snapshot()
    for label, getter_name in (
        ("mlx_active_bytes", "get_active_memory"),
        ("mlx_allocator_cache_bytes", "get_cache_memory"),
        ("mlx_peak_bytes", "get_peak_memory"),
    ):
        getter = getattr(mx, getter_name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except Exception:
            result[label] = None
    return result


def evaluate_release_gate(
    baseline: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    references_cleared: bool,
    allocator_purge_available: bool,
    tolerance_bytes: int = RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
) -> dict[str, Any]:
    baseline_active = baseline.get("mlx_active_bytes")
    after_active = after.get("mlx_active_bytes")
    after_cache = after.get("mlx_allocator_cache_bytes")
    active_available = isinstance(baseline_active, int) and isinstance(after_active, int)
    active_within = active_available and after_active <= baseline_active + tolerance_bytes
    cache_zero = after_cache == 0
    passed = bool(references_cleared and allocator_purge_available and active_within and cache_zero)
    return {
        "passed": passed,
        "references_cleared": bool(references_cleared),
        "allocator_purge_available": bool(allocator_purge_available),
        "active_memory_baseline_bytes": baseline_active,
        "active_memory_after_bytes": after_active,
        "active_memory_tolerance_bytes": int(tolerance_bytes),
        "active_memory_gate_available": bool(active_available),
        "active_memory_within_tolerance": bool(active_within),
        "allocator_cache_after_bytes": after_cache,
        "final_allocator_cache_gate": bool(cache_zero),
    }


def release_runtime(
    mx: Any,
    references: dict[str, Any],
    baseline: Mapping[str, Any],
    *,
    tolerance_bytes: int,
) -> dict[str, Any]:
    cleanup_errors: list[BaseException] = []
    try:
        references_cleared = clear_runtime_references(references)
    except BaseException as exc:
        references_cleared = False
        cleanup_errors.append(exc)
    try:
        gc.collect()
    except BaseException as exc:
        cleanup_errors.append(exc)
    before_purge = memory_snapshot(mx)
    clear_cache = getattr(mx, "clear_cache", None)
    purge_available = callable(clear_cache)
    purge_error: BaseException | None = None
    if purge_available:
        try:
            clear_cache()
        except BaseException as exc:
            purge_error = exc
            cleanup_errors.append(exc)
    after_purge = memory_snapshot(mx)
    result = evaluate_release_gate(
        baseline,
        after_purge,
        references_cleared=references_cleared and not cleanup_errors,
        allocator_purge_available=purge_available and purge_error is None,
        tolerance_bytes=tolerance_bytes,
    )
    result.update(
        {
            "memory_before_allocator_purge": before_purge,
            "memory_after_allocator_purge": after_purge,
            "allocator_purge_error": None
            if purge_error is None
            else {"type": type(purge_error).__name__, "message": str(purge_error)},
            "cleanup_errors": [_json_error(error) for error in cleanup_errors],
        }
    )
    if cleanup_errors:
        raise ReleaseGateError("decoder runtime cleanup failed", result) from cleanup_errors[0]
    if not result["passed"]:
        raise ReleaseGateError("decoder release gate failed", result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_file_checksums(checkpoint_root: Path) -> dict[str, str]:
    return {
        relative: sha256_file(checkpoint_root / relative)
        for relative in CONFIG_RELATIVE_PATHS
        if (checkpoint_root / relative).is_file()
    }


def output_paths(output_root: Path) -> dict[str, Any]:
    return {
        "root": str(output_root),
        "frames": str(output_root / "frames"),
        "audio_wav": str(output_root / "decoder-audio.wav"),
        "report": str(output_root / "decoder-report.json"),
    }


def ensure_output_namespace(output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing v0.5a outputs: {output_root}; pass --overwrite explicitly"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    frames = output_root / "frames"
    if overwrite and frames.exists():
        for path in frames.glob("frame_*.png"):
            if path.is_file():
                path.unlink()
    if overwrite:
        for path in (output_root / "decoder-audio.wav", output_root / "decoder-report.json"):
            if path.is_file():
                path.unlink()
    return output_paths(output_root)


def frame_file_metadata(frames_directory: Path, expected_count: int) -> list[dict[str, str]]:
    """Require an exact, contiguous frame sequence and checksum every frame."""
    if expected_count < 0:
        raise ValueError("expected video frame count must be nonnegative")
    frame_paths = sorted(frames_directory.glob("frame_*.png"))
    expected_paths = [frames_directory / f"frame_{index:05d}.png" for index in range(expected_count)]
    if len(frame_paths) != expected_count:
        raise ValueError(
            f"video frame-file count mismatch: {len(frame_paths)} files for {expected_count} decoded frames"
        )
    if frame_paths != expected_paths or any(not path.is_file() for path in expected_paths):
        raise ValueError("video frame-file paths do not match every expected decoded frame")
    checksums = [{"path": str(path), "sha256": sha256_file(path)} for path in expected_paths]
    if any(len(item["sha256"]) != 64 for item in checksums):
        raise ValueError("every video frame must have a SHA-256 checksum")
    return checksums


def validate_wav_metadata(
    metadata: Mapping[str, Any],
    waveform: np.ndarray,
    config: Any,
) -> None:
    expected_sample_count = int(waveform.shape[1])
    expected_sample_rate = int(config.sampling_rate)
    if metadata.get("channels") != 2:
        raise ValueError(f"WAV channel count must be 2, got {metadata.get('channels')}")
    if metadata.get("sample_rate") != expected_sample_rate:
        raise ValueError(
            f"WAV sample rate mismatch: {metadata.get('sample_rate')} != {expected_sample_rate}"
        )
    if metadata.get("sample_count") != expected_sample_count:
        raise ValueError(
            f"WAV sample count mismatch: {metadata.get('sample_count')} != {expected_sample_count}"
        )
    expected_duration = expected_sample_count / expected_sample_rate
    if not math.isclose(
        float(metadata.get("duration_seconds", -1.0)),
        expected_duration,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("WAV duration does not match sample count divided by sample rate")
    checksum = metadata.get("sha256")
    if not isinstance(checksum, str) or len(checksum) != 64:
        raise ValueError("WAV checksum is missing or malformed")


def wav_metadata(path: Path) -> dict[str, Any]:
    import wave

    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        sample_width = handle.getsampwidth()
    return {
        "channels": channels,
        "sample_rate": sample_rate,
        "sample_count": sample_count,
        "sample_width_bytes": sample_width,
        "duration_seconds": sample_count / sample_rate if sample_rate else 0.0,
        "sha256": sha256_file(path),
    }


def validate_report(report: Mapping[str, Any]) -> None:
    keys = set(report)
    if keys != set(SUCCESS_REPORT_KEYS):
        raise ValueError(
            f"strict v0.5a report schema mismatch: missing={sorted(SUCCESS_REPORT_KEYS - keys)}, "
            f"unexpected={sorted(keys - SUCCESS_REPORT_KEYS)}"
        )
    if report.get("probe_format") != ARTIFACT_FORMAT or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v0.5a report identity mismatch")
    if report.get("status") == "success":
        if report.get("failure") is not None:
            raise ValueError("successful v0.5a report must have failure=null")
        if not report["video_memory"]["release_gate"]["passed"]:
            raise ValueError("successful report cannot contain a failed video release gate")
        if not report["audio_memory"]["release_gate"]["passed"]:
            raise ValueError("successful report cannot contain a failed audio release gate")
        for field in (
            "video_vae_ever_loaded",
            "audio_vae_ever_loaded",
        ):
            if report.get(field) is not True:
                raise ValueError(f"successful report must prove {field}=true")
        for field in (
            "video_vae_currently_resident",
            "audio_vae_currently_resident",
        ):
            if report.get(field) is not False:
                raise ValueError(f"successful report must prove {field}=false")
        validate_phase_order(report["phase_order"])
        video_media = report.get("video_media", {})
        frame_count = video_media.get("frame_count")
        frame_checksums = video_media.get("frame_checksums")
        if not isinstance(frame_count, int) or not isinstance(frame_checksums, list):
            raise ValueError("successful report must include video frame count and checksums")
        if len(frame_checksums) != frame_count:
            raise ValueError("successful report frame checksum count must equal decoded frame count")
        audio_media = report.get("audio_media", {})
        wav = audio_media.get("wav")
        if not isinstance(wav, Mapping):
            raise ValueError("successful report must include WAV metadata")
        if wav.get("channels") != 2 or len(str(wav.get("sha256", ""))) != 64:
            raise ValueError("successful report must include stereo WAV metadata and checksum")
    elif report.get("status") == "failed":
        failure = report.get("failure")
        if not isinstance(failure, Mapping):
            raise ValueError("failed report must preserve a failure receipt")
        for field in ("active_phase", "completed_stages", "residency", "partial_output_paths", "error"):
            if field not in failure:
                raise ValueError(f"failed report is missing failure field {field}")
    else:
        raise ValueError("v0.5a report status must be success or failed")


def _json_error(exc: BaseException) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)}


def _clear_exception_frames(exc: BaseException) -> None:
    """Drop traceback-held decoder locals before failure cleanup reaches the gate."""
    try:
        traceback.clear_frames(exc.__traceback__)
    except BaseException:
        pass


def _append_stage(stages: list[str], phase_order: list[str], stage: str) -> None:
    stages.append(stage)
    phase_order.append(stage)


def _normalization_arrays(mx: Any, config: Any, *, audio: bool) -> tuple[Any, Any]:
    mean = np.asarray(config.latents_mean, dtype=np.float32)
    std = np.asarray(config.latents_std, dtype=np.float32)
    expected = int(config.latent_channels)
    if mean.shape != (expected,) or std.shape != (expected,):
        label = "audio" if audio else "video"
        raise ValueError(f"{label} latent normalization metadata must have shape [{expected}]")
    if audio:
        return (
            mx.array(mean, dtype=mx.float32).reshape(1, expected, 1),
            mx.array(std, dtype=mx.float32).reshape(1, expected, 1),
        )
    return (
        mx.array(mean, dtype=mx.float32).reshape(1, expected, 1, 1, 1),
        mx.array(std, dtype=mx.float32).reshape(1, expected, 1, 1, 1),
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _runtime_imports() -> tuple[Any, Any, Any, Any, Any, Any]:
    """Import only the two VAE config/load pairs and the existing media helpers."""
    import mlx.core as mx

    from minimax_h3_mlx.load import (
        load_audio_vae,
        load_audio_vae_config,
        load_video_vae,
        load_video_vae_config,
    )
    from minimax_h3_mlx.media import save_frames, save_wav

    return mx, load_video_vae_config, load_video_vae, load_audio_vae_config, load_audio_vae, (save_frames, save_wav)


def _runtime_run(args: argparse.Namespace, paths: Mapping[str, Any]) -> dict[str, Any]:
    mx, load_video_config, load_video, load_audio_config, load_audio, media_helpers = _runtime_imports()
    save_frames, save_wav = media_helpers
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    video_dir = checkpoint_root / "video_vae"
    audio_dir = checkpoint_root / "audio_vae"
    tolerance = int(args.active_memory_tolerance_bytes)
    phase_order: list[str] = []
    completed_stages: list[str] = []
    video_refs = {key: None for key in RUNTIME_REFERENCE_KEYS}
    audio_refs = {key: None for key in RUNTIME_REFERENCE_KEYS}
    video_memory: dict[str, Any] = {}
    audio_memory: dict[str, Any] = {}
    video_geometry: dict[str, Any] = {}
    audio_geometry: dict[str, Any] = {}
    video_media: dict[str, Any] = {}
    audio_media: dict[str, Any] = {}
    source_contracts: dict[str, Any] = {"inspected_files": list(SOURCE_INSPECTION_FILES)}
    residency = {
        "video_vae_ever_loaded": False,
        "video_vae_currently_resident": False,
        "audio_vae_ever_loaded": False,
        "audio_vae_currently_resident": False,
    }
    partial_output_paths: list[str] = []

    def record_partial_outputs() -> None:
        for frame_path in sorted(Path(paths["frames"]).glob("frame_*.png")):
            value = str(frame_path)
            if value not in partial_output_paths:
                partial_output_paths.append(value)
        audio_path = str(paths["audio_wav"])
        if Path(audio_path).is_file() and audio_path not in partial_output_paths:
            partial_output_paths.append(audio_path)

    def decorate_failure(failure: ProbeFailure, phase: str, geometry: Mapping[str, Any]) -> None:
        record_partial_outputs()
        failure.phase = phase
        failure.completed_stages = list(completed_stages)
        failure.residency = dict(residency)
        failure.partial_output_paths = list(partial_output_paths)
        failure.geometries = {
            "video": dict(video_geometry),
            "audio": dict(audio_geometry),
        }
        failure.memory = {"video": dict(video_memory), "audio": dict(audio_memory)}
        failure.phase_order = list(phase_order)

    def phase_state() -> dict[str, Any]:
        return {
            "completed_stages": completed_stages,
            "phase_order": phase_order,
            "residency": residency,
            "partial_output_paths": partial_output_paths,
            "geometries": {"video": video_geometry, "audio": audio_geometry},
            "memory": {"video": video_memory, "audio": audio_memory},
        }

    def run_video() -> Mapping[str, Any]:
        run_video.probe_state = phase_state()
        _append_stage(completed_stages, phase_order, "video-baseline")
        video_memory["memory_before_config"] = memory_snapshot(mx)

        def worker() -> Mapping[str, Any]:
            _append_stage(completed_stages, phase_order, "video-config-load")
            config = load_video_config(video_dir)
            video_refs["config"] = config
            layout = resolve_video_decode_layout(config)
            video_geometry.update(select_minimum_video_geometry(config, layout))
            source_contracts["video"] = build_video_source_contract(config, layout)
            latent = make_deterministic_latent(
                mx,
                (
                    video_geometry["batch"],
                    video_geometry["latent_channels"],
                    video_geometry["latent_frames"],
                    video_geometry["latent_height"],
                    video_geometry["latent_width"],
                ),
                salt=0,
            )
            video_refs["latent"] = latent
            validate_native_video_latent(latent, config, video_geometry, mx)
            mx.eval(latent)
            video_memory["latent_fingerprint"] = array_fingerprint(latent, mx)
            _append_stage(completed_stages, phase_order, "video-latent-validated")

            _append_stage(completed_stages, phase_order, "video-vae-load")
            # A loader may allocate before raising; cleanup must not depend on a return value.
            residency["video_vae_currently_resident"] = True
            decoder = load_video(video_dir)
            video_refs["decoder"] = decoder
            residency["video_vae_ever_loaded"] = True
            video_memory["memory_after_load"] = memory_snapshot(mx)
            scaled_mean, scaled_std = _normalization_arrays(mx, config, audio=False)
            video_refs["temporary_normalization"] = (scaled_mean, scaled_std)
            scaled = latent * scaled_std + scaled_mean
            video_refs["scaled_latent"] = scaled
            mx.eval(scaled)
            _append_stage(completed_stages, phase_order, "video-decode")
            raw = decoder.decode(scaled.astype(mx.float32))
            video_refs["raw_decoded"] = raw
            mx.eval(raw)
            raw_np = np.array(raw.astype(mx.float32), dtype=np.float32, copy=True)
            video_refs["raw_decoded"] = raw_np
            frames = video_frames_from_raw(raw_np)
            video_refs["converted_output"] = frames
            validate_video_output(raw_np, frames, config, video_geometry, layout)
            video_media.update(
                {
                    "raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": str(raw_np.dtype)},
                    "final_shape_dtype": {"shape": list(frames.shape), "dtype": str(frames.dtype)},
                    "frame_count": int(frames.shape[0]),
                    "height": int(frames.shape[1]),
                    "width": int(frames.shape[2]),
                }
            )
            _append_stage(completed_stages, phase_order, "video-output-write")
            save_frames(paths["frames"], frames)
            frame_checksums = frame_file_metadata(
                Path(paths["frames"]), int(frames.shape[0])
            )
            video_media["frame_checksums"] = frame_checksums
            partial_output_paths.extend(item["path"] for item in frame_checksums)
            return {"written_output_paths": [item["path"] for item in frame_checksums]}

        def before_video_release() -> None:
            video_memory["memory_before_release"] = memory_snapshot(mx)
            _append_stage(completed_stages, phase_order, "video-runtime-clear")

        def after_video_release(_: Mapping[str, Any]) -> None:
            residency["video_vae_currently_resident"] = False
            _append_stage(completed_stages, phase_order, "video-release-gate")

        try:
            result = execute_scoped_phase(
                worker,
                phase="video",
                mx=mx,
                references=video_refs,
                baseline=video_memory["memory_before_config"],
                tolerance_bytes=tolerance,
                on_runtime_clear=before_video_release,
                on_release_success=after_video_release,
            )
        except ProbeFailure as failure:
            if isinstance(failure.original_error, ReleaseGateError):
                video_memory["release_gate"] = failure.original_error.result
            if isinstance(failure.cleanup_error, ReleaseGateError):
                video_memory["release_gate"] = failure.cleanup_error.result
            if failure.cleanup_result is not None:
                video_memory["release_gate"] = dict(failure.cleanup_result)
            decorate_failure(failure, "video", video_geometry)
            raise
        video_memory["release_gate"] = result["release_gate"]
        return {
            **result,
            "video_vae_ever_loaded": residency["video_vae_ever_loaded"],
            "video_vae_currently_resident": residency["video_vae_currently_resident"],
        }

    def run_audio() -> Mapping[str, Any]:
        run_audio.probe_state = phase_state()
        if residency["video_vae_currently_resident"]:
            raise RuntimeError("video VAE must be nonresident before audio loading begins")
        _append_stage(completed_stages, phase_order, "audio-baseline")
        audio_memory["memory_before_config"] = memory_snapshot(mx)

        def worker() -> Mapping[str, Any]:
            _append_stage(completed_stages, phase_order, "audio-config-load")
            config = load_audio_config(audio_dir)
            audio_refs["config"] = config
            source_contracts["audio"] = build_audio_source_contract(config)
            audio_geometry.update(select_minimum_audio_geometry(config))
            latent = make_deterministic_latent(
                mx,
                (audio_geometry["batch"], audio_geometry["latent_channels"], audio_geometry["latent_length"]),
                salt=1,
            )
            audio_refs["latent"] = latent
            validate_native_audio_latent(latent, config, audio_geometry, mx)
            mx.eval(latent)
            audio_memory["latent_fingerprint"] = array_fingerprint(latent, mx)
            _append_stage(completed_stages, phase_order, "audio-latent-validated")

            _append_stage(completed_stages, phase_order, "audio-vae-load")
            # Treat the load attempt as potentially resident so an allocating loader
            # that raises still receives the same cleanup path.
            residency["audio_vae_currently_resident"] = True
            audio_refs["decoder"] = load_audio(audio_dir)
            residency["audio_vae_ever_loaded"] = True
            audio_memory["memory_after_load"] = memory_snapshot(mx)
            scaled_mean, scaled_std = _normalization_arrays(mx, config, audio=True)
            audio_refs["temporary_normalization"] = (scaled_mean, scaled_std)
            scaled = latent * scaled_std + scaled_mean
            audio_refs["scaled_latent"] = scaled
            mx.eval(scaled)
            _append_stage(completed_stages, phase_order, "audio-decode")
            raw = audio_refs["decoder"].decode(scaled.astype(mx.float32))
            audio_refs["raw_decoded"] = raw
            mx.eval(raw)
            raw_np = np.array(raw.astype(mx.float32), dtype=np.float32, copy=True)
            audio_refs["raw_decoded"] = raw_np
            waveform = raw_np[:, 0, :].astype(np.float32, copy=True)
            audio_refs["converted_output"] = waveform
            validate_audio_output(raw_np, waveform, config, audio_geometry)
            audio_media.update(
                {
                    "raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": str(raw_np.dtype)},
                    "final_shape_dtype": {"shape": list(waveform.shape), "dtype": str(waveform.dtype)},
                    "sample_rate": int(config.sampling_rate),
                    "sample_count": int(waveform.shape[1]),
                    "duration_seconds": float(waveform.shape[1] / config.sampling_rate),
                    "peak_absolute_amplitude": float(np.max(np.abs(waveform))),
                    "rms": float(np.sqrt(np.mean(np.square(waveform), dtype=np.float64))),
                    "decoder_clipping": "mx.clip(output, -1, 1) in BigVGANDecoder",
                    "media_helper_clipping": "save_wav clips to [-1, 1] before int16 PCM",
                }
            )
            _append_stage(completed_stages, phase_order, "audio-output-write")
            save_wav(paths["audio_wav"], waveform, int(config.sampling_rate))
            audio_media["wav"] = wav_metadata(Path(paths["audio_wav"]))
            validate_wav_metadata(audio_media["wav"], waveform, config)
            partial_output_paths.append(paths["audio_wav"])
            return {"written_output_paths": [paths["audio_wav"]]}

        def before_audio_release() -> None:
            audio_memory["memory_before_release"] = memory_snapshot(mx)
            _append_stage(completed_stages, phase_order, "audio-runtime-clear")

        def after_audio_release(_: Mapping[str, Any]) -> None:
            residency["audio_vae_currently_resident"] = False
            _append_stage(completed_stages, phase_order, "audio-release-gate")

        try:
            result = execute_scoped_phase(
                worker,
                phase="audio",
                mx=mx,
                references=audio_refs,
                baseline=audio_memory["memory_before_config"],
                tolerance_bytes=tolerance,
                on_runtime_clear=before_audio_release,
                on_release_success=after_audio_release,
            )
        except ProbeFailure as failure:
            if isinstance(failure.original_error, ReleaseGateError):
                audio_memory["release_gate"] = failure.original_error.result
            if isinstance(failure.cleanup_error, ReleaseGateError):
                audio_memory["release_gate"] = failure.cleanup_error.result
            if failure.cleanup_result is not None:
                audio_memory["release_gate"] = dict(failure.cleanup_result)
            decorate_failure(failure, "audio", audio_geometry)
            raise
        audio_memory["release_gate"] = result["release_gate"]
        return {
            **result,
            "audio_vae_ever_loaded": residency["audio_vae_ever_loaded"],
            "audio_vae_currently_resident": residency["audio_vae_currently_resident"],
        }

    video_result, audio_result = run_sequential_phases(run_video, run_audio)
    validate_phase_order(phase_order)
    final_memory = memory_snapshot(mx)
    report = _build_success_report(
        checkpoint_root,
        video_dir,
        audio_dir,
        paths,
        video_geometry,
        audio_geometry,
        video_result,
        audio_result,
        phase_order,
        video_memory,
        audio_memory,
        final_memory,
        video_media,
        audio_media,
        config_file_checksums(checkpoint_root),
        source_contracts,
        residency,
        partial_output_paths,
    )
    validate_report(report)
    return report


def _build_base_report(
    checkpoint_root: Path,
    video_dir: Path,
    audio_dir: Path,
    paths: Mapping[str, Any],
    checksums: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "status": "failed",
        "probe_format": ARTIFACT_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "committed_baseline": COMMITTED_BASELINE,
        "checkpoint_root": str(checkpoint_root),
        "component_paths": {"video_vae": str(video_dir), "audio_vae": str(audio_dir)},
        "config_file_checksums": dict(checksums),
        "source_derived_contracts": {},
        "selected_minimum_geometries": {},
        "deterministic_inputs": {},
        "latent_scaling_formulas": {
            "video": "scaled = deterministic_latent * latents_std + latents_mean",
            "audio": "scaled = deterministic_latent * latents_std + latents_mean",
        },
        "output_shapes_and_dtypes": {},
        "video_media": {},
        "audio_media": {},
        "phase_order": [],
        "generation_components": {
            "text_encoder_loaded": False,
            "transformer_loaded": False,
            "scheduler_loaded": False,
            "adaln_cache_loaded": False,
            "denoising_executed": False,
            "generation_loader_functions_invoked": [],
        },
        "video_vae_ever_loaded": False,
        "video_vae_currently_resident": False,
        "audio_vae_ever_loaded": False,
        "audio_vae_currently_resident": False,
        "video_memory": {},
        "audio_memory": {},
        "final_memory": {},
        "output_paths": dict(paths),
        "failure": None,
    }


def _build_success_report(
    checkpoint_root: Path,
    video_dir: Path,
    audio_dir: Path,
    paths: Mapping[str, Any],
    video_geometry: Mapping[str, Any],
    audio_geometry: Mapping[str, Any],
    video_result: Mapping[str, Any],
    audio_result: Mapping[str, Any],
    phase_order: Sequence[str],
    video_memory: Mapping[str, Any],
    audio_memory: Mapping[str, Any],
    final_memory: Mapping[str, Any],
    video_media: Mapping[str, Any],
    audio_media: Mapping[str, Any],
    checksums: Mapping[str, str],
    source_contracts: Mapping[str, Any],
    residency: Mapping[str, bool],
    partial_output_paths: Sequence[str],
) -> dict[str, Any]:
    report = _build_base_report(checkpoint_root, video_dir, audio_dir, paths, checksums)
    report.update(
        {
            "status": "success",
            "selected_minimum_geometries": {
                "video": dict(video_geometry),
                "audio": dict(audio_geometry),
            },
            "deterministic_inputs": {
                "method": DETERMINISTIC_INPUT_METHOD,
                "fingerprint_method": FINGERPRINT_METHOD,
                "video": {
                    "shape": [
                        video_geometry["batch"],
                        video_geometry["latent_channels"],
                        video_geometry["latent_frames"],
                        video_geometry["latent_height"],
                        video_geometry["latent_width"],
                    ],
                    "dtype": "float32",
                    "salt": 0,
                    "fingerprint": video_memory["latent_fingerprint"],
                },
                "audio": {
                    "shape": [
                        audio_geometry["batch"],
                        audio_geometry["latent_channels"],
                        audio_geometry["latent_length"],
                    ],
                    "dtype": "float32",
                    "salt": 1,
                    "fingerprint": audio_memory["latent_fingerprint"],
                },
            },
            "output_shapes_and_dtypes": {
                "video": video_media.get("final_shape_dtype"),
                "audio": audio_media.get("final_shape_dtype"),
            },
            "source_derived_contracts": dict(source_contracts),
            "video_media": dict(video_media),
            "audio_media": dict(audio_media),
            "phase_order": list(phase_order),
            "video_memory": dict(video_memory),
            "audio_memory": dict(audio_memory),
            "final_memory": dict(final_memory),
            "generation_components": {
                "text_encoder_loaded": False,
                "transformer_loaded": False,
                "scheduler_loaded": False,
                "adaln_cache_loaded": False,
                "denoising_executed": False,
                "generation_loader_functions_invoked": [],
            },
            "video_vae_ever_loaded": bool(residency["video_vae_ever_loaded"]),
            "video_vae_currently_resident": bool(residency["video_vae_currently_resident"]),
            "audio_vae_ever_loaded": bool(residency["audio_vae_ever_loaded"]),
            "audio_vae_currently_resident": bool(residency["audio_vae_currently_resident"]),
            "output_paths": {
                **dict(paths),
                "written_media": list(partial_output_paths),
            },
        }
    )
    return report


def _build_failure_report(
    base: Mapping[str, Any],
    failure: ProbeFailure,
    *,
    phase_order: Sequence[str],
    partial_output_paths: Sequence[str],
    geometries: Mapping[str, Mapping[str, Any]],
    memory: Mapping[str, Mapping[str, Any]],
    residency: Mapping[str, bool],
) -> dict[str, Any]:
    report = dict(base)
    report["status"] = "failed"
    report["phase_order"] = list(phase_order)
    report["selected_minimum_geometries"] = {key: dict(value) for key, value in geometries.items()}
    report["video_memory"] = dict(memory.get("video", {}))
    report["audio_memory"] = dict(memory.get("audio", {}))
    report["video_vae_ever_loaded"] = bool(residency["video_vae_ever_loaded"])
    report["video_vae_currently_resident"] = bool(residency["video_vae_currently_resident"])
    report["audio_vae_ever_loaded"] = bool(residency["audio_vae_ever_loaded"])
    report["audio_vae_currently_resident"] = bool(residency["audio_vae_currently_resident"])
    report["failure"] = {
        "active_phase": failure.phase,
        "completed_stages": list(failure.completed_stages),
        "residency": dict(residency),
        "partial_output_paths": list(partial_output_paths),
        "input_geometry": {key: dict(value) for key, value in geometries.items()},
        "error": _json_error(failure.original_error),
        "cleanup_error": None if failure.cleanup_error is None else _json_error(failure.cleanup_error),
    }
    return report


def cmd_decode_deterministic_media(args: argparse.Namespace) -> int:
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    paths = ensure_output_namespace(output_root, overwrite=bool(args.overwrite))
    checksums = config_file_checksums(checkpoint_root)
    base = _build_base_report(
        checkpoint_root,
        checkpoint_root / "video_vae",
        checkpoint_root / "audio_vae",
        paths,
        checksums,
    )
    try:
        report = _runtime_run(args, paths)
    except ProbeFailure as failure:
        report = _build_failure_report(
            base,
            failure,
            phase_order=getattr(failure, "phase_order", failure.completed_stages),
            partial_output_paths=getattr(failure, "partial_output_paths", ()),
            geometries=getattr(failure, "geometries", {}),
            memory=getattr(failure, "memory", {}),
            residency=getattr(
                failure,
                "residency",
                {
                    "video_vae_ever_loaded": False,
                    "video_vae_currently_resident": False,
                    "audio_vae_ever_loaded": False,
                    "audio_vae_currently_resident": False,
                },
            ),
        )
        validate_report(report)
        _write_json(Path(paths["report"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    except BaseException as exc:
        failure = ProbeFailure("setup", exc)
        report = _build_failure_report(
            base,
            failure,
            phase_order=(),
            partial_output_paths=(),
            geometries={},
            memory={},
            residency={
                "video_vae_ever_loaded": False,
                "video_vae_currently_resident": False,
                "audio_vae_ever_loaded": False,
                "audio_vae_currently_resident": False,
            },
        )
        validate_report(report)
        _write_json(Path(paths["report"]), report)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 1
    _write_json(Path(paths["report"]), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

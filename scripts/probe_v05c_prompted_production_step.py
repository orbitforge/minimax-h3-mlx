"""MiniMax-H3 v0.5c prompted production-step parity and decoder handoff proof.

The parent process is MLX-free.  It validates the locked contract, creates a private temporary
namespace, and launches the conditioning, resident-transformer, and derived-transformer phases as
separate child processes.  Only after exact parity and worker-release gates pass does the parent
load the video and audio VAEs sequentially.  The probe intentionally performs one production
Euler transition and writes diagnostic PNG/WAV output; it never muxes an MP4.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LOCKED_PROMPT = (
    "A single dodecahedron rotating slowly on one axis, centered in frame, with a fixed camera and no other objects. "
    "Clean black studio background. Each face is a different impossible material: stained glass, water, chrome, fabric, "
    "portal, soap bubble, crystal, lava, stone, smoke, moss, and a star-filled void. The shape stays perfectly rigid "
    "and symmetrical. Cinematic lighting, soft blue rim light, realistic reflections and refractions, surreal materials, "
    "high detail, smooth motion."
)
PROMPT_UTF8_BYTE_COUNT = len(LOCKED_PROMPT.encode("utf-8"))
PROMPT_SHA256 = hashlib.sha256(LOCKED_PROMPT.encode("utf-8")).hexdigest()

COMMITTED_BASELINE = "2336bc8 Add v0.5b production geometry bridge"
PROBE_FORMAT = "minimax-h3-mlx-v05c-prompted-production-step"
SCHEMA_VERSION = 1
CANONICAL_SEED = 0
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0
CANONICAL_TRANSITION_COUNT = 1
EXPECTED_BLOCK_COUNT = 50
EXPECTED_VIDEO_NATIVE_SHAPE = (1, 24, 9, 8, 8)
EXPECTED_AUDIO_NATIVE_SHAPE = (2, 32, 50)
EXPECTED_TARGET_VIDEO_ROWS = 144
EXPECTED_TARGET_AUDIO_ROWS = 100
EXPECTED_TARGET_ROWS = 244
EXPECTED_VIDEO_FPS = 24
EXPECTED_AUDIO_SAMPLE_RATE = 32_000
EXPECTED_AUDIO_SAMPLES = 40_000
DEFAULT_TOLERANCE_BYTES = 1 * 1024 * 1024
FINGERPRINT_METHOD = "sha256-logical-shape-dtype-plus-canonical-float32-values-v1"
RNG_METHOD = "mlx.core.random.seed(0)+mlx.core.random.normal-float32-video-then-audio-v1"

SOURCE_INSPECTION_FILES = (
    "minimax_h3_mlx/text_encoder.py",
    "minimax_h3_mlx/packing.py",
    "minimax_h3_mlx/pipeline.py",
    "minimax_h3_mlx/dit.py",
    "minimax_h3_mlx/adaln.py",
    "minimax_h3_mlx/streamed_adaln.py",
    "minimax_h3_mlx/scheduler.py",
    "minimax_h3_mlx/geometry.py",
    "minimax_h3_mlx/video_vae.py",
    "minimax_h3_mlx/audio_vae.py",
    "minimax_h3_mlx/load.py",
    "scripts/probe_v04a_one_step.py",
    "scripts/probe_v04b_multistep.py",
    "scripts/probe_v04c_conditioned.py",
    "scripts/probe_v05a_decoders.py",
    "scripts/probe_v05b_geometry_bridge.py",
    "tests/test_packing_parity.py",
    "tests/test_pipeline_staged_loading.py",
    "tests/test_streamed_adaln.py",
)

SOURCE_LOCATIONS = {
    "prompt_tokenization": "minimax_h3_mlx/text_encoder.py:200-239,269-306",
    "conditioning_attention_mask": "minimax_h3_mlx/text_encoder.py:251-267",
    "text_row_packing": "minimax_h3_mlx/packing.py:278-370",
    "padless_attention_policy": "minimax_h3_mlx/packing.py:19-21",
    "video_audio_pack_unpack": "minimax_h3_mlx/packing.py:186-240",
    "initial_noise_order": "minimax_h3_mlx/pipeline.py:568-585",
    "production_forward": "minimax_h3_mlx/dit.py:361-446",
    "one_step_loop": "minimax_h3_mlx/denoise.py:177-315",
    "scheduler": "minimax_h3_mlx/scheduler.py:1-5,43-189,218-312",
    "geometry": "minimax_h3_mlx/geometry.py:40-126",
    "resident_loader": "minimax_h3_mlx/load.py:300-420",
    "derived_loader": "minimax_h3_mlx/load.py:104-221",
    "streamed_adaln_cache": "minimax_h3_mlx/streamed_adaln.py:371-610",
    "final_layer_adaln": "minimax_h3_mlx/dit.py:231-259",
    "video_decoder": "minimax_h3_mlx/pipeline.py:655-680,708-725",
    "audio_decoder": "minimax_h3_mlx/pipeline.py:682-704,727-735",
    "video_loader": "minimax_h3_mlx/load.py:423-499",
    "audio_loader": "minimax_h3_mlx/load.py:502-577",
    "v05b_geometry": "scripts/probe_v05b_geometry_bridge.py:118-246",
}

GENERATION_EXCLUSIONS = {
    "second_denoising_step": False,
    "quality_judgment": False,
    "mp4_muxing": False,
    "ffmpeg_invoked": False,
    "image_conditioning": False,
    "negative_prompt": False,
    "prompt_rewritten": False,
    "checkpoint_mutation": False,
}

REPORT_KEYS = frozenset(
    {
        "status",
        "schema_version",
        "probe_identity",
        "committed_baseline",
        "source_contracts",
        "canonical_host_command",
        "prompt",
        "tokenizer_receipts",
        "conditioning_receipts",
        "checkpoint_paths",
        "checkpoint_checksums",
        "native_geometry",
        "deterministic_inputs",
        "complete_packed_sequence_contract",
        "scheduler_receipts",
        "process_isolation",
        "resident_worker",
        "derived_worker",
        "streamed_adaln_lifecycle",
        "exact_parity_gates",
        "unpacked_native_latents",
        "conditioning_memory_release",
        "resident_memory_release",
        "derived_memory_release",
        "video_media",
        "audio_media",
        "video_memory",
        "audio_memory",
        "final_memory",
        "phase_order",
        "output_paths",
        "generation_exclusions",
        "failure",
    }
)

FAILURE_KEYS = frozenset(
    {
        "active_phase",
        "worker_identity",
        "completed_stages",
        "original_error",
        "cleanup_error",
        "subprocess_exit_status",
        "subprocess_log_path",
        "partial_conditioning_metadata",
        "partial_packed_metadata",
        "partial_parity_metadata",
        "sidecar_state",
        "residency_state",
        "partial_media_paths",
        "memory_receipts",
        "later_phase_suppression",
    }
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run-prompted-production-step",
        help="run the locked-prompt, one-transition, resident-versus-derived production proof",
    )
    run.add_argument("--checkpoint-root", required=True)
    run.add_argument("--resident-transformer", required=True)
    run.add_argument("--derived-transformer", required=True)
    run.add_argument("--output-root", required=True)
    run.add_argument("--prompt", required=True)
    run.add_argument("--seed", required=True, type=int)
    run.add_argument("--active-memory-tolerance-bytes", required=True, type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=run_command)
    return parser


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _shape(value: Any) -> list[int]:
    return [int(item) for item in value.shape]


def shape_dtype(value: Any, *, logical_dtype: str | None = None) -> dict[str, Any]:
    return {"shape": _shape(value), "dtype": logical_dtype or _dtype_name(value.dtype)}


def _as_float32_numpy(value: Any, mx: Any | None = None) -> np.ndarray:
    is_mlx = getattr(value, "__mlx_array__", False) or value.__class__.__module__.startswith("mlx.")
    if mx is not None and is_mlx:
        value = value.astype(mx.float32)
        mx.eval(value)
    return np.array(value, dtype=np.float32, copy=True)


def array_fingerprint(value: Any, *, logical_dtype: str | None = None, mx: Any | None = None) -> str:
    """Hash logical shape/dtype and evaluated canonical float32 values."""
    array = np.ascontiguousarray(_as_float32_numpy(value, mx), dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("fingerprints require finite values")
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


def validate_locked_prompt(prompt: str) -> None:
    if prompt != LOCKED_PROMPT:
        raise ValueError("prompt must equal the locked v0.5c UTF-8 prompt exactly")


def validate_seed(seed: int) -> None:
    if seed != CANONICAL_SEED:
        raise ValueError(f"canonical v0.5c proof accepts only seed {CANONICAL_SEED}, got {seed}")


def prompt_receipt(prompt: str, token_count: int | None = None, token_ids: Any | None = None) -> dict[str, Any]:
    validate_locked_prompt(prompt)
    result: dict[str, Any] = {
        "text": prompt,
        "utf8_byte_count": len(prompt.encode("utf-8")),
        "sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_is_literal": True,
        "negative_prompt": None,
        "image_conditioning": False,
    }
    if token_count is not None:
        result["token_count"] = int(token_count)
    if token_ids is not None:
        result["token_ids"] = np.asarray(token_ids, dtype=np.int32).tolist()
    return result


def deterministic_input_receipt(video: np.ndarray, audio: np.ndarray) -> dict[str, Any]:
    return {
        "rng_implementation": RNG_METHOD,
        "seed": CANONICAL_SEED,
        "fingerprint_method": FINGERPRINT_METHOD,
        "video": {
            "shape": list(video.shape),
            "dtype": "float32",
            "fingerprint": array_fingerprint(video, logical_dtype="float32"),
        },
        "audio": {
            "shape": list(audio.shape),
            "dtype": "float32",
            "fingerprint": array_fingerprint(audio, logical_dtype="float32"),
        },
    }


def canonical_geometry_contract() -> dict[str, Any]:
    """The immutable v0.5b geometry values, used for MLX-free contract checks."""
    return {
        "video_native_latent_shape": list(EXPECTED_VIDEO_NATIVE_SHAPE),
        "audio_native_latent_shape": list(EXPECTED_AUDIO_NATIVE_SHAPE),
        "video_output_shape": [1, 3, 30, 128, 128],
        "video_rgb_shape": [30, 128, 128, 3],
        "video_fps": EXPECTED_VIDEO_FPS,
        "video_duration_seconds": 1.25,
        "audio_raw_shape": [2, 1, EXPECTED_AUDIO_SAMPLES],
        "audio_waveform_shape": [2, EXPECTED_AUDIO_SAMPLES],
        "audio_sample_rate": EXPECTED_AUDIO_SAMPLE_RATE,
        "audio_duration_seconds": 1.25,
        "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
        "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
        "target_rows": EXPECTED_TARGET_ROWS,
        "audio_feature_width": 32,
        "video_feature_width": 96,
        "video_patch_size": [1, 2, 2],
    }


def derive_row_ranges(text_rows: int, audio_rows: int = EXPECTED_TARGET_AUDIO_ROWS, video_rows: int = EXPECTED_TARGET_VIDEO_ROWS) -> dict[str, Any]:
    if not isinstance(text_rows, int) or isinstance(text_rows, bool) or text_rows <= 0:
        raise ValueError("text row count must be a positive integer")
    if audio_rows != EXPECTED_TARGET_AUDIO_ROWS or video_rows != EXPECTED_TARGET_VIDEO_ROWS:
        raise ValueError("v0.5c target geometry must remain the established v0.5b geometry")
    audio_start = text_rows
    video_start = audio_start + audio_rows
    return {
        "text": [0, text_rows],
        "target_audio": [audio_start, video_start],
        "target_video": [video_start, video_start + video_rows],
        "text_rows": text_rows,
        "target_audio_rows": audio_rows,
        "target_video_rows": video_rows,
        "total_rows": text_rows + audio_rows + video_rows,
    }


def validate_packed_contract(contract: Mapping[str, Any]) -> None:
    ranges = contract.get("row_ranges")
    if not isinstance(ranges, Mapping):
        raise ValueError("packed contract has no row ranges")
    expected = derive_row_ranges(int(ranges["text"][1]))
    if dict(ranges) != expected:
        raise ValueError(f"packed row ranges changed: {ranges!r}")
    if contract.get("row_order") != "[text | target-audio | target-video]":
        raise ValueError("packed row order is not [text | target-audio | target-video]")
    if contract.get("attention_mask") is not None:
        raise ValueError("v0.5c production packing must not emit an attention mask")
    if contract.get("padding_rows") != 0:
        raise ValueError("v0.5c production packing must not emit padding rows")
    if contract.get("position_ids_shape") != [expected["total_rows"], 3]:
        raise ValueError("packed position-ID shape is inconsistent with complete sequence rows")
    if contract.get("token_tags_shape") != [expected["total_rows"]]:
        raise ValueError("packed token-tag shape is inconsistent with complete sequence rows")
    if contract.get("timestep_indices_shape") != [expected["total_rows"]]:
        raise ValueError("packed timestep-index shape is inconsistent with complete sequence rows")


def canonical_scheduler_receipt(video_scheduler: Any, audio_scheduler: Any, transition: Mapping[str, Any], timestep: Any, timestep_indices: Any) -> dict[str, Any]:
    def values(scheduler: Any) -> list[float]:
        return [float(value) for value in np.asarray(scheduler.sigmas.tolist(), dtype=np.float32)]

    return {
        "identity": "MiniMaxH3MultimodalScheduler",
        "video": {
            "identity": "MiniMaxH3Scheduler",
            "shift": float(video_scheduler.shift),
            "base_sigma_grid": [1.0, 0.0],
            "shifted_sigma_grid": values(video_scheduler),
            "starting_sigma": float(video_scheduler.sigmas[0].item()),
            "ending_sigma": float(video_scheduler.sigmas[1].item()),
            "delta_sigma": float(video_scheduler.sigmas[1].item() - video_scheduler.sigmas[0].item()),
        },
        "audio": {
            "identity": "MiniMaxH3Scheduler",
            "shift": float(audio_scheduler.shift),
            "base_sigma_grid": [1.0, 0.0],
            "shifted_sigma_grid": values(audio_scheduler),
            "starting_sigma": float(audio_scheduler.sigmas[0].item()),
            "ending_sigma": float(audio_scheduler.sigmas[1].item()),
            "delta_sigma": float(audio_scheduler.sigmas[1].item() - audio_scheduler.sigmas[0].item()),
        },
        "transition": dict(transition),
        "timestep_table": shape_dtype(timestep),
        "timestep_indices": shape_dtype(timestep_indices),
        "scheduler_implementation": "MiniMaxH3MultimodalScheduler.step",
        "transformer_forwards": 1,
        "scheduler_updates": 1,
        "transition_count": 1,
        "configurable_step_count_exposed": False,
    }


def exact_gate(left: Any, right: Any, *, left_dtype: str | None = None, right_dtype: str | None = None) -> dict[str, Any]:
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    shape_equal = tuple(left_array.shape) == tuple(right_array.shape)
    dtype_equal = (left_dtype or str(left_array.dtype)) == (right_dtype or str(right_array.dtype))
    value_equal = bool(shape_equal and np.array_equal(left_array, right_array))
    left_fp = array_fingerprint(left_array, logical_dtype=left_dtype or str(left_array.dtype))
    right_fp = array_fingerprint(right_array, logical_dtype=right_dtype or str(right_array.dtype))
    fingerprint_equal = left_fp == right_fp
    return {
        "shape_equal": shape_equal,
        "dtype_equal": dtype_equal,
        "value_equal": value_equal,
        "fingerprint_equal": fingerprint_equal,
        "left_fingerprint": left_fp,
        "right_fingerprint": right_fp,
        "exact_equality": bool(shape_equal and dtype_equal and value_equal and fingerprint_equal),
    }


def validate_one_step_counts(receipt: Mapping[str, Any], *, derived: bool = False) -> None:
    if receipt.get("transformer_calls") != 1 or receipt.get("scheduler_updates") != 1:
        raise ValueError("v0.5c worker must perform exactly one transformer forward and one scheduler update")
    expected = 1 if derived else 0
    if receipt.get("cache_acquisitions") != expected or receipt.get("cache_releases") != expected:
        raise ValueError("worker cache session counts are inconsistent with resident/derived mode")
    if receipt.get("completed_steps") != 1:
        raise ValueError("v0.5c worker completed more or fewer than one denoising step")


def source_contracts() -> dict[str, Any]:
    return {
        "inspected_files": list(SOURCE_INSPECTION_FILES),
        "source_locations": dict(SOURCE_LOCATIONS),
        "prompt_tokenization": {
            "call": "MiniMaxH3TextEncoder.tokenizer(prompt, add_special_tokens=False)",
            "special_tokens": False,
            "prompt_literal": True,
            "images": None,
            "chat_template": None,
        },
        "conditioning": {
            "shape": "(1, runtime_token_count, 5120)",
            "dtype": "bfloat16",
            "state": "unnormalized hidden state before final norm",
            "attention_mask_policy": "create_attention_mask(hidden_states, None)",
            "token_presence_mask": "metadata-only all-ones (1, token_count); not passed to the encoder",
        },
        "packing": {
            "row_order": "[text | target-audio | target-video]",
            "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
            "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
            "target_rows": EXPECTED_TARGET_ROWS,
            "padding_rows": 0,
            "attention_mask": None,
            "modality_tags": {"video": 0, "text": 1, "audio": 2, "padding": -1},
            "position_axes": "(t,h,w)",
        },
        "initial_noise": {
            "source": "minimax_h3_mlx/pipeline.py:568-585",
            "seed": 0,
            "draw_order": ["video_native", "audio_native"],
            "dtype": "float32",
        },
        "scheduler": {
            "video_shift": VIDEO_SHIFT,
            "audio_shift": AUDIO_SHIFT,
            "schedule_points": 2,
            "euler_transitions": 1,
            "implementation": "MiniMaxH3MultimodalScheduler.step",
        },
        "resident_transformer": {
            "loader": "minimax_h3_mlx.load.load_dit",
            "construction_mode": "resident",
        },
        "derived_transformer": {
            "loader": "minimax_h3_mlx.load.load_dit",
            "construction_mode": "cache_only",
            "dense_block_adaln": "absent",
            "final_layer_adaln": "resident",
        },
        "decoder_handoff": {
            "order": ["parity", "transformer-release", "video-load-decode-release", "audio-load-decode-release"],
            "video_scaling": "latents * latents_std + latents_mean; decode; ImageNet inverse; clip/uint8",
            "audio_scaling": "latents * latents_std + latents_mean; decode; raw[:,0,:]",
        },
    }


def _config_checksums(root: Path, resident: Path, derived: Path) -> dict[str, str]:
    paths = {
        "checkpoint_root/model_index.json": root / "model_index.json",
        "checkpoint_root/text_encoder/config.json": root / "text_encoder" / "config.json",
        "checkpoint_root/video_vae/config.json": root / "video_vae" / "config.json",
        "checkpoint_root/video_vae/source/config.json": root / "video_vae" / "source" / "config.json",
        "checkpoint_root/audio_vae/config.json": root / "audio_vae" / "config.json",
        "checkpoint_root/audio_vae/metadata.json": root / "audio_vae" / "metadata.json",
        "resident/config.json": resident / "config.json",
        "resident/model.safetensors.index.json": resident / "model.safetensors.index.json",
        "resident/quant_config.json": resident / "quant_config.json",
        "derived/config.json": derived / "config.json",
        "derived/base/model.safetensors.index.json": derived / "base" / "model.safetensors.index.json",
        "derived/quant_config.json": derived / "quant_config.json",
        "derived/conversion_manifest.json": derived / "conversion_manifest.json",
        "derived/adaln/manifest.json": derived / "adaln" / "manifest.json",
    }
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}


def ensure_output_namespace(root: Path, overwrite: bool) -> dict[str, str]:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite nonempty v0.5c output root: {root}; pass --overwrite")
    root.mkdir(parents=True, exist_ok=True)
    frames = root / "frames"
    if overwrite and frames.is_dir():
        for path in frames.glob("frame_*.png"):
            if path.is_file():
                path.unlink()
    for path in (root / "prompted-step-audio.wav", root / "prompted-step-report.json"):
        if overwrite and path.is_file():
            path.unlink()
    return {
        "root": str(root),
        "frames": str(frames),
        "audio_wav": str(root / "prompted-step-audio.wav"),
        "report": str(root / "prompted-step-report.json"),
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True, default=str) + "\n")


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{key: np.ascontiguousarray(np.asarray(value)) for key, value in arrays.items()})


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.array(loaded[key], copy=True) for key in loaded.files}


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, name in (("active", "get_active_memory"), ("allocator_cache", "get_cache_memory"), ("peak", "get_peak_memory")):
        getter = getattr(mx, name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except Exception:
            result[label] = None
    return result


def _release_runtime(mx: Any, references: dict[str, Any], baseline: Mapping[str, Any] | None, tolerance: int) -> dict[str, Any]:
    for key in list(references):
        references[key] = None
    gc.collect()
    before = _memory_snapshot(mx)
    clear_cache = getattr(mx, "clear_cache", None)
    purge_error: dict[str, str] | None = None
    if not callable(clear_cache):
        purge_error = {"type": "RuntimeError", "message": "MLX allocator cache purge is unavailable"}
    else:
        try:
            clear_cache()
        except BaseException as exc:
            purge_error = {"type": type(exc).__name__, "message": str(exc)}
    after = _memory_snapshot(mx)
    baseline_active = baseline.get("active") if isinstance(baseline, Mapping) else None
    after_active = after.get("active")
    active_available = baseline_active is not None and after_active is not None
    active_within = bool(active_available and after_active <= baseline_active + tolerance)
    cache_zero = after.get("allocator_cache") == 0
    passed = purge_error is None and active_within and cache_zero
    return {
        "status": "success" if passed else "failed",
        "passed": passed,
        "memory_before_allocator_purge": before,
        "memory_after_allocator_purge": after,
        "active_memory_baseline_bytes": baseline_active,
        "active_memory_tolerance_bytes": tolerance,
        "active_memory_gate_available": active_available,
        "active_memory_within_tolerance": active_within,
        "allocator_cache_after_bytes": after.get("allocator_cache"),
        "allocator_cache_zero": cache_zero,
        "allocator_purge_error": purge_error,
    }


def _error(error: BaseException | Mapping[str, Any] | None) -> dict[str, str] | None:
    if error is None:
        return None
    if isinstance(error, Mapping):
        return {"type": str(error.get("type", "RuntimeError")), "message": str(error.get("message", error))}
    return {"type": type(error).__name__, "message": str(error)}


class PhaseFailure(RuntimeError):
    def __init__(self, phase: str, original: BaseException | Mapping[str, Any], *, cleanup: BaseException | Mapping[str, Any] | None = None, details: Mapping[str, Any] | None = None):
        self.phase = phase
        self.original_error = _error(original)
        self.cleanup_error = _error(cleanup)
        self.details = dict(details or {})
        super().__init__(self.original_error["message"] if self.original_error else f"{phase} failed")


def _validate_checkpoint_paths(root: Path, resident: Path, derived: Path) -> None:
    required_dirs = {
        "checkpoint root": root,
        "resident transformer": resident,
        "derived transformer": derived,
        "text encoder": root / "text_encoder",
        "video VAE": root / "video_vae",
        "audio VAE": root / "audio_vae",
    }
    missing = [f"{name}: {path}" for name, path in required_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError("missing required checkpoint directories: " + "; ".join(missing))
    for name, path in (("resident config", resident / "config.json"), ("derived config", derived / "config.json")):
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    if resident.resolve() == derived.resolve():
        raise ValueError("resident and derived transformer paths must be distinct explicit checkpoints")


def _canonical_host_command(args: argparse.Namespace) -> str:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "run-prompted-production-step",
        "--checkpoint-root", str(Path(args.checkpoint_root).expanduser().resolve()),
        "--resident-transformer", str(Path(args.resident_transformer).expanduser().resolve()),
        "--derived-transformer", str(Path(args.derived_transformer).expanduser().resolve()),
        "--output-root", str(Path(args.output_root).expanduser().resolve()),
        "--prompt", LOCKED_PROMPT,
        "--seed", str(args.seed),
        "--active-memory-tolerance-bytes", str(args.active_memory_tolerance_bytes),
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.verbose:
        command.append("--verbose")
    return shlex.join(command)


def _conditioning_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--resident-transformer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _transformer_worker_parser(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--role", choices=("resident", "derived"), required=True)
    parser.add_argument("--transformer", required=True)
    parser.add_argument("--conditioning-artifact", required=True)
    parser.add_argument("--conditioning-receipt", required=True)
    parser.add_argument("--output-artifact", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--tolerance", required=True, type=int)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def _worker_base(role: str) -> dict[str, Any]:
    return {
        "status": "failed",
        "worker_identity": role,
        "pid": os.getpid(),
        "completed_stages": [],
        "transformer_calls": 0,
        "scheduler_updates": 0,
        "completed_steps": 0,
        "cache_acquisitions": 0,
        "cache_releases": 0,
        "error": None,
        "cleanup_error": None,
        "memory_receipts": {},
        "residency": {},
        "sidecar_state": {},
        "partial_artifact_paths": [],
    }


def _write_worker_failure(path: Path, receipt: dict[str, Any], original: BaseException | Mapping[str, Any], cleanup: BaseException | Mapping[str, Any] | None = None) -> None:
    receipt["status"] = "failed"
    receipt["error"] = _error(original)
    receipt["cleanup_error"] = _error(cleanup)
    receipt["partial_artifact_paths"] = [str(value) for value in receipt.get("partial_artifact_paths", [])]
    _write_json(path, receipt)


def _conditioning_worker_main(argv: Sequence[str]) -> int:
    args = _conditioning_worker_parser(argv)
    receipt = _worker_base("conditioning")
    artifact_path = Path(args.artifact).resolve()
    receipt["artifact_path"] = str(artifact_path)
    mx = None
    references: dict[str, Any] = {}
    baseline = None
    original: BaseException | None = None
    cleanup: BaseException | None = None
    encoder = input_ids = token_tags = vision_inputs = token_presence = None
    conditioning = encoded_tags = video_native = audio_native = None
    layout = timestep = timestep_indices = None
    video_scheduler = audio_scheduler = scheduler = None
    try:
        validate_locked_prompt(args.prompt)
        validate_seed(args.seed)
        import mlx.core as mx
        from minimax_h3_mlx.config import DiTConfig
        from minimax_h3_mlx.geometry import ProductionMultimodalGeometry
        from minimax_h3_mlx.load import load_audio_vae_config, load_video_vae_config
        from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps
        from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder
        from minimax_h3_mlx.video_decode_layout import resolve_video_decode_layout

        root = Path(args.checkpoint_root).resolve()
        resident = Path(args.resident_transformer).resolve()
        receipt["completed_stages"].append("conditioning-baseline")
        baseline = _memory_snapshot(mx)
        receipt["memory_receipts"]["before_text_encoder_load"] = baseline
        encoder = MiniMaxH3TextEncoder(root / "text_encoder", dtype=mx.bfloat16, load_vision=False, verbose=args.verbose)
        references["encoder"] = encoder
        receipt["completed_stages"].append("text-encoder-loaded")
        input_ids, token_tags, vision_inputs = encoder.build_request(args.prompt, None)
        if vision_inputs is not None:
            raise ValueError("locked v0.5c conditioning unexpectedly produced image inputs")
        token_presence = mx.ones(input_ids.shape, dtype=mx.int32)
        conditioning, encoded_tags = encoder.encode(args.prompt, None)
        if list(np.asarray(token_tags)) != list(np.asarray(encoded_tags)):
            raise ValueError("encoder tag output differs from build_request tags")
        if conditioning.ndim != 3 or tuple(conditioning.shape)[0] != 1 or tuple(conditioning.shape)[2] != 5120:
            raise ValueError(f"conditioning shape is not (1, token_count, 5120): {conditioning.shape}")
        if _dtype_name(conditioning.dtype) != "bfloat16":
            raise ValueError(f"conditioning dtype must be bfloat16, got {conditioning.dtype}")
        receipt["completed_stages"].append("prompt-tokenized-and-conditioned")

        # This is the production pipeline's no-image draw order, materialized once and then shared
        # by both isolated transformer workers.  The CPU artifact is the only input source thereafter.
        mx.random.seed(args.seed)
        video_native = mx.random.normal(EXPECTED_VIDEO_NATIVE_SHAPE).astype(mx.float32)
        audio_native = mx.random.normal(EXPECTED_AUDIO_NATIVE_SHAPE).astype(mx.float32)
        mx.eval(conditioning, input_ids, token_presence, video_native, audio_native)

        dit_config = DiTConfig.from_json(resident / "config.json")
        video_config = load_video_vae_config(root / "video_vae")
        audio_config = load_audio_vae_config(root / "audio_vae")
        layout_info = resolve_video_decode_layout(video_config)
        geometry = ProductionMultimodalGeometry.canonical(video_config, audio_config, dit_config, layout_info)
        if tuple(geometry.video_latent_shape) != EXPECTED_VIDEO_NATIVE_SHAPE or tuple(geometry.audio_latent_shape) != EXPECTED_AUDIO_NATIVE_SHAPE:
            raise ValueError("runtime geometry differs from locked v0.5b geometry")
        raw_tags = np.asarray(token_tags, dtype=np.int64)
        layout = build_packed_sequence(
            raw_tags,
            geometry.video_latent_shape[2], geometry.video_latent_shape[3], geometry.video_latent_shape[4],
            geometry.audio_latent_shape[2], tuple(dit_config.patch_size), keyframe_anchors=(),
        )
        video_scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        audio_scheduler = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        video_scheduler.set_timesteps(2)
        audio_scheduler.set_timesteps(2)
        scheduler = MiniMaxH3MultimodalScheduler(video_scheduler, audio_scheduler)
        transition = vars(scheduler.transition(0))
        timestep, timestep_indices = build_row_timesteps(
            layout,
            transition["video_current_timestep"], transition["audio_current_timestep"],
            0.999, 1.0,
        )
        for value in (layout.position_ids, layout.token_tags, layout.video_indices, layout.audio_indices, layout.text_indices, timestep, timestep_indices):
            mx.eval(value)

        token_count = int(input_ids.shape[1])
        ranges = derive_row_ranges(token_count)
        packed_contract = {
            "row_order": "[text | target-audio | target-video]",
            "row_ranges": ranges,
            "text_rows": token_count,
            "target_audio_rows": EXPECTED_TARGET_AUDIO_ROWS,
            "target_video_rows": EXPECTED_TARGET_VIDEO_ROWS,
            "target_rows": EXPECTED_TARGET_ROWS,
            "total_rows": int(layout.sequence_length),
            "feature_widths": {"text": 5120, "target_audio": 32, "target_video": 96},
            "modality_tags": {"video": 0, "text": 1, "audio": 2, "padding": -1},
            "row_tag_values": np.asarray(layout.token_tags, dtype=np.int32).tolist(),
            "position_ids_shape": _shape(layout.position_ids),
            "position_ids_dtype": _dtype_name(layout.position_ids.dtype),
            "token_tags_shape": _shape(layout.token_tags),
            "token_tags_dtype": _dtype_name(layout.token_tags.dtype),
            "video_indices_shape": _shape(layout.video_indices),
            "audio_indices_shape": _shape(layout.audio_indices),
            "text_indices_shape": _shape(layout.text_indices),
            "timestep_table_shape": _shape(timestep),
            "timestep_table_dtype": _dtype_name(timestep.dtype),
            "timestep_indices_shape": _shape(timestep_indices),
            "timestep_indices_dtype": _dtype_name(timestep_indices.dtype),
            "attention_mask": None,
            "padding_rows": 0,
            "text_encoder_attention_mask_policy": "create_attention_mask(hidden_states, None)",
        }
        validate_packed_contract(packed_contract)

        conditioning_np = _as_float32_numpy(conditioning, mx)
        token_ids_np = np.asarray(input_ids, dtype=np.int32)
        presence_np = np.asarray(token_presence, dtype=np.int32)
        token_tags_np = np.asarray(token_tags, dtype=np.int32)
        video_np = _as_float32_numpy(video_native, mx)
        audio_np = _as_float32_numpy(audio_native, mx)
        _write_npz(
            artifact_path,
            {
                "text_conditioning": conditioning_np,
                "token_ids": token_ids_np,
                "token_presence_mask": presence_np,
                "text_token_tags": token_tags_np,
                "initial_video_native": video_np,
                "initial_audio_native": audio_np,
                "packed_position_ids": np.asarray(layout.position_ids, dtype=np.float32),
                "packed_token_tags": np.asarray(layout.token_tags, dtype=np.int32),
                "packed_video_indices": np.asarray(layout.video_indices, dtype=np.int32),
                "packed_audio_indices": np.asarray(layout.audio_indices, dtype=np.int32),
                "packed_text_indices": np.asarray(layout.text_indices, dtype=np.int32),
                "timestep_table": np.asarray(timestep, dtype=np.float32),
                "timestep_indices": np.asarray(timestep_indices, dtype=np.int32),
            },
        )
        receipt["completed_stages"].append("conditioning-and-input-artifact-written")
        receipt["prompt"] = prompt_receipt(args.prompt, token_count, token_ids_np)
        receipt["tokenizer"] = {
            "entrypoint": "MiniMaxH3TextEncoder.tokenizer",
            "call": "tokenizer(prompt, add_special_tokens=False)",
            "add_special_tokens": False,
            "token_count": token_count,
            "token_ids": token_ids_np.tolist(),
            "token_presence_mask": presence_np.tolist(),
            "token_presence_mask_description": "metadata-only two-dimensional all-ones mask; not passed to the encoder",
        }
        receipt["conditioning"] = {
            "shape": list(conditioning_np.shape),
            "dtype": "bfloat16",
            "fingerprint": array_fingerprint(conditioning_np, logical_dtype="bfloat16"),
            "attention_mask_policy": "create_attention_mask(hidden_states, None)",
            "token_tags_shape": list(token_tags_np.shape),
        }
        receipt["deterministic_inputs"] = deterministic_input_receipt(video_np, audio_np)
        receipt["geometry"] = {
            **canonical_geometry_contract(),
            "alignment_evidence": list(geometry.alignment_evidence),
            "video_duration_fraction": str(geometry.video_duration),
            "audio_duration_fraction": str(geometry.audio_duration),
        }
        receipt["packed_contract"] = packed_contract
        receipt["packed_arrays"] = {
            "position_ids": array_fingerprint(np.asarray(layout.position_ids, dtype=np.float32), logical_dtype="float32"),
            "token_tags": array_fingerprint(np.asarray(layout.token_tags, dtype=np.int32), logical_dtype="int32"),
            "video_indices": array_fingerprint(np.asarray(layout.video_indices, dtype=np.int32), logical_dtype="int32"),
            "audio_indices": array_fingerprint(np.asarray(layout.audio_indices, dtype=np.int32), logical_dtype="int32"),
            "text_indices": array_fingerprint(np.asarray(layout.text_indices, dtype=np.int32), logical_dtype="int32"),
            "timestep_table": array_fingerprint(np.asarray(timestep, dtype=np.float32), logical_dtype="float32"),
            "timestep_indices": array_fingerprint(np.asarray(timestep_indices, dtype=np.int32), logical_dtype="int32"),
        }
        receipt["scheduler"] = canonical_scheduler_receipt(video_scheduler, audio_scheduler, transition, timestep, timestep_indices)
        receipt["memory_receipts"]["after_conditioning_materialization"] = _memory_snapshot(mx)
        references.update({"encoder": None, "conditioning": conditioning, "input_ids": input_ids, "token_presence": token_presence, "token_tags": token_tags, "encoded_tags": encoded_tags, "video_native": video_native, "audio_native": audio_native, "layout": layout, "timestep": timestep, "timestep_indices": timestep_indices, "scheduler": scheduler})
        receipt["memory_receipts"]["before_conditioning_reference_clear"] = _memory_snapshot(mx)
    except BaseException as exc:
        original = exc
    finally:
        if mx is not None:
            encoder = input_ids = token_tags = vision_inputs = token_presence = None
            conditioning = encoded_tags = video_native = audio_native = None
            layout = timestep = timestep_indices = None
            video_scheduler = audio_scheduler = scheduler = None
            release = _release_runtime(mx, references, baseline, args.tolerance)
            receipt["conditioning_release"] = release
            receipt["memory_receipts"]["after_encoder_release_and_allocator_purge"] = release.get("memory_after_allocator_purge")
            if original is None and not release.get("passed"):
                original = RuntimeError("conditioning release gate failed")
            elif original is not None and not release.get("passed"):
                cleanup = RuntimeError("conditioning cleanup/release gate failed")
    if original is not None:
        _write_worker_failure(Path(args.receipt), receipt, original, cleanup)
        return 1
    receipt["status"] = "success"
    receipt["completed_stages"].append("conditioning-released-before-transformer")
    receipt["partial_artifact_paths"] = [str(artifact_path), str(Path(args.receipt).resolve())]
    _write_json(Path(args.receipt).resolve(), receipt)
    return 0


def _asdict(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {key: _asdict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_asdict(item) for item in value]
    return value


@contextmanager
def _observe_blocks(dit: Any):
    original_blocks = list(dit.blocks)
    observations: list[list[int]] = []
    active_call: int | None = None

    class BlockProxy:
        def __init__(self, index: int, block: Any):
            self.index = index
            self.block = block

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if active_call is None:
                raise RuntimeError("transformer block executed outside an active forward")
            observations[active_call].append(self.index)
            return self.block(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.block, name)

    dit.blocks[:] = [BlockProxy(index, block) for index, block in enumerate(original_blocks)]

    class TransformerProxy:
        def __init__(self, transformer: Any):
            self.transformer = transformer
            self.forward_count = 0

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal active_call
            call_index = len(observations)
            observations.append([])
            active_call = call_index
            self.forward_count += 1
            try:
                return self.transformer(*args, **kwargs)
            finally:
                active_call = None

        def __getattr__(self, name: str) -> Any:
            return getattr(self.transformer, name)

    proxy = TransformerProxy(dit)
    try:
        yield proxy, observations
    finally:
        dit.blocks[:] = original_blocks


class _StreamedCacheSessionProvider:
    def __init__(self, dit: Any, mx: Any):
        self.dit = dit
        self.mx = mx
        self.records: list[dict[str, Any]] = []
        self.active = False
        self.next_session = 0
        self.last_session_token: int | None = None

    def cache_for_step(self, step_index: int, timestep: Any):
        if self.active:
            raise RuntimeError("streamed AdaLN cache sessions overlapped")
        from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache

        self.next_session += 1
        record: dict[str, Any] = {
            "step_index": step_index,
            "cache_session_token": self.next_session,
            "events": [{"event": "session-acquire-start", "step_index": step_index, "cache_session_token": self.next_session}],
            "sidecar_open_events": [],
            "sidecar_release_events": [],
        }
        self.records.append(record)

        def telemetry(event: str, details: Mapping[str, Any]) -> None:
            if event == "sidecar_opening":
                record["sidecar_open_events"].append({"block_index": details.get("block_index"), "path": str(details.get("path"))})
            elif event == "sidecar_released":
                record["sidecar_release_events"].append({"block_index": details.get("block_index"), "path": str(details.get("path"))})

        cache, stats = build_streamed_modulation_cache(
            self.dit,
            timestep,
            dtype=self.mx.bfloat16,
            telemetry=telemetry,
        )
        record["stats"] = _asdict(stats)
        record["events"].append({"event": "session-acquire-complete", "step_index": step_index, "cache_session_token": self.next_session})
        self.active = True
        self.last_session_token = self.next_session
        return cache

    def release_step(self, step_index: int, cache: Any) -> None:
        record = self.records[-1]
        record["events"].append({"event": "session-release-start", "step_index": step_index, "cache_session_token": record["cache_session_token"]})
        cache = None
        self.active = False
        record["events"].append({"event": "session-release-complete", "step_index": step_index, "cache_session_token": record["cache_session_token"]})


def _transformer_worker_main(argv: Sequence[str]) -> int:
    args = _transformer_worker_parser(argv)
    receipt = _worker_base(args.role)
    receipt["transformer_path"] = str(Path(args.transformer).resolve())
    output_path = Path(args.output_artifact).resolve()
    receipt_path = Path(args.receipt).resolve()
    mx = None
    references: dict[str, Any] = {}
    baseline = None
    original: BaseException | None = None
    cleanup: BaseException | None = None
    provider: _StreamedCacheSessionProvider | None = None
    observations: list[list[int]] = []
    dit = text = initial_video_native = initial_audio_native = None
    video_rows = audio_rows = position_ids = token_tags = None
    video_indices = audio_indices = text_indices = timestep = timestep_indices = None
    scheduler_video = scheduler_audio = scheduler = result = step = None
    observed_transformer = refs_for_observation = None
    pred_video = pred_audio = updated_video_packed = updated_audio_packed = None
    unpacked_video = unpacked_audio = None
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
        from minimax_h3_mlx.config import DiTConfig
        from minimax_h3_mlx.denoise import denoise_loop
        from minimax_h3_mlx.load import load_dit
        from minimax_h3_mlx.packing import pack_audio_latents, patchify_video_latents, unpatchify_video_tokens, unpack_audio_tokens
        from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler

        conditioning_path = Path(args.conditioning_artifact).resolve()
        conditioning_receipt = json.loads(Path(args.conditioning_receipt).read_text())
        arrays = _load_npz(conditioning_path)
        required = {
            "text_conditioning", "token_ids", "token_presence_mask", "text_token_tags", "initial_video_native", "initial_audio_native",
            "packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices",
        }
        missing = sorted(required - set(arrays))
        if missing:
            raise ValueError(f"conditioning artifact is missing arrays: {missing}")
        config = DiTConfig.from_json(Path(args.transformer).resolve() / "config.json")
        text = mx.array(arrays["text_conditioning"], dtype=mx.float32).astype(mx.bfloat16)
        initial_video_native = mx.array(arrays["initial_video_native"], dtype=mx.float32)
        initial_audio_native = mx.array(arrays["initial_audio_native"], dtype=mx.float32)
        video_rows = patchify_video_latents(initial_video_native, tuple(config.patch_size)).astype(mx.bfloat16)[None]
        audio_rows = pack_audio_latents(initial_audio_native).astype(mx.bfloat16)[None]
        position_ids = mx.array(arrays["packed_position_ids"], dtype=mx.float32)
        token_tags = mx.array(arrays["packed_token_tags"], dtype=mx.int32)
        video_indices = mx.array(arrays["packed_video_indices"], dtype=mx.int32)
        audio_indices = mx.array(arrays["packed_audio_indices"], dtype=mx.int32)
        text_indices = mx.array(arrays["packed_text_indices"], dtype=mx.int32)
        timestep = mx.array(arrays["timestep_table"], dtype=mx.float32)
        timestep_indices = mx.array(arrays["timestep_indices"], dtype=mx.int32)
        mx.eval(text, initial_video_native, initial_audio_native, video_rows, audio_rows, position_ids, token_tags, video_indices, audio_indices, text_indices, timestep, timestep_indices)

        scheduler_video = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
        scheduler_audio = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
        scheduler_video.set_timesteps(2)
        scheduler_audio.set_timesteps(2)
        scheduler = MiniMaxH3MultimodalScheduler(scheduler_video, scheduler_audio)
        transition = vars(scheduler.transition(0))
        baseline = _memory_snapshot(mx)
        receipt["memory_receipts"]["before_transformer_load"] = baseline
        receipt["completed_stages"].append("worker-baseline-and-inputs-ready")
        dit = load_dit(Path(args.transformer).resolve(), verbose=args.verbose)
        references["dit"] = dit
        receipt["completed_stages"].append("transformer-loaded")
        if args.role == "resident" and getattr(dit, "construction_mode", None) != "resident":
            raise ValueError("resident worker did not load a resident transformer")
        if args.role == "derived" and getattr(dit, "construction_mode", None) != "cache_only":
            raise ValueError("derived worker did not load a cache-only transformer")
        parameter_keys = [key for key, _ in tree_flatten(dit.parameters())]
        dense_block_keys = sorted(key for key in parameter_keys if key.startswith("blocks.") and ".adaln_proj." in key)
        final_layer_keys = sorted(key for key in parameter_keys if key.startswith("final_layer.adaln_proj.linear."))
        receipt["residency"] = {
            "construction_mode": getattr(dit, "construction_mode", None),
            "configured_block_count": len(dit.blocks),
            "dense_block_adaln_parameter_keys": dense_block_keys,
            "dense_block_adaln_projection_reconstructed": False,
            "final_layer_adaln_parameter_keys": final_layer_keys,
            "final_layer_adaln_resident": all(key in final_layer_keys for key in ("final_layer.adaln_proj.linear.weight", "final_layer.adaln_proj.linear.bias")),
        }
        if args.role == "derived":
            if dense_block_keys:
                raise ValueError("derived cache-only transformer unexpectedly contains dense block AdaLN parameters")
            info = getattr(dit, "checkpoint_format_info", None)
            if getattr(info, "checkpoint_format", None) != "derived" or getattr(info, "adaln_manifest_path", None) is None:
                raise ValueError("derived worker lacks validated sidecar manifest metadata")
            receipt["residency"]["sidecar_manifest_present"] = Path(info.adaln_manifest_path).is_file()
            if not receipt["residency"]["sidecar_manifest_present"]:
                raise ValueError("derived sidecar manifest is not present")

        refs_for_observation = None
        if args.role == "derived":
            provider = _StreamedCacheSessionProvider(dit, mx)
            refs_for_observation = provider
        else:
            refs_for_observation = None
        with _observe_blocks(dit) as (observed_transformer, observations):
            result = denoise_loop(
                observed_transformer,
                scheduler,
                initial_video_latent=video_rows,
                initial_audio_latent=audio_rows,
                text_embedding=text,
                timestep_provider=lambda _step, _transition: (timestep, timestep_indices),
                token_tags=token_tags,
                position_ids=position_ids,
                video_indices=video_indices,
                audio_indices=audio_indices,
                text_indices=text_indices,
                modulation_cache_provider=refs_for_observation,
                step_indices=(0,),
                transition_count=1,
                expected_video_shape=(1, EXPECTED_TARGET_VIDEO_ROWS, 96),
                expected_audio_shape=(1, EXPECTED_TARGET_AUDIO_ROWS, 32),
                expected_text_shape=(1, int(text.shape[1]), 5120),
            )
        receipt["completed_stages"].append("one-production-forward-and-euler-update")
        validate_one_step_counts(
            {
                "transformer_calls": result.transformer_calls,
                "scheduler_updates": result.scheduler_updates,
                "cache_acquisitions": result.cache_acquisitions,
                "cache_releases": result.cache_releases,
                "completed_steps": result.completed_steps,
            },
            derived=args.role == "derived",
        )
        receipt.update({
            "transformer_calls": result.transformer_calls,
            "scheduler_updates": result.scheduler_updates,
            "completed_steps": result.completed_steps,
            "cache_acquisitions": result.cache_acquisitions,
            "cache_releases": result.cache_releases,
        })
        if len(observations) != 1 or observations[0] != list(range(EXPECTED_BLOCK_COUNT)):
            raise ValueError(f"transformer block execution did not observe exactly 0..49 once: {observations!r}")
        receipt["forward_observation"] = {"forward_count": len(observations), "block_indices": [list(item) for item in observations]}
        receipt["scheduler"] = canonical_scheduler_receipt(scheduler_video, scheduler_audio, transition, timestep, timestep_indices)
        receipt["conditioning"] = {
            "prompt": conditioning_receipt.get("prompt"),
            "token_ids": arrays["token_ids"].tolist(),
            "token_presence_mask": arrays["token_presence_mask"].tolist(),
            "shape": list(text.shape),
            "dtype": "bfloat16",
            "fingerprint": array_fingerprint(text, logical_dtype="bfloat16", mx=mx),
        }
        receipt["initial_inputs"] = deterministic_input_receipt(
            arrays["initial_video_native"].astype(np.float32), arrays["initial_audio_native"].astype(np.float32)
        )
        receipt["packed_contract"] = conditioning_receipt["packed_contract"]
        receipt["packed_arrays"] = conditioning_receipt["packed_arrays"]
        step = result.step_receipts[0]
        pred_video = _as_float32_numpy(step.video_prediction, mx)
        pred_audio = _as_float32_numpy(step.audio_prediction, mx)
        updated_video_packed = _as_float32_numpy(step.updated_video_latent, mx)
        updated_audio_packed = _as_float32_numpy(step.updated_audio_latent, mx)
        unpacked_video = _as_float32_numpy(unpatchify_video_tokens(step.updated_video_latent[0], EXPECTED_VIDEO_NATIVE_SHAPE[2], EXPECTED_VIDEO_NATIVE_SHAPE[3], EXPECTED_VIDEO_NATIVE_SHAPE[4], EXPECTED_VIDEO_NATIVE_SHAPE[1], tuple(config.patch_size)), mx)
        unpacked_audio = _as_float32_numpy(unpack_audio_tokens(step.updated_audio_latent[0], EXPECTED_AUDIO_NATIVE_SHAPE[2]), mx)
        _write_npz(
            output_path,
            {
                "text_conditioning": arrays["text_conditioning"],
                "token_ids": arrays["token_ids"],
                "token_presence_mask": arrays["token_presence_mask"],
                "initial_video_native": arrays["initial_video_native"],
                "initial_audio_native": arrays["initial_audio_native"],
                "packed_video_prediction": pred_video,
                "packed_audio_prediction": pred_audio,
                "updated_packed_video": updated_video_packed,
                "updated_packed_audio": updated_audio_packed,
                "updated_native_video": unpacked_video,
                "updated_native_audio": unpacked_audio,
                "packed_position_ids": arrays["packed_position_ids"],
                "packed_token_tags": arrays["packed_token_tags"],
                "packed_video_indices": arrays["packed_video_indices"],
                "packed_audio_indices": arrays["packed_audio_indices"],
                "packed_text_indices": arrays["packed_text_indices"],
                "timestep_table": arrays["timestep_table"],
                "timestep_indices": arrays["timestep_indices"],
            },
        )
        receipt["predictions"] = {
            "packed_video": shape_dtype(pred_video, logical_dtype="float32") | {"fingerprint": array_fingerprint(pred_video, logical_dtype="float32")},
            "packed_audio": shape_dtype(pred_audio, logical_dtype="float32") | {"fingerprint": array_fingerprint(pred_audio, logical_dtype="float32")},
        }
        receipt["updated_latents"] = {
            "packed_video": shape_dtype(updated_video_packed, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(updated_video_packed, logical_dtype="bfloat16")},
            "packed_audio": shape_dtype(updated_audio_packed, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(updated_audio_packed, logical_dtype="bfloat16")},
            "native_video": shape_dtype(unpacked_video, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(unpacked_video, logical_dtype="bfloat16")},
            "native_audio": shape_dtype(unpacked_audio, logical_dtype="bfloat16") | {"fingerprint": array_fingerprint(unpacked_audio, logical_dtype="bfloat16")},
        }
        if provider is not None:
            receipt["sidecar_state"] = {
                "cache_sessions_created": len(provider.records),
                "cache_sessions_released": sum(1 for record in provider.records if any(event["event"] == "session-release-complete" for event in record["events"])),
                "sessions": provider.records,
                "sidecar_open_count": sum(len(record["sidecar_open_events"]) for record in provider.records),
                "sidecar_release_count": sum(len(record["sidecar_release_events"]) for record in provider.records),
                "maximum_simultaneous_sidecars": 1,
                "no_sidecar_open_after_forward": not provider.active,
                "dense_temporary_projection_created": False,
            }
        else:
            receipt["sidecar_state"] = {
                "cache_sessions_created": 0,
                "cache_sessions_released": 0,
                "sidecar_open_count": 0,
                "sidecar_release_count": 0,
                "maximum_simultaneous_sidecars": 0,
                "no_sidecar_open_after_forward": True,
                "dense_temporary_projection_created": False,
            }
        receipt["completed_stages"].append("cpu-output-artifact-written")
        references.update({
            "text": text, "initial_video_native": initial_video_native, "initial_audio_native": initial_audio_native,
            "video_rows": video_rows, "audio_rows": audio_rows, "position_ids": position_ids, "token_tags": token_tags,
            "video_indices": video_indices, "audio_indices": audio_indices, "text_indices": text_indices,
            "timestep": timestep, "timestep_indices": timestep_indices, "scheduler": scheduler, "result": result,
            "provider": provider, "pred_video": pred_video, "pred_audio": pred_audio, "updated_video_packed": updated_video_packed,
            "updated_audio_packed": updated_audio_packed, "unpacked_video": unpacked_video, "unpacked_audio": unpacked_audio,
        })
        receipt["memory_receipts"]["before_transformer_reference_clear"] = _memory_snapshot(mx)
    except BaseException as exc:
        original = exc
        receipt["forward_observation"] = {"forward_count": len(observations), "block_indices": [list(item) for item in observations]}
        if provider is not None:
            receipt["sidecar_state"] = {
                "cache_sessions_created": len(provider.records),
                "cache_sessions_released": sum(1 for record in provider.records if any(event["event"] == "session-release-complete" for event in record["events"])),
                "sessions": provider.records,
                "sidecar_open_count": sum(len(record["sidecar_open_events"]) for record in provider.records),
                "sidecar_release_count": sum(len(record["sidecar_release_events"]) for record in provider.records),
                "maximum_simultaneous_sidecars": 1,
                "no_sidecar_open_after_forward": not provider.active,
                "dense_temporary_projection_created": False,
            }
    finally:
        if mx is not None:
            if provider is not None and provider.active:
                provider.active = False
            if provider is not None:
                provider.dit = None
            references["dit"] = None
            dit = text = initial_video_native = initial_audio_native = None
            video_rows = audio_rows = position_ids = token_tags = None
            video_indices = audio_indices = text_indices = timestep = timestep_indices = None
            scheduler_video = scheduler_audio = scheduler = result = step = None
            observed_transformer = refs_for_observation = None
            pred_video = pred_audio = updated_video_packed = updated_audio_packed = None
            unpacked_video = unpacked_audio = None
            release = _release_runtime(mx, references, baseline, args.tolerance)
            receipt["transformer_release"] = release
            receipt["memory_receipts"]["after_transformer_release_and_allocator_purge"] = release.get("memory_after_allocator_purge")
            if original is None and not release.get("passed"):
                original = RuntimeError(f"{args.role} transformer release gate failed")
            elif original is not None and not release.get("passed"):
                cleanup = RuntimeError(f"{args.role} transformer cleanup/release gate failed")
    if original is not None:
        receipt["partial_artifact_paths"] = [str(output_path), str(receipt_path)] if output_path.exists() else [str(receipt_path)]
        _write_worker_failure(receipt_path, receipt, original, cleanup)
        return 1
    receipt["status"] = "success"
    receipt["completed_stages"].append("transformer-released-before-decoder")
    receipt["partial_artifact_paths"] = [str(output_path), str(receipt_path)]
    _write_json(receipt_path, receipt)
    return 0


def _child_command(worker: str, args: Sequence[str]) -> list[str]:
    return [sys.executable, str(Path(__file__).resolve()), worker, *args]


def _run_child(worker: str, args: Sequence[str], log_path: Path, receipt_path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        _child_command(worker, args),
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    log_path.write_text(completed.stdout + ("\n--- stderr ---\n" + completed.stderr if completed.stderr else ""))
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            receipt = _worker_base(worker.removeprefix("__").removesuffix("-worker"))
            receipt["error"] = _error(exc)
    else:
        receipt = _worker_base(worker.removeprefix("__").removesuffix("-worker"))
        receipt["error"] = {"type": "ChildProcessError", "message": f"worker exited {completed.returncode} without a receipt"}
    receipt["subprocess"] = {
        "command": shlex.join(_child_command(worker, args)),
        "exit_status": completed.returncode,
        "log_path": str(log_path),
    }
    if completed.returncode != 0 and receipt.get("status") == "success":
        receipt["status"] = "failed"
        receipt["error"] = {"type": "ChildProcessError", "message": f"worker exited {completed.returncode}"}
    return receipt


def _worker_success(receipt: Mapping[str, Any], *, derived: bool) -> None:
    if receipt.get("status") != "success":
        raise ValueError(f"{receipt.get('worker_identity', 'worker')} did not report success")
    validate_one_step_counts(receipt, derived=derived)
    release = receipt.get("transformer_release") if receipt.get("worker_identity") in {"resident", "derived"} else receipt.get("conditioning_release")
    if not isinstance(release, Mapping) or not release.get("passed"):
        raise ValueError(f"{receipt.get('worker_identity', 'worker')} release gate did not pass")


def compare_worker_artifacts(conditioning: Mapping[str, Any], resident: Mapping[str, Any], derived: Mapping[str, Any], resident_arrays: Mapping[str, np.ndarray], derived_arrays: Mapping[str, np.ndarray], resident_packed: Mapping[str, np.ndarray], derived_packed: Mapping[str, np.ndarray]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    prompt_texts = [
        conditioning.get("prompt", {}).get("text"),
        resident.get("prompt", {}).get("text") or resident.get("conditioning", {}).get("prompt", {}).get("text"),
        derived.get("prompt", {}).get("text") or derived.get("conditioning", {}).get("prompt", {}).get("text"),
    ]
    comparisons["prompt_text"] = {"exact_equality": len(set(prompt_texts)) == 1 and prompt_texts[0] == LOCKED_PROMPT}
    comparisons["token_ids"] = exact_gate(resident_arrays["token_ids"], derived_arrays["token_ids"], left_dtype="int32", right_dtype="int32")
    comparisons["token_presence_metadata"] = exact_gate(resident_arrays["token_presence_mask"], derived_arrays["token_presence_mask"], left_dtype="int32", right_dtype="int32")
    comparisons["conditioning"] = {
        **exact_gate(resident_arrays["text_conditioning"], derived_arrays["text_conditioning"], left_dtype="bfloat16", right_dtype="bfloat16"),
        "shape_equal": resident.get("conditioning", {}).get("shape") == derived.get("conditioning", {}).get("shape"),
        "dtype_equal": resident.get("conditioning", {}).get("dtype") == derived.get("conditioning", {}).get("dtype"),
        "fingerprint_equal": resident.get("conditioning", {}).get("fingerprint") == derived.get("conditioning", {}).get("fingerprint"),
        "value_equal": bool(np.array_equal(resident_arrays["text_conditioning"], derived_arrays["text_conditioning"])),
    }
    comparisons["conditioning"]["exact_equality"] = all(comparisons["conditioning"].get(key) for key in ("shape_equal", "dtype_equal", "value_equal", "fingerprint_equal"))
    for name, key, logical_dtype in (
        ("initial_video", "initial_video_native", "float32"),
        ("initial_audio", "initial_audio_native", "float32"),
        ("packed_metadata_position_ids", "packed_position_ids", "float32"),
        ("packed_metadata_token_tags", "packed_token_tags", "int32"),
        ("packed_metadata_video_indices", "packed_video_indices", "int32"),
        ("packed_metadata_audio_indices", "packed_audio_indices", "int32"),
        ("packed_metadata_text_indices", "packed_text_indices", "int32"),
        ("packed_metadata_timestep_table", "timestep_table", "float32"),
        ("packed_metadata_timestep_indices", "timestep_indices", "int32"),
    ):
        left = resident_arrays.get(key, resident_packed.get(key))
        right = derived_arrays.get(key, derived_packed.get(key))
        if left is None or right is None:
            raise ValueError(f"worker artifacts are missing exact comparison array {key}")
        comparisons[name] = exact_gate(left, right, left_dtype=logical_dtype, right_dtype=logical_dtype)
    for name, key, logical_dtype in (
        ("packed_audio_prediction", "packed_audio_prediction", "float32"),
        ("packed_video_prediction", "packed_video_prediction", "float32"),
        ("updated_packed_audio", "updated_packed_audio", "bfloat16"),
        ("updated_packed_video", "updated_packed_video", "bfloat16"),
        ("native_audio_latent", "updated_native_audio", "bfloat16"),
        ("native_video_latent", "updated_native_video", "bfloat16"),
    ):
        comparisons[name] = exact_gate(resident_arrays[key], derived_arrays[key], left_dtype=logical_dtype, right_dtype=logical_dtype)
    exact_flags = [bool(value.get("exact_equality")) for value in comparisons.values() if isinstance(value, Mapping) and "exact_equality" in value]
    comparisons["all_exact_gates"] = bool(comparisons["prompt_text"]["exact_equality"] and exact_flags and all(exact_flags))
    return comparisons


def _frame_file_metadata(frames_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(frames_dir.glob("frame_*.png"))
    expected = [frames_dir / f"frame_{index:05d}.png" for index in range(30)]
    if paths != expected:
        raise ValueError(f"video frame names are not exactly frame_00000.png..frame_00029.png: {[path.name for path in paths]}")
    from PIL import Image

    metadata = []
    for index, path in enumerate(paths):
        with Image.open(path) as image:
            if image.size != (128, 128) or image.mode != "RGB":
                raise ValueError(f"frame {index} is not 128x128 RGB")
        metadata.append({"index": index, "filename": path.name, "sha256": sha256_file(path)})
    return metadata


def _wav_metadata(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        metadata = {
            "channels": handle.getnchannels(),
            "sample_width_bytes": handle.getsampwidth(),
            "sample_rate": handle.getframerate(),
            "sample_count": handle.getnframes(),
            "duration_seconds": handle.getnframes() / handle.getframerate(),
        }
    metadata["sha256"] = sha256_file(path)
    return metadata


def _decode_media(args: argparse.Namespace, paths: Mapping[str, str], geometry: Mapping[str, Any], native_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_audio_vae, load_audio_vae_config, load_video_vae, load_video_vae_config
    from minimax_h3_mlx.media import save_frames, save_wav
    from minimax_h3_mlx.video_decode_layout import resolve_video_decode_layout

    root = Path(args.checkpoint_root).resolve()
    video_dir, audio_dir = root / "video_vae", root / "audio_vae"
    result: dict[str, Any] = {"video_media": {}, "audio_media": {}, "video_memory": {}, "audio_memory": {}, "final_memory": {}}
    video_refs: dict[str, Any] = {}
    audio_refs: dict[str, Any] = {}
    video_config = load_video_vae_config(video_dir)
    audio_config = load_audio_vae_config(audio_dir)
    layout = resolve_video_decode_layout(video_config)
    video_baseline = _memory_snapshot(mx)
    result["video_memory"]["memory_before_load"] = video_baseline
    video_original: BaseException | None = None
    video_cleanup: BaseException | None = None
    latent = mean = std = scaled = raw = None
    try:
        video_refs["decoder"] = load_video_vae(video_dir)
        result["video_memory"]["loaded"] = True
        latent = mx.array(native_arrays["updated_native_video"], dtype=mx.float32)
        mean = mx.array(np.asarray(video_config.latents_mean, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.asarray(video_config.latents_std, dtype=np.float32)).reshape(1, -1, 1, 1, 1)
        scaled = latent * std + mean
        raw = video_refs["decoder"].decode(scaled.astype(mx.float32))
        mx.eval(raw)
        raw_np = _as_float32_numpy(raw, mx)
        pixel_mean = np.asarray((0.485, 0.456, 0.406), dtype=np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.asarray((0.229, 0.224, 0.225), dtype=np.float32).reshape(1, 3, 1, 1, 1)
        frames = np.clip(raw_np * pixel_std + pixel_mean, 0.0, 1.0)[0].transpose(1, 2, 3, 0)
        frames = (frames * 255.0 + 0.5).astype(np.uint8)
        if frames.shape != (30, 128, 128, 3) or frames.dtype != np.uint8:
            raise ValueError(f"decoded video output is {frames.shape}/{frames.dtype}, expected (30,128,128,3)/uint8")
        save_frames(paths["frames"], frames)
        result["video_media"] = {
            "raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": "float32"},
            "final_shape_dtype": {"shape": list(frames.shape), "dtype": "uint8"},
            "frame_count": 30,
            "frame_checksums": _frame_file_metadata(Path(paths["frames"])),
            "diagnostic_only": True,
        }
    except BaseException as exc:
        video_original = exc
    finally:
        latent = mean = std = scaled = raw = None
        video_refs.update({key: None for key in video_refs})
        video_release = _release_runtime(mx, video_refs, video_baseline, args.active_memory_tolerance_bytes)
        result["video_memory"]["release_gate"] = video_release
        result["video_memory"]["memory_after_release"] = video_release.get("memory_after_allocator_purge")
        if video_original is None and not video_release.get("passed"):
            video_original = RuntimeError("video VAE release gate failed; audio loading suppressed")
        elif video_original is not None and not video_release.get("passed"):
            video_cleanup = RuntimeError("video VAE cleanup/release gate failed")
    if video_original is not None:
        raise PhaseFailure("video-decoder", video_original, cleanup=video_cleanup, details=result)

    audio_baseline = _memory_snapshot(mx)
    result["audio_memory"]["memory_before_load"] = audio_baseline
    audio_original: BaseException | None = None
    audio_cleanup: BaseException | None = None
    latent = mean = std = scaled = raw = None
    try:
        audio_refs["decoder"] = load_audio_vae(audio_dir)
        result["audio_memory"]["loaded"] = True
        latent = mx.array(native_arrays["updated_native_audio"], dtype=mx.float32)
        mean = mx.array(np.asarray(audio_config.latents_mean, dtype=np.float32)).reshape(1, -1, 1)
        std = mx.array(np.asarray(audio_config.latents_std, dtype=np.float32)).reshape(1, -1, 1)
        scaled = latent * std + mean
        raw = audio_refs["decoder"].decode(scaled.astype(mx.float32))
        mx.eval(raw)
        raw_np = _as_float32_numpy(raw, mx)
        waveform = raw_np[:, 0, :].astype(np.float32, copy=True)
        if raw_np.shape != (2, 1, EXPECTED_AUDIO_SAMPLES) or waveform.shape != (2, EXPECTED_AUDIO_SAMPLES):
            raise ValueError("decoded audio output does not match (2,1,40000)/(2,40000)")
        save_wav(paths["audio_wav"], waveform, EXPECTED_AUDIO_SAMPLE_RATE)
        wav = _wav_metadata(Path(paths["audio_wav"]))
        if wav["channels"] != 2 or wav["sample_rate"] != EXPECTED_AUDIO_SAMPLE_RATE or wav["sample_count"] != EXPECTED_AUDIO_SAMPLES or wav["sample_width_bytes"] != 2 or wav["duration_seconds"] != 1.25:
            raise ValueError(f"WAV metadata is not the locked stereo 32kHz/40000/16-bit contract: {wav}")
        result["audio_media"] = {
            "raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": "float32"},
            "final_shape_dtype": {"shape": list(waveform.shape), "dtype": "float32"},
            "sample_rate": EXPECTED_AUDIO_SAMPLE_RATE,
            "sample_count": EXPECTED_AUDIO_SAMPLES,
            "duration_seconds": 1.25,
            "wav": wav,
            "diagnostic_only": True,
        }
    except BaseException as exc:
        audio_original = exc
    finally:
        latent = mean = std = scaled = raw = None
        audio_refs.update({key: None for key in audio_refs})
        audio_release = _release_runtime(mx, audio_refs, audio_baseline, args.active_memory_tolerance_bytes)
        result["audio_memory"]["release_gate"] = audio_release
        result["audio_memory"]["memory_after_release"] = audio_release.get("memory_after_allocator_purge")
        if audio_original is None and not audio_release.get("passed"):
            audio_original = RuntimeError("audio VAE release gate failed")
        elif audio_original is not None and not audio_release.get("passed"):
            audio_cleanup = RuntimeError("audio VAE cleanup/release gate failed")
    result["final_memory"] = _memory_snapshot(mx)
    if audio_original is not None:
        raise PhaseFailure("audio-decoder", audio_original, cleanup=audio_cleanup, details=result)
    return result


def _base_report(args: argparse.Namespace, paths: Mapping[str, str], root: Path, resident: Path, derived: Path) -> dict[str, Any]:
    try:
        initial_prompt = prompt_receipt(args.prompt)
    except BaseException as exc:
        initial_prompt = {
            "text": getattr(args, "prompt", ""),
            "utf8_byte_count": len(str(getattr(args, "prompt", "")).encode("utf-8")),
            "sha256": hashlib.sha256(str(getattr(args, "prompt", "")).encode("utf-8")).hexdigest(),
            "prompt_is_literal": False,
            "negative_prompt": None,
            "image_conditioning": False,
            "validation_error": _error(exc),
        }
    return {
        "status": "failed",
        "schema_version": SCHEMA_VERSION,
        "probe_identity": PROBE_FORMAT,
        "committed_baseline": COMMITTED_BASELINE,
        "source_contracts": source_contracts(),
        "canonical_host_command": _canonical_host_command(args),
        "prompt": initial_prompt,
        "tokenizer_receipts": {},
        "conditioning_receipts": {},
        "checkpoint_paths": {
            "checkpoint_root": str(root),
            "resident_transformer": str(resident),
            "derived_transformer": str(derived),
            "video_vae": str(root / "video_vae"),
            "audio_vae": str(root / "audio_vae"),
            "text_encoder": str(root / "text_encoder"),
        },
        "checkpoint_checksums": _config_checksums(root, resident, derived),
        "native_geometry": canonical_geometry_contract(),
        "deterministic_inputs": {},
        "complete_packed_sequence_contract": {},
        "scheduler_receipts": {},
        "process_isolation": {
            "conditioning_worker_separate_process": True,
            "resident_worker_separate_process": True,
            "derived_worker_separate_process": True,
            "resident_and_derived_transformers_coexisted": False,
            "max_simultaneous_transformer_workers": 1,
        },
        "resident_worker": {},
        "derived_worker": {},
        "streamed_adaln_lifecycle": {},
        "exact_parity_gates": {"all_exact_gates": False},
        "unpacked_native_latents": {},
        "conditioning_memory_release": {},
        "resident_memory_release": {},
        "derived_memory_release": {},
        "video_media": {},
        "audio_media": {},
        "video_memory": {},
        "audio_memory": {},
        "final_memory": {},
        "phase_order": [],
        "output_paths": dict(paths),
        "generation_exclusions": dict(GENERATION_EXCLUSIONS),
        "failure": None,
    }


def validate_report(report: Mapping[str, Any]) -> None:
    actual = set(report)
    if actual != REPORT_KEYS:
        raise ValueError(f"strict v0.5c report schema mismatch: missing={sorted(REPORT_KEYS - actual)}, unexpected={sorted(actual - REPORT_KEYS)}")
    if report.get("schema_version") != SCHEMA_VERSION or report.get("probe_identity") != PROBE_FORMAT:
        raise ValueError("v0.5c report identity mismatch")
    if report.get("status") == "success":
        if report.get("failure") is not None:
            raise ValueError("successful v0.5c report must have failure=null")
        if not report.get("exact_parity_gates", {}).get("all_exact_gates"):
            raise ValueError("successful v0.5c report must prove all exact parity gates")
        if report.get("resident_memory_release", {}).get("passed") is not True or report.get("derived_memory_release", {}).get("passed") is not True:
            raise ValueError("successful v0.5c report must prove both transformer release gates")
        if report.get("conditioning_memory_release", {}).get("passed") is not True:
            raise ValueError("successful v0.5c report must prove conditioning release")
        if report.get("video_media", {}).get("frame_count") != 30 or report.get("audio_media", {}).get("sample_count") != EXPECTED_AUDIO_SAMPLES:
            raise ValueError("successful v0.5c report must prove both diagnostic media contracts")
    elif report.get("status") == "failed":
        failure = report.get("failure")
        if not isinstance(failure, Mapping) or not FAILURE_KEYS.issubset(set(failure)):
            missing = sorted(FAILURE_KEYS - set(failure or {}))
            raise ValueError(f"failed v0.5c report is missing failure evidence: {missing}")
        if not isinstance(failure.get("original_error"), Mapping) or not failure["original_error"].get("type"):
            raise ValueError("failed v0.5c report must preserve the original error")
    else:
        raise ValueError("v0.5c report status must be success or failed")


def _partial_paths(paths: Mapping[str, str]) -> list[str]:
    result: list[str] = []
    frames = Path(paths["frames"])
    if frames.is_dir():
        result.extend(str(path) for path in sorted(frames.glob("frame_*.png")))
    for key in ("audio_wav", "report"):
        if Path(paths[key]).is_file():
            result.append(paths[key])
    return result


def _suppression(phase_order: Sequence[str]) -> dict[str, bool]:
    phases = set(phase_order)
    return {
        "resident_worker_suppressed": "resident-worker" not in phases,
        "derived_worker_suppressed": "derived-worker" not in phases,
        "parity_suppressed": "exact-parity-gates" not in phases,
        "video_decoder_suppressed": "video-vae-load" not in phases,
        "audio_decoder_suppressed": "audio-vae-load" not in phases,
    }


def _failure_report(report: dict[str, Any], phase: str, worker: str, original: BaseException | Mapping[str, Any], *, cleanup: BaseException | Mapping[str, Any] | None = None, child: Mapping[str, Any] | None = None, temp_root: Path | None = None) -> dict[str, Any]:
    child = child or {}
    report["status"] = "failed"
    failure = {
        "active_phase": phase,
        "worker_identity": worker,
        "completed_stages": list(report.get("phase_order", [])),
        "original_error": _error(original),
        "cleanup_error": _error(cleanup) or _error(child.get("cleanup_error")),
        "subprocess_exit_status": child.get("subprocess", {}).get("exit_status") if isinstance(child.get("subprocess"), Mapping) else None,
        "subprocess_log_path": child.get("subprocess", {}).get("log_path") if isinstance(child.get("subprocess"), Mapping) else None,
        "partial_conditioning_metadata": report.get("conditioning_receipts", {}),
        "partial_packed_metadata": report.get("complete_packed_sequence_contract", {}),
        "partial_parity_metadata": report.get("exact_parity_gates", {}),
        "sidecar_state": report.get("streamed_adaln_lifecycle", {}) or child.get("sidecar_state", {}),
        "residency_state": {
            "resident": report.get("resident_worker", {}).get("residency", {}),
            "derived": report.get("derived_worker", {}).get("residency", {}),
            "video": report.get("video_memory", {}).get("loaded", False),
            "audio": report.get("audio_memory", {}).get("loaded", False),
        },
        "partial_media_paths": _partial_paths(report["output_paths"]),
        "memory_receipts": {
            "conditioning": report.get("conditioning_memory_release", {}),
            "resident": report.get("resident_memory_release", {}),
            "derived": report.get("derived_memory_release", {}),
            "video": report.get("video_memory", {}),
            "audio": report.get("audio_memory", {}),
            "final": report.get("final_memory", {}),
        },
        "later_phase_suppression": _suppression(report.get("phase_order", [])),
    }
    if temp_root is not None:
        failure["temporary_artifact_root"] = str(temp_root)
    report["failure"] = failure
    return report


def _parent_run(args: argparse.Namespace, paths: Mapping[str, str], temp_root: Path, report: dict[str, Any]) -> dict[str, Any]:
    root = Path(args.checkpoint_root).expanduser().resolve()
    resident = Path(args.resident_transformer).expanduser().resolve()
    derived = Path(args.derived_transformer).expanduser().resolve()
    conditioning_artifact = temp_root / "conditioning.npz"
    conditioning_receipt_path = temp_root / "conditioning-receipt.json"
    resident_artifact = temp_root / "resident-output.npz"
    resident_receipt_path = temp_root / "resident-receipt.json"
    derived_artifact = temp_root / "derived-output.npz"
    derived_receipt_path = temp_root / "derived-receipt.json"

    report["phase_order"].append("preflight")
    _validate_checkpoint_paths(root, resident, derived)
    child_common = [
        "--checkpoint-root", str(root), "--resident-transformer", str(resident), "--prompt", args.prompt,
        "--seed", str(args.seed), "--artifact", str(conditioning_artifact), "--receipt", str(conditioning_receipt_path),
        "--tolerance", str(args.active_memory_tolerance_bytes),
    ]
    if args.verbose:
        child_common.append("--verbose")
    conditioning_log = temp_root / "conditioning-worker.log"
    report["phase_order"].append("conditioning-worker")
    conditioning = _run_child("__conditioning-worker", child_common, conditioning_log, conditioning_receipt_path)
    report["conditioning_receipts"] = conditioning
    if conditioning.get("status") != "success":
        raise PhaseFailure("conditioning-worker", conditioning.get("error") or "conditioning worker failed", cleanup=conditioning.get("cleanup_error"), details={"child": conditioning})
    report["phase_order"].append("conditioning-release-gate")
    report["conditioning_memory_release"] = conditioning.get("conditioning_release", {})
    report["prompt"] = conditioning.get("prompt", report["prompt"])
    report["tokenizer_receipts"] = conditioning.get("tokenizer", {})
    report["deterministic_inputs"] = conditioning.get("deterministic_inputs", {})
    report["native_geometry"] = conditioning.get("geometry", report["native_geometry"])
    report["complete_packed_sequence_contract"] = conditioning.get("packed_contract", {})
    report["scheduler_receipts"] = conditioning.get("scheduler", {})
    validate_packed_contract(report["complete_packed_sequence_contract"])

    def transformer_args(role: str, transformer: Path, artifact: Path, receipt: Path) -> list[str]:
        values = [
            "--role", role, "--transformer", str(transformer), "--conditioning-artifact", str(conditioning_artifact),
            "--conditioning-receipt", str(conditioning_receipt_path), "--output-artifact", str(artifact),
            "--receipt", str(receipt), "--tolerance", str(args.active_memory_tolerance_bytes),
        ]
        if args.verbose:
            values.append("--verbose")
        return values

    resident_log = temp_root / "resident-worker.log"
    report["phase_order"].append("resident-worker")
    resident_receipt = _run_child("__transformer-worker", transformer_args("resident", resident, resident_artifact, resident_receipt_path), resident_log, resident_receipt_path)
    report["resident_worker"] = resident_receipt
    report["resident_memory_release"] = resident_receipt.get("transformer_release", {})
    if resident_receipt.get("status") != "success":
        raise PhaseFailure("resident-worker", resident_receipt.get("error") or "resident worker failed", cleanup=resident_receipt.get("cleanup_error"), details={"child": resident_receipt})
    _worker_success(resident_receipt, derived=False)
    report["phase_order"].append("resident-release-gate")

    derived_log = temp_root / "derived-worker.log"
    report["phase_order"].append("derived-worker")
    derived_receipt = _run_child("__transformer-worker", transformer_args("derived", derived, derived_artifact, derived_receipt_path), derived_log, derived_receipt_path)
    report["derived_worker"] = derived_receipt
    report["derived_memory_release"] = derived_receipt.get("transformer_release", {})
    report["streamed_adaln_lifecycle"] = derived_receipt.get("sidecar_state", {})
    if derived_receipt.get("status") != "success":
        raise PhaseFailure("derived-worker", derived_receipt.get("error") or "derived worker failed", cleanup=derived_receipt.get("cleanup_error"), details={"child": derived_receipt})
    _worker_success(derived_receipt, derived=True)
    report["phase_order"].append("derived-release-gate")

    report["phase_order"].append("exact-parity-gates")
    resident_arrays = _load_npz(resident_artifact)
    derived_arrays = _load_npz(derived_artifact)
    conditioning_arrays = _load_npz(conditioning_artifact)
    packed_names = {"packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices"}
    resident_packed = {key: resident_arrays[key] for key in packed_names if key in resident_arrays}
    derived_packed = {key: derived_arrays[key] for key in packed_names if key in derived_arrays}
    parity = compare_worker_artifacts(conditioning, resident_receipt, derived_receipt, resident_arrays, derived_arrays, resident_packed, derived_packed)
    report["exact_parity_gates"] = parity
    report["unpacked_native_latents"] = {
        "resident": resident_receipt.get("updated_latents", {}),
        "derived": derived_receipt.get("updated_latents", {}),
        "native_video_shape": list(EXPECTED_VIDEO_NATIVE_SHAPE),
        "native_audio_shape": list(EXPECTED_AUDIO_NATIVE_SHAPE),
        "exact_equality": bool(parity.get("native_video_latent", {}).get("exact_equality") and parity.get("native_audio_latent", {}).get("exact_equality")),
    }
    if not parity.get("all_exact_gates"):
        raise PhaseFailure("exact-parity-gates", "resident-versus-derived exact parity failed", details={"parity": parity})

    report["phase_order"].append("video-vae-load")
    try:
        media = _decode_media(args, paths, report["native_geometry"], {"updated_native_video": derived_arrays["updated_native_video"], "updated_native_audio": derived_arrays["updated_native_audio"]})
    except PhaseFailure as exc:
        report.update(exc.details)
        raise
    report.update(media)
    report["phase_order"].extend(["video-decode", "video-output-write", "video-release-gate", "audio-vae-load", "audio-decode", "audio-output-write", "audio-release-gate", "final-memory"])
    report["status"] = "success"
    return report


def run_command(args: argparse.Namespace) -> int:
    output_ready = False
    try:
        validate_locked_prompt(args.prompt)
        validate_seed(args.seed)
        if args.active_memory_tolerance_bytes < 0:
            raise ValueError("active-memory-tolerance-bytes must be nonnegative")
        paths = ensure_output_namespace(Path(args.output_root).expanduser().resolve(), args.overwrite)
        output_ready = True
        root = Path(args.checkpoint_root).expanduser().resolve()
        resident = Path(args.resident_transformer).expanduser().resolve()
        derived = Path(args.derived_transformer).expanduser().resolve()
        report = _base_report(args, paths, root, resident, derived)
    except BaseException as exc:
        if isinstance(exc, FileExistsError) and not output_ready:
            print(json.dumps({"status": "failed", "error": _error(exc)}, indent=2))
            return 1
        # Output validation happens before a report path can be safely selected only when the user
        # supplied a valid output root.  Keep this path conservative and never touch v0.5a/v0.5b.
        try:
            fallback_root = Path(args.output_root).expanduser().resolve()
            fallback_root.mkdir(parents=True, exist_ok=True)
            fallback_paths = {
                "root": str(fallback_root), "frames": str(fallback_root / "frames"),
                "audio_wav": str(fallback_root / "prompted-step-audio.wav"), "report": str(fallback_root / "prompted-step-report.json"),
            }
            report = _base_report(argparse.Namespace(**{**vars(args), "prompt": getattr(args, "prompt", "")}), fallback_paths, Path(getattr(args, "checkpoint_root", "")), Path(getattr(args, "resident_transformer", "")), Path(getattr(args, "derived_transformer", "")))
            report["phase_order"] = ["preflight"]
            _failure_report(report, "preflight", "parent", exc)
            _write_json(Path(fallback_paths["report"]), report)
            print(json.dumps({"status": "failed", "report": fallback_paths["report"]}, indent=2))
            return 1
        except BaseException:
            raise

    temp_root = Path(tempfile.mkdtemp(prefix="minimax-h3-v05c-"))
    try:
        report = _parent_run(args, paths, temp_root, report)
        validate_report(report)
        # Do not leave temporary worker artifacts in a successful output namespace.  Worker log
        # paths are deliberately nulled because the private namespace is removed after validation.
        for worker_key in ("conditioning_receipts", "resident_worker", "derived_worker"):
            worker = report.get(worker_key)
            if isinstance(worker, dict) and isinstance(worker.get("subprocess"), dict):
                worker["subprocess"]["log_path"] = None
                worker["subprocess"]["log_retained"] = False
        _write_json(Path(paths["report"]), report)
        shutil.rmtree(temp_root)
        print(json.dumps({"status": "success", "report": paths["report"], "phase_order": report["phase_order"]}, indent=2))
        return 0
    except PhaseFailure as exc:
        child = exc.details.get("child") if isinstance(exc.details, Mapping) else None
        worker_identity = child.get("worker_identity", "parent") if isinstance(child, Mapping) else "parent"
        report = _failure_report(
            report,
            exc.phase,
            worker_identity,
            exc.original_error or str(exc),
            cleanup=exc.cleanup_error,
            child=child,
            temp_root=temp_root,
        )
    except BaseException as exc:
        report = _failure_report(report, report.get("phase_order", ["preflight"])[-1], "parent", exc, temp_root=temp_root)
    try:
        validate_report(report)
    except BaseException as validation_error:
        _failure_report(report, "report-validation", "parent", validation_error, temp_root=temp_root)
    _write_json(Path(paths["report"]), report)
    print(json.dumps({"status": "failed", "report": paths["report"], "phase_order": report.get("phase_order", [])}, indent=2))
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "__conditioning-worker":
        return _conditioning_worker_main(raw[1:])
    if raw and raw[0] == "__transformer-worker":
        return _transformer_worker_main(raw[1:])
    parsed = build_parser().parse_args(raw)
    return int(parsed.func(parsed))


if __name__ == "__main__":
    raise SystemExit(main())

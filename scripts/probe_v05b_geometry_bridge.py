"""MiniMax-H3 v0.5b production geometry, packing, and decoder-entrance proof.

The module is MLX-free on import. Runtime imports happen only after the output namespace is checked
and the MLX-free contract tests have been run. It never loads the text encoder or transformer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CHECKPOINT_ROOT = Path("/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va")
DEFAULT_OUTPUT_ROOT = ROOT / "out" / "v0.5b"
COMMITTED_BASELINE = "7fd9322 Add v0.5a decoder lifecycle proof"
PROBE_FORMAT = "minimax-h3-mlx-v05b-production-multimodal-geometry-bridge"
SCHEMA_VERSION = 1
RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES = 1 * 1024 * 1024
DETERMINISTIC_INPUT_METHOD = "flat-index-sine-pattern-float32-v1"
FINGERPRINT_METHOD = "sha256-shape-dtype-canonical-float32-values-v1"
SOURCE_INSPECTION_FILES = (
    "minimax_h3_mlx/packing.py",
    "minimax_h3_mlx/pipeline.py",
    "minimax_h3_mlx/dit.py",
    "minimax_h3_mlx/video_decode_layout.py",
    "minimax_h3_mlx/video_vae.py",
    "minimax_h3_mlx/audio_vae.py",
    "minimax_h3_mlx/load.py",
    "tests/test_packing_parity.py",
    "tests/test_video_vae_parity.py",
    "tests/test_audio_vae_parity.py",
    "scripts/probe_v04a_one_step.py",
    "scripts/probe_v05a_decoders.py",
)
GENERATION_EXCLUSIONS = {
    "text_encoder_loaded": False,
    "transformer_loaded": False,
    "scheduler_loaded": False,
    "adaln_cache_loaded": False,
    "streamed_adaln_sidecars_loaded": False,
    "denoising_executed": False,
    "prompt_encoded": False,
    "image_conditioning_executed": False,
    "final_media_muxing_executed": False,
}


def _v05a():
    spec = importlib.util.spec_from_file_location("probe_v05a_decoders", ROOT / "scripts" / "probe_v05a_decoders.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("decode-deterministic-geometry", help="prove production packing and sequential VAE decoding")
    run.add_argument("--checkpoint-root", default=str(DEFAULT_CHECKPOINT_ROOT))
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--active-memory-tolerance-bytes", type=int, default=RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--verbose", action="store_true")
    run.set_defaults(func=run_command)
    return parser


def deterministic_values(element_count: int, salt: int) -> tuple[float, ...]:
    if not isinstance(element_count, int) or isinstance(element_count, bool) or element_count <= 0:
        raise ValueError("deterministic latent element count must be a positive integer")
    if salt not in (0, 1):
        raise ValueError("deterministic latent salt must be 0 for video or 1 for audio")
    return tuple(
        float(0.125 * np.sin((index + 1 + salt * 17) * 0.173) + 0.03125 * np.cos((index + 1 + salt * 17) * 0.071))
        for index in range(element_count)
    )


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(tuple(int(item) for item in value.shape)), "dtype": str(value.dtype).removeprefix("mlx.core.")}


def array_fingerprint(value: Any, mx: Any | None = None) -> str:
    if mx is not None and getattr(value, "__mlx_array__", False):
        canonical = value.astype(mx.float32)
        mx.eval(canonical)
        array = np.array(canonical, dtype=np.float32, copy=True)
    else:
        array = np.array(value, dtype=np.float32, copy=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"|float32|")
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _as_numpy(value: Any, mx: Any | None = None) -> np.ndarray:
    if mx is not None and getattr(value, "__mlx_array__", False):
        value = value.astype(mx.float32)
        mx.eval(value)
    return np.array(value, dtype=np.float32, copy=True)


def geometry_mapping(geometry: Any) -> dict[str, Any]:
    return {
        "batch": geometry.video_latent_shape[0],
        "latent_channels": geometry.video_latent_shape[1],
        "latent_frames": geometry.video_latent_shape[2],
        "latent_height": geometry.video_latent_shape[3],
        "latent_width": geometry.video_latent_shape[4],
        "audio_batch": geometry.audio_latent_shape[0],
        "audio_latent_channels": geometry.audio_latent_shape[1],
        "audio_latent_length": geometry.audio_latent_shape[2],
        "video_duration_seconds": geometry.duration_seconds,
        "audio_duration_seconds": float(geometry.audio_duration),
    }


def pack_native_latents(packing: Any, video_latent: Any, audio_latent: Any, geometry: Any) -> dict[str, Any]:
    """Call the production pack/layout APIs; no probe-local reshape is performed here."""
    video_rows = packing.patchify_video_latents(video_latent, geometry.video_patch_size)
    audio_rows = packing.pack_audio_latents(audio_latent)
    layout = packing.build_packed_sequence(
        np.empty((0,), dtype=np.int64),
        geometry.video_latent_shape[2],
        geometry.video_latent_shape[3],
        geometry.video_latent_shape[4],
        geometry.audio_latent_shape[2],
        geometry.video_patch_size,
    )
    timesteps, timestep_indices = packing.build_row_timesteps(layout, 0.5, 0.5, 0.999, 1.0)
    return {
        "video_rows": video_rows,
        "audio_rows": audio_rows,
        "layout": layout,
        "timesteps": timesteps,
        "timestep_indices": timestep_indices,
    }


def validate_native_geometry(video_shape: Sequence[int], audio_shape: Sequence[int], patch_size: Sequence[int]) -> None:
    video_shape = tuple(int(value) for value in video_shape)
    audio_shape = tuple(int(value) for value in audio_shape)
    patch_size = tuple(int(value) for value in patch_size)
    if len(video_shape) != 5 or video_shape[0] != 1:
        raise ValueError("video native geometry must be (1,C,F,H,W)")
    if len(patch_size) != 3 or any(value <= 0 for value in patch_size):
        raise ValueError("video patch size must contain three positive dimensions")
    if any(dimension % patch for dimension, patch in zip(video_shape[2:], patch_size)):
        raise ValueError(f"video latent geometry {video_shape} is not divisible by patch {patch_size}")
    if len(audio_shape) != 3 or audio_shape[0] != 2 or audio_shape[2] <= 0:
        raise ValueError("audio native geometry must use two mono batch items: (2,C,L) with positive L")


def exact_parity(packing: Any, packed: Mapping[str, Any], video_latent: Any, audio_latent: Any, geometry: Any, mx: Any) -> dict[str, Any]:
    video_back = packing.unpatchify_video_tokens(
        packed["video_rows"], geometry.video_latent_shape[2], geometry.video_latent_shape[3],
        geometry.video_latent_shape[4], geometry.video_latent_shape[1], geometry.video_patch_size,
    )
    audio_back = packing.unpack_audio_tokens(packed["audio_rows"], geometry.audio_latent_shape[2])
    video_np = _as_numpy(video_latent, mx)
    audio_np = _as_numpy(audio_latent, mx)
    video_back_np = _as_numpy(video_back, mx)
    audio_back_np = _as_numpy(audio_back, mx)
    result = {
        "video_shape_equal": tuple(video_back_np.shape) == tuple(video_np.shape),
        "video_dtype_equal": str(video_back.dtype).removeprefix("mlx.core.") == str(video_latent.dtype).removeprefix("mlx.core."),
        "video_value_equal": bool(np.array_equal(video_back_np, video_np)),
        "video_fingerprint_equal": array_fingerprint(video_back, mx) == array_fingerprint(video_latent, mx),
        "audio_shape_equal": tuple(audio_back_np.shape) == tuple(audio_np.shape),
        "audio_dtype_equal": str(audio_back.dtype).removeprefix("mlx.core.") == str(audio_latent.dtype).removeprefix("mlx.core."),
        "audio_value_equal": bool(np.array_equal(audio_back_np, audio_np)),
        "audio_fingerprint_equal": array_fingerprint(audio_back, mx) == array_fingerprint(audio_latent, mx),
        "unpacked_video": _shape_dtype(video_back),
        "unpacked_audio": _shape_dtype(audio_back),
    }
    result["all_exact_gates"] = all(result[key] for key in result if key.endswith("_equal"))
    return result


def source_contracts(video_config: Any, audio_config: Any, dit_config: Any, layout: Any, geometry: Any) -> dict[str, Any]:
    return {
        "inspected_files": list(SOURCE_INSPECTION_FILES),
        "source_locations": {
            "video_decode_layout": "minimax_h3_mlx/video_decode_layout.py:43-84",
            "video_decode": "minimax_h3_mlx/video_vae.py:598-658",
            "video_patchify": "minimax_h3_mlx/packing.py:186-222",
            "audio_pack_unpack": "minimax_h3_mlx/packing.py:225-245",
            "packed_sequence": "minimax_h3_mlx/packing.py:269-362",
            "dit_call_contract": "minimax_h3_mlx/dit.py:361-446",
            "pipeline_pack_and_decode": "minimax_h3_mlx/pipeline.py:523-734",
            "video_loader": "minimax_h3_mlx/load.py:423-499",
            "audio_loader": "minimax_h3_mlx/load.py:502-577",
        },
        "video": {
            "native_latent_axis_order": "(B,C,F,H,W)",
            "patch_size": list(geometry.video_patch_size),
            "latent_shape": list(geometry.video_latent_shape),
            "spatial_compression_ratio": int(video_config.spatial_compression_ratio),
            "temporal_layout": vars(layout),
            "decoded_frame_formula": "VideoVAE.decode: num_tokens=F+token_drop; pad to tokens_chunk_size; decode chunk overlap; exact repeated-tail trim",
            "decoded_frame_count": geometry.video_frames,
            "spatial_output_formula": "latent_height/width * spatial_compression_ratio",
            "divisibility": "F%pt=0, H%ph=0, W%pw=0 required by patchify_video_latents; decoder itself pads temporal tails",
        },
        "audio": {
            "native_latent_axis_order": "(B,C,L)",
            "stereo_representation": "two mono batch items",
            "latent_shape": list(geometry.audio_latent_shape),
            "latent_rate": geometry.audio_latent_rate,
            "sample_rate": geometry.audio_sample_rate,
            "sample_formula": "L * prod(decoder_rates)",
            "decoder_rates": list(audio_config.decoder_rates),
            "hop_length": int(audio_config.hop_length),
            "divisibility": "positive L; decoder_rates product must equal hop_length",
        },
        "packing": {
            "row_order": "[text | condition-video | target-audio | target-video]; with no text/conditions: [audio | video]",
            "video_token_formula": "B * F/pt * H/ph * W/pw",
            "audio_token_formula": "AUDIO_CHANNELS * L",
            "video_token_count": geometry.video_token_count,
            "audio_token_count": geometry.audio_token_count,
            "total_token_count": geometry.total_token_count,
            "video_feature_width": geometry.video_patch_width,
            "audio_feature_width": geometry.audio_patch_width,
            "modality_tags": {"video": 0, "text": 1, "audio": 2, "padding": -1},
            "position_ids": {"shape": [geometry.total_token_count, 3], "dtype": "float32", "axes": "(t,h,w)"},
            "token_tags": {"shape": [geometry.total_token_count], "dtype": "int32"},
            "mask": {"attention_mask": None, "padding_rows": 0, "reason": "padless production sequence"},
            "timesteps": {"distinct_shape": [1], "row_indices_shape": [geometry.total_token_count], "dtype": "float32/int32"},
        },
    }


def ensure_output_namespace(root: Path, overwrite: bool) -> dict[str, str]:
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing v0.5b outputs: {root}; pass --overwrite explicitly")
    root.mkdir(parents=True, exist_ok=True)
    frames = root / "frames"
    if overwrite and frames.exists():
        for path in frames.glob("frame_*.png"):
            if path.is_file():
                path.unlink()
    for path in (root / "geometry-audio.wav", root / "geometry-report.json"):
        if overwrite and path.is_file():
            path.unlink()
    return {"root": str(root), "frames": str(frames), "audio_wav": str(root / "geometry-audio.wav"), "report": str(root / "geometry-report.json")}


REPORT_KEYS = frozenset({
    "status", "probe_format", "schema_version", "committed_baseline", "checkpoint_root", "component_paths",
    "config_file_checksums", "locked_downstream_render_target", "source_derived_contracts", "geometry",
    "deterministic_inputs", "packing", "parity", "packing_memory", "video_media", "audio_media",
    "video_memory", "audio_memory", "final_memory", "phase_order", "generation_components",
    "residency", "output_paths", "failure",
})


def validate_report(report: Mapping[str, Any]) -> None:
    keys = set(report)
    if keys != REPORT_KEYS:
        raise ValueError(f"strict v0.5b report schema mismatch: missing={sorted(REPORT_KEYS - keys)}, unexpected={sorted(keys - REPORT_KEYS)}")
    if report.get("probe_format") != PROBE_FORMAT or report.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("v0.5b report identity mismatch")
    if report.get("status") == "success":
        if report.get("failure") is not None or not report.get("parity", {}).get("all_exact_gates"):
            raise ValueError("successful v0.5b report must prove exact parity and failure=null")
    elif report.get("status") == "failed":
        failure = report.get("failure")
        if not isinstance(failure, Mapping) or not all(field in failure for field in ("active_phase", "completed_stages", "error", "residency")):
            raise ValueError("failed v0.5b report must preserve a failure receipt")
    else:
        raise ValueError("v0.5b report status must be success or failed")


def _runtime_imports():
    import mlx.core as mx
    from minimax_h3_mlx import packing
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.load import load_audio_vae, load_audio_vae_config, load_video_vae, load_video_vae_config
    from minimax_h3_mlx.media import save_frames, save_wav
    from minimax_h3_mlx.video_decode_layout import resolve_video_decode_layout
    return mx, packing, DiTConfig, load_video_vae_config, load_video_vae, load_audio_vae_config, load_audio_vae, save_frames, save_wav, resolve_video_decode_layout


def _config_checksums(root: Path) -> dict[str, str]:
    result = {}
    for relative in ("transformer/config.json", "video_vae/config.json", "video_vae/source/config.json", "audio_vae/config.json", "audio_vae/metadata.json"):
        path = root / relative
        if path.is_file():
            result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")


def _record_report_validation_failure(report: dict[str, Any], error: BaseException) -> dict[str, Any]:
    """Convert an outer final-validation error into a diagnostic failure receipt."""
    phase_order = list(report.get("phase_order", []))
    if not phase_order or phase_order[-1] != "report-validation":
        phase_order.append("report-validation")
    report["status"] = "failed"
    report["phase_order"] = phase_order
    report.setdefault("video_memory", {})
    report.setdefault("audio_memory", {})
    report.setdefault("final_memory", {})
    output_paths = report.get("output_paths", {})
    partial_output_paths = list(output_paths.values()) if isinstance(output_paths, Mapping) else []
    report["failure"] = {
        "active_phase": "report-validation",
        "completed_stages": phase_order,
        "error": {"type": type(error).__name__, "message": str(error)},
        "cleanup_error": None,
        "partial_geometry": report.get("geometry", {}),
        "partial_pack_metadata": report.get("packing", {}),
        "partial_output_paths": partial_output_paths,
        "residency": report.get("residency", {}),
        "memory_receipts": {
            "packing": report.get("packing_memory", {}),
            "video": report.get("video_memory", {}),
            "audio": report.get("audio_memory", {}),
            "final": report.get("final_memory", {}),
        },
        "later_phase_suppression": {
            "audio_suppressed": "audio-baseline" not in phase_order,
            "video_suppressed": "video-vae-load" not in phase_order,
        },
    }
    return report


def _runtime_run(args: argparse.Namespace, paths: Mapping[str, str]) -> dict[str, Any]:
    v05a = _v05a()
    mx, packing, DiTConfig, load_vcfg, load_video, load_acfg, load_audio, save_frames, save_wav, resolve_layout = _runtime_imports()
    root = Path(args.checkpoint_root).expanduser().resolve()
    video_dir, audio_dir = root / "video_vae", root / "audio_vae"
    tolerance = int(args.active_memory_tolerance_bytes)
    phase_order: list[str] = []
    refs = {key: None for key in v05a.RUNTIME_REFERENCE_KEYS}
    video_refs = {key: None for key in v05a.RUNTIME_REFERENCE_KEYS}
    audio_refs = {key: None for key in v05a.RUNTIME_REFERENCE_KEYS}
    memory = {"packing": {}, "video": {}, "audio": {}}
    residency = {"video_vae_ever_loaded": False, "video_vae_currently_resident": False, "audio_vae_ever_loaded": False, "audio_vae_currently_resident": False}
    geometry = None
    unpacked_cpu: dict[str, np.ndarray] = {}
    report: dict[str, Any] = {
        "status": "failed", "probe_format": PROBE_FORMAT, "schema_version": SCHEMA_VERSION,
        "committed_baseline": COMMITTED_BASELINE, "checkpoint_root": str(root),
        "component_paths": {"transformer_config": str(root / "transformer" / "config.json"), "video_vae": str(video_dir), "audio_vae": str(audio_dir)},
        "config_file_checksums": _config_checksums(root), "locked_downstream_render_target": {"target_id": "locked-dodecahedron-materials-v1", "encoded": False, "prompt": None},
        "source_derived_contracts": {}, "geometry": {}, "deterministic_inputs": {}, "packing": {},
        "parity": {}, "packing_memory": {}, "video_media": {}, "audio_media": {},
        "video_memory": {}, "audio_memory": {}, "final_memory": {}, "phase_order": phase_order,
        "generation_components": dict(GENERATION_EXCLUSIONS), "residency": residency, "output_paths": dict(paths), "failure": None,
    }
    try:
        # Lightweight configs are metadata only; no VAE weights are loaded here.
        dit_config = DiTConfig.from_json(root / "transformer" / "config.json")
        video_config = load_vcfg(video_dir)
        audio_config = load_acfg(audio_dir)
        layout = resolve_layout(video_config)
        from minimax_h3_mlx.geometry import ProductionMultimodalGeometry
        geometry = ProductionMultimodalGeometry.canonical(video_config, audio_config, dit_config, layout)
        report["geometry"] = {**geometry_mapping(geometry), "video_latent_shape": list(geometry.video_latent_shape), "audio_latent_shape": list(geometry.audio_latent_shape), "video_duration_fraction": str(geometry.video_duration), "audio_duration_fraction": str(geometry.audio_duration), "alignment_evidence": list(geometry.alignment_evidence)}
        report["source_derived_contracts"] = source_contracts(video_config, audio_config, dit_config, layout, geometry)

        def pack_worker():
            phase_order.append("deterministic-latent-construction")
            video_latent = mx.array(np.asarray(deterministic_values(int(np.prod(geometry.video_latent_shape)), 0), dtype=np.float32).reshape(geometry.video_latent_shape), dtype=mx.float32)
            audio_latent = mx.array(np.asarray(deterministic_values(int(np.prod(geometry.audio_latent_shape)), 1), dtype=np.float32).reshape(geometry.audio_latent_shape), dtype=mx.float32)
            refs["latent"] = (video_latent, audio_latent)
            validate_native_geometry(geometry.video_latent_shape, geometry.audio_latent_shape, geometry.video_patch_size)
            phase_order.append("production-packing")
            packed = pack_native_latents(packing, video_latent, audio_latent, geometry)
            for value in (packed["video_rows"], packed["audio_rows"], packed["layout"].position_ids, packed["layout"].token_tags, packed["timesteps"], packed["timestep_indices"]):
                mx.eval(value)
            phase_order.append("production-unpacking")
            parity = exact_parity(packing, packed, video_latent, audio_latent, geometry, mx)
            unpacked_cpu["video"] = _as_numpy(packing.unpatchify_video_tokens(packed["video_rows"], geometry.video_latent_shape[2], geometry.video_latent_shape[3], geometry.video_latent_shape[4], geometry.video_latent_shape[1], geometry.video_patch_size), mx)
            unpacked_cpu["audio"] = _as_numpy(packing.unpack_audio_tokens(packed["audio_rows"], geometry.audio_latent_shape[2]), mx)
            phase_order.append("exact-parity-gates")
            if not parity["all_exact_gates"]:
                raise ValueError(f"exact pack/unpack parity failed: {parity}")
            report["deterministic_inputs"] = {"method": DETERMINISTIC_INPUT_METHOD, "fingerprint_method": FINGERPRINT_METHOD, "video": {"shape": list(geometry.video_latent_shape), "dtype": "float32", "salt": 0, "fingerprint": array_fingerprint(video_latent, mx)}, "audio": {"shape": list(geometry.audio_latent_shape), "dtype": "float32", "salt": 1, "fingerprint": array_fingerprint(audio_latent, mx)}}
            report["packing"] = {"video_rows": _shape_dtype(packed["video_rows"]), "audio_rows": _shape_dtype(packed["audio_rows"]), "video_token_count": int(packed["video_rows"].shape[0]), "audio_token_count": int(packed["audio_rows"].shape[0]), "total_token_count": packed["layout"].sequence_length, "feature_widths": {"video": int(packed["video_rows"].shape[1]), "audio": int(packed["audio_rows"].shape[1])}, "modality_order": "audio then video", "position_ids": _shape_dtype(packed["layout"].position_ids), "token_tags": _shape_dtype(packed["layout"].token_tags), "video_indices": _shape_dtype(packed["layout"].video_indices), "audio_indices": _shape_dtype(packed["layout"].audio_indices), "timestep_table": _shape_dtype(packed["timesteps"]), "timestep_indices": _shape_dtype(packed["timestep_indices"]), "attention_mask": None}
            report["parity"] = parity
            return {"worker": "cpu-receipts-materialized"}

        phase_order.append("packing-baseline")
        memory["packing"]["memory_before"] = v05a.memory_snapshot(mx)
        pack_result = v05a.execute_scoped_phase(pack_worker, phase="packing", mx=mx, references=refs, baseline=memory["packing"]["memory_before"], tolerance_bytes=tolerance, on_runtime_clear=lambda: phase_order.append("packing-runtime-clear"))
        memory["packing"]["release_gate"] = pack_result["release_gate"]
        phase_order.append("packing-release-gate")
        if not memory["packing"]["release_gate"]["passed"]:
            raise RuntimeError("packing release gate failed; VAE loading suppressed")

        # Video and audio phases reuse the hardened v0.5a scoped lifecycle helper.
        def video_worker():
            phase_order.append("video-config-load")
            video_refs["config"] = video_config
            video_refs["latent"] = mx.array(unpacked_cpu["video"], dtype=mx.float32)
            phase_order.append("video-vae-load")
            residency["video_vae_currently_resident"] = True
            video_refs["decoder"] = load_video(video_dir)
            residency["video_vae_ever_loaded"] = True
            scaled_mean, scaled_std = v05a._normalization_arrays(mx, video_config, audio=False)
            video_refs["scaled_latent"] = video_refs["latent"] * scaled_std + scaled_mean
            mx.eval(video_refs["scaled_latent"])
            phase_order.append("video-decode")
            raw = video_refs["decoder"].decode(video_refs["scaled_latent"].astype(mx.float32))
            mx.eval(raw)
            raw_np = np.array(raw.astype(mx.float32), dtype=np.float32, copy=True)
            frames = v05a.video_frames_from_raw(raw_np)
            v05a.validate_video_output(raw_np, frames, video_config, geometry_mapping(geometry), layout)
            save_frames(paths["frames"], frames)
            phase_order.append("video-output-write")
            metadata = v05a.frame_file_metadata(Path(paths["frames"]), geometry.video_frames)
            report["video_media"] = {"raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": "float32"}, "final_shape_dtype": {"shape": list(frames.shape), "dtype": "uint8"}, "frame_count": len(metadata), "frame_checksums": metadata}
            return {"worker": "video-cpu-receipt"}

        memory["video"]["memory_before"] = v05a.memory_snapshot(mx)
        video_result = v05a.execute_scoped_phase(video_worker, phase="video", mx=mx, references=video_refs, baseline=memory["video"]["memory_before"], tolerance_bytes=tolerance, on_runtime_clear=lambda: phase_order.append("video-runtime-clear"), on_release_success=lambda _: (phase_order.append("video-release-gate"), residency.update(video_vae_currently_resident=False)))
        memory["video"]["release_gate"] = video_result["release_gate"]
        if not memory["video"]["release_gate"]["passed"] or residency["video_vae_currently_resident"]:
            raise RuntimeError("video release gate failed; audio loading suppressed")

        def audio_worker():
            phase_order.append("audio-config-load")
            audio_refs["config"] = audio_config
            audio_refs["latent"] = mx.array(unpacked_cpu["audio"], dtype=mx.float32)
            phase_order.append("audio-vae-load")
            residency["audio_vae_currently_resident"] = True
            audio_refs["decoder"] = load_audio(audio_dir)
            residency["audio_vae_ever_loaded"] = True
            scaled_mean, scaled_std = v05a._normalization_arrays(mx, audio_config, audio=True)
            audio_refs["scaled_latent"] = audio_refs["latent"] * scaled_std + scaled_mean
            mx.eval(audio_refs["scaled_latent"])
            phase_order.append("audio-decode")
            raw = audio_refs["decoder"].decode(audio_refs["scaled_latent"].astype(mx.float32))
            mx.eval(raw)
            raw_np = np.array(raw.astype(mx.float32), dtype=np.float32, copy=True)
            waveform = raw_np[:, 0, :].astype(np.float32, copy=True)
            v05a.validate_audio_output(raw_np, waveform, audio_config, {"batch": 2, "latent_channels": geometry.audio_latent_shape[1], "latent_length": geometry.audio_latent_shape[2]})
            save_wav(paths["audio_wav"], waveform, geometry.audio_sample_rate)
            phase_order.append("audio-output-write")
            wav = v05a.wav_metadata(Path(paths["audio_wav"]))
            v05a.validate_wav_metadata(wav, waveform, audio_config)
            report["audio_media"] = {"raw_shape_dtype": {"shape": list(raw_np.shape), "dtype": "float32"}, "final_shape_dtype": {"shape": list(waveform.shape), "dtype": "float32"}, "sample_rate": geometry.audio_sample_rate, "sample_count": geometry.audio_samples, "duration_seconds": float(geometry.audio_duration), "wav": wav}
            return {"worker": "audio-cpu-receipt"}

        memory["audio"]["memory_before"] = v05a.memory_snapshot(mx)
        audio_result = v05a.execute_scoped_phase(audio_worker, phase="audio", mx=mx, references=audio_refs, baseline=memory["audio"]["memory_before"], tolerance_bytes=tolerance, on_runtime_clear=lambda: phase_order.append("audio-runtime-clear"), on_release_success=lambda _: (phase_order.append("audio-release-gate"), residency.update(audio_vae_currently_resident=False)))
        memory["audio"]["release_gate"] = audio_result["release_gate"]
        report.update({"status": "success", "packing_memory": memory["packing"], "residency": residency, "phase_order": phase_order, "video_memory": memory["video"], "audio_memory": memory["audio"], "final_memory": v05a.memory_snapshot(mx)})
        phase_order.append("report-validation")
        validate_report(report)
        return report
    except BaseException as exc:
        try:
            traceback.clear_frames(exc.__traceback__)
        except BaseException:
            pass
        for reference in (refs, video_refs, audio_refs):
            for key in reference:
                reference[key] = None
        gc.collect()
        original = getattr(exc, "original_error", exc)
        cleanup = getattr(exc, "cleanup_error", None)
        partial_outputs = [str(path) for path in Path(paths["frames"]).glob("frame_*.png")] if Path(paths["frames"]).exists() else []
        if Path(paths["audio_wav"]).is_file():
            partial_outputs.append(paths["audio_wav"])
        report["status"] = "failed"
        report["failure"] = {
            "active_phase": phase_order[-1] if phase_order else "preflight",
            "completed_stages": list(phase_order),
            "error": {"type": type(original).__name__, "message": str(original)},
            "cleanup_error": None if cleanup is None else {"type": type(cleanup).__name__, "message": str(cleanup)},
            "partial_geometry": report["geometry"],
            "partial_pack_metadata": report["packing"],
            "partial_output_paths": partial_outputs,
            "residency": dict(residency),
            "memory_receipts": memory,
            "later_phase_suppression": {"audio_suppressed": "audio-baseline" not in phase_order, "video_suppressed": "video-vae-load" not in phase_order},
        }
        report["packing_memory"] = report.get("packing_memory") or memory["packing"]
        report["video_memory"] = report.get("video_memory") or memory["video"]
        report["audio_memory"] = report.get("audio_memory") or memory["audio"]
        return report


def run_command(args: argparse.Namespace) -> int:
    paths = ensure_output_namespace(Path(args.output_root).expanduser().resolve(), args.overwrite)
    report = _runtime_run(args, paths)
    try:
        validate_report(report)
    except BaseException as exc:
        report = _record_report_validation_failure(report, exc)
        _write_report(Path(paths["report"]), report)
        print(json.dumps({"status": report["status"], "report": paths["report"], "phase_order": report.get("phase_order", [])}, indent=2))
        return 1
    _write_report(Path(paths["report"]), report)
    print(json.dumps({"status": report["status"], "report": paths["report"], "phase_order": report.get("phase_order", [])}, indent=2))
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    raise SystemExit(parsed.func(parsed))

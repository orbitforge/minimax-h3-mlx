"""v0.4a resident-versus-cache-only parity for one production denoising step."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ORIGINAL = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit"
DEFAULT_DERIVED = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_ARTIFACT = ROOT / "out" / "v0.4a" / "one-step-reference.safetensors"
DEFAULT_METADATA = ROOT / "out" / "v0.4a" / "one-step-reference.json"
DEFAULT_REPORT = ROOT / "out" / "v0.4a" / "one-step-parity-report.json"

ARTIFACT_FORMAT = "minimax-h3-mlx-v04a-one-production-denoising-step"
ARTIFACT_SCHEMA_VERSION = 1
FINGERPRINT_METHOD = "sha256(index-json)+sha256(full-content-of-every-indexed-resident-shard)-v1"
DETERMINISTIC_INPUT_METHOD = "tensor-index-pattern-v1"
RELATIVE_DENOMINATOR_FLOOR = 1.0e-6
ALLCLOSE_ATOL = 1.0e-5
ALLCLOSE_RTOL = 1.0e-5
EXPECTED_BLOCK_COUNT = 50
# The scheduler API receives sigma-grid points; three points contain two real model transitions.
INFERENCE_STEPS = 2

ARTIFACT_KEYS = (
    "original_video_latent", "original_audio_latent", "text_input", "timestep", "timestep_indices",
    "token_tags", "position_ids", "video_indices", "audio_indices", "text_indices",
    "resident_video_prediction", "resident_audio_prediction", "resident_updated_video_latent",
    "resident_updated_audio_latent", "selected_step_index", "video_current_timestep",
    "video_next_timestep", "video_current_sigma", "video_next_sigma", "audio_current_timestep",
    "audio_next_timestep", "audio_current_sigma", "audio_next_sigma",
)
METADATA_KEYS = frozenset({
    "artifact_format", "artifact_schema_version", "artifact_file_format", "reference_checkpoint",
    "derived_checkpoint", "resident_checkpoint_fingerprint", "fingerprint_method", "resident_config_sha256",
    "derived_config_sha256", "conversion_manifest_sha256", "sidecar_manifest_sha256", "artifact_sha256",
    "deterministic_input_method", "deterministic_input_spec", "tensor_keys", "tensor_inventory",
    "packed_layout", "scheduler_identity", "scheduler_configuration", "prediction_parameterization",
    "inference_step_count", "selected_step_index", "video_current_timestep", "video_next_timestep",
    "video_current_sigma", "video_next_sigma", "audio_current_timestep", "audio_next_timestep",
    "audio_current_sigma", "audio_next_sigma", "timestep_row_convention", "model_input_scaling",
    "update_method", "configured_resident_block_count",
    "observed_resident_block_count", "observed_resident_block_indices", "transformer_construction_mode",
})


def artifact_metadata_path(artifact: Path) -> Path:
    if artifact.suffix != ".safetensors":
        raise ValueError(f"v0.4a reference artifact must use .safetensors: {artifact}")
    return artifact.with_suffix(".json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def existing_artifact_paths(*paths: Path | str) -> list[str]:
    return [str(path) for path in paths if Path(path).is_file()]


def deterministic_input_parameters() -> dict[str, Any]:
    return {"modulus": 23, "offset": 1.0, "scale_base": 0.001, "scale_step": 0.0001,
            "allowed_salts": (0, 1, 2)}


def deterministic_input_specification() -> dict[str, Any]:
    parameters = deterministic_input_parameters()
    return {"method": DETERMINISTIC_INPUT_METHOD, **parameters,
            "allowed_salts": list(parameters["allowed_salts"])}


def deterministic_input_values(element_count: int, salt: int) -> tuple[float, ...]:
    if not isinstance(element_count, int) or isinstance(element_count, bool) or element_count <= 0:
        raise ValueError("deterministic input element count must be a strictly positive integer")
    parameters = deterministic_input_parameters()
    if salt not in parameters["allowed_salts"]:
        raise ValueError("deterministic input salt must be one of [0, 1, 2]")
    scale = parameters["scale_base"] + salt * parameters["scale_step"]
    return tuple((((index + salt + parameters["offset"]) % parameters["modulus"]) + parameters["offset"]) * scale
                 for index in range(element_count))


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": [int(item) for item in value.shape], "dtype": _dtype_name(value.dtype)}


def _finite(mx: Any, value: Any, label: str) -> None:
    result = mx.all(mx.isfinite(value)) if mx is not None else np.all(np.isfinite(np.asarray(value)))
    if mx is not None and hasattr(mx, "eval"):
        mx.eval(result)
    if not bool(result.item() if hasattr(result, "item") else result):
        raise ValueError(f"{label} contains nonfinite values")


def _exact(mx: Any, left: Any, right: Any) -> bool:
    result = mx.all(left == right) if mx is not None else np.all(np.asarray(left) == np.asarray(right))
    if mx is not None and hasattr(mx, "eval"):
        mx.eval(result)
    return bool(result.item() if hasattr(result, "item") else result)


def _metric_report(mx: Any, resident: Any, derived: Any) -> dict[str, Any]:
    if tuple(resident.shape) != tuple(derived.shape) or _dtype_name(resident.dtype) != _dtype_name(derived.dtype):
        raise ValueError(f"parity shape/dtype mismatch: {_shape_dtype(resident)} vs {_shape_dtype(derived)}")
    if mx is None:
        left = np.asarray(resident, dtype=np.float32)
        right = np.asarray(derived, dtype=np.float32)
        difference = np.abs(left - right)
        denominator = np.maximum(np.abs(left), RELATIVE_DENOMINATOR_FLOOR)
        exact_equality = bool(np.array_equal(np.asarray(resident), np.asarray(derived)))
        max_difference = float(np.max(difference))
        mean_difference = float(np.mean(difference))
        rms_difference = float(np.sqrt(np.mean(np.square(left - right))))
        max_relative = float(np.max(difference / denominator))
        allclose = bool(np.allclose(left, right, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL))
        mismatched = int(np.count_nonzero(np.asarray(resident) != np.asarray(derived)))
    else:
        # MLX BF16 arrays must stay on the MLX path.  Float32 is used only for diagnostics; exact
        # equality is evaluated against the original stored dtype values above.
        exact = mx.all(resident == derived)
        left = resident.astype(mx.float32)
        right = derived.astype(mx.float32)
        difference = mx.abs(left - right)
        denominator = mx.maximum(mx.abs(left), RELATIVE_DENOMINATOR_FLOOR)
        max_value = mx.max(difference)
        mean_value = mx.mean(difference)
        rms_value = mx.sqrt(mx.mean(mx.square(left - right)))
        relative_value = mx.max(difference / denominator)
        allclose_value = mx.all(difference <= ALLCLOSE_ATOL + ALLCLOSE_RTOL * mx.abs(left))
        mismatched_value = mx.sum((resident != derived).astype(mx.int32))
        mx.eval(exact, max_value, mean_value, rms_value, relative_value, allclose_value, mismatched_value)
        exact_equality = bool(exact.item())
        max_difference = float(max_value.item())
        mean_difference = float(mean_value.item())
        rms_difference = float(rms_value.item())
        max_relative = float(relative_value.item())
        allclose = bool(allclose_value.item())
        mismatched = int(mismatched_value.item())
    return {
        "exact_equality": exact_equality,
        "maximum_absolute_difference": max_difference,
        "mean_absolute_difference": mean_difference,
        "root_mean_square_difference": rms_difference,
        "maximum_relative_difference": max_relative,
        "relative_denominator_floor": RELATIVE_DENOMINATOR_FLOOR,
        "allclose": allclose,
        "allclose_atol": ALLCLOSE_ATOL, "allclose_rtol": ALLCLOSE_RTOL,
        "element_count": int(math.prod(resident.shape)),
        "mismatched_element_count_exact": mismatched,
        "resident": _shape_dtype(resident), "derived": _shape_dtype(derived),
    }


def validate_combined_exact_parity(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    expected = {"video_prediction", "audio_prediction", "updated_video_latent", "updated_audio_latent"}
    if set(metrics) != expected or not all(bool(metrics[key]["exact_equality"]) for key in expected):
        raise ValueError("one-step parity failed: exact equality is required for all four comparisons")


def write_diagnostic_report(report_path: Path, report: Mapping[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True, default=str) + "\n")


def validate_parity_after_report(report_path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    if not report_path.is_file():
        raise ValueError(f"diagnostic report must exist before parity validation: {report_path}")
    validate_combined_exact_parity(metrics)


def emit_parity_success_message(parity_validated: bool) -> None:
    if not parity_validated:
        raise ValueError("cannot emit one-step parity success before parity validation")
    print("ONE-STEP PARITY PASSED", flush=True)


def validate_artifact_tensor_keys(keys: Any) -> None:
    if list(keys) != list(ARTIFACT_KEYS):
        raise ValueError(f"artifact tensor key contract mismatch: got {keys!r}, expected {list(ARTIFACT_KEYS)!r}")


TRANSITION_FIELDS = (
    "selected_step_index", "video_current_timestep", "video_next_timestep", "video_current_sigma",
    "video_next_sigma", "audio_current_timestep", "audio_next_timestep", "audio_current_sigma",
    "audio_next_sigma",
)


def transition_tensor_values(mx: Any, arrays: Mapping[str, Any]) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    for field in TRANSITION_FIELDS:
        value = arrays[field]
        if tuple(value.shape) != (1,):
            raise ValueError(f"serialized transition tensor {field} must have shape [1]")
        scalar = value[0]
        if mx is not None:
            mx.eval(scalar)
        scalar = scalar.item() if hasattr(scalar, "item") else scalar
        values[field] = int(scalar) if field == "selected_step_index" else float(scalar)
    return values


def artifact_transition_values(transition: Mapping[str, Any]) -> dict[str, float | int]:
    if "step_index" in transition:
        selected = transition["step_index"]
        required = {"step_index", *(field for field in TRANSITION_FIELDS if field != "selected_step_index")}
    elif "selected_step_index" in transition:
        selected = transition["selected_step_index"]
        required = set(TRANSITION_FIELDS)
    else:
        raise ValueError("transition is missing required selected step field: step_index")
    missing = sorted(required - set(transition))
    if missing:
        raise ValueError(f"transition is missing required fields: {missing}")
    return {field: (int(selected) if field == "selected_step_index"
                    else float(np.float32(transition[field])))
            for field in TRANSITION_FIELDS}


def validate_serialized_transition(mx: Any, arrays: Mapping[str, Any], metadata: Mapping[str, Any],
                                   expected: Mapping[str, Any]) -> None:
    missing_metadata = sorted(set(TRANSITION_FIELDS) - set(metadata))
    if missing_metadata:
        raise ValueError(f"serialized transition metadata is missing required fields: {missing_metadata}")
    serialized = transition_tensor_values(mx, arrays)
    expected_values = artifact_transition_values(expected)
    for field in TRANSITION_FIELDS:
        expected_value = expected_values[field]
        if serialized[field] != expected_value or metadata[field] != expected_value:
            raise ValueError(f"serialized transition mismatch for {field}")


def validate_metadata_keys(metadata: Mapping[str, Any]) -> None:
    missing = sorted(METADATA_KEYS - set(metadata))
    unexpected = sorted(set(metadata) - METADATA_KEYS)
    if missing or unexpected:
        raise ValueError(f"metadata key contract mismatch: missing={missing}, unexpected={unexpected}")


def validate_reference_metadata(metadata: Mapping[str, Any], *, original: Path, derived: Path,
                                expected_checksums: Mapping[str, str], expected_inventory: Mapping[str, Any],
                                expected_layout: Mapping[str, Any], expected_transition: Mapping[str, Any]) -> None:
    validate_metadata_keys(metadata)
    transition = artifact_transition_values(expected_transition)
    expected = {
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors", "reference_checkpoint": str(original.resolve()),
        "derived_checkpoint": str(derived.resolve()), "fingerprint_method": FINGERPRINT_METHOD,
        "deterministic_input_method": DETERMINISTIC_INPUT_METHOD, "tensor_keys": list(ARTIFACT_KEYS),
        "packed_layout": dict(expected_layout), "scheduler_identity": "MiniMaxH3MultimodalScheduler",
        "scheduler_configuration": {"identity": "MiniMaxH3MultimodalScheduler",
                                     "video": {"identity": "MiniMaxH3Scheduler", "shift": 12.0},
                                     "audio": {"identity": "MiniMaxH3Scheduler", "shift": 3.0},
                                     "num_inference_steps": INFERENCE_STEPS,
                                     "prediction_parameterization": "velocity", "input_scaling": "identity",
                                     "update_method": "rectified-flow-euler-data-ward-velocity-v1"},
        "prediction_parameterization": "velocity", "inference_step_count": INFERENCE_STEPS,
        **transition,
        "timestep_row_convention": {"text": "video_current_timestep", "conditioning_video": "0.999",
                                     "conditioning_audio": "0.999", "target_video": "video_current_timestep",
                                     "target_audio": "audio_current_timestep"},
        "model_input_scaling": "identity",
        "update_method": "rectified-flow-euler-data-ward-velocity-v1", "configured_resident_block_count": EXPECTED_BLOCK_COUNT,
        "observed_resident_block_count": EXPECTED_BLOCK_COUNT, "observed_resident_block_indices": list(range(EXPECTED_BLOCK_COUNT)),
        "transformer_construction_mode": "resident",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata mismatch for {key}: got {metadata.get(key)!r}, expected {value!r}")
    if metadata.get("deterministic_input_spec") != deterministic_input_specification():
        raise ValueError("metadata deterministic input specification mismatch")
    for key, value in expected_checksums.items():
        if metadata.get(key) != value:
            raise ValueError(f"metadata checksum mismatch for {key}")
    if metadata.get("tensor_inventory") != dict(expected_inventory):
        raise ValueError("metadata tensor inventory does not match the serialized artifact")


def _resident_shard_paths(checkpoint: Path) -> tuple[Path, ...]:
    index_path = checkpoint / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("resident checkpoint index has no weight_map")
    values = list(weight_map.values())
    if not all(isinstance(name, str) for name in values):
        raise ValueError("resident checkpoint index shard names must all be strings")
    names = sorted(set(values))
    if not names or any(not name or Path(name).name != name or name in {".", ".."} for name in names):
        raise ValueError("resident checkpoint index contains unsafe or empty shard basenames")
    paths = tuple(checkpoint / name for name in names)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"resident checkpoint index references missing shards: {missing}")
    return paths


def resident_checkpoint_fingerprint(checkpoint: Path) -> str:
    index = checkpoint / "model.safetensors.index.json"
    payload = {"method": FINGERPRINT_METHOD, "index_sha256": sha256_file(index),
               "shards": [{"filename": path.name, "sha256": sha256_file(path)} for path in _resident_shard_paths(checkpoint)]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class _ObservedTransformerBlock:
    def __init__(self, index: int, block: Any, observed: list[int]):
        self.index, self.block, self.observed = index, block, observed

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.observed.append(self.index)
        return self.block(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.block, name)


@contextmanager
def observe_transformer_block_execution(dit: Any):
    blocks = getattr(dit, "blocks", None)
    if not isinstance(blocks, list):
        raise TypeError("transformer block observation requires a mutable list of blocks")
    original = list(blocks)
    observed: list[int] = []
    blocks[:] = [_ObservedTransformerBlock(index, block, observed) for index, block in enumerate(original)]
    try:
        yield observed
    finally:
        blocks[:] = original


def validate_observed_block_indices(configured_count: int, observed: list[int] | tuple[int, ...]) -> None:
    expected = list(range(configured_count))
    if list(observed) != expected:
        raise ValueError(f"transformer block execution mismatch: got {list(observed)}, expected {expected}")


def validate_complete_cache_stats(*, cache_table_count: int, configured_block_count: int,
                                  stats: Any, actual_sidecar_names: list[str]) -> None:
    expected_names = [f"block-{index:03d}.safetensors" for index in range(configured_block_count)]
    for label, value in (("cache tables", cache_table_count), ("blocks completed", stats.blocks_completed),
                         ("sidecar files opened", stats.sidecar_files_opened),
                         ("unique sidecars opened", stats.unique_sidecar_files_opened),
                         ("successful payload opens", stats.successful_payload_opens),
                         ("completed payload releases", stats.completed_payload_releases)):
        if value != configured_block_count:
            raise ValueError(f"complete cache contract violation: {label}={value}, expected {configured_block_count}")
    if actual_sidecar_names != expected_names:
        raise ValueError(f"sidecar order mismatch: got {actual_sidecar_names}, expected {expected_names}")
    for label, value, expected in (("every_sidecar_released_before_next_opened", stats.every_sidecar_released_before_next_opened, True),
                                   ("sidecar_overlap_observed", stats.sidecar_overlap_observed, False),
                                   ("next_sidecar_opened_before_previous_release", stats.next_sidecar_opened_before_previous_release, False),
                                   ("dense_temporary_projection_created", stats.dense_temporary_projection_created, False)):
        if value is not expected:
            raise ValueError(f"cache lifecycle contract violation: {label}={value!r}, expected {expected!r}")


def _build_canonical_scheduler(num_steps: int = INFERENCE_STEPS):
    from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler
    video = MiniMaxH3Scheduler(shift=12.0)
    audio = MiniMaxH3Scheduler(shift=3.0)
    if num_steps != INFERENCE_STEPS:
        raise ValueError(f"v0.4a canonical scheduler requires exactly {INFERENCE_STEPS} transitions")
    video.set_timesteps(num_steps + 1)
    audio.set_timesteps(num_steps + 1)
    scheduler = MiniMaxH3MultimodalScheduler(video, audio)
    transition = scheduler.transition(0)
    if scheduler.num_inference_steps != 2 or transition.step_index != 0:
        raise ValueError("canonical scheduler must expose exactly two transitions and select step 0")
    if transition.video_current_sigma != 1.0 or transition.audio_current_sigma != 1.0:
        raise ValueError("canonical selected transition must start both modalities at sigma 1.0")
    if not (0.0 < transition.video_next_sigma < 1.0 and 0.0 < transition.audio_next_sigma < 1.0):
        raise ValueError("canonical selected transition next sigmas must be strictly between zero and one")
    if transition.video_next_sigma == transition.audio_next_sigma:
        raise ValueError("canonical video and audio next sigmas must differ")
    for scalar, prefix in ((video, "video"), (audio, "audio")):
        scalar_transition = scalar.transition(0)
        for field in ("current_timestep", "next_timestep", "current_sigma", "next_sigma"):
            if getattr(transition, f"{prefix}_{field}") != getattr(scalar_transition, field):
                raise ValueError(f"canonical {prefix} transition does not agree with its scalar scheduler")
    return scheduler


def _build_layout_and_inputs(mx: Any, dit: Any, transition: Mapping[str, Any]):
    from minimax_h3_mlx.config import TAG_TEXT
    from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps
    layout = build_packed_sequence(np.array([TAG_TEXT], dtype=np.int64), 1, 2, 2, 1,
                                   tuple(dit.config.patch_size), keyframe_anchors=())
    timestep, timestep_indices = build_row_timesteps(
        layout, transition["video_current_timestep"], transition["audio_current_timestep"],
        0.999, 0.999)
    def pattern(shape: tuple[int, ...], salt: int):
        values = mx.array(deterministic_input_values(math.prod(shape), salt), dtype=mx.float32)
        return values.reshape(shape).astype(mx.bfloat16)
    video = pattern((1, 1, int(dit.config.video_patch_dim)), 0)
    audio = pattern((1, 2, int(dit.config.audio_latents_dim)), 1)
    text = pattern((1, 1, int(dit.config.text_dim)), 2)
    for label, value in (("video latent", video), ("audio latent", audio), ("text input", text)):
        _finite(mx, value, label)
        if not bool(mx.all(value != 0).item()):
            raise ValueError(f"{label} violates the strictly-nonzero input contract")
    return layout, timestep, timestep_indices, video, audio, text


def _layout_contract(layout: Any, timestep: Any) -> dict[str, Any]:
    return {"sequence_length": int(layout.sequence_length), "text_token_count": int(layout.text_indices.shape[0]),
            "video_token_count": int(layout.video_indices.shape[0]), "audio_token_count": int(layout.audio_indices.shape[0]),
            "video_shape": [1, 1, 96], "audio_shape": [1, 2, 32], "text_shape": [1, 1, 5120],
            "timestep_values": [float(value) for value in timestep.tolist()],
            "timestep_dtype": _dtype_name(timestep.dtype)}


def _load_reference_arrays(mx: Any, artifact: Path) -> dict[str, Any]:
    if not artifact.is_file():
        raise FileNotFoundError(f"reference artifact is missing: {artifact}")
    arrays = mx.load(str(artifact))
    if set(arrays) != set(ARTIFACT_KEYS):
        raise ValueError(f"serialized tensor key contract mismatch: got {sorted(arrays)}")
    return arrays


def _write_artifact(mx: Any, artifact: Path, metadata_path: Path, arrays: Mapping[str, Any], metadata: Mapping[str, Any], overwrite: bool) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and (artifact.exists() or metadata_path.exists()):
        raise FileExistsError(f"reference artifact already exists; pass --overwrite to replace {artifact}")
    mx.save_safetensors(str(artifact), dict(arrays), metadata={"artifact_format": ARTIFACT_FORMAT,
                       "artifact_schema_version": str(ARTIFACT_SCHEMA_VERSION),
                       "tensor_keys": json.dumps(list(ARTIFACT_KEYS), separators=(",", ":"))})
    receipt = dict(metadata)
    receipt["artifact_sha256"] = sha256_file(artifact)
    metadata_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def _release(mx: Any) -> dict[str, Any]:
    gc.collect()
    clear = getattr(mx, "clear_cache", None)
    available = callable(clear)
    if available:
        clear()
    return {"allocator_cache_purge_available": available, "allocator_cache_purged": available}


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, getter_name in (("active_memory", "get_active_memory"), ("allocator_cache", "get_cache_memory"),
                               ("peak_memory", "get_peak_memory")):
        getter = getattr(mx, getter_name, None)
        try:
            result[label] = int(getter()) if callable(getter) else None
        except Exception:
            result[label] = None
    return result


def emit_phase(phase: str, started: float, before: Mapping[str, Any], after: Mapping[str, Any], **details: Any) -> None:
    receipt = {"phase": phase, "wall_clock_seconds": round(time.perf_counter() - started, 6),
               "memory_before": dict(before), "memory_after": dict(after), **details}
    print(f"phase_receipt={json.dumps(receipt, sort_keys=True, default=str)}", flush=True)


def _detach(exc: BaseException) -> BaseException:
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None
    exc.__suppress_context__ = True
    return exc


def _validate_model(dit: Any, mode: str) -> None:
    if getattr(dit, "construction_mode", None) != mode:
        raise ValueError(f"transformer construction mode is {getattr(dit, 'construction_mode', None)!r}, expected {mode!r}")
    if int(dit.config.num_layers) != EXPECTED_BLOCK_COUNT or len(dit.blocks) != EXPECTED_BLOCK_COUNT:
        raise ValueError("transformer must expose exactly 50 configured blocks")


def _prediction_dtype(dit: Any) -> str:
    from minimax_h3_mlx.dit import param_dtype
    return _dtype_name(param_dtype(dit.final_layer.video_out))


def _run_one_step(mx: Any, dit: Any, scheduler: Any, arrays: Mapping[str, Any], layout: Any,
                  timestep: Any, step_index: int, cache: Any = None):
    from minimax_h3_mlx.denoise import one_step_denoise
    kwargs = {"modulation_cache": cache} if cache is not None else {}
    with observe_transformer_block_execution(dit) as observed:
        result = one_step_denoise(
            dit, scheduler, video_latent=arrays["original_video_latent"], audio_latent=arrays["original_audio_latent"],
            text_embedding=arrays["text_input"], timestep=timestep, timestep_indices=arrays["timestep_indices"],
            token_tags=arrays["token_tags"], position_ids=arrays["position_ids"], video_indices=arrays["video_indices"],
            audio_indices=arrays["audio_indices"], text_indices=arrays["text_indices"], step_index=step_index,
            expected_prediction_dtype=_prediction_dtype(dit), **kwargs)
    validate_observed_block_indices(len(dit.blocks), observed)
    return result, {"configured_transformer_block_count": len(dit.blocks),
                    "observed_transformer_block_indices": list(observed), "observed_transformer_block_count": len(observed)}


def _metadata(*, original: Path, derived: Path, scheduler: Any, transition: Mapping[str, Any], dit: Any,
              layout: Any, timestep: Any, arrays: Mapping[str, Any], observed: list[int]) -> dict[str, Any]:
    transition = artifact_transition_values(transition)
    return {"artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_file_format": "safetensors", "reference_checkpoint": str(original.resolve()),
            "derived_checkpoint": str(derived.resolve()), "resident_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
            "fingerprint_method": FINGERPRINT_METHOD, "resident_config_sha256": sha256_file(original / "config.json"),
            "derived_config_sha256": sha256_file(derived / "config.json"),
            "conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
            "sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json"),
            "artifact_sha256": "pending", "deterministic_input_method": DETERMINISTIC_INPUT_METHOD,
            "deterministic_input_spec": deterministic_input_specification(), "tensor_keys": list(ARTIFACT_KEYS),
            "tensor_inventory": {key: _shape_dtype(value) for key, value in arrays.items()},
            "packed_layout": _layout_contract(layout, timestep), "scheduler_identity": "MiniMaxH3MultimodalScheduler",
            "scheduler_configuration": scheduler.configuration(), "prediction_parameterization": "velocity",
            "inference_step_count": scheduler.num_inference_steps,
            "selected_step_index": transition["selected_step_index"],
            "video_current_timestep": transition["video_current_timestep"],
            "video_next_timestep": transition["video_next_timestep"],
            "video_current_sigma": transition["video_current_sigma"], "video_next_sigma": transition["video_next_sigma"],
            "audio_current_timestep": transition["audio_current_timestep"],
            "audio_next_timestep": transition["audio_next_timestep"],
            "audio_current_sigma": transition["audio_current_sigma"], "audio_next_sigma": transition["audio_next_sigma"],
            "timestep_row_convention": {"text": "video_current_timestep", "conditioning_video": "0.999",
                                         "conditioning_audio": "0.999", "target_video": "video_current_timestep",
                                         "target_audio": "audio_current_timestep"},
            "model_input_scaling": "identity", "update_method": scheduler.update_method,
            "configured_resident_block_count": int(len(dit.blocks)), "observed_resident_block_count": len(observed),
            "observed_resident_block_indices": list(observed), "transformer_construction_mode": dit.construction_mode}


def cmd_create_reference(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.load import load_dit
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    artifact, metadata_path = Path(args.artifact).resolve(), Path(args.metadata).resolve()
    dit = arrays = None
    failure: BaseException | None = None
    started = time.perf_counter()
    before = _memory_snapshot(mx)
    try:
        scheduler = _build_canonical_scheduler()
        transition = vars(scheduler.transition(args.step_index))
        if args.step_index != 0:
            raise ValueError("v0.4a reference creation requires selected step index 0")
        dit = load_dit(original, verbose=True)
        _validate_model(dit, "resident")
        layout, timestep, timestep_indices, video, audio, text = _build_layout_and_inputs(mx, dit, transition)
        arrays = {"original_video_latent": video, "original_audio_latent": audio, "text_input": text,
                  "timestep": timestep, "timestep_indices": timestep_indices, "token_tags": layout.token_tags,
                  "position_ids": layout.position_ids, "video_indices": layout.video_indices,
                  "audio_indices": layout.audio_indices, "text_indices": layout.text_indices}
        result, execution = _run_one_step(mx, dit, scheduler, arrays, layout, timestep, args.step_index)
        arrays.update({"resident_video_prediction": result.video_prediction, "resident_audio_prediction": result.audio_prediction,
                       "resident_updated_video_latent": result.updated_video_latent, "resident_updated_audio_latent": result.updated_audio_latent,
                       "selected_step_index": mx.array([args.step_index], dtype=mx.int32),
                       "video_current_timestep": mx.array([transition["video_current_timestep"]], dtype=mx.float32),
                       "video_next_timestep": mx.array([transition["video_next_timestep"]], dtype=mx.float32),
                       "video_current_sigma": mx.array([transition["video_current_sigma"]], dtype=mx.float32),
                       "video_next_sigma": mx.array([transition["video_next_sigma"]], dtype=mx.float32),
                       "audio_current_timestep": mx.array([transition["audio_current_timestep"]], dtype=mx.float32),
                       "audio_next_timestep": mx.array([transition["audio_next_timestep"]], dtype=mx.float32),
                       "audio_current_sigma": mx.array([transition["audio_current_sigma"]], dtype=mx.float32),
                       "audio_next_sigma": mx.array([transition["audio_next_sigma"]], dtype=mx.float32)})
        validate_serialized_transition(mx, arrays, artifact_transition_values(transition), transition)
        metadata = _metadata(original=original, derived=derived, scheduler=scheduler, transition=transition, dit=dit,
                             layout=layout, timestep=timestep, arrays=arrays, observed=execution["observed_transformer_block_indices"])
        _write_artifact(mx, artifact, metadata_path, arrays, metadata, args.overwrite)
        print(f"reference_artifact={artifact}", flush=True)
        print(f"reference_metadata={metadata_path}", flush=True)
        print(f"schedule={json.dumps(scheduler.configuration(), sort_keys=True)}", flush=True)
        print(f"transition={json.dumps(transition, sort_keys=True)}", flush=True)
        print(f"block_execution={json.dumps(execution, sort_keys=True)}", flush=True)
        emit_phase("resident_one_step_reference", started, before, _memory_snapshot(mx),
                   outputs={"artifact": str(artifact), "metadata": str(metadata_path)}, block_execution=execution)
    except BaseException as exc:
        failure = _detach(exc)
    arrays = layout = timestep = timestep_indices = video = audio = text = result = scheduler = None
    dit = None
    release = _release(mx)
    emit_phase("resident_reference_release", started, before, _memory_snapshot(mx), release_result=release,
               artifacts=existing_artifact_paths(artifact, metadata_path))
    if failure is not None:
        raise failure.with_traceback(None)
    print("ONE-STEP RESIDENT REFERENCE CREATED", flush=True)
    return 0


def _read_metadata(path: Path) -> dict[str, Any]:
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"reference metadata is not valid JSON: {path}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("reference metadata must be a JSON object")
    validate_metadata_keys(metadata)
    return metadata


def cmd_compare_derived(args: argparse.Namespace) -> int:
    import mlx.core as mx
    from minimax_h3_mlx.config import DiTConfig
    from minimax_h3_mlx.load import load_dit
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    artifact, metadata_path, report_path = Path(args.artifact).resolve(), Path(args.metadata).resolve(), Path(args.report).resolve()
    dit = arrays = cache = resident_combined = derived_combined = None
    report: dict[str, Any] = {"artifact": str(artifact), "reference_metadata": str(metadata_path), "status": "failed"}
    failure: BaseException | None = None
    parity_validated = False
    metrics: dict[str, Mapping[str, Any]] | None = None
    execution: Mapping[str, Any] | None = None
    started = time.perf_counter()
    before = _memory_snapshot(mx)
    try:
        metadata = _read_metadata(metadata_path)
        if metadata["artifact_sha256"] != sha256_file(artifact):
            raise ValueError("reference artifact checksum does not match metadata")
        expected_checksums = {"artifact_sha256": sha256_file(artifact),
                              "resident_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
                              "resident_config_sha256": sha256_file(original / "config.json"),
                              "derived_config_sha256": sha256_file(derived / "config.json"),
                              "conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
                              "sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json")}
        # Validate all file/checkpoint bindings before opening the derived transformer.
        if any(metadata[key] != value for key, value in expected_checksums.items()):
            raise ValueError("reference metadata checkpoint or artifact checksum binding failed")
        derived_config = DiTConfig.from_json(derived / "config.json")
        arrays = _load_reference_arrays(mx, artifact)
        inventory = {key: _shape_dtype(value) for key, value in arrays.items()}
        scheduler = _build_canonical_scheduler(int(metadata["inference_step_count"]))
        transition = vars(scheduler.transition(int(metadata["selected_step_index"])))
        validate_serialized_transition(mx, arrays, metadata, transition)
        expected_layout = {"sequence_length": int(arrays["token_tags"].shape[0]),
                           "text_token_count": int(arrays["text_indices"].shape[0]),
                           "video_token_count": int(arrays["video_indices"].shape[0]),
                           "audio_token_count": int(arrays["audio_indices"].shape[0]),
                           "video_shape": [1, 1, 96], "audio_shape": [1, 2, 32],
                           "text_shape": [1, 1, 5120],
                           "timestep_values": [float(value) for value in arrays["timestep"].tolist()],
                           "timestep_dtype": _dtype_name(arrays["timestep"].dtype)}
        validate_reference_metadata(metadata, original=original, derived=derived, expected_checksums=expected_checksums,
                                    expected_inventory=inventory, expected_layout=expected_layout, expected_transition=transition)
        for key, expected in metadata["tensor_inventory"].items():
            if _shape_dtype(arrays[key]) != expected:
                raise ValueError(f"serialized tensor inventory mismatch for {key}")
            _finite(mx, arrays[key], f"reference {key}")
        # Reuse the serialized layout/input tensors directly; no second deterministic formula is used.
        layout = SimpleNamespace(sequence_length=int(expected_layout["sequence_length"]),
                                 token_tags=arrays["token_tags"], position_ids=arrays["position_ids"],
                                 video_indices=arrays["video_indices"], audio_indices=arrays["audio_indices"],
                                 text_indices=arrays["text_indices"], num_condition_video_rows=0,
                                 num_condition_audio_rows=0)
        dit = load_dit(derived, verbose=True)
        _validate_model(dit, "cache_only")
        from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache
        sidecars: list[str] = []
        cache, stats = build_streamed_modulation_cache(dit, arrays["timestep"], dtype=mx.bfloat16,
                                                       telemetry=lambda event, details: sidecars.append(str(details["path"])) if event == "sidecar_opening" else None)
        validate_complete_cache_stats(cache_table_count=len(cache.tables), configured_block_count=len(dit.blocks),
                                      stats=stats, actual_sidecar_names=[Path(value).name for value in sidecars])
        result, execution = _run_one_step(mx, dit, scheduler, arrays, layout, arrays["timestep"], int(metadata["selected_step_index"]), cache)
        metrics = {"video_prediction": _metric_report(mx, arrays["resident_video_prediction"], result.video_prediction),
                   "audio_prediction": _metric_report(mx, arrays["resident_audio_prediction"], result.audio_prediction),
                   "updated_video_latent": _metric_report(mx, arrays["resident_updated_video_latent"], result.updated_video_latent),
                   "updated_audio_latent": _metric_report(mx, arrays["resident_updated_audio_latent"], result.updated_audio_latent)}
        report.update({"status": "parity-evaluated", "artifact_format": ARTIFACT_FORMAT,
                       "video_prediction": metrics["video_prediction"], "audio_prediction": metrics["audio_prediction"],
                       "updated_video_latent": metrics["updated_video_latent"], "updated_audio_latent": metrics["updated_audio_latent"],
                       "execution_observation": execution,
                       "cache_lifecycle": {key: value for key, value in vars(stats).items() if key != "per_block"},
                       "checkpoint_open_count": "derived transformer loaded", "sidecar_open_count": stats.sidecar_files_opened})
        validate_combined_exact_parity(metrics)
        parity_validated = True
    except BaseException as exc:
        failure = _detach(exc)
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    # The model-work try is over before any owned MLX reference is released.  Success is not
    # observable until this cleanup and the final report write have both completed.
    arrays = layout = timestep = timestep_indices = video = audio = text = result = scheduler = None
    cache = None
    dit = None
    resident_combined = derived_combined = metrics = execution = None
    release = _release(mx)
    report["status"] = "passed" if parity_validated else "failed"
    report["parity_validated"] = parity_validated
    report["memory_release"] = release
    write_diagnostic_report(report_path, report)
    emit_phase("derived_one_step_release", started, before, _memory_snapshot(mx), release_result=release,
               artifacts=existing_artifact_paths(artifact, metadata_path, report_path),
               parity_validated=parity_validated)
    if failure is not None:
        raise failure.with_traceback(None)
    print(f"parity_report={report_path}", flush=True)
    emit_parity_success_message(parity_validated)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-reference", help="create one resident production denoising step")
    create.add_argument("--original", default=DEFAULT_ORIGINAL)
    create.add_argument("--derived", default=DEFAULT_DERIVED)
    create.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    create.add_argument("--metadata", default=str(DEFAULT_METADATA))
    create.add_argument("--step-index", type=int, default=0)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=cmd_create_reference)
    compare = subparsers.add_parser("compare-derived", help="compare one derived cache-only production step")
    compare.add_argument("--original", default=DEFAULT_ORIGINAL)
    compare.add_argument("--derived", default=DEFAULT_DERIVED)
    compare.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    compare.add_argument("--metadata", default=str(DEFAULT_METADATA))
    compare.add_argument("--report", default=str(DEFAULT_REPORT))
    compare.set_defaults(func=cmd_compare_derived)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

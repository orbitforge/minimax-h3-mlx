"""v0.4b resident-versus-derived parity for the canonical two-step loop.

The two subcommands deliberately run in separate processes.  This module is MLX-free until a
subcommand actually enters its model-runtime function, which keeps its contract tests safe on
machines without Metal or the large checkpoints.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import json
import math
from pathlib import Path
import pprint
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_ORIGINAL = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit"
DEFAULT_DERIVED = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_ARTIFACT = ROOT / "out" / "v0.4b" / "multistep-reference.safetensors"
DEFAULT_METADATA = ROOT / "out" / "v0.4b" / "multistep-reference.json"
DEFAULT_REPORT = ROOT / "out" / "v0.4b" / "multistep-parity-report.json"

ARTIFACT_FORMAT = "minimax-h3-mlx-v04b-multistep-denoising"
ARTIFACT_SCHEMA_VERSION = 2
FINGERPRINT_METHOD = "sha256(index-json)+sha256(full-content-of-every-indexed-resident-shard)-v1"
DETERMINISTIC_INPUT_METHOD = "tensor-index-pattern-v1"
EXPECTED_BLOCK_COUNT = 50
CANONICAL_TRANSITION_COUNT = 2
CANONICAL_STEP_INDICES = (0, 1)
ALLCLOSE_ATOL = 1.0e-5
ALLCLOSE_RTOL = 1.0e-5
RELATIVE_DENOMINATOR_FLOOR = 1.0e-6
RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES = 1 * 1024 * 1024

TRANSITION_FIELDS = (
    "selected_step_index", "video_current_timestep", "video_next_timestep",
    "video_current_sigma", "video_next_sigma", "audio_current_timestep",
    "audio_next_timestep", "audio_current_sigma", "audio_next_sigma",
)


def _transition_tensor_key(step: int, field: str) -> str:
    return f"step_{step}_{field}"


ARTIFACT_KEYS = (
    "initial_video_latent", "initial_audio_latent", "text_input", "token_tags", "position_ids",
    "video_indices", "audio_indices", "text_indices", "step_0_timestep", "step_0_timestep_indices",
    "step_1_timestep", "step_1_timestep_indices",
    *(_transition_tensor_key(step, field) for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS),
    "step_0_video_prediction", "step_0_audio_prediction", "step_0_updated_video_latent",
    "step_0_updated_audio_latent", "step_1_video_prediction", "step_1_audio_prediction",
    "step_1_updated_video_latent", "step_1_updated_audio_latent",
    "final_video_latent", "final_audio_latent",
)

PARITY_COMPARISONS = (
    "step_0_video_prediction", "step_0_audio_prediction", "step_0_updated_video_latent",
    "step_0_updated_audio_latent", "step_1_video_prediction", "step_1_audio_prediction",
    "final_video_latent", "final_audio_latent",
)

REFERENCE_METADATA_REQUIRED_KEYS = frozenset({
    "artifact_format", "artifact_schema_version", "artifact_file_format", "reference_checkpoint",
    "derived_checkpoint", "resident_checkpoint_fingerprint", "fingerprint_method",
    "reference_config_sha256", "derived_config_sha256", "derived_conversion_manifest_sha256",
    "derived_sidecar_manifest_sha256", "artifact_sha256", "deterministic_input_method",
    "deterministic_input_spec", "tensor_keys", "tensor_inventory", "packed_layout",
    "scheduler_identity", "scheduler_configuration", "prediction_parameterization", "input_scaling",
    "update_method", "transition_count", "selected_step_indices", "transitions", "timestep_row_convention",
    "configured_transformer_block_count", "observed_transformer_block_counts",
    "observed_transformer_block_indices", "expected_cache_construction_count", "parity_comparisons",
    "transition_tensor_keys",
})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create-reference", help="create the resident two-step reference")
    _add_common_arguments(create)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=cmd_create_reference)
    compare = subparsers.add_parser("compare-derived", help="compare the derived two-step loop")
    _add_common_arguments(compare)
    compare.add_argument("--report", default=str(DEFAULT_REPORT))
    compare.set_defaults(func=cmd_compare_derived)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--derived", default=DEFAULT_DERIVED)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))


def deterministic_input_parameters() -> dict[str, Any]:
    return {"modulus": 23, "offset": 1.0, "scale_base": 0.001, "scale_step": 0.0001,
            "allowed_salts": (0, 1, 2)}


def deterministic_input_specification() -> dict[str, Any]:
    value = deterministic_input_parameters()
    return {"method": DETERMINISTIC_INPUT_METHOD, **value, "allowed_salts": list(value["allowed_salts"])}


def deterministic_input_values(element_count: int, salt: int) -> tuple[float, ...]:
    if not isinstance(element_count, int) or isinstance(element_count, bool) or element_count <= 0:
        raise ValueError("deterministic input element count must be strictly positive")
    parameters = deterministic_input_parameters()
    if salt not in parameters["allowed_salts"]:
        raise ValueError("deterministic input salt must be one of [0, 1, 2]")
    scale = parameters["scale_base"] + salt * parameters["scale_step"]
    return tuple((((i + salt + parameters["offset"]) % parameters["modulus"]) + parameters["offset"]) * scale
                 for i in range(element_count))


def validate_canonical_schedule(transitions: Sequence[Mapping[str, Any]]) -> None:
    if len(transitions) != CANONICAL_TRANSITION_COUNT:
        raise ValueError("canonical loop must contain exactly two transitions")
    normalized = tuple(_transition_mapping(item) for item in transitions)
    if [int(item["selected_step_index"]) for item in normalized] != list(CANONICAL_STEP_INDICES):
        raise ValueError("canonical loop step order must be exactly [0, 1]")
    first, terminal = normalized
    for prefix in ("video", "audio"):
        if float(first[f"{prefix}_current_sigma"]) != 1.0:
            raise ValueError(f"step 0 {prefix} current sigma must be 1.0")
        if not 0.0 < float(first[f"{prefix}_next_sigma"]) < 1.0:
            raise ValueError(f"step 0 {prefix} next sigma must be strictly between zero and one")
        if float(terminal[f"{prefix}_next_sigma"]) != 0.0:
            raise ValueError(f"step 1 {prefix} must be terminal")
        if not 0.0 < float(terminal[f"{prefix}_current_sigma"]) < 1.0:
            raise ValueError(f"step 1 {prefix} current sigma must be nonzero")


def validate_step_receipts(receipts: Sequence[Any]) -> None:
    if len(receipts) != CANONICAL_TRANSITION_COUNT:
        raise ValueError("loop must return exactly two step receipts")
    if [receipt.step_index for receipt in receipts] != list(CANONICAL_STEP_INDICES):
        raise ValueError("step receipts must be ordered [0, 1]")
    if not _exact_equal(receipts[1].input_video_latent, receipts[0].updated_video_latent):
        raise ValueError("step-1 video input does not equal step-0 updated video latent")
    if not _exact_equal(receipts[1].input_audio_latent, receipts[0].updated_audio_latent):
        raise ValueError("step-1 audio input does not equal step-0 updated audio latent")
    validate_canonical_schedule([vars(receipt) for receipt in receipts])


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": [int(item) for item in value.shape], "dtype": _dtype_name(value.dtype)}


def _mlx_core_for(value: Any) -> Any | None:
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


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _mlx_eval(mx: Any, *values: Any) -> None:
    evaluate = getattr(mx, "eval", None)
    if callable(evaluate):
        evaluate(*values)


def _exact_equal(left: Any, right: Any) -> bool:
    mx = _mlx_core_for(left) or _mlx_core_for(right)
    if mx is not None:
        result = mx.all(left == right)
        _mlx_eval(mx, result)
        return bool(_scalar(result))
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception as exc:
        raise ValueError("exact equality validation failed") from exc


def metric_report(resident: Any, derived: Any) -> dict[str, Any]:
    if tuple(resident.shape) != tuple(derived.shape) or _dtype_name(resident.dtype) != _dtype_name(derived.dtype):
        raise ValueError(f"parity shape/dtype mismatch: {_shape_dtype(resident)} vs {_shape_dtype(derived)}")
    mx = _mlx_core_for(resident) or _mlx_core_for(derived)
    if mx is not None:
        exact = mx.all(resident == derived)
        mismatch = mx.sum(resident != derived)
        left32 = mx.array(resident, dtype=mx.float32)
        right32 = mx.array(derived, dtype=mx.float32)
        difference = mx.abs(left32 - right32)
        denominator = mx.maximum(mx.abs(left32), RELATIVE_DENOMINATOR_FLOOR)
        mse = mx.mean(mx.square(left32 - right32))
        allclose = mx.all(difference <= (ALLCLOSE_ATOL + ALLCLOSE_RTOL * mx.abs(right32)))
        values = (exact, mismatch, mx.max(difference), mx.mean(difference), mx.sqrt(mse),
                  mx.max(difference / denominator), allclose)
        _mlx_eval(mx, *values)
        scalar_values = [_scalar(value) for value in values]
        return {
            "exact_equality": bool(scalar_values[0]),
            "mismatched_element_count_exact": int(scalar_values[1]),
            "maximum_absolute_difference": float(scalar_values[2]),
            "mean_absolute_difference": float(scalar_values[3]),
            "root_mean_square_difference": float(scalar_values[4]),
            "maximum_relative_difference": float(scalar_values[5]),
            "allclose": bool(scalar_values[6]),
            "shape": [int(item) for item in resident.shape], "dtype": _dtype_name(resident.dtype),
            "element_count": int(math.prod(resident.shape)),
        }
    left, right = np.asarray(resident), np.asarray(derived)
    left32, right32 = left.astype(np.float32), right.astype(np.float32)
    difference = np.abs(left32 - right32)
    denominator = np.maximum(np.abs(left32), RELATIVE_DENOMINATOR_FLOOR)
    return {
        "exact_equality": bool(np.array_equal(left, right)),
        "mismatched_element_count_exact": int(np.count_nonzero(left != right)),
        "maximum_absolute_difference": float(np.max(difference)),
        "mean_absolute_difference": float(np.mean(difference)),
        "root_mean_square_difference": float(np.sqrt(np.mean(np.square(left32 - right32)))),
        "maximum_relative_difference": float(np.max(difference / denominator)),
        "allclose": bool(np.allclose(left32, right32, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL)),
        "shape": [int(item) for item in left.shape], "dtype": _dtype_name(resident.dtype),
        "element_count": int(math.prod(left.shape)),
    }


def validate_exact_parity(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    if set(metrics) != set(PARITY_COMPARISONS):
        raise ValueError("parity report comparison set is incomplete")
    failures = [name for name in PARITY_COMPARISONS if not bool(metrics[name]["exact_equality"])]
    if failures:
        raise ValueError(f"exact parity failed for {failures}")


def validate_report_before_parity(report_path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    if not report_path.is_file():
        raise ValueError("diagnostic report must exist before parity validation")
    validate_exact_parity(metrics)


def validate_artifact_tensor_keys(keys: Any) -> None:
    """Validate exact canonical tensor membership and iteration order."""
    actual = list(keys.keys()) if isinstance(keys, Mapping) else list(keys)
    if actual != list(ARTIFACT_KEYS):
        missing = [key for key in ARTIFACT_KEYS if key not in actual]
        extra = [key for key in actual if key not in ARTIFACT_KEYS]
        raise ValueError(
            f"artifact tensor key order or membership mismatch: missing={missing}, extra={extra}, keys={actual!r}"
        )


def validate_loaded_artifact_tensor_membership(keys: Any) -> None:
    """Validate only the tensor-key set returned by a loaded artifact."""
    actual = list(keys.keys()) if isinstance(keys, Mapping) else list(keys)
    expected = set(ARTIFACT_KEYS)
    actual_set = set(actual)
    missing = [key for key in ARTIFACT_KEYS if key not in actual_set]
    extra = sorted(key for key in actual_set if key not in expected)
    if missing or extra:
        raise ValueError(
            f"loaded artifact tensor key membership mismatch: missing={missing}, extra={extra}"
        )


def _canonicalize_loaded_artifact_arrays(loaded_arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize a safetensors load whose mapping iteration order is unspecified."""
    validate_loaded_artifact_tensor_membership(loaded_arrays)
    arrays = {key: loaded_arrays[key] for key in ARTIFACT_KEYS}
    validate_artifact_tensor_keys(arrays)
    return arrays


def validate_artifact_tensor_inventory(inventory: Mapping[str, Any]) -> None:
    expected = _canonical_inventory()
    actual_keys = list(inventory)
    expected_keys = list(expected)
    missing = [key for key in expected_keys if key not in inventory]
    extra = sorted(key for key in actual_keys if key not in expected)
    shape_mismatches = []
    dtype_mismatches = []
    for key in expected_keys:
        if key not in inventory:
            continue
        actual_entry = inventory[key]
        expected_entry = expected[key]
        actual_shape = actual_entry.get("shape") if isinstance(actual_entry, Mapping) else "<missing>"
        expected_shape = expected_entry["shape"]
        if actual_shape != expected_shape:
            shape_mismatches.append((key, actual_shape, expected_shape))
        actual_dtype = actual_entry.get("dtype") if isinstance(actual_entry, Mapping) else "<missing>"
        expected_dtype = expected_entry["dtype"]
        if actual_dtype != expected_dtype:
            dtype_mismatches.append((key, actual_dtype, expected_dtype))

    order_mismatch = actual_keys != expected_keys
    if not (missing or extra or shape_mismatches or dtype_mismatches or order_mismatch):
        return

    ordered_actual = {
        key: inventory[key]
        for key in [*([key for key in expected_keys if key in inventory]), *extra]
    }
    lines = ["artifact tensor inventory mismatch (canonical 40-tensor schema):"]
    lines.append(f"missing keys: {missing!r}")
    lines.append(f"extra keys: {extra!r}")
    if shape_mismatches:
        lines.append("shape mismatches:")
        lines.extend(f"  {key}: actual={actual!r}, expected={expected_value!r}"
                     for key, actual, expected_value in shape_mismatches)
    else:
        lines.append("shape mismatches: []")
    if dtype_mismatches:
        lines.append("dtype mismatches:")
        lines.extend(f"  {key}: actual={actual!r}, expected={expected_value!r}"
                     for key, actual, expected_value in dtype_mismatches)
    else:
        lines.append("dtype mismatches: []")
    if order_mismatch:
        lines.append(f"key order: actual={actual_keys!r}, expected={expected_keys!r}")
    lines.append("actual inventory:")
    lines.append(pprint.pformat(ordered_actual, sort_dicts=False))
    lines.append("expected inventory:")
    lines.append(pprint.pformat(expected, sort_dicts=False))
    raise ValueError("\n".join(lines))


def validate_metadata_keys(metadata: Mapping[str, Any]) -> None:
    missing = sorted(REFERENCE_METADATA_REQUIRED_KEYS - set(metadata))
    unexpected = sorted(set(metadata) - REFERENCE_METADATA_REQUIRED_KEYS)
    if missing or unexpected:
        raise ValueError(f"metadata key contract mismatch: missing={missing}, unexpected={unexpected}")


def validate_reference_metadata(metadata: Mapping[str, Any], *, original: Path, derived: Path,
                                artifact: Path, inventory: Mapping[str, Any]) -> None:
    validate_metadata_keys(metadata)
    expected = {
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors", "reference_checkpoint": str(original.resolve()),
        "derived_checkpoint": str(derived.resolve()), "fingerprint_method": FINGERPRINT_METHOD,
        "deterministic_input_method": DETERMINISTIC_INPUT_METHOD, "tensor_keys": list(ARTIFACT_KEYS),
        "scheduler_identity": "MiniMaxH3MultimodalScheduler", "prediction_parameterization": "velocity",
        "input_scaling": "identity", "update_method": "rectified-flow-euler-data-ward-velocity-v1",
        "transition_count": 2, "selected_step_indices": [0, 1],
        "timestep_row_convention": {"text": "video_current_timestep", "conditioning_video": "0.999",
                                     "conditioning_audio": "0.999", "target_video": "video_current_timestep",
                                     "target_audio": "audio_current_timestep"},
        "configured_transformer_block_count": EXPECTED_BLOCK_COUNT,
        "observed_transformer_block_counts": [EXPECTED_BLOCK_COUNT, EXPECTED_BLOCK_COUNT],
        "observed_transformer_block_indices": [list(range(EXPECTED_BLOCK_COUNT)), list(range(EXPECTED_BLOCK_COUNT))],
        "expected_cache_construction_count": 2, "parity_comparisons": list(PARITY_COMPARISONS),
        "transition_tensor_keys": [_transition_tensor_key(step, field)
                                    for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"reference metadata mismatch for {key}")
    if metadata.get("deterministic_input_spec") != deterministic_input_specification():
        raise ValueError("reference deterministic input specification mismatch")
    expected_configuration = {
        "identity": "MiniMaxH3MultimodalScheduler",
        "video": {"identity": "MiniMaxH3Scheduler", "shift": 12.0},
        "audio": {"identity": "MiniMaxH3Scheduler", "shift": 3.0},
        "num_inference_steps": 2, "prediction_parameterization": "velocity", "input_scaling": "identity",
        "update_method": "rectified-flow-euler-data-ward-velocity-v1",
    }
    if metadata.get("scheduler_configuration") != expected_configuration:
        raise ValueError("reference scheduler configuration mismatch")
    expected_layout = {"sequence_length": 4, "text_token_count": 1, "video_token_count": 1,
                       "audio_token_count": 2, "video_shape": [1, 1, 96], "audio_shape": [1, 2, 32],
                       "text_shape": [1, 1, 5120]}
    if metadata.get("packed_layout") != expected_layout:
        raise ValueError("reference packed-layout contract mismatch")
    validate_canonical_schedule(metadata["transitions"])
    validate_artifact_tensor_inventory(inventory)
    bindings = _checkpoint_bindings(original, derived, artifact)
    for key, value in bindings.items():
        if metadata.get(key) != value:
            raise ValueError(f"reference metadata binding mismatch for {key}")


def _canonical_inventory() -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {
        "initial_video_latent": {"shape": [1, 1, 96], "dtype": "bfloat16"},
        "initial_audio_latent": {"shape": [1, 2, 32], "dtype": "bfloat16"},
        "text_input": {"shape": [1, 1, 5120], "dtype": "bfloat16"},
        "token_tags": {"shape": [4], "dtype": "int32"},
        "position_ids": {"shape": [4, 3], "dtype": "float32"},
        "video_indices": {"shape": [1], "dtype": "int32"},
        "audio_indices": {"shape": [2], "dtype": "int32"},
        "text_indices": {"shape": [1], "dtype": "int32"},
        "step_0_timestep": {"shape": [1], "dtype": "float32"},
        "step_0_timestep_indices": {"shape": [4], "dtype": "int32"},
        "step_1_timestep": {"shape": [2], "dtype": "float32"},
        "step_1_timestep_indices": {"shape": [4], "dtype": "int32"},
    }
    for step in CANONICAL_STEP_INDICES:
        inventory.update({
            _transition_tensor_key(step, "selected_step_index"): {"shape": [], "dtype": "int32"},
            **{_transition_tensor_key(step, field): {"shape": [], "dtype": "float32"}
               for field in TRANSITION_FIELDS if field != "selected_step_index"},
        })
    inventory.update({
        "step_0_video_prediction": {"shape": [1, 1, 96], "dtype": "float32"},
        "step_0_audio_prediction": {"shape": [1, 2, 32], "dtype": "float32"},
        "step_0_updated_video_latent": {"shape": [1, 1, 96], "dtype": "bfloat16"},
        "step_0_updated_audio_latent": {"shape": [1, 2, 32], "dtype": "bfloat16"},
        "step_1_video_prediction": {"shape": [1, 1, 96], "dtype": "float32"},
        "step_1_audio_prediction": {"shape": [1, 2, 32], "dtype": "float32"},
        "step_1_updated_video_latent": {"shape": [1, 1, 96], "dtype": "bfloat16"},
        "step_1_updated_audio_latent": {"shape": [1, 2, 32], "dtype": "bfloat16"},
        "final_video_latent": {"shape": [1, 1, 96], "dtype": "bfloat16"},
        "final_audio_latent": {"shape": [1, 2, 32], "dtype": "bfloat16"},
    })
    inventory = {key: inventory[key] for key in ARTIFACT_KEYS}
    validate_artifact_tensor_keys(inventory)
    return inventory


def _transition_mapping(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError("canonical scheduler transition must be a mapping")
    if "selected_step_index" not in value and "step_index" in value:
        value = {**value, "selected_step_index": value["step_index"]}
    required = set(TRANSITION_FIELDS)
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"canonical scheduler transition is missing fields: {missing}")
    return {field: value[field] for field in TRANSITION_FIELDS}


def _float32_value(value: Any) -> float:
    return float(np.float32(value))


def validate_transition_bindings(metadata: Mapping[str, Any], artifact_arrays: Mapping[str, Any],
                                 scheduler: Any) -> tuple[dict[str, Any], ...]:
    canonical = tuple(_transition_mapping(scheduler.transition(step)) for step in CANONICAL_STEP_INDICES)
    validate_canonical_schedule(canonical)
    metadata_transitions = metadata.get("transitions")
    if not isinstance(metadata_transitions, Sequence) or len(metadata_transitions) != len(canonical):
        raise ValueError("metadata must contain both canonical transition records")
    for step, (expected, actual) in enumerate(zip(canonical, metadata_transitions)):
        actual = _transition_mapping(actual)
        if int(actual["selected_step_index"]) != step:
            raise ValueError(f"metadata transition step {step} index mismatch")
        for field in TRANSITION_FIELDS:
            if field == "selected_step_index":
                continue
            if _float32_value(actual[field]) != _float32_value(expected[field]):
                raise ValueError(f"metadata transition mismatch at step {step}: {field}")
        selected = artifact_arrays[_transition_tensor_key(step, "selected_step_index")]
        if _dtype_name(selected.dtype) != "int32" or tuple(selected.shape) != () or int(_scalar(selected)) != step:
            raise ValueError(f"serialized step {step} selected index is not canonical")
        for field in TRANSITION_FIELDS:
            if field == "selected_step_index":
                continue
            serialized = artifact_arrays[_transition_tensor_key(step, field)]
            if _dtype_name(serialized.dtype) != "float32" or tuple(serialized.shape) != ():
                raise ValueError(f"serialized transition field {step}:{field} is not a float32 scalar")
            mx = _mlx_core_for(serialized)
            if mx is not None:
                expected_value = mx.array(np.float32(expected[field]), dtype=mx.float32)
                if not _exact_equal(serialized, expected_value):
                    raise ValueError(f"serialized transition mismatch at step {step}: {field}")
            elif float(serialized) != _float32_value(expected[field]):
                raise ValueError(f"serialized transition mismatch at step {step}: {field}")
    return canonical


def validate_timestep_reconstruction(artifact_arrays: Mapping[str, Any], layout: Any,
                                     canonical: Sequence[Mapping[str, Any]],
                                     rebuilt_timesteps: Mapping[int, Any],
                                     rebuilt_indices: Mapping[int, Any]) -> None:
    for step in CANONICAL_STEP_INDICES:
        serialized_timestep = artifact_arrays[f"step_{step}_timestep"]
        serialized_indices = artifact_arrays[f"step_{step}_timestep_indices"]
        if not _exact_equal(serialized_timestep, rebuilt_timesteps[step]):
            raise ValueError(f"serialized step {step} timestep does not match canonical reconstruction")
        if not _exact_equal(serialized_indices, rebuilt_indices[step]):
            raise ValueError(f"serialized step {step} timestep indices do not match canonical reconstruction")
        expected_timestep_shape = tuple(rebuilt_timesteps[step].shape)
        if tuple(serialized_timestep.shape) != expected_timestep_shape or tuple(serialized_indices.shape) != (4,):
            raise ValueError(f"serialized step {step} packed timetable shape is not canonical")
        # The inverse indices encode row semantics.  Rebuilding the layout is the authoritative
        # check that each row points to the correct modality/conditioning timestep.
        expected_rows = np.full(layout.sequence_length, _float32_value(canonical[step]["video_current_timestep"]), dtype=np.float32)
        video_indices = np.asarray(layout.video_indices.tolist(), dtype=np.int64)
        audio_indices = np.asarray(layout.audio_indices.tolist(), dtype=np.int64)
        expected_rows[video_indices[:layout.num_condition_video_rows]] = np.float32(0.999)
        expected_rows[audio_indices[layout.num_condition_audio_rows:]] = np.float32(canonical[step]["audio_current_timestep"])
        expected_rows[audio_indices[:layout.num_condition_audio_rows]] = np.float32(0.999)
        mx = _mlx_core_for(serialized_timestep) or _mlx_core_for(serialized_indices)
        if mx is not None:
            rows = mx.take(serialized_timestep, serialized_indices)
            expected = mx.array(expected_rows, dtype=mx.float32)
            if not _exact_equal(rows, expected):
                raise ValueError(f"serialized step {step} packed timetable row semantics mismatch")
        else:
            rows = np.asarray(serialized_timestep)[np.asarray(serialized_indices)]
            if not np.array_equal(rows, expected_rows):
                raise ValueError(f"serialized step {step} packed timetable row semantics mismatch")
    if _exact_equal(artifact_arrays["step_0_timestep"], artifact_arrays["step_1_timestep"]):
        raise ValueError("serialized canonical timestep tensors must be distinct")


def validate_cache_lifecycle(per_step: Sequence[Mapping[str, Any]]) -> None:
    if len(per_step) != CANONICAL_TRANSITION_COUNT:
        raise ValueError("cache lifecycle must have one record per transition")
    for index, record in enumerate(per_step):
        expected = {"cache_table_count": 50, "blocks_completed": 50, "sidecar_files_opened": 50,
                    "unique_sidecars_opened": 50, "successful_payload_opens": 50,
                    "completed_payload_releases": 50, "every_sidecar_released_before_next_opened": True,
                    "sidecar_overlap_observed": False, "next_sidecar_opened_before_previous_release": False,
                    "dense_temporary_projection_created": False}
        for key, value in expected.items():
            if record.get(key) != value:
                raise ValueError(f"step {index} cache lifecycle violation: {key}={record.get(key)!r}")
        token = record.get("cache_session_token")
        if not isinstance(token, int) or isinstance(token, bool) or token <= 0:
            raise ValueError(f"step {index} cache lifecycle violation: cache_session_token={token!r}")
        if not isinstance(record.get("complete_cache_statistics"), Mapping):
            raise ValueError(f"step {index} cache lifecycle violation: complete cache statistics missing")
        events = record.get("events")
        if not isinstance(events, Sequence) or len(events) != 4:
            raise ValueError(f"step {index} cache lifecycle violation: events={events!r}")
        names = [event.get("event") if isinstance(event, Mapping) else None for event in events]
        if names != ["acquire-start", "acquire-complete", "release-start", "release-complete"]:
            raise ValueError(f"step {index} cache lifecycle violation: events={events!r}")
        token = record.get("cache_session_token")
        step_index = record.get("step_index", index)
        numbers = []
        for event in events:
            if not isinstance(event, Mapping) or event.get("step_index") != step_index \
                    or event.get("cache_session_token") != token:
                raise ValueError(f"step {index} cache lifecycle violation: event identity={event!r}")
            number = event.get("global_event_number")
            if not isinstance(number, int) or isinstance(number, bool):
                raise ValueError(f"step {index} cache lifecycle violation: event number={number!r}")
            numbers.append(number)
        if numbers != list(range(numbers[0], numbers[0] + 4)):
            raise ValueError(f"step {index} cache lifecycle violation: event numbers={numbers!r}")
        expected_names = [f"block-{block:03d}.safetensors" for block in range(50)]
        if record.get("sidecar_names") != expected_names:
            raise ValueError(f"step {index} cache lifecycle sidecar order mismatch")
    if sum(int(item["sidecar_files_opened"]) for item in per_step) != 100:
        raise ValueError("total sidecar opens must be exactly 100")
    if sum(int(item["completed_payload_releases"]) for item in per_step) != 100:
        raise ValueError("total payload releases must be exactly 100")
    tokens = [item["cache_session_token"] for item in per_step]
    if len(set(tokens)) != len(tokens):
        raise ValueError("cache session tokens must be distinct across constructions")
    all_event_numbers = [event["global_event_number"] for record in per_step for event in record["events"]]
    if all_event_numbers != sorted(set(all_event_numbers)):
        raise ValueError("cache lifecycle event numbers must be globally monotonic and unique")
    release_zero = per_step[0]["events"][-1]["global_event_number"]
    acquire_one = per_step[1]["events"][0]["global_event_number"]
    if not isinstance(release_zero, int) or not isinstance(acquire_one, int) or release_zero >= acquire_one:
        raise ValueError("cache lifecycle ordering violation: step-0 release must complete before step-1 acquisition")
    for index, record in enumerate(per_step):
        if record["events"][-1]["global_event_number"] - record["events"][0]["global_event_number"] != 3:
            raise ValueError(f"step {index} cache lifecycle event span is not canonical")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_resident_shard_names(weight_map: Mapping[str, Any]) -> tuple[str, ...]:
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("resident checkpoint index has no weight_map")
    names = list(weight_map.values())
    if not all(isinstance(name, str) for name in names):
        raise ValueError("resident checkpoint index shard names must all be strings")
    unique = tuple(sorted(set(names)))
    if any(not name or Path(name).name != name or name in {".", ".."} for name in unique):
        raise ValueError("resident checkpoint index contains unsafe or empty shard basenames")
    return unique


def resident_checkpoint_fingerprint(checkpoint: Path) -> str:
    index_path = checkpoint / "model.safetensors.index.json"
    payload = json.loads(index_path.read_text())
    names = validate_resident_shard_names(payload.get("weight_map"))
    shards = []
    for name in names:
        path = checkpoint / name
        if not path.is_file():
            raise FileNotFoundError(f"resident checkpoint index references missing shard: {path}")
        shards.append({"filename": name, "sha256": sha256_file(path)})
    binding = {"method": FINGERPRINT_METHOD, "index_sha256": sha256_file(index_path), "shards": shards}
    return hashlib.sha256(json.dumps(binding, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _detach(exc: BaseException) -> BaseException:
    exc.__traceback__ = None
    exc.__context__ = None
    exc.__cause__ = None
    return exc


def _receipt_evidence(receipts: Sequence[Any]) -> list[dict[str, Any]]:
    evidence = []
    for receipt in receipts:
        evidence.append({
            "step_index": receipt.step_index,
            "video_current_timestep": receipt.video_current_timestep,
            "video_next_timestep": receipt.video_next_timestep,
            "video_current_sigma": receipt.video_current_sigma,
            "video_next_sigma": receipt.video_next_sigma,
            "audio_current_timestep": receipt.audio_current_timestep,
            "audio_next_timestep": receipt.audio_next_timestep,
            "audio_current_sigma": receipt.audio_current_sigma,
            "audio_next_sigma": receipt.audio_next_sigma,
            "video_prediction": _shape_dtype(receipt.video_prediction),
            "audio_prediction": _shape_dtype(receipt.audio_prediction),
            "updated_video_latent": _shape_dtype(receipt.updated_video_latent),
            "updated_audio_latent": _shape_dtype(receipt.updated_audio_latent),
        })
    return evidence


def _complete_stats(stats: Any) -> dict[str, Any]:
    if is_dataclass(stats):
        return asdict(stats)
    if isinstance(stats, Mapping):
        return dict(stats)
    if hasattr(stats, "__dict__"):
        return dict(vars(stats))
    raise ValueError("cache statistics must be a dataclass, mapping, or object with fields")


@contextmanager
def observe_transformer_block_execution(dit: Any):
    """Observe each top-level transformer call in its own block-index bucket.

    The proxy is the object passed to denoising.  The cache builder continues to receive ``dit``
    itself, so cache construction never accidentally sees the observer as its model contract.
    """
    original_blocks = list(dit.blocks)
    observed: list[list[int]] = []
    active_call: int | None = None

    class ObservedBlock:
        def __init__(self, index: int, block: Any):
            self.index = index
            self.block = block

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            if active_call is None:
                raise RuntimeError("transformer block executed outside an active top-level call")
            observed[active_call].append(self.index)
            return self.block(*args, **kwargs)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.block, name)

    dit.blocks[:] = [ObservedBlock(index, block) for index, block in enumerate(original_blocks)]

    class TransformerObserver:
        def __init__(self, transformer: Any):
            self._transformer = transformer

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            nonlocal active_call
            if self._transformer is None:
                raise RuntimeError("transformer observer is detached")
            call_index = len(observed)
            observed.append([])
            active_call = call_index
            try:
                return self._transformer(*args, **kwargs)
            finally:
                active_call = None

        @property
        def observations(self) -> list[list[int]]:
            return observed

        def __getattr__(self, name: str) -> Any:
            transformer = self._transformer
            if transformer is None:
                raise RuntimeError("transformer observer is detached")
            return getattr(transformer, name)

    observer = TransformerObserver(dit)
    try:
        yield observer
    finally:
        active_call = None
        dit.blocks[:] = original_blocks
        observer._transformer = None


def validate_per_step_block_observations(observed: Sequence[Sequence[int]]) -> None:
    expected = list(range(EXPECTED_BLOCK_COUNT))
    if len(observed) != CANONICAL_TRANSITION_COUNT or any(list(item) != expected for item in observed):
        raise ValueError(f"per-step transformer block observations mismatch: {observed!r}")


def _release_runtime(mx: Any, *, active_memory_baseline: Mapping[str, Any] | int | None = None) -> dict[str, Any]:
    """Only collect, purge, and report; caller-owned references are cleared by callers."""
    gc.collect()
    memory_before_purge = _memory_snapshot(mx)
    clear = getattr(mx, "clear_cache", None)
    available = callable(clear)
    if available:
        clear()
    cache_getter = getattr(mx, "get_cache_memory", None)
    cache_after = None
    baseline_active = (active_memory_baseline.get("active")
                       if isinstance(active_memory_baseline, Mapping)
                       else active_memory_baseline)
    memory_after_purge: dict[str, int | None] | None = None
    try:
        if callable(cache_getter):
            cache_after = int(cache_getter())
            if cache_after != 0:
                raise RuntimeError(f"allocator cache was not zero after purge: {cache_after}")
        memory_after_purge = _memory_snapshot(mx)
        active_after = memory_after_purge.get("active")
        active_gate_available = (baseline_active is not None and active_after is not None
                                 and callable(getattr(mx, "get_active_memory", None)))
        if active_gate_available:
            active_limit = int(baseline_active) + RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES
            if int(active_after) > active_limit:
                raise RuntimeError(
                    "active memory after release exceeded baseline: "
                    f"active={active_after}, baseline={baseline_active}, limit={active_limit}"
                )
    except BaseException as exc:
        exc.memory_before_allocator_purge = memory_before_purge
        exc.memory_after_allocator_purge = memory_after_purge or _memory_snapshot(mx)
        exc.release_active_memory_baseline = baseline_active
        exc.release_active_memory_tolerance = RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES
        raise
    assert memory_after_purge is not None
    return {"allocator_cache_purge_available": available, "allocator_cache_purged": available,
            "allocator_cache_after": cache_after, "memory_before_allocator_purge": memory_before_purge,
            "memory_after_allocator_purge": memory_after_purge,
            "active_memory_baseline": baseline_active,
            "active_memory_tolerance_bytes": RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            "active_memory_gate_available": (baseline_active is not None
                                              and memory_after_purge.get("active") is not None
                                              and callable(getattr(mx, "get_active_memory", None)))}


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    snapshot: dict[str, int | None] = {}
    for label, name in (("active", "get_active_memory"), ("allocator_cache", "get_cache_memory"),
                        ("peak", "get_peak_memory")):
        getter = getattr(mx, name, None)
        try:
            snapshot[label] = int(getter()) if callable(getter) else None
        except Exception:
            snapshot[label] = None
    return snapshot


def _build_canonical_scheduler():
    from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler
    video, audio = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(3)
    audio.set_timesteps(3)
    scheduler = MiniMaxH3MultimodalScheduler(video, audio)
    validate_canonical_schedule([vars(scheduler.transition(i)) for i in CANONICAL_STEP_INDICES])
    return scheduler


def _checkpoint_bindings(original: Path, derived: Path, artifact: Path | None = None) -> dict[str, str]:
    values = {
        "resident_checkpoint_fingerprint": resident_checkpoint_fingerprint(original),
        "reference_config_sha256": sha256_file(original / "config.json"),
        "derived_config_sha256": sha256_file(derived / "config.json"),
        "derived_conversion_manifest_sha256": sha256_file(derived / "conversion_manifest.json"),
        "derived_sidecar_manifest_sha256": sha256_file(derived / "adaln" / "manifest.json"),
    }
    if artifact is not None:
        values["artifact_sha256"] = sha256_file(artifact)
    return values


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")


def _load_artifact_arrays(path: Path) -> Mapping[str, Any]:
    import mlx.core as mx
    return _canonicalize_loaded_artifact_arrays(mx.load(str(path)))


def _load_derived_transformer(path: Path) -> Any:
    from minimax_h3_mlx.load import load_dit
    return load_dit(path, verbose=True)


def _canonical_layout_and_timesteps(config: Any, scheduler: Any):
    from minimax_h3_mlx.config import TAG_TEXT
    from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps
    layout = build_packed_sequence(np.array([TAG_TEXT], dtype=np.int64), 1, 2, 2, 1,
                                   tuple(config.patch_size), keyframe_anchors=())
    transitions = [_transition_mapping(scheduler.transition(i)) for i in CANONICAL_STEP_INDICES]
    timesteps = {}
    timestep_indices = {}
    for step, transition in enumerate(transitions):
        timesteps[step], timestep_indices[step] = build_row_timesteps(
            layout, transition["video_current_timestep"], transition["audio_current_timestep"], 0.999, 0.999)
    return layout, transitions, timesteps, timestep_indices


def _runtime_inputs(mx: Any, dit: Any, scheduler: Any):
    layout, transitions, timesteps, timestep_indices = _canonical_layout_and_timesteps(dit.config, scheduler)
    def pattern(shape: tuple[int, ...], salt: int):
        return mx.array(deterministic_input_values(math.prod(shape), salt), dtype=mx.float32).reshape(shape).astype(mx.bfloat16)
    static = {"token_tags": layout.token_tags, "position_ids": layout.position_ids,
              "video_indices": layout.video_indices, "audio_indices": layout.audio_indices,
              "text_indices": layout.text_indices}
    arrays = {"initial_video_latent": pattern((1, 1, 96), 0), "initial_audio_latent": pattern((1, 2, 32), 1),
              "text_input": pattern((1, 1, 5120), 2), **static}
    return layout, transitions, timesteps, timestep_indices, arrays


def _artifact_arrays_from_result(mx: Any, arrays: Mapping[str, Any], timesteps: Mapping[int, Any],
                                 timestep_indices: Mapping[int, Any], transitions: Sequence[Mapping[str, Any]],
                                 result: Any) -> dict[str, Any]:
    first, second = result.step_receipts
    artifact = {**arrays, "step_0_timestep": timesteps[0], "step_0_timestep_indices": timestep_indices[0],
            "step_1_timestep": timesteps[1], "step_1_timestep_indices": timestep_indices[1],
            "step_0_video_prediction": first.video_prediction, "step_0_audio_prediction": first.audio_prediction,
            "step_0_updated_video_latent": first.updated_video_latent, "step_0_updated_audio_latent": first.updated_audio_latent,
            "step_1_video_prediction": second.video_prediction, "step_1_audio_prediction": second.audio_prediction,
            "step_1_updated_video_latent": second.updated_video_latent, "step_1_updated_audio_latent": second.updated_audio_latent,
            "final_video_latent": result.final_video_latent, "final_audio_latent": result.final_audio_latent}
    for step, raw_transition in enumerate(transitions):
        transition = _transition_mapping(raw_transition)
        for field in TRANSITION_FIELDS:
            dtype = mx.int32 if field == "selected_step_index" else mx.float32
            artifact[_transition_tensor_key(step, field)] = mx.array(transition[field], dtype=dtype)
    artifact = {key: artifact[key] for key in ARTIFACT_KEYS}
    validate_artifact_tensor_keys(artifact)
    if not _exact_equal(artifact["step_1_updated_video_latent"], artifact["final_video_latent"]):
        raise ValueError("serialized step-1 video update must equal final video latent")
    if not _exact_equal(artifact["step_1_updated_audio_latent"], artifact["final_audio_latent"]):
        raise ValueError("serialized step-1 audio update must equal final audio latent")
    return artifact


def _existing_artifact_paths(*paths: Path) -> list[str]:
    return [str(path) for path in paths if path.is_file()]


def _loop_receipt_values(result: Any, failure: BaseException | None) -> tuple[int, dict[str, int], Sequence[Any]]:
    if result is not None:
        return result.completed_steps, {
            "transformer_calls": result.transformer_calls,
            "scheduler_updates": result.scheduler_updates,
            "cache_acquisitions": result.cache_acquisitions,
            "cache_releases": result.cache_releases,
        }, result.step_receipts
    partial = getattr(failure, "denoise_step_receipts", ()) if failure is not None else ()
    return getattr(failure, "denoise_completed_steps", len(partial)) if failure is not None else 0, {
        name: getattr(failure, f"denoise_{name}", 0) if failure is not None else 0
        for name in ("transformer_calls", "scheduler_updates", "cache_acquisitions", "cache_releases")
    }, partial


def _run_resident(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    from minimax_h3_mlx.denoise import denoise_loop
    from minimax_h3_mlx.load import load_dit
    artifact_path = Path(args.artifact).resolve()
    metadata_path = Path(args.metadata).resolve()
    receipt: dict[str, Any] = {"phase": "runtime", "status": "failed",
                               "artifacts": _existing_artifact_paths(artifact_path, metadata_path),
                               "artifact_paths": [str(artifact_path), str(metadata_path)]}
    scheduler = dit = layout = transitions = timesteps = timestep_indices = arrays = None
    result = artifact_arrays = metadata = None
    original_blocks = observed_wrappers = receipts = observer = None
    observed_per_step: list[list[int]] = []
    failure: BaseException | None = None
    try:
        receipt["memory_before_runtime"] = _memory_snapshot(mx)
        scheduler = _build_canonical_scheduler()
        dit = load_dit(Path(args.original), verbose=True)
        if getattr(dit, "construction_mode", None) != "resident":
            raise ValueError("resident command did not construct a resident transformer")
        layout, transitions, timesteps, timestep_indices, arrays = _runtime_inputs(mx, dit, scheduler)
        with observe_transformer_block_execution(dit) as observer:
            observed_wrappers = dit.blocks
            result = denoise_loop(observer, scheduler, initial_video_latent=arrays["initial_video_latent"],
                initial_audio_latent=arrays["initial_audio_latent"], text_embedding=arrays["text_input"],
                timestep_provider=lambda step, transition: (timesteps[step], timestep_indices[step]),
                token_tags=arrays["token_tags"], position_ids=arrays["position_ids"], video_indices=arrays["video_indices"],
                audio_indices=arrays["audio_indices"], text_indices=arrays["text_indices"])
        observed_per_step = [list(item) for item in observer.observations]
        validate_per_step_block_observations(observed_per_step)
        validate_step_receipts(result.step_receipts)
        receipts = result.step_receipts
        artifact_arrays = _artifact_arrays_from_result(mx, arrays, timesteps, timestep_indices, transitions, result)
        metadata = _metadata(args, scheduler, dit, layout, transitions, artifact_arrays, observed_per_step)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and (artifact_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"reference artifact already exists: {artifact_path}")
        mx.save_safetensors(str(artifact_path), artifact_arrays)
        metadata["artifact_sha256"] = sha256_file(artifact_path)
        _write_json(metadata_path, metadata)
        receipt.update({"completed_steps": result.completed_steps,
                        "counters": {"transformer_calls": result.transformer_calls,
                                     "scheduler_updates": result.scheduler_updates,
                                     "cache_acquisitions": result.cache_acquisitions,
                                     "cache_releases": result.cache_releases},
                        "observed_transformer_block_indices": observed_per_step})
    except BaseException as exc:
        failure = _detach(exc)
        if observer is not None:
            observed_per_step = [list(item) for item in observer.observations]
        completed_steps, counters, partial = _loop_receipt_values(result, exc)
        receipt.update({"error": {"type": type(exc).__name__, "message": str(exc)},
                        "completed_steps": completed_steps, "counters": counters,
                        "partial_step_evidence": _receipt_evidence(partial) if partial else [],
                        "observed_transformer_block_indices": observed_per_step})
    finally:
        receipt["memory_before_reference_clear"] = _memory_snapshot(mx)
        observer = None
        dit = scheduler = layout = transitions = timesteps = timestep_indices = arrays = None
        result = artifact_arrays = metadata = original_blocks = observed_wrappers = receipts = None
        release_started = time.perf_counter()
        try:
            release_result = _release_runtime(mx, active_memory_baseline=receipt.get("memory_before_runtime"))
            receipt["release"] = {"phase": "release", "elapsed_seconds": round(time.perf_counter() - release_started, 6),
                                  **release_result}
        except BaseException as release_exc:
            receipt["release"] = {"phase": "release", "status": "failed",
                                  "error": {"type": type(release_exc).__name__, "message": str(release_exc)},
                                  "memory_before_allocator_purge": getattr(release_exc, "memory_before_allocator_purge", None),
                                  "memory_after_allocator_purge": getattr(release_exc, "memory_after_allocator_purge", _memory_snapshot(mx))}
            if failure is None:
                failure = _detach(release_exc)
        receipt["artifacts"] = _existing_artifact_paths(artifact_path, metadata_path)
        receipt["status"] = "runtime-complete" if failure is None else "failed"
        print(f"phase_receipt={json.dumps(receipt, sort_keys=True, default=str)}", flush=True)
    if failure is not None:
        raise failure.with_traceback(None)
    return receipt


def _metadata(args: argparse.Namespace, scheduler: Any, dit: Any, layout: Any,
              transitions: Sequence[Mapping[str, Any]], artifact_arrays: Mapping[str, Any],
              observed: Sequence[Sequence[int]]) -> dict[str, Any]:
    validate_artifact_tensor_keys(artifact_arrays)
    inventory = {key: _shape_dtype(value) for key, value in artifact_arrays.items()}
    validate_artifact_tensor_inventory(inventory)
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    bindings = _checkpoint_bindings(original, derived)
    return {"artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_file_format": "safetensors", "reference_checkpoint": str(Path(args.original).resolve()),
            "derived_checkpoint": str(Path(args.derived).resolve()), **bindings,
            "fingerprint_method": FINGERPRINT_METHOD, "artifact_sha256": "pending",
            "deterministic_input_method": DETERMINISTIC_INPUT_METHOD, "deterministic_input_spec": deterministic_input_specification(),
            "tensor_keys": list(ARTIFACT_KEYS), "tensor_inventory": inventory,
            "transition_tensor_keys": [_transition_tensor_key(step, field)
                                       for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS],
            "packed_layout": {"sequence_length": int(layout.sequence_length), "text_token_count": int(layout.text_indices.shape[0]),
                              "video_token_count": int(layout.video_indices.shape[0]), "audio_token_count": int(layout.audio_indices.shape[0]),
                              "video_shape": [1, 1, 96], "audio_shape": [1, 2, 32], "text_shape": [1, 1, 5120]},
            "scheduler_identity": "MiniMaxH3MultimodalScheduler", "scheduler_configuration": scheduler.configuration(),
            "prediction_parameterization": "velocity", "input_scaling": "identity",
            "update_method": scheduler.update_method, "transition_count": len(transitions), "selected_step_indices": list(CANONICAL_STEP_INDICES),
            "transitions": [dict(transition) for transition in transitions],
            "timestep_row_convention": {"text": "video_current_timestep", "conditioning_video": "0.999",
                                         "conditioning_audio": "0.999", "target_video": "video_current_timestep", "target_audio": "audio_current_timestep"},
            "configured_transformer_block_count": len(dit.blocks),
            "observed_transformer_block_counts": [len(item) for item in observed],
            "observed_transformer_block_indices": [list(item) for item in observed],
            "expected_cache_construction_count": len(transitions), "parity_comparisons": list(PARITY_COMPARISONS)}


def cmd_create_reference(args: argparse.Namespace) -> int:
    _run_resident(args)
    print("MULTISTEP RESIDENT REFERENCE CREATED", flush=True)
    return 0


def cmd_compare_derived(args: argparse.Namespace) -> int:
    """Validate the reference, then run the derived loop in the external MLX process."""
    report_path = Path(args.report).resolve()
    artifact_path = Path(args.artifact).resolve()
    metadata_path = Path(args.metadata).resolve()
    report: dict[str, Any] = {"artifact": str(artifact_path), "reference_metadata": str(metadata_path), "status": "failed"}
    failure: BaseException | None = None
    parity_validated = False
    mx = dit = arrays = scheduler = cache_provider = result = None
    layout = transitions = timesteps = timestep_indices = artifact_arrays = metadata = None
    canonical = derived_config = rebuilt_layout = rebuilt_transitions = rebuilt_timesteps = rebuilt_indices = None
    inventory = None
    original_blocks = observed_wrappers = receipts = metrics = lifecycle = None
    observed_per_step: list[list[int]] = []
    observer = None
    memory_before_runtime = None
    try:
        # Everything through this point is reference-only validation.  In particular, no derived
        # transformer object is constructed before the small artifact and all bindings pass.
        metadata = json.loads(metadata_path.read_text())
        validate_metadata_keys(metadata)
        validate_artifact_tensor_keys(metadata["tensor_keys"])
        if metadata["artifact_sha256"] != sha256_file(artifact_path):
            raise ValueError("reference artifact checksum does not match metadata")
        original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
        arrays = _load_artifact_arrays(artifact_path)
        mx = _mlx_core_for(next(iter(arrays.values())))
        if mx is None:
            import mlx.core as mx
        memory_before_runtime = _memory_snapshot(mx)
        validate_artifact_tensor_keys(arrays)
        inventory = {key: _shape_dtype(value) for key, value in arrays.items()}
        if inventory != metadata["tensor_inventory"]:
            raise ValueError("reference artifact tensor inventory mismatch")
        validate_artifact_tensor_inventory(inventory)
        # This performs the resident fingerprint plus all derived config/manifest bindings while
        # the derived transformer is still unloaded.
        validate_reference_metadata(metadata, original=original, derived=derived, artifact=artifact_path, inventory=inventory)
        scheduler = _build_canonical_scheduler()
        canonical = validate_transition_bindings(metadata, arrays, scheduler)
        from minimax_h3_mlx.config import DiTConfig
        derived_config = DiTConfig.from_json(derived / "config.json")
        rebuilt_layout, rebuilt_transitions, rebuilt_timesteps, rebuilt_indices = _canonical_layout_and_timesteps(
            derived_config, scheduler
        )
        validate_timestep_reconstruction(arrays, rebuilt_layout, canonical, rebuilt_timesteps, rebuilt_indices)
        validate_artifact_tensor_keys(arrays)

        # Only after all metadata, transition, timetable, index, and packed-row checks pass may
        # the derived loader construct the cache-only transformer.
        from minimax_h3_mlx.denoise import denoise_loop
        dit = _load_derived_transformer(derived)
        if getattr(dit, "construction_mode", None) != "cache_only":
            raise ValueError("derived command did not construct a cache-only transformer")

        from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache
        class StreamedProvider:
            def __init__(self):
                self.records: list[dict[str, Any]] = []
                self.active = False
                self.event_number = 0
                self.next_session_token = 0

            def _event(self, record: dict[str, Any], name: str) -> None:
                self.event_number += 1
                record["events"].append({"global_event_number": self.event_number,
                                         "step_index": record["step_index"],
                                         "cache_session_token": record["cache_session_token"],
                                         "event": name})

            def cache_for_step(self, step_index, timestep):
                if self.active:
                    raise ValueError("cache overlap between denoising steps")
                self.next_session_token += 1
                record = {"step_index": step_index, "cache_session_token": self.next_session_token,
                          "events": [], "sidecar_names": []}
                self.records.append(record)
                self._event(record, "acquire-start")
                sidecars: list[str] = []
                cache, stats = build_streamed_modulation_cache(
                    dit, timestep, dtype=mx.bfloat16,
                    telemetry=lambda event, details: sidecars.append(str(details["path"])) if event == "sidecar_opening" else None)
                tables = getattr(cache, "tables", None)
                if tables is None:
                    raise ValueError("constructed modulation cache has no tables collection")
                self.last_session_token = record["cache_session_token"]
                record.update({"stats": stats, "cache_table_count": len(tables),
                               "sidecar_names": [Path(path).name for path in sidecars]})
                self.active = True
                self._event(record, "acquire-complete")
                return cache

            def release_step(self, step_index, cache):
                record = self.records[-1]
                self._event(record, "release-start")
                release = getattr(cache, "release", None)
                if callable(release):
                    release()
                self.active = False
                self._event(record, "release-complete")

        cache_provider = StreamedProvider()
        layout = rebuilt_layout
        with observe_transformer_block_execution(dit) as observer:
            result = denoise_loop(observer, scheduler, initial_video_latent=arrays["initial_video_latent"],
                initial_audio_latent=arrays["initial_audio_latent"], text_embedding=arrays["text_input"],
                timestep_provider=lambda step, transition: (arrays[f"step_{step}_timestep"], arrays[f"step_{step}_timestep_indices"]),
                token_tags=layout.token_tags, position_ids=layout.position_ids, video_indices=layout.video_indices,
                audio_indices=layout.audio_indices, text_indices=layout.text_indices, modulation_cache_provider=cache_provider)
        observed_per_step = [list(item) for item in observer.observations]
        validate_per_step_block_observations(observed_per_step)
        validate_step_receipts(result.step_receipts)
        lifecycle = []
        for record in cache_provider.records:
            stats = record["stats"]
            lifecycle.append({"step_index": record["step_index"], "cache_table_count": record["cache_table_count"],
                              "cache_session_token": record["cache_session_token"],
                              "complete_cache_statistics": _complete_stats(stats),
                              "events": record["events"],
                              "blocks_completed": stats.blocks_completed, "sidecar_files_opened": stats.sidecar_files_opened,
                              "unique_sidecars_opened": stats.unique_sidecar_files_opened,
                              "successful_payload_opens": stats.successful_payload_opens,
                              "completed_payload_releases": stats.completed_payload_releases,
                              "every_sidecar_released_before_next_opened": stats.every_sidecar_released_before_next_opened,
                              "sidecar_overlap_observed": stats.sidecar_overlap_observed,
                              "next_sidecar_opened_before_previous_release": stats.next_sidecar_opened_before_previous_release,
                              "dense_temporary_projection_created": stats.dense_temporary_projection_created,
                              "sidecar_names": record["sidecar_names"]})
        validate_cache_lifecycle(lifecycle)
        metrics = {}
        receipts = result.step_receipts
        for name in PARITY_COMPARISONS:
            metrics[name] = metric_report(arrays[name], {
                "step_0_video_prediction": result.step_receipts[0].video_prediction,
                "step_0_audio_prediction": result.step_receipts[0].audio_prediction,
                "step_0_updated_video_latent": result.step_receipts[0].updated_video_latent,
                "step_0_updated_audio_latent": result.step_receipts[0].updated_audio_latent,
                "step_1_video_prediction": result.step_receipts[1].video_prediction,
                "step_1_audio_prediction": result.step_receipts[1].audio_prediction,
                "final_video_latent": result.final_video_latent,
                "final_audio_latent": result.final_audio_latent,
            }[name])
        report.update({"status": "parity-evaluated", "metrics": metrics, "cache_lifecycle": lifecycle,
                       "completed_steps": result.completed_steps,
                       "transformer_calls": result.transformer_calls,
                       "scheduler_updates": result.scheduler_updates,
                       "cache_acquisitions": result.cache_acquisitions,
                       "cache_releases": result.cache_releases,
                       "cache_construction_sessions": len(cache_provider.records),
                       "observed_transformer_block_indices": observed_per_step,
                       "artifacts": {"artifact": str(artifact_path) if artifact_path.is_file() else None,
                                     "metadata": str(metadata_path) if metadata_path.is_file() else None,
                                     "report": str(report_path) if report_path.is_file() else None}})
        _write_json(report_path, report)
        validate_report_before_parity(report_path, metrics)
        parity_validated = True
    except BaseException as exc:
        if observer is not None:
            observed_per_step = [list(item) for item in observer.observations]
        completed_steps, counters, partial = _loop_receipt_values(result, exc)
        if partial:
            report["partial_step_evidence"] = _receipt_evidence(partial)
        report["completed_steps_before_failure"] = completed_steps
        report.update(counters)
        report["observed_transformer_block_indices"] = observed_per_step
        if cache_provider is not None:
            report["cache_lifecycle_events"] = [record.get("events", []) for record in cache_provider.records]
        failure = _detach(exc)
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        memory_before = _memory_snapshot(mx) if mx is not None else None
        observer = None
        cache_provider = None
        result = arrays = scheduler = dit = layout = transitions = timesteps = timestep_indices = None
        artifact_arrays = metadata = original_blocks = observed_wrappers = receipts = metrics = lifecycle = None
        rebuilt_layout = rebuilt_transitions = rebuilt_timesteps = rebuilt_indices = None
        canonical = derived_config = inventory = None
        if mx is not None:
            try:
                report["memory_release"] = _release_runtime(
                    mx, active_memory_baseline=memory_before_runtime
                )
            except BaseException as release_exc:
                report["memory_release"] = {"status": "failed", "error": str(release_exc),
                                             "memory_before_allocator_purge": getattr(release_exc, "memory_before_allocator_purge", None),
                                             "memory_after_allocator_purge": getattr(release_exc, "memory_after_allocator_purge", _memory_snapshot(mx))}
                if failure is None:
                    failure = _detach(release_exc)
            report["memory_before_runtime_release"] = memory_before_runtime or memory_before
            report["memory_before_reference_clear"] = memory_before
            report["memory_after_release"] = _memory_snapshot(mx)
        release_failed = isinstance(report.get("memory_release"), Mapping) and report["memory_release"].get("status") == "failed"
        report["release_phase"] = {"status": "failed" if release_failed else "complete"}
        if release_failed:
            parity_validated = False
        report["status"] = "passed" if parity_validated and failure is None else "failed"
        report["artifacts"] = {"artifact": str(artifact_path) if artifact_path.is_file() else None,
                                "metadata": str(metadata_path) if metadata_path.is_file() else None,
                                "report": str(report_path)}
    _write_json(report_path, report)
    if failure is not None:
        raise failure.with_traceback(None)
    print(f"parity_report={report_path}", flush=True)
    print("MULTISTEP PARITY PASSED", flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

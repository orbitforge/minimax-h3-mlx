"""v0.4c real text-conditioning parity for the canonical two-step loop.

The resident and derived commands are intentionally separate process entry points.  Importing this
module is MLX-free; the model runtime is imported only after a command has entered its runtime
phase.  The artifact namespace is independent of v0.4b because its text input is a retained Qwen
conditioning result rather than a synthetic tensor.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_V04B_SPEC = importlib.util.spec_from_file_location(
    "probe_v04b_multistep_for_v04c", ROOT / "scripts" / "probe_v04b_multistep.py"
)
assert _V04B_SPEC is not None and _V04B_SPEC.loader is not None
_V04B = importlib.util.module_from_spec(_V04B_SPEC)
_V04B_SPEC.loader.exec_module(_V04B)

DEFAULT_ORIGINAL = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit"
DEFAULT_DERIVED = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
DEFAULT_CONDITIONING_CHECKPOINT = "/Volumes/models/MiniMax-H3/FL2VA"
DEFAULT_ARTIFACT = ROOT / "out" / "v0.4c" / "conditioned-reference.safetensors"
DEFAULT_METADATA = ROOT / "out" / "v0.4c" / "conditioned-reference.json"
DEFAULT_REPORT = ROOT / "out" / "v0.4c" / "conditioned-parity-report.json"

ARTIFACT_FORMAT = "minimax-h3-mlx-v04c-real-text-conditioning"
ARTIFACT_SCHEMA_VERSION = 1
FINGERPRINT_METHOD = "sha256-logical-shape-dtype-plus-canonical-float32-values-v1"
PROMPT = "A calm blue sky over a quiet meadow."
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()
TOKENIZER_CONFIGURATION = {
    "entrypoint": "MiniMaxH3TextEncoder.tokenizer",
    "call": "tokenizer(prompt, add_special_tokens=False)",
    "add_special_tokens": False,
    "images": None,
    "processor": None,
    "chat_template": None,
    "prompt_is_literal": True,
}
TOKEN_PRESENCE_MASK_DESCRIPTION = (
    "two-dimensional token-presence mask; all ones for this unpadded fixed prompt; "
    "metadata-only evidence; not passed into MiniMaxH3TextEncoder.encode; "
    "not the causal attention mask"
)
ENCODER_ATTENTION_POLICY = "create_attention_mask(hidden_states, None)"
EXPECTED_SCHEDULER_CONFIGURATION = {
    "identity": "MiniMaxH3MultimodalScheduler",
    "video": {"identity": "MiniMaxH3Scheduler", "shift": 12.0},
    "audio": {"identity": "MiniMaxH3Scheduler", "shift": 3.0},
    "num_inference_steps": 2,
    "prediction_parameterization": "velocity",
    "input_scaling": "identity",
    "update_method": "rectified-flow-euler-data-ward-velocity-v1",
}
EXPECTED_TIMESTEP_ROW_CONVENTION = {
    "text": "video_current_timestep",
    "conditioning_video": "0.999",
    "conditioning_audio": "0.999",
    "target_video": "video_current_timestep",
    "target_audio": "audio_current_timestep",
}
EXPECTED_TEXT_HIDDEN_SIZE = 5120
EXPECTED_TEXT_DTYPE = "bfloat16"
EXPECTED_BLOCK_COUNT = 50
CANONICAL_TRANSITION_COUNT = 2
CANONICAL_STEP_INDICES = (0, 1)
RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES = _V04B.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES

TRANSITION_FIELDS = _V04B.TRANSITION_FIELDS
PARITY_COMPARISONS = (
    "text_conditioning",
    "step_0_video_prediction", "step_0_audio_prediction",
    "step_0_updated_video_latent", "step_0_updated_audio_latent",
    "step_1_video_prediction", "step_1_audio_prediction",
    "final_video_latent", "final_audio_latent",
)


def _transition_tensor_key(step: int, field: str) -> str:
    return f"step_{step}_{field}"


ARTIFACT_KEYS = (
    "initial_video_latent", "initial_audio_latent", "text_conditioning", "token_ids",
    "token_presence_mask", "token_tags", "position_ids", "video_indices", "audio_indices",
    "text_indices", "step_0_timestep", "step_0_timestep_indices", "step_1_timestep",
    "step_1_timestep_indices",
    *(_transition_tensor_key(step, field) for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS),
    "step_0_video_prediction", "step_0_audio_prediction", "step_0_updated_video_latent",
    "step_0_updated_audio_latent", "step_1_video_prediction", "step_1_audio_prediction",
    "step_1_updated_video_latent", "step_1_updated_audio_latent", "final_video_latent",
    "final_audio_latent",
)

REFERENCE_METADATA_REQUIRED_KEYS = frozenset({
    "artifact_format", "artifact_schema_version", "artifact_file_format", "reference_checkpoint",
    "derived_checkpoint", "conditioning_checkpoint", "resident_checkpoint_fingerprint",
    "fingerprint_method", "reference_config_sha256", "derived_config_sha256",
    "derived_conversion_manifest_sha256", "derived_sidecar_manifest_sha256", "artifact_sha256",
    "prompt", "prompt_sha256", "tokenizer_configuration", "token_ids", "token_presence_mask",
    "token_presence_mask_description", "encoder_attention_policy",
    "conditioning_shape", "conditioning_dtype", "conditioning_fingerprint",
    "conditioning_fingerprint_method", "tensor_keys", "tensor_inventory", "packed_layout",
    "scheduler_identity", "scheduler_configuration", "prediction_parameterization", "input_scaling",
    "update_method", "transition_count", "selected_step_indices", "transitions",
    "timestep_row_convention", "configured_transformer_block_count", "observed_transformer_block_counts",
    "observed_transformer_block_indices", "expected_cache_construction_count", "parity_comparisons",
    "transition_tensor_keys", "process_isolation", "conditioning_release_contract",
})

FAILURE_RECEIPT_REQUIRED_KEYS = frozenset({
    "status", "active_phase", "completed_conditioning_stages", "completed_steps_before_failure",
    "transformer_calls", "scheduler_updates", "cache_acquisitions", "cache_releases",
    "partial_artifact_paths", "partial_block_observations", "partial_cache_lifecycle_events",
    "memory_snapshots", "error",
})
CONDITIONING_RECEIPT_REQUIRED_KEYS = frozenset({
    "memory_before_text_encoder_load", "memory_after_conditioning_materialization",
    "memory_before_encoder_reference_clear", "memory_after_encoder_release_and_allocator_purge",
    "retained_conditioning_shape_after_purge", "retained_conditioning_dtype_after_purge",
    "conditioning_fingerprint", "conditioning_release_status",
})
GENERATION_RECEIPT_REQUIRED_KEYS = frozenset({
    "memory_before_transformer_load", "peak_memory", "memory_before_reference_clear",
    "memory_after_allocator_purge", "allocator_cache_after_purge",
})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "create-conditioned-reference", help="encode the fixed prompt and create the resident reference"
    )
    _add_common_arguments(create)
    create.add_argument("--overwrite", action="store_true")
    create.set_defaults(func=cmd_create_conditioned_reference)
    compare = subparsers.add_parser(
        "compare-conditioned-derived", help="validate conditioned evidence and compare the derived loop"
    )
    _add_common_arguments(compare)
    compare.add_argument("--report", default=str(DEFAULT_REPORT))
    compare.set_defaults(func=cmd_compare_conditioned_derived)
    return parser


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--original", default=DEFAULT_ORIGINAL)
    parser.add_argument("--derived", default=DEFAULT_DERIVED)
    parser.add_argument("--conditioning-checkpoint", default=DEFAULT_CONDITIONING_CHECKPOINT)
    parser.add_argument("--artifact", default=str(DEFAULT_ARTIFACT))
    parser.add_argument("--metadata", default=str(DEFAULT_METADATA))


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
    return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _copy_array(value: Any) -> Any:
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    mx = _mlx_core_for(value)
    if mx is not None:
        return mx.array(value)
    return np.array(value, copy=True)


def metric_report(resident: Any, derived: Any) -> dict[str, Any]:
    return _V04B.metric_report(resident, derived)


def validate_exact_parity(metrics: Mapping[str, Mapping[str, Any]]) -> None:
    if set(metrics) != set(PARITY_COMPARISONS):
        raise ValueError("v0.4c exact parity comparison set is incomplete")
    failures = [name for name in PARITY_COMPARISONS if not bool(metrics[name]["exact_equality"])]
    if failures:
        raise ValueError(f"exact parity failed for {failures}")


def validate_report_before_parity(report_path: Path, metrics: Mapping[str, Mapping[str, Any]]) -> None:
    if not report_path.is_file():
        raise ValueError("conditioned diagnostic report must exist before parity validation")
    validate_exact_parity(metrics)


def validate_artifact_tensor_keys(keys: Any) -> None:
    actual = list(keys.keys()) if isinstance(keys, Mapping) else list(keys)
    if actual != list(ARTIFACT_KEYS):
        missing = [key for key in ARTIFACT_KEYS if key not in actual]
        extra = [key for key in actual if key not in ARTIFACT_KEYS]
        raise ValueError(
            f"v0.4c artifact tensor key order or membership mismatch: missing={missing}, extra={extra}, keys={actual!r}"
        )


def validate_loaded_artifact_tensor_membership(keys: Any) -> None:
    actual = list(keys.keys()) if isinstance(keys, Mapping) else list(keys)
    expected = set(ARTIFACT_KEYS)
    actual_set = set(actual)
    missing = [key for key in ARTIFACT_KEYS if key not in actual_set]
    extra = sorted(key for key in actual_set if key not in expected)
    if missing or extra:
        raise ValueError(f"loaded v0.4c artifact membership mismatch: missing={missing}, extra={extra}")


def canonicalize_loaded_artifact_arrays(loaded_arrays: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize the unordered mapping returned by a safetensors loader."""
    validate_loaded_artifact_tensor_membership(loaded_arrays)
    arrays = {key: loaded_arrays[key] for key in ARTIFACT_KEYS}
    validate_artifact_tensor_keys(arrays)
    return arrays


def _inventory_entry(value: Any) -> dict[str, Any]:
    return _shape_dtype(value)


def _expected_inventory(condition_shape: Sequence[int], token_count: int, sequence_length: int) -> dict[str, dict[str, Any]]:
    inventory: dict[str, dict[str, Any]] = {
        "initial_video_latent": {"shape": [1, 1, 96], "dtype": "bfloat16"},
        "initial_audio_latent": {"shape": [1, 2, 32], "dtype": "bfloat16"},
        "text_conditioning": {"shape": list(condition_shape), "dtype": EXPECTED_TEXT_DTYPE},
        "token_ids": {"shape": [1, token_count], "dtype": "int32"},
        "token_presence_mask": {"shape": [1, token_count], "dtype": "int32"},
        "token_tags": {"shape": [sequence_length], "dtype": "int32"},
        "position_ids": {"shape": [sequence_length, 3], "dtype": "float32"},
        "video_indices": {"shape": [1], "dtype": "int32"},
        "audio_indices": {"shape": [2], "dtype": "int32"},
        "text_indices": {"shape": [token_count], "dtype": "int32"},
        "step_0_timestep": {"shape": [1], "dtype": "float32"},
        "step_0_timestep_indices": {"shape": [sequence_length], "dtype": "int32"},
        "step_1_timestep": {"shape": [2], "dtype": "float32"},
        "step_1_timestep_indices": {"shape": [sequence_length], "dtype": "int32"},
    }
    for step in CANONICAL_STEP_INDICES:
        inventory[_transition_tensor_key(step, "selected_step_index")] = {"shape": [], "dtype": "int32"}
        for field in TRANSITION_FIELDS:
            if field != "selected_step_index":
                inventory[_transition_tensor_key(step, field)] = {"shape": [], "dtype": "float32"}
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
    return {key: inventory[key] for key in ARTIFACT_KEYS}


def validate_artifact_tensor_inventory(
    inventory: Mapping[str, Any], *, condition_shape: Sequence[int], token_count: int, sequence_length: int
) -> None:
    expected = _expected_inventory(condition_shape, token_count, sequence_length)
    actual_keys = list(inventory)
    expected_keys = list(expected)
    problems: list[str] = []
    if actual_keys != expected_keys:
        problems.append(f"key order: actual={actual_keys!r}, expected={expected_keys!r}")
    for key in expected_keys:
        if key not in inventory:
            problems.append(f"missing key: {key}")
            continue
        if inventory[key] != expected[key]:
            problems.append(f"{key}: actual={inventory[key]!r}, expected={expected[key]!r}")
    extra = sorted(set(actual_keys) - set(expected_keys))
    if extra:
        problems.append(f"extra keys: {extra!r}")
    if problems:
        raise ValueError("v0.4c artifact tensor inventory mismatch: " + "; ".join(problems))


def validate_metadata_keys(metadata: Mapping[str, Any]) -> None:
    missing = sorted(REFERENCE_METADATA_REQUIRED_KEYS - set(metadata))
    unexpected = sorted(set(metadata) - REFERENCE_METADATA_REQUIRED_KEYS)
    if missing or unexpected:
        raise ValueError(f"v0.4c metadata key contract mismatch: missing={missing}, unexpected={unexpected}")


def _array_fingerprint(value: Any) -> str:
    """Fingerprint logical shape/dtype plus evaluated, canonical float32 host bytes.

    MLX BF16 objects must never be handed directly to NumPy: NumPy does not understand MLX's
    logical BF16 dtype on every supported host.  The logical descriptor is captured before the
    value conversion, while the value bytes are always evaluated and transferred as float32.
    """
    logical_shape = [int(item) for item in value.shape]
    logical_dtype = _dtype_name(value.dtype)
    mx = _mlx_core_for(value)
    if mx is not None:
        _mlx_eval(mx, value)
        canonical = value.astype(mx.float32)
        _mlx_eval(mx, canonical)
        finite_test = getattr(mx, "isfinite", None)
        all_values = getattr(mx, "all", None)
        if callable(finite_test) and callable(all_values):
            finite = all_values(finite_test(canonical))
            _mlx_eval(mx, finite)
            if not bool(_scalar(finite)):
                raise ValueError("conditioning fingerprint requires finite values")
        data = np.ascontiguousarray(np.asarray(canonical, dtype=np.float32), dtype=np.float32)
    else:
        data = np.ascontiguousarray(np.asarray(value, dtype=np.float32), dtype=np.float32)
    if not np.all(np.isfinite(data)):
        raise ValueError("conditioning fingerprint requires finite values")
    descriptor = json.dumps(
        {"shape": logical_shape, "dtype": logical_dtype},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(descriptor + b"\0" + data.tobytes(order="C")).hexdigest()


def validate_prompt_contract(
    metadata: Mapping[str, Any], token_ids: Any, token_presence_mask: Any, conditioning: Any
) -> None:
    if metadata.get("prompt") != PROMPT or metadata.get("prompt_sha256") != PROMPT_SHA256:
        raise ValueError("fixed prompt identity mismatch")
    if metadata.get("tokenizer_configuration") != TOKENIZER_CONFIGURATION:
        raise ValueError("tokenizer configuration mismatch")
    if metadata.get("token_ids") != np.asarray(token_ids).tolist():
        raise ValueError("token IDs do not match the fixed prompt metadata")
    if metadata.get("token_presence_mask") != np.asarray(token_presence_mask).tolist():
        raise ValueError("token-presence mask does not match the fixed prompt metadata")
    expected_shape = metadata.get("conditioning_shape")
    if list(conditioning.shape) != expected_shape or metadata.get("conditioning_dtype") != _dtype_name(conditioning.dtype):
        raise ValueError("conditioning shape/dtype does not match metadata")
    validate_conditioning_shape_dtype(conditioning)
    if metadata.get("conditioning_fingerprint") != _array_fingerprint(conditioning):
        raise ValueError("exact conditioning fingerprint mismatch")


def validate_conditioning_shape_dtype(conditioning: Any) -> None:
    shape = tuple(int(item) for item in conditioning.shape)
    dtype = _dtype_name(conditioning.dtype)
    if len(shape) != 3 or shape[0] != 1 or shape[1] <= 0 or shape[2] != EXPECTED_TEXT_HIDDEN_SIZE:
        raise ValueError(f"conditioning must have shape (1, token_count, 5120), got {shape}")
    if dtype != EXPECTED_TEXT_DTYPE:
        raise ValueError(f"conditioning must have dtype {EXPECTED_TEXT_DTYPE}, got {dtype}")


def validate_conditioning_inventory(metadata: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
    shape = metadata.get("conditioning_shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError("metadata conditioning shape is malformed")
    token_count = int(shape[1])
    sequence_length = int(metadata["packed_layout"]["sequence_length"])
    validate_artifact_tensor_inventory(
        inventory, condition_shape=shape, token_count=token_count, sequence_length=sequence_length
    )
    if metadata.get("tensor_keys") != list(ARTIFACT_KEYS):
        raise ValueError("metadata tensor key order mismatch")


def validate_reference_metadata(
    metadata: Mapping[str, Any], *, original: Path, derived: Path, artifact: Path,
    inventory: Mapping[str, Any], conditioning_checkpoint: Path | None = None,
) -> None:
    validate_metadata_keys(metadata)
    raw_conditioning_path = metadata.get("conditioning_checkpoint")
    if not isinstance(raw_conditioning_path, str) or not raw_conditioning_path:
        raise ValueError("conditioning checkpoint path metadata is malformed")
    conditioning_path = Path(raw_conditioning_path).resolve()
    if conditioning_checkpoint is not None:
        conditioning_path = conditioning_checkpoint.resolve()
    expected = {
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors", "reference_checkpoint": str(original.resolve()),
        "derived_checkpoint": str(derived.resolve()), "conditioning_checkpoint": str(conditioning_path),
        "fingerprint_method": FINGERPRINT_METHOD, "conditioning_fingerprint_method": FINGERPRINT_METHOD,
        "prompt": PROMPT, "prompt_sha256": PROMPT_SHA256, "tokenizer_configuration": TOKENIZER_CONFIGURATION,
        "token_presence_mask_description": TOKEN_PRESENCE_MASK_DESCRIPTION,
        "encoder_attention_policy": ENCODER_ATTENTION_POLICY,
        "scheduler_identity": "MiniMaxH3MultimodalScheduler", "prediction_parameterization": "velocity",
        "input_scaling": "identity", "update_method": "rectified-flow-euler-data-ward-velocity-v1",
        "transition_count": 2, "selected_step_indices": [0, 1], "expected_cache_construction_count": 2,
        "parity_comparisons": list(PARITY_COMPARISONS), "tensor_keys": list(ARTIFACT_KEYS),
        "process_isolation": {"resident_command": "create-conditioned-reference",
                               "derived_command": "compare-conditioned-derived",
                               "transformers_per_process": 1, "shared_conditioning_artifact": True},
        "conditioning_release_contract": {
            "materialize_before_release": True, "encoder_released_before_transformer_load": True,
            "allocator_purge_after_gc": True, "active_memory_tolerance_bytes": RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        },
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"conditioned reference metadata mismatch for {key}")
    fingerprint = metadata.get("conditioning_fingerprint")
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("conditioning fingerprint metadata is malformed")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise ValueError("conditioning fingerprint metadata is not hexadecimal") from exc
    shape = metadata.get("conditioning_shape")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[2] != EXPECTED_TEXT_HIDDEN_SIZE:
        raise ValueError("conditioning shape metadata is malformed")
    if metadata.get("conditioning_dtype") != EXPECTED_TEXT_DTYPE:
        raise ValueError("conditioning dtype metadata is not bfloat16")
    token_presence_mask = metadata.get("token_presence_mask")
    if (
        not isinstance(token_presence_mask, list)
        or len(token_presence_mask) != 1
        or not isinstance(token_presence_mask[0], list)
        or len(token_presence_mask[0]) != int(shape[1])
        or any(value != 1 for value in token_presence_mask[0])
    ):
        raise ValueError("conditioned token-presence mask must be one all-ones row")
    if metadata.get("artifact_sha256") != _V04B.sha256_file(artifact):
        raise ValueError("conditioned artifact checksum does not match metadata")
    bindings = _checkpoint_bindings(original, derived, artifact)
    for key, value in bindings.items():
        if metadata.get(key) != value:
            raise ValueError(f"conditioned reference metadata binding mismatch for {key}")
    if metadata.get("scheduler_configuration") != EXPECTED_SCHEDULER_CONFIGURATION:
        raise ValueError("conditioned scheduler configuration mismatch")
    token_count = int(shape[1])
    expected_layout = {
        "sequence_length": token_count + 3,
        "text_token_count": token_count,
        "video_token_count": 1,
        "audio_token_count": 2,
        "video_shape": [1, 1, 96],
        "audio_shape": [1, 2, 32],
        "text_shape": list(shape),
    }
    if metadata.get("packed_layout") != expected_layout:
        raise ValueError("conditioned packed-layout contract mismatch")
    if metadata.get("timestep_row_convention") != EXPECTED_TIMESTEP_ROW_CONVENTION:
        raise ValueError("conditioned timestep-row convention mismatch")
    if metadata.get("configured_transformer_block_count") != EXPECTED_BLOCK_COUNT:
        raise ValueError("conditioned configured transformer block count mismatch")
    if metadata.get("observed_transformer_block_counts") != [EXPECTED_BLOCK_COUNT, EXPECTED_BLOCK_COUNT]:
        raise ValueError("conditioned observed transformer block counts mismatch")
    if metadata.get("observed_transformer_block_indices") != [list(range(EXPECTED_BLOCK_COUNT)), list(range(EXPECTED_BLOCK_COUNT))]:
        raise ValueError("conditioned observed transformer block indices mismatch")
    if metadata.get("transition_tensor_keys") != [
        _transition_tensor_key(step, field)
        for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS
    ]:
        raise ValueError("conditioned transition tensor keys mismatch")
    transitions = metadata.get("transitions")
    if not isinstance(transitions, Sequence) or len(transitions) != CANONICAL_TRANSITION_COUNT:
        raise ValueError("conditioned transitions must contain two records")
    try:
        _V04B.validate_canonical_schedule(transitions)
    except ValueError as exc:
        raise ValueError("conditioned transition schedule is malformed") from exc
    validate_conditioning_inventory(metadata, inventory)
    validate_conditioning_shape_dtype(
        SimpleNamespace(shape=metadata["conditioning_shape"], dtype=metadata["conditioning_dtype"])
    )


def validate_exact_conditioning_fingerprint(metadata: Mapping[str, Any], conditioning: Any) -> None:
    validate_conditioning_shape_dtype(conditioning)
    if _array_fingerprint(conditioning) != metadata.get("conditioning_fingerprint"):
        raise ValueError("exact conditioning fingerprint validation failed")


def validate_failure_receipt(receipt: Mapping[str, Any]) -> None:
    missing = sorted(FAILURE_RECEIPT_REQUIRED_KEYS - set(receipt))
    if missing:
        raise ValueError(f"failure receipt is missing evidence: {missing}")
    error = receipt.get("error")
    if not isinstance(error, Mapping) or not error.get("type") or not isinstance(error.get("message"), str):
        raise ValueError("failure receipt must preserve the original error type and message")
    if receipt.get("status") == "passed":
        raise ValueError("failure receipt cannot claim success")


def validate_conditioning_receipt(receipt: Mapping[str, Any]) -> None:
    missing = sorted(CONDITIONING_RECEIPT_REQUIRED_KEYS - set(receipt))
    if missing:
        raise ValueError(f"conditioning receipt is missing evidence: {missing}")
    if receipt.get("conditioning_release_status") != "success":
        raise ValueError("conditioning receipt cannot claim encoder release success")
    if receipt.get("retained_conditioning_dtype_after_purge") != EXPECTED_TEXT_DTYPE:
        raise ValueError("conditioning receipt has an invalid retained dtype")
    shape = receipt.get("retained_conditioning_shape_after_purge")
    if not isinstance(shape, list) or len(shape) != 3 or shape[0] != 1 or shape[2] != EXPECTED_TEXT_HIDDEN_SIZE:
        raise ValueError("conditioning receipt has an invalid retained shape")


def validate_generation_receipt(receipt: Mapping[str, Any]) -> None:
    missing = sorted(GENERATION_RECEIPT_REQUIRED_KEYS - set(receipt))
    if missing:
        raise ValueError(f"generation receipt is missing evidence: {missing}")


def validate_resident_release_gates(receipt: Mapping[str, Any]) -> None:
    expected = {
        "transformer_release_to_post_conditioning_baseline",
        "final_process_release_to_pre_conditioning_baseline",
    }
    statuses = receipt.get("release_gate_statuses")
    if not isinstance(statuses, Mapping) or set(statuses) != expected or any(
        statuses[name] != "success" for name in expected
    ):
        raise ValueError("resident release gates are incomplete or unsuccessful")


def _memory_snapshot(mx: Any) -> dict[str, int | None]:
    return _V04B._memory_snapshot(mx)


def _detach(exc: BaseException) -> BaseException:
    return _V04B._detach(exc)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n")


def _existing_artifact_paths(*paths: Path) -> list[str]:
    return [str(path) for path in paths if path.is_file()]


def _zero_counters() -> dict[str, int]:
    return {"transformer_calls": 0, "scheduler_updates": 0, "cache_acquisitions": 0, "cache_releases": 0}


def _new_receipt(artifact_path: Path, metadata_path: Path) -> dict[str, Any]:
    return {
        "status": "failed", "active_phase": "conditioning", "completed_conditioning_stages": [],
        "completed_steps_before_failure": 0, **_zero_counters(),
        "partial_artifact_paths": _existing_artifact_paths(artifact_path, metadata_path),
        "partial_block_observations": [], "partial_cache_lifecycle_events": [], "memory_snapshots": {},
    }


def _record_error(receipt: dict[str, Any], exc: BaseException) -> None:
    receipt["error"] = {"type": type(exc).__name__, "message": str(exc)}
    validate_failure_receipt(receipt)


def _release_conditioning_runtime(
    mx: Any, *, active_memory_baseline: Mapping[str, Any] | int | None = None
) -> dict[str, Any]:
    gc.collect()
    before = _memory_snapshot(mx)
    baseline = active_memory_baseline.get("active") if isinstance(active_memory_baseline, Mapping) else active_memory_baseline
    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        raise RuntimeError("allocator cache purge is unavailable during conditioning release")
    try:
        clear_cache()
    except BaseException as exc:
        error = RuntimeError(f"conditioning allocator purge failed: {exc}")
        error.memory_before_allocator_purge = before
        error.memory_after_allocator_purge = _memory_snapshot(mx)
        raise error from exc
    after = _memory_snapshot(mx)
    active_after = after.get("active")
    if baseline is not None and active_after is not None and active_after > baseline + RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES:
        error = RuntimeError(
            f"conditioning active memory after release exceeded baseline: {active_after} > "
            f"{baseline} + {RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES}"
        )
        error.memory_before_allocator_purge = before
        error.memory_after_allocator_purge = after
        raise error
    return {
        "status": "success", "memory_before_allocator_purge": before,
        "memory_after_allocator_purge": after, "allocator_cache_after_purge": after.get("allocator_cache"),
        "active_memory_baseline": baseline, "active_memory_tolerance_bytes": RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        "active_memory_gate_available": baseline is not None and active_after is not None,
    }


def _conditioning_checkpoint_path(args: argparse.Namespace) -> Path:
    root = Path(args.conditioning_checkpoint).resolve()
    path = root / "text_encoder"
    if not path.is_dir():
        raise FileNotFoundError(f"conditioning checkpoint has no text_encoder directory: {path}")
    return path


def _prepare_conditioning(mx: Any, args: argparse.Namespace, receipt: dict[str, Any]) -> dict[str, Any]:
    """Encode, evaluate, retain, and release the fixed prompt before any transformer load."""
    from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

    baseline = _memory_snapshot(mx)
    receipt["memory_snapshots"]["memory_before_text_encoder_load"] = baseline
    encoder = input_ids = token_tags = encoded_tags = prompt_embeds = token_presence_mask = None
    tokenizer = processor = None
    retained_conditioning = retained_input_ids = retained_token_presence_mask = retained_token_tags = None
    retained: dict[str, Any] | None = None
    failure: BaseException | None = None
    release_result: dict[str, Any] | None = None
    try:
        encoder = MiniMaxH3TextEncoder(_conditioning_checkpoint_path(args), dtype=mx.bfloat16, load_vision=False, verbose=True)
        receipt["completed_conditioning_stages"].append("text_encoder_loaded")
        receipt["memory_snapshots"]["memory_after_text_encoder_load"] = _memory_snapshot(mx)

        input_ids, token_tags, vision_inputs = encoder.build_request(PROMPT, None)
        if vision_inputs is not None:
            raise ValueError("fixed v0.4c prompt unexpectedly produced vision inputs")
        token_presence_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        # This artifact tensor is two-dimensional presence evidence only.  MiniMaxH3TextEncoder
        # builds its actual causal policy internally as create_attention_mask(hidden_states, None).
        prompt_embeds, encoded_tags = encoder.encode(PROMPT, None)
        if list(np.asarray(token_tags)) != list(np.asarray(encoded_tags)):
            raise ValueError("token tags from build_request and encode disagree")
        receipt["completed_conditioning_stages"].append("prompt_tokenized_and_encoded")

        # Materialize and copy only the arrays required after encoder reclamation.  The explicit
        # eval/copy boundary prevents a lazy hidden-state graph from retaining Qwen parameters.
        retained_conditioning = _copy_array(prompt_embeds.astype(mx.bfloat16))
        retained_input_ids = _copy_array(input_ids.astype(mx.int32))
        retained_token_presence_mask = _copy_array(token_presence_mask)
        retained_token_tags = mx.array(np.asarray(token_tags, dtype=np.int32))
        _mlx_eval(mx, retained_conditioning, retained_input_ids, retained_token_presence_mask, retained_token_tags)
        validate_conditioning_shape_dtype(retained_conditioning)
        receipt["completed_conditioning_stages"].append("conditioning_materialized")
        receipt["memory_snapshots"]["memory_after_conditioning_materialization"] = _memory_snapshot(mx)
        retained = {
            "text_conditioning": retained_conditioning,
            "token_ids": retained_input_ids,
            "token_presence_mask": retained_token_presence_mask,
            "token_tags": retained_token_tags,
        }
        receipt["conditioning_shape_after_purge"] = list(retained_conditioning.shape)
        receipt["conditioning_dtype_after_purge"] = _dtype_name(retained_conditioning.dtype)
    except BaseException as exc:
        failure = _detach(exc)
    finally:
        receipt["memory_snapshots"]["memory_before_encoder_reference_clear"] = _memory_snapshot(mx)
        # Clear tokenizer, processor, encoder, tokenization objects, hidden-state handles, and all
        # temporary arrays.  `retained` is deliberately the only live conditioning state here.
        tokenizer = getattr(encoder, "_tokenizer", None) if encoder is not None else None
        processor = getattr(encoder, "_processor", None) if encoder is not None else None
        encoder = None
        input_ids = token_tags = encoded_tags = prompt_embeds = token_presence_mask = None
        tokenizer = processor = None
        gc.collect()
        try:
            release_result = _release_conditioning_runtime(mx, active_memory_baseline=baseline)
            receipt["conditioning_release"] = release_result
            receipt["memory_snapshots"]["memory_after_encoder_release_and_allocator_purge"] = (
                release_result["memory_after_allocator_purge"]
            )
            receipt["conditioning_release_status"] = "success"
        except BaseException as release_exc:
            receipt["conditioning_release_status"] = "failed"
            receipt["conditioning_release"] = {
                "status": "failed", "error": {"type": type(release_exc).__name__, "message": str(release_exc)},
                "memory_before_allocator_purge": getattr(release_exc, "memory_before_allocator_purge", None),
                "memory_after_allocator_purge": getattr(release_exc, "memory_after_allocator_purge", _memory_snapshot(mx)),
            }
            receipt["memory_snapshots"]["memory_after_encoder_release_and_allocator_purge"] = (
                receipt["conditioning_release"].get("memory_after_allocator_purge")
            )
            if failure is None:
                failure = _detach(release_exc)
            else:
                receipt["cleanup_error"] = {"type": type(release_exc).__name__, "message": str(release_exc)}
    if failure is not None:
        raise failure.with_traceback(None)
    assert retained is not None
    _mlx_eval(
        mx, retained["text_conditioning"], retained["token_ids"],
        retained["token_presence_mask"], retained["token_tags"]
    )
    validate_conditioning_shape_dtype(retained["text_conditioning"])
    receipt["retained_conditioning_shape_after_purge"] = list(retained["text_conditioning"].shape)
    receipt["retained_conditioning_dtype_after_purge"] = _dtype_name(retained["text_conditioning"].dtype)
    receipt["conditioning_fingerprint"] = _array_fingerprint(retained["text_conditioning"])
    receipt["completed_conditioning_stages"].append("conditioning_fingerprint_validated")
    snapshots = receipt["memory_snapshots"]
    receipt.update({
        "memory_before_text_encoder_load": snapshots["memory_before_text_encoder_load"],
        "memory_after_conditioning_materialization": snapshots["memory_after_conditioning_materialization"],
        "memory_before_encoder_reference_clear": snapshots["memory_before_encoder_reference_clear"],
        "memory_after_encoder_release_and_allocator_purge": snapshots["memory_after_encoder_release_and_allocator_purge"],
    })
    validate_conditioning_receipt(receipt)
    return retained


def _build_canonical_scheduler():
    from minimax_h3_mlx.scheduler import MiniMaxH3MultimodalScheduler, MiniMaxH3Scheduler
    video, audio = MiniMaxH3Scheduler(shift=12.0), MiniMaxH3Scheduler(shift=3.0)
    video.set_timesteps(3)
    audio.set_timesteps(3)
    scheduler = MiniMaxH3MultimodalScheduler(video, audio)
    _V04B.validate_canonical_schedule([vars(scheduler.transition(i)) for i in CANONICAL_STEP_INDICES])
    return scheduler


def _canonical_layout_and_timesteps(config: Any, scheduler: Any, token_tags: Any):
    from minimax_h3_mlx.packing import build_packed_sequence, build_row_timesteps
    raw_tags = np.asarray(token_tags.tolist() if hasattr(token_tags, "tolist") else token_tags, dtype=np.int64)
    layout = build_packed_sequence(raw_tags, 1, 2, 2, 1, tuple(config.patch_size), keyframe_anchors=())
    transitions = [_V04B._transition_mapping(scheduler.transition(i)) for i in CANONICAL_STEP_INDICES]
    timesteps: dict[int, Any] = {}
    timestep_indices: dict[int, Any] = {}
    for step, transition in enumerate(transitions):
        timesteps[step], timestep_indices[step] = build_row_timesteps(
            layout, transition["video_current_timestep"], transition["audio_current_timestep"], 0.999, 0.999
        )
    return layout, transitions, timesteps, timestep_indices


def validate_timestep_reconstruction(artifact_arrays: Mapping[str, Any], layout: Any,
                                     canonical: Sequence[Mapping[str, Any]],
                                     rebuilt_timesteps: Mapping[int, Any],
                                     rebuilt_indices: Mapping[int, Any]) -> None:
    """Validate the packed timetable without v0.4b's fixed one-token shape assumption."""
    for step in CANONICAL_STEP_INDICES:
        serialized_timestep = artifact_arrays[f"step_{step}_timestep"]
        serialized_indices = artifact_arrays[f"step_{step}_timestep_indices"]
        if not _exact_equal(serialized_timestep, rebuilt_timesteps[step]):
            raise ValueError(f"serialized step {step} timestep does not match canonical reconstruction")
        if not _exact_equal(serialized_indices, rebuilt_indices[step]):
            raise ValueError(f"serialized step {step} timestep indices do not match canonical reconstruction")
        if tuple(serialized_timestep.shape) != tuple(rebuilt_timesteps[step].shape):
            raise ValueError(f"serialized step {step} timestep shape is not canonical")
        if tuple(serialized_indices.shape) != tuple(rebuilt_indices[step].shape):
            raise ValueError(f"serialized step {step} timestep-index shape is not canonical")
        expected_rows = np.full(
            layout.sequence_length, _V04B._float32_value(canonical[step]["video_current_timestep"]), dtype=np.float32
        )
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


def _runtime_inputs(mx: Any, dit: Any, scheduler: Any, conditioned: Mapping[str, Any]):
    layout, transitions, timesteps, timestep_indices = _canonical_layout_and_timesteps(
        dit.config, scheduler, conditioned["token_tags"]
    )

    def pattern(shape: tuple[int, ...], salt: int):
        values = _V04B.deterministic_input_values(math.prod(shape), salt)
        return mx.array(values, dtype=mx.float32).reshape(shape).astype(mx.bfloat16)

    arrays = {
        "initial_video_latent": pattern((1, 1, 96), 0),
        "initial_audio_latent": pattern((1, 2, 32), 1),
        "text_conditioning": conditioned["text_conditioning"],
        "token_ids": conditioned["token_ids"],
        "token_presence_mask": conditioned["token_presence_mask"],
        "token_tags": layout.token_tags,
        "position_ids": layout.position_ids,
        "video_indices": layout.video_indices,
        "audio_indices": layout.audio_indices,
        "text_indices": layout.text_indices,
    }
    return layout, transitions, timesteps, timestep_indices, arrays


def _artifact_arrays_from_result(mx: Any, arrays: Mapping[str, Any], timesteps: Mapping[int, Any],
                                 timestep_indices: Mapping[int, Any], transitions: Sequence[Mapping[str, Any]],
                                 result: Any) -> dict[str, Any]:
    first, second = result.step_receipts
    artifact = {
        **arrays,
        "step_0_timestep": timesteps[0], "step_0_timestep_indices": timestep_indices[0],
        "step_1_timestep": timesteps[1], "step_1_timestep_indices": timestep_indices[1],
        "step_0_video_prediction": first.video_prediction, "step_0_audio_prediction": first.audio_prediction,
        "step_0_updated_video_latent": first.updated_video_latent, "step_0_updated_audio_latent": first.updated_audio_latent,
        "step_1_video_prediction": second.video_prediction, "step_1_audio_prediction": second.audio_prediction,
        "step_1_updated_video_latent": second.updated_video_latent, "step_1_updated_audio_latent": second.updated_audio_latent,
        "final_video_latent": result.final_video_latent, "final_audio_latent": result.final_audio_latent,
    }
    for step, raw_transition in enumerate(transitions):
        transition = _V04B._transition_mapping(raw_transition)
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


def _checkpoint_bindings(original: Path, derived: Path, artifact: Path | None = None) -> dict[str, str]:
    values = {
        "resident_checkpoint_fingerprint": _V04B.resident_checkpoint_fingerprint(original),
        "reference_config_sha256": _V04B.sha256_file(original / "config.json"),
        "derived_config_sha256": _V04B.sha256_file(derived / "config.json"),
        "derived_conversion_manifest_sha256": _V04B.sha256_file(derived / "conversion_manifest.json"),
        "derived_sidecar_manifest_sha256": _V04B.sha256_file(derived / "adaln" / "manifest.json"),
    }
    if artifact is not None:
        values["artifact_sha256"] = _V04B.sha256_file(artifact)
    return values


def _metadata(args: argparse.Namespace, scheduler: Any, dit: Any, layout: Any,
              transitions: Sequence[Mapping[str, Any]], artifact_arrays: Mapping[str, Any],
              receipt: Mapping[str, Any], observed: Sequence[Sequence[int]]) -> dict[str, Any]:
    validate_artifact_tensor_keys(artifact_arrays)
    inventory = {key: _inventory_entry(value) for key, value in artifact_arrays.items()}
    condition_shape = inventory["text_conditioning"]["shape"]
    token_count = int(condition_shape[1])
    validate_artifact_tensor_inventory(inventory, condition_shape=condition_shape, token_count=token_count,
                                       sequence_length=int(layout.sequence_length))
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    token_ids = np.asarray(artifact_arrays["token_ids"]).tolist()
    token_presence_mask = np.asarray(artifact_arrays["token_presence_mask"]).tolist()
    return {
        "artifact_format": ARTIFACT_FORMAT, "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors", "reference_checkpoint": str(original),
        "derived_checkpoint": str(derived), "conditioning_checkpoint": str(Path(args.conditioning_checkpoint).resolve()),
        **_checkpoint_bindings(original, derived), "fingerprint_method": FINGERPRINT_METHOD,
        "artifact_sha256": "pending", "prompt": PROMPT, "prompt_sha256": PROMPT_SHA256,
        "tokenizer_configuration": TOKENIZER_CONFIGURATION, "token_ids": token_ids,
        "token_presence_mask": token_presence_mask,
        "token_presence_mask_description": TOKEN_PRESENCE_MASK_DESCRIPTION,
        "encoder_attention_policy": ENCODER_ATTENTION_POLICY,
        "conditioning_shape": condition_shape,
        "conditioning_dtype": inventory["text_conditioning"]["dtype"],
        "conditioning_fingerprint": _array_fingerprint(artifact_arrays["text_conditioning"]),
        "conditioning_fingerprint_method": FINGERPRINT_METHOD, "tensor_keys": list(ARTIFACT_KEYS),
        "tensor_inventory": inventory,
        "packed_layout": {"sequence_length": int(layout.sequence_length), "text_token_count": token_count,
                          "video_token_count": int(layout.video_indices.shape[0]),
                          "audio_token_count": int(layout.audio_indices.shape[0]),
                          "video_shape": [1, 1, 96], "audio_shape": [1, 2, 32],
                          "text_shape": condition_shape},
        "scheduler_identity": "MiniMaxH3MultimodalScheduler", "scheduler_configuration": scheduler.configuration(),
        "prediction_parameterization": "velocity", "input_scaling": "identity",
        "update_method": scheduler.update_method, "transition_count": len(transitions),
        "selected_step_indices": list(CANONICAL_STEP_INDICES), "transitions": [dict(item) for item in transitions],
        "timestep_row_convention": dict(EXPECTED_TIMESTEP_ROW_CONVENTION),
        "configured_transformer_block_count": len(dit.blocks),
        "observed_transformer_block_counts": [len(item) for item in observed],
        "observed_transformer_block_indices": [list(item) for item in observed],
        "expected_cache_construction_count": 2, "parity_comparisons": list(PARITY_COMPARISONS),
        "transition_tensor_keys": [_transition_tensor_key(step, field)
                                    for step in CANONICAL_STEP_INDICES for field in TRANSITION_FIELDS],
        "process_isolation": {"resident_command": "create-conditioned-reference",
                               "derived_command": "compare-conditioned-derived",
                               "transformers_per_process": 1, "shared_conditioning_artifact": True},
        "conditioning_release_contract": {
            "materialize_before_release": True, "encoder_released_before_transformer_load": True,
            "allocator_purge_after_gc": True, "active_memory_tolerance_bytes": RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        },
    }


def _loop_receipt_values(result: Any, failure: BaseException | None) -> tuple[int, dict[str, int], Sequence[Any]]:
    if result is not None:
        return result.completed_steps, {
            "transformer_calls": result.transformer_calls, "scheduler_updates": result.scheduler_updates,
            "cache_acquisitions": result.cache_acquisitions, "cache_releases": result.cache_releases,
        }, result.step_receipts
    partial = getattr(failure, "denoise_step_receipts", ()) if failure is not None else ()
    return len(partial), {
        "transformer_calls": int(getattr(failure, "denoise_transformer_calls", 0)),
        "scheduler_updates": int(getattr(failure, "denoise_scheduler_updates", 0)),
        "cache_acquisitions": int(getattr(failure, "denoise_cache_acquisitions", 0)),
        "cache_releases": int(getattr(failure, "denoise_cache_releases", 0)),
    }, partial


def _receipt_evidence(receipts: Sequence[Any]) -> list[dict[str, Any]]:
    return _V04B._receipt_evidence(receipts)


def _complete_runtime_release(
    mx: Any, receipt: dict[str, Any], baseline: Mapping[str, Any] | None, gate_name: str
) -> None:
    try:
        release = _V04B._release_runtime(mx, active_memory_baseline=baseline)
        receipt.setdefault("release_gates", {})[gate_name] = release
        receipt["generation_release"] = release
        receipt["memory_snapshots"]["memory_after_allocator_purge"] = release["memory_after_allocator_purge"]
        receipt["allocator_cache_after_purge"] = release["allocator_cache_after"]
        receipt["memory_after_allocator_purge"] = release["memory_after_allocator_purge"]
        receipt.setdefault("release_gate_statuses", {})[gate_name] = "success"
        receipt["release_status"] = "success"
    except BaseException as exc:
        receipt.setdefault("release_gate_statuses", {})[gate_name] = "failed"
        receipt["release_status"] = "failed"
        receipt.setdefault("release_gates", {})[gate_name] = {
            "status": "failed", "error": {"type": type(exc).__name__, "message": str(exc)},
            "memory_before_allocator_purge": getattr(exc, "memory_before_allocator_purge", None),
            "memory_after_allocator_purge": getattr(exc, "memory_after_allocator_purge", _memory_snapshot(mx)),
        }
        raise


@contextmanager
def _observe_transformer_block_execution(dit: Any):
    with _V04B.observe_transformer_block_execution(dit) as observer:
        yield observer


def _run_resident(args: argparse.Namespace) -> dict[str, Any]:
    from minimax_h3_mlx.denoise import denoise_loop
    from minimax_h3_mlx.load import load_dit

    artifact_path, metadata_path = Path(args.artifact).resolve(), Path(args.metadata).resolve()
    receipt = _new_receipt(artifact_path, metadata_path)
    failure: BaseException | None = None
    mx = dit = scheduler = layout = transitions = timesteps = timestep_indices = arrays = None
    conditioned = result = receipts = artifact_arrays = metadata = None
    observer = wrapped_blocks = original_blocks = observed_wrappers = None
    observed_per_step: list[list[int]] = []
    generation_baseline = None
    try:
        import mlx.core as mx
        conditioned = _prepare_conditioning(mx, args, receipt)
        receipt["completed_conditioning_stages"].append("text_encoder_released_before_transformer_load")
        receipt["active_phase"] = "transformer"
        receipt["memory_snapshots"]["memory_before_transformer_load"] = _memory_snapshot(mx)
        receipt["memory_before_transformer_load"] = receipt["memory_snapshots"]["memory_before_transformer_load"]
        # The transformer phase retains the already-materialized conditioning, so its release gate
        # compares against the post-conditioning, pre-transformer baseline rather than the earlier
        # text-encoder baseline.
        generation_baseline = receipt["memory_snapshots"]["memory_before_transformer_load"]
        scheduler = _build_canonical_scheduler()
        dit = load_dit(Path(args.original), verbose=True)
        if getattr(dit, "construction_mode", None) != "resident":
            raise ValueError("resident command did not construct a resident transformer")
        receipt["memory_snapshots"]["memory_after_transformer_load"] = _memory_snapshot(mx)
        layout, transitions, timesteps, timestep_indices, arrays = _runtime_inputs(mx, dit, scheduler, conditioned)
        receipt["active_phase"] = "denoising"
        with _observe_transformer_block_execution(dit) as observer:
            wrapped_blocks = dit.blocks
            original_blocks = wrapped_blocks
            observed_wrappers = wrapped_blocks
            result = denoise_loop(
                observer, scheduler, initial_video_latent=arrays["initial_video_latent"],
                initial_audio_latent=arrays["initial_audio_latent"], text_embedding=arrays["text_conditioning"],
                timestep_provider=lambda step, transition: (timesteps[step], timestep_indices[step]),
                token_tags=arrays["token_tags"], position_ids=arrays["position_ids"],
                video_indices=arrays["video_indices"], audio_indices=arrays["audio_indices"],
                text_indices=arrays["text_indices"], expected_text_shape=tuple(arrays["text_conditioning"].shape),
            )
        observed_per_step = [list(item) for item in observer.observations]
        _V04B.validate_per_step_block_observations(observed_per_step)
        _V04B.validate_step_receipts(result.step_receipts)
        receipt["completed_steps"] = result.completed_steps
        receipt["completed_steps_before_failure"] = result.completed_steps
        receipt.update({"transformer_calls": result.transformer_calls, "scheduler_updates": result.scheduler_updates,
                        "cache_acquisitions": result.cache_acquisitions, "cache_releases": result.cache_releases})
        artifact_arrays = _artifact_arrays_from_result(mx, arrays, timesteps, timestep_indices, transitions, result)
        metadata = _metadata(args, scheduler, dit, layout, transitions, artifact_arrays, receipt, observed_per_step)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if not args.overwrite and (artifact_path.exists() or metadata_path.exists()):
            raise FileExistsError(f"conditioned reference artifact already exists: {artifact_path}")
        # The ordered mapping is the writer contract; loaders are canonicalized independently.
        mx.save_safetensors(str(artifact_path), artifact_arrays)
        metadata["artifact_sha256"] = _V04B.sha256_file(artifact_path)
        _write_json(metadata_path, metadata)
        receipt["status"] = "runtime-complete"
    except BaseException as exc:
        failure = _detach(exc)
        if observer is not None:
            observed_per_step = [list(item) for item in observer.observations]
        completed_steps, counters, partial = _loop_receipt_values(result, exc)
        receipt["completed_steps_before_failure"] = completed_steps
        receipt.update(counters)
        receipt["partial_block_observations"] = observed_per_step
        receipt["partial_step_evidence"] = _receipt_evidence(partial) if partial else []
        if getattr(exc, "cache_lifecycle_events", None) is not None:
            receipt["partial_cache_lifecycle_events"] = exc.cache_lifecycle_events
        _record_error(receipt, exc)
    finally:
        receipt["memory_snapshots"]["memory_before_reference_clear"] = (
            _memory_snapshot(mx) if mx is not None else None
        )
        receipt["memory_before_reference_clear"] = receipt["memory_snapshots"]["memory_before_reference_clear"]
        if mx is not None:
            receipt["peak_memory"] = _memory_snapshot(mx).get("peak")
        observer = wrapped_blocks = original_blocks = observed_wrappers = None
        dit = scheduler = layout = transitions = timesteps = timestep_indices = arrays = None
        result = receipts = artifact_arrays = metadata = None
        pre_conditioning_baseline = receipt["memory_snapshots"].get("memory_before_text_encoder_load")
        if mx is not None:
            try:
                _complete_runtime_release(
                    mx, receipt, generation_baseline,
                    "transformer_release_to_post_conditioning_baseline",
                )
            except BaseException as release_exc:
                if failure is None:
                    failure = _detach(release_exc)
                    _record_error(receipt, release_exc)
                else:
                    receipt["cleanup_error"] = {"type": type(release_exc).__name__, "message": str(release_exc)}
            # The first gate proves transformer release while the retained conditioning result is
            # still owned by this process.  Only the final gate runs after that retained state dies.
            conditioned = None
            try:
                _complete_runtime_release(
                    mx, receipt, pre_conditioning_baseline,
                    "final_process_release_to_pre_conditioning_baseline",
                )
            except BaseException as release_exc:
                if failure is None:
                    failure = _detach(release_exc)
                    _record_error(receipt, release_exc)
                else:
                    receipt["cleanup_error"] = {"type": type(release_exc).__name__, "message": str(release_exc)}
            if any(status != "success" for status in receipt.get("release_gate_statuses", {}).values()):
                receipt["release_status"] = "failed"
        if failure is None and receipt.get("release_status") == "success":
            validate_resident_release_gates(receipt)
            validate_generation_receipt(receipt)
        receipt["partial_artifact_paths"] = _existing_artifact_paths(artifact_path, metadata_path)
        if failure is not None:
            receipt["status"] = "failed"
        print(f"phase_receipt={json.dumps(receipt, sort_keys=True, default=str)}", flush=True)
    if failure is not None:
        raise failure.with_traceback(None)
    return receipt


def _load_artifact_arrays(path: Path) -> Mapping[str, Any]:
    import mlx.core as mx
    return canonicalize_loaded_artifact_arrays(mx.load(str(path)))


def _load_derived_transformer(path: Path) -> Any:
    from minimax_h3_mlx.load import load_dit
    return load_dit(path, verbose=True)


def _validate_reference_before_derived_load(args: argparse.Namespace, artifact_path: Path, metadata_path: Path):
    metadata = json.loads(metadata_path.read_text())
    arrays = _load_artifact_arrays(artifact_path)
    inventory = {key: _inventory_entry(value) for key, value in arrays.items()}
    original, derived = Path(args.original).resolve(), Path(args.derived).resolve()
    validate_reference_metadata(
        metadata,
        original=original,
        derived=derived,
        artifact=artifact_path,
        inventory=inventory,
        conditioning_checkpoint=Path(args.conditioning_checkpoint),
    )
    return metadata, arrays


class _StreamedCacheProvider:
    def __init__(self, dit: Any, mx: Any):
        self.dit, self.mx = dit, mx
        self.records: list[dict[str, Any]] = []
        self.active = False
        self.event_number = 0
        self.next_session_token = 0
        self.last_session_token = None

    def _event(self, record: dict[str, Any], name: str) -> None:
        self.event_number += 1
        record["events"].append({"global_event_number": self.event_number, "step_index": record["step_index"],
                                 "cache_session_token": record["cache_session_token"], "event": name})

    def cache_for_step(self, step_index, timestep):
        if self.active:
            raise ValueError("cache overlap between conditioned denoising steps")
        from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache
        self.next_session_token += 1
        record = {"step_index": step_index, "cache_session_token": self.next_session_token, "events": [], "sidecar_names": []}
        self.records.append(record)
        self._event(record, "acquire-start")
        sidecars: list[str] = []
        cache, stats = build_streamed_modulation_cache(
            self.dit, timestep, dtype=self.mx.bfloat16,
            telemetry=lambda event, details: sidecars.append(str(details["path"])) if event == "sidecar_opening" else None,
        )
        tables = getattr(cache, "tables", None)
        if tables is None:
            raise ValueError("constructed streamed cache has no tables collection")
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


def _cache_lifecycle_records(provider: _StreamedCacheProvider) -> list[dict[str, Any]]:
    records = []
    for record in provider.records:
        stats = record["stats"]
        records.append({
            "step_index": record["step_index"], "cache_table_count": record["cache_table_count"],
            "cache_session_token": record["cache_session_token"], "complete_cache_statistics": _V04B._complete_stats(stats),
            "events": record["events"], "blocks_completed": stats.blocks_completed,
            "sidecar_files_opened": stats.sidecar_files_opened, "unique_sidecars_opened": stats.unique_sidecar_files_opened,
            "successful_payload_opens": stats.successful_payload_opens, "completed_payload_releases": stats.completed_payload_releases,
            "every_sidecar_released_before_next_opened": stats.every_sidecar_released_before_next_opened,
            "sidecar_overlap_observed": stats.sidecar_overlap_observed,
            "next_sidecar_opened_before_previous_release": stats.next_sidecar_opened_before_previous_release,
            "dense_temporary_projection_created": stats.dense_temporary_projection_created,
            "sidecar_names": record["sidecar_names"],
        })
    _V04B.validate_cache_lifecycle(records)
    return records


def cmd_create_conditioned_reference(args: argparse.Namespace) -> int:
    _run_resident(args)
    print("CONDITIONED RESIDENT REFERENCE CREATED", flush=True)
    return 0


def cmd_compare_conditioned_derived(args: argparse.Namespace) -> int:
    from minimax_h3_mlx.denoise import denoise_loop

    artifact_path, metadata_path, report_path = (
        Path(args.artifact).resolve(), Path(args.metadata).resolve(), Path(args.report).resolve()
    )
    report: dict[str, Any] = {
        "artifact": str(artifact_path), "reference_metadata": str(metadata_path), "status": "failed",
        "active_phase": "reference-validation", "partial_artifact_paths": _existing_artifact_paths(artifact_path, metadata_path),
        "partial_block_observations": [], "partial_cache_lifecycle_events": [], "memory_snapshots": {},
        **_zero_counters(), "completed_conditioning_stages": [],
        "completed_steps_before_failure": 0,
    }
    failure: BaseException | None = None
    parity_validated = False
    mx = dit = scheduler = arrays = reference_arrays = result = layout = None
    transitions = timesteps = timestep_indices = None
    derived_config = canonical = rebuilt_layout = rebuilt_transitions = None
    rebuilt_timesteps = rebuilt_indices = inventory = None
    provider = observer = wrapped_blocks = original_blocks = observed_wrappers = None
    lifecycle = metrics = derived_outputs = receipts = None
    observed_per_step: list[list[int]] = []
    derived_baseline = None
    try:
        import mlx.core as mx
        # Capture the process baseline before the reference artifact is loaded.  Final derived
        # release must return to this baseline after both artifact arrays and transformer state die.
        derived_baseline = _memory_snapshot(mx)
        report["memory_snapshots"]["memory_before_reference_load"] = derived_baseline
        report["memory_before_reference_load"] = derived_baseline
        metadata, arrays = _validate_reference_before_derived_load(args, artifact_path, metadata_path)
        reference_arrays = arrays
        report["completed_conditioning_stages"].append("reference_artifact_validated")
        validate_prompt_contract(
            metadata, arrays["token_ids"], arrays["token_presence_mask"], arrays["text_conditioning"]
        )
        validate_exact_conditioning_fingerprint(metadata, arrays["text_conditioning"])
        report["completed_conditioning_stages"].append("conditioning_fingerprint_validated")
        report["memory_snapshots"]["memory_before_transformer_load"] = _memory_snapshot(mx)
        report["memory_before_transformer_load"] = report["memory_snapshots"]["memory_before_transformer_load"]
        from minimax_h3_mlx.config import DiTConfig
        derived = Path(args.derived).resolve()
        derived_config = DiTConfig.from_json(derived / "config.json")
        scheduler = _build_canonical_scheduler()
        text_token_count = int(metadata["conditioning_shape"][1])
        layout, transitions, timesteps, timestep_indices = _canonical_layout_and_timesteps(
            derived_config, scheduler, np.ones(text_token_count, dtype=np.int64)
        )
        if not _exact_equal(arrays["token_tags"], layout.token_tags):
            raise ValueError("packed token-tag layout does not match conditioned reference")
        _V04B.validate_canonical_schedule(metadata["transitions"])
        _V04B.validate_transition_bindings(metadata, arrays, scheduler)
        validate_timestep_reconstruction(
            arrays, layout, metadata["transitions"], timesteps, timestep_indices
        )
        if metadata["packed_layout"]["sequence_length"] != layout.sequence_length:
            raise ValueError("derived packed layout does not match conditioned reference")
        report["completed_conditioning_stages"].append("derived_configuration_validated")
        dit = _load_derived_transformer(derived)
        if getattr(dit, "construction_mode", None) != "cache_only":
            raise ValueError("conditioned derived command did not construct a cache-only transformer")
        report["completed_conditioning_stages"].append("derived_transformer_loaded")
        report["memory_snapshots"]["memory_after_transformer_load"] = _memory_snapshot(mx)
        provider = _StreamedCacheProvider(dit, mx)
        with _observe_transformer_block_execution(dit) as observer:
            wrapped_blocks = dit.blocks
            original_blocks = wrapped_blocks
            observed_wrappers = wrapped_blocks
            result = denoise_loop(
                observer, scheduler, initial_video_latent=arrays["initial_video_latent"],
                initial_audio_latent=arrays["initial_audio_latent"], text_embedding=arrays["text_conditioning"],
                timestep_provider=lambda step, transition: (arrays[f"step_{step}_timestep"], arrays[f"step_{step}_timestep_indices"]),
                token_tags=arrays["token_tags"], position_ids=arrays["position_ids"],
                video_indices=arrays["video_indices"], audio_indices=arrays["audio_indices"],
                text_indices=arrays["text_indices"], modulation_cache_provider=provider,
                expected_text_shape=tuple(arrays["text_conditioning"].shape),
            )
        observed_per_step = [list(item) for item in observer.observations]
        _V04B.validate_per_step_block_observations(observed_per_step)
        _V04B.validate_step_receipts(result.step_receipts)
        lifecycle = _cache_lifecycle_records(provider)
        report["completed_conditioning_stages"].append("derived_generation_completed")
        metrics = {"text_conditioning": metric_report(
            arrays["text_conditioning"], _copy_array(arrays["text_conditioning"])
        )}
        derived_outputs = {
            "step_0_video_prediction": result.step_receipts[0].video_prediction,
            "step_0_audio_prediction": result.step_receipts[0].audio_prediction,
            "step_0_updated_video_latent": result.step_receipts[0].updated_video_latent,
            "step_0_updated_audio_latent": result.step_receipts[0].updated_audio_latent,
            "step_1_video_prediction": result.step_receipts[1].video_prediction,
            "step_1_audio_prediction": result.step_receipts[1].audio_prediction,
            "final_video_latent": result.final_video_latent, "final_audio_latent": result.final_audio_latent,
        }
        for name in PARITY_COMPARISONS[1:]:
            metrics[name] = metric_report(arrays[name], derived_outputs[name])
        validate_exact_parity(metrics)
        report["completed_conditioning_stages"].append("derived_parity_validated")
        report.update({"status": "parity-evaluated", "metrics": metrics, "cache_lifecycle": lifecycle,
                       "completed_steps": result.completed_steps, "completed_steps_before_failure": result.completed_steps,
                       "transformer_calls": result.transformer_calls, "scheduler_updates": result.scheduler_updates,
                       "cache_acquisitions": result.cache_acquisitions, "cache_releases": result.cache_releases,
                       "cache_construction_sessions": len(provider.records),
                       "observed_transformer_block_indices": observed_per_step, "active_phase": "complete"})
        _write_json(report_path, report)
        # Validation is intentionally after report creation, matching the v0.4b report contract.
        validate_report_before_parity(report_path, metrics)
        parity_validated = True
    except BaseException as exc:
        failure = _detach(exc)
        if observer is not None:
            observed_per_step = [list(item) for item in observer.observations]
        completed_steps, counters, partial = _loop_receipt_values(result, exc)
        report["completed_steps_before_failure"] = completed_steps
        report.update(counters)
        report["partial_block_observations"] = observed_per_step
        report["partial_step_evidence"] = _receipt_evidence(partial) if partial else []
        report["partial_cache_lifecycle_events"] = [record.get("events", []) for record in provider.records] if provider else []
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["memory_snapshots"]["memory_before_reference_clear"] = _memory_snapshot(mx) if mx is not None else None
        report["memory_before_reference_clear"] = report["memory_snapshots"]["memory_before_reference_clear"]
        if mx is not None:
            report["peak_memory"] = _memory_snapshot(mx).get("peak")
        # Clear every loaded reference, rebuilt timetable, cache/lifecycle object, and parity
        # output before the final process-release gate.  The provider owns the transformer too.
        observer = wrapped_blocks = original_blocks = observed_wrappers = None
        provider = None
        dit = scheduler = arrays = reference_arrays = result = layout = None
        transitions = timesteps = timestep_indices = None
        derived_config = canonical = rebuilt_layout = rebuilt_transitions = None
        rebuilt_timesteps = rebuilt_indices = inventory = None
        lifecycle = metrics = derived_outputs = receipts = metadata = None
        if mx is not None:
            try:
                _complete_runtime_release(
                    mx, report, derived_baseline,
                    "final_process_release_to_pre_conditioning_baseline",
                )
            except BaseException as release_exc:
                report["release_status"] = "failed"
                if failure is None:
                    failure = _detach(release_exc)
                    report["error"] = {"type": type(release_exc).__name__, "message": str(release_exc)}
                else:
                    report["cleanup_error"] = {"type": type(release_exc).__name__, "message": str(release_exc)}
        if failure is None and report.get("release_status") == "success":
            validate_generation_receipt(report)
        if report.get("release_status") == "failed":
            parity_validated = False
        report["release_phase"] = {"status": "complete" if report.get("release_status") != "failed" else "failed"}
        report["status"] = "passed" if parity_validated and failure is None else "failed"
        if failure is not None:
            validate_failure_receipt(report)
        report["partial_artifact_paths"] = _existing_artifact_paths(artifact_path, metadata_path)
        report["report"] = str(report_path)
        _write_json(report_path, report)
    if failure is not None:
        raise failure.with_traceback(None)
    print(f"parity_report={report_path}", flush=True)
    print("CONDITIONED PARITY PASSED", flush=True)
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

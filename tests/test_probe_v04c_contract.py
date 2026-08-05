"""MLX-free contracts for v0.4c real text conditioning and process-isolated parity."""

from __future__ import annotations

import copy
import gc
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v04c_conditioned", ROOT / "scripts" / "probe_v04c_conditioned.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeMLXArray:
    """MLX-like value whose BF16 object rejects direct NumPy conversion."""

    def __init__(self, value, dtype="bfloat16", core=None):
        self.data = np.asarray(value, dtype=np.float32)
        self.dtype = dtype
        self.__mlx_array__ = core is not None
        self.__mlx_core__ = core

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype):
        return FakeMLXArray(self.data, dtype=dtype, core=self.__mlx_core__)

    def copy(self):
        return FakeMLXArray(self.data.copy(), self.dtype, self.__mlx_core__)

    def tolist(self):
        return self.data.tolist()

    def __array__(self, dtype=None):
        if self.__mlx_array__ and self.dtype == "bfloat16":
            raise AssertionError("NumPy received the BF16 MLX object directly")
        return np.asarray(self.data, dtype=dtype)


class FakeMLXCore:
    bfloat16 = "bfloat16"
    float32 = "float32"
    int32 = np.int32

    @staticmethod
    def eval(*values):
        return None

    @staticmethod
    def isfinite(value):
        return np.isfinite(value.data)

    @staticmethod
    def all(value):
        return np.all(value)

    @staticmethod
    def array(value, dtype=None):
        if isinstance(value, FakeMLXArray):
            value = value.data
        return FakeMLXArray(value, dtype=dtype or "float32", core=FAKE_MX)

    @staticmethod
    def ones(shape, dtype=None):
        return FakeMLXArray(np.ones(shape), dtype=dtype or "float32", core=FAKE_MX)


FAKE_MX = FakeMLXCore()


class FakeArray:
    def __init__(self, value, dtype="bfloat16"):
        self.data = np.asarray(value, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return FakeArray(self.data.copy(), self.dtype)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


class FakeRuntimeMemory:
    def __init__(self, active: int, cache: int, peak: int = 0):
        self.active = active
        self.cache = cache
        self.peak = peak

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.cache

    def get_peak_memory(self):
        return self.peak

    def clear_cache(self):
        self.cache = 0


class FakeNumpyCompatibleValue:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __array__(self, dtype=None, copy=None):
        if copy:
            return np.array(self.value, dtype=dtype, copy=True)
        return np.asarray(self.value, dtype=dtype)


def condition_metadata(conditioning: FakeArray, token_ids, token_presence_mask):
    return {
        "prompt": probe.PROMPT,
        "prompt_sha256": probe.PROMPT_SHA256,
        "tokenizer_configuration": probe.TOKENIZER_CONFIGURATION,
        "token_ids": np.asarray(token_ids).tolist(),
        "token_presence_mask": np.asarray(token_presence_mask).tolist(),
        "token_presence_mask_description": probe.TOKEN_PRESENCE_MASK_DESCRIPTION,
        "encoder_attention_policy": probe.ENCODER_ATTENTION_POLICY,
        "conditioning_shape": list(conditioning.shape),
        "conditioning_dtype": conditioning.dtype,
        "conditioning_fingerprint": probe._array_fingerprint(conditioning),
    }


def transition_records():
    return [
        {
            "selected_step_index": 0,
            "video_current_timestep": 0.0,
            "video_next_timestep": 0.5,
            "video_current_sigma": 1.0,
            "video_next_sigma": 0.5,
            "audio_current_timestep": 0.0,
            "audio_next_timestep": 0.5,
            "audio_current_sigma": 1.0,
            "audio_next_sigma": 0.5,
        },
        {
            "selected_step_index": 1,
            "video_current_timestep": 0.5,
            "video_next_timestep": 1.0,
            "video_current_sigma": 0.5,
            "video_next_sigma": 0.0,
            "audio_current_timestep": 0.5,
            "audio_next_timestep": 1.0,
            "audio_current_sigma": 0.5,
            "audio_next_sigma": 0.0,
        },
    ]


def valid_reference_metadata():
    shape = [1, 3, 5120]
    inventory = probe._expected_inventory(shape, token_count=3, sequence_length=6)
    original = Path("/tmp/v04c-original")
    derived = Path("/tmp/v04c-derived")
    conditioning = Path("/tmp/v04c-conditioning")
    return {
        "artifact_format": probe.ARTIFACT_FORMAT,
        "artifact_schema_version": probe.ARTIFACT_SCHEMA_VERSION,
        "artifact_file_format": "safetensors",
        "reference_checkpoint": str(original.resolve()),
        "derived_checkpoint": str(derived.resolve()),
        "conditioning_checkpoint": str(conditioning.resolve()),
        "resident_checkpoint_fingerprint": "resident",
        "fingerprint_method": probe.FINGERPRINT_METHOD,
        "reference_config_sha256": "reference-config",
        "derived_config_sha256": "derived-config",
        "derived_conversion_manifest_sha256": "conversion",
        "derived_sidecar_manifest_sha256": "sidecars",
        "artifact_sha256": "a" * 64,
        "prompt": probe.PROMPT,
        "prompt_sha256": probe.PROMPT_SHA256,
        "tokenizer_configuration": probe.TOKENIZER_CONFIGURATION,
        "token_ids": [[17, 23, 42]],
        "token_presence_mask": [[1, 1, 1]],
        "token_presence_mask_description": probe.TOKEN_PRESENCE_MASK_DESCRIPTION,
        "encoder_attention_policy": probe.ENCODER_ATTENTION_POLICY,
        "conditioning_shape": shape,
        "conditioning_dtype": "bfloat16",
        "conditioning_fingerprint": "b" * 64,
        "conditioning_fingerprint_method": probe.FINGERPRINT_METHOD,
        "tensor_keys": list(probe.ARTIFACT_KEYS),
        "tensor_inventory": inventory,
        "packed_layout": {
            "sequence_length": 6,
            "text_token_count": 3,
            "video_token_count": 1,
            "audio_token_count": 2,
            "video_shape": [1, 1, 96],
            "audio_shape": [1, 2, 32],
            "text_shape": shape,
        },
        "scheduler_identity": "MiniMaxH3MultimodalScheduler",
        "scheduler_configuration": probe.EXPECTED_SCHEDULER_CONFIGURATION,
        "prediction_parameterization": "velocity",
        "input_scaling": "identity",
        "update_method": "rectified-flow-euler-data-ward-velocity-v1",
        "transition_count": 2,
        "selected_step_indices": [0, 1],
        "transitions": transition_records(),
        "timestep_row_convention": probe.EXPECTED_TIMESTEP_ROW_CONVENTION,
        "configured_transformer_block_count": 50,
        "observed_transformer_block_counts": [50, 50],
        "observed_transformer_block_indices": [list(range(50)), list(range(50))],
        "expected_cache_construction_count": 2,
        "parity_comparisons": list(probe.PARITY_COMPARISONS),
        "transition_tensor_keys": [
            probe._transition_tensor_key(step, field)
            for step in probe.CANONICAL_STEP_INDICES
            for field in probe.TRANSITION_FIELDS
        ],
        "process_isolation": {
            "resident_command": "create-conditioned-reference",
            "derived_command": "compare-conditioned-derived",
            "transformers_per_process": 1,
            "shared_conditioning_artifact": True,
        },
        "conditioning_release_contract": {
            "materialize_before_release": True,
            "encoder_released_before_transformer_load": True,
            "allocator_purge_after_gc": True,
            "active_memory_tolerance_bytes": probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        },
    }, original, derived, conditioning, inventory


class WriterMX:
    int32 = np.int32
    float32 = np.float32

    @staticmethod
    def array(value, dtype=None):
        return np.asarray(value, dtype=dtype)


class ProbeV04CContractTests(unittest.TestCase):
    def test_copy_array_does_not_reference_v04b_copy(self):
        source = inspect.getsource(probe._copy_array)
        self.assertNotIn("_V04B._copy", source)

    def test_copy_array_uses_callable_copy_method(self):
        class CopyTrackingValue:
            def __init__(self):
                self.copy_calls = 0
                self.copied = object()

            def copy(self):
                self.copy_calls += 1
                return self.copied

        value = CopyTrackingValue()
        self.assertIs(probe._copy_array(value), value.copied)
        self.assertEqual(value.copy_calls, 1)

    def test_copy_array_uses_mlx_array_without_entering_numpy(self):
        value = FakeMLXArray([1.0, 2.0], core=FAKE_MX)
        value.copy = None
        with patch.object(FAKE_MX, "array", wraps=FAKE_MX.array) as array:
            copied = probe._copy_array(value)
        array.assert_called_once_with(value)
        self.assertIsInstance(copied, FakeMLXArray)
        self.assertIs(copied.__mlx_core__, FAKE_MX)

    def test_copy_array_mlx_like_value_never_enters_numpy(self):
        value = FakeMLXArray([1.0, 2.0], core=FAKE_MX)
        value.copy = None
        copied = probe._copy_array(value)
        self.assertEqual(copied.tolist(), [1.0, 2.0])

    def test_copy_array_numpy_fallback_is_independent(self):
        value = FakeNumpyCompatibleValue([1.0, 2.0])
        copied = probe._copy_array(value)
        value.value[0] = 99.0
        self.assertIsInstance(copied, np.ndarray)
        np.testing.assert_array_equal(copied, [1.0, 2.0])

    def test_conditioning_failure_releases_to_pre_encoder_baseline(self):
        class FailingCopyEncoder:
            def __init__(self, *args, **kwargs):
                self._tokenizer = object()
                self._processor = object()

            def build_request(self, prompt, images):
                return (
                    FAKE_MX.array([[17, 23]], dtype=FAKE_MX.int32),
                    np.array([0, 0], dtype=np.int64),
                    None,
                )

            def encode(self, prompt, images):
                return (
                    FakeMLXArray(np.ones((1, 2, 5120)), core=FAKE_MX),
                    np.array([0, 0], dtype=np.int64),
                )

        class FakeTextEncoderModule:
            MiniMaxH3TextEncoder = FailingCopyEncoder

        runtime = FakeRuntimeMemory(active=0, cache=4096)
        runtime.bfloat16 = FAKE_MX.bfloat16
        runtime.int32 = FAKE_MX.int32
        runtime.array = FAKE_MX.array
        runtime.ones = FAKE_MX.ones
        args = SimpleNamespace(conditioning_checkpoint="/tmp/fake-conditioning")
        receipt = probe._new_receipt(Path("/tmp/no-v04c-artifact"), Path("/tmp/no-v04c-metadata"))
        with patch.dict(sys.modules, {"minimax_h3_mlx.text_encoder": FakeTextEncoderModule}), \
             patch.object(probe, "_conditioning_checkpoint_path", return_value=Path("/tmp/fake-text-encoder")), \
             patch.object(probe, "_copy_array", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                probe._prepare_conditioning(runtime, args, receipt)
        release = receipt["conditioning_release"]
        self.assertEqual(release["active_memory_baseline"], 0)
        self.assertTrue(release["active_memory_gate_available"])
        self.assertEqual(receipt["memory_snapshots"]["memory_before_text_encoder_load"]["active"], 0)

    def test_cli_surface_is_exactly_two_conditioned_subcommands(self):
        action = next(item for item in probe.build_parser()._actions if item.dest == "command")
        self.assertEqual(set(action.choices), {
            "create-conditioned-reference", "compare-conditioned-derived"
        })
        parser_text = (ROOT / "scripts" / "probe_v04c_conditioned.py").read_text()
        self.assertNotIn('add_argument("--prompt"', parser_text)

    def test_prompt_identity_is_fixed_and_text_only(self):
        self.assertEqual(probe.PROMPT, "A calm blue sky over a quiet meadow.")
        self.assertEqual(len(probe.PROMPT_SHA256), 64)
        self.assertIsNone(probe.TOKENIZER_CONFIGURATION["images"])
        self.assertFalse(probe.TOKENIZER_CONFIGURATION["add_special_tokens"])
        self.assertTrue(probe.TOKENIZER_CONFIGURATION["prompt_is_literal"])

    def test_artifact_keeps_the_complete_42_tensor_namespace(self):
        self.assertEqual(len(probe.ARTIFACT_KEYS), 42)
        self.assertIn("text_conditioning", probe.ARTIFACT_KEYS)
        self.assertIn("token_presence_mask", probe.ARTIFACT_KEYS)
        self.assertNotIn("text_input", probe.ARTIFACT_KEYS)
        self.assertNotIn("attention_mask", probe.ARTIFACT_KEYS)
        self.assertEqual(len(probe.PARITY_COMPARISONS), 9)

    def test_conditioning_shape_and_dtype_contract(self):
        conditioning = FakeArray(np.ones((1, 3, 5120)), "bfloat16")
        probe.validate_conditioning_shape_dtype(conditioning)
        with self.assertRaisesRegex(ValueError, "5120"):
            probe.validate_conditioning_shape_dtype(FakeArray(np.ones((1, 3, 16)), "bfloat16"))
        with self.assertRaisesRegex(ValueError, "bfloat16"):
            probe.validate_conditioning_shape_dtype(FakeArray(np.ones((1, 3, 5120)), "float32"))

    def test_bfloat16_conditioning_fingerprint_never_enters_numpy_directly(self):
        value = FakeMLXArray(np.arange(6).reshape(1, 2, 3), core=FAKE_MX)
        self.assertEqual(probe._array_fingerprint(value), probe._array_fingerprint(value.copy()))

    def test_fingerprint_includes_original_logical_shape_and_dtype(self):
        same_values = np.arange(6).reshape(1, 2, 3)
        bfloat = FakeMLXArray(same_values, dtype="bfloat16", core=FAKE_MX)
        float32 = FakeMLXArray(same_values, dtype="float32", core=FAKE_MX)
        different_shape = FakeMLXArray(same_values.reshape(1, 3, 2), dtype="bfloat16", core=FAKE_MX)
        self.assertNotEqual(probe._array_fingerprint(bfloat), probe._array_fingerprint(float32))
        self.assertNotEqual(probe._array_fingerprint(bfloat), probe._array_fingerprint(different_shape))

    def test_fingerprint_survives_save_load_equivalent_value_conversion(self):
        values = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 3, 4)
        saved = FakeMLXArray(values, dtype="bfloat16", core=FAKE_MX)
        loaded = FakeMLXArray(np.asarray(saved.astype(FAKE_MX.float32)), dtype="bfloat16", core=FAKE_MX)
        self.assertEqual(probe._array_fingerprint(saved), probe._array_fingerprint(loaded))

    def test_nonfinite_conditioning_fails_fingerprint_creation(self):
        value = FakeMLXArray([[np.nan, 1.0]], dtype="bfloat16", core=FAKE_MX)
        with self.assertRaisesRegex(ValueError, "finite"):
            probe._array_fingerprint(value)

    def test_prompt_token_ids_presence_mask_and_fingerprint_are_exactly_validated(self):
        conditioning = FakeArray(np.arange(3 * 5120).reshape(1, 3, 5120), "bfloat16")
        token_ids = np.array([[17, 23, 42]], dtype=np.int32)
        token_presence_mask = np.ones((1, 3), dtype=np.int32)
        metadata = condition_metadata(conditioning, token_ids, token_presence_mask)
        probe.validate_prompt_contract(metadata, token_ids, token_presence_mask, conditioning)
        broken = dict(metadata, conditioning_fingerprint="0" * 64)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            probe.validate_prompt_contract(broken, token_ids, token_presence_mask, conditioning)

    def test_token_presence_mask_is_truthful_and_encoder_policy_is_real(self):
        self.assertIn("two-dimensional", probe.TOKEN_PRESENCE_MASK_DESCRIPTION)
        self.assertIn("all ones", probe.TOKEN_PRESENCE_MASK_DESCRIPTION)
        self.assertIn("not passed", probe.TOKEN_PRESENCE_MASK_DESCRIPTION)
        self.assertIn("not the causal attention mask", probe.TOKEN_PRESENCE_MASK_DESCRIPTION)
        self.assertEqual(probe.ENCODER_ATTENTION_POLICY, "create_attention_mask(hidden_states, None)")
        source = (ROOT / "scripts" / "probe_v04c_conditioned.py").read_text()
        self.assertNotIn('"attention_mask"', source)
        self.assertIn("encoder.encode(PROMPT, None)", source)

    def test_schema_inventory_is_dynamic_for_real_token_count(self):
        inventory = probe._expected_inventory((1, 3, 5120), token_count=3, sequence_length=6)
        probe.validate_artifact_tensor_inventory(
            inventory, condition_shape=(1, 3, 5120), token_count=3, sequence_length=6
        )
        self.assertEqual(inventory["token_presence_mask"], {"shape": [1, 3], "dtype": "int32"})
        self.assertEqual(inventory["token_tags"], {"shape": [6], "dtype": "int32"})
        self.assertEqual(len(inventory), len(probe.ARTIFACT_KEYS))

    def test_complete_writer_output_and_dynamic_inventory_pass_behaviorally(self):
        transitions = transition_records()
        arrays = {
            "initial_video_latent": np.zeros((1, 1, 96), dtype=np.float32),
            "initial_audio_latent": np.zeros((1, 2, 32), dtype=np.float32),
            "text_conditioning": np.zeros((1, 3, 5120), dtype=np.float32),
            "token_ids": np.zeros((1, 3), dtype=np.int32),
            "token_presence_mask": np.ones((1, 3), dtype=np.int32),
            "token_tags": np.ones((6,), dtype=np.int32),
            "position_ids": np.zeros((6, 3), dtype=np.float32),
            "video_indices": np.zeros((1,), dtype=np.int32),
            "audio_indices": np.zeros((2,), dtype=np.int32),
            "text_indices": np.zeros((3,), dtype=np.int32),
        }
        timesteps = {0: np.zeros((1,), dtype=np.float32), 1: np.ones((2,), dtype=np.float32)}
        indices = {0: np.zeros((6,), dtype=np.int32), 1: np.ones((6,), dtype=np.int32)}
        step = lambda: SimpleNamespace(
            video_prediction=np.zeros((1, 1, 96), dtype=np.float32),
            audio_prediction=np.zeros((1, 2, 32), dtype=np.float32),
            updated_video_latent=np.zeros((1, 1, 96), dtype=np.float32),
            updated_audio_latent=np.zeros((1, 2, 32), dtype=np.float32),
        )
        result = SimpleNamespace(step_receipts=(step(), step()), final_video_latent=np.zeros((1, 1, 96), dtype=np.float32),
                                 final_audio_latent=np.zeros((1, 2, 32), dtype=np.float32))
        artifact = probe._artifact_arrays_from_result(
            WriterMX(), arrays, timesteps, indices, transitions, result
        )
        self.assertEqual(list(artifact), list(probe.ARTIFACT_KEYS))
        self.assertEqual(len(artifact), 42)

    def test_schema_rejects_reordered_writer_and_metadata_keys(self):
        keys = list(probe.ARTIFACT_KEYS)
        keys[0], keys[1] = keys[1], keys[0]
        with self.assertRaisesRegex(ValueError, "order"):
            probe.validate_artifact_tensor_keys(keys)
        source = inspect.getsource(probe._artifact_arrays_from_result)
        self.assertIn("artifact = {key: artifact[key] for key in ARTIFACT_KEYS}", source)
        metadata_source = inspect.getsource(probe._metadata)
        self.assertIn('"tensor_keys": list(ARTIFACT_KEYS)', metadata_source)

    def test_loaded_safetensors_membership_is_unordered_but_canonicalized(self):
        unordered = {key: key for key in reversed(probe.ARTIFACT_KEYS)}
        canonical = probe.canonicalize_loaded_artifact_arrays(unordered)
        self.assertEqual(list(canonical), list(probe.ARTIFACT_KEYS))
        broken = dict(unordered)
        broken.pop(probe.ARTIFACT_KEYS[0])
        with self.assertRaisesRegex(ValueError, "missing"):
            probe.canonicalize_loaded_artifact_arrays(broken)

    def test_materialization_precedes_encoder_reference_clear(self):
        source = inspect.getsource(probe._prepare_conditioning)
        self.assertLess(source.index("_mlx_eval(mx, retained_conditioning"), source.rindex("encoder = None"))
        self.assertLess(source.index("retained_conditioning = _copy_array"), source.index("gc.collect()"))

    def test_encoder_release_evidence_is_recorded_before_transformer_loading(self):
        source = inspect.getsource(probe._run_resident)
        self.assertLess(source.index("conditioned = _prepare_conditioning"), source.index('append("text_encoder_released_before_transformer_load")'))
        self.assertLess(source.index('append("text_encoder_released_before_transformer_load")'), source.index("dit = load_dit"))

    def test_conditioning_receipt_requires_all_memory_boundaries(self):
        receipt = {
            "memory_before_text_encoder_load": {"active": 1},
            "memory_after_conditioning_materialization": {"active": 2},
            "memory_before_encoder_reference_clear": {"active": 2},
            "memory_after_encoder_release_and_allocator_purge": {"active": 1},
            "retained_conditioning_shape_after_purge": [1, 3, 5120],
            "retained_conditioning_dtype_after_purge": "bfloat16",
            "conditioning_fingerprint": "x",
            "conditioning_release_status": "success",
        }
        probe.validate_conditioning_receipt(receipt)
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            probe.validate_conditioning_receipt({"conditioning_release_status": "success"})

    def test_generation_receipt_requires_peak_and_post_purge_memory(self):
        valid = {key: {} for key in probe.GENERATION_RECEIPT_REQUIRED_KEYS}
        probe.validate_generation_receipt(valid)
        valid.pop("peak_memory")
        with self.assertRaisesRegex(ValueError, "peak_memory"):
            probe.validate_generation_receipt(valid)

    def test_failure_receipt_preserves_original_error_evidence(self):
        receipt = probe._new_receipt(Path("/tmp/no-artifact"), Path("/tmp/no-metadata"))
        receipt["error"] = {"type": "ValueError", "message": "original failure"}
        probe.validate_failure_receipt(receipt)
        broken = dict(receipt)
        broken.pop("partial_block_observations")
        with self.assertRaisesRegex(ValueError, "missing evidence"):
            probe.validate_failure_receipt(broken)

    def test_resident_has_two_explicit_release_gates_and_cleanup_nulls_references(self):
        source = inspect.getsource(probe._run_resident)
        self.assertIn("transformer_release_to_post_conditioning_baseline", source)
        self.assertIn("final_process_release_to_pre_conditioning_baseline", source)
        cleanup = source[source.index("finally:"):]
        for name in ("observer", "wrapped_blocks", "dit", "scheduler", "layout", "transitions", "timesteps",
                     "timestep_indices", "arrays", "conditioned", "result", "receipts", "artifact_arrays", "metadata"):
            self.assertIn(name, cleanup)
        self.assertLess(cleanup.index("result = receipts = artifact_arrays = metadata = None"),
                        cleanup.index("_complete_runtime_release"))

    def test_derived_cleanup_clears_loaded_arrays_configuration_timetables_and_outputs(self):
        source = inspect.getsource(probe.cmd_compare_conditioned_derived)
        cleanup = source[source.index("finally:"):]
        for name in ("reference_arrays", "arrays", "derived_config", "canonical", "rebuilt_layout",
                     "rebuilt_transitions", "rebuilt_timesteps", "rebuilt_indices", "provider", "lifecycle",
                     "metrics", "derived_outputs", "metadata", "observer", "wrapped_blocks"):
            self.assertIn(name, cleanup)
        self.assertLess(cleanup.index("derived_config = canonical = rebuilt_layout = rebuilt_transitions = None"),
                        cleanup.index("_complete_runtime_release"))

    def test_weak_references_to_fake_mlx_arrays_expire_before_release(self):
        value = FakeMLXArray([1.0, 2.0], core=FAKE_MX)
        reference = weakref.ref(value)
        value = None
        gc.collect()
        self.assertIsNone(reference())

    def test_derived_completion_stages_begin_empty_and_append_after_success(self):
        source = inspect.getsource(probe.cmd_compare_conditioned_derived)
        self.assertIn('"completed_conditioning_stages": []', source)
        self.assertLess(source.index("metadata, arrays = _validate_reference_before_derived_load"),
                        source.index('append("reference_artifact_validated")'))
        self.assertLess(source.index('append("reference_artifact_validated")'),
                        source.index('append("conditioning_fingerprint_validated")'))
        self.assertLess(source.index('append("conditioning_fingerprint_validated")'),
                        source.index("dit = _load_derived_transformer"))

    def test_cleanup_failure_suppresses_resident_and_derived_success(self):
        resident = inspect.getsource(probe._run_resident)
        derived = inspect.getsource(probe.cmd_compare_conditioned_derived)
        self.assertIn('receipt["status"] = "failed"', resident)
        self.assertIn("if any(status != \"success\"", resident)
        self.assertIn("parity_validated = False", derived)
        self.assertIn('report["status"] = "passed" if parity_validated and failure is None else "failed"', derived)

    def test_reference_metadata_validates_every_proof_bearing_field_before_derived_loading(self):
        metadata, original, derived, conditioning, inventory = valid_reference_metadata()
        tamper_values = {
            "conditioning_checkpoint": "/tmp/tampered-conditioning",
            "fingerprint_method": "wrong",
            "conditioning_fingerprint_method": "wrong",
            "token_presence_mask": [[0, 1, 1]],
            "token_presence_mask_description": "causal mask",
            "encoder_attention_policy": "token_presence_mask",
            "packed_layout": dict(metadata["packed_layout"], sequence_length=999),
            "scheduler_configuration": dict(metadata["scheduler_configuration"], num_inference_steps=9),
            "timestep_row_convention": dict(metadata["timestep_row_convention"], text="wrong"),
            "configured_transformer_block_count": 49,
            "observed_transformer_block_counts": [50, 49],
            "observed_transformer_block_indices": [list(range(50)), list(range(49))],
            "expected_cache_construction_count": 1,
            "transition_tensor_keys": ["tampered"],
            "process_isolation": dict(metadata["process_isolation"], transformers_per_process=2),
            "conditioning_release_contract": dict(metadata["conditioning_release_contract"], materialize_before_release=False),
            "parity_comparisons": list(metadata["parity_comparisons"][:-1]),
        }
        bindings = {key: metadata[key] for key in (
            "resident_checkpoint_fingerprint", "reference_config_sha256", "derived_config_sha256",
            "derived_conversion_manifest_sha256", "derived_sidecar_manifest_sha256", "artifact_sha256"
        )}
        for field, value in tamper_values.items():
            with self.subTest(field=field):
                broken = copy.deepcopy(metadata)
                broken[field] = value
                with patch.object(probe._V04B, "sha256_file", return_value="a" * 64), \
                     patch.object(probe, "_checkpoint_bindings", return_value=bindings), \
                     patch.object(probe, "_load_derived_transformer", side_effect=AssertionError("must not load")) as load:
                    with self.assertRaises(ValueError):
                        probe.validate_reference_metadata(
                            broken, original=original, derived=derived, artifact=Path("/tmp/artifact"),
                            inventory=inventory, conditioning_checkpoint=conditioning,
                        )
                    load.assert_not_called()

    def test_exact_nine_parity_gates_are_independently_tested(self):
        for name in probe.PARITY_COMPARISONS:
            metrics = {item: {"exact_equality": True} for item in probe.PARITY_COMPARISONS}
            metrics[name] = {"exact_equality": False}
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                probe.validate_exact_parity(metrics)

    def test_two_steps_two_scheduler_updates_and_two_cache_sessions_are_fixed_contracts(self):
        self.assertEqual(probe.CANONICAL_TRANSITION_COUNT, 2)
        self.assertEqual(probe.CANONICAL_STEP_INDICES, (0, 1))
        self.assertEqual(probe.EXPECTED_BLOCK_COUNT, 50)
        source = (ROOT / "scripts" / "probe_v04c_conditioned.py").read_text()
        self.assertIn("acquire-start", source)
        self.assertIn("release-complete", source)
        self.assertIn("_V04B.validate_cache_lifecycle(records)", inspect.getsource(probe._cache_lifecycle_records))

    def test_independent_fifty_block_observations_are_required(self):
        observations = [list(range(50)), list(range(50))]
        probe._V04B.validate_per_step_block_observations(observations)
        with self.assertRaisesRegex(ValueError, "per-step"):
            probe._V04B.validate_per_step_block_observations([list(range(100))])

    def test_step_zero_release_precedes_step_one_acquisition(self):
        events = [
            "step0:acquire-start", "step0:acquire-complete", "step0:release-start", "step0:release-complete",
            "step1:acquire-start", "step1:acquire-complete", "step1:release-start", "step1:release-complete",
        ]
        self.assertLess(events.index("step0:release-complete"), events.index("step1:acquire-start"))

    def test_active_memory_gate_enforces_conditioning_baseline(self):
        baseline = 8 * 1024 * 1024
        runtime = FakeRuntimeMemory(active=baseline, cache=4096, peak=baseline + 1)
        result = probe._release_conditioning_runtime(runtime, active_memory_baseline={"active": baseline})
        self.assertEqual(result["allocator_cache_after_purge"], 0)
        self.assertTrue(result["active_memory_gate_available"])
        failing = FakeRuntimeMemory(active=baseline + probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES + 1, cache=0)
        with self.assertRaisesRegex(RuntimeError, "baseline"):
            probe._release_conditioning_runtime(failing, active_memory_baseline={"active": baseline})

    def test_conditioning_release_failure_is_not_hidden_by_original_failure(self):
        source = inspect.getsource(probe._prepare_conditioning)
        self.assertIn("if failure is None:", source)
        self.assertIn('receipt["cleanup_error"]', source)
        self.assertIn("raise failure.with_traceback(None)", source)

    def test_parity_report_must_exist_before_exact_gate(self):
        with self.assertRaisesRegex(ValueError, "must exist"):
            probe.validate_report_before_parity(Path("/tmp/conditioned-report-does-not-exist"), {})
        with self.assertRaisesRegex(ValueError, "comparison set"):
            path = Path("/tmp/conditioned-report-contract.json")
            path.write_text(json.dumps({"status": "parity-evaluated"}))
            try:
                probe.validate_report_before_parity(path, {})
            finally:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

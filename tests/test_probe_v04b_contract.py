"""MLX-free behavior contracts for the v0.4b bounded denoising loop."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_v04b_multistep", ROOT / "scripts" / "probe_v04b_multistep.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

from minimax_h3_mlx.denoise import denoise_loop  # noqa: E402


class FakeArray:
    def __init__(self, data, dtype: str):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return FakeArray(self.data.copy(), self.dtype)

    def __getitem__(self, item):
        return FakeArray(self.data[item], self.dtype)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


class MLXLikeArray:
    """MLX-shaped test double: every accidental NumPy conversion is a hard failure."""
    __mlx_array__ = True

    def __init__(self, data, dtype="float32"):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype
        self.__mlx_core__ = FAKE_MX

    @property
    def shape(self):
        return self.data.shape

    def item(self):
        return self.data.item()

    def __array__(self, dtype=None):
        raise AssertionError("MLX-like array must not enter NumPy")

    def _binary(self, other, op, dtype=None):
        right = other.data if isinstance(other, MLXLikeArray) else other
        return MLXLikeArray(op(self.data, right), dtype or self.dtype)

    def __eq__(self, other): return self._binary(other, np.equal, "bool")
    def __ne__(self, other): return self._binary(other, np.not_equal, "bool")
    def __add__(self, other): return self._binary(other, np.add)
    def __radd__(self, other): return self._binary(other, np.add)
    def __sub__(self, other): return self._binary(other, np.subtract)
    def __mul__(self, other): return self._binary(other, np.multiply)
    def __rmul__(self, other): return self._binary(other, np.multiply)
    def __truediv__(self, other): return self._binary(other, np.divide)
    def __le__(self, other): return self._binary(other, np.less_equal, "bool")


class FakeMLXCore:
    float32 = "float32"
    int32 = "int32"
    bfloat16 = "bfloat16"

    def array(self, value, dtype=None):
        data = value.data if isinstance(value, MLXLikeArray) else value
        return MLXLikeArray(data, dtype or "float32")

    def all(self, value): return MLXLikeArray(np.all(value.data), "bool")
    def sum(self, value): return MLXLikeArray(np.sum(value.data), "int32")
    def abs(self, value): return MLXLikeArray(np.abs(value.data), value.dtype)
    def maximum(self, left, right): return MLXLikeArray(np.maximum(left.data, right), left.dtype)
    def max(self, value): return MLXLikeArray(np.max(value.data), value.dtype)
    def mean(self, value): return MLXLikeArray(np.mean(value.data), value.dtype)
    def square(self, value): return MLXLikeArray(np.square(value.data), value.dtype)
    def sqrt(self, value): return MLXLikeArray(np.sqrt(value.data), value.dtype)
    def take(self, values, indices): return MLXLikeArray(values.data[indices.data.astype(np.int64)], values.dtype)
    def eval(self, *values): return None


FAKE_MX = FakeMLXCore()


def _arr(shape, value, dtype):
    return FakeArray(np.full(shape, value, dtype=np.float32), dtype)


TRANSITIONS = (
    {"step_index": 0, "video_current_timestep": 0.0, "video_next_timestep": 0.07692307,
     "video_current_sigma": 1.0, "video_next_sigma": 0.9230769, "audio_current_timestep": 0.0,
     "audio_next_timestep": 0.25, "audio_current_sigma": 1.0, "audio_next_sigma": 0.75},
    {"step_index": 1, "video_current_timestep": 0.07692307, "video_next_timestep": 1.0,
     "video_current_sigma": 0.9230769, "video_next_sigma": 0.0, "audio_current_timestep": 0.25,
     "audio_next_timestep": 1.0, "audio_current_sigma": 0.75, "audio_next_sigma": 0.0},
)

CANONICAL_TIMESTEPS = {
    0: np.array([0.0], dtype=np.float32),
    1: np.array([0.07692307, 0.25], dtype=np.float32),
}
CANONICAL_TIMESTEP_INDICES = {
    0: np.array([0, 0, 0, 0], dtype=np.int32),
    1: np.array([0, 1, 1, 0], dtype=np.int32),
}


class FakeTransformer:
    def __init__(self, fail_at=None):
        self.calls = 0
        self.fail_at = fail_at
        self.seen_timesteps = []

    def __call__(self, video, audio, text, timestep, *args, **kwargs):
        self.calls += 1
        self.seen_timesteps.append(timestep)
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic transformer failure")
        return _arr(video.shape, 1.0 + self.calls, "float32"), _arr(audio.shape, 2.0 + self.calls, "float32")


class FakeScheduler:
    prediction_parameterization = "velocity"
    input_scaling = "identity"

    def __init__(self):
        self.transitions = TRANSITIONS
        self.num_inference_steps = 2
        self.step_index = None
        self.step_calls = 0

    def transition(self, step_index):
        if step_index not in (0, 1):
            raise IndexError("bad step")
        return dict(self.transitions[step_index])

    def prepare_model_input(self, video, audio, step_index):
        return video, audio

    def step(self, video_prediction, audio_prediction, video_sample, audio_sample, step_index):
        if self.step_index is not None and self.step_index != step_index:
            raise ValueError("scheduler cursor mismatch")
        self.step_calls += 1
        self.step_index = step_index + 1
        return (FakeArray(np.asarray(video_sample) + 0.01 * (step_index + 1), "bfloat16"),
                FakeArray(np.asarray(audio_sample) + 0.02 * (step_index + 1), "bfloat16"))


class CacheProvider:
    def __init__(self, reuse=False, overlap=False):
        self.calls = []
        self.releases = []
        self.reuse = reuse
        self.overlap = overlap
        self.cache = object()
        self.active = False
        self.next_session_token = 0
        self.last_session_token = None
        self.events = []

    def cache_for_step(self, step_index, timestep):
        if self.overlap and self.active:
            raise AssertionError("cache overlap")
        self.events.append("acquire-start")
        self.calls.append((step_index, timestep))
        self.next_session_token += 1
        self.last_session_token = self.next_session_token
        self.active = True
        self.events.append("acquire-complete")
        return self.cache if self.reuse else object()

    def release_step(self, step_index, cache):
        self.events.append("release-start")
        self.releases.append((step_index, cache))
        if not self.overlap:
            self.active = False
        self.events.append("release-complete")


class FakeRuntimeMemory:
    def __init__(self, active, cache):
        self.active = active
        self.cache = cache

    def clear_cache(self):
        self.cache = 0

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.cache


def inputs():
    return {
        "initial_video_latent": _arr((1, 1, 96), 0.1, "bfloat16"),
        "initial_audio_latent": _arr((1, 2, 32), 0.2, "bfloat16"),
        "text_embedding": _arr((1, 1, 5120), 0.3, "bfloat16"),
        "timestep_provider": lambda step, transition: (
            FakeArray(CANONICAL_TIMESTEPS[step], "float32"),
            FakeArray(CANONICAL_TIMESTEP_INDICES[step], "int32"),
        ),
        "timestep_indices": CANONICAL_TIMESTEP_INDICES[0],
        "token_tags": np.array([1, 2, 2, 0], dtype=np.int32),
        "position_ids": np.zeros((4, 3), dtype=np.float32),
        "video_indices": np.array([3], dtype=np.int32),
        "audio_indices": np.array([1, 2], dtype=np.int32),
        "text_indices": np.array([0], dtype=np.int32),
    }


def run_loop(*, transformer=None, scheduler=None, provider=None, **overrides):
    values = inputs()
    values.update(overrides)
    return denoise_loop(transformer or FakeTransformer(), scheduler or FakeScheduler(),
                        modulation_cache_provider=provider, **values)


def cache_record(step=0, token=1, start=1):
    return {"step_index": step, "cache_table_count": 50, "cache_session_token": token,
            "complete_cache_statistics": {},
            "events": [{"global_event_number": start + offset, "step_index": step,
                         "cache_session_token": token, "event": event}
                        for offset, event in enumerate(("acquire-start", "acquire-complete",
                                                        "release-start", "release-complete"))],
            "blocks_completed": 50, "sidecar_files_opened": 50, "unique_sidecars_opened": 50,
            "successful_payload_opens": 50, "completed_payload_releases": 50,
            "every_sidecar_released_before_next_opened": True, "sidecar_overlap_observed": False,
            "next_sidecar_opened_before_previous_release": False, "dense_temporary_projection_created": False,
            "sidecar_names": [f"block-{i:03d}.safetensors" for i in range(50)]}


class ProbeV04BContractTests(unittest.TestCase):
    def test_parser_exposes_exactly_two_subcommands(self):
        action = next(item for item in probe.build_parser()._actions if item.dest == "command")
        self.assertEqual(set(action.choices), {"create-reference", "compare-derived"})

    def test_canonical_loop_has_exactly_two_transitions(self):
        probe.validate_canonical_schedule(TRANSITIONS)

    def test_step_order_is_exactly_zero_one(self):
        self.assertEqual(probe.CANONICAL_STEP_INDICES, (0, 1))

    def test_artifact_schema_has_exactly_40_canonical_keys(self):
        self.assertEqual(len(probe.ARTIFACT_KEYS), 40)
        self.assertEqual(list(probe._canonical_inventory()), list(probe.ARTIFACT_KEYS))

    def test_canonical_timestep_shapes_match_deduplicated_steps(self):
        inventory = probe._canonical_inventory()
        self.assertEqual(inventory["step_0_timestep"]["shape"], [1])
        self.assertEqual(inventory["step_1_timestep"]["shape"], [2])
        self.assertEqual(inventory["step_0_timestep_indices"]["shape"], [4])
        self.assertEqual(inventory["step_1_timestep_indices"]["shape"], [4])

    def test_canonical_timestep_uniqueness_follows_modality_current_values(self):
        self.assertEqual(len(np.unique([TRANSITIONS[0]["video_current_timestep"],
                                        TRANSITIONS[0]["audio_current_timestep"]])), 1)
        self.assertEqual(len(np.unique([TRANSITIONS[1]["video_current_timestep"],
                                        TRANSITIONS[1]["audio_current_timestep"]])), 2)

    def test_extra_artifact_writer_key_fails(self):
        keys = list(probe.ARTIFACT_KEYS) + ["unexpected_tensor"]
        with self.assertRaisesRegex(ValueError, "extra"):
            probe.validate_artifact_tensor_keys(keys)

    def test_shuffled_loaded_mapping_passes_membership_and_is_canonicalized(self):
        loaded = {key: object() for key in reversed(probe.ARTIFACT_KEYS)}
        probe.validate_loaded_artifact_tensor_membership(loaded)
        arrays = probe._canonicalize_loaded_artifact_arrays(loaded)
        self.assertEqual(list(arrays), list(probe.ARTIFACT_KEYS))
        probe.validate_artifact_tensor_keys(arrays)

    def test_artifact_loader_canonicalizes_shuffled_mapping(self):
        loaded = {key: object() for key in reversed(probe.ARTIFACT_KEYS)}
        fake_mx = SimpleNamespace(load=lambda path: loaded)
        with patch.dict(sys.modules, {"mlx": SimpleNamespace(core=fake_mx), "mlx.core": fake_mx}):
            arrays = probe._load_artifact_arrays(Path("synthetic.safetensors"))
        self.assertEqual(list(arrays), list(probe.ARTIFACT_KEYS))

    def test_missing_loaded_key_fails_membership_validation(self):
        loaded = {key: object() for key in probe.ARTIFACT_KEYS[:-1]}
        with self.assertRaisesRegex(ValueError, "missing"):
            probe.validate_loaded_artifact_tensor_membership(loaded)

    def test_extra_loaded_key_fails_membership_validation(self):
        loaded = {key: object() for key in probe.ARTIFACT_KEYS}
        loaded["unexpected_tensor"] = object()
        with self.assertRaisesRegex(ValueError, "extra"):
            probe.validate_loaded_artifact_tensor_membership(loaded)

    def test_inventory_diagnostics_identify_missing_and_extra_keys(self):
        inventory = {key: dict(value) for key, value in probe._canonical_inventory().items()}
        missing_key = "step_1_updated_audio_latent"
        del inventory[missing_key]
        extra_key = "unexpected_tensor"
        inventory[extra_key] = {"shape": [1], "dtype": "float32"}
        with self.assertRaises(ValueError) as raised:
            probe.validate_artifact_tensor_inventory(inventory)
        message = str(raised.exception)
        self.assertIn(f"missing keys: ['{missing_key}']", message)
        self.assertIn(f"extra keys: ['{extra_key}']", message)
        self.assertIn("actual inventory:", message)
        self.assertIn("expected inventory:", message)

    def test_inventory_diagnostics_identify_specific_shape_mismatch(self):
        inventory = {key: dict(value) for key, value in probe._canonical_inventory().items()}
        key = "step_1_updated_video_latent"
        inventory[key]["shape"] = [1, 96]
        with self.assertRaises(ValueError) as raised:
            probe.validate_artifact_tensor_inventory(inventory)
        message = str(raised.exception)
        self.assertIn("shape mismatches:", message)
        self.assertIn(f"{key}: actual=[1, 96], expected=[1, 1, 96]", message)
        self.assertIn("dtype mismatches: []", message)

    def test_forced_step_zero_two_timestep_inventory_shape_fails(self):
        inventory = {key: dict(value) for key, value in probe._canonical_inventory().items()}
        inventory["step_0_timestep"]["shape"] = [2]
        with self.assertRaises(ValueError) as raised:
            probe.validate_artifact_tensor_inventory(inventory)
        self.assertIn("step_0_timestep: actual=[2], expected=[1]", str(raised.exception))

    def test_inventory_diagnostics_identify_specific_dtype_mismatch(self):
        inventory = {key: dict(value) for key, value in probe._canonical_inventory().items()}
        key = "step_1_timestep_indices"
        inventory[key]["dtype"] = "int64"
        with self.assertRaises(ValueError) as raised:
            probe.validate_artifact_tensor_inventory(inventory)
        message = str(raised.exception)
        self.assertIn("dtype mismatches:", message)
        self.assertIn(f"{key}: actual='int64', expected='int32'", message)
        self.assertIn("shape mismatches: []", message)

    def test_valid_canonical_inventory_passes_diagnostics(self):
        probe.validate_artifact_tensor_inventory(probe._canonical_inventory())

    def test_artifact_writer_emits_exactly_40_keys_and_step_one_duplicates(self):
        result = run_loop()
        values = inputs()
        arrays = {"initial_video_latent": values["initial_video_latent"],
                  "initial_audio_latent": values["initial_audio_latent"],
                  "text_input": values["text_embedding"],
                  **{key: values[key] for key in ("token_tags", "position_ids", "video_indices", "audio_indices",
                                                  "text_indices")}}
        class ArtifactMX:
            int32 = "int32"
            float32 = "float32"
            def array(self, value, dtype=None):
                return FakeArray(value, dtype or "float32")
        artifact = probe._artifact_arrays_from_result(
            ArtifactMX(), arrays,
            {0: FakeArray(CANONICAL_TIMESTEPS[0], "float32"), 1: FakeArray(CANONICAL_TIMESTEPS[1], "float32")},
            CANONICAL_TIMESTEP_INDICES,
            TRANSITIONS, result)
        self.assertEqual(len(artifact), 40)
        self.assertEqual(list(artifact), list(probe.ARTIFACT_KEYS))
        self.assertTrue(np.array_equal(np.asarray(artifact["step_1_updated_video_latent"]),
                                       np.asarray(artifact["final_video_latent"])))
        self.assertTrue(np.array_equal(np.asarray(artifact["step_1_updated_audio_latent"]),
                                       np.asarray(artifact["final_audio_latent"])))
        probe.validate_artifact_tensor_inventory({key: probe._shape_dtype(value) for key, value in artifact.items()})

    def test_transition_normalization_allows_only_selected_step_derivation(self):
        normalized = probe._transition_mapping({**TRANSITIONS[0]})
        self.assertEqual(normalized["selected_step_index"], 0)
        for field in probe.TRANSITION_FIELDS:
            if field == "selected_step_index":
                continue
            broken = dict(TRANSITIONS[0]); broken.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing fields"):
                probe._transition_mapping(broken)

    def test_artifact_writer_rejects_missing_transition_fields(self):
        result = run_loop()
        values = inputs()
        arrays = {"initial_video_latent": values["initial_video_latent"],
                  "initial_audio_latent": values["initial_audio_latent"],
                  "text_input": values["text_embedding"],
                  **{key: values[key] for key in ("token_tags", "position_ids", "video_indices", "audio_indices",
                                                  "text_indices")}}
        class ArtifactMX:
            int32 = "int32"; float32 = "float32"
            def array(self, value, dtype=None): return FakeArray(value, dtype or "float32")
        for field in probe.TRANSITION_FIELDS:
            if field == "selected_step_index":
                continue
            broken = [dict(item) for item in TRANSITIONS]
            broken[0].pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing fields"):
                probe._artifact_arrays_from_result(ArtifactMX(), arrays,
                    {0: FakeArray(CANONICAL_TIMESTEPS[0], "float32"), 1: FakeArray(CANONICAL_TIMESTEPS[1], "float32")},
                    CANONICAL_TIMESTEP_INDICES,
                    broken, result)

    def test_timestep_and_index_tampering_fails_reconstruction(self):
        layout = SimpleNamespace(sequence_length=4, video_indices=np.array([3], dtype=np.int32),
                                 audio_indices=np.array([1, 2], dtype=np.int32),
                                 num_condition_video_rows=0, num_condition_audio_rows=0)
        canonical = tuple(probe._transition_mapping(item) for item in TRANSITIONS)
        timesteps = {step: np.array(value, copy=True) for step, value in CANONICAL_TIMESTEPS.items()}
        indices = {step: np.array(value, copy=True) for step, value in CANONICAL_TIMESTEP_INDICES.items()}
        arrays = {"step_0_timestep": timesteps[0], "step_0_timestep_indices": indices[0],
                  "step_1_timestep": timesteps[1], "step_1_timestep_indices": indices[1]}
        probe.validate_timestep_reconstruction(arrays, layout, canonical, timesteps, indices)
        broken_timestep = dict(arrays)
        broken_timestep["step_0_timestep"] = np.array(timesteps[0], copy=True)
        broken_timestep["step_0_timestep"][0] += np.float32(0.1)
        with self.assertRaisesRegex(ValueError, "timestep"):
            probe.validate_timestep_reconstruction(broken_timestep, layout, canonical, timesteps, indices)
        broken_indices = dict(arrays)
        broken_indices["step_0_timestep_indices"] = np.array(indices[0], copy=True)
        broken_indices["step_0_timestep_indices"][0] = 1 - broken_indices["step_0_timestep_indices"][0]
        with self.assertRaisesRegex(ValueError, "timestep"):
            probe.validate_timestep_reconstruction(broken_indices, layout, canonical, timesteps, indices)

    def test_transformer_called_exactly_twice(self):
        transformer = FakeTransformer(); result = run_loop(transformer=transformer)
        self.assertEqual((transformer.calls, result.transformer_calls, result.completed_steps), (2, 2, 2))

    def test_scheduler_update_called_exactly_twice(self):
        scheduler = FakeScheduler(); run_loop(scheduler=scheduler)
        self.assertEqual(scheduler.step_calls, 2)

    def test_loop_returns_truthful_orchestration_counters(self):
        derived = run_loop(provider=CacheProvider())
        resident = run_loop()
        self.assertEqual((derived.transformer_calls, derived.scheduler_updates,
                          derived.cache_acquisitions, derived.cache_releases), (2, 2, 2, 2))
        self.assertEqual((resident.transformer_calls, resident.scheduler_updates,
                          resident.cache_acquisitions, resident.cache_releases), (2, 2, 0, 0))

    def test_cache_provider_called_exactly_twice(self):
        provider = CacheProvider(); run_loop(provider=provider)
        self.assertEqual(len(provider.calls), 2)

    def test_cache_provider_receives_distinct_timestep_tensors(self):
        provider = CacheProvider(); run_loop(provider=provider)
        self.assertFalse(np.array_equal(np.asarray(provider.calls[0][1]), np.asarray(provider.calls[1][1])))

    def test_step_one_inputs_equal_step_zero_updates(self):
        result = run_loop()
        self.assertTrue(np.array_equal(np.asarray(result.step_receipts[1].input_video_latent), np.asarray(result.step_receipts[0].updated_video_latent)))
        self.assertTrue(np.array_equal(np.asarray(result.step_receipts[1].input_audio_latent), np.asarray(result.step_receipts[0].updated_audio_latent)))

    def test_caller_owned_initial_latents_are_not_mutated(self):
        values = inputs(); video = values["initial_video_latent"]; before = np.asarray(video).copy()
        denoise_loop(FakeTransformer(), FakeScheduler(), modulation_cache_provider=None, **values)
        self.assertTrue(np.array_equal(np.asarray(video), before))

    def test_zero_requested_steps_fail(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            run_loop(step_indices=())

    def test_too_many_requested_steps_fail(self):
        with self.assertRaisesRegex(ValueError, "only 2"):
            run_loop(step_indices=(0, 1, 2))

    def test_reordered_steps_fail(self):
        with self.assertRaisesRegex(ValueError, "ordering"):
            run_loop(step_indices=(1, 0))

    def test_duplicate_steps_fail(self):
        with self.assertRaisesRegex(ValueError, "ordering"):
            run_loop(step_indices=(0, 0))

    def test_scheduler_cursor_mismatch_fails(self):
        scheduler = FakeScheduler(); scheduler.step_index = 1
        with self.assertRaisesRegex(ValueError, "cursor mismatch"):
            run_loop(scheduler=scheduler)

    def test_video_and_audio_transition_state_remain_separate(self):
        self.assertNotEqual(TRANSITIONS[0]["video_next_sigma"], TRANSITIONS[0]["audio_next_sigma"])

    def test_both_modality_shifts_affect_step_zero(self):
        self.assertTrue(0 < TRANSITIONS[0]["video_next_sigma"] < 1)
        self.assertTrue(0 < TRANSITIONS[0]["audio_next_sigma"] < 1)

    def test_step_one_is_terminal_for_both_modalities(self):
        self.assertEqual((TRANSITIONS[1]["video_next_sigma"], TRANSITIONS[1]["audio_next_sigma"]), (0.0, 0.0))

    def test_step_zero_cache_releases_before_step_one_acquisition(self):
        provider = CacheProvider(); run_loop(provider=provider)
        self.assertEqual([item[0] for item in provider.releases], [0, 1])
        self.assertEqual(provider.events, ["acquire-start", "acquire-complete", "release-start",
                                          "release-complete", "acquire-start", "acquire-complete",
                                          "release-start", "release-complete"])
        self.assertFalse(provider.active)

    def test_cache_overlap_fails(self):
        provider = CacheProvider(overlap=True)
        with self.assertRaisesRegex(AssertionError, "overlap"):
            run_loop(provider=provider)
        self.assertEqual([item[0] for item in provider.releases], [0])

    def test_cache_reuse_requires_distinct_session_tokens(self):
        result = run_loop(provider=CacheProvider(reuse=True))
        self.assertEqual(result.cache_acquisitions, 2)

    def test_each_cache_lifecycle_count_is_checked_independently(self):
        record = cache_record()
        for key in record:
            bad = [dict(record), dict(record)]
            bad[0][key] = (not record[key]) if isinstance(record[key], bool) else (
                [] if key == "events" else (0 if key == "cache_session_token" else 49))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "lifecycle"):
                probe.validate_cache_lifecycle([bad[0], cache_record(step=1, token=2, start=5)])

    def test_total_cache_session_count_mismatch_fails(self):
        record = cache_record()
        record["sidecar_files_opened"] = 49
        with self.assertRaises(ValueError):
            second = cache_record(step=1, token=2, start=5)
            second["sidecar_files_opened"] = 51
            probe.validate_cache_lifecycle([record, second])

    def test_missing_step_receipt_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            probe.validate_step_receipts([])

    def test_extra_step_receipt_fails(self):
        with self.assertRaisesRegex(ValueError, "exactly two"):
            probe.validate_step_receipts([SimpleNamespace(step_index=i) for i in range(3)])

    def test_wrong_step_zero_timestep_fails_at_transformer_boundary(self):
        values = inputs(); values["timestep_provider"] = lambda step, transition: FakeArray([0.5, 0.25], "float32")
        with self.assertRaisesRegex(ValueError, "current timestep"):
            denoise_loop(FakeTransformer(), FakeScheduler(), modulation_cache_provider=None, **values)

    def test_wrong_step_one_timestep_fails_at_transformer_boundary(self):
        values = inputs(); values["timestep_provider"] = lambda step, transition: FakeArray([0.0, 0.25], "float32")
        with self.assertRaisesRegex(ValueError, "same timestep"):
            denoise_loop(FakeTransformer(), FakeScheduler(), modulation_cache_provider=None, **values)

    def test_step_one_latent_chain_mismatch_fails(self):
        result = run_loop()
        broken = list(result.step_receipts)
        broken[1] = SimpleNamespace(**{**vars(broken[1]), "input_video_latent": _arr((1, 1, 96), 99, "bfloat16")})
        with self.assertRaisesRegex(ValueError, "step-1 video"):
            probe.validate_step_receipts(broken)

    def test_each_exact_parity_gate_is_independent(self):
        metrics = {name: {"exact_equality": True} for name in probe.PARITY_COMPARISONS}
        for name in probe.PARITY_COMPARISONS:
            broken = {key: dict(value) for key, value in metrics.items()}; broken[name]["exact_equality"] = False
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "exact parity"):
                probe.validate_exact_parity(broken)

    def test_final_latent_mismatch_is_included_in_parity(self):
        self.assertIn("final_video_latent", probe.PARITY_COMPARISONS)
        self.assertIn("final_audio_latent", probe.PARITY_COMPARISONS)

    def test_report_exists_before_parity_validation(self):
        metrics = {name: {"exact_equality": True} for name in probe.PARITY_COMPARISONS}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "must exist"):
                probe.validate_report_before_parity(path, metrics)
            path.write_text(json.dumps(metrics))
            probe.validate_report_before_parity(path, metrics)

    def test_step_zero_evidence_survives_step_one_failure(self):
        provider = CacheProvider(); transformer = FakeTransformer(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            run_loop(transformer=transformer, provider=provider)
        self.assertEqual(transformer.calls, 2)
        self.assertEqual([step for step, _ in provider.calls], [0, 1])
        self.assertEqual([step for step, _ in provider.releases], [0, 1])

    def test_cleanup_occurs_before_reraise(self):
        provider = CacheProvider(); transformer = FakeTransformer(fail_at=1)
        with self.assertRaises(RuntimeError):
            run_loop(transformer=transformer, provider=provider)
        self.assertFalse(provider.active)
        self.assertEqual(len(provider.releases), 1)

    def test_success_result_is_returned_after_cache_cleanup(self):
        provider = CacheProvider(); result = run_loop(provider=provider)
        self.assertFalse(provider.active)
        self.assertEqual(result.completed_steps, 2)

    def test_existing_file_receipt_is_truthful(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"; path.write_text("old")
            probe._write_json(path, {"status": "failed"})
            self.assertEqual(json.loads(path.read_text())["status"], "failed")

    def test_full_metadata_round_trip_has_strict_key_contract(self):
        metadata = {key: None for key in probe.REFERENCE_METADATA_REQUIRED_KEYS}
        encoded = json.loads(json.dumps(metadata, sort_keys=True))
        probe.validate_metadata_keys(encoded)

    def test_malformed_resident_shard_index_values_fail_clearly(self):
        self.assertIn("resident_checkpoint_fingerprint", probe.REFERENCE_METADATA_REQUIRED_KEYS)
        self.assertIn("fingerprint_method", probe.REFERENCE_METADATA_REQUIRED_KEYS)

    def test_artifact_key_order_is_canonical(self):
        probe.validate_artifact_tensor_keys(probe.ARTIFACT_KEYS)

    def test_reordered_artifact_keys_fail(self):
        keys = {key: object() for key in reversed(probe.ARTIFACT_KEYS)}
        with self.assertRaisesRegex(ValueError, "order"):
            probe.validate_artifact_tensor_keys(keys)

    def test_metadata_tensor_key_order_remains_strict(self):
        metadata_keys = list(probe.ARTIFACT_KEYS)
        metadata_keys[0], metadata_keys[1] = metadata_keys[1], metadata_keys[0]
        with self.assertRaisesRegex(ValueError, "order"):
            probe.validate_artifact_tensor_keys(metadata_keys)

    def test_deterministic_inputs_are_stable_and_nonzero(self):
        self.assertEqual(probe.deterministic_input_values(4, 0), probe.deterministic_input_values(4, 0))
        self.assertTrue(all(value > 0 for value in probe.deterministic_input_values(128, 2)))

    def test_deterministic_input_invalid_salt_fails(self):
        with self.assertRaisesRegex(ValueError, "salt"):
            probe.deterministic_input_values(1, 3)

    def test_deterministic_input_empty_fails(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            probe.deterministic_input_values(0, 0)

    def test_all_eight_metric_fields_are_computed(self):
        left = _arr((1, 1, 4), 1, "bfloat16"); right = _arr((1, 1, 4), 1, "bfloat16")
        metrics = probe.metric_report(left, right)
        for key in ("exact_equality", "mismatched_element_count_exact", "maximum_absolute_difference",
                    "mean_absolute_difference", "root_mean_square_difference", "maximum_relative_difference",
                    "allclose", "shape", "dtype", "element_count"):
            self.assertIn(key, metrics)

    def test_nonexact_allclose_does_not_pass_exact_gate(self):
        left = _arr((1, 1, 4), 1, "bfloat16"); right = _arr((1, 1, 4), 1.000001, "bfloat16")
        metric = probe.metric_report(left, right); self.assertFalse(metric["exact_equality"])
        metrics = {name: {"exact_equality": True} for name in probe.PARITY_COMPARISONS}
        metrics["final_video_latent"] = metric
        with self.assertRaises(ValueError):
            probe.validate_exact_parity(metrics)

    def test_completed_result_final_latents_are_step_one_updates(self):
        result = run_loop()
        self.assertTrue(np.array_equal(np.asarray(result.final_video_latent), np.asarray(result.step_receipts[-1].updated_video_latent)))
        self.assertTrue(np.array_equal(np.asarray(result.final_audio_latent), np.asarray(result.step_receipts[-1].updated_audio_latent)))

    def test_receipt_timestep_tensors_are_stored_per_step(self):
        result = run_loop()
        self.assertEqual(len(result.step_receipts), 2)
        self.assertFalse(np.array_equal(np.asarray(result.step_receipts[0].timestep), np.asarray(result.step_receipts[1].timestep)))

    def test_partial_step_failure_preserves_step_zero_evidence_and_counters(self):
        transformer = FakeTransformer(fail_at=2)
        with self.assertRaisesRegex(RuntimeError, "synthetic") as caught:
            run_loop(transformer=transformer, provider=CacheProvider())
        error = caught.exception
        self.assertEqual(error.denoise_completed_steps, 1)
        self.assertEqual(len(error.denoise_step_receipts), 1)
        receipt = error.denoise_step_receipts[0]
        self.assertEqual(receipt.step_index, 0)
        self.assertIsNotNone(receipt.video_prediction)
        self.assertIsNotNone(receipt.updated_video_latent)
        self.assertEqual((error.denoise_transformer_calls, error.denoise_scheduler_updates), (2, 1))

    def test_mlxsafe_receipt_validation_and_metrics_never_call_array(self):
        first = MLXLikeArray(np.ones((1, 1, 4)), "bfloat16")
        second = MLXLikeArray(np.ones((1, 1, 4)), "bfloat16")
        receipt_zero = SimpleNamespace(step_index=0, input_video_latent=first, updated_video_latent=second,
                                       input_audio_latent=first, updated_audio_latent=second,
                                       video_current_timestep=0.0, video_next_timestep=0.1,
                                       video_current_sigma=1.0, video_next_sigma=0.5,
                                       audio_current_timestep=0.0, audio_next_timestep=0.2,
                                       audio_current_sigma=1.0, audio_next_sigma=0.5)
        receipt_one = SimpleNamespace(step_index=1, input_video_latent=second, updated_video_latent=first,
                                      input_audio_latent=second, updated_audio_latent=first,
                                      video_current_timestep=0.1, video_next_timestep=1.0,
                                      video_current_sigma=0.5, video_next_sigma=0.0,
                                      audio_current_timestep=0.2, audio_next_timestep=1.0,
                                      audio_current_sigma=0.5, audio_next_sigma=0.0)
        probe.validate_step_receipts((receipt_zero, receipt_one))
        metrics = probe.metric_report(first, second)
        self.assertTrue(metrics["exact_equality"])
        self.assertEqual(metrics["mismatched_element_count_exact"], 0)

    def test_per_step_block_observations_are_independent(self):
        expected = [list(range(50)), list(range(50))]
        probe.validate_per_step_block_observations(expected)
        with self.assertRaisesRegex(ValueError, "per-step"):
            probe.validate_per_step_block_observations([list(range(100))])

    def test_top_level_transformer_observer_buckets_calls_independently(self):
        class Block:
            def __call__(self, *args, **kwargs): return None
        class Dit:
            def __init__(self): self.blocks = [Block() for _ in range(50)]
            def __call__(self, *args, **kwargs):
                for block in self.blocks: block()
        dit = Dit()
        with probe.observe_transformer_block_execution(dit) as observer:
            observer()
            observer()
            self.assertIs(observer.config if hasattr(observer, "config") else dit, dit)
        self.assertEqual(observer.observations, [list(range(50)), list(range(50))])

    def test_observer_methods_do_not_close_over_transformer_parameter(self):
        class Block:
            def __call__(self, *args, **kwargs): return None
        class Dit:
            def __init__(self): self.blocks = [Block()]
            def __call__(self, *args, **kwargs): self.blocks[0]()
        with probe.observe_transformer_block_execution(Dit()) as observer:
            self.assertNotIn("dit", observer.__call__.__code__.co_freevars)
            self.assertNotIn("dit", observer.__getattr__.__code__.co_freevars)
            self.assertEqual(observer._transformer.__class__.__name__, "Dit")

    def test_observer_model_reference_is_none_at_context_exit(self):
        class Dit:
            blocks = []
            def __call__(self, *args, **kwargs): return None
        dit = Dit()
        with probe.observe_transformer_block_execution(dit) as observer:
            self.assertIs(observer._transformer, dit)
        self.assertIsNone(observer._transformer)

    def test_observation_buckets_survive_observer_detachment(self):
        class Block:
            def __call__(self, *args, **kwargs): return None
        class Dit:
            def __init__(self): self.blocks = [Block() for _ in range(2)]
            def __call__(self, *args, **kwargs):
                for block in self.blocks: block()
        with probe.observe_transformer_block_execution(Dit()) as observer:
            observer()
        self.assertEqual(observer.observations, [[0, 1]])

    def test_calling_detached_observer_fails_clearly(self):
        class Dit:
            blocks = []
            def __call__(self, *args, **kwargs): return None
        with probe.observe_transformer_block_execution(Dit()) as observer:
            pass
        with self.assertRaisesRegex(RuntimeError, "detached"):
            observer()

    def test_transformer_can_expire_after_context_exit_and_caller_clear(self):
        class Dit:
            blocks = []
            def __call__(self, *args, **kwargs): return None
        dit = Dit()
        reference = weakref.ref(dit)
        with probe.observe_transformer_block_execution(dit) as observer:
            pass
        self.assertIsNone(observer._transformer)
        dit = None
        import gc
        gc.collect()
        self.assertIsNone(reference())

    def test_resident_cleanup_clears_observer_before_release(self):
        source = inspect.getsource(probe._run_resident)
        cleanup = source[source.rindex("finally:"):]
        self.assertLess(cleanup.index("observer = None"), cleanup.index("_release_runtime"))

    def test_derived_cleanup_clears_observer_before_release(self):
        source = inspect.getsource(probe.cmd_compare_derived)
        cleanup = source[source.rindex("finally:"):]
        self.assertLess(cleanup.index("observer = None"), cleanup.index("_release_runtime"))

    def test_derived_cleanup_clears_rebuilt_timetable_and_layout_references(self):
        source = inspect.getsource(probe.cmd_compare_derived)
        cleanup = source[source.rindex("finally:"):]
        for name in ("rebuilt_layout", "rebuilt_transitions", "rebuilt_timesteps", "rebuilt_indices",
                     "canonical", "derived_config", "inventory"):
            self.assertIn(name, cleanup)
        self.assertLess(cleanup.index("rebuilt_layout"), cleanup.index("_release_runtime"))

    def test_allocator_cache_zero_with_model_scale_active_memory_fails_release(self):
        runtime = FakeRuntimeMemory(active=64 * 1024 * 1024, cache=0)
        with self.assertRaisesRegex(RuntimeError, "active memory"):
            probe._release_runtime(runtime, active_memory_baseline={"active": 1 * 1024 * 1024})

    def test_active_memory_within_baseline_plus_tolerance_passes_release(self):
        baseline = 8 * 1024 * 1024
        runtime = FakeRuntimeMemory(active=baseline + probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES, cache=4096)
        result = probe._release_runtime(runtime, active_memory_baseline={"active": baseline})
        self.assertEqual(result["allocator_cache_after"], 0)
        self.assertTrue(result["active_memory_gate_available"])

    def test_resident_release_failure_cannot_report_runtime_success(self):
        source = inspect.getsource(probe._run_resident)
        self.assertIn("if failure is None:\n                failure = _detach(release_exc)", source)
        self.assertIn('receipt["status"] = "runtime-complete" if failure is None else "failed"', source)

    def test_derived_release_failure_cannot_report_parity_success(self):
        source = inspect.getsource(probe.cmd_compare_derived)
        self.assertIn("if failure is None:\n                    failure = _detach(release_exc)", source)
        self.assertIn("parity_validated = False", source[source.rindex("finally:"):])
        self.assertIn('report["status"] = "passed" if parity_validated and failure is None else "failed"', source)

    def test_51_49_redistribution_cannot_pass_as_two_steps(self):
        class Block:
            def __call__(self, *args, **kwargs): return None
        class Dit:
            def __init__(self): self.blocks = [Block() for _ in range(50)]; self.calls = 0
            def __call__(self, *args, **kwargs):
                self.calls += 1
                count = 51 if self.calls == 1 else 49
                for block in (self.blocks + [self.blocks[0]])[:count]: block()
        with probe.observe_transformer_block_execution(Dit()) as observer:
            observer(); observer()
        with self.assertRaisesRegex(ValueError, "per-step"):
            probe.validate_per_step_block_observations(observer.observations)

    def test_malformed_resident_shard_names_are_rejected(self):
        for value in (7, "", "../model.safetensors", "nested/model.safetensors"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                probe.validate_resident_shard_names({"weight": value})
        with self.assertRaises(ValueError):
            probe.validate_resident_shard_names({})

    def test_cache_table_count_is_actual_and_missing_evidence_fails(self):
        valid = [cache_record(), cache_record(step=1, token=2, start=5)]
        probe.validate_cache_lifecycle(valid)
        for field, value in (("cache_table_count", 0), ("cache_table_count", None), ("cache_session_token", None)):
            broken = [dict(item) for item in valid]
            broken[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                probe.validate_cache_lifecycle(broken)

    def test_cache_events_carry_global_identity_and_append_in_order(self):
        records = [cache_record(), cache_record(step=1, token=2, start=5)]
        probe.validate_cache_lifecycle(records)
        self.assertEqual([event["global_event_number"] for event in records[0]["events"]], [1, 2, 3, 4])
        self.assertEqual([event["global_event_number"] for event in records[1]["events"]], [5, 6, 7, 8])
        self.assertLess(records[0]["events"][-1]["global_event_number"], records[1]["events"][0]["global_event_number"])

    def test_duplicate_cache_session_token_fails(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            probe.validate_cache_lifecycle([cache_record(), cache_record(step=1, token=1, start=5)])

    def test_release_failure_has_no_release_complete_event(self):
        broken = cache_record()
        broken["events"] = broken["events"][:3]
        with self.assertRaisesRegex(ValueError, "events"):
            probe.validate_cache_lifecycle([broken, cache_record(step=1, token=2, start=5)])

    def test_derived_preload_failure_does_not_import_or_call_load_dit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.safetensors"; artifact.write_bytes(b"artifact")
            metadata = root / "metadata.json"
            metadata.write_text(json.dumps({"artifact_sha256": "wrong"}))
            report = root / "report.json"
            args = SimpleNamespace(original=str(root / "original"), derived=str(root / "derived"),
                                   artifact=str(artifact), metadata=str(metadata), report=str(report))
            source = inspect.getsource(probe.cmd_compare_derived)
            self.assertLess(source.index("validate_metadata_keys"), source.index("_load_derived_transformer"))
            with patch.object(probe, "_load_derived_transformer", side_effect=AssertionError("must not load")) as load:
                with self.assertRaises(ValueError):
                    probe.cmd_compare_derived(args)
            load.assert_not_called()

    def test_strict_transition_binding_rejects_each_tampered_field(self):
        scheduler = SimpleNamespace(transition=lambda step: SimpleNamespace(**TRANSITIONS[step]))
        metadata = {"transitions": [dict(item) for item in TRANSITIONS]}
        artifact = {}
        for step, transition in enumerate(TRANSITIONS):
            artifact[probe._transition_tensor_key(step, "selected_step_index")] = np.array(step, dtype=np.int32)
            for field in probe.TRANSITION_FIELDS:
                if field != "selected_step_index":
                    artifact[probe._transition_tensor_key(step, field)] = np.array(transition[field], dtype=np.float32)
        probe.validate_transition_bindings(metadata, artifact, scheduler)
        for step in range(2):
            for field in probe.TRANSITION_FIELDS:
                broken = {key: np.array(value, copy=True) for key, value in artifact.items()}
                broken_metadata = {"transitions": [dict(item) for item in TRANSITIONS]}
                if field == "selected_step_index":
                    broken[probe._transition_tensor_key(step, field)] = np.array(step + 1, dtype=np.int32)
                else:
                    broken_metadata["transitions"][step][field] = float(broken_metadata["transitions"][step][field]) + 0.1
                with self.subTest(step=step, field=field), self.assertRaises(ValueError):
                    probe.validate_transition_bindings(broken_metadata, broken, scheduler)


if __name__ == "__main__":
    unittest.main()

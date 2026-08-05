"""MLX-free v0.4a scheduler, denoise, artifact, and cleanup contract tests."""

from __future__ import annotations

import importlib.util
import inspect
import json
from contextlib import ExitStack
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_v04a_one_step", ROOT / "scripts" / "probe_v04a_one_step.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

from minimax_h3_mlx.denoise import _exact_equal, _finite, one_step_denoise  # noqa: E402
from minimax_h3_mlx.scheduler import (  # noqa: E402
    MiniMaxH3MultimodalScheduler,
    MiniMaxH3ScheduleTransition,
)


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


def fake_inputs():
    return {
        "video_latent": FakeArray(np.full((1, 1, 96), 0.1), "bfloat16"),
        "audio_latent": FakeArray(np.full((1, 2, 32), 0.2), "bfloat16"),
        "text_embedding": FakeArray(np.full((1, 1, 5120), 0.3), "bfloat16"),
        "timestep": FakeArray([0.0, 0.25], "float32"),
        "timestep_indices": np.array([0, 0, 1, 1], dtype=np.int32),
        "token_tags": np.array([1, 2, 2, 0], dtype=np.int32),
        "position_ids": np.zeros((4, 3), dtype=np.float32),
        "video_indices": np.array([3], dtype=np.int32),
        "audio_indices": np.array([1, 2], dtype=np.int32),
        "text_indices": np.array([0], dtype=np.int32),
        "step_index": 0,
    }


MULTIMODAL_TRANSITION = {
    "step_index": 0,
    "video_current_timestep": 0.0,
    "video_next_timestep": 0.076923072,
    "video_current_sigma": 1.0,
    "video_next_sigma": 0.923076928,
    "audio_current_timestep": 0.0,
    "audio_next_timestep": 0.25,
    "audio_current_sigma": 1.0,
    "audio_next_sigma": 0.75,
}


class FakeTransformer:
    def __init__(self, *, nonfinite=False):
        self.calls = 0
        self.seen = None
        self.nonfinite = nonfinite

    def __call__(self, video, audio, text, *args, **kwargs):
        self.calls += 1
        self.seen = (video, audio, text, args, kwargs)
        video_data = np.full(video.shape, np.nan if self.nonfinite else 1.0)
        audio_data = np.full(audio.shape, np.nan if self.nonfinite else 2.0)
        return FakeArray(video_data, "float32"), FakeArray(audio_data, "float32")


class FakeScheduler:
    prediction_parameterization = "velocity"
    input_scaling = "identity"

    def __init__(self, *, unchanged_video=False, unchanged_audio=False, nonfinite_video=False, nonfinite_audio=False):
        self.prepare_calls = 0
        self.step_calls = 0
        self.seen = None
        self.unchanged_video = unchanged_video
        self.unchanged_audio = unchanged_audio
        self.nonfinite_video = nonfinite_video
        self.nonfinite_audio = nonfinite_audio

    def transition(self, step_index):
        if step_index != 0:
            raise IndexError("bad step")
        return dict(MULTIMODAL_TRANSITION)

    def prepare_model_input(self, video, audio, step_index):
        self.prepare_calls += 1
        return video, audio

    def step(self, video_prediction, audio_prediction, video_sample, audio_sample, step_index):
        self.step_calls += 1
        self.seen = (video_prediction, audio_prediction, video_sample, audio_sample, step_index)
        video_data = np.asarray(video_sample).copy() if self.unchanged_video else np.asarray(video_sample) + 0.01
        audio_data = np.asarray(audio_sample).copy() if self.unchanged_audio else np.asarray(audio_sample) + 0.02
        if self.nonfinite_video:
            video_data[...] = np.nan
        if self.nonfinite_audio:
            audio_data[...] = np.nan
        return FakeArray(video_data, "bfloat16"), FakeArray(audio_data, "bfloat16")


def run_one(transformer=None, scheduler=None, **overrides):
    values = fake_inputs()
    values.update(overrides)
    return one_step_denoise(transformer or FakeTransformer(), scheduler or FakeScheduler(), **values)


class ScalarScheduler:
    def __init__(self, sigmas, timesteps, label):
        self.sigmas = np.asarray(sigmas, dtype=np.float32)
        self.timesteps = np.asarray(timesteps, dtype=np.float32)
        self.num_inference_steps = len(self.sigmas) - 1
        self._step_index = None
        self.label = label
        self.calls = []

    @property
    def shift(self):
        return 12.0 if self.label == "video" else 3.0

    @property
    def step_index(self):
        return self._step_index

    def transition(self, step_index):
        return MiniMaxH3ScheduleTransition(
            step_index=step_index,
            current_timestep=float(self.timesteps[step_index]),
            next_timestep=float(1.0 - self.sigmas[step_index + 1]),
            current_sigma=float(self.sigmas[step_index]),
            next_sigma=float(self.sigmas[step_index + 1]),
        )

    def step(self, model_output, timestep, sample):
        self.calls.append((timestep, sample.shape))
        index = int(np.where(self.timesteps == np.float32(timestep))[0][0])
        if self._step_index is not None and self._step_index != index:
            raise ValueError("fake scalar cursor mismatch")
        self._step_index = index + 1
        return np.asarray(sample) + np.asarray(model_output)


def production_adapter():
    video = ScalarScheduler([1.0, 0.8, 0.0], [0.0, 0.2], "video")
    audio = ScalarScheduler([1.0, 0.6, 0.0], [0.0, 0.4], "audio")
    return MiniMaxH3MultimodalScheduler(video, audio), video, audio


class NoNumpyMLXArray:
    __mlx_array__ = True

    def __init__(self, data, dispatch):
        self.data = np.asarray(data)
        self.__mlx_core__ = dispatch

    @property
    def shape(self):
        return self.data.shape

    def __eq__(self, other):
        return self.data == getattr(other, "data", other)

    def __array__(self, *args, **kwargs):
        raise AssertionError("NumPy conversion was attempted on an MLX array")


class MinimalMLXDispatch:
    def isfinite(self, value):
        return value.data == value.data

    def all(self, value):
        return np.all(value)

    def eval(self, *values):
        return None


class DispatchArray(NoNumpyMLXArray):
    def astype(self, dtype):
        return DispatchArray(self.data.astype(np.float32), self.__mlx_core__, str(dtype))

    def __init__(self, data, dispatch, dtype="bfloat16"):
        super().__init__(data, dispatch)
        self.dtype = dtype

    def _other(self, value):
        return getattr(value, "data", value)

    def __add__(self, other):
        return DispatchArray(self.data + self._other(other), self.__mlx_core__, self.dtype)

    __radd__ = __add__

    def __sub__(self, other):
        return DispatchArray(self.data - self._other(other), self.__mlx_core__, self.dtype)

    def __mul__(self, other):
        return DispatchArray(self.data * self._other(other), self.__mlx_core__, self.dtype)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return DispatchArray(self.data / self._other(other), self.__mlx_core__, self.dtype)

    def __le__(self, other):
        return DispatchArray(self.data <= self._other(other), self.__mlx_core__, "bool")

    def __ne__(self, other):
        return DispatchArray(self.data != self._other(other), self.__mlx_core__, "bool")

    def __eq__(self, other):
        return DispatchArray(self.data == self._other(other), self.__mlx_core__, "bool")

    def item(self):
        return self.data.item()


class MetricMLXDispatch(MinimalMLXDispatch):
    float32 = "float32"
    int32 = "int32"

    def _wrap(self, value, dtype=None):
        return DispatchArray(value, self, dtype or "float32")

    def all(self, value):
        return self._wrap(np.all(value.data), "bool")

    def abs(self, value):
        return self._wrap(np.abs(value.data), value.dtype)

    def maximum(self, left, right):
        return self._wrap(np.maximum(left.data, right), left.dtype)

    def max(self, value):
        return self._wrap(np.max(value.data), value.dtype)

    def mean(self, value):
        return self._wrap(np.mean(value.data), value.dtype)

    def square(self, value):
        return self._wrap(np.square(value.data), value.dtype)

    def sqrt(self, value):
        return self._wrap(np.sqrt(value.data), value.dtype)

    def sum(self, value):
        return self._wrap(np.sum(value.data), "int32")


class ProbeV04AContractTests(unittest.TestCase):
    def test_parser_exposes_exactly_two_subcommands(self):
        parser = probe.build_parser()
        action = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(set(action.choices), {"create-reference", "compare-derived"})

    def test_artifact_inventory_exposes_all_multimodal_transition_fields(self):
        self.assertEqual(len(probe.ARTIFACT_KEYS), 23)
        self.assertTrue(set(probe.TRANSITION_FIELDS).issubset(probe.ARTIFACT_KEYS))
        probe.validate_artifact_tensor_keys(list(probe.ARTIFACT_KEYS))

    def test_metadata_inventory_rejects_missing_and_unexpected_keys(self):
        metadata = {key: None for key in probe.METADATA_KEYS}
        probe.validate_metadata_keys(metadata)
        with self.assertRaises(ValueError):
            probe.validate_metadata_keys({key: value for key, value in metadata.items() if key != "audio_next_sigma"})
        with self.assertRaises(ValueError):
            probe.validate_metadata_keys({**metadata, "unexpected": True})

    def test_scheduler_contract_requires_all_modality_fields(self):
        broken = dict(MULTIMODAL_TRANSITION)
        del broken["audio_next_timestep"]
        scheduler = FakeScheduler()
        scheduler.transition = lambda _: broken
        with self.assertRaisesRegex(ValueError, "audio_next_timestep"):
            run_one(scheduler=scheduler)

    def test_tampered_video_next_timestep_fails_strict_metadata_validation(self):
        metadata, transition, inventory = self._metadata_fixture()
        metadata["video_next_timestep"] += 0.1
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)

    def test_tampered_audio_next_timestep_fails_strict_metadata_validation(self):
        metadata, transition, inventory = self._metadata_fixture()
        metadata["audio_next_timestep"] += 0.1
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)

    def test_tampered_video_sigma_fails_strict_metadata_validation(self):
        metadata, transition, inventory = self._metadata_fixture()
        metadata["video_next_sigma"] += 0.1
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)

    def test_tampered_audio_sigma_fails_strict_metadata_validation(self):
        metadata, transition, inventory = self._metadata_fixture()
        metadata["audio_next_sigma"] += 0.1
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)

    def test_full_canonical_metadata_survives_sorted_json_round_trip(self):
        metadata, transition, inventory = self._metadata_fixture()
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "metadata.json"
            path.write_text(json.dumps(metadata, sort_keys=True))
            restored = json.loads(path.read_text())
            self._validate_fixture(restored, transition, inventory)

    def test_missing_and_unexpected_inventory_keys_fail_strict_validation(self):
        metadata, transition, inventory = self._metadata_fixture()
        metadata["tensor_inventory"] = {key: value for key, value in inventory.items()
                                         if key != "audio_next_sigma"}
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)
        metadata, transition, inventory = self._metadata_fixture()
        metadata["tensor_inventory"] = {**inventory, "unexpected": {"shape": [1], "dtype": "float32"}}
        with self.assertRaises(ValueError):
            self._validate_fixture(metadata, transition, inventory)

    def _metadata_fixture(self):
        transition = dict(MULTIMODAL_TRANSITION)
        inventory = {key: {"shape": [1], "dtype": "float32"} for key in probe.ARTIFACT_KEYS}
        metadata = {key: None for key in probe.METADATA_KEYS}
        with tempfile.TemporaryDirectory() as temp:
            original, derived = Path(temp) / "r", Path(temp) / "d"
            metadata.update({
                "artifact_format": probe.ARTIFACT_FORMAT, "artifact_schema_version": 1,
                "artifact_file_format": "safetensors", "reference_checkpoint": str(original.resolve()),
                "derived_checkpoint": str(derived.resolve()), "fingerprint_method": probe.FINGERPRINT_METHOD,
                "deterministic_input_method": probe.DETERMINISTIC_INPUT_METHOD,
                "tensor_keys": list(probe.ARTIFACT_KEYS), "packed_layout": {"sequence_length": 4},
                "scheduler_identity": "MiniMaxH3MultimodalScheduler",
                "scheduler_configuration": {"identity": "MiniMaxH3MultimodalScheduler",
                    "video": {"identity": "MiniMaxH3Scheduler", "shift": 12.0},
                    "audio": {"identity": "MiniMaxH3Scheduler", "shift": 3.0},
                    "num_inference_steps": 2, "prediction_parameterization": "velocity",
                    "input_scaling": "identity", "update_method": "rectified-flow-euler-data-ward-velocity-v1"},
                "prediction_parameterization": "velocity", "inference_step_count": 2,
                **probe.artifact_transition_values(transition), "timestep_row_convention": {"text": "video_current_timestep",
                    "conditioning_video": "0.999", "conditioning_audio": "0.999",
                    "target_video": "video_current_timestep", "target_audio": "audio_current_timestep"},
                "model_input_scaling": "identity", "update_method": "rectified-flow-euler-data-ward-velocity-v1",
                "configured_resident_block_count": 50, "observed_resident_block_count": 50,
                "observed_resident_block_indices": list(range(50)), "transformer_construction_mode": "resident",
                "tensor_inventory": inventory, "deterministic_input_spec": probe.deterministic_input_specification(),
            })
            metadata.update({key: "x" for key in ("artifact_sha256", "resident_checkpoint_fingerprint",
                                                   "resident_config_sha256", "derived_config_sha256",
                                                   "conversion_manifest_sha256", "sidecar_manifest_sha256")})
        return metadata, transition, inventory

    def _validate_fixture(self, metadata, transition, inventory):
        with tempfile.TemporaryDirectory() as temp:
            original, derived = Path(temp) / "r", Path(temp) / "d"
            metadata["reference_checkpoint"] = str(original.resolve())
            metadata["derived_checkpoint"] = str(derived.resolve())
            checksums = {key: "x" for key in ("artifact_sha256", "resident_checkpoint_fingerprint",
                                                "resident_config_sha256", "derived_config_sha256",
                                                "conversion_manifest_sha256", "sidecar_manifest_sha256")}
            probe.validate_reference_metadata(metadata, original=original, derived=derived,
                                              expected_checksums=checksums, expected_inventory=inventory,
                                              expected_layout={"sequence_length": 4}, expected_transition=transition)

    def test_create_metadata_accepts_canonical_transition_and_writes_selected_step_zero(self):
        metadata = self._build_reference_metadata(MULTIMODAL_TRANSITION)
        self.assertEqual(metadata["selected_step_index"], 0)

    def test_create_metadata_propagates_nonzero_synthetic_selected_step(self):
        transition = {**MULTIMODAL_TRANSITION, "step_index": 1}
        metadata = self._build_reference_metadata(transition)
        self.assertEqual(metadata["selected_step_index"], 1)

    def test_missing_selected_step_metadata_fails_intentionally(self):
        transition = {key: value for key, value in MULTIMODAL_TRANSITION.items() if key != "step_index"}
        with self.assertRaisesRegex(ValueError, "required selected step field"):
            probe.artifact_transition_values(transition)

    def _build_reference_metadata(self, transition):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / "original"
            derived = root / "derived"
            (original / "model.safetensors").parent.mkdir(parents=True)
            (derived / "adaln").mkdir(parents=True)
            (original / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"weight": "model.safetensors"}}))
            (original / "model.safetensors").write_bytes(b"synthetic")
            for path in (original / "config.json", derived / "config.json",
                         derived / "conversion_manifest.json", derived / "adaln" / "manifest.json"):
                path.write_text("{}")
            layout = SimpleNamespace(sequence_length=4, text_indices=np.array([0], dtype=np.int32),
                                     video_indices=np.array([3], dtype=np.int32),
                                     audio_indices=np.array([1, 2], dtype=np.int32))
            timestep = np.array([0.0, 0.25], dtype=np.float32)
            arrays = {"synthetic": np.zeros((1,), dtype=np.float32)}
            scheduler = SimpleNamespace(
                num_inference_steps=2,
                update_method="rectified-flow-euler-data-ward-velocity-v1",
                configuration=lambda: {"identity": "synthetic"},
            )
            dit = SimpleNamespace(blocks=[None] * 50, construction_mode="resident")
            return probe._metadata(original=original, derived=derived, scheduler=scheduler,
                                   transition=transition, dit=dit, layout=layout,
                                   timestep=timestep, arrays=arrays, observed=list(range(50)))

    def test_create_reference_release_runs_when_artifact_construction_fails(self):
        events, fake_mx = [], SimpleNamespace(int32=np.int32, float32=np.float32,
                                               array=lambda value, dtype=None: np.asarray(value, dtype=dtype),
                                               eval=lambda *values: None)
        with self._fake_create_context(fake_mx, events, write_error=RuntimeError("artifact construction failed")):
            with self.assertRaisesRegex(RuntimeError, "artifact construction failed"):
                probe.cmd_create_reference(self._create_args())
        self.assertEqual(events, ["artifact", "release"])

    def test_create_reference_success_is_reported_after_artifact_and_release(self):
        events, fake_mx = [], SimpleNamespace(int32=np.int32, float32=np.float32,
                                               array=lambda value, dtype=None: np.asarray(value, dtype=dtype),
                                               eval=lambda *values: None)
        with self._fake_create_context(fake_mx, events):
            def capture_print(*args, **kwargs):
                if args and args[0] == "ONE-STEP RESIDENT REFERENCE CREATED":
                    events.append("success")
            with mock.patch("builtins.print", side_effect=capture_print):
                self.assertEqual(probe.cmd_create_reference(self._create_args()), 0)
        self.assertEqual(events, ["artifact", "release", "success"])

    def _create_args(self):
        return SimpleNamespace(original="/synthetic/original", derived="/synthetic/derived",
                               artifact="/synthetic/artifact.safetensors",
                               metadata="/synthetic/artifact.json", step_index=0, overwrite=False)

    def _fake_create_context(self, fake_mx, events, write_error=None):
        scheduler = SimpleNamespace(
            num_inference_steps=2,
            transition=lambda index: SimpleNamespace(**{**MULTIMODAL_TRANSITION, "step_index": index}),
            configuration=lambda: {"identity": "synthetic"},
        )
        dit = SimpleNamespace()
        layout = SimpleNamespace(token_tags=np.zeros(4, dtype=np.int32), position_ids=np.zeros((4, 3), dtype=np.float32),
                                 video_indices=np.array([3], dtype=np.int32), audio_indices=np.array([1, 2], dtype=np.int32),
                                 text_indices=np.array([0], dtype=np.int32))
        fake_load = types.ModuleType("minimax_h3_mlx.load")
        fake_load.load_dit = lambda *args, **kwargs: dit

        def write_artifact(*args, **kwargs):
            events.append("artifact")
            if write_error is not None:
                raise write_error

        def release(*args, **kwargs):
            events.append("release")
            return {"allocator_cache_purge_available": True, "allocator_cache_purged": True}

        fake_context = mock.patch.dict(sys.modules, {
            "mlx": SimpleNamespace(core=fake_mx), "mlx.core": fake_mx,
            "minimax_h3_mlx.load": fake_load,
        })
        patches = [
            fake_context,
            mock.patch.object(probe, "_build_canonical_scheduler", return_value=scheduler),
            mock.patch.object(probe, "_validate_model"),
            mock.patch.object(probe, "_build_layout_and_inputs", return_value=(
                layout, np.array([0.0, 0.25], dtype=np.float32), np.zeros(4, dtype=np.int32),
                np.zeros((1, 1, 96), dtype=np.float32), np.zeros((1, 2, 32), dtype=np.float32),
                np.zeros((1, 1, 5120), dtype=np.float32))),
            mock.patch.object(probe, "_run_one_step", return_value=(SimpleNamespace(
                video_prediction=np.zeros((1, 1, 96), dtype=np.float32),
                audio_prediction=np.zeros((1, 2, 32), dtype=np.float32),
                updated_video_latent=np.ones((1, 1, 96), dtype=np.float32),
                updated_audio_latent=np.ones((1, 2, 32), dtype=np.float32)),
                {"observed_transformer_block_indices": list(range(50))})),
            mock.patch.object(probe, "_metadata", return_value={"selected_step_index": 0}),
            mock.patch.object(probe, "_write_artifact", side_effect=write_artifact),
            mock.patch.object(probe, "_release", side_effect=release),
            mock.patch.object(probe, "emit_phase"),
        ]
        stack = ExitStack()
        for patcher in patches:
            stack.enter_context(patcher)
        return stack

    def test_serialized_transition_mismatch_fails_independently(self):
        arrays = {field: np.array([value], dtype=np.int32 if field == "selected_step_index" else np.float32)
                  for field, value in probe.artifact_transition_values(MULTIMODAL_TRANSITION).items()}
        for field in probe.TRANSITION_FIELDS:
            tampered = dict(arrays)
            tampered[field] = tampered[field].copy()
            tampered[field][0] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_serialized_transition(None, tampered,
                                                      probe.artifact_transition_values(MULTIMODAL_TRANSITION),
                                                      MULTIMODAL_TRANSITION)

    def test_deterministic_runtime_values_use_one_canonical_parameter_helper(self):
        self.assertEqual(probe.deterministic_input_values(128, 2), probe.deterministic_input_values(128, 2))
        source = inspect.getsource(probe._build_layout_and_inputs)
        self.assertNotIn("0.001 +", source)
        self.assertNotIn("% 23", source)

    def test_mlxsafe_finite_and_exact_helpers_never_convert_to_numpy(self):
        dispatch = MinimalMLXDispatch()
        left = NoNumpyMLXArray([1.0, 2.0], dispatch)
        right = NoNumpyMLXArray([1.0, 2.0], dispatch)
        _finite(left, "left")
        self.assertTrue(_exact_equal(left, right))

    def test_probe_metrics_use_native_dispatch_without_numpy_conversion(self):
        dispatch = MetricMLXDispatch()
        left = DispatchArray([1.0, 2.0], dispatch)
        right = DispatchArray([1.0, 2.0], dispatch)
        metrics = probe._metric_report(dispatch, left, right)
        self.assertTrue(metrics["exact_equality"])
        self.assertEqual(metrics["mismatched_element_count_exact"], 0)

    def test_current_timestep_mismatch_fails(self):
        scheduler = FakeScheduler()
        scheduler.transition = lambda _: {**MULTIMODAL_TRANSITION, "audio_current_timestep": 0.5}
        with self.assertRaisesRegex(ValueError, "current timestep"):
            run_one(scheduler=scheduler)

    def test_transformer_and_scheduler_each_execute_once(self):
        transformer, scheduler = FakeTransformer(), FakeScheduler()
        run_one(transformer=transformer, scheduler=scheduler)
        self.assertEqual((transformer.calls, scheduler.prepare_calls, scheduler.step_calls), (1, 1, 1))

    def test_scheduler_parameterization_and_scaling_mismatches_fail(self):
        scheduler = FakeScheduler()
        scheduler.prediction_parameterization = "noise"
        with self.assertRaisesRegex(ValueError, "parameterization"):
            run_one(scheduler=scheduler)
        scheduler = FakeScheduler()
        scheduler.input_scaling = "sigma"
        with self.assertRaisesRegex(ValueError, "scaling"):
            run_one(scheduler=scheduler)

    def test_nonfinite_and_unchanged_outputs_fail(self):
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            run_one(transformer=FakeTransformer(nonfinite=True))
        with self.assertRaisesRegex(ValueError, "video latent unchanged"):
            run_one(scheduler=FakeScheduler(unchanged_video=True))
        with self.assertRaisesRegex(ValueError, "audio latent unchanged"):
            run_one(scheduler=FakeScheduler(unchanged_audio=True))

    def test_production_adapter_exposes_both_transitions_and_different_shifts(self):
        scheduler, _, _ = production_adapter()
        transition = scheduler.transition(0)
        self.assertEqual(transition.video_current_sigma, 1.0)
        self.assertEqual(transition.audio_current_sigma, 1.0)
        self.assertNotEqual(transition.video_next_sigma, transition.audio_next_sigma)
        self.assertTrue(0 < transition.video_next_sigma / transition.video_current_sigma < 1)
        self.assertTrue(0 < transition.audio_next_sigma / transition.audio_current_sigma < 1)

    def test_production_adapter_calls_each_scalar_once_with_own_timestep(self):
        scheduler, video, audio = production_adapter()
        video_sample = np.ones((1, 1, 4))
        audio_sample = np.ones((1, 2, 3))
        scheduler.step(np.ones_like(video_sample), np.ones_like(audio_sample), video_sample, audio_sample, 0)
        self.assertEqual(len(video.calls), 1)
        self.assertEqual(len(audio.calls), 1)
        self.assertEqual(video.calls[0][0], 0.0)
        self.assertEqual(audio.calls[0][0], 0.0)
        self.assertEqual((video.step_index, audio.step_index), (1, 1))

    def test_production_adapter_preserves_batch_dimensions(self):
        scheduler, _, _ = production_adapter()
        updated_video, updated_audio = scheduler.step(np.ones((2, 1, 4)), np.ones((2, 2, 3)),
                                                       np.ones((2, 1, 4)), np.ones((2, 2, 3)), 0)
        self.assertEqual(updated_video.shape, (1, 1, 4))
        self.assertEqual(updated_audio.shape, (1, 2, 3))

    def test_production_adapter_cursor_mismatch_fails(self):
        scheduler, video, _ = production_adapter()
        video._step_index = 1
        with self.assertRaisesRegex(ValueError, "cursor mismatch"):
            scheduler.step(np.ones((1, 1, 4)), np.ones((1, 2, 3)), np.ones((1, 1, 4)), np.ones((1, 2, 3)), 0)

    def test_production_adapter_step_index_mismatch_fails(self):
        scheduler, _, _ = production_adapter()
        with self.assertRaises(IndexError):
            scheduler.step(np.ones((1, 1, 4)), np.ones((1, 2, 3)), np.ones((1, 1, 4)), np.ones((1, 2, 3)), 2)

    def test_fingerprint_rejects_mixed_type_and_unsafe_shard_names(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoint = Path(temp)
            index = checkpoint / "model.safetensors.index.json"
            index.write_text(json.dumps({"weight_map": {"x": "shard.safetensors", "y": 1}}))
            with self.assertRaisesRegex(ValueError, "must all be strings"):
                probe.resident_checkpoint_fingerprint(checkpoint)
            index.write_text(json.dumps({"weight_map": {"x": "../shard.safetensors"}}))
            with self.assertRaisesRegex(ValueError, "unsafe"):
                probe.resident_checkpoint_fingerprint(checkpoint)

    def test_cleanup_detaches_exception_and_success_is_guarded(self):
        original = RuntimeError("boom")
        detached = probe._detach(original)
        self.assertIsNone(detached.__traceback__)
        with self.assertRaises(ValueError):
            probe.emit_parity_success_message(False)

    def test_json_report_and_existing_receipts_are_truthful(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "report.json"
            probe.write_diagnostic_report(path, {"z": 1, "a": {"b": 2}})
            self.assertEqual(json.loads(path.read_text()), {"a": {"b": 2}, "z": 1})
            present = Path(temp) / "present"
            present.write_text("x")
            self.assertEqual(probe.existing_artifact_paths(present, Path(temp) / "missing"), [str(present)])


if __name__ == "__main__":
    unittest.main()

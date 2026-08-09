"""MLX-free lifecycle contracts for one derived streamed transition."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.denoise import StreamedTransitionSession  # noqa: E402


class FakeArray:
    def __init__(self, value, dtype: str):
        self.data = np.asarray(value, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return type(self)(self.data.copy(), self.dtype)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


class LazyPrediction(FakeArray):
    def __init__(self, value, dtype: str, events: list[str], label: str, fail: bool = False):
        super().__init__(value, dtype)
        self.events = events
        self.label = label
        self.fail = fail
        self.materialized = False

    def materialize(self) -> None:
        self.events.append(f"materialize-{self.label}")
        if self.fail:
            raise RuntimeError(f"{self.label} materialization failure")
        self.materialized = True


class FakeCache:
    def __init__(self, serial: int):
        self.serial = serial
        self.tables = [f"table-{serial}"]
        self.timesteps = object()
        self.released = False


class FakeScheduler:
    prediction_parameterization = "velocity"
    input_scaling = "identity"

    def __init__(self):
        self.scheduler_updates = 0

    def transition(self, step_index: int):
        values = (
            (0.0, 0.5, 0.25, 0.75),
            (0.25, 0.75, 1.0, 1.0),
        )
        video_current, audio_current, video_next, audio_next = values[step_index]
        return {
            "step_index": step_index,
            "video_current_timestep": video_current,
            "video_next_timestep": video_next,
            "video_current_sigma": 1.0 - video_current,
            "video_next_sigma": 1.0 - video_next,
            "audio_current_timestep": audio_current,
            "audio_next_timestep": audio_next,
            "audio_current_sigma": 1.0 - audio_current,
            "audio_next_sigma": 1.0 - audio_next,
        }

    def prepare_model_input(self, video, audio, _step_index):
        return video, audio


class FakeTransformer:
    def __init__(self, events: list[str], *, fail_forward: bool = False, fail_audio_materialization: bool = False):
        self.events = events
        self.fail_forward = fail_forward
        self.fail_audio_materialization = fail_audio_materialization
        self.calls = 0
        self.last_inputs = None
        self.last_predictions: tuple[LazyPrediction, LazyPrediction] | None = None

    def __call__(
        self,
        video,
        audio,
        text,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        **_kwargs,
    ):
        self.calls += 1
        self.events.append("forward")
        self.last_inputs = {
            "video": video,
            "audio": audio,
            "text": text,
            "timestep": timestep,
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
        }
        if self.fail_forward:
            raise RuntimeError("forward failure")
        predictions = (
            LazyPrediction(np.ones_like(video.data), "float32", self.events, "video"),
            LazyPrediction(
                np.ones_like(audio.data),
                "float32",
                self.events,
                "audio",
                fail=self.fail_audio_materialization,
            ),
        )
        self.last_predictions = predictions
        return predictions


class FakeBuilder:
    def __init__(self, events: list[str], *, fail: bool = False):
        self.events = events
        self.fail = fail
        self.calls = 0
        self.caches: list[FakeCache] = []

    def __call__(self, _transformer, _timestep):
        self.events.append("build")
        self.calls += 1
        if self.fail:
            raise RuntimeError("cache build failure")
        if self.caches and not self.caches[-1].released:
            raise AssertionError("cache overlap")
        cache = FakeCache(self.calls)
        self.caches.append(cache)
        return cache, {"build_number": self.calls}


def inputs(step_index: int = 0, *, with_conditions: bool = False) -> dict[str, object]:
    if with_conditions:
        return {
            "video_latent": FakeArray(
                np.array([[[10.0] * 96, [11.0] * 96, [20.0] * 96, [21.0] * 96]]),
                "bfloat16",
            ),
            "audio_latent": FakeArray(np.array([[[30.0] * 32, [31.0] * 32, [32.0] * 32]]), "bfloat16"),
            "text_embedding": FakeArray(np.zeros((1, 2, 5120)), "bfloat16"),
            "timestep": FakeArray([0.0, 0.5, 0.999], "float32"),
            "timestep_indices": FakeArray([0, 0, 2, 2, 1, 1, 1, 0, 0, 0, 0], "int32"),
            "token_tags": FakeArray([1, 1, 0, 0, 2, 2, 2, 0, 0, 0, 0], "int32"),
            "position_ids": FakeArray(np.arange(33).reshape(11, 3), "float32"),
            "video_indices": FakeArray([2, 3, 7, 8, 9, 10], "int32"),
            "audio_indices": FakeArray([4, 5, 6], "int32"),
            "text_indices": FakeArray([0, 1], "int32"),
            "step_index": 0,
            "expected_video_shape": (1, 4, 96),
            "expected_audio_shape": (1, 3, 32),
            "expected_text_shape": (1, 2, 5120),
        }
    if step_index == 0:
        timestep = [0.0, 0.5]
        timestep_indices = [0, 0, 1, 0]
    else:
        timestep = [0.25, 0.75]
        timestep_indices = [0, 1, 1, 0]
    return {
        "video_latent": FakeArray(np.full((1, 1, 96), 0.1 + step_index), "bfloat16"),
        "audio_latent": FakeArray(np.full((1, 2, 32), 0.2 + step_index), "bfloat16"),
        "text_embedding": FakeArray(np.full((1, 1, 5120), 0.3), "bfloat16"),
        "timestep": FakeArray(timestep, "float32"),
        "timestep_indices": FakeArray(timestep_indices, "int32"),
        "token_tags": FakeArray([1, 2, 2, 0], "int32"),
        "position_ids": FakeArray(np.zeros((4, 3)), "float32"),
        "video_indices": FakeArray([3], "int32"),
        "audio_indices": FakeArray([1, 2], "int32"),
        "text_indices": FakeArray([0], "int32"),
        "step_index": step_index,
    }


class StreamedTransitionSessionTests(unittest.TestCase):
    def run_transition(self, session, scheduler=None, *, step_index: int = 0, with_conditions: bool = False):
        return session.run(scheduler or FakeScheduler(), **inputs(step_index, with_conditions=with_conditions))

    def test_happy_path_returns_only_after_materialization_and_release(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)

        def release(cache):
            self.assertTrue(all(pred.materialized for pred in transformer.last_predictions or ()))
            events.append("release")
            cache.tables.clear()
            cache.timesteps = None
            cache.released = True

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        result = self.run_transition(session)
        events.append("scheduler")

        self.assertEqual(
            events,
            ["build", "forward", "materialize-video", "materialize-audio", "release", "scheduler"],
        )
        self.assertEqual(result.step_index, 0)
        self.assertFalse(hasattr(result, "cache"))
        self.assertFalse(session.active)

    def test_optional_runtime_observer_reports_lifecycle_without_owning_it(self):
        events: list[str] = []
        observed: list[tuple[str, int]] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)

        def release(cache):
            events.append("release")
            cache.tables.clear()
            cache.timesteps = None

        session = StreamedTransitionSession(
            transformer,
            cache_builder=builder,
            cache_releaser=release,
            observer=lambda event, details: observed.append((event, details["step_index"])),
        )
        self.run_transition(session)

        self.assertEqual(
            [event for event, _step in observed],
            [
                "transition-start",
                "cache-build-start",
                "cache-build-complete",
                "forward-start",
                "forward-complete",
                "materialize-start",
                "materialize-complete",
                "transition-succeeded",
                "cache-release-start",
                "cache-release-complete",
            ],
        )
        self.assertTrue(all(step == 0 for _event, step in observed))
        self.assertEqual(events[-1], "release")
        self.assertFalse(session.active)

    def test_session_reuses_sequentially_without_overlapping_caches(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)

        def release(cache):
            events.append(f"release-{cache.serial}")
            cache.tables.clear()
            cache.timesteps = None
            cache.released = True

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        self.run_transition(session, step_index=0)
        self.run_transition(session, step_index=1)

        self.assertEqual(builder.calls, 2)
        self.assertEqual([cache.serial for cache in builder.caches], [1, 2])
        self.assertTrue(all(cache.released for cache in builder.caches))
        self.assertEqual(
            [event for event in events if event.startswith("build") or event.startswith("release-")],
            ["build", "release-1", "build", "release-2"],
        )

    def test_reentrant_transition_is_rejected(self):
        events: list[str] = []
        holder: dict[str, StreamedTransitionSession] = {}
        transformer = FakeTransformer(events)
        cache = FakeCache(1)

        def builder(_transformer, _timestep):
            events.append("build")
            with self.assertRaisesRegex(RuntimeError, "already owns"):
                self.run_transition(holder["session"])
            events.append("overlap-rejected")
            return cache

        def release(value):
            events.append("release")
            value.tables.clear()
            value.timesteps = None
            value.released = True

        holder["session"] = StreamedTransitionSession(
            transformer,
            cache_builder=builder,
            cache_releaser=release,
        )
        self.run_transition(holder["session"])
        self.assertEqual(events[:3], ["build", "overlap-rejected", "forward"])
        self.assertTrue(cache.released)

    def test_default_release_clears_cache_owned_storage(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)
        session = StreamedTransitionSession(transformer, cache_builder=builder)

        self.run_transition(session)

        self.assertEqual(builder.caches[0].tables, [])
        self.assertIsNone(builder.caches[0].timesteps)
        self.assertFalse(session.active)

    def test_cache_build_failure_blocks_forward_and_scheduler_path(self):
        events: list[str] = []
        builder = FakeBuilder(events, fail=True)
        transformer = FakeTransformer(events)
        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=lambda _cache: events.append("release"))

        with self.assertRaisesRegex(RuntimeError, "cache build failure") as raised:
            self.run_transition(session)

        self.assertEqual(transformer.calls, 0)
        self.assertEqual(events, ["build"])
        self.assertFalse(session.active)
        self.assertFalse(raised.exception.streamed_transition_cleanup_started)

    def test_forward_failure_releases_cache_and_preserves_primary_error(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events, fail_forward=True)

        def release(cache):
            events.append("release")
            cache.released = True

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        with self.assertRaisesRegex(RuntimeError, "forward failure") as raised:
            self.run_transition(session)

        self.assertEqual(events, ["build", "forward", "release"])
        self.assertTrue(raised.exception.streamed_transition_cleanup_started)
        self.assertTrue(raised.exception.streamed_transition_cleanup_succeeded)
        self.assertIsNone(raised.exception.streamed_transition_cleanup_error)
        self.assertFalse(session.active)

    def test_materialization_failure_releases_cache_and_returns_no_scheduler_result(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events, fail_audio_materialization=True)

        def release(cache):
            events.append("release")
            cache.released = True

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        with self.assertRaisesRegex(RuntimeError, "audio materialization failure") as raised:
            self.run_transition(session)

        self.assertEqual(events, ["build", "forward", "materialize-video", "materialize-audio", "release"])
        self.assertTrue(raised.exception.streamed_transition_cleanup_succeeded)
        self.assertFalse(session.active)

    def test_release_failure_is_visible_and_does_not_return_scheduler_result(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)

        def release(_cache):
            events.append("release")
            raise RuntimeError("release failure")

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        with self.assertRaisesRegex(RuntimeError, "release failure") as raised:
            self.run_transition(session)

        self.assertEqual(events, ["build", "forward", "materialize-video", "materialize-audio", "release"])
        self.assertFalse(raised.exception.streamed_transition_cleanup_succeeded)
        self.assertFalse(session.active)

    def test_primary_and_release_failures_remain_distinct(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events, fail_forward=True)

        def release(_cache):
            events.append("release")
            raise RuntimeError("release failure")

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        with self.assertRaisesRegex(RuntimeError, "forward failure") as raised:
            self.run_transition(session)

        self.assertEqual(str(raised.exception.streamed_transition_cleanup_error), "release failure")
        self.assertFalse(raised.exception.streamed_transition_cleanup_succeeded)
        self.assertFalse(session.active)

    def test_condition_rows_pass_through_and_target_only_schedule_can_follow_release(self):
        events: list[str] = []
        builder = FakeBuilder(events)
        transformer = FakeTransformer(events)

        def release(cache):
            events.append("release")
            cache.tables.clear()
            cache.timesteps = None
            cache.released = True

        session = StreamedTransitionSession(transformer, cache_builder=builder, cache_releaser=release)
        values = inputs(with_conditions=True)
        scheduler = FakeScheduler()
        result = session.run(scheduler, **values)

        self.assertIs(transformer.last_inputs["token_tags"], values["token_tags"])
        self.assertIs(transformer.last_inputs["position_ids"], values["position_ids"])
        self.assertIs(transformer.last_inputs["video_indices"], values["video_indices"])
        self.assertIs(transformer.last_inputs["audio_indices"], values["audio_indices"])
        self.assertIs(transformer.last_inputs["text_indices"], values["text_indices"])
        np.testing.assert_array_equal(
            transformer.last_inputs["video"].data[0, :2],
            np.asarray(values["video_latent"]).copy()[0, :2],
        )

        video_before = result.forward.input_video_latent.data.copy()
        audio_before = result.forward.input_audio_latent.data.copy()
        updated_video = video_before.copy()
        updated_audio = audio_before.copy()
        updated_video[:, 2:] += 1.0  # target video rows only; the first two are keyframe conditions
        updated_audio += 1.0  # all audio rows are targets in the public packed layout
        events.append("scheduler")

        np.testing.assert_array_equal(updated_video[:, :2], video_before[:, :2])
        np.testing.assert_array_equal(updated_video[:, 2:], video_before[:, 2:] + 1.0)
        np.testing.assert_array_equal(updated_audio, audio_before + 1.0)
        self.assertEqual(events[-1], "scheduler")
        self.assertEqual(result.step_index, 0)


if __name__ == "__main__":
    unittest.main()

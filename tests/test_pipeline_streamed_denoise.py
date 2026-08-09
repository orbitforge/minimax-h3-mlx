"""MLX-free contracts for the production pipeline's derived transition seam."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.denoise import run_streamed_transition  # noqa: E402


class FakeArray:
    def __init__(self, value, dtype: str):
        self.data = np.asarray(value)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    def tolist(self):
        return self.data.tolist()

    def astype(self, dtype: str):
        return type(self)(self.data.copy(), dtype)

    def copy(self):
        return type(self)(self.data.copy(), self.dtype)

    def __getitem__(self, item):
        return type(self)(self.data[item], self.dtype)

    def __array__(self, dtype=None):
        return np.asarray(self.data, dtype=dtype)


class FakePrediction(FakeArray):
    pass


class TraceScheduler:
    def __init__(self, label: str, events: list[str], fail_step: int | None = None):
        self.label = label
        self.events = events
        self.fail_step = fail_step
        self.calls = 0

    def step(self, _prediction, _timestep: float, sample: FakeArray):
        step_index = self.calls
        self.calls += 1
        self.events.append(f"scheduler-{self.label}-{step_index}")
        if self.fail_step == step_index:
            raise RuntimeError(f"{self.label} scheduler failure at {step_index}")
        return FakeArray(sample.data + 1.0, sample.dtype)


class FakeTransformer:
    def __init__(self, events: list[str], *, fail_stage: str | None = None, fail_step: int | None = None):
        self.events = events
        self.construction_mode = "cache_only"
        self.fail_stage = fail_stage
        self.fail_step = fail_step


class FakeStreamedTransitionSession:
    instances: list["FakeStreamedTransitionSession"] = []

    def __init__(self, transformer: FakeTransformer):
        self.transformer = transformer
        self.events = transformer.events
        self.session_number = len(self.instances)
        self.instances.append(self)
        self.events.append(f"session-{self.session_number}")
        self.calls: list[dict[str, object]] = []

    def run(self, _scheduler, **kwargs):
        step_index = int(kwargs["step_index"])
        self.calls.append(kwargs)
        self.events.append(f"build-{step_index}")
        if self.transformer.fail_stage == "build" and self.transformer.fail_step == step_index:
            raise RuntimeError(f"cache build failure at {step_index}")

        self.events.append(f"forward-{step_index}")
        if self.transformer.fail_stage == "forward" and self.transformer.fail_step == step_index:
            self.events.append(f"release-{step_index}")
            raise RuntimeError(f"forward failure at {step_index}")

        self.events.append(f"materialize-{step_index}")
        if self.transformer.fail_stage == "materialize" and self.transformer.fail_step == step_index:
            self.events.append(f"release-{step_index}")
            raise RuntimeError(f"materialize failure at {step_index}")

        self.events.append(f"release-{step_index}")
        if self.transformer.fail_stage == "release" and self.transformer.fail_step == step_index:
            raise RuntimeError(f"cache release failure at {step_index}")

        video = kwargs["video_latent"]
        audio = kwargs["audio_latent"]
        return SimpleNamespace(
            step_index=step_index,
            forward=SimpleNamespace(
                video_prediction=FakePrediction(np.ones(video.shape), "float32"),
                audio_prediction=FakePrediction(np.ones(audio.shape), "float32"),
                input_video_latent=video.copy(),
                input_audio_latent=audio.copy(),
            ),
        )


def layout():
    return SimpleNamespace(
        token_tags=object(),
        position_ids=object(),
        video_indices=object(),
        audio_indices=object(),
        text_indices=object(),
    )


def concatenate(values: list[FakeArray]) -> FakeArray:
    return FakeArray(np.concatenate([value.data for value in values]), values[0].dtype)


def run_two_transitions(
    transformer: FakeTransformer,
    video_scheduler: TraceScheduler,
    audio_scheduler: TraceScheduler,
    *,
    condition_rows: int = 0,
):
    video_rows = FakeArray(
        np.array([[10.0], [20.0], [30.0]]) if condition_rows else np.array([[20.0], [30.0]]),
        "bfloat16",
    )
    audio_rows = FakeArray(np.array([[40.0], [50.0]]), "bfloat16")
    text = FakeArray(np.zeros((1, 1, 2)), "bfloat16")
    row_layout = layout()
    transformer.layout = row_layout
    if condition_rows:
        timesteps = (FakeArray([0.0, 0.5, 0.999], "float32"), FakeArray([0.5, 0.75, 0.999], "float32"))
        plans = (FakeArray([0, 2, 1, 1, 0, 0], "int32"), FakeArray([0, 2, 1, 1, 0, 0], "int32"))
    else:
        timesteps = (FakeArray([0.0, 0.5], "float32"), FakeArray([0.5, 0.75], "float32"))
        plans = (FakeArray([0, 1, 1, 0, 0], "int32"), FakeArray([0, 1, 1, 0, 0], "int32"))

    for step_index, plan in enumerate(plans):
        video_rows, audio_rows, _video_prediction, _audio_prediction = run_streamed_transition(
            transformer,
            object(),
            video_scheduler,
            audio_scheduler,
            video_model_input=FakeArray(video_rows.data[None], "bfloat16"),
            audio_model_input=FakeArray(audio_rows.data[None], "bfloat16"),
            text_embedding=text,
            timestep=timesteps[step_index],
            timestep_indices=plan,
            layout=row_layout,
            step_index=step_index,
            video_timestep=(0.0, 0.5)[step_index],
            audio_timestep=(0.5, 0.75)[step_index],
            num_condition_video_rows=condition_rows,
            num_condition_audio_rows=0,
            prediction_cast=lambda value: value.astype("float32"),
            concatenate=concatenate,
            session_factory=FakeStreamedTransitionSession,
        )
    return video_rows, audio_rows, transformer


class PipelineStreamedDenoiseTests(unittest.TestCase):
    def setUp(self):
        FakeStreamedTransitionSession.instances = []

    def make_transformer(self, *, fail_stage: str | None = None, fail_step: int | None = None):
        events: list[str] = []
        transformer = FakeTransformer(events, fail_stage=fail_stage, fail_step=fail_step)
        return transformer, events

    def test_derived_path_has_one_session_and_exact_release_before_scheduler_order(self):
        transformer, events = self.make_transformer()
        video_scheduler = TraceScheduler("video", events)
        audio_scheduler = TraceScheduler("audio", events)

        run_two_transitions(transformer, video_scheduler, audio_scheduler)

        self.assertEqual(
            [event for event in events if not event.startswith("session-")],
            [
                "build-0",
                "forward-0",
                "materialize-0",
                "release-0",
                "scheduler-video-0",
                "scheduler-audio-0",
                "build-1",
                "forward-1",
                "materialize-1",
                "release-1",
                "scheduler-video-1",
                "scheduler-audio-1",
            ],
        )
        self.assertEqual(len(FakeStreamedTransitionSession.instances), 2)
        self.assertEqual([len(session.calls) for session in FakeStreamedTransitionSession.instances], [1, 1])

    def test_condition_rows_remain_fixed_and_metadata_is_passed_unchanged(self):
        transformer, events = self.make_transformer()
        video_scheduler = TraceScheduler("video", events)
        audio_scheduler = TraceScheduler("audio", events)

        final_video, final_audio, _ = run_two_transitions(
            transformer,
            video_scheduler,
            audio_scheduler,
            condition_rows=1,
        )
        np.testing.assert_array_equal(final_video.data[0], [10.0])
        np.testing.assert_array_equal(final_video.data[1:], [[22.0], [32.0]])
        np.testing.assert_array_equal(final_audio.data, [[42.0], [52.0]])
        first_call = FakeStreamedTransitionSession.instances[0].calls[0]
        self.assertIs(first_call["token_tags"], transformer.layout.token_tags)
        self.assertIs(first_call["position_ids"], transformer.layout.position_ids)
        self.assertIs(first_call["video_indices"], transformer.layout.video_indices)
        self.assertIs(first_call["audio_indices"], transformer.layout.audio_indices)
        self.assertIs(first_call["text_indices"], transformer.layout.text_indices)
        self.assertEqual(first_call["timestep"].data.tolist(), [0.0, 0.5, 0.999])
        self.assertEqual(first_call["timestep_indices"].data.tolist(), [0, 2, 1, 1, 0, 0])

    def test_resident_mode_does_not_select_streamed_helper(self):
        source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
        self.assertIn('getattr(self.dit, "construction_mode", "resident") == CACHE_ONLY_CONSTRUCTION', source)
        self.assertEqual(source.count("self._ensure_cache(timestep_table, drop_adaln, verbose)"), 1)
        self.assertIn("run_streamed_transition(", source)
        self.assertIn("session_factory=StreamedTransitionSession", source)
        self.assertIn("apply_target_scheduler_updates(", source)
        self.assertNotIn("video_sched.step(", source)
        self.assertNotIn("audio_sched.step(", source)
        self.assertIn("modulation_cache=self._cache", source)

    def test_build_forward_materialization_release_and_scheduler_failures_stop_before_step_one(self):
        cases = ("build", "forward", "materialize", "release", "scheduler")
        for stage in cases:
            with self.subTest(stage=stage):
                transformer, events = self.make_transformer(fail_stage=stage, fail_step=0)
                video_scheduler = TraceScheduler("video", events, fail_step=0 if stage == "scheduler" else None)
                audio_scheduler = TraceScheduler("audio", events)
                with self.assertRaisesRegex(RuntimeError, f"{stage}.*failure"):
                    run_two_transitions(transformer, video_scheduler, audio_scheduler)
                self.assertNotIn("build-1", events)
                self.assertNotIn("forward-1", events)
                self.assertNotIn("scheduler-video-1", events)

    def test_derived_keep_adaln_request_and_ensure_cache_guard_remain_fail_closed(self):
        source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
        self.assertIn("sidecar cache construction has not occurred", source)
        self.assertIn("keep-adaln is not supported for derived checkpoints", source)

    def test_rng_and_cli_control_flow_have_no_new_public_surface(self):
        source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
        generation_source = source[source.index("    def __call__(") :]
        self.assertEqual(generation_source.count("mx.random.seed(seed)"), 1)
        self.assertEqual(generation_source.count("mx.random.normal"), 3)
        self.assertLess(generation_source.index("condition_noise"), generation_source.index("latents = mx.random.normal"))
        self.assertLess(generation_source.index("latents = mx.random.normal"), generation_source.index("audio_latents = mx.random.normal"))

        cli_source = (ROOT / "scripts" / "generate.py").read_text()
        self.assertIn('parser.add_argument("--keep-adaln"', cli_source)
        self.assertNotIn("streamed", cli_source)
        self.assertNotIn("sidecar", cli_source)


if __name__ == "__main__":
    unittest.main()

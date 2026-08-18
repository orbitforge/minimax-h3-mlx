"""Mocked regression tests for MiniMax-H3's staged component lifecycle."""

from __future__ import annotations

import gc
import io
import json
import sys
import tempfile
import weakref
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.config import PipelineConfig
from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
import minimax_h3_mlx.pipeline as pipeline_module


@contextmanager
def patched_attribute(owner, name: str, value):
    original = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, original)


class FakeEncoder:
    def __init__(self, events: list[str], load_vision: bool = False):
        self.events = events
        self.load_vision = load_vision

    def encode(self, prompt: str, images):
        self.events.append(f"encode:{'images' if images else 'text'}")
        return mx.array([[[1.0, 2.0], [3.0, 4.0]]]), np.array([1, 1], dtype=np.int64)


def make_pipeline(encoder, events: list[str], unload: bool = True, failing: bool = False):
    def load_dit():
        events.append("transformer")
        if failing:
            raise ValueError("fixture transformer failure")
        return object()

    def load_video_vae():
        events.append("video_vae")
        return object()

    def load_audio_vae():
        events.append("audio_vae")
        return object()

    return MiniMaxH3Pipeline(
        None,
        encoder,
        None,
        None,
        PipelineConfig(),
        unload_text_encoder=unload,
        generation_loaders={
            "dit": load_dit,
            "video_vae": load_video_vae,
            "audio_vae": load_audio_vae,
        },
    )


def test_conditioning_precedes_generation_loads() -> None:
    events: list[str] = []
    encoder = FakeEncoder(events)
    pipe = make_pipeline(encoder, events)

    embeds, tags = pipe._prepare_conditioning("prompt", None, verbose=False)
    assert events == ["encode:text"], events
    assert pipe.text_encoder is None

    pipe._load_generation_components(verbose=False)
    assert events == ["encode:text", "transformer", "video_vae", "audio_vae"], events
    assert tuple(np.asarray(embeds).shape) == (1, 2, 2)
    assert tags.tolist() == [1, 1]
    mx.eval(embeds)


def test_conditioning_survives_encoder_reclamation() -> None:
    events: list[str] = []
    encoder = FakeEncoder(events)
    encoder_ref = weakref.ref(encoder)
    pipe = make_pipeline(encoder, events)

    embeds, tags = pipe._prepare_conditioning("prompt", None, verbose=False)
    del encoder
    gc.collect()

    assert encoder_ref() is None
    assert pipe.text_encoder is None
    mx.eval(embeds)
    assert np.asarray(embeds).tolist() == [[[1.0, 2.0], [3.0, 4.0]]]
    assert tags.tolist() == [1, 1]


def test_keep_text_encoder_retains_intentionally() -> None:
    events: list[str] = []
    encoder = FakeEncoder(events)
    pipe = make_pipeline(encoder, events, unload=False)

    pipe._prepare_conditioning("prompt", None, verbose=False)
    assert pipe.text_encoder is encoder
    pipe._load_generation_components(verbose=False)
    assert events == ["encode:text", "transformer", "video_vae", "audio_vae"], events


def test_image_conditioning_is_forwarded() -> None:
    events: list[str] = []
    encoder = FakeEncoder(events)
    pipe = make_pipeline(encoder, events)

    pipe._prepare_conditioning("prompt", [object()], verbose=False)
    assert events == ["encode:images"], events


def test_repeated_staged_conditioning_is_explicitly_rejected() -> None:
    events: list[str] = []
    pipe = make_pipeline(FakeEncoder(events), events)
    pipe._prepare_conditioning("prompt", None, verbose=False)

    try:
        pipe._prepare_conditioning("second prompt", None, verbose=False)
    except RuntimeError as exc:
        assert "one-shot" in str(exc)
    else:
        raise AssertionError("released staged encoder was unexpectedly reusable")


def test_stage_failure_is_wrapped() -> None:
    events: list[str] = []
    pipe = make_pipeline(FakeEncoder(events), events, failing=True)
    pipe._prepare_conditioning("prompt", None, verbose=False)

    clear_calls: list[int] = []
    with patched_attribute(pipeline_module.mx, "clear_cache", lambda: clear_calls.append(1)):
        try:
            pipe._load_generation_components(verbose=False)
        except RuntimeError as exc:
            assert "Stage failed" in str(exc)
            assert pipe.dit is None
        else:
            raise AssertionError("fixture loader failure was not reported")
    assert len(clear_calls) == 1  # partial-load failure cleanup; text release happened before this patch


def test_from_pretrained_passes_vision_choice_and_defers_generation(monkeypatch=None) -> None:
    """Exercise the public factory without touching a checkpoint."""
    import minimax_h3_mlx.load as load_module
    import minimax_h3_mlx.text_encoder as encoder_module

    events: list[bool] = []
    original_encoder = encoder_module.MiniMaxH3TextEncoder
    original_dit = load_module.load_dit
    original_video = load_module.load_video_vae
    original_audio = load_module.load_audio_vae
    original_video_config = load_module.load_video_vae_config
    original_audio_config = load_module.load_audio_vae_config

    class FactoryEncoder(FakeEncoder):
        def __init__(self, *args, **kwargs):
            events.append(bool(kwargs.get("load_vision", False)))
            super().__init__([], kwargs.get("load_vision", False))

    encoder_module.MiniMaxH3TextEncoder = FactoryEncoder
    load_module.load_dit = lambda *args, **kwargs: object()
    load_module.load_video_vae = lambda *args, **kwargs: object()
    load_module.load_audio_vae = lambda *args, **kwargs: object()
    load_module.load_video_vae_config = lambda *args, **kwargs: object()
    load_module.load_audio_vae_config = lambda *args, **kwargs: object()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "model_index.json").write_text(json.dumps({}))
            (root / "transformer").mkdir()
            (root / "transformer" / "config.json").write_text(json.dumps({}))
            text_only = MiniMaxH3Pipeline.from_pretrained(root, load_vision=False, verbose=False)
            image = MiniMaxH3Pipeline.from_pretrained(root, load_vision=True, verbose=False)
            assert events == [False, True], events
            assert text_only.dit is None and text_only.video_vae is None and text_only.audio_vae is None
            assert image.dit is None and image.video_vae is None and image.audio_vae is None
    finally:
        encoder_module.MiniMaxH3TextEncoder = original_encoder
        load_module.load_dit = original_dit
        load_module.load_video_vae = original_video
        load_module.load_audio_vae = original_audio
        load_module.load_video_vae_config = original_video_config
        load_module.load_audio_vae_config = original_audio_config


class FakeSchedule:
    def __init__(self):
        self.timesteps = mx.array([0.0, 0.5])

    def step(self, prediction, timestep, current):
        return current


class FakeDenoiser:
    def __init__(self, events: list[str]):
        self.events = events

    def __call__(self, video, audio, *args, **kwargs):
        self.events.append("denoise")
        return mx.zeros_like(video), mx.zeros_like(audio)


class FakeModulationCache:
    def __init__(self, events: list[str]):
        self.events = events
        self.tables = [(mx.array([7.0]),)]
        self.timesteps = mx.array([0.0])
        self.lora_identity = None

    def materialize(self) -> None:
        self.events.append("cache_materialize")
        mx.eval(self.timesteps, self.tables[0][0])

    def nbytes(self) -> int:
        return self.tables[0][0].nbytes


def fake_configs():
    return (
        SimpleNamespace(patch_size=(1, 2, 2)),
        SimpleNamespace(
            spatial_compression_ratio=16,
            latent_channels=24,
            latents_mean=(0.0,) * 24,
            latents_std=(1.0,) * 24,
        ),
        SimpleNamespace(
            latent_channels=32,
            sampling_rate=32000,
            latents_mean=(0.0,) * 32,
            latents_std=(1.0,) * 32,
        ),
    )


def run_fake_generation(images: bool, verbose: bool = False) -> list[str]:
    events: list[str] = []
    dit_config, video_config, audio_config = fake_configs()

    def load_dit():
        events.append("transformer_load")
        return FakeDenoiser(events)

    def load_video():
        events.append("video_load")
        return object()

    def load_audio():
        events.append("audio_load")
        return object()

    def fake_layout(*args, **kwargs):
        condition_count = 1 if images else 0
        return SimpleNamespace(
            sequence_length=4,
            num_condition_video_rows=condition_count,
            num_condition_audio_rows=0,
            token_tags=mx.array([1, 1, 0, 2], dtype=mx.int32),
            position_ids=mx.zeros((4, 3), dtype=mx.int32),
            video_indices=mx.array([2], dtype=mx.int32),
            audio_indices=mx.array([3], dtype=mx.int32),
            text_indices=mx.array([0, 1], dtype=mx.int32),
        )

    original_layout = pipeline_module.build_packed_sequence
    pipeline_module.build_packed_sequence = fake_layout
    try:
        pipe = MiniMaxH3Pipeline(
            None,
            FakeEncoder(events),
            None,
            None,
            PipelineConfig(),
            unload_text_encoder=True,
            generation_loaders={"dit": load_dit, "video_vae": load_video, "audio_vae": load_audio},
            dit_config=dit_config,
            video_vae_config=video_config,
            audio_vae_config=audio_config,
        )

        original_prepare = pipe._prepare_conditioning
        original_cache = pipe._ensure_cache
        original_decode_video = pipe._decode_video
        original_decode_audio = pipe._decode_audio
        original_release = pipe._release_component
        original_allocator_purge = pipe._purge_allocator_cache

        def prepare(*args, **kwargs):
            result = original_prepare(*args, **kwargs)
            events.append("conditioning_ready")
            return result

        def cache(*args, **kwargs):
            events.append("adaln_cache")
            pipe._cache = FakeModulationCache(events)
            pipe._cache.materialize()
            events.append("adaln_projection_removal")
            pipe._purge_adaln_allocator_cache(verbose=False)

        original_purge = pipe._purge_adaln_allocator_cache

        def purge(*args, **kwargs):
            events.append("cache_purge")
            return original_purge(*args, **kwargs)

        def allocator_purge(reason, *args, **kwargs):
            events.append(f"allocator_purge:{reason}")
            return original_allocator_purge(reason, *args, **kwargs)

        def encode_keyframes(*args, **kwargs):
            events.append("keyframe_encode")
            return mx.zeros((1, 96), dtype=mx.float32)

        def decode_video(*args, **kwargs):
            events.append("video_decode")
            return np.zeros((1, 1, 1, 3), dtype=np.uint8)

        def decode_audio(*args, **kwargs):
            events.append("audio_decode")
            return np.zeros((2, 4), dtype=np.float32)

        def release(attr, label, verbose, purge_reason=None):
            events.append(f"{attr}_release")
            return original_release(attr, label, verbose, purge_reason)

        pipe._prepare_conditioning = prepare
        pipe._ensure_cache = cache
        pipe._purge_adaln_allocator_cache = purge
        pipe._purge_allocator_cache = allocator_purge
        pipe._encode_keyframes = encode_keyframes
        pipe._decode_video = decode_video
        pipe._decode_audio = decode_audio
        pipe._release_component = release
        pipe._build_schedules = lambda steps, turbo_schedule=None: (FakeSchedule(), FakeSchedule())
        pipe._row_timestep_plan = lambda *args: (
            mx.array([0.0]),
            [mx.zeros((4,), dtype=mx.int32), mx.zeros((4,), dtype=mx.int32)],
        )

        pipe(
            "prompt",
            duration_seconds=5,
            height=32,
            width=32,
            num_inference_steps=2,
            images=[object()] if images else None,
            keyframe_anchors=("first",) if images else (),
            verbose=verbose,
        )
        return events
    finally:
        pipeline_module.build_packed_sequence = original_layout


def test_text_only_decoder_residency_is_phase_isolated() -> None:
    events = run_fake_generation(images=False)
    assert events.index("transformer_load") < events.index("adaln_cache") < events.index("denoise")
    assert events.index("dit_release") < events.index("video_load")
    assert events.index("video_vae_release") < events.index("audio_load")
    assert events.index("audio_vae_release") > events.index("audio_decode")
    assert "keyframe_encode" not in events


def test_image_conditioning_reloads_video_vae_only_for_decode() -> None:
    events = run_fake_generation(images=True)
    assert events.index("video_load") < events.index("keyframe_encode")
    assert events.index("video_vae_release") < events.index("transformer_load")
    assert events.index("dit_release") < events.index("video_load", events.index("dit_release"))
    assert events.index("video_decode") < events.index("video_vae_release", events.index("dit_release"))
    assert events.count("allocator_purge:video-vae-release") == 2


def test_materialized_decode_outputs_are_retained_after_release() -> None:
    events = run_fake_generation(images=False)
    assert events[-2] == "audio_vae_release", events


def test_adaln_cache_purge_order_is_post_removal_and_pre_denoise() -> None:
    events = run_fake_generation(images=False)
    assert events.index("adaln_cache") < events.index("cache_materialize")
    assert events.index("adaln_projection_removal") < events.index("cache_purge")
    assert events.index("cache_purge") < events.index("denoise")
    assert events.count("cache_purge") == 1
    assert events.count("allocator_purge:adaln-pre-denoise") == 1


def test_component_release_purges_are_separate_from_adaln_purge() -> None:
    events = run_fake_generation(images=False)
    reasons = [event.split(":", 1)[1] for event in events if event.startswith("allocator_purge:")]
    assert reasons == [
        "text-encoder-release",
        "adaln-pre-denoise",
        "transformer-release",
        "video-vae-release",
        "audio-vae-release",
    ], reasons


def test_adaln_cache_materialization_is_repeated_after_projection_drop() -> None:
    events: list[str] = []
    cache = FakeModulationCache(events)
    dit = SimpleNamespace(parameters=lambda: [mx.array([1.0])])
    pipe = MiniMaxH3Pipeline(dit, None, None, None, PipelineConfig())

    def build(*args, **kwargs):
        events.append("cache_build")
        return cache

    def drop(_dit):
        events.append("adaln_drop")
        return 123

    clear_calls: list[int] = []
    with patched_attribute(pipeline_module.ModulationCache, "build", build):
        with patched_attribute(pipeline_module, "drop_adaln_weights", drop):
            with patched_attribute(pipeline_module.mx, "clear_cache", lambda: clear_calls.append(1)):
                pipe._ensure_cache(mx.array([0.0]), drop_adaln=True, verbose=False)

    assert events == ["cache_build", "cache_materialize", "adaln_drop", "cache_materialize"], events
    assert clear_calls == [1]
    assert pipe._adaln_purge_attempts == 1
    assert pipe._adaln_purge_status == "success"


def test_adaln_cache_arrays_remain_usable_after_purge() -> None:
    cache = ModulationCache([(mx.array([3.0, 4.0]),)], mx.array([0.0]))
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())
    calls: list[int] = []
    with patched_attribute(pipeline_module.mx, "clear_cache", lambda: calls.append(1)):
        assert pipe._purge_adaln_allocator_cache(verbose=False) == "success"
    cache.materialize()
    assert np.asarray(cache.get(0)[0]).tolist() == [3.0, 4.0]
    assert calls == [1]


def test_adaln_cache_purge_reports_success_telemetry() -> None:
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())
    output = io.StringIO()
    with patched_attribute(pipeline_module.mx, "clear_cache", lambda: None):
        with redirect_stdout(output):
            status = pipe._purge_adaln_allocator_cache(verbose=True)
    assert status == "success"
    assert "AdaLN allocator purge: success" in output.getvalue()
    assert "immediately before AdaLN allocator purge" in output.getvalue()
    assert "immediately after AdaLN allocator purge" in output.getvalue()


def test_adaln_cache_purge_missing_api_is_nonfatal_and_reported() -> None:
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())
    output = io.StringIO()
    with patched_attribute(pipeline_module.mx, "clear_cache", None):
        with redirect_stdout(output):
            status = pipe._purge_adaln_allocator_cache(verbose=True)
    assert status == "unavailable"
    assert pipe._adaln_purge_attempts == 1
    assert "AdaLN allocator purge: unavailable" in output.getvalue()


def test_adaln_cache_purge_failure_is_nonfatal_and_reported() -> None:
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())

    def fail():
        raise RuntimeError("fixture allocator failure")

    output = io.StringIO()
    with patched_attribute(pipeline_module.mx, "clear_cache", fail):
        with redirect_stdout(output):
            status = pipe._purge_adaln_allocator_cache(verbose=True)
    assert status == "failed"
    assert pipe._adaln_purge_attempts == 1
    assert "failed nonfatally" in output.getvalue()
    assert "AdaLN allocator purge: failed" in output.getvalue()


def test_adaln_cache_purge_is_not_repeated_inside_denoise_loop() -> None:
    clear_calls: list[int] = []
    with patched_attribute(pipeline_module.mx, "clear_cache", lambda: clear_calls.append(1)):
        events = run_fake_generation(images=False)
    assert len(clear_calls) == 5  # four release boundaries plus the dedicated AdaLN purge
    assert events.index("cache_purge") < events.index("denoise")
    assert events.count("allocator_purge:adaln-pre-denoise") == 1
    assert not any(event == "allocator_purge:denoise-loop" for event in events)


def test_regular_release_purge_failure_is_nonfatal() -> None:
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())
    pipe.dit = object()

    def fail():
        raise RuntimeError("fixture allocator failure")

    with patched_attribute(pipeline_module.mx, "clear_cache", fail):
        pipe._release_component("dit", "transformer", verbose=False)
    assert pipe.dit is None


def test_release_and_adaln_telemetry_distinguish_purge_reason() -> None:
    pipe = MiniMaxH3Pipeline(None, None, None, None, PipelineConfig())
    pipe.dit = object()
    output = io.StringIO()
    with redirect_stdout(output):
        with patched_attribute(pipeline_module.mx, "clear_cache", lambda: None):
            pipe._purge_adaln_allocator_cache(verbose=True)
            pipe._release_component("dit", "transformer", verbose=True)
    text = output.getvalue()
    assert "AdaLN allocator purge: success" in text
    assert "reason=adaln-pre-denoise" in text
    assert "allocator purge: success" in text
    assert "reason=transformer-release" in text
    assert "immediately after Python object reclamation (transformer-release)" in text


def test_denoising_timing_telemetry_distinguishes_first_and_later_steps() -> None:
    output = io.StringIO()
    with redirect_stdout(output):
        run_fake_generation(images=False, verbose=True)
    text = output.getvalue()
    assert "first denoising step:" in text
    assert "later denoising step mean:" in text
    assert "MLX cache after first denoising step:" in text


def main() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    failures: list[str] = []
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures.append(f"{test.__name__}: {exc}")
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        print("\n".join(failures))
        return 1
    print(f"\nstaged loading tests passed ({len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

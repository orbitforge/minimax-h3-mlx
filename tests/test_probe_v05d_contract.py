"""MLX-free behavioral contracts for the v0.5d derived full-schedule harness."""

from __future__ import annotations

import importlib.util
import ast
import gc
import json
import inspect
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
import wave
import weakref

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v05d_derived_full_schedule",
    ROOT / "scripts" / "probe_v05d_derived_full_schedule.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeTypedArray:
    def __init__(self, data, dtype):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype

    @property
    def shape(self):
        return self.data.shape

    def copy(self):
        return FakeTypedArray(self.data.copy(), self.dtype)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.data, dtype=dtype)

    def __add__(self, other):
        return FakeTypedArray(self.data + other, self.dtype)


class FakeMLXArray:
    def __init__(self, data, dtype, mx):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype
        self.mx = mx
        self.materialized = False

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype):
        self.mx.events.append(("astype", self.dtype, dtype))
        return FakeMLXArray(self.data.copy(), dtype, self.mx)

    def reshape(self, *shape):
        return FakeMLXArray(self.data.reshape(*shape), self.dtype, self.mx)

    def copy(self):
        return FakeMLXArray(self.data.copy(), self.dtype, self.mx)

    def __array__(self, dtype=None, copy=None):
        if not self.materialized:
            raise AssertionError("fake MLX array was inspected before materialization")
        return np.asarray(self.data, dtype=dtype)

    def __mul__(self, other):
        value = other.data if isinstance(other, FakeMLXArray) else other
        return FakeMLXArray(self.data * value, self.dtype, self.mx)

    def __add__(self, other):
        value = other.data if isinstance(other, FakeMLXArray) else other
        return FakeMLXArray(self.data + value, self.dtype, self.mx)


class FakeMLX:
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.events = []
        self.active = 0
        self.cache = 0
        self.peak = 0

    def array(self, value, dtype):
        self.events.append(("array", dtype))
        return FakeMLXArray(value, dtype, self)

    def eval(self, *values):
        self.events.append(("eval", len(values)))
        for value in values:
            if isinstance(value, FakeMLXArray):
                value.materialized = True

    def get_active_memory(self):
        return self.active

    def get_cache_memory(self):
        return self.cache

    def get_peak_memory(self):
        return self.peak

    def clear_cache(self):
        self.events.append(("clear_cache",))
        self.cache = 0


class FakeRuntimeMLXArray:
    """MLX-like bfloat16 value whose host-buffer conversion is deliberately forbidden."""

    __mlx_array__ = True

    def __init__(self, data, dtype, mx):
        self._data = np.asarray(data, dtype=np.float32).copy()
        self.dtype = dtype
        self.__mlx_core__ = mx
        self.materialized = False

    @property
    def shape(self):
        return self._data.shape

    def astype(self, dtype):
        return FakeRuntimeMLXArray(self._data, dtype, self.__mlx_core__)

    def copy(self):
        raise AssertionError("runtime MLX values must use mx.asarray(..., copy=True)")

    def __array__(self, dtype=None, copy=None):
        if not self.materialized:
            raise AssertionError("runtime MLX value was converted before materialization")
        if self.dtype == "bfloat16" and dtype in (None, "bfloat16"):
            raise AssertionError("runtime bfloat16 value must not cross through NumPy")
        return np.asarray(self._data, dtype=dtype)

    def __add__(self, other):
        value = other._data if isinstance(other, FakeRuntimeMLXArray) else other
        return FakeRuntimeMLXArray(self._data + value, self.dtype, self.__mlx_core__)


class FakeRuntimeMLX:
    float32 = "float32"
    bfloat16 = "bfloat16"

    def __init__(self):
        self.events = []

    def asarray(self, value, *, copy=None):
        self.events.append(("asarray", copy, value.dtype))
        if copy is not True:
            raise AssertionError("runtime MLX copy must request copy=True")
        return FakeRuntimeMLXArray(value._data, value.dtype, self)

    def eval(self, *values):
        self.events.append(("eval", len(values)))
        for value in values:
            if isinstance(value, FakeRuntimeMLXArray):
                value.materialized = True

    def isfinite(self, value):
        return np.isfinite(value._data)

    def all(self, value):
        return bool(np.all(value))


class RuntimeSentinel:
    pass


def exception_with_retained_sentinel(sentinel, message):
    retained = sentinel
    try:
        raise RuntimeError(message)
    except RuntimeError as error:
        return error


class FakeVideoConfig:
    latent_channels = 24
    out_channels = 3
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    clip_length = 17
    token_drop = 3
    latents_mean = tuple(0.0 for _ in range(24))
    latents_std = tuple(1.0 for _ in range(24))


class FakeAudioConfig:
    latent_channels = 32
    decoder_rates = (5, 5, 2, 2, 2, 2, 2)
    sampling_rate = 32000
    latents_mean = tuple(0.0 for _ in range(32))
    latents_std = tuple(1.0 for _ in range(32))

    @property
    def hop_length(self):
        return 800


def write_stereo_wav(path: Path, *, channels=2, sample_rate=32000, sample_count=40000, sample_width=2, truncate=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    silence = b"\x00" * channels * sample_count * sample_width
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(silence)
    if truncate:
        path.write_bytes(path.read_bytes()[:-7])


def write_rgb_frame_set(directory: Path, *, mode="RGB", size=(128, 128), count=30, empty_index=None):
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        path = directory / f"frame_{index:05d}.png"
        if empty_index == index:
            path.write_bytes(b"")
        else:
            Image.new(mode, size, 0).save(path)


def valid_ffprobe_json(**overrides):
    video = {
        "codec_type": "video",
        "codec_name": "h264",
        "width": 128,
        "height": 128,
        "avg_frame_rate": "24/1",
        "r_frame_rate": "24/1",
        "pix_fmt": "yuv420p",
        "nb_frames": "30",
        "duration": "1.25",
    }
    audio = {
        "codec_type": "audio",
        "codec_name": "aac",
        "channels": 2,
        "sample_rate": "32000",
    }
    container = {"duration": "1.25", "size": "12"}
    video.update(overrides.pop("video", {}))
    audio.update(overrides.pop("audio", {}))
    container.update(overrides.pop("format", {}))
    result = {"streams": [video, audio], "format": container}
    result.update(overrides)
    return result


class FakeMuxRunner:
    def __init__(
        self,
        *,
        ffmpeg_returncode=0,
        ffprobe_returncode=0,
        write_output=True,
        empty_output=False,
        ffprobe_json=None,
        timeout_tool=None,
        ffmpeg_stderr="",
        ffprobe_stderr="",
    ):
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffprobe_returncode = ffprobe_returncode
        self.write_output = write_output
        self.empty_output = empty_output
        self.ffprobe_json = ffprobe_json or valid_ffprobe_json()
        self.timeout_tool = timeout_tool
        self.ffmpeg_stderr = ffmpeg_stderr
        self.ffprobe_stderr = ffprobe_stderr
        self.calls = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        tool = Path(argv[0]).name
        if tool == "ffmpeg":
            if self.timeout_tool == "ffmpeg":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr="ffmpeg timeout")
            output = Path(argv[-1])
            if self.write_output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"" if self.empty_output else b"synthetic-mp4")
            return subprocess.CompletedProcess(argv, self.ffmpeg_returncode, stdout="ffmpeg stdout", stderr=self.ffmpeg_stderr)
        if tool == "ffprobe":
            if self.timeout_tool == "ffprobe":
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr="ffprobe timeout")
            return subprocess.CompletedProcess(
                argv,
                self.ffprobe_returncode,
                stdout=json.dumps(self.ffprobe_json),
                stderr=self.ffprobe_stderr,
            )
        raise AssertionError(f"unexpected tool: {tool}")


def mp4_media_fixture(root: Path):
    frames = root / "frames"
    write_rgb_frame_set(frames)
    video_manifest_path = root / "video-frame-manifest.json"
    probe._write_json(video_manifest_path, probe.build_video_frame_manifest(frames, attempt_identifier="attempt-1"))
    wav_path = root / "audio.wav"
    write_stereo_wav(wav_path)
    audio_manifest_path = root / "audio-manifest.json"
    probe._write_json(audio_manifest_path, probe.build_audio_wav_manifest(wav_path, attempt_identifier="attempt-1"))
    return {
        "frames": frames,
        "video_frame_manifest": video_manifest_path,
        "audio_wav": wav_path,
        "audio_manifest": audio_manifest_path,
        "mp4_partial": root / "dodecahedron.partial.mp4",
        "mp4": root / "dodecahedron.mp4",
        "mp4_manifest": root / "mp4-manifest.json",
        "attempt_identifier": "attempt-1",
    }


def successful_mux_gate():
    return {
        "derived_phase_status": "completed",
        "video_status": "completed",
        "audio_status": "completed",
        "standalone_media_status": "completed",
        "video_release_gate_passed": True,
        "audio_release_gate_passed": True,
        "video_worker_termination_confirmed": True,
        "audio_worker_termination_confirmed": True,
        "frame_manifest_valid": True,
        "wav_manifest_valid": True,
        "passed": True,
    }


def mux_report_fixture(paths):
    return {
        "attempt": {"attempt_identifier": paths["attempt_identifier"]},
        "output_paths": dict(paths),
        "latent_generation_status": "completed",
        "video_status": "completed",
        "audio_status": "completed",
        "standalone_media_status": "completed",
        "video_decoder": {
            "release_gate_passed": True,
            "worker_termination_confirmed": True,
            "worker_receipt": {"worker_termination_confirmed": True},
        },
        "audio_decoder": {
            "release_gate_passed": True,
            "worker_termination_confirmed": True,
            "worker_receipt": {"worker_termination_confirmed": True},
        },
        "video_artifacts": {},
        "audio_artifacts": {},
        "standalone_media": {"status": "completed"},
        "phase_order": ["video-worker", "audio-worker"],
        "invocation": {},
    }


class FakeMultimodalScheduler:
    prediction_parameterization = "velocity"
    input_scaling = "identity"

    def __init__(self, plan, *, fail_step: int | None = None, update_dtype: str | None = None):
        self.plan = plan
        self.num_inference_steps = len(plan.transitions)
        self.events: list[tuple[str, int]] = []
        self.fail_step = fail_step
        self.update_dtype = update_dtype
        self.prediction_dtypes: list[tuple[str, str]] = []

    def transition(self, step_index: int):
        return self.plan.transitions[step_index]

    def prepare_model_input(self, video, audio, step_index):
        self.events.append(("prepare", step_index))
        return video, audio

    def step(self, video_prediction, audio_prediction, video_sample, audio_sample, step_index):
        self.events.append(("scheduler", step_index))
        self.prediction_dtypes.append((video_prediction.dtype, audio_prediction.dtype))
        if self.fail_step == step_index:
            raise RuntimeError(f"scheduler failure at {step_index}")
        updated_video = video_sample + 1.0
        updated_audio = audio_sample + 1.0
        if self.update_dtype is not None:
            updated_video = FakeTypedArray(np.asarray(updated_video), self.update_dtype)
            updated_audio = FakeTypedArray(np.asarray(updated_audio), self.update_dtype)
        return updated_video, updated_audio


class FakeCache:
    pass


class DeferredPrediction:
    def __init__(self, shape, *, dtype="float32", fail=False):
        self._data = np.ones(shape, dtype=np.float32)
        self.dtype = dtype
        self.materialize_calls = 0
        self.materialized = False
        self.fail = fail

    @property
    def shape(self):
        return self._data.shape

    def materialize(self):
        self.materialize_calls += 1
        if self.fail:
            raise RuntimeError("deferred prediction materialization failed")
        self.materialized = True

    def __array__(self, dtype=None):
        if not self.materialized:
            raise AssertionError("deferred prediction was inspected before materialization")
        return np.asarray(self._data, dtype=dtype)


class FakeCacheProvider(probe.StreamedCacheSessionProvider):
    def __init__(self, *, fail_build_step: int | None = None, events=None, fail_cleanup_step: int | None = None):
        self.test_events = events if events is not None else []
        self.fail_build_step = fail_build_step
        self.fail_cleanup_step = fail_cleanup_step

        def builder(step_index, _timestep, telemetry):
            if self.fail_build_step == step_index:
                raise RuntimeError(f"cache failure at {step_index}")
            for block_index in range(probe.EXPECTED_BLOCK_COUNT):
                path = f"block-{block_index:03d}.safetensors"
                telemetry("sidecar_opening", {"block_index": block_index, "path": path})
                telemetry("sidecar_released", {"block_index": block_index, "path": path})
            return FakeCache(), {
                "blocks_completed": probe.EXPECTED_BLOCK_COUNT,
                "sidecar_files_opened": probe.EXPECTED_BLOCK_COUNT,
                "unique_sidecar_files_opened": probe.EXPECTED_BLOCK_COUNT,
                "successful_payload_opens": probe.EXPECTED_BLOCK_COUNT,
                "completed_payload_releases": probe.EXPECTED_BLOCK_COUNT,
                "every_sidecar_released_before_next_opened": True,
                "sidecar_overlap_observed": False,
                "next_sidecar_opened_before_previous_release": False,
                "dense_temporary_projection_created": False,
            }

        def cleanup(step_index, _cache):
            self.test_events.append(("cache-release", step_index))
            if self.fail_cleanup_step == step_index:
                raise RuntimeError(f"cleanup failure at {step_index}")

        super().__init__(builder, event_sink=self._record_event, cleanup_hook=cleanup)

    def _record_event(self, event, details):
        self.test_events.append((event, details.get("step_index", details.get("transition_index"))))


def run_fake(
    *,
    fail_cache: int | None = None,
    fail_forward: int | None = None,
    fail_scheduler: int | None = None,
    fail_cleanup: int | None = None,
    deferred: bool = False,
    fail_materialization: bool = False,
    prediction_dtype: str = "float32",
    expected_prediction_dtype: str = probe.CANONICAL_PREDICTION_DTYPE,
    latent_dtype: str = "float32",
    updated_dtype: str | None = None,
):
    plan = probe.build_full_schedule()
    scheduler = FakeMultimodalScheduler(plan, fail_step=fail_scheduler, update_dtype=updated_dtype)
    events: list[tuple[str, object]] = []
    provider = FakeCacheProvider(fail_build_step=fail_cache, events=events, fail_cleanup_step=fail_cleanup)
    calls: list[int] = []
    initial_video = np.zeros((1, 2), dtype=np.float32)
    initial_audio = np.zeros((1, 2), dtype=np.float32)
    if latent_dtype != "float32":
        initial_video = FakeTypedArray(initial_video, latent_dtype)
        initial_audio = FakeTypedArray(initial_audio, latent_dtype)

    def forward(step_index, _transition, video, audio, _timestep, _indices, _cache, _packed):
        calls.append(step_index)
        if fail_forward == step_index:
            raise RuntimeError(f"forward failure at {step_index}")
        if deferred:
            return probe.ForwardExecution(
                DeferredPrediction(video.shape, dtype=prediction_dtype, fail=fail_materialization),
                DeferredPrediction(audio.shape, dtype=prediction_dtype, fail=fail_materialization),
                timing_seconds=0.0,
            )
        video_prediction = np.asarray(video, dtype=np.float32) + 0.5
        audio_prediction = np.asarray(audio, dtype=np.float32) + 0.5
        if prediction_dtype != "float32":
            video_prediction = FakeTypedArray(video_prediction, prediction_dtype)
            audio_prediction = FakeTypedArray(audio_prediction, prediction_dtype)
        return probe.ForwardExecution(video_prediction, audio_prediction)

    result = probe.run_full_schedule(
        object(),
        scheduler,
        plan,
        initial_video_latent=initial_video,
        initial_audio_latent=initial_audio,
        timestep_provider=lambda _index, transition: (
            np.asarray([transition["video_current_timestep"], transition["audio_current_timestep"]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int32),
        ),
        packed_inputs={
            "token_tags": np.asarray([0, 2], dtype=np.int32),
            "position_ids": np.zeros((2, 3), dtype=np.float32),
            "video_indices": np.asarray([0], dtype=np.int32),
            "audio_indices": np.asarray([1], dtype=np.int32),
            "text_indices": np.asarray([0], dtype=np.int32),
        },
        cache_provider=provider,
        forward_runner=forward,
        memory_snapshot=lambda: {"active": 1, "allocator_cache": 0, "peak": 1},
        expected_video_dtype=latent_dtype,
        expected_audio_dtype=latent_dtype,
        expected_prediction_dtype=expected_prediction_dtype,
    )
    return result, scheduler, provider, calls, events


def valid_final_artifact():
    video = np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32)
    audio = np.zeros(probe.AUDIO_NATIVE_SHAPE, dtype=np.float32)
    artifact = {
        "artifact_identity": "minimax-h3-mlx-v05d-final-native-latent",
        "schema_version": probe.FINAL_ARTIFACT_SCHEMA_VERSION,
        "attempt_identifier": "attempt-1",
        "native_video": probe.shape_dtype(video, logical_dtype="bfloat16") | {"fingerprint": probe.array_fingerprint(video, logical_dtype="bfloat16")},
        "native_audio": probe.shape_dtype(audio, logical_dtype="bfloat16") | {"fingerprint": probe.array_fingerprint(audio, logical_dtype="bfloat16")},
        "packed_final_state_fingerprint": "packed",
        "schedule_contract": probe.build_full_schedule().receipt(),
        "completed_transition_count": 15,
        "transformer_forward_count": 15,
        "scheduler_update_counts": {"video": 15, "audio": 15},
        "streamed_adaln_lifecycle": probe.expected_lifecycle_totals(),
        "transformer_release_receipt": {
            "passed": True,
            "allocator_cache_zero": True,
            "active_memory_within_tolerance": True,
            "memory_after_allocator_purge": {"active": 0, "allocator_cache": 0},
        },
        "final_active_memory": 0,
        "final_allocator_cache": 0,
        "final_allocator_cache_zero": True,
        "final_artifact_npz_sha256": "artifact",
        "metadata_sha256": None,
        "worker_identity": "derived",
        "worker_exit_receipt": {
            "worker_started": True,
            "worker_exit_observed": True,
            "worker_exit_code": 0,
            "worker_pid": 123,
            "worker_termination_confirmed": True,
        },
        "memory_receipt": {},
        "git_identity": {},
        "checkpoint_identity": {},
    }
    artifact["streamed_adaln_lifecycle"]["sessions"] = probe.expected_lifecycle_sessions()
    artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
    return artifact, {"final_video_native": video, "final_audio_native": audio}


def full_builder_stats(**overrides):
    stats = {**probe.SESSION_STAT_FIELDS}
    stats.update(overrides)
    return stats


def provider_for_sidecar_events(events, *, stats=None, cleanup_hook=True):
    def builder(step_index, _timestep, telemetry):
        for event, block_index, path in events:
            telemetry(event, {"block_index": block_index, "path": path})
        return FakeCache(), stats or full_builder_stats()

    hook = (lambda _step, _cache: None) if cleanup_hook else None
    return probe.StreamedCacheSessionProvider(builder, cleanup_hook=hook)


def valid_sidecar_events():
    events = []
    for index in range(probe.EXPECTED_BLOCK_COUNT):
        path = f"block-{index:03d}.safetensors"
        events.extend([("sidecar_opening", index, path), ("sidecar_released", index, path)])
    return events


def write_valid_event_stream(path: Path):
    writer = probe.JsonlEventWriter(path)
    for transition_index in range(probe.EXPECTED_DENOISING_TRANSITIONS):
        session_id = f"cache-session-{transition_index + 1:02d}"
        details = {"step_index": transition_index, "transition_index": transition_index, "cache_session_id": session_id}
        writer("session-acquire-start", details)
        for block_index in range(probe.EXPECTED_BLOCK_COUNT):
            sidecar = {**details, "block_index": block_index, "path": f"block-{block_index:03d}.safetensors"}
            writer("sidecar_opening", sidecar)
            writer("sidecar_released", sidecar)
        writer("session-acquire-complete", details)
        writer("session-release-start", details)
        writer("session-release-complete", details)
    return writer


def conditioning_fixture(root: Path):
    ranges = probe.derive_row_ranges(probe.EXPECTED_TOKEN_COUNT)
    arrays = {
        "text_conditioning": np.zeros(probe.EXPECTED_CONDITIONING_SHAPE, dtype=np.float32),
        "token_ids": np.arange(probe.EXPECTED_TOKEN_COUNT, dtype=np.int32).reshape(1, -1),
        "token_presence_mask": np.ones((1, probe.EXPECTED_TOKEN_COUNT), dtype=np.int32),
        "text_token_tags": np.zeros((probe.EXPECTED_TOKEN_COUNT,), dtype=np.int32),
        "initial_video_native": np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32),
        "initial_audio_native": np.zeros(probe.AUDIO_NATIVE_SHAPE, dtype=np.float32),
        "packed_position_ids": np.zeros((probe.EXPECTED_TOTAL_ROWS, 3), dtype=np.float32),
        "packed_token_tags": np.concatenate(
            [
                np.zeros((probe.EXPECTED_TOKEN_COUNT,), dtype=np.int32),
                np.full(probe.EXPECTED_TARGET_AUDIO_ROWS, 2, dtype=np.int32),
                np.full(probe.EXPECTED_TARGET_VIDEO_ROWS, 0, dtype=np.int32),
            ]
        ),
        "packed_video_indices": np.arange(*ranges["target_video"], dtype=np.int32),
        "packed_audio_indices": np.arange(*ranges["target_audio"], dtype=np.int32),
        "packed_text_indices": np.arange(*ranges["text"], dtype=np.int32),
    }
    path = root / "conditioning.npz"
    probe._write_npz(path, arrays)
    receipt = {
        "status": "success",
        "prompt": probe.prompt_receipt(probe.LOCKED_PROMPT, arrays["token_ids"]),
        "tokenizer": {
            "token_ids": arrays["token_ids"].tolist(),
            "token_presence_mask": arrays["token_presence_mask"].tolist(),
        },
        "conditioning": {
            "shape": list(probe.EXPECTED_CONDITIONING_SHAPE),
            "dtype": probe.EXPECTED_CONDITIONING_DTYPE,
            "fingerprint": probe.array_fingerprint(arrays["text_conditioning"], logical_dtype=probe.EXPECTED_CONDITIONING_DTYPE),
        },
        "deterministic_inputs": probe.deterministic_input_receipt(arrays["initial_video_native"], arrays["initial_audio_native"]),
        "packing": {
            **probe.packed_contract(),
            "position_ids_shape": list(arrays["packed_position_ids"].shape),
            "token_tags_shape": list(arrays["packed_token_tags"].shape),
            "video_indices_shape": list(arrays["packed_video_indices"].shape),
            "audio_indices_shape": list(arrays["packed_audio_indices"].shape),
            "text_indices_shape": list(arrays["packed_text_indices"].shape),
        },
    }
    receipt["conditioning_artifact"] = probe.conditioning_artifact_binding(path, arrays)
    return path, arrays, receipt


def derived_gate_fixture(root: Path):
    artifact, arrays = valid_final_artifact()
    artifact["checkpoint_identity"] = {"checkpoint": "derived-checkpoint"}
    artifact_path = root / "final-native-latent.npz"
    metadata_path = root / "final-native-latent.json"
    probe._write_npz(artifact_path, arrays)
    artifact["final_artifact_npz_sha256"] = probe.sha256_file(artifact_path)
    artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
    probe._write_json(metadata_path, artifact)
    receipt = {
        **probe.decoder_worker_receipt("derived"),
        "status": "success",
        "transformer_release": artifact["transformer_release_receipt"],
    }
    gate = probe.validate_derived_decoder_gate(
        receipt,
        artifact_path,
        metadata_path,
        derived_worker_attempts=1,
        expected_attempt_identifier=artifact["attempt_identifier"],
        expected_checkpoint_identity=artifact["checkpoint_identity"],
    )
    return gate, receipt, artifact_path, metadata_path, artifact, arrays


def decoder_artifact_fixture(root: Path, worker_identity: str = "video"):
    if worker_identity == "video":
        key = "decoded_video"
        array = np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32)
    else:
        key = "decoded_audio"
        array = np.zeros(probe.AUDIO_NATIVE_SHAPE, dtype=np.float32)
    artifact_path = root / f"{worker_identity}-decoder.npz"
    metadata_path = root / f"{worker_identity}-decoder.json"
    probe._write_npz(artifact_path, {key: array})
    spec = probe.DecoderArtifactSpec(
        artifact_path,
        metadata_path,
        frozenset({key}),
        {key: {"shape": list(array.shape), "logical_dtype": "bfloat16", "storage_dtype": "float32"}},
        "attempt-1",
        {"checkpoint": "derived-checkpoint"},
        worker_identity,
    )
    metadata = probe.build_decoder_artifact_metadata(
        artifact_path,
        metadata_path=metadata_path,
        expected_keys=spec.expected_keys,
        array_specs=spec.array_specs,
        attempt_identifier=spec.attempt_identifier,
        checkpoint_identity=spec.checkpoint_identity,
        worker_identity=spec.worker_identity,
    )
    return spec, metadata, {key: array}


def run_decoder_orchestrator(
    derived_gate,
    *,
    video_receipt=None,
    audio_receipt=None,
    video_artifact=None,
    audio_artifact=None,
    implemented_phase_scope=None,
):
    events = []
    receipts = {
        "video": video_receipt or probe.decoder_worker_receipt("video"),
        "audio": audio_receipt or probe.decoder_worker_receipt("audio"),
    }

    def launch(identity):
        events.append(identity)
        return receipts[identity]

    result = probe.DecoderPhaseOrchestrator(
        derived_gate=derived_gate,
        worker_launcher=launch,
        video_artifact=video_artifact,
        audio_artifact=audio_artifact,
        implemented_phase_scope=implemented_phase_scope,
    ).run()
    return result, events


class ProbeV05DContractTests(unittest.TestCase):
    def test_numpy_and_existing_fake_typed_arrays_keep_their_copy_semantics(self):
        numpy_value = np.arange(4, dtype=np.float32)
        numpy_copy = probe._copy(numpy_value)
        self.assertIsNot(numpy_copy, numpy_value)
        np.testing.assert_array_equal(numpy_copy, numpy_value)

        fake_value = FakeTypedArray([1.0, 2.0], "bfloat16")
        fake_copy = probe._copy(fake_value)
        self.assertIsNot(fake_copy, fake_value)
        self.assertEqual(fake_copy.dtype, "bfloat16")
        np.testing.assert_array_equal(fake_copy.data, fake_value.data)

    def test_runtime_bfloat16_copy_is_mlx_native_materialized_and_independent(self):
        fake_mx = FakeRuntimeMLX()
        original = FakeRuntimeMLXArray([1.0, 2.0], fake_mx.bfloat16, fake_mx)
        copied = probe._copy(original)

        self.assertIsNot(copied, original)
        self.assertEqual(copied.dtype, fake_mx.bfloat16)
        self.assertTrue(copied.materialized)
        self.assertEqual(fake_mx.events[:2], [("asarray", True, "bfloat16"), ("eval", 1)])
        copied._data[0] = 99.0
        self.assertEqual(original._data[0], 1.0)
        self.assertEqual(copied._data[0], 99.0)

    def test_full_schedule_does_not_convert_bfloat16_latents_through_numpy(self):
        fake_mx = FakeRuntimeMLX()
        plan = probe.build_full_schedule()
        scheduler = FakeMultimodalScheduler(plan)
        cache_provider = FakeCacheProvider()
        initial_video = FakeRuntimeMLXArray(np.zeros((1, 2), dtype=np.float32), fake_mx.bfloat16, fake_mx)
        initial_audio = FakeRuntimeMLXArray(np.zeros((1, 2), dtype=np.float32), fake_mx.bfloat16, fake_mx)

        def forward(_step, _transition, video, audio, _timestep, _indices, _cache, _packed):
            return probe.ForwardExecution(
                FakeRuntimeMLXArray(np.ones(video.shape, dtype=np.float32), fake_mx.float32, fake_mx),
                FakeRuntimeMLXArray(np.ones(audio.shape, dtype=np.float32), fake_mx.float32, fake_mx),
            )

        packed = {
            "token_tags": np.asarray([0, 2], dtype=np.int32),
            "position_ids": np.zeros((2, 3), dtype=np.float32),
            "video_indices": np.asarray([0], dtype=np.int32),
            "audio_indices": np.asarray([1], dtype=np.int32),
            "text_indices": np.asarray([0], dtype=np.int32),
        }
        original_array = probe.np.array

        def reject_bfloat16_host_conversion(value, *args, **kwargs):
            if isinstance(value, FakeRuntimeMLXArray) and value.dtype == fake_mx.bfloat16:
                raise AssertionError("run_full_schedule attempted a NumPy conversion of bfloat16 latent")
            return original_array(value, *args, **kwargs)

        probe.np.array = reject_bfloat16_host_conversion
        try:
            result = probe.run_full_schedule(
                object(),
                scheduler,
                plan,
                initial_video_latent=initial_video,
                initial_audio_latent=initial_audio,
                timestep_provider=lambda _index, transition: (
                    np.asarray([transition["video_current_timestep"], transition["audio_current_timestep"]], dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int32),
                ),
                packed_inputs=packed,
                cache_provider=cache_provider,
                forward_runner=forward,
                memory_snapshot=lambda: {"active": 1, "allocator_cache": 0, "peak": 1},
                expected_video_dtype=fake_mx.bfloat16,
                expected_audio_dtype=fake_mx.bfloat16,
            )
        finally:
            probe.np.array = original_array

        self.assertEqual(result.final_video_latent.dtype, fake_mx.bfloat16)
        self.assertEqual(result.final_audio_latent.dtype, fake_mx.bfloat16)
        self.assertGreaterEqual(sum(event[0] == "asarray" for event in fake_mx.events), 2)

    def test_failure_evidence_detaches_traceback_and_collects_runtime_sentinel(self):
        sentinel = RuntimeSentinel()
        sentinel_ref = weakref.ref(sentinel)
        primary = exception_with_retained_sentinel(sentinel, "primary failure")
        del sentinel
        gc.collect()
        self.assertIsNotNone(sentinel_ref())

        evidence = probe.serialize_and_detach_error(primary)
        self.assertEqual(evidence["type"], "RuntimeError")
        self.assertEqual(evidence["message"], "primary failure")
        self.assertIn("RuntimeError: primary failure", evidence["traceback"])
        self.assertIsNone(primary.__traceback__)
        del primary
        gc.collect()
        self.assertIsNone(sentinel_ref())

    def test_primary_and_cleanup_evidence_remain_independently_serialized(self):
        primary = exception_with_retained_sentinel(RuntimeSentinel(), "primary failure")
        cleanup = exception_with_retained_sentinel(RuntimeSentinel(), "cleanup failure")
        primary_evidence, cleanup_evidence = probe.serialize_and_detach_failure(primary, cleanup)

        self.assertEqual(primary_evidence["type"], "RuntimeError")
        self.assertEqual(primary_evidence["message"], "primary failure")
        self.assertIn("RuntimeError: primary failure", primary_evidence["traceback"])
        self.assertEqual(cleanup_evidence["type"], "RuntimeError")
        self.assertEqual(cleanup_evidence["message"], "cleanup failure")
        self.assertIn("RuntimeError: cleanup failure", cleanup_evidence["traceback"])

    def test_release_failure_is_cleanup_and_cannot_replace_primary_evidence(self):
        primary = probe.serialize_and_detach_error(
            exception_with_retained_sentinel(RuntimeSentinel(), "primary failure")
        )
        preserved_primary, cleanup = probe.preserve_primary_on_release_failure(
            primary,
            None,
            RuntimeError("release failure"),
        )

        self.assertEqual(preserved_primary["message"], "primary failure")
        self.assertEqual(cleanup["message"], "release failure")

    def test_canonical_prediction_dtype_is_float32(self):
        self.assertEqual(probe.CANONICAL_PREDICTION_DTYPE, "float32")
        self.assertEqual(
            inspect.signature(probe.run_full_schedule).parameters["expected_prediction_dtype"].default,
            probe.CANONICAL_PREDICTION_DTYPE,
        )

    def test_derived_worker_passes_canonical_prediction_dtype_to_full_schedule(self):
        source = textwrap.dedent(inspect.getsource(probe._derived_worker_main))
        tree = ast.parse(source)
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "run_full_schedule"
        ]
        self.assertEqual(len(calls), 1)
        keyword = next(item for item in calls[0].keywords if item.arg == "expected_prediction_dtype")
        self.assertIsInstance(keyword.value, ast.Name)
        self.assertEqual(keyword.value.id, "CANONICAL_PREDICTION_DTYPE")
        self.assertNotIn('expected_prediction_dtype="bfloat16"', source)

    def test_float32_predictions_are_accepted_by_the_full_schedule_contract(self):
        result, scheduler, _provider, _calls, _events = run_fake(
            prediction_dtype=probe.CANONICAL_PREDICTION_DTYPE,
        )
        self.assertEqual(result.transformer_forwards, probe.EXPECTED_TRANSFORMER_FORWARDS)
        self.assertTrue(all(dtypes == ("float32", "float32") for dtypes in scheduler.prediction_dtypes))

    def test_bfloat16_predictions_are_rejected_by_the_float32_production_contract(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(prediction_dtype="bfloat16", deferred=True)
        self.assertIn("video transformer prediction dtype mismatch", str(context.exception))
        self.assertIn("expected 'float32'", str(context.exception))

    def test_scheduler_updated_rows_and_native_latents_remain_bfloat16(self):
        result, scheduler, _provider, _calls, _events = run_fake(
            latent_dtype="bfloat16",
            updated_dtype="bfloat16",
        )
        self.assertTrue(all(dtypes == ("float32", "float32") for dtypes in scheduler.prediction_dtypes))
        self.assertEqual(result.final_video_latent.dtype, "bfloat16")
        self.assertEqual(result.final_audio_latent.dtype, "bfloat16")
        self.assertEqual(result.final_native_video.dtype, "bfloat16")
        self.assertEqual(result.final_native_audio.dtype, "bfloat16")
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(latent_dtype="bfloat16", updated_dtype="float32")
        self.assertIn("updated video latent dtype mismatch", str(context.exception))

    def test_locked_prompt_literal_bytes_hash_and_token_policy(self):
        receipt = probe.prompt_receipt(probe.LOCKED_PROMPT)
        self.assertEqual(receipt["utf8_byte_count"], 482)
        self.assertEqual(receipt["sha256"], "c7d57d0bf61aa78dfe79d3267c13fc74b91bc397e09f1d73c35d12f4179dd00a")
        self.assertTrue(receipt["prompt_is_literal"])
        self.assertFalse(receipt["special_tokens"])
        self.assertIsNone(receipt["chat_template"])
        self.assertIsNone(receipt["negative_prompt"])

    def test_prompt_rejects_rewrite_and_nonzero_seed(self):
        with self.assertRaises(ValueError):
            probe.validate_locked_prompt(probe.LOCKED_PROMPT + " extra")
        with self.assertRaisesRegex(ValueError, "seed 0"):
            probe.validate_seed(1)

    def test_geometry_is_locked(self):
        geometry = probe.canonical_geometry_contract()
        self.assertEqual(geometry["resolution"], [128, 128])
        self.assertEqual(geometry["frames"], 30)
        self.assertEqual(tuple(geometry["video_native_latent_shape"]), (1, 24, 9, 8, 8))
        self.assertEqual(tuple(geometry["audio_native_latent_shape"]), (2, 32, 50))
        self.assertEqual(geometry["total_rows"], 347)

    def test_row_ranges_and_order_are_locked(self):
        contract = probe.packed_contract()
        probe.validate_packed_contract(contract)
        self.assertEqual(contract["row_ranges"], {"text": [0, 103], "target_audio": [103, 203], "target_video": [203, 347], "text_rows": 103, "target_audio_rows": 100, "target_video_rows": 144, "total_rows": 347})
        with self.assertRaisesRegex(ValueError, "row order"):
            probe.validate_packed_contract({**contract, "row_order": "[audio | video]"})

    def test_schedule_is_built_from_float32_grid_and_shifts(self):
        plan = probe.build_full_schedule()
        receipt = plan.receipt()
        self.assertEqual(receipt["requested_sigma_points"], 16)
        self.assertEqual(receipt["effective_video_sigma_points"], 16)
        self.assertEqual(receipt["effective_audio_sigma_points"], 16)
        self.assertEqual(receipt["denoising_transitions"], 15)
        self.assertEqual(receipt["transformer_forwards"], 15)
        self.assertEqual(receipt["video_shift"], 12.0)
        self.assertEqual(receipt["audio_shift"], 3.0)
        self.assertEqual(receipt["base_sigma_grid"][0], 1.0)
        self.assertEqual(receipt["base_sigma_grid"][-1], 0.0)
        self.assertEqual(receipt["video_shifted_sigma_grid"][-1], 0.0)
        self.assertEqual(receipt["audio_shifted_sigma_grid"][-1], 0.0)

    def test_schedule_validator_rejects_terminal_zero_missing(self):
        broken = probe.build_full_schedule().receipt()
        broken["video_shifted_sigma_grid"][-1] = 0.1
        with self.assertRaisesRegex(ValueError, "terminal zero"):
            probe.validate_schedule_contract(broken)

    def test_schedule_validator_rejects_reordered_transition(self):
        broken = probe.build_full_schedule().receipt()
        broken["transitions"][2]["step_index"] = 7
        with self.assertRaisesRegex(ValueError, "ordered"):
            probe.validate_schedule_contract(broken)

    def test_schedule_validator_rejects_effective_video_count_mismatch(self):
        broken = probe.build_full_schedule().receipt()
        broken["effective_video_sigma_points"] = 15
        with self.assertRaisesRegex(ValueError, "counts"):
            probe.validate_schedule_contract(broken)

    def test_schedule_validator_rejects_requested_count_mismatch(self):
        broken = probe.build_full_schedule().receipt()
        broken["requested_sigma_points"] = 17
        with self.assertRaisesRegex(ValueError, "counts"):
            probe.validate_schedule_contract(broken)

    def test_schedule_validator_rejects_forward_count_mismatch(self):
        broken = probe.build_full_schedule().receipt()
        broken["transformer_forwards"] = 14
        with self.assertRaisesRegex(ValueError, "counts"):
            probe.validate_schedule_contract(broken)

    def test_full_schedule_completes_exactly_fifteen_transitions(self):
        result, scheduler, provider, calls, _events = run_fake()
        self.assertEqual(len(result.transitions), 15)
        self.assertEqual([item["step_index"] for item in result.transitions], list(range(15)))
        self.assertEqual(calls, list(range(15)))
        self.assertEqual([item[1] for item in scheduler.events if item[0] == "scheduler"], list(range(15)))
        self.assertEqual(result.transformer_forwards, 15)
        self.assertEqual(result.video_scheduler_updates, 15)
        self.assertEqual(result.audio_scheduler_updates, 15)
        self.assertEqual(result.lifecycle["cache_sessions"], 15)
        self.assertEqual(result.lifecycle["cache_sessions_released"], 15)
        self.assertEqual(result.lifecycle["sidecar_opens"], 750)
        self.assertEqual(result.lifecycle["sidecar_releases"], 750)
        self.assertEqual(result.lifecycle["blocks_completed"], 750)
        self.assertEqual(result.lifecycle["maximum_simultaneous_sidecars"], 1)
        self.assertEqual(result.lifecycle["overlap_violations"], 0)
        self.assertEqual(result.lifecycle["dense_temporary_reconstructions"], 0)
        self.assertEqual(result.lifecycle["open_sidecars_after_cleanup"], 0)

    def test_cache_release_precedes_scheduler_update(self):
        result, scheduler, _provider, _calls, events = run_fake()
        self.assertEqual(len(result.transitions), 15)
        for index in range(15):
            release_positions = [position for position, item in enumerate(events) if item == ("cache-release", index)]
            scheduler_positions = [position for position, item in enumerate(scheduler.events) if item == ("scheduler", index)]
            self.assertEqual(len(release_positions), 1)
            self.assertEqual(len(scheduler_positions), 1)
        self.assertEqual(events.count(("cache-release", 14)), 1)

    def test_provider_rejects_overlapping_session(self):
        provider = FakeCacheProvider()
        provider.active = True
        with self.assertRaisesRegex(RuntimeError, "overlapped"):
            provider.cache_for_step(0, np.asarray([0.0], dtype=np.float32))

    def test_provider_records_each_sidecar_event(self):
        provider = FakeCacheProvider()
        cache = provider.cache_for_step(0, np.asarray([0.0], dtype=np.float32))
        provider.release_step(0, cache)
        record = provider.records[0]
        self.assertEqual(record["sidecar_opens"], 50)
        self.assertEqual(record["sidecar_releases"], 50)
        self.assertEqual(record["open_sidecars"], 0)
        self.assertEqual(record["status"], "released")

    def test_provider_never_reports_dense_reconstruction_for_fake_builder(self):
        provider = FakeCacheProvider()
        cache = provider.cache_for_step(0, np.asarray([0.0], dtype=np.float32))
        provider.release_step(0, cache)
        self.assertEqual(provider.aggregate()["dense_temporary_reconstructions"], 0)

    def test_exact_one_forward_per_transition(self):
        result, _scheduler, _provider, calls, _events = run_fake()
        self.assertEqual(calls, list(range(15)))
        self.assertTrue(all(item["transformer_forwards"] == 1 for item in result.transitions))

    def test_deferred_predictions_materialize_before_release_and_timing_ends_after_materialization(self):
        result, _scheduler, provider, _calls, _events = run_fake(deferred=True)
        self.assertEqual(result.transformer_forwards, 15)
        self.assertTrue(all(item["prediction_materialized"] for item in result.transitions))
        self.assertTrue(all(item["timings"]["materialization_completed_before_release"] for item in result.transitions))
        self.assertTrue(all(record["forward_materialized_before_release"] for record in provider.records))
        self.assertTrue(all(record["local_cache_reference_dropped"] for record in provider.records))
        self.assertTrue(all(record["provider_cache_reference_dropped"] for record in provider.records))
        self.assertTrue(all(item["timings"]["transformer_forward_seconds"] > 0.0 for item in result.transitions))

    def test_failed_materialization_does_not_count_a_completed_forward_or_normal_release(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(deferred=True, fail_materialization=True)
        state = context.exception.state
        self.assertEqual(state["transformer_forward_count"], 0)
        self.assertEqual(state["transitions"][0]["transformer_forwards"], 0)
        self.assertEqual(state["streamed_adaln_lifecycle"]["cache_sessions_released"], 0)
        self.assertEqual(state["streamed_adaln_lifecycle"]["sessions"][0]["status"], "failed-cleanup")

    def test_cache_reference_drop_is_recorded_before_scheduler_update(self):
        result, scheduler, provider, _calls, _events = run_fake()
        self.assertEqual(len(result.transitions), len(scheduler.events) // 2)
        for index, transition in enumerate(result.transitions):
            self.assertTrue(transition["cache_reference_dropped_before_scheduler"])
            self.assertTrue(provider.records[index]["local_cache_reference_dropped"])
            self.assertEqual(scheduler.events[index * 2][0], "prepare")
            self.assertEqual(scheduler.events[index * 2 + 1][0], "scheduler")

    def test_missing_cache_release_path_is_not_reported_as_released(self):
        provider = provider_for_sidecar_events(
            valid_sidecar_events(),
            stats=full_builder_stats(blocks_completed=50, sidecar_files_opened=50, unique_sidecar_files_opened=50,
                                     successful_payload_opens=50, completed_payload_releases=50),
            cleanup_hook=False,
        )
        cache = provider.cache_for_step(0, np.asarray([0.0], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, "bounded release path"):
            provider.release_step(0, cache)
        self.assertEqual(provider.aggregate()["cache_sessions_released"], 0)

    def test_builder_stats_are_required_and_dense_reconstruction_is_fail_loud(self):
        for overrides, message in (
            ({"dense_temporary_projection_created": None}, "dense_temporary_projection_created"),
            ({"successful_payload_opens": None}, "successful_payload_opens"),
            ({"blocks_completed": 49}, "blocks_completed"),
            ({"blocks_completed": 51}, "blocks_completed"),
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, message):
                provider = provider_for_sidecar_events(valid_sidecar_events(), stats=full_builder_stats(**overrides))
                provider.cache_for_step(0, np.asarray([0.0], dtype=np.float32))

    def test_sidecar_state_machine_rejects_release_without_open_duplicate_open_and_out_of_order(self):
        cases = {
            "release without open": [("sidecar_released", 0, "block-000.safetensors")],
            "duplicate open": [("sidecar_opening", 0, "block-000.safetensors"), ("sidecar_opening", 0, "block-000.safetensors")],
            "out of order": [("sidecar_opening", 1, "block-001.safetensors")],
            "mismatched release": [("sidecar_opening", 0, "block-000.safetensors"), ("sidecar_released", 1, "block-001.safetensors")],
            "duplicate release": [("sidecar_opening", 0, "block-000.safetensors"), ("sidecar_released", 0, "block-000.safetensors"), ("sidecar_released", 0, "block-000.safetensors")],
        }
        for label, events in cases.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                provider_for_sidecar_events(events).cache_for_step(0, np.asarray([0.0], dtype=np.float32))

    def test_aggregate_totals_cannot_hide_an_invalid_individual_session(self):
        _result, _scheduler, provider, _calls, _events = run_fake()
        provider.records[0]["blocks_completed"] = 49
        provider.records[1]["blocks_completed"] = 51
        observed = provider.aggregate()
        self.assertEqual(observed["blocks_completed"], 750)
        with self.assertRaisesRegex(ValueError, "session 0 blocks_completed"):
            probe.validate_lifecycle_totals(observed)

    def test_jsonl_validation_reports_exact_record_categories_and_rejects_identity_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            write_valid_event_stream(path)
            summary = probe.validate_event_stream(path)
            self.assertEqual(summary["cache_session_count"], 15)
            self.assertEqual(summary["sidecar_open_event_count"], 750)
            self.assertEqual(summary["sidecar_release_event_count"], 750)
            self.assertEqual(summary["validated_block_pairs"], 750)
            lines = path.read_text().splitlines()
            lines[2] = lines[2].replace('"block_index": 0', '"block_index": 1')
            path.write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "does not match"):
                probe.validate_event_stream(path)

    def test_conditioning_artifact_binding_rejects_bytes_array_and_key_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arrays, receipt = conditioning_fixture(Path(directory))
            probe.validate_conditioning_artifact_binding(receipt, path)
            original = path.read_bytes()
            path.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                probe.validate_conditioning_artifact_binding(receipt, path)
            probe._write_npz(path, {**arrays, "text_conditioning": arrays["text_conditioning"] + 1.0})
            receipt["conditioning_artifact"]["sha256"] = probe.sha256_file(path)
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                probe.validate_conditioning_artifact_binding(receipt, path)
            probe._write_npz(path, {key: value for key, value in arrays.items() if key != "token_ids"})
            with self.assertRaisesRegex(ValueError, "key set"):
                probe.validate_conditioning_artifact_binding(receipt, path)

    def test_derived_worker_validates_conditioning_binding_before_transformer_load(self):
        source = __import__("inspect").getsource(probe._derived_worker_main)
        self.assertLess(source.index("validate_conditioning_artifact_binding"), source.index("load_dit("))

    def test_final_artifact_requires_post_release_memory_and_stable_metadata_evidence(self):
        artifact, arrays = valid_final_artifact()
        probe.validate_final_artifact(artifact, arrays=arrays)
        artifact["final_allocator_cache_zero"] = False
        artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
        with self.assertRaisesRegex(ValueError, "allocator-cache-zero"):
            probe.validate_final_artifact(artifact, arrays=arrays)

    def test_finalization_consumes_worker_transformer_release_boundary(self):
        artifact, _arrays = valid_final_artifact()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata_path = root / "metadata.json"
            artifact_path = root / "artifact.npz"
            probe._write_json(metadata_path, artifact)
            probe._write_npz(
                artifact_path,
                {
                    "final_video_native": np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32),
                    "final_audio_native": np.zeros(probe.AUDIO_NATIVE_SHAPE, dtype=np.float32),
                },
            )
            boundary = {
                "worker_started": True,
                "worker_exit_observed": True,
                "worker_exit_code": 0,
                "worker_pid": 123,
                "worker_termination_confirmed": True,
                "transformer_release": artifact["transformer_release_receipt"],
            }
            finalized = probe._finalize_artifact_worker_exit(metadata_path, boundary, artifact_path=artifact_path)
            self.assertEqual(finalized["transformer_release_receipt"], artifact["transformer_release_receipt"])
            self.assertEqual(finalized["final_artifact_npz_sha256"], probe.sha256_file(artifact_path))

    def test_expected_lifecycle_totals_are_observed_not_copied(self):
        _result, _scheduler, provider, _calls, _events = run_fake()
        observed = provider.aggregate()
        expected = {**probe.expected_lifecycle_totals(), "transformer_forwards": 0, "video_scheduler_updates": 0, "audio_scheduler_updates": 0}
        self.assertEqual({key: observed[key] for key in expected}, expected)
        complete = {**probe.expected_lifecycle_totals(), "sessions": probe.expected_lifecycle_sessions()}
        with self.assertRaisesRegex(ValueError, "sidecar_opens"):
            probe.validate_lifecycle_totals({**complete, "sidecar_opens": 749})

    def test_final_artifact_schema_and_fingerprints(self):
        artifact, arrays = valid_final_artifact()
        probe.validate_final_artifact(artifact, arrays=arrays)
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            probe.validate_final_artifact({**artifact, "unexpected": True})

    def test_final_artifact_requires_confirmed_worker_termination(self):
        artifact, arrays = valid_final_artifact()
        artifact["worker_exit_receipt"]["worker_termination_confirmed"] = False
        with self.assertRaisesRegex(ValueError, "termination"):
            probe.validate_final_artifact(artifact, arrays=arrays)

    def test_final_artifact_rejects_incomplete_transition_count(self):
        artifact, arrays = valid_final_artifact()
        artifact["completed_transition_count"] = 14
        with self.assertRaisesRegex(ValueError, "transition"):
            probe.validate_final_artifact(artifact, arrays=arrays)

    def test_final_artifact_rejects_lifecycle_total_mismatch(self):
        artifact, arrays = valid_final_artifact()
        artifact["streamed_adaln_lifecycle"]["sidecar_releases"] = 749
        with self.assertRaisesRegex(ValueError, "sidecar_releases"):
            probe.validate_final_artifact(artifact, arrays=arrays)

    def test_final_artifact_rejects_open_sidecars(self):
        artifact, arrays = valid_final_artifact()
        artifact["streamed_adaln_lifecycle"]["open_sidecars_after_cleanup"] = 1
        with self.assertRaisesRegex(ValueError, "open_sidecars"):
            probe.validate_final_artifact(artifact, arrays=arrays)

    def test_worker_boundary_requires_exit_and_receipt(self):
        valid = {"worker_identity": "derived", "status": "success", "worker_started": True, "worker_exit_observed": True, "worker_exit_code": 0, "worker_pid": 123, "worker_termination_confirmed": True, "worker_receipt_valid": True}
        probe.validate_worker_boundary(valid, identity="derived")
        for field, value in (("worker_receipt_valid", False), ("worker_termination_confirmed", False), ("worker_exit_observed", False)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_worker_boundary({**valid, field: value}, identity="derived")

    def test_worker_boundary_rejects_missing_receipt_identity(self):
        with self.assertRaisesRegex(ValueError, "identity"):
            probe.validate_worker_boundary({"status": "success"}, identity="conditioning")

    def test_conditioning_receipt_requires_locked_prompt_and_release(self):
        receipt = {
            "status": "success",
            "prompt": probe.prompt_receipt(probe.LOCKED_PROMPT, np.zeros((1, 103), dtype=np.int32)),
            "tokenizer": {"token_ids": [[0] * 103]},
            "conditioning": {"shape": [1, 103, 5120], "dtype": "bfloat16", "fingerprint": "conditioning"},
            "packing": probe.packed_contract(),
            "geometry": probe.canonical_geometry_contract(),
            "conditioning_release": {"passed": True},
            "conditioning_artifact": {"sha256": "artifact", "array_keys": sorted(probe.CONDITIONING_ARRAY_KEYS), "arrays": {}},
        }
        probe.validate_conditioning_receipt(receipt)
        receipt["prompt"]["sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "prompt"):
            probe.validate_conditioning_receipt(receipt)

    def test_conditioning_receipt_rejects_allocator_gate_failure(self):
        receipt = {
            "status": "success",
            "prompt": probe.prompt_receipt(probe.LOCKED_PROMPT, np.zeros((1, 103), dtype=np.int32)),
            "tokenizer": {"token_ids": [[0] * 103]},
            "conditioning": {"shape": [1, 103, 5120], "dtype": "bfloat16", "fingerprint": "conditioning"},
            "packing": probe.packed_contract(),
            "geometry": probe.canonical_geometry_contract(),
            "conditioning_release": {"passed": False},
            "conditioning_artifact": {"sha256": "artifact", "array_keys": sorted(probe.CONDITIONING_ARRAY_KEYS), "arrays": {}},
        }
        with self.assertRaisesRegex(ValueError, "release"):
            probe.validate_conditioning_receipt(receipt)

    def test_fresh_namespace_refuses_nonempty_and_preserves_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt"
            paths = probe.ensure_attempt_namespace(root)
            evidence = root / "prior-evidence.json"
            evidence.write_text("keep")
            with self.assertRaises(FileExistsError):
                probe.ensure_attempt_namespace(root)
            self.assertEqual(evidence.read_text(), "keep")
            self.assertTrue(paths["namespace_newly_created"])

    def test_empty_precreated_namespace_records_not_new(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "attempt"
            root.mkdir()
            paths = probe.ensure_attempt_namespace(root)
            self.assertFalse(paths["namespace_newly_created"])

    def test_attempt_paths_contain_video_and_audio_publication_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            path_text = " ".join(map(str, paths.values()))
            self.assertIn("audio.partial.wav", path_text)
            self.assertIn("audio.wav", path_text)
            self.assertIn("audio-manifest.json", path_text)
            self.assertIn("audio-worker-receipt.json", path_text)
            self.assertIn("audio-worker.log", path_text)
            self.assertIn("frames.partial", path_text)
            self.assertIn("dodecahedron.partial.mp4", path_text)
            self.assertIn("dodecahedron.mp4", path_text)
            self.assertIn("mp4-manifest.json", path_text)

    def test_initial_report_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            args = type("Args", (), {"prompt": probe.LOCKED_PROMPT, "operator_declared_uncontended": False})()
            report = probe._base_report(args, paths)
            self.assertEqual(report["status"], "incomplete")
            self.assertEqual(report["run_state"], "incomplete")
            probe.validate_report(report)

    def test_failure_stops_after_cache_failure_without_retry(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_cache=4)
        failure = context.exception
        self.assertEqual(failure.state["completed_transition_count"], 4)
        self.assertEqual([item["step_index"] for item in failure.state["transitions"]], [0, 1, 2, 3, 4])
        self.assertEqual(failure.state["transitions"][-1]["status"], "incomplete")
        self.assertEqual(failure.state["streamed_adaln_lifecycle"]["cache_sessions"], 5)

    def test_failure_stops_after_forward_failure_without_retry(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_forward=6)
        self.assertEqual(context.exception.state["completed_transition_count"], 6)
        self.assertEqual(context.exception.state["transformer_forward_count"], 6)
        self.assertEqual(len(context.exception.state["transitions"]), 7)

    def test_failure_stops_after_scheduler_failure_without_retry(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_scheduler=8)
        self.assertEqual(context.exception.state["completed_transition_count"], 8)
        self.assertEqual(context.exception.state["scheduler_update_counts"], {"video": 8, "audio": 8})
        self.assertEqual(len(context.exception.state["transitions"]), 9)

    def test_scheduler_failure_does_not_count_an_update(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_scheduler=1)
        self.assertEqual(context.exception.state["scheduler_update_counts"], {"video": 1, "audio": 1})
        self.assertEqual(context.exception.state["transitions"][-1]["video_scheduler_updates"], 0)
        self.assertEqual(context.exception.state["transitions"][-1]["audio_scheduler_updates"], 0)

    def test_cache_failure_records_primary_and_cleanup_fields(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_cache=0)
        fields = context.exception.state["failure"]
        self.assertEqual(fields["primary_error_type"], "RuntimeError")
        self.assertTrue(fields["cleanup_attempted"])
        self.assertTrue(fields["cleanup_succeeded"])

    def test_primary_error_is_preserved_when_cache_cleanup_succeeds(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_forward=3)
        failure = context.exception
        self.assertEqual(type(failure.primary_error).__name__, "RuntimeError")
        self.assertIsNone(failure.cleanup_error)
        self.assertTrue(failure.state["failure"]["cleanup_succeeded"])
        self.assertIsNone(failure.state["failure"]["cleanup_error_type"])

    def test_primary_and_cleanup_errors_are_both_preserved(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_forward=3, fail_cleanup=3)
        failure = context.exception
        self.assertIn("forward failure", str(failure.primary_error))
        self.assertIn("cleanup failure", str(failure.cleanup_error))
        self.assertEqual(failure.state["failure"]["primary_error_type"], "RuntimeError")
        self.assertEqual(failure.state["failure"]["cleanup_error_type"], "RuntimeError")
        self.assertFalse(failure.state["failure"]["cleanup_succeeded"])

    def test_partial_transition_telemetry_is_retained(self):
        with self.assertRaises(probe.DenoisingFailure) as context:
            run_fake(fail_forward=2)
        partial = context.exception.state["transitions"][-1]
        self.assertEqual(partial["step_index"], 2)
        self.assertEqual(partial["status"], "incomplete")
        self.assertIn("failure", partial)
        self.assertIn("memory", partial)
        self.assertIn("timings", partial)

    def test_no_retry_policy_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            args = type("Args", (), {"prompt": probe.LOCKED_PROMPT, "operator_declared_uncontended": False})()
            report = probe._base_report(args, paths)
            policy = report["invocation"]["retry_policy"]
            self.assertEqual(report["invocation"]["attempts"], 1)
            self.assertFalse(policy["internal_retry_loop"])
            self.assertFalse(policy["automatic_worker_replacement"])
            self.assertFalse(policy["second_generation_attempt"])

    def test_failure_report_suppresses_derived_when_conditioning_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            args = type("Args", (), {"prompt": probe.LOCKED_PROMPT, "operator_declared_uncontended": False})()
            report = probe._base_report(args, paths)
            report["phase_order"] = ["preflight", "conditioning-worker"]
            probe._failure_report(report, "conditioning-worker", "conditioning", RuntimeError("bad prompt"), cleanup_attempted=True)
            self.assertTrue(report["failure"]["later_phase_suppression"]["derived_worker_suppressed"])
            self.assertTrue(report["failure"]["later_phase_suppression"]["decoder_suppressed"])
            probe.validate_report(report)

    def test_cleanup_gate_failure_is_a_failure(self):
        class FakeMX:
            def get_active_memory(self):
                return 10

            def get_cache_memory(self):
                return 3

            def get_peak_memory(self):
                return 10

            def clear_cache(self):
                raise RuntimeError("purge unavailable")

        receipt = probe._release_runtime(FakeMX(), {"model": object()}, {"active": 0}, 0)
        self.assertFalse(receipt["passed"])
        self.assertFalse(receipt["allocator_cache_zero"])
        self.assertIsNotNone(receipt["allocator_purge_error"])

    def test_event_file_checksum_linkage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            writer = probe.JsonlEventWriter(path)
            for index in range(750):
                writer("block", {"block_index": index % 50, "transition_index": index // 50})
            summary = writer.summary()
            report = {"event_file_path": summary["path"], "event_file_record_count": summary["record_count"], "event_file_sha256": summary["sha256"]}
            probe.validate_event_file_linkage(report, path)
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "stale"):
                probe.validate_event_file_linkage(report, path)

    def test_host_contention_declaration_does_not_prove_eligibility(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            args = type("Args", (), {"prompt": probe.LOCKED_PROMPT, "operator_declared_uncontended": True})()
            report = probe._base_report(args, paths)
            self.assertTrue(report["host_contention"]["operator_declared_uncontended"])
            self.assertFalse(report["host_contention"]["process_snapshot_captured"])
            self.assertFalse(report["host_contention"]["canonical_timing_eligible"])

    def test_import_and_help_are_mlxfree(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        code = "import runpy, sys; runpy.run_path(sys.argv[1], run_name='v05d_import'); print('mlx.core' in sys.modules)"
        imported = subprocess.run([sys.executable, "-c", code, str(ROOT / "scripts" / "probe_v05d_derived_full_schedule.py")], cwd="/tmp", env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(imported.returncode, 0, imported.stderr)
        self.assertEqual(imported.stdout.strip(), "False")
        help_result = subprocess.run([sys.executable, str(ROOT / "scripts" / "probe_v05d_derived_full_schedule.py"), "--help"], cwd="/tmp", env=environment, capture_output=True, text=True, check=False)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("run-derived-full-schedule", help_result.stdout)

    def test_generation_exclusions_distinguish_implemented_mux_from_current_execution(self):
        self.assertEqual(probe.GENERATION_EXCLUSIONS, probe.EXPECTED_GENERATION_EXCLUSIONS)
        self.assertFalse(probe.GENERATION_EXCLUSIONS["decoder_phase"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["video_decode"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["png_output"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["audio_decode"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["wav_output"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["ffmpeg_invoked"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["ffprobe_invoked"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["mp4_muxing"])
        self.assertTrue(probe.GENERATION_EXCLUSIONS["resident_comparison_path"])

    def test_valid_derived_artifacts_permit_video_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(gate)
            self.assertEqual(events, ["video", "audio"])
            self.assertEqual(result["video_decoder"]["status"], "completed")
            self.assertEqual(result["audio_decoder"]["status"], "completed")
            self.assertEqual(result["decoder_phase"]["implemented_scope"], "video_then_audio")
            self.assertEqual(result["decoder_phase"]["worker_launches"]["video"], 1)
            self.assertEqual(result["decoder_phase"]["worker_launches"]["audio"], 1)
            self.assertTrue(result["decoder_phase_order"]["valid"])

    def _assert_invalid_derived_gate_suppresses_decoders(self, mutate, expected_gate=None):
        with tempfile.TemporaryDirectory() as directory:
            gate, receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(Path(directory))
            mutate(artifact, artifact_path, metadata_path)
            with self.assertRaises(probe.DecoderGateFailure) as context:
                probe.validate_derived_decoder_gate(
                    receipt,
                    artifact_path,
                    metadata_path,
                    derived_worker_attempts=1,
                    expected_attempt_identifier=artifact["attempt_identifier"],
                    expected_checkpoint_identity=artifact["checkpoint_identity"],
                )
            failed_gate = context.exception.gate_receipt
            result, events = run_decoder_orchestrator(failed_gate)
            self.assertEqual(events, [])
            self.assertEqual(result["video_decoder"]["status"], "suppressed")
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertIsNotNone(result["decoder_failure"])
            if expected_gate is not None:
                self.assertEqual(result["decoder_failure"]["failed_gate"], expected_gate)

    @staticmethod
    def _rewrite_final_metadata(artifact, metadata_path):
        artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
        probe._write_json(metadata_path, artifact)

    def test_invalid_npz_checksum_suppresses_video(self):
        self._assert_invalid_derived_gate_suppresses_decoders(
            lambda _artifact, artifact_path, _metadata_path: artifact_path.write_bytes(artifact_path.read_bytes() + b"corrupt"),
            "final-latent-npz-sha256",
        )

    def test_invalid_metadata_linkage_suppresses_video(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["attempt_identifier"] = "wrong-attempt"
            probe._write_json(metadata_path, artifact)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "final-native-latent-metadata-linkage")

    def test_wrong_video_shape_suppresses_video(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["native_video"]["shape"] = [1, 24, 9, 8, 7]
            self._rewrite_final_metadata(artifact, metadata_path)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "video-latent-shape-and-logical-dtype")

    def test_wrong_audio_shape_suppresses_both_decoders(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["native_audio"]["shape"] = [2, 32, 49]
            self._rewrite_final_metadata(artifact, metadata_path)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "audio-latent-shape-and-logical-dtype")

    def test_wrong_logical_dtype_suppresses_decoders(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["native_video"]["dtype"] = "float32"
            self._rewrite_final_metadata(artifact, metadata_path)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "video-latent-shape-and-logical-dtype")

    def test_failed_transformer_release_gate_suppresses_decoders(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["transformer_release_receipt"]["passed"] = False
            self._rewrite_final_metadata(artifact, metadata_path)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "transformer-release-receipt")

    def test_nonzero_allocator_cache_suppresses_decoders(self):
        def mutate(artifact, _artifact_path, metadata_path):
            artifact["final_allocator_cache"] = 1
            artifact["final_allocator_cache_zero"] = False
            artifact["transformer_release_receipt"]["allocator_cache_zero"] = False
            self._rewrite_final_metadata(artifact, metadata_path)

        self._assert_invalid_derived_gate_suppresses_decoders(mutate, "final-allocator-cache-zero")

    def test_video_launches_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(gate)
            self.assertEqual(events.count("video"), 1)
            self.assertEqual(result["video_decoder"]["worker_launch_count"], 1)

    def test_audio_cannot_launch_before_video_termination(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_exit_observed=False,
                    worker_termination_confirmed=False,
                    worker_receipt_valid=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertEqual(result["decoder_failure"]["failed_gate"], "video-worker-termination")

    def test_audio_cannot_launch_before_video_release_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_receipt_valid=False,
                    release_gate_passed=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertEqual(result["decoder_failure"]["failed_gate"], "video-release-gate")

    def test_video_failure_suppresses_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_exit_code=7,
                    worker_receipt_valid=False,
                    release_gate_passed=False,
                    allocator_cache_zero=False,
                    published_artifact_valid=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["video_decoder"]["status"], "failed")
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")

    def test_video_termination_uncertainty_suppresses_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_termination_confirmed=False,
                    worker_receipt_valid=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertEqual(result["decoder_failure"]["failed_gate"], "video-worker-termination")

    def test_video_release_failure_suppresses_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_receipt_valid=False,
                    release_gate_passed=False,
                    allocator_cache_zero=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertEqual(result["decoder_failure"]["failed_gate"], "video-release-gate")

    def test_successful_exit_without_receipt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt={
                    "worker_identity": "video",
                    "worker_started": True,
                    "worker_pid": 123,
                    "worker_exit_observed": True,
                    "worker_exit_code": 0,
                    "worker_termination_confirmed": True,
                    "worker_receipt_valid": False,
                },
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")
            self.assertIn("missing fields", result["decoder_failure"]["primary_error"]["message"])

    def test_receipt_without_confirmed_termination_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_termination_confirmed=False,
                    worker_receipt_valid=False,
                ),
            )
            self.assertEqual(events, ["video"])
            self.assertEqual(result["decoder_failure"]["failed_gate"], "video-worker-termination")

    def test_duplicate_worker_launch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            calls = []

            def launch(identity):
                calls.append(identity)
                return probe.decoder_worker_receipt(identity)

            orchestrator = probe.DecoderPhaseOrchestrator(derived_gate=gate, worker_launcher=launch)
            self.assertTrue(orchestrator._launch("video"))
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                orchestrator._launch("video")
            self.assertEqual(calls, ["video"])

    def test_no_retry_or_replacement_path_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, _events = run_decoder_orchestrator(
                gate,
                video_receipt=probe.decoder_worker_receipt(
                    "video",
                    worker_exit_code=1,
                    worker_receipt_valid=False,
                    release_gate_passed=False,
                    allocator_cache_zero=False,
                    published_artifact_valid=False,
                ),
            )
            self.assertFalse(result["decoder_phase"]["retry_allowed"])
            self.assertFalse(result["decoder_phase"]["replacement_worker_allowed"])
            self.assertTrue(result["decoder_failure"]["retry_suppressed"])
            self.assertTrue(result["decoder_failure"]["replacement_worker_suppressed"])

    def test_decoder_artifact_seam_validates_key_checksum_linkage_identity_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, metadata, arrays = decoder_artifact_fixture(Path(directory))
            validated = probe.validate_decoder_artifact(
                spec.artifact_path,
                spec.metadata_path,
                expected_keys=spec.expected_keys,
                array_specs=spec.array_specs,
                attempt_identifier=spec.attempt_identifier,
                checkpoint_identity=spec.checkpoint_identity,
                worker_identity=spec.worker_identity,
                artifact_identity=metadata["artifact_identity"],
                arrays=arrays,
            )
            self.assertTrue(validated["passed"])
            spec.artifact_path.write_bytes(spec.artifact_path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                probe.validate_decoder_artifact(
                    spec.artifact_path,
                    spec.metadata_path,
                    expected_keys=spec.expected_keys,
                    array_specs=spec.array_specs,
                    attempt_identifier=spec.attempt_identifier,
                    checkpoint_identity=spec.checkpoint_identity,
                    worker_identity=spec.worker_identity,
                )

    def test_decoder_phase_order_rejects_early_audio_start(self):
        with self.assertRaisesRegex(ValueError, "audio worker starts"):
            probe.validate_decoder_phase_order(["derived-finalization", "audio-worker-start"])

    def test_mux_contract_is_mlxfree_and_decoder_workers_remain_media_free(self):
        self.assertFalse(probe.GENERATION_EXCLUSIONS["ffmpeg_invoked"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["ffprobe_invoked"])
        self.assertNotIn("import mlx", inspect.getsource(probe.validate_decoder_artifact))
        source = (ROOT / "scripts" / "probe_v05d_derived_full_schedule.py").read_text()
        class_source = source[source.index("class DecoderPhaseOrchestrator") : source.index("def validate_locked_prompt")]
        self.assertNotIn("ffmpeg", class_source)
        self.assertNotIn("ffprobe", class_source)
        self.assertNotIn("shell=True", source)

    def test_video_input_metadata_and_fingerprint_fail_before_vae_loader(self):
        mutations = {
            "metadata": lambda artifact, _path: artifact["native_video"].update({"dtype": "float32"}),
            "fingerprint": lambda artifact, _path: artifact["native_video"].update({"fingerprint": "wrong"}),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root)
                mutate(artifact, metadata_path)
                artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
                probe._write_json(metadata_path, artifact)
                loader_calls = []
                with self.assertRaises(ValueError):
                    probe.execute_video_decode_once(
                        artifact_path=artifact_path,
                        metadata_path=metadata_path,
                        expected_attempt_identifier=artifact["attempt_identifier"],
                        expected_checkpoint_identity=artifact["checkpoint_identity"],
                        video_root=root / "video_vae",
                        frames_partial=root / "frames.partial",
                        frames_final=root / "frames",
                        manifest_path=root / "video-frame-manifest.json",
                        mx=FakeMLX(),
                        load_video_config=lambda _path: loader_calls.append("config"),
                        load_video_vae=lambda _path: loader_calls.append("vae"),
                    )
                self.assertEqual(loader_calls, [], label)

    def test_video_input_gate_checks_attempt_checkpoint_worker_termination_release_and_cache(self):
        checks = {
            "attempt": lambda artifact: artifact.update({"attempt_identifier": "other"}),
            "checkpoint": lambda artifact: artifact.update({"checkpoint_identity": {"checkpoint": "other"}}),
            "worker": lambda artifact: artifact.update({"worker_identity": "other"}),
            "termination": lambda artifact: artifact["worker_exit_receipt"].update({"worker_termination_confirmed": False}),
            "release": lambda artifact: artifact["transformer_release_receipt"].update({"passed": False}),
            "cache": lambda artifact: artifact.update({"final_allocator_cache": 1, "final_allocator_cache_zero": False}),
        }
        for label, mutate in checks.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root)
                mutate(artifact)
                artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
                probe._write_json(metadata_path, artifact)
                with self.assertRaises(ValueError):
                    probe.validate_final_video_input_artifact(
                        artifact_path,
                        metadata_path,
                        expected_attempt_identifier="attempt-1",
                        expected_checkpoint_identity={"checkpoint": "derived-checkpoint"},
                    )

    def test_logical_bfloat16_restoration_precedes_normalization(self):
        stored = np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32)
        fake_mx = FakeMLX()
        expected = probe.array_fingerprint(stored, logical_dtype="bfloat16")
        materialization_events = []

        def materialize(value):
            materialization_events.append(value.dtype)
            value.materialized = True

        restored, fingerprint = probe.restore_video_latent_logical_bfloat16(
            stored,
            fake_mx,
            expected_fingerprint=expected,
            materialize=materialize,
        )
        self.assertEqual(restored.dtype, "bfloat16")
        self.assertEqual(fingerprint, expected)
        self.assertEqual(materialization_events, ["bfloat16"])
        self.assertEqual(fake_mx.events[:2], [("array", "float32"), ("astype", "float32", "bfloat16")])

    def test_decoder_input_is_materialized_float32_after_normalization(self):
        fake_mx = FakeMLX()
        latent = FakeMLXArray(np.zeros(probe.VIDEO_NATIVE_SHAPE, dtype=np.float32), "bfloat16", fake_mx)
        latent.materialized = True
        materialized = []
        decoder_input = probe.normalize_video_latent_for_decode(
            latent,
            FakeVideoConfig(),
            fake_mx,
            materialize=lambda value: (materialized.append(value.dtype), setattr(value, "materialized", True)),
        )
        self.assertEqual(decoder_input.dtype, "float32")
        self.assertEqual(materialized, ["float32"])

    def test_decode_loads_video_vae_once_decodes_once_and_publishes_video_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root / "gate")
            fake_mx = FakeMLX()
            calls = []

            class Decoder:
                def __init__(self):
                    self.decode_calls = 0

                def decode(self, value):
                    self.decode_calls += 1
                    self.input_dtype = value.dtype
                    return FakeMLXArray(np.zeros(probe.VIDEO_RAW_SHAPE, dtype=np.float32), "float32", fake_mx)

            decoder = Decoder()

            def save_frames(directory_path, frames):
                write_rgb_frame_set(Path(directory_path), count=len(frames))

            result = probe.execute_video_decode_once(
                artifact_path=artifact_path,
                metadata_path=metadata_path,
                expected_attempt_identifier=artifact["attempt_identifier"],
                expected_checkpoint_identity=artifact["checkpoint_identity"],
                video_root=root / "video_vae",
                frames_partial=root / "frames.partial",
                frames_final=root / "frames",
                manifest_path=root / "video-frame-manifest.json",
                mx=fake_mx,
                load_video_config=lambda _path: (calls.append("config"), FakeVideoConfig())[1],
                load_video_vae=lambda _path: (calls.append("vae"), decoder)[1],
                save_frames=save_frames,
                materialize=lambda value: setattr(value, "materialized", True),
                memory_snapshot=lambda: {"active": 1, "allocator_cache": 0, "peak": 2},
            )
            self.assertEqual(calls, ["config", "vae"])
            self.assertEqual(decoder.decode_calls, 1)
            self.assertEqual(decoder.input_dtype, "float32")
            self.assertEqual(result["raw_shape_dtype"]["shape"], list(probe.VIDEO_RAW_SHAPE))
            self.assertEqual(result["rgb_shape_dtype"]["shape"], list(probe.VIDEO_RGB_SHAPE))
            self.assertEqual(result["frame_manifest"]["publication_state"], "published")
            self.assertTrue((root / "frames").is_dir())
            self.assertFalse((root / "frames.partial").exists())
            probe.validate_video_frame_manifest(
                root / "video-frame-manifest.json",
                root / "frames",
                expected_attempt_identifier=artifact["attempt_identifier"],
            )

    def test_raw_shape_dtype_and_finite_output_gates_are_strict(self):
        fake_mx = FakeMLX()
        accepted = FakeMLXArray(np.zeros(probe.VIDEO_RAW_SHAPE, dtype=np.float32), "float32", fake_mx)
        raw_np, receipt = probe.materialize_and_validate_video_raw_output(accepted, fake_mx)
        self.assertEqual(tuple(raw_np.shape), probe.VIDEO_RAW_SHAPE)
        self.assertEqual(receipt["dtype"], "float32")
        cases = [
            (np.zeros((1, 3, 29, 128, 128), dtype=np.float32), "shape"),
            (np.full(probe.VIDEO_RAW_SHAPE, np.nan, dtype=np.float32), "non-finite"),
            (np.zeros(probe.VIDEO_RAW_SHAPE, dtype=np.float32), "float32"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                dtype = "bfloat16" if label == "float32" else "float32"
                with self.assertRaises(ValueError):
                    probe.materialize_and_validate_video_raw_output(
                        FakeMLXArray(data, dtype, fake_mx),
                        fake_mx,
                    )

    def test_exact_rgb_shape_and_uint8_are_accepted_and_wrong_output_is_rejected(self):
        frames, receipt = probe.convert_and_validate_video_rgb(np.zeros(probe.VIDEO_RAW_SHAPE, dtype=np.float32))
        self.assertEqual(tuple(frames.shape), probe.VIDEO_RGB_SHAPE)
        self.assertEqual(frames.dtype, np.uint8)
        self.assertEqual(receipt["dtype"], "uint8")
        with self.assertRaisesRegex(ValueError, "shape"):
            probe.validate_video_rgb_output(np.zeros((29, 128, 128, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "dtype"):
            probe.validate_video_rgb_output(np.zeros(probe.VIDEO_RGB_SHAPE, dtype=np.int16))

    def test_exact_contiguous_png_set_rejects_missing_duplicate_and_extra_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_rgb_frame_set(root)
            self.assertEqual(probe._inspect_video_frame_set(root)["frame_count"], 30)
            (root / "frame_00007.png").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous"):
                probe._inspect_video_frame_set(root)
            write_rgb_frame_set(root)
            (root / "frame_00030.png").write_bytes((root / "frame_00000.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "contiguous"):
                probe._inspect_video_frame_set(root)
            (root / "frame_00030.png").unlink()
            write_rgb_frame_set(root)
            manifest = probe.build_video_frame_manifest(root, attempt_identifier="attempt-1")
            manifest["frames"][-1] = dict(manifest["frames"][0])
            manifest["manifest_sha256"] = probe.stable_video_frame_manifest_sha256(manifest)
            manifest_path = root.parent / "manifest.json"
            probe._write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "checksum linkage"):
                probe.validate_video_frame_manifest(manifest_path, root, expected_attempt_identifier="attempt-1")

    def test_wrong_png_mode_dimensions_and_empty_file_are_rejected(self):
        cases = [("L", (128, 128), None), ("RGB", (127, 128), None), ("RGB", (128, 128), 3)]
        for mode, size, empty_index in cases:
            with self.subTest(mode=mode, size=size, empty_index=empty_index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                write_rgb_frame_set(root, mode=mode, size=size, empty_index=empty_index)
                with self.assertRaises(ValueError):
                    probe._inspect_video_frame_set(root)

    def test_final_frames_directory_is_absent_during_prepublication_validation_and_publish_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            write_rgb_frame_set(partial)
            observed = []

            def rename(source, destination):
                observed.append((Path(source).resolve() == partial.resolve(), not destination.exists()))
                source.rename(destination)

            published = probe.publish_video_frames_atomically(
                partial,
                final,
                manifest,
                attempt_identifier="attempt-1",
                rename=rename,
            )
            self.assertEqual(observed, [(True, True)])
            self.assertFalse(partial.exists())
            self.assertTrue(final.is_dir())
            self.assertEqual(published["frame_count"], 30)

    def test_manifest_checksum_linkage_is_enforced_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            write_rgb_frame_set(partial)
            probe.publish_video_frames_atomically(partial, final, manifest, attempt_identifier="attempt-1")
            from PIL import Image

            Image.new("RGB", (128, 128), 255).save(final / "frame_00000.png")
            with self.assertRaisesRegex(ValueError, "checksum linkage"):
                probe.validate_video_frame_manifest(manifest, final, expected_attempt_identifier="attempt-1")

    def test_publication_failure_preserves_only_staged_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            write_rgb_frame_set(partial)

            def fail_rename(_source, _destination):
                raise OSError("synthetic rename failure")

            with self.assertRaises(OSError):
                probe.publish_video_frames_atomically(
                    partial,
                    final,
                    manifest,
                    attempt_identifier="attempt-1",
                    rename=fail_rename,
                )
            self.assertTrue(partial.is_dir())
            self.assertFalse(final.exists())

    def test_existing_final_frames_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "frames.partial"
            final = root / "frames"
            write_rgb_frame_set(partial)
            final.mkdir()
            with self.assertRaisesRegex(FileExistsError, "final video frames"):
                probe.publish_video_frames_atomically(
                    partial,
                    final,
                    root / "video-frame-manifest.json",
                    attempt_identifier="attempt-1",
                )

    def test_release_failure_prevents_video_success_and_preserves_primary_gate(self):
        class PurgeFailureMX(FakeMLX):
            def __init__(self):
                super().__init__()
                self.cache = 7

            def clear_cache(self):
                raise RuntimeError("purge failed")

        fake_mx = PurgeFailureMX()
        release = probe.release_video_decoder(
            fake_mx,
            {"decoder": object(), "latent": object()},
            {"active": 0},
            0,
        )
        self.assertFalse(release["passed"])
        self.assertFalse(release["allocator_cache_zero"])
        self.assertIsNotNone(release["allocator_purge_error"])

    def _execute_audio_fixture(self, root: Path, *, save_wav=None, materialize=None, raw=None):
        _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root / "gate")
        fake_mx = FakeMLX()
        calls = []

        class Decoder:
            def __init__(self):
                self.decode_calls = 0
                self.input_dtype = None

            def decode(self, value):
                self.decode_calls += 1
                self.input_dtype = value.dtype
                value = raw if raw is not None else np.zeros(probe.AUDIO_RAW_SHAPE, dtype=np.float32)
                return FakeMLXArray(value, "float32", fake_mx)

        decoder = Decoder()

        def save(path, waveform, sample_rate):
            calls.append(("save", Path(path), tuple(waveform.shape), sample_rate))
            write_stereo_wav(Path(path), sample_rate=sample_rate, sample_count=waveform.shape[1])

        result = probe.execute_audio_decode_once(
            artifact_path=artifact_path,
            metadata_path=metadata_path,
            expected_attempt_identifier=artifact["attempt_identifier"],
            expected_checkpoint_identity=artifact["checkpoint_identity"],
            audio_root=root / "audio_vae",
            wav_partial=root / "audio.partial.wav",
            wav_final=root / "audio.wav",
            manifest_path=root / "audio-manifest.json",
            mx=fake_mx,
            load_audio_config=lambda _path: (calls.append("config"), FakeAudioConfig())[1],
            load_audio_vae=lambda _path: (calls.append("vae"), decoder)[1],
            save_wav=save_wav or save,
            materialize=materialize or (lambda value: setattr(value, "materialized", True)),
            memory_snapshot=lambda: {"active": 0, "allocator_cache": 0, "peak": 1},
        )
        return result, artifact, fake_mx, decoder, calls, artifact_path, metadata_path

    def test_audio_launches_after_every_video_gate_and_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            result, events = run_decoder_orchestrator(gate)
            self.assertEqual(events, ["video", "audio"])
            self.assertEqual(result["decoder_phase"]["worker_launches"], {"video": 1, "audio": 1})
            self.assertEqual(result["decoder_phase_order"]["observed"], list(probe.DECODER_PHASE_ORDER))
            self.assertFalse(result["decoder_phase"]["retry_allowed"])
            self.assertFalse(result["decoder_phase"]["replacement_worker_allowed"])

    def test_failed_video_publication_suppresses_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            gate, _receipt, _artifact_path, _metadata_path, _artifact, _arrays = derived_gate_fixture(Path(directory))
            events = []

            def launch(identity):
                events.append(identity)
                return probe.decoder_worker_receipt(identity)

            def reject_video():
                raise ValueError("synthetic frame manifest failure")

            result = probe.DecoderPhaseOrchestrator(
                derived_gate=gate,
                worker_launcher=launch,
                implemented_phase_scope={"video": True, "audio": True},
                artifact_validators={"video": reject_video},
            ).run()
            self.assertEqual(events, ["video"])
            self.assertEqual(result["video_decoder"]["status"], "failed")
            self.assertEqual(result["audio_decoder"]["status"], "suppressed")

    def test_invalid_audio_latent_metadata_prevents_vae_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root / "gate")
            artifact["native_audio"]["shape"] = [2, 32, 49]
            artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
            probe._write_json(metadata_path, artifact)
            calls = []
            with self.assertRaises(ValueError):
                probe.execute_audio_decode_once(
                    artifact_path=artifact_path,
                    metadata_path=metadata_path,
                    expected_attempt_identifier=artifact["attempt_identifier"],
                    expected_checkpoint_identity=artifact["checkpoint_identity"],
                    audio_root=root / "audio_vae",
                    wav_partial=root / "audio.partial.wav",
                    wav_final=root / "audio.wav",
                    manifest_path=root / "audio-manifest.json",
                    mx=FakeMLX(),
                    load_audio_config=lambda _path: calls.append("config"),
                    load_audio_vae=lambda _path: calls.append("vae"),
                )
            self.assertEqual(calls, [])

    def test_invalid_audio_fingerprint_prevents_vae_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _gate, _receipt, artifact_path, metadata_path, artifact, _arrays = derived_gate_fixture(root / "gate")
            artifact["native_audio"]["fingerprint"] = "0" * 64
            artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
            probe._write_json(metadata_path, artifact)
            calls = []
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                probe.execute_audio_decode_once(
                    artifact_path=artifact_path,
                    metadata_path=metadata_path,
                    expected_attempt_identifier=artifact["attempt_identifier"],
                    expected_checkpoint_identity=artifact["checkpoint_identity"],
                    audio_root=root / "audio_vae",
                    wav_partial=root / "audio.partial.wav",
                    wav_final=root / "audio.wav",
                    manifest_path=root / "audio-manifest.json",
                    mx=FakeMLX(),
                    load_audio_config=lambda _path: calls.append("config"),
                    load_audio_vae=lambda _path: calls.append("vae"),
                )
            self.assertEqual(calls, [])

    def test_audio_logical_bfloat16_restoration_precedes_normalization(self):
        stored = np.zeros(probe.AUDIO_NATIVE_SHAPE, dtype=np.float32)
        fake_mx = FakeMLX()
        expected = probe.array_fingerprint(stored, logical_dtype="bfloat16")
        events = []

        def materialize(value):
            events.append(value.dtype)
            value.materialized = True

        latent, fingerprint = probe.restore_audio_latent_logical_bfloat16(
            stored,
            fake_mx,
            expected_fingerprint=expected,
            materialize=materialize,
        )
        decoder_input = probe.normalize_audio_latent_for_decode(
            latent,
            FakeAudioConfig(),
            fake_mx,
            materialize=materialize,
        )
        self.assertEqual(latent.dtype, "bfloat16")
        self.assertEqual(fingerprint, expected)
        self.assertEqual(decoder_input.dtype, "float32")
        self.assertEqual(events, ["bfloat16", "float32"])

    def test_audio_decoder_input_is_materialized_float32_and_decode_is_exactly_once(self):
        with tempfile.TemporaryDirectory() as directory:
            result, _artifact, _fake_mx, decoder, calls, _artifact_path, _metadata_path = self._execute_audio_fixture(Path(directory))
            self.assertEqual(calls[:2], ["config", "vae"])
            self.assertEqual(decoder.decode_calls, 1)
            self.assertEqual(decoder.input_dtype, "float32")
            self.assertEqual(result["raw_shape_dtype"]["shape"], list(probe.AUDIO_RAW_SHAPE))
            self.assertEqual(result["waveform_shape_dtype"]["shape"], list(probe.AUDIO_WAVEFORM_SHAPE))
            self.assertEqual(result["wav_manifest"]["publication_state"], "published")

    def test_audio_raw_shape_dtype_and_finite_output_gates_are_strict(self):
        fake_mx = FakeMLX()
        accepted = FakeMLXArray(np.zeros(probe.AUDIO_RAW_SHAPE, dtype=np.float32), "float32", fake_mx)
        raw_np, receipt = probe.materialize_and_validate_audio_raw_output(accepted, fake_mx)
        self.assertEqual(tuple(raw_np.shape), probe.AUDIO_RAW_SHAPE)
        self.assertEqual(receipt["dtype"], "float32")
        cases = [
            (np.zeros((2, 1, 39999), dtype=np.float32), "shape"),
            (np.full(probe.AUDIO_RAW_SHAPE, np.nan, dtype=np.float32), "non-finite"),
            (np.zeros(probe.AUDIO_RAW_SHAPE, dtype=np.float32), "dtype"),
        ]
        for data, label in cases:
            with self.subTest(label=label):
                dtype = "bfloat16" if label == "dtype" else "float32"
                with self.assertRaises(ValueError):
                    probe.materialize_and_validate_audio_raw_output(FakeMLXArray(data, dtype, fake_mx), fake_mx)

    def test_audio_waveform_shape_and_finite_values_are_strict(self):
        waveform, receipt = probe.convert_and_validate_audio_waveform(np.zeros(probe.AUDIO_RAW_SHAPE, dtype=np.float32))
        self.assertEqual(tuple(waveform.shape), probe.AUDIO_WAVEFORM_SHAPE)
        self.assertEqual(receipt["dtype"], "float32")
        with self.assertRaisesRegex(ValueError, "shape"):
            probe.convert_and_validate_audio_waveform(np.zeros((2, 1, 39999), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            broken = np.zeros(probe.AUDIO_RAW_SHAPE, dtype=np.float32)
            broken[0, 0, 0] = np.nan
            probe.convert_and_validate_audio_waveform(broken)

    def test_wav_metadata_requires_stereo_rate_count_width_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                ({"channels": 1}, "channel"),
                ({"sample_rate": 44100}, "rate"),
                ({"sample_count": 39999}, "count"),
                ({"sample_width": 1}, "width"),
            ]
            for overrides, label in cases:
                with self.subTest(label=label):
                    path = root / f"{label}.wav"
                    write_stereo_wav(path, **overrides)
                    metadata = probe._wav_metadata(path)
                    with self.assertRaises(ValueError):
                        probe.validate_audio_wav_metadata(metadata)
            metadata = {
                "channels": 2,
                "sample_rate": 32000,
                "sample_count": 40000,
                "sample_width_bytes": 2,
                "duration_seconds": 1.0,
                "size_bytes": 1,
                "wav_sha256": "a" * 64,
            }
            with self.assertRaisesRegex(ValueError, "duration"):
                probe.validate_audio_wav_metadata(metadata)

    def test_truncated_or_empty_wav_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truncated = root / "truncated.wav"
            write_stereo_wav(truncated, truncate=True)
            with self.assertRaisesRegex(ValueError, "truncated"):
                probe._wav_metadata(truncated)
            empty = root / "empty.wav"
            empty.write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "empty"):
                probe._wav_metadata(empty)

    def test_wav_manifest_checksum_linkage_and_atomic_publication_are_strict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "audio.partial.wav"
            final = root / "audio.wav"
            manifest = root / "audio-manifest.json"
            write_stereo_wav(partial)
            observed = []

            def rename(source, destination):
                observed.append((Path(source).resolve() == partial.resolve(), not destination.exists()))
                source.rename(destination)

            published = probe.publish_audio_wav_atomically(
                partial,
                final,
                manifest,
                attempt_identifier="attempt-1",
                rename=rename,
            )
            self.assertEqual(observed, [(True, True)])
            self.assertFalse(partial.exists())
            self.assertTrue(final.is_file())
            self.assertEqual(published["sample_count"], probe.AUDIO_SAMPLE_COUNT)
            final_bytes = bytearray(final.read_bytes())
            final_bytes[-1] ^= 1
            final.write_bytes(final_bytes)
            with self.assertRaisesRegex(ValueError, "linkage"):
                probe.validate_audio_wav_manifest(manifest, final, expected_attempt_identifier="attempt-1")
            final_bytes[-1] ^= 1
            final.write_bytes(final_bytes)
            manifest_data = probe._read_json_object(manifest, "manifest")
            manifest_data["manifest_sha256"] = "0" * 64
            probe._write_json(manifest, manifest_data)
            with self.assertRaisesRegex(ValueError, "checksum linkage"):
                probe.validate_audio_wav_manifest(manifest, final, expected_attempt_identifier="attempt-1")

    def test_existing_final_audio_is_refused_and_failure_preserves_staged_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "audio.partial.wav"
            final = root / "audio.wav"
            manifest = root / "audio-manifest.json"
            write_stereo_wav(partial)
            final.write_bytes(b"existing")
            with self.assertRaisesRegex(FileExistsError, "existing final audio"):
                probe.publish_audio_wav_atomically(partial, final, manifest, attempt_identifier="attempt-1")
            final.unlink()

            def fail_rename(_source, _destination):
                raise OSError("synthetic audio rename failure")

            with self.assertRaises(OSError):
                probe.publish_audio_wav_atomically(
                    partial,
                    final,
                    manifest,
                    attempt_identifier="attempt-1",
                    rename=fail_rename,
                )
            self.assertTrue(partial.is_file())
            self.assertFalse(final.exists())
            self.assertTrue(manifest.is_file())

    def test_audio_release_failure_prevents_standalone_success(self):
        class PurgeFailureMX(FakeMLX):
            def __init__(self):
                super().__init__()
                self.cache = 7

            def clear_cache(self):
                raise RuntimeError("audio purge failed")

        release = probe.release_audio_decoder(
            PurgeFailureMX(),
            {"decoder": object(), "waveform": object()},
            {"active": 0},
            0,
        )
        self.assertFalse(release["passed"])
        self.assertFalse(release["allocator_cache_zero"])
        self.assertIsNotNone(release["allocator_purge_error"])

    def test_audio_worker_receipt_requires_exactly_once_counts(self):
        receipt = probe.decoder_worker_receipt("audio")
        receipt.update({"wav_manifest_valid": True, "decode_count": 1, "audio_vae_load_count": 1})
        self.assertEqual(receipt["worker_exit_code"], 0)
        self.assertEqual(receipt["worker_termination_confirmed"], True)
        self.assertNotEqual(receipt["worker_exit_code"], 1)

    def test_audio_worker_has_no_ffmpeg_ffprobe_or_mp4_path(self):
        source = inspect.getsource(probe._video_worker_main)
        for forbidden in ("audio_vae", "load_audio", "ffmpeg", "ffprobe", "save_mp4", "mp4"):
            self.assertNotIn(forbidden, source)
        audio_source = inspect.getsource(probe._audio_worker_main).lower()
        for forbidden in ("text_encoder", "load_dit", "MiniMaxH3DiT", "scheduler", "adaln", "video_vae", "ffmpeg", "ffprobe", "mp4"):
            self.assertNotIn(forbidden, audio_source)
        self.assertEqual(probe.DecoderPhaseOrchestrator(derived_gate={}, worker_launcher=lambda _identity: {}).implemented_phase_scope, {"video": True, "audio": True})

    def test_attempt_namespace_contains_slice_3c_publication_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_attempt_namespace(Path(directory) / "attempt")
            self.assertTrue(paths["frames_partial"].endswith("frames.partial"))
            self.assertTrue(paths["frames"].endswith("frames"))
            self.assertTrue(paths["video_frame_manifest"].endswith("video-frame-manifest.json"))
            self.assertTrue(paths["video_worker_receipt"].endswith("video-worker-receipt.json"))
            self.assertTrue(paths["video_worker_log"].endswith("video-worker.log"))
            self.assertTrue(paths["audio_partial"].endswith("audio.partial.wav"))
            self.assertTrue(paths["audio_wav"].endswith("audio.wav"))
            self.assertTrue(paths["audio_manifest"].endswith("audio-manifest.json"))
            self.assertTrue(paths["audio_worker_receipt"].endswith("audio-worker-receipt.json"))
            self.assertTrue(paths["audio_worker_log"].endswith("audio-worker.log"))

    def test_report_validation_accepts_only_exact_standalone_media_exclusion_map(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate, _receipt, _artifact_path, _metadata_path, artifact, _arrays = derived_gate_fixture(root / "gate")
            output_paths = probe.ensure_attempt_namespace(root / "output")
            args = type("Args", (), {"prompt": probe.LOCKED_PROMPT, "operator_declared_uncontended": False})()
            report = probe._base_report(args, output_paths)
            report["schedule_contract"] = artifact["schedule_contract"]
            report["streamed_adaln_lifecycle"] = artifact["streamed_adaln_lifecycle"]
            report["final_artifact"] = artifact
            report["conditioning_worker"] = {
                **probe.decoder_worker_receipt("conditioning"),
                "status": "success",
                "conditioning_release": {"passed": True},
            }
            report["derived_worker"] = {
                **probe.decoder_worker_receipt("derived"),
                "status": "success",
                "transformer_release": artifact["transformer_release_receipt"],
            }
            def worker(identity):
                receipt = probe.decoder_worker_receipt(identity)
                if identity == "audio":
                    receipt.update({"wav_manifest_valid": True, "decode_count": 1, "audio_vae_load_count": 1})
                return receipt

            decoder_result = probe.DecoderPhaseOrchestrator(
                derived_gate=gate,
                worker_launcher=worker,
            ).run()
            report["decoder_phase"] = decoder_result["decoder_phase"]
            report["video_decoder"] = decoder_result["video_decoder"]
            report["audio_decoder"] = decoder_result["audio_decoder"]
            report["decoder_phase_order"] = decoder_result["decoder_phase_order"]
            report["video_artifacts"] = {
                "publication_state": "published",
                "frame_count": 30,
                "width": 128,
                "height": 128,
                "fps": 24,
                "duration_seconds": 1.25,
                "manifest_sha256": "a" * 64,
            }
            report["audio_artifacts"] = {
                "publication_state": "published",
                "channels": 2,
                "sample_rate": 32000,
                "sample_count": 40000,
                "sample_width_bytes": 2,
                "duration_seconds": 1.25,
                "wav_sha256": "c" * 64,
                "manifest_sha256": "d" * 64,
            }
            report["decoder_memory"] = {"video": {}, "audio": {}}
            report["decoder_timing"] = {"video": {}, "audio": {}}
            report["event_file_record_count"] = 750
            report["total_event_records"] = 750
            report["cache_session_count"] = 15
            report["sidecar_open_event_count"] = 750
            report["sidecar_release_event_count"] = 750
            report["validated_block_pairs"] = 750
            report["event_file_sha256"] = "b" * 64
            report["status"] = "success"
            report["run_state"] = "successful"
            report["failure"] = None
            report["latent_generation_status"] = "completed"
            report["video_status"] = "completed"
            report["audio_status"] = "completed"
            report["standalone_media_status"] = "completed"
            report["mp4_mux_status"] = "completed"
            report["standalone_media"] = {
                "status": "completed",
                "latent_generation_status": "completed",
                "video_status": "completed",
                "audio_status": "completed",
                "standalone_media_status": "completed",
                "mp4_mux_status": "completed",
            }
            report["mp4_mux"] = {
                "status": "completed",
                "invoked": True,
                "output_path": str(output_paths["mp4"]),
                "launch_gate": successful_mux_gate(),
                "retry_suppressed": True,
                "invocation_counts": {"ffmpeg": 1, "ffprobe": 1},
                "ffmpeg": {"invoked": True, "returncode": 0, "argv": ["ffmpeg"]},
                "ffprobe": {"invoked": True, "returncode": 0, "argv": ["ffprobe"]},
            }
            report["mp4_artifact"] = {
                "publication_state": "published",
                "mp4_path": str(output_paths["mp4"]),
                "manifest_path": str(output_paths["mp4_manifest"]),
                "mp4_sha256": "e" * 64,
                "manifest_sha256": "f" * 64,
                "size_bytes": 1,
            }
            report["mux_timing"] = {"total_seconds": 0.01, "executed": True}
            report["mux_failure"] = None
            probe.validate_report(report)
            report["audio_decoder"]["release_gate_passed"] = False
            with self.assertRaisesRegex(ValueError, "release"):
                probe.validate_report(report)
            report["audio_decoder"]["release_gate_passed"] = True
            report["generation_exclusions"] = {**probe.EXPECTED_GENERATION_EXCLUSIONS, "video_decode": True}
            with self.assertRaisesRegex(ValueError, "exclusions"):
                probe.validate_report(report)


class MP4MuxContractTests(unittest.TestCase):
    def _run_success(self, root, *, runner=None, probe_json=None, rename=None):
        paths = mp4_media_fixture(Path(root))
        fake = runner or FakeMuxRunner(ffprobe_json=probe_json)
        result = probe.execute_mp4_mux(
            frames_directory=paths["frames"],
            video_manifest_path=paths["video_frame_manifest"],
            wav_path=paths["audio_wav"],
            audio_manifest_path=paths["audio_manifest"],
            mp4_partial_path=paths["mp4_partial"],
            mp4_final_path=paths["mp4"],
            mp4_manifest_path=paths["mp4_manifest"],
            attempt_identifier=paths["attempt_identifier"],
            launch_gate=successful_mux_gate(),
            subprocess_runner=fake,
            rename=rename,
        )
        return paths, fake, result

    def test_mux_starts_only_after_standalone_media_success(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            report = mux_report_fixture(paths)
            report["standalone_media_status"] = "not_started"
            report["standalone_media"]["status"] = "not_started"
            runner = FakeMuxRunner()
            result = probe.apply_mp4_mux_report(report, paths, subprocess_runner=runner)
            self.assertEqual(result["mp4_mux_status"], "suppressed")
            self.assertEqual(runner.calls, [])
            self.assertFalse(paths["mp4"].exists())

    def test_failed_video_gate_suppresses_mux(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            report = mux_report_fixture(paths)
            report["video_decoder"]["release_gate_passed"] = False
            runner = FakeMuxRunner()
            result = probe.apply_mp4_mux_report(report, paths, subprocess_runner=runner)
            self.assertEqual(result["mp4_mux_status"], "suppressed")
            self.assertEqual(runner.calls, [])

    def test_failed_audio_gate_suppresses_mux(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            report = mux_report_fixture(paths)
            report["audio_decoder"]["release_gate_passed"] = False
            runner = FakeMuxRunner()
            result = probe.apply_mp4_mux_report(report, paths, subprocess_runner=runner)
            self.assertEqual(result["mp4_mux_status"], "suppressed")
            self.assertEqual(runner.calls, [])

    def test_ffmpeg_launches_exactly_once_and_ffprobe_once_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            paths, runner, result = self._run_success(Path(directory))
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg", "ffprobe"])
            self.assertEqual(result["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})
            self.assertEqual(result["ffmpeg"]["argv"], runner.calls[0][0])
            self.assertEqual(result["ffprobe"]["argv"], runner.calls[1][0])

    def test_ffprobe_never_runs_after_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            runner = FakeMuxRunner(ffmpeg_returncode=7, ffmpeg_stderr="encoder failed")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                probe.execute_mp4_mux(
                    frames_directory=paths["frames"],
                    video_manifest_path=paths["video_frame_manifest"],
                    wav_path=paths["audio_wav"],
                    audio_manifest_path=paths["audio_manifest"],
                    mp4_partial_path=paths["mp4_partial"],
                    mp4_final_path=paths["mp4"],
                    mp4_manifest_path=paths["mp4_manifest"],
                    attempt_identifier="attempt-1",
                    launch_gate=successful_mux_gate(),
                    subprocess_runner=runner,
                )
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(context.exception.receipt["ffmpeg"]["stderr"], "encoder failed")
            self.assertEqual(context.exception.receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 0})

    def test_nonzero_ffmpeg_exit_is_failure_and_partial_is_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            runner = FakeMuxRunner(ffmpeg_returncode=2)
            with self.assertRaises(probe.MP4MuxFailure) as context:
                self._run_mux_direct(paths, runner)
            self.assertFalse(paths["mp4_partial"].exists())
            self.assertEqual(context.exception.receipt["partial_cleanup_policy"], "delete_partial_mp4_after_failure")

    def test_zero_exit_missing_or_empty_mp4_is_failure_without_ffprobe(self):
        cases = [("missing", False, False), ("empty", True, True)]
        for label, write_output, empty_output in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                paths = mp4_media_fixture(Path(directory))
                runner = FakeMuxRunner(write_output=write_output, empty_output=empty_output)
                with self.assertRaises(probe.MP4MuxFailure):
                    self._run_mux_direct(paths, runner)
                self.assertEqual(len(runner.calls), 1)
                self.assertFalse(paths["mp4_partial"].exists())

    def test_timeout_fails_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            runner = FakeMuxRunner(timeout_tool="ffmpeg")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                self._run_mux_direct(paths, runner)
            receipt = context.exception.receipt
            self.assertTrue(receipt["ffmpeg"]["timed_out"])
            self.assertEqual(receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 0})
            self.assertTrue(receipt["retry_suppressed"])
            self.assertFalse(paths["mp4_partial"].exists())

    def test_ffmpeg_command_requests_locked_media_properties(self):
        command = probe.build_ffmpeg_command(Path("/tmp/frames"), Path("/tmp/audio.wav"), Path("/tmp/out.mp4"))
        self.assertEqual(command[command.index("-framerate") + 1], "24")
        self.assertEqual(command[command.index("-c:v") + 1], "libx264")
        self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
        self.assertEqual(command[command.index("-c:a") + 1], "aac")
        self.assertEqual(command[command.index("-b:a") + 1], "192k")
        self.assertIn("-shortest", command)
        self.assertTrue(any("frame_%05d.png" in argument for argument in command))
        self.assertIn("-frames:v", command)
        self.assertEqual(command[command.index("-frames:v") + 1], "30")

    def test_injected_subprocess_path_has_no_shell_true(self):
        source = inspect.getsource(probe._default_mux_subprocess_runner)
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", (ROOT / "scripts" / "probe_v05d_derived_full_schedule.py").read_text())

    def test_wrong_video_properties_fail_before_publication(self):
        cases = [
            ({"video": {"width": 127}}, "dimensions"),
            ({"video": {"height": 127}}, "dimensions"),
            ({"video": {"avg_frame_rate": "25/1", "r_frame_rate": "25/1"}}, "frame rate"),
            ({"video": {"codec_name": "vp9"}}, "codec"),
            ({"video": {"pix_fmt": "rgb24"}}, "pixel format"),
        ]
        for overrides, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                paths = mp4_media_fixture(Path(directory))
                runner = FakeMuxRunner(ffprobe_json=valid_ffprobe_json(**overrides))
                with self.assertRaises(probe.MP4MuxFailure):
                    self._run_mux_direct(paths, runner)
                self.assertFalse(paths["mp4"].exists())
                self.assertFalse(paths["mp4_partial"].exists())

    def test_missing_audio_stream_mono_audio_wrong_rate_and_codec_fail(self):
        cases = [
            (valid_ffprobe_json(streams=[valid_ffprobe_json()["streams"][0]]), "audio stream"),
            (valid_ffprobe_json(audio={"channels": 1}), "channels"),
            (valid_ffprobe_json(audio={"sample_rate": "44100"}), "sample rate"),
            (valid_ffprobe_json(audio={"codec_name": "mp3"}), "codec"),
        ]
        for payload, label in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                paths = mp4_media_fixture(Path(directory))
                runner = FakeMuxRunner(ffprobe_json=payload)
                with self.assertRaises(probe.MP4MuxFailure):
                    self._run_mux_direct(paths, runner)
                self.assertFalse(paths["mp4"].exists())

    def test_duration_outside_explicit_tolerance_and_missing_probe_data_fail(self):
        cases = [
            valid_ffprobe_json(format={"duration": str(probe.MP4_EXPECTED_DURATION_SECONDS + probe.MP4_DURATION_TOLERANCE_SECONDS + 0.001)}),
            valid_ffprobe_json(video={"width": None}),
            {"streams": valid_ffprobe_json()["streams"]},
        ]
        for payload in cases:
            with tempfile.TemporaryDirectory() as directory:
                paths = mp4_media_fixture(Path(directory))
                runner = FakeMuxRunner(ffprobe_json=payload)
                with self.assertRaises(probe.MP4MuxFailure):
                    self._run_mux_direct(paths, runner)

    def test_final_mp4_does_not_exist_before_inspection_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            observed = []
            fake = FakeMuxRunner()

            def runner(argv, **kwargs):
                if Path(argv[0]).name == "ffprobe":
                    observed.append(paths["mp4"].exists())
                return fake(argv, **kwargs)

            result = self._run_success(Path(directory), runner=runner)[2]
            self.assertEqual(observed, [False])
            self.assertTrue(paths["mp4"].is_file())
            self.assertFalse(paths["mp4_partial"].exists())
            self.assertEqual(result["status"], "completed")

    def test_successful_inspection_publishes_atomically_and_validates_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            observed = []
            root = Path(directory)
            paths = mp4_media_fixture(root)

            def rename(source, destination):
                observed.append((Path(source).name, not Path(destination).exists()))
                source.rename(destination)

            runner = FakeMuxRunner()
            result = self._run_success(root, runner=runner, rename=rename)[2]
            self.assertEqual(observed, [("dodecahedron.partial.mp4", True)])
            self.assertTrue(paths["mp4"].is_file())
            self.assertFalse(paths["mp4_partial"].exists())
            validated = probe.validate_mp4_manifest(
                paths["mp4_manifest"],
                paths["mp4"],
                expected_attempt_identifier="attempt-1",
                expected_published_path=paths["mp4"],
                expected_video_manifest_path=paths["video_frame_manifest"],
                expected_audio_manifest_path=paths["audio_manifest"],
            )
            self.assertEqual(validated["mp4_sha256"], result["mp4_artifact"]["mp4_sha256"])
            paths["mp4"].write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum linkage"):
                probe.validate_mp4_manifest(
                    paths["mp4_manifest"],
                    paths["mp4"],
                    expected_attempt_identifier="attempt-1",
                    expected_published_path=paths["mp4"],
                    expected_video_manifest_path=paths["video_frame_manifest"],
                    expected_audio_manifest_path=paths["audio_manifest"],
                )

    def test_existing_final_mp4_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            paths["mp4"].write_bytes(b"keep-existing")
            runner = FakeMuxRunner()
            with self.assertRaisesRegex(FileExistsError, "existing final MP4"):
                self._run_mux_direct(paths, runner)
            self.assertEqual(paths["mp4"].read_bytes(), b"keep-existing")
            self.assertEqual(runner.calls, [])

    def test_mux_failure_preserves_png_and_wav_and_records_cleanup_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            runner = FakeMuxRunner(ffmpeg_returncode=1, ffmpeg_stderr="synthetic stderr")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                self._run_mux_direct(paths, runner)
            receipt = context.exception.receipt
            self.assertTrue(receipt["frames_preserved"])
            self.assertTrue(receipt["wav_preserved"])
            self.assertEqual(receipt["ffmpeg"]["stderr"], "synthetic stderr")
            self.assertFalse(paths["mp4"].exists())
            self.assertFalse(paths["mp4_partial"].exists())

    def test_no_retry_or_replacement_execution_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            runner = FakeMuxRunner(ffprobe_returncode=1, ffprobe_stderr="probe failed")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                self._run_mux_direct(paths, runner)
            receipt = context.exception.receipt
            self.assertEqual(len(runner.calls), 2)
            self.assertEqual(receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})
            self.assertTrue(receipt["retry_suppressed"])
            self.assertEqual(receipt["ffprobe"]["stderr"], "probe failed")

    def test_apply_report_records_completed_mp4_artifact_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            report = mux_report_fixture(paths)
            result = probe.apply_mp4_mux_report(report, paths, subprocess_runner=FakeMuxRunner())
            self.assertEqual(result["mp4_mux_status"], "completed")
            self.assertEqual(result["standalone_media"]["mp4_mux_status"], "completed")
            self.assertEqual(result["mp4_mux"]["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})
            self.assertIsNone(result["mux_failure"])
            self.assertTrue(result["mux_timing"]["executed"])

    def test_apply_report_records_failed_mux_without_losing_standalone_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = mp4_media_fixture(Path(directory))
            report = mux_report_fixture(paths)
            result = probe.apply_mp4_mux_report(
                report,
                paths,
                subprocess_runner=FakeMuxRunner(ffmpeg_returncode=1),
            )
            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["standalone_media_status"], "completed")
            self.assertEqual(result["video_status"], "completed")
            self.assertEqual(result["audio_status"], "completed")
            self.assertEqual(result["mp4_mux_status"], "failed")
            self.assertIsInstance(result["mux_failure"], dict)
            self.assertTrue(result["mux_failure"]["retry_suppressed"])

    def test_real_ffmpeg_and_ffprobe_are_not_used_by_fake_contract_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            _paths, runner, _result = self._run_success(Path(directory))
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg", "ffprobe"])
            self.assertTrue(all(call[1]["check"] is False for call in runner.calls))

    def test_public_parser_surface_remains_single_run_command(self):
        parser = probe.build_parser()
        parsed = parser.parse_args(
            [
                "run-derived-full-schedule",
                "--checkpoint-root", "checkpoint",
                "--derived-transformer", "derived",
                "--output-root", "output",
                "--prompt", probe.LOCKED_PROMPT,
                "--seed", "0",
                "--active-memory-tolerance-bytes", "0",
            ]
        )
        self.assertEqual(parsed.command, "run-derived-full-schedule")
        self.assertFalse(hasattr(parsed, "ffmpeg"))
        self.assertFalse(hasattr(parsed, "ffprobe"))

    def _run_mux_direct(self, paths, runner):
        return probe.execute_mp4_mux(
            frames_directory=paths["frames"],
            video_manifest_path=paths["video_frame_manifest"],
            wav_path=paths["audio_wav"],
            audio_manifest_path=paths["audio_manifest"],
            mp4_partial_path=paths["mp4_partial"],
            mp4_final_path=paths["mp4"],
            mp4_manifest_path=paths["mp4_manifest"],
            attempt_identifier=paths["attempt_identifier"],
            launch_gate=successful_mux_gate(),
            subprocess_runner=runner,
        )


if __name__ == "__main__":
    unittest.main()

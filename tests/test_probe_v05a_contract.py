"""MLX-free behavioral contracts for the v0.5a decoder-only proof."""

from __future__ import annotations

import importlib.util
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile
import unittest
import wave
import weakref

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v05a_decoders", ROOT / "scripts" / "probe_v05a_decoders.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class VideoConfig:
    latent_channels = 24
    out_channels = 3
    temporal_compression_ratio = 4
    spatial_compression_ratio = 16
    token_drop = 3
    clip_length = 17
    latents_mean = tuple(0.0 for _ in range(24))
    latents_std = tuple(1.0 for _ in range(24))


class AudioConfig:
    latent_channels = 32
    sampling_rate = 32000
    decoder_rates = (5, 5, 2, 2, 2, 2, 2)
    latents_mean = tuple(0.0 for _ in range(32))
    latents_std = tuple(1.0 for _ in range(32))

    @property
    def hop_length(self):
        return 800


class FakeMLXArray:
    __mlx_array__ = True

    def __init__(self, data, dtype="bfloat16"):
        self.data = np.asarray(data, dtype=np.float32)
        self.dtype = dtype
        self.astype_calls = []

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype):
        self.astype_calls.append(dtype)
        return FakeMLXArray(self.data, str(dtype))

    def __array__(self, dtype=None, copy=None):
        if self.dtype == "bfloat16":
            raise AssertionError("BF16 MLX array was converted directly")
        return np.asarray(self.data, dtype=dtype)


class FakeMLX:
    float32 = "float32"

    @staticmethod
    def eval(*values):
        return None


class FakeAllocator:
    def __init__(self, *, purge_error=None):
        self.cache = 4096
        self.purge_calls = 0
        self.purge_error = purge_error
        self.on_purge = None

    def get_active_memory(self):
        return 0

    def get_cache_memory(self):
        return self.cache

    def get_peak_memory(self):
        return 0

    def clear_cache(self):
        self.purge_calls += 1
        if self.purge_error is not None:
            raise self.purge_error
        self.cache = 0
        if self.on_purge is not None:
            self.on_purge()


class Finalizable:
    def __init__(self, label):
        self.label = label


class ProbeV05AContractTests(unittest.TestCase):
    @staticmethod
    def video_layout():
        return probe.resolve_video_decode_layout(VideoConfig())

    def test_import_bootstrap_exposes_repository_root(self):
        self.assertIn(str(probe.ROOT), sys.path)

    def test_cli_help_succeeds_outside_repository(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "probe_v05a_decoders.py"), "--help"],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("decode-deterministic-media", completed.stdout)

    def test_direct_script_bootstrap_imports_package_outside_repository(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        bootstrap_check = (
            "import runpy, sys\n"
            "from pathlib import Path\n"
            "script = Path(sys.argv[1]).resolve()\n"
            "runpy.run_path(str(script), run_name='probe_v05a_bootstrap')\n"
            "import minimax_h3_mlx\n"
            "print(next(iter(minimax_h3_mlx.__path__)))\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", bootstrap_check, str(ROOT / "scripts" / "probe_v05a_decoders.py")],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(str(ROOT / "minimax_h3_mlx"), completed.stdout)

    def test_exact_cli_surface(self):
        parser = probe.build_parser()
        action = next(item for item in parser._actions if item.dest == "command")
        self.assertEqual(set(action.choices), {"decode-deterministic-media"})

    def test_cli_has_no_generation_arguments(self):
        parser = probe.build_parser()
        action = next(item for item in parser._actions if item.dest == "command")
        help_text = action.choices["decode-deterministic-media"].format_help().lower()
        for forbidden in ("prompt", "transformer", "scheduler", "denois", "text-encoder"):
            self.assertNotIn(forbidden, help_text)

    def test_source_inspection_contract_is_explicit(self):
        self.assertEqual(len(probe.SOURCE_INSPECTION_FILES), 10)
        self.assertIn("minimax_h3_mlx/video_decode_layout.py", probe.SOURCE_INSPECTION_FILES)
        self.assertIn("minimax_h3_mlx/video_vae.py", probe.SOURCE_INSPECTION_FILES)
        self.assertIn("minimax_h3_mlx/audio_vae.py", probe.SOURCE_INSPECTION_FILES)
        self.assertIn("tests/test_packing_parity.py", probe.SOURCE_INSPECTION_FILES)
        contracts = probe.build_source_contracts(VideoConfig(), AudioConfig(), self.video_layout())
        self.assertEqual(contracts["video"]["native_latent_axis_order"], "(B, C, F, H, W)")
        self.assertEqual(contracts["audio"]["native_latent_axis_order"], "(B, C, L)")

    def test_fixture_uses_only_authentic_video_config_fields(self):
        config = VideoConfig()
        for field_name in ("clip_length", "token_drop", "temporal_compression_ratio"):
            self.assertTrue(hasattr(config, field_name))
        for derived_name in ("tokens_chunk_size", "token_overlap", "frame_pre_padding"):
            self.assertFalse(hasattr(config, derived_name))

    def test_resolve_video_decode_layout_matches_decoder_formulas(self):
        layout = self.video_layout()
        self.assertEqual(layout.clip_length, VideoConfig.clip_length)
        self.assertEqual(layout.temporal_compression_ratio, VideoConfig.temporal_compression_ratio)
        self.assertEqual(layout.tokens_chunk_size, 5)
        self.assertEqual(layout.token_drop, 3)
        self.assertEqual(layout.token_overlap, 2)
        self.assertEqual(layout.frame_pre_padding, 3)
        self.assertEqual(layout.frame_overlap, 5)
        self.assertEqual(layout.chunk_num_frames, 20)
        self.assertEqual(layout.tail_trim_remainder, 1)
        self.assertEqual(layout.minimum_latent_frames, 7)
        with self.assertRaises((AttributeError, TypeError)):
            layout.tokens_chunk_size = 6

    def test_missing_authentic_video_config_field_fails_clearly(self):
        config = SimpleNamespace(temporal_compression_ratio=4, token_drop=3)
        with self.assertRaisesRegex(ValueError, "clip_length"):
            probe.resolve_video_decode_layout(config)

    def test_video_geometry_validation_and_minimum(self):
        layout = self.video_layout()
        geometry = probe.select_minimum_video_geometry(VideoConfig(), layout)
        self.assertEqual(geometry["latent_frames"], 7)
        self.assertEqual(geometry["latent_height"], 1)
        self.assertEqual(geometry["latent_width"], 1)
        with self.assertRaisesRegex(ValueError, "at least 7"):
            probe.validate_video_geometry(VideoConfig(), {**geometry, "latent_frames": 6}, layout)

    def test_video_expected_frame_count_uses_the_resolved_layout(self):
        layout = self.video_layout()
        geometry = probe.select_minimum_video_geometry(VideoConfig(), layout)
        self.assertEqual(probe.video_decoded_frame_count(geometry["latent_frames"], layout), 22)
        self.assertEqual(
            probe.expected_video_output_shape(VideoConfig(), geometry, layout),
            (1, 3, 22, 16, 16),
        )

    def test_authentic_video_contract_report_generation_succeeds(self):
        layout = self.video_layout()
        contract = probe.build_video_source_contract(VideoConfig(), layout)
        self.assertEqual(contract["tokens_chunk_size"], layout.tokens_chunk_size)
        self.assertEqual(contract["frame_overlap"], layout.frame_overlap)
        self.assertEqual(contract["minimum_accepted_latent_frame_count"], 7)
        self.assertIn("tail trim uses clip_length%ratio_t", contract["decoded_frame_count_formula"])

    def test_audio_geometry_validation_and_minimum(self):
        geometry = probe.select_minimum_audio_geometry(AudioConfig())
        self.assertEqual(geometry["batch"], 2)
        self.assertEqual(geometry["latent_length"], 1)
        with self.assertRaises(ValueError):
            probe.validate_audio_geometry(AudioConfig(), {**geometry, "latent_length": 0})

    def test_deterministic_latents_are_reproducible(self):
        first = probe.deterministic_values(32, 0)
        second = probe.deterministic_values(32, 0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, probe.deterministic_values(32, 1))
        self.assertTrue(np.isfinite(np.asarray(first)).all())

    def test_mlx_safe_fingerprint_casts_before_numpy(self):
        value = FakeMLXArray([[1.0, 2.0]])
        fingerprint = probe.array_fingerprint(value, FakeMLX())
        self.assertEqual(len(fingerprint), 64)
        self.assertEqual(value.astype_calls, ["float32"])

    def test_video_phase_precedes_audio_phase(self):
        events = [
            "video-baseline", "video-config-load", "video-vae-load", "video-decode",
            "video-runtime-clear", "video-release-gate", "audio-baseline",
            "audio-config-load", "audio-vae-load", "audio-decode",
            "audio-runtime-clear", "audio-release-gate",
        ]
        probe.validate_phase_order(events)
        with self.assertRaisesRegex(ValueError, "sequential"):
            probe.validate_phase_order(list(reversed(events)))

    def test_video_release_completes_before_audio_load(self):
        calls = []

        def video():
            calls.append("video-release")
            return {"release_gate": {"passed": True}}

        def audio():
            calls.append("audio-load")
            return {"release_gate": {"passed": True}}

        probe.run_sequential_phases(video, audio)
        self.assertEqual(calls, ["video-release", "audio-load"])

    def test_no_decoder_load_occurs_before_layout_validation(self):
        allocator = FakeAllocator()
        decoder_calls = []
        original_runtime_imports = probe._runtime_imports

        def load_video_config(_):
            return SimpleNamespace(temporal_compression_ratio=4, token_drop=3)

        def load_video(_):
            decoder_calls.append("video")
            raise AssertionError("decoder load must follow layout validation")

        try:
            probe._runtime_imports = lambda: (
                allocator,
                load_video_config,
                load_video,
                lambda _: AudioConfig(),
                lambda _: object(),
                (lambda *_: None, lambda *_: None),
            )
            with tempfile.TemporaryDirectory() as directory:
                paths = {
                    "frames": str(Path(directory) / "frames"),
                    "audio_wav": str(Path(directory) / "audio.wav"),
                }
                args = SimpleNamespace(
                    checkpoint_root=directory,
                    active_memory_tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
                )
                with self.assertRaises(probe.ProbeFailure) as caught:
                    probe._runtime_run(args, paths)
        finally:
            probe._runtime_imports = original_runtime_imports

        self.assertIn("clip_length", str(caught.exception.original_error))
        self.assertEqual(decoder_calls, [])

    def test_video_output_shape_and_dtype(self):
        layout = self.video_layout()
        geometry = probe.select_minimum_video_geometry(VideoConfig(), layout)
        raw = np.zeros(probe.expected_video_output_shape(VideoConfig(), geometry, layout), dtype=np.float32)
        frames = probe.video_frames_from_raw(raw)
        probe.validate_video_output(raw, frames, VideoConfig(), geometry, layout)
        self.assertEqual(frames.shape, (22, 16, 16, 3))
        self.assertEqual(frames.dtype, np.uint8)

    def test_rgb_range_validation(self):
        layout = self.video_layout()
        geometry = probe.select_minimum_video_geometry(VideoConfig(), layout)
        raw = np.zeros(probe.expected_video_output_shape(VideoConfig(), geometry, layout), dtype=np.float32)
        frames = probe.video_frames_from_raw(raw)
        frames[0, 0, 0, 0] = 255
        probe.validate_video_output(raw, frames, VideoConfig(), geometry, layout)
        with self.assertRaisesRegex(ValueError, "uint8"):
            probe.validate_video_output(raw, frames.astype(np.int16), VideoConfig(), geometry, layout)

    def test_audio_stereo_shape_validation(self):
        geometry = probe.select_minimum_audio_geometry(AudioConfig())
        raw = np.zeros(probe.expected_audio_output_shape(AudioConfig(), geometry), dtype=np.float32)
        waveform = raw[:, 0, :].copy()
        probe.validate_audio_output(raw, waveform, AudioConfig(), geometry)
        with self.assertRaisesRegex(ValueError, "stereo waveform shape"):
            probe.validate_audio_output(raw, waveform[:1], AudioConfig(), geometry)

    def test_audio_sample_count_validation(self):
        geometry = probe.select_minimum_audio_geometry(AudioConfig())
        expected = probe.expected_audio_output_shape(AudioConfig(), geometry)
        self.assertEqual(expected, (2, 1, 800))
        raw = np.zeros((2, 1, 799), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "raw audio shape"):
            probe.validate_audio_output(raw, raw[:, 0, :], AudioConfig(), geometry)

    def test_finite_waveform_enforcement(self):
        geometry = probe.select_minimum_audio_geometry(AudioConfig())
        raw = np.zeros(probe.expected_audio_output_shape(AudioConfig(), geometry), dtype=np.float32)
        raw[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            probe.validate_audio_output(raw, raw[:, 0, :], AudioConfig(), geometry)

    def test_wav_metadata_and_checksum_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(32000)
                handle.writeframes(b"\x00\x00" * 2 * 800)
            metadata = probe.wav_metadata(path)
            self.assertEqual(metadata["channels"], 2)
            self.assertEqual(metadata["sample_rate"], 32000)
            self.assertEqual(metadata["sample_count"], 800)
            self.assertEqual(len(metadata["sha256"]), 64)
            with wave.open(str(path), "rb") as handle:
                self.assertEqual(handle.getsampwidth(), 2)

    def test_existing_output_refusal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v0.5a"
            output.mkdir()
            (output / "decoder-report.json").write_text("{}")
            with self.assertRaisesRegex(FileExistsError, "refusing"):
                probe.ensure_output_namespace(output, overwrite=False)
            probe.ensure_output_namespace(output, overwrite=True)

    def test_strict_report_schema(self):
        base = probe._build_base_report(
            Path("/checkpoint"),
            Path("/checkpoint/video_vae"),
            Path("/checkpoint/audio_vae"),
            {"report": "/out/report.json"},
            {},
        )
        failure = probe.ProbeFailure("video", RuntimeError("synthetic"))
        report = probe._build_failure_report(
            base,
            failure,
            phase_order=(),
            partial_output_paths=(),
            geometries={},
            memory={},
            residency={
                "video_vae_ever_loaded": False,
                "video_vae_currently_resident": False,
                "audio_vae_ever_loaded": False,
                "audio_vae_currently_resident": False,
            },
        )
        probe.validate_report(report)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe.validate_report({**report, "extra": True})

    def test_failure_receipt_preservation(self):
        base = probe._build_base_report(
            Path("/checkpoint"),
            Path("/checkpoint/video_vae"),
            Path("/checkpoint/audio_vae"),
            {},
            {},
        )
        original = RuntimeError("decode broke")
        cleanup = RuntimeError("purge broke")
        failure = probe.ProbeFailure(
            "video",
            original,
            cleanup_error=cleanup,
            completed_stages=("video-baseline", "video-decode"),
        )
        report = probe._build_failure_report(
            base,
            failure,
            phase_order=failure.completed_stages,
            partial_output_paths=("/out/frame_00000.png",),
            geometries={"video": {"latent_frames": 7}},
            memory={"video": {"memory_before_config": {"mlx_active_bytes": 1}}},
            residency={
                "video_vae_ever_loaded": True,
                "video_vae_currently_resident": True,
                "audio_vae_ever_loaded": False,
                "audio_vae_currently_resident": False,
            },
        )
        self.assertEqual(report["failure"]["active_phase"], "video")
        self.assertEqual(report["failure"]["error"]["message"], "decode broke")
        self.assertEqual(report["failure"]["cleanup_error"]["message"], "purge broke")
        self.assertEqual(report["failure"]["partial_output_paths"], ["/out/frame_00000.png"])

    def test_video_cleanup_failure_suppresses_audio(self):
        audio_calls = []

        def video():
            return {"release_gate": {"passed": False, "active_memory_within_tolerance": False}}

        def audio():
            audio_calls.append("called")
            return {"release_gate": {"passed": True}}

        with self.assertRaisesRegex(probe.ReleaseGateError, "audio phase was suppressed"):
            probe.run_sequential_phases(video, audio)
        self.assertEqual(audio_calls, [])

    def test_audio_cleanup_failure_suppresses_success(self):
        def video():
            return {"release_gate": {"passed": True}}

        def audio():
            return {"release_gate": {"passed": False, "final_allocator_cache_gate": False}}

        with self.assertRaisesRegex(probe.ReleaseGateError, "overall proof"):
            probe.run_sequential_phases(video, audio)

    def test_active_memory_gate_enforcement(self):
        baseline = {"mlx_active_bytes": 100, "mlx_allocator_cache_bytes": 1000}
        after = {"mlx_active_bytes": 100 + probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES + 1,
                 "mlx_allocator_cache_bytes": 0}
        result = probe.evaluate_release_gate(
            baseline, after, references_cleared=True, allocator_purge_available=True
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["active_memory_within_tolerance"])

    def test_final_allocator_cache_gate(self):
        baseline = {"mlx_active_bytes": 100, "mlx_allocator_cache_bytes": 1000}
        after = {"mlx_active_bytes": 100, "mlx_allocator_cache_bytes": 1}
        result = probe.evaluate_release_gate(
            baseline, after, references_cleared=True, allocator_purge_available=True
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["final_allocator_cache_gate"])

    def test_all_runtime_references_are_cleared(self):
        references = {key: object() for key in probe.RUNTIME_REFERENCE_KEYS}
        self.assertTrue(probe.clear_runtime_references(references))
        self.assertTrue(all(value is None for value in references.values()))

    def test_video_worker_objects_are_collected_before_release_gate(self):
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}
        baseline = {"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096}
        finalized = []

        def worker():
            latent = Finalizable("video-latent")
            raw = Finalizable("video-raw")
            weak_latent = weakref.ref(latent, lambda _: finalized.append("latent"))
            weak_raw = weakref.ref(raw, lambda _: finalized.append("raw"))
            references["latent"] = latent
            references["raw_decoded"] = raw
            allocator.on_purge = lambda: self.assertTrue(
                weak_latent() is None and weak_raw() is None
            )
            return {"weakrefs": (weak_latent, weak_raw)}

        result = probe.execute_scoped_phase(
            worker,
            phase="video",
            mx=allocator,
            references=references,
            baseline=baseline,
            tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        )
        gc.collect()
        self.assertTrue(result["release_gate"]["passed"])
        self.assertEqual(set(finalized), {"latent", "raw"})

    def test_audio_worker_objects_are_collected_before_release_gate(self):
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}
        baseline = {"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096}
        weak_objects = []

        def worker():
            latent = Finalizable("audio-latent")
            raw = Finalizable("audio-raw")
            weak_objects.extend((weakref.ref(latent), weakref.ref(raw)))
            references["latent"] = latent
            references["raw_decoded"] = raw
            return {"written_output_paths": []}

        allocator.on_purge = lambda: self.assertTrue(all(reference() is None for reference in weak_objects))
        result = probe.execute_scoped_phase(
            worker,
            phase="audio",
            mx=allocator,
            references=references,
            baseline=baseline,
            tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
        )
        self.assertTrue(result["release_gate"]["passed"])
        self.assertTrue(all(reference() is None for reference in weak_objects))

    def test_configuration_failure_still_runs_cleanup(self):
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}

        def worker():
            raise ValueError("configuration failed")

        with self.assertRaises(probe.ProbeFailure) as caught:
            probe.execute_scoped_phase(
                worker,
                phase="video",
                mx=allocator,
                references=references,
                baseline={"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096},
                tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            )
        self.assertEqual(str(caught.exception.original_error), "configuration failed")
        self.assertIsNone(caught.exception.cleanup_error)
        self.assertEqual(allocator.purge_calls, 1)

    def test_latent_fingerprint_failure_still_runs_cleanup(self):
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}
        weak_latent = None

        def worker():
            nonlocal weak_latent
            latent = Finalizable("latent")
            references["latent"] = latent
            weak_latent = weakref.ref(latent)
            raise ValueError("latent fingerprint failed")

        with self.assertRaises(probe.ProbeFailure) as caught:
            probe.execute_scoped_phase(
                worker,
                phase="video",
                mx=allocator,
                references=references,
                baseline={"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096},
                tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            )
        self.assertEqual(str(caught.exception.original_error), "latent fingerprint failed")
        self.assertIsNotNone(weak_latent)
        self.assertIsNone(weak_latent())
        self.assertEqual(allocator.purge_calls, 1)

    def test_allocating_decoder_loader_failure_still_triggers_purge(self):
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}
        weak_decoder = None

        def worker():
            nonlocal weak_decoder
            decoder = Finalizable("decoder")
            references["decoder"] = decoder
            weak_decoder = weakref.ref(decoder)
            raise RuntimeError("decoder loader allocated then raised")

        with self.assertRaises(probe.ProbeFailure) as caught:
            probe.execute_scoped_phase(
                worker,
                phase="audio",
                mx=allocator,
                references=references,
                baseline={"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096},
                tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            )
        self.assertEqual(str(caught.exception.original_error), "decoder loader allocated then raised")
        self.assertIsNotNone(weak_decoder)
        self.assertIsNone(weak_decoder())
        self.assertEqual(allocator.purge_calls, 1)

    def test_original_and_cleanup_errors_are_both_preserved(self):
        allocator = FakeAllocator(purge_error=RuntimeError("purge broke"))
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}

        def worker():
            raise RuntimeError("decode broke")

        with self.assertRaises(probe.ProbeFailure) as caught:
            probe.execute_scoped_phase(
                worker,
                phase="video",
                mx=allocator,
                references=references,
                baseline={"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096},
                tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            )
        self.assertEqual(str(caught.exception.original_error), "decode broke")
        self.assertIsNotNone(caught.exception.cleanup_error)
        self.assertEqual(str(caught.exception.cleanup_error), "purge broke")
        self.assertIsNotNone(caught.exception.cleanup_result)
        self.assertIn("memory_before_allocator_purge", caught.exception.cleanup_result)
        self.assertIn("memory_after_allocator_purge", caught.exception.cleanup_result)

    def test_current_residency_is_false_only_after_release_success(self):
        state = {"currently_resident": True}
        allocator = FakeAllocator()
        references = {key: None for key in probe.RUNTIME_REFERENCE_KEYS}

        def worker():
            return {"residency_at_worker_return": state["currently_resident"]}

        def after_release(gate):
            self.assertTrue(gate["passed"])
            state["currently_resident"] = False

        result = probe.execute_scoped_phase(
            worker,
            phase="video",
            mx=allocator,
            references=references,
            baseline={"mlx_active_bytes": 0, "mlx_allocator_cache_bytes": 4096},
            tolerance_bytes=probe.RELEASE_ACTIVE_MEMORY_TOLERANCE_BYTES,
            on_release_success=after_release,
        )
        self.assertTrue(result["release_gate"]["passed"])
        self.assertFalse(state["currently_resident"])

    def test_video_release_gate_precedes_audio_loading(self):
        calls = []
        residency = {"video_vae_currently_resident": True}

        def video():
            calls.append(("video-release", residency["video_vae_currently_resident"]))
            residency["video_vae_currently_resident"] = False
            return {"release_gate": {"passed": True}, "video_vae_currently_resident": False}

        def audio():
            calls.append(("audio-load", residency["video_vae_currently_resident"]))
            return {"release_gate": {"passed": True}}

        probe.run_sequential_phases(video, audio)
        self.assertEqual(calls, [("video-release", True), ("audio-load", False)])

    def test_overwrite_removes_only_known_prior_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "v0.5a"
            frames = output / "frames"
            frames.mkdir(parents=True)
            (frames / "frame_00000.png").write_bytes(b"old")
            (output / "decoder-audio.wav").write_bytes(b"old")
            (output / "decoder-report.json").write_text("old")
            unrelated = output / "keep.me"
            unrelated.write_text("keep")
            probe.ensure_output_namespace(output, overwrite=True)
            self.assertFalse((frames / "frame_00000.png").exists())
            self.assertFalse((output / "decoder-audio.wav").exists())
            self.assertFalse((output / "decoder-report.json").exists())
            self.assertTrue(unrelated.exists())

    def test_frame_file_count_must_equal_decoded_count(self):
        with tempfile.TemporaryDirectory() as directory:
            frames = Path(directory)
            (frames / "frame_00000.png").write_bytes(b"one")
            with self.assertRaisesRegex(ValueError, "count mismatch"):
                probe.frame_file_metadata(frames, 2)
            (frames / "frame_00001.png").write_bytes(b"two")
            checksums = probe.frame_file_metadata(frames, 2)
            self.assertEqual(len(checksums), 2)
            self.assertTrue(all(len(item["sha256"]) == 64 for item in checksums))

    def test_wav_metadata_must_match_decoded_waveform(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.wav"
            waveform = np.zeros((2, 800), dtype=np.float32)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(32000)
                handle.writeframes(b"\x00\x00" * 2 * 800)
            metadata = probe.wav_metadata(path)
            probe.validate_wav_metadata(metadata, waveform, AudioConfig())
            with self.assertRaisesRegex(ValueError, "sample count mismatch"):
                probe.validate_wav_metadata({**metadata, "sample_count": 799}, waveform, AudioConfig())

    def test_no_generation_component_loaders_are_invoked(self):
        self.assertEqual(
            probe._runtime_imports.__doc__,
            "Import only the two VAE config/load pairs and the existing media helpers.",
        )
        self.assertNotIn("load_dit", Path(probe.__file__).read_text())
        self.assertNotIn("MiniMaxH3Pipeline", Path(probe.__file__).read_text())


if __name__ == "__main__":
    unittest.main()

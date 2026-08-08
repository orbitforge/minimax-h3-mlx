"""MLX-free Slice 3B3A proof-side 256x256 video decoder geometry contracts."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v05d_derived_full_schedule.py"
SPEC = importlib.util.spec_from_file_location("probe_v05e_slice3b3a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class FakeMLXArray:
    def __init__(self, data, dtype, mx):
        self.data = np.asarray(data, dtype=np.float32).copy()
        self.dtype = dtype
        self.mx = mx
        self.materialized = False

    @property
    def shape(self):
        return self.data.shape

    def astype(self, dtype):
        return FakeMLXArray(self.data, dtype, self.mx)

    def reshape(self, *shape):
        return FakeMLXArray(self.data.reshape(*shape), self.dtype, self.mx)

    def __array__(self, dtype=None, copy=None):
        if not self.materialized:
            raise AssertionError("fake MLX value was converted before materialization")
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

    def array(self, value, dtype):
        self.events.append(("array", dtype))
        return FakeMLXArray(value, dtype, self)

    def eval(self, *values):
        self.events.append(("eval", len(values)))
        for value in values:
            if isinstance(value, FakeMLXArray):
                value.materialized = True


def geometry(size: int) -> dict[str, object]:
    return probe.canonical_geometry_contract(size)


def video_config():
    config = probe._proof_video_config()
    config.latents_mean = tuple(0.0 for _ in range(24))
    config.latents_std = tuple(1.0 for _ in range(24))
    return config


def valid_final_video_fixture(root: Path, selected_geometry: dict[str, object]):
    video = np.zeros(tuple(selected_geometry["video_native_shape"]), dtype=np.float32)
    audio = np.zeros(tuple(selected_geometry["audio_native_shape"]), dtype=np.float32)
    artifact = {
        "artifact_identity": "minimax-h3-mlx-v05d-final-native-latent",
        "schema_version": probe.FINAL_ARTIFACT_SCHEMA_VERSION,
        "attempt_identifier": "slice3b3a-test-attempt",
        "geometry": selected_geometry,
        "native_video": probe.shape_dtype(video, logical_dtype="bfloat16")
        | {"fingerprint": probe.array_fingerprint(video, logical_dtype="bfloat16")},
        "native_audio": probe.shape_dtype(audio, logical_dtype="bfloat16")
        | {"fingerprint": probe.array_fingerprint(audio, logical_dtype="bfloat16")},
        "packed_final_state_fingerprint": "slice3b3a-packed-state",
        "schedule_contract": probe.build_full_schedule().receipt(),
        "completed_transition_count": probe.EXPECTED_DENOISING_TRANSITIONS,
        "transformer_forward_count": probe.EXPECTED_TRANSFORMER_FORWARDS,
        "scheduler_update_counts": {
            "video": probe.EXPECTED_DENOISING_TRANSITIONS,
            "audio": probe.EXPECTED_DENOISING_TRANSITIONS,
        },
        "streamed_adaln_lifecycle": probe.expected_lifecycle_totals(),
        "worker_identity": "derived",
        "worker_exit_receipt": {
            "worker_started": True,
            "worker_exit_observed": True,
            "worker_exit_code": 0,
            "worker_pid": 123,
            "worker_termination_confirmed": True,
        },
        "transformer_release_receipt": {
            "passed": True,
            "allocator_cache_zero": True,
            "active_memory_within_tolerance": True,
            "memory_after_allocator_purge": {"active": 0, "allocator_cache": 0},
        },
        "final_active_memory": 0,
        "final_allocator_cache": 0,
        "final_allocator_cache_zero": True,
        "final_artifact_npz_sha256": None,
        "metadata_sha256": None,
        "memory_receipt": {},
        "git_identity": {},
        "checkpoint_identity": {"checkpoint": "slice3b3a-test"},
    }
    artifact["streamed_adaln_lifecycle"]["sessions"] = probe.expected_lifecycle_sessions()
    artifact_path = root / f"final-video-{selected_geometry['video_width']}.npz"
    metadata_path = root / f"final-video-{selected_geometry['video_width']}.json"
    probe._write_npz(artifact_path, {"final_video_native": video, "final_audio_native": audio})
    artifact["final_artifact_npz_sha256"] = probe.sha256_file(artifact_path)
    artifact["metadata_sha256"] = probe.stable_metadata_sha256(artifact)
    probe._write_json(metadata_path, artifact)
    return artifact_path, metadata_path, artifact, {"final_video_native": video, "final_audio_native": audio}


class Slice3B3ADecoderGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry_128 = geometry(128)
        cls.geometry_256 = geometry(256)

    def test_128_latent_is_accepted_under_128_geometry(self):
        stored = np.zeros((1, 24, 9, 8, 8), dtype=np.float32)
        expected = probe.array_fingerprint(stored, logical_dtype="bfloat16")
        restored, fingerprint = probe.restore_video_latent_logical_bfloat16(
            stored,
            FakeMLX(),
            expected_fingerprint=expected,
            materialize=lambda value: setattr(value, "materialized", True),
            geometry=self.geometry_128,
        )
        self.assertEqual(tuple(restored.shape), (1, 24, 9, 8, 8))
        self.assertEqual(fingerprint, expected)

    def test_256_latent_is_accepted_under_256_geometry(self):
        stored = np.zeros((1, 24, 9, 16, 16), dtype=np.float32)
        expected = probe.array_fingerprint(stored, logical_dtype="bfloat16")
        restored, fingerprint = probe.restore_video_latent_logical_bfloat16(
            stored,
            FakeMLX(),
            expected_fingerprint=expected,
            materialize=lambda value: setattr(value, "materialized", True),
            geometry=self.geometry_256,
        )
        self.assertEqual(tuple(restored.shape), (1, 24, 9, 16, 16))
        self.assertEqual(fingerprint, expected)

    def test_128_latent_is_rejected_under_256_geometry(self):
        stored = np.zeros((1, 24, 9, 8, 8), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            probe.restore_video_latent_logical_bfloat16(
                stored,
                FakeMLX(),
                expected_fingerprint="unused",
                geometry=self.geometry_256,
            )

    def test_256_latent_is_rejected_under_128_geometry(self):
        stored = np.zeros((1, 24, 9, 16, 16), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            probe.restore_video_latent_logical_bfloat16(
                stored,
                FakeMLX(),
                expected_fingerprint="unused",
                geometry=self.geometry_128,
            )

    def test_locked_video_config_returns_exact_128_decoder_shapes(self):
        result = probe.validate_locked_video_config(
            video_config(), probe._proof_video_layout(), geometry=self.geometry_128
        )
        self.assertEqual(result["video_native_shape"], [1, 24, 9, 8, 8])
        self.assertEqual(result["video_raw_shape"], [1, 3, 30, 128, 128])
        self.assertEqual(result["video_rgb_shape"], [30, 128, 128, 3])
        self.assertEqual(result["frame_count"], 30)

    def test_locked_video_config_returns_exact_256_decoder_shapes(self):
        result = probe.validate_locked_video_config(
            video_config(), probe._proof_video_layout(), geometry=self.geometry_256
        )
        self.assertEqual(result["video_native_shape"], [1, 24, 9, 16, 16])
        self.assertEqual(result["video_raw_shape"], [1, 3, 30, 256, 256])
        self.assertEqual(result["video_rgb_shape"], [30, 256, 256, 3])
        self.assertEqual(result["frame_count"], 30)

    def test_128_raw_output_remains_exact(self):
        fake_mx = FakeMLX()
        raw = FakeMLXArray(np.zeros((1, 3, 30, 128, 128), dtype=np.float32), "float32", fake_mx)
        raw_np, receipt = probe.materialize_and_validate_video_raw_output(
            raw,
            fake_mx,
            geometry=self.geometry_128,
            materialize=lambda value: setattr(value, "materialized", True),
        )
        self.assertEqual(tuple(raw_np.shape), (1, 3, 30, 128, 128))
        self.assertEqual(receipt["shape"], [1, 3, 30, 128, 128])

    def test_256_raw_output_is_exact(self):
        fake_mx = FakeMLX()
        raw = FakeMLXArray(np.zeros((1, 3, 30, 256, 256), dtype=np.float32), "float32", fake_mx)
        raw_np, receipt = probe.materialize_and_validate_video_raw_output(
            raw,
            fake_mx,
            geometry=self.geometry_256,
            materialize=lambda value: setattr(value, "materialized", True),
        )
        self.assertEqual(tuple(raw_np.shape), (1, 3, 30, 256, 256))
        self.assertEqual(receipt["shape"], [1, 3, 30, 256, 256])

    def test_wrong_256_width_fails(self):
        fake_mx = FakeMLX()
        raw = FakeMLXArray(np.zeros((1, 3, 30, 256, 255), dtype=np.float32), "float32", fake_mx)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            probe.materialize_and_validate_video_raw_output(
                raw,
                fake_mx,
                geometry=self.geometry_256,
                materialize=lambda value: setattr(value, "materialized", True),
            )

    def test_wrong_256_height_fails(self):
        fake_mx = FakeMLX()
        raw = FakeMLXArray(np.zeros((1, 3, 30, 255, 256), dtype=np.float32), "float32", fake_mx)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            probe.materialize_and_validate_video_raw_output(
                raw,
                fake_mx,
                geometry=self.geometry_256,
                materialize=lambda value: setattr(value, "materialized", True),
            )

    def test_wrong_frame_count_fails(self):
        fake_mx = FakeMLX()
        raw = FakeMLXArray(np.zeros((1, 3, 29, 256, 256), dtype=np.float32), "float32", fake_mx)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            probe.materialize_and_validate_video_raw_output(
                raw,
                fake_mx,
                geometry=self.geometry_256,
                materialize=lambda value: setattr(value, "materialized", True),
            )

    def test_raw_dtype_and_finite_checks_remain_strict(self):
        fake_mx = FakeMLX()
        bad_dtype = FakeMLXArray(np.zeros((1, 3, 30, 256, 256), dtype=np.float32), "bfloat16", fake_mx)
        with self.assertRaisesRegex(ValueError, "float32"):
            probe.materialize_and_validate_video_raw_output(
                bad_dtype, fake_mx, geometry=self.geometry_256
            )
        nonfinite = np.zeros((1, 3, 30, 256, 256), dtype=np.float32)
        nonfinite[0, 0, 0, 0, 0] = np.nan
        bad_values = FakeMLXArray(nonfinite, "float32", fake_mx)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            probe.materialize_and_validate_video_raw_output(
                bad_values,
                fake_mx,
                geometry=self.geometry_256,
                materialize=lambda value: setattr(value, "materialized", True),
            )

    def test_128_rgb_conversion_remains_exact(self):
        frames, receipt = probe.convert_and_validate_video_rgb(
            np.zeros((1, 3, 30, 128, 128), dtype=np.float32), geometry=self.geometry_128
        )
        self.assertEqual(tuple(frames.shape), (30, 128, 128, 3))
        self.assertEqual(frames.dtype, np.dtype(np.uint8))
        self.assertEqual(receipt["shape"], [30, 128, 128, 3])

    def test_256_rgb_conversion_is_exact(self):
        frames, receipt = probe.convert_and_validate_video_rgb(
            np.zeros((1, 3, 30, 256, 256), dtype=np.float32), geometry=self.geometry_256
        )
        self.assertEqual(tuple(frames.shape), (30, 256, 256, 3))
        self.assertEqual(frames.dtype, np.dtype(np.uint8))
        self.assertEqual(receipt["shape"], [30, 256, 256, 3])

    def test_rgb_shape_and_dtype_checks_remain_strict(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            probe.validate_video_rgb_output(
                np.zeros((30, 256, 255, 3), dtype=np.uint8), geometry=self.geometry_256
            )
        with self.assertRaisesRegex(ValueError, "dtype"):
            probe.validate_video_rgb_output(
                np.zeros((30, 256, 256, 3), dtype=np.float32), geometry=self.geometry_256
            )

    def test_video_receipt_geometry_validates_for_128(self):
        receipt = probe.decoder_worker_receipt("video", geometry=self.geometry_128)
        result = probe.validate_video_decoder_receipt_geometry(
            receipt, expected_geometry=self.geometry_128
        )
        self.assertEqual(result["video_raw_shape"], [1, 3, 30, 128, 128])
        probe.validate_decoder_worker_receipt(
            receipt, identity="video", expected_geometry=self.geometry_128
        )

    def test_video_receipt_geometry_validates_for_256(self):
        receipt = probe.decoder_worker_receipt("video", geometry=self.geometry_256)
        result = probe.validate_video_decoder_receipt_geometry(
            receipt, expected_geometry=self.geometry_256
        )
        self.assertEqual(result["video_native_shape"], [1, 24, 9, 16, 16])
        self.assertEqual(receipt["frame_count"], 30)

    def test_video_receipt_cross_size_mismatch_fails(self):
        receipt = probe.decoder_worker_receipt("video", geometry=self.geometry_128)
        with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
            probe.validate_video_decoder_receipt_geometry(
                receipt, expected_geometry=self.geometry_256
            )

    def test_video_receipt_shape_mutation_fails(self):
        receipt = probe.decoder_worker_receipt("video", geometry=self.geometry_256)
        receipt["video_raw_shape"] = [1, 3, 30, 256, 255]
        with self.assertRaisesRegex(ValueError, "video_raw_shape"):
            probe.validate_video_decoder_receipt_geometry(
                receipt, expected_geometry=self.geometry_256
            )

    def test_128_video_artifact_input_validates_under_128_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path, metadata_path, artifact, _arrays = valid_final_video_fixture(
                Path(directory), self.geometry_128
            )
            result = probe.validate_final_video_input_artifact(
                artifact_path,
                metadata_path,
                expected_attempt_identifier=artifact["attempt_identifier"],
                expected_checkpoint_identity=artifact["checkpoint_identity"],
                expected_geometry=self.geometry_128,
            )
            self.assertEqual(result["video_geometry"]["video_rgb_shape"], [30, 128, 128, 3])

    def test_256_video_artifact_input_validates_under_256_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path, metadata_path, artifact, _arrays = valid_final_video_fixture(
                Path(directory), self.geometry_256
            )
            result = probe.validate_final_video_input_artifact(
                artifact_path,
                metadata_path,
                expected_attempt_identifier=artifact["attempt_identifier"],
                expected_checkpoint_identity=artifact["checkpoint_identity"],
                expected_geometry=self.geometry_256,
            )
            self.assertEqual(result["video_shape"], [1, 24, 9, 16, 16])

    def test_128_video_artifact_is_rejected_under_256_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path, metadata_path, artifact, _arrays = valid_final_video_fixture(
                Path(directory), self.geometry_128
            )
            with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
                probe.validate_final_video_input_artifact(
                    artifact_path,
                    metadata_path,
                    expected_attempt_identifier=artifact["attempt_identifier"],
                    expected_checkpoint_identity=artifact["checkpoint_identity"],
                    expected_geometry=self.geometry_256,
                )

    def test_256_video_artifact_is_rejected_under_128_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_path, metadata_path, artifact, _arrays = valid_final_video_fixture(
                Path(directory), self.geometry_256
            )
            with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
                probe.validate_final_video_input_artifact(
                    artifact_path,
                    metadata_path,
                    expected_attempt_identifier=artifact["attempt_identifier"],
                    expected_checkpoint_identity=artifact["checkpoint_identity"],
                    expected_geometry=self.geometry_128,
                )

    def test_audio_contracts_remain_unchanged(self):
        self.assertEqual(probe.AUDIO_NATIVE_SHAPE, (2, 32, 50))
        self.assertEqual(probe.AUDIO_RAW_SHAPE, (2, 1, 40000))
        self.assertEqual(probe.AUDIO_WAVEFORM_SHAPE, (2, 40000))
        self.assertEqual(probe.AUDIO_SAMPLE_RATE, 32000)
        self.assertEqual(probe.AUDIO_DURATION_SECONDS, 1.25)

    def test_frozen_media_contract_remains_128_only(self):
        self.assertEqual(probe.VIDEO_FRAME_COUNT, 30)
        self.assertEqual(probe.VIDEO_FRAME_WIDTH, 128)
        self.assertEqual(probe.VIDEO_FRAME_HEIGHT, 128)
        self.assertEqual(probe.VIDEO_RAW_SHAPE, (1, 3, 30, 128, 128))
        self.assertEqual(probe.VIDEO_RGB_SHAPE, (30, 128, 128, 3))
        self.assertIn("30 frames at 128x128", inspect.getsource(probe.validate_video_frame_manifest))

    def test_256_real_full_run_remains_blocked_before_attempt_creation(self):
        args = probe.build_parser().parse_args(
            [
                "run-derived-full-schedule",
                "--checkpoint-root",
                "/nonexistent/checkpoint",
                "--derived-transformer",
                "/nonexistent/transformer",
                "--output-root",
                "/private/tmp/slice3b3a-no-output",
                "--prompt",
                probe.LOCKED_PROMPT,
                "--seed",
                str(probe.CANONICAL_SEED),
                "--video-size",
                "256",
                "--active-memory-tolerance-bytes",
                "0",
            ]
        )
        result = probe.run_command(args)
        self.assertEqual(result, 1)

    def test_128_real_full_run_gate_remains_permitted(self):
        self.assertEqual(probe.validate_full_run_video_size(128), 128)
        args = probe.build_parser().parse_args(
            [
                "run-derived-full-schedule",
                "--checkpoint-root",
                "checkpoint",
                "--derived-transformer",
                "transformer",
                "--output-root",
                "output",
                "--prompt",
                probe.LOCKED_PROMPT,
                "--seed",
                str(probe.CANONICAL_SEED),
                "--active-memory-tolerance-bytes",
                "0",
            ]
        )
        self.assertEqual(args.video_size, 128)

    def test_parent_import_remains_mlx_free(self):
        code = (
            "import importlib.util, sys; "
            f"spec=importlib.util.spec_from_file_location('probe_parent_import', {str(SCRIPT)!r}); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={"PATH": str(Path(sys.executable).parent), "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

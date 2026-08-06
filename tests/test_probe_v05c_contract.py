"""MLX-free behavioral contracts for the v0.5c prompted production-step probe."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v05c_prompted_production_step",
    ROOT / "scripts" / "probe_v05c_prompted_production_step.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def exact_worker_receipt(role: str, *, derived: bool = False) -> dict:
    arrays = {
        "text_conditioning": np.arange(12, dtype=np.float32).reshape(1, 3, 4),
        "token_ids": np.array([[1, 2, 3]], dtype=np.int32),
        "token_presence_mask": np.ones((1, 3), dtype=np.int32),
        "initial_video_native": np.arange(1 * 24 * 9 * 8 * 8, dtype=np.float32).reshape(1, 24, 9, 8, 8),
        "initial_audio_native": np.arange(2 * 32 * 50, dtype=np.float32).reshape(2, 32, 50),
        "packed_position_ids": np.zeros((247, 3), dtype=np.float32),
        "packed_token_tags": np.zeros(247, dtype=np.int32),
        "packed_video_indices": np.arange(144, dtype=np.int32),
        "packed_audio_indices": np.arange(100, dtype=np.int32),
        "packed_text_indices": np.arange(3, dtype=np.int32),
        "timestep_table": np.array([0.0], dtype=np.float32),
        "timestep_indices": np.zeros(247, dtype=np.int32),
        "packed_video_prediction": np.ones((1, 144, 96), dtype=np.float32),
        "packed_audio_prediction": np.ones((1, 100, 32), dtype=np.float32),
        "updated_packed_video": np.full((1, 144, 96), 2, dtype=np.float32),
        "updated_packed_audio": np.full((1, 100, 32), 2, dtype=np.float32),
        "updated_native_video": np.full((1, 24, 9, 8, 8), 2, dtype=np.float32),
        "updated_native_audio": np.full((2, 32, 50), 2, dtype=np.float32),
    }
    prompt = probe.prompt_receipt(probe.LOCKED_PROMPT, 3, arrays["token_ids"])
    return {
        "status": "success",
        "worker_identity": role,
        "transformer_calls": 1,
        "scheduler_updates": 1,
        "completed_steps": 1,
        "cache_acquisitions": 1 if derived else 0,
        "cache_releases": 1 if derived else 0,
        "conditioning": {
            "prompt": prompt,
            "token_ids": arrays["token_ids"].tolist(),
            "token_presence_mask": arrays["token_presence_mask"].tolist(),
            "shape": [1, 3, 4],
            "dtype": "bfloat16",
            "fingerprint": probe.array_fingerprint(arrays["text_conditioning"], logical_dtype="bfloat16"),
        },
        "transformer_release": {"passed": True},
        "residency": {},
        "sidecar_state": {},
        "subprocess": {"exit_status": 0, "log_path": "/tmp/worker.log"},
        "arrays": arrays,
    }


class ProbeV05CContractTests(unittest.TestCase):
    def test_locked_prompt_is_exact_and_utf8_receipts_are_stable(self):
        receipt = probe.prompt_receipt(probe.LOCKED_PROMPT)
        self.assertEqual(receipt["utf8_byte_count"], 482)
        self.assertEqual(receipt["sha256"], "c7d57d0bf61aa78dfe79d3267c13fc74b91bc397e09f1d73c35d12f4179dd00a")
        self.assertEqual(receipt["text"], probe.LOCKED_PROMPT)
        with self.assertRaisesRegex(ValueError, "locked"):
            probe.validate_locked_prompt(probe.LOCKED_PROMPT + " extra")

    def test_prompt_token_receipt_preserves_runtime_token_ids(self):
        receipt = probe.prompt_receipt(probe.LOCKED_PROMPT, 3, np.array([[11, 12, 13]], dtype=np.int32))
        self.assertEqual(receipt["token_count"], 3)
        self.assertEqual(receipt["token_ids"], [[11, 12, 13]])
        self.assertIsNone(receipt["negative_prompt"])
        self.assertFalse(receipt["image_conditioning"])

    def test_only_seed_zero_is_accepted(self):
        probe.validate_seed(0)
        with self.assertRaisesRegex(ValueError, "only seed 0"):
            probe.validate_seed(1)

    def test_locked_v05b_geometry_is_unchanged(self):
        geometry = probe.canonical_geometry_contract()
        self.assertEqual(tuple(geometry["video_native_latent_shape"]), (1, 24, 9, 8, 8))
        self.assertEqual(tuple(geometry["audio_native_latent_shape"]), (2, 32, 50))
        self.assertEqual(geometry["target_audio_rows"], 100)
        self.assertEqual(geometry["target_video_rows"], 144)
        self.assertEqual(geometry["target_rows"], 244)
        self.assertEqual(geometry["video_rgb_shape"], [30, 128, 128, 3])
        self.assertEqual(geometry["audio_waveform_shape"], [2, 40000])

    def test_complete_sequence_includes_runtime_text_rows(self):
        ranges = probe.derive_row_ranges(17)
        self.assertEqual(ranges["text"], [0, 17])
        self.assertEqual(ranges["target_audio"], [17, 117])
        self.assertEqual(ranges["target_video"], [117, 261])
        self.assertEqual(ranges["total_rows"], 261)

    def test_exact_row_order_and_ranges_are_locked(self):
        contract = {
            "row_order": "[text | target-audio | target-video]",
            "row_ranges": probe.derive_row_ranges(3),
            "attention_mask": None,
            "padding_rows": 0,
            "position_ids_shape": [247, 3],
            "token_tags_shape": [247],
            "timestep_indices_shape": [247],
        }
        probe.validate_packed_contract(contract)
        contract["row_order"] = "[audio | video]"
        with self.assertRaisesRegex(ValueError, "row order"):
            probe.validate_packed_contract(contract)

    def test_packed_contract_rejects_padding_and_attention_mask(self):
        base = {
            "row_order": "[text | target-audio | target-video]",
            "row_ranges": probe.derive_row_ranges(3),
            "attention_mask": None,
            "padding_rows": 0,
            "position_ids_shape": [247, 3],
            "token_tags_shape": [247],
            "timestep_indices_shape": [247],
        }
        for field, value in (("attention_mask", []), ("padding_rows", 1)):
            broken = dict(base, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                probe.validate_packed_contract(broken)

    def test_source_contracts_record_real_scheduler_and_encoder_policies(self):
        contracts = probe.source_contracts()
        self.assertEqual(contracts["scheduler"]["video_shift"], 12.0)
        self.assertEqual(contracts["scheduler"]["audio_shift"], 3.0)
        self.assertEqual(contracts["scheduler"]["schedule_points"], 2)
        self.assertEqual(contracts["scheduler"]["euler_transitions"], 1)
        self.assertEqual(contracts["conditioning"]["attention_mask_policy"], "create_attention_mask(hidden_states, None)")
        self.assertIsNone(contracts["packing"]["attention_mask"])

    def test_all_required_source_files_are_named(self):
        for required in ("text_encoder.py", "packing.py", "pipeline.py", "dit.py", "adaln.py", "streamed_adaln.py", "scheduler.py", "geometry.py", "video_vae.py", "audio_vae.py", "load.py"):
            self.assertTrue(any(path.endswith(required) for path in probe.SOURCE_INSPECTION_FILES))

    def test_worker_counts_allow_exactly_one_transition_only(self):
        probe.validate_one_step_counts({"transformer_calls": 1, "scheduler_updates": 1, "completed_steps": 1, "cache_acquisitions": 0, "cache_releases": 0})
        probe.validate_one_step_counts({"transformer_calls": 1, "scheduler_updates": 1, "completed_steps": 1, "cache_acquisitions": 1, "cache_releases": 1}, derived=True)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            probe.validate_one_step_counts({"transformer_calls": 2, "scheduler_updates": 1, "completed_steps": 1, "cache_acquisitions": 0, "cache_releases": 0})

    def test_deterministic_fingerprint_is_shape_and_logical_dtype_bound(self):
        value = np.zeros((2, 3), dtype=np.float32)
        self.assertNotEqual(
            probe.array_fingerprint(value, logical_dtype="float32"),
            probe.array_fingerprint(value.reshape(3, 2), logical_dtype="float32"),
        )
        self.assertNotEqual(
            probe.array_fingerprint(value, logical_dtype="float32"),
            probe.array_fingerprint(value, logical_dtype="bfloat16"),
        )

    def test_exact_gate_requires_shape_dtype_and_values(self):
        passing = probe.exact_gate(np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32), left_dtype="int32", right_dtype="int32")
        self.assertTrue(passing["exact_equality"])
        wrong_dtype = probe.exact_gate(np.array([1, 2], dtype=np.int32), np.array([1, 2], dtype=np.int32), left_dtype="int32", right_dtype="float32")
        self.assertFalse(wrong_dtype["exact_equality"])
        wrong_value = probe.exact_gate(np.array([1, 2], dtype=np.int32), np.array([1, 3], dtype=np.int32), left_dtype="int32", right_dtype="int32")
        self.assertFalse(wrong_value["exact_equality"])

    def test_process_isolation_is_explicit_in_base_report(self):
        args = type("Args", (), {
            "prompt": probe.LOCKED_PROMPT,
            "checkpoint_root": "/tmp/checkpoint",
            "resident_transformer": "/tmp/resident",
            "derived_transformer": "/tmp/derived",
            "output_root": "/tmp/v05c",
            "seed": 0,
            "active_memory_tolerance_bytes": 1048576,
            "overwrite": False,
            "verbose": False,
        })()
        report = probe._base_report(args, {"root": "/tmp/v05c", "frames": "/tmp/v05c/frames", "audio_wav": "/tmp/v05c/a.wav", "report": "/tmp/v05c/r.json"}, Path("/tmp/checkpoint"), Path("/tmp/resident"), Path("/tmp/derived"))
        self.assertTrue(report["process_isolation"]["resident_worker_separate_process"])
        self.assertFalse(report["process_isolation"]["resident_and_derived_transformers_coexisted"])
        self.assertEqual(report["process_isolation"]["max_simultaneous_transformer_workers"], 1)

    def test_compare_worker_artifacts_proves_exact_prediction_and_native_parity(self):
        resident = exact_worker_receipt("resident")
        derived = exact_worker_receipt("derived", derived=True)
        parity = probe.compare_worker_artifacts(
            {"prompt": resident["conditioning"]["prompt"], "conditioning": resident["conditioning"]},
            resident,
            derived,
            resident["arrays"],
            derived["arrays"],
            {key: resident["arrays"][key] for key in ("packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices")},
            {key: derived["arrays"][key] for key in ("packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices")},
        )
        self.assertTrue(parity["packed_video_prediction"]["exact_equality"])
        self.assertTrue(parity["native_audio_latent"]["exact_equality"])
        self.assertTrue(parity["all_exact_gates"])
        derived["arrays"]["packed_audio_prediction"][0, 0, 0] += 1
        broken = probe.compare_worker_artifacts(
            {"prompt": resident["conditioning"]["prompt"], "conditioning": resident["conditioning"]},
            resident,
            derived,
            resident["arrays"],
            derived["arrays"],
            {key: resident["arrays"][key] for key in ("packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices")},
            {key: derived["arrays"][key] for key in ("packed_position_ids", "packed_token_tags", "packed_video_indices", "packed_audio_indices", "packed_text_indices", "timestep_table", "timestep_indices")},
        )
        self.assertFalse(broken["all_exact_gates"])

    def test_output_namespace_refuses_nonempty_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v0.5c"
            root.mkdir()
            (root / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                probe.ensure_output_namespace(root, False)

    def test_overwrite_removes_only_known_v05c_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v0.5c"
            (root / "frames").mkdir(parents=True)
            (root / "frames" / "frame_00000.png").write_bytes(b"known")
            (root / "keep.bin").write_bytes(b"keep")
            (root / "prompted-step-audio.wav").write_bytes(b"known")
            paths = probe.ensure_output_namespace(root, True)
            self.assertFalse((root / "frames" / "frame_00000.png").exists())
            self.assertTrue((root / "keep.bin").is_file())
            self.assertFalse((root / "prompted-step-audio.wav").exists())

    def test_cli_help_works_from_tmp_without_mlxd_import(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        completed = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "probe_v05c_prompted_production_step.py"), "--help"],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("run-prompted-production-step", completed.stdout)
        self.assertNotIn("--steps", completed.stdout)
        self.assertNotIn("--num-inference-steps", completed.stdout)

    def test_direct_script_import_from_tmp_is_mlxfree(self):
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        code = (
            "import runpy, sys\n"
            "from pathlib import Path\n"
            "runpy.run_path(str(Path(sys.argv[1]).resolve()), run_name='v05c_import')\n"
            "print('mlx' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code, str(ROOT / "scripts" / "probe_v05c_prompted_production_step.py")],
            cwd="/tmp",
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "False")

    def test_failure_suppression_after_conditioning_failure(self):
        suppression = probe._suppression(["preflight", "conditioning-worker"])
        self.assertTrue(suppression["resident_worker_suppressed"])
        self.assertTrue(suppression["derived_worker_suppressed"])
        self.assertTrue(suppression["parity_suppressed"])
        self.assertTrue(suppression["video_decoder_suppressed"])
        self.assertTrue(suppression["audio_decoder_suppressed"])

    def test_failure_suppression_after_parity_failure(self):
        suppression = probe._suppression(["preflight", "conditioning-worker", "resident-worker", "derived-worker", "exact-parity-gates"])
        self.assertFalse(suppression["parity_suppressed"])
        self.assertTrue(suppression["video_decoder_suppressed"])
        self.assertTrue(suppression["audio_decoder_suppressed"])

    def test_success_and_failure_report_schemas_are_strict(self):
        args = type("Args", (), {
            "prompt": probe.LOCKED_PROMPT,
            "checkpoint_root": "/tmp/checkpoint",
            "resident_transformer": "/tmp/resident",
            "derived_transformer": "/tmp/derived",
            "output_root": "/tmp/v05c",
            "seed": 0,
            "active_memory_tolerance_bytes": 1048576,
            "overwrite": False,
            "verbose": False,
        })()
        paths = {"root": "/tmp/v05c", "frames": "/tmp/v05c/frames", "audio_wav": "/tmp/v05c/a.wav", "report": "/tmp/v05c/r.json"}
        report = probe._base_report(args, paths, Path("/tmp/checkpoint"), Path("/tmp/resident"), Path("/tmp/derived"))
        report["status"] = "success"
        report["exact_parity_gates"] = {"all_exact_gates": True}
        report["conditioning_memory_release"] = {"passed": True}
        report["resident_memory_release"] = {"passed": True}
        report["derived_memory_release"] = {"passed": True}
        report["video_media"] = {"frame_count": 30}
        report["audio_media"] = {"sample_count": 40000}
        probe.validate_report(report)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe.validate_report({**report, "extra": True})
        failed = probe._failure_report(report, "test", "parent", RuntimeError("original"))
        probe.validate_report(failed)
        self.assertIsNone(failed["failure"]["cleanup_error"])

    def test_generation_exclusions_forbid_quality_and_mux_claims(self):
        self.assertFalse(probe.GENERATION_EXCLUSIONS["second_denoising_step"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["quality_judgment"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["mp4_muxing"])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["image_conditioning"])

    def test_wav_metadata_reader_records_exact_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(2)
                handle.setsampwidth(2)
                handle.setframerate(32000)
                handle.writeframes(b"\x00\x00" * 2 * 40000)
            metadata = probe._wav_metadata(path)
            self.assertEqual(metadata["channels"], 2)
            self.assertEqual(metadata["sample_width_bytes"], 2)
            self.assertEqual(metadata["sample_rate"], 32000)
            self.assertEqual(metadata["sample_count"], 40000)
            self.assertEqual(metadata["duration_seconds"], 1.25)
            self.assertEqual(len(metadata["sha256"]), 64)

    def test_no_mp4_path_is_created_by_output_namespace(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_output_namespace(Path(directory) / "v0.5c", False)
            self.assertNotIn("mp4", " ".join(paths.values()))


if __name__ == "__main__":
    unittest.main()

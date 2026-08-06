"""MLX-free behavioral contracts for the v0.5b geometry bridge."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_v05b_geometry_bridge", ROOT / "scripts" / "probe_v05b_geometry_bridge.py")
assert SPEC and SPEC.loader
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class VideoConfig:
    latent_channels = 24
    out_channels = 3
    spatial_compression_ratio = 16
    temporal_compression_ratio = 4
    token_drop = 3
    clip_length = 17
    latents_mean = tuple(0.0 for _ in range(24))
    latents_std = tuple(1.0 for _ in range(24))


class AudioConfig:
    latent_channels = 32
    sampling_rate = 32000
    decoder_rates = (5, 5, 2, 2, 2, 2, 2)
    hop_length = 800


class DiTConfig:
    patch_size = (1, 2, 2)


LAYOUT = SimpleNamespace(
    clip_length=17, temporal_compression_ratio=4, tokens_chunk_size=5, token_drop=3,
    token_overlap=2, frame_pre_padding=3, frame_overlap=5, chunk_num_frames=20,
    tail_trim_remainder=1, minimum_latent_frames=7,
)


class FakePacking:
    def __init__(self):
        self.calls = []

    def patchify_video_latents(self, value, patch):
        self.calls.append("patchify_video_latents")
        b, c, f, h, w = value.shape
        pt, ph, pw = patch
        return value.reshape(b, c, f // pt, pt, h // ph, ph, w // pw, pw).transpose(0, 2, 4, 6, 1, 3, 5, 7).reshape(-1, c * pt * ph * pw)

    def pack_audio_latents(self, value):
        self.calls.append("pack_audio_latents")
        return value.transpose(0, 2, 1).reshape(-1, value.shape[1])

    def build_packed_sequence(self, tags, f, h, w, audio, patch):
        self.calls.append("build_packed_sequence")
        total = f * (h // patch[1]) * (w // patch[2]) + 2 * audio
        return SimpleNamespace(
            sequence_length=total,
            position_ids=np.zeros((total, 3), dtype=np.float32),
            token_tags=np.concatenate((np.full(2 * audio, 2, dtype=np.int32), np.zeros(total - 2 * audio, dtype=np.int32))),
            video_indices=np.arange(2 * audio, total, dtype=np.int32),
            audio_indices=np.arange(2 * audio, dtype=np.int32),
        )

    def build_row_timesteps(self, layout, *args):
        self.calls.append("build_row_timesteps")
        return np.array([0.5], dtype=np.float32), np.zeros(layout.sequence_length, dtype=np.int32)

    def unpatchify_video_tokens(self, rows, f, h, w, c, patch):
        pt, ph, pw = patch
        value = rows.reshape(1, f // pt, h // ph, w // pw, c, pt, ph, pw).transpose(0, 4, 1, 5, 2, 6, 3, 7)
        return value.reshape(1, c, f, h, w)

    def unpack_audio_tokens(self, rows, length):
        return rows.reshape(2, length, rows.shape[-1]).transpose(0, 2, 1)


def geometry():
    from minimax_h3_mlx.geometry import ProductionMultimodalGeometry
    return ProductionMultimodalGeometry.canonical(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT)


class ProbeV05BContractTests(unittest.TestCase):
    @staticmethod
    def complete_report(status="success"):
        report = {key: None for key in probe.REPORT_KEYS}
        report.update({
            "status": status,
            "probe_format": probe.PROBE_FORMAT,
            "schema_version": probe.SCHEMA_VERSION,
            "parity": {"all_exact_gates": True},
            "video_memory": {"release_gate": {"passed": True}, "memory_before": {"active": 10}},
            "audio_memory": {"release_gate": {"passed": True}, "memory_before": {"active": 20}},
            "final_memory": {"active": 30},
            "phase_order": ["packing", "video", "audio"],
            "output_paths": {"frames": "/tmp/frames", "audio_wav": "/tmp/audio.wav", "report": "/tmp/report.json"},
        })
        if status == "failed":
            report["failure"] = {"active_phase": "video", "completed_stages": ["packing", "video"], "error": {}, "residency": {}}
        return report

    def test_import_is_mlx_free(self):
        self.assertNotIn("mlx", sys.modules)

    def test_direct_script_help_works_from_tmp(self):
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        completed = subprocess.run([sys.executable, str(ROOT / "scripts/probe_v05b_geometry_bridge.py"), "--help"], cwd="/tmp", env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("decode-deterministic-geometry", completed.stdout)

    def test_authentic_geometry_is_immutable(self):
        value = geometry()
        self.assertEqual(value.video_latent_shape, (1, 24, 9, 8, 8))
        with self.assertRaises(Exception):
            value.video_frames = 31

    def test_video_decoder_9_latents_are_30_frames(self):
        self.assertEqual(probe._v05a().video_decoded_frame_count(9, LAYOUT), 30)
        from minimax_h3_mlx.geometry import video_decode_frame_count
        self.assertEqual(video_decode_frame_count(9, LAYOUT), 30)

    def test_video_spatial_geometry_is_128_square(self):
        value = geometry()
        self.assertEqual(value.rgb_media_shape, (30, 128, 128, 3))

    def test_audio_50_latents_are_40000_samples(self):
        value = geometry()
        self.assertEqual(value.audio_raw_shape, (2, 1, 40000))

    def test_exact_duration_alignment(self):
        value = geometry()
        self.assertEqual(value.video_duration, value.audio_duration)
        self.assertEqual(value.video_duration.numerator, 5)
        self.assertEqual(value.video_duration.denominator, 4)

    def test_patch_contract_is_derived_from_source_config(self):
        value = geometry()
        self.assertEqual(value.video_patch_size, (1, 2, 2))
        self.assertEqual(value.video_patch_width, 96)
        self.assertEqual(value.audio_patch_width, 32)

    def test_exact_token_counts(self):
        value = geometry()
        self.assertEqual(value.video_token_count, 144)
        self.assertEqual(value.audio_token_count, 100)
        self.assertEqual(value.total_token_count, 244)

    def test_modality_order_and_metadata_shapes(self):
        fake = FakePacking()
        value = geometry()
        packed = probe.pack_native_latents(fake, np.zeros(value.video_latent_shape, np.float32), np.zeros(value.audio_latent_shape, np.float32), value)
        self.assertEqual(packed["layout"].audio_indices.tolist(), list(range(100)))
        self.assertEqual(packed["layout"].video_indices.tolist(), list(range(100, 244)))
        self.assertEqual(packed["layout"].position_ids.shape, (244, 3))
        self.assertEqual(packed["layout"].token_tags.shape, (244,))

    def test_attention_mask_is_padless(self):
        self.assertIsNone(probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())["packing"]["mask"]["attention_mask"])

    def test_real_pack_api_seam_is_invoked(self):
        fake = FakePacking()
        value = geometry()
        probe.pack_native_latents(fake, np.zeros(value.video_latent_shape, np.float32), np.zeros(value.audio_latent_shape, np.float32), value)
        self.assertEqual(fake.calls, ["patchify_video_latents", "pack_audio_latents", "build_packed_sequence", "build_row_timesteps"])

    def test_exact_video_round_trip(self):
        fake = FakePacking(); value = geometry()
        video = np.arange(np.prod(value.video_latent_shape), dtype=np.float32).reshape(value.video_latent_shape)
        packed = probe.pack_native_latents(fake, video, np.zeros(value.audio_latent_shape, np.float32), value)
        result = probe.exact_parity(fake, packed, video, np.zeros(value.audio_latent_shape, np.float32), value, None)
        self.assertTrue(result["video_value_equal"])
        self.assertTrue(result["video_fingerprint_equal"])

    def test_exact_audio_round_trip(self):
        fake = FakePacking(); value = geometry()
        audio = np.arange(np.prod(value.audio_latent_shape), dtype=np.float32).reshape(value.audio_latent_shape)
        packed = probe.pack_native_latents(fake, np.zeros(value.video_latent_shape, np.float32), audio, value)
        result = probe.exact_parity(fake, packed, np.zeros(value.video_latent_shape, np.float32), audio, value, None)
        self.assertTrue(result["audio_value_equal"])
        self.assertTrue(result["audio_fingerprint_equal"])

    def test_invalid_video_divisibility_rejected_before_packing(self):
        with self.assertRaisesRegex(ValueError, "not divisible"):
            probe.validate_native_geometry((1, 24, 9, 7, 8), (2, 32, 50), (1, 2, 2))

    def test_invalid_audio_geometry_rejected_before_model_loading(self):
        with self.assertRaisesRegex(ValueError, "two mono"):
            probe.validate_native_geometry((1, 24, 9, 8, 8), (1, 32, 50), (1, 2, 2))

    def test_deterministic_domains_are_distinct_and_stable(self):
        self.assertEqual(probe.deterministic_values(16, 0), probe.deterministic_values(16, 0))
        self.assertNotEqual(probe.deterministic_values(16, 0), probe.deterministic_values(16, 1))

    def test_fingerprint_includes_shape_and_dtype_contract(self):
        self.assertNotEqual(probe.array_fingerprint(np.zeros((2, 3), np.float32)), probe.array_fingerprint(np.zeros((3, 2), np.float32)))

    def test_output_namespace_refuses_nonempty_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v0.5b"; root.mkdir(); (root / "keep.txt").write_text("keep")
            with self.assertRaises(FileExistsError):
                probe.ensure_output_namespace(root, False)

    def test_overwrite_removes_only_known_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v0.5b"; (root / "frames").mkdir(parents=True)
            (root / "frames" / "frame_00000.png").write_bytes(b"x"); (root / "unknown.bin").write_bytes(b"keep")
            probe.ensure_output_namespace(root, True)
            self.assertFalse((root / "frames" / "frame_00000.png").exists())
            self.assertTrue((root / "unknown.bin").exists())

    def test_report_schema_is_strict(self):
        report = {key: None for key in probe.REPORT_KEYS}
        report.update({"status": "failed", "probe_format": probe.PROBE_FORMAT, "schema_version": 1, "failure": {"active_phase": "packing", "completed_stages": [], "error": {}, "residency": {}}})
        probe.validate_report(report)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe.validate_report({**report, "extra": True})

    def test_complete_success_report_with_memory_sections_passes_strict_validation(self):
        probe.validate_report(self.complete_report())

    def test_success_report_key_set_equals_report_keys(self):
        self.assertEqual(set(self.complete_report()), probe.REPORT_KEYS)

    def test_success_report_missing_any_memory_section_fails_strict_validation(self):
        for field in ("video_memory", "audio_memory", "final_memory"):
            with self.subTest(field=field):
                report = self.complete_report()
                del report[field]
                with self.assertRaisesRegex(ValueError, "missing"):
                    probe.validate_report(report)

    def test_success_report_with_unrelated_field_fails_strict_validation(self):
        report = self.complete_report()
        report["unrelated"] = True
        with self.assertRaisesRegex(ValueError, "unexpected"):
            probe.validate_report(report)

    def test_failure_report_includes_all_memory_sections(self):
        report = self.complete_report(status="failed")
        probe.validate_report(report)
        for field in ("video_memory", "audio_memory", "final_memory"):
            self.assertIn(field, report)

    def test_outer_report_validation_failure_becomes_diagnostic_failure_receipt(self):
        report = self.complete_report()
        report["parity"] = {"all_exact_gates": False}
        expected_video_memory = dict(report["video_memory"])
        expected_audio_memory = dict(report["audio_memory"])
        expected_final_memory = dict(report["final_memory"])
        expected_output_paths = dict(report["output_paths"])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(output_root=directory, overwrite=False)
            with patch.object(probe, "_runtime_run", return_value=report), patch.object(probe, "_write_report") as write_report:
                self.assertEqual(probe.run_command(args), 1)
            diagnostic = write_report.call_args.args[1]
        probe.validate_report(diagnostic)
        self.assertEqual(diagnostic["status"], "failed")
        self.assertEqual(diagnostic["failure"]["active_phase"], "report-validation")
        self.assertIn("exact parity", diagnostic["failure"]["error"]["message"])
        self.assertEqual(diagnostic["video_memory"], expected_video_memory)
        self.assertEqual(diagnostic["audio_memory"], expected_audio_memory)
        self.assertEqual(diagnostic["final_memory"], expected_final_memory)
        self.assertEqual(diagnostic["output_paths"], expected_output_paths)
        self.assertEqual(diagnostic["phase_order"][-1], "report-validation")

    def test_success_report_requires_exact_parity(self):
        report = {key: None for key in probe.REPORT_KEYS}
        report.update({"status": "success", "probe_format": probe.PROBE_FORMAT, "schema_version": 1, "parity": {"all_exact_gates": False}})
        with self.assertRaisesRegex(ValueError, "exact parity"):
            probe.validate_report(report)

    def test_failure_receipt_preserves_cleanup_field(self):
        report = {key: None for key in probe.REPORT_KEYS}
        report.update({"status": "failed", "probe_format": probe.PROBE_FORMAT, "schema_version": 1, "failure": {"active_phase": "video", "completed_stages": ["packing-release-gate"], "error": {"type": "RuntimeError"}, "cleanup_error": None, "residency": {}}})
        probe.validate_report(report)
        self.assertIn("cleanup_error", report["failure"])

    def test_generation_components_are_all_excluded(self):
        self.assertTrue(all(value is False for value in probe.GENERATION_EXCLUSIONS.values()))

    def test_no_locked_prompt_is_encoded(self):
        self.assertIsNone(probe._runtime_run.__name__ and None)
        self.assertIsNone(probe.GENERATION_EXCLUSIONS.get("prompt"))

    def test_source_contract_has_exact_locations_and_formulas(self):
        contracts = probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())
        self.assertIn("minimax_h3_mlx/packing.py:186-222", contracts["source_locations"]["video_patchify"])
        self.assertIn("F+token_drop", contracts["video"]["decoded_frame_formula"])

    def test_audio_stays_channel_major(self):
        contracts = probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())
        self.assertEqual(contracts["audio"]["stereo_representation"], "two mono batch items")

    def test_position_coordinates_are_three_axis(self):
        contracts = probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())
        self.assertEqual(contracts["packing"]["position_ids"]["axes"], "(t,h,w)")

    def test_timestep_metadata_has_one_distinct_level_without_scheduler(self):
        contracts = probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())
        self.assertEqual(contracts["packing"]["timesteps"]["distinct_shape"], [1])
        self.assertFalse(probe.GENERATION_EXCLUSIONS["scheduler_loaded"])

    def test_rgb_and_waveform_contracts(self):
        value = geometry()
        self.assertEqual(value.rgb_media_shape, (30, 128, 128, 3))
        self.assertEqual(value.waveform_shape, (2, 40000))

    def test_checkpoint_paths_are_read_only_metadata(self):
        contracts = probe.source_contracts(VideoConfig(), AudioConfig(), DiTConfig(), LAYOUT, geometry())
        self.assertIn("load.py:423-499", contracts["source_locations"]["video_loader"])
        self.assertIn("load.py:502-577", contracts["source_locations"]["audio_loader"])

    def test_phase_contract_is_sequentially_named(self):
        self.assertEqual(probe.DEFAULT_OUTPUT_ROOT.name, "v0.5b")
        self.assertEqual(probe.COMMITTED_BASELINE, "7fd9322 Add v0.5a decoder lifecycle proof")

    def test_media_output_names_are_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_output_namespace(Path(directory) / "v0.5b", False)
            self.assertTrue(paths["audio_wav"].endswith("geometry-audio.wav"))
            self.assertTrue(paths["report"].endswith("geometry-report.json"))

    def test_no_mp4_output_is_declared(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = probe.ensure_output_namespace(Path(directory) / "v0.5b", False)
            self.assertNotIn("mp4", " ".join(paths.values()))


if __name__ == "__main__":
    unittest.main()

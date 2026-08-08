"""MLX-free Slice 3B2 core 128x128/256x256 geometry contracts."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v05d_derived_full_schedule.py"
SPEC = importlib.util.spec_from_file_location("probe_v05e_slice3b2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def conditioning_fixture(root: Path, geometry: dict[str, object]):
    shapes = probe._conditioning_array_shapes(geometry)
    ranges = geometry["row_ranges"]
    text_rows = int(geometry["text_rows"])
    arrays = {
        "text_conditioning": np.zeros(probe.EXPECTED_CONDITIONING_SHAPE, dtype=np.float32),
        "token_ids": np.arange(text_rows, dtype=np.int32).reshape(1, -1),
        "token_presence_mask": np.ones((1, text_rows), dtype=np.int32),
        "text_token_tags": np.zeros((text_rows,), dtype=np.int32),
        "initial_video_native": np.zeros(tuple(shapes["initial_video_native"]), dtype=np.float32),
        "initial_audio_native": np.zeros(tuple(shapes["initial_audio_native"]), dtype=np.float32),
        "packed_position_ids": np.zeros(tuple(shapes["packed_position_ids"]), dtype=np.float32),
        "packed_token_tags": np.concatenate(
            [
                np.zeros((text_rows,), dtype=np.int32),
                np.full(int(geometry["target_audio_rows"]), 2, dtype=np.int32),
                np.full(int(geometry["target_video_rows"]), 0, dtype=np.int32),
            ]
        ),
        "packed_video_indices": np.arange(*ranges["target_video"], dtype=np.int32),
        "packed_audio_indices": np.arange(*ranges["target_audio"], dtype=np.int32),
        "packed_text_indices": np.arange(*ranges["text"], dtype=np.int32),
    }
    path = root / f"conditioning-{geometry['video_width']}.npz"
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
            "fingerprint": probe.array_fingerprint(
                arrays["text_conditioning"], logical_dtype=probe.EXPECTED_CONDITIONING_DTYPE
            ),
        },
        "deterministic_inputs": probe.deterministic_input_receipt(
            arrays["initial_video_native"], arrays["initial_audio_native"]
        ),
        "packing": {
            **probe.packed_contract(geometry=geometry),
            "position_ids_shape": list(arrays["packed_position_ids"].shape),
            "token_tags_shape": list(arrays["packed_token_tags"].shape),
            "video_indices_shape": list(arrays["packed_video_indices"].shape),
            "audio_indices_shape": list(arrays["packed_audio_indices"].shape),
            "text_indices_shape": list(arrays["packed_text_indices"].shape),
        },
        "geometry": geometry,
        "conditioning_release": {"passed": True},
    }
    receipt["conditioning_artifact"] = probe.conditioning_artifact_binding(path, arrays, geometry=geometry)
    return path, arrays, receipt


def final_artifact_fixture(geometry: dict[str, object]):
    video = np.zeros(tuple(geometry["video_native_shape"]), dtype=np.float32)
    audio = np.zeros(tuple(geometry["audio_native_shape"]), dtype=np.float32)
    artifact = {
        "artifact_identity": "minimax-h3-mlx-v05d-final-native-latent",
        "schema_version": probe.FINAL_ARTIFACT_SCHEMA_VERSION,
        "attempt_identifier": "slice3b2-test-attempt",
        "native_video": probe.shape_dtype(video, logical_dtype="bfloat16")
        | {"fingerprint": probe.array_fingerprint(video, logical_dtype="bfloat16")},
        "native_audio": probe.shape_dtype(audio, logical_dtype="bfloat16")
        | {"fingerprint": probe.array_fingerprint(audio, logical_dtype="bfloat16")},
        "packed_final_state_fingerprint": "slice3b2-packed-state",
        "schedule_contract": probe.build_full_schedule().receipt(),
        "completed_transition_count": probe.EXPECTED_DENOISING_TRANSITIONS,
        "transformer_forward_count": probe.EXPECTED_TRANSFORMER_FORWARDS,
        "scheduler_update_counts": {
            "video": probe.EXPECTED_DENOISING_TRANSITIONS,
            "audio": probe.EXPECTED_DENOISING_TRANSITIONS,
        },
        "streamed_adaln_lifecycle": probe.expected_lifecycle_totals(),
        "worker_identity": "derived",
        "worker_exit_receipt": {},
        "transformer_release_receipt": {},
        "final_active_memory": None,
        "final_allocator_cache": None,
        "final_allocator_cache_zero": False,
        "final_artifact_npz_sha256": None,
        "metadata_sha256": None,
        "memory_receipt": {},
        "git_identity": {},
        "checkpoint_identity": {"checkpoint": "slice3b2-test"},
        "geometry": geometry,
    }
    artifact["streamed_adaln_lifecycle"]["sessions"] = probe.expected_lifecycle_sessions()
    return artifact, {"final_video_native": video, "final_audio_native": audio}


def attribution_stats() -> dict[str, object]:
    blocks = []
    for index in range(probe.EXPECTED_BLOCK_COUNT):
        blocks.append(
            {
                "block_index": index,
                "sidecar_filename": f"block-{index:03d}.safetensors",
                "sidecar_io_and_reconstruction_seconds": 0.001,
                "projection_compute_seconds": 0.002,
                "materialization_evaluation_seconds": 0.003,
                "cache_entry_assembly_bookkeeping_seconds": 0.003,
                "release_purge_seconds": 0.002,
                "total_block_cache_construction_seconds": 0.02,
            }
        )
    return {
        "per_block": blocks,
        "elapsed_total_seconds": 1.0,
        "elapsed_shared_timestep_embedding_seconds": 0.01,
        "elapsed_cache_finalize_materialization_seconds": 0.01,
    }


class Slice3B2GeometryContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.geometry_128 = probe.canonical_geometry_contract(128)
        cls.geometry_256 = probe.canonical_geometry_contract(256)

    def test_default_selector_preserves_128_geometry(self):
        geometry = probe.resolve_core_geometry()
        self.assertEqual((geometry.video_width, geometry.video_height), (128, 128))
        self.assertEqual(geometry.text_token_count, probe.EXPECTED_TOKEN_COUNT)

    def test_default_native_latent_shape_is_unchanged(self):
        self.assertEqual(self.geometry_128["video_native_shape"], [1, 24, 9, 8, 8])

    def test_default_video_rows_are_unchanged(self):
        self.assertEqual(self.geometry_128["video_rows"], 144)
        self.assertEqual(self.geometry_128["target_video_rows"], probe.EXPECTED_TARGET_VIDEO_ROWS)

    def test_default_total_packed_rows_are_unchanged(self):
        self.assertEqual(self.geometry_128["total_packed_rows"], probe.EXPECTED_TOTAL_ROWS)
        self.assertEqual(self.geometry_128["row_ranges"]["total_rows"], probe.EXPECTED_TOTAL_ROWS)

    def test_256_selector_resolves_square_resolution(self):
        self.assertEqual(self.geometry_256["resolution"], [256, 256])
        self.assertEqual(self.geometry_256["video_width"], 256)
        self.assertEqual(self.geometry_256["video_height"], 256)

    def test_256_native_latent_shape_is_spatially_parameterized(self):
        self.assertEqual(self.geometry_256["video_native_shape"], [1, 24, 9, 16, 16])

    def test_256_video_rows_are_576(self):
        self.assertEqual(self.geometry_256["video_rows"], 576)
        self.assertEqual(self.geometry_256["video_indices_shape"], [576])

    def test_256_total_packed_rows_are_779(self):
        self.assertEqual(self.geometry_256["total_packed_rows"], 779)
        self.assertEqual(self.geometry_256["position_ids_shape"], [779, 3])

    def test_256_row_ranges_retain_locked_text_audio_video_order(self):
        self.assertEqual(
            self.geometry_256["row_ranges"],
            {
                "text": [0, 103],
                "target_audio": [103, 203],
                "target_video": [203, 779],
                "text_rows": 103,
                "target_audio_rows": 100,
                "target_video_rows": 576,
                "total_rows": 779,
            },
        )

    def test_256_index_and_tag_shapes_are_geometry_bound(self):
        self.assertEqual(self.geometry_256["token_tags_shape"], [779])
        self.assertEqual(self.geometry_256["video_indices_shape"], [576])
        self.assertEqual(self.geometry_256["audio_indices_shape"], [100])
        self.assertEqual(self.geometry_256["text_indices_shape"], [103])

    def test_audio_geometry_remains_unchanged_at_256(self):
        self.assertEqual(self.geometry_128["audio_native_shape"], [2, 32, 50])
        self.assertEqual(self.geometry_256["audio_native_shape"], [2, 32, 50])
        self.assertEqual(self.geometry_256["audio_rows"], 100)
        self.assertEqual(self.geometry_256["audio_sample_rate"], 32000)
        self.assertEqual(self.geometry_256["audio_samples_per_channel"], 40000)

    def test_rng_receipt_retains_order_and_records_cross_geometry_limitation(self):
        self.assertEqual(self.geometry_256["rng_draw_order"], ["video_native", "audio_native"])
        self.assertIn("not claimed bit-identical", self.geometry_256["rng_methodology_limitation"])
        receipt = probe.deterministic_input_receipt(
            np.zeros((1, 24, 9, 16, 16), dtype=np.float32),
            np.zeros((2, 32, 50), dtype=np.float32),
        )
        self.assertEqual(receipt["draw_order"], ["video_native", "audio_native"])
        self.assertEqual(receipt["methodology_limitation"], probe.RNG_METHODOLOGY_LIMITATION)

    def test_proof_selector_rejects_unsupported_resolution(self):
        with self.assertRaisesRegex(ValueError, "one of 128 or 256"):
            probe.validate_video_size(192)

    def test_core_geometry_rejects_rectangular_dimensions(self):
        with self.assertRaisesRegex(ValueError, "must be square"):
            probe.ProductionMultimodalGeometry.canonical(
                probe._proof_video_config(),
                probe._proof_audio_config(),
                probe._proof_dit_config(),
                probe._proof_video_layout(),
                width=256,
                height=128,
                text_token_count=probe.EXPECTED_TOKEN_COUNT,
            )

    def test_256_packed_contract_round_trips_with_dynamic_rows(self):
        packed = probe.packed_contract(geometry=self.geometry_256)
        probe.validate_packed_contract(packed, geometry=self.geometry_256)
        self.assertEqual(packed["total_rows"], 779)
        self.assertEqual(packed["target_video_rows"], 576)

    def test_conditioning_artifact_round_trip_binds_256_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arrays, receipt = conditioning_fixture(Path(directory), self.geometry_256)
            probe.validate_conditioning_receipt(receipt, expected_geometry=self.geometry_256)
            loaded = probe.validate_conditioning_artifact_binding(
                receipt, path, arrays=arrays, geometry=self.geometry_256
            )
            self.assertEqual(loaded["initial_video_native"].shape, (1, 24, 9, 16, 16))
            self.assertEqual(receipt["geometry"]["total_packed_rows"], 779)

    def test_conditioning_artifact_rejects_128_under_256_expectation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arrays, receipt = conditioning_fixture(Path(directory), self.geometry_128)
            with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
                probe.validate_conditioning_artifact_binding(
                    receipt, path, arrays=arrays, geometry=self.geometry_256
                )

    def test_conditioning_artifact_rejects_256_under_128_expectation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, arrays, receipt = conditioning_fixture(Path(directory), self.geometry_256)
            with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
                probe.validate_conditioning_artifact_binding(
                    receipt, path, arrays=arrays, geometry=self.geometry_128
                )

    def test_final_artifact_round_trip_binds_256_geometry(self):
        artifact, arrays = final_artifact_fixture(self.geometry_256)
        probe.validate_final_artifact(
            artifact, arrays=arrays, require_worker_termination=False, geometry=self.geometry_256
        )
        self.assertEqual(artifact["native_video"]["shape"], [1, 24, 9, 16, 16])

    def test_final_artifact_rejects_cross_geometry_native_shape(self):
        artifact, arrays = final_artifact_fixture(self.geometry_256)
        with self.assertRaisesRegex(ValueError, "does not match the selected core contract"):
            probe.validate_final_artifact(
                artifact, arrays=arrays, require_worker_termination=False, geometry=self.geometry_128
            )

    def test_full_run_256_selector_is_open_and_parser_default_is_128(self):
        self.assertEqual(probe.validate_full_run_video_size(256), 256)

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

    def test_full_run_128_selector_is_open(self):
        self.assertEqual(probe.validate_full_run_video_size(128), 128)

    def test_worker_parsers_carry_the_same_proof_only_selector(self):
        conditioning = probe._conditioning_worker_parser(
            [
                "--checkpoint-root",
                "checkpoint",
                "--derived-transformer",
                "transformer",
                "--prompt",
                probe.LOCKED_PROMPT,
                "--seed",
                "0",
                "--video-size",
                "256",
                "--artifact",
                "artifact",
                "--receipt",
                "receipt",
                "--tolerance",
                "0",
            ]
        )
        derived = probe._derived_worker_parser(
            [
                "--checkpoint-root",
                "checkpoint",
                "--derived-transformer",
                "transformer",
                "--attempt-identifier",
                "attempt",
                "--video-size",
                "256",
                "--conditioning-artifact",
                "conditioning",
                "--conditioning-receipt",
                "receipt",
                "--final-artifact",
                "final",
                "--final-artifact-metadata",
                "metadata",
                "--event-file",
                "events",
                "--receipt",
                "receipt",
                "--tolerance",
                "0",
            ]
        )
        self.assertEqual(conditioning.video_size, 256)
        self.assertEqual(derived.video_size, 256)

    def test_help_exposes_only_128_and_256_video_sizes(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "run-derived-full-schedule", "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--video-size {128,256}", result.stdout)

    def test_attribution_telemetry_schema_remains_valid_for_full_schedule(self):
        sessions = [
            probe.build_cache_session_attribution(
                attribution_stats(), session_index=index, wall_clock_seconds=1.0
            )
            for index in range(probe.EXPECTED_DENOISING_TRANSITIONS)
        ]
        aggregate = probe.build_cache_attribution_aggregate(sessions)
        probe.validate_cache_attribution(aggregate)
        self.assertEqual(aggregate["session_count"], 15)
        self.assertEqual(aggregate["block_count"], 750)
        self.assertEqual(set(aggregate["category_percentages_of_cache_wall_time"]), set(probe.ATTRIBUTION_REPORT_CATEGORY_FIELDS))

    def test_import_path_is_mlx_free(self):
        code = (
            "import importlib.util, sys; "
            f"spec=importlib.util.spec_from_file_location('probe_import_check', {str(SCRIPT)!r}); "
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

    def test_script_has_no_top_level_mlx_import(self):
        tree = ast.parse(SCRIPT.read_text())
        top_level_mlx_imports = []
        for node in tree.body:
            candidates = []
            if isinstance(node, ast.Import):
                candidates.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                candidates.append(node.module or "")
            top_level_mlx_imports.extend(name for name in candidates if name == "mlx" or name.startswith("mlx."))
        self.assertEqual(top_level_mlx_imports, [])

    def test_geometry_contract_rejects_missing_identity_field(self):
        incomplete = dict(self.geometry_256)
        del incomplete["total_packed_rows"]
        with self.assertRaisesRegex(ValueError, "missing fields"):
            probe.validate_geometry_contract(incomplete)

    def test_legacy_default_packed_contract_stays_byte_for_byte_structurally_compatible(self):
        self.assertEqual(
            probe.packed_contract(),
            {
                "row_order": "[text | target audio | target video]",
                "row_ranges": {
                    "text": [0, 103],
                    "target_audio": [103, 203],
                    "target_video": [203, 347],
                    "text_rows": 103,
                    "target_audio_rows": 100,
                    "target_video_rows": 144,
                    "total_rows": 347,
                },
                "text_rows": 103,
                "target_audio_rows": 100,
                "target_video_rows": 144,
                "total_rows": 347,
                "feature_widths": {"text": 5120, "target_audio": 32, "target_video": 96},
                "attention_mask": None,
                "padding_rows": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()

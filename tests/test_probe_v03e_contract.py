"""MLX-free contract checks for the v0.3e external probe."""

from __future__ import annotations

import importlib.util
import inspect
import json
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe_v03e_adaln", ROOT / "scripts" / "probe_v03e_adaln.py")
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeV03EContractTests(unittest.TestCase):
    def test_parser_exposes_all_four_expected_subcommands(self) -> None:
        parser = probe.build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(
            set(subparsers.choices),
            {"create-reference", "verify-resident-release", "compare-sidecars", "forward"},
        )

    def test_selected_block_contract_is_canonical(self) -> None:
        self.assertEqual(probe.SELECTED_BLOCKS, (0, 5, 49))
        probe.validate_selected_blocks([0, 5, 49])
        self.assertEqual(len(probe.artifact_tensor_keys()), 18)
        self.assertEqual(
            probe.artifact_tensor_keys()[:6],
            [f"block_000_{name}" for name in probe.TENSOR_ORDER],
        )

    def test_duplicate_missing_and_reordered_blocks_fail_loudly(self) -> None:
        for blocks, message in (
            ([0, 0, 49], "duplicate"),
            ([0, 5], "missing"),
            ([49, 5, 0], "reordered"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                probe.validate_selected_blocks(blocks)

    def test_exact_equality_true_passes(self) -> None:
        self.assertIsNone(probe.validate_exact_equality(True))

    def test_exact_equality_false_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact_equality=False"):
            probe.validate_exact_equality(False)

    def test_reported_false_result_cannot_reach_success_path(self) -> None:
        reported = {"exact_equality": False}
        with self.assertRaisesRegex(ValueError, "streamed AdaLN parity failed"):
            probe.validate_exact_equality(reported["exact_equality"])

        source = inspect.getsource(probe.cmd_compare_sidecars)
        diagnostic_offset = source.index('print(f"exact_equality={exact}"')
        validation_offset = source.index("validate_exact_equality(exact)")
        success_marker_offset = source.index('print("STREAMED ADALN PARITY REPORT GENERATED"')
        success_return_offset = source.index("return 0", validation_offset)
        self.assertLess(diagnostic_offset, validation_offset)
        self.assertLess(validation_offset, success_marker_offset)
        self.assertLess(validation_offset, success_return_offset)

    def test_artifact_metadata_path_requires_safetensors(self) -> None:
        self.assertEqual(
            probe.artifact_metadata_path(Path("reference.safetensors")),
            Path("reference.json"),
        )
        with self.assertRaisesRegex(ValueError, "safetensors"):
            probe.artifact_metadata_path(Path("reference.npy"))

    def test_checkpoint_fingerprint_metadata_fields_are_required(self) -> None:
        self.assertIn("reference_checkpoint_fingerprint", probe.REFERENCE_METADATA_REQUIRED_KEYS)
        self.assertIn("reference_checkpoint_fingerprint_method", probe.REFERENCE_METADATA_REQUIRED_KEYS)
        self.assertEqual(probe.ARTIFACT_SCHEMA_VERSION, 2)

    def test_checkpoint_fingerprint_binds_full_indexed_shard_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            index = {
                "weight_map": {
                    "tensor_a": "model-00002-of-00002.safetensors",
                    "tensor_b": "model-00001-of-00002.safetensors",
                }
            }
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
            original = probe.resident_checkpoint_fingerprint(checkpoint)
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"changed")
            self.assertNotEqual(original, probe.resident_checkpoint_fingerprint(checkpoint))

    def test_cache_stat_validation_rejects_each_count_and_lifecycle_contract(self) -> None:
        stats = SimpleNamespace(
            blocks_completed=3,
            sidecar_files_opened=3,
            unique_sidecar_files_opened=3,
            successful_payload_opens=3,
            completed_payload_releases=3,
            every_sidecar_released_before_next_opened=True,
            sidecar_overlap_observed=False,
            next_sidecar_opened_before_previous_release=False,
            dense_temporary_projection_created=False,
        )
        kwargs = {
            "cache_table_count": 3,
            "configured_block_count": 3,
            "stats": stats,
            "actual_sidecar_names": [f"block-{index:03d}.safetensors" for index in range(3)],
        }
        probe.validate_complete_cache_stats(**kwargs)

        for field in ("cache_table_count", "blocks_completed", "sidecar_files_opened", "unique_sidecar_files_opened", "successful_payload_opens", "completed_payload_releases"):
            if field == "cache_table_count":
                bad_kwargs = {**kwargs, field: 2}
            else:
                bad_stats = SimpleNamespace(**{**vars(stats), field: 2})
                bad_kwargs = {**kwargs, "stats": bad_stats}
            with self.assertRaisesRegex(ValueError, "complete cache contract violation"):
                probe.validate_complete_cache_stats(**bad_kwargs)

        for field, bad_value in (
            ("every_sidecar_released_before_next_opened", False),
            ("sidecar_overlap_observed", True),
            ("next_sidecar_opened_before_previous_release", True),
            ("dense_temporary_projection_created", True),
        ):
            bad_stats = SimpleNamespace(**{**vars(stats), field: bad_value})
            with self.assertRaisesRegex(ValueError, field):
                probe.validate_complete_cache_stats(**{**kwargs, "stats": bad_stats})

        with self.assertRaisesRegex(ValueError, "sidecar filename order"):
            probe.validate_complete_cache_stats(**{
                **kwargs,
                "actual_sidecar_names": list(reversed(kwargs["actual_sidecar_names"])),
            })

    def test_execution_observation_restores_patched_block_state(self) -> None:
        class Block:
            def __init__(self, index: int) -> None:
                self.index = index

            def __call__(self, value: int) -> int:
                return value + self.index

        original_blocks = [Block(0), Block(1)]
        dit = SimpleNamespace(blocks=original_blocks)
        with probe.observe_transformer_block_execution(dit) as observed:
            self.assertEqual([dit.blocks[i](10) for i in range(2)], [10, 11])
            self.assertEqual(observed, [0, 1])
        self.assertIs(dit.blocks, original_blocks)
        self.assertEqual([block.index for block in dit.blocks], [0, 1])

    def test_configured_block_count_is_not_an_observed_execution_receipt(self) -> None:
        receipt = probe.build_execution_receipt(3, [0, 1, 2])
        self.assertEqual(receipt["configured_transformer_block_count"], 3)
        self.assertEqual(receipt["observed_transformer_block_indices"], [0, 1, 2])
        self.assertEqual(receipt["observed_transformer_block_count"], 3)
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            probe.build_execution_receipt(3, [0, 1])


if __name__ == "__main__":
    unittest.main()

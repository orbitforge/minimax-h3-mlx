"""MLX-free contract tests for the v0.3f full transformer parity probe."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v03f_transformer_parity", ROOT / "scripts" / "probe_v03f_transformer_parity.py"
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class ProbeV03FContractTests(unittest.TestCase):
    def _metadata_fixture(self):
        inventory = {key: {"shape": [1], "dtype": "float32"} for key in probe.ARTIFACT_KEYS}
        checksums = {
            "artifact_sha256": "artifact",
            "reference_checkpoint_fingerprint": "reference-fingerprint",
            "reference_config_sha256": "reference-config",
            "derived_config_sha256": "derived-config",
            "derived_conversion_manifest_sha256": "conversion-manifest",
            "derived_sidecar_manifest_sha256": "sidecar-manifest",
        }
        original = Path("original").resolve()
        derived = Path("derived").resolve()
        metadata = {
            "artifact_format": probe.ARTIFACT_FORMAT,
            "artifact_schema_version": probe.ARTIFACT_SCHEMA_VERSION,
            "artifact_file_format": "safetensors",
            "reference_checkpoint": str(original),
            "derived_checkpoint": str(derived),
            "reference_checkpoint_fingerprint": checksums["reference_checkpoint_fingerprint"],
            "reference_checkpoint_fingerprint_method": probe.REFERENCE_CHECKPOINT_FINGERPRINT_METHOD,
            "reference_config_sha256": checksums["reference_config_sha256"],
            "derived_config_sha256": checksums["derived_config_sha256"],
            "derived_conversion_manifest_sha256": checksums["derived_conversion_manifest_sha256"],
            "derived_sidecar_manifest_sha256": checksums["derived_sidecar_manifest_sha256"],
            "artifact_sha256": checksums["artifact_sha256"],
            "deterministic_input_method": probe.DETERMINISTIC_INPUT_METHOD,
            "timestep_values": [probe.TIMESTEP_VALUE],
            "timestep_dtype": probe.CANONICAL_TIMESTEP_DTYPE,
            "configured_resident_block_count": probe.EXPECTED_BLOCK_COUNT,
            "observed_resident_block_count": probe.EXPECTED_BLOCK_COUNT,
            "observed_resident_block_indices": list(range(probe.EXPECTED_BLOCK_COUNT)),
            "tensor_keys": list(probe.ARTIFACT_KEYS),
            "tensor_inventory": {key: dict(value) for key, value in inventory.items()},
            "packed_layout": {"canonical": True},
        }
        return metadata, checksums, inventory, {"canonical": True}, original, derived

    def _validate_metadata(self, metadata, checksums, inventory, packed_layout, original, derived):
        probe.validate_reference_metadata(
            metadata,
            original=original,
            derived=derived,
            expected_checksums=checksums,
            expected_tensor_inventory=inventory,
            expected_packed_layout=packed_layout,
            expected_observed_block_indices=list(range(probe.EXPECTED_BLOCK_COUNT)),
        )

    def test_parser_exposes_exactly_create_reference_and_compare_derived(self) -> None:
        parser = probe.build_parser()
        subparsers = next(action for action in parser._actions if action.dest == "command")
        self.assertEqual(set(subparsers.choices), {"create-reference", "compare-derived"})

    def test_artifact_schema_and_complete_required_metadata_key_set_are_canonical(self) -> None:
        expected = {
            "artifact_format", "artifact_schema_version", "artifact_file_format",
            "reference_checkpoint", "derived_checkpoint", "reference_checkpoint_fingerprint",
            "reference_checkpoint_fingerprint_method", "reference_config_sha256",
            "derived_config_sha256", "derived_conversion_manifest_sha256",
            "derived_sidecar_manifest_sha256", "artifact_sha256", "deterministic_input_method",
            "timestep_values", "timestep_dtype", "configured_resident_block_count",
            "observed_resident_block_count", "observed_resident_block_indices", "tensor_keys",
            "tensor_inventory", "packed_layout",
        }
        self.assertEqual(probe.ARTIFACT_SCHEMA_VERSION, 2)
        self.assertEqual(probe.REFERENCE_METADATA_REQUIRED_KEYS, frozenset(expected))
        self.assertNotIn("derived_adaln_manifest_sha256", probe.REFERENCE_METADATA_REQUIRED_KEYS)
        self.assertEqual(probe.artifact_metadata_path(Path("reference.safetensors")), Path("reference.json"))
        with self.assertRaisesRegex(ValueError, "safetensors"):
            probe.artifact_metadata_path(Path("reference.npy"))

    def test_full_resident_fingerprint_changes_when_indexed_shard_content_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            index = {"weight_map": {"a": "model-00002-of-00002.safetensors",
                                     "b": "model-00001-of-00002.safetensors"}}
            (checkpoint / "model.safetensors.index.json").write_text(json.dumps(index))
            (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
            original = probe.resident_checkpoint_fingerprint(checkpoint)
            (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"changed")
            self.assertNotEqual(original, probe.resident_checkpoint_fingerprint(checkpoint))

    def test_non_string_and_unsafe_shard_names_fail_clearly(self) -> None:
        for weight_map, message in (
            ({"a": 7}, "must all be strings"),
            ({"a": "nested/model.safetensors"}, "unsafe shard names"),
            ({"a": ""}, "unsafe shard names"),
        ):
            with self.subTest(weight_map=weight_map):
                with self.assertRaisesRegex(ValueError, message):
                    probe.validate_resident_shard_names(weight_map)

    def test_deterministic_input_specification_is_stable_bounded_and_strictly_nonzero(self) -> None:
        first = probe.deterministic_input_specification()
        second = probe.deterministic_input_specification()
        self.assertEqual(first, second)
        self.assertEqual(
            probe.deterministic_input_pattern_parameters(2),
            (first["modulus"], first["offset"], first["scale_base"] + 2 * first["scale_step"]),
        )
        values = probe.deterministic_input_values(256, 2)
        self.assertTrue(values)
        self.assertTrue(all(value > 0 for value in values))
        self.assertLessEqual(max(values), first["modulus"] * (first["scale_base"] + 2 * first["scale_step"]))
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            probe.deterministic_input_values(0, 0)
        with self.assertRaisesRegex(ValueError, "salt"):
            probe.deterministic_input_values(1, 3)

    def test_timestep_dtype_name_normalization_is_prefix_only(self) -> None:
        for dtype in ("float32", "mlx.core.float32"):
            with self.subTest(dtype=dtype):
                probe.validate_canonical_timestep_dtype(dtype)
                self.assertEqual(probe.normalize_dtype_name(dtype), "float32")
        for dtype in ("mlx.core.bfloat16", "float64"):
            with self.subTest(dtype=dtype):
                with self.assertRaisesRegex(ValueError, "not canonical"):
                    probe.validate_canonical_timestep_dtype(dtype)
                self.assertNotEqual(probe.normalize_dtype_name(dtype), "float32")

    def test_artifact_keys_equal_complete_canonical_ordered_tuple(self) -> None:
        self.assertEqual(probe.ARTIFACT_KEYS, (
            "video_input", "audio_input", "text_input", "timestep", "timestep_indices",
            "token_tags", "position_ids", "video_indices", "audio_indices", "text_indices",
            "resident_video_output", "resident_audio_output",
        ))

    def test_missing_artifact_keys_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "key order or membership"):
            probe.validate_artifact_tensor_keys(list(probe.ARTIFACT_KEYS[:-1]))

    def test_unexpected_artifact_keys_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "key order or membership"):
            probe.validate_artifact_tensor_keys(list(probe.ARTIFACT_KEYS) + ["unexpected"])

    def test_missing_tensor_inventory_key_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        del metadata["tensor_inventory"][probe.ARTIFACT_KEYS[-1]]
        with self.assertRaisesRegex(ValueError, "missing"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_unexpected_tensor_inventory_key_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["tensor_inventory"]["unexpected"] = {"shape": [1], "dtype": "float32"}
        with self.assertRaisesRegex(ValueError, "unexpected"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_non_dictionary_tensor_inventory_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["tensor_inventory"] = []
        with self.assertRaisesRegex(ValueError, "must be a dictionary"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_duplicated_metadata_tensor_keys_fail(self) -> None:
        keys = list(probe.ARTIFACT_KEYS)
        keys[-1] = keys[-2]
        with self.assertRaisesRegex(ValueError, "key order or membership"):
            probe.validate_artifact_tensor_keys(keys)

    def test_reordered_metadata_tensor_keys_fail(self) -> None:
        keys = list(probe.ARTIFACT_KEYS)
        keys[0], keys[1] = keys[1], keys[0]
        with self.assertRaisesRegex(ValueError, "key order or membership"):
            probe.validate_artifact_tensor_keys(keys)

    def test_canonical_metadata_passes(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_sort_keys_json_round_trip_preserves_valid_metadata_contract(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        round_tripped = json.loads(json.dumps(metadata, sort_keys=True))
        self._validate_metadata(round_tripped, checksums, inventory, layout, original, derived)

    def test_shape_mismatch_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["tensor_inventory"]["video_input"] = {"shape": [2], "dtype": "float32"}
        with self.assertRaisesRegex(ValueError, "tensor inventory"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_dtype_mismatch_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["tensor_inventory"]["video_input"] = {"shape": [1], "dtype": "bfloat16"}
        with self.assertRaisesRegex(ValueError, "tensor inventory"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_metadata_fingerprint_method_mismatch_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["reference_checkpoint_fingerprint_method"] = "wrong"
        with self.assertRaisesRegex(ValueError, "fingerprint_method"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_metadata_timestep_value_mismatch_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["timestep_values"] = [0.25]
        with self.assertRaisesRegex(ValueError, "timestep_values"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_metadata_timestep_dtype_mismatch_fails(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        metadata["timestep_dtype"] = "bfloat16"
        with self.assertRaisesRegex(ValueError, "timestep_dtype"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_metadata_missing_and_unexpected_keys_fail(self) -> None:
        metadata, checksums, inventory, layout, original, derived = self._metadata_fixture()
        del metadata["artifact_format"]
        metadata["diagnostic_only"] = True
        with self.assertRaisesRegex(ValueError, "key contract"):
            self._validate_metadata(metadata, checksums, inventory, layout, original, derived)

    def test_block_observation_records_calls_and_restores_original_list(self) -> None:
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

    def test_observed_block_delegates_adaln_projection_and_records_one_call(self) -> None:
        calls = []

        class Projection:
            def __call__(self, value: int) -> int:
                calls.append(value)
                return value + 1

        class Block:
            adaln_proj = Projection()

            def __call__(self, value: int) -> int:
                return value * 2

        observed = []
        wrapper = probe._ObservedTransformerBlock(7, Block(), observed)
        self.assertIs(wrapper.adaln_proj, wrapper.block.adaln_proj)
        self.assertEqual(wrapper.adaln_proj(4), 5)
        self.assertEqual(observed, [])
        self.assertEqual(wrapper(4), 8)
        self.assertEqual(observed, [7])
        self.assertEqual(calls, [4])

    def test_observed_block_unknown_attribute_uses_wrapped_attribute_error(self) -> None:
        class Block:
            def __call__(self) -> None:
                pass

        wrapper = probe._ObservedTransformerBlock(0, Block(), [])
        with self.assertRaises(AttributeError):
            _ = wrapper.missing_attribute

    def test_observed_block_restores_original_list_after_exception(self) -> None:
        class Block:
            def __call__(self) -> None:
                raise RuntimeError("forward failed")

        original_blocks = [Block()]
        dit = SimpleNamespace(blocks=original_blocks)
        with self.assertRaisesRegex(RuntimeError, "forward failed"):
            with probe.observe_transformer_block_execution(dit):
                dit.blocks[0]()
        self.assertIs(dit.blocks, original_blocks)

    def test_existing_artifact_paths_report_only_existing_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            existing = root / "transformer-reference.safetensors"
            existing.write_bytes(b"artifact")
            missing = root / "transformer-reference.json"
            self.assertEqual(probe.existing_artifact_paths(existing), [str(existing)])
            self.assertEqual(probe.existing_artifact_paths(missing), [])
            self.assertEqual(probe.existing_artifact_paths(existing, missing), [str(existing)])

    def test_detach_exception_removes_traceback_context_and_cause(self) -> None:
        cause = ValueError("cause")
        context = LookupError("context")
        original_exception = None
        try:
            try:
                raise cause
            except ValueError as caught_cause:
                try:
                    raise TypeError("model failure") from caught_cause
                except TypeError as caught:
                    original_exception = caught
                    caught.__context__ = context
                    detached = probe.detach_exception(caught)
        except Exception as unexpected:  # pragma: no cover - defensive test failure path
            self.fail(f"detaching should not raise: {unexpected}")
        self.assertIs(detached, original_exception)
        self.assertIsNone(detached.__traceback__)
        self.assertIsNone(detached.__context__)
        self.assertIsNone(detached.__cause__)

    def test_cleanup_precedes_reraising_original_failure_and_success_is_after_failure_gate(self) -> None:
        events = []
        failure = None
        try:
            raise LookupError("original message")
        except BaseException as exc:
            failure = probe.detach_exception(exc)
        events.append("cleanup")
        with self.assertRaisesRegex(LookupError, "original message") as raised:
            if failure is not None:
                raise failure.with_traceback(None)
            events.append("success")
        self.assertEqual(events, ["cleanup"])
        self.assertIs(type(raised.exception), LookupError)

    def test_command_cleanup_and_success_order_is_structurally_gated(self) -> None:
        for command in (probe.cmd_create_reference, probe.cmd_compare_derived):
            source = inspect.getsource(command)
            capture = source.index("except BaseException as exc")
            release = source.index("release_receipt = _release(mx)")
            failure_gate = source.index("if failure is not None:")
            success = source.index("return 0", failure_gate)
            self.assertLess(capture, release)
            self.assertLess(release, failure_gate)
            self.assertLess(failure_gate, success)

    def test_create_reference_releases_before_reraising_model_failure(self) -> None:
        fake_mlx = ModuleType("mlx.core")
        fake_mlx_package = ModuleType("mlx")
        fake_mlx_package.core = fake_mlx
        fake_load = ModuleType("minimax_h3_mlx.load")
        fake_load.load_dit = lambda *args, **kwargs: object()
        args = SimpleNamespace(
            original="/missing/original",
            derived="/missing/derived",
            artifact="/tmp/nonexistent-transformer-reference.safetensors",
            overwrite=False,
        )
        events = []

        def fake_begin(_mx):
            events.append("begin")
            return 0.0, {}

        def fake_release(_mx):
            events.append("release")
            return {"memory": {}, "allocator_cache_purge_available": True, "allocator_cache_purged": True}

        with mock.patch.dict(sys.modules, {"mlx": fake_mlx_package, "mlx.core": fake_mlx, "minimax_h3_mlx.load": fake_load}):
            with mock.patch.object(probe, "begin_phase", side_effect=fake_begin), \
                 mock.patch.object(probe, "_validate_transformer_config", side_effect=RuntimeError("model failed")), \
                 mock.patch.object(probe, "_release", side_effect=fake_release), \
                 mock.patch.object(probe, "emit_phase", side_effect=lambda *args, **kwargs: events.append("emit")), \
                 mock.patch("builtins.print") as printer:
                with self.assertRaisesRegex(RuntimeError, "model failed"):
                    probe.cmd_create_reference(args)
        self.assertLess(events.index("release"), events.index("emit"))
        self.assertNotIn("RESIDENT TRANSFORMER REFERENCE CREATED", " ".join(str(call) for call in printer.call_args_list))

    def test_compare_derived_releases_before_reraising_model_failure(self) -> None:
        fake_mlx = ModuleType("mlx.core")
        fake_mlx_package = ModuleType("mlx")
        fake_mlx_package.core = fake_mlx
        fake_config = ModuleType("minimax_h3_mlx.config")

        class FakeDiTConfig:
            @classmethod
            def from_json(cls, _path):
                return object()

        fake_config.DiTConfig = FakeDiTConfig
        fake_load = ModuleType("minimax_h3_mlx.load")
        fake_load.load_dit = lambda *args, **kwargs: object()
        args = SimpleNamespace(
            original="/missing/original",
            derived="/missing/derived",
            artifact="/tmp/nonexistent-transformer-reference.safetensors",
            report="/tmp/nonexistent-transformer-report.json",
        )
        events = []

        def fake_begin(_mx):
            events.append("begin")
            return 0.0, {}

        def fake_release(_mx):
            events.append("release")
            return {"memory": {}, "allocator_cache_purge_available": True, "allocator_cache_purged": True}

        with mock.patch.dict(sys.modules, {
            "mlx": fake_mlx_package,
            "mlx.core": fake_mlx,
            "minimax_h3_mlx.config": fake_config,
            "minimax_h3_mlx.load": fake_load,
        }):
            with mock.patch.object(probe, "begin_phase", side_effect=fake_begin), \
                 mock.patch.object(probe, "_load_and_validate_reference", side_effect=ValueError("reference failed")), \
                 mock.patch.object(probe, "_release", side_effect=fake_release), \
                 mock.patch.object(probe, "emit_phase", side_effect=lambda *args, **kwargs: events.append("emit")), \
                 mock.patch("builtins.print") as printer:
                with self.assertRaisesRegex(ValueError, "reference failed"):
                    probe.cmd_compare_derived(args)
        self.assertLess(events.index("release"), events.index("emit"))
        self.assertNotIn("TRANSFORMER PARITY PASSED", " ".join(str(call) for call in printer.call_args_list))

    def test_missing_block_observation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            probe.validate_observed_block_indices(2, [])

    def test_duplicate_block_observation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            probe.validate_observed_block_indices(2, [0, 0])

    def test_reordered_block_observation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            probe.validate_observed_block_indices(3, [0, 2, 1])

    def test_partial_block_observation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "observation mismatch"):
            probe.validate_observed_block_indices(3, [0, 1])

    def test_each_cache_count_contract_fails_independently(self) -> None:
        stats = SimpleNamespace(
            blocks_completed=50, sidecar_files_opened=50, unique_sidecar_files_opened=50,
            successful_payload_opens=50, completed_payload_releases=50,
            every_sidecar_released_before_next_opened=True, sidecar_overlap_observed=False,
            next_sidecar_opened_before_previous_release=False, dense_temporary_projection_created=False,
        )
        names = [f"block-{i:03d}.safetensors" for i in range(50)]
        for field in ("cache_table_count", "blocks_completed", "sidecar_files_opened",
                      "unique_sidecar_files_opened", "successful_payload_opens", "completed_payload_releases"):
            with self.subTest(field=field):
                kwargs = {"cache_table_count": 50}
                broken = stats
                if field == "cache_table_count":
                    kwargs["cache_table_count"] = 49
                else:
                    broken = SimpleNamespace(**{**vars(stats), field: 49})
                with self.assertRaisesRegex(ValueError, "complete cache contract violation"):
                    probe.validate_complete_cache_stats(
                        configured_block_count=50, stats=broken,
                        actual_sidecar_names=names, **kwargs,
                    )

    def test_sidecar_order_mismatch_fails(self) -> None:
        stats = SimpleNamespace(
            blocks_completed=2, sidecar_files_opened=2, unique_sidecar_files_opened=2,
            successful_payload_opens=2, completed_payload_releases=2,
            every_sidecar_released_before_next_opened=True, sidecar_overlap_observed=False,
            next_sidecar_opened_before_previous_release=False, dense_temporary_projection_created=False,
        )
        with self.assertRaisesRegex(ValueError, "sidecar filename order"):
            probe.validate_complete_cache_stats(
                cache_table_count=2, configured_block_count=2, stats=stats,
                actual_sidecar_names=["block-001.safetensors", "block-000.safetensors"],
            )

    def test_each_lifecycle_flag_fails_independently(self) -> None:
        base = {
            "blocks_completed": 2, "sidecar_files_opened": 2, "unique_sidecar_files_opened": 2,
            "successful_payload_opens": 2, "completed_payload_releases": 2,
            "every_sidecar_released_before_next_opened": True, "sidecar_overlap_observed": False,
            "next_sidecar_opened_before_previous_release": False, "dense_temporary_projection_created": False,
        }
        names = ["block-000.safetensors", "block-001.safetensors"]
        for field, value in (
            ("every_sidecar_released_before_next_opened", False),
            ("sidecar_overlap_observed", True),
            ("next_sidecar_opened_before_previous_release", True),
            ("dense_temporary_projection_created", True),
        ):
            with self.subTest(field=field):
                values = {**base, field: value}
                with self.assertRaisesRegex(ValueError, "complete cache contract violation"):
                    probe.validate_complete_cache_stats(
                        cache_table_count=2, configured_block_count=2,
                        stats=SimpleNamespace(**values), actual_sidecar_names=names,
                    )

    def test_combined_exact_parity_requires_both_video_and_audio(self) -> None:
        probe.validate_combined_exact_parity(True, True)

    def test_false_combined_parity_raises(self) -> None:
        for video, audio in ((False, True), (True, False), (False, False)):
            with self.subTest(video=video, audio=audio):
                with self.assertRaisesRegex(ValueError, "full transformer parity failed"):
                    probe.validate_combined_exact_parity(video, audio)

    def test_diagnostic_report_is_written_before_fail_loud_parity_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            probe.write_diagnostic_report(report_path, {"diagnostic": True})
            with self.assertRaisesRegex(ValueError, "full transformer parity failed"):
                probe.validate_parity_after_report(report_path, False, True)
            self.assertTrue(report_path.is_file())
            self.assertEqual(json.loads(report_path.read_text()), {"diagnostic": True})

    def test_final_success_message_requires_prior_parity_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "before parity validation"):
            probe.emit_parity_success_message(False)
        with mock.patch("builtins.print") as printer:
            probe.emit_parity_success_message(True)
        printer.assert_called_once_with("TRANSFORMER PARITY PASSED", flush=True)

    def test_derived_cleanup_explicitly_clears_combined_comparison_arrays(self) -> None:
        resident_combined, derived_combined = probe.clear_derived_comparison_arrays(object(), object())
        self.assertIsNone(resident_combined)
        self.assertIsNone(derived_combined)


if __name__ == "__main__":
    unittest.main()

"""Focused v0.3d tests; execute these only from the external MLX Terminal."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import mlx.nn as nn

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import CACHE_ONLY_CONSTRUCTION, MiniMaxH3DiT, RESIDENT_CONSTRUCTION, timestep_embedding
from minimax_h3_mlx.load import CheckpointFormatInfo
from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache


FORMAT = "minimax-h3-mlx-streamed-adaln-v1"


def tiny_config() -> DiTConfig:
    return DiTConfig(
        hidden_size=8,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=1,
        attention_head_dim=8,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 1, 1),
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=64,
        adaln_out_features=6 * 3 * 8,
        final_adaln_out_features=2 * 8,
        rope_inv_freq_len=1,
    )


class StreamedAdaLNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="streamed-adaln-"))
        self.config = tiny_config()
        self.dit = MiniMaxH3DiT(self.config, construction_mode=CACHE_ONLY_CONSTRUCTION)
        self.sidecar_arrays = self._synthetic_sidecar()
        self._write_metadata()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _synthetic_sidecar(self):
        linear = nn.Linear(self.config.time_embed_dim, self.config.adaln_out_features)
        quantized = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=8)
        return {
            "blocks.0.adaln_proj.linear.weight": quantized.weight,
            "blocks.0.adaln_proj.linear.scales": quantized.scales.astype(mx.bfloat16),
            "blocks.0.adaln_proj.linear.biases": quantized.biases.astype(mx.bfloat16),
            "blocks.0.adaln_proj.linear.bias": linear.bias.astype(mx.bfloat16),
        }

    def _write_metadata(self) -> None:
        (self.temp / "base").mkdir(parents=True)
        (self.temp / "adaln").mkdir()
        (self.temp / "conversion_manifest.json").write_text(json.dumps({
            "format_identifier": FORMAT,
            "schema_version": 1,
            "bounded": False,
            "verification_status": "verified",
            "selected_blocks": list(range(50)),
        }))
        (self.temp / "quant_config.json").write_text(json.dumps({
            "bits": 6,
            "group_size": 64,
            "quantize_adaln": True,
            "adaln_bits": 8,
        }))
        projection = {
            "quantization_bits": 8,
            "quantization_group_size": 64,
            "logical_input_features": 64,
            "logical_output_features": 144,
            "packed_weight_shape": [144, 16],
            "scales_shape": [144, 1],
            "quantization_biases_shape": [144, 1],
            "learned_bias_shape": [144],
        }
        blocks = {}
        for block in range(50):
            filename = f"block-{block:03d}.safetensors"
            if block < len(self.dit.blocks):
                (self.temp / "adaln" / filename).write_bytes(b"fixture")
            tensors = []
            for key, role, dtype, shape, quant_format, bits, group_size in (
                (f"blocks.{block}.adaln_proj.linear.bias", "learned_bias", "BF16", [144], "unquantized", None, None),
                (f"blocks.{block}.adaln_proj.linear.biases", "quantization_biases", "BF16", [144, 1], "affine", 8, 64),
                (f"blocks.{block}.adaln_proj.linear.scales", "scales", "BF16", [144, 1], "affine", 8, 64),
                (f"blocks.{block}.adaln_proj.linear.weight", "packed_weight", "U32", [144, 16], "affine", 8, 64),
            ):
                tensors.append({
                    "tensor_key": key,
                    "tensor_role": role,
                    "source_dtype": dtype,
                    "source_shape": shape,
                    "byte_count": 1,
                    "tensor_checksum": "fixture",
                    "quantization_format": quant_format,
                    "quantization_bits": bits,
                    "group_size": group_size,
                })
            blocks[str(block)] = {
                "block_index": block,
                "sidecar_filename": filename,
                "projection": projection,
                "tensors": tensors,
            }
        (self.temp / "adaln" / "manifest.json").write_text(json.dumps({
            "format_identifier": FORMAT,
            "schema_version": 1,
            "bounded": False,
            "blocks": blocks,
        }))
        self.dit.checkpoint_format_info = CheckpointFormatInfo(
            checkpoint_format="derived",
            derived_root=self.temp,
            base_root=self.temp / "base",
            conversion_manifest_path=self.temp / "conversion_manifest.json",
            adaln_manifest_path=self.temp / "adaln" / "manifest.json",
            construction_mode=CACHE_ONLY_CONSTRUCTION,
            base_shards=(),
        )

    def _loader(self, calls: list[str]):
        def load(path: str):
            calls.append(path)
            block = int(Path(path).stem.split("-")[1])
            return {
                key.replace("blocks.0.", f"blocks.{block}."): value
                for key, value in self.sidecar_arrays.items()
            }
        return load

    def test_resident_model_is_rejected(self) -> None:
        self.dit.construction_mode = RESIDENT_CONSTRUCTION
        with self.assertRaisesRegex(ValueError, "cache-only"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]))

    def test_missing_validated_format_info_is_rejected(self) -> None:
        del self.dit.checkpoint_format_info
        with self.assertRaisesRegex(ValueError, "validated checkpoint format"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]))

    def test_unsupported_schema_and_recipe_are_rejected(self) -> None:
        conversion = self.temp / "conversion_manifest.json"
        raw = json.loads(conversion.read_text())
        raw["schema_version"] = 99
        conversion.write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "schema"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]))
        raw["schema_version"] = 1
        conversion.write_text(json.dumps(raw))
        recipe = self.temp / "quant_config.json"
        quant = json.loads(recipe.read_text())
        quant["adaln_bits"] = 6
        recipe.write_text(json.dumps(quant))
        with self.assertRaisesRegex(ValueError, "bit width"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]))

    def test_timetable_rejects_empty_and_duplicate_values(self) -> None:
        for timetable in (mx.array([]), mx.array([0.0, 0.0])):
            with self.assertRaisesRegex(ValueError, "timetable"):
                build_streamed_modulation_cache(self.dit, timetable)

    def test_sidecars_are_opened_once_in_block_order_without_preload(self) -> None:
        calls: list[str] = []
        cache, stats = build_streamed_modulation_cache(
            self.dit,
            mx.array([0.0, 0.5]),
            sidecar_loader=self._loader(calls),
        )
        self.assertEqual([Path(path).name for path in calls], ["block-000.safetensors", "block-001.safetensors"])
        self.assertEqual(stats.sidecar_files_opened, 2)
        self.assertEqual(stats.unique_sidecar_files_opened, 2)
        self.assertTrue(stats.every_sidecar_released_before_next_opened)
        self.assertFalse(stats.dense_temporary_projection_created)
        self.assertEqual(len(cache.tables), 2)

    def test_lifecycle_events_alternate_and_openings_see_no_live_previous_payload(self) -> None:
        events: list[tuple[str, dict]] = []
        cache, stats = build_streamed_modulation_cache(
            self.dit,
            mx.array([0.0]),
            sidecar_loader=self._loader([]),
            telemetry=lambda event, details: events.append((event, dict(details))),
        )
        lifecycle = [event for event, _ in events if event in {"sidecar_opening", "sidecar_released"}]
        self.assertEqual(lifecycle, ["sidecar_opening", "sidecar_released"] * 2)
        openings = [details for event, details in events if event == "sidecar_opening"]
        self.assertEqual([details["block_index"] for details in openings], [0, 1])
        self.assertTrue(all(details["builder_payload_live"] is False for details in openings))
        self.assertEqual(stats.successful_payload_opens, 2)
        self.assertEqual(stats.completed_payload_releases, 2)
        self.assertFalse(stats.sidecar_overlap_observed)
        self.assertFalse(stats.next_sidecar_opened_before_previous_release)
        self.assertEqual(len(cache.tables), 2)

    def test_exact_structure_dtype_shape_and_validate_for(self) -> None:
        cache, _ = build_streamed_modulation_cache(
            self.dit, mx.array([0.0, 0.5]), sidecar_loader=self._loader([])
        )
        self.assertEqual(len(cache.tables), 2)
        self.assertTrue(all(len(table) == 6 for table in cache.tables))
        self.assertTrue(all(array.shape == (6, 8) for table in cache.tables for array in table))
        self.assertTrue(all(array.dtype == mx.bfloat16 for table in cache.tables for array in table))
        cache.validate_for(self.config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)

    def test_shared_timestep_embedding_is_computed_once(self) -> None:
        calls = 0
        original = self.dit.time_embedder

        class CountingEmbedder(nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def __call__(self, value):
                nonlocal calls
                calls += 1
                return self.wrapped(value)

        self.dit.time_embedder = CountingEmbedder(original)
        build_streamed_modulation_cache(self.dit, mx.array([0.0, 0.5]), sidecar_loader=self._loader([]))
        self.assertEqual(calls, 1)

    def test_wrong_payload_key_shape_dtype_and_midstream_failure_are_rejected(self) -> None:
        def extra(path):
            payload = self._loader([])(path)
            payload["extra"] = next(iter(payload.values()))
            return payload

        with self.assertRaisesRegex(ValueError, "exactly four"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]), sidecar_loader=extra)

        def wrong_dtype(path):
            payload = self._loader([])(path)
            key = "blocks.0.adaln_proj.linear.weight" if "000" in path else "blocks.1.adaln_proj.linear.weight"
            payload[key] = payload[key].astype(mx.int32)
            return payload

        with self.assertRaisesRegex(ValueError, "dtype"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]), sidecar_loader=wrong_dtype)

        def failing(path):
            if "001" in path:
                raise OSError("fixture failure")
            return self._loader([])(path)

        with self.assertRaisesRegex(ValueError, "block 1"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]), sidecar_loader=failing)

    def test_missing_and_wrong_block_payload_keys_are_rejected(self) -> None:
        def missing(path):
            payload = self._loader([])(path)
            payload.pop(next(iter(payload)))
            return payload

        with self.assertRaisesRegex(ValueError, "exactly four"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]), sidecar_loader=missing)

        def wrong_block(path):
            payload = self._loader([])(path)
            return {
                key.replace("blocks.0.", "blocks.1."): value
                for key, value in payload.items()
            }

        with self.assertRaisesRegex(ValueError, "exactly four"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]), sidecar_loader=wrong_block)

    def test_unsupported_group_size_is_rejected(self) -> None:
        recipe = self.temp / "quant_config.json"
        quant = json.loads(recipe.read_text())
        quant["group_size"] = 32
        recipe.write_text(json.dumps(quant))
        with self.assertRaisesRegex(ValueError, "group size"):
            build_streamed_modulation_cache(self.dit, mx.array([0.0]))

    def test_float16_storage_and_unknown_injected_dense_status(self) -> None:
        def executor(activation, weight, scales, biases, learned_bias):
            return mx.zeros((activation.shape[0], self.config.adaln_out_features), dtype=mx.bfloat16)

        cache, stats = build_streamed_modulation_cache(
            self.dit,
            mx.array([0.0]),
            dtype=mx.float16,
            sidecar_loader=self._loader([]),
            projection_executor=executor,
        )
        self.assertTrue(all(array.dtype == mx.float16 for array in cache.tables[0]))
        self.assertIsNone(stats.dense_temporary_projection_created)

    def test_failure_after_table_construction_clears_partial_cache_before_purge(self) -> None:
        events: list[tuple[str, dict]] = []
        purge_observations: list[tuple[str, dict]] = []

        def executor(activation, weight, scales, biases, learned_bias):
            return mx.full((1, self.config.adaln_out_features), float("nan"), dtype=mx.bfloat16)

        def purge():
            purge_observations.append((events[-1][0], dict(events[-1][1])))
            return True

        def telemetry(event, details):
            events.append((event, dict(details)))

        cache_result = None
        with self.assertRaisesRegex(ValueError, "block 0 sidecar"):
            cache_result = build_streamed_modulation_cache(
                self.dit,
                mx.array([0.0]),
                sidecar_loader=self._loader([]),
                projection_executor=executor,
                telemetry=telemetry,
                allocator_purge=purge,
            )
        self.assertIsNone(cache_result)
        cleanup = [details for event, details in events if event == "failure_cleanup_ready"]
        self.assertEqual(len(cleanup), 1)
        self.assertEqual(cleanup[0]["block_index"], 0)
        self.assertEqual(cleanup[0]["purge_kind"], "failure-cleanup")
        self.assertFalse(cleanup[0]["builder_payload_live"])
        self.assertFalse(cleanup[0]["current_table_live"])
        self.assertEqual(cleanup[0]["partial_table_count"], 0)
        self.assertFalse(cleanup[0]["shared_embedding_live"])
        self.assertFalse(cleanup[0]["adaln_activation_live"])
        self.assertFalse(cleanup[0]["remaining_intermediates_live"])
        self.assertEqual(cleanup[0]["successful_payload_opens"], 1)
        self.assertEqual(cleanup[0]["completed_payload_releases"], 1)
        self.assertFalse(cleanup[0]["sidecar_overlap_observed"])
        self.assertFalse(cleanup[0]["next_sidecar_opened_before_previous_release"])
        self.assertEqual(len(purge_observations), 1)
        self.assertEqual(purge_observations[0][0], "failure_cleanup_ready")
        self.assertFalse(purge_observations[0][1]["builder_payload_live"])
        self.assertFalse(purge_observations[0][1]["current_table_live"])
        self.assertEqual(purge_observations[0][1]["partial_table_count"], 0)
        self.assertFalse(purge_observations[0][1]["shared_embedding_live"])
        self.assertFalse(purge_observations[0][1]["adaln_activation_live"])
        self.assertFalse(purge_observations[0][1]["remaining_intermediates_live"])

    def test_no_dense_projection_and_projection_executor_seam(self) -> None:
        seen = []

        def executor(activation, weight, scales, biases, learned_bias):
            seen.append((activation.shape, weight.shape, scales.shape, biases.shape, learned_bias.shape))
            return mx.quantized_matmul(
                activation, weight, scales=scales, biases=biases,
                transpose=True, group_size=64, bits=8, mode="affine",
            ) + learned_bias

        build_streamed_modulation_cache(
            self.dit, mx.array([0.0]), sidecar_loader=self._loader([]), projection_executor=executor
        )
        self.assertEqual(seen, [((1, 64), (144, 16), (144, 1), (144, 1), (144,))] * 2)

    def test_tiny_synthetic_sidecar_matches_live_quantized_projection(self) -> None:
        resident = nn.QuantizedLinear(64, 144, group_size=64, bits=8)
        resident.weight = self.sidecar_arrays["blocks.0.adaln_proj.linear.weight"]
        resident.scales = self.sidecar_arrays["blocks.0.adaln_proj.linear.scales"]
        resident.biases = self.sidecar_arrays["blocks.0.adaln_proj.linear.biases"]
        resident.bias = self.sidecar_arrays["blocks.0.adaln_proj.linear.bias"]
        timesteps = mx.array([0.0, 0.5])
        temb = self.dit.time_embedder(timestep_embedding(timesteps, self.config.timestep_input_dim))
        expected = resident(nn.silu(temb).astype(resident.scales.dtype))
        cache, _ = build_streamed_modulation_cache(
            self.dit, timesteps, sidecar_loader=self._loader([])
        )
        got = mx.concatenate(cache.tables[0], axis=-1).reshape(2, 144)
        mx.eval(expected, got)
        self.assertEqual(float(mx.max(mx.abs(expected.astype(mx.bfloat16) - got)).item()), 0.0)

    def test_resident_modulation_build_remains_available(self) -> None:
        resident = MiniMaxH3DiT(self.config, construction_mode=RESIDENT_CONSTRUCTION)
        cache = ModulationCache.build(resident, mx.array([0.0, 0.5]), dtype=mx.bfloat16)
        self.assertEqual(len(cache.tables), 2)


if __name__ == "__main__":
    unittest.main()

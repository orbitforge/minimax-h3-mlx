"""Focused v0.3c tests for format routing, cache-only construction, and strict base gates."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.adaln import ModulationCache
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.dit import CACHE_ONLY_CONSTRUCTION, MiniMaxH3DiT
from minimax_h3_mlx.load import (
    inspect_checkpoint_format,
    collect_weight_payloads,
    load_dit,
    validate_derived_base_index,
)
import minimax_h3_mlx.load as load_module


FORMAT = "minimax-h3-mlx-streamed-adaln-v1"


def tiny_config() -> DiTConfig:
    return DiTConfig(
        hidden_size=16,
        num_layers=2,
        token_refiner_num_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        ffn_hidden_size=16,
        latents_dim=2,
        audio_latents_dim=4,
        patch_size=(1, 1, 1),
        text_dim=8,
        timestep_input_dim=8,
        time_embed_hidden_size=16,
        time_embed_dim=8,
        adaln_out_features=6 * 3 * 16,
        final_adaln_out_features=2 * 16,
        rope_inv_freq_len=1,
    )


def write_derived_fixture(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text(json.dumps({"hidden_size": 16}) + "\n")
    base = root / "base"
    base.mkdir()
    shards = [f"model-{i:05d}-of-00005.safetensors" for i in range(1, 6)]
    ordinary = [f"ordinary.{i}" for i in range(848)]
    keys = ordinary + [
        "final_layer.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
    ]
    weight_map = {key: shards[index % 5] for index, key in enumerate(keys)}
    (base / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}) + "\n")
    for shard in shards:
        (base / shard).write_bytes(b"")

    adaln = root / "adaln"
    adaln.mkdir()
    blocks = {}
    for index in range(50):
        filename = f"block-{index:03d}.safetensors"
        (adaln / filename).write_bytes(b"")
        blocks[str(index)] = {"block_index": index, "sidecar_filename": filename}
    (adaln / "manifest.json").write_text(
        json.dumps({"format_identifier": FORMAT, "schema_version": 1, "bounded": False, "blocks": blocks})
        + "\n"
    )
    (root / "conversion_manifest.json").write_text(
        json.dumps(
            {
                "format_identifier": FORMAT,
                "schema_version": 1,
                "bounded": False,
                "verification_status": "verified",
                "derived_base_tensor_count": 850,
                "total_logical_tensor_count": 1050,
                "sidecar_count": 50,
                "sidecar_tensor_count": 200,
                "selected_blocks": list(range(50)),
            }
        )
        + "\n"
    )


class CacheOnlyLoaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="cache-only-loader-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def test_original_checkpoint_routes_to_resident_without_manifest(self) -> None:
        original = self.temp / "original"
        original.mkdir()
        info = inspect_checkpoint_format(original)
        self.assertEqual(info.checkpoint_format, "original")
        self.assertEqual(info.construction_mode, "resident")
        self.assertEqual(info.base_root, original)

    def test_complete_derived_checkpoint_routes_to_cache_only(self) -> None:
        derived = self.temp / "derived"
        write_derived_fixture(derived)
        info = inspect_checkpoint_format(derived)
        self.assertEqual(info.checkpoint_format, "derived")
        self.assertEqual(info.construction_mode, CACHE_ONLY_CONSTRUCTION)
        self.assertEqual(info.base_root, derived / "base")
        self.assertEqual(info.adaln_manifest_path, derived / "adaln" / "manifest.json")

    def test_unindexed_extra_base_safetensors_file_is_rejected(self) -> None:
        derived = self.temp / "derived-extra-base"
        write_derived_fixture(derived)
        (derived / "base" / "unindexed-extra.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "unindexed-extra.safetensors"):
            inspect_checkpoint_format(derived)

    def test_unmanifested_extra_adaln_sidecar_is_rejected(self) -> None:
        derived = self.temp / "derived-extra-sidecar"
        write_derived_fixture(derived)
        (derived / "adaln" / "block-050.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "block-050.safetensors"):
            inspect_checkpoint_format(derived)

    def test_keep_adaln_is_rejected_before_derived_model_loading(self) -> None:
        derived = self.temp / "derived"
        write_derived_fixture(derived)
        with self.assertRaisesRegex(ValueError, "sidecars"):
            load_dit(derived, keep_adaln=True)

    def test_invalid_derived_manifest_states_fail_without_fallback(self) -> None:
        for field, value in (
            ("schema_version", 999),
            ("verification_status", "pending"),
            ("bounded", True),
            ("format_identifier", "wrong-format"),
        ):
            derived = self.temp / field
            write_derived_fixture(derived)
            manifest_path = derived / "conversion_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest[field] = value
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError, msg=field):
                inspect_checkpoint_format(derived)

    def test_missing_base_index_sidecar_manifest_or_sidecar_fails(self) -> None:
        for relative in (
            "base/model.safetensors.index.json",
            "adaln/manifest.json",
            "adaln/block-017.safetensors",
        ):
            derived = self.temp / relative.replace("/", "-")
            write_derived_fixture(derived)
            (derived / relative).unlink()
            with self.assertRaises((ValueError, FileNotFoundError), msg=relative):
                inspect_checkpoint_format(derived)

    def test_cache_only_blocks_have_no_block_adaln_parameters(self) -> None:
        model = MiniMaxH3DiT(tiny_config(), construction_mode=CACHE_ONLY_CONSTRUCTION)
        keys = {key for key, _ in tree_flatten(model.parameters())}
        self.assertEqual(len(model.blocks), 2)
        self.assertFalse(any(key.startswith("blocks.") and ".adaln_proj." in key for key in keys))
        self.assertIn("final_layer.adaln_proj.linear.weight", keys)
        self.assertIn("final_layer.adaln_proj.linear.bias", keys)
        self.assertTrue(all(not hasattr(block.adaln_proj, "linear") for block in model.blocks))

    def test_complete_cache_boundary_is_accepted(self) -> None:
        config = tiny_config()
        timesteps = mx.array([0.0, 0.5])
        table = tuple(mx.zeros((6, config.hidden_size)) for _ in range(6))
        cache = ModulationCache([table, table], timesteps)
        cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)

    def _cache(self, *, timesteps=None, blocks=2, width=None, tensor_count=6, dtype=None):
        config = tiny_config()
        timesteps = mx.array([0.0, 0.5] if timesteps is None else timesteps)
        width = config.hidden_size if width is None else width
        dtype = mx.float32 if dtype is None else dtype
        table = tuple(mx.zeros((timesteps.shape[0] * 3, width), dtype=dtype) for _ in range(tensor_count))
        return config, ModulationCache([table for _ in range(blocks)], timesteps)

    def test_cache_timetable_order_missing_extra_and_duplicate_are_rejected(self) -> None:
        config, cache = self._cache()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            cache.validate_for(config, mx.array([0.5, 0.0]), mx.array([0, 1, 0]), 3)
        with self.assertRaisesRegex(ValueError, "timetable count"):
            cache.validate_for(config, mx.array([0.0]), mx.array([0, 1, 0]), 3)
        with self.assertRaisesRegex(ValueError, "timetable count"):
            cache.validate_for(config, mx.array([0.0, 0.5, 1.0]), mx.array([0, 1, 0]), 3)
        _, duplicate = self._cache(timesteps=[0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            duplicate.validate_for(config, mx.array([0.0, 0.0]), mx.array([0, 1, 0]), 3)

    def test_cache_indices_require_integer_dtype_and_sequence_boundary(self) -> None:
        config, cache = self._cache()
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0.0, 1.0, 0.0]), 3)
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0.0, 1.5, 0.0]), 3)
        with self.assertRaisesRegex(ValueError, "sequence length"):
            cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1]), 3)
        with self.assertRaisesRegex(ValueError, "index"):
            cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([-1, 1, 0]), 3)
        with self.assertRaisesRegex(ValueError, "index"):
            cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 2, 0]), 3)

    def test_cache_modulation_shape_dtype_block_and_tensor_counts_are_rejected(self) -> None:
        config, wrong_dtype = self._cache(dtype=mx.int32)
        with self.assertRaisesRegex(ValueError, "dtype"):
            wrong_dtype.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)
        _, inconsistent = self._cache()
        inconsistent.tables[1] = tuple(mx.zeros((6, config.hidden_size), dtype=mx.float16) for _ in range(6))
        with self.assertRaisesRegex(ValueError, "consistent"):
            inconsistent.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)
        _, wrong_blocks = self._cache(blocks=1)
        with self.assertRaisesRegex(ValueError, "block entries"):
            wrong_blocks.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)
        _, wrong_width = self._cache(width=config.hidden_size + 1)
        with self.assertRaisesRegex(ValueError, "shape"):
            wrong_width.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)
        _, wrong_tensor_count = self._cache(tensor_count=5)
        with self.assertRaisesRegex(ValueError, "six"):
            wrong_tensor_count.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1, 0]), 3)

    def test_partial_malformed_and_missing_timestep_cache_is_rejected(self) -> None:
        config = tiny_config()
        timesteps = mx.array([0.0, 0.5])
        good = tuple(mx.zeros((6, config.hidden_size)) for _ in range(6))
        cases = (
            ModulationCache([good], timesteps),
            ModulationCache([good[:5], good], timesteps),
            ModulationCache([tuple(mx.zeros((5, config.hidden_size)) for _ in range(6)), good], timesteps),
        )
        for cache in cases:
            with self.assertRaises(ValueError):
                cache.validate_for(config, mx.array([0.0, 0.5]), mx.array([0, 1]), 2)
        with self.assertRaises(ValueError):
            ModulationCache([good, good], mx.array([0.0])).validate_for(
                config, mx.array([0.0, 0.5]), mx.array([0, 0]), 2
            )

    def test_cache_only_forward_rejects_missing_cache_before_computation(self) -> None:
        model = MiniMaxH3DiT(tiny_config(), construction_mode=CACHE_ONLY_CONSTRUCTION)
        with self.assertRaisesRegex(RuntimeError, "complete modulation cache"):
            model(*([None] * 10))

    def test_exact_base_index_validation_rejects_missing_extra_and_block_keys(self) -> None:
        derived = self.temp / "derived"
        write_derived_fixture(derived)
        info = inspect_checkpoint_format(derived)
        index = json.loads((derived / "base" / "model.safetensors.index.json").read_text())
        expected = set(index["weight_map"])
        validate_derived_base_index(info, expected)
        with self.assertRaises(KeyError):
            validate_derived_base_index(info, expected - {"ordinary.0"})
        with self.assertRaises(KeyError):
            validate_derived_base_index(info, expected | {"ordinary.extra"})

        index["weight_map"]["blocks.0.adaln_proj.linear.weight"] = next(iter(index["weight_map"].values()))
        (derived / "base" / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaises(KeyError):
            validate_derived_base_index(info, expected)

    def test_derived_strict_false_cannot_bypass_validation(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaisesRegex(KeyError, "unexpected physical"):
            collect_weight_payloads(
                [("base-1", {"a": tensor, "extra": tensor})],
                {"a"},
                strict=False,
                derived=True,
            )

    def test_derived_rejects_unexpected_physical_base_tensor(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaises(KeyError):
            collect_weight_payloads([("base-1", {"a": tensor, "extra": tensor})], {"a"}, derived=True)

    def test_derived_rejects_duplicate_physical_tensor_across_shards(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaisesRegex(KeyError, "duplicate physical"):
            collect_weight_payloads(
                [("base-1", {"a": tensor}), ("base-2", {"a": tensor})],
                {"a"},
                derived=True,
            )

    def test_derived_rejects_missing_physical_tensor_even_when_index_claims_it(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaisesRegex(KeyError, "missing"):
            collect_weight_payloads(
                [("base-1", {"a": tensor})],
                {"a", "b"},
                derived=True,
                index_weight_map={"a": "base-1", "b": "base-2"},
            )

    def test_derived_rejects_physical_block_adaln_tensor(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaisesRegex(KeyError, "block-level AdaLN"):
            collect_weight_payloads(
                [("base-1", {"a": tensor, "blocks.0.adaln_proj.linear.weight": tensor})],
                {"a"},
                derived=True,
            )

    def test_derived_rejects_all_expected_tensors_plus_one_extra(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaises(KeyError):
            collect_weight_payloads(
                [("base-1", {"a": tensor, "b": tensor, "extra": tensor})],
                {"a", "b"},
                strict=False,
                derived=True,
            )

    def test_derived_rejects_payload_index_disagreement(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        with self.assertRaisesRegex(KeyError, "payload/index"):
            collect_weight_payloads(
                [("base-1", {"a": tensor})],
                {"a"},
                derived=True,
                index_weight_map={"a": "base-2"},
            )

    def test_original_strict_false_compatibility_remains_unchanged(self) -> None:
        tensor = SimpleNamespace(nbytes=4)
        weights, logical_bytes = collect_weight_payloads(
            [("base-1", {"a": tensor, "extra": tensor})],
            {"a", "missing"},
            strict=False,
        )
        self.assertEqual(set(weights), {"a"})
        self.assertEqual(logical_bytes, 4)

    def test_format_inspection_does_not_call_mlx_sidecar_loader(self) -> None:
        derived = self.temp / "derived"
        write_derived_fixture(derived)
        calls: list[str] = []
        with mock.patch.object(load_module.mx, "load", side_effect=lambda path: calls.append(str(path))):
            inspect_checkpoint_format(derived)
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()

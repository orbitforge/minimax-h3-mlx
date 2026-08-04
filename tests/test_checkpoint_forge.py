"""Focused source-safety, exactness, corruption, and consumer tests for the checkpoint forge."""

from __future__ import annotations

import json
import os
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.checkpoint_forge import tensor_io
from minimax_h3_mlx.checkpoint_forge.forge import ForgeOptions, _space_report, forge_checkpoint
from minimax_h3_mlx.checkpoint_forge.tensor_io import DTYPE_BYTES, read_safetensors_header, sha256_file, sha256_range
from minimax_h3_mlx.checkpoint_forge.topology import BLOCK_ROLES, SourceTopology, parse_block_selection
from minimax_h3_mlx.checkpoint_forge.verify import verify_checkpoint


def payload(dtype: str, shape: tuple[int, ...], seed: int) -> bytes:
    size = DTYPE_BYTES[dtype]
    count = 1
    for dimension in shape:
        count *= dimension
    return bytes(((seed + index) % 251 for index in range(count * size)))


def write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    offset = 0
    entries = {"__metadata__": {"format": "mlx"}}
    for name in sorted(tensors):
        dtype, shape, raw = tensors[name]
        expected = 1
        for dimension in shape:
            expected *= dimension
        expected *= DTYPE_BYTES[dtype]
        if len(raw) != expected:
            raise AssertionError(f"fixture payload mismatch for {name}: {len(raw)} != {expected}")
        entries[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    header = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header)))
        handle.write(header)
        for name in sorted(tensors):
            handle.write(tensors[name][2])


def synthetic_checkpoint(root: Path) -> None:
    root.mkdir()
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    projection = {
        "weight": ("U32", (12, 2)),
        "scales": ("BF16", (12, 2)),
        "biases": ("BF16", (12, 2)),
        "bias": ("BF16", (12,)),
    }
    for block in range(50):
        for role, (dtype, shape) in projection.items():
            tensors[f"blocks.{block}.adaln_proj.linear.{role}"] = (dtype, shape, payload(dtype, shape, block + len(role)))
    tensors["final_layer.adaln_proj.linear.bias"] = ("BF16", (4,), payload("BF16", (4,), 201))
    tensors["final_layer.adaln_proj.linear.weight"] = ("U32", (4,), payload("U32", (4,), 202))
    tensors["ordinary.weight"] = ("U32", (4,), payload("U32", (4,), 203))
    tensors["ordinary.bias"] = ("BF16", (4,), payload("BF16", (4,), 204))
    names = sorted(tensors)
    shards = [names[::3], names[1::3], names[2::3]]
    weight_map = {}
    for index, names_for_shard in enumerate(shards, 1):
        filename = f"model-{index:05d}-of-00003.safetensors"
        write_raw_safetensors(root / filename, {name: tensors[name] for name in names_for_shard})
        weight_map.update({name: filename for name in names_for_shard})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(len(item[2]) for item in tensors.values())}, "weight_map": weight_map}, sort_keys=True)
    )
    (root / "config.json").write_text(json.dumps({"num_layers": 50, "time_embed_dim": 8, "adaln_out_features": 12, "synthetic": True}) + "\n")
    (root / "quant_config.json").write_text(json.dumps({"bits": 6, "adaln_bits": 8, "group_size": 4, "quantize_adaln": True}) + "\n")


def mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


class CheckpointForgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="checkpoint-forge-test-"))
        self.source = self.temp / "source"
        synthetic_checkpoint(self.source)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def forge(self, name: str = "derived", blocks: tuple[int, ...] | None = (5,)) -> Path:
        output = self.temp / name
        forge_checkpoint(ForgeOptions(self.source, output, blocks=blocks))
        return output

    def test_topology_classification_and_cross_shard_reconstruction(self) -> None:
        topology = SourceTopology.load(self.source)
        self.assertEqual(topology.source_tensor_count, 204)
        self.assertEqual(len(topology.ordinary), 2)
        self.assertEqual(len(topology.final_adaln), 2)
        self.assertEqual(len(topology.block_adaln), 50)
        self.assertGreater(len({record.shard for record in topology.block_adaln[5]}), 1)
        self.assertEqual({record.name.rsplit(".", 1)[-1] for record in topology.block_adaln[5]}, set(BLOCK_ROLES))
        self.assertEqual(topology.block_projection_metadata()["logical_input_features"], 8)

    def test_raw_bf16_and_u32_bytes_are_exact(self) -> None:
        output = self.forge(blocks=None)
        self.assertTrue(verify_checkpoint(self.source, output).ok)
        source = SourceTopology.load(self.source)
        derived_header = read_safetensors_header(output / "adaln" / "block-005.safetensors")
        source_record = source.block_adaln[5][0]
        derived = derived_header.tensor_map()[source_record.name]
        self.assertEqual(source_record.checksum(), sha256_range(output / "adaln" / "block-005.safetensors", derived_header.data_start + derived.start, derived_header.data_start + derived.end))
        self.assertEqual(derived.dtype, source_record.dtype)

    def test_bounded_dry_run_and_conversion_write_no_base_payload(self) -> None:
        dry = self.temp / "dry"
        result = forge_checkpoint(ForgeOptions(self.source, dry, dry_run=True, blocks=(0, 5)))
        self.assertIsNone(result.output)
        self.assertFalse(dry.exists())
        output = self.forge("bounded", (0, 5))
        self.assertTrue((output / "base" / "classification.json").exists())
        self.assertFalse((output / "base" / "model.safetensors.index.json").exists())
        self.assertTrue(verify_checkpoint(self.source, output).ok)

    def test_equal_source_output_is_rejected_with_and_without_force_before_temp_creation(self) -> None:
        before = {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()}
        for force in (False, True):
            with self.assertRaises(ValueError):
                forge_checkpoint(ForgeOptions(self.source, self.source, force=force))
            self.assertEqual(list(self.temp.glob(f".{self.source.name}.incomplete-*")), [])
        self.assertEqual(before, {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()})

    def test_output_below_source_is_rejected_before_temp_creation(self) -> None:
        output = self.source / "derived"
        with self.assertRaises(ValueError):
            forge_checkpoint(ForgeOptions(self.source, output, force=True))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.source.glob(".derived.incomplete-*")), [])

    def test_source_below_output_is_rejected_before_temp_creation(self) -> None:
        output = self.temp
        with self.assertRaises(ValueError):
            forge_checkpoint(ForgeOptions(self.source, output, force=True))
        self.assertFalse(list(self.temp.glob(".checkpoint-forge-test-*.incomplete-*")))

    def test_existing_output_refusal_and_verify_only(self) -> None:
        output = self.forge(blocks=(0,))
        with self.assertRaises(ValueError):
            forge_checkpoint(ForgeOptions(self.source, output, blocks=(0,)))
        self.assertTrue(forge_checkpoint(ForgeOptions(self.source, output, verify_only=True, blocks=(0,))).message.startswith("exact"))

    def test_manifest_and_index_are_deterministically_ordered(self) -> None:
        output = self.forge(blocks=(5,))
        manifest = json.loads((output / "adaln" / "manifest.json").read_text())
        self.assertEqual(list(manifest["blocks"]), ["5"])
        self.assertEqual([item["tensor_key"] for item in manifest["blocks"]["5"]["tensors"]], sorted(item["tensor_key"] for item in manifest["blocks"]["5"]["tensors"]))
        self.assertEqual(manifest["blocks"]["5"]["tensors"][0]["tensor_role"], "learned_bias")

    def test_selection_parser(self) -> None:
        self.assertEqual(parse_block_selection("5"), (5,))
        self.assertEqual(parse_block_selection("0,5,7-8"), (0, 5, 7, 8))
        with self.assertRaises(ValueError):
            parse_block_selection("50")

    def test_no_bfloat16_numpy_conversion_path(self) -> None:
        self.assertNotIn("numpy", tensor_io.__dict__)

    def test_source_original_is_unchanged(self) -> None:
        before = {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()}
        self.forge(blocks=(5,))
        self.assertEqual(before, {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()})

    def test_missing_sidecar_is_rejected(self) -> None:
        output = self.forge(blocks=(5,))
        (output / "adaln" / "block-005.safetensors").unlink()
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_unexpected_sidecar_is_rejected(self) -> None:
        output = self.forge(blocks=(5,))
        shutil.copyfile(output / "adaln" / "block-005.safetensors", output / "adaln" / "block-006.safetensors")
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_corrupted_sidecar_is_rejected(self) -> None:
        output = self.forge(blocks=(5,))
        with (output / "adaln" / "block-005.safetensors").open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_corrupted_base_shard_and_appended_bytes_are_rejected(self) -> None:
        output = self.forge(blocks=None)
        shard = next((output / "base").glob("*.safetensors"))
        with shard.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_config_quant_config_base_index_adaln_manifest_and_report_corruption_are_rejected(self) -> None:
        artifacts = ["config.json", "quant_config.json", "adaln/manifest.json", "verification_report.txt"]
        for relative in artifacts:
            output = self.forge(relative.replace("/", "-"), blocks=None)
            path = output / relative
            if path.suffix == ".json":
                mutate_json(path, lambda value: value.update({"corrupted": True}))
            else:
                path.write_text("{\"status\": \"corrupted\"}\n")
            with self.assertRaises(ValueError, msg=relative):
                verify_checkpoint(self.source, output)
        output = self.forge("corrupt-index", blocks=None)
        mutate_json(output / "base" / "model.safetensors.index.json", lambda value: value["metadata"].update({"total_size": 0}))
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_root_byte_and_tensor_count_corruption_is_rejected(self) -> None:
        for field in ("source_logical_payload_byte_count", "total_logical_tensor_count"):
            output = self.forge(field, blocks=(5,))
            mutate_json(output / "conversion_manifest.json", lambda value, field=field: value.update({field: value[field] + 1}))
            with self.assertRaises(ValueError, msg=field):
                verify_checkpoint(self.source, output)

    def test_sidecar_tensor_metadata_and_extra_manifest_block_are_rejected(self) -> None:
        output = self.forge(blocks=(5,))
        mutate_json(output / "adaln" / "manifest.json", lambda value: value["blocks"]["5"]["tensors"][0].update({"byte_count": 0}))
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)
        output = self.forge("extra-manifest-block", blocks=(5,))
        mutate_json(output / "adaln" / "manifest.json", lambda value: value["blocks"].update({"6": value["blocks"]["5"]}))
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_unexpected_base_shard_is_rejected(self) -> None:
        output = self.forge(blocks=None)
        write_raw_safetensors(output / "base" / "unexpected.safetensors", {"unexpected": ("U32", (1,), payload("U32", (1,), 1))})
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_per_file_checksum_enforcement_is_genuine(self) -> None:
        output = self.forge(blocks=(5,))
        (output / "config.json").write_text((output / "config.json").read_text() + "\n")
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_changed_source_topology_is_rejected_by_completion_verification(self) -> None:
        output = self.forge(blocks=(5,))
        index = json.loads((self.source / "model.safetensors.index.json").read_text())
        index["weight_map"]["new.tensor"] = next(iter(index["weight_map"].values()))
        (self.source / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            verify_checkpoint(self.source, output)

    def test_source_missing_companion_is_rejected(self) -> None:
        index = json.loads((self.source / "model.safetensors.index.json").read_text())
        del index["weight_map"]["blocks.0.adaln_proj.linear.bias"]
        (self.source / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            SourceTopology.load(self.source)

    def test_duplicate_logical_tensor_across_source_shards_is_rejected(self) -> None:
        index = json.loads((self.source / "model.safetensors.index.json").read_text())["weight_map"]
        original_name = "blocks.0.adaln_proj.linear.bias"
        original_shard = index[original_name]
        target_shard = next(shard for name, shard in index.items() if name != original_name and shard != original_shard)
        parsed = read_safetensors_header(self.source / target_shard)
        tensors = {item.name: (item.dtype, item.shape, (self.source / target_shard).read_bytes()[parsed.data_start + item.start:parsed.data_start + item.end]) for item in parsed.tensors}
        source = SourceTopology.load(self.source)
        record = next(item for item in source.records if item.name == original_name)
        tensors[original_name] = (record.dtype, record.shape, (record.path).read_bytes()[record.data_start:record.data_start + record.nbytes])
        write_raw_safetensors(self.source / target_shard, tensors)
        with self.assertRaises(ValueError):
            SourceTopology.load(self.source)

    def test_unexpected_fifth_adaln_companion_is_rejected(self) -> None:
        index = json.loads((self.source / "model.safetensors.index.json").read_text())
        target = next(name for name in index["weight_map"] if name == "blocks.0.adaln_proj.linear.weight")
        shard = index["weight_map"][target]
        parsed = read_safetensors_header(self.source / shard)
        tensors = {item.name: (item.dtype, item.shape, (self.source / shard).read_bytes()[parsed.data_start + item.start:parsed.data_start + item.end]) for item in parsed.tensors}
        tensors["blocks.0.adaln_proj.linear.extra"] = ("U32", (1,), payload("U32", (1,), 9))
        write_raw_safetensors(self.source / shard, tensors)
        index["weight_map"]["blocks.0.adaln_proj.linear.extra"] = shard
        (self.source / "model.safetensors.index.json").write_text(json.dumps(index))
        with self.assertRaises(ValueError):
            SourceTopology.load(self.source)

    def test_force_replacement_success_and_source_is_not_deleted(self) -> None:
        output = self.forge("force", blocks=(0,))
        before = {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()}
        forge_checkpoint(ForgeOptions(self.source, output, force=True, blocks=(5,)))
        self.assertEqual(json.loads((output / "conversion_manifest.json").read_text())["selected_blocks"], [5])
        self.assertEqual(before, {path: sha256_file(path) for path in self.source.iterdir() if path.is_file()})
        self.assertEqual(list(self.temp.glob(".force.previous-*")), [])

    def test_force_failure_before_backup_rename_leaves_output(self) -> None:
        output = self.forge("force-before", blocks=(0,))
        original = (output / "conversion_manifest.json").read_bytes()
        with mock.patch("minimax_h3_mlx.checkpoint_forge.forge.os.replace", side_effect=OSError("before backup")):
            with self.assertRaises(OSError):
                forge_checkpoint(ForgeOptions(self.source, output, force=True, blocks=(5,)))
        self.assertEqual((output / "conversion_manifest.json").read_bytes(), original)
        self.assertEqual(list(self.temp.glob(".force-before.previous-*")), [])

    def test_force_failure_after_backup_rename_restores_prior_output(self) -> None:
        output = self.forge("force-after", blocks=(0,))
        original = (output / "conversion_manifest.json").read_bytes()
        real_replace = os.replace
        calls = 0

        def fail_after_backup(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("after backup")
            return real_replace(source, destination)

        with mock.patch("minimax_h3_mlx.checkpoint_forge.forge.os.replace", side_effect=fail_after_backup):
            with self.assertRaises(OSError):
                forge_checkpoint(ForgeOptions(self.source, output, force=True, blocks=(5,)))
        self.assertEqual((output / "conversion_manifest.json").read_bytes(), original)
        self.assertEqual(list(self.temp.glob(".force-after.previous-*")), [])

    def test_atomic_failure_cleans_incomplete_directory(self) -> None:
        output = self.temp / "failed"
        with mock.patch("minimax_h3_mlx.checkpoint_forge.forge.verify_checkpoint", side_effect=ValueError("synthetic failure")):
            with self.assertRaises(ValueError):
                forge_checkpoint(ForgeOptions(self.source, output, blocks=(5,)))
        self.assertFalse(output.exists())
        self.assertEqual(list(self.temp.glob(".failed.incomplete-*")), [])

    def test_safetensors_reader_rejects_shape_mismatch_overlap_hole_trailing_and_out_of_range(self) -> None:
        cases = {
            "shape": {"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, 4]}},
            "overlap": {
                "a": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "U32", "shape": [1], "data_offsets": [2, 6]},
            },
            "hole": {
                "a": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]},
                "b": {"dtype": "U32", "shape": [1], "data_offsets": [8, 12]},
            },
            "trailing": {"x": {"dtype": "U32", "shape": [1], "data_offsets": [0, 4]}},
            "outside": {"x": {"dtype": "U32", "shape": [2], "data_offsets": [0, 8]}},
        }
        for name, descriptors in cases.items():
            path = self.temp / f"bad-{name}.safetensors"
            header = json.dumps({"__metadata__": {"format": "mlx"}, **descriptors}, sort_keys=True, separators=(",", ":")).encode()
            raw = b"\0" * ({"shape": 4, "overlap": 6, "hole": 12, "trailing": 8, "outside": 4}[name])
            path.write_bytes(struct.pack("<Q", len(header)) + header + raw)
            with self.assertRaises(ValueError, msg=name):
                read_safetensors_header(path)

    def test_supported_dtype_sizes_are_enforced(self) -> None:
        for index, dtype in enumerate(("BF16", "U32", "F16", "F32", "I8", "I16", "I32", "I64", "U8", "U16", "U64", "F64", "BOOL")):
            path = self.temp / f"dtype-{index}.safetensors"
            write_raw_safetensors(path, {"x": (dtype, (2,), payload(dtype, (2,), index))})
            self.assertEqual(read_safetensors_header(path).tensor_map()["x"].dtype, dtype)

    def test_synthetic_complete_checkpoint_loads_with_mlx_when_runtime_is_available(self) -> None:
        try:
            import mlx.core as mx
            mx.get_active_memory()
        except Exception as exc:
            self.skipTest(f"MLX device unavailable: {exc}")
        output = self.forge("mlx", blocks=None)
        expected_dtype = {"BF16": mx.bfloat16, "U32": mx.uint32}
        for path in sorted((output / "base").glob("*.safetensors")) + sorted((output / "adaln").glob("*.safetensors")):
            loaded = mx.load(str(path))
            header = read_safetensors_header(path)
            self.assertEqual(set(loaded), set(header.tensor_map()))
            for name, descriptor in header.tensor_map().items():
                array = loaded[name]
                self.assertEqual(tuple(array.shape), descriptor.shape)
                self.assertEqual(str(array.dtype), str(expected_dtype[descriptor.dtype]))
            mx.eval(*loaded.values())
            del loaded
            mx.clear_cache()

    @unittest.skipUnless(os.environ.get("MINIMAX_H3_CHECKPOINT"), "set MINIMAX_H3_CHECKPOINT for bounded real-checkpoint MLX probes")
    def test_real_bounded_sidecars_load_with_mlx(self) -> None:
        import mlx.core as mx

        try:
            mx.get_active_memory()
        except Exception as exc:
            self.skipTest(f"MLX device unavailable: {exc}")
        real_source = Path(os.environ["MINIMAX_H3_CHECKPOINT"]).expanduser().resolve()
        output = self.temp / "real-bounded"
        forge_checkpoint(ForgeOptions(real_source, output, blocks=(0, 5)))
        topology = SourceTopology.load(real_source)
        memory = []
        for block in (0, 5):
            path = output / "adaln" / f"block-{block:03d}.safetensors"
            memory.append((block, "before_load", mx.get_active_memory(), mx.get_cache_memory()))
            loaded = mx.load(str(path))
            self.assertEqual(set(loaded), {record.name for record in topology.block_adaln[block]})
            for record in topology.block_adaln[block]:
                self.assertEqual(tuple(loaded[record.name].shape), record.shape)
                self.assertEqual(str(loaded[record.name].dtype), str({"BF16": mx.bfloat16, "U32": mx.uint32}[record.dtype]))
            memory.append((block, "after_load", mx.get_active_memory(), mx.get_cache_memory()))
            mx.eval(*loaded.values())
            memory.append((block, "after_evaluation", mx.get_active_memory(), mx.get_cache_memory()))
            del loaded
            mx.clear_cache()
            memory.append((block, "after_release", mx.get_active_memory(), mx.get_cache_memory()))
        print("real MLX sidecar memory:", memory)


if __name__ == "__main__":
    unittest.main()

"""MLX-free contracts for the Slice 024 monolithic Q6 converter."""

from __future__ import annotations

import inspect
import json
import shutil
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_monolithic_quant import main as converter_main
from minimax_h3_mlx.checkpoint_forge.tensor_io import read_safetensors_header
from minimax_h3_mlx.config import DiTConfig
from minimax_h3_mlx.monolithic_quant import (
    BytesQuantizedArray,
    QuantizedResult,
    build_conversion_plan,
    convert,
    quantized_tensor_shapes,
    verify_output,
)
from minimax_h3_mlx.monolithic_source import (
    MonolithicSafetensorsSource,
    MonolithicSourceError,
    SourceStaleError,
    classify_source,
    decode_bfloat16_to_float32,
    expected_source_dtype,
    expected_tensor_shapes,
    extract_embedded_config,
)
import minimax_h3_mlx.monolithic_quant as monolithic_quant


def write_raw_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    metadata: dict[str, str] | None = None,
) -> None:
    offset = 0
    entries: dict[str, object] = {"__metadata__": dict(sorted((metadata or {}).items()))}
    payloads: list[bytes] = []
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        entries[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        payloads.append(payload)
        offset += len(payload)
    raw_header = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"".join(payloads))


def payload(dtype: str, shape: tuple[int, ...], seed: int = 0) -> bytes:
    count = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if dtype == "BF16":
        pattern = np.array([0x3F80, 0x4000, 0xC000, 0x0000, 0x3F00], dtype="<u2")
        return np.resize(pattern, count).astype("<u2", copy=False).tobytes()
    if dtype == "F32":
        values = (np.arange(count, dtype=np.float32) + np.float32(seed + 1)) / np.float32(8.0)
        return values.astype("<f4", copy=False).tobytes()
    if dtype == "F16":
        return np.zeros(count, dtype="<f2").tobytes()
    raise AssertionError(dtype)


def tiny_raw_config() -> dict[str, object]:
    return {
        "_class_name": "MiniMaxH3DiTModel",
        "_diffusers_version": "0.32.2",
        "hidden_size": 64,
        "num_layers": 50,
        "token_refiner_num_layers": 2,
        "num_attention_heads": 1,
        "attention_head_dim": 64,
        "ffn_hidden_size": 64,
        "latents_dim": 4,
        "audio_latents_dim": 8,
        "patch_size": [1, 1, 1],
        "text_dim": 64,
        "timestep_input_dim": 16,
        "time_embed_hidden_size": 64,
        "time_embed_dim": 64,
        "adaln_out_features": 192,
        "final_adaln_out_features": 128,
        "rope_inv_freq_len": 1,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5,
        "final_norm_eps": 1e-5,
    }


def write_complete_fixture(root: Path) -> tuple[Path, dict[str, object]]:
    config_raw = tiny_raw_config()
    config = DiTConfig.from_dict(config_raw)
    tensors = {
        name: (expected_source_dtype(name), shape, payload(expected_source_dtype(name), shape, index))
        for index, (name, shape) in enumerate(expected_tensor_shapes(config).items())
    }
    path = root / "beta.safetensors"
    write_raw_safetensors(path, tensors, {"config": json.dumps(config_raw, sort_keys=True)})
    return path, config_raw


def write_small_fixture(root: Path, metadata: dict[str, str] | None = None) -> Path:
    path = root / "small.safetensors"
    write_raw_safetensors(
        path,
        {
            "a": ("BF16", (2,), b"\x80\x3f\x00\x40"),
            "b": ("F32", (3,), b"\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40"),
        },
        metadata,
    )
    return path


def write_planned_output(
    output: Path,
    plan,
    *,
    renamed_tensors: dict[str, str] | None = None,
) -> None:
    """Write a synthetic header-valid output topology without reading source payloads."""
    output.mkdir()
    renamed_tensors = renamed_tensors or {}
    shard_name = "model-00001-of-00001.safetensors"
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for tensor in plan.output_tensors:
        name = renamed_tensors.get(tensor.name, tensor.name)
        if name in tensors:
            raise AssertionError(f"duplicate synthetic output tensor: {name}")
        tensors[name] = (tensor.dtype, tensor.shape, b"\0" * tensor.nbytes)
    write_raw_safetensors(output / shard_name, tensors, {"format": "mlx"})
    (output / "config.json").write_text(json.dumps(plan.config_raw, indent=2, sort_keys=True) + "\n")
    (output / "quant_config.json").write_text(json.dumps(plan.quant_config, indent=2, sort_keys=True) + "\n")
    metadata = {
        "total_size": sum(len(item[2]) for item in tensors.values()),
        "bounded": plan.bounded,
        "source_identity": plan.source.identity,
        "source_size": plan.source.source_size,
        "selected_quantized_weights": list(plan.selected_quantized_weights),
        "quantized_layers": plan.quantized_counts,
    }
    index = {
        "metadata": metadata,
        "weight_map": {name: shard_name for name in tensors},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )


class FakeQuantizer:
    def __init__(self, plan):
        self.plan = plan
        self.calls: list[tuple[str, bytes | None]] = []
        self.releases = 0

    def quantize(self, parent: str, bias_bytes: bytes | None = None) -> QuantizedResult:
        self.calls.append((parent, bias_bytes))
        item = self.plan.classification.by_name[parent]
        self.plan.source.read_tensor(parent)
        bias_name = f"{parent[:-len('.weight')]}.bias"
        if bias_bytes is None and bias_name in self.plan.classification.by_name:
            bias_bytes = self.plan.source.read_tensor(bias_name)
            self.calls[-1] = (parent, bias_bytes)
        shapes = quantized_tensor_shapes(item.descriptor, bits=item.bits or 0, group_size=64)
        arrays = {
            role: BytesQuantizedArray(
                "U32" if role == "weight" else "BF16",
                shape,
                b"\0" * ((int(np.prod(shape, dtype=np.int64)) if shape else 1) * (4 if role == "weight" else 2)),
            )
            for role, shape in shapes.items()
        }
        return QuantizedResult(arrays)

    def release(self, result: QuantizedResult) -> None:
        self.releases += 1


class MonolithicQuantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="slice024-monolithic-"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp)

    def _convert_bounded_q6(self, output_name: str):
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(
            source,
            selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",),
        )
        output = self.temp / output_name
        convert(plan, output, target_shard_bytes=300, quantizer=FakeQuantizer(plan))
        return source, plan, output

    def test_header_inspection_reads_zero_payload_bytes(self) -> None:
        path = write_small_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        self.assertEqual(source.payload_bytes_read, 0)
        self.assertEqual(source.range_read_count, 0)
        self.assertEqual(source.header_bytes_read, source.header.data_start)

    def test_named_range_fetch_reads_only_selected_tensor_bytes(self) -> None:
        path = write_small_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        expected = source.descriptor("b").nbytes
        self.assertEqual(source.read_tensor("b"), b"\x00\x00\x80\x3f\x00\x00\x00\x40\x00\x00\x40\x40")
        self.assertEqual(source.payload_bytes_read, expected)
        self.assertEqual(source.range_read_count, 1)

    def test_source_staleness_is_detected_before_payload_read(self) -> None:
        path = write_small_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        with path.open("r+b") as handle:
            handle.seek(source.header.data_start)
            original = handle.read(1)
            handle.seek(source.header.data_start)
            handle.write(bytes([original[0] ^ 1]))
        with self.assertRaises(SourceStaleError):
            source.read_tensor("a")
        self.assertEqual(source.payload_bytes_read, 0)

    def test_malformed_offsets_are_rejected(self) -> None:
        path = self.temp / "malformed.safetensors"
        raw_header = json.dumps(
            {
                "__metadata__": {},
                "x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 1]},
            },
            separators=(",", ":"),
        ).encode()
        path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + b"\0\0")
        with self.assertRaises((ValueError, MonolithicSourceError)):
            MonolithicSafetensorsSource(path)

    def test_unsupported_dtype_is_rejected(self) -> None:
        path = self.temp / "unsupported.safetensors"
        write_raw_safetensors(path, {"x": ("F16", (1,), b"\0\0")})
        with self.assertRaises(MonolithicSourceError):
            MonolithicSafetensorsSource(path)

    def test_embedded_config_extraction_is_explicit_and_strict(self) -> None:
        config = tiny_raw_config()
        extracted = extract_embedded_config({"config": json.dumps(config)})
        self.assertEqual(extracted["hidden_size"], 64)
        wrapped = extract_embedded_config({"config": json.dumps({"transformer": config})})
        self.assertEqual(wrapped["hidden_size"], 64)
        with self.assertRaises(MonolithicSourceError):
            extract_embedded_config({"model_config": json.dumps(config)})
        incomplete = dict(config)
        del incomplete["adaln_out_features"]
        with self.assertRaises(MonolithicSourceError):
            from minimax_h3_mlx.monolithic_source import validate_config_contract

            validate_config_contract(incomplete)

    def test_full_classification_counts_and_never_quantize_policy(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        classification = classify_source(source)
        self.assertEqual(len(source.tensors), 535)
        self.assertEqual(classification.counts, {
            "learned_bias": 50,
            "ordinary": 226,
            "q6_core_weight": 208,
            "q8_block_adaln_weight": 50,
            "recomputed": 1,
        })
        self.assertEqual(classification.by_name["rope.inv_freq"].role, "recomputed")
        self.assertEqual(classification.by_name["blocks.0.attn.qkv_proj.weight"].role, "q6_core_weight")
        self.assertEqual(classification.by_name["token_refiner.blocks.1.mlp.fc2.weight"].role, "q6_core_weight")
        self.assertEqual(classification.by_name["blocks.49.adaln_proj.linear.weight"].role, "q8_block_adaln_weight")
        for name in (
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "condition_proj.weight",
            "time_embedder.proj_in.weight",
            "final_layer.adaln_proj.linear.weight",
            "final_layer.video_out.weight",
            "final_layer.audio_out.weight",
        ):
            self.assertEqual(classification.by_name[name].role, "ordinary", name)

    def test_complete_output_arithmetic_proves_1050_tensors(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        plan = build_conversion_plan(MonolithicSafetensorsSource(path))
        self.assertFalse(plan.bounded)
        self.assertEqual(plan.source_tensor_count, 535)
        self.assertEqual(plan.stored_source_tensor_count, 534)
        self.assertEqual(plan.output_tensor_count, 1050)
        self.assertEqual(plan.quantized_counts, {"6": 208, "8": 50})

    def test_output_only_full_verification_proves_exact_canonical_names(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(source)
        expected_names = {tensor.name for tensor in plan.output_tensors}
        self.assertEqual(len(expected_names), 1050)
        output = self.temp / "synthetic-full-output"
        write_planned_output(output, plan)

        output_only = verify_output(output)
        self.assertEqual(output_only.tensor_count, 1050)
        self.assertFalse(output_only.bounded)
        self.assertFalse(output_only.source_checked)
        source_linked = verify_output(output, source=source)
        self.assertTrue(source_linked.source_checked)
        self.assertEqual(source.payload_bytes_read, 0)

    def test_full_verification_rejects_name_substitution_at_same_count(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(source)
        original = plan.output_tensors[0].name
        output = self.temp / "forged-full-output"
        write_planned_output(
            output,
            plan,
            renamed_tensors={original: f"forged.{original}"},
        )
        index = json.loads((output / "model.safetensors.index.json").read_text())
        self.assertEqual(len(index["weight_map"]), 1050)
        with self.assertRaisesRegex(MonolithicSourceError, "exact canonical conversion topology"):
            verify_output(output)
        self.assertEqual(source.payload_bytes_read, 0)

    def test_q6_q8_output_keys_dtypes_and_shapes(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        plan = build_conversion_plan(MonolithicSafetensorsSource(path))
        outputs = {tensor.name: (tensor.dtype, tensor.shape) for tensor in plan.output_tensors}
        self.assertEqual(outputs["blocks.0.attn.qkv_proj.weight"], ("U32", (192, 12)))
        self.assertEqual(outputs["blocks.0.attn.qkv_proj.scales"], ("BF16", (192, 1)))
        self.assertEqual(outputs["blocks.0.attn.qkv_proj.biases"], ("BF16", (192, 1)))
        self.assertEqual(outputs["blocks.0.adaln_proj.linear.weight"], ("U32", (192, 16)))
        self.assertEqual(outputs["blocks.0.adaln_proj.linear.scales"], ("BF16", (192, 1)))
        self.assertEqual(outputs["blocks.0.adaln_proj.linear.biases"], ("BF16", (192, 1)))

    def test_learned_bias_is_retained_with_quantized_parent(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        plan = build_conversion_plan(
            MonolithicSafetensorsSource(path),
            selected_quantized_weights=("blocks.0.adaln_proj.linear.weight",),
        )
        bias = next(item for item in plan.output_tensors if item.name == "blocks.0.adaln_proj.linear.bias")
        self.assertEqual((bias.dtype, bias.shape), ("BF16", (192,)))
        self.assertEqual(bias.quantized_parent, "blocks.0.adaln_proj.linear.weight")
        self.assertEqual(bias.quantized_role, "bias")

    def test_bfloat16_decode_preserves_representative_values(self) -> None:
        raw = np.array([0x3F80, 0x4000, 0xC000, 0x0000, 0x3F00], dtype="<u2").tobytes()
        decoded = decode_bfloat16_to_float32(raw, (5,))
        np.testing.assert_array_equal(decoded, np.array([1.0, 2.0, -2.0, 0.0, 0.5], dtype=np.float32))

    def test_sharding_and_index_order_are_deterministic(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        plan = build_conversion_plan(MonolithicSafetensorsSource(path), selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",))
        first = [(shard.filename, tuple(item.name for item in shard.tensors)) for shard in plan.shards(300)]
        second = [(shard.filename, tuple(item.name for item in shard.tensors)) for shard in plan.shards(300)]
        self.assertEqual(first, second)
        self.assertEqual(first[0][0], "model-00001-of-00003.safetensors")

    def test_swapped_weight_map_shard_assignments_fail(self) -> None:
        source, _, output = self._convert_bounded_q6("swapped-map-output")
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        by_shard: dict[str, list[str]] = {}
        for name, shard in index["weight_map"].items():
            by_shard.setdefault(shard, []).append(name)
        first_shard, second_shard = sorted(by_shard)[:2]
        first_name = by_shard[first_shard][0]
        second_name = by_shard[second_shard][0]
        index["weight_map"][first_name] = second_shard
        index["weight_map"][second_name] = first_shard
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")

        for linked_source in (None, source):
            with self.subTest(source_linked=linked_source is not None):
                with self.assertRaisesRegex(MonolithicSourceError, "physically present"):
                    verify_output(output, source=linked_source)

    def test_weight_map_nonexistent_shard_fails(self) -> None:
        _, _, output = self._convert_bounded_q6("missing-map-shard-output")
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        first_name = next(iter(index["weight_map"]))
        index["weight_map"][first_name] = "model-99999-of-99999.safetensors"
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(MonolithicSourceError, "nonexistent shards"):
            verify_output(output)

    def test_duplicate_physical_tensor_across_shards_fails(self) -> None:
        _, _, output = self._convert_bounded_q6("duplicate-physical-output")
        index = json.loads((output / "model.safetensors.index.json").read_text())
        first_shard, second_shard = sorted(set(index["weight_map"].values()))[:2]
        first_descriptor = read_safetensors_header(output / first_shard).tensors[0]
        second_header = read_safetensors_header(output / second_shard)
        tensors = {
            descriptor.name: (
                descriptor.dtype,
                descriptor.shape,
                b"\0" * descriptor.nbytes,
            )
            for descriptor in second_header.tensors
        }
        tensors[first_descriptor.name] = (
            first_descriptor.dtype,
            first_descriptor.shape,
            b"\0" * first_descriptor.nbytes,
        )
        write_raw_safetensors(output / second_shard, tensors, {"format": "mlx"})
        with self.assertRaisesRegex(MonolithicSourceError, "duplicated across shards"):
            verify_output(output)

    def test_unindexed_safetensors_shard_fails(self) -> None:
        _, _, output = self._convert_bounded_q6("unexpected-shard-output")
        write_raw_safetensors(
            output / "model-unexpected.safetensors",
            {"unexpected": ("BF16", (1,), b"\0\0")},
            {"format": "mlx"},
        )
        with self.assertRaisesRegex(MonolithicSourceError, "unexpected safetensors shard"):
            verify_output(output)

    def test_non_string_weight_map_value_fails_cleanly(self) -> None:
        _, _, output = self._convert_bounded_q6("malformed-map-output")
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        first_name = next(iter(index["weight_map"]))
        index["weight_map"][first_name] = ["not", "a", "shard"]
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(MonolithicSourceError, "string pairs"):
            verify_output(output)

    def test_quant_config_contract(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        plan = build_conversion_plan(MonolithicSafetensorsSource(path))
        self.assertEqual(
            plan.quant_config,
            {
                "bits": 6,
                "group_size": 64,
                "quantize_adaln": True,
                "adaln_bits": 8,
                "quantized_layers": {"6": 208, "8": 50},
            },
        )

    def test_no_whole_file_mlx_load_fallback_exists(self) -> None:
        source_text = inspect.getsource(monolithic_quant)
        self.assertNotIn("mx.load", source_text)
        self.assertNotIn("load_dit", source_text)

    def test_bounded_selector_writes_only_authorized_output_and_verifies(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(source, selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",))
        fake = FakeQuantizer(plan)
        output = self.temp / "bounded-output"
        receipt = convert(plan, output, target_shard_bytes=300, quantizer=fake)
        self.assertTrue(receipt.output.exists())
        self.assertEqual(receipt.output_tensor_count, 3)
        self.assertEqual(receipt.payload_bytes_read, source.descriptor("blocks.0.attn.qkv_proj.weight").nbytes)
        self.assertEqual(fake.releases, 1)
        output_only = verify_output(output)
        self.assertEqual(output_only.tensor_count, 3)
        self.assertFalse(output_only.source_checked)
        self.assertEqual(verify_output(output, source=source).tensor_count, 3)
        index = json.loads((output / "model.safetensors.index.json").read_text())
        self.assertEqual(set(index["weight_map"]), {
            "blocks.0.attn.qkv_proj.weight",
            "blocks.0.attn.qkv_proj.scales",
            "blocks.0.attn.qkv_proj.biases",
        })

    def test_incomplete_bounded_output_cannot_be_relabelled_full(self) -> None:
        source, _, output = self._convert_bounded_q6("forged-incomplete-full-output")
        full_plan = build_conversion_plan(source)
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["metadata"]["bounded"] = False
        index["metadata"]["selected_quantized_weights"] = list(
            full_plan.selected_quantized_weights
        )
        index["metadata"]["quantized_layers"] = full_plan.quantized_counts
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        quant_config_path = output / "quant_config.json"
        quant_config = json.loads(quant_config_path.read_text())
        quant_config["quantized_layers"] = full_plan.quantized_counts
        quant_config_path.write_text(json.dumps(quant_config, indent=2, sort_keys=True) + "\n")

        with self.assertRaisesRegex(MonolithicSourceError, "exact canonical conversion topology"):
            verify_output(output)

    def test_output_only_bounded_verification_requires_selected_topology(self) -> None:
        _, _, output = self._convert_bounded_q6("forged-bounded-selection-output")
        index_path = output / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        index["metadata"]["selected_quantized_weights"] = [
            "blocks.1.attn.qkv_proj.weight"
        ]
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        with self.assertRaisesRegex(MonolithicSourceError, "exact canonical conversion topology"):
            verify_output(output)

    def test_bounded_learned_bias_is_read_once_and_preserved(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(source, selected_quantized_weights=("blocks.0.adaln_proj.linear.weight",))
        fake = FakeQuantizer(plan)
        output = self.temp / "bounded-adaln"
        convert(plan, output, target_shard_bytes=300, quantizer=fake)
        expected_read = source.descriptor("blocks.0.adaln_proj.linear.weight").nbytes + source.descriptor("blocks.0.adaln_proj.linear.bias").nbytes
        self.assertEqual(source.payload_bytes_read, expected_read)
        self.assertIsNotNone(fake.calls[0][1])
        self.assertEqual(len(fake.calls[0][1] or b""), source.descriptor("blocks.0.adaln_proj.linear.bias").nbytes)
        self.assertEqual(verify_output(output, source=source).tensor_count, 4)

    def test_source_output_overlap_and_overwrite_are_refused(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(source, selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",))
        with self.assertRaises(MonolithicSourceError):
            convert(plan, self.temp, quantizer=FakeQuantizer(plan))
        existing = self.temp / "existing"
        existing.mkdir()
        with self.assertRaises(MonolithicSourceError):
            convert(plan, existing, quantizer=FakeQuantizer(plan))
        existing_file = self.temp / "existing-file"
        existing_file.write_bytes(b"do-not-overwrite")
        with self.assertRaises(MonolithicSourceError):
            convert(plan, existing_file, quantizer=FakeQuantizer(plan))
        self.assertEqual(list(self.temp.glob(".existing.incomplete-*")), [])

    def test_existing_output_symlinks_including_dangling_fail_closed(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(
            source,
            selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",),
        )
        live_target = self.temp / "live-target"
        live_target.mkdir()
        live_link = self.temp / "live-link"
        live_link.symlink_to(live_target, target_is_directory=True)
        dangling_link = self.temp / "dangling-link"
        dangling_link.symlink_to(self.temp / "missing-target", target_is_directory=True)

        for output in (live_link, dangling_link):
            with self.subTest(output=output.name):
                with self.assertRaisesRegex(MonolithicSourceError, "symlink"):
                    convert(plan, output, quantizer=FakeQuantizer(plan))
                self.assertTrue(output.is_symlink())
                self.assertEqual(list(self.temp.glob(f".{output.name}.incomplete-*")), [])

    def test_cli_dry_run_does_not_create_output(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        output = self.temp / "cli-output"
        result = converter_main([
            "--source", str(path),
            "--output", str(output),
            "--dry-run",
            "--tensor", "blocks.0.attn.qkv_proj.weight",
        ])
        self.assertEqual(result, 0)
        self.assertFalse(output.exists())

    def test_cli_duplicate_tensor_selector_fails_before_source_construction(self) -> None:
        stderr = StringIO()
        selector = "blocks.0.attn.qkv_proj.weight"
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                converter_main([
                    "--source", str(self.temp / "must-not-be-opened.safetensors"),
                    "--output", str(self.temp / "duplicate-selector-output"),
                    "--tensor", selector,
                    "--tensor", selector,
                ])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("duplicate --tensor selector", stderr.getvalue())

    def test_cli_tensor_and_full_scopes_are_mutually_exclusive(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                converter_main([
                    "--source", str(self.temp / "must-not-be-opened.safetensors"),
                    "--output", str(self.temp / "ambiguous-scope-output"),
                    "--tensor", "blocks.0.attn.qkv_proj.weight",
                    "--full",
                ])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("exactly one scope", stderr.getvalue())

    def test_cli_neither_scope_fails_before_source_construction(self) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as raised:
                converter_main([
                    "--source", str(self.temp / "must-not-be-opened.safetensors"),
                    "--output", str(self.temp / "missing-scope-output"),
                ])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("exactly one scope", stderr.getvalue())

    def test_cli_explicit_full_dry_run_plans_1050_without_payload_reads(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        output = self.temp / "full-dry-run-output"
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = converter_main([
                "--source", str(path),
                "--output", str(output),
                "--dry-run",
                "--full",
            ])
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertFalse(receipt["bounded"])
        self.assertEqual(receipt["source_tensor_count"], 535)
        self.assertEqual(receipt["output_tensor_count"], 1050)
        self.assertEqual(receipt["payload_bytes_read"], 0)
        self.assertFalse(output.exists())

    def test_cli_verify_remains_scope_free(self) -> None:
        path, _ = write_complete_fixture(self.temp)
        source = MonolithicSafetensorsSource(path)
        plan = build_conversion_plan(
            source,
            selected_quantized_weights=("blocks.0.attn.qkv_proj.weight",),
        )
        output = self.temp / "cli-verify-output"
        write_planned_output(output, plan)
        stdout = StringIO()
        with redirect_stdout(stdout):
            result = converter_main(["--output", str(output), "--verify"])
        receipt = json.loads(stdout.getvalue())
        self.assertEqual(result, 0)
        self.assertEqual(receipt["status"], "VERIFIED")
        self.assertTrue(receipt["bounded"])
        self.assertFalse(receipt["source_checked"])


if __name__ == "__main__":
    unittest.main()

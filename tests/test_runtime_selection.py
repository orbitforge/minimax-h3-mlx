"""MLX-free admission tests for the explicit beta-0.6 runtime profile."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import minimax_h3_mlx.runtime_selection as runtime_selection
from minimax_h3_mlx.runtime_selection import RuntimeSelectionError, resolve_runtime


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mutate_json(path: Path, mutate) -> None:
    value = json.loads(path.read_text())
    mutate(value)
    write_json(path, value)


def load_generate_module():
    module_name = "slice026_generate_test_module"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / "generate.py")
    if spec is None or spec.loader is None:
        raise AssertionError("could not load generate.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class BetaProfileFixture:
    """Small metadata-only profile with the accepted beta semantic contract."""

    def __init__(self, root: Path):
        self.root = root
        self.assets = root / "assets"
        self.profile = self.assets / "beta-0.6"
        self.checkpoint = root / "checkpoint"
        self.conventional = root / "conventional"
        self.transformer = root / "transformer"
        self._build_conventional()
        self._build_checkpoint()
        self._build_transformer()
        self.profile.mkdir(parents=True)
        os.symlink(self.checkpoint, self.profile / "checkpoint")
        os.symlink(self.transformer, self.profile / "transformer")
        os.symlink(self.conventional, self.profile / "conventional")

    def _build_conventional(self) -> None:
        self.conventional.mkdir(parents=True)
        source_identity = "safetensors:test-beta-source:fixture"
        metadata = {
            "source_size": runtime_selection.BETA_SOURCE_BYTES,
            "source_identity": source_identity,
            "qkv_source_layout": runtime_selection.BETA_QKV_SOURCE_LAYOUT,
            "qkv_canonical_layout": runtime_selection.BETA_QKV_CANONICAL_LAYOUT,
            "qkv_row_reconciliation_applied": True,
            "qkv_tensors_reconciled": runtime_selection.BETA_QKV_TENSOR_COUNT,
            "quantized_layers": dict(runtime_selection.BETA_QUANTIZED_LAYER_COUNTS),
            "qkv_layout_authorization": (
                f"payload_receipt:{runtime_selection.BETA_QKV_RECEIPT_SCHEMA};"
                f"receipt_sha256={runtime_selection.BETA_QKV_RECEIPT_SHA256};"
                f"source_sha256={runtime_selection.BETA_SOURCE_SHA256};"
                f"source_identity={source_identity};"
                f"layout={runtime_selection.BETA_QKV_SOURCE_LAYOUT}"
            ),
        }
        weight_map = {
            f"fixture.tensor.{index:04d}": f"model-{index % 5 + 1:05d}-of-00005.safetensors"
            for index in range(runtime_selection.BETA_LOGICAL_TENSOR_COUNT)
        }
        write_json(
            self.conventional / "model.safetensors.index.json",
            {"metadata": metadata, "weight_map": weight_map},
        )
        write_json(
            self.conventional / "config.json",
            {
                "adaln_out_features": 96768,
                "attention_head_dim": 128,
                "audio_latents_dim": 32,
                "ffn_hidden_size": 14336,
                "final_adaln_out_features": 10752,
                "final_norm_eps": 1e-5,
                "hidden_size": 5376,
                "image_model": "minimax_h3",
                "latents_dim": 24,
                "norm_eps": 1e-5,
                "num_attention_heads": 56,
                "num_layers": 50,
                "patch_size": [1, 2, 2],
                "qk_norm_eps": 1e-5,
                "rope_inv_freq_len": 16,
                "text_dim": 5120,
                "time_embed_dim": 2688,
                "time_embed_hidden_size": 5376,
                "timestep_input_dim": 256,
                "token_refiner_num_layers": 2,
            },
        )
        write_json(
            self.conventional / "quant_config.json",
            {"bits": 6, "adaln_bits": 8, "group_size": 64, "quantize_adaln": True},
        )
        self.index_sha256 = sha256_file(self.conventional / "model.safetensors.index.json")
        self.config_sha256 = sha256_file(self.conventional / "config.json")
        self.quant_config_sha256 = sha256_file(self.conventional / "quant_config.json")

    def _build_checkpoint(self) -> None:
        self.checkpoint.mkdir(parents=True)
        write_json(
            self.checkpoint / "model_index.json",
            {
                "_class_name": "MiniMaxH3Pipeline",
                "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
                "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
                "scheduler": None,
                "processor": ["transformers", "Qwen3VLProcessor"],
                "_minimax_h3": {
                    "schema_version": 1,
                    "partition": "fl2va",
                    "tasks": ["t2va", "fl2va"],
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                },
            },
        )

        qwen = self.checkpoint / "text_encoder"
        qwen.mkdir()
        qwen_quant = {
            "bits": runtime_selection.BETA_QWEN_QUANT_BITS,
            "group_size": runtime_selection.BETA_QWEN_QUANT_GROUP_SIZE,
            "mode": runtime_selection.BETA_QWEN_QUANT_MODE,
        }
        write_json(
            qwen / "config.json",
            {
                "model_type": runtime_selection.BETA_QWEN_MODEL_TYPE,
                "architectures": [runtime_selection.BETA_QWEN_ARCHITECTURE],
                "text_config": {
                    "hidden_size": runtime_selection.BETA_QWEN_TEXT_HIDDEN_SIZE,
                    "num_hidden_layers": runtime_selection.BETA_QWEN_TEXT_LAYER_COUNT,
                },
                "quantization": qwen_quant,
                "quantization_config": qwen_quant,
            },
        )
        tokenizer_config = {
            "tokenizer_class": "Qwen2Tokenizer",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "added_tokens_decoder": {
                "151652": {"content": "<|vision_start|>"},
                "151653": {"content": "<|vision_end|>"},
                "151655": {"content": "<|image_pad|>"},
                "151656": {"content": "<|video_pad|>"},
            },
        }
        tokenizer = self.checkpoint / "tokenizer"
        tokenizer.mkdir()
        write_json(tokenizer / "tokenizer_config.json", tokenizer_config)
        write_json(tokenizer / "tokenizer.json", {"model": {"type": "BPE"}})
        write_json(tokenizer / "vocab.json", {})
        (tokenizer / "merges.txt").write_text("")

        processor = self.checkpoint / "processor"
        processor.mkdir()
        write_json(processor / "tokenizer_config.json", tokenizer_config)
        write_json(processor / "tokenizer.json", {"model": {"type": "BPE"}})
        write_json(processor / "vocab.json", {})
        (processor / "merges.txt").write_text("")
        write_json(
            processor / "preprocessor_config.json",
            {
                "size": {"longest_edge": 16_777_216, "shortest_edge": 65_536},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast",
            },
        )
        write_json(
            processor / "video_preprocessor_config.json",
            {
                "size": {"longest_edge": 25_165_824, "shortest_edge": 4_096},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "video_processor_type": "Qwen3VLVideoProcessor",
            },
        )
        write_json(processor / "chat_template.json", {"chat_template": "accepted-qwen3-vl-template"})

        video = self.checkpoint / "video_vae"
        (video / "source").mkdir(parents=True)
        write_json(
            video / "config.json",
            {"_class_name": "MiniMaxH3VideoVAE", "latent_channels": runtime_selection.BETA_VIDEO_LATENT_CHANNELS},
        )
        write_json(
            video / "source" / "config.json",
            {
                "z_channels": runtime_selection.BETA_VIDEO_LATENT_CHANNELS,
                "vae_ratio": runtime_selection.BETA_VIDEO_SPATIAL_RATIO,
                "vae_ratio_t": runtime_selection.BETA_VIDEO_TEMPORAL_RATIO,
            },
        )

        audio = self.checkpoint / "audio_vae"
        audio.mkdir()
        write_json(
            audio / "config.json",
            {
                "_class_name": "MiniMaxH3AudioVAE",
                "sample_rate": runtime_selection.BETA_AUDIO_SAMPLE_RATE,
                "latent_channels": runtime_selection.BETA_AUDIO_LATENT_CHANNELS,
            },
        )
        write_json(
            audio / "metadata.json",
            {
                "metadata": {
                    "kwargs": {
                        "sample_rate": runtime_selection.BETA_AUDIO_SAMPLE_RATE,
                        "vae_latent_channels": runtime_selection.BETA_AUDIO_LATENT_CHANNELS,
                    }
                }
            },
        )

    def _build_transformer(self) -> None:
        self.transformer.mkdir(parents=True)
        write_json(
            self.transformer / "config.json",
            {
                "adaln_out_features": 96768,
                "attention_head_dim": 128,
                "audio_latents_dim": 32,
                "ffn_hidden_size": 14336,
                "final_adaln_out_features": 10752,
                "final_norm_eps": 1e-5,
                "hidden_size": 5376,
                "image_model": "minimax_h3",
                "latents_dim": 24,
                "norm_eps": 1e-5,
                "num_attention_heads": 56,
                "num_layers": 50,
                "patch_size": [1, 2, 2],
                "qk_norm_eps": 1e-5,
                "rope_inv_freq_len": 16,
                "text_dim": 5120,
                "time_embed_dim": 2688,
                "time_embed_hidden_size": 5376,
                "timestep_input_dim": 256,
                "token_refiner_num_layers": 2,
            },
        )
        write_json(
            self.transformer / "conversion_manifest.json",
            {
                "format_identifier": runtime_selection.BETA_STREAMED_FORMAT,
                "schema_version": runtime_selection.BETA_STREAMED_SCHEMA_VERSION,
                "bounded": False,
                "verification_status": "verified",
                "selected_blocks": list(range(runtime_selection.BETA_BLOCK_COUNT)),
                "source_checkpoint": {"logical_identity": self.index_sha256},
                "source_safetensors_index_checksum": self.index_sha256,
                "source_configuration_checksum": self.config_sha256,
                "source_quantization_configuration_checksum": self.quant_config_sha256,
                "source_tensor_count": runtime_selection.BETA_LOGICAL_TENSOR_COUNT,
                "derived_base_tensor_count": runtime_selection.BETA_RESIDENT_TENSOR_COUNT,
                "sidecar_count": runtime_selection.BETA_SIDECAR_COUNT,
                "sidecar_tensor_count": runtime_selection.BETA_SIDECAR_TENSOR_COUNT,
                "total_logical_tensor_count": runtime_selection.BETA_LOGICAL_TENSOR_COUNT,
                "original_checkpoint_modified": False,
            },
        )
        write_json(
            self.transformer / "quant_config.json",
            {
                "bits": runtime_selection.BETA_CORE_BITS,
                "adaln_bits": runtime_selection.BETA_ADALN_BITS,
                "group_size": runtime_selection.BETA_GROUP_SIZE,
                "quantize_adaln": True,
                "quantized_layers": dict(runtime_selection.BETA_QUANTIZED_LAYER_COUNTS),
            },
        )

        base = self.transformer / "base"
        base.mkdir()
        shard_names = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
        weight_map = {
            f"fixture.base.{index:04d}": shard_names[index % 5]
            for index in range(runtime_selection.BETA_RESIDENT_TENSOR_COUNT - 2)
        }
        weight_map["final_layer.adaln_proj.linear.bias"] = shard_names[0]
        weight_map["final_layer.adaln_proj.linear.weight"] = shard_names[1]
        write_json(base / "model.safetensors.index.json", {"weight_map": weight_map})
        for shard in shard_names:
            (base / shard).touch()

        adaln = self.transformer / "adaln"
        adaln.mkdir()
        projection = {
            "quantization_bits": runtime_selection.BETA_ADALN_BITS,
            "quantization_group_size": runtime_selection.BETA_GROUP_SIZE,
            "logical_input_features": 2688,
            "logical_output_features": 96768,
            "packed_weight_shape": [96768, 672],
            "scales_shape": [96768, 42],
            "quantization_biases_shape": [96768, 42],
            "learned_bias_shape": [96768],
        }
        role_contract = {
            "bias": ("learned_bias", "BF16", "unquantized", None, None),
            "biases": ("quantization_biases", "BF16", "affine", 8, 64),
            "scales": ("scales", "BF16", "affine", 8, 64),
            "weight": ("packed_weight", "U32", "affine", 8, 64),
        }
        blocks = {}
        for index in range(runtime_selection.BETA_BLOCK_COUNT):
            tensors = []
            for suffix in ("bias", "biases", "scales", "weight"):
                role, dtype, quant_format, bits, group_size = role_contract[suffix]
                tensors.append(
                    {
                        "tensor_key": f"blocks.{index}.adaln_proj.linear.{suffix}",
                        "tensor_role": role,
                        "source_dtype": dtype,
                        "quantization_format": quant_format,
                        "quantization_bits": bits,
                        "group_size": group_size,
                    }
                )
            blocks[str(index)] = {
                "block_index": index,
                "sidecar_filename": f"block-{index:03d}.safetensors",
                "projection": projection,
                "tensors": tensors,
            }
            (adaln / f"block-{index:03d}.safetensors").touch()
        write_json(
            adaln / "manifest.json",
            {
                "format_identifier": runtime_selection.BETA_STREAMED_FORMAT,
                "schema_version": runtime_selection.BETA_STREAMED_SCHEMA_VERSION,
                "bounded": False,
                "blocks": blocks,
            },
        )


class RuntimeSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="runtime-selection-test-"))
        self.fixture = BetaProfileFixture(self.temp)
        self.constants = mock.patch.multiple(
            runtime_selection,
            BETA_CONVENTIONAL_INDEX_SHA256=self.fixture.index_sha256,
            BETA_CONVENTIONAL_CONFIG_SHA256=self.fixture.config_sha256,
            BETA_CONVENTIONAL_QUANT_CONFIG_SHA256=self.fixture.quant_config_sha256,
        )
        self.constants.start()

    def tearDown(self) -> None:
        self.constants.stop()
        shutil.rmtree(self.temp)

    def resolve(self):
        return resolve_runtime("beta-0.6", self.fixture.assets)

    def test_accepted_beta_resolution(self) -> None:
        resolved = self.resolve()
        self.assertEqual(resolved.runtime_id, "beta-0.6")
        self.assertEqual(resolved.checkpoint_root, self.fixture.checkpoint.resolve())
        self.assertEqual(resolved.transformer_root, self.fixture.transformer.resolve())
        self.assertEqual(resolved.conventional_root, self.fixture.conventional.resolve())
        self.assertEqual(resolved.surrounding_identity["scheduler"]["video_shift"], 12.0)
        self.assertEqual(resolved.surrounding_identity["scheduler"]["audio_shift"], 3.0)
        self.assertEqual(resolved.surrounding_identity["qwen"]["model_type"], "qwen3_vl")
        self.assertEqual(resolved.surrounding_identity["tokenizer"]["config"]["tokenizer_class"], "Qwen2Tokenizer")
        self.assertEqual(resolved.surrounding_identity["processor"]["image_config"]["processor_class"], "Qwen3VLProcessor")
        self.assertEqual(resolved.surrounding_identity["video_vae"]["latent_channels"], 24)
        self.assertEqual(resolved.surrounding_identity["audio_vae"]["sample_rate"], 32_000)

    def test_accepted_transformer_config_is_parsed_and_normalized(self) -> None:
        resolved = self.resolve()
        identity = resolved.transformer_identity
        self.assertEqual(identity["config_path"], str((self.fixture.transformer / "config.json").resolve()))
        self.assertEqual(identity["config_sha256"], sha256_file(self.fixture.transformer / "config.json"))
        self.assertEqual(identity["config_fields"]["num_layers"], 50)
        self.assertEqual(identity["config_fields"]["hidden_size"], 5376)
        self.assertEqual(identity["config_fields"]["patch_size"], [1, 2, 2])
        self.assertEqual(identity["config_fields"]["rope_theta"], 10000.0)
        self.assertEqual(identity["defaults_applied"], {"rope_theta": 10000.0})
        self.assertEqual(identity["derived_contract"]["inner_dim"], 7168)
        self.assertEqual(identity["derived_contract"]["video_patch_dim"], 96)
        self.assertEqual(identity["derived_contract"]["rotary_dim"], 96)

    def test_transformer_root_and_validated_config_are_the_same_target(self) -> None:
        resolved = self.resolve()
        self.assertEqual(
            Path(resolved.transformer_identity["config_path"]),
            resolved.transformer_root / "config.json",
        )

    def test_accepted_streamed_manifest_semantics(self) -> None:
        identity = self.resolve().transformer_identity
        self.assertEqual(identity["format"], "minimax-h3-mlx-streamed-adaln-v1")
        self.assertEqual(identity["source_sha256"], runtime_selection.BETA_SOURCE_SHA256)
        self.assertEqual(identity["qkv_source_layout"], "grouped_qkv")
        self.assertEqual(identity["qkv_canonical_layout"], "runtime_interleaved")
        self.assertEqual(identity["qkv_tensors_reconciled"], 52)
        self.assertEqual(identity["qkv_authorization_receipt_sha256"], runtime_selection.BETA_QKV_RECEIPT_SHA256)
        self.assertEqual(identity["core_bits"], 6)
        self.assertEqual(identity["adaln_bits"], 8)
        self.assertEqual(identity["group_size"], 64)
        self.assertEqual(identity["logical_tensor_count"], 1_050)
        self.assertEqual(identity["resident_tensor_count"], 850)
        self.assertEqual(identity["resident_block_adaln_tensor_count"], 0)
        self.assertEqual(identity["sidecar_count"], 50)
        self.assertEqual(identity["sidecar_tensor_count"], 200)

    def test_wrong_source_identity_fails(self) -> None:
        mutate_json(
            self.fixture.transformer / "conversion_manifest.json",
            lambda value: value["source_checkpoint"].update({"logical_identity": "wrong-source"}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "logical linkage"):
            self.resolve()

    def test_wrong_qkv_contract_fails(self) -> None:
        path = self.fixture.conventional / "model.safetensors.index.json"
        mutate_json(path, lambda value: value["metadata"].update({"qkv_source_layout": "runtime_interleaved"}))
        runtime_selection.BETA_CONVENTIONAL_INDEX_SHA256 = sha256_file(path)
        with self.assertRaisesRegex(RuntimeSelectionError, "QKV source layout"):
            self.resolve()

    def test_wrong_qkv_count_fails(self) -> None:
        path = self.fixture.conventional / "model.safetensors.index.json"
        mutate_json(path, lambda value: value["metadata"].update({"qkv_tensors_reconciled": 51}))
        runtime_selection.BETA_CONVENTIONAL_INDEX_SHA256 = sha256_file(path)
        with self.assertRaisesRegex(RuntimeSelectionError, "reconciled QKV count"):
            self.resolve()

    def test_wrong_streamed_topology_fails(self) -> None:
        mutate_json(
            self.fixture.transformer / "conversion_manifest.json",
            lambda value: value.update({"derived_base_tensor_count": 849}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "resident tensor count"):
            self.resolve()

    def test_wrong_quantization_contract_fails(self) -> None:
        mutate_json(
            self.fixture.transformer / "quant_config.json",
            lambda value: value.update({"adaln_bits": 6}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "AdaLN quantization"):
            self.resolve()

    def test_missing_transformer_config_fails_before_pipeline_loading(self) -> None:
        (self.fixture.transformer / "config.json").unlink()
        with self.assertRaisesRegex(RuntimeSelectionError, "streamed transformer config is missing"):
            self.resolve()

    def test_malformed_transformer_config_fails_before_pipeline_loading(self) -> None:
        (self.fixture.transformer / "config.json").write_text("{not-json")
        with self.assertRaisesRegex(RuntimeSelectionError, "streamed transformer config is not valid JSON"):
            self.resolve()

    def test_architecture_changing_transformer_config_fields_fail_closed(self) -> None:
        for field, value in (
            ("num_layers", 49),
            ("hidden_size", 5375),
            ("num_attention_heads", 55),
            ("attention_head_dim", 64),
            ("ffn_hidden_size", 14335),
            ("latents_dim", 23),
            ("audio_latents_dim", 31),
            ("patch_size", [1, 1, 2]),
            ("text_dim", 5119),
            ("adaln_out_features", 1),
            ("final_adaln_out_features", 1),
            ("rope_inv_freq_len", 8),
        ):
            with self.subTest(field=field):
                mutate_json(self.fixture.transformer / "config.json", lambda raw, f=field, v=value: raw.update({f: v}))
                with self.assertRaisesRegex(RuntimeSelectionError, f"field {field!r}"):
                    self.resolve()
                write_json(
                    self.fixture.transformer / "config.json",
                    {
                        "adaln_out_features": 96768,
                        "attention_head_dim": 128,
                        "audio_latents_dim": 32,
                        "ffn_hidden_size": 14336,
                        "final_adaln_out_features": 10752,
                        "final_norm_eps": 1e-5,
                        "hidden_size": 5376,
                        "image_model": "minimax_h3",
                        "latents_dim": 24,
                        "norm_eps": 1e-5,
                        "num_attention_heads": 56,
                        "num_layers": 50,
                        "patch_size": [1, 2, 2],
                        "qk_norm_eps": 1e-5,
                        "rope_inv_freq_len": 16,
                        "text_dim": 5120,
                        "time_embed_dim": 2688,
                        "time_embed_hidden_size": 5376,
                        "timestep_input_dim": 256,
                        "token_refiner_num_layers": 2,
                    },
                )

    def test_tokenizer_identity_is_checked(self) -> None:
        mutate_json(
            self.fixture.checkpoint / "tokenizer" / "tokenizer_config.json",
            lambda value: value.update({"tokenizer_class": "WrongTokenizer"}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "tokenizer class"):
            self.resolve()

    def test_processor_identity_is_checked(self) -> None:
        mutate_json(
            self.fixture.checkpoint / "processor" / "preprocessor_config.json",
            lambda value: value.update({"processor_class": "WrongProcessor"}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "processor class"):
            self.resolve()

    def test_model_index_tokenizer_and_processor_mappings_are_checked(self) -> None:
        mutate_json(
            self.fixture.checkpoint / "model_index.json",
            lambda value: value.update({"tokenizer": ["transformers", "WrongTokenizer"]}),
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "tokenizer contract"):
            self.resolve()

        write_json(
            self.fixture.checkpoint / "model_index.json",
            {
                "_class_name": "MiniMaxH3Pipeline",
                "text_encoder": ["transformers", "MiniMaxH3Qwen3VLHFEncoder"],
                "tokenizer": ["transformers", "Qwen2TokenizerFast"],
                "video_vae": ["diffusers", "MiniMaxH3VideoVAE"],
                "audio_vae": ["diffusers", "MiniMaxH3AudioVAE"],
                "scheduler": None,
                "processor": ["transformers", "WrongProcessor"],
                "_minimax_h3": {
                    "schema_version": 1,
                    "partition": "fl2va",
                    "tasks": ["t2va", "fl2va"],
                    "sigma_shift_scales": {"video": 12.0, "audio": 3.0},
                },
            },
        )
        with self.assertRaisesRegex(RuntimeSelectionError, "processor contract"):
            self.resolve()

    def test_missing_transformer_fails_clearly(self) -> None:
        (self.fixture.profile / "transformer").unlink()
        with self.assertRaisesRegex(RuntimeSelectionError, "RUNTIME_ASSET_MISSING"):
            self.resolve()

    def test_profile_entries_must_be_non_copying_symbolic_links(self) -> None:
        (self.fixture.profile / "checkpoint").unlink()
        shutil.copytree(self.fixture.checkpoint, self.fixture.profile / "checkpoint")
        with self.assertRaisesRegex(RuntimeSelectionError, "must be a symbolic link"):
            self.resolve()

    def test_unknown_profile_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeSelectionError, "UNKNOWN_RUNTIME"):
            resolve_runtime("beta-0.6-unknown", self.fixture.assets)

    def test_resolved_receipt_reports_actual_assets(self) -> None:
        receipt = self.resolve().receipt
        self.assertEqual(receipt["runtime_id"], "beta-0.6")
        self.assertEqual(Path(receipt["checkpoint_root"]), self.fixture.checkpoint.resolve())
        self.assertEqual(Path(receipt["transformer_root"]), self.fixture.transformer.resolve())
        self.assertEqual(receipt["transformer"]["source_sha256"], runtime_selection.BETA_SOURCE_SHA256)
        self.assertEqual(receipt["transformer"]["config_path"], str((self.fixture.transformer / "config.json").resolve()))
        self.assertEqual(receipt["transformer"]["config_sha256"], sha256_file(self.fixture.transformer / "config.json"))
        self.assertEqual(receipt["transformer"]["qkv_tensors_reconciled"], 52)
        self.assertEqual(receipt["qwen"]["path"], str((self.fixture.checkpoint / "text_encoder").resolve()))
        self.assertEqual(receipt["tokenizer"]["path"], str((self.fixture.checkpoint / "tokenizer").resolve()))
        self.assertEqual(receipt["processor"]["path"], str((self.fixture.checkpoint / "processor").resolve()))
        self.assertFalse(receipt["explicit_override_used"])

    def test_cli_conflicting_override_fails_before_pipeline_loader(self) -> None:
        module = load_generate_module()
        argv = [
            "generate.py",
            "prompt",
            "--runtime",
            "beta-0.6",
            "--runtime-assets",
            str(self.fixture.assets),
            "--transformer",
            str(self.temp / "other-transformer"),
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(module, "_load_pipeline_class") as load_pipeline:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        module.main()
        self.assertEqual(raised.exception.code, 2)
        load_pipeline.assert_not_called()

    @staticmethod
    def _fake_pipeline(calls):
        class FakePipeline:
            @classmethod
            def from_pretrained(cls, checkpoint, **kwargs):
                calls["checkpoint"] = checkpoint
                calls["transformer"] = kwargs["transformer_dir"]
                return cls()

            def __call__(self, *args, **kwargs):
                return SimpleNamespace(
                    video=__import__("numpy").zeros((1, 2, 2, 3), dtype="uint8"),
                    fps=1,
                    audio=__import__("numpy").zeros((1, 4), dtype="float32"),
                    sample_rate=32_000,
                    seconds_per_step=0.1,
                    total_seconds=0.1,
                )

        return FakePipeline

    def test_explicit_runtime_assets_without_named_runtime_still_rejects(self) -> None:
        module = load_generate_module()
        argv = [
            "generate.py",
            "prompt",
            "--runtime-assets",
            str(self.fixture.assets),
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(module, "_load_pipeline_class") as load_pipeline:
                with contextlib.redirect_stderr(io.StringIO()) as stderr:
                    with self.assertRaises(SystemExit) as raised:
                        module.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--runtime-assets requires --runtime", stderr.getvalue())
        load_pipeline.assert_not_called()

    def test_ambient_runtime_assets_without_named_runtime_stays_manual(self) -> None:
        module = load_generate_module()
        calls = {}
        argv = [
            "generate.py",
            "ambient manual prompt",
            "--output",
            str(self.temp / "ambient-manual.mp4"),
        ]
        with mock.patch.dict(module.os.environ, {module.RUNTIME_ASSETS_ENV: str(self.fixture.assets)}):
            with mock.patch.object(module, "resolve_runtime", side_effect=AssertionError("named runtime selected")) as resolve:
                with mock.patch.object(module, "_load_pipeline_class", return_value=self._fake_pipeline(calls)):
                    with mock.patch.object(module, "save_mp4"):
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(module.main(), 0)
        resolve.assert_not_called()
        self.assertEqual(calls["checkpoint"], module.DEFAULT_CHECKPOINT)
        self.assertIsNone(calls["transformer"])

    def test_legacy_manual_workflow_is_not_forced_through_beta_profile(self) -> None:
        module = load_generate_module()
        calls = {}
        fake_pipeline = self._fake_pipeline(calls)

        argv = [
            "generate.py",
            "legacy prompt",
            "--checkpoint",
            str(self.temp / "manual-checkpoint"),
            "--transformer",
            str(self.temp / "manual-transformer"),
            "--megapixels",
            "0.2",
            "--duration",
            "5",
            "--output",
            str(self.temp / "legacy.mp4"),
        ]
        with mock.patch.dict(module.os.environ, {module.RUNTIME_ASSETS_ENV: str(self.fixture.assets)}):
            with mock.patch.object(module, "resolve_runtime", side_effect=AssertionError("beta profile was selected")) as resolve:
                with mock.patch.object(module, "_load_pipeline_class", return_value=fake_pipeline):
                    with mock.patch.object(module, "save_mp4"):
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(module.main(), 0)
        resolve.assert_not_called()
        self.assertEqual(calls["checkpoint"], str(self.temp / "manual-checkpoint"))
        self.assertEqual(calls["transformer"], str(self.temp / "manual-transformer"))

    def test_named_runtime_consumes_ambient_runtime_assets(self) -> None:
        module = load_generate_module()
        calls = {}
        argv = [
            "generate.py",
            "ambient beta prompt",
            "--runtime",
            "beta-0.6",
            "--output",
            str(self.temp / "ambient-beta.mp4"),
        ]
        with mock.patch.dict(module.os.environ, {module.RUNTIME_ASSETS_ENV: str(self.fixture.assets)}):
            with mock.patch.object(module, "resolve_runtime", wraps=module.resolve_runtime) as resolve:
                with mock.patch.object(module, "_load_pipeline_class", return_value=self._fake_pipeline(calls)):
                    with mock.patch.object(module, "save_mp4"):
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(module.main(), 0)
        resolve.assert_called_once_with("beta-0.6", str(self.fixture.assets))
        self.assertEqual(calls["checkpoint"], str(self.fixture.checkpoint.resolve()))
        self.assertEqual(calls["transformer"], str(self.fixture.transformer.resolve()))

    def test_named_runtime_cli_assets_override_ambient_runtime_assets(self) -> None:
        module = load_generate_module()
        calls = {}
        explicit_assets = self.fixture.assets
        argv = [
            "generate.py",
            "explicit beta prompt",
            "--runtime",
            "beta-0.6",
            "--runtime-assets",
            str(explicit_assets),
            "--output",
            str(self.temp / "explicit-beta.mp4"),
        ]
        with mock.patch.dict(
            module.os.environ,
            {module.RUNTIME_ASSETS_ENV: str(self.temp / "ambient-assets-that-must-not-win")},
        ):
            with mock.patch.object(module, "resolve_runtime", wraps=module.resolve_runtime) as resolve:
                with mock.patch.object(module, "_load_pipeline_class", return_value=self._fake_pipeline(calls)):
                    with mock.patch.object(module, "save_mp4"):
                        with mock.patch.object(sys, "argv", argv):
                            with contextlib.redirect_stdout(io.StringIO()):
                                self.assertEqual(module.main(), 0)
        resolve.assert_called_once_with("beta-0.6", str(explicit_assets))
        self.assertEqual(calls["checkpoint"], str(self.fixture.checkpoint.resolve()))
        self.assertEqual(calls["transformer"], str(self.fixture.transformer.resolve()))

    def test_invalid_profile_is_rejected_before_heavy_loader(self) -> None:
        mutate_json(
            self.fixture.transformer / "conversion_manifest.json",
            lambda value: value.update({"total_logical_tensor_count": 999}),
        )
        module = load_generate_module()
        argv = [
            "generate.py",
            "prompt",
            "--runtime",
            "beta-0.6",
            "--runtime-assets",
            str(self.fixture.assets),
        ]
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(module, "_load_pipeline_class") as load_pipeline:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        module.main()
        self.assertEqual(raised.exception.code, 2)
        load_pipeline.assert_not_called()
        self.assertNotIn("minimax_h3_mlx.pipeline", sys.modules)

    def test_invalid_transformer_config_is_rejected_before_heavy_loader(self) -> None:
        mutate_json(
            self.fixture.transformer / "config.json",
            lambda value: value.update({"hidden_size": 1}),
        )
        module = load_generate_module()
        argv = [
            "generate.py",
            "prompt",
            "--runtime",
            "beta-0.6",
            "--runtime-assets",
            str(self.fixture.assets),
        ]
        sys.modules.pop("minimax_h3_mlx.pipeline", None)
        with mock.patch.object(sys, "argv", argv):
            with mock.patch.object(module, "_load_pipeline_class") as load_pipeline:
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        module.main()
        self.assertEqual(raised.exception.code, 2)
        load_pipeline.assert_not_called()
        self.assertNotIn("minimax_h3_mlx.pipeline", sys.modules)


if __name__ == "__main__":
    unittest.main()

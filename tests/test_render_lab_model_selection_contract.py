"""MLX-free contracts for Slice 029 Render Lab model selection."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from minimax_h3_mlx.checkpoint_format import inspect_checkpoint_format
from tools.render_lab.runner import (
    BETA_MODEL_ID,
    BETA_TRANSFORMER_NAME,
    CANONICAL_TRANSFORMER_MODE,
    CURRENT_MODEL_ID,
    DEFAULT_MODEL_ID,
    DEFAULT_TURBO_PRESET_NFE,
    DEFAULT_TURBO_PRESET_ID,
    FIRST_LAST,
    FL2V_STORYBOARD_SEGMENT_WORKFLOW,
    FL2V_STORYBOARD_WORKFLOW,
    RenderRequest,
    RenderController,
    RenderValidationError,
    build_generation_command,
    build_render_config,
    build_storyboard_config,
    execute_storyboard,
    initialize_run,
    model_transformer_path,
    model_selection_payload,
    reserve_storyboard_namespace,
    reserve_run_namespace,
    streamed_transformer_asset_available,
    validate_render_request,
    validate_storyboard_request,
    _validate_transformer_safety,
)
from tools.render_lab.server import PAGE, _render_request_from_fields
from tools.render_lab.turbo_presets import REFERENCE_TURBO_PRESET_ID, turbo_preset_by_id


ROOT = Path(__file__).resolve().parents[1]
CURRENT_TRANSFORMER_NAME = "minimax-h3-mlx-6bit-streamed-adaln"
CURRENT_TRANSFORMER_PATH = Path(
    "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/"
    "minimax-h3-mlx-6bit-streamed-adaln"
)
BETA_TRANSFORMER_PATH = Path(
    "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/"
    "minimax-h3-mlx-beta-0.6-q6-q8-corrected-slice-025-streamed-adaln"
)
LIGHTX_VARIANT = "fl2va-turbo-4step-v0.1"

VALID_TRANSFORMER_CONFIG = {
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": [1, 2, 2],
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}
VALID_QUANT_CONFIG = {
    "bits": 6,
    "group_size": 64,
    "quantize_adaln": True,
    "adaln_bits": 8,
    "quantized_layers": {"6": 208, "8": 50},
}
VALID_SIDECAR_PROJECTION = {
    "quantization_bits": 8,
    "quantization_group_size": 64,
    "logical_input_features": 2688,
    "logical_output_features": 96768,
    "packed_weight_shape": [96768, 672],
    "scales_shape": [96768, 42],
    "quantization_biases_shape": [96768, 42],
    "learned_bias_shape": [96768],
}
TEST_DTYPE_BYTES = {"BF16": 2, "U32": 4}
VALID_DERIVED_BASE_BYTE_COUNT = 16_464_048_640
VALID_SIDECAR_BYTE_COUNT = 13_828_147_200


def _independent_payload_byte_count(shape: list[int], dtype: str) -> int:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return element_count * TEST_DTYPE_BYTES[dtype]


def _request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "mode": "T2V",
        "prompt": "a bounded Render Lab contract prompt",
        "resolution_id": None,
        "width": 512,
        "height": 512,
        "steps": 4,
        "duration_seconds": 5.0,
        "seed": 1701,
        "output_root": root / "render-lab",
        "output_name": "render.mp4",
        "checkpoint_root": root / "checkpoint",
        "model_id": DEFAULT_MODEL_ID,
        "turbo_preset_id": REFERENCE_TURBO_PRESET_ID,
    }
    values.update(changes)
    return RenderRequest(**values)


def _validated(root: Path, **changes: object):
    return validate_render_request(
        _request(root, **changes),
        repo_root=ROOT,
        check_images=False,
        verify_runtime_geometry=False,
    )


def _cards(root: Path, count: int = 3) -> tuple[Path, ...]:
    cards = []
    for index in range(count):
        path = root / f"card-{index + 1}.png"
        Image.new("RGB", (4, 4), color=(index * 30, 20, 10)).save(path)
        cards.append(path)
    return tuple(cards)


def _write_complete_streamed_fixture(path: Path) -> None:
    """Write only the metadata/file structure consumed by the shared format inspection."""
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(VALID_TRANSFORMER_CONFIG) + "\n", encoding="utf-8")
    (path / "quant_config.json").write_text(json.dumps(VALID_QUANT_CONFIG) + "\n", encoding="utf-8")
    base = path / "base"
    base.mkdir()
    shards = [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)]
    keys = [f"ordinary.{index:03d}" for index in range(848)] + [
        "final_layer.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
    ]
    weight_map = {key: shards[index % len(shards)] for index, key in enumerate(keys)}
    (base / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {"total_size": VALID_DERIVED_BASE_BYTE_COUNT},
            "weight_map": weight_map,
        }) + "\n",
        encoding="utf-8",
    )
    for shard in shards:
        (base / shard).write_bytes(b"")

    adaln = path / "adaln"
    adaln.mkdir()
    blocks = {}
    sidecar_byte_count = 0
    for index in range(50):
        filename = f"block-{index:03d}.safetensors"
        (adaln / filename).write_bytes(b"")
        prefix = f"blocks.{index}.adaln_proj.linear."
        tensor_specs = (
            ("bias", "learned_bias", "BF16", [96768], "unquantized", None, None),
            ("biases", "quantization_biases", "BF16", [96768, 42], "affine", 8, 64),
            ("scales", "scales", "BF16", [96768, 42], "affine", 8, 64),
            ("weight", "packed_weight", "U32", [96768, 672], "affine", 8, 64),
        )
        tensors = []
        for suffix, role, dtype, shape, quantization_format, bits, group_size in tensor_specs:
            byte_count = _independent_payload_byte_count(shape, dtype)
            sidecar_byte_count += byte_count
            tensors.append({
                "tensor_key": prefix + suffix,
                "tensor_role": role,
                "source_shape": shape,
                "source_dtype": dtype,
                "byte_count": byte_count,
                "quantization_format": quantization_format,
                "quantization_bits": bits,
                "group_size": group_size,
            })
        blocks[str(index)] = {
            "block_index": index,
            "sidecar_filename": filename,
            "projection": VALID_SIDECAR_PROJECTION,
            "tensors": tensors,
        }
    (adaln / "manifest.json").write_text(
        json.dumps({
            "format_identifier": "minimax-h3-mlx-streamed-adaln-v1",
            "schema_version": 1,
            "bounded": False,
            "blocks": blocks,
        })
        + "\n",
        encoding="utf-8",
    )
    (path / "conversion_manifest.json").write_text(
        json.dumps({
            "format_identifier": "minimax-h3-mlx-streamed-adaln-v1",
            "schema_version": 1,
            "bounded": False,
            "derived_base_byte_count": VALID_DERIVED_BASE_BYTE_COUNT,
            "verification_status": "verified",
            "derived_base_tensor_count": 850,
            "total_logical_tensor_count": 1050,
            "sidecar_count": 50,
            "sidecar_tensor_count": 200,
            "sidecar_byte_count": sidecar_byte_count,
            "selected_blocks": list(range(50)),
        })
        + "\n",
        encoding="utf-8",
    )


class RenderLabModelSelectionContractTests(unittest.TestCase):
    def test_default_request_api_and_normalized_effective_defaults_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = RenderRequest(
                mode="T2V",
                prompt="default contract",
                width=512,
                height=512,
                output_root=root / "render-lab",
                checkpoint_root=root / "checkpoint",
            )
            self.assertEqual(request.model_id, BETA_MODEL_ID)
            self.assertEqual(request.turbo_preset_id, DEFAULT_TURBO_PRESET_ID)
            self.assertEqual(request.steps, DEFAULT_TURBO_PRESET_NFE)
            validated = validate_render_request(
                request,
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            self.assertEqual(validated.request.model_id, BETA_MODEL_ID)
            self.assertEqual(validated.request.turbo_preset_id, DEFAULT_TURBO_PRESET_ID)
            self.assertEqual(validated.request.steps, DEFAULT_TURBO_PRESET_NFE)
            self.assertEqual(validated.request.turbo_steps, DEFAULT_TURBO_PRESET_NFE)

        payload = RenderController(ROOT).config_payload()
        self.assertEqual(payload["defaults"]["model_id"], BETA_MODEL_ID)
        self.assertEqual(payload["defaults"]["turbo_preset_id"], DEFAULT_TURBO_PRESET_ID)
        self.assertEqual(payload["defaults"]["steps"], DEFAULT_TURBO_PRESET_NFE)
        self.assertEqual(payload["defaults"]["turbo_steps"], DEFAULT_TURBO_PRESET_NFE)

        server_request = _render_request_from_fields({
            "mode": "T2V",
            "prompt": "server omitted turbo fields",
            "steps": "4",
            "duration_seconds": "5",
            "seed": "0",
            "width": "512",
            "height": "512",
        })
        self.assertEqual(server_request.turbo_preset_id, DEFAULT_TURBO_PRESET_ID)
        self.assertIsNone(server_request.turbo_steps)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = validate_render_request(
                replace(server_request, output_root=root / "render-lab", checkpoint_root=root / "checkpoint"),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            self.assertEqual(validated.request.turbo_steps, DEFAULT_TURBO_PRESET_NFE)

    def test_real_current_and_beta_assets_pass_metadata_only_admission(self) -> None:
        expected = {
            CURRENT_MODEL_ID: CURRENT_TRANSFORMER_PATH,
            BETA_MODEL_ID: BETA_TRANSFORMER_PATH,
        }
        for model_id, path in expected.items():
            with self.subTest(model_id=model_id):
                self.assertTrue(path.is_dir(), path)
                available, reason = streamed_transformer_asset_available(path)
                self.assertTrue(available, reason)
                self.assertIsNone(reason)

    def test_structurally_complete_fixture_passes_and_config_only_assets_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_root = root / "config-fixture"
            complete_root = root / "complete-fixture"
            for model_id, transformer_name in (
                (CURRENT_MODEL_ID, CURRENT_TRANSFORMER_NAME),
                (BETA_MODEL_ID, BETA_TRANSFORMER_NAME),
            ):
                config_only = config_root / "models" / transformer_name
                config_only.mkdir(parents=True)
                (config_only / "config.json").write_text("{}\n", encoding="utf-8")
                available, reason = streamed_transformer_asset_available(config_only)
                self.assertFalse(available)
                self.assertIn("complete derived streamed-AdaLN", reason or "")

                complete = complete_root / "models" / transformer_name
                _write_complete_streamed_fixture(complete)
                available, reason = streamed_transformer_asset_available(complete)
                self.assertTrue(available, reason)

            selected = model_selection_payload(config_root / "repo")
            self.assertEqual([item["id"] for item in selected], [CURRENT_MODEL_ID, BETA_MODEL_ID])
            self.assertFalse(selected[0]["available"])
            self.assertFalse(selected[1]["available"])

    def test_streamed_metadata_admission_rejects_empty_malformed_and_inconsistent_metadata(self) -> None:
        cases = (
            ("empty-config", lambda root: (root / "config.json").write_text("{}\n"), "missing required DiT fields"),
            ("empty-quant", lambda root: (root / "quant_config.json").write_text("{}\n"), "core bits"),
            ("empty-config-and-quant", self._write_empty_config_and_quant, "missing required DiT fields"),
            ("malformed-config", lambda root: (root / "config.json").write_text("{not-json\n"), "not valid JSON"),
            ("malformed-quant", lambda root: (root / "quant_config.json").write_text("{not-json\n"), "not valid JSON"),
            (
                "missing-architecture-field",
                lambda root: self._mutate_json(root / "config.json", lambda value: value.pop("num_layers")),
                "missing required DiT fields",
            ),
            (
                "missing-quantization-field",
                lambda root: self._mutate_json(root / "quant_config.json", lambda value: value.pop("quantized_layers")),
                "layer counts",
            ),
            (
                "inconsistent-sidecar-tensors",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"].pop(),
                ),
                "tensor keys/count",
            ),
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate, message in cases:
                with self.subTest(case=name):
                    fixture = root / name
                    _write_complete_streamed_fixture(fixture)
                    mutate(fixture)
                    with self.assertRaisesRegex(ValueError, message):
                        inspect_checkpoint_format(fixture)

    def test_streamed_byte_count_metadata_rejects_missing_zero_negative_non_integer_and_mismatch(self) -> None:
        cases = (
            (
                "omitted-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].pop("byte_count"),
                ),
            ),
            (
                "zero-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].update(byte_count=0),
                ),
            ),
            (
                "negative-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].update(byte_count=-1),
                ),
            ),
            (
                "non-integer-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].update(byte_count=193536.0),
                ),
            ),
            (
                "boolean-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].update(byte_count=True),
                ),
            ),
            (
                "mismatched-tensor-byte-count",
                lambda root: self._mutate_json(
                    root / "adaln" / "manifest.json",
                    lambda value: value["blocks"]["0"]["tensors"][0].update(byte_count=193537),
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutate in cases:
                with self.subTest(case=name):
                    fixture = root / name
                    _write_complete_streamed_fixture(fixture)
                    mutate(fixture)
                    with self.assertRaisesRegex(ValueError, "byte"):
                        inspect_checkpoint_format(fixture)

    def test_streamed_root_byte_count_metadata_rejects_missing_zero_and_mismatch(self) -> None:
        cases = (
            (
                "omitted-root-derived-base-byte-count",
                lambda value: value.pop("derived_base_byte_count"),
            ),
            (
                "zero-root-derived-base-byte-count",
                lambda value: value.update(derived_base_byte_count=0),
            ),
            (
                "mismatched-root-derived-base-byte-count",
                lambda value: value.update(derived_base_byte_count=VALID_DERIVED_BASE_BYTE_COUNT + 1),
            ),
            (
                "omitted-root-sidecar-byte-count",
                lambda value: value.pop("sidecar_byte_count"),
            ),
            (
                "zero-root-sidecar-byte-count",
                lambda value: value.update(sidecar_byte_count=0),
            ),
            (
                "mismatched-root-sidecar-byte-count",
                lambda value: value.update(sidecar_byte_count=VALID_SIDECAR_BYTE_COUNT + 1),
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, mutation in cases:
                with self.subTest(case=name):
                    fixture = root / name
                    _write_complete_streamed_fixture(fixture)
                    self._mutate_json(fixture / "conversion_manifest.json", mutation)
                    with self.assertRaisesRegex(ValueError, "byte"):
                        inspect_checkpoint_format(fixture)

    @staticmethod
    def _mutate_json(path: Path, mutation) -> None:
        value = json.loads(path.read_text(encoding="utf-8"))
        mutation(value)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")

    @staticmethod
    def _write_empty_config_and_quant(root: Path) -> None:
        (root / "config.json").write_text("{}\n", encoding="utf-8")
        (root / "quant_config.json").write_text("{}\n", encoding="utf-8")

    def test_config_only_current_and_beta_reject_before_any_child_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo_root = root / "repo"
            repo_root.mkdir()
            checkpoint = root / "checkpoints" / "minimax-h3-fl2va"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model_index.json").write_text("{}\n", encoding="utf-8")
            for model_id in (CURRENT_MODEL_ID, BETA_MODEL_ID):
                transformer = model_transformer_path(model_id, repo_root)
                transformer.mkdir(parents=True)
                (transformer / "config.json").write_text("{}\n", encoding="utf-8")
                request = RenderRequest(
                    mode="T2V",
                    prompt="config-only rejection",
                    width=512,
                    height=512,
                    output_root=root / f"render-{model_id}",
                    checkpoint_root=checkpoint,
                    model_id=model_id,
                    turbo_preset_id=REFERENCE_TURBO_PRESET_ID,
                )
                with self.subTest(model_id=model_id), patch("tools.render_lab.runner.subprocess.Popen") as popen:
                    with self.assertRaisesRegex(RenderValidationError, "unavailable"):
                        RenderController(repo_root).start(request)
                    popen.assert_not_called()

    def test_exact_transformer_mappings_are_independently_bound(self) -> None:
        self.assertEqual(model_transformer_path(CURRENT_MODEL_ID, ROOT), CURRENT_TRANSFORMER_PATH)
        self.assertEqual(model_transformer_path(BETA_MODEL_ID, ROOT), BETA_TRANSFORMER_PATH)

    def test_config_exposes_exact_choices_and_beta_default(self) -> None:
        payload = RenderController(ROOT).config_payload()
        self.assertEqual(
            [(item["id"], item["label"]) for item in payload["models"]],
            [(CURRENT_MODEL_ID, "Current"), (BETA_MODEL_ID, "Beta 0.6")],
        )
        self.assertEqual(payload["defaults"]["model_id"], BETA_MODEL_ID)
        self.assertEqual(set(payload["runtime"]["models"]), {CURRENT_MODEL_ID, BETA_MODEL_ID})
        self.assertNotIn("transformer_path", payload["models"][0])
        self.assertNotIn("transformer_path", payload["models"][1])

    def test_current_and_beta_resolve_to_exact_streamed_transformers_and_commands(self) -> None:
        expected = {
            CURRENT_MODEL_ID: CURRENT_TRANSFORMER_NAME,
            BETA_MODEL_ID: BETA_TRANSFORMER_NAME,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id, transformer_name in expected.items():
                with self.subTest(model_id=model_id):
                    validated = _validated(root, model_id=model_id)
                    self.assertEqual(validated.request.model_id, model_id)
                    self.assertEqual(validated.transformer_path, model_transformer_path(model_id, ROOT))
                    self.assertEqual(validated.transformer_path.name, transformer_name)
                    command = build_generation_command(validated, python="python")
                    self.assertEqual(
                        Path(command[command.index("--transformer") + 1]),
                        model_transformer_path(model_id, ROOT),
                    )
                    self.assertEqual(
                        Path(command[command.index("--checkpoint") + 1]),
                        (root / "checkpoint").resolve(),
                    )

    def test_unknown_model_and_arbitrary_transformer_override_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RenderValidationError, "Unknown Render Lab model"):
                _validated(root, model_id="q8")
            with self.assertRaisesRegex(RenderValidationError, "arbitrary transformer overrides"):
                _validated(
                    root,
                    model_id=CURRENT_MODEL_ID,
                    transformer_path=root / "arbitrary-transformer",
                )

    def test_stale_volume_old_resident_and_unrecognized_transformers_remain_rejected(self) -> None:
        with self.assertRaisesRegex(RenderValidationError, "stale /Volumes/models"):
            _validate_transformer_safety(
                Path("/Volumes/models/minimax-h3-mlx-6bit-streamed-adaln"),
                repo_root=ROOT,
                check_runtime_paths=False,
            )
        with self.assertRaisesRegex(RenderValidationError, "non-streamed"):
            _validate_transformer_safety(
                ROOT.parent / "models" / "minimax-h3-mlx-6bit",
                repo_root=ROOT,
                check_runtime_paths=False,
            )
        with self.assertRaisesRegex(RenderValidationError, "only the Current or Beta 0.6"):
            _validate_transformer_safety(
                ROOT.parent / "models" / "unrecognized-transformer",
                repo_root=ROOT,
                check_runtime_paths=False,
            )

    def test_evidence_records_model_identity_resolved_path_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = _validated(root, model_id=BETA_MODEL_ID)
            namespace = reserve_run_namespace(validated)
            config = build_render_config(
                validated,
                namespace,
                build_generation_command(validated, python="python"),
                repo_root=ROOT,
            )
            self.assertEqual(config["model_id"], BETA_MODEL_ID)
            self.assertEqual(config["model"]["label"], "Beta 0.6")
            self.assertEqual(config["model"]["transformer_path"], str(model_transformer_path(BETA_MODEL_ID, ROOT)))
            self.assertEqual(config["model"]["transformer_mode"], CANONICAL_TRANSFORMER_MODE)
            self.assertEqual(config["runtime_identity"]["model_id"], BETA_MODEL_ID)

    def test_current_and_beta_lightx_commands_keep_explicit_four_step_shape(self) -> None:
        preset = turbo_preset_by_id(DEFAULT_TURBO_PRESET_ID)
        self.assertIsNotNone(preset)
        assert preset is not None
        self.assertEqual(preset.runtime_variant, LIGHTX_VARIANT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for model_id in (CURRENT_MODEL_ID, BETA_MODEL_ID):
                with self.subTest(model_id=model_id):
                    validated = _validated(
                        root,
                        model_id=model_id,
                        steps=4,
                        turbo_preset_id=DEFAULT_TURBO_PRESET_ID,
                    )
                    command = build_generation_command(validated, python="python")
                    self.assertEqual(command[command.index("--steps") + 1], "4")
                    self.assertEqual(command[command.index("--lightx-lora") + 1], str(preset.resolve_asset_path(ROOT)))
                    self.assertEqual(command[command.index("--lightx-variant") + 1], LIGHTX_VARIANT)
                    self.assertEqual(command[command.index("--turbo-steps") + 1], "4")
                    self.assertEqual(command[command.index("--transformer") + 1], str(model_transformer_path(model_id, ROOT)))

    def test_storyboard_model_is_global_and_each_segment_child_is_sequentially_isolated(self) -> None:
        for model_id in (CURRENT_MODEL_ID, BETA_MODEL_ID):
            with self.subTest(model_id=model_id), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                cards = _cards(root)
                request = _request(
                    root,
                    workflow=FL2V_STORYBOARD_WORKFLOW,
                    mode=FIRST_LAST,
                    storyboard_card_paths=cards,
                    model_id=model_id,
                    steps=4,
                    turbo_preset_id=DEFAULT_TURBO_PRESET_ID,
                )
                validated = validate_storyboard_request(
                    request,
                    cards,
                    repo_root=ROOT,
                    check_runtime_paths=False,
                    check_images=True,
                    verify_runtime_geometry=False,
                )
                parent = reserve_storyboard_namespace(validated.shared.output_root)  # type: ignore[union-attr]
                initialize_run(parent, build_storyboard_config(validated, parent, repo_root=ROOT))
                parent_config = json.loads(parent.config_path.read_text(encoding="utf-8"))
                shared = parent_config["storyboard"]["shared_global_settings"]
                self.assertEqual(shared["model_id"], model_id)
                self.assertEqual(shared["model"]["transformer_path"], str(model_transformer_path(model_id, ROOT)))

                calls: list[tuple[list[str], object]] = []

                def fake_runner(command, _cwd, namespace):
                    calls.append((list(command), namespace))
                    namespace.output_path.write_bytes(f"segment-{len(calls)}".encode("utf-8"))
                    return 0, 0.01, "", ""

                result = execute_storyboard(
                    parent,
                    validated,
                    repo_root=ROOT,
                    telemetry=lambda _output_root: {"test": True},
                    command_runner=fake_runner,
                    check_runtime_paths=False,
                )
                self.assertTrue(result.success)
                self.assertEqual(len(calls), len(cards) - 1)
                self.assertEqual([item[1].run_id for item in calls], [f"{parent.run_id}-segment-01", f"{parent.run_id}-segment-02"])
                for command, namespace in calls:
                    self.assertEqual(command[command.index("--transformer") + 1], str(model_transformer_path(model_id, ROOT)))
                    child_config = json.loads(Path(namespace.config_path).read_text(encoding="utf-8"))
                    self.assertEqual(child_config["workflow"], FL2V_STORYBOARD_SEGMENT_WORKFLOW)
                    self.assertEqual(child_config["model_id"], model_id)
                    self.assertEqual(child_config["model"]["transformer_path"], str(model_transformer_path(model_id, ROOT)))
                manifest = json.loads(parent.output_path.read_text(encoding="utf-8"))
                self.assertEqual(manifest["execution"]["parallel"], False)
                self.assertEqual(manifest["execution"]["isolation"], "one child render process per segment")
                self.assertEqual(manifest["shared_global_settings"]["model_id"], model_id)

    def test_server_and_launcher_use_logical_model_selector_and_render_lab(self) -> None:
        request = _render_request_from_fields({
            "mode": "T2V",
            "prompt": "selector contract",
            "model_id": BETA_MODEL_ID,
            "steps": "4",
            "duration_seconds": "5",
            "seed": "0",
            "width": "512",
            "height": "512",
        })
        self.assertEqual(request.model_id, BETA_MODEL_ID)
        self.assertIn('id="model"', PAGE)
        self.assertIn("form.set('model_id', $('model').value);", PAGE)
        self.assertIn("$('model').value = config.defaults.model_id;", PAGE)
        self.assertIn("$('turbo-preset').value = config.defaults.turbo_preset_id;", PAGE)
        self.assertIn("$('steps').value = config.defaults.steps;", PAGE)
        self.assertNotIn("config.defaults.model_id || 'beta-0.6'", PAGE)
        self.assertNotIn("config.defaults.turbo_preset_id || 'none'", PAGE)
        launcher = (ROOT / "Launch MiniMax H3.command").read_text(encoding="utf-8")
        self.assertIn("tools/render_lab/server.py --open", launcher)
        self.assertIn(".venv/bin/python", launcher)
        self.assertIn('required repository interpreter is unavailable', launcher)
        self.assertNotIn('/usr/bin/python3', launcher)
        self.assertNotIn("scripts/minimax_h3_surface.py", launcher)

    def test_launcher_missing_repository_interpreter_fails_closed(self) -> None:
        launcher_source = (ROOT / "Launch MiniMax H3.command").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "Launch MiniMax H3.command"
            launcher.write_text(launcher_source, encoding="utf-8")
            result = subprocess.run(
                ["/bin/zsh", str(launcher)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required repository interpreter is unavailable", result.stderr)


if __name__ == "__main__":
    unittest.main()

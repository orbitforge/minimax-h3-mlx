"""MLX-free contracts for the curated Render Lab Turbo surface."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.render_lab.runner import (
    BETA_TRANSFORMER_NAME,
    CANONICAL_TRANSFORMER_NAME,
    CANONICAL_TRANSFORMER_MODE,
    CURRENT_MODEL_ID,
    DEFAULT_TURBO_PRESET_ID,
    RenderController,
    RenderRequest,
    RenderValidationError,
    build_generation_command,
    build_render_config,
    default_transformer_path,
    reserve_run_namespace,
    validate_render_request,
)
from tools.render_lab.turbo_presets import (
    REFERENCE_TURBO_PRESET_ID,
    TURBO_PRESETS,
    turbo_preset_by_id,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT.parent / "checkpoints" / "minimax-h3-fl2va"
STREAMED_TRANSFORMER = ROOT.parent / "models" / CANONICAL_TRANSFORMER_NAME


def make_request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "mode": "T2V",
        "prompt": "a test prompt",
        "resolution_id": None,
        "width": 512,
        "height": 512,
        "steps": 16,
        "duration_seconds": 5.0,
        "seed": 0,
        "output_root": root / "render-lab",
        "output_name": "test.mp4",
        "checkpoint_root": root / "checkpoint",
        "turbo_preset_id": REFERENCE_TURBO_PRESET_ID,
    }
    values.update(changes)
    return RenderRequest(**values)


class RenderLabTurboContractTests(unittest.TestCase):
    def test_default_and_exact_five_production_presets_are_exposed_once(self) -> None:
        payload = RenderController(ROOT).config_payload()
        self.assertEqual(payload["defaults"]["turbo_preset_id"], DEFAULT_TURBO_PRESET_ID)
        self.assertEqual(payload["turbo_presets"][0]["label"], "None / Reference")

        surfaced = [item for item in payload["turbo_presets"] if item["id"] != REFERENCE_TURBO_PRESET_ID]
        self.assertEqual(len(surfaced), 5)
        self.assertEqual(
            [item["label"] for item in surfaced],
            [preset.label for preset in TURBO_PRESETS],
        )
        self.assertEqual(len({item["id"] for item in surfaced}), 5)
        for preset in TURBO_PRESETS:
            self.assertEqual(
                sum(item["id"] == preset.preset_id for item in payload["turbo_presets"]),
                1,
            )

    def test_each_preset_owns_family_asset_nfe_and_cli_shape(self) -> None:
        expected = {
            "lightx-4step-v01": ("LightX2V", 4, "--lightx-lora", "fl2va-turbo-4step-v0.1"),
            "lightx-8step-v10": ("LightX2V", 8, "--lightx-lora", "fl2va-turbo-8step-v1.0"),
            "lightx-4step-v10-768p": ("LightX2V", 4, "--lightx-lora", "fl2va-turbo-4step-v1.0-768p"),
            "larry-v4-step600": ("Larry", 8, "--turbo-lora", None),
            "larry-850": ("Larry", 4, "--turbo-lora", None),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for preset_id, (family, nfe, adapter_flag, variant) in expected.items():
                with self.subTest(preset_id=preset_id):
                    preset = turbo_preset_by_id(preset_id)
                    assert preset is not None
                    self.assertEqual((preset.family, preset.nfe, preset.adapter_flag, preset.runtime_variant), (family, nfe, adapter_flag, variant))
                    request = make_request(
                        root,
                        steps=nfe,
                        lora_path=root / "wrong-adapter.safetensors",
                        lora_scale=1.0,
                        turbo_preset_id=preset_id,
                    )
                    validated = validate_render_request(
                        request,
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )
                    command = build_generation_command(validated, python="python")
                    self.assertIn(adapter_flag, command)
                    self.assertEqual(
                        Path(command[command.index(adapter_flag) + 1]),
                        preset.resolve_asset_path(ROOT),
                    )
                    self.assertEqual(command[command.index("--steps") + 1], str(nfe))
                    self.assertEqual(command[command.index("--turbo-steps") + 1], str(nfe))
                    self.assertEqual(command[command.index("--lora-scale") + 1], "1")
                    if variant is None:
                        self.assertNotIn("--lightx-variant", command)
                    else:
                        self.assertEqual(command[command.index("--lightx-variant") + 1], variant)
                    self.assertNotIn("--lora", command)

    def test_preset_rejects_mismatched_steps_and_scale_and_none_preserves_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for changes, message in (
                ({"turbo_preset_id": "larry-850", "steps": 8}, "owns 4 NFE"),
                ({"turbo_preset_id": "larry-850", "steps": 4, "turbo_steps": 8}, "owns 4 NFE"),
                ({"turbo_preset_id": "larry-850", "steps": 4, "lora_scale": 0.5}, "fixes LoRA scale"),
            ):
                with self.subTest(changes=changes), self.assertRaisesRegex(RenderValidationError, message):
                    validate_render_request(
                        make_request(root, **changes),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

            reference = validate_render_request(
                make_request(
                    root,
                    turbo_preset_id=REFERENCE_TURBO_PRESET_ID,
                    lora_enabled=False,
                    lora_path=root / "unused.safetensors",
                    lora_scale=-99,
                    steps=16,
                ),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            command = build_generation_command(reference, python="python")
            self.assertNotIn("--lora", command)
            self.assertIsNone(reference.turbo_preset)
            self.assertEqual(reference.request.steps, 16)

    def test_run_evidence_records_preset_and_streamed_transformer_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = turbo_preset_by_id("lightx-4step-v10-768p")
            assert preset is not None
            validated = validate_render_request(
                make_request(
                    root,
                    width=1344,
                    height=768,
                    steps=4,
                    model_id=CURRENT_MODEL_ID,
                    turbo_preset_id=preset.preset_id,
                    checkpoint_root=CHECKPOINT,
                    transformer_path=STREAMED_TRANSFORMER,
                ),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            namespace = reserve_run_namespace(validated)
            config = build_render_config(
                validated,
                namespace,
                build_generation_command(validated, python="python"),
                repo_root=ROOT,
            )
            turbo = config["turbo"]
            self.assertEqual(turbo["preset_id"], preset.preset_id)
            self.assertEqual(turbo["family"], "LightX2V")
            self.assertEqual(turbo["logical_asset"], "lightx_v10_768p")
            self.assertEqual(turbo["effective_nfe"], 4)
            self.assertEqual(turbo["runtime_variant"], "fl2va-turbo-4step-v1.0-768p")
            self.assertEqual(turbo["effective_scheduler"], {"video_shift": 6.0, "audio_shift": 3.0, "source": "LightX2V manifest"})
            self.assertEqual(config["transformer_mode"], CANONICAL_TRANSFORMER_MODE)
            self.assertEqual(config["runtime_identity"]["transformer_name"], CANONICAL_TRANSFORMER_NAME)

    def test_normal_render_lab_rejects_non_streamed_q6_and_uses_streamed_default(self) -> None:
        with patch.dict(os.environ, {"H3_TRANSFORMER": ""}):
            self.assertEqual(default_transformer_path(ROOT).name, BETA_TRANSFORMER_NAME)
        ordinary = ROOT.parent / "models" / "minimax-h3-mlx-6bit"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RenderValidationError, "arbitrary transformer overrides"):
                validate_render_request(
                    make_request(
                        Path(directory),
                        checkpoint_root=CHECKPOINT,
                        transformer_path=ordinary,
                    ),
                    repo_root=ROOT,
                    check_runtime_paths=True,
                    check_images=False,
                    verify_runtime_geometry=False,
                )


if __name__ == "__main__":
    unittest.main()

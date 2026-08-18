"""MLX-free contracts for Slice 021B Render Lab auxiliary LoRA plumbing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.render_lab.runner import (
    AdditionalLoRA,
    RenderRequest,
    RenderValidationError,
    build_generation_command,
    build_render_config,
    parse_additional_loras_payload,
    reserve_run_namespace,
    validate_render_request,
)
from tools.render_lab.server import PAGE
from tools.render_lab.turbo_presets import REFERENCE_TURBO_PRESET_ID, turbo_preset_by_id


ROOT = Path(__file__).resolve().parents[1]


def make_request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "mode": "T2V",
        "prompt": "a test prompt",
        "resolution_id": None,
        "width": 512,
        "height": 512,
        "steps": 16,
        "duration_seconds": 5.0,
        "seed": 7,
        "output_root": root / "render-lab",
        "output_name": "test.mp4",
        "checkpoint_root": root / "checkpoint",
        "transformer_path": root / "transformer",
        "turbo_preset_id": REFERENCE_TURBO_PRESET_ID,
    }
    values.update(changes)
    return RenderRequest(**values)


def validate(root: Path, **changes: object):
    return validate_render_request(
        make_request(root, **changes),
        repo_root=ROOT,
        check_images=False,
        verify_runtime_geometry=False,
    )


def flag_values(command: list[str], flag: str) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command) if value == flag]


class RenderLabAdditionalLoRAContractTests(unittest.TestCase):
    def test_none_reference_supports_zero_one_and_multiple_additional_loras(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = (
                (),
                (AdditionalLoRA(root / "style-a.safetensors", 0.80),),
                (
                    AdditionalLoRA(root / "style-a.safetensors", 0.80),
                    AdditionalLoRA(root / "style-b.safetensors", 0.45),
                ),
            )
            for entries in cases:
                with self.subTest(entries=entries):
                    validated = validate(root, additional_loras=entries)
                    command = build_generation_command(validated, python="python")
                    self.assertEqual(flag_values(command, "--additional-lora"), [str(entry.path.resolve()) for entry in entries])
                    self.assertEqual(
                        flag_values(command, "--additional-lora-scale"),
                        [format(entry.scale, ".6g") for entry in entries],
                    )
                    self.assertNotIn("--turbo-lora", command)
                    self.assertNotIn("--lightx-lora", command)
                    self.assertNotIn("--turbo", command)

    def test_turbo_supports_zero_one_and_multiple_additional_loras_without_consuming_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = turbo_preset_by_id("larry-850")
            assert preset is not None
            cases = (
                (),
                (AdditionalLoRA(root / "style-a.safetensors", 0.80),),
                (
                    AdditionalLoRA(root / "style-a.safetensors", 0.80),
                    AdditionalLoRA(root / "style-b.safetensors", 0.45),
                ),
            )
            for entries in cases:
                with self.subTest(entries=entries):
                    validated = validate(
                        root,
                        steps=preset.nfe,
                        turbo_preset_id=preset.preset_id,
                        additional_loras=entries,
                    )
                    command = build_generation_command(validated, python="python")
                    self.assertEqual(command[command.index("--turbo-lora") + 1], str(preset.resolve_asset_path(ROOT)))
                    self.assertEqual(flag_values(command, "--additional-lora"), [str(entry.path.resolve()) for entry in entries])
                    self.assertEqual(
                        flag_values(command, "--additional-lora-scale"),
                        [format(entry.scale, ".6g") for entry in entries],
                    )
                    self.assertNotIn("--lora", command)

    def test_additional_scales_stay_associated_with_deterministic_row_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = (
                AdditionalLoRA(root / "first.safetensors", 0.125),
                AdditionalLoRA(root / "second.safetensors", 2.0),
                AdditionalLoRA(root / "third.safetensors", 0.45),
            )
            validated = validate(root, additional_loras=entries)
            command = build_generation_command(validated, python="python")
            serialized_pairs = list(zip(flag_values(command, "--additional-lora"), flag_values(command, "--additional-lora-scale")))
            self.assertEqual(
                serialized_pairs,
                [(str(entry.path.resolve()), format(entry.scale, ".6g")) for entry in entries],
            )

    def test_turbo_selection_does_not_overwrite_auxiliary_paths_or_scales(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = (
                AdditionalLoRA(root / "style-a.safetensors", 0.80),
                AdditionalLoRA(root / "style-b.safetensors", 0.45),
            )
            reference = validate(root, steps=4, additional_loras=entries)
            preset = validate(root, steps=4, turbo_preset_id="lightx-4step-v01", additional_loras=entries)
            self.assertEqual(reference.additional_loras, preset.additional_loras)
            self.assertEqual(
                [(entry.path, entry.scale) for entry in preset.additional_loras],
                [((root / "style-a.safetensors").resolve(), 0.80), ((root / "style-b.safetensors").resolve(), 0.45)],
            )

    def test_auxiliary_changes_do_not_change_turbo_nfe_or_lightx_variant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = validate(root, steps=4, turbo_preset_id="lightx-4step-v01", additional_loras=())
            changed = validate(
                root,
                steps=4,
                turbo_preset_id="lightx-4step-v01",
                additional_loras=(AdditionalLoRA(root / "style.safetensors", 0.25),),
            )
            self.assertEqual(base.request.steps, changed.request.steps)
            self.assertEqual(base.request.turbo_steps, changed.request.turbo_steps)
            self.assertEqual(base.turbo_preset.runtime_variant, changed.turbo_preset.runtime_variant)
            self.assertEqual(
                build_generation_command(base, python="python")[: build_generation_command(base, python="python").index("--turbo-steps") + 2],
                build_generation_command(changed, python="python")[: build_generation_command(changed, python="python").index("--turbo-steps") + 2],
            )

    def test_malformed_additional_rows_fail_before_command_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = {"additional_loras": (AdditionalLoRA(root / "style.safetensors", 0.8),)}
            malformed = (
                ("empty path", (AdditionalLoRA("", 0.8),), "no adapter path"),
                ("invalid scale", (AdditionalLoRA(root / "style.safetensors", "not-a-number"),), "finite nonnegative"),
                ("NaN scale", (AdditionalLoRA(root / "style.safetensors", float("nan")),), "finite nonnegative"),
                ("infinite scale", (AdditionalLoRA(root / "style.safetensors", float("inf")),), "finite nonnegative"),
            )
            for label, entries, message in malformed:
                with self.subTest(label=label), self.assertRaisesRegex(RenderValidationError, message):
                    validate(root, **(base | {"additional_loras": entries}))

    def test_duplicate_and_scheduling_collision_are_rejected_by_021a_stack_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = (
                AdditionalLoRA(root / "style.safetensors", 0.8),
                AdditionalLoRA(root / "./style.safetensors", 0.45),
            )
            with self.assertRaisesRegex(RenderValidationError, "duplicate normalized auxiliary"):
                validate(root, additional_loras=duplicate)

            preset = turbo_preset_by_id("larry-850")
            assert preset is not None
            collision = (AdditionalLoRA(preset.resolve_asset_path(ROOT), 0.8),)
            with self.assertRaisesRegex(RenderValidationError, "duplicated as an auxiliary"):
                validate(
                    root,
                    steps=preset.nfe,
                    turbo_preset_id=preset.preset_id,
                    additional_loras=collision,
                )

    def test_request_payload_round_trip_preserves_order_and_scales(self) -> None:
        payload = json.dumps([
            {"path": "/models/style-a.safetensors", "scale": "0.80"},
            {"path": "/models/style-b.safetensors", "scale": "0.45"},
        ])
        entries = parse_additional_loras_payload(payload)
        self.assertEqual([entry.path for entry in entries], ["/models/style-a.safetensors", "/models/style-b.safetensors"])
        self.assertEqual([entry.scale for entry in entries], ["0.80", "0.45"])

    def test_evidence_separates_ordered_auxiliary_stack_from_turbo_owner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preset = turbo_preset_by_id("larry-850")
            assert preset is not None
            validated = validate(
                root,
                steps=preset.nfe,
                turbo_preset_id=preset.preset_id,
                additional_loras=(AdditionalLoRA(root / "style-a.safetensors", 0.8), AdditionalLoRA(root / "style-b.safetensors", 0.45)),
            )
            namespace = reserve_run_namespace(validated)
            config = build_render_config(
                validated,
                namespace,
                build_generation_command(validated, python="python"),
                repo_root=ROOT,
            )
            self.assertEqual(
                [(item["path"], item["scale"]) for item in config["additional_loras"]],
                [(str((root / "style-a.safetensors").resolve()), 0.8), (str((root / "style-b.safetensors").resolve()), 0.45)],
            )
            self.assertEqual(config["auxiliary_lora_stack"]["scheduling_owner"], "turbo-preset")
            self.assertEqual(config["turbo"]["scheduling_owner"], "turbo-preset")
            self.assertEqual(config["turbo"]["adapter_asset"]["path"], str(preset.resolve_asset_path(ROOT)))
            self.assertNotIn(str(preset.resolve_asset_path(ROOT)), [item["path"] for item in config["additional_loras"]])

    def test_legacy_manual_turbo_command_and_evidence_stay_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manual_path = root / "manual-turbo.safetensors"
            validated = validate(
                root,
                lora_enabled=True,
                lora_path=manual_path,
                lora_scale=0.75,
                turbo_enabled=True,
                turbo_steps=8,
                steps=8,
                turbo_preset_id=None,
            )
            command = build_generation_command(validated, python="python")
            self.assertEqual(command[command.index("--lora") + 1], str(manual_path.resolve()))
            self.assertEqual(command[command.index("--lora-scale") + 1], "0.75")
            self.assertIn("--turbo", command)
            self.assertEqual(command[command.index("--turbo-steps") + 1], "8")

            namespace = reserve_run_namespace(validated)
            config = build_render_config(
                validated,
                namespace,
                command,
                repo_root=ROOT,
            )
            turbo = config["turbo"]
            self.assertEqual(turbo["scheduling_owner"], "legacy-turbo-adapter")
            self.assertEqual(turbo["adapter_asset"]["flag"], "--lora")
            self.assertEqual(turbo["adapter_asset"]["path"], str(manual_path.resolve()))
            self.assertEqual(turbo["effective_scale"], 0.75)
            self.assertEqual(turbo["effective_nfe"], 8)
            self.assertEqual(config["auxiliary_lora_stack"]["scheduling_owner"], turbo["scheduling_owner"])

    def test_none_reference_without_turbo_has_no_scheduling_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = validate(
                root,
                turbo_preset_id=REFERENCE_TURBO_PRESET_ID,
                turbo_enabled=False,
                lora_enabled=False,
                lora_path=root / "unused.safetensors",
                lora_scale=-99,
            )
            namespace = reserve_run_namespace(validated)
            config = build_render_config(
                validated,
                namespace,
                build_generation_command(validated, python="python"),
                repo_root=ROOT,
            )
            turbo = config["turbo"]
            self.assertEqual(turbo["scheduling_owner"], "none-reference")
            self.assertIsNone(turbo["adapter_asset"])
            self.assertIsNone(turbo["effective_scale"])
            self.assertEqual(config["auxiliary_lora_stack"]["scheduling_owner"], "none-reference")

    def test_browser_keeps_additional_rows_independent_when_turbo_changes(self) -> None:
        self.assertIn('id="additional-lora-rows"', PAGE)
        self.assertIn("function addAdditionalLora()", PAGE)
        self.assertIn("function removeAdditionalLora(index)", PAGE)
        self.assertIn("form.set('additional_loras', JSON.stringify(selectedAdditionalLoras()));", PAGE)
        self.assertIn("$('steps').value = preset.nfe;", PAGE)
        self.assertNotIn("lora-enabled", PAGE)
        self.assertNotIn("referenceControlState", PAGE)
        self.assertNotIn("preset.adapter_asset_path", PAGE)


if __name__ == "__main__":
    unittest.main()

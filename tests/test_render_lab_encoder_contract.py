"""MLX-free contracts for the Render Lab text-encoder boundary."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from minimax_h3_mlx.conditioning_artifact import (
    CONDITIONING_WIDTH,
    HERETIC_BRIDGE_SHA256,
    build_encoder_provenance,
    create_conditioning_artifact_from_bits,
    validate_conditioning_artifact,
)
from tools.render_lab.encoder_catalog import (
    CANONICAL_ENCODER_ID,
    HERETIC_ENCODER_ID,
    HERETIC_IMAGE_MODE_MESSAGE,
    HERETIC_SOURCE_WIDTH,
    HERETIC_STATE,
    HERETIC_TARGET_WIDTH,
    probe_heretic_assets,
    text_encoder_payload,
    validate_text_encoder_selection,
)
from tools.render_lab import heretic_encoder
from tools.render_lab.runner import (
    FIRST_LAST,
    T2V,
    RenderRequest,
    RenderController,
    RenderValidationError,
    build_generation_command,
    build_heretic_encoder_command,
    build_render_config,
    execute_heretic_run,
    reserve_run_namespace,
    validate_render_request,
)
from tools.render_lab.server import PAGE


ROOT = Path(__file__).resolve().parents[1]


def _request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "mode": T2V,
        "prompt": "a jaguar prowls through dense jungle foliage.",
        "resolution_id": None,
        "width": 512,
        "height": 512,
        "steps": 16,
        "duration_seconds": 5.0,
        "seed": 0,
        "output_root": root / "render-lab",
        "output_name": "test.mp4",
        "checkpoint_root": root / "checkpoint",
        "transformer_path": root / "minimax-h3-mlx-6bit-streamed-adaln",
    }
    values.update(changes)
    return RenderRequest(**values)


def _checkpoint(root: Path) -> dict:
    (root / "text_encoder").mkdir(parents=True, exist_ok=True)
    (root / "tokenizer").mkdir(exist_ok=True)
    (root / "processor").mkdir(exist_ok=True)
    (root / "text_encoder" / "config.json").write_text(
        json.dumps({
            "model_type": "qwen3_vl",
            "text_config": {"num_hidden_layers": 64, "hidden_size": CONDITIONING_WIDTH},
        }),
        encoding="utf-8",
    )
    (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "processor" / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    (root / "model_index.json").write_text("{}", encoding="utf-8")
    return build_encoder_provenance(root)


def _heretic_provenance(checkpoint: Path, source_root: Path, token_ids: list[int]) -> dict:
    canonical = build_encoder_provenance(checkpoint)
    pieces = ["a", " jaguar"]
    return {
        "encoder_id": HERETIC_ENCODER_ID,
        "experimental": True,
        "family": "qwen3_5_moe",
        "config": {
            "sha256": "source-config",
            "model_type": "qwen3_5_moe",
            "hidden_size": HERETIC_SOURCE_WIDTH,
            "full_decoder_layers": 40,
        },
        "selected_state": {
            "hidden_state": "hidden_states[28]",
            "normalization": "unnormalized-pre-final-norm",
            "selected_decoder_layer": HERETIC_STATE,
            "logical_dtype": "bfloat16",
        },
        "source_model": {
            "path": str(source_root / "heretic-model"),
            "config_sha256": "source-config",
            "maximum_executed_state": 28,
            "layers_29_through_40_executed": False,
        },
        "weights": {"source": "heretic-source-model", "manifest": {"present": True}},
        "tokenizer": {"source": "heretic-source-model", "manifest": {"present": True}},
        "processor": {"source": "not-used-for-text-only-encode", "manifest": {"present": False}},
        "canonical_h3_encoder": canonical,
        "bridge": {
            "path": str(source_root / "stable-bridge.npz"),
            "sha256": HERETIC_BRIDGE_SHA256,
            "input_width": HERETIC_SOURCE_WIDTH,
            "target_width": HERETIC_TARGET_WIDTH,
            "keys": ["input_mean", "input_scale", "target_mean", "weights"],
            "shapes": {
                "input_mean": [HERETIC_SOURCE_WIDTH],
                "input_scale": [HERETIC_SOURCE_WIDTH],
                "target_mean": [HERETIC_TARGET_WIDTH],
                "weights": [HERETIC_SOURCE_WIDTH, HERETIC_TARGET_WIDTH],
            },
            "operation": "(state28 - input_mean) / input_scale @ weights + target_mean",
        },
        "token_alignment": {
            "exact_token_piece_alignment": True,
            "canonical_h3_token_ids": token_ids,
            "heretic_token_ids": [value + 1000 for value in token_ids],
            "token_pieces": pieces,
        },
    }


class RenderLabEncoderContractTests(unittest.TestCase):
    def test_surface_exposes_exactly_two_choices_and_canonical_default(self) -> None:
        payload = RenderController(ROOT).config_payload()
        self.assertEqual([item["id"] for item in payload["text_encoders"]], [CANONICAL_ENCODER_ID, HERETIC_ENCODER_ID])
        self.assertEqual(payload["text_encoders"][0]["label"], "Canonical Qwen3-VL")
        self.assertEqual(payload["text_encoders"][1]["label"], "Heretic 35B-A3B · Experimental")
        self.assertEqual(payload["defaults"]["text_encoder_id"], CANONICAL_ENCODER_ID)
        self.assertEqual(payload["text_encoders"][1]["allowed_modes"], [T2V])
        self.assertIn("state 28 + learned H3 conditioning bridge", payload["text_encoders"][1]["hint"])

    def test_heretic_is_t2v_only_and_fails_before_runtime_asset_checks(self) -> None:
        with self.assertRaisesRegex(ValueError, HERETIC_IMAGE_MODE_MESSAGE):
            validate_text_encoder_selection(
                HERETIC_ENCODER_ID,
                FIRST_LAST,
                repo_root=ROOT,
                check_runtime_paths=True,
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RenderValidationError, "text-only"):
                validate_render_request(
                    _request(root, mode=FIRST_LAST, text_encoder_id=HERETIC_ENCODER_ID, image_paths=(root / "a", root / "b")),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )

    def test_missing_or_wrong_bridge_is_unavailable_before_encoder_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "heretic"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "qwen3_5_moe", "text_config": {"hidden_size": 2048, "num_hidden_layers": 40}}),
                encoding="utf-8",
            )
            bridge = root / "bridge.npz"
            bridge.write_bytes(b"not-the-approved-bridge")
            with patch.dict(os.environ, {"H3_HERETIC_MODEL": str(model), "H3_HERETIC_BRIDGE": str(bridge)}):
                assets = probe_heretic_assets(ROOT)
            self.assertFalse(assets.available)
            self.assertIn("SHA-256 mismatch", assets.reason or "")

    def test_heretic_command_is_artifact_replay_and_keeps_turbo_shape_orthogonal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            heretic_request = _request(root, text_encoder_id=HERETIC_ENCODER_ID)
            canonical = validate_render_request(
                _request(root), repo_root=ROOT, check_images=False, verify_runtime_geometry=False
            )
            heretic = validate_render_request(
                heretic_request,
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            self.assertEqual(canonical.request.steps, heretic.request.steps)
            self.assertEqual(canonical.request.duration_seconds, heretic.request.duration_seconds)
            self.assertEqual(canonical.width, heretic.width)
            canonical_command = build_generation_command(canonical, python="python")
            heretic_command = build_generation_command(heretic, python="python")
            self.assertNotIn("--conditioning-artifact", canonical_command)
            self.assertIn("--conditioning-artifact", heretic_command)
            self.assertNotIn(heretic_request.prompt, heretic_command)
            self.assertNotIn("--image", heretic_command)

            namespace = reserve_run_namespace(heretic)
            encoder_command = build_heretic_encoder_command(heretic, namespace, python="python")
            self.assertIn("heretic_encoder.py", " ".join(encoder_command))
            self.assertIn(str(namespace.encoder_evidence_path), encoder_command)
            self.assertIn(str(namespace.encoder_release_path), encoder_command)

    def test_stale_volumes_transformer_and_ordinary_q6_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for transformer in (
                ROOT.parent / "models" / "minimax-h3-mlx-6bit",
                Path("/Volumes/models/minimax-h3-mlx-6bit-streamed-adaln"),
            ):
                with self.subTest(transformer=transformer), self.assertRaises(RenderValidationError):
                    validate_render_request(
                        _request(root, transformer_path=transformer),
                        repo_root=ROOT,
                        check_runtime_paths=True,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

    def test_artifact_accepts_heretic_provenance_but_replays_against_canonical_h3_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            _checkpoint(checkpoint)
            ids = [11, 12]
            provenance = _heretic_provenance(checkpoint, root, ids)
            bits = np.full((1, 2, CONDITIONING_WIDTH), 0x3F80, dtype=np.uint16)
            artifact = create_conditioning_artifact_from_bits(
                root / "conditioning.npz",
                prompt="a jaguar prowls through dense jungle foliage.",
                conditioning_bits=bits,
                text_token_tags=np.ones((2,), dtype=np.int32),
                token_ids=np.asarray([ids], dtype=np.int32),
                encoder_provenance=provenance,
            )
            validated = validate_conditioning_artifact(
                artifact,
                checkpoint_root=checkpoint,
                prompt="a jaguar prowls through dense jungle foliage.",
            )
            self.assertEqual(validated.metadata["postprocessing"]["selected_state"], "hidden_states[28]")
            self.assertEqual(validated.metadata["postprocessing"]["projection"], "state28-standardize-affine-bridge-before-H3-condition_proj")
            self.assertEqual(validated.metadata["encoder"]["token_alignment"]["canonical_h3_token_ids"], ids)

    def test_encoder_and_h3_children_are_strictly_sequential(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            _checkpoint(checkpoint)
            request = _request(root, checkpoint_root=checkpoint)
            validated = validate_render_request(
                request,
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            namespace = reserve_run_namespace(validated)
            h3_command = ["h3-child"]
            encoder_command = ["encoder-child"]
            config = build_render_config(validated, namespace, h3_command, repo_root=ROOT, encoder_command=encoder_command)
            namespace.config_path.write_text(json.dumps(config), encoding="utf-8")
            calls: list[str] = []

            def fake_runner(command, _cwd, current_namespace):
                calls.append(command[0])
                if command[0] == "encoder-child":
                    ids = [11, 12]
                    artifact = create_conditioning_artifact_from_bits(
                        current_namespace.conditioning_artifact_path,
                        prompt=request.prompt,
                        conditioning_bits=np.full((1, 2, CONDITIONING_WIDTH), 0x3F80, dtype=np.uint16),
                        text_token_tags=np.ones((2,), dtype=np.int32),
                        token_ids=np.asarray([ids], dtype=np.int32),
                        encoder_provenance=_checkpoint(checkpoint),
                    )
                    current_namespace.encoder_evidence_path.write_text(
                        json.dumps({
                            "status": "complete",
                            "release_gate": True,
                            "h3_launched_before_encoder_exit": False,
                            "conditioning_artifact": {
                                "path": str(artifact.path),
                                "identity": artifact.artifact_identity,
                                "tensor_checksum": artifact.tensor_checksum,
                            },
                        }),
                        encoding="utf-8",
                    )
                    current_namespace.encoder_release_path.write_text(json.dumps({"clean": True}), encoding="utf-8")
                    return 0, 0.1, "", ""
                current_namespace.output_path.write_bytes(b"fake mp4")
                return 0, 0.1, "", ""

            result = execute_heretic_run(
                namespace,
                encoder_command,
                h3_command,
                repo_root=ROOT,
                command_runner=fake_runner,
            )
            self.assertEqual(
                calls,
                ["encoder-child", "h3-child"],
                json.dumps(result.benchmark) + " stderr=" + namespace.stderr_path.read_text(encoding="utf-8"),
            )
            self.assertTrue(result.success)
            benchmark = json.loads(namespace.benchmark_path.read_text(encoding="utf-8"))
            self.assertFalse(benchmark["h3_launched_before_encoder_exit"])
            self.assertTrue(benchmark["h3_launched_after_encoder_exit"])
            self.assertEqual(benchmark["encoder_process_exit_code"], 0)

    def test_page_contains_selector_hint_and_form_field(self) -> None:
        self.assertIn('id="text-encoder"', PAGE)
        self.assertIn("Heretic is currently text-only; image-conditioned modes require Canonical Qwen3-VL.", PAGE)
        self.assertIn("form.set('text_encoder_id', $('text-encoder').value);", PAGE)

    def test_encoder_helper_keeps_mlx_imports_inside_child_execution_and_caps_state(self) -> None:
        path = ROOT / "tools" / "render_lab" / "heretic_encoder.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        top_level_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
        self.assertFalse(
            any(
                (node.module or "").split(".", 1)[0] == "mlx"
                for node in top_level_imports
                if isinstance(node, ast.ImportFrom)
            )
        )
        self.assertFalse(any(alias.name.split(".", 1)[0] == "mlx" for node in top_level_imports if isinstance(node, ast.Import) for alias in node.names))
        self.assertIn("layers[:HERETIC_STATE]", source)
        self.assertIn("mx.bfloat16", source)
        self.assertIn("layers_29_through_40_executed", source)
        self.assertNotIn("Qwen3_5MoeForConditionalGeneration", source)

    def test_state28_timing_stops_after_single_materialization(self) -> None:
        events: list[object] = []
        source_state = object()

        def fake_clock() -> float:
            events.append("clock")
            return float(len(events))

        class FakeMX:
            def eval(self, value: object) -> None:
                events.append(("eval", value))

        with patch.object(heretic_encoder, "_manual_state28_forward", return_value=source_state) as forward:
            evaluated_state, elapsed = heretic_encoder._timed_state28_forward(
                object(),
                object(),
                FakeMX(),
                clock=fake_clock,
            )

        self.assertIs(evaluated_state, source_state)
        self.assertEqual(elapsed, 2.0)
        self.assertEqual(forward.call_count, 1)
        self.assertEqual(events, ["clock", ("eval", source_state), "clock"])


if __name__ == "__main__":
    unittest.main()

"""MLX-free contracts for Render Lab canonical conditioning-artifact replay."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from minimax_h3_mlx.conditioning_artifact import (
    CONDITIONING_WIDTH,
    build_encoder_provenance,
    create_conditioning_artifact_from_bits,
)
from tools.render_lab.encoder_catalog import HERETIC_ENCODER_ID
from tools.render_lab.runner import (
    FIRST_LAST,
    I2V,
    T2V,
    RenderRequest,
    RenderValidationError,
    build_generation_command,
    build_render_config,
    reserve_run_namespace,
    validate_render_request,
)
from tools.render_lab.server import PAGE, _render_request_from_fields


ROOT = Path(__file__).resolve().parents[1]


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


def _request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "mode": T2V,
        "prompt": "a replayable conditioning prompt",
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


def _artifact(root: Path, *, checkpoint: Path | None = None, prompt: str = "a replayable conditioning prompt"):
    checkpoint = checkpoint or root / "checkpoint"
    provenance = _checkpoint(checkpoint)
    path = root / "conditioning-artifact.npz"
    artifact = create_conditioning_artifact_from_bits(
        path,
        prompt=prompt,
        conditioning_bits=np.full((1, 3, CONDITIONING_WIDTH), 0x3F80, dtype=np.uint16),
        text_token_tags=np.ones((3,), dtype=np.int32),
        token_ids=np.asarray([[11, 12, 13]], dtype=np.int32),
        encoder_provenance=provenance,
    )
    return artifact


class RenderLabConditioningArtifactContractTests(unittest.TestCase):
    def test_blank_path_preserves_live_canonical_qwen_behavior_and_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _checkpoint(root / "checkpoint")
            validated = validate_render_request(
                _request(root),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            command = build_generation_command(validated, python="python")
            self.assertIn("a replayable conditioning prompt", command)
            self.assertNotIn("--conditioning-artifact", command)
            namespace = reserve_run_namespace(validated)
            config = build_render_config(validated, namespace, command, repo_root=ROOT)
            self.assertEqual(config["conditioning_source"], "live-encoder")
            self.assertIsNone(config["conditioning_artifact"])
            self.assertIsNone(config["conditioning_artifact_path"])

    def test_valid_canonical_artifact_is_admitted_and_replay_command_omits_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _artifact(root)
            validated = validate_render_request(
                _request(root, conditioning_artifact_path=str(artifact.path)),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            self.assertEqual(validated.conditioning_artifact_path, artifact.path.resolve())
            self.assertIsNotNone(validated.conditioning_artifact_evidence)
            evidence = validated.conditioning_artifact_evidence
            assert evidence is not None
            self.assertEqual(evidence.artifact_identity, artifact.artifact_identity)
            self.assertEqual(evidence.token_count, artifact.token_count)
            self.assertEqual(evidence.conditioning_shape, artifact.conditioning_shape)
            self.assertEqual(evidence.tensor_checksum, artifact.tensor_checksum)

            command = build_generation_command(validated, python="python")
            self.assertNotIn("a replayable conditioning prompt", command)
            self.assertEqual(
                command[command.index("--conditioning-artifact") + 1],
                str(artifact.path.resolve()),
            )

            namespace = reserve_run_namespace(validated)
            config = build_render_config(validated, namespace, command, repo_root=ROOT)
            self.assertEqual(config["conditioning_source"], "artifact-replay")
            self.assertEqual(config["conditioning_artifact_path"], str(artifact.path.resolve()))
            self.assertEqual(
                config["conditioning_artifact"],
                {
                    "path": str(artifact.path.resolve()),
                    "artifact_identity": artifact.artifact_identity,
                    "token_count": artifact.token_count,
                    "conditioning_shape": list(artifact.conditioning_shape),
                    "tensor_checksum": artifact.tensor_checksum,
                },
            )
            self.assertEqual(config["text_encoder"]["conditioning_source"], "artifact-replay")

    def test_replay_validation_rejects_prompt_missing_corrupt_and_provenance_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _artifact(root)
            cases = (
                ("prompt mismatch", _request(root, prompt="a different prompt", conditioning_artifact_path=artifact.path), "prompt identity"),
                ("missing artifact", _request(root, conditioning_artifact_path=root / "missing.npz"), "not a readable file"),
            )
            for label, request, message in cases:
                with self.subTest(label=label), self.assertRaisesRegex(RenderValidationError, message):
                    validate_render_request(
                        request,
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

            corrupt = root / "corrupt.npz"
            corrupt.write_bytes(b"not a conditioning artifact")
            with self.assertRaisesRegex(RenderValidationError, "could not read conditioning artifact"):
                validate_render_request(
                    _request(root, conditioning_artifact_path=corrupt),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )

            other_checkpoint = root / "other-checkpoint"
            _checkpoint(other_checkpoint)
            (other_checkpoint / "tokenizer" / "tokenizer_config.json").write_text(
                '{"changed": true}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RenderValidationError, "identity does not match"):
                validate_render_request(
                    _request(
                        root,
                        checkpoint_root=other_checkpoint,
                        conditioning_artifact_path=artifact.path,
                    ),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )

    def test_replay_is_t2v_canonical_only_and_rejects_heretic_and_image_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = _artifact(root)
            for label, changes, message in (
                ("I2V", {"mode": I2V, "image_paths": (root / "image.png",)}, "T2V-only"),
                ("FIRST_LAST", {"mode": FIRST_LAST, "image_paths": (root / "first.png", root / "last.png")}, "T2V-only"),
                ("Heretic", {"text_encoder_id": HERETIC_ENCODER_ID}, "Canonical Qwen3-VL"),
            ):
                with self.subTest(label=label), self.assertRaisesRegex(RenderValidationError, message):
                    validate_render_request(
                        _request(root, conditioning_artifact_path=artifact.path, **changes),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

    def test_browser_and_server_serialization_preserve_optional_path_and_blank_absence(self) -> None:
        self.assertIn('id="conditioning-artifact"', PAGE)
        self.assertIn("Conditioning artifact (optional)", PAGE)
        self.assertIn("if (conditioningArtifactPath) form.set('conditioning_artifact_path', conditioningArtifactPath);", PAGE)
        fields: dict[str, object] = {
            "mode": T2V,
            "prompt": "a replayable conditioning prompt",
            "steps": "16",
            "duration_seconds": "5",
            "seed": "0",
            "width": "512",
            "height": "512",
            "output_root": "out/render-lab",
            "output_name": "render.mp4",
            "conditioning_artifact_path": "  /tmp/conditioning-artifact.npz  ",
        }
        request = _render_request_from_fields(fields)
        self.assertEqual(request.normalized().conditioning_artifact_path, Path("/tmp/conditioning-artifact.npz"))
        blank = _render_request_from_fields(fields | {"conditioning_artifact_path": "   "})
        self.assertIsNone(blank.normalized().conditioning_artifact_path)

    def test_render_lab_admission_path_is_mlxfree_and_heretic_internal_command_remains_separate(self) -> None:
        source = (ROOT / "tools" / "render_lab" / "runner.py").read_text(encoding="utf-8")
        self.assertNotIn("import mlx", source)
        self.assertNotIn("MiniMaxH3TextEncoder", source)
        self.assertIn("load_conditioning_artifact", source)
        self.assertIn("validate_conditioning_artifact", source)


if __name__ == "__main__":
    unittest.main()

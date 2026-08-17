"""MLX-free production-entry routing contracts for conditioning encode/replay."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ConditioningReplayContractTests(unittest.TestCase):
    def test_encode_entry_is_conditioner_only(self) -> None:
        path = ROOT / "scripts" / "encode_conditioning.py"
        tree = ast.parse(path.read_text())
        imported = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        self.assertNotIn("minimax_h3_mlx.pipeline", imported)
        source = path.read_text()
        self.assertIn("MiniMaxH3TextEncoder", source)
        self.assertNotIn("load_dit", source)
        self.assertNotIn("load_video_vae", source)
        self.assertNotIn("load_audio_vae", source)
        self.assertIn("create_conditioning_artifact", source)
        self.assertIn("clean process exit", source)

    def test_replay_factory_validates_before_live_qwen_import_and_skips_construction(self) -> None:
        source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
        tree = ast.parse(source)
        factory = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "from_pretrained"
        )
        function_source = ast.get_source_segment(source, factory)
        self.assertIsNotNone(function_source)
        assert function_source is not None
        self.assertIn("load_conditioning_artifact", function_source)
        self.assertIn("validate_conditioning_artifact", function_source)
        self.assertIn("conditioning_artifact=loaded_conditioning_artifact", function_source)
        self.assertIn("text_encoder = None", function_source)
        self.assertIn("Qwen/text encoder construction: skipped (artifact replay)", function_source)
        self.assertLess(
            function_source.index("if loaded_conditioning_artifact is None:"),
            function_source.index("from .text_encoder import MiniMaxH3TextEncoder"),
        )

    def test_replay_conditioning_source_has_no_live_fallback(self) -> None:
        source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
        tree = ast.parse(source)
        prepare = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_conditioning"
        )
        function_source = ast.get_source_segment(source, prepare)
        self.assertIsNotNone(function_source)
        assert function_source is not None
        self.assertIn("self._conditioning_artifact is not None", function_source)
        self.assertIn("conditioning_mlx", function_source)
        self.assertIn("artifact replay is text-only", function_source)
        replay_block = function_source.split("if self._conditioning_artifact is not None:", 1)[1].split(
            "if self.text_encoder is None:", 1
        )[0]
        self.assertNotIn("self.text_encoder.encode", replay_block)

    def test_generate_cli_exposes_artifact_replay_without_changing_live_default(self) -> None:
        source = (ROOT / "scripts" / "generate.py").read_text()
        self.assertIn('parser.add_argument(\n        "prompt",', source)
        self.assertIn('nargs="?"', source)
        self.assertIn('"--conditioning-artifact"', source)
        self.assertIn("conditioning_artifact=args.conditioning_artifact", source)
        self.assertIn("prompt is required unless --conditioning-artifact is supplied", source)

    def test_generate_cli_routes_omitted_prompt_to_artifact_replay(self) -> None:
        calls: dict[str, object] = {}

        class FakePipeline:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["factory"] = {"args": args, **kwargs}
                return cls()

            def __call__(self, prompt, **kwargs):
                calls["generation"] = {"prompt": prompt, **kwargs}
                return SimpleNamespace(
                    video=SimpleNamespace(shape=(1, 1, 1, 3)),
                    audio=SimpleNamespace(shape=(2, 4)),
                    fps=24,
                    sample_rate=32_000,
                    seconds_per_step=1.0,
                    total_seconds=1.0,
                )

        pipeline_stub = types.ModuleType("minimax_h3_mlx.pipeline")
        pipeline_stub.MiniMaxH3Pipeline = FakePipeline
        media_stub = types.ModuleType("minimax_h3_mlx.media")
        media_stub.FFmpegUnavailableError = type("FFmpegUnavailableError", (Exception,), {})
        media_stub.save_frames = lambda *args, **kwargs: None
        media_stub.save_mp4 = lambda *args, **kwargs: None
        media_stub.save_wav = lambda *args, **kwargs: None

        spec = importlib.util.spec_from_file_location("conditioning_replay_generate", ROOT / "scripts" / "generate.py")
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules,
            {
                "minimax_h3_mlx.pipeline": pipeline_stub,
                "minimax_h3_mlx.media": media_stub,
            },
        ), patch.object(
            sys,
            "argv",
            [
                "generate.py",
                "--conditioning-artifact",
                "/tmp/conditioning-artifact.npz",
                "--checkpoint",
                "/models/FL2VA",
                "--transformer",
                "/models/h3-6bit",
                "--lightx-lora",
                "/models/lightx.safetensors",
                "--steps",
                "8",
                "--turbo-steps",
                "8",
                "--height",
                "544",
                "--width",
                "960",
                "--duration",
                "5.1666667",
                "--output",
                "/tmp/conditioning-replay.mp4",
            ],
        ):
            spec.loader.exec_module(module)
            self.assertEqual(module.main(), 0)

        factory = calls["factory"]
        generation = calls["generation"]
        self.assertIsNone(generation["prompt"])
        self.assertEqual(factory["conditioning_artifact"], "/tmp/conditioning-artifact.npz")
        self.assertTrue(factory["unload_text_encoder"])


if __name__ == "__main__":
    unittest.main()

"""MLX-free contracts for MiniMax-H3 still-image conditioning."""

from __future__ import annotations

import importlib.machinery
import json
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


def _install_mlx_stub() -> types.ModuleType:
    """Keep request-construction tests away from MLX/Metal entirely."""
    core = types.ModuleType("mlx.core")
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
    core.bfloat16 = object()
    core.array = lambda value, *args, **kwargs: np.asarray(value)

    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    return core


_MLX_STUB = _install_mlx_stub()

from minimax_h3_mlx.config import TAG_TEXT, TAG_VIDEO  # noqa: E402
from minimax_h3_mlx import text_encoder as text_encoder_module  # noqa: E402
from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder  # noqa: E402

text_encoder_module.mx = _MLX_STUB


ROOT = Path(__file__).resolve().parents[1]
H3_IMAGE_PROCESSOR_CONFIG = {
    "size": {"longest_edge": 16777216, "shortest_edge": 65536},
    "patch_size": 16,
    "temporal_patch_size": 2,
    "merge_size": 2,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "processor_class": "Qwen3VLProcessor",
    "image_processor_type": "Qwen2VLImageProcessorFast",
}


class FakeTokenizer:
    special_ids = {
        "<|vision_start|>": 10,
        "<|image_pad|>": 11,
        "<|vision_end|>": 12,
    }

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_ids[token]

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        if text.startswith("<Picture "):
            return {"input_ids": [20, 21, 22]}
        return {"input_ids": [100 + index for index in range(len(text))]}


def make_encoder(root: Path) -> MiniMaxH3TextEncoder:
    processor_dir = root / "processor"
    processor_dir.mkdir()
    (processor_dir / "preprocessor_config.json").write_text(
        json.dumps(H3_IMAGE_PROCESSOR_CONFIG), encoding="utf-8"
    )
    model_dir = root / "text_encoder"
    model_dir.mkdir()

    encoder = object.__new__(MiniMaxH3TextEncoder)
    encoder._model_dir = model_dir
    encoder._tokenizer = FakeTokenizer()
    encoder._image_processor = None
    return encoder


def as_numpy(value) -> np.ndarray:
    return np.asarray(value)


class TextEncoderImageConditioningTests(unittest.TestCase):
    def test_text_only_request_does_not_construct_an_image_processor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(Path(directory))
            input_ids, token_tags, vision_inputs = encoder.build_request("go", None)

        self.assertIsNone(vision_inputs)
        np.testing.assert_array_equal(as_numpy(input_ids), [[100, 101]])
        np.testing.assert_array_equal(token_tags, [TAG_TEXT, TAG_TEXT])
        self.assertIsNone(encoder._image_processor)

    def test_image_request_uses_pil_processor_without_composite_processor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(Path(directory))
            image = Image.new("RGB", (8, 8), (17, 34, 51))

            with patch("transformers.AutoProcessor.from_pretrained", side_effect=AssertionError):
                input_ids, token_tags, vision_inputs = encoder.build_request("go", [image])

            self.assertEqual(type(encoder.image_processor).__name__, "Qwen2VLImageProcessorPil")
            self.assertEqual(encoder.image_processor.backend, "pil")
            self.assertFalse(hasattr(encoder.image_processor, "video_processor"))

        self.assertIsNotNone(vision_inputs)
        pixel_values, image_grid_thw = vision_inputs
        self.assertEqual(pixel_values.shape, (256, 1536))
        self.assertEqual(pixel_values.dtype, np.float32)
        np.testing.assert_array_equal(image_grid_thw, [[1, 16, 16]])
        np.testing.assert_allclose(
            pixel_values[0],
            np.concatenate(
                [
                    np.full(512, -0.8666667, dtype=np.float32),
                    np.full(512, -0.7333333, dtype=np.float32),
                    np.full(512, -0.6, dtype=np.float32),
                ]
            ),
            rtol=0,
            atol=1e-6,
        )

        expected_ids = [[20, 21, 22, 10, *([11] * 64), 12, 100, 101]]
        np.testing.assert_array_equal(as_numpy(input_ids), expected_ids)
        np.testing.assert_array_equal(
            token_tags,
            [TAG_TEXT] * 3 + [TAG_VIDEO] * 66 + [TAG_TEXT] * 2,
        )

    def test_multiple_still_images_preserve_per_image_grid_and_token_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(Path(directory))
            images = [
                Image.new("RGB", (8, 8), (0, 0, 0)),
                Image.new("RGB", (8, 12), (255, 255, 255)),
            ]
            input_ids, token_tags, vision_inputs = encoder.build_request("x", images)

        pixel_values, image_grid_thw = vision_inputs
        self.assertEqual(pixel_values.shape, (536, 1536))
        np.testing.assert_array_equal(image_grid_thw, [[1, 16, 16], [1, 20, 14]])
        expected_ids = [
            20, 21, 22, 10, *([11] * 64), 12,
            20, 21, 22, 10, *([11] * 70), 12,
            100,
        ]
        np.testing.assert_array_equal(as_numpy(input_ids), [expected_ids])
        np.testing.assert_array_equal(
            token_tags,
            [TAG_TEXT] * 3
            + [TAG_VIDEO] * 66
            + [TAG_TEXT] * 3
            + [TAG_VIDEO] * 72
            + [TAG_TEXT],
        )

    def test_t2v_request_remains_text_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            encoder = make_encoder(Path(directory))
            input_ids, token_tags, vision_inputs = encoder.build_request("a prompt", [])

        self.assertIsNone(vision_inputs)
        self.assertIsNone(encoder._image_processor)
        np.testing.assert_array_equal(as_numpy(input_ids), [[100 + i for i in range(8)]])
        np.testing.assert_array_equal(token_tags, [TAG_TEXT] * 8)

    def test_bounded_smoke_proves_no_torch_video_processor_or_torchvision_import(self) -> None:
        smoke = textwrap.dedent(
            r'''
            import builtins
            import importlib.machinery
            import json
            import sys
            import tempfile
            import types
            from pathlib import Path

            import numpy as np
            from PIL import Image

            core = types.ModuleType("mlx.core")
            core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
            core.bfloat16 = object()
            core.array = lambda value, *args, **kwargs: np.asarray(value)
            mlx = types.ModuleType("mlx")
            mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
            mlx.__path__ = []
            mlx.core = core
            sys.modules["mlx"] = mlx
            sys.modules["mlx.core"] = core

            blocked = []
            original_import = builtins.__import__

            def guarded_import(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    blocked.append(name)
                    raise AssertionError(f"unexpected torch import: {name}")
                if name == "torchvision" or name.startswith("torchvision."):
                    blocked.append(name)
                    raise AssertionError(f"unexpected torchvision import: {name}")
                return original_import(name, *args, **kwargs)

            builtins.__import__ = guarded_import

            from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

            config = {
                "size": {"longest_edge": 16777216, "shortest_edge": 65536},
                "patch_size": 16,
                "temporal_patch_size": 2,
                "merge_size": 2,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
                "processor_class": "Qwen3VLProcessor",
                "image_processor_type": "Qwen2VLImageProcessorFast",
            }

            class Tokenizer:
                def convert_tokens_to_ids(self, token):
                    return {"<|vision_start|>": 10, "<|image_pad|>": 11, "<|vision_end|>": 12}[token]

                def __call__(self, text, *, add_special_tokens):
                    return {"input_ids": list(range(len(text)))}

            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "processor").mkdir()
                (root / "processor" / "preprocessor_config.json").write_text(json.dumps(config))
                (root / "text_encoder").mkdir()
                encoder = object.__new__(MiniMaxH3TextEncoder)
                encoder._model_dir = root / "text_encoder"
                encoder._tokenizer = Tokenizer()
                encoder._image_processor = None
                _, _, vision = encoder.build_request("x", [Image.new("RGB", (8, 8), (1, 2, 3))])

            assert blocked == [], blocked
            assert "transformers.models.qwen3_vl.video_processing_qwen3_vl" not in sys.modules
            assert vision[0].shape == (256, 1536)
            assert vision[1].tolist() == [[1, 16, 16]]
            print(json.dumps({"blocked_imports": blocked, "video_processor_module": False, "grid": vision[1].tolist()}))
            '''
        )
        result = subprocess.run(
            [sys.executable, "-c", smoke],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(receipt, {"blocked_imports": [], "video_processor_module": False, "grid": [[1, 16, 16]]})


if __name__ == "__main__":
    unittest.main()

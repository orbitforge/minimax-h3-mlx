"""MLX-free contract for the intentional no-crop keyframe policy."""

from __future__ import annotations

import importlib.machinery
import sys
import types
import unittest

import numpy as np
from PIL import Image


def _install_mlx_stub() -> None:
    core = types.ModuleType("mlx.core")
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core


_install_mlx_stub()

from minimax_h3_mlx.packing import prepare_keyframe_image  # noqa: E402


class KeyframeImageResizeContractTests(unittest.TestCase):
    def test_aspect_mismatch_is_resized_without_crop_or_padding(self) -> None:
        source = Image.fromarray(
            np.array(
                [
                    [[255, 0, 0], [255, 32, 0], [0, 32, 255], [0, 0, 255]],
                    [[255, 64, 0], [255, 96, 0], [0, 96, 255], [0, 64, 255]],
                ],
                dtype=np.uint8,
            ),
            mode="RGB",
        )
        expected = source.resize((4, 4), Image.Resampling.LANCZOS)

        actual = prepare_keyframe_image(source, 4, 4)

        self.assertEqual(actual.size, (4, 4))
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


if __name__ == "__main__":
    unittest.main()

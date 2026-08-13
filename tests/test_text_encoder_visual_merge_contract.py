"""MLX-free synthetic tests for the Qwen visual-row merge contract."""

from __future__ import annotations

import importlib.machinery
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np


def _install_mlx_stub() -> types.ModuleType:
    core = types.ModuleType("mlx.core")
    core.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None)
    core.bfloat16 = object()
    core.array = lambda value, *args, **kwargs: np.asarray(value)
    core.sum = lambda value, *args, **kwargs: np.asarray(value).sum(*args, **kwargs)
    core.broadcast_to = np.broadcast_to

    mlx = types.ModuleType("mlx")
    mlx.__spec__ = importlib.machinery.ModuleSpec("mlx", loader=None, is_package=True)
    mlx.__path__ = []
    mlx.core = core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = core
    return core


_MLX_STUB = _install_mlx_stub()

from minimax_h3_mlx import text_encoder as text_encoder_module  # noqa: E402

text_encoder_module.mx = _MLX_STUB


def _reference_masked_scatter(final_embedding, image_mask_expanded, scaled_image_features):
    result = np.array(final_embedding, copy=True)
    positions = np.flatnonzero(np.asarray(image_mask_expanded).reshape(-1))
    result.reshape(-1)[positions] = np.asarray(scaled_image_features).reshape(-1)
    return result


def _reference_modules() -> dict[str, types.ModuleType]:
    root = types.ModuleType("mlx_vlm")
    root.__path__ = []
    models = types.ModuleType("mlx_vlm.models")
    models.__path__ = []
    qwen = types.ModuleType("mlx_vlm.models.qwen3_vl")
    qwen.__path__ = []
    implementation = types.ModuleType("mlx_vlm.models.qwen3_vl.qwen3_vl")
    implementation.masked_scatter = _reference_masked_scatter
    root.models = models
    models.qwen3_vl = qwen
    qwen.qwen3_vl = implementation
    return {
        "mlx_vlm": root,
        "mlx_vlm.models": models,
        "mlx_vlm.models.qwen3_vl": qwen,
        "mlx_vlm.models.qwen3_vl.qwen3_vl": implementation,
    }


class TextEncoderVisualMergeContractTests(unittest.TestCase):
    def setUp(self) -> None:
        # The existing request-construction tests install a smaller MLX stub at
        # module import time; restore this contract fixture before each test.
        text_encoder_module.mx = _MLX_STUB

    def test_compact_rows_insert_only_at_selected_positions_and_preserve_dtype(self) -> None:
        base = np.arange(1 * 6 * 2, dtype=np.float32).reshape(1, 6, 2)
        mask = np.array([[False, True, True, False, True, False]])
        visual = np.array([[100, 101], [200, 201], [300, 301]], dtype=np.float64)

        with patch.dict(sys.modules, _reference_modules()):
            merged = text_encoder_module._merge_visual_embeddings(base, mask, visual)

        expected = base.copy()
        expected[0, 1] = [100, 101]
        expected[0, 2] = [200, 201]
        expected[0, 4] = [300, 301]
        np.testing.assert_array_equal(merged, expected)
        self.assertEqual(merged.dtype, base.dtype)

    def test_one_image_ordering_is_compact_row_order(self) -> None:
        base = np.zeros((1, 5, 1), dtype=np.float32)
        mask = np.array([[False, True, True, True, False]])
        visual = np.array([[11], [22], [33]], dtype=np.float32)

        with patch.dict(sys.modules, _reference_modules()):
            merged = text_encoder_module._merge_visual_embeddings(base, mask, visual)

        np.testing.assert_array_equal(merged[0, :, 0], [0, 11, 22, 33, 0])

    def test_two_image_ordering_is_image_one_then_image_two(self) -> None:
        base = np.zeros((1, 9, 1), dtype=np.float32)
        # Two visual blocks separated by ordinary label/text rows.
        mask = np.array([[False, True, True, False, False, True, True, True, False]])
        visual = np.array([[101], [102], [201], [202], [203]], dtype=np.float32)

        with patch.dict(sys.modules, _reference_modules()):
            merged = text_encoder_module._merge_visual_embeddings(base, mask, visual)

        np.testing.assert_array_equal(merged[0, :, 0], [0, 101, 102, 0, 0, 201, 202, 203, 0])

    def test_visual_cardinality_mismatch_is_rejected(self) -> None:
        base = np.zeros((1, 4, 2), dtype=np.float32)
        mask = np.array([[False, True, True, False]])
        visual = np.zeros((3, 2), dtype=np.float32)

        with self.assertRaisesRegex(ValueError, "mask rows 2"):
            text_encoder_module._merge_visual_embeddings(base, mask, visual)

    def test_deepstack_cardinality_and_width_are_rejected(self) -> None:
        mask = np.array([[False, True, True, False]])

        with self.assertRaisesRegex(ValueError, "deepstack rows"):
            text_encoder_module._validate_visual_embedding_rows(
                "deepstack rows", mask, np.zeros((1, 3), dtype=np.float32), 3
            )
        with self.assertRaisesRegex(ValueError, "deepstack width"):
            text_encoder_module._validate_visual_embedding_rows(
                "deepstack width", mask, np.zeros((2, 2), dtype=np.float32), 3
            )


if __name__ == "__main__":
    unittest.main()

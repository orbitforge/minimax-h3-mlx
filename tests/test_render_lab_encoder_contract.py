"""MLX-free baseline Render Lab safety contract."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools.render_lab.runner import (
    T2V,
    RenderRequest,
    RenderValidationError,
    validate_render_request,
)


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


class RenderLabBaselineSafetyTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

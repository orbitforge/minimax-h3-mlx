"""MLX-free contracts for the legacy browser surface transformer seam."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SURFACE_PATH = ROOT / "scripts" / "minimax_h3_surface.py"


def _load_surface():
    spec = importlib.util.spec_from_file_location("legacy_surface_contract", SURFACE_PATH)
    assert spec is not None and spec.loader is not None
    surface = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(surface)
    return surface


SURFACE = _load_surface()


class LegacySurfaceTransformerContractTests(unittest.TestCase):
    def test_surface_default_uses_streamed_adaln_basename(self) -> None:
        self.assertEqual(SURFACE.TRANSFORMER.resolve().name, "minimax-h3-mlx-6bit-streamed-adaln")

    def test_surface_default_does_not_use_stale_resident_basename(self) -> None:
        self.assertNotEqual(SURFACE.TRANSFORMER.resolve().name, "minimax-h3-mlx-6bit")

    def test_start_job_propagates_transformer_path_to_generate_child(self) -> None:
        popen_calls: list[tuple[list[str], dict[str, object]]] = []
        thread_starts: list[object] = []

        class FakeProcess:
            stdout = ()

        def fake_popen(command, **kwargs):
            popen_calls.append((list(command), dict(kwargs)))
            return FakeProcess()

        class FakeThread:
            def __init__(self, *, target, args, daemon):
                self.target = target
                self.args = args
                self.daemon = daemon

            def start(self) -> None:
                thread_starts.append(self)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "output.mp4"
            with SURFACE.JOB_LOCK:
                SURFACE.JOB.update(process=None, log="", exit_code=None)
            try:
                with patch.object(SURFACE.subprocess, "Popen", side_effect=fake_popen), patch.object(
                    SURFACE.threading, "Thread", FakeThread
                ):
                    result, status = SURFACE.start_job(
                        {"prompt": "a contract prompt", "length": 5, "megapixels": "0.2", "output": str(output)}
                    )
            finally:
                with SURFACE.JOB_LOCK:
                    SURFACE.JOB.update(process=None, log="", exit_code=None)

        self.assertEqual((result, status), ({"ok": True}, 200))
        self.assertEqual(len(popen_calls), 1)
        command, kwargs = popen_calls[0]
        transformer_index = command.index("--transformer") + 1
        self.assertEqual(command[transformer_index], str(SURFACE.TRANSFORMER))
        self.assertEqual(Path(command[transformer_index]).name, "minimax-h3-mlx-6bit-streamed-adaln")
        self.assertEqual(Path(command[command.index("-u") + 1]).name, "generate.py")
        self.assertEqual(kwargs["cwd"], SURFACE.REPO)
        self.assertEqual(len(thread_starts), 1)


if __name__ == "__main__":
    unittest.main()

"""MLX-free contracts for the legacy browser surface runtime seam."""

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
    @staticmethod
    def _payload(output: Path, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "prompt": "a contract prompt",
            "length": 5,
            "megapixels": "0.2",
            "output": str(output),
        }
        payload.update(overrides)
        return payload

    def _start_with_fake_process(self, payload: dict[str, object]):
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

        with SURFACE.JOB_LOCK:
            SURFACE.JOB.update(
                process=None,
                log="",
                exit_code=None,
                runtime_id=SURFACE.DEFAULT_RUNTIME_ID,
                runtime_label=SURFACE.RUNTIME_LABELS[SURFACE.DEFAULT_RUNTIME_ID],
            )
        try:
            with patch.object(SURFACE.subprocess, "Popen", side_effect=fake_popen), patch.object(
                SURFACE.threading, "Thread", FakeThread
            ):
                result, status = SURFACE.start_job(payload)
            with SURFACE.JOB_LOCK:
                snapshot = dict(SURFACE.JOB)
        finally:
            with SURFACE.JOB_LOCK:
                SURFACE.JOB.update(
                    process=None,
                    log="",
                    exit_code=None,
                    runtime_id=SURFACE.DEFAULT_RUNTIME_ID,
                    runtime_label=SURFACE.RUNTIME_LABELS[SURFACE.DEFAULT_RUNTIME_ID],
                )
        return result, status, popen_calls, thread_starts, snapshot

    def test_runtime_selector_exposes_beta_choice_without_artifact_details(self) -> None:
        self.assertIn('<select id="runtime">', SURFACE.PAGE)
        self.assertIn('<option value="beta-0.6">Beta 0.6</option>', SURFACE.PAGE)
        self.assertIn("runtime: $('runtime').value", SURFACE.PAGE)
        self.assertNotIn("Slice 025", SURFACE.PAGE)
        self.assertNotIn("transformer", SURFACE.PAGE.lower())

    def test_beta_selection_has_named_runtime_identity(self) -> None:
        self.assertEqual(SURFACE.BETA_RUNTIME_ID, "beta-0.6")
        self.assertEqual(SURFACE.RUNTIME_LABELS[SURFACE.BETA_RUNTIME_ID], "Beta 0.6")

    def test_surface_default_uses_streamed_adaln_basename(self) -> None:
        self.assertEqual(SURFACE.TRANSFORMER.resolve().name, "minimax-h3-mlx-6bit-streamed-adaln")

    def test_surface_default_does_not_use_stale_resident_basename(self) -> None:
        self.assertNotEqual(SURFACE.TRANSFORMER.resolve().name, "minimax-h3-mlx-6bit")

    def test_start_job_propagates_transformer_path_to_generate_child(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "output.mp4"
            result, status, popen_calls, thread_starts, _snapshot = self._start_with_fake_process(
                self._payload(output)
            )

        self.assertEqual(
            (result, status),
            ({"ok": True, "runtime_id": "current", "runtime_label": "Current"}, 200),
        )
        self.assertEqual(len(popen_calls), 1)
        command, kwargs = popen_calls[0]
        transformer_index = command.index("--transformer") + 1
        self.assertEqual(command[transformer_index], str(SURFACE.TRANSFORMER))
        self.assertEqual(Path(command[transformer_index]).name, "minimax-h3-mlx-6bit-streamed-adaln")
        self.assertEqual(Path(command[command.index("-u") + 1]).name, "generate.py")
        self.assertEqual(kwargs["cwd"], SURFACE.REPO)
        self.assertEqual(len(thread_starts), 1)

    def test_missing_runtime_selection_preserves_current_manual_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "current.mp4"
            result, status, popen_calls, _threads, _snapshot = self._start_with_fake_process(
                self._payload(output)
            )

        self.assertEqual(status, 200)
        self.assertEqual(result["runtime_id"], SURFACE.DEFAULT_RUNTIME_ID)
        command = popen_calls[0][0]
        self.assertIn("--checkpoint", command)
        self.assertIn("--transformer", command)
        self.assertNotIn("--runtime", command)

    def test_beta_launch_uses_named_runtime_and_preserves_existing_parameters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "beta.mp4"
            runtime_assets = Path(directory) / "runtime-assets"
            with patch.dict(SURFACE.os.environ, {SURFACE.RUNTIME_ASSETS_ENV: str(runtime_assets)}):
                result, status, popen_calls, _threads, snapshot = self._start_with_fake_process(
                    self._payload(
                        output,
                        runtime=SURFACE.BETA_RUNTIME_ID,
                        prompt="beta prompt",
                        length=9,
                        megapixels="0.5",
                    )
                )

        self.assertEqual(
            (result, status),
            ({"ok": True, "runtime_id": "beta-0.6", "runtime_label": "Beta 0.6"}, 200),
        )
        command, kwargs = popen_calls[0]
        self.assertEqual(command[command.index("--runtime") + 1], "beta-0.6")
        self.assertNotIn("--checkpoint", command)
        self.assertNotIn("--transformer", command)
        self.assertIn("beta prompt", command)
        self.assertEqual(command[command.index("--megapixels") + 1], "0.5")
        self.assertEqual(command[command.index("--duration") + 1], "9")
        self.assertEqual(command[command.index("--output") + 1], str(output))
        self.assertEqual(kwargs["env"][SURFACE.RUNTIME_ASSETS_ENV], str(runtime_assets))
        self.assertEqual(snapshot["runtime_id"], "beta-0.6")
        self.assertEqual(snapshot["runtime_label"], "Beta 0.6")
        self.assertIn("runtime: Beta 0.6 (beta-0.6)", snapshot["log"])
        self.assertIn("--runtime beta-0.6", snapshot["log"])

    def test_beta_launch_never_falls_back_to_manual_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "beta.mp4"
            command = SURFACE.build_generation_command(
                SURFACE.BETA_RUNTIME_ID,
                "prompt",
                "0.2",
                5,
                output,
            )

        self.assertEqual(command[command.index("--runtime") + 1], "beta-0.6")
        self.assertNotIn(str(SURFACE.CHECKPOINT), command)
        self.assertNotIn(str(SURFACE.TRANSFORMER), command)

    def test_beta_admission_failure_remains_visible_without_legacy_retry(self) -> None:
        class FailedProcess:
            stdout = ("RUNTIME_ASSET_INVALID: profile rejected before model loading\n",)

            def wait(self) -> int:
                return 2

        with SURFACE.JOB_LOCK:
            SURFACE.JOB.update(
                process=FailedProcess(),
                log="runtime: Beta 0.6 (beta-0.6)\n",
                exit_code=None,
                runtime_id=SURFACE.BETA_RUNTIME_ID,
                runtime_label=SURFACE.RUNTIME_LABELS[SURFACE.BETA_RUNTIME_ID],
            )
        try:
            SURFACE.consume_output(FailedProcess())
            with SURFACE.JOB_LOCK:
                snapshot = dict(SURFACE.JOB)
        finally:
            with SURFACE.JOB_LOCK:
                SURFACE.JOB.update(
                    process=None,
                    log="",
                    exit_code=None,
                    runtime_id=SURFACE.DEFAULT_RUNTIME_ID,
                    runtime_label=SURFACE.RUNTIME_LABELS[SURFACE.DEFAULT_RUNTIME_ID],
                )

        self.assertEqual(snapshot["exit_code"], 2)
        self.assertEqual(snapshot["runtime_id"], "beta-0.6")
        self.assertIn("RUNTIME_ASSET_INVALID", snapshot["log"])
        self.assertNotIn(str(SURFACE.TRANSFORMER), snapshot["log"])

    def test_invalid_runtime_is_rejected_before_child_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "invalid.mp4"
            with patch.object(SURFACE.subprocess, "Popen") as popen:
                result, status = SURFACE.start_job(
                    self._payload(output, runtime="unsupported-runtime")
                )

        self.assertEqual((result, status), ({"error": "Choose a supported runtime."}, 400))
        popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""MLX-free tests for ffmpeg availability and encoding failure classification."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx import media


def _load_generate_module():
    fake_pipeline = types.ModuleType("minimax_h3_mlx.pipeline")
    fake_pipeline.MiniMaxH3Pipeline = object
    spec = importlib.util.spec_from_file_location(
        "generate_media_failure_tests",
        ROOT / "scripts" / "generate.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"minimax_h3_mlx.pipeline": fake_pipeline}):
        spec.loader.exec_module(module)
    return module


generate = _load_generate_module()


class FakeResult:
    video = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    audio = np.zeros((2, 4), dtype=np.float32)
    sample_rate = 32000
    fps = 24
    seconds_per_step = 0.1
    total_seconds = 0.2


class FakePipeline:
    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        return cls()

    def __call__(self, *_args, **_kwargs):
        return FakeResult()


class FakeFFmpeg:
    def __init__(self, *, returncode: int = 0, payload: bytes = b"synthetic-mp4") -> None:
        self.returncode = returncode
        self.payload = payload
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        Path(argv[-1]).write_bytes(self.payload)
        stderr = b"encoder failed" if self.returncode else b""
        return subprocess.CompletedProcess(argv, self.returncode, stdout=b"", stderr=stderr)


class MediaFailureClassificationTests(unittest.TestCase):
    def _run_generate(self, output: Path, save_mp4) -> int:
        with (
            mock.patch.object(generate, "MiniMaxH3Pipeline", FakePipeline),
            mock.patch.object(generate, "save_mp4", side_effect=save_mp4),
            mock.patch.object(sys, "argv", ["generate.py", "prompt", "-o", str(output)]),
        ):
            return generate.main()

    def test_missing_ffmpeg_has_unavailable_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "clip.mp4"
            with (
                mock.patch.object(media.shutil, "which", return_value=None),
                mock.patch.object(media.subprocess, "run") as run,
            ):
                with self.assertRaises(media.FFmpegUnavailableError):
                    media.save_mp4(output, FakeResult.video, 24, FakeResult.audio)
            run.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

    def test_missing_ffmpeg_triggers_png_wav_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip.mp4"
            save_mp4 = mock.Mock(side_effect=media.FFmpegUnavailableError("missing"))
            save_frames = mock.Mock()
            save_wav = mock.Mock(wraps=media.save_wav)
            with (
                mock.patch.object(generate, "save_frames", save_frames),
                mock.patch.object(generate, "save_wav", save_wav),
            ):
                result = self._run_generate(output, save_mp4)

            self.assertEqual(result, 0)
            self.assertEqual(save_mp4.call_args.args[0], output)
            self.assertIs(save_mp4.call_args.args[1], FakeResult.video)
            self.assertIs(save_mp4.call_args.args[3], FakeResult.audio)
            self.assertEqual(save_frames.call_args.args, (output.with_suffix(""), FakeResult.video))
            self.assertEqual(save_wav.call_args.args, (output.with_suffix(".wav"), FakeResult.audio, 32000))
            self.assertEqual(sorted(path.name for path in Path(directory).iterdir()), ["clip.wav"])

    def test_encoding_failure_does_not_trigger_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip.mp4"
            save_mp4 = mock.Mock(side_effect=media.FFmpegEncodingError("encoder failed"))
            save_frames = mock.Mock()
            save_wav = mock.Mock()
            with (
                mock.patch.object(generate, "save_frames", save_frames),
                mock.patch.object(generate, "save_wav", save_wav),
            ):
                with self.assertRaises(media.FFmpegEncodingError):
                    self._run_generate(output, save_mp4)

            save_frames.assert_not_called()
            save_wav.assert_not_called()

    def test_nonzero_ffmpeg_exit_propagates_as_encoding_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "clip.mp4"
            runner = FakeFFmpeg(returncode=7)
            with (
                mock.patch.object(media.shutil, "which", return_value="/fake/ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=runner),
            ):
                with self.assertRaisesRegex(media.FFmpegEncodingError, "encoder failed"):
                    media.save_mp4(output, FakeResult.video, 24)

            self.assertFalse(output.exists())
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                [],
            )

    def test_successful_mp4_still_returns_published_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "clip.mp4"
            runner = FakeFFmpeg(payload=b"encoded")
            with (
                mock.patch.object(media.shutil, "which", return_value="/fake/ffmpeg"),
                mock.patch.object(media.subprocess, "run", side_effect=runner),
            ):
                result = media.save_mp4(output, FakeResult.video, 24)

            self.assertEqual(result, output)
            self.assertEqual(output.read_bytes(), b"encoded")
            self.assertEqual(len(runner.calls), 1)


if __name__ == "__main__":
    unittest.main()

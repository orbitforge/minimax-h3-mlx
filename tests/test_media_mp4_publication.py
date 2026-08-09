"""MLX-free tests for atomic MP4 publication."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx import media


REAL_OS_REPLACE = os.replace


class FakeFFmpeg:
    def __init__(
        self,
        final_path: Path,
        events: list[tuple[object, ...]],
        *,
        returncode: int = 0,
        payload: bytes = b"synthetic-mp4",
        stderr: bytes = b"",
        raised: BaseException | None = None,
    ) -> None:
        self.final_path = final_path
        self.events = events
        self.returncode = returncode
        self.payload = payload
        self.stderr = stderr
        self.raised = raised
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.audio_inputs: list[tuple[Path, bool]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        for value in argv:
            if isinstance(value, str) and value.endswith(".wav"):
                audio_path = Path(value)
                self.audio_inputs.append((audio_path, audio_path.exists()))
        staged_path = Path(argv[-1])
        self.events.append(("ffmpeg", self.final_path.exists(), staged_path.exists(), staged_path))
        staged_path.write_bytes(self.payload)
        if self.raised is not None:
            raise self.raised
        return subprocess.CompletedProcess(argv, self.returncode, stdout=b"", stderr=self.stderr)


class RecordingReplace:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.calls: list[tuple[Path, Path]] = []

    def __call__(self, source, destination) -> None:
        source = Path(source)
        destination = Path(destination)
        self.calls.append((source, destination))
        self.events.append(("replace", destination.exists(), source.exists(), source, destination))
        REAL_OS_REPLACE(source, destination)


class AtomicMP4PublicationTests(unittest.TestCase):
    @staticmethod
    def _video() -> np.ndarray:
        return np.zeros((2, 2, 2, 3), dtype=np.uint8)

    @staticmethod
    def _partial_paths(directory: Path) -> list[Path]:
        return sorted(
            path for path in directory.iterdir() if path.name.endswith(".partial.mp4")
        )

    def _save_mp4(
        self,
        final_path: Path,
        runner: FakeFFmpeg,
        publisher: RecordingReplace,
        audio: np.ndarray | None = None,
    ) -> Path:
        with (
            mock.patch.object(media.shutil, "which", return_value="/fake/ffmpeg"),
            mock.patch.object(media.subprocess, "run", side_effect=runner),
            mock.patch.object(media.os, "replace", side_effect=publisher),
        ):
            return media.save_mp4(final_path, self._video(), 24, audio)

    def test_audio_mux_uses_temporary_sibling_wav_and_removes_it_after_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            events: list[tuple[object, ...]] = []
            publisher = RecordingReplace(events)
            runner = FakeFFmpeg(final_path, events, payload=b"encoded")

            result = self._save_mp4(
                final_path, runner, publisher, np.zeros((2, 4), dtype=np.float32)
            )

            self.assertEqual(result, final_path)
            self.assertEqual(final_path.read_bytes(), b"encoded")
            self.assertEqual(publisher.calls[0][1], final_path)
            self.assertEqual(len(runner.audio_inputs), 1)
            mux_wav_path, existed_at_ffmpeg = runner.audio_inputs[0]
            self.assertTrue(existed_at_ffmpeg)
            self.assertEqual(mux_wav_path.parent, final_path.parent)
            self.assertTrue(mux_wav_path.name.startswith(f".{final_path.name}.mux-"))
            self.assertTrue(mux_wav_path.name.endswith(".wav"))
            self.assertNotEqual(mux_wav_path, final_path.with_suffix(".wav"))
            self.assertFalse(mux_wav_path.exists())
            self.assertFalse(final_path.with_suffix(".wav").exists())

    def test_audio_mux_failure_cleans_temporary_wav_and_never_leaves_final_wav(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            events: list[tuple[object, ...]] = []
            runner = FakeFFmpeg(final_path, events, returncode=7, stderr=b"encoder failed")
            publisher = RecordingReplace(events)

            with self.assertRaisesRegex(media.FFmpegEncodingError, "encoder failed"):
                self._save_mp4(final_path, runner, publisher, np.zeros((2, 4), dtype=np.float32))

            mux_wav_path, existed_at_ffmpeg = runner.audio_inputs[0]
            self.assertTrue(existed_at_ffmpeg)
            self.assertFalse(mux_wav_path.exists())
            self.assertFalse(final_path.with_suffix(".wav").exists())
            self.assertFalse(final_path.exists())
            self.assertEqual(publisher.calls, [])

    def test_success_stages_to_unique_sibling_then_publishes_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            events: list[tuple[object, ...]] = []
            publisher = RecordingReplace(events)

            first_runner = FakeFFmpeg(final_path, events, payload=b"first")
            second_runner = FakeFFmpeg(final_path, events, payload=b"second")
            self._save_mp4(final_path, first_runner, publisher)
            self._save_mp4(final_path, second_runner, publisher)

            first_staged = Path(first_runner.calls[0][0][-1])
            second_staged = Path(second_runner.calls[0][0][-1])
            self.assertNotEqual(first_staged, second_staged)
            for staged_path in (first_staged, second_staged):
                self.assertEqual(staged_path.parent, final_path.parent)
                self.assertTrue(staged_path.name.endswith(".partial.mp4"))
                self.assertEqual(staged_path.suffix, ".mp4")
                self.assertNotEqual(staged_path, final_path)
                self.assertFalse(staged_path.exists())

            self.assertEqual(
                [(event[0], event[1], event[2]) for event in events],
                [("ffmpeg", False, True), ("replace", False, True),
                 ("ffmpeg", True, True), ("replace", True, True)],
            )
            self.assertEqual(final_path.read_bytes(), b"second")
            self.assertEqual(len(publisher.calls), 2)
            self.assertEqual([call[1] for call in publisher.calls], [final_path, final_path])

            argv = first_runner.calls[0][0]
            self.assertEqual(argv[-1], str(first_staged))
            self.assertIn("-c:v", argv)
            self.assertEqual(argv[argv.index("-c:v") + 1], "libx264")
            self.assertEqual(argv[argv.index("-crf") + 1], "18")
            pix_fmt_indices = [index for index, value in enumerate(argv) if value == "-pix_fmt"]
            self.assertEqual([argv[index + 1] for index in pix_fmt_indices], ["rgb24", "yuv420p"])

    def test_ffmpeg_failure_never_publishes_requested_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            events: list[tuple[object, ...]] = []
            runner = FakeFFmpeg(final_path, events, returncode=7, stderr=b"encoder failed")
            publisher = RecordingReplace(events)

            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                self._save_mp4(final_path, runner, publisher)

            self.assertFalse(final_path.exists())
            self.assertEqual(publisher.calls, [])
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(self._partial_paths(root), [])

    def test_failed_replacement_preserves_existing_final_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            final_path.write_bytes(b"original")
            events: list[tuple[object, ...]] = []
            runner = FakeFFmpeg(final_path, events, returncode=9, stderr=b"replacement failed")
            publisher = RecordingReplace(events)

            with self.assertRaisesRegex(RuntimeError, "replacement failed"):
                self._save_mp4(final_path, runner, publisher)

            self.assertEqual(final_path.read_bytes(), b"original")
            self.assertEqual(publisher.calls, [])
            self.assertEqual(self._partial_paths(root), [])

    def test_successful_replacement_atomically_overwrites_existing_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            final_path.write_bytes(b"old")
            events: list[tuple[object, ...]] = []
            runner = FakeFFmpeg(final_path, events, payload=b"new")
            publisher = RecordingReplace(events)

            result = self._save_mp4(final_path, runner, publisher)

            self.assertEqual(result, final_path)
            self.assertEqual(final_path.read_bytes(), b"new")
            self.assertEqual(len(publisher.calls), 1)
            self.assertEqual(publisher.calls[0][1], final_path)
            self.assertTrue(events[0][1])
            self.assertEqual(events[1][0], "replace")
            self.assertTrue(events[1][1])
            self.assertTrue(events[1][2])

    def test_python_failure_cleans_staged_artifact_and_preserves_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final_path = root / "clip.mp4"
            final_path.write_bytes(b"original")
            events: list[tuple[object, ...]] = []
            runner = FakeFFmpeg(
                final_path,
                events,
                raised=OSError("subprocess seam failed"),
            )
            publisher = RecordingReplace(events)

            with self.assertRaisesRegex(OSError, "subprocess seam failed"):
                self._save_mp4(final_path, runner, publisher, np.zeros((2, 4), dtype=np.float32))

            self.assertEqual(final_path.read_bytes(), b"original")
            self.assertEqual(publisher.calls, [])
            self.assertEqual(self._partial_paths(root), [])
            mux_wav_path, existed_at_ffmpeg = runner.audio_inputs[0]
            self.assertTrue(existed_at_ffmpeg)
            self.assertFalse(mux_wav_path.exists())
            self.assertFalse(final_path.with_suffix(".wav").exists())


if __name__ == "__main__":
    unittest.main()

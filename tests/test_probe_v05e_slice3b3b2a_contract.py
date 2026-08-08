"""MLX-free Slice 3B3B2A geometry-aware ffmpeg staging contracts."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v05d_derived_full_schedule.py"
SPEC = importlib.util.spec_from_file_location("probe_v05e_slice3b3b2a", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def geometry(size: int) -> dict[str, object]:
    return probe.canonical_geometry_contract(size)


def write_rgb_frame_set(directory: Path, *, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(probe.VIDEO_FRAME_COUNT):
        Image.new("RGB", (size, size), (index % 256, 17, 29)).save(
            directory / f"frame_{index:05d}.png"
        )


def write_stereo_wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(probe.AUDIO_SAMPLE_WIDTH_BYTES)
        handle.setframerate(probe.AUDIO_SAMPLE_RATE)
        handle.writeframes(b"\0" * (2 * probe.AUDIO_SAMPLE_COUNT * probe.AUDIO_SAMPLE_WIDTH_BYTES))


def media_fixture(root: Path, *, size: int) -> dict[str, object]:
    selected_geometry = geometry(size)
    frames = root / "frames"
    write_rgb_frame_set(frames, size=size)
    frame_manifest = root / "video-frame-manifest.json"
    probe._write_json(
        frame_manifest,
        probe.build_video_frame_manifest(
            frames,
            attempt_identifier="slice3b3b2a-attempt",
            geometry=selected_geometry,
        ),
    )
    wav = root / "audio.wav"
    write_stereo_wav(wav)
    wav_manifest = root / "audio-manifest.json"
    probe._write_json(
        wav_manifest,
        probe.build_audio_wav_manifest(wav, attempt_identifier="slice3b3b2a-attempt"),
    )
    paths = {
        "frames": frames,
        "video_frame_manifest": frame_manifest,
        "audio_wav": wav,
        "audio_manifest": wav_manifest,
        "mp4_partial": root / probe.MP4_PARTIAL_FILENAME,
        "mp4": root / probe.MP4_FINAL_FILENAME,
        "mp4_manifest": root / probe.MP4_MANIFEST_FILENAME,
        "attempt_identifier": "slice3b3b2a-attempt",
    }
    report = {
        "attempt": {"attempt_identifier": paths["attempt_identifier"]},
        "geometry": selected_geometry,
        "latent_generation_status": "completed",
        "video_status": "completed",
        "audio_status": "completed",
        "standalone_media_status": "completed",
        "video_decoder": {
            "release_gate_passed": True,
            "allocator_cache_zero": True,
            "worker_termination_confirmed": True,
            "worker_receipt": probe.decoder_worker_receipt("video", geometry=selected_geometry),
        },
        "audio_decoder": {
            "release_gate_passed": True,
            "allocator_cache_zero": True,
            "worker_termination_confirmed": True,
            "worker_receipt": probe.decoder_worker_receipt("audio"),
        },
    }
    paths["launch_gate"] = probe.build_mux_launch_gate(report, paths, geometry=selected_geometry)
    return paths


class FakeFFmpegRunner:
    def __init__(
        self,
        *,
        returncode: int = 0,
        write_output: bool = True,
        empty_output: bool = False,
        timeout: bool = False,
        stderr: str = "",
    ) -> None:
        self.returncode = returncode
        self.write_output = write_output
        self.empty_output = empty_output
        self.timeout = timeout
        self.stderr = stderr
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        if Path(argv[0]).name != "ffmpeg":
            raise AssertionError(f"unexpected subprocess: {argv[0]}")
        if self.timeout:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr=self.stderr or "timed out")
        output = Path(argv[-1])
        if self.write_output:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"" if self.empty_output else b"synthetic-partial-mp4")
        return subprocess.CompletedProcess(
            argv,
            self.returncode,
            stdout="synthetic stdout",
            stderr=self.stderr,
        )


def run_staging(paths: dict[str, object], *, runner: FakeFFmpegRunner):
    size = int(probe._read_json_object(Path(paths["video_frame_manifest"]), "manifest")["video_width"])
    return probe.execute_ffmpeg_staging(
        frames_directory=paths["frames"],
        video_manifest_path=paths["video_frame_manifest"],
        wav_path=paths["audio_wav"],
        audio_manifest_path=paths["audio_manifest"],
        mp4_partial_path=paths["mp4_partial"],
        mp4_final_path=paths["mp4"],
        mp4_manifest_path=paths["mp4_manifest"],
        attempt_identifier=paths["attempt_identifier"],
        launch_gate=paths["launch_gate"],
        geometry=geometry(size),
        video_decoder_receipt=probe.decoder_worker_receipt("video", geometry=geometry(size)),
        subprocess_runner=runner,
    )


class Slice3B3B2AFFmpegStagingTests(unittest.TestCase):
    def test_128_and_256_validated_media_reach_the_same_staging_path(self):
        for size in (128, 256):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=size)
                runner = FakeFFmpegRunner()
                receipt = run_staging(paths, runner=runner)
                self.assertEqual(receipt["status"], "staged")
                self.assertEqual(receipt["video_width"], size)
                self.assertEqual(receipt["video_height"], size)
                self.assertTrue(receipt["staged_mp4_nonzero"])
                self.assertTrue(Path(receipt["staged_mp4_path"]).is_file())
                self.assertFalse(paths["mp4"].exists())
                self.assertFalse(paths["mp4_manifest"].exists())
                self.assertEqual(len(runner.calls), 1)

    def test_cross_size_frame_manifest_fails_before_ffmpeg(self):
        for actual_size, selected_size in ((128, 256), (256, 128)):
            with self.subTest(actual_size=actual_size, selected_size=selected_size), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=actual_size)
                report_gate = probe.build_mux_launch_gate(
                    {
                        "attempt": {"attempt_identifier": paths["attempt_identifier"]},
                        "geometry": geometry(selected_size),
                        "latent_generation_status": "completed",
                        "video_status": "completed",
                        "audio_status": "completed",
                        "standalone_media_status": "completed",
                        "video_decoder": {
                            "release_gate_passed": True,
                            "allocator_cache_zero": True,
                            "worker_termination_confirmed": True,
                            "worker_receipt": probe.decoder_worker_receipt("video", geometry=geometry(selected_size)),
                        },
                        "audio_decoder": {
                            "release_gate_passed": True,
                            "allocator_cache_zero": True,
                            "worker_termination_confirmed": True,
                            "worker_receipt": probe.decoder_worker_receipt("audio"),
                        },
                    },
                    paths,
                    geometry=geometry(selected_size),
                )
                self.assertFalse(report_gate["passed"])
                runner = FakeFFmpegRunner()
                paths["launch_gate"] = report_gate
                with self.assertRaises(ValueError):
                    run_staging(paths, runner=runner)
                self.assertEqual(runner.calls, [])

    def test_ffmpeg_argv_is_deterministic_and_native_resolution_is_implicit(self):
        commands = []
        for size in (128, 256):
            command = probe.build_ffmpeg_command(
                Path(f"/synthetic/{size}/frames"),
                Path(f"/synthetic/{size}/audio.wav"),
                Path(f"/synthetic/{size}/dodecahedron.partial.mp4"),
            )
            commands.append(command)
            self.assertEqual(command[command.index("-framerate") + 1], "24")
            self.assertEqual(command[command.index("-frames:v") + 1], "30")
            self.assertEqual(command[command.index("-c:v") + 1], "libx264")
            self.assertEqual(command[command.index("-pix_fmt") + 1], "yuv420p")
            self.assertEqual(command[command.index("-c:a") + 1], "aac")
            self.assertEqual(command[command.index("-b:a") + 1], "192k")
            self.assertIn("-shortest", command)
            self.assertIn("frame_%05d.png", command[command.index("-i") + 1])
            self.assertNotIn("-vf", command)
            self.assertNotIn("scale", " ".join(command))
            self.assertEqual(
                command,
                probe.build_ffmpeg_command(
                    Path(f"/synthetic/{size}/frames"),
                    Path(f"/synthetic/{size}/audio.wav"),
                    Path(f"/synthetic/{size}/dodecahedron.partial.mp4"),
                ),
            )
        self.assertNotEqual(commands[0][-1], commands[1][-1])
        self.assertEqual(
            [item for item in commands[0] if item in {"-y", "-framerate", "24", "-start_number", "0", "-map", "0:v:0", "1:a:0", "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest"}],
            [item for item in commands[1] if item in {"-y", "-framerate", "24", "-start_number", "0", "-map", "0:v:0", "1:a:0", "-frames:v", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest"}],
        )

    def test_staging_receipt_binds_geometry_and_manifest_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            receipt = run_staging(paths, runner=FakeFFmpegRunner())
            validated = probe.validate_ffmpeg_staging_receipt(
                receipt,
                expected_geometry=geometry(256),
            )
            self.assertEqual(validated["frame_manifest_identity"], probe.VIDEO_FRAME_MANIFEST_IDENTITY)
            self.assertEqual(validated["wav_manifest_identity"], probe.AUDIO_WAV_MANIFEST_IDENTITY)
            self.assertEqual(validated["ffmpeg_exit_code"], 0)
            self.assertEqual(validated["invocation_counts"], {"ffmpeg": 1, "ffprobe": 0})
            self.assertFalse(validated["final_mp4_published"])

    def test_nonzero_exit_preserves_stderr_and_has_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeFFmpegRunner(returncode=7, stderr="encoder diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_staging(paths, runner=runner)
            receipt = context.exception.receipt
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(receipt["ffmpeg"]["stderr"], "encoder diagnostics")
            self.assertEqual(receipt["ffmpeg_exit_code"], 7)
            self.assertTrue(receipt["retry_suppressed"])
            self.assertFalse(paths["mp4_partial"].exists())

    def test_timeout_is_failure_without_retry_or_ffprobe(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeFFmpegRunner(timeout=True, stderr="timeout diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_staging(paths, runner=runner)
            receipt = context.exception.receipt
            self.assertEqual(len(runner.calls), 1)
            self.assertTrue(receipt["ffmpeg"]["timed_out"])
            self.assertEqual(receipt["ffprobe"]["invoked"], False)
            self.assertFalse(paths["mp4_partial"].exists())

    def test_zero_exit_missing_or_empty_partial_is_failure(self):
        for write_output, empty_output in ((False, False), (True, True)):
            with self.subTest(write_output=write_output, empty_output=empty_output), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=256)
                runner = FakeFFmpegRunner(write_output=write_output, empty_output=empty_output)
                with self.assertRaises(probe.MP4MuxFailure) as context:
                    run_staging(paths, runner=runner)
                self.assertEqual(len(runner.calls), 1)
                self.assertFalse(paths["mp4_partial"].exists())
                self.assertFalse(context.exception.receipt["staged_mp4_nonzero"])

    def test_existing_final_and_existing_staged_paths_are_refused(self):
        for existing_key in ("mp4", "mp4_partial"):
            with self.subTest(existing_key=existing_key), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=128)
                Path(paths[existing_key]).write_bytes(b"pre-existing")
                runner = FakeFFmpegRunner()
                with self.assertRaises(FileExistsError):
                    run_staging(paths, runner=runner)
                self.assertEqual(runner.calls, [])

    def test_256_execute_mp4_mux_dispatches_to_staging_without_ffprobe(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeFFmpegRunner()
            result = probe.execute_mp4_mux(
                frames_directory=paths["frames"],
                video_manifest_path=paths["video_frame_manifest"],
                wav_path=paths["audio_wav"],
                audio_manifest_path=paths["audio_manifest"],
                mp4_partial_path=paths["mp4_partial"],
                mp4_final_path=paths["mp4"],
                mp4_manifest_path=paths["mp4_manifest"],
                attempt_identifier=paths["attempt_identifier"],
                launch_gate=paths["launch_gate"],
                geometry=geometry(256),
                subprocess_runner=runner,
            )
            self.assertEqual(result["status"], "staged")
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg"])
            self.assertFalse(paths["mp4"].exists())
            self.assertFalse(paths["mp4_manifest"].exists())

    def test_128_legacy_mux_still_reaches_ffprobe_and_final_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=128)

            def runner(argv, **kwargs):
                if Path(argv[0]).name == "ffmpeg":
                    Path(argv[-1]).write_bytes(b"synthetic-mp4")
                    return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
                self.assertEqual(Path(argv[0]).name, "ffprobe")
                payload = {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "width": 128,
                            "height": 128,
                            "avg_frame_rate": "24/1",
                            "r_frame_rate": "24/1",
                            "pix_fmt": "yuv420p",
                            "nb_frames": "30",
                            "duration": "1.25",
                        },
                        {
                            "codec_type": "audio",
                            "codec_name": "aac",
                            "channels": 2,
                            "sample_rate": "32000",
                        },
                    ],
                    "format": {"duration": "1.25", "size": "12"},
                }
                return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(payload), stderr="")

            result = probe.execute_mp4_mux(
                frames_directory=paths["frames"],
                video_manifest_path=paths["video_frame_manifest"],
                wav_path=paths["audio_wav"],
                audio_manifest_path=paths["audio_manifest"],
                mp4_partial_path=paths["mp4_partial"],
                mp4_final_path=paths["mp4"],
                mp4_manifest_path=paths["mp4_manifest"],
                attempt_identifier=paths["attempt_identifier"],
                launch_gate=paths["launch_gate"],
                subprocess_runner=runner,
            )
            self.assertEqual(result["status"], "completed")
            self.assertTrue(paths["mp4"].is_file())
            self.assertFalse(paths["mp4_partial"].exists())

    def test_subprocess_seam_is_shell_free_and_parent_import_is_mlx_free(self):
        source = SCRIPT.read_text()
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        tree = ast.parse(source)
        top_level_mlx_imports = []
        for node in tree.body:
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            top_level_mlx_imports.extend(
                module for module in modules if module == "mlx" or module.startswith("mlx.")
            )
        self.assertEqual(top_level_mlx_imports, [])
        code = (
            "import importlib.util, sys; "
            f"spec=importlib.util.spec_from_file_location('probe_import_check', {str(SCRIPT)!r}); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_256_full_run_gate_remains_before_attempt_creation(self):
        args = probe.build_parser().parse_args(
            [
                "run-derived-full-schedule",
                "--checkpoint-root", "/nonexistent/checkpoint",
                "--derived-transformer", "/nonexistent/transformer",
                "--output-root", "/private/tmp/slice3b3b2a-no-output",
                "--prompt", probe.LOCKED_PROMPT,
                "--seed", str(probe.CANONICAL_SEED),
                "--video-size", "256",
                "--active-memory-tolerance-bytes", "0",
            ]
        )
        self.assertEqual(probe.validate_full_run_video_size(128), 128)
        self.assertEqual(probe.run_command(args), 1)


if __name__ == "__main__":
    unittest.main()

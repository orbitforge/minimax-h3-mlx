"""MLX-free Slice 3B3B2B1 geometry-aware ffprobe validation contracts."""

from __future__ import annotations

import ast
import copy
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
SPEC = importlib.util.spec_from_file_location("probe_v05e_slice3b3b2b1", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def geometry(size: int) -> dict[str, object]:
    return probe.canonical_geometry_contract(size)


def valid_ffprobe_json(size: int = 128) -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": size,
                "height": size,
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


def override_probe(
    size: int = 128,
    *,
    video: dict[str, object] | None = None,
    audio: dict[str, object] | None = None,
    format_info: dict[str, object] | None = None,
    streams: list[object] | None = None,
) -> dict[str, object]:
    payload = valid_ffprobe_json(size)
    if video:
        payload["streams"][0].update(video)
    if audio:
        payload["streams"][1].update(audio)
    if format_info:
        payload["format"].update(format_info)
    if streams is not None:
        payload["streams"] = streams
    return payload


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
            attempt_identifier="slice3b3b2b1-attempt",
            geometry=selected_geometry,
        ),
    )
    wav = root / "audio.wav"
    write_stereo_wav(wav)
    wav_manifest = root / "audio-manifest.json"
    probe._write_json(
        wav_manifest,
        probe.build_audio_wav_manifest(wav, attempt_identifier="slice3b3b2b1-attempt"),
    )
    paths = {
        "frames": frames,
        "video_frame_manifest": frame_manifest,
        "audio_wav": wav,
        "audio_manifest": wav_manifest,
        "mp4_partial": root / probe.MP4_PARTIAL_FILENAME,
        "mp4": root / probe.MP4_FINAL_FILENAME,
        "mp4_manifest": root / probe.MP4_MANIFEST_FILENAME,
        "attempt_identifier": "slice3b3b2b1-attempt",
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


class FakeMuxRunner:
    def __init__(
        self,
        *,
        ffprobe_json: object | None = None,
        ffprobe_stdout: object | None = None,
        ffmpeg_returncode: int = 0,
        ffprobe_returncode: int = 0,
        ffmpeg_stderr: str = "",
        ffprobe_stderr: str = "",
        timeout_tool: str | None = None,
        write_output: bool = True,
        empty_output: bool = False,
    ) -> None:
        self.ffprobe_json = ffprobe_json if ffprobe_json is not None else valid_ffprobe_json()
        self.ffprobe_stdout = ffprobe_stdout
        self.ffmpeg_returncode = ffmpeg_returncode
        self.ffprobe_returncode = ffprobe_returncode
        self.ffmpeg_stderr = ffmpeg_stderr
        self.ffprobe_stderr = ffprobe_stderr
        self.timeout_tool = timeout_tool
        self.write_output = write_output
        self.empty_output = empty_output
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        tool = Path(argv[0]).name
        if tool == "ffmpeg":
            if self.timeout_tool == tool:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr=self.ffmpeg_stderr or "ffmpeg timeout")
            output = Path(argv[-1])
            if self.write_output:
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"" if self.empty_output else b"synthetic-partial-mp4")
            return subprocess.CompletedProcess(
                argv,
                self.ffmpeg_returncode,
                stdout="ffmpeg stdout",
                stderr=self.ffmpeg_stderr,
            )
        if tool == "ffprobe":
            if self.timeout_tool == tool:
                raise subprocess.TimeoutExpired(argv, kwargs["timeout"], stderr=self.ffprobe_stderr or "ffprobe timeout")
            stdout = self.ffprobe_stdout
            if stdout is None:
                stdout = json.dumps(self.ffprobe_json)
            return subprocess.CompletedProcess(
                argv,
                self.ffprobe_returncode,
                stdout=stdout,
                stderr=self.ffprobe_stderr,
            )
        raise AssertionError(f"unexpected subprocess: {tool}")


def run_mux(paths: dict[str, object], *, size: int, runner: FakeMuxRunner) -> dict[str, object]:
    return probe.execute_mp4_mux(
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
        subprocess_runner=runner,
    )


class Slice3B3B2B1FFprobeValidationTests(unittest.TestCase):
    def test_valid_128_and_256_probe_json_pass_under_selected_geometry(self):
        for size in (128, 256):
            with self.subTest(size=size):
                result = probe.validate_ffprobe_json(
                    valid_ffprobe_json(size),
                    expected_geometry=geometry(size),
                )
                self.assertTrue(result["passed"])
                self.assertEqual(result["video"]["width"], size)
                self.assertEqual(result["video"]["height"], size)
                self.assertEqual(result["video"]["frame_count"], 30)

    def test_cross_size_probe_dimensions_fail_closed(self):
        for actual_size, selected_size in ((128, 256), (256, 128)):
            with self.subTest(actual_size=actual_size, selected_size=selected_size):
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    probe.validate_ffprobe_json(
                        valid_ffprobe_json(actual_size),
                        expected_geometry=geometry(selected_size),
                    )

    def test_wrong_video_contract_fields_fail_closed(self):
        cases = [
            ({"codec_name": "vp9"}, "codec"),
            ({"pix_fmt": "rgb24"}, "pixel format"),
            ({"avg_frame_rate": "25/1", "r_frame_rate": "25/1"}, "frame rate"),
            ({"nb_frames": "29"}, "frame count"),
            ({"width": 256, "height": 128}, "dimensions"),
        ]
        for overrides, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    probe.validate_ffprobe_json(
                        override_probe(256, video=overrides),
                        expected_geometry=geometry(256),
                    )

    def test_missing_video_or_audio_stream_fails_closed(self):
        base = valid_ffprobe_json(256)
        cases = [
            (base["streams"][1:], "video stream"),
            (base["streams"][:1], "audio stream"),
        ]
        for streams, label in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, "stream"):
                    probe.validate_ffprobe_json(
                        {"streams": streams, "format": base["format"]},
                        expected_geometry=geometry(256),
                    )

    def test_wrong_audio_contract_fields_fail_closed(self):
        cases = [
            ({"codec_name": "mp3"}, "codec"),
            ({"channels": 1}, "channels"),
            ({"sample_rate": "44100"}, "sample rate"),
        ]
        for overrides, label in cases:
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    probe.validate_ffprobe_json(
                        override_probe(256, audio=overrides),
                        expected_geometry=geometry(256),
                    )

    def test_invalid_duration_and_contradictory_stream_container_duration_fail(self):
        outside = probe.MP4_EXPECTED_DURATION_SECONDS + probe.MP4_DURATION_TOLERANCE_SECONDS + 0.001
        cases = [
            override_probe(256, format_info={"duration": str(outside)}),
            override_probe(256, video={"duration": "1.10"}),
            override_probe(256, video={"duration": "1.25"}, format_info={"duration": "1.10"}),
        ]
        for payload in cases:
            with self.assertRaises(ValueError):
                probe.validate_ffprobe_json(payload, expected_geometry=geometry(256))

    def test_malformed_json_fails_after_one_ffprobe_call_and_no_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(ffprobe_stdout="{not-json", ffprobe_stderr="malformed diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_mux(paths, size=256, runner=runner)
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg", "ffprobe"])
            self.assertEqual(context.exception.receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})
            self.assertTrue(context.exception.receipt["retry_suppressed"])
            self.assertEqual(context.exception.receipt["ffprobe"]["stderr"], "malformed diagnostics")
            self.assertFalse(paths["mp4"].exists())
            self.assertFalse(paths["mp4_manifest"].exists())

    def test_nonzero_ffprobe_exit_retains_stderr_and_does_not_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(ffprobe_returncode=9, ffprobe_stderr="ffprobe diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_mux(paths, size=256, runner=runner)
            self.assertEqual(len(runner.calls), 2)
            self.assertEqual(context.exception.receipt["ffprobe_exit_code"], 9)
            self.assertEqual(context.exception.receipt["ffprobe"]["stderr"], "ffprobe diagnostics")
            self.assertTrue(context.exception.receipt["retry_suppressed"])

    def test_ffprobe_timeout_is_one_shot_and_retains_timeout_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(timeout_tool="ffprobe", ffprobe_stderr="timeout diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_mux(paths, size=256, runner=runner)
            self.assertEqual(len(runner.calls), 2)
            self.assertTrue(context.exception.receipt["ffprobe"]["timed_out"])
            self.assertEqual(context.exception.receipt["ffprobe"]["stderr"], "timeout diagnostics")
            self.assertEqual(context.exception.receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})

    def test_ffprobe_cannot_run_after_failed_ffmpeg_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(ffmpeg_returncode=7, ffmpeg_stderr="ffmpeg diagnostics")
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_mux(paths, size=256, runner=runner)
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg"])
            self.assertEqual(context.exception.receipt["invocation_counts"], {"ffmpeg": 1, "ffprobe": 0})
            self.assertFalse(context.exception.receipt["ffprobe"]["invoked"])
            self.assertEqual(context.exception.receipt["ffmpeg"]["stderr"], "ffmpeg diagnostics")

    def test_valid_256_receipt_binds_geometry_staging_and_all_media_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(ffprobe_json=valid_ffprobe_json(256))
            result = run_mux(paths, size=256, runner=runner)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["geometry"], geometry(256))
            self.assertEqual(result["video_width"], 256)
            self.assertEqual(result["video_height"], 256)
            self.assertEqual(result["video_codec"], "h264")
            self.assertEqual(result["pixel_format"], "yuv420p")
            self.assertEqual(result["fps"], 24)
            self.assertEqual(result["frame_count"], 30)
            self.assertEqual(result["audio_codec"], "aac")
            self.assertEqual(result["audio_channels"], 2)
            self.assertEqual(result["audio_sample_rate"], 32000)
            self.assertEqual(result["duration"], 1.25)
            self.assertEqual(result["ffprobe_exit_code"], 0)
            self.assertEqual(result["ffprobe_argv"], runner.calls[1][0])
            self.assertEqual(
                result["frame_manifest_identity"],
                probe.VIDEO_FRAME_MANIFEST_IDENTITY,
            )
            self.assertEqual(result["wav_manifest_identity"], probe.AUDIO_WAV_MANIFEST_IDENTITY)
            self.assertEqual(result["ffmpeg_staging_receipt"]["status"], "staged")
            self.assertEqual(result["invocation_counts"], {"ffmpeg": 1, "ffprobe": 1})
            self.assertFalse(paths["mp4_partial"].exists())
            self.assertTrue(paths["mp4"].is_file())
            self.assertTrue(paths["mp4_manifest"].is_file())

    def test_receipt_cross_size_and_field_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            staging_runner = FakeMuxRunner()
            staging = probe.execute_ffmpeg_staging(
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
                subprocess_runner=staging_runner,
            )
            probe_runner = FakeMuxRunner(ffprobe_json=valid_ffprobe_json(256))
            with self.assertRaises(ValueError):
                probe.execute_ffprobe_validation(
                    staging,
                    geometry=geometry(128),
                    subprocess_runner=probe_runner,
                )
            self.assertEqual(probe_runner.calls, [])

            result = probe.execute_ffprobe_validation(
                staging,
                geometry=geometry(256),
                subprocess_runner=probe_runner,
            )
            tampered = copy.deepcopy(result)
            tampered["video_width"] = 128
            with self.assertRaises(ValueError):
                probe.validate_ffprobe_staging_receipt(tampered, expected_geometry=geometry(256))

    def test_staged_mp4_identity_is_bound_before_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            staging = probe.execute_ffmpeg_staging(
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
                subprocess_runner=FakeMuxRunner(),
            )
            result = probe.execute_ffprobe_validation(
                staging,
                geometry=geometry(256),
                subprocess_runner=FakeMuxRunner(ffprobe_json=valid_ffprobe_json(256)),
            )
            identity = result["staged_mp4_identity"]
            self.assertEqual(identity["path"], str(paths["mp4_partial"].resolve()))
            self.assertEqual(identity["size_bytes"], paths["mp4_partial"].stat().st_size)
            self.assertEqual(identity["sha256"], probe.sha256_file(paths["mp4_partial"]))
            self.assertFalse(result["final_mp4_published"])
            self.assertFalse(result["mp4_manifest_created"])
            self.assertFalse(paths["mp4"].exists())
            self.assertFalse(paths["mp4_manifest"].exists())
            self.assertTrue(paths["mp4_partial"].exists())

    def test_128_legacy_path_remains_publishable_with_same_probe_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=128)
            result = run_mux(paths, size=128, runner=FakeMuxRunner(ffprobe_json=valid_ffprobe_json(128)))
            self.assertEqual(result["status"], "completed")
            self.assertTrue(paths["mp4"].is_file())
            self.assertFalse(paths["mp4_partial"].exists())
            self.assertTrue(paths["mp4_manifest"].is_file())

    def test_256_full_run_selector_passes_size_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            args = probe.build_parser().parse_args(
                [
                    "run-derived-full-schedule",
                    "--checkpoint-root", "/nonexistent/checkpoint",
                    "--derived-transformer", "/nonexistent/transformer",
                    "--output-root", str(Path(directory) / "output"),
                    "--prompt", probe.LOCKED_PROMPT,
                    "--seed", str(probe.CANONICAL_SEED),
                    "--video-size", "256",
                    "--active-memory-tolerance-bytes", "0",
                ]
            )
            self.assertEqual(probe.validate_full_run_video_size(256), 256)
            self.assertEqual(probe.validate_full_run_video_size(128), 128)
            self.assertEqual(probe.run_command(args), 1)

    def test_ffprobe_command_is_deterministic_and_shell_free(self):
        command = probe.build_ffprobe_command(Path("/tmp/staged.mp4"))
        self.assertEqual(
            command,
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                "-count_frames",
                str(Path("/tmp/staged.mp4").resolve()),
            ],
        )
        source = SCRIPT.read_text()
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)

    def test_parent_import_is_mlx_free(self):
        tree = ast.parse(SCRIPT.read_text())
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


if __name__ == "__main__":
    unittest.main()

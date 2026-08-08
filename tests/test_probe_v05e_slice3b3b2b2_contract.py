"""MLX-free Slice 3B3B2B2 final MP4 publication contracts."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "probe_v05d_derived_full_schedule.py"
SPEC = importlib.util.spec_from_file_location("probe_v05e_slice3b3b2b2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def geometry(size: int) -> dict[str, object]:
    return probe.canonical_geometry_contract(size)


def valid_ffprobe_json(size: int) -> dict[str, object]:
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


def write_rgb_frames(directory: Path, size: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(probe.VIDEO_FRAME_COUNT):
        Image.new("RGB", (size, size), (index % 256, 23, 41)).save(
            directory / f"frame_{index:05d}.png"
        )


def write_stereo_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(probe.AUDIO_SAMPLE_WIDTH_BYTES)
        handle.setframerate(probe.AUDIO_SAMPLE_RATE)
        handle.writeframes(b"\0" * (2 * probe.AUDIO_SAMPLE_COUNT * probe.AUDIO_SAMPLE_WIDTH_BYTES))


class FakeMuxRunner:
    def __init__(
        self,
        size: int,
        *,
        write_output: bool = True,
        empty_output: bool = False,
        ffprobe_size: int | None = None,
        mutate_after_ffprobe: bool = False,
    ) -> None:
        self.size = size
        self.write_output = write_output
        self.empty_output = empty_output
        self.ffprobe_size = ffprobe_size or size
        self.mutate_after_ffprobe = mutate_after_ffprobe
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.final_exists_during_probe: list[bool] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append((argv, dict(kwargs)))
        tool = Path(argv[0]).name
        if tool == "ffmpeg":
            output = Path(argv[-1])
            if self.write_output:
                output.write_bytes(b"" if self.empty_output else b"synthetic-partial-mp4")
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        if tool == "ffprobe":
            self.final_exists_during_probe.append(Path(argv[-1]).with_name(probe.MP4_FINAL_FILENAME).exists())
            if self.mutate_after_ffprobe:
                Path(argv[-1]).write_bytes(b"changed-after-ffprobe")
            return subprocess.CompletedProcess(
                argv,
                0,
                stdout=json.dumps(valid_ffprobe_json(self.ffprobe_size)),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess: {tool}")


def media_fixture(root: Path, *, size: int) -> dict[str, object]:
    selected_geometry = geometry(size)
    frames = root / "frames"
    write_rgb_frames(frames, size)
    video_manifest = root / "video-frame-manifest.json"
    probe._write_json(
        video_manifest,
        probe.build_video_frame_manifest(
            frames,
            attempt_identifier="slice3b3b2b2-attempt",
            geometry=selected_geometry,
        ),
    )
    wav = root / "audio.wav"
    write_stereo_wav(wav)
    audio_manifest = root / "audio-manifest.json"
    probe._write_json(
        audio_manifest,
        probe.build_audio_wav_manifest(wav, attempt_identifier="slice3b3b2b2-attempt"),
    )
    paths: dict[str, object] = {
        "frames": frames,
        "video_frame_manifest": video_manifest,
        "audio_wav": wav,
        "audio_manifest": audio_manifest,
        "mp4_partial": root / probe.MP4_PARTIAL_FILENAME,
        "mp4": root / probe.MP4_FINAL_FILENAME,
        "mp4_manifest": root / probe.MP4_MANIFEST_FILENAME,
        "attempt_identifier": "slice3b3b2b2-attempt",
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


def run_mux(paths: dict[str, object], *, size: int, runner: FakeMuxRunner, rename=None) -> dict[str, object]:
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
        rename=rename,
    )


class Slice3B3B2B2PublicationTests(unittest.TestCase):
    def test_valid_128_and_256_evidence_produce_one_geometry_bound_manifest(self):
        for size in (128, 256):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=size)
                result = run_mux(paths, size=size, runner=FakeMuxRunner(size))
                manifest = probe._read_json_object(paths["mp4_manifest"], "MP4 manifest")
                self.assertEqual(set(manifest), probe.MP4_MANIFEST_KEYS)
                self.assertEqual(manifest["geometry"], geometry(size))
                self.assertEqual(manifest["video_width"], size)
                self.assertEqual(manifest["video_height"], size)
                self.assertEqual(manifest["frame_count"], 30)
                self.assertEqual(manifest["fps"], 24)
                self.assertEqual(manifest["duration"], 1.25)
                self.assertEqual(manifest["video_codec"], "h264")
                self.assertEqual(manifest["pixel_format"], "yuv420p")
                self.assertEqual(manifest["audio_codec"], "aac")
                self.assertEqual(manifest["audio_channels"], 2)
                self.assertEqual(manifest["audio_sample_rate"], 32000)
                self.assertEqual(result["mp4_artifact"]["geometry"], geometry(size))

    def test_staged_sha_and_size_are_flat_bound_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            result = run_mux(paths, size=256, runner=FakeMuxRunner(256))
            manifest = probe._read_json_object(paths["mp4_manifest"], "MP4 manifest")
            identity = result["staged_mp4_identity"]
            self.assertEqual(manifest["staged_mp4_path"], identity["path"])
            self.assertEqual(manifest["staged_mp4_size_bytes"], identity["size_bytes"])
            self.assertEqual(manifest["staged_mp4_sha256"], identity["sha256"])
            self.assertEqual(result["final_mp4_sha256"], identity["sha256"])
            self.assertEqual(result["final_mp4_size_bytes"], identity["size_bytes"])

    def test_ffprobe_is_bound_to_the_same_staged_path_and_final_is_absent_during_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(256)
            result = run_mux(paths, size=256, runner=runner)
            self.assertEqual(runner.final_exists_during_probe, [False])
            self.assertEqual(Path(result["ffprobe_argv"][-1]), paths["mp4_partial"].resolve())
            self.assertEqual(Path(result["ffmpeg_argv"][-1]), paths["mp4_partial"].resolve())

    def test_changed_staged_file_after_ffprobe_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            runner = FakeMuxRunner(256, mutate_after_ffprobe=True)
            with self.assertRaises(probe.MP4MuxFailure) as context:
                run_mux(paths, size=256, runner=runner)
            self.assertEqual([Path(call[0][0]).name for call in runner.calls], ["ffmpeg", "ffprobe"])
            self.assertIn("identity", str(context.exception))
            self.assertFalse(paths["mp4"].exists())
            self.assertFalse(paths["mp4_manifest"].exists())
            self.assertFalse(paths["mp4_partial"].exists())

    def test_cross_size_ffprobe_and_manifest_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            with self.assertRaises(probe.MP4MuxFailure):
                run_mux(paths, size=256, runner=FakeMuxRunner(256, ffprobe_size=128))
            self.assertFalse(paths["mp4"].exists())

        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            run_mux(paths, size=256, runner=FakeMuxRunner(256))
            manifest = probe._read_json_object(paths["mp4_manifest"], "MP4 manifest")
            manifest["video_width"] = 128
            manifest["manifest_sha256"] = probe.stable_mp4_manifest_sha256(manifest)
            probe._write_json(paths["mp4_manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "geometry"):
                probe.validate_mp4_manifest(
                    paths["mp4_manifest"],
                    paths["mp4"],
                    expected_attempt_identifier=paths["attempt_identifier"],
                    expected_published_path=paths["mp4"],
                    expected_video_manifest_path=paths["video_frame_manifest"],
                    expected_audio_manifest_path=paths["audio_manifest"],
                    expected_geometry=geometry(256),
                )

    def test_wrong_staged_sha_fails_manifest_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            result = run_mux(paths, size=256, runner=FakeMuxRunner(256))
            manifest = probe._read_json_object(paths["mp4_manifest"], "MP4 manifest")
            manifest["staged_mp4_sha256"] = "0" * 64
            manifest["manifest_sha256"] = probe.stable_mp4_manifest_sha256(manifest)
            probe._write_json(paths["mp4_manifest"], manifest)
            with self.assertRaisesRegex(ValueError, "staged MP4 identity"):
                probe.validate_mp4_manifest(
                    paths["mp4_manifest"],
                    paths["mp4"],
                    expected_attempt_identifier=paths["attempt_identifier"],
                    expected_published_path=paths["mp4"],
                    expected_video_manifest_path=paths["video_frame_manifest"],
                    expected_audio_manifest_path=paths["audio_manifest"],
                    expected_geometry=geometry(256),
                    expected_staged_mp4_identity=result["staged_mp4_identity"],
                )

    def test_existing_final_blocks_publication_and_missing_or_empty_partial_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            paths["mp4"].write_bytes(b"keep-existing")
            runner = FakeMuxRunner(256)
            with self.assertRaises(FileExistsError):
                run_mux(paths, size=256, runner=runner)
            self.assertEqual(runner.calls, [])
            self.assertEqual(paths["mp4"].read_bytes(), b"keep-existing")

        for write_output, empty_output in ((False, False), (True, True)):
            with self.subTest(write_output=write_output, empty_output=empty_output), tempfile.TemporaryDirectory() as directory:
                paths = media_fixture(Path(directory), size=256)
                with self.assertRaises(probe.MP4MuxFailure):
                    run_mux(paths, size=256, runner=FakeMuxRunner(256, write_output=write_output, empty_output=empty_output))
                self.assertFalse(paths["mp4_partial"].exists())
                self.assertFalse(paths["mp4"].exists())

    def test_successful_publication_is_atomic_and_consumes_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            observed: list[tuple[str, bool]] = []

            def rename(source, destination):
                observed.append((Path(source).name, not Path(destination).exists()))
                source.rename(destination)

            result = run_mux(paths, size=256, runner=FakeMuxRunner(256), rename=rename)
            self.assertEqual(observed, [(probe.MP4_PARTIAL_FILENAME, True)])
            self.assertFalse(paths["mp4_partial"].exists())
            self.assertTrue(paths["mp4"].is_file())
            self.assertEqual(probe.sha256_file(paths["mp4"]), result["staged_mp4_sha256"])
            self.assertEqual(paths["mp4"].stat().st_size, result["staged_mp4_size_bytes"])

    def test_failed_publication_is_not_functional_success(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            report = {
                "attempt": {"attempt_identifier": paths["attempt_identifier"]},
                "geometry": geometry(256),
                "latent_generation_status": "completed",
                "video_status": "completed",
                "audio_status": "completed",
                "standalone_media_status": "completed",
                "video_decoder": {
                    "release_gate_passed": True,
                    "allocator_cache_zero": True,
                    "worker_termination_confirmed": True,
                    "worker_receipt": probe.decoder_worker_receipt("video", geometry=geometry(256)),
                },
                "audio_decoder": {
                    "release_gate_passed": True,
                    "allocator_cache_zero": True,
                    "worker_termination_confirmed": True,
                    "worker_receipt": probe.decoder_worker_receipt("audio"),
                },
                "video_artifacts": {},
                "audio_artifacts": {},
                "standalone_media": {"status": "completed"},
                "phase_order": [],
                "invocation": {},
            }

            def fail_rename(_source, _destination):
                raise OSError("synthetic publication failure")

            result = probe.apply_mp4_mux_report(
                report,
                paths,
                geometry=geometry(256),
                subprocess_runner=FakeMuxRunner(256),
                rename=fail_rename,
            )
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["functional_success"])
            self.assertEqual(result["mp4_mux_status"], "failed")
            self.assertFalse(paths["mp4"].exists())

    def test_final_report_binds_geometry_identity_and_exact_invocation_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = media_fixture(Path(directory), size=256)
            report = {
                "attempt": {"attempt_identifier": paths["attempt_identifier"]},
                "geometry": geometry(256),
                "latent_generation_status": "completed",
                "video_status": "completed",
                "audio_status": "completed",
                "standalone_media_status": "completed",
                "video_decoder": {
                    "release_gate_passed": True,
                    "allocator_cache_zero": True,
                    "worker_termination_confirmed": True,
                    "worker_receipt": probe.decoder_worker_receipt("video", geometry=geometry(256)),
                },
                "audio_decoder": {
                    "release_gate_passed": True,
                    "allocator_cache_zero": True,
                    "worker_termination_confirmed": True,
                    "worker_receipt": probe.decoder_worker_receipt("audio"),
                },
                "video_artifacts": {},
                "audio_artifacts": {},
                "standalone_media": {"status": "completed"},
                "phase_order": [],
                "invocation": {},
            }
            result = probe.apply_mp4_mux_report(
                report,
                paths,
                geometry=geometry(256),
                subprocess_runner=FakeMuxRunner(256),
            )
            mux = result["mp4_mux"]
            self.assertEqual(mux["video_width"], 256)
            self.assertEqual(mux["video_height"], 256)
            self.assertEqual(mux["frame_count"], 30)
            self.assertEqual(mux["fps"], 24)
            self.assertEqual(mux["duration"], 1.25)
            self.assertEqual(mux["final_mp4_path"], str(paths["mp4"].resolve()))
            self.assertEqual(mux["ffmpeg_invocations"], 1)
            self.assertEqual(mux["ffprobe_invocations"], 1)
            self.assertEqual(mux["retry_count"], 0)
            self.assertTrue(mux["functional_success"])

    def test_gate_removal_preserves_default_and_rejects_unsupported_sizes(self):
        self.assertEqual(probe.validate_full_run_video_size(256), 256)
        self.assertEqual(probe.validate_full_run_video_size(128), 128)
        with self.assertRaisesRegex(ValueError, "one of 128 or 256"):
            probe.validate_full_run_video_size(192)
        parsed = probe.build_parser().parse_args(
            [
                "run-derived-full-schedule",
                "--checkpoint-root", "checkpoint",
                "--derived-transformer", "derived",
                "--output-root", "output",
                "--prompt", probe.LOCKED_PROMPT,
                "--seed", "0",
                "--active-memory-tolerance-bytes", "0",
            ]
        )
        self.assertEqual(parsed.video_size, 128)

    def test_256_size_preflight_does_not_launch_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            args = probe.build_parser().parse_args(
                [
                    "run-derived-full-schedule",
                    "--checkpoint-root", "/nonexistent/checkpoint",
                    "--derived-transformer", "/nonexistent/transformer",
                    "--output-root", str(Path(directory) / "output"),
                    "--prompt", probe.LOCKED_PROMPT,
                    "--seed", "0",
                    "--video-size", "256",
                    "--active-memory-tolerance-bytes", "0",
                ]
            )
            with mock.patch.object(probe, "_run_child", side_effect=AssertionError("worker launched")) as run_child:
                with contextlib.redirect_stdout(io.StringIO()):
                    result = probe.run_command(args)
            self.assertEqual(result, 1)
            run_child.assert_not_called()

    def test_attribution_schema_and_parent_import_remain_mlxfree(self):
        self.assertEqual(probe.ATTRIBUTION_SCHEMA_VERSION, 1)
        self.assertEqual(len(probe.ATTRIBUTION_COMPONENT_FIELDS), 5)
        source = SCRIPT.read_text()
        tree = ast.parse(source)
        top_level_mlx = []
        for node in tree.body:
            modules = []
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
            top_level_mlx.extend(module for module in modules if module == "mlx" or module.startswith("mlx."))
        self.assertEqual(top_level_mlx, [])
        code = (
            "import importlib.util, sys; "
            f"spec=importlib.util.spec_from_file_location('probe_import_check', {str(SCRIPT)!r}); "
            "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
            "assert 'mlx' not in sys.modules and 'mlx.core' not in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()

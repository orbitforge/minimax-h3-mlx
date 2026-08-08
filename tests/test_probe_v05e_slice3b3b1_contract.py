"""MLX-free Slice 3B3B1 frame staging, manifest, and publication contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "probe_v05d_derived_full_schedule",
    ROOT / "scripts" / "probe_v05d_derived_full_schedule.py",
)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def geometry(size: int) -> dict[str, object]:
    return probe.canonical_geometry_contract(size)


def write_rgb_frame_set(directory: Path, *, size: int = 128, count: int = 30) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        Image.new("RGB", (size, size), (index % 256, 17, 29)).save(
            directory / f"frame_{index:05d}.png"
        )


def valid_receipt(size: int) -> dict[str, object]:
    return probe.decoder_worker_receipt("video", geometry=geometry(size))


class Slice3B3B1FramePublicationTests(unittest.TestCase):
    def test_128_frame_set_stages_and_publishes_with_legacy_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            frames = np.zeros((30, 128, 128, 3), dtype=np.uint8)
            staged_receipt = probe.stage_video_frames(staged, frames, geometry=geometry(128))
            self.assertEqual(staged_receipt["frame_count"], 30)
            published = probe.publish_video_frames_atomically(
                staged,
                final,
                manifest,
                attempt_identifier="slice3b3b1-128",
                geometry=geometry(128),
                decoder_receipt=valid_receipt(128),
                decoded_rgb_geometry={"shape": [30, 128, 128, 3], "dtype": "uint8"},
            )
            self.assertEqual(published["width"], 128)
            self.assertEqual(published["height"], 128)
            self.assertEqual(published["frame_count"], 30)
            self.assertTrue(final.is_dir())
            self.assertFalse(staged.exists())

    def test_256_frame_set_stages_exactly_30_rgb_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "frames.partial"
            result = probe.stage_video_frames(
                staged,
                np.zeros((30, 256, 256, 3), dtype=np.uint8),
                geometry=geometry(256),
            )
            self.assertEqual(result["frame_count"], 30)
            self.assertEqual(result["width"], 256)
            self.assertEqual(result["height"], 256)
            self.assertEqual(
                sorted(path.name for path in staged.iterdir()),
                [f"frame_{index:05d}.png" for index in range(30)],
            )
            for path in sorted(staged.iterdir()):
                with Image.open(path) as image:
                    self.assertEqual(image.mode, "RGB")
                    self.assertEqual(image.size, (256, 256))

    def test_256_manifest_binds_geometry_filenames_and_sha256_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            manifest_path = root / "video-frame-manifest.json"
            write_rgb_frame_set(staged, size=256)
            result = probe.publish_video_frames_atomically(
                staged,
                final,
                manifest_path,
                attempt_identifier="slice3b3b1-256",
                geometry=geometry(256),
                decoder_receipt=valid_receipt(256),
                decoded_rgb_geometry={"shape": [30, 256, 256, 3], "dtype": "uint8"},
            )
            self.assertEqual(result["video_width"], 256)
            self.assertEqual(result["video_height"], 256)
            self.assertEqual(result["frame_filename_pattern"], probe.VIDEO_FRAME_FILENAME_PATTERN)
            self.assertEqual(len(result["per_frame_sha256"]), 30)
            self.assertEqual(
                result["per_frame_sha256"]["frame_00000.png"],
                probe.sha256_file(final / "frame_00000.png"),
            )
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["geometry"]["video_width"], 256)
            self.assertEqual(manifest["geometry_identity"]["video_rgb_shape"], [30, 256, 256, 3])
            self.assertEqual(manifest["attempt_identifier"], "slice3b3b1-256")
            probe.validate_video_frame_manifest(
                manifest_path,
                final,
                expected_attempt_identifier="slice3b3b1-256",
            )

    def test_cross_size_frame_sets_fail_closed(self):
        for actual_size, selected_size in ((128, 256), (256, 128)):
            with self.subTest(actual_size=actual_size, selected_size=selected_size), tempfile.TemporaryDirectory() as directory:
                staged = Path(directory) / "frames.partial"
                write_rgb_frame_set(staged, size=actual_size)
                with self.assertRaisesRegex(ValueError, "dimensions"):
                    probe._inspect_video_frame_set(staged, geometry=geometry(selected_size))

    def test_missing_and_unexpected_frames_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "frames.partial"
            write_rgb_frame_set(staged, size=256)
            (staged / "frame_00007.png").unlink()
            with self.assertRaisesRegex(ValueError, "contiguous"):
                probe._inspect_video_frame_set(staged, geometry=geometry(256))

    def test_manifest_duplicate_or_index_inconsistency_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            write_rgb_frame_set(staged, size=256)
            manifest = probe.build_video_frame_manifest(
                staged,
                attempt_identifier="slice3b3b1-index",
                geometry=geometry(256),
            )
            manifest["frames"][1] = dict(manifest["frames"][0])
            manifest["manifest_sha256"] = probe.stable_video_frame_manifest_sha256(manifest)
            manifest_path = root / "manifest.json"
            probe._write_json(manifest_path, manifest)
            with self.assertRaisesRegex(ValueError, "checksum linkage"):
                probe.validate_video_frame_manifest(
                    manifest_path,
                    staged,
                    expected_attempt_identifier="slice3b3b1-index",
                    geometry=geometry(256),
                )
            write_rgb_frame_set(staged, size=256)
            (staged / "frame_00030.png").write_bytes((staged / "frame_00000.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "contiguous"):
                probe._inspect_video_frame_set(staged, geometry=geometry(256))

    def test_zero_byte_invalid_png_and_non_rgb_frames_fail_closed(self):
        cases = ("empty", "signature", "mode")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                staged = Path(directory) / "frames.partial"
                write_rgb_frame_set(staged, size=256)
                target = staged / "frame_00000.png"
                if case == "empty":
                    target.write_bytes(b"")
                elif case == "signature":
                    target.write_bytes(b"not-a-png")
                else:
                    Image.new("L", (256, 256), 0).save(target)
                with self.assertRaises(ValueError):
                    probe._inspect_video_frame_set(staged, geometry=geometry(256))

    def test_manifest_geometry_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            write_rgb_frame_set(staged, size=256)
            probe.publish_video_frames_atomically(
                staged,
                final,
                manifest,
                attempt_identifier="slice3b3b1-mismatch",
                geometry=geometry(256),
            )
            with self.assertRaisesRegex(ValueError, "geometry"):
                probe.validate_video_frame_manifest(
                    manifest,
                    final,
                    expected_attempt_identifier="slice3b3b1-mismatch",
                    geometry=geometry(128),
                )

    def test_decoder_receipt_and_decoded_rgb_mismatch_block_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            manifest = root / "video-frame-manifest.json"
            write_rgb_frame_set(staged, size=256)
            with self.assertRaisesRegex(ValueError, "selected core contract|selected geometry"):
                probe.publish_video_frames_atomically(
                    staged,
                    final,
                    manifest,
                    attempt_identifier="slice3b3b1-receipt-mismatch",
                    geometry=geometry(256),
                    decoder_receipt=valid_receipt(128),
                    decoded_rgb_geometry={"shape": [30, 256, 256, 3], "dtype": "uint8"},
                )
            self.assertFalse(final.exists())

            with self.assertRaisesRegex(ValueError, "decoded RGB geometry"):
                probe.publish_video_frames_atomically(
                    staged,
                    final,
                    root / "second-manifest.json",
                    attempt_identifier="slice3b3b1-rgb-mismatch",
                    geometry=geometry(256),
                    decoded_rgb_geometry={"shape": [30, 128, 128, 3], "dtype": "uint8"},
                )
            self.assertFalse(final.exists())

    def test_atomic_publication_validates_before_rename_and_preserves_existing_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            manifest = root / "manifest.json"
            write_rgb_frame_set(staged, size=256)
            observed: list[bool] = []

            def rename(source: Path, destination: Path) -> None:
                observed.append(not destination.exists())
                source.rename(destination)

            result = probe.publish_video_frames_atomically(
                staged,
                final,
                manifest,
                attempt_identifier="slice3b3b1-atomic",
                geometry=geometry(256),
                rename=rename,
            )
            self.assertEqual(observed, [True])
            self.assertEqual(result["publication_state"], "published")

            existing_frame = final / "frame_00000.png"
            existing_bytes = existing_frame.read_bytes()
            second_staged = root / "second.partial"
            write_rgb_frame_set(second_staged, size=256)
            with self.assertRaises(FileExistsError):
                probe.publish_video_frames_atomically(
                    second_staged,
                    final,
                    root / "second.json",
                    attempt_identifier="slice3b3b1-existing",
                    geometry=geometry(256),
                )
            self.assertEqual(existing_frame.read_bytes(), existing_bytes)

    def test_publication_failure_does_not_expose_final_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staged = root / "frames.partial"
            final = root / "frames"
            write_rgb_frame_set(staged, size=256)

            def fail_rename(_source: Path, _destination: Path) -> None:
                raise OSError("synthetic rename failure")

            with self.assertRaisesRegex(OSError, "synthetic rename"):
                probe.publish_video_frames_atomically(
                    staged,
                    final,
                    root / "manifest.json",
                    attempt_identifier="slice3b3b1-failure",
                    geometry=geometry(256),
                    rename=fail_rename,
                )
            self.assertTrue(staged.is_dir())
            self.assertFalse(final.exists())

    def test_256_full_run_selector_is_valid_after_publication_contract(self):
        self.assertEqual(probe.validate_full_run_video_size(256), 256)
        self.assertEqual(probe.validate_full_run_video_size(128), 128)


if __name__ == "__main__":
    unittest.main()

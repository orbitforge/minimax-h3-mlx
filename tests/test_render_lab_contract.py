"""MLX-free contracts for the local H3 Render Lab orchestration."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tools.render_lab.resolutions import (
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    RESOLUTION_STEP,
    RESOLUTION_PRESETS,
    authoritative_dimensions,
    preset_payload,
)
from tools.render_lab.runner import (
    FIRST_LAST,
    I2V,
    T2V,
    RenderController,
    RenderFileLock,
    RenderRequest,
    RenderValidationError,
    RenderBusyError,
    build_generation_command,
    build_render_config,
    build_generation_command_for_namespace,
    execute_run,
    history_rows,
    parse_runtime_metrics,
    reserve_run_namespace,
    validate_render_request,
    anchors_for_mode,
    recognize_output_artifact,
    sha256_file,
)
from tools.render_lab.server import PAGE


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_CATALOG_DIMENSIONS = {
    (256, 256), (320, 320), (384, 384), (448, 448), (512, 512),
    (384, 256), (448, 256), (512, 288), (512, 384), (640, 360), (640, 384),
    (768, 432), (768, 512), (896, 512), (1024, 576), (1344, 768),
    (256, 384), (256, 448), (288, 512), (384, 512), (360, 640), (384, 640),
    (432, 768), (512, 768), (512, 896), (576, 1024), (768, 1344),
}


def request(tmp: Path, *, mode: str = T2V, images: tuple[Path, ...] = (), **changes: object) -> RenderRequest:
    values = {
        "mode": mode,
        "prompt": "a test prompt",
        "resolution_id": "quality-256-square-v05e",
        "steps": 16,
        "duration_seconds": 5.0,
        "seed": 0,
        "output_root": tmp / "render-lab",
        "output_name": "test.mp4",
        "image_paths": images,
        "checkpoint_root": tmp / "checkpoint",
        "transformer_path": tmp / "transformer",
    }
    values.update(changes)
    return RenderRequest(**values)


class RenderLabContractTests(unittest.TestCase):
    def test_mode_to_anchor_mapping_and_exact_image_counts(self) -> None:
        self.assertEqual(anchors_for_mode(T2V, 0), ())
        self.assertEqual(anchors_for_mode(I2V, 1), ("first",))
        self.assertEqual(anchors_for_mode(FIRST_LAST, 2), ("first", "last"))
        with self.assertRaises(RenderValidationError):
            anchors_for_mode(T2V, 1)
        with self.assertRaises(RenderValidationError):
            anchors_for_mode(I2V, 0)
        with self.assertRaises(RenderValidationError):
            anchors_for_mode(FIRST_LAST, 1)
        with self.assertRaises(RenderValidationError):
            anchors_for_mode(FIRST_LAST, 3)

    def test_missing_and_unreadable_images_rejected_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = root / "missing.png"
            with self.assertRaises(RenderValidationError):
                validate_render_request(
                    request(root, mode=I2V, images=(missing,)),
                    repo_root=ROOT,
                    check_images=True,
                    verify_runtime_geometry=False,
                )
            unreadable = root / "unreadable.png"
            unreadable.write_bytes(b"not an image")
            with self.assertRaises(RenderValidationError):
                validate_render_request(
                    request(root, mode=I2V, images=(unreadable,)),
                    repo_root=ROOT,
                    check_images=True,
                    verify_runtime_geometry=False,
                )

    def test_t2v_i2v_and_first_last_command_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            t2v = validate_render_request(request(root), repo_root=ROOT, check_images=False, verify_runtime_geometry=False)
            t2v_command = build_generation_command(t2v, python="python")
            self.assertNotIn("--image", t2v_command)
            self.assertNotIn("--anchor", t2v_command)

            image1 = root / "one.png"
            image2 = root / "two.png"
            image1.write_bytes(b"not validated in this construction test")
            image2.write_bytes(b"not validated in this construction test")
            i2v = validate_render_request(
                request(root, mode=I2V, images=(image1,)),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            i2v_command = build_generation_command(i2v, python="python")
            self.assertEqual(i2v_command.count("--image"), 1)
            self.assertEqual(i2v_command[i2v_command.index("--anchor") + 1], "first")

            first_last = validate_render_request(
                request(root, mode=FIRST_LAST, images=(image1, image2)),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            first_last_command = build_generation_command(first_last, python="python")
            anchors = [first_last_command[index + 1] for index, value in enumerate(first_last_command) if value == "--anchor"]
            self.assertEqual(anchors, ["first", "last"])

    def test_independent_dimensions_accept_square_landscape_and_portrait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for width, height in ((512, 512), (768, 512), (512, 768)):
                with self.subTest(width=width, height=height):
                    validated = validate_render_request(
                        request(root, resolution_id=None, width=width, height=height),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )
                    self.assertEqual((validated.width, validated.height), (width, height))
                    command = build_generation_command(validated, python="python")
                    self.assertEqual(command[command.index("--width") + 1], str(width))
                    self.assertEqual(command[command.index("--height") + 1], str(height))

    def test_invalid_dimensions_fail_closed_with_positive_and_alignment_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            invalid = (
                ({"width": 500, "height": 512}, "Width must be divisible by 32"),
                ({"width": 512, "height": 514}, "Height must be divisible by 32"),
                ({"width": 0, "height": 512}, "Width must be positive"),
                ({"width": -32, "height": 512}, "Width must be positive"),
            )
            for changes, message in invalid:
                with self.subTest(changes=changes), self.assertRaisesRegex(RenderValidationError, message):
                    validate_render_request(
                        request(root, resolution_id=None, **changes),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

    def test_dimension_contract_bounds_and_slider_step_are_exposed(self) -> None:
        controller = RenderController(ROOT)
        payload = controller.config_payload()
        self.assertEqual(
            payload["resolution_contract"],
            {
                "min_dimension": MIN_RESOLUTION,
                "max_dimension": MAX_RESOLUTION,
                "step": RESOLUTION_STEP,
                "positive": True,
                "source": "tools/render_lab/resolutions.py:independent-dimensions-v1",
            },
        )
        self.assertIn('id="width-range" type="range" min="128" max="1344" step="32"', PAGE)
        self.assertIn('id="height-range" type="range" min="128" max="1344" step="32"', PAGE)
        self.assertIn("form.set('width', $('width').value); form.set('height', $('height').value);", PAGE)

    def test_lora_controls_are_optional_and_map_to_runtime_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            disabled = validate_render_request(
                request(
                    root,
                    resolution_id=None,
                    width=512,
                    height=512,
                    lora_enabled=False,
                    lora_path=root / "not-used.safetensors",
                    lora_scale=-99,
                ),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            disabled_command = build_generation_command(disabled, python="python")
            self.assertNotIn("--lora", disabled_command)
            self.assertNotIn("--lora-scale", disabled_command)

            enabled = validate_render_request(
                request(
                    root,
                    resolution_id=None,
                    width=512,
                    height=512,
                    lora_enabled=True,
                    lora_path=root / "adapter.safetensors",
                    lora_scale=0.5,
                ),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            enabled_command = build_generation_command(enabled, python="python")
            self.assertEqual(
                Path(enabled_command[enabled_command.index("--lora") + 1]),
                (root / "adapter.safetensors").resolve(),
            )
            self.assertEqual(enabled_command[enabled_command.index("--lora-scale") + 1], "0.5")

    def test_lora_enabled_requires_path_and_finite_nonnegative_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = dict(
                resolution_id=None,
                width=512,
                height=512,
                lora_enabled=True,
                lora_path=root / "adapter.safetensors",
            )
            with self.assertRaisesRegex(RenderValidationError, "no adapter path"):
                validate_render_request(
                    request(root, **(base | {"lora_path": None})),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )
            for scale in (-0.1, float("nan"), float("inf")):
                with self.subTest(scale=scale), self.assertRaisesRegex(RenderValidationError, "finite nonnegative"):
                    validate_render_request(
                        request(root, **(base | {"lora_scale": scale})),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

    def test_turbo_requires_matching_ordinary_and_turbo_steps_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changes = {
                "resolution_id": None,
                "width": 512,
                "height": 512,
                "lora_enabled": True,
                "lora_path": root / "adapter.safetensors",
                "turbo_enabled": True,
                "turbo_steps": 8,
            }
            with self.assertRaisesRegex(RenderValidationError, "ordinary inference steps and Turbo steps"):
                validate_render_request(
                    request(root, steps=16, **changes),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )
            validated = validate_render_request(
                request(root, steps=8, **changes),
                repo_root=ROOT,
                check_images=False,
                verify_runtime_geometry=False,
            )
            command = build_generation_command(validated, python="python")
            self.assertIn("--turbo", command)
            self.assertEqual(command[command.index("--turbo-steps") + 1], "8")

    def test_known_resolution_presets_match_authoritative_runtime_helper(self) -> None:
        for preset in RESOLUTION_PRESETS:
            with self.subTest(preset=preset.preset_id):
                self.assertEqual(authoritative_dimensions(preset), preset.dimensions)

    def test_curated_resolution_catalog_contains_required_approved_dimensions(self) -> None:
        payload = preset_payload()
        exposed_dimensions = {(item["width"], item["height"]) for item in payload}
        self.assertTrue(REQUIRED_CATALOG_DIMENSIONS <= exposed_dimensions)
        self.assertEqual(
            len(payload),
            sum(1 for preset in RESOLUTION_PRESETS if preset.project_approved),
        )
        self.assertTrue(all(item["project_approved"] for item in payload))
        self.assertEqual({item["orientation"] for item in payload}, {"square", "landscape", "portrait"})
        self.assertEqual(
            next(item for item in payload if item["width"] == 640 and item["height"] == 360)["runtime_dimensions"],
            "640 × 384",
        )

    def test_unapproved_presets_are_not_exposed_to_the_browser(self) -> None:
        hidden = replace(RESOLUTION_PRESETS[0], preset_id="unapproved-test", project_approved=False)
        with patch(
            "tools.render_lab.resolutions.RESOLUTION_PRESETS",
            RESOLUTION_PRESETS + (hidden,),
        ):
            payload = preset_payload()
        self.assertNotIn("unapproved-test", {item["id"] for item in payload})
        self.assertTrue(all(item["project_approved"] for item in payload))

    def test_terminal_render_status_stops_polling_without_removing_active_updates(self) -> None:
        self.assertIn("const TERMINAL_STATUSES = new Set(['succeeded', 'failed']);", PAGE)
        self.assertIn("function stopStatusPolling()", PAGE)
        self.assertIn("const terminal = isTerminalSnapshot(snapshot);", PAGE)
        self.assertIn("$('preview').innerHTML = terminal ? previewFor(snapshot) : '';", PAGE)
        self.assertIn("if (active) { setStatus('Rendering…'); $('render').disabled = true; scheduleStatusPolling(); }", PAGE)
        self.assertIn("else if (terminal)", PAGE)
        self.assertNotIn("setTimeout(refreshStatus, 1000);", PAGE)

    def test_invalid_resolution_step_duration_and_output_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = request(root)
            for changes in (
                {"resolution_id": "not-a-preset"},
                {"steps": 1},
                {"steps": 41},
                {"duration_seconds": 4.9},
                {"duration_seconds": 15.1},
                {"output_name": "../overwrite.mp4"},
                {"output_root": "/"},
            ):
                with self.subTest(changes=changes), self.assertRaises(RenderValidationError):
                    validate_render_request(
                        RenderRequest(**(base.__dict__ | changes)),
                        repo_root=ROOT,
                        check_images=False,
                        verify_runtime_geometry=False,
                    )

    def test_immutable_run_directory_creation_and_config_serialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from PIL import Image

            image = root / "input.png"
            Image.new("RGB", (2, 2), color=(12, 34, 56)).save(image)
            validated = validate_render_request(
                request(root, mode=I2V, images=(image,)),
                repo_root=ROOT,
                check_images=True,
                verify_runtime_geometry=False,
            )
            first = reserve_run_namespace(validated)
            second = reserve_run_namespace(validated)
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertTrue(first.stdout_path.is_file())
            self.assertTrue(first.stderr_path.is_file())
            config = build_render_config(validated, first, ["python", "scripts/generate.py"], repo_root=ROOT)
            first.config_path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(json.loads(first.config_path.read_text())["resolution"]["preset_id"], validated.preset.preset_id)
            self.assertEqual(config["input_images"][0]["sha256"], sha256_file(image))
            self.assertEqual(config["image_anchors"], ["first"])
            with self.assertRaises(FileExistsError):
                from tools.render_lab.runner import initialize_run

                initialize_run(first, config)

    def test_benchmark_parses_only_emitted_runtime_metrics(self) -> None:
        metrics = parse_runtime_metrics(
            "  step 1/15 2.5s eta 0.5 min\n"
            "  step 2/15 3.5s eta 0.4 min\n"
            "  [memory] after denoising: mlx_peak=1.5GB mlx_cache=0B\n"
            "  video decoding: 4.0s\n"
            "2.0s per step, 0.1 min total\n"
        )
        self.assertEqual(metrics["actual_transformer_forward_count"], 2)
        self.assertEqual(metrics["seconds_per_forward"], 3.0)
        self.assertEqual(metrics["peak_mlx_memory_bytes"], int(1.5 * 1024**3))
        self.assertEqual(metrics["stage_timings_seconds"]["video decoding"], [4.0])
        self.assertEqual(metrics["runtime_reported_total_seconds"], 6.0)

    def test_failed_run_preserves_namespace_logs_and_benchmark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = validate_render_request(request(root), repo_root=ROOT, check_images=False, verify_runtime_geometry=False)
            namespace = reserve_run_namespace(validated)
            namespace.output_path.write_bytes(b"stale output is not produced by this failure")
            namespace.output_path.unlink()

            def fake_telemetry(_root: Path) -> dict[str, object]:
                return {"filesystem_free_bytes": 123}

            fake_command = [sys.executable, "-c", "print('hello'); print('bad', file=__import__('sys').stderr); raise SystemExit(3)"]
            result = execute_run(
                namespace,
                fake_command,
                repo_root=ROOT,
                telemetry=fake_telemetry,
                command_runner=lambda command, cwd, ns: __import__("tools.render_lab.runner", fromlist=["_run_command_text"])._run_command_text(command, cwd, ns),
            )
            self.assertFalse(result.success)
            self.assertEqual(result.exit_code, 3)
            self.assertTrue(namespace.run_dir.is_dir())
            self.assertTrue(namespace.stdout_path.is_file())
            self.assertTrue(namespace.stderr_path.is_file())
            benchmark = json.loads(namespace.benchmark_path.read_text())
            self.assertFalse(benchmark["success"])
            self.assertEqual(json.loads(namespace.status_path.read_text())["status"], "failed")

    def test_successful_run_artifact_recognition_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validated = validate_render_request(request(root), repo_root=ROOT, check_images=False, verify_runtime_geometry=False)
            namespace = reserve_run_namespace(validated)
            namespace.output_path.write_bytes(b"fake mp4")
            self.assertEqual(recognize_output_artifact(namespace.output_path), namespace.output_path)
            config = build_render_config(validated, namespace, ["fake"], repo_root=ROOT)
            namespace.config_path.write_text(json.dumps(config), encoding="utf-8")
            namespace.benchmark_path.write_text(json.dumps({"success": True, "total_elapsed_seconds": 2.5, "output_artifact": {"path": str(namespace.output_path), "kind": "mp4"}}), encoding="utf-8")
            namespace.status_path.write_text(json.dumps({"status": "succeeded"}), encoding="utf-8")
            rows = history_rows(validated.output_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "succeeded")
            self.assertEqual(rows[0]["artifact_name"], "test.mp4")

    def test_no_parallel_render_admission(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "render-lab"
            first = RenderFileLock(root)
            second = RenderFileLock(root)
            first.acquire()
            try:
                with self.assertRaises(RenderBusyError):
                    second.acquire()
            finally:
                first.release()
            second.acquire()
            second.release()


if __name__ == "__main__":
    unittest.main()

"""MLX-free contracts for the phase-one FL2V storyboard workflow."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.render_lab.runner import (
    FIRST_LAST,
    FL2V_STORYBOARD_SEGMENT_WORKFLOW,
    FL2V_STORYBOARD_WORKFLOW,
    I2V,
    RenderRequest,
    RenderValidationError,
    SINGLE_RENDER_WORKFLOW,
    T2V,
    _stage_storyboard_paths,
    build_generation_command,
    build_storyboard_config,
    build_storyboard_segment_jobs,
    execute_storyboard,
    initialize_run,
    reserve_run_namespace,
    reserve_storyboard_namespace,
    validate_render_request,
    validate_storyboard_request,
)
from tools.render_lab.server import PAGE, _render_request_from_fields


ROOT = Path(__file__).resolve().parents[1]


def _request(root: Path, **changes: object) -> RenderRequest:
    values: dict[str, object] = {
        "workflow": FL2V_STORYBOARD_WORKFLOW,
        "mode": FIRST_LAST,
        "prompt": "a shared storyboard prompt",
        "resolution_id": None,
        "width": 512,
        "height": 512,
        "steps": 16,
        "duration_seconds": 5.0,
        "seed": 1701,
        "output_root": root / "render-lab",
        "output_name": "segment.mp4",
        "checkpoint_root": root / "checkpoint",
        "transformer_path": root / "minimax-h3-mlx-6bit-streamed-adaln",
    }
    values.update(changes)
    return RenderRequest(**values)


def _cards(root: Path, count: int = 4) -> tuple[Path, ...]:
    cards = []
    for index in range(count):
        path = root / f"card-{index + 1}.png"
        Image.new("RGB", (4, 4), color=(index * 30, 20, 10)).save(path)
        cards.append(path)
    return tuple(cards)


class RenderLabStoryboardContractTests(unittest.TestCase):
    def test_storyboard_validation_requires_at_least_two_cards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            with self.assertRaisesRegex(RenderValidationError, "at least 2 cards"):
                validate_storyboard_request(
                    request,
                    (root / "only-card.png",),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )

    def test_empty_and_missing_card_paths_reject_before_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = _request(root)
            with self.assertRaisesRegex(RenderValidationError, "card 1 path must be non-empty"):
                validate_storyboard_request(
                    request,
                    ("", root / "card-2.png"),
                    repo_root=ROOT,
                    check_images=False,
                    verify_runtime_geometry=False,
                )
            with self.assertRaisesRegex(RenderValidationError, "not readable"):
                validate_storyboard_request(
                    request,
                    (root / "missing-1.png", root / "missing-2.png"),
                    repo_root=ROOT,
                    check_images=True,
                    verify_runtime_geometry=False,
                )

    def test_ordered_cards_create_exact_adjacent_segment_pairs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root)
            request = _request(root, storyboard_card_paths=cards)
            jobs = build_storyboard_segment_jobs(request)
            self.assertEqual(len(jobs), 3)
            self.assertEqual(
                [(job.start_card_index, job.end_card_index) for job in jobs],
                [(1, 2), (2, 3), (3, 4)],
            )
            self.assertEqual(
                [(job.start_path, job.end_path) for job in jobs],
                [
                    (cards[0].resolve(), cards[1].resolve()),
                    (cards[1].resolve(), cards[2].resolve()),
                    (cards[2].resolve(), cards[3].resolve()),
                ],
            )

    def test_global_settings_propagate_unchanged_to_every_segment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 3)
            request = _request(root, storyboard_card_paths=cards, prompt="same prompt", seed=42, steps=8)
            jobs = build_storyboard_segment_jobs(request)
            for job in jobs:
                with self.subTest(segment=job.segment_index):
                    self.assertEqual(job.request.prompt, request.prompt)
                    self.assertEqual(job.request.resolution_id, request.resolution_id)
                    self.assertEqual(job.request.width, request.width)
                    self.assertEqual(job.request.height, request.height)
                    self.assertEqual(job.request.duration_seconds, request.duration_seconds)
                    self.assertEqual(job.request.seed, request.seed)
                    self.assertEqual(job.request.steps, request.steps)
                    self.assertEqual(job.request.output_root, request.output_root)
                    self.assertEqual(job.request.output_name, request.output_name)
                    self.assertEqual(job.request.mode, FIRST_LAST)
                    self.assertEqual(job.request.workflow, "SINGLE_RENDER")
                    self.assertEqual(job.request.storyboard_card_paths, ())

    def test_storyboard_input_cards_are_staged_under_parent_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 3)
            request = _request(root, storyboard_card_paths=cards)
            validated = validate_storyboard_request(
                request,
                cards,
                repo_root=ROOT,
                check_images=True,
                verify_runtime_geometry=False,
            )
            parent = reserve_storyboard_namespace(validated.shared.output_root)  # type: ignore[union-attr]
            staged = _stage_storyboard_paths(parent, cards)
            self.assertEqual(
                [path.relative_to(parent.run_dir).as_posix() for path in staged],
                ["inputs/card-01.png", "inputs/card-02.png", "inputs/card-03.png"],
            )
            self.assertEqual([path.read_bytes() for path in staged], [path.read_bytes() for path in cards])

    def _execute(
        self,
        root: Path,
        cards: tuple[Path, ...],
        fake_runner,
        *,
        output_name: str = "segment.mp4",
    ) -> tuple[object, object, list[tuple[list[str], object]]]:
        request = _request(root, storyboard_card_paths=cards, output_name=output_name)
        validated = validate_storyboard_request(
            request,
            cards,
            repo_root=ROOT,
            check_runtime_paths=False,
            check_images=True,
            verify_runtime_geometry=False,
        )
        parent = reserve_storyboard_namespace(validated.shared.output_root)  # type: ignore[union-attr]
        initialize_run(parent, build_storyboard_config(validated, parent, repo_root=ROOT))
        calls: list[tuple[list[str], object]] = []
        result = execute_storyboard(
            parent,
            validated,
            repo_root=ROOT,
            telemetry=lambda _output_root: {"test": True},
            command_runner=fake_runner(calls),
            check_runtime_paths=False,
        )
        return result, parent, calls

    def test_each_segment_is_an_independent_child_and_prior_evidence_is_finalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 3)

            def fake_runner(calls):
                def run(command, _cwd, namespace):
                    if calls:
                        self.assertEqual(json.loads(Path(calls[-1][1].status_path).read_text())["status"], "succeeded")
                    calls.append((list(command), namespace))
                    namespace.output_path.write_bytes(f"segment-{len(calls)}".encode())
                    return 0, 0.01, "", ""

                return run

            result, parent, calls = self._execute(
                root,
                cards,
                fake_runner,
                output_name="sword-fight.mp4",
            )
            self.assertTrue(result.success)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(calls[0][1].run_id, calls[1][1].run_id)
            configs = [json.loads(Path(namespace.config_path).read_text()) for _, namespace in calls]
            self.assertTrue(all(config["workflow"] == FL2V_STORYBOARD_SEGMENT_WORKFLOW for config in configs))
            self.assertEqual(
                [value for command, _ in calls for value in command if value == "--anchor"],
                ["--anchor", "--anchor", "--anchor", "--anchor"],
            )
            manifest = json.loads(parent.output_path.read_text())
            self.assertEqual(manifest["overall_status"], "succeeded")
            self.assertEqual(manifest["per_segment_run_ids"], [namespace.run_id for _, namespace in calls])
            self.assertEqual(len(manifest["segments"]), 2)
            self.assertEqual([item["start_card_index"] for item in manifest["segments"]], [1, 2])
            self.assertEqual([item["end_card_index"] for item in manifest["segments"]], [2, 3])
            self.assertEqual([item["output_path"] for item in manifest["segments"]], [str(namespace.output_path) for _, namespace in calls])
            self.assertEqual([item["sha256"] for item in manifest["cards"]], [item["identity"] for item in manifest["cards"]])
            self.assertEqual(
                sorted(path.name for path in parent.run_dir.glob("*.mp4")),
                ["sword-fight-01.mp4", "sword-fight-02.mp4"],
            )
            self.assertEqual(
                sorted(path.name for path in parent.run_dir.parent.iterdir() if path.is_dir()),
                [parent.run_dir.name],
            )
            self.assertEqual(
                [namespace.run_dir for _, namespace in calls],
                [parent.run_dir, parent.run_dir],
            )
            self.assertEqual(
                len({namespace.run_id for _, namespace in calls}),
                2,
            )
            for segment, (_, child_namespace) in zip(manifest["segments"], calls):
                with self.subTest(segment=segment["segment_index"]):
                    self.assertEqual(segment["child_run_id"], child_namespace.run_id)
                    self.assertEqual(segment["child_run_directory"], str(parent.run_dir))
                    self.assertEqual(segment["output_path"], str(parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.mp4"))
                    expected_paths = {
                        "output_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.mp4",
                        "config_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.config.json",
                        "status_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.status.json",
                        "stdout_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.stdout.log",
                        "stderr_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.stderr.log",
                        "benchmark_path": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.benchmark.json",
                        "telemetry_dir": parent.run_dir / f"sword-fight-{segment['segment_index']:02d}.telemetry",
                    }
                    for key, expected in expected_paths.items():
                        self.assertEqual(Path(segment[key]), expected)
                        self.assertTrue(expected.exists())
                    self.assertEqual(
                        Path(segment["telemetry_before_path"]),
                        expected_paths["telemetry_dir"] / "before.json",
                    )
                    self.assertEqual(
                        Path(segment["telemetry_after_path"]),
                        expected_paths["telemetry_dir"] / "after.json",
                    )
                    self.assertEqual(
                        json.loads(child_namespace.config_path.read_text())["run_identifier"],
                        child_namespace.run_id,
                    )
                    self.assertEqual(
                        json.loads(child_namespace.status_path.read_text())["run_id"],
                        child_namespace.run_id,
                    )
                    self.assertEqual(
                        json.loads(child_namespace.benchmark_path.read_text())["run_id"],
                        child_namespace.run_id,
                    )

    def test_failure_stops_before_later_segment_launch_and_records_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 4)

            def fake_runner(calls):
                def run(command, _cwd, namespace):
                    calls.append((list(command), namespace))
                    if len(calls) == 1:
                        namespace.output_path.write_bytes(b"completed segment")
                        return 0, 0.01, "", ""
                    return 7, 0.01, "", "synthetic segment failure"

                return run

            result, parent, calls = self._execute(root, cards, fake_runner)
            self.assertFalse(result.success)
            self.assertEqual(len(calls), 2)
            manifest = json.loads(parent.output_path.read_text())
            self.assertEqual(manifest["overall_status"], "failed")
            self.assertEqual(manifest["failure_segment_index"], 2)
            self.assertEqual(manifest["segments"][0]["child_exit_code"], 0)
            self.assertTrue(manifest["segments"][0]["success"])
            self.assertTrue(Path(manifest["segments"][0]["output_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][0]["config_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][0]["status_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][0]["benchmark_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][0]["telemetry_before_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][0]["telemetry_after_path"]).is_file())
            self.assertEqual(manifest["segments"][1]["child_exit_code"], 7)
            self.assertFalse(manifest["segments"][1]["success"])
            self.assertTrue(Path(manifest["segments"][1]["config_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][1]["status_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][1]["benchmark_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][1]["telemetry_before_path"]).is_file())
            self.assertTrue(Path(manifest["segments"][1]["telemetry_after_path"]).is_file())
            self.assertEqual(len(manifest["segments"]), 2)
            self.assertEqual(manifest["segment_count"], 3)

    def test_ordinary_first_last_render_remains_single_child_behavior(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 2)
            request = _request(
                root,
                workflow=SINGLE_RENDER_WORKFLOW,
                image_paths=cards,
                storyboard_card_paths=(),
            )
            validated = validate_render_request(
                request,
                repo_root=ROOT,
                check_runtime_paths=False,
                check_images=True,
                verify_runtime_geometry=False,
            )
            command = build_generation_command(validated, python="python")
            self.assertEqual(validated.request.workflow, "SINGLE_RENDER")
            self.assertEqual(command.count("--image"), 2)
            self.assertEqual(
                [command[index + 1] for index, value in enumerate(command) if value == "--anchor"],
                ["first", "last"],
            )

    def test_ordinary_t2v_i2v_and_first_last_keep_one_run_directory_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cards = _cards(root, 2)
            cases = (
                (T2V, (), "t2v.mp4"),
                (I2V, (cards[0],), "i2v.mp4"),
                (FIRST_LAST, cards, "first-last.mp4"),
            )
            for mode, image_paths, output_name in cases:
                with self.subTest(mode=mode):
                    request = _request(
                        root,
                        workflow=SINGLE_RENDER_WORKFLOW,
                        mode=mode,
                        image_paths=image_paths,
                        storyboard_card_paths=(),
                        output_name=output_name,
                    )
                    validated = validate_render_request(
                        request,
                        repo_root=ROOT,
                        check_runtime_paths=False,
                        check_images=True,
                        verify_runtime_geometry=False,
                    )
                    namespace = reserve_run_namespace(validated)
                    self.assertIsNone(namespace.artifact_prefix)
                    self.assertEqual(namespace.output_path, namespace.run_dir / output_name)
                    self.assertEqual(namespace.config_path.name, "render-config.json")
                    self.assertEqual(namespace.status_path.name, "run-status.json")
                    self.assertEqual(namespace.benchmark_path.name, "benchmark.json")
                    self.assertEqual(namespace.stdout_path.name, "stdout.log")
                    self.assertEqual(namespace.stderr_path.name, "stderr.log")
                    self.assertEqual(namespace.telemetry_dir.name, "telemetry")

    def test_browser_exposes_simple_ordered_card_picker_and_drop_surface(self) -> None:
        self.assertIn("id=\"workflow\"", PAGE)
        self.assertIn("id=\"storyboard-cards\"", PAGE)
        self.assertIn("function addStoryboardCard()", PAGE)
        self.assertIn("function removeStoryboardCard(index)", PAGE)
        self.assertIn("event.dataTransfer.files[0]", PAGE)
        self.assertIn("Card ${index + 1}", PAGE)
        self.assertIn("form.append('storyboard_card', card.file, card.file.name);", PAGE)

    def test_server_payload_preserves_storyboard_workflow_and_card_order(self) -> None:
        request = _render_request_from_fields({
            "workflow": FL2V_STORYBOARD_WORKFLOW,
            "mode": FIRST_LAST,
            "prompt": "a shared storyboard prompt",
            "width": "512",
            "height": "512",
            "steps": "16",
            "duration_seconds": "5",
            "seed": "1701",
            "output_root": "out/render-lab",
            "output_name": "segment.mp4",
            "storyboard_card_paths": json.dumps(["/tmp/card-1.png", "/tmp/card-2.png"]),
        })
        self.assertEqual(request.normalized().workflow, FL2V_STORYBOARD_WORKFLOW)
        self.assertEqual(
            request.normalized().storyboard_card_paths,
            (Path("/tmp/card-1.png"), Path("/tmp/card-2.png")),
        )


if __name__ == "__main__":
    unittest.main()

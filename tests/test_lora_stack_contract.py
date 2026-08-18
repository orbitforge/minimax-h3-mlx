"""MLX-free contracts for Slice 021A adapter-stack admission and composition."""

from __future__ import annotations

import ast
import importlib.util
import json
import struct
import sys
import tempfile
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.h3_lora import validate_h3_lora_compatibility
from minimax_h3_mlx.lora import (
    LoRAError,
    LoRARegistry,
    LoRAStack,
    load_lora_safetensors,
)
from minimax_h3_mlx.turbo import TurboSchedule


def _write_safetensors(path: Path, *, down: np.ndarray, up: np.ndarray) -> None:
    tensors = {
        "blocks.0.mlp.fc2.lora_A.weight": np.asarray(down, dtype=np.float32),
        "blocks.0.mlp.fc2.lora_B.weight": np.asarray(up, dtype=np.float32),
    }
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    for name, value in tensors.items():
        raw = value.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(value.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def _expect_error(callable_value, phrase: str) -> None:
    try:
        callable_value()
    except LoRAError as exc:
        assert phrase in str(exc), str(exc)
    else:  # pragma: no cover
        raise AssertionError(f"expected an error containing {phrase!r}")


def test_stack_normalizes_order_and_independent_scales() -> None:
    stack = LoRAStack.from_sources(
        scheduling_path="./schedule.safetensors",
        auxiliary_paths=("./style-a.safetensors", "./style-b.safetensors"),
        auxiliary_scales=(0.80, 0.45),
    )
    assert stack.scheduling is not None
    assert stack.scheduling.source == (Path.cwd() / "schedule.safetensors").resolve()
    assert [entry.source.name for entry in stack.auxiliary] == ["style-a.safetensors", "style-b.safetensors"]
    assert [entry.scale for entry in stack.auxiliary] == [0.80, 0.45]
    assert [entry.scale for entry in stack.ordered_entries] == [1.0, 0.80, 0.45]


def test_stack_rejects_cardinality_nonfinite_and_duplicate_sources() -> None:
    _expect_error(
        lambda: LoRAStack.from_sources(
            auxiliary_paths=("a.safetensors", "b.safetensors"),
            auxiliary_scales=(0.8,),
        ),
        "cardinality mismatch",
    )
    _expect_error(
        lambda: LoRAStack.from_sources(
            auxiliary_paths=("a.safetensors",),
            auxiliary_scales=(float("nan"),),
        ),
        "must be finite",
    )
    _expect_error(
        lambda: LoRAStack.from_sources(
            auxiliary_paths=("./a.safetensors", "a.safetensors"),
        ),
        "duplicate normalized auxiliary",
    )
    _expect_error(
        lambda: LoRAStack.from_sources(
            scheduling_path="schedule.safetensors",
            auxiliary_paths=("./schedule.safetensors",),
        ),
        "duplicated as an auxiliary",
    )
    _expect_error(
        lambda: LoRAStack.from_sources(
            auxiliary_paths=("ordinary.safetensors", "style-a.safetensors", "style-b.safetensors"),
            auxiliary_scales=(0.7, 0.8),
        ),
        "cardinality mismatch",
    )


def test_stack_represents_legacy_ordinary_and_turbo_roles_without_pipeline_import() -> None:
    ordinary = LoRAStack.from_sources(
        auxiliary_paths=("ordinary.safetensors", "style-a.safetensors", "style-b.safetensors"),
        auxiliary_scales=(1.0, 0.80, 0.45),
    )
    assert ordinary.scheduling is None
    assert [entry.source.name for entry in ordinary.auxiliary] == [
        "ordinary.safetensors",
        "style-a.safetensors",
        "style-b.safetensors",
    ]

    turbo = LoRAStack.from_sources(
        scheduling_path="schedule.safetensors",
        auxiliary_paths=("style-a.safetensors", "style-b.safetensors"),
        auxiliary_scales=(0.80, 0.45),
    )
    assert turbo.scheduling is not None
    assert turbo.scheduling.source.name == "schedule.safetensors"
    assert [entry.source.name for entry in turbo.auxiliary] == [
        "style-a.safetensors",
        "style-b.safetensors",
    ]


def test_two_header_only_sources_merge_additively_and_preserve_lazy_payloads() -> None:
    with tempfile.TemporaryDirectory(prefix="lora-stack-") as directory:
        root = Path(directory)
        schedule_path = root / "schedule.safetensors"
        auxiliary_path = root / "style.safetensors"
        auxiliary_b_path = root / "style-b.safetensors"
        _write_safetensors(
            schedule_path,
            down=np.array([[1.0, 0.0]], dtype=np.float32),
            up=np.array([[1.0], [0.0]], dtype=np.float32),
        )
        _write_safetensors(
            auxiliary_path,
            down=np.array([[0.0, 1.0]], dtype=np.float32),
            up=np.array([[0.0], [2.0]], dtype=np.float32),
        )
        _write_safetensors(
            auxiliary_b_path,
            down=np.array([[1.0, 1.0]], dtype=np.float32),
            up=np.array([[1.0], [1.0]], dtype=np.float32),
        )

        scheduling = load_lora_safetensors(schedule_path, scheduling_owner=True)
        auxiliary = load_lora_safetensors(
            auxiliary_path,
            scale=0.5,
            adapter_name_prefix="auxiliary-0",
            scheduling_owner=False,
        )
        auxiliary_b = load_lora_safetensors(
            auxiliary_b_path,
            scale=0.25,
            adapter_name_prefix="auxiliary-1",
            scheduling_owner=False,
        )
        assert all(source.payload_bytes_read == 0 for source in scheduling.sources + auxiliary.sources)
        assert all(source.payload_bytes_read == 0 for source in auxiliary_b.sources)

        scheduling.merge(auxiliary)
        scheduling.merge(auxiliary_b)
        assert len(scheduling.adapters_for("blocks.0.mlp.fc2")) == 3
        assert all(source.payload_bytes_read == 0 for source in scheduling.sources)
        got = scheduling.delta("blocks.0.mlp.fc2", np.array([[3.0, 4.0]], dtype=np.float32))
        np.testing.assert_allclose(got, [[4.75, 5.75]])
        assert [source.path.name for source in scheduling.sources] == [
            "schedule.safetensors",
            "style.safetensors",
            "style-b.safetensors",
        ]
        assert scheduling.sources[0].fetch_count == 2
        assert scheduling.sources[1].fetch_count == 2
        assert scheduling.sources[2].fetch_count == 2


def test_auxiliary_metadata_cannot_change_scheduling_owner_or_cache_identity() -> None:
    scheduling = LoRARegistry(
        metadata={"turbo_steps": "4", "turbo_sigmas": "[1, 0.75, 0.5, 0.25, 0]"}
    )
    scheduling.register(
        "blocks.0.mlp.fc2",
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        adapter_name="schedule",
    )
    auxiliary = LoRARegistry()
    auxiliary.metadata["turbo_steps"] = "99"
    auxiliary.register(
        "blocks.1.mlp.fc2",
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        adapter_name="auxiliary",
        scale=0.45,
    )
    before = scheduling.cache_identity
    scheduling.merge(auxiliary)
    assert TurboSchedule.from_registry(scheduling).steps == 4
    assert TurboSchedule.from_registry(scheduling).sigmas == (1.0, 0.75, 0.5, 0.25, 0.0)
    assert scheduling.cache_identity != before

    reverse = LoRARegistry()
    reverse.register(
        "blocks.1.mlp.fc2",
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        adapter_name="auxiliary",
        scale=0.45,
    )
    reverse.register(
        "blocks.0.mlp.fc2",
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        adapter_name="schedule",
    )
    assert reverse.cache_identity != scheduling.cache_identity


def test_registry_merge_rejects_two_scheduling_owners_even_on_disjoint_targets() -> None:
    first = LoRARegistry(metadata={"turbo_steps": "4"})
    first.register(
        "blocks.0.mlp.fc2",
        np.ones((1, 2), dtype=np.float32),
        np.ones((3, 1), dtype=np.float32),
    )
    second = LoRARegistry(metadata={"turbo_steps": "8"})
    second.register(
        "blocks.1.mlp.fc2",
        np.ones((1, 2), dtype=np.float32),
        np.ones((3, 1), dtype=np.float32),
    )
    _expect_error(lambda: first.merge(second), "more than one scheduling adapter")


def test_each_auxiliary_is_admitted_before_pipeline_transformer_load() -> None:
    source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
    loader = source[source.index("    def _load_transformer"):source.index("    def _load_video_vae")]
    assert "validate_h3_lora_compatibility(auxiliary_registry" in loader
    assert "adapter_name_prefix=f\"auxiliary-{index}\"" in loader
    assert loader.index("validate_h3_lora_compatibility(auxiliary_registry") < loader.index(
        "self._load_component(\"dit\", \"transformer\", verbose)"
    )

    incompatible = LoRARegistry()
    incompatible.register(
        "foreign.layers.0.proj",
        np.ones((1, 2), dtype=np.float32),
        np.ones((3, 1), dtype=np.float32),
    )
    _expect_error(
        lambda: validate_h3_lora_compatibility(incompatible, adapter_path="bad-style.safetensors"),
        "compatible H3 targets=0",
    )


def test_pipeline_normalizes_legacy_ordinary_plus_default_auxiliary_scales() -> None:
    source = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
    tree = ast.parse(source)
    constructor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    function_source = ast.get_source_segment(source, constructor)
    assert function_source is not None
    assert "if not auxiliary_scales:" in function_source
    assert "auxiliary_scales = [1.0] * len(auxiliary_paths)" in function_source
    assert function_source.index("if not auxiliary_scales:") < function_source.index(
        "auxiliary_paths.insert(0, lora_path)"
    )

    # Exercise the exact effective stack expected from the constructor's normalization seam
    # without importing the MLX pipeline module on a host with no Metal device.
    auxiliary_paths = ["style-a.safetensors", "style-b.safetensors"]
    auxiliary_scales: list[float] = []
    if not auxiliary_scales:
        auxiliary_scales = [1.0] * len(auxiliary_paths)
    auxiliary_paths.insert(0, "ordinary.safetensors")
    auxiliary_scales.insert(0, 0.7)
    stack = LoRAStack.from_sources(
        auxiliary_paths=tuple(auxiliary_paths),
        auxiliary_scales=tuple(auxiliary_scales),
    )
    assert [entry.source.name for entry in stack.auxiliary] == [
        "ordinary.safetensors",
        "style-a.safetensors",
        "style-b.safetensors",
    ]
    assert [entry.scale for entry in stack.auxiliary] == [0.7, 1.0, 1.0]


def _load_generator_with_stubs():
    calls: dict[str, object] = {"factory": None, "generation": None}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["factory"] = {"args": args, **kwargs}
            return cls()

        def __call__(self, prompt: str | None, **kwargs):
            calls["prompt"] = prompt
            calls["generation"] = kwargs
            return SimpleNamespace(
                video=SimpleNamespace(shape=(1, 1, 1, 3)),
                audio=SimpleNamespace(shape=(2, 4)),
                fps=24,
                sample_rate=32_000,
                seconds_per_step=1.0,
                total_seconds=8.0,
            )

    pipeline_stub = types.ModuleType("minimax_h3_mlx.pipeline")
    pipeline_stub.MiniMaxH3Pipeline = FakePipeline
    media_stub = types.ModuleType("minimax_h3_mlx.media")
    media_stub.FFmpegUnavailableError = type("FFmpegUnavailableError", (Exception,), {})
    media_stub.save_frames = lambda *args, **kwargs: None
    media_stub.save_mp4 = lambda *args, **kwargs: None
    media_stub.save_wav = lambda *args, **kwargs: None

    module_name = "lora_stack_generate_contract_module"
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "scripts" / "generate.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "minimax_h3_mlx.pipeline": pipeline_stub,
            "minimax_h3_mlx.media": media_stub,
        },
    ):
        spec.loader.exec_module(module)
    return module, calls


def _run_cli(argv: list[str]):
    generate_cli, calls = _load_generator_with_stubs()
    with patch.object(sys, "argv", ["generate.py", *argv]):
        result = generate_cli.main()
    return result, calls


def test_cli_propagates_scheduling_source_and_ordered_auxiliary_scales() -> None:
    result, calls = _run_cli(
        [
            "stack prompt",
            "--turbo-lora",
            "/models/schedule.safetensors",
            "--additional-lora",
            "/models/style-a.safetensors",
            "--additional-lora-scale",
            "0.8",
            "--additional-lora",
            "/models/style-b.safetensors",
            "--additional-lora-scale",
            "0.45",
        ]
    )
    assert result == 0
    factory = calls["factory"]
    assert factory["lora_path"] == "/models/schedule.safetensors"
    assert factory["turbo"] is True
    assert factory["additional_lora_paths"] == (
        "/models/style-a.safetensors",
        "/models/style-b.safetensors",
    )
    assert factory["additional_lora_scales"] == (0.8, 0.45)


def test_cli_ordinary_lora_plus_default_auxiliary_scale_succeeds() -> None:
    result, calls = _run_cli(
        [
            "ordinary plus style prompt",
            "--lora",
            "/models/ordinary.safetensors",
            "--additional-lora",
            "/models/style.safetensors",
        ]
    )
    assert result == 0
    factory = calls["factory"]
    assert factory["lora_path"] == "/models/ordinary.safetensors"
    assert factory["additional_lora_paths"] == ("/models/style.safetensors",)
    # The CLI leaves omitted auxiliary scales absent; the production pipeline constructor applies
    # the default 1.0 while preserving the legacy ordinary --lora scale at its own seam.
    assert factory["additional_lora_scales"] is None


def test_cli_preserves_legacy_ordinary_lora_route() -> None:
    result, calls = _run_cli(["ordinary prompt", "--lora", "/models/ordinary.safetensors"])
    assert result == 0
    factory = calls["factory"]
    assert factory["lora_path"] == "/models/ordinary.safetensors"
    assert factory["turbo"] is False
    assert factory["additional_lora_paths"] == ()
    assert factory["additional_lora_scales"] is None


def test_cli_accepts_auxiliary_only_and_rejects_malformed_stack() -> None:
    result, calls = _run_cli(
        [
            "style prompt",
            "--additional-lora",
            "/models/style-a.safetensors",
            "--additional-lora-scale",
            "0.8",
            "--additional-lora",
            "/models/style-b.safetensors",
            "--additional-lora-scale",
            "0.45",
        ]
    )
    assert result == 0
    assert calls["factory"]["lora_path"] is None
    assert calls["factory"]["turbo"] is False

    generate_cli, malformed_calls = _load_generator_with_stubs()
    with patch.object(
        sys,
        "argv",
        [
            "generate.py",
            "style prompt",
            "--additional-lora",
            "a.safetensors",
            "--additional-lora",
            "b.safetensors",
            "--additional-lora-scale",
            "0.8",
        ],
    ):
        try:
            generate_cli.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("malformed auxiliary path/scale cardinality was accepted")
    assert malformed_calls["factory"] is None


def main() -> int:
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    failures: list[str] = []
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures.append(f"{test.__name__}: {exc}")
            print(f"FAIL {test.__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    if failures:
        print(f"\nFAILED ({len(failures)}):\n" + "\n".join(failures))
        return 1
    print(f"\nLoRA stack contracts passed ({len(tests)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

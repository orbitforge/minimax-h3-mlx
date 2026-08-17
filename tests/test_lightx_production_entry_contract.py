"""Host-independent contracts for the native LightX production pipeline and CLI entry points."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
PIPELINE_SOURCE = (ROOT / "minimax_h3_mlx" / "pipeline.py").read_text()
GENERATOR_PATH = ROOT / "scripts" / "generate.py"
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.lora import (
    LIGHTX_FL2VA_TURBO_4STEP_V0_1,
    LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P,
    LIGHTX_FL2VA_TURBO_8STEP_V1_0,
    LIGHTX_NATIVE_REPRESENTATION,
    LIGHTX_QKV_PROJECTIONS,
    LightXQKVOutputPermutation,
    normalize_lightx_target,
)


def _load_generator_with_stubs():
    calls: dict[str, object] = {"factory": None, "generation": None}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["factory"] = {"args": args, **kwargs}
            return cls()

        def __call__(self, prompt: str, **kwargs):
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

    module_name = "lightx_generate_contract_module"
    spec = importlib.util.spec_from_file_location(module_name, GENERATOR_PATH)
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


def test_native_lightx_cli_passes_explicit_manifest_entry_to_generation() -> None:
    result, calls = _run_cli(
        [
            "official parkour prompt",
            "--checkpoint",
            "/models/FL2VA",
            "--transformer",
            "/models/h3-6bit",
            "--lightx-lora",
            "/models/lightx.safetensors",
            "--steps",
            "8",
            "--turbo-steps",
            "8",
            "--height",
            "544",
            "--width",
            "960",
            "--duration",
            "5.1666667",
            "--seed",
            "0",
            "--output",
            "/tmp/lightx-contract.mp4",
        ]
    )

    assert result == 0
    factory_kwargs = calls["factory"]
    generation_kwargs = calls["generation"]
    assert factory_kwargs["lightx_path"] == "/models/lightx.safetensors"
    assert factory_kwargs["lora_path"] is None
    assert factory_kwargs["lora_scale"] == 1.0
    assert factory_kwargs["turbo"] is True
    assert factory_kwargs["turbo_steps"] == 8
    assert generation_kwargs["num_inference_steps"] == 8
    assert generation_kwargs["turbo"] is True
    assert generation_kwargs["turbo_steps"] == 8
    assert generation_kwargs["height"] == 544
    assert generation_kwargs["width"] == 960
    assert generation_kwargs["duration_seconds"] == 5.1666667
    assert generation_kwargs["seed"] == 0


def test_native_lightx_v01_cli_binds_four_step_manifest_and_schedule() -> None:
    result, calls = _run_cli(
        [
            "official parkour prompt",
            "--checkpoint",
            "/models/FL2VA",
            "--transformer",
            "/models/h3-6bit",
            "--lightx-lora",
            "/models/minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
            "--lightx-variant",
            "fl2va-turbo-4step-v0.1",
            "--steps",
            "4",
            "--turbo-steps",
            "4",
            "--height",
            "544",
            "--width",
            "960",
            "--duration",
            "5.1666666667",
            "--seed",
            "0",
            "--output",
            "/tmp/lightx-v01-contract.mp4",
        ]
    )

    assert result == 0
    factory_kwargs = calls["factory"]
    generation_kwargs = calls["generation"]
    manifest = factory_kwargs["lightx_manifest"]
    assert manifest == LIGHTX_FL2VA_TURBO_4STEP_V0_1
    assert manifest.task == "FL2VA/T2VA"
    assert manifest.nfe == 4
    assert manifest.video_shift == 12.0
    assert manifest.audio_shift == 3.0
    assert manifest.alpha == 8.0
    assert manifest.runtime_scale_default == 1.0
    assert manifest.effective_alpha_rank_multiplier == 0.0625
    assert manifest.representation == LIGHTX_NATIVE_REPRESENTATION
    assert factory_kwargs["turbo"] is True
    assert factory_kwargs["turbo_steps"] == 4
    assert generation_kwargs["num_inference_steps"] == 4
    assert generation_kwargs["turbo"] is True
    assert generation_kwargs["turbo_steps"] == 4
    assert generation_kwargs["height"] == 544
    assert generation_kwargs["width"] == 960
    assert generation_kwargs["duration_seconds"] == 5.1666666667
    assert generation_kwargs["seed"] == 0


def test_native_lightx_768p_v10_cli_selects_exact_manifest_and_schedule() -> None:
    result, calls = _run_cli(
        [
            "official parkour prompt",
            "--checkpoint",
            "/models/FL2VA",
            "--transformer",
            "/models/h3-6bit",
            "--lightx-lora",
            "/models/minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
            "--lightx-variant",
            "fl2va-turbo-4step-v1.0-768p",
            "--lora-scale",
            "1.0",
            "--steps",
            "4",
            "--turbo-steps",
            "4",
            "--height",
            "768",
            "--width",
            "1344",
            "--duration",
            "5.1666666667",
            "--seed",
            "0",
            "--output",
            "/tmp/lightx-768p-v10-contract.mp4",
        ]
    )

    assert result == 0
    factory_kwargs = calls["factory"]
    generation_kwargs = calls["generation"]
    manifest = factory_kwargs["lightx_manifest"]
    assert manifest == LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P
    assert manifest.variant_id == "lightx2v-fl2va-turbo-4step-v1.0-768p"
    assert manifest.task == "FL2VA/T2VA"
    assert manifest.representation == LIGHTX_NATIVE_REPRESENTATION
    assert manifest.rank == 128
    assert manifest.alpha == 128.0
    assert manifest.runtime_scale_default == 1.0
    assert manifest.effective_alpha_rank_multiplier == 1.0
    assert manifest.nfe == 4
    assert manifest.video_shift == 6.0
    assert manifest.audio_shift == 3.0
    assert factory_kwargs["lightx_path"].endswith(
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    )
    assert factory_kwargs["lora_scale"] == 1.0
    assert factory_kwargs["turbo"] is True
    assert factory_kwargs["turbo_steps"] == 4
    assert generation_kwargs["num_inference_steps"] == 4
    assert generation_kwargs["turbo"] is True
    assert generation_kwargs["turbo_steps"] == 4
    assert generation_kwargs["height"] == 768
    assert generation_kwargs["width"] == 1344
    assert generation_kwargs["duration_seconds"] == 5.1666666667
    assert generation_kwargs["seed"] == 0


def test_native_lightx_v01_entry_preserves_split_qkv_rank_and_local_layout() -> None:
    manifest = LIGHTX_FL2VA_TURBO_4STEP_V0_1
    specs = [
        normalize_lightx_target(
            f"transformer_blocks.0.attn.to_{projection}",
            manifest=manifest,
        )
        for projection in LIGHTX_QKV_PROJECTIONS
    ]
    assert {spec.role for spec in specs} == set(LIGHTX_QKV_PROJECTIONS)
    assert {spec.local_target for spec in specs} == {"blocks.0.attn.qkv_proj"}
    assert all(isinstance(spec.output_transform, LightXQKVOutputPermutation) for spec in specs)
    assert all(spec.output_transform.projection in LIGHTX_QKV_PROJECTIONS for spec in specs)
    assert all(spec.output_transform.num_attention_heads == 56 for spec in specs)
    assert all(spec.output_transform.attention_head_dim == 128 for spec in specs)
    assert manifest.rank == 128
    assert manifest.alpha == 8.0


def test_larry_and_no_lora_cli_routes_remain_generic_and_normal() -> None:
    _result, larry_calls = _run_cli(["larry prompt", "--turbo-lora", "/models/larry.safetensors"])
    larry_factory = larry_calls["factory"]
    larry_generation = larry_calls["generation"]
    assert larry_factory["lora_path"] == "/models/larry.safetensors"
    assert larry_factory["lightx_path"] is None
    assert larry_factory["turbo"] is True
    assert larry_generation["turbo"] is True

    _result, plain_calls = _run_cli(["plain prompt"])
    plain_factory = plain_calls["factory"]
    plain_generation = plain_calls["generation"]
    assert plain_factory["lora_path"] is None
    assert plain_factory["lightx_path"] is None
    assert plain_factory["turbo"] is False
    assert plain_generation["turbo"] is False
    assert plain_generation["num_inference_steps"] is None


def test_lightx_cli_rejects_noncanonical_nfe_before_pipeline_admission() -> None:
    generate_cli, calls = _load_generator_with_stubs()
    with patch.object(sys, "argv", ["generate.py", "prompt", "--lightx", "/models/lightx.safetensors", "--steps", "4"]):
        try:
            generate_cli.main()
        except SystemExit as exc:
            assert exc.code == 2
        else:  # pragma: no cover
            raise AssertionError("native LightX accepted a noncanonical NFE")
    assert calls["factory"] is None


def test_pipeline_source_binds_manifest_to_native_loader_schedule_and_cache() -> None:
    parsed = ast.parse(PIPELINE_SOURCE)
    assert isinstance(parsed, ast.Module)
    for marker in (
        "lightx_manifest=lightx_manifest",
        "LIGHTX_FL2VA_TURBO_8STEP_V1_0",
        "load_lightx_safetensors(",
        "variant=self._lightx_manifest or LIGHTX_FL2VA_TURBO_8STEP_V1_0",
        "self._lightx_manifest.video_shift",
        "self._lightx_manifest.audio_shift",
        "self._lightx_manifest.nfe",
        "lora_registry=self._lora_registry",
        "native LightX production requires its manifest-bound Turbo schedule",
    ):
        assert marker in PIPELINE_SOURCE, marker

    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.names: set[str] = set()

        def visit_FunctionDef(self, node):
            self.names.add(node.name)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(parsed)
    assert {"_load_transformer", "_build_schedules", "_ensure_cache"}.issubset(visitor.names)
    assert LIGHTX_FL2VA_TURBO_8STEP_V1_0.nfe == 8
    assert LIGHTX_FL2VA_TURBO_8STEP_V1_0.video_shift == 12.0
    assert LIGHTX_FL2VA_TURBO_8STEP_V1_0.audio_shift == 3.0


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("LightX production-entry contracts passed")

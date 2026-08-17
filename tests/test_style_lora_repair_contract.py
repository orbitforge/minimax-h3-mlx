"""Focused offline contracts for the style-LoRA target normalization repair."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.h3_lora import (
    H3LoRACompatibilityError,
    is_h3_compatible_target,
    validate_h3_lora_compatibility,
)
from minimax_h3_mlx.lora import (
    LoRARegistry,
    apply_lora,
    canonical_target,
    load_lora_payload,
    load_lora_safetensors,
)
from tools.render_lab.runner import (
    REFERENCE_TURBO_PRESET_ID,
    T2V,
    RenderRequest,
    build_generation_command,
    validate_render_request,
)


MMH3_PATH = Path("/Users/elbancol/Downloads/MMH3-V1.safetensors")
MMH3_HEADER_SHA256 = "0746bc380117d6e1919e8bdfb1f9ea81a1e9b97e6d867a55b7ac82acc3dd80fa"


def _pair_payload(target: str) -> dict[str, np.ndarray]:
    return {
        f"{target}.lora_A.weight": np.array([[1.0, 0.0]], dtype=np.float32),
        f"{target}.lora_B.weight": np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
    }


def _registry(*targets: str) -> LoRARegistry:
    registry = LoRARegistry()
    for target in targets:
        registry.register(
            target,
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[1.0], [2.0], [3.0]], dtype=np.float32),
            alpha=1.0,
        )
    return registry


def test_diffusion_model_targets_canonicalize_to_local_h3_paths() -> None:
    assert canonical_target("diffusion_model.blocks.0.attn.qkv_proj") == "blocks.0.attn.qkv_proj"
    assert canonical_target("diffusion_model.blocks.49.mlp.fc2") == "blocks.49.mlp.fc2"


def test_known_wrapper_composition_is_deterministic_and_bounded() -> None:
    expected = "blocks.3.mlp.fc1"
    for target in (
        "base_model.model.diffusion_model.transformer.transformer_blocks.3.mlp.fc1",
        "transformer.diffusion_model.base_model.model.transformer_blocks.3.mlp.fc1",
    ):
        assert canonical_target(target) == expected

    too_many = ".".join(("model",) * 6) + ".blocks.0.mlp.fc1"
    try:
        canonical_target(too_many)
    except ValueError as exc:
        assert "too many composed wrapper scopes" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unbounded wrapper composition was accepted")


def test_unknown_wrapper_is_not_silently_stripped() -> None:
    target = "exporter.diffusion_model.blocks.0.attn.qkv_proj"
    assert canonical_target(target) == target
    assert not is_h3_compatible_target(target)


def test_existing_wrapper_semantics_and_target_identity_remain_stable() -> None:
    assert canonical_target("base_model.model.blocks.0.mlp.fc2") == "blocks.0.mlp.fc2"
    assert canonical_target("base_model.blocks.0.mlp.fc2") == "blocks.0.mlp.fc2"
    assert canonical_target("model.blocks.0.mlp.fc2") == "blocks.0.mlp.fc2"
    assert canonical_target("transformer.blocks.0.mlp.fc2") == "blocks.0.mlp.fc2"
    assert canonical_target("transformer_blocks.0.mlp.fc2") == "blocks.0.mlp.fc2"

    down = np.array([[1.0, 0.0]], dtype=np.float32)
    up = np.array([[1.0], [2.0]], dtype=np.float32)
    wrapped = LoRARegistry()
    wrapped.register("diffusion_model.blocks.0.mlp.fc2", down, up, source_identity="same")
    local = LoRARegistry()
    local.register("blocks.0.mlp.fc2", down, up, source_identity="same")
    assert wrapped.cache_identity == local.cache_identity


def test_real_mmh3_header_admission_resolves_exactly_200_h3_core_targets() -> None:
    assert MMH3_PATH.is_file()
    registry = load_lora_safetensors(MMH3_PATH, strict=True, scale=1.0)
    source = registry.sources[0]
    report = validate_h3_lora_compatibility(registry, adapter_path=MMH3_PATH)

    assert source.header_sha256 == MMH3_HEADER_SHA256
    assert len(source.keys) == 600
    assert registry.adapter_count == 200
    assert len(registry.targets) == 200
    assert report.registered_adapter_count == 200
    assert report.registered_target_count == 200
    assert report.compatible_target_count == 200
    assert report.incompatible_target_count == 0
    assert report.topology == {
        "total": 200,
        "resident_core": 200,
        "streamed_block_adaln": 0,
        "resident_final_adaln": 0,
        "other": 0,
    }
    assert source.fetch_count == 200
    assert source.payload_bytes_read == 800
    assert all(
        hasattr(adapter.down, "materialize") and hasattr(adapter.up, "materialize")
        for target in registry.targets
        for adapter in registry.adapters_for(target)
    )
    assert all(
        adapter.rank == 32
        and adapter.alpha == 32.0
        and adapter.scale == 1.0
        and adapter.multiplier == 1.0
        for target in registry.targets
        for adapter in registry.adapters_for(target)
    )

    for target in (
        "blocks.0.attn.qkv_proj",
        "blocks.25.attn.out_proj",
        "blocks.49.mlp.fc1",
        "blocks.49.mlp.fc2",
    ):
        assert registry.has(target)


def test_normalized_target_is_applied_at_the_existing_projection_seam() -> None:
    registry = load_lora_payload(_pair_payload("diffusion_model.blocks.0.attn.qkv_proj"))
    activation = np.array([[2.0, 4.0]], dtype=np.float32)

    def layer(value: np.ndarray) -> np.ndarray:
        return np.zeros((value.shape[0], 3), dtype=np.float32)

    assert registry.has("blocks.0.attn.qkv_proj")
    output = apply_lora(
        layer,
        activation,
        target="blocks.0.attn.qkv_proj",
        registry=registry,
    )
    np.testing.assert_allclose(output, [[2.0, 4.0, 6.0]])
    assert float(np.linalg.norm(output)) > 0.0


def test_zero_compatible_h3_targets_fail_closed_with_actionable_evidence() -> None:
    registry = _registry("exporter.layers.0.proj")
    try:
        validate_h3_lora_compatibility(registry, adapter_path="/tmp/style.safetensors")
    except H3LoRACompatibilityError as exc:
        message = str(exc)
        assert "adapter=/tmp/style.safetensors" in message
        assert "registered targets=1" in message
        assert "compatible H3 targets=0" in message
        assert "examples='exporter.layers.0.proj'" in message
    else:  # pragma: no cover
        raise AssertionError("zero-compatible H3 adapter was admitted")


def test_partial_h3_compatibility_is_admitted_and_reports_unsupported_targets() -> None:
    report = validate_h3_lora_compatibility(
        _registry("blocks.0.attn.qkv_proj", "other_runtime.layers.0.proj"),
        adapter_path="/tmp/partial.safetensors",
    )
    assert report.registered_target_count == 2
    assert report.compatible_target_count == 1
    assert report.incompatible_target_count == 1
    assert report.incompatible_targets == ("other_runtime.layers.0.proj",)


def test_render_lab_none_reference_passes_mmh3_path_and_scale_unchanged() -> None:
    with tempfile.TemporaryDirectory(prefix="style-lora-render-lab-") as directory:
        root = Path(directory)
        request = RenderRequest(
            mode=T2V,
            prompt="a style test",
            resolution_id="quality-256-square-v05e",
            steps=16,
            duration_seconds=5.0,
            seed=7,
            output_root=root / "render-lab",
            output_name="style.mp4",
            checkpoint_root=root / "checkpoint",
            transformer_path=root / "transformer",
            lora_enabled=True,
            lora_path=MMH3_PATH,
            lora_scale=1.0,
            turbo_preset_id=REFERENCE_TURBO_PRESET_ID,
        )
        validated = validate_render_request(
            request,
            check_images=False,
            verify_runtime_geometry=False,
        )
        command = build_generation_command(validated, python="python")
        assert command[command.index("--lora") + 1] == str(MMH3_PATH)
        assert command[command.index("--lora-scale") + 1] == "1"


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("Style LoRA repair contracts passed")

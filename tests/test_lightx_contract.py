"""CPU-only native LightX2V normalization contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.lora import (
    LIGHTX_FL2VA_TURBO_4STEP_V0_1,
    LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P,
    LIGHTX_FL2VA_TURBO_8STEP_V1_0,
    LIGHTX_NATIVE_REPRESENTATION,
    LIGHTX_QKV_PROJECTIONS,
    LightX2VManifest,
    LightXFC1ValueGateToGateValue,
    LightXQKVOutputPermutation,
    LoRARegistry,
    is_core_projection_target,
    is_streamed_adaln_target,
    load_lightx_safetensors,
    load_lora_safetensors,
    normalize_lightx_target,
)


LIGHTX_PATH = Path(
    "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/"
    "minimax-h3-turbo/lightx2v/Minimax-h3-Turbo/"
    "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
)
LIGHTX_4STEP_PATH = LIGHTX_PATH.with_name("minimax_h3_fl2v_turbo_4step_v0.1.safetensors")
LIGHTX_4STEP_V1_0_768P_PATH = LIGHTX_PATH.with_name(
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
)


def _small_manifest() -> LightX2VManifest:
    return LightX2VManifest(
        variant_id="test-lightx",
        task="FL2VA/T2VA",
        nfe=8,
        video_shift=12.0,
        audio_shift=3.0,
        rank=2,
        alpha=2.0,
        runtime_scale_default=1.0,
        effective_alpha_rank_multiplier=1.0,
        representation=LIGHTX_NATIVE_REPRESENTATION,
        artifact_name="test.safetensors",
        num_attention_heads=2,
        attention_head_dim=3,
        hidden_size=4,
        ffn_hidden_size=5,
        main_block_count=1,
        token_refiner_block_count=1,
    )


def test_manifest_is_explicit_and_not_filename_derived() -> None:
    manifest = LIGHTX_FL2VA_TURBO_8STEP_V1_0
    assert manifest.task == "FL2VA/T2VA"
    assert manifest.nfe == 8
    assert manifest.video_shift == 12.0
    assert manifest.audio_shift == 3.0
    assert manifest.rank == 128
    assert manifest.alpha == 8.0
    assert manifest.runtime_scale_default == 1.0
    assert manifest.effective_alpha_rank_multiplier == 0.0625
    assert manifest.representation == LIGHTX_NATIVE_REPRESENTATION
    assert manifest.cache_identity.startswith("lightx2v:")


def test_v01_manifest_binds_the_authoritative_four_step_contract() -> None:
    manifest = LIGHTX_FL2VA_TURBO_4STEP_V0_1
    assert manifest.task == "FL2VA/T2VA"
    assert manifest.nfe == 4
    assert manifest.video_shift == 12.0
    assert manifest.audio_shift == 3.0
    assert manifest.rank == 128
    assert manifest.alpha == 8.0
    assert manifest.runtime_scale_default == 1.0
    assert manifest.effective_alpha_rank_multiplier == 0.0625
    assert manifest.representation == LIGHTX_NATIVE_REPRESENTATION
    assert manifest.artifact_name == "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"


def test_768p_v10_manifest_binds_the_authoritative_four_step_contract() -> None:
    manifest = LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P
    assert manifest.variant_id == "lightx2v-fl2va-turbo-4step-v1.0-768p"
    assert manifest.task == "FL2VA/T2VA"
    assert manifest.nfe == 4
    assert manifest.video_shift == 6.0
    assert manifest.audio_shift == 3.0
    assert manifest.rank == 128
    assert manifest.alpha == 128.0
    assert manifest.runtime_scale_default == 1.0
    assert manifest.effective_alpha_rank_multiplier == 1.0
    assert manifest.representation == LIGHTX_NATIVE_REPRESENTATION
    assert manifest.artifact_name == "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"


def test_native_lightx_v01_header_contract_uses_manifest_alpha_and_split_qkv() -> None:
    assert LIGHTX_4STEP_PATH.is_file()
    registry = load_lightx_safetensors(
        LIGHTX_4STEP_PATH,
        variant=LIGHTX_FL2VA_TURBO_4STEP_V0_1,
    )
    report = registry.lightx_report
    assert report is not None
    assert report.variant_id == LIGHTX_FL2VA_TURBO_4STEP_V0_1.variant_id
    assert report.native_tensor_count == 624
    assert report.native_pair_count == 312
    assert report.qkv_triplet_count == 52
    assert registry.metadata["turbo_steps"] == 4
    assert "alpha" not in registry.sources[0].metadata
    assert "lora_alpha" not in registry.sources[0].metadata

    qkv_targets = [target for target in registry.targets if target.endswith("attn.qkv_proj")]
    assert len(qkv_targets) == 52
    for target in qkv_targets:
        adapters = registry.adapters_for(target)
        assert len(adapters) == 3
        assert {adapter.metadata["lightx_role"] for adapter in adapters} == set(LIGHTX_QKV_PROJECTIONS)
        assert all(adapter.rank == 128 for adapter in adapters)
        assert all(adapter.alpha == 8.0 for adapter in adapters)
        assert all(adapter.multiplier == 0.0625 for adapter in adapters)


def test_native_lightx_768p_v10_header_contract_uses_manifest_alpha_and_split_qkv() -> None:
    assert LIGHTX_4STEP_V1_0_768P_PATH.is_file()
    registry = load_lightx_safetensors(
        LIGHTX_4STEP_V1_0_768P_PATH,
        variant=LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P,
    )
    report = registry.lightx_report
    assert report is not None
    assert report.variant_id == LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P.variant_id
    assert report.native_tensor_count == 624
    assert report.native_pair_count == 312
    assert report.qkv_triplet_count == 52
    assert registry.metadata["turbo_steps"] == 4
    assert registry.metadata["lightx_task"] == "FL2VA/T2VA"
    assert registry.metadata["lightx_video_shift"] == 6.0
    assert registry.metadata["lightx_audio_shift"] == 3.0
    assert registry.metadata["lightx_rank"] == 128
    assert registry.metadata["lightx_alpha"] == 128.0
    assert registry.metadata["lightx_runtime_scale_default"] == 1.0
    assert registry.metadata["lightx_effective_alpha_rank_multiplier"] == 1.0
    assert registry.metadata["lightx_representation"] == LIGHTX_NATIVE_REPRESENTATION

    qkv_targets = [target for target in registry.targets if target.endswith("attn.qkv_proj")]
    assert len(qkv_targets) == 52
    for target in qkv_targets:
        adapters = registry.adapters_for(target)
        assert len(adapters) == 3
        assert {adapter.metadata["lightx_role"] for adapter in adapters} == set(LIGHTX_QKV_PROJECTIONS)
        assert all(adapter.rank == 128 for adapter in adapters)
        assert all(adapter.alpha == 128.0 for adapter in adapters)
        assert all(adapter.scale == 1.0 for adapter in adapters)
        assert all(adapter.multiplier == 1.0 for adapter in adapters)


def test_native_target_normalization_covers_token_refiner_and_main_block() -> None:
    token_q = normalize_lightx_target("token_refiner.refiner_blocks.1.attn.to_q")
    assert token_q.local_target == "token_refiner.blocks.1.attn.qkv_proj"
    assert token_q.role == "q"
    assert isinstance(token_q.output_transform, LightXQKVOutputPermutation)

    main_fc1 = normalize_lightx_target("transformer_blocks.49.ff.net.0.proj")
    assert main_fc1.local_target == "blocks.49.mlp.fc1"
    assert main_fc1.role == "fc1"
    assert isinstance(main_fc1.output_transform, LightXFC1ValueGateToGateValue)

    assert normalize_lightx_target("transformer_blocks.0.attn.to_out.0").local_target == (
        "blocks.0.attn.out_proj"
    )
    assert normalize_lightx_target("transformer_blocks.0.ff.net.2").local_target == "blocks.0.mlp.fc2"
    assert normalize_lightx_target("transformer_blocks.0.attn.to_out.0").output_transform is None
    assert normalize_lightx_target("transformer_blocks.0.ff.net.2").output_transform is None

    for invalid in (
        "transformer_blocks.50.attn.to_q",
        "token_refiner.refiner_blocks.2.attn.to_q",
        "transformer_blocks.0.adaln_proj.linear",
        "transformer_blocks.0.attn.to_qkv",
    ):
        try:
            normalize_lightx_target(invalid)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"invalid LightX target was accepted: {invalid}")


def test_qkv_and_fc1_transforms_are_explicit_and_layout_exact() -> None:
    native_q = np.arange(12, dtype=np.float32).reshape(2, 6)
    for projection, slot in zip(LIGHTX_QKV_PROJECTIONS, range(3)):
        transform = LightXQKVOutputPermutation(projection, 2, 3)
        got = transform(native_q)
        expected = np.zeros((2, 2, 3, 3), dtype=np.float32)
        expected[:, :, slot, :] = native_q.reshape(2, 2, 3)
        np.testing.assert_array_equal(got, expected.reshape(2, 18))

    native_fc1 = np.arange(20, dtype=np.float32).reshape(2, 10)
    got_fc1 = LightXFC1ValueGateToGateValue(5)(native_fc1)
    np.testing.assert_array_equal(got_fc1, np.concatenate([native_fc1[:, 5:], native_fc1[:, :5]], axis=-1))


def test_native_lightx_header_contract_recognizes_all_pairs_and_triplets() -> None:
    assert LIGHTX_PATH.is_file()
    registry = load_lightx_safetensors(LIGHTX_PATH)
    report = registry.lightx_report
    assert report is not None
    assert report.native_tensor_count == 624
    assert report.native_pair_count == 312
    assert report.normalized_adapter_count == 312
    assert report.qkv_triplet_count == 52
    assert report.adaln_target_count == 0
    assert len(report.normalized_targets) == 208
    assert registry.adapter_count == 312
    assert registry.representation_identity == LIGHTX_FL2VA_TURBO_8STEP_V1_0.cache_identity
    assert registry.lightx_manifest == LIGHTX_FL2VA_TURBO_8STEP_V1_0
    assert registry.sources[0].payload_bytes_read == 0
    assert registry.metadata["turbo_steps"] == 8

    assert all(is_core_projection_target(target) for target in registry.targets)
    assert all(not is_streamed_adaln_target(target) for target in registry.targets)
    assert registry.topology_counts == {
        "total": 312,
        "resident_core": 312,
        "streamed_block_adaln": 0,
        "resident_final_adaln": 0,
        "other": 0,
    }

    for target in registry.targets:
        for adapter in registry.adapters_for(target):
            assert adapter.rank == 128
            assert adapter.alpha == 8.0
            assert adapter.scale == 1.0
            assert adapter.multiplier == 0.0625
            assert adapter.source_identity == registry.sources[0].identity
            if adapter.output_transform is not None:
                assert adapter.runtime_identity[-1].startswith("lightx-")

    qkv_targets = [target for target in registry.targets if target.endswith("attn.qkv_proj")]
    assert len(qkv_targets) == 52
    for target in qkv_targets:
        adapters = registry.adapters_for(target)
        assert len(adapters) == 3
        assert {adapter.metadata["lightx_role"] for adapter in adapters} == set(LIGHTX_QKV_PROJECTIONS)
        assert all(adapter.down.shape == (128, 5376) for adapter in adapters)
        assert all(adapter.up.shape == (7168, 128) for adapter in adapters)
        assert all(adapter.rank != 384 for adapter in adapters)

    # The generic entry point requires the same explicit manifest; it does not infer native format
    # from the filename or incomplete metadata.
    via_generic = load_lora_safetensors(LIGHTX_PATH, variant=LIGHTX_FL2VA_TURBO_8STEP_V1_0)
    assert via_generic.lightx_report == report


def test_representative_native_deltas_match_local_layout_transforms() -> None:
    registry = load_lightx_safetensors(LIGHTX_PATH)
    activation_by_target = {
        "blocks.0.attn.qkv_proj": np.linspace(-0.25, 0.25, 5376, dtype=np.float32).reshape(1, -1),
        "blocks.0.attn.out_proj": np.linspace(-0.2, 0.2, 7168, dtype=np.float32).reshape(1, -1),
        "blocks.0.mlp.fc1": np.linspace(-0.15, 0.15, 5376, dtype=np.float32).reshape(1, -1),
        "blocks.0.mlp.fc2": np.linspace(-0.1, 0.1, 14336, dtype=np.float32).reshape(1, -1),
    }
    for target, activation in activation_by_target.items():
        for adapter in registry.adapters_for(target):
            down = adapter.down.materialize()
            up = adapter.up.materialize()
            native_delta = np.matmul(np.matmul(activation, down.T), up.T) * np.float32(adapter.multiplier)
            expected = (
                native_delta
                if adapter.output_transform is None
                else adapter.output_transform(native_delta)
            )
            np.testing.assert_allclose(adapter.delta(activation), expected, rtol=0.0, atol=0.0)


def test_transformed_pairs_remain_rank_128_when_registered() -> None:
    manifest = _small_manifest()
    registry = LoRARegistry(representation_identity=manifest.cache_identity)
    activation = np.arange(4, dtype=np.float32).reshape(1, 4)
    q_specs = [normalize_lightx_target(f"transformer_blocks.0.attn.to_{role}", manifest=manifest) for role in LIGHTX_QKV_PROJECTIONS]
    for index, spec in enumerate(q_specs):
        down = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32) + index
        up = np.arange(12, dtype=np.float32).reshape(6, 2) + index
        registry.register(
            spec.local_target,
            down,
            up,
            alpha=manifest.alpha,
            adapter_name=f"test:{spec.role}",
            output_transform=spec.output_transform,
        )
    got = registry.delta("blocks.0.attn.qkv_proj", activation)
    assert got.shape == (1, 18)
    assert all(adapter.rank == 2 for adapter in registry.adapters_for("blocks.0.attn.qkv_proj"))


def test_variant_representation_participates_in_cache_identity() -> None:
    manifest = _small_manifest()
    down = np.ones((2, 4), dtype=np.float32)
    up = np.ones((6, 2), dtype=np.float32)
    generic = LoRARegistry()
    generic.register("blocks.0.attn.qkv_proj", down, up, alpha=2.0, source_identity="same")
    native = LoRARegistry(representation_identity=manifest.cache_identity)
    native.register("blocks.0.attn.qkv_proj", down, up, alpha=2.0, source_identity="same")
    assert generic.cache_identity != native.cache_identity


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("LightX contracts passed")

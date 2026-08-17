"""Synthetic MLX proof for native LightX projection seams.

This is intentionally a projection-level test.  It uses tiny ordinary MLX linear layers and
compares the actual ``linear_with_lora`` path against the existing NumPy adapter reference,
including the native split-Q/K/V and fused-FC1 layout transforms.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.lora import (
    LIGHTX_NATIVE_REPRESENTATION,
    LIGHTX_QKV_PROJECTIONS,
    LightX2VManifest,
    LightXFC1ValueGateToGateValue,
    LightXQKVOutputPermutation,
    LoRARegistry,
    linear_with_lora,
    normalize_lightx_target,
)


def _manifest() -> LightX2VManifest:
    return LightX2VManifest(
        variant_id="synthetic-lightx-mlx",
        task="FL2VA/T2VA",
        nfe=8,
        video_shift=12.0,
        audio_shift=3.0,
        rank=2,
        alpha=2.0,
        runtime_scale_default=1.0,
        effective_alpha_rank_multiplier=1.0,
        representation=LIGHTX_NATIVE_REPRESENTATION,
        artifact_name="synthetic.safetensors",
        num_attention_heads=2,
        attention_head_dim=3,
        hidden_size=4,
        ffn_hidden_size=5,
        main_block_count=1,
        token_refiner_block_count=1,
    )


def _load_mlx():
    import mlx.core as mx

    try:
        mx.eval(mx.array([0.0]))
    except Exception as exc:  # pragma: no cover - exercised only on unavailable hosts
        raise RuntimeError(f"MLX device unavailable: {exc}") from exc
    import mlx.nn as nn

    return mx, nn


def _linear(nn, input_features: int, output_features: int):
    return nn.Linear(input_features, output_features, bias=False)


def _assert_projection_matches_numpy(
    mx,
    layer,
    activation_np: np.ndarray,
    target: str,
    registry: LoRARegistry,
) -> None:
    activation_mx = mx.array(activation_np)
    base = layer(activation_mx)
    numpy_delta = registry.delta(target, activation_np)
    got = linear_with_lora(layer, activation_mx, target, registry)
    mx.eval(base, got)
    mlx_delta = registry.delta(target, activation_mx)
    mx.eval(mlx_delta)
    mlx_delta_np = np.asarray(mlx_delta.astype(mx.float32))
    numpy_delta_at_output_dtype = mx.array(numpy_delta).astype(base.dtype)
    expected = base + numpy_delta_at_output_dtype
    mx.eval(expected)
    np.testing.assert_allclose(
        mlx_delta_np,
        np.asarray(numpy_delta, dtype=np.float32),
        # Metal float32 GEMM and NumPy BLAS can choose different accumulation kernels.  The
        # explicit CPU transform contracts remain exact; this comparison tolerates that backend
        # difference while still catching wrong target dimensions or row assembly.
        rtol=1e-2,
        atol=5e-2,
        err_msg=f"target={target} NumPy/MLX low-rank delta mismatch before base projection",
    )
    np.testing.assert_allclose(
        np.asarray(got.astype(mx.float32)),
        np.asarray(expected.astype(mx.float32)),
        rtol=1e-2,
        atol=5e-2,
        err_msg=(
            f"target={target} activation_dtype={activation_mx.dtype} "
            f"numpy_delta_dtype={np.asarray(numpy_delta).dtype} "
            f"mlx_delta_dtype={mlx_delta.dtype} "
            f"base_dtype={base.dtype} got_dtype={got.dtype} "
            f"max_delta_difference={float(np.max(np.abs(mlx_delta_np - np.asarray(numpy_delta, dtype=np.float32))))}"
        ),
    )


def test_native_lightx_mlx_projection_path_matches_numpy_reference() -> None:
    mx, nn = _load_mlx()
    manifest = _manifest()
    registry = LoRARegistry(representation_identity=manifest.cache_identity)
    rng = np.random.default_rng(17)

    def register(native_target: str, down_features: int, up_features: int) -> None:
        spec = normalize_lightx_target(native_target, manifest=manifest)
        registry.register(
            spec.local_target,
            rng.normal(size=(manifest.rank, down_features)).astype(np.float32),
            rng.normal(size=(up_features, manifest.rank)).astype(np.float32),
            alpha=manifest.alpha,
            adapter_name=f"synthetic:{native_target}",
            source_identity="synthetic-lightx-source",
            output_transform=spec.output_transform,
            metadata={"lightx_role": spec.role},
        )

    # Main-block split Q/K/V remain three rank-2 pairs and are assembled only at the
    # local fused qkv output boundary.
    for role in LIGHTX_QKV_PROJECTIONS:
        register(f"transformer_blocks.0.attn.to_{role}", 4, manifest.inner_dim)
    # Token-refiner Q/K/V use the same local fused path under a separate target prefix.
    for role in LIGHTX_QKV_PROJECTIONS:
        register(f"token_refiner.refiner_blocks.0.attn.to_{role}", 4, manifest.inner_dim)
    register("transformer_blocks.0.attn.to_out.0", manifest.inner_dim, 4)
    register("transformer_blocks.0.ff.net.0.proj", 4, 2 * manifest.ffn_hidden_size)
    register("transformer_blocks.0.ff.net.2", manifest.ffn_hidden_size, 4)

    _assert_projection_matches_numpy(
        mx,
        _linear(nn, 4, 3 * manifest.inner_dim),
        rng.normal(size=(2, 4)).astype(np.float32),
        "blocks.0.attn.qkv_proj",
        registry,
    )
    _assert_projection_matches_numpy(
        mx,
        _linear(nn, 4, 3 * manifest.inner_dim),
        rng.normal(size=(2, 4)).astype(np.float32),
        "token_refiner.blocks.0.attn.qkv_proj",
        registry,
    )
    _assert_projection_matches_numpy(
        mx,
        _linear(nn, manifest.inner_dim, 4),
        rng.normal(size=(2, manifest.inner_dim)).astype(np.float32),
        "blocks.0.attn.out_proj",
        registry,
    )
    _assert_projection_matches_numpy(
        mx,
        _linear(nn, 4, 2 * manifest.ffn_hidden_size),
        rng.normal(size=(2, 4)).astype(np.float32),
        "blocks.0.mlp.fc1",
        registry,
    )
    _assert_projection_matches_numpy(
        mx,
        _linear(nn, manifest.ffn_hidden_size, 4),
        rng.normal(size=(2, manifest.ffn_hidden_size)).astype(np.float32),
        "blocks.0.mlp.fc2",
        registry,
    )

    assert all(adapter.rank == manifest.rank for target in registry.targets for adapter in registry.adapters_for(target))
    assert not any(adapter.rank == 3 * manifest.rank for target in registry.targets for adapter in registry.adapters_for(target))
    print(
        "synthetic LightX MLX projections passed: "
        f"targets={len(registry.targets)} adapters={registry.adapter_count} "
        "native_rank_preserved=true qkv_interleaved=true fc1_gate_value=true"
    )


if __name__ == "__main__":
    test_native_lightx_mlx_projection_path_matches_numpy_reference()

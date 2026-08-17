"""MLX-free LoRA registry, loader, and quantized-projection contracts."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.lora import (
    LoRAError,
    LoRARegistry,
    LoRASafetensorsSource,
    is_core_projection_target,
    is_final_adaln_target,
    is_streamed_adaln_target,
    load_lora_payload,
    load_lora_safetensors,
    linear_with_lora,
)
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler, _linspace_1_to_0
from minimax_h3_mlx.turbo import TurboSchedule


class DummyLinear:
    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None):
        self.weight = np.asarray(weight, dtype=np.float32)
        self.bias = None if bias is None else np.asarray(bias, dtype=np.float32)
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        result = np.asarray(x, dtype=np.float32) @ self.weight.T
        return result if self.bias is None else result + self.bias


class QuantizedStyleLinear(DummyLinear):
    """A packed-storage stand-in: its dense weight is not available to LoRA math."""

    def __init__(self, weight: np.ndarray):
        self.packed_weight = np.asarray(np.rint(weight * 16), dtype=np.int8)
        self.scales = np.full((weight.shape[0], 1), 1.0 / 16.0, dtype=np.float32)
        self.weight = None
        self.bias = None
        self.calls = 0

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self.calls += 1
        dense = self.packed_weight.astype(np.float32) * self.scales
        return np.asarray(x, dtype=np.float32) @ dense.T


class _NumpyMLX:
    """Small scheduler-only stand-in; this test must not initialize MLX/Metal."""

    float32 = np.float32

    @staticmethod
    def array(values, dtype=None):
        return np.asarray(values, dtype=dtype)


def _bf16_bytes(value: np.ndarray) -> bytes:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return (bits >> np.uint32(16)).astype("<u2").tobytes()


def _write_safetensors(path: Path, tensors: list[tuple[str, str, tuple[int, ...], bytes]]) -> None:
    header: dict[str, object] = {}
    payload = bytearray()
    offset = 0
    item_sizes = {"BF16": 2, "F32": 4, "I8": 1}
    for name, dtype, shape, raw in tensors:
        assert len(raw) == (int(np.prod(shape, dtype=np.int64)) if shape else 1) * item_sizes[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(raw)]}
        payload.extend(raw)
        offset += len(raw)
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_loader_pairs_common_names_and_alpha() -> None:
    payload = {
        "base_model.model.blocks.0.attn.qkv_proj.lora_A.weight": np.array(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        "base_model.model.blocks.0.attn.qkv_proj.lora_B.weight": np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
        ),
        "base_model.model.blocks.0.attn.qkv_proj.alpha": np.array(4.0, dtype=np.float32),
    }
    registry = load_lora_payload(payload)
    assert registry.targets == ("blocks.0.attn.qkv_proj",)
    adapter = registry.adapters_for("blocks.0.attn.qkv_proj")[0]
    assert adapter.rank == 2
    assert adapter.alpha == 4.0
    assert adapter.multiplier == 2.0


def test_lora_adds_expected_delta_to_ordinary_projection() -> None:
    registry = LoRARegistry()
    registry.register(
        "blocks.0.mlp.fc2",
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[3.0], [4.0]], dtype=np.float32),
        alpha=1.0,
    )
    layer = DummyLinear(np.zeros((2, 2), dtype=np.float32))
    activation = np.array([[5.0, 6.0]], dtype=np.float32)
    output = linear_with_lora(layer, activation, "blocks.0.mlp.fc2", registry)
    # down(activation) = 17; up.T turns it into [51, 68]; alpha/rank = 1.
    np.testing.assert_allclose(output, [[51.0, 68.0]])
    assert layer.calls == 1


def test_lora_wraps_quantized_style_projection_without_dense_base_weight() -> None:
    base_weight = np.array([[0.5, -0.5], [1.0, 0.25]], dtype=np.float32)
    layer = QuantizedStyleLinear(base_weight)
    registry = LoRARegistry()
    registry.register(
        "blocks.3.attn.out_proj",
        np.eye(2, dtype=np.float32),
        np.array([[0.25, 0.0], [0.0, -0.5]], dtype=np.float32),
        alpha=2.0,
    )
    activation = np.array([[2.0, 4.0]], dtype=np.float32)
    base = layer(activation)
    got = linear_with_lora(layer, activation, "blocks.3.attn.out_proj", registry)
    expected_delta = activation @ np.eye(2, dtype=np.float32) @ np.array(
        [[0.25, 0.0], [0.0, -0.5]], dtype=np.float32
    ).T
    np.testing.assert_allclose(got, base + expected_delta)
    assert layer.weight is None
    assert layer.calls == 2


def test_loader_rejects_orphan_pair() -> None:
    payload = {"blocks.0.mlp.fc1.lora_down.weight": np.ones((1, 2), dtype=np.float32)}
    try:
        load_lora_payload(payload)
    except LoRAError as exc:
        assert "exactly one down and one up" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("orphan LoRA pair was accepted")


def test_header_first_loader_fetches_only_one_named_bf16_pair() -> None:
    with tempfile.TemporaryDirectory(prefix="lora-selective-") as directory:
        path = Path(directory) / "adapter.safetensors"
        tensors = [
            (
                "blocks.0.attn.qkv_proj.lora_A.weight",
                "BF16",
                (2, 3),
                _bf16_bytes(np.arange(6, dtype=np.float32).reshape(2, 3)),
            ),
            (
                "blocks.0.attn.qkv_proj.lora_B.weight",
                "BF16",
                (4, 2),
                _bf16_bytes(np.arange(8, dtype=np.float32).reshape(4, 2)),
            ),
            (
                "blocks.1.attn.qkv_proj.lora_A.weight",
                "BF16",
                (2, 3),
                _bf16_bytes(np.ones((2, 3), dtype=np.float32)),
            ),
            (
                "blocks.1.attn.qkv_proj.lora_B.weight",
                "BF16",
                (4, 2),
                _bf16_bytes(np.ones((4, 2), dtype=np.float32)),
            ),
        ]
        _write_safetensors(path, tensors)
        source = LoRASafetensorsSource(path)
        assert len(source.keys) == 4
        assert source.payload_bytes_read == 0

        registry = load_lora_safetensors(path)
        loaded = registry.sources[0]
        assert loaded.payload_bytes_read == 0
        adapter = registry.adapters_for("blocks.0.attn.qkv_proj")[0]
        got = adapter.delta(np.ones((1, 3), dtype=np.float32), transient=True)
        assert got.shape == (1, 4)
        assert adapter.prepared_pair_count == 0
        assert loaded.fetch_count == 2
        assert loaded.payload_bytes_read == 28
        np.testing.assert_allclose(adapter.down.materialize()[:1], [[0.0, 1.0, 2.0]])
        assert loaded.fetch_count == 3


def test_selective_loader_rejects_wrong_dtype_and_shape_from_header() -> None:
    with tempfile.TemporaryDirectory(prefix="lora-header-validation-") as directory:
        root = Path(directory)
        wrong_dtype = root / "wrong-dtype.safetensors"
        _write_safetensors(
            wrong_dtype,
            [
                ("blocks.0.mlp.fc1.lora_A.weight", "I8", (1, 2), b"\x00\x01"),
                ("blocks.0.mlp.fc1.lora_B.weight", "I8", (3, 1), b"\x00\x01\x02"),
            ],
        )
        try:
            load_lora_safetensors(wrong_dtype)
        except LoRAError as exc:
            assert "unsupported dtype" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("integer LoRA tensors were accepted")

        wrong_shape = root / "wrong-shape.safetensors"
        _write_safetensors(
            wrong_shape,
            [
                ("blocks.0.mlp.fc1.lora_A.weight", "F32", (2, 3), np.zeros((2, 3), dtype=np.float32).tobytes()),
                ("blocks.0.mlp.fc1.lora_B.weight", "F32", (4, 4), np.zeros((4, 4), dtype=np.float32).tobytes()),
            ],
        )
        try:
            load_lora_safetensors(wrong_shape)
        except LoRAError as exc:
            assert "rank mismatch" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("wrong-rank LoRA tensors were accepted")


def test_named_fetch_fails_closed_when_source_file_identity_changes() -> None:
    with tempfile.TemporaryDirectory(prefix="lora-stale-source-") as directory:
        path = Path(directory) / "adapter.safetensors"
        raw = np.zeros((1, 2), dtype=np.float32).tobytes()
        _write_safetensors(
            path,
            [
                ("blocks.0.mlp.fc1.lora_A.weight", "F32", (1, 2), raw),
                ("blocks.0.mlp.fc1.lora_B.weight", "F32", (3, 1), np.zeros((3, 1), dtype=np.float32).tobytes()),
            ],
        )
        source = LoRASafetensorsSource(path)
        path.write_bytes(path.read_bytes() + b"x")
        try:
            source.reference("blocks.0.mlp.fc1.lora_A.weight").materialize()
        except LoRAError as exc:
            assert "identity changed" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("stale LoRA source was read")


def test_resident_adapter_preparation_is_cached_across_functional_calls() -> None:
    registry = LoRARegistry()
    adapter = registry.register(
        "blocks.0.mlp.fc2",
        np.array([[1.0, 2.0]], dtype=np.float32),
        np.array([[3.0], [4.0]], dtype=np.float32),
    )
    layer = DummyLinear(np.zeros((2, 2), dtype=np.float32))
    activation = np.array([[5.0, 6.0]], dtype=np.float32)
    for _ in range(3):
        linear_with_lora(layer, activation, "blocks.0.mlp.fc2", registry)
    assert adapter.prepare_count == 1
    assert adapter.prepared_pair_count == 1


def test_missing_alpha_scale_one_and_scale_zero_preserve_exact_gate() -> None:
    registry = LoRARegistry()
    adapter = registry.register(
        "blocks.0.mlp.fc2",
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32),
    )
    assert adapter.alpha == adapter.rank == 2
    assert adapter.multiplier == 1.0
    layer = DummyLinear(np.array([[4.0, 0.0], [0.0, 5.0]], dtype=np.float32))
    activation = np.array([[2.0, 3.0]], dtype=np.float32)
    base = layer(activation)
    registry.set_scale(0.0)
    np.testing.assert_array_equal(linear_with_lora(layer, activation, adapter.target, registry), base)
    assert adapter.multiplier == 0.0


def test_multiple_stacked_loras_remain_additive_with_independent_strengths() -> None:
    registry = LoRARegistry()
    target = "blocks.0.mlp.fc2"
    first = registry.register(
        target,
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[1.0], [0.0]], dtype=np.float32),
        adapter_name="first",
        alpha=1.0,
        scale=1.0,
    )
    second = registry.register(
        target,
        np.array([[0.0, 1.0]], dtype=np.float32),
        np.array([[0.0], [2.0]], dtype=np.float32),
        adapter_name="second",
        alpha=1.0,
        scale=0.5,
    )
    got = registry.delta(target, np.array([[3.0, 4.0]], dtype=np.float32))
    expected = np.array([[3.0, 4.0]], dtype=np.float32) @ first.down.T @ first.up.T
    expected += 0.5 * (np.array([[3.0, 4.0]], dtype=np.float32) @ second.down.T @ second.up.T)
    np.testing.assert_allclose(got, expected)


def test_streamed_block_pair_is_transient_and_final_pair_is_resident_eligible() -> None:
    registry = LoRARegistry()
    block = registry.register(
        "blocks.7.adaln_proj.linear",
        np.ones((2, 3), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
    )
    final = registry.register(
        "final_layer.adaln_proj.linear",
        np.ones((2, 3), dtype=np.float32),
        np.ones((4, 2), dtype=np.float32),
    )
    activation = np.ones((1, 3), dtype=np.float32)
    registry.delta(block.target, activation, transient=True)
    assert block.prepared_pair_count == 0
    registry.delta(block.target, activation, transient=True)
    assert block.prepare_count == 2
    registry.delta(final.target, activation)
    registry.delta(final.target, activation)
    assert final.prepare_count == 1
    assert final.prepared_pair_count == 1


def test_canonical_turbo_topology_is_header_validated_without_payload_reads() -> None:
    path = Path(
        "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/"
        "minimax-h3-loras/larryvrh/minimax_h3_turbo_v4_step600_ema.safetensors"
    )
    assert path.is_file()
    registry = load_lora_safetensors(path)
    assert registry.adapter_count == 259
    assert registry.topology_counts == {
        "total": 259,
        "resident_core": 208,
        "streamed_block_adaln": 50,
        "resident_final_adaln": 1,
        "other": 0,
    }
    assert registry.sources[0].payload_bytes_read == 0
    block = registry.adapters_for("blocks.49.adaln_proj.linear")[0]
    assert block.alpha == block.rank
    assert block.multiplier == 1.0
    assert TurboSchedule.from_registry(registry).steps == 8


def test_topology_classification_and_cache_identity_include_strength_source_and_stack_order() -> None:
    assert is_core_projection_target("base_model.model.blocks.0.attn.qkv_proj")
    assert is_streamed_adaln_target("blocks.49.adaln_proj.linear")
    assert not is_streamed_adaln_target("blocks.50.adaln_proj.linear")
    assert is_final_adaln_target("final_layer.adaln_proj.linear")

    def make_registry(order: tuple[str, str] = ("first", "second"), source: str = "source-a"):
        result = LoRARegistry()
        for name in order:
            result.register(
                "blocks.0.mlp.fc2",
                np.ones((1, 2), dtype=np.float32),
                np.ones((3, 1), dtype=np.float32),
                adapter_name=name,
                source_identity=source,
            )
        return result

    same = make_registry()
    same_again = make_registry()
    assert same.cache_identity == same_again.cache_identity
    same.set_scale(0.5)
    assert same.cache_identity != same_again.cache_identity
    assert make_registry(source="source-b").cache_identity != same_again.cache_identity
    assert make_registry(order=("second", "first")).cache_identity != same_again.cache_identity


def test_turbo_schedule_is_explicit_and_metadata_aware() -> None:
    registry = LoRARegistry(metadata={"turbo_steps": "6"})
    assert TurboSchedule.from_registry(registry).steps == 6
    assert TurboSchedule.from_registry(registry, steps=8).name == "turbo-8"
    sigma_registry = LoRARegistry(metadata={"turbo_sigmas": "[1.0, 0.5, 0.0]"})
    assert TurboSchedule.from_registry(sigma_registry).steps == 2
    assert TurboSchedule.from_registry(sigma_registry).sigmas == (1.0, 0.5, 0.0)
    assert TurboSchedule(2, sigmas=(1.0, 0.5, 0.0)).steps == 2
    try:
        TurboSchedule(1)
    except ValueError as exc:
        assert "at least 2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid Turbo schedule was accepted")


def test_turbo_nfe_contract_reports_grid_shifts_transitions_and_forwards() -> None:
    expected_schedules = {
        4: {
            "q": (1.0, 0.75, 0.5, 0.25, 0.0),
            "video": (1.0, 0.9729729890823364, 0.9230769276618958, 0.800000011920929, 0.0),
            "audio": (1.0, 0.8999999761581421, 0.75, 0.5, 0.0),
        },
        6: {
            "q": (1.0, 0.8333333134651184, 0.6666666269302368, 0.5, 0.3333333432674408,
                 0.1666666716337204, 0.0),
            "video": (1.0, 0.9836066365242004, 0.9599999785423279, 0.9230769276618958,
                      0.8571428060531616, 0.7058823108673096, 0.0),
            "audio": (1.0, 0.9375000596046448, 0.8571428060531616, 0.75, 0.5999999642372131,
                      0.375, 0.0),
        },
        8: {
            "q": (1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0),
            "video": (1.0, 0.9882352948188782, 0.9729729890823364, 0.9523809552192688,
                      0.9230769276618958, 0.8780487775802612, 0.800000011920929,
                      0.6315789222717285, 0.0),
            "audio": (1.0, 0.9545454382896423, 0.8999999761581421, 0.8333333134651184,
                      0.75, 0.6428571343421936, 0.5, 0.30000001192092896, 0.0),
        },
    }
    for nfe in (4, 6, 8):
        with patch("minimax_h3_mlx.scheduler._mx", return_value=_NumpyMLX):
            video = MiniMaxH3Scheduler(shift=12.0)
            audio = MiniMaxH3Scheduler(shift=3.0)
            TurboSchedule(nfe).configure(video, audio)

        unshifted_q = _linspace_1_to_0(nfe + 1)
        expected = expected_schedules[nfe]
        np.testing.assert_array_equal(unshifted_q, np.asarray(expected["q"], dtype=np.float32))
        shift_video = np.float32(12.0)
        shift_audio = np.float32(3.0)
        expected_video = (
            shift_video * unshifted_q
            / (np.float32(1.0) + np.float32(12.0 - 1.0) * unshifted_q)
        ).astype(np.float32)
        expected_audio = (
            shift_audio * unshifted_q
            / (np.float32(1.0) + np.float32(3.0 - 1.0) * unshifted_q)
        ).astype(np.float32)

        np.testing.assert_array_equal(expected_video, np.asarray(expected["video"], dtype=np.float32))
        np.testing.assert_array_equal(expected_audio, np.asarray(expected["audio"], dtype=np.float32))
        np.testing.assert_array_equal(video.sigmas, expected_video)
        np.testing.assert_array_equal(audio.sigmas, expected_audio)
        assert len(unshifted_q) == nfe + 1
        assert video.sigmas[-1] == 0.0
        assert audio.sigmas[-1] == 0.0
        assert video.num_inference_steps == nfe
        assert audio.num_inference_steps == nfe
        assert len(video.timesteps) == nfe
        assert len(audio.timesteps) == nfe
        print(
            f"Turbo NFE={nfe}: sigma_grid_points={len(unshifted_q)} "
            f"unshifted_q={tuple(float(value) for value in unshifted_q)} "
            f"video_sigmas={tuple(float(value) for value in video.sigmas)} "
            f"audio_sigmas={tuple(float(value) for value in audio.sigmas)} "
            f"transitions={video.num_inference_steps} transformer_forwards={len(video.timesteps)}"
        )


if __name__ == "__main__":
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            value()
    print("LoRA contracts passed")

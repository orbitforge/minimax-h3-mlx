"""Generic low-rank adapter loading and projection math.

The H3 transformer uses both ordinary ``nn.Linear`` modules and MLX quantized
linear modules.  A LoRA adapter is deliberately applied *around* the callable
projection rather than merged into its weight: the base callable therefore
continues to own quantized storage, while the low-rank path works at the
projection's activation dtype.

The registry is intentionally independent of the H3 module tree.  It can be
used with a NumPy or test-double projection as well as with MLX, which keeps
adapter format and numerical contracts testable without loading a 33B model.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
import struct
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


class LoRAError(ValueError):
    """Raised when an adapter is malformed or cannot match a projection."""


def _is_mlx_array(value: Any) -> bool:
    module = type(value).__module__
    return bool(getattr(value, "__mlx_array__", False)) or module.startswith("mlx.")


def _mlx():
    import mlx.core as mx

    return mx


_SAFETENSORS_HEADER_BYTES = 8
_SAFETENSORS_FLOAT_DTYPES = {"BF16", "F16", "F32", "F64"}
_SAFETENSORS_NUMPY_DTYPES = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "U16": np.dtype("<u2"),
    "I16": np.dtype("<i2"),
    "U32": np.dtype("<u4"),
    "I32": np.dtype("<i4"),
    "U64": np.dtype("<u8"),
    "I64": np.dtype("<i8"),
    "F16": np.dtype("<f2"),
    "F32": np.dtype("<f4"),
    "F64": np.dtype("<f8"),
}
_SAFETENSORS_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "U32": 4,
    "I32": 4,
    "U64": 8,
    "I64": 8,
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
}


LIGHTX_FAMILY = "LightX2V"
LIGHTX_TASK_FL2VA_T2VA = "FL2VA/T2VA"
LIGHTX_TASK_REF2VA = "Ref2VA"
LIGHTX_TRANSFORMER_PARTITION = "transformer"
LIGHTX_TRANSFORMER_REF_PARTITION = "transformer_ref"
LIGHTX_NATIVE_REPRESENTATION = "native_diffusers_peft_split_projections"
LIGHTX_QKV_PROJECTIONS = ("q", "k", "v")

# Native task identity is not present in the safetensors metadata.  These immutable, local
# source identities are therefore an admission allowlist, not task inference from a filename.
# The selected manifest still owns the task; the path and header digest prove that the selected
# manifest is being applied to the intended native artifact rather than a renamed/converter output.
LIGHTX_NATIVE_ARTIFACT_ROOT = Path(
    "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/"
    "minimax-h3-turbo/lightx2v/Minimax-h3-Turbo"
)
LIGHTX_NATIVE_SOURCE_IDENTITIES = {
    "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors": {
        "header_sha256": "415893e26ad2fec16d17ba17e71a0158c024a0211a8a7f37cd6d2d1f3964309a",
    },
    "minimax_h3_fl2v_turbo_4step_v0.1.safetensors": {
        "header_sha256": "7e776ee3ab39492903c88444b31f15701438de78a99ba27429befaf12fb559a7",
    },
    "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors": {
        "header_sha256": "e0819519eba8ebf33378dad706ec318840de024ecc39389a0f8293acc35843e3",
    },
    "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors": {
        "header_sha256": "8852df1d0e3eedd19acd5096b81661b972fa42af2087dea1a5d0b8a769b80723",
    },
}


@dataclass(frozen=True)
class LightX2VManifest:
    """Explicit runtime contract for one supported native LightX2V adapter.

    The adapter filename and safetensors metadata are provenance only.  Values that affect
    scheduling, LoRA math, layout transforms, or cache identity live here so a native adapter
    cannot silently select a different H3 variant from an incomplete header.
    """

    variant_id: str
    task: str
    nfe: int
    video_shift: float
    audio_shift: float
    rank: int
    alpha: float
    runtime_scale_default: float
    effective_alpha_rank_multiplier: float
    representation: str
    artifact_name: str
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    hidden_size: int = 5376
    ffn_hidden_size: int = 14336
    main_block_count: int = 50
    token_refiner_block_count: int = 2
    family: str = LIGHTX_FAMILY
    transformer_partition: str = LIGHTX_TRANSFORMER_PARTITION
    canonical_source_path: str | None = None
    source_header_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.family != LIGHTX_FAMILY:
            raise ValueError(f"unsupported LightX family {self.family!r}")
        if not self.variant_id or not self.task or not self.representation or not self.artifact_name:
            raise ValueError("LightX2V manifest identity fields must be non-empty")
        if self.task not in {LIGHTX_TASK_FL2VA_T2VA, LIGHTX_TASK_REF2VA}:
            raise ValueError(f"unsupported LightX2V task {self.task!r}")
        expected_partition = (
            LIGHTX_TRANSFORMER_REF_PARTITION
            if self.task == LIGHTX_TASK_REF2VA
            else LIGHTX_TRANSFORMER_PARTITION
        )
        if self.transformer_partition != expected_partition:
            raise ValueError(
                f"LightX task {self.task} requires transformer partition {expected_partition!r}, "
                f"got {self.transformer_partition!r}"
            )
        if self.representation != LIGHTX_NATIVE_REPRESENTATION:
            raise ValueError(f"unsupported LightX2V representation {self.representation!r}")
        if self.canonical_source_path is None and self.source_header_sha256 is not None:
            raise ValueError("LightX source header identity requires a canonical source path")
        if self.task == LIGHTX_TASK_REF2VA and (
            self.canonical_source_path is None or self.source_header_sha256 is None
        ):
            raise ValueError("Ref2VA manifests require an explicit canonical source identity")
        if self.canonical_source_path is not None:
            source_path = Path(self.canonical_source_path)
            if not source_path.is_absolute() or source_path.name != self.artifact_name:
                raise ValueError(
                    "LightX canonical source path must be absolute and end with artifact_name"
                )
        if self.source_header_sha256 is not None:
            if not re.fullmatch(r"[0-9a-f]{64}", self.source_header_sha256):
                raise ValueError("LightX source header SHA-256 must be 64 lowercase hex characters")
        if isinstance(self.nfe, bool) or not isinstance(self.nfe, int) or self.nfe < 2:
            raise ValueError(f"LightX2V NFE must be an integer at least 2, got {self.nfe!r}")
        if any(not math.isfinite(float(value)) for value in (self.video_shift, self.audio_shift)):
            raise ValueError("LightX2V scheduler shifts must be finite")
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ValueError(f"LightX2V rank must be positive, got {self.rank!r}")
        if not math.isfinite(float(self.alpha)) or self.alpha <= 0:
            raise ValueError("LightX2V alpha must be finite and positive")
        if not math.isfinite(float(self.runtime_scale_default)):
            raise ValueError("LightX2V runtime scale default must be finite")
        expected_multiplier = float(self.alpha) / float(self.rank)
        if not math.isclose(
            expected_multiplier,
            float(self.effective_alpha_rank_multiplier),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "LightX2V effective alpha/rank multiplier does not match the explicit rank and alpha"
            )
        for name in (
            "num_attention_heads",
            "attention_head_dim",
            "hidden_size",
            "ffn_hidden_size",
            "main_block_count",
            "token_refiner_block_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"LightX2V {name} must be a positive integer, got {value!r}")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def expected_qkv_triplet_count(self) -> int:
        return self.main_block_count + self.token_refiner_block_count

    @property
    def expected_native_pair_count(self) -> int:
        return self.expected_qkv_triplet_count * 6

    @property
    def metadata(self) -> dict[str, Any]:
        """Expose the manifest values to existing schedule/report surfaces."""
        return {
            "lightx_family": self.family,
            "turbo_steps": self.nfe,
            "lightx_variant_id": self.variant_id,
            "lightx_task": self.task,
            "lightx_transformer_partition": self.transformer_partition,
            "lightx_video_shift": float(self.video_shift),
            "lightx_audio_shift": float(self.audio_shift),
            "lightx_rank": self.rank,
            "lightx_alpha": float(self.alpha),
            "lightx_runtime_scale_default": float(self.runtime_scale_default),
            "lightx_effective_alpha_rank_multiplier": float(self.effective_alpha_rank_multiplier),
            "lightx_representation": self.representation,
            "lightx_artifact_name": self.artifact_name,
            "lightx_canonical_source_path": self.canonical_source_path,
            "lightx_source_header_sha256": self.source_header_sha256,
        }

    def identity_payload(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "variant_id": self.variant_id,
            "task": self.task,
            "transformer_partition": self.transformer_partition,
            "nfe": self.nfe,
            "video_shift": float(self.video_shift),
            "audio_shift": float(self.audio_shift),
            "rank": self.rank,
            "alpha": float(self.alpha),
            "runtime_scale_default": float(self.runtime_scale_default),
            "effective_alpha_rank_multiplier": float(self.effective_alpha_rank_multiplier),
            "representation": self.representation,
            "artifact_name": self.artifact_name,
            "num_attention_heads": self.num_attention_heads,
            "attention_head_dim": self.attention_head_dim,
            "hidden_size": self.hidden_size,
            "ffn_hidden_size": self.ffn_hidden_size,
            "main_block_count": self.main_block_count,
            "token_refiner_block_count": self.token_refiner_block_count,
            "canonical_source_path": self.canonical_source_path,
            "source_header_sha256": self.source_header_sha256,
        }

    @property
    def cache_identity(self) -> str:
        encoded = json.dumps(self.identity_payload(), separators=(",", ":"), sort_keys=True).encode()
        return f"lightx2v:{hashlib.sha256(encoded).hexdigest()}"


LIGHTX_FL2VA_TURBO_4STEP_V0_1 = LightX2VManifest(
    variant_id="lightx2v-fl2va-turbo-4step-v0.1",
    task=LIGHTX_TASK_FL2VA_T2VA,
    nfe=4,
    video_shift=12.0,
    audio_shift=3.0,
    rank=128,
    alpha=8.0,
    runtime_scale_default=1.0,
    effective_alpha_rank_multiplier=0.0625,
    representation=LIGHTX_NATIVE_REPRESENTATION,
    artifact_name="minimax_h3_fl2v_turbo_4step_v0.1.safetensors",
    canonical_source_path=str(LIGHTX_NATIVE_ARTIFACT_ROOT / "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"),
    source_header_sha256=LIGHTX_NATIVE_SOURCE_IDENTITIES[
        "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
    ]["header_sha256"],
)

LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P = LightX2VManifest(
    variant_id="lightx2v-fl2va-turbo-4step-v1.0-768p",
    task=LIGHTX_TASK_FL2VA_T2VA,
    nfe=4,
    video_shift=6.0,
    audio_shift=3.0,
    rank=128,
    alpha=128.0,
    runtime_scale_default=1.0,
    effective_alpha_rank_multiplier=1.0,
    representation=LIGHTX_NATIVE_REPRESENTATION,
    artifact_name="minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors",
    canonical_source_path=str(
        LIGHTX_NATIVE_ARTIFACT_ROOT / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    ),
    source_header_sha256=LIGHTX_NATIVE_SOURCE_IDENTITIES[
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    ]["header_sha256"],
)

LIGHTX_FL2VA_TURBO_8STEP_V1_0 = LightX2VManifest(
    variant_id="lightx2v-fl2va-turbo-8step-v1.0",
    task=LIGHTX_TASK_FL2VA_T2VA,
    nfe=8,
    video_shift=12.0,
    audio_shift=3.0,
    rank=128,
    alpha=8.0,
    runtime_scale_default=1.0,
    effective_alpha_rank_multiplier=0.0625,
    representation=LIGHTX_NATIVE_REPRESENTATION,
    artifact_name="minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors",
    canonical_source_path=str(
        LIGHTX_NATIVE_ARTIFACT_ROOT / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    ),
    source_header_sha256=LIGHTX_NATIVE_SOURCE_IDENTITIES[
        "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
    ]["header_sha256"],
)

LIGHTX_REF2VA_TURBO_4STEP_V0_1 = LightX2VManifest(
    variant_id="lightx2v-ref2va-turbo-4step-v0.1",
    task=LIGHTX_TASK_REF2VA,
    nfe=4,
    video_shift=12.0,
    audio_shift=3.0,
    rank=128,
    alpha=8.0,
    runtime_scale_default=1.0,
    effective_alpha_rank_multiplier=0.0625,
    representation=LIGHTX_NATIVE_REPRESENTATION,
    artifact_name="minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors",
    transformer_partition=LIGHTX_TRANSFORMER_REF_PARTITION,
    canonical_source_path=str(
        LIGHTX_NATIVE_ARTIFACT_ROOT / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    ),
    source_header_sha256=LIGHTX_NATIVE_SOURCE_IDENTITIES[
        "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    ]["header_sha256"],
)

LIGHTX_DEFAULT_VARIANT = "fl2va-turbo-8step-v1.0"
LIGHTX_VARIANTS: Mapping[str, LightX2VManifest] = {
    "fl2va-turbo-4step-v0.1": LIGHTX_FL2VA_TURBO_4STEP_V0_1,
    "fl2va-turbo-4step-v1.0-768p": LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P,
    LIGHTX_DEFAULT_VARIANT: LIGHTX_FL2VA_TURBO_8STEP_V1_0,
    "ref2va-turbo-4step-v0.1": LIGHTX_REF2VA_TURBO_4STEP_V0_1,
}

# The shorter aliases keep the manifest discoverable without introducing a second contract.
LightX2VVariant = LightX2VManifest
LIGHTX2V_FL2VA_TURBO_4STEP_V0_1 = LIGHTX_FL2VA_TURBO_4STEP_V0_1
LIGHTX2V_FL2VA_TURBO_4STEP_V1_0_768P = LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P
LIGHTX2V_FL2VA_TURBO_8STEP_V1_0 = LIGHTX_FL2VA_TURBO_8STEP_V1_0
LIGHTX2V_REF2VA_TURBO_4STEP_V0_1 = LIGHTX_REF2VA_TURBO_4STEP_V0_1
LIGHTX_FL2VA_TURBO_4STEP_V0 = LIGHTX_FL2VA_TURBO_4STEP_V0_1
LIGHTX_FL2VA_TURBO_4STEP_V1_0_768 = LIGHTX_FL2VA_TURBO_4STEP_V1_0_768P
LIGHTX_FL2VA_TURBO_8STEP_V1 = LIGHTX_FL2VA_TURBO_8STEP_V1_0


@dataclass(frozen=True)
class LightXQKVOutputPermutation:
    """Map one native Diffusers Q/K/V output into local H3 interleaved QKV rows."""

    projection: str
    num_attention_heads: int
    attention_head_dim: int

    def __post_init__(self) -> None:
        if self.projection not in LIGHTX_QKV_PROJECTIONS:
            raise ValueError(f"unsupported LightX QKV projection {self.projection!r}")
        if self.num_attention_heads <= 0 or self.attention_head_dim <= 0:
            raise ValueError("LightX QKV permutation dimensions must be positive")

    @property
    def identity(self) -> str:
        return (
            f"lightx-qkv-output-interleave:{self.projection}:"
            f"{self.num_attention_heads}x{self.attention_head_dim}"
        )

    def __call__(self, value: Any) -> Any:
        shape = tuple(int(item) for item in getattr(value, "shape", ()))
        inner_dim = self.num_attention_heads * self.attention_head_dim
        if not shape or shape[-1] != inner_dim:
            raise LoRAError(
                f"LightX {self.projection} output permutation expects last dimension {inner_dim}, got {shape}"
            )
        leading = shape[:-1]
        reshaped = value.reshape(leading + (self.num_attention_heads, self.attention_head_dim))
        zero = (
            _mlx().zeros(leading + (self.num_attention_heads, self.attention_head_dim), dtype=value.dtype)
            if _is_mlx_array(value)
            else np.zeros(leading + (self.num_attention_heads, self.attention_head_dim), dtype=np.asarray(value).dtype)
        )
        parts = {
            "q": (reshaped, zero, zero),
            "k": (zero, reshaped, zero),
            "v": (zero, zero, reshaped),
        }[self.projection]
        if _is_mlx_array(value):
            interleaved = _mlx().stack(parts, axis=-2)
        else:
            interleaved = np.stack(parts, axis=-2)
        return interleaved.reshape(leading + (3 * inner_dim,))


@dataclass(frozen=True)
class LightXFC1ValueGateToGateValue:
    """Swap native Diffusers ``[value; gate]`` rows into local H3 ``[gate; value]`` rows."""

    ffn_hidden_size: int

    def __post_init__(self) -> None:
        if self.ffn_hidden_size <= 0:
            raise ValueError("LightX fc1 hidden size must be positive")

    @property
    def identity(self) -> str:
        return f"lightx-fc1-value-gate-to-gate-value:{self.ffn_hidden_size}"

    def __call__(self, value: Any) -> Any:
        shape = tuple(int(item) for item in getattr(value, "shape", ()))
        expected = 2 * self.ffn_hidden_size
        if not shape or shape[-1] != expected:
            raise LoRAError(f"LightX fc1 transform expects last dimension {expected}, got {shape}")
        first, second = value[..., : self.ffn_hidden_size], value[..., self.ffn_hidden_size :]
        if _is_mlx_array(value):
            return _mlx().concatenate([second, first], axis=-1)
        return np.concatenate([second, first], axis=-1)


@dataclass(frozen=True)
class LightXTargetSpec:
    """One native LightX target normalized to a local H3 target and output seam."""

    native_target: str
    local_target: str
    role: str
    output_transform: Callable[[Any], Any] | None = None

    @property
    def transform_identity(self) -> str:
        if self.output_transform is None:
            return "direct"
        return str(getattr(self.output_transform, "identity", type(self.output_transform).__name__))


@dataclass(frozen=True)
class LightXNormalizationReport:
    """Header-level proof summary emitted by the native LightX loader."""

    variant_id: str
    native_tensor_count: int
    native_pair_count: int
    normalized_adapter_count: int
    qkv_triplet_count: int
    adaln_target_count: int
    normalized_targets: tuple[str, ...]


def normalize_lightx_target(
    native_target: str,
    *,
    manifest: LightX2VManifest = LIGHTX_FL2VA_TURBO_8STEP_V1_0,
) -> LightXTargetSpec:
    """Normalize one native Diffusers PEFT target without changing its low-rank orientation."""
    if not isinstance(native_target, str) or not native_target:
        raise LoRAError(f"LightX target must be a non-empty string, got {native_target!r}")
    token_match = re.fullmatch(r"token_refiner\.refiner_blocks\.(\d+)\.(.+)", native_target)
    main_match = re.fullmatch(r"transformer_blocks\.(\d+)\.(.+)", native_target)
    if token_match is not None:
        stack, index, suffix = "token_refiner.blocks", int(token_match.group(1)), token_match.group(2)
        if index >= manifest.token_refiner_block_count:
            raise LoRAError(f"LightX token-refiner block index {index} is outside the manifest")
    elif main_match is not None:
        stack, index, suffix = "blocks", int(main_match.group(1)), main_match.group(2)
        if index >= manifest.main_block_count:
            raise LoRAError(f"LightX main block index {index} is outside the manifest")
    else:
        raise LoRAError(f"unrecognized native LightX target {native_target!r}")

    role_by_suffix = {
        "attn.to_q": "q",
        "attn.to_k": "k",
        "attn.to_v": "v",
        "attn.to_out.0": "out_proj",
        "ff.net.0.proj": "fc1",
        "ff.net.2": "fc2",
    }
    try:
        role = role_by_suffix[suffix]
    except KeyError as exc:
        raise LoRAError(f"unrecognized native LightX projection target {native_target!r}") from exc

    local_prefix = f"{stack}.{index}"
    if role in LIGHTX_QKV_PROJECTIONS:
        return LightXTargetSpec(
            native_target=native_target,
            local_target=f"{local_prefix}.attn.qkv_proj",
            role=role,
            output_transform=LightXQKVOutputPermutation(
                role, manifest.num_attention_heads, manifest.attention_head_dim
            ),
        )
    if role == "out_proj":
        local_target = f"{local_prefix}.attn.out_proj"
    elif role == "fc1":
        local_target = f"{local_prefix}.mlp.fc1"
    else:
        local_target = f"{local_prefix}.mlp.fc2"
    return LightXTargetSpec(
        native_target=native_target,
        local_target=local_target,
        role=role,
        output_transform=(
            LightXFC1ValueGateToGateValue(manifest.ffn_hidden_size) if role == "fc1" else None
        ),
    )


@dataclass(frozen=True)
class LoRATensorSpec:
    """Header-only description of one tensor in a LoRA safetensors file."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def byte_count(self) -> int:
        return self.data_offsets[1] - self.data_offsets[0]


def _decode_safetensors_tensor(raw: bytes, spec: LoRATensorSpec) -> np.ndarray:
    """Decode exactly one safetensors payload without asking NumPy to understand BF16."""
    itemsize = _SAFETENSORS_ITEMSIZE.get(spec.dtype)
    if itemsize is None:
        raise LoRAError(f"unsupported LoRA safetensors dtype {spec.dtype!r} for {spec.name!r}")
    expected_count = int(np.prod(spec.shape, dtype=np.int64)) if spec.shape else 1
    expected_bytes = expected_count * itemsize
    if len(raw) != expected_bytes:
        raise LoRAError(
            f"LoRA tensor {spec.name!r} payload length {len(raw)} does not match header length "
            f"{expected_bytes}"
        )
    if spec.dtype == "BF16":
        # NumPy has no portable BF16 dtype in the supported environment. Decode the 16-bit
        # storage representation to float32 without touching any other tensor payload.
        bits = np.frombuffer(raw, dtype=np.dtype("<u2"), count=expected_count)
        decoded_bits = bits.astype(np.uint32) << np.uint32(16)
        return decoded_bits.view(np.float32).reshape(spec.shape)
    dtype = _SAFETENSORS_NUMPY_DTYPES.get(spec.dtype)
    if dtype is None:
        raise LoRAError(f"unsupported LoRA safetensors dtype {spec.dtype!r} for {spec.name!r}")
    return np.frombuffer(raw, dtype=dtype, count=expected_count).reshape(spec.shape)


class LoRATensorReference:
    """Lazy reference to one named tensor; registration never materializes its payload."""

    def __init__(self, source: "LoRASafetensorsSource", spec: LoRATensorSpec):
        self.source = source
        self.spec = spec

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def dtype(self) -> str:
        return self.spec.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.spec.shape

    @property
    def nbytes(self) -> int:
        return self.spec.byte_count

    @property
    def identity(self) -> str:
        return (
            f"{self.source.current_identity}:{self.spec.name}:{self.spec.dtype}:"
            f"{self.spec.shape}:{self.spec.data_offsets}"
        )

    def materialize(self) -> np.ndarray:
        return self.source.fetch(self.spec.name)


class TransposedLoRATensorReference:
    """Header-preserving transpose view for exporters using ``(rank, output)`` up matrices."""

    def __init__(self, base: LoRATensorReference):
        if len(base.shape) != 2:
            raise LoRAError(f"cannot transpose non-matrix LoRA tensor {base.name!r}")
        self.base = base
        self.source = base.source
        self.name = base.name
        self.dtype = base.dtype
        self.shape = (base.shape[1], base.shape[0])
        self.nbytes = base.nbytes

    @property
    def identity(self) -> str:
        return f"{self.base.identity}:transpose"

    def materialize(self) -> np.ndarray:
        return self.base.materialize().T


class LoRASafetensorsSource:
    """Header-first, named-range safetensors access for generic LoRA adapters.

    The source reads the 8-byte header length and JSON header at construction. Every later fetch
    seeks directly to one tensor's byte range. It deliberately has no whole-file fallback.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise LoRAError(f"could not stat LoRA safetensors {self.path}: {exc}") from exc
        try:
            with self.path.open("rb") as handle:
                raw_length = handle.read(_SAFETENSORS_HEADER_BYTES)
                if len(raw_length) != _SAFETENSORS_HEADER_BYTES:
                    raise LoRAError(f"LoRA safetensors {self.path} has a truncated header length")
                header_length = struct.unpack("<Q", raw_length)[0]
                if header_length <= 0 or header_length > stat.st_size - _SAFETENSORS_HEADER_BYTES:
                    raise LoRAError(f"LoRA safetensors {self.path} has an invalid header length {header_length}")
                header_bytes = handle.read(header_length)
                if len(header_bytes) != header_length:
                    raise LoRAError(f"LoRA safetensors {self.path} has a truncated JSON header")
        except LoRAError:
            raise
        except OSError as exc:
            raise LoRAError(f"could not read LoRA safetensors header {self.path}: {exc}") from exc
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LoRAError(f"LoRA safetensors {self.path} has invalid JSON header") from exc
        if not isinstance(header, dict):
            raise LoRAError(f"LoRA safetensors {self.path} header must be an object")
        payload_size = stat.st_size - _SAFETENSORS_HEADER_BYTES - header_length

        specs: dict[str, LoRATensorSpec] = {}
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(name, str) or not isinstance(entry, dict):
                raise LoRAError(f"LoRA safetensors {self.path} contains a malformed tensor header")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            offsets = entry.get("data_offsets")
            if not isinstance(dtype, str) or not isinstance(shape, list) or not isinstance(offsets, list):
                raise LoRAError(f"LoRA tensor {name!r} has malformed header metadata")
            if dtype not in _SAFETENSORS_ITEMSIZE:
                raise LoRAError(f"unsupported LoRA safetensors dtype {dtype!r} for {name!r}")
            try:
                normalized_shape = tuple(int(value) for value in shape)
                normalized_offsets = tuple(int(value) for value in offsets)
            except (TypeError, ValueError) as exc:
                raise LoRAError(f"LoRA tensor {name!r} has non-integer shape or offsets") from exc
            if any(value < 0 for value in normalized_shape) or len(normalized_offsets) != 2:
                raise LoRAError(f"LoRA tensor {name!r} has invalid shape or offsets")
            if normalized_offsets[0] < 0 or normalized_offsets[1] <= normalized_offsets[0]:
                raise LoRAError(f"LoRA tensor {name!r} has invalid data offsets")
            if normalized_offsets[1] > payload_size:
                raise LoRAError(f"LoRA tensor {name!r} extends beyond the file payload")
            expected_bytes = (int(np.prod(normalized_shape, dtype=np.int64)) if normalized_shape else 1) * _SAFETENSORS_ITEMSIZE[dtype]
            if normalized_offsets[1] - normalized_offsets[0] != expected_bytes:
                raise LoRAError(
                    f"LoRA tensor {name!r} offset length {normalized_offsets[1] - normalized_offsets[0]} "
                    f"does not match shape/dtype byte length {expected_bytes}"
                )
            specs[name] = LoRATensorSpec(name, dtype, normalized_shape, normalized_offsets)
        if not specs:
            raise LoRAError(f"LoRA safetensors {self.path} contains no tensors")
        metadata = header.get("__metadata__", {})
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise LoRAError(f"LoRA safetensors {self.path} metadata must be an object")

        self.header_length = int(header_length)
        self.header_sha256 = hashlib.sha256(header_bytes).hexdigest()
        self._specs = specs
        self._metadata = dict(metadata)
        self.payload_bytes_read = 0
        self.fetch_count = 0
        self._stat_fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        self.identity = self._identity_for_stat(stat)

    def _identity_for_stat(self, stat: Any) -> str:
        return (
            f"safetensors:{self.path.resolve()}:{stat.st_dev}:{stat.st_ino}:"
            f"{stat.st_size}:{stat.st_mtime_ns}:{self.header_sha256}"
        )

    @property
    def current_identity(self) -> str:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise LoRAError(f"LoRA safetensors source is unavailable: {self.path}") from exc
        return self._identity_for_stat(stat)

    def validate_current(self) -> None:
        try:
            stat = self.path.stat()
        except OSError as exc:
            raise LoRAError(f"LoRA safetensors source is unavailable: {self.path}") from exc
        fingerprint = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        if fingerprint != self._stat_fingerprint:
            raise LoRAError(f"LoRA safetensors source identity changed: {self.path}")

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._specs)

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self._metadata

    @property
    def tensor_specs(self) -> Mapping[str, LoRATensorSpec]:
        return self._specs

    def reference(self, name: str) -> LoRATensorReference:
        try:
            return LoRATensorReference(self, self._specs[name])
        except KeyError as exc:
            raise LoRAError(f"LoRA tensor {name!r} is missing from {self.path}") from exc

    def fetch(self, name: str) -> np.ndarray:
        reference = self.reference(name)
        start, end = reference.spec.data_offsets
        absolute_start = _SAFETENSORS_HEADER_BYTES + self.header_length + start
        try:
            self.validate_current()
            with self.path.open("rb") as handle:
                handle.seek(absolute_start)
                raw = handle.read(end - start)
        except LoRAError:
            raise
        except OSError as exc:
            raise LoRAError(f"could not read LoRA tensor {name!r} from {self.path}: {exc}") from exc
        if len(raw) != end - start:
            raise LoRAError(f"LoRA tensor {name!r} payload is truncated in {self.path}")
        self.payload_bytes_read += len(raw)
        self.fetch_count += 1
        return _decode_safetensors_tensor(raw, reference.spec)


def canonical_target(target: str) -> str:
    """Normalize common wrapper prefixes without changing a generic module path."""
    if not isinstance(target, str) or not target.strip():
        raise LoRAError(f"LoRA target must be a non-empty string, got {target!r}")
    value = target.strip()
    for prefix in ("base_model.model.", "base_model.", "model."):
        if value.startswith(prefix):
            value = value[len(prefix):]
            break
    # Diffusers' H3 reference names the main stack `transformer_blocks`; the
    # MLX port calls the same stack `blocks`.
    if value.startswith("transformer."):
        value = value[len("transformer."):]
    value = value.replace(".transformer_blocks.", ".blocks.")
    if value.startswith("transformer_blocks."):
        value = "blocks." + value[len("transformer_blocks."):]
    return value


def _shape(value: Any) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.shape)
    except AttributeError as exc:
        raise LoRAError("LoRA tensors must expose a two-dimensional shape") from exc


def _materialize_tensor(value: Any) -> Any:
    materialize = getattr(value, "materialize", None)
    return materialize() if callable(materialize) else value


def _tensor_dtype(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        raise LoRAError("LoRA tensors must expose a dtype")
    normalized = str(dtype).replace("numpy.", "").upper()
    return {
        "BFLOAT16": "BF16",
        "FLOAT16": "F16",
        "FLOAT32": "F32",
        "FLOAT64": "F64",
    }.get(normalized, normalized)


def _tensor_identity(value: Any) -> str:
    identity = getattr(value, "identity", None)
    if isinstance(identity, str) and identity:
        return identity
    if _is_mlx_array(value):
        return f"mlx:{_shape(value)}:{_tensor_dtype(value)}"
    try:
        array = np.ascontiguousarray(np.asarray(value))
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        return f"numpy:{array.shape}:{array.dtype}:{digest}"
    except (TypeError, ValueError) as exc:
        raise LoRAError("LoRA tensor identity could not be derived") from exc


def _transpose_tensor(value: Any) -> Any:
    if isinstance(value, LoRATensorReference):
        return TransposedLoRATensorReference(value)
    if isinstance(value, TransposedLoRATensorReference):
        return value.base
    return value.T if not _is_mlx_array(value) else _mlx().transpose(value)


def _to_compute_array(value: Any, like: Any) -> Any:
    """Convert an adapter tensor to the backend and dtype of an activation."""
    value = _materialize_tensor(value)
    if _is_mlx_array(like):
        mx = _mlx()
        result = value if _is_mlx_array(value) else mx.array(value)
        return result.astype(like.dtype)
    activation = np.asarray(like)
    return np.asarray(value, dtype=activation.dtype)


@dataclass
class LoRAAdapter:
    """One normalized ``down``/``up`` low-rank pair.

    ``down`` has shape ``(rank, input_features)`` and ``up`` has shape
    ``(output_features, rank)``.  The effective multiplier is
    ``scale * alpha / rank``.
    """

    target: str
    down: Any
    up: Any
    alpha: float
    scale: float = 1.0
    adapter_name: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_identity: str | None = None
    output_transform: Callable[[Any], Any] | None = None
    _prepared: dict[tuple[str, str], tuple[Any, Any]] = field(default_factory=dict, init=False, repr=False)
    _prepare_count: int = field(default=0, init=False, repr=False)
    _identity: tuple[str, str, str] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        target = canonical_target(self.target)
        object.__setattr__(self, "target", target)
        down_shape, up_shape = _shape(self.down), _shape(self.up)
        if len(down_shape) != 2 or len(up_shape) != 2:
            raise LoRAError(
                f"LoRA target {target!r} requires rank-2 tensors, got {down_shape} and {up_shape}"
            )
        if down_shape[0] <= 0 or down_shape[1] <= 0 or up_shape[0] <= 0:
            raise LoRAError(f"LoRA target {target!r} contains an empty tensor")
        if down_shape[0] != up_shape[1]:
            raise LoRAError(
                f"LoRA target {target!r} rank mismatch: down {down_shape}, up {up_shape}"
            )
        if not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0:
            raise LoRAError(f"LoRA target {target!r} alpha must be finite and positive")
        if not math.isfinite(float(self.scale)):
            raise LoRAError(f"LoRA target {target!r} scale must be finite")
        for name, tensor in (("down", self.down), ("up", self.up)):
            dtype = _tensor_dtype(tensor)
            if dtype not in _SAFETENSORS_FLOAT_DTYPES and not (
                not isinstance(tensor, (LoRATensorReference, TransposedLoRATensorReference))
                and np.asarray(tensor).dtype.kind in "fc"
            ):
                raise LoRAError(f"LoRA target {target!r} {name} tensor has unsupported dtype {dtype!r}")
        object.__setattr__(
            self,
            "_identity",
            (_tensor_identity(self.down), _tensor_identity(self.up), self.source_identity or ""),
        )

    @property
    def rank(self) -> int:
        return int(self.down.shape[0])

    @property
    def input_features(self) -> int:
        return int(self.down.shape[1])

    @property
    def output_features(self) -> int:
        return int(self.up.shape[0])

    @property
    def multiplier(self) -> float:
        return float(self.scale) * float(self.alpha) / float(self.rank)

    @property
    def prepared_pair_count(self) -> int:
        return len(self._prepared)

    @property
    def prepare_count(self) -> int:
        return self._prepare_count

    @property
    def runtime_identity(self) -> tuple[Any, ...]:
        down_identity, up_identity, source_identity = self._identity
        if isinstance(self.down, (LoRATensorReference, TransposedLoRATensorReference)):
            down_identity = _tensor_identity(self.down)
        if isinstance(self.up, (LoRATensorReference, TransposedLoRATensorReference)):
            up_identity = _tensor_identity(self.up)
        identity = (
            self.target,
            self.adapter_name,
            (down_identity, up_identity, source_identity),
            self.alpha,
            self.scale,
            self.rank,
            self.input_features,
            self.output_features,
        )
        transform_identity = getattr(self.output_transform, "identity", None)
        if transform_identity is not None:
            identity += (str(transform_identity),)
        return identity

    def _prepare_pair(self, activation: Any, *, cache: bool) -> tuple[Any, Any]:
        if _is_mlx_array(activation):
            key = ("mlx", str(activation.dtype))
        else:
            key = ("numpy", np.asarray(activation).dtype.str)
        if cache and key in self._prepared:
            for tensor in (self.down, self.up):
                source = getattr(tensor, "source", None)
                validate_current = getattr(source, "validate_current", None)
                if callable(validate_current):
                    validate_current()
            return self._prepared[key]
        down = _to_compute_array(self.down, activation)
        up = _to_compute_array(self.up, activation)
        self._prepare_count += 1
        if cache:
            self._prepared[key] = (down, up)
        return down, up

    def prepare(self, activation: Any) -> tuple[Any, Any]:
        """Materialize and convert this adapter once for an activation backend/dtype."""
        return self._prepare_pair(activation, cache=True)

    def clear_prepared(self) -> None:
        """Drop resident backend arrays while retaining header references and metadata."""
        self._prepared.clear()

    def delta(self, activation: Any, *, transient: bool = False) -> Any:
        """Return ``activation @ down.T @ up.T * alpha/rank * scale``."""
        activation_shape = _shape(activation)
        if not activation_shape or activation_shape[-1] != self.input_features:
            raise LoRAError(
                f"LoRA target {self.target!r} expects {self.input_features} input features, "
                f"got activation shape {activation_shape}"
            )
        down, up = self._prepare_pair(activation, cache=not transient)
        if _is_mlx_array(activation):
            mx = _mlx()
            result = mx.matmul(mx.matmul(activation, mx.transpose(down)), mx.transpose(up))
            result = result * np.float32(self.multiplier)
        else:
            result = np.matmul(np.matmul(np.asarray(activation), np.asarray(down).T), np.asarray(up).T)
            result = result * np.float32(self.multiplier)
        if self.output_transform is not None:
            result = self.output_transform(result)
        if transient and _is_mlx_array(result):
            # Streamed AdaLN owns the pair only until this numerical result is evaluated.
            _mlx().eval(result)
        return result


class LoRARegistry:
    """Target-indexed collection of one or more active LoRA adapters."""

    def __init__(
        self,
        adapters: Mapping[str, LoRAAdapter] | None = None,
        *,
        metadata: Mapping[str, Any] | None = None,
        representation_identity: str | None = None,
    ) -> None:
        self._adapters: dict[str, dict[str, LoRAAdapter]] = defaultdict(dict)
        self._aliases: dict[str, str] = {}
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._sources: dict[str, LoRASafetensorsSource] = {}
        self._representation_identity = representation_identity
        self._lightx_manifest: LightX2VManifest | None = None
        self._lightx_report: Any = None
        if adapters:
            for name, adapter in adapters.items():
                self.register(
                    adapter.target,
                    adapter.down,
                    adapter.up,
                    alpha=adapter.alpha,
                    scale=adapter.scale,
                    adapter_name=name,
                    metadata=adapter.metadata,
                    source_identity=adapter.source_identity,
                    output_transform=adapter.output_transform,
                )

    def register(
        self,
        target: str,
        down: Any,
        up: Any,
        *,
        alpha: float | None = None,
        scale: float = 1.0,
        adapter_name: str = "default",
        metadata: Mapping[str, Any] | None = None,
        replace: bool = False,
        source_identity: str | None = None,
        output_transform: Callable[[Any], Any] | None = None,
    ) -> LoRAAdapter:
        """Register a pair using canonical ``(rank, in)`` / ``(out, rank)`` orientation."""
        target = canonical_target(target)
        if not isinstance(adapter_name, str) or not adapter_name:
            raise LoRAError(f"adapter name for {target!r} must be non-empty")
        down_shape, up_shape = _shape(down), _shape(up)
        if len(down_shape) != 2 or len(up_shape) != 2:
            raise LoRAError(f"LoRA target {target!r} requires rank-2 tensors")
        if down_shape[0] != up_shape[1]:
            # A few exporters store the up matrix as (rank, out). It is
            # unambiguous when its first dimension is the down rank.
            if down_shape[0] == up_shape[0]:
                if isinstance(up, (LoRATensorReference, TransposedLoRATensorReference)):
                    up = _transpose_tensor(up)
                else:
                    up = up.T if not _is_mlx_array(up) else _mlx().transpose(up)
                up_shape = _shape(up)
            else:
                raise LoRAError(
                    f"LoRA target {target!r} rank mismatch: down {down_shape}, up {up_shape}"
                )
        if alpha is None:
            alpha = float(down_shape[0])
        adapter = LoRAAdapter(
            target=target,
            down=down,
            up=up,
            alpha=float(alpha),
            scale=float(scale),
            adapter_name=adapter_name,
            metadata=dict(metadata or {}),
            source_identity=source_identity,
            output_transform=output_transform,
        )
        if adapter_name in self._adapters[target] and not replace:
            raise LoRAError(f"duplicate LoRA adapter {adapter_name!r} for target {target!r}")
        self._adapters[target][adapter_name] = adapter
        return adapter

    def _register_source(self, source: LoRASafetensorsSource) -> None:
        self._sources[source.identity] = source

    @property
    def representation_identity(self) -> str | None:
        return self._representation_identity

    def bind_representation_identity(self, identity: str) -> None:
        if not isinstance(identity, str) or not identity:
            raise LoRAError("adapter representation identity must be a non-empty string")
        if self._representation_identity is not None and self._representation_identity != identity:
            raise LoRAError(
                "cannot combine LoRA adapters with different representation identities"
            )
        self._representation_identity = identity

    @property
    def lightx_manifest(self) -> LightX2VManifest | None:
        return self._lightx_manifest

    @property
    def lightx_report(self) -> Any:
        return self._lightx_report

    @property
    def sources(self) -> tuple[LoRASafetensorsSource, ...]:
        return tuple(self._sources.values())

    def register_alias(self, alias: str, target: str) -> None:
        self._aliases[canonical_target(alias)] = canonical_target(target)

    def _candidate_targets(self, target: str) -> tuple[str, ...]:
        canonical = canonical_target(target)
        candidates = [canonical]
        mapped = self._aliases.get(canonical)
        if mapped is not None:
            candidates.append(mapped)
        # Permit an adapter exported with an explicit H3 `transformer` scope
        # to resolve against the local target even if it was registered before
        # canonicalization rules were extended.
        for alias, mapped_target in self._aliases.items():
            if mapped_target == canonical:
                candidates.append(alias)
        return tuple(dict.fromkeys(candidates))

    def adapters_for(self, target: str) -> tuple[LoRAAdapter, ...]:
        result: list[LoRAAdapter] = []
        for candidate in self._candidate_targets(target):
            result.extend(self._adapters.get(candidate, {}).values())
        return tuple(result)

    def has(self, target: str) -> bool:
        return bool(self.adapters_for(target))

    def delta(self, target: str, activation: Any, *, transient: bool = False) -> Any:
        adapters = self.adapters_for(target)
        if not adapters:
            return None
        result = None
        for adapter in adapters:
            value = adapter.delta(activation, transient=transient)
            result = value if result is None else result + value
        if transient and _is_mlx_array(result):
            _mlx().eval(result)
        return result

    def apply(self, target: str, activation: Any, base_output: Any, *, transient: bool = False) -> Any:
        """Add the active adapter delta to a base projection output."""
        delta = self.delta(target, activation, transient=transient)
        if delta is None:
            return base_output
        if _is_mlx_array(base_output):
            delta = delta.astype(base_output.dtype)
        else:
            delta = np.asarray(delta, dtype=np.asarray(base_output).dtype)
        output_shape = _shape(base_output)
        delta_shape = _shape(delta)
        if output_shape != delta_shape:
            raise LoRAError(
                f"LoRA target {canonical_target(target)!r} output shape mismatch: "
                f"base {output_shape}, delta {delta_shape}"
            )
        return base_output + delta

    def set_scale(self, scale: float, *, adapter_name: str | None = None) -> None:
        """Set an active multiplier while preserving each adapter's alpha/rank."""
        if not math.isfinite(float(scale)):
            raise LoRAError("LoRA scale must be finite")
        for target, adapters in list(self._adapters.items()):
            for name, adapter in list(adapters.items()):
                if adapter_name is None or name == adapter_name:
                    adapter.scale = float(scale)

    def prepare_target(self, target: str, activation: Any) -> int:
        """Prepare all adapters for one target and return the number of newly cached pairs."""
        prepared = 0
        for adapter in self.adapters_for(target):
            before = adapter.prepared_pair_count
            adapter.prepare(activation)
            prepared += adapter.prepared_pair_count - before
        return prepared

    def prepare_resident(self, exemplars: Mapping[str, Any] | Any) -> tuple[str, ...]:
        """Prepare known resident targets from activation exemplars.

        ``exemplars`` may map target names to representative activations. A single exemplar is
        also accepted for small generic registries. Block AdaLN targets are deliberately omitted;
        their callers use ``transient=True`` while building streamed modulation tables.
        """
        prepared_targets: list[str] = []
        for target in self.targets:
            if is_streamed_adaln_target(target):
                continue
            if isinstance(exemplars, Mapping):
                exemplar = exemplars.get(target)
                if exemplar is None:
                    continue
            else:
                exemplar = exemplars
            self.prepare_target(target, exemplar)
            prepared_targets.append(target)
        return tuple(prepared_targets)

    @property
    def cache_identity(self) -> str | None:
        """Stable identity of adapter sources, stack order, and strengths for cache validation."""
        if not self._adapters:
            return None
        entries: list[Any] = []
        for target in sorted(self._adapters):
            # Preserve insertion order within each target: stacked adapter ordering is part of the
            # runtime contract even though ordinary real-valued addition is usually commutative.
            for adapter in self._adapters[target].values():
                entries.append(adapter.runtime_identity)
        identity_payload: Any = entries
        if self._representation_identity is not None:
            identity_payload = {
                "representation_identity": self._representation_identity,
                "adapters": entries,
            }
        encoded = json.dumps(identity_payload, separators=(",", ":"), sort_keys=False, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def topology_counts(self) -> dict[str, int]:
        counts = {
            "total": self.adapter_count,
            "resident_core": 0,
            "streamed_block_adaln": 0,
            "resident_final_adaln": 0,
            "other": 0,
        }
        for target in self.targets:
            count = len(self._adapters[target])
            if is_core_projection_target(target):
                counts["resident_core"] += count
            elif is_streamed_adaln_target(target):
                counts["streamed_block_adaln"] += count
            elif is_final_adaln_target(target):
                counts["resident_final_adaln"] += count
            else:
                counts["other"] += count
        return counts

    @property
    def targets(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    @property
    def adapter_count(self) -> int:
        return sum(len(items) for items in self._adapters.values())

    @property
    def turbo_steps(self) -> int | None:
        for key in ("turbo_steps", "num_inference_steps", "steps"):
            value = self.metadata.get(key)
            if value is None:
                continue
            try:
                value = int(value)
            except (TypeError, ValueError):
                continue
            if value >= 2:
                return value
        return None


def linear_with_lora(
    layer: Callable[[Any], Any],
    activation: Any,
    target: str,
    registry: LoRARegistry | None = None,
    *,
    transient: bool = False,
) -> Any:
    """Call one projection and add a registry delta when its target is active.

    This helper is intentionally layer-agnostic.  MLX ``QuantizedLinear``
    instances and small NumPy test doubles both satisfy the same callable
    contract.
    """
    output = layer(activation)
    if registry is None or not registry.has(target):
        return output
    return registry.apply(target, activation, output, transient=transient)


def apply_lora(
    layer: Callable[[Any], Any],
    activation: Any,
    *,
    target: str,
    registry: LoRARegistry | None,
    transient: bool = False,
) -> Any:
    """Named wrapper for callers that prefer an explicit adapter operation."""
    return linear_with_lora(layer, activation, target, registry, transient=transient)


_BLOCK_ADALN_TARGET = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear$")
_CORE_PROJECTION_TARGET = re.compile(
    r"^(?:blocks|token_refiner\.blocks)\.\d+\.(?:attn\.(?:qkv_proj|out_proj)|mlp\.(?:fc1|fc2))$"
)


def is_streamed_adaln_target(target: str) -> bool:
    match = _BLOCK_ADALN_TARGET.fullmatch(canonical_target(target))
    return match is not None and 0 <= int(match.group(1)) < 50


def is_final_adaln_target(target: str) -> bool:
    return canonical_target(target) == "final_layer.adaln_proj.linear"


def is_core_projection_target(target: str) -> bool:
    return _CORE_PROJECTION_TARGET.fullmatch(canonical_target(target)) is not None


_PAIR_MARKERS = {"lora_A", "lora_B", "lora_down", "lora_up"}
_DOWN_MARKERS = {"lora_A", "lora_down"}
_UP_MARKERS = {"lora_B", "lora_up"}
_ALPHA_MARKERS = {"alpha", "lora_alpha"}


def _parse_lora_key(key: str) -> tuple[str, str, str]:
    parts = key.split(".")
    for index, marker in enumerate(parts):
        if marker in _PAIR_MARKERS:
            suffix = parts[index + 1:]
            adapter_name = "default"
            if suffix == ["weight"]:
                pass
            elif len(suffix) == 2 and suffix[1] == "weight":
                adapter_name = suffix[0]
            elif suffix:
                raise LoRAError(f"unsupported LoRA tensor suffix in key {key!r}")
            target = ".".join(parts[:index])
            return canonical_target(target), marker, adapter_name
    for index, marker in enumerate(parts):
        if marker in _ALPHA_MARKERS:
            suffix = parts[index + 1:]
            adapter_name = "default"
            if suffix:
                if len(suffix) != 1:
                    raise LoRAError(f"unsupported LoRA alpha suffix in key {key!r}")
                adapter_name = suffix[0]
            target = ".".join(parts[:index])
            return canonical_target(target), marker, adapter_name
    raise LoRAError(f"unrecognized LoRA tensor key {key!r}")


def _scalar(value: Any, key: str) -> float:
    raw = value.tolist() if hasattr(value, "tolist") else value
    array = np.asarray(raw).reshape(-1)
    if array.size != 1:
        raise LoRAError(f"LoRA alpha {key!r} must be scalar, got shape {array.shape}")
    result = float(array[0])
    if not math.isfinite(result) or result <= 0:
        raise LoRAError(f"LoRA alpha {key!r} must be finite and positive")
    return result


def _read_safetensors(path: Path) -> LoRASafetensorsSource:
    """Open only the safetensors header; named payloads are fetched by the source later."""
    return LoRASafetensorsSource(path)


def _validate_lightx_source_identity(
    source: LoRASafetensorsSource,
    manifest: LightX2VManifest,
) -> None:
    """Require the manifest's explicit native source identity before topology admission.

    Native Ref2VA and FL2VA headers intentionally share the same tensor inventory.  A canonical
    path plus the immutable header digest is therefore checked before any target/shape validation;
    this is provenance admission, not task inference from a filename or tensor topology.
    """
    expected_path = manifest.canonical_source_path
    expected_header = manifest.source_header_sha256
    if expected_path is None:
        return
    actual_path = source.path.resolve()
    if actual_path != Path(expected_path).resolve():
        raise LoRAError(
            f"native LightX source identity mismatch for {manifest.variant_id}: "
            f"expected {Path(expected_path).resolve()}, got {actual_path}"
        )
    if source.path.name != manifest.artifact_name:
        raise LoRAError(
            f"native LightX artifact name mismatch for {manifest.variant_id}: "
            f"expected {manifest.artifact_name!r}, got {source.path.name!r}"
        )
    if expected_header is not None and source.header_sha256 != expected_header:
        raise LoRAError(
            f"native LightX source header identity mismatch for {manifest.variant_id}: "
            f"expected {expected_header}, got {source.header_sha256}"
        )


_LIGHTX_KEY_PATTERN = re.compile(r"^(?P<target>.+)\.(?P<marker>lora_[AB])\.default\.weight$")
_LIGHTX_ROLE_SUFFIXES = {
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
}


def _looks_like_native_lightx_key(key: str) -> bool:
    """Recognize native Diffusers LightX naming without assigning its task."""
    if not isinstance(key, str):
        return False
    match = re.fullmatch(r"(?P<target>.+)\.(?P<marker>lora_[AB])(?:\.default)?\.weight", key)
    if match is None:
        return False
    target = match.group("target")
    return any(target.endswith(f".{suffix}") for suffix in _LIGHTX_ROLE_SUFFIXES)


def _lightx_expected_native_targets(manifest: LightX2VManifest) -> tuple[str, ...]:
    targets: list[str] = []
    for stack, count in (
        ("token_refiner.refiner_blocks", manifest.token_refiner_block_count),
        ("transformer_blocks", manifest.main_block_count),
    ):
        for index in range(count):
            prefix = f"{stack}.{index}"
            targets.extend(f"{prefix}.{suffix}" for suffix in sorted(_LIGHTX_ROLE_SUFFIXES))
    return tuple(targets)


def _lightx_expected_pair_shapes(
    spec: LightXTargetSpec,
    manifest: LightX2VManifest,
) -> tuple[tuple[int, int], tuple[int, int]]:
    if spec.role in LIGHTX_QKV_PROJECTIONS:
        return (manifest.rank, manifest.hidden_size), (manifest.inner_dim, manifest.rank)
    if spec.role == "out_proj":
        return (manifest.rank, manifest.inner_dim), (manifest.hidden_size, manifest.rank)
    if spec.role == "fc1":
        return (manifest.rank, manifest.hidden_size), (2 * manifest.ffn_hidden_size, manifest.rank)
    if spec.role == "fc2":
        return (manifest.rank, manifest.ffn_hidden_size), (manifest.hidden_size, manifest.rank)
    raise LoRAError(f"unsupported LightX normalized role {spec.role!r}")


def load_lightx_safetensors(
    path: str | Path,
    *,
    variant: LightX2VManifest = LIGHTX_FL2VA_TURBO_8STEP_V1_0,
    registry: LoRARegistry | None = None,
    scale: float | None = None,
    strict: bool = True,
) -> LoRARegistry:
    """Header-validate and lazily normalize one native Diffusers LightX2V adapter.

    Registration reads no tensor payload.  The native split Q/K/V and rank-128 shapes remain in
    the registry; only the output delta is reshaped/permuted when a local fused projection is
    called.  The explicit manifest is required so no filename or partial metadata can choose the
    task, schedule, scale, or representation.
    """
    if not isinstance(variant, LightX2VManifest):
        raise LoRAError(f"expected LightX2VManifest, got {type(variant).__name__}")
    source = _read_safetensors(Path(path))
    _validate_lightx_source_identity(source, variant)
    expected_targets = set(_lightx_expected_native_targets(variant))
    pairs: dict[str, dict[str, LoRATensorReference]] = defaultdict(dict)
    specs: dict[str, LightXTargetSpec] = {}
    unrecognized: list[str] = []

    for key in source.keys:
        match = _LIGHTX_KEY_PATTERN.fullmatch(key)
        if match is None:
            unrecognized.append(key)
            continue
        native_target = match.group("target")
        try:
            spec = normalize_lightx_target(native_target, manifest=variant)
        except LoRAError:
            unrecognized.append(key)
            continue
        if spec.native_target in pairs and match.group("marker") in pairs[spec.native_target]:
            raise LoRAError(f"duplicate native LightX tensor {key!r}")
        specs[native_target] = spec
        pairs[native_target][match.group("marker")] = source.reference(key)

    if unrecognized and strict:
        raise LoRAError(f"unrecognized native LightX tensor keys: {unrecognized[:4]}")
    actual_targets = set(pairs)
    missing = sorted(expected_targets - actual_targets)
    unexpected = sorted(actual_targets - expected_targets)
    if missing or unexpected:
        raise LoRAError(
            f"native LightX target inventory mismatch: missing={missing[:4]}, unexpected={unexpected[:4]}"
        )
    if len(pairs) != variant.expected_native_pair_count:
        raise LoRAError(
            f"native LightX pair count {len(pairs)} does not match manifest count "
            f"{variant.expected_native_pair_count}"
        )

    for native_target, spec in specs.items():
        pair = pairs[native_target]
        if set(pair) != {"lora_A", "lora_B"}:
            raise LoRAError(f"native LightX target {native_target!r} must have exactly one A/B pair")
        expected_down, expected_up = _lightx_expected_pair_shapes(spec, variant)
        if pair["lora_A"].shape != expected_down or pair["lora_B"].shape != expected_up:
            raise LoRAError(
                f"native LightX target {native_target!r} has shapes "
                f"{pair['lora_A'].shape}/{pair['lora_B'].shape}, expected {expected_down}/{expected_up}"
            )

    qkv_roles: dict[str, set[str]] = defaultdict(set)
    normalized_targets: set[str] = set()
    for spec in specs.values():
        normalized_targets.add(spec.local_target)
        if spec.role in LIGHTX_QKV_PROJECTIONS:
            qkv_roles[spec.local_target].add(spec.role)
    incomplete_triplets = sorted(
        target for target, roles in qkv_roles.items() if roles != set(LIGHTX_QKV_PROJECTIONS)
    )
    if incomplete_triplets or len(qkv_roles) != variant.expected_qkv_triplet_count:
        raise LoRAError(
            f"native LightX Q/K/V triplet inventory is incomplete: "
            f"count={len(qkv_roles)}, incomplete={incomplete_triplets[:4]}"
        )

    metadata_alpha = None
    for key in ("alpha", "lora_alpha"):
        if key in source.metadata:
            metadata_alpha = _scalar(source.metadata[key], f"metadata.{key}")
            break
    if metadata_alpha is not None and not math.isclose(
        metadata_alpha, float(variant.alpha), rel_tol=0.0, abs_tol=1e-12
    ):
        raise LoRAError(
            f"native LightX metadata alpha {metadata_alpha} disagrees with manifest alpha {variant.alpha}"
        )
    effective_scale = variant.runtime_scale_default if scale is None else float(scale)
    if not math.isfinite(float(effective_scale)):
        raise LoRAError("native LightX runtime scale must be finite")

    registrations: list[tuple[LightXTargetSpec, str, LoRATensorReference, LoRATensorReference]] = []
    for native_target in sorted(specs):
        spec = specs[native_target]
        pair = pairs[native_target]
        registrations.append((spec, native_target, pair["lora_A"], pair["lora_B"]))

    result = registry or LoRARegistry()
    result.bind_representation_identity(variant.cache_identity)
    result._register_source(source)
    result.metadata.update(variant.metadata)
    for spec, native_target, down, up in registrations:
        result.register(
            spec.local_target,
            down,
            up,
            alpha=variant.alpha,
            scale=effective_scale,
            adapter_name=f"{variant.variant_id}:{spec.role}",
            metadata={
                **variant.metadata,
                "lightx_native_target": native_target,
                "lightx_role": spec.role,
                "lightx_transform": spec.transform_identity,
            },
            source_identity=source.identity,
            output_transform=spec.output_transform,
        )

    report = LightXNormalizationReport(
        variant_id=variant.variant_id,
        native_tensor_count=len(source.keys),
        native_pair_count=len(registrations),
        normalized_adapter_count=len(registrations),
        qkv_triplet_count=len(qkv_roles),
        adaln_target_count=sum(
            1 for spec, _native_target, _down, _up in registrations if is_streamed_adaln_target(spec.local_target)
        ),
        normalized_targets=tuple(sorted(normalized_targets)),
    )
    result._lightx_manifest = variant
    result._lightx_report = report
    return result


def load_lora_safetensors(
    path: str | Path,
    *,
    registry: LoRARegistry | None = None,
    tensor_loader: Callable[[str], Mapping[str, Any]] | None = None,
    adapter_name: str | None = None,
    scale: float | None = None,
    strict: bool = True,
    variant: LightX2VManifest | None = None,
) -> LoRARegistry:
    """Load a generic LoRA ``.safetensors`` payload into a registry.

    Supported pair names are ``lora_A``/``lora_B`` and
    ``lora_down``/``lora_up`` with an optional ``.weight`` and optional adapter
    name component. Unknown tensor keys fail in strict mode so a partially
    loaded adapter cannot be mistaken for a complete one.
    """
    if variant is not None:
        if tensor_loader is not None:
            raise LoRAError("native LightX normalization does not support a custom tensor loader")
        return load_lightx_safetensors(
            path,
            variant=variant,
            registry=registry,
            scale=scale,
            strict=strict,
        )

    path = Path(path)
    if tensor_loader is None:
        source = _read_safetensors(path)
        payload: Mapping[str, Any] = {key: source.reference(key) for key in source.keys}
        metadata = dict(source.metadata)
    else:
        loaded = tensor_loader(str(path))
        if not isinstance(loaded, Mapping):
            raise LoRAError(f"custom LoRA tensor loader returned {type(loaded).__name__}, expected a mapping")
        payload, metadata = dict(loaded), {}

    if variant is None and any(_looks_like_native_lightx_key(key) for key in payload):
        raise LoRAError(
            "native LightX adapter requires an explicit LightX2V variant manifest; "
            "generic LoRA loading cannot assign its task or representation"
        )

    result = registry or LoRARegistry()
    if tensor_loader is None:
        result._register_source(source)
    result.metadata.update(metadata)
    generic_scale = 1.0 if scale is None else float(scale)
    pairs: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    unrecognized: list[str] = []
    for key, tensor in payload.items():
        if not isinstance(key, str):
            raise LoRAError(f"LoRA tensor keys must be strings, got {key!r}")
        try:
            target, marker, named_adapter = _parse_lora_key(key)
        except LoRAError:
            unrecognized.append(key)
            continue
        if adapter_name is not None and named_adapter != adapter_name:
            continue
        pair = pairs[(target, named_adapter)]
        if marker in pair:
            raise LoRAError(f"duplicate LoRA tensor {key!r}")
        pair[marker] = tensor

    if strict and unrecognized:
        raise LoRAError(f"unrecognized LoRA tensor keys: {unrecognized[:4]}")
    if not pairs:
        raise LoRAError(f"LoRA payload {path} contains no adapter pairs")

    global_alpha = None
    for key in ("lora_alpha", "alpha"):
        if key in metadata:
            global_alpha = _scalar(metadata[key], f"metadata.{key}")
            break

    for (target, named_adapter), pair in sorted(pairs.items()):
        down_keys = _DOWN_MARKERS.intersection(pair)
        up_keys = _UP_MARKERS.intersection(pair)
        if len(down_keys) != 1 or len(up_keys) != 1:
            if strict:
                raise LoRAError(
                    f"LoRA target {target!r} adapter {named_adapter!r} must contain exactly one down and one up tensor"
                )
            continue
        alpha = global_alpha
        for alpha_key in _ALPHA_MARKERS.intersection(pair):
            alpha = _scalar(_materialize_tensor(pair[alpha_key]), f"{target}.{alpha_key}")
        result.register(
            target,
            pair[next(iter(down_keys))],
            pair[next(iter(up_keys))],
            alpha=alpha,
            scale=generic_scale,
            adapter_name=named_adapter,
            metadata=metadata,
            source_identity=(source.identity if tensor_loader is None else None),
        )
    return result


def load_lora_payload(
    payload: Mapping[str, Any],
    *,
    registry: LoRARegistry | None = None,
    adapter_name: str | None = None,
    scale: float = 1.0,
    strict: bool = True,
) -> LoRARegistry:
    """MLX-free convenience loader used by synthetic and format contract tests."""
    return load_lora_safetensors(
        "<in-memory>",
        registry=registry,
        tensor_loader=lambda _path: payload,
        adapter_name=adapter_name,
        scale=scale,
        strict=strict,
    )

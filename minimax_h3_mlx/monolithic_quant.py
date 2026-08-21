"""Bounded-memory monolithic BF16 -> conventional MLX Q6/Q8 conversion.

The planner is MLX-free and proves the full H3 topology before any payload is materialized.  The
default quantizer imports MLX lazily, creates one temporary ``nn.Linear`` for one source weight,
uses ``nn.QuantizedLinear.from_linear`` as the sole packing authority, and releases that result
before moving to the next logical linear.  Output shards are written with the shared raw
safetensors streaming writer rather than an in-memory parameter dictionary.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping, Protocol

import numpy as np

from .checkpoint_forge.tensor_io import (
    DTYPE_BYTES,
    TensorHeader,
    copy_range,
    read_safetensors_header,
    write_safetensors_stream,
)
from .monolithic_source import (
    MonolithicSafetensorsSource,
    MonolithicSourceError,
    QKV_BIAS_SUFFIX,
    QKV_SOURCE_LAYOUT_GROUPED,
    QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED,
    QKV_WEIGHT_SUFFIX,
    QKVLayoutDecision,
    SourceClassification,
    SourceStaleError,
    TensorClassification,
    classify_expected_source_config,
    classify_source,
    decode_bfloat16_to_float32,
    reconcile_qkv_rows,
)


MAX_SHARD_BYTES = 5 * 1024**3
OUTPUT_RECIPE = {
    "bits": 6,
    "group_size": 64,
    "quantize_adaln": True,
    "adaln_bits": 8,
}
QUANTIZED_ROLES = frozenset({"q6_core_weight", "q8_block_adaln_weight"})


@dataclass(frozen=True)
class OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    source_name: str | None = None
    quantized_parent: str | None = None
    quantized_role: str | None = None

    @property
    def nbytes(self) -> int:
        count = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        return count * DTYPE_BYTES[self.dtype]


@dataclass(frozen=True)
class OutputShard:
    filename: str
    tensors: tuple[OutputTensor, ...]

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors)


@dataclass(frozen=True)
class ConversionPlan:
    source: MonolithicSafetensorsSource
    classification: SourceClassification
    output_tensors: tuple[OutputTensor, ...]
    selected_quantized_weights: tuple[str, ...]
    bounded: bool

    @property
    def config(self):
        return self.classification.config

    @property
    def config_raw(self) -> Mapping[str, object]:
        return self.classification.config_raw

    @property
    def source_tensor_count(self) -> int:
        return len(self.classification.tensors)

    @property
    def stored_source_tensor_count(self) -> int:
        return self.source_tensor_count - self.classification.counts["recomputed"]

    @property
    def output_tensor_count(self) -> int:
        return len(self.output_tensors)

    @property
    def quantized_counts(self) -> dict[str, int]:
        return _quantized_counts(self.classification, self.selected_quantized_weights)

    @property
    def quant_config(self) -> dict[str, object]:
        return {**OUTPUT_RECIPE, "quantized_layers": self.quantized_counts}

    @property
    def qkv_tensors_planned(self) -> int:
        if not self.classification.qkv_layout.row_reconciliation_required:
            return 0
        return sum(name.endswith(QKV_WEIGHT_SUFFIX) for name in self.selected_quantized_weights)

    def shards(self, target_bytes: int = MAX_SHARD_BYTES) -> tuple[OutputShard, ...]:
        if target_bytes <= 0:
            raise ValueError("target shard bytes must be positive")
        groups: list[list[OutputTensor]] = [[]]
        sizes = [0]
        for tensor in self.output_tensors:
            if groups[-1] and sizes[-1] + tensor.nbytes > target_bytes:
                groups.append([])
                sizes.append(0)
            groups[-1].append(tensor)
            sizes[-1] += tensor.nbytes
        width = max(5, len(str(len(groups))))
        total = len(groups)
        return tuple(
            OutputShard(
                f"model-{index:0{width}d}-of-{total:0{width}d}.safetensors",
                tuple(group),
            )
            for index, group in enumerate(groups, 1)
        )


def quantized_tensor_shapes(
    source_descriptor: TensorHeader,
    *,
    bits: int,
    group_size: int,
) -> dict[str, tuple[int, ...]]:
    """Return the MLX affine output shapes for one logical ``nn.Linear`` weight."""
    if len(source_descriptor.shape) != 2:
        raise MonolithicSourceError(
            f"quantized source weight {source_descriptor.name} must be rank 2; "
            f"got {source_descriptor.shape}"
        )
    output_features, input_features = source_descriptor.shape
    if input_features % group_size:
        raise MonolithicSourceError(
            f"quantized source weight {source_descriptor.name} input features {input_features} "
            f"are not divisible by group size {group_size}"
        )
    packed_numerator = input_features * bits
    if packed_numerator % 32:
        raise MonolithicSourceError(
            f"quantized source weight {source_descriptor.name} cannot pack {bits}-bit values "
            f"from input width {input_features} into U32"
        )
    return {
        "weight": (output_features, packed_numerator // 32),
        "scales": (output_features, input_features // group_size),
        "biases": (output_features, input_features // group_size),
    }


def _quantized_output_entries(item: TensorClassification) -> tuple[OutputTensor, ...]:
    if item.bits is None:
        raise MonolithicSourceError(f"quantized classification has no bit width: {item.name}")
    shapes = quantized_tensor_shapes(item.descriptor, bits=item.bits, group_size=64)
    parent = item.name
    return tuple(
        OutputTensor(
            f"{parent[:-len('.weight')]}.{role}",
            "U32" if role == "weight" else "BF16",
            shape,
            quantized_parent=parent,
            quantized_role=role,
        )
        for role, shape in shapes.items()
    )


def _quantized_counts(
    classification: SourceClassification,
    selected_quantized_weights: tuple[str, ...],
) -> dict[str, int]:
    counts = {"6": 0, "8": 0}
    by_name = classification.by_name
    for name in selected_quantized_weights:
        counts[str(by_name[name].bits)] += 1
    return counts


def _build_output_topology(
    classification: SourceClassification,
    selected_quantized_weights: tuple[str, ...] | list[str] | None,
) -> tuple[tuple[OutputTensor, ...], tuple[str, ...], bool]:
    """Derive exact full or selected output descriptors from admitted classification only."""
    all_quantized = tuple(item.name for item in classification.quantized_weights)
    if selected_quantized_weights is None:
        selected = all_quantized
        bounded = False
    else:
        supplied = tuple(selected_quantized_weights)
        if not supplied:
            raise MonolithicSourceError("bounded conversion selector cannot be empty")
        if not all(isinstance(name, str) for name in supplied):
            raise MonolithicSourceError("bounded conversion selectors must be tensor-name strings")
        seen: set[str] = set()
        duplicates: set[str] = set()
        for name in supplied:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        if duplicates:
            raise MonolithicSourceError(
                f"duplicate bounded conversion selector(s): {sorted(duplicates)}"
            )
        selected = tuple(sorted(supplied))
        by_name = classification.by_name
        unknown = sorted(set(selected) - set(by_name))
        if unknown:
            raise MonolithicSourceError(f"bounded selector names missing from source: {unknown}")
        invalid = sorted(
            name for name in selected if by_name[name].role not in QUANTIZED_ROLES
        )
        if invalid:
            raise MonolithicSourceError(
                "bounded selector must name logical quantized weights, not ordinary or recomputed "
                f"tensors: {invalid}"
            )
        bounded = set(selected) != set(all_quantized)

    selected_set = set(selected)
    entries: list[OutputTensor] = []
    for item in classification.tensors:
        if item.role == "recomputed":
            continue
        if item.role in QUANTIZED_ROLES:
            if item.name in selected_set:
                entries.extend(_quantized_output_entries(item))
            continue
        if not bounded:
            entries.append(
                OutputTensor(
                    item.name,
                    item.descriptor.dtype,
                    item.descriptor.shape,
                    source_name=item.name,
                    quantized_parent=(
                        f"{item.module_path}.weight"
                        if item.role == "learned_bias" and item.module_path
                        else None
                    ),
                    quantized_role="bias" if item.role == "learned_bias" else None,
                )
            )
            continue
        if item.role == "learned_bias":
            parent = f"{item.module_path}.weight" if item.module_path else None
            if parent in selected_set:
                entries.append(OutputTensor(item.name, item.descriptor.dtype, item.descriptor.shape, source_name=item.name, quantized_parent=parent, quantized_role="bias"))

    entries.sort(key=lambda item: item.name)
    if not entries:
        raise MonolithicSourceError("conversion plan contains no output tensors")
    names = [item.name for item in entries]
    if len(names) != len(set(names)):
        raise MonolithicSourceError("conversion plan contains duplicate output tensor names")
    expected_count = 1050 if not bounded else len(entries)
    if not bounded and len(entries) != expected_count:
        raise MonolithicSourceError(
            f"complete output tensor arithmetic failed: got {len(entries)}, expected {expected_count}"
        )
    return tuple(entries), tuple(sorted(selected)), bounded


def build_conversion_plan(
    source: MonolithicSafetensorsSource,
    *,
    selected_quantized_weights: tuple[str, ...] | list[str] | None = None,
) -> ConversionPlan:
    """Build a deterministic full or bounded output plan without reading any payload bytes."""
    classification = classify_source(source)
    entries, selected, bounded = _build_output_topology(
        classification,
        selected_quantized_weights,
    )
    return ConversionPlan(source, classification, entries, selected, bounded)


class QuantizedArray(Protocol):
    dtype: str
    shape: tuple[int, ...]

    def write_to(self, destination: BinaryIO, temp_dir: Path) -> str:
        ...


@dataclass(frozen=True)
class BytesQuantizedArray:
    """Small test/offline quantized-array implementation for the streaming writer seam."""

    dtype: str
    shape: tuple[int, ...]
    payload: bytes

    def write_to(self, destination: BinaryIO, temp_dir: Path) -> str:
        expected = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        expected *= DTYPE_BYTES[self.dtype]
        if len(self.payload) != expected:
            raise MonolithicSourceError(
                f"synthetic quantized payload has {len(self.payload)} bytes; expected {expected}"
            )
        destination.write(self.payload)
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class QuantizedResult:
    arrays: Mapping[str, QuantizedArray]


class QKVReconciliationExecution:
    """Actual per-invocation QKV transformations that completed before quantization."""

    def __init__(self) -> None:
        self.reconciled_tensor_names: set[str] = set()

    def record(self, tensor_name: str) -> None:
        self.reconciled_tensor_names.add(tensor_name)

    @property
    def qkv_tensors_reconciled(self) -> int:
        return len(self.reconciled_tensor_names)

    @property
    def qkv_row_reconciliation_applied(self) -> bool:
        return bool(self.reconciled_tensor_names)


class Quantizer(Protocol):
    def quantize(self, parent: str, bias_bytes: bytes | None = None) -> QuantizedResult:
        ...

    def release(self, result: QuantizedResult) -> None:
        ...


class MlxArrayPayload:
    """One MLX array serialized through a one-tensor temporary safetensors file."""

    def __init__(self, value, mx_module, dtype: str, shape: tuple[int, ...]):
        self.value = value
        self.mx = mx_module
        self.dtype = dtype
        self.shape = shape

    def write_to(self, destination: BinaryIO, temp_dir: Path) -> str:
        temp_dir.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix="mlx-array-", suffix=".safetensors", dir=temp_dir)
        os.close(fd)
        temporary = Path(temporary_name)
        try:
            self.mx.save_safetensors(
                str(temporary),
                {"value": self.value},
                metadata={"format": "mlx"},
            )
            header = read_safetensors_header(temporary)
            if set(header.tensor_map()) != {"value"}:
                raise MonolithicSourceError("MLX one-array serialization produced unexpected tensor keys")
            descriptor = header.tensor_map()["value"]
            if descriptor.dtype != self.dtype or descriptor.shape != self.shape:
                raise MonolithicSourceError(
                    f"MLX quantized output descriptor mismatch: got dtype={descriptor.dtype}, shape={descriptor.shape}; "
                    f"expected dtype={self.dtype}, shape={self.shape}"
                )
            with temporary.open("rb") as source_file:
                return copy_range(
                    source_file,
                    destination,
                    header.data_start + descriptor.start,
                    header.data_start + descriptor.end,
                )
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _decode_source_array(raw: bytes, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if dtype == "BF16":
        return decode_bfloat16_to_float32(raw, shape)
    if dtype == "F32":
        expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if len(raw) != expected * DTYPE_BYTES["F32"]:
            raise MonolithicSourceError(
                f"F32 payload has {len(raw)} bytes; expected {expected * DTYPE_BYTES['F32']}"
            )
        return np.frombuffer(raw, dtype=np.dtype("<f4"), count=expected).reshape(shape)
    raise MonolithicSourceError(f"cannot decode source dtype for MLX quantization: {dtype}")


def prepare_source_array_for_quantization(
    raw: bytes,
    dtype: str,
    shape: tuple[int, ...],
    *,
    tensor_name: str,
    qkv_layout: QKVLayoutDecision | None,
    num_attention_heads: int,
    attention_head_dim: int,
    reconciliation_execution: QKVReconciliationExecution | None = None,
) -> np.ndarray:
    """Decode one source tensor and reconcile only fused QKV rows before MLX sees the weight."""
    decoded = _decode_source_array(raw, dtype, shape)
    if tensor_name.endswith(QKV_WEIGHT_SUFFIX) or tensor_name.endswith(QKV_BIAS_SUFFIX):
        if qkv_layout is None:
            raise MonolithicSourceError(
                f"QKV tensor {tensor_name} reached quantization without an admitted layout decision"
            )
        if tensor_name not in (*qkv_layout.weight_names, *qkv_layout.bias_names):
            raise MonolithicSourceError(
                f"QKV tensor {tensor_name} is not part of the admitted QKV surface"
            )
        if qkv_layout.source_layout == QKV_SOURCE_LAYOUT_GROUPED:
            transformed = reconcile_qkv_rows(
                decoded,
                source_layout=qkv_layout.source_layout,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                tensor_name=tensor_name,
            )
            if reconciliation_execution is not None:
                reconciliation_execution.record(tensor_name)
            return transformed
        if qkv_layout.source_layout != QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED:
            raise MonolithicSourceError(
                f"QKV tensor {tensor_name} reached quantization with unknown source layout "
                f"{qkv_layout.source_layout!r}"
            )
    return decoded


def _decode_to_mlx(
    raw: bytes,
    dtype: str,
    shape: tuple[int, ...],
    mx_module,
    *,
    tensor_name: str,
    qkv_layout: QKVLayoutDecision | None,
    num_attention_heads: int,
    attention_head_dim: int,
    reconciliation_execution: QKVReconciliationExecution | None = None,
):
    decoded = prepare_source_array_for_quantization(
        raw,
        dtype,
        shape,
        tensor_name=tensor_name,
        qkv_layout=qkv_layout,
        num_attention_heads=num_attention_heads,
        attention_head_dim=attention_head_dim,
        reconciliation_execution=reconciliation_execution,
    )
    if dtype == "BF16":
        result = mx_module.array(decoded).astype(mx_module.bfloat16)
    else:
        result = mx_module.array(decoded).astype(mx_module.float32)
    del decoded
    return result


class MlxIsolatedQuantizer:
    """MLX-backed one-linear quantizer; importing MLX is intentionally deferred to construction."""

    def __init__(
        self,
        source: MonolithicSafetensorsSource,
        classification: SourceClassification,
        temp_dir: Path,
        reconciliation_execution: QKVReconciliationExecution | None = None,
    ):
        import mlx.core as mx
        import mlx.nn as nn

        self.source = source
        self.classification = classification
        self.temp_dir = temp_dir
        self.mx = mx
        self.nn = nn
        self.memory_snapshots: list[dict[str, int | None]] = []
        self.reconciliation_execution = reconciliation_execution
        self._snapshot("quantizer_ready")

    def bind_reconciliation_execution(self, execution: QKVReconciliationExecution) -> None:
        self.reconciliation_execution = execution

    def _snapshot(self, label: str) -> None:
        values: dict[str, object] = {"label": label}
        for key, method_name in (
            ("active_bytes", "get_active_memory"),
            ("cache_bytes", "get_cache_memory"),
            ("peak_bytes", "get_peak_memory"),
        ):
            getter = getattr(self.mx, method_name, None)
            try:
                values[key] = int(getter()) if callable(getter) else None
            except Exception:
                values[key] = None
        self.memory_snapshots.append(values)

    def quantize(self, parent: str, bias_bytes: bytes | None = None) -> QuantizedResult:
        by_name = self.classification.by_name
        item = by_name.get(parent)
        if item is None or item.role not in QUANTIZED_ROLES or item.bits is None:
            raise MonolithicSourceError(f"MLX quantizer received a non-quantized parent: {parent}")
        self._snapshot(f"before_{parent}")
        raw_weight = self.source.read_tensor(parent)
        output_features, input_features = item.descriptor.shape
        weight = _decode_to_mlx(
            raw_weight,
            item.descriptor.dtype,
            item.descriptor.shape,
            self.mx,
            tensor_name=parent,
            qkv_layout=self.classification.qkv_layout,
            num_attention_heads=self.classification.config.num_attention_heads,
            attention_head_dim=self.classification.config.attention_head_dim,
            reconciliation_execution=self.reconciliation_execution,
        )
        del raw_weight

        bias_name = f"{parent[:-len('.weight')]}.bias"
        has_bias = bias_name in by_name and by_name[bias_name].role == "learned_bias"
        linear = self.nn.Linear(input_features, output_features, bias=has_bias)
        linear.weight = weight
        del weight
        if has_bias:
            if bias_bytes is None:
                bias_bytes = self.source.read_tensor(bias_name)
            bias_descriptor = by_name[bias_name].descriptor
            linear.bias = _decode_to_mlx(
                bias_bytes,
                bias_descriptor.dtype,
                bias_descriptor.shape,
                self.mx,
                tensor_name=bias_name,
                qkv_layout=self.classification.qkv_layout,
                num_attention_heads=self.classification.config.num_attention_heads,
                attention_head_dim=self.classification.config.attention_head_dim,
                reconciliation_execution=self.reconciliation_execution,
            )

        quantized = self.nn.QuantizedLinear.from_linear(
            linear,
            group_size=64,
            bits=item.bits,
        )
        expected = quantized_tensor_shapes(item.descriptor, bits=item.bits, group_size=64)
        values = {role: getattr(quantized, role, None) for role in ("weight", "scales", "biases")}
        if any(value is None for value in values.values()):
            raise MonolithicSourceError(f"MLX quantizer omitted a packed output for {parent}")
        # The production MLX checkpoint contract stores affine companions as BF16 even when a
        # backend exposes the quantizer's intermediate scale/bias arrays as another float dtype.
        values["scales"] = values["scales"].astype(self.mx.bfloat16)
        values["biases"] = values["biases"].astype(self.mx.bfloat16)
        self.mx.eval(*values.values())
        arrays: dict[str, QuantizedArray] = {}
        for role, value in values.items():
            shape = tuple(int(dimension) for dimension in value.shape)
            if shape != expected[role]:
                raise MonolithicSourceError(
                    f"MLX {item.bits}-bit {role} shape for {parent} is {shape}; expected {expected[role]}"
                )
            arrays[role] = MlxArrayPayload(
                value,
                self.mx,
                "U32" if role == "weight" else "BF16",
                expected[role],
            )
        self._snapshot(f"after_{parent}")
        return QuantizedResult(arrays)

    def release(self, result: QuantizedResult) -> None:
        del result
        clear_cache = getattr(self.mx, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
        self._snapshot("after_release")


def _ensure_output_disjoint(source: MonolithicSafetensorsSource, output: Path) -> Path:
    requested = output.expanduser()
    if os.path.lexists(requested) and requested.is_symlink():
        raise MonolithicSourceError(
            f"refusing output path that is a symlink, including a dangling symlink: {requested}"
        )
    target = requested.resolve()
    if target == source.path:
        raise MonolithicSourceError("source and output are the same path")
    if source.path.is_relative_to(target):
        raise MonolithicSourceError(
            f"output directory encloses the read-only source; source={source.path}, output={target}"
        )
    if target.exists() and not target.is_dir():
        raise MonolithicSourceError(f"output exists but is not a directory: {target}")
    return target


def _index_metadata(
    plan: ConversionPlan,
    *,
    qkv_tensors_reconciled: int = 0,
    qkv_row_reconciliation_applied: bool = False,
) -> dict[str, object]:
    qkv_layout = plan.classification.qkv_layout
    return {
        "total_size": sum(tensor.nbytes for tensor in plan.output_tensors),
        "bounded": plan.bounded,
        "source_identity": plan.source.identity,
        "source_size": plan.source.source_size,
        "selected_quantized_weights": list(plan.selected_quantized_weights),
        "quantized_layers": plan.quantized_counts,
        "qkv_source_layout": qkv_layout.source_layout,
        "qkv_canonical_layout": qkv_layout.canonical_layout,
        "qkv_row_reconciliation_applied": qkv_row_reconciliation_applied,
        "qkv_tensors_reconciled": qkv_tensors_reconciled,
        "qkv_layout_source_identity": qkv_layout.source_identity or plan.source.identity,
        "qkv_layout_authorization": qkv_layout.authorization,
    }


def _qkv_receipt_metadata(
    plan: ConversionPlan,
    *,
    qkv_tensors_reconciled: int = 0,
    qkv_row_reconciliation_applied: bool = False,
) -> dict[str, object]:
    metadata = _index_metadata(
        plan,
        qkv_tensors_reconciled=qkv_tensors_reconciled,
        qkv_row_reconciliation_applied=qkv_row_reconciliation_applied,
    )
    return {
        key: metadata[key]
        for key in (
            "qkv_source_layout",
            "qkv_canonical_layout",
            "qkv_row_reconciliation_applied",
            "qkv_tensors_reconciled",
            "qkv_layout_source_identity",
            "qkv_layout_authorization",
        )
    }


def _dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class ConversionReceipt:
    output: Path
    shard_names: tuple[str, ...]
    output_tensor_count: int
    source_identity: str
    source_size: int
    header_bytes_read: int
    payload_bytes_read: int
    range_read_count: int
    memory_snapshots: tuple[Mapping[str, object], ...]
    qkv_source_layout: str
    qkv_canonical_layout: str
    qkv_row_reconciliation_applied: bool
    qkv_tensors_reconciled: int
    qkv_layout_source_identity: str
    qkv_layout_authorization: str


def convert(
    plan: ConversionPlan,
    output: str | Path,
    *,
    target_shard_bytes: int = MAX_SHARD_BYTES,
    quantizer: Quantizer | None = None,
) -> ConversionReceipt:
    """Atomically publish one full or bounded output directory, refusing overwrite."""
    target = _ensure_output_disjoint(plan.source, Path(output))
    if target.exists():
        raise MonolithicSourceError(f"refusing to overwrite existing output: {target}")
    if target_shard_bytes <= 0:
        raise ValueError("target shard bytes must be positive")
    target.parent.mkdir(parents=True, exist_ok=True)
    shards = plan.shards(target_shard_bytes)
    temporary: Path | None = Path(tempfile.mkdtemp(prefix=f".{target.name}.incomplete-", dir=target.parent))
    active_quantizer = quantizer
    reconciliation_execution = QKVReconciliationExecution()
    states: dict[str, QuantizedResult] = {}
    remaining: dict[str, int] = {name: 3 for name in plan.selected_quantized_weights}
    cached_biases: dict[str, bytes] = {}
    try:
        _dump_json(temporary / "config.json", plan.config_raw)
        _dump_json(temporary / "quant_config.json", plan.quant_config)
        if active_quantizer is None:
            active_quantizer = MlxIsolatedQuantizer(
                plan.source,
                plan.classification,
                temporary / ".mlx-arrays",
                reconciliation_execution,
            )
        else:
            bind_execution = getattr(active_quantizer, "bind_reconciliation_execution", None)
            if callable(bind_execution):
                bind_execution(reconciliation_execution)

        output_by_name = {tensor.name: tensor for tensor in plan.output_tensors}
        shard_names: list[str] = []
        weight_map: dict[str, str] = {}

        def write_payload(descriptor: TensorHeader, destination: BinaryIO) -> str:
            entry = output_by_name[descriptor.name]
            if entry.source_name is not None:
                if entry.quantized_role == "bias":
                    raw = cached_biases.get(entry.source_name)
                    if raw is None:
                        raw = plan.source.read_tensor(entry.source_name)
                        cached_biases[entry.source_name] = raw
                    destination.write(raw)
                    return hashlib.sha256(raw).hexdigest()
                return plan.source.copy_tensor_to(entry.source_name, destination)

            parent = entry.quantized_parent
            role = entry.quantized_role
            if parent is None or role not in {"weight", "scales", "biases"}:
                raise MonolithicSourceError(f"output tensor has no valid payload source: {entry}")
            result = states.get(parent)
            if result is None:
                bias_name = f"{parent[:-len('.weight')]}.bias"
                result = active_quantizer.quantize(parent, cached_biases.get(bias_name))
                states[parent] = result
            array = result.arrays.get(role)
            if array is None:
                raise MonolithicSourceError(f"quantizer did not produce {role} for {parent}")
            expected_dtype = entry.dtype
            expected_shape = entry.shape
            if array.dtype != expected_dtype or tuple(array.shape) != expected_shape:
                raise MonolithicSourceError(
                    f"quantizer output mismatch for {entry.name}: got dtype={array.dtype}, shape={array.shape}; "
                    f"expected dtype={expected_dtype}, shape={expected_shape}"
                )
            checksum = array.write_to(destination, temporary / ".mlx-arrays")
            remaining[parent] -= 1
            if remaining[parent] == 0:
                del states[parent]
                active_quantizer.release(result)
            return checksum

        for shard in shards:
            descriptors = tuple(
                TensorHeader(tensor.name, tensor.dtype, tensor.shape, 0, tensor.nbytes)
                for tensor in shard.tensors
            )
            shard_path = temporary / shard.filename
            write_safetensors_stream(shard_path, descriptors, write_payload, {"format": "mlx"})
            shard_names.append(shard.filename)
            for tensor in shard.tensors:
                weight_map[tensor.name] = shard.filename

        if states or any(value != 0 for value in remaining.values()):
            raise MonolithicSourceError("conversion ended with unreleased quantized linear state")
        source_identity = plan.source.identity
        _dump_json(
            temporary / "model.safetensors.index.json",
            {
                "metadata": _index_metadata(
                    plan,
                    qkv_tensors_reconciled=reconciliation_execution.qkv_tensors_reconciled,
                    qkv_row_reconciliation_applied=(
                        reconciliation_execution.qkv_row_reconciliation_applied
                    ),
                ),
                "weight_map": weight_map,
            },
        )
        plan.source.validate_current()
        os.replace(temporary, target)
        temporary = None
        memory = tuple(getattr(active_quantizer, "memory_snapshots", ()))
        return ConversionReceipt(
            target,
            tuple(shard_names),
            plan.output_tensor_count,
            source_identity,
            plan.source.source_size,
            plan.source.header_bytes_read,
            plan.source.payload_bytes_read,
            plan.source.range_read_count,
            memory,
            plan.classification.qkv_layout.source_layout,
            plan.classification.qkv_layout.canonical_layout,
            reconciliation_execution.qkv_row_reconciliation_applied,
            reconciliation_execution.qkv_tensors_reconciled,
            plan.classification.qkv_layout.source_identity or plan.source.identity,
            plan.classification.qkv_layout.authorization,
        )
    except Exception:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        raise


@dataclass(frozen=True)
class VerificationReceipt:
    output: Path
    tensor_count: int
    total_size: int
    bounded: bool
    source_checked: bool


_QKV_RECEIPT_KEYS = frozenset(
    {
        "qkv_source_layout",
        "qkv_canonical_layout",
        "qkv_row_reconciliation_applied",
        "qkv_tensors_reconciled",
        "qkv_layout_source_identity",
        "qkv_layout_authorization",
    }
)


def _validate_qkv_receipt_metadata(
    metadata: Mapping[str, object],
    *,
    plan: ConversionPlan | None,
) -> None:
    present = _QKV_RECEIPT_KEYS & set(metadata)
    if not present:
        # Existing Slice 024 receipts predate this field group and remain readable.
        return
    if present != _QKV_RECEIPT_KEYS:
        raise MonolithicSourceError(
            "output index QKV layout receipt is incomplete; expected all layout decision fields"
        )
    if metadata["qkv_source_layout"] not in {
        "grouped_qkv",
        "runtime_interleaved",
    }:
        raise MonolithicSourceError("output index QKV source layout is unknown")
    if metadata["qkv_canonical_layout"] != "runtime_interleaved":
        raise MonolithicSourceError("output index QKV canonical layout is not runtime_interleaved")
    if not isinstance(metadata["qkv_row_reconciliation_applied"], bool):
        raise MonolithicSourceError("output index QKV reconciliation flag is not boolean")
    reconciled = metadata["qkv_tensors_reconciled"]
    if not isinstance(reconciled, int) or isinstance(reconciled, bool) or reconciled < 0:
        raise MonolithicSourceError("output index QKV reconciled tensor count is invalid")
    if metadata["qkv_row_reconciliation_applied"] != (reconciled > 0):
        raise MonolithicSourceError(
            "output index QKV reconciliation flag does not match the actual reconciled count"
        )
    if metadata["qkv_source_layout"] == QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED and reconciled != 0:
        raise MonolithicSourceError(
            "runtime-interleaved output cannot report reconciled QKV tensors"
        )
    for key in ("qkv_layout_source_identity", "qkv_layout_authorization"):
        if not isinstance(metadata[key], str) or not metadata[key]:
            raise MonolithicSourceError(f"output index {key} is invalid")
    if plan is not None:
        expected = _qkv_receipt_metadata(plan)
        static_keys = _QKV_RECEIPT_KEYS - {
            "qkv_row_reconciliation_applied",
            "qkv_tensors_reconciled",
        }
        actual_static = {key: metadata[key] for key in static_keys}
        expected_static = {key: expected[key] for key in static_keys}
        if actual_static != expected_static:
            raise MonolithicSourceError(
                "output index QKV layout authorization does not match the source-derived conversion plan"
            )
        if metadata["qkv_source_layout"] == QKV_SOURCE_LAYOUT_GROUPED:
            if reconciled > plan.qkv_tensors_planned:
                raise MonolithicSourceError(
                    "output index QKV reconciled count exceeds the selected source plan"
                )


def verify_output(
    output: str | Path,
    *,
    source: MonolithicSafetensorsSource | None = None,
) -> VerificationReceipt:
    """Verify output headers/index/config/recipe, optionally against a header-only source plan."""
    root = Path(output).expanduser().resolve()
    if not root.is_dir():
        raise MonolithicSourceError(f"output directory does not exist: {root}")
    try:
        config_raw = json.loads((root / "config.json").read_text())
        quant_config = json.loads((root / "quant_config.json").read_text())
        index = json.loads((root / "model.safetensors.index.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MonolithicSourceError(f"output metadata is not valid: {exc}") from exc
    if not isinstance(config_raw, dict) or not isinstance(quant_config, dict) or not isinstance(index, dict):
        raise MonolithicSourceError("output config, quant_config, and index must be JSON objects")
    from .monolithic_source import validate_config_contract

    validate_config_contract(config_raw)
    if any(quant_config.get(key) != value for key, value in OUTPUT_RECIPE.items()):
        raise MonolithicSourceError(f"output quant_config recipe mismatch: {quant_config}")
    if not isinstance(quant_config.get("quantized_layers"), dict):
        raise MonolithicSourceError("output quant_config is missing quantized_layers")
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or not isinstance(metadata, dict):
        raise MonolithicSourceError("output index must contain metadata and weight_map objects")
    if not weight_map or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise MonolithicSourceError("output index weight_map must contain non-empty string pairs")
    if not isinstance(metadata.get("bounded"), bool):
        raise MonolithicSourceError("output index metadata must declare boolean bounded")
    bounded = metadata["bounded"]
    selected = metadata.get("selected_quantized_weights")
    if not isinstance(selected, list) or not all(isinstance(item, str) for item in selected):
        raise MonolithicSourceError("output index selected_quantized_weights metadata is malformed")

    canonical_classification = classify_expected_source_config(config_raw)
    expected_tensors, canonical_selected, topology_bounded = _build_output_topology(
        canonical_classification,
        tuple(selected) if bounded else None,
    )
    if topology_bounded is not bounded:
        raise MonolithicSourceError(
            "output bounded metadata does not match the selected canonical topology"
        )
    if tuple(selected) != canonical_selected:
        raise MonolithicSourceError(
            "output selected_quantized_weights metadata is not the exact canonical selection"
        )
    expected_quantized_counts = _quantized_counts(
        canonical_classification,
        canonical_selected,
    )
    if quant_config.get("quantized_layers") != expected_quantized_counts:
        raise MonolithicSourceError(
            "output quant_config quantized_layers does not match the selected canonical topology"
        )
    if metadata.get("quantized_layers") != expected_quantized_counts:
        raise MonolithicSourceError(
            "output index quantized_layers does not match the selected canonical topology"
        )

    shard_names = set(weight_map.values())
    if not all(isinstance(name, str) and Path(name).name == name and name.endswith(".safetensors") for name in shard_names):
        raise MonolithicSourceError("output index contains unsafe shard names")
    shard_entries = {
        path.name: path
        for path in root.iterdir()
        if path.name.endswith(".safetensors")
    }
    invalid_shards = sorted(
        name for name, path in shard_entries.items() if path.is_symlink() or not path.is_file()
    )
    if invalid_shards:
        raise MonolithicSourceError(
            f"output shard entries must be regular files: {invalid_shards}"
        )
    actual_shards = set(shard_entries)
    missing_shards = sorted(shard_names - actual_shards)
    unexpected_shards = sorted(actual_shards - shard_names)
    if missing_shards:
        raise MonolithicSourceError(
            f"output index maps tensors to nonexistent shards: {missing_shards}"
        )
    if unexpected_shards:
        raise MonolithicSourceError(
            f"output contains unexpected safetensors shard files: {unexpected_shards}"
        )
    actual: dict[str, tuple[str, tuple[int, ...]]] = {}
    physical_shards: dict[str, str] = {}
    total_size = 0
    for shard in sorted(actual_shards):
        header = read_safetensors_header(root / shard)
        for descriptor in header.tensors:
            if descriptor.name in actual:
                raise MonolithicSourceError(
                    f"output tensor is duplicated across shards: {descriptor.name}; "
                    f"shards={[physical_shards[descriptor.name], shard]}"
                )
            mapped_shard = weight_map.get(descriptor.name)
            if mapped_shard != shard:
                raise MonolithicSourceError(
                    f"output index maps tensor {descriptor.name!r} to {mapped_shard!r}, "
                    f"but it is physically present in {shard!r}"
                )
            actual[descriptor.name] = (descriptor.dtype, descriptor.shape)
            physical_shards[descriptor.name] = shard
            total_size += descriptor.nbytes
    if set(weight_map) != set(actual):
        raise MonolithicSourceError("output index/header tensor sets differ")
    expected = {
        tensor.name: (tensor.dtype, tensor.shape)
        for tensor in expected_tensors
    }
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(
            name
            for name in set(expected) & set(actual)
            if expected[name] != actual[name]
        )
        raise MonolithicSourceError(
            "output tensor descriptors differ from the exact canonical conversion topology; "
            f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
        )
    if metadata.get("total_size") != total_size:
        raise MonolithicSourceError(
            f"output index total_size {metadata.get('total_size')} does not match headers {total_size}"
        )
    source_checked = False
    source_plan: ConversionPlan | None = None
    if source is not None:
        source.validate_current()
        source_plan = build_conversion_plan(
            source,
            selected_quantized_weights=None if not bounded else tuple(selected),
        )
        source_expected = {tensor.name: (tensor.dtype, tensor.shape) for tensor in source_plan.output_tensors}
        if source_expected != actual:
            raise MonolithicSourceError("output tensor descriptors differ from the source-derived conversion plan")
        if dict(source_plan.config_raw) != config_raw:
            raise MonolithicSourceError("output config does not match the source-embedded config")
        if metadata.get("source_identity") != source.identity:
            raise SourceStaleError("output source identity does not match the currently registered source")
        if metadata.get("source_size") != source.source_size:
            raise SourceStaleError("output source size does not match the currently registered source")
        source_checked = True
    _validate_qkv_receipt_metadata(metadata, plan=source_plan)
    return VerificationReceipt(root, len(actual), total_size, bounded, source_checked)

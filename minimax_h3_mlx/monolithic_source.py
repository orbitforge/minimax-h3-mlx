"""Header-first access and topology admission for a monolithic H3 safetensors source.

The beta checkpoint is one large safetensors file rather than an MLX directory.  This module is
deliberately independent of MLX: registration parses only the header, and materialization is
limited to one named payload range at a time.  The actual MLX quantizer lives in
``monolithic_quant`` and is imported only when conversion is requested.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Mapping

import numpy as np

from .checkpoint_forge.tensor_io import (
    COPY_CHUNK_BYTES,
    DTYPE_BYTES,
    SafetensorsHeader,
    TensorHeader,
    read_safetensors_header,
)
from .config import DiTConfig
from .quantize import CORE_LINEARS, NEVER_QUANTIZE, QuantConfig


SUPPORTED_SOURCE_DTYPES = frozenset({"BF16", "F32"})
CONFIG_METADATA_KEY = "config"

# The MLX DiT consumes fused QKV rows as ``[head0:q,k,v][head1:q,k,v]...``.  The accepted
# PinkCherry beta master is a distinct source lineage whose fused rows are ``[Q_all;K_all;V_all]``.
# Shapes alone cannot distinguish those layouts, so admission is payload-receipt-bound and fail-closed.
QKV_WEIGHT_SUFFIX = ".attn.qkv_proj.weight"
QKV_BIAS_SUFFIX = ".attn.qkv_proj.bias"
QKV_SOURCE_LAYOUT_METADATA_KEY = "qkv_source_layout"
QKV_SOURCE_LAYOUT_GROUPED = "grouped_qkv"
QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED = "runtime_interleaved"
QKV_CANONICAL_LAYOUT = QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED
QKV_MASTER_FORMAT_METADATA_KEY = "pinkcherry_h3_master_format"
QKV_ACCEPTED_MASTER_FORMAT = "bf16_convrot_lineage_consolidation_v1"
QKV_SOURCE_FINGERPRINT_METADATA_KEY = "pinkcherry_h3_source_fingerprint"
QKV_ACCEPTED_SOURCE_FINGERPRINT = "41490c636f0f1158eadb37fab5b5c4b2fa16a4548f542db919756a59c40e20df"
QKV_LAYOUT_AUTHORIZATION_SCHEMA = "slice025.qkv_layout_authorization.v1"
QKV_LAYOUT_AUTHORIZATION_RECEIPT_NAME = "qkv_layout_authorization.json"
QKV_LAYOUT_AUTHORIZATION_RECEIPT_PATH = Path(__file__).with_name(QKV_LAYOUT_AUTHORIZATION_RECEIPT_NAME)
QKV_LAYOUT_AUTHORIZATION_RECEIPT_SHA256 = (
    "62095531d253bc129c39b0033fa4b6c167067588aa5d0dfd851720d8753e10b6"
)
QKV_ACCEPTED_SOURCE_PATH = (
    "/Users/elbancol/Downloads/PinkCherry-beta-redownload/beta-0.6-fl2va/"
    "PinkCherry_fl2va_MiniMax_H3_bf16_beta-0.6.safetensors"
)
QKV_ACCEPTED_SOURCE_BYTES = 66288818760
QKV_ACCEPTED_SOURCE_SHA256 = "16f1950cc83bd686106d49588c8611281fbb5e9ae46f8cd1ae7945fd4e00357d"
QKV_ACCEPTED_SOURCE_IDENTITY = (
    "safetensors:/Users/elbancol/Downloads/PinkCherry-beta-redownload/beta-0.6-fl2va/"
    "PinkCherry_fl2va_MiniMax_H3_bf16_beta-0.6.safetensors:16777233:76930510:66288818760:"
    "1787194704372543937:1787194704374750655:46d8e0071c98dbc5690b5e8bf990d536b18289c8b7b4c97724973fab88d5b35b"
)

# These are the fields present in the canonical production transformer's config.json.  The
# loader's DiTConfig has a deliberate rope_theta default because the production file omits it;
# this adapter does not synthesize any other field or topology value.
PRODUCTION_CONFIG_KEYS = (
    "adaln_out_features",
    "attention_head_dim",
    "audio_latents_dim",
    "ffn_hidden_size",
    "final_adaln_out_features",
    "final_norm_eps",
    "hidden_size",
    "latents_dim",
    "norm_eps",
    "num_attention_heads",
    "num_layers",
    "patch_size",
    "qk_norm_eps",
    "rope_inv_freq_len",
    "text_dim",
    "time_embed_dim",
    "time_embed_hidden_size",
    "timestep_input_dim",
    "token_refiner_num_layers",
)
FP32_SOURCE_PREFIXES = (
    "video_patch_proj.",
    "audio_patch_proj.",
    "time_embedder.",
    "final_layer.video_out.",
    "final_layer.audio_out.",
)


class MonolithicSourceError(ValueError):
    """Base error for header, identity, and source-topology admission failures."""


class SourceStaleError(MonolithicSourceError):
    """Raised when the registered source's local identity changes during conversion."""


@dataclass(frozen=True)
class FusedQKVSurface:
    """The complete fused-QKV weight/bias surface derived from admitted source names."""

    weight_names: tuple[str, ...]
    bias_names: tuple[str, ...]

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)


@dataclass(frozen=True)
class QKVLayoutDecision:
    """The source-to-runtime QKV layout contract carried by a conversion plan."""

    source_layout: str
    canonical_layout: str
    row_reconciliation_required: bool
    weight_names: tuple[str, ...]
    bias_names: tuple[str, ...]
    authorization: str
    source_identity: str | None = None

    @property
    def weight_count(self) -> int:
        return len(self.weight_names)


def is_fused_qkv_weight(name: str) -> bool:
    return name.endswith(QKV_WEIGHT_SUFFIX)


def is_fused_qkv_bias(name: str) -> bool:
    return name.endswith(QKV_BIAS_SUFFIX)


def enumerate_fused_qkv_surface(tensor_names: tuple[str, ...] | list[str] | set[str]) -> FusedQKVSurface:
    """Derive every fused QKV weight and bias from the source topology without a fixed count."""
    names = tuple(sorted(tensor_names))
    qkvish = tuple(name for name in names if ".attn.qkv_proj." in name)
    weights = tuple(name for name in qkvish if is_fused_qkv_weight(name))
    biases = tuple(name for name in qkvish if is_fused_qkv_bias(name))
    unsupported = tuple(
        name for name in qkvish if not is_fused_qkv_weight(name) and not is_fused_qkv_bias(name)
    )
    if unsupported:
        raise MonolithicSourceError(
            f"unsupported fused QKV tensor names in source topology: {list(unsupported)}"
        )
    weight_parents = {name[:-len(".weight")] for name in weights}
    orphan_biases = tuple(
        name for name in biases if name[:-len(".bias")] not in weight_parents
    )
    if orphan_biases:
        raise MonolithicSourceError(
            f"fused QKV bias has no corresponding weight: {list(orphan_biases)}"
        )
    return FusedQKVSurface(weights, biases)


def _layout_decision(
    source_layout: str,
    surface: FusedQKVSurface,
    *,
    authorization: str,
    source_identity: str | None,
) -> QKVLayoutDecision:
    if source_layout not in {QKV_SOURCE_LAYOUT_GROUPED, QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED}:
        raise MonolithicSourceError(f"unknown QKV source layout: {source_layout!r}")
    if not surface.weight_names:
        raise MonolithicSourceError("admitted source topology contains no fused QKV weights")
    return QKVLayoutDecision(
        source_layout=source_layout,
        canonical_layout=QKV_CANONICAL_LAYOUT,
        row_reconciliation_required=source_layout == QKV_SOURCE_LAYOUT_GROUPED,
        weight_names=surface.weight_names,
        bias_names=surface.bias_names,
        authorization=authorization,
        source_identity=source_identity,
    )


@dataclass(frozen=True)
class SourceFingerprint:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    header_sha256: str

    @classmethod
    def capture(cls, path: Path, header_sha256: str) -> "SourceFingerprint":
        try:
            stat = path.stat()
        except OSError as exc:
            raise MonolithicSourceError(f"could not stat source {path}: {exc}") from exc
        return cls(
            device=int(stat.st_dev),
            inode=int(stat.st_ino),
            size=int(stat.st_size),
            mtime_ns=int(stat.st_mtime_ns),
            ctime_ns=int(stat.st_ctime_ns),
            header_sha256=header_sha256,
        )

    def identity(self, path: Path) -> str:
        return (
            f"safetensors:{path.resolve()}:{self.device}:{self.inode}:{self.size}:"
            f"{self.mtime_ns}:{self.ctime_ns}:{self.header_sha256}"
        )


def _header_sha256(path: Path, header: SafetensorsHeader) -> str:
    digest = hashlib.sha256()
    remaining = header.data_start
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise MonolithicSourceError(f"unexpected EOF while reading header from {path}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


class MonolithicSafetensorsSource:
    """A validated, named-range view of one local safetensors file.

    Construction reads the 8-byte length and JSON header through the shared safetensors parser;
    no payload bytes are read.  ``read_tensor`` and ``copy_tensor_to`` are the only payload access
    methods, and both revalidate the source fingerprint before and after the range read.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise MonolithicSourceError(f"monolithic source is not a file: {self.path}")
        try:
            header = read_safetensors_header(self.path)
        except (OSError, ValueError) as exc:
            raise MonolithicSourceError(f"invalid monolithic safetensors source {self.path}: {exc}") from exc
        unsupported = sorted({tensor.dtype for tensor in header.tensors} - SUPPORTED_SOURCE_DTYPES)
        if unsupported:
            raise MonolithicSourceError(
                f"unsupported source dtype(s) in {self.path}: {unsupported}; "
                f"expected only {sorted(SUPPORTED_SOURCE_DTYPES)}"
            )
        header_hash = _header_sha256(self.path, header)
        self.header = header
        self._tensor_map = header.tensor_map()
        self._fingerprint = SourceFingerprint.capture(self.path, header_hash)
        self.header_bytes_read = header.data_start
        self.payload_bytes_read = 0
        self.range_read_count = 0

    @property
    def metadata(self) -> Mapping[str, str]:
        return self.header.metadata

    @property
    def tensors(self) -> tuple[TensorHeader, ...]:
        return self.header.tensors

    @property
    def tensor_names(self) -> tuple[str, ...]:
        return tuple(tensor.name for tensor in self.header.tensors)

    @property
    def source_size(self) -> int:
        return self._fingerprint.size

    @property
    def fingerprint(self) -> SourceFingerprint:
        return self._fingerprint

    @property
    def identity(self) -> str:
        return self._fingerprint.identity(self.path)

    def descriptor(self, name: str) -> TensorHeader:
        try:
            return self._tensor_map[name]
        except KeyError as exc:
            raise MonolithicSourceError(f"source tensor is missing: {name}") from exc

    def _current_stat_fingerprint(self) -> SourceFingerprint:
        return SourceFingerprint.capture(self.path, self._fingerprint.header_sha256)

    def validate_current(self) -> None:
        try:
            current = self._current_stat_fingerprint()
        except MonolithicSourceError as exc:
            raise SourceStaleError(f"source is unavailable: {self.path}") from exc
        if current != self._fingerprint:
            raise SourceStaleError(
                f"source identity changed after registration: {self.path}; "
                f"registered={self.identity}, current={current.identity(self.path)}"
            )

    def _absolute_range(self, name: str) -> tuple[TensorHeader, int, int]:
        descriptor = self.descriptor(name)
        return (
            descriptor,
            self.header.data_start + descriptor.start,
            self.header.data_start + descriptor.end,
        )

    def read_tensor(self, name: str) -> bytes:
        """Read exactly one tensor payload and account for the exact bytes read."""
        descriptor, start, end = self._absolute_range(name)
        self.validate_current()
        output = bytearray()
        try:
            with self.path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start
                while remaining:
                    chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise MonolithicSourceError(
                            f"unexpected EOF while reading {name} from {self.path}"
                        )
                    output.extend(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            raise MonolithicSourceError(f"could not read source tensor {name}: {exc}") from exc
        self.payload_bytes_read += len(output)
        self.range_read_count += 1
        self.validate_current()
        if len(output) != descriptor.nbytes:
            raise MonolithicSourceError(
                f"source tensor {name} read {len(output)} bytes; expected {descriptor.nbytes}"
            )
        return bytes(output)

    def copy_tensor_to(self, name: str, destination: BinaryIO) -> str:
        """Copy exactly one tensor payload to an open output stream without whole-tensor buffering."""
        descriptor, start, end = self._absolute_range(name)
        self.validate_current()
        digest = hashlib.sha256()
        copied = 0
        try:
            with self.path.open("rb") as handle:
                handle.seek(start)
                remaining = end - start
                while remaining:
                    chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
                    if not chunk:
                        raise MonolithicSourceError(
                            f"unexpected EOF while copying {name} from {self.path}"
                        )
                    destination.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    remaining -= len(chunk)
        except OSError as exc:
            raise MonolithicSourceError(f"could not copy source tensor {name}: {exc}") from exc
        self.payload_bytes_read += copied
        self.range_read_count += 1
        self.validate_current()
        if copied != descriptor.nbytes:
            raise MonolithicSourceError(
                f"source tensor {name} copied {copied} bytes; expected {descriptor.nbytes}"
            )
        return digest.hexdigest()


def _load_qkv_layout_authorization_receipt() -> Mapping[str, object]:
    """Load and validate the checked-in payload-bound Slice 025 layout receipt."""
    receipt_path = QKV_LAYOUT_AUTHORIZATION_RECEIPT_PATH
    try:
        raw = receipt_path.read_bytes()
    except OSError as exc:
        raise MonolithicSourceError(
            f"QKV layout authorization receipt is unavailable: {receipt_path}"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != QKV_LAYOUT_AUTHORIZATION_RECEIPT_SHA256:
        raise MonolithicSourceError(
            "QKV layout authorization receipt checksum does not match the admitted evidence"
        )
    try:
        receipt = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MonolithicSourceError("QKV layout authorization receipt is not valid JSON") from exc
    if not isinstance(receipt, dict) or receipt.get("receipt_schema") != QKV_LAYOUT_AUTHORIZATION_SCHEMA:
        raise MonolithicSourceError("QKV layout authorization receipt schema is not admitted")

    beta = receipt.get("beta_source")
    if not isinstance(beta, dict):
        raise MonolithicSourceError("QKV layout authorization receipt has no beta source record")
    if {
        beta.get("path"),
        beta.get("bytes"),
        beta.get("sha256"),
        beta.get("source_identity"),
    } != {
        QKV_ACCEPTED_SOURCE_PATH,
        QKV_ACCEPTED_SOURCE_BYTES,
        QKV_ACCEPTED_SOURCE_SHA256,
        QKV_ACCEPTED_SOURCE_IDENTITY,
    }:
        raise MonolithicSourceError(
            "QKV layout authorization receipt beta source identity is not admitted"
        )
    if beta.get("topology") != {"tensor_count": 535, "bf16": 522, "f32": 13}:
        raise MonolithicSourceError("QKV layout authorization receipt topology is not admitted")

    coverage = receipt.get("coverage")
    if not isinstance(coverage, dict) or coverage != {
        "all_expected_qkv_examined": True,
        "examined_qkv_count": 52,
        "expected_qkv_count": 52,
        "main_block_count": 50,
        "token_refiner_count": 2,
    }:
        raise MonolithicSourceError("QKV layout authorization receipt coverage is incomplete")
    conclusion = receipt.get("layout_conclusion")
    if not isinstance(conclusion, dict) or conclusion.get("status") != "GROUPED_QKV_PAYLOAD_SUPPORTED":
        raise MonolithicSourceError("QKV layout authorization receipt has no grouped-QKV conclusion")
    if conclusion.get("accepted_source_layout") != QKV_SOURCE_LAYOUT_GROUPED:
        raise MonolithicSourceError("QKV layout authorization receipt does not authorize grouped QKV")
    if conclusion.get("canonical_runtime_layout") != QKV_CANONICAL_LAYOUT:
        raise MonolithicSourceError("QKV layout authorization receipt has an invalid runtime layout")

    aggregate = receipt.get("aggregate")
    if not isinstance(aggregate, dict):
        raise MonolithicSourceError("QKV layout authorization receipt has no aggregate metrics")
    direct_metrics = aggregate.get("direct_orientation")
    grouped_metrics = aggregate.get("grouped_to_runtime_orientation")
    if not isinstance(direct_metrics, dict) or not isinstance(grouped_metrics, dict):
        raise MonolithicSourceError("QKV layout authorization receipt aggregate metrics are incomplete")
    decisive_metrics = (
        direct_metrics.get("mean_relative_l2"),
        direct_metrics.get("max_cosine"),
        grouped_metrics.get("max_relative_l2"),
        grouped_metrics.get("min_cosine"),
    )
    if any(
        not isinstance(value, (int, float)) or isinstance(value, bool)
        for value in decisive_metrics
    ):
        raise MonolithicSourceError("QKV layout authorization receipt aggregate metrics are invalid")
    if not (
        decisive_metrics[0] > 1.0
        and decisive_metrics[1] < 0.1
        and decisive_metrics[2] < 0.1
        and decisive_metrics[3] > 0.99
    ):
        raise MonolithicSourceError(
            "QKV layout authorization receipt does not decisively support grouped QKV"
        )

    tensors = receipt.get("tensors")
    if not isinstance(tensors, list) or len(tensors) != 52:
        raise MonolithicSourceError("QKV layout authorization receipt does not contain all tensors")
    names: set[str] = set()
    expected_names = {
        *(f"blocks.{block}.attn.qkv_proj.weight" for block in range(50)),
        *(f"token_refiner.blocks.{block}.attn.qkv_proj.weight" for block in range(2)),
    }
    for tensor in tensors:
        if not isinstance(tensor, dict) or not isinstance(tensor.get("name"), str):
            raise MonolithicSourceError("QKV layout authorization receipt has an invalid tensor record")
        name = tensor["name"]
        if name in names:
            raise MonolithicSourceError("QKV layout authorization receipt has duplicate tensor records")
        names.add(name)
        for orientation in ("direct", "grouped_to_runtime"):
            metrics = tensor.get(orientation)
            if (
                not isinstance(metrics, dict)
                or not isinstance(metrics.get("relative_l2"), (int, float))
                or not isinstance(metrics.get("cosine"), (int, float))
            ):
                raise MonolithicSourceError(
                    f"QKV layout authorization receipt has incomplete metrics for {name}"
                )
    if names != expected_names:
        raise MonolithicSourceError(
            "QKV layout authorization receipt tensor surface is not the admitted 52-tensor QKV surface"
        )
    return receipt


def _authorize_grouped_qkv_source(
    source: MonolithicSafetensorsSource,
    *,
    master_format: str | None,
    source_fingerprint: str | None,
) -> tuple[str, str]:
    """Authorize grouped rows only for the exact source bound by the independent receipt."""
    if (
        master_format != QKV_ACCEPTED_MASTER_FORMAT
        or source_fingerprint != QKV_ACCEPTED_SOURCE_FINGERPRINT
    ):
        raise MonolithicSourceError(
            "grouped QKV source layout requires the exact accepted beta lineage metadata"
        )
    receipt = _load_qkv_layout_authorization_receipt()
    beta = receipt["beta_source"]
    if not isinstance(beta, dict):
        raise MonolithicSourceError("QKV layout authorization receipt beta source record is invalid")
    if source.path.as_posix() != QKV_ACCEPTED_SOURCE_PATH or source.source_size != QKV_ACCEPTED_SOURCE_BYTES:
        raise MonolithicSourceError(
            "grouped QKV authorization is bound to the exact accepted beta source path and size"
        )
    if source.identity != QKV_ACCEPTED_SOURCE_IDENTITY or source.identity != beta["source_identity"]:
        raise MonolithicSourceError(
            "grouped QKV authorization source identity does not match the payload-bound receipt"
        )
    authorization = (
        f"payload_receipt:{QKV_LAYOUT_AUTHORIZATION_SCHEMA};"
        f"receipt_sha256={QKV_LAYOUT_AUTHORIZATION_RECEIPT_SHA256};"
        f"source_sha256={QKV_ACCEPTED_SOURCE_SHA256};"
        f"source_identity={source.identity};"
        f"layout={QKV_SOURCE_LAYOUT_GROUPED}"
    )
    return authorization, source.identity


def resolve_qkv_layout(
    source: MonolithicSafetensorsSource,
    surface: FusedQKVSurface,
) -> QKVLayoutDecision:
    """Resolve the source QKV layout from explicit metadata and trusted beta lineage evidence."""
    explicit = source.metadata.get(QKV_SOURCE_LAYOUT_METADATA_KEY)
    master_format = source.metadata.get(QKV_MASTER_FORMAT_METADATA_KEY)
    source_fingerprint = source.metadata.get(QKV_SOURCE_FINGERPRINT_METADATA_KEY)

    if explicit is not None:
        if explicit not in {QKV_SOURCE_LAYOUT_GROUPED, QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED}:
            raise MonolithicSourceError(
                f"unknown QKV source layout metadata {explicit!r}; refusing to guess"
            )
        if explicit == QKV_SOURCE_LAYOUT_GROUPED:
            authorization, source_identity = _authorize_grouped_qkv_source(
                source,
                master_format=master_format,
                source_fingerprint=source_fingerprint,
            )
            return _layout_decision(
                explicit,
                surface,
                authorization=authorization,
                source_identity=source_identity,
            )
        elif master_format == QKV_ACCEPTED_MASTER_FORMAT:
            raise MonolithicSourceError(
                "QKV source layout metadata conflicts with the accepted grouped beta lineage"
            )
        elif master_format is not None or source_fingerprint is not None:
            raise MonolithicSourceError(
                "QKV source layout metadata is ambiguous with additional source lineage metadata"
            )
        return _layout_decision(
            explicit,
            surface,
            authorization=f"metadata:{QKV_SOURCE_LAYOUT_METADATA_KEY}={explicit}",
            source_identity=source.identity,
        )

    if master_format == QKV_ACCEPTED_MASTER_FORMAT:
        authorization, source_identity = _authorize_grouped_qkv_source(
            source,
            master_format=master_format,
            source_fingerprint=source_fingerprint,
        )
        return _layout_decision(
            QKV_SOURCE_LAYOUT_GROUPED,
            surface,
            authorization=authorization,
            source_identity=source_identity,
        )

    if master_format is not None or source_fingerprint is not None:
        raise MonolithicSourceError(
            "QKV source layout is unknown for the supplied source lineage; refusing to guess"
        )
    raise MonolithicSourceError(
        f"QKV source layout metadata {QKV_SOURCE_LAYOUT_METADATA_KEY!r} is missing; refusing to guess"
    )


def reconcile_qkv_rows(
    value: np.ndarray,
    *,
    source_layout: str,
    num_attention_heads: int,
    attention_head_dim: int,
    tensor_name: str,
) -> np.ndarray:
    """Canonicalize grouped QKV rows into runtime interleaving without changing feature axes."""
    if source_layout not in {QKV_SOURCE_LAYOUT_GROUPED, QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED}:
        raise MonolithicSourceError(
            f"cannot reconcile {tensor_name}: unknown QKV source layout {source_layout!r}"
        )
    if num_attention_heads <= 0 or attention_head_dim <= 0:
        raise MonolithicSourceError(
            f"cannot reconcile {tensor_name}: QKV dimensions must be positive"
        )
    expected_rows = 3 * num_attention_heads * attention_head_dim
    if value.ndim == 2:
        if value.shape[0] != expected_rows or value.shape[1] <= 0:
            raise MonolithicSourceError(
                f"{tensor_name} fused QKV weight shape {tuple(value.shape)} is incompatible with "
                f"{num_attention_heads} heads x {attention_head_dim} head_dim; "
                f"expected ({expected_rows}, input_features>0)"
            )
        if source_layout == QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED:
            return np.array(value, copy=True)
        return (
            value.reshape(3, num_attention_heads, attention_head_dim, value.shape[1])
            .transpose(1, 0, 2, 3)
            .reshape(value.shape)
            .copy()
        )
    if value.ndim == 1:
        if value.shape[0] != expected_rows:
            raise MonolithicSourceError(
                f"{tensor_name} fused QKV bias shape {tuple(value.shape)} is incompatible with "
                f"{num_attention_heads} heads x {attention_head_dim} head_dim; "
                f"expected ({expected_rows},)"
            )
        if source_layout == QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED:
            return np.array(value, copy=True)
        return (
            value.reshape(3, num_attention_heads, attention_head_dim)
            .transpose(1, 0, 2)
            .reshape(value.shape)
            .copy()
        )
    raise MonolithicSourceError(
        f"{tensor_name} fused QKV tensor must be rank 1 bias or rank 2 weight; got rank {value.ndim}"
    )


def extract_embedded_config(
    metadata: Mapping[str, str],
    *,
    metadata_key: str = CONFIG_METADATA_KEY,
) -> dict[str, object]:
    """Decode the one explicitly selected embedded JSON config metadata value."""
    if metadata_key not in metadata:
        available = sorted(metadata)
        raise MonolithicSourceError(
            f"embedded config metadata key {metadata_key!r} is missing; available keys={available}"
        )
    raw = metadata[metadata_key]
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise MonolithicSourceError(
            f"embedded config metadata {metadata_key!r} is not JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise MonolithicSourceError(
            f"embedded config metadata {metadata_key!r} must decode to an object"
        )
    # The beta header's exact format is a single explicit production wrapper.  Unwrap only this
    # proven shape; unknown wrappers remain an admission failure rather than a guessed fallback.
    if set(value) == {"transformer"}:
        transformer = value["transformer"]
        if not isinstance(transformer, dict):
            raise MonolithicSourceError("embedded config transformer wrapper must contain an object")
        value = transformer
    return value


def validate_config_contract(raw: Mapping[str, object]) -> DiTConfig:
    """Validate that embedded config is sufficient for the existing production DiTConfig."""
    missing = [key for key in PRODUCTION_CONFIG_KEYS if key not in raw]
    if missing:
        raise MonolithicSourceError(
            f"embedded beta config is insufficient for the production config contract; "
            f"missing={missing}"
        )
    if raw.get("_class_name") not in (None, "MiniMaxH3DiTModel"):
        raise MonolithicSourceError(
            f"unsupported embedded config class: {raw.get('_class_name')!r}"
        )

    integer_fields = {
        key
        for key in PRODUCTION_CONFIG_KEYS
        if key not in {"patch_size", "final_norm_eps", "norm_eps", "qk_norm_eps"}
    }
    for key in integer_fields:
        value = raw[key]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise MonolithicSourceError(f"embedded config field {key!r} must be a positive integer")
    patch = raw["patch_size"]
    if (
        not isinstance(patch, (list, tuple))
        or len(patch) != 3
        or not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in patch)
    ):
        raise MonolithicSourceError("embedded config patch_size must be three positive integers")
    for key in ("final_norm_eps", "norm_eps", "qk_norm_eps"):
        value = raw[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise MonolithicSourceError(f"embedded config field {key!r} must be a positive number")
    if "rope_theta" in raw:
        value = raw["rope_theta"]
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
            raise MonolithicSourceError("embedded config rope_theta must be a positive number")

    config = DiTConfig.from_dict(dict(raw))
    if config.num_layers != 50 or config.token_refiner_num_layers != 2:
        raise MonolithicSourceError(
            "this beta converter requires the proven 50-block/2-token-refiner H3 topology; "
            f"got num_layers={config.num_layers}, token_refiner_num_layers={config.token_refiner_num_layers}"
        )
    return config


def _add(tensors: dict[str, tuple[int, ...]], name: str, shape: tuple[int, ...]) -> None:
    if name in tensors:
        raise AssertionError(f"duplicate expected tensor name: {name}")
    tensors[name] = shape


def expected_tensor_shapes(config: DiTConfig) -> dict[str, tuple[int, ...]]:
    """Derive the exact 535-tensor raw H3 topology from the admitted config."""
    tensors: dict[str, tuple[int, ...]] = {}
    hidden = config.hidden_size
    inner = config.inner_dim

    for prefix, input_dim in (
        ("video_patch_proj", config.video_patch_dim),
        ("audio_patch_proj", config.audio_latents_dim),
        ("condition_proj", config.text_dim),
    ):
        _add(tensors, f"{prefix}.weight", (hidden, input_dim))
        _add(tensors, f"{prefix}.bias", (hidden,))
    _add(tensors, "time_embedder.proj_in.weight", (config.time_embed_hidden_size, config.timestep_input_dim))
    _add(tensors, "time_embedder.proj_in.bias", (config.time_embed_hidden_size,))
    _add(tensors, "time_embedder.proj_out.weight", (config.time_embed_dim, config.time_embed_hidden_size))
    _add(tensors, "time_embedder.proj_out.bias", (config.time_embed_dim,))

    def add_stack_block(prefix: str, *, with_adaln: bool) -> None:
        _add(tensors, f"{prefix}.norm1.weight", (hidden,))
        _add(tensors, f"{prefix}.attn.qkv_proj.weight", (3 * inner, hidden))
        _add(tensors, f"{prefix}.attn.q_norm.weight", (config.attention_head_dim,))
        _add(tensors, f"{prefix}.attn.k_norm.weight", (config.attention_head_dim,))
        _add(tensors, f"{prefix}.attn.out_proj.weight", (hidden, inner))
        _add(tensors, f"{prefix}.norm2.weight", (hidden,))
        _add(tensors, f"{prefix}.mlp.fc1.weight", (2 * config.ffn_hidden_size, hidden))
        _add(tensors, f"{prefix}.mlp.fc2.weight", (hidden, config.ffn_hidden_size))
        if with_adaln:
            _add(tensors, f"{prefix}.adaln_proj.linear.weight", (config.adaln_out_features, config.time_embed_dim))
            _add(tensors, f"{prefix}.adaln_proj.linear.bias", (config.adaln_out_features,))

    for index in range(config.token_refiner_num_layers):
        add_stack_block(f"token_refiner.blocks.{index}", with_adaln=False)
    _add(tensors, "token_refiner.final_norm.weight", (hidden,))
    for index in range(config.num_layers):
        add_stack_block(f"blocks.{index}", with_adaln=True)

    _add(tensors, "final_layer.norm.weight", (hidden,))
    _add(tensors, "final_layer.adaln_proj.linear.weight", (config.final_adaln_out_features, config.time_embed_dim))
    _add(tensors, "final_layer.adaln_proj.linear.bias", (config.final_adaln_out_features,))
    _add(tensors, "final_layer.video_out.weight", (config.video_patch_dim, hidden))
    _add(tensors, "final_layer.video_out.bias", (config.video_patch_dim,))
    _add(tensors, "final_layer.audio_out.weight", (config.audio_latents_dim, hidden))
    _add(tensors, "final_layer.audio_out.bias", (config.audio_latents_dim,))
    _add(tensors, "rope.inv_freq", (config.rope_inv_freq_len,))
    return tensors


def expected_source_dtype(name: str) -> str:
    if name == "rope.inv_freq" or name.startswith(FP32_SOURCE_PREFIXES):
        return "F32"
    return "BF16"


def _is_core_module(module_path: str, config: DiTConfig) -> bool:
    for stack, count in (
        ("blocks", config.num_layers),
        ("token_refiner.blocks", config.token_refiner_num_layers),
    ):
        for index in range(count):
            prefix = f"{stack}.{index}"
            if any(module_path == f"{prefix}{suffix}" for suffix in CORE_LINEARS):
                return True
    return False


def _is_block_adaln_module(module_path: str, config: DiTConfig) -> bool:
    return any(
        module_path == f"blocks.{index}.adaln_proj.linear"
        for index in range(config.num_layers)
    )


@dataclass(frozen=True)
class TensorClassification:
    name: str
    descriptor: TensorHeader
    role: str
    bits: int | None = None
    module_path: str | None = None


@dataclass(frozen=True)
class SourceClassification:
    config: DiTConfig
    config_raw: Mapping[str, object]
    tensors: tuple[TensorClassification, ...]
    counts: Mapping[str, int]
    qkv_layout: QKVLayoutDecision

    @property
    def by_name(self) -> dict[str, TensorClassification]:
        return {item.name: item for item in self.tensors}

    @property
    def quantized_weights(self) -> tuple[TensorClassification, ...]:
        return tuple(
            item
            for item in self.tensors
            if item.role in {"q6_core_weight", "q8_block_adaln_weight"}
        )


def _classify_admitted_descriptors(
    config: DiTConfig,
    config_raw: Mapping[str, object],
    descriptors: Mapping[str, TensorHeader],
    qkv_layout: QKVLayoutDecision,
) -> SourceClassification:
    """Apply the production QuantConfig policy to an already admitted canonical topology."""
    policy = QuantConfig(bits=6, group_size=64, quantize_adaln=True, adaln_bits=8)
    preliminary: list[TensorClassification] = []
    for name in sorted(descriptors):
        descriptor = descriptors[name]
        if name == "rope.inv_freq":
            preliminary.append(TensorClassification(name, descriptor, "recomputed"))
            continue
        if name.endswith(".weight"):
            module_path = name[:-len(".weight")]
            policy_bits = policy.bits_for(module_path)
            if _is_core_module(module_path, config):
                if policy_bits != 6:
                    raise MonolithicSourceError(
                        f"QuantConfig policy did not admit core module {module_path} at Q6"
                    )
                preliminary.append(TensorClassification(name, descriptor, "q6_core_weight", 6, module_path))
            elif _is_block_adaln_module(module_path, config):
                if policy_bits != 8:
                    raise MonolithicSourceError(
                        f"QuantConfig policy did not admit block AdaLN module {module_path} at Q8"
                    )
                preliminary.append(TensorClassification(name, descriptor, "q8_block_adaln_weight", 8, module_path))
            elif policy_bits is not None:
                raise MonolithicSourceError(
                    f"unexpected policy-quantized non-canonical module {module_path} at {policy_bits} bits"
                )
            else:
                preliminary.append(TensorClassification(name, descriptor, "ordinary", module_path=module_path))
        else:
            preliminary.append(TensorClassification(name, descriptor, "ordinary"))

    quantized_parents = {
        item.name[:-len(".weight")]
        for item in preliminary
        if item.role in {"q6_core_weight", "q8_block_adaln_weight"}
    }
    classified: list[TensorClassification] = []
    for item in preliminary:
        if item.name.endswith(".bias") and item.name[:-len(".bias")] in quantized_parents:
            classified.append(
                TensorClassification(item.name, item.descriptor, "learned_bias", module_path=item.name[:-len(".bias")])
            )
        else:
            classified.append(item)

    counts: dict[str, int] = {}
    for item in classified:
        counts[item.role] = counts.get(item.role, 0) + 1
    expected_counts = {
        "recomputed": 1,
        "q6_core_weight": 208,
        "q8_block_adaln_weight": 50,
        "learned_bias": 50,
        "ordinary": 226,
    }
    if counts != expected_counts:
        raise MonolithicSourceError(
            f"source classification count mismatch: actual={dict(sorted(counts.items()))}, "
            f"expected={expected_counts}"
        )
    if len(classified) - counts["recomputed"] != 534:
        raise MonolithicSourceError("stored source tensor count after rope exclusion is not 534")
    return SourceClassification(
        config,
        config_raw,
        tuple(classified),
        dict(sorted(counts.items())),
        qkv_layout,
    )


def classify_expected_source_config(config_raw: Mapping[str, object]) -> SourceClassification:
    """Derive and classify the exact canonical source topology from an admitted config only."""
    try:
        config = validate_config_contract(config_raw)
    except MonolithicSourceError:
        raise
    except Exception as exc:
        raise MonolithicSourceError(f"could not validate H3 config: {exc}") from exc

    descriptors: dict[str, TensorHeader] = {}
    offset = 0
    for name, shape in sorted(expected_tensor_shapes(config).items()):
        dtype = expected_source_dtype(name)
        element_count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        end = offset + element_count * DTYPE_BYTES[dtype]
        descriptors[name] = TensorHeader(name, dtype, shape, offset, end)
        offset = end
    surface = enumerate_fused_qkv_surface(tuple(descriptors))
    qkv_layout = _layout_decision(
        QKV_SOURCE_LAYOUT_RUNTIME_INTERLEAVED,
        surface,
        authorization="canonical-output-topology",
        source_identity=None,
    )
    return _classify_admitted_descriptors(config, dict(config_raw), descriptors, qkv_layout)


def classify_source(
    source: MonolithicSafetensorsSource,
    *,
    metadata_key: str = CONFIG_METADATA_KEY,
) -> SourceClassification:
    """Admit the complete beta topology and classify every raw source tensor."""
    try:
        config_raw = extract_embedded_config(source.metadata, metadata_key=metadata_key)
        config = validate_config_contract(config_raw)
    except MonolithicSourceError:
        raise
    except Exception as exc:
        raise MonolithicSourceError(f"could not validate embedded beta config: {exc}") from exc

    expected = expected_tensor_shapes(config)
    actual = set(source.tensor_names)
    expected_names = set(expected)
    missing = sorted(expected_names - actual)
    extra = sorted(actual - expected_names)
    if missing or extra:
        raise MonolithicSourceError(
            f"source topology does not match the canonical H3 contract; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    descriptors = {tensor.name: tensor for tensor in source.tensors}
    for name, shape in expected.items():
        descriptor = descriptors[name]
        expected_dtype = expected_source_dtype(name)
        if descriptor.dtype != expected_dtype:
            raise MonolithicSourceError(
                f"source tensor {name} has dtype {descriptor.dtype}; expected {expected_dtype}"
            )
        if descriptor.shape != shape:
            raise MonolithicSourceError(
                f"source tensor {name} has shape {descriptor.shape}; expected {shape}"
            )

    surface = enumerate_fused_qkv_surface(tuple(descriptors))
    qkv_layout = resolve_qkv_layout(source, surface)
    return _classify_admitted_descriptors(config, config_raw, descriptors, qkv_layout)


def decode_bfloat16_to_float32(raw: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Decode BF16 storage using uint16 bits; NumPy never receives a native BF16 dtype."""
    expected = int(np.prod(shape, dtype=np.int64)) if shape else 1
    if len(raw) != expected * DTYPE_BYTES["BF16"]:
        raise MonolithicSourceError(
            f"BF16 payload has {len(raw)} bytes; expected {expected * DTYPE_BYTES['BF16']}"
        )
    bits = np.frombuffer(raw, dtype=np.dtype("<u2"), count=expected)
    decoded = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return decoded.reshape(shape)

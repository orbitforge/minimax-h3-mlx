"""Raw safetensors header parsing and byte-range copying.

This module intentionally does not import NumPy or MLX.  Tensor payloads are copied as
opaque bytes so BF16 bit patterns and packed U32 quantized weights cannot be changed by a
framework conversion.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable


MAX_HEADER_BYTES = 100 * 1024 * 1024
COPY_CHUNK_BYTES = 8 * 1024 * 1024

# These are the safetensors element widths used by the released transformer and
# the standard scalar types accepted by the local reader.  U32 is a packed
# quantized payload, not a logical four-byte weight, but its on-disk element
# width is still four bytes.
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


@dataclass(frozen=True)
class TensorHeader:
    name: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class SafetensorsHeader:
    path: Path
    data_start: int
    metadata: dict[str, str]
    tensors: tuple[TensorHeader, ...]

    def tensor_map(self) -> dict[str, TensorHeader]:
        return {tensor.name: tensor for tensor in self.tensors}


def read_safetensors_header(path: Path) -> SafetensorsHeader:
    with path.open("rb") as handle:
        raw_size = handle.read(8)
        if len(raw_size) != 8:
            raise ValueError(f"{path} is truncated before its safetensors header")
        header_size = struct.unpack("<Q", raw_size)[0]
        if header_size > MAX_HEADER_BYTES:
            raise ValueError(f"{path} has an unreasonable safetensors header size: {header_size}")
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError(f"{path} is truncated inside its safetensors header")
    try:
        decoded = json.loads(raw_header)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} has invalid safetensors JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{path} safetensors header is not an object")
    metadata = decoded.pop("__metadata__", {})
    if not isinstance(metadata, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
        raise ValueError(f"{path} has invalid __metadata__; safetensors metadata must be string pairs")
    tensors: list[TensorHeader] = []
    for name, item in decoded.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError(f"{path} has an invalid tensor entry for {name!r}")
        dtype = item.get("dtype")
        shape = item.get("shape")
        offsets = item.get("data_offsets")
        if not isinstance(dtype, str) or dtype not in DTYPE_BYTES or not isinstance(shape, list) or not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{path} has an invalid tensor descriptor for {name}")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in shape + offsets):
            raise ValueError(f"{path} has non-negative integer requirements violated for {name}")
        start, end = offsets
        if end < start:
            raise ValueError(f"{path} has reversed data offsets for {name}")
        element_count = 1
        for dimension in shape:
            element_count *= dimension
        expected_bytes = element_count * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(
                f"{path} tensor {name} has {end - start} payload bytes; "
                f"dtype {dtype} and shape {shape} require {expected_bytes}"
            )
        tensors.append(TensorHeader(name, dtype, tuple(shape), start, end))
    tensors.sort(key=lambda tensor: tensor.name)
    data_start = 8 + header_size
    file_size = path.stat().st_size
    payload_bytes = file_size - data_start
    if payload_bytes < 0:
        raise ValueError(f"{path} is shorter than its declared header")
    if any(data_start + tensor.end > file_size for tensor in tensors):
        raise ValueError(f"{path} has tensor data outside the file")
    by_offset = sorted(tensors, key=lambda tensor: (tensor.start, tensor.end, tensor.name))
    expected_start = 0
    for tensor in by_offset:
        if tensor.start != expected_start:
            if tensor.start < expected_start:
                raise ValueError(f"{path} has overlapping tensor data ranges near {tensor.name}")
            raise ValueError(f"{path} has a hole before tensor {tensor.name}")
        expected_start = tensor.end
    if expected_start != payload_bytes:
        raise ValueError(f"{path} has trailing unindexed payload bytes: {payload_bytes - expected_start}")
    return SafetensorsHeader(path, data_start, dict(metadata), tuple(tensors))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(COPY_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_range(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = end - start
        while remaining:
            chunk = handle.read(min(COPY_CHUNK_BYTES, remaining))
            if not chunk:
                raise ValueError(f"unexpected EOF while hashing {path} bytes {start}:{end}")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def copy_range(source: BinaryIO, destination: BinaryIO, start: int, end: int) -> str:
    digest = hashlib.sha256()
    source.seek(start)
    remaining = end - start
    while remaining:
        chunk = source.read(min(COPY_CHUNK_BYTES, remaining))
        if not chunk:
            raise ValueError(f"unexpected EOF while copying byte range {start}:{end}")
        destination.write(chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    return digest.hexdigest()


def _header_bytes(tensors: Iterable[TensorHeader], metadata: dict[str, str]) -> bytes:
    offset = 0
    entries: dict[str, object] = {"__metadata__": dict(sorted(metadata.items()))}
    for tensor in sorted(tensors, key=lambda item: item.name):
        entries[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes
    return json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_safetensors_fast(
    destination: Path,
    selected: Iterable[tuple[Path, SafetensorsHeader, TensorHeader]],
    metadata: dict[str, str],
) -> tuple[str, dict[str, str]]:
    """Like write_safetensors, with parsed headers supplied so each shard is read once per tensor."""
    ordered = sorted(selected, key=lambda item: item[2].name)
    raw_header = _header_bytes((tensor for _, _, tensor in ordered), metadata)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensor_checksums: dict[str, str] = {}
    with destination.open("wb") as output:
        output.write(struct.pack("<Q", len(raw_header)))
        output.write(raw_header)
        for source_path, source_header, tensor in ordered:
            with source_path.open("rb") as source:
                checksum = copy_range(
                    source,
                    output,
                    source_header.data_start + tensor.start,
                    source_header.data_start + tensor.end,
                )
            tensor_checksums[tensor.name] = checksum
    return sha256_file(destination), tensor_checksums

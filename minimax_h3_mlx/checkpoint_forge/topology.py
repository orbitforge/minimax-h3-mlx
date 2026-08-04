"""Source checkpoint topology and tensor-role classification."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .tensor_io import SafetensorsHeader, TensorHeader, read_safetensors_header, sha256_file, sha256_range


FORMAT_IDENTIFIER = "minimax-h3-mlx-streamed-adaln-v1"
BLOCK_COUNT = 50
BLOCK_ROLES = ("bias", "biases", "scales", "weight")
BLOCK_PATTERN = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.(bias|biases|scales|weight)$")
BLOCK_PREFIX_PATTERN = re.compile(r"^blocks\.(\d+)\.adaln_proj\.linear\.")
FINAL_PATTERN = re.compile(r"^final_layer\.adaln_proj\.linear\.(bias|weight)$")


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shard: str
    path: Path
    header: SafetensorsHeader
    descriptor: TensorHeader

    @property
    def dtype(self) -> str:
        return self.descriptor.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.descriptor.shape

    @property
    def nbytes(self) -> int:
        return self.descriptor.nbytes

    @property
    def data_start(self) -> int:
        return self.header.data_start + self.descriptor.start

    def checksum(self) -> str:
        return sha256_range(self.path, self.data_start, self.header.data_start + self.descriptor.end)


@dataclass(frozen=True)
class SourceTopology:
    root: Path
    index_path: Path
    index: dict
    config_path: Path
    quant_config_path: Path | None
    config: dict
    quant_config: dict | None
    records: tuple[TensorRecord, ...]
    ordinary: tuple[TensorRecord, ...]
    final_adaln: tuple[TensorRecord, ...]
    block_adaln: dict[int, tuple[TensorRecord, ...]]
    shard_headers: dict[str, SafetensorsHeader]

    @classmethod
    def load(cls, root: Path) -> "SourceTopology":
        root = root.expanduser().resolve()
        if not root.is_dir():
            raise ValueError(f"source checkpoint directory does not exist: {root}")
        index_path = root / "model.safetensors.index.json"
        config_path = root / "config.json"
        if not index_path.is_file() or not config_path.is_file():
            raise ValueError(f"source must contain config.json and model.safetensors.index.json: {root}")
        try:
            index = json.loads(index_path.read_text())
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"source metadata is not valid JSON: {exc}") from exc
        if not isinstance(index, dict) or not isinstance(config, dict):
            raise ValueError("source index and config must be JSON objects")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("source safetensors index has no weight_map")
        shard_names = sorted(set(weight_map.values()))
        if not all(isinstance(name, str) and Path(name).name == name for name in shard_names):
            raise ValueError("source index contains unsafe or non-string shard names")
        shard_headers: dict[str, SafetensorsHeader] = {}
        records: list[TensorRecord] = []
        seen: set[str] = set()
        for shard in shard_names:
            path = root / shard
            if not path.is_file():
                raise ValueError(f"source index references missing shard: {path}")
            parsed = read_safetensors_header(path)
            shard_headers[shard] = parsed
            for descriptor in parsed.tensors:
                if descriptor.name in seen:
                    raise ValueError(f"duplicate logical tensor across source shards: {descriptor.name}")
                seen.add(descriptor.name)
                records.append(TensorRecord(descriptor.name, shard, path, parsed, descriptor))
        if set(weight_map) != seen:
            missing = sorted(set(weight_map) - seen)
            extra = sorted(seen - set(weight_map))
            raise ValueError(f"source index/header mismatch; missing={missing[:3]} extra={extra[:3]}")
        for name, shard in weight_map.items():
            if records_by_name(records)[name].shard != shard:
                raise ValueError(f"source index maps {name} to {shard}, but header places it elsewhere")
        records.sort(key=lambda record: record.name)
        ordinary: list[TensorRecord] = []
        final_adaln: list[TensorRecord] = []
        blocks: dict[int, list[TensorRecord]] = {index: [] for index in range(BLOCK_COUNT)}
        for record in records:
            block_match = BLOCK_PATTERN.match(record.name)
            if block_match:
                block = int(block_match.group(1))
                if block not in blocks:
                    raise ValueError(f"unexpected block AdaLN index {block} in {record.name}")
                blocks[block].append(record)
            elif BLOCK_PREFIX_PATTERN.match(record.name):
                raise ValueError(f"unexpected block AdaLN companion: {record.name}")
            elif FINAL_PATTERN.match(record.name):
                final_adaln.append(record)
            else:
                ordinary.append(record)
        for block, block_records in blocks.items():
            roles = [BLOCK_PATTERN.match(record.name).group(2) for record in block_records]
            if sorted(roles) != sorted(BLOCK_ROLES):
                raise ValueError(f"block {block} must contain exactly {BLOCK_ROLES}; found {sorted(roles)}")
            block_records.sort(key=lambda record: record.name)
        if len(final_adaln) != 2:
            raise ValueError(f"expected exactly two final-layer AdaLN tensors, found {len(final_adaln)}")
        quant_config_path = root / "quant_config.json" if (root / "quant_config.json").is_file() else None
        quant_config = None
        if quant_config_path:
            try:
                quant_config = json.loads(quant_config_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"source quant_config.json is not valid JSON: {exc}") from exc
            if not isinstance(quant_config, dict):
                raise ValueError("source quant_config.json must be a JSON object")
        topology = cls(
            root,
            index_path,
            index,
            config_path,
            quant_config_path,
            config,
            quant_config,
            tuple(records),
            tuple(ordinary),
            tuple(sorted(final_adaln, key=lambda record: record.name)),
            {block: tuple(records) for block, records in blocks.items()},
            shard_headers,
        )
        topology.block_projection_metadata()
        return topology

    @property
    def source_tensor_count(self) -> int:
        return len(self.records)

    @property
    def block_tensor_count(self) -> int:
        return sum(len(records) for records in self.block_adaln.values())

    @property
    def base_records(self) -> tuple[TensorRecord, ...]:
        return tuple(sorted((*self.ordinary, *self.final_adaln), key=lambda record: record.name))

    @property
    def logical_payload_bytes(self) -> int:
        return sum(record.nbytes for record in self.records)

    @property
    def physical_source_bytes(self) -> int:
        return sum((self.root / shard).stat().st_size for shard in self.shard_headers)

    def source_checksums(self) -> dict[str, str]:
        return {shard: sha256_file(self.root / shard) for shard in sorted(self.shard_headers)}

    def block_projection_metadata(self) -> dict[str, object]:
        """Return logical AdaLN projection metadata proven by config and descriptors."""
        logical_input = self.config.get("time_embed_dim")
        logical_output = self.config.get("adaln_out_features")
        quant_config = self.quant_config or {}
        bits = quant_config.get("adaln_bits", quant_config.get("bits"))
        group_size = quant_config.get("group_size")
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in (logical_input, logical_output, bits, group_size)):
            raise ValueError("cannot prove AdaLN logical dimensions and quantization metadata from source")
        if bits > 32 or 32 % bits:
            raise ValueError(f"unsupported AdaLN quantization bits for packed U32 weights: {bits}")
        if logical_input % group_size:
            raise ValueError("AdaLN logical input features are not divisible by quantization group size")
        pack_factor = 32 // bits
        if logical_input % pack_factor:
            raise ValueError("AdaLN logical input features are not divisible by the U32 packing factor")
        projection = {
            "quantization_bits": bits,
            "quantization_group_size": group_size,
            "logical_input_features": logical_input,
            "logical_output_features": logical_output,
            "packed_weight_shape": [logical_output, logical_input // pack_factor],
            "scales_shape": [logical_output, logical_input // group_size],
            "quantization_biases_shape": [logical_output, logical_input // group_size],
            "learned_bias_shape": [logical_output],
        }
        expected_shapes = {
            "weight": tuple(projection["packed_weight_shape"]),
            "scales": tuple(projection["scales_shape"]),
            "biases": tuple(projection["quantization_biases_shape"]),
            "bias": tuple(projection["learned_bias_shape"]),
        }
        for block, records in self.block_adaln.items():
            for record in records:
                role = record.name.rsplit(".", 1)[-1]
                if record.shape != expected_shapes[role]:
                    raise ValueError(
                        f"block {block} AdaLN {role} shape {record.shape} does not match "
                        f"proven projection shape {expected_shapes[role]}"
                    )
        return projection


def records_by_name(records: list[TensorRecord]) -> dict[str, TensorRecord]:
    return {record.name: record for record in records}


def parse_block_selection(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("--blocks contains an empty selection")
        if "-" in part:
            pieces = part.split("-")
            if len(pieces) != 2:
                raise ValueError(f"invalid block range: {part}")
            start, end = (int(piece) for piece in pieces)
            if end < start:
                raise ValueError(f"reversed block range: {part}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    if not selected or any(block < 0 or block >= BLOCK_COUNT for block in selected):
        raise ValueError(f"block selection must be within 0..{BLOCK_COUNT - 1}")
    return tuple(sorted(selected))

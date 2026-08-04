"""Exact verification of derived raw tensor payloads and manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .manifest import load_json
from .tensor_io import read_safetensors_header, sha256_file, sha256_range
from .topology import BLOCK_COUNT, FORMAT_IDENTIFIER, SourceTopology, TensorRecord


SCHEMA_VERSION = 1
SEMANTIC_ROLES = {
    "weight": "packed_weight",
    "scales": "scales",
    "biases": "quantization_biases",
    "bias": "learned_bias",
}


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    checked_tensors: int
    checked_files: int
    message: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}


def _compare_record(source: TensorRecord, output_path: Path, output_header, output_descriptor) -> None:
    if output_descriptor.dtype != source.dtype or output_descriptor.shape != source.shape:
        raise ValueError(f"descriptor mismatch for {source.name}")
    source_hash = source.checksum()
    output_hash = sha256_range(
        output_path,
        output_header.data_start + output_descriptor.start,
        output_header.data_start + output_descriptor.end,
    )
    if source_hash != output_hash:
        raise ValueError(f"exact payload checksum mismatch for {source.name}")


def _verify_source_snapshot(topology: SourceTopology, manifest: dict) -> None:
    snapshot = manifest.get("source_snapshot")
    _require(isinstance(snapshot, dict), "missing source snapshot")
    expected_shards = sorted(topology.shard_headers)
    actual_shards = snapshot.get("shards")
    _require(isinstance(actual_shards, dict) and sorted(actual_shards) == expected_shards, "source shard snapshot is incomplete or unexpected")
    for shard in expected_shards:
        expected = actual_shards[shard]
        _require(isinstance(expected, dict) and set(expected) == {"size", "mtime_ns"}, f"invalid source snapshot for shard {shard}")
        path = topology.root / shard
        actual = {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        _require(actual == expected, f"source shard changed after forge: {shard}")
    expected_files = {"model.safetensors.index.json", "config.json"}
    if topology.quant_config_path:
        expected_files.add("quant_config.json")
    actual_files = snapshot.get("files")
    _require(isinstance(actual_files, dict) and set(actual_files) == expected_files, "source metadata snapshot is incomplete or unexpected")
    for relative in sorted(expected_files):
        path = topology.root / relative
        _require(sha256_file(path) == actual_files[relative], f"source metadata file changed after forge: {relative}")


def _expected_tensor_manifest(record: TensorRecord, projection: dict[str, object]) -> dict[str, object]:
    role = record.name.rsplit(".", 1)[-1]
    semantic_role = SEMANTIC_ROLES[role]
    quantized = semantic_role != "learned_bias"
    return {
        "tensor_key": record.name,
        "tensor_role": semantic_role,
        "source_shard": record.shard,
        "source_dtype": record.dtype,
        "source_shape": list(record.shape),
        "byte_count": record.nbytes,
        "tensor_checksum": record.checksum(),
        "quantization_format": "affine" if quantized else "unquantized",
        "quantization_bits": projection["quantization_bits"] if quantized else None,
        "group_size": projection["quantization_group_size"] if quantized else None,
    }


def _expected_payload_files(
    output: Path,
    topology: SourceTopology,
    bounded: bool,
    selected_blocks: tuple[int, ...],
    verification_status: str,
) -> set[str]:
    files = {"config.json", "adaln/manifest.json"}
    if topology.quant_config_path:
        files.add("quant_config.json")
    if bounded:
        files.add("base/classification.json")
    else:
        index = load_json(output / "base" / "model.safetensors.index.json")
        _require(isinstance(index, dict), "complete output base index must be a JSON object")
        weight_map = index.get("weight_map")
        _require(isinstance(weight_map, dict) and weight_map, "complete output base index has no weight_map")
        for shard in weight_map.values():
            _require(isinstance(shard, str) and Path(shard).name == shard, "complete output base index contains an unsafe shard name")
            files.add(f"base/{shard}")
        files.add("base/model.safetensors.index.json")
    files.update(f"adaln/block-{block:03d}.safetensors" for block in selected_blocks)
    if verification_status == "verified":
        files.add("verification_report.txt")
    return files


def _verify_base(
    output: Path,
    topology: SourceTopology,
    bounded: bool,
) -> int:
    expected_base = {record.name: record for record in topology.base_records}
    if bounded:
        classification_path = output / "base" / "classification.json"
        classification = load_json(classification_path)
        _require(isinstance(classification, dict), "bounded base classification must be a JSON object")
        _require(classification.get("bounded") is True, "bounded base classification is not marked bounded")
        _require(classification.get("tensor_count") == len(expected_base), "bounded base tensor count mismatch")
        _require(classification.get("byte_count") == sum(record.nbytes for record in expected_base.values()), "bounded base byte count mismatch")
        _require(classification.get("tensor_names") == sorted(expected_base), "bounded base classification does not match source base tensor set")
        _require(not (output / "base" / "model.safetensors.index.json").exists(), "bounded output unexpectedly contains a base safetensors index")
        _require(not list((output / "base").glob("*.safetensors")), "bounded output unexpectedly contains base payload shards")
        return 0

    index_path = output / "base" / "model.safetensors.index.json"
    base_index = load_json(index_path)
    _require(isinstance(base_index, dict), "complete output base index must be a JSON object")
    weight_map = base_index.get("weight_map")
    _require(isinstance(weight_map, dict) and set(weight_map) == set(expected_base), "derived base tensor set does not match ordinary plus final AdaLN tensors")
    metadata = base_index.get("metadata")
    _require(isinstance(metadata, dict) and metadata.get("total_size") == sum(record.nbytes for record in expected_base.values()), "derived base index byte total mismatch")
    _require(all(isinstance(shard, str) for shard in weight_map.values()), "derived base index contains a non-string shard name")
    shard_names = sorted(set(weight_map.values()))
    _require(all(Path(shard).name == shard for shard in shard_names), "derived base index contains an unsafe shard name")
    output_headers = {}
    for shard in shard_names:
        path = output / "base" / shard
        _require(path.is_file(), f"missing indexed base shard: {shard}")
        output_headers[shard] = read_safetensors_header(path)
    seen: set[str] = set()
    for shard, header in output_headers.items():
        for descriptor in header.tensors:
            _require(descriptor.name in expected_base, f"unindexed or unexpected tensor in derived base shard: {descriptor.name}")
            _require(descriptor.name not in seen, f"duplicate derived base tensor: {descriptor.name}")
            _require(weight_map[descriptor.name] == shard, f"derived base index places {descriptor.name} in the wrong shard")
            _compare_record(expected_base[descriptor.name], output / "base" / shard, header, descriptor)
            seen.add(descriptor.name)
    _require(seen == set(expected_base), "derived base shards do not cover every expected tensor")
    return len(expected_base)


def _verify_sidecars(
    output: Path,
    topology: SourceTopology,
    bounded: bool,
    selected_blocks: tuple[int, ...],
    sidecar_manifest: dict,
) -> int:
    _require(sidecar_manifest.get("format_identifier") == FORMAT_IDENTIFIER, "unexpected AdaLN sidecar-manifest format identifier")
    _require(sidecar_manifest.get("schema_version") == SCHEMA_VERSION, "unexpected AdaLN sidecar-manifest schema version")
    _require(sidecar_manifest.get("bounded") is bounded, "AdaLN sidecar-manifest bounded flag disagrees with root manifest")
    blocks = sidecar_manifest.get("blocks")
    _require(isinstance(blocks, dict) and set(blocks) == {str(block) for block in selected_blocks}, "AdaLN sidecar-manifest block set is stale or inconsistent")
    projection = topology.block_projection_metadata()
    sidecar_root = output / "adaln"
    expected_sidecars = {f"block-{block:03d}.safetensors" for block in selected_blocks}
    actual_sidecars = {path.name for path in sidecar_root.glob("block-*.safetensors")}
    _require(actual_sidecars == expected_sidecars, f"sidecar set mismatch: expected {sorted(expected_sidecars)}, found {sorted(actual_sidecars)}")
    checked = 0
    for block in selected_blocks:
        filename = f"block-{block:03d}.safetensors"
        entry = blocks[str(block)]
        _require(isinstance(entry, dict), f"invalid sidecar-manifest entry for block {block}")
        _require(entry.get("block_index") == block, f"invalid block index in sidecar manifest for block {block}")
        _require(entry.get("sidecar_filename") == filename, f"invalid sidecar filename for block {block}")
        _require(entry.get("projection") == projection, f"projection metadata mismatch for block {block}")
        parsed = read_safetensors_header(sidecar_root / filename)
        expected_records = {record.name: record for record in topology.block_adaln[block]}
        _require(set(parsed.tensor_map()) == set(expected_records), f"sidecar tensor set mismatch for block {block}")
        tensors = entry.get("tensors")
        _require(isinstance(tensors, list) and all(isinstance(item, dict) for item in tensors), f"invalid tensor metadata in sidecar manifest for block {block}")
        _require([item.get("tensor_key") for item in tensors] == sorted(expected_records), f"sidecar tensor manifest ordering mismatch for block {block}")
        for item in tensors:
            name = item.get("tensor_key")
            _require(name in expected_records, f"unexpected tensor metadata in sidecar manifest: {name}")
            expected_item = _expected_tensor_manifest(expected_records[name], projection)
            _require(item == expected_item, f"sidecar tensor metadata mismatch for {name}")
            descriptor = parsed.tensor_map().get(name)
            _require(descriptor is not None, f"sidecar tensor missing from payload: {name}")
            _compare_record(expected_records[name], sidecar_root / filename, parsed, descriptor)
            checked += 1
        _require(entry.get("sidecar_checksum") == sha256_file(sidecar_root / filename), f"sidecar checksum mismatch for block {block}")
    return checked


def verify_checkpoint(
    source: Path,
    output: Path,
    manifest_path: Path | None = None,
    allow_pending: bool = False,
) -> VerificationResult:
    topology = SourceTopology.load(source)
    output = output.expanduser().resolve()
    _require(output.is_dir(), f"derived output directory does not exist: {output}")
    manifest_path = manifest_path or (output / "conversion_manifest.json")
    _require(manifest_path.is_file(), f"missing completion manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    _require(isinstance(manifest, dict), "completion manifest must be a JSON object")
    _require(manifest.get("format_identifier") == FORMAT_IDENTIFIER, "unexpected derived checkpoint format identifier")
    _require(manifest.get("schema_version") == SCHEMA_VERSION, "unexpected derived checkpoint schema version")
    status = manifest.get("verification_status")
    _require(status == "verified" or (allow_pending and status == "pending"), "derived checkpoint is not marked verified")
    if status == "pending":
        _require(manifest.get("verification_timestamp") is None, "pending completion metadata has a verification timestamp")
    else:
        _require(isinstance(manifest.get("verification_timestamp"), str) and manifest["verification_timestamp"], "verified completion metadata has no verification timestamp")
    _require(manifest.get("completion_record") == "conversion_manifest.json is the completion record and is intentionally excluded from per_file_checksums", "stale completion-record metadata")
    _require(manifest.get("original_checkpoint_modified") is False, "derived manifest claims the source checkpoint was modified")
    _verify_source_snapshot(topology, manifest)

    bounded = manifest.get("bounded")
    _require(isinstance(bounded, bool), "derived manifest bounded flag is invalid")
    selected_raw = manifest.get("selected_blocks")
    _require(isinstance(selected_raw, list) and all(isinstance(block, int) and not isinstance(block, bool) for block in selected_raw), "derived manifest selected block list is invalid")
    selected_blocks = tuple(selected_raw)
    _require(selected_blocks == tuple(sorted(set(selected_blocks))), "derived manifest selected blocks are not uniquely ordered")
    _require(selected_blocks and all(0 <= block < BLOCK_COUNT for block in selected_blocks), "derived manifest has an invalid selected block set")
    if not bounded:
        _require(selected_blocks == tuple(range(BLOCK_COUNT)), "complete derived output must contain all 50 AdaLN sidecars")

    source_shards = manifest.get("source_shards")
    expected_source_shards = [{"filename": shard, "size_bytes": (topology.root / shard).stat().st_size} for shard in sorted(topology.shard_headers)]
    _require(source_shards == expected_source_shards, "root source shard metadata is stale or inconsistent")
    _require(manifest.get("source_tensor_count") == topology.source_tensor_count, "root source tensor count mismatch")
    _require(manifest.get("source_logical_payload_byte_count") == topology.logical_payload_bytes, "root source logical byte total mismatch")
    _require(manifest.get("source_physical_byte_count") == topology.physical_source_bytes, "root source physical byte total mismatch")
    _require(manifest.get("source_configuration_checksum") == sha256_file(topology.config_path), "source config checksum mismatch")
    _require(manifest.get("source_quantization_configuration_checksum") == (sha256_file(topology.quant_config_path) if topology.quant_config_path else None), "source quant-config checksum mismatch")
    _require(manifest.get("source_safetensors_index_checksum") == sha256_file(topology.index_path), "source index checksum mismatch")
    source_checkpoint = manifest.get("source_checkpoint")
    _require(isinstance(source_checkpoint, dict) and source_checkpoint.get("logical_identity") == sha256_file(topology.index_path), "source logical identity mismatch")

    expected_base_count = len(topology.base_records)
    expected_base_bytes = 0 if bounded else sum(record.nbytes for record in topology.base_records)
    expected_sidecar_count = len(selected_blocks)
    expected_sidecar_tensor_count = sum(len(topology.block_adaln[block]) for block in selected_blocks)
    expected_sidecar_bytes = sum(record.nbytes for block in selected_blocks for record in topology.block_adaln[block])
    _require(manifest.get("derived_base_tensor_count") == expected_base_count, "root derived base tensor count mismatch")
    _require(manifest.get("derived_base_byte_count") == expected_base_bytes, "root derived base byte total mismatch")
    _require(manifest.get("sidecar_count") == expected_sidecar_count, "root sidecar count mismatch")
    _require(manifest.get("sidecar_tensor_count") == expected_sidecar_tensor_count, "root sidecar tensor count mismatch")
    _require(manifest.get("sidecar_byte_count") == expected_sidecar_bytes, "root sidecar byte total mismatch")
    _require(manifest.get("total_logical_tensor_count") == expected_base_count + expected_sidecar_tensor_count, "root logical tensor count mismatch")
    expected_final = [
        {"key": record.name, "shape": list(record.shape), "dtype": record.dtype, "byte_count": record.nbytes, "checksum": record.checksum()}
        for record in topology.final_adaln
    ]
    _require(manifest.get("final_layer_adaln") == expected_final, "root final-layer AdaLN metadata mismatch")

    checked_base = _verify_base(output, topology, bounded)
    sidecar_manifest_path = output / "adaln" / "manifest.json"
    sidecar_manifest = load_json(sidecar_manifest_path)
    _require(isinstance(sidecar_manifest, dict), "AdaLN sidecar manifest must be a JSON object")
    checked_sidecars = _verify_sidecars(output, topology, bounded, selected_blocks, sidecar_manifest)

    expected_files = _expected_payload_files(output, topology, bounded, selected_blocks, status)
    completion_name = manifest_path.relative_to(output).as_posix() if manifest_path.is_relative_to(output) else manifest_path.name
    expected_tree = expected_files | {completion_name}
    if status == "pending":
        _require(completion_name == "conversion_manifest.json.pending", "pending completion record has an unexpected name")
    else:
        _require(completion_name == "conversion_manifest.json", "verified completion record has an unexpected name")
    _require(_relative_files(output) == expected_tree, "derived output contains missing or unexpected payload files")

    checksums = manifest.get("per_file_checksums")
    _require(isinstance(checksums, dict) and set(checksums) == expected_files, "per_file_checksums is missing expected files or contains unexpected entries")
    for relative, expected_checksum in sorted(checksums.items()):
        path = output / relative
        _require(path.is_file(), f"per-file checksum references missing artifact: {relative}")
        _require(sha256_file(path) == expected_checksum, f"per-file checksum mismatch: {relative}")

    if status == "verified":
        report_path = output / "verification_report.txt"
        report = load_json(report_path)
        _require(isinstance(report, dict), "verification report must be a JSON object")
        _require(report.get("status") == "verified" and report.get("source_unchanged") is True, "stale or inconsistent verification report")
        _require(report.get("checked_source_tensors") == topology.source_tensor_count, "verification report source tensor count mismatch")
        _require(report.get("checked_derived_tensors") == manifest["total_logical_tensor_count"], "verification report derived tensor count mismatch")

    checked_files = len(expected_tree)
    return VerificationResult(True, checked_base + checked_sidecars, checked_files, "exact verification passed")

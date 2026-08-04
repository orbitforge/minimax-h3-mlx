"""Atomic construction of the complete or bounded derived checkpoint."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .manifest import dump_json
from .tensor_io import sha256_file, write_safetensors_fast
from .topology import BLOCK_COUNT, FORMAT_IDENTIFIER, SourceTopology, TensorRecord
from .verify import verify_checkpoint


TARGET_SHARD_BYTES = 4_000_000_000
SAFETY_OVERHEAD_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class ForgeOptions:
    source: Path
    output: Path
    dry_run: bool = False
    verify_only: bool = False
    force: bool = False
    blocks: tuple[int, ...] | None = None
    target_shard_bytes: int = TARGET_SHARD_BYTES


@dataclass(frozen=True)
class ForgeResult:
    output: Path | None
    bounded: bool
    selected_blocks: tuple[int, ...]
    message: str


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_identity() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repository, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"], cwd=repository, check=True, capture_output=True, text=True
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unavailable", None
    return {"commit": commit or "unavailable", "dirty": dirty}


def _available_bytes(path: Path) -> int:
    anchor = path if path.exists() else path.parent
    while not anchor.exists():
        anchor = anchor.parent
    return shutil.disk_usage(anchor).free


def _resolve_and_protect_paths(source: Path, output: Path) -> tuple[Path, Path]:
    """Resolve paths and reject source/output overlap before any mutation."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve(strict=False)
    output_parent = output.parent.resolve(strict=False)
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("source and output checkpoint paths must be disjoint")
    if output_parent == source or source in output_parent.parents:
        raise ValueError("temporary conversion directory would be created inside the source checkpoint")
    return source, output


def _base_shards(records: tuple[TensorRecord, ...], target: int) -> list[list[TensorRecord]]:
    shards: list[list[TensorRecord]] = []
    current: list[TensorRecord] = []
    current_size = 0
    for record in records:
        if current and current_size + record.nbytes > target:
            shards.append(current)
            current = []
            current_size = 0
        current.append(record)
        current_size += record.nbytes
    if current:
        shards.append(current)
    return shards


def _source_snapshot(topology: SourceTopology) -> dict:
    return {
        "shards": {
            shard: {"size": (topology.root / shard).stat().st_size, "mtime_ns": (topology.root / shard).stat().st_mtime_ns}
            for shard in sorted(topology.shard_headers)
        },
        "files": {
            "model.safetensors.index.json": sha256_file(topology.index_path),
            "config.json": sha256_file(topology.config_path),
            **({"quant_config.json": sha256_file(topology.quant_config_path)} if topology.quant_config_path else {}),
        },
    }


def _space_report(
    topology: SourceTopology,
    output: Path,
    force: bool,
    base_bytes: int,
    sidecar_bytes: int,
    target_shard_bytes: int = TARGET_SHARD_BYTES,
) -> dict:
    expected_final = base_bytes + sidecar_bytes
    existing_bytes = 0
    if output.exists():
        existing_bytes = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
    replacement_overhead = existing_bytes if force and output.exists() else 0
    temporary_peak = expected_final + SAFETY_OVERHEAD_BYTES + replacement_overhead
    available = _available_bytes(output)
    return {
        "source_checkpoint_bytes": topology.physical_source_bytes,
        "expected_derived_base_bytes": base_bytes,
        "expected_adaln_sidecar_bytes": sidecar_bytes,
        "expected_final_derived_bytes": expected_final,
        "expected_temporary_peak_bytes": temporary_peak,
        "available_filesystem_bytes": available,
        "replacement_overhead_bytes": replacement_overhead,
        "sufficient_space": available >= temporary_peak,
        "target_base_shard_bytes": target_shard_bytes,
        "units": "bytes; GB uses decimal 10^9 and GiB uses 2^30",
    }


def _print_space_report(report: dict) -> None:
    for key, value in report.items():
        if key.endswith("_bytes"):
            print(f"{key}: {value:,} bytes ({value / 1_000_000_000:.3f} GB; {value / 2**30:.3f} GiB)")
        else:
            print(f"{key}: {value}")


def _record_manifest(record: TensorRecord, checksum: str, projection: dict[str, object]) -> dict:
    role = record.name.rsplit(".", 1)[-1]
    semantic_role = {
        "weight": "packed_weight",
        "scales": "scales",
        "biases": "quantization_biases",
        "bias": "learned_bias",
    }[role]
    quantized = semantic_role != "learned_bias"
    return {
        "tensor_key": record.name,
        "tensor_role": semantic_role,
        "source_shard": record.shard,
        "source_dtype": record.dtype,
        "source_shape": list(record.shape),
        "byte_count": record.nbytes,
        "tensor_checksum": checksum,
        "quantization_format": "affine" if quantized else "unquantized",
        "quantization_bits": projection["quantization_bits"] if quantized else None,
        "group_size": projection["quantization_group_size"] if quantized else None,
    }


def forge_checkpoint(options: ForgeOptions) -> ForgeResult:
    source, output = _resolve_and_protect_paths(options.source, options.output)
    topology = SourceTopology.load(source)
    selected_blocks = options.blocks or tuple(range(BLOCK_COUNT))
    if options.verify_only:
        result = verify_checkpoint(topology.root, output)
        print(f"{result.message}; tensors={result.checked_tensors}; files={result.checked_files}")
        return ForgeResult(output, bool(options.blocks), selected_blocks, result.message)
    if options.dry_run:
        report = _space_report(
            topology,
            output,
            options.force,
            sum(record.nbytes for record in topology.base_records) if options.blocks is None else 0,
            sum(record.nbytes for block in selected_blocks for record in topology.block_adaln[block]),
            options.target_shard_bytes,
        )
        print(f"source tensors: {topology.source_tensor_count}")
        print(f"ordinary tensors: {len(topology.ordinary)}")
        print(f"final-layer AdaLN tensors: {len(topology.final_adaln)}")
        print(f"block AdaLN tensors: {topology.block_tensor_count} across {len(topology.block_adaln)} blocks")
        print(f"base shard strategy: {options.target_shard_bytes:,} bytes target; {_base_shards(topology.base_records, options.target_shard_bytes).__len__()} shards")
        print(f"selected bounded blocks: {','.join(map(str, selected_blocks)) if options.blocks else 'all 50 (full conversion)'}")
        _print_space_report(report)
        if not report["sufficient_space"]:
            print("WARNING: insufficient free space for the requested temporary output")
        return ForgeResult(None, options.blocks is not None, selected_blocks, "dry-run complete")
    if output.exists() and not output.is_dir():
        raise ValueError(f"output exists but is not a directory: {output}")
    if output.exists() and not options.force:
        raise ValueError(f"output already exists; refusing overwrite without --force: {output}")
    base_bytes = sum(record.nbytes for record in topology.base_records) if options.blocks is None else 0
    sidecar_bytes = sum(record.nbytes for block in selected_blocks for record in topology.block_adaln[block])
    report = _space_report(topology, output, options.force, base_bytes, sidecar_bytes, options.target_shard_bytes)
    _print_space_report(report)
    if not report["sufficient_space"]:
        raise ValueError("insufficient filesystem space for safe atomic conversion")
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.incomplete-", dir=parent))
    backup: Path | None = None
    try:
        (temporary / "base").mkdir()
        (temporary / "adaln").mkdir()
        shutil.copyfile(topology.config_path, temporary / "config.json")
        if topology.quant_config_path:
            shutil.copyfile(topology.quant_config_path, temporary / "quant_config.json")
        base_files: dict[str, str] = {}
        base_index = {"metadata": {"total_size": base_bytes}, "weight_map": {}}
        if options.blocks is None:
            shards = _base_shards(topology.base_records, options.target_shard_bytes)
            width = max(5, len(str(len(shards))))
            for index, records in enumerate(shards, 1):
                filename = f"model-{index:0{width}d}-of-{len(shards):0{width}d}.safetensors"
                selected = [(record.path, record.header, record.descriptor) for record in records]
                checksum, _ = write_safetensors_fast(temporary / "base" / filename, selected, {"format": "mlx"})
                base_files[f"base/{filename}"] = checksum
                for record in records:
                    base_index["weight_map"][record.name] = filename
                print(f"base shard {index}/{len(shards)}: {sum(record.nbytes for record in records):,} bytes written")
            dump_json(temporary / "base" / "model.safetensors.index.json", base_index)
            base_files["base/model.safetensors.index.json"] = sha256_file(temporary / "base" / "model.safetensors.index.json")
        else:
            dump_json(temporary / "base" / "classification.json", {
                "bounded": True,
                "tensor_count": len(topology.base_records),
                "byte_count": sum(record.nbytes for record in topology.base_records),
                "tensor_names": [record.name for record in topology.base_records],
                "excluded_from_bounded_payload": "base safetensors shards",
            })
            base_files["base/classification.json"] = sha256_file(temporary / "base" / "classification.json")
        sidecar_entries: dict[str, dict] = {}
        projection = topology.block_projection_metadata()
        for block in selected_blocks:
            filename = f"block-{block:03d}.safetensors"
            records = topology.block_adaln[block]
            selected = [(record.path, record.header, record.descriptor) for record in records]
            checksum, tensor_checksums = write_safetensors_fast(temporary / "adaln" / filename, selected, {"format": "mlx", "block": str(block)})
            sidecar_entries[str(block)] = {
                "block_index": block,
                "sidecar_filename": filename,
                "sidecar_checksum": checksum,
                "projection": projection,
                "tensors": [_record_manifest(record, tensor_checksums[record.name], projection) for record in records],
            }
            print(f"sidecar block {selected_blocks.index(block) + 1}/{len(selected_blocks)}: {sum(record.nbytes for record in records):,} bytes written")
        dump_json(temporary / "adaln" / "manifest.json", {
            "format_identifier": FORMAT_IDENTIFIER,
            "schema_version": 1,
            "bounded": options.blocks is not None,
            "blocks": sidecar_entries,
        })
        file_checksums = dict(base_files)
        file_checksums["adaln/manifest.json"] = sha256_file(temporary / "adaln" / "manifest.json")
        for block in selected_blocks:
            filename = f"adaln/block-{block:03d}.safetensors"
            file_checksums[filename] = sha256_file(temporary / filename)
        file_checksums["config.json"] = sha256_file(temporary / "config.json")
        if topology.quant_config_path:
            file_checksums["quant_config.json"] = sha256_file(temporary / "quant_config.json")
        manifest = {
            "format_identifier": FORMAT_IDENTIFIER,
            "schema_version": 1,
            "bounded": options.blocks is not None,
            "selected_blocks": list(selected_blocks),
            "source_checkpoint": {"path": str(topology.root), "logical_identity": sha256_file(topology.index_path)},
            "source_tensor_count": topology.source_tensor_count,
            "source_logical_payload_byte_count": topology.logical_payload_bytes,
            "source_physical_byte_count": topology.physical_source_bytes,
            "source_shards": [{"filename": shard, "size_bytes": (topology.root / shard).stat().st_size} for shard in sorted(topology.shard_headers)],
            "source_configuration_checksum": sha256_file(topology.config_path),
            "source_quantization_configuration_checksum": sha256_file(topology.quant_config_path) if topology.quant_config_path else None,
            "source_safetensors_index_checksum": sha256_file(topology.index_path),
            "conversion_timestamp": _now(),
            "converter_repository": _git_identity(),
            "derived_base_tensor_count": len(topology.base_records),
            "derived_base_byte_count": base_bytes,
            "sidecar_count": len(selected_blocks),
            "sidecar_tensor_count": sum(len(topology.block_adaln[block]) for block in selected_blocks),
            "sidecar_byte_count": sidecar_bytes,
            "final_layer_adaln": [{"key": record.name, "shape": list(record.shape), "dtype": record.dtype, "byte_count": record.nbytes, "checksum": record.checksum()} for record in topology.final_adaln],
            "total_logical_tensor_count": len(topology.base_records) + sum(len(topology.block_adaln[block]) for block in selected_blocks),
            "verification_status": "pending",
            "verification_timestamp": None,
            "per_file_checksums": file_checksums,
            "original_checkpoint_modified": False,
            "source_snapshot": _source_snapshot(topology),
            "disk_space_report": report,
            "base_shard_strategy": {"target_bytes": options.target_shard_bytes, "expected_shards": len(_base_shards(topology.base_records, options.target_shard_bytes)) if options.blocks is None else 0},
            "completion_record": "conversion_manifest.json is the completion record and is intentionally excluded from per_file_checksums",
        }
        dump_json(temporary / "conversion_manifest.json.pending", manifest)
        # Verify the provisional tree before writing the completion manifest. The final completion
        # manifest is the last required artifact and is published only after this exact pass.
        print("verification phase: provisional artifact checks")
        verification = verify_checkpoint(
            topology.root,
            temporary,
            manifest_path=temporary / "conversion_manifest.json.pending",
            allow_pending=True,
        )
        if not verification.ok:
            raise ValueError(verification.message)
        dump_json(temporary / "verification_report.txt", {
            "status": "verified",
            "method": "raw safetensors payload checksums plus exact dtype/shape descriptors",
            "source_unchanged": True,
            "checked_source_tensors": topology.source_tensor_count,
            "checked_derived_tensors": manifest["total_logical_tensor_count"],
        })
        file_checksums["verification_report.txt"] = sha256_file(temporary / "verification_report.txt")
        manifest["verification_status"] = "verified"
        manifest["verification_timestamp"] = _now()
        manifest["per_file_checksums"] = file_checksums
        dump_json(temporary / "conversion_manifest.json", manifest)
        (temporary / "conversion_manifest.json.pending").unlink()
        print("verification phase: final artifact checks")
        verification = verify_checkpoint(topology.root, temporary)
        if not verification.ok:
            raise ValueError(verification.message)
        print("publication phase: replacing destination with verified output")
        if output.exists():
            backup = parent / f".{output.name}.previous-{next(tempfile._get_candidate_names())}"
            os.replace(output, backup)
            try:
                os.replace(temporary, output)
            except Exception:
                if backup.exists() and not output.exists():
                    os.replace(backup, output)
                raise
        else:
            os.replace(temporary, output)
        if backup is not None:
            try:
                shutil.rmtree(backup)
            except OSError as exc:
                print(f"warning: verified output published; prior backup retained at {backup}: {exc}")
        print(f"published verified output: {output}")
        print(f"exact verification: {verification.message}; tensors={verification.checked_tensors}; files={verification.checked_files}")
        return ForgeResult(output, options.blocks is not None, selected_blocks, "conversion and verification complete")
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        if backup is not None and not output.exists() and backup.exists():
            os.replace(backup, output)
        raise

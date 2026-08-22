"""MLX-free inspection of the MiniMax-H3 derived streamed-AdaLN format.

The loader and operator surfaces share this metadata-only contract.  It validates manifests,
indexes, filenames, and required sidecar structure without hashing or opening tensor payloads.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoint_forge.tensor_io import DTYPE_BYTES
from .checkpoint_forge.topology import BLOCK_COUNT, FORMAT_IDENTIFIER
from .config import DiTConfig
from .runtime_selection import (
    BETA_ADALN_BITS,
    BETA_CORE_BITS,
    BETA_DIT_CONFIG_VALUES,
    BETA_DIT_INTEGER_CONFIG_FIELDS,
    BETA_DIT_REQUIRED_CONFIG_FIELDS,
    BETA_GROUP_SIZE,
    BETA_QUANTIZED_LAYER_COUNTS,
)


RESIDENT_CONSTRUCTION = "resident"
CACHE_ONLY_CONSTRUCTION = "cache_only"
SUPPORTED_DERIVED_SCHEMA_VERSION = 1
DERIVED_BASE_TENSOR_COUNT = 850
DERIVED_SIDECAR_COUNT = 50
DERIVED_SIDECAR_TENSOR_COUNT = 200
DERIVED_BASE_SHARD_COUNT = 5
SIDECAR_ROLE_TO_SUFFIX = {
    "packed_weight": "weight",
    "scales": "scales",
    "quantization_biases": "biases",
    "learned_bias": "bias",
}
SIDECAR_DTYPE_NAMES = {
    "packed_weight": "U32",
    "scales": "BF16",
    "quantization_biases": "BF16",
    "learned_bias": "BF16",
}


@dataclass(frozen=True)
class CheckpointFormatInfo:
    """Validated, lightweight format routing information for a transformer checkpoint."""

    checkpoint_format: str
    derived_root: Path | None
    base_root: Path
    conversion_manifest_path: Path | None
    adaln_manifest_path: Path | None
    construction_mode: str
    base_shards: tuple[str, ...]


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _positive_integer(value: object, label: str) -> int:
    _require(type(value) is int and value > 0, f"{label} must be a positive integer")
    return value


def _expected_payload_byte_count(shape: list[int], dtype: str) -> int:
    element_count = 1
    for dimension in shape:
        element_count *= dimension
    return element_count * DTYPE_BYTES[dtype]


def validate_streamed_transformer_config(model_dir: str | Path) -> DiTConfig:
    """Validate the H3 architecture metadata consumed by ``DiTConfig.from_json``.

    ``DiTConfig`` supplies defaults for every field, so the runtime-selection contract separately
    requires every architecture field except its intentional ``rope_theta`` default.  The exact
    accepted values are shared with the named-runtime validator; this function only reads JSON and
    constructs the pure config object.
    """
    root = Path(model_dir)
    config_path = root / "config.json"
    raw = _read_json(config_path, "derived transformer config")
    missing = sorted(BETA_DIT_REQUIRED_CONFIG_FIELDS - set(raw))
    _require(
        not missing,
        "derived transformer config is missing required DiT fields: " + ", ".join(missing),
    )
    for field in BETA_DIT_INTEGER_CONFIG_FIELDS:
        _require(
            type(raw.get(field)) is int,
            f"derived transformer config field {field!r} must be an integer",
        )
    patch_size = raw.get("patch_size")
    _require(
        isinstance(patch_size, (list, tuple))
        and len(patch_size) == 3
        and all(type(value) is int for value in patch_size),
        "derived transformer config patch_size must be a three-integer sequence",
    )
    try:
        parsed = DiTConfig.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"derived transformer config cannot be interpreted by DiTConfig: {config_path}: {exc}"
        ) from exc

    for field, expected in BETA_DIT_CONFIG_VALUES.items():
        actual = getattr(parsed, field)
        if isinstance(expected, tuple):
            actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
        _require(
            actual == expected,
            f"derived transformer config field {field!r} is not accepted: "
            f"expected {expected!r}, got {actual!r}",
        )
    return parsed


def validate_streamed_quantization_config(model_dir: str | Path) -> dict[str, Any]:
    """Validate the Q6/Q8 recipe required by the derived transformer structure."""
    root = Path(model_dir)
    quant_path = root / "quant_config.json"
    quant_config = _read_json(quant_path, "derived quantization config")
    _require(
        quant_config.get("bits") == BETA_CORE_BITS,
        "derived quantization config core bits are not Q6",
    )
    _require(
        quant_config.get("adaln_bits") == BETA_ADALN_BITS,
        "derived quantization config AdaLN bits are not Q8",
    )
    _require(
        quant_config.get("group_size") == BETA_GROUP_SIZE,
        "derived quantization config group size is not 64",
    )
    _require(
        quant_config.get("quantize_adaln") is True,
        "derived quantization config must enable AdaLN quantization",
    )
    _require(
        quant_config.get("quantized_layers") == BETA_QUANTIZED_LAYER_COUNTS,
        "derived quantization config layer counts are not Q6 core/Q8 AdaLN",
    )
    return quant_config


def _sidecar_projection(config: DiTConfig) -> dict[str, Any]:
    return {
        "quantization_bits": BETA_ADALN_BITS,
        "quantization_group_size": BETA_GROUP_SIZE,
        "logical_input_features": int(config.time_embed_dim),
        "logical_output_features": int(config.adaln_out_features),
        "packed_weight_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 4],
        "scales_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 64],
        "quantization_biases_shape": [int(config.adaln_out_features), int(config.time_embed_dim) // 64],
        "learned_bias_shape": [int(config.adaln_out_features)],
    }


def validate_streamed_sidecar_manifest(
    model_dir: str | Path,
    config: DiTConfig,
    quant_config: Mapping[str, Any],
) -> Path:
    """Validate declared sidecar topology and tensor metadata without opening payloads."""
    root = Path(model_dir)
    manifest_path = root / "adaln" / "manifest.json"
    manifest = _read_json(manifest_path, "derived AdaLN sidecar manifest")
    _require(
        set(manifest) == {"format_identifier", "schema_version", "bounded", "blocks"},
        "unsupported derived AdaLN sidecar manifest schema",
    )
    _require(
        manifest.get("format_identifier") == FORMAT_IDENTIFIER,
        "invalid derived AdaLN sidecar-manifest format identifier",
    )
    _require(
        manifest.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION,
        "unsupported derived AdaLN sidecar-manifest schema version",
    )
    _require(
        manifest.get("bounded") is False,
        "bounded derived AdaLN sidecar manifest is not loadable",
    )
    blocks = manifest.get("blocks")
    _require(
        isinstance(blocks, dict) and set(blocks) == {str(i) for i in range(BLOCK_COUNT)},
        "derived AdaLN sidecar manifest must describe all 50 blocks",
    )

    expected_projection = _sidecar_projection(config)
    expected_sidecars = {f"block-{index:03d}.safetensors" for index in range(BLOCK_COUNT)}
    actual_sidecars = {
        path.name for path in (root / "adaln").glob("*.safetensors") if path.is_file()
    }
    _require(
        actual_sidecars == expected_sidecars,
        "derived AdaLN sidecar payload file set mismatch: "
        f"missing={sorted(expected_sidecars - actual_sidecars)}, "
        f"unexpected={sorted(actual_sidecars - expected_sidecars)}",
    )

    expected_tensor_count = 0
    for block_index in range(BLOCK_COUNT):
        entry = blocks[str(block_index)]
        expected_filename = f"block-{block_index:03d}.safetensors"
        _require(
            isinstance(entry, dict),
            f"malformed derived AdaLN sidecar manifest entry for block {block_index}",
        )
        _require(
            entry.get("block_index") == block_index,
            f"reordered derived AdaLN sidecar manifest entry for block {block_index}",
        )
        _require(
            entry.get("sidecar_filename") == expected_filename,
            f"malformed derived AdaLN sidecar filename for block {block_index}",
        )
        _require(
            entry.get("projection") == expected_projection,
            f"derived AdaLN projection metadata mismatch for block {block_index}",
        )
        tensors = entry.get("tensors")
        _require(
            isinstance(tensors, list) and all(isinstance(item, dict) for item in tensors),
            f"derived AdaLN tensor metadata for block {block_index} must be a list of objects",
        )
        expected_keys = sorted(
            f"blocks.{block_index}.adaln_proj.linear.{suffix}"
            for suffix in SIDECAR_ROLE_TO_SUFFIX.values()
        )
        actual_keys = [item.get("tensor_key") for item in tensors]
        _require(
            all(isinstance(key, str) for key in actual_keys),
            f"derived AdaLN tensor metadata for block {block_index} has non-string tensor keys",
        )
        _require(
            len(tensors) == len(expected_keys)
            and len(set(actual_keys)) == len(expected_keys)
            and actual_keys == expected_keys,
            f"derived AdaLN tensor metadata for block {block_index} has inconsistent tensor keys/count",
        )

        shape_by_role = {
            "packed_weight": "packed_weight_shape",
            "scales": "scales_shape",
            "quantization_biases": "quantization_biases_shape",
            "learned_bias": "learned_bias_shape",
        }
        for item in tensors:
            role = item.get("tensor_role")
            _require(
                isinstance(role, str) and role in SIDECAR_ROLE_TO_SUFFIX,
                f"unsupported derived AdaLN tensor role in block {block_index}: {role!r}",
            )
            expected_key = (
                f"blocks.{block_index}.adaln_proj.linear.{SIDECAR_ROLE_TO_SUFFIX[role]}"
            )
            _require(
                item.get("tensor_key") == expected_key,
                f"derived AdaLN tensor role/key mismatch in block {block_index}",
            )
            _require(
                item.get("source_shape") == expected_projection[shape_by_role[role]],
                f"derived AdaLN {role} shape metadata mismatch in block {block_index}",
            )
            _require(
                item.get("source_dtype") == SIDECAR_DTYPE_NAMES[role],
                f"derived AdaLN {role} dtype metadata mismatch in block {block_index}",
            )
            declared_byte_count = _positive_integer(
                item.get("byte_count"),
                f"derived AdaLN {role} byte metadata in block {block_index}",
            )
            expected_byte_count = _expected_payload_byte_count(
                item["source_shape"],
                item["source_dtype"],
            )
            _require(
                declared_byte_count == expected_byte_count,
                f"derived AdaLN {role} byte metadata mismatch in block {block_index}: "
                f"declared {declared_byte_count}, expected {expected_byte_count}",
            )
            if role == "learned_bias":
                _require(
                    item.get("quantization_format") == "unquantized"
                    and item.get("quantization_bits") is None
                    and item.get("group_size") is None,
                    f"derived AdaLN learned-bias quantization metadata is invalid in block {block_index}",
                )
            else:
                _require(
                    item.get("quantization_format") == "affine"
                    and item.get("quantization_bits") == BETA_ADALN_BITS
                    and item.get("group_size") == BETA_GROUP_SIZE,
                    f"derived AdaLN quantization metadata is invalid in block {block_index} for {role}",
                )
            expected_tensor_count += 1

    _require(
        expected_tensor_count == DERIVED_SIDECAR_TENSOR_COUNT,
        "derived AdaLN sidecar tensor metadata count is incomplete",
    )
    _require(
        quant_config.get("adaln_bits") == expected_projection["quantization_bits"]
        and quant_config.get("group_size") == expected_projection["quantization_group_size"],
        "derived AdaLN sidecar metadata disagrees with quantization config",
    )
    return manifest_path


def _validated_base_shards(weight_map: Mapping[str, object]) -> tuple[str, ...]:
    values = list(weight_map.values())
    _require(
        all(isinstance(name, str) for name in values),
        "derived base index shard names must be strings",
    )
    names = tuple(sorted(set(values)))
    _require(
        len(names) == DERIVED_BASE_SHARD_COUNT,
        f"derived base index must reference exactly {DERIVED_BASE_SHARD_COUNT} shards",
    )
    for name in names:
        _require(
            Path(name).name == name
            and name.startswith("model-")
            and name.endswith("-of-00005.safetensors"),
            f"unsafe derived base shard filename: {name!r}",
        )
    return names


def inspect_checkpoint_format(model_dir: str | Path) -> CheckpointFormatInfo:
    """Route a transformer directory using explicit derived-format metadata.

    This is a lightweight structural/runtime check. It validates manifests, indexes, filenames,
    and required files, but deliberately does not hash the 30 GB source or open any AdaLN sidecar
    payload. Complete forensic verification remains the forge verifier's separate operation.
    """
    root = Path(model_dir)
    conversion_path = root / "conversion_manifest.json"
    if not conversion_path.exists():
        return CheckpointFormatInfo(
            checkpoint_format="original",
            derived_root=None,
            base_root=root,
            conversion_manifest_path=None,
            adaln_manifest_path=None,
            construction_mode=RESIDENT_CONSTRUCTION,
            base_shards=(),
        )

    manifest = _read_json(conversion_path, "derived conversion manifest")
    _require(
        manifest.get("format_identifier") == FORMAT_IDENTIFIER,
        f"unsupported derived checkpoint format identifier: {manifest.get('format_identifier')!r}",
    )
    _require(
        manifest.get("schema_version") == SUPPORTED_DERIVED_SCHEMA_VERSION,
        f"unsupported derived checkpoint schema version: {manifest.get('schema_version')!r}",
    )
    _require(manifest.get("bounded") is False, "bounded derived checkpoints cannot load a transformer base")
    _require(
        manifest.get("verification_status") == "verified",
        "derived checkpoint verification_status must be 'verified'",
    )
    _require(
        manifest.get("derived_base_tensor_count") == DERIVED_BASE_TENSOR_COUNT,
        "derived base tensor count is not complete",
    )
    _require(
        manifest.get("total_logical_tensor_count") == 1050,
        "derived checkpoint tensor count is incomplete",
    )
    _require(
        manifest.get("sidecar_count") == DERIVED_SIDECAR_COUNT,
        "derived sidecar count is incomplete",
    )
    _require(
        manifest.get("sidecar_tensor_count") == DERIVED_SIDECAR_TENSOR_COUNT,
        "derived sidecar tensor count is incomplete",
    )
    _require(
        manifest.get("selected_blocks") == list(range(BLOCK_COUNT)),
        "derived checkpoint does not contain all 50 blocks",
    )
    derived_base_byte_count = _positive_integer(
        manifest.get("derived_base_byte_count"),
        "derived_base_byte_count",
    )
    sidecar_byte_count = _positive_integer(
        manifest.get("sidecar_byte_count"),
        "sidecar_byte_count",
    )

    config = validate_streamed_transformer_config(root)
    quant_config = validate_streamed_quantization_config(root)

    base_root = root / "base"
    base_index_path = base_root / "model.safetensors.index.json"
    base_index = _read_json(base_index_path, "derived base index")
    base_metadata = base_index.get("metadata")
    _require(
        isinstance(base_metadata, dict),
        "derived base index metadata is missing",
    )
    indexed_base_byte_count = _positive_integer(
        base_metadata.get("total_size"),
        "derived base index metadata.total_size",
    )
    _require(
        derived_base_byte_count == indexed_base_byte_count,
        "derived_base_byte_count disagrees with derived base index metadata.total_size",
    )
    weight_map = base_index.get("weight_map")
    _require(
        isinstance(weight_map, dict) and len(weight_map) == DERIVED_BASE_TENSOR_COUNT,
        "derived base index must contain exactly 850 tensors",
    )
    base_shards = _validated_base_shards(weight_map)
    _require(
        not any(key.startswith("blocks.") and ".adaln_proj." in key for key in weight_map),
        "derived base index contains block-level AdaLN tensors",
    )
    _require(
        "final_layer.adaln_proj.linear.weight" in weight_map,
        "derived base index is missing final-layer AdaLN weight",
    )
    _require(
        "final_layer.adaln_proj.linear.bias" in weight_map,
        "derived base index is missing final-layer AdaLN bias",
    )
    actual_base_shards = {
        path.name for path in base_root.glob("*.safetensors") if path.is_file()
    }
    missing_base_shards = sorted(set(base_shards) - actual_base_shards)
    unexpected_base_shards = sorted(actual_base_shards - set(base_shards))
    _require(
        not missing_base_shards and not unexpected_base_shards,
        "derived base payload file set mismatch: "
        f"missing={missing_base_shards}, unexpected={unexpected_base_shards}",
    )

    adaln_path = validate_streamed_sidecar_manifest(root, config, quant_config)
    sidecar_manifest = _read_json(adaln_path, "derived AdaLN sidecar manifest")
    sidecar_blocks = sidecar_manifest.get("blocks")
    _require(isinstance(sidecar_blocks, dict), "derived AdaLN sidecar manifest has no blocks")
    computed_sidecar_byte_count = sum(
        item["byte_count"]
        for block in sidecar_blocks.values()
        for item in block["tensors"]
    )
    _require(
        sidecar_byte_count == computed_sidecar_byte_count,
        "sidecar_byte_count disagrees with declared sidecar tensor byte counts",
    )

    return CheckpointFormatInfo(
        checkpoint_format="derived",
        derived_root=root,
        base_root=base_root,
        conversion_manifest_path=conversion_path,
        adaln_manifest_path=adaln_path,
        construction_mode=CACHE_ONLY_CONSTRUCTION,
        base_shards=base_shards,
    )

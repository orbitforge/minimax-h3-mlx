"""Explicit, MLX-free runtime selection and beta asset admission.

The production CLI historically accepted an upstream checkpoint root plus an arbitrary
transformer override.  This module owns the smaller named-runtime seam used by the CLI:
``<assets-root>/beta-0.6/{checkpoint,transformer,conventional}`` are non-copying host links,
while the accepted beta identity is established from the linked metadata rather than from a
directory name.

Only JSON metadata and filesystem names are inspected here.  No MLX import, safetensors payload,
model construction, or model evaluation is performed.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config import DiTConfig


BETA_RUNTIME_ID = "beta-0.6"
BETA_RUNTIME_ASSET_DIR = BETA_RUNTIME_ID
BETA_CHECKPOINT_LINK = "checkpoint"
BETA_TRANSFORMER_LINK = "transformer"
BETA_CONVENTIONAL_LINK = "conventional"

BETA_STREAMED_FORMAT = "minimax-h3-mlx-streamed-adaln-v1"
BETA_STREAMED_SCHEMA_VERSION = 1
BETA_BLOCK_COUNT = 50
BETA_SOURCE_BYTES = 66_288_818_760
BETA_SOURCE_SHA256 = "16f1950cc83bd686106d49588c8611281fbb5e9ae46f8cd1ae7945fd4e00357d"
BETA_SOURCE_TENSOR_COUNT = 535
BETA_SOURCE_DTYPE_COUNTS = {"BF16": 522, "F32": 13}
BETA_QKV_SOURCE_LAYOUT = "grouped_qkv"
BETA_QKV_CANONICAL_LAYOUT = "runtime_interleaved"
BETA_QKV_TENSOR_COUNT = 52
BETA_QKV_RECEIPT_SCHEMA = "slice025.qkv_layout_authorization.v1"
BETA_QKV_RECEIPT_SHA256 = "62095531d253bc129c39b0033fa4b6c167067588aa5d0dfd851720d8753e10b6"

BETA_CORE_BITS = 6
BETA_ADALN_BITS = 8
BETA_GROUP_SIZE = 64
BETA_QUANTIZED_LAYER_COUNTS = {"6": 208, "8": 50}
BETA_LOGICAL_TENSOR_COUNT = 1_050
BETA_RESIDENT_TENSOR_COUNT = 850
BETA_RESIDENT_BLOCK_ADALN_TENSOR_COUNT = 0
BETA_SIDECAR_COUNT = 50
BETA_SIDECAR_TENSOR_COUNT = 200

# The conventional index/config/recipe are generated artifacts, but these three identities are
# stable source-linkage fields.  The streamed conversion manifest's timestamp and host path are
# intentionally not pinned as the profile identity.
BETA_CONVENTIONAL_INDEX_SHA256 = "5d372ee509f981b020166051b7296bd1bd91f6d3d5222e80259d9123f0a4f592"
BETA_CONVENTIONAL_CONFIG_SHA256 = "bd97e5da656ee83da7cf4d83146a19c06521ec8b455bd62775521c27bdb08ebf"
BETA_CONVENTIONAL_QUANT_CONFIG_SHA256 = "7c23b9aa18b61e518058acda3ffa3d26ef1f3a60809f1aac0bd37ace43bee39c"

BETA_QWEN_MODEL_TYPE = "qwen3_vl"
BETA_QWEN_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
BETA_QWEN_TEXT_HIDDEN_SIZE = 5120
BETA_QWEN_TEXT_LAYER_COUNT = 64
BETA_QWEN_QUANT_BITS = 4
BETA_QWEN_QUANT_GROUP_SIZE = 64
BETA_QWEN_QUANT_MODE = "affine"
BETA_TOKENIZER_MODEL_INDEX = ["transformers", "Qwen2TokenizerFast"]
BETA_PROCESSOR_MODEL_INDEX = ["transformers", "Qwen3VLProcessor"]
BETA_TOKENIZER_CLASS = "Qwen2Tokenizer"
BETA_PROCESSOR_CLASS = "Qwen3VLProcessor"
BETA_IMAGE_PROCESSOR_TYPE = "Qwen2VLImageProcessorFast"
BETA_VIDEO_PROCESSOR_TYPE = "Qwen3VLVideoProcessor"
BETA_PROCESSOR_PATCH_SIZE = 16
BETA_PROCESSOR_TEMPORAL_PATCH_SIZE = 2
BETA_PROCESSOR_MERGE_SIZE = 2
BETA_PROCESSOR_IMAGE_MEAN = [0.5, 0.5, 0.5]
BETA_PROCESSOR_IMAGE_STD = [0.5, 0.5, 0.5]
BETA_PROCESSOR_IMAGE_SIZE = {"longest_edge": 16_777_216, "shortest_edge": 65_536}
BETA_PROCESSOR_VIDEO_SIZE = {"longest_edge": 25_165_824, "shortest_edge": 4_096}
BETA_TOKENIZER_REQUIRED_FILES = ("tokenizer_config.json", "tokenizer.json", "vocab.json", "merges.txt")
BETA_PROCESSOR_REQUIRED_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "chat_template.json",
)
BETA_TOKENIZER_SPECIAL_TOKENS = {
    "151652": "<|vision_start|>",
    "151653": "<|vision_end|>",
    "151655": "<|image_pad|>",
    "151656": "<|video_pad|>",
}
BETA_DIT_CONFIG_VALUES: Mapping[str, Any] = {
    "hidden_size": 5376,
    "num_layers": 50,
    "token_refiner_num_layers": 2,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_hidden_size": 14336,
    "latents_dim": 24,
    "audio_latents_dim": 32,
    "patch_size": (1, 2, 2),
    "text_dim": 5120,
    "timestep_input_dim": 256,
    "time_embed_hidden_size": 5376,
    "time_embed_dim": 2688,
    "adaln_out_features": 96768,
    "final_adaln_out_features": 10752,
    "rope_inv_freq_len": 16,
    "rope_theta": 10000.0,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}
BETA_DIT_REQUIRED_CONFIG_FIELDS = frozenset(BETA_DIT_CONFIG_VALUES) - {"rope_theta"}
BETA_DIT_INTEGER_CONFIG_FIELDS = frozenset(
    {
        "hidden_size",
        "num_layers",
        "token_refiner_num_layers",
        "num_attention_heads",
        "attention_head_dim",
        "ffn_hidden_size",
        "latents_dim",
        "audio_latents_dim",
        "timestep_input_dim",
        "time_embed_hidden_size",
        "time_embed_dim",
        "adaln_out_features",
        "final_adaln_out_features",
        "rope_inv_freq_len",
    }
)
BETA_VIDEO_LATENT_CHANNELS = 24
BETA_VIDEO_SPATIAL_RATIO = 16
BETA_VIDEO_TEMPORAL_RATIO = 4
BETA_AUDIO_SAMPLE_RATE = 32_000
BETA_AUDIO_LATENT_CHANNELS = 32
BETA_VIDEO_SCHEDULER_SHIFT = 12.0
BETA_AUDIO_SCHEDULER_SHIFT = 3.0

QKV_LAYOUT_AUTHORIZATION_RECEIPT_PATH = Path(__file__).with_name("qkv_layout_authorization.json")


class RuntimeSelectionError(ValueError):
    """Fail-closed named-runtime error with a stable operator-facing code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class ResolvedRuntime:
    """Resolved component paths and metadata for one named runtime."""

    runtime_id: str
    asset_profile_root: Path
    checkpoint_root: Path
    transformer_root: Path
    conventional_root: Path
    qwen_root: Path
    video_vae_root: Path
    audio_vae_root: Path
    transformer_identity: Mapping[str, Any]
    surrounding_identity: Mapping[str, Any]
    explicit_override_used: bool = False

    @property
    def receipt(self) -> dict[str, Any]:
        """Return a JSON-safe resolved-fact receipt for pre-load operator telemetry."""
        return {
            "runtime_id": self.runtime_id,
            "selection": "named-runtime",
            "asset_profile_root": str(self.asset_profile_root),
            "checkpoint_root": str(self.checkpoint_root),
            "transformer_root": str(self.transformer_root),
            "conventional_root": str(self.conventional_root),
            "qwen": dict(self.surrounding_identity["qwen"]),
            "video_vae": dict(self.surrounding_identity["video_vae"]),
            "audio_vae": dict(self.surrounding_identity["audio_vae"]),
            "scheduler": dict(self.surrounding_identity["scheduler"]),
            "tokenizer": dict(self.surrounding_identity["tokenizer"]),
            "processor": dict(self.surrounding_identity["processor"]),
            "transformer": dict(self.transformer_identity),
            "explicit_override_used": self.explicit_override_used,
        }


def _resolved(path: Path) -> Path:
    try:
        return path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"could not resolve runtime asset path: {path}: {exc}") from exc


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"{label} must be a JSON object: {path}")
    return value


def _require(condition: bool, detail: str) -> None:
    if not condition:
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", detail)


def _require_file(path: Path, label: str) -> None:
    _require(path.is_file(), f"{label} is missing: {path}")


def _require_directory(path: Path, label: str) -> None:
    _require(path.is_dir(), f"{label} is missing or not a directory: {path}")


def _sha256_file(path: Path, label: str) -> str:
    _require_file(path, label)
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"could not hash {label}: {path}: {exc}") from exc
    return digest.hexdigest()


def _file_identity(path: Path, label: str) -> dict[str, Any]:
    """Return metadata for one admitted JSON identity file, without loading model payloads."""
    _require_file(path, label)
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"could not stat {label}: {path}: {exc}") from exc
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "sha256": _sha256_file(path, label),
    }


def _json_identity(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one metadata JSON object and return both its value and its file identity."""
    return _read_json(path, label), _file_identity(path, label)


def _resolve_link(profile_root: Path, name: str) -> Path:
    requested = profile_root / name
    if not os.path.lexists(requested):
        raise RuntimeSelectionError(
            "RUNTIME_ASSET_MISSING",
            f"beta runtime asset link is missing: {requested}",
        )
    _require(
        requested.is_symlink(),
        f"beta runtime asset {name!r} must be a symbolic link so the profile cannot silently select "
        f"a copied or arbitrary directory: {requested}",
    )
    resolved = _resolved(requested)
    _require_directory(resolved, f"beta runtime asset {name!r}")
    return resolved


def _parse_semicolon_fields(value: str) -> tuple[str, dict[str, str]]:
    parts = value.split(";")
    prefix = parts[0]
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"malformed authorization field: {part!r}")
        key, field_value = part.split("=", 1)
        if not key or key in fields:
            raise RuntimeSelectionError("RUNTIME_ASSET_INVALID", f"malformed authorization field: {part!r}")
        fields[key] = field_value
    return prefix, fields


def _normalize_config_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _validate_transformer_config(transformer_root: Path) -> dict[str, Any]:
    """Validate the exact JSON semantics consumed by ``DiTConfig.from_json``.

    The released config intentionally omits ``rope_theta``; ``DiTConfig`` supplies its stable
    default. Every other recognized field is required so a missing architecture field cannot
    silently fall back to a constructor default. Unknown decorative fields remain ignored exactly
    as they are by the production parser.
    """
    config_path = transformer_root / "config.json"
    _require_file(config_path, "streamed transformer config")
    raw, file_identity = _json_identity(config_path, "streamed transformer config")

    missing = sorted(BETA_DIT_REQUIRED_CONFIG_FIELDS - set(raw))
    _require(
        not missing,
        "streamed transformer config is missing required DiT fields: " + ", ".join(missing),
    )
    for field in BETA_DIT_INTEGER_CONFIG_FIELDS:
        _require(
            type(raw.get(field)) is int,
            f"streamed transformer config field {field!r} must be an integer",
        )
    patch_size = raw.get("patch_size")
    _require(
        isinstance(patch_size, (list, tuple))
        and len(patch_size) == 3
        and all(type(value) is int for value in patch_size),
        "streamed transformer config patch_size must be a three-integer sequence",
    )

    try:
        parsed = DiTConfig.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeSelectionError(
            "RUNTIME_ASSET_INVALID",
            f"streamed transformer config cannot be interpreted by DiTConfig: {config_path}: {exc}",
        ) from exc

    normalized = {
        field: _normalize_config_value(getattr(parsed, field))
        for field in BETA_DIT_CONFIG_VALUES
    }
    for field, expected in BETA_DIT_CONFIG_VALUES.items():
        actual = getattr(parsed, field)
        if isinstance(expected, tuple):
            actual = tuple(actual) if isinstance(actual, (list, tuple)) else actual
        _require(
            actual == expected,
            f"streamed transformer config field {field!r} is not accepted: expected {expected!r}, got {actual!r}",
        )

    defaults_applied: dict[str, Any] = {}
    if "rope_theta" not in raw:
        defaults_applied["rope_theta"] = BETA_DIT_CONFIG_VALUES["rope_theta"]

    derived = {
        "inner_dim": parsed.inner_dim,
        "video_patch_dim": parsed.video_patch_dim,
        "audio_patch_dim": parsed.audio_latents_dim,
        "rotary_dim": parsed.rotary_dim,
        "adaln_modulation_features": parsed.adaln_out_features,
        "final_adaln_features": parsed.final_adaln_out_features,
        "final_video_output_dim": parsed.video_patch_dim,
        "final_audio_output_dim": parsed.audio_latents_dim,
    }
    ignored_fields = sorted(
        set(raw) - set(BETA_DIT_CONFIG_VALUES) - set(defaults_applied)
    )
    return {
        "config_schema": "minimax_h3_mlx.config.DiTConfig",
        "config_identity_policy": "semantic-fields-plus-sha256",
        "config_path": file_identity["path"],
        "config_bytes": file_identity["bytes"],
        "config_sha256": file_identity["sha256"],
        "config_fields": normalized,
        "defaults_applied": defaults_applied,
        "derived_contract": derived,
        "ignored_config_fields": ignored_fields,
    }


def _validate_tokenizer_config(path: Path, label: str) -> dict[str, Any]:
    config, identity = _json_identity(path, label)
    _require(config.get("tokenizer_class") == BETA_TOKENIZER_CLASS, f"{label} tokenizer class is not Qwen2Tokenizer")
    _require(config.get("eos_token") == "<|im_end|>", f"{label} EOS token is not Qwen3-VL compatible")
    _require(config.get("pad_token") == "<|endoftext|>", f"{label} pad token is not Qwen3-VL compatible")
    added_tokens = config.get("added_tokens_decoder")
    _require(isinstance(added_tokens, dict), f"{label} added token metadata is missing")
    for token_id, token in BETA_TOKENIZER_SPECIAL_TOKENS.items():
        entry = added_tokens.get(token_id)
        _require(
            isinstance(entry, dict) and entry.get("content") == token,
            f"{label} special token {token!r} is not mapped to its accepted ID",
        )
    return {
        **identity,
        "tokenizer_class": config["tokenizer_class"],
        "eos_token": config["eos_token"],
        "pad_token": config["pad_token"],
        "special_token_ids": dict(BETA_TOKENIZER_SPECIAL_TOKENS),
    }


def _validate_tokenizer_root(tokenizer_root: Path) -> dict[str, Any]:
    _require_directory(tokenizer_root, "Qwen tokenizer root")
    for filename in BETA_TOKENIZER_REQUIRED_FILES:
        _require_file(tokenizer_root / filename, f"Qwen tokenizer {filename}")
    config_identity = _validate_tokenizer_config(
        tokenizer_root / "tokenizer_config.json",
        "Qwen tokenizer config",
    )
    return {
        "path": str(tokenizer_root),
        "identity_policy": "semantic-config; vocabulary payload presence-only",
        "required_files": list(BETA_TOKENIZER_REQUIRED_FILES),
        "config": config_identity,
    }


def _validate_processor_config(path: Path, label: str, *, video: bool) -> dict[str, Any]:
    config, identity = _json_identity(path, label)
    _require(config.get("processor_class") == BETA_PROCESSOR_CLASS, f"{label} processor class is not Qwen3VLProcessor")
    if video:
        _require(
            config.get("video_processor_type") == BETA_VIDEO_PROCESSOR_TYPE,
            f"{label} video processor type is not Qwen3VLVideoProcessor",
        )
        expected_size = BETA_PROCESSOR_VIDEO_SIZE
    else:
        _require(
            config.get("image_processor_type") == BETA_IMAGE_PROCESSOR_TYPE,
            f"{label} image processor type is not Qwen2VLImageProcessorFast",
        )
        expected_size = BETA_PROCESSOR_IMAGE_SIZE
    _require(config.get("patch_size") == BETA_PROCESSOR_PATCH_SIZE, f"{label} patch size is not 16")
    _require(
        config.get("temporal_patch_size") == BETA_PROCESSOR_TEMPORAL_PATCH_SIZE,
        f"{label} temporal patch size is not 2",
    )
    _require(config.get("merge_size") == BETA_PROCESSOR_MERGE_SIZE, f"{label} merge size is not 2")
    _require(config.get("image_mean") == BETA_PROCESSOR_IMAGE_MEAN, f"{label} image mean is not accepted")
    _require(config.get("image_std") == BETA_PROCESSOR_IMAGE_STD, f"{label} image std is not accepted")
    _require(config.get("size") == expected_size, f"{label} resize geometry is not accepted")
    return {
        **identity,
        "processor_class": config["processor_class"],
        "processor_type": config.get("video_processor_type", config.get("image_processor_type")),
        "patch_size": config["patch_size"],
        "temporal_patch_size": config["temporal_patch_size"],
        "merge_size": config["merge_size"],
        "size": dict(config["size"]),
    }


def _validate_processor_root(processor_root: Path) -> dict[str, Any]:
    _require_directory(processor_root, "Qwen processor root")
    for filename in BETA_PROCESSOR_REQUIRED_FILES:
        _require_file(processor_root / filename, f"Qwen processor {filename}")
    tokenizer_identity = _validate_tokenizer_config(
        processor_root / "tokenizer_config.json",
        "Qwen processor tokenizer config",
    )
    image_identity = _validate_processor_config(
        processor_root / "preprocessor_config.json",
        "Qwen image processor config",
        video=False,
    )
    video_identity = _validate_processor_config(
        processor_root / "video_preprocessor_config.json",
        "Qwen video processor config",
        video=True,
    )
    chat_template, chat_identity = _json_identity(
        processor_root / "chat_template.json",
        "Qwen processor chat template",
    )
    _require(isinstance(chat_template.get("chat_template"), str) and chat_template["chat_template"], "Qwen processor chat template is missing")
    return {
        "path": str(processor_root),
        "identity_policy": "semantic-config; tokenizer and processor payload presence-only",
        "required_files": list(BETA_PROCESSOR_REQUIRED_FILES),
        "tokenizer_config": tokenizer_identity,
        "image_config": image_identity,
        "video_config": video_identity,
        "chat_template": chat_identity,
    }


def _validate_qkv_authorization_receipt() -> dict[str, Any]:
    actual_sha256 = _sha256_file(QKV_LAYOUT_AUTHORIZATION_RECEIPT_PATH, "QKV authorization receipt")
    _require(
        actual_sha256 == BETA_QKV_RECEIPT_SHA256,
        "QKV authorization receipt SHA-256 does not match the accepted Slice 025 receipt",
    )
    receipt = _read_json(QKV_LAYOUT_AUTHORIZATION_RECEIPT_PATH, "QKV authorization receipt")
    _require(receipt.get("receipt_schema") == BETA_QKV_RECEIPT_SCHEMA, "QKV authorization receipt schema is not accepted")

    beta_source = receipt.get("beta_source")
    _require(isinstance(beta_source, dict), "QKV authorization receipt has no beta source record")
    _require(beta_source.get("sha256") == BETA_SOURCE_SHA256, "QKV authorization source SHA-256 is not accepted")
    _require(beta_source.get("bytes") == BETA_SOURCE_BYTES, "QKV authorization source byte count is not accepted")
    _require(beta_source.get("expected_bytes") == BETA_SOURCE_BYTES, "QKV authorization expected source byte count is not accepted")
    _require(beta_source.get("qkv_count_examined") == BETA_QKV_TENSOR_COUNT, "QKV authorization examined count is not 52")
    _require(beta_source.get("source_range_count") == BETA_QKV_TENSOR_COUNT, "QKV authorization range count is not 52")
    _require(beta_source.get("source_stat_unchanged") is True, "QKV authorization source was not unchanged")
    _require(
        beta_source.get("topology") == {
            "bf16": BETA_SOURCE_DTYPE_COUNTS["BF16"],
            "f32": BETA_SOURCE_DTYPE_COUNTS["F32"],
            "tensor_count": BETA_SOURCE_TENSOR_COUNT,
        },
        "QKV authorization source topology is not accepted",
    )

    coverage = receipt.get("coverage")
    _require(isinstance(coverage, dict), "QKV authorization coverage is missing")
    _require(coverage.get("all_expected_qkv_examined") is True, "QKV authorization coverage is incomplete")
    _require(coverage.get("examined_qkv_count") == BETA_QKV_TENSOR_COUNT, "QKV authorization examined QKV count is not 52")
    _require(coverage.get("expected_qkv_count") == BETA_QKV_TENSOR_COUNT, "QKV authorization expected QKV count is not 52")
    _require(coverage.get("main_block_count") == BETA_BLOCK_COUNT, "QKV authorization main-block count is not 50")
    _require(coverage.get("token_refiner_count") == 2, "QKV authorization token-refiner count is not 2")

    conclusion = receipt.get("layout_conclusion")
    _require(isinstance(conclusion, dict), "QKV authorization layout conclusion is missing")
    _require(conclusion.get("accepted_source_layout") == BETA_QKV_SOURCE_LAYOUT, "QKV authorization source layout is not grouped_qkv")
    _require(conclusion.get("canonical_runtime_layout") == BETA_QKV_CANONICAL_LAYOUT, "QKV authorization canonical layout is not runtime_interleaved")
    _require(conclusion.get("status") == "GROUPED_QKV_PAYLOAD_SUPPORTED", "QKV authorization status is not accepted")

    return {
        "receipt_schema": BETA_QKV_RECEIPT_SCHEMA,
        "receipt_sha256": actual_sha256,
        "source_sha256": BETA_SOURCE_SHA256,
        "source_bytes": BETA_SOURCE_BYTES,
        "source_tensor_count": BETA_SOURCE_TENSOR_COUNT,
        "source_dtype_counts": dict(BETA_SOURCE_DTYPE_COUNTS),
        "source_layout": BETA_QKV_SOURCE_LAYOUT,
        "canonical_layout": BETA_QKV_CANONICAL_LAYOUT,
        "qkv_tensors_reconciled": BETA_QKV_TENSOR_COUNT,
    }


def _validate_conventional_root(conventional_root: Path, qkv_receipt: Mapping[str, Any]) -> dict[str, Any]:
    index_path = conventional_root / "model.safetensors.index.json"
    config_path = conventional_root / "config.json"
    quant_config_path = conventional_root / "quant_config.json"
    _require_file(index_path, "conventional transformer index")
    _require_file(config_path, "conventional transformer config")
    _require_file(quant_config_path, "conventional quantization config")

    index_sha256 = _sha256_file(index_path, "conventional transformer index")
    config_sha256 = _sha256_file(config_path, "conventional transformer config")
    quant_config_sha256 = _sha256_file(quant_config_path, "conventional quantization config")
    _require(index_sha256 == BETA_CONVENTIONAL_INDEX_SHA256, "conventional transformer index identity is not the accepted corrected beta index")
    _require(config_sha256 == BETA_CONVENTIONAL_CONFIG_SHA256, "conventional transformer config identity is not accepted")
    _require(quant_config_sha256 == BETA_CONVENTIONAL_QUANT_CONFIG_SHA256, "conventional quantization config identity is not accepted")

    index = _read_json(index_path, "conventional transformer index")
    metadata = index.get("metadata")
    weight_map = index.get("weight_map")
    _require(isinstance(metadata, dict), "conventional transformer index metadata is missing")
    _require(isinstance(weight_map, dict), "conventional transformer index weight_map is missing")
    _require(len(weight_map) == BETA_LOGICAL_TENSOR_COUNT, "conventional transformer logical tensor count is not 1,050")
    _require(metadata.get("source_size") == BETA_SOURCE_BYTES, "conventional transformer source byte identity is not accepted")
    _require(metadata.get("qkv_source_layout") == BETA_QKV_SOURCE_LAYOUT, "conventional transformer QKV source layout is not grouped_qkv")
    _require(metadata.get("qkv_canonical_layout") == BETA_QKV_CANONICAL_LAYOUT, "conventional transformer QKV canonical layout is not runtime_interleaved")
    _require(metadata.get("qkv_row_reconciliation_applied") is True, "conventional transformer QKV reconciliation is not recorded")
    _require(metadata.get("qkv_tensors_reconciled") == BETA_QKV_TENSOR_COUNT, "conventional transformer reconciled QKV count is not 52")
    _require(metadata.get("quantized_layers") == BETA_QUANTIZED_LAYER_COUNTS, "conventional transformer quantized-layer policy is not Q6/Q8")

    authorization = metadata.get("qkv_layout_authorization")
    _require(isinstance(authorization, str), "conventional transformer QKV authorization is missing")
    prefix, fields = _parse_semicolon_fields(authorization)
    _require(prefix == f"payload_receipt:{BETA_QKV_RECEIPT_SCHEMA}", "conventional transformer QKV authorization schema is not accepted")
    _require(fields.get("receipt_sha256") == BETA_QKV_RECEIPT_SHA256, "conventional transformer QKV authorization receipt identity is not accepted")
    _require(fields.get("source_sha256") == BETA_SOURCE_SHA256, "conventional transformer QKV authorization source identity is not accepted")
    _require(fields.get("layout") == BETA_QKV_SOURCE_LAYOUT, "conventional transformer QKV authorization layout is not grouped_qkv")
    _require(fields.get("source_identity") == metadata.get("source_identity"), "conventional transformer QKV authorization/source identity linkage is inconsistent")

    return {
        "index_path": str(index_path.resolve()),
        "index_sha256": index_sha256,
        "config_sha256": config_sha256,
        "quant_config_sha256": quant_config_sha256,
        "logical_tensor_count": len(weight_map),
        "source_bytes": metadata["source_size"],
        "qkv_source_layout": metadata["qkv_source_layout"],
        "qkv_canonical_layout": metadata["qkv_canonical_layout"],
        "qkv_tensors_reconciled": metadata["qkv_tensors_reconciled"],
        "qkv_row_reconciliation_applied": metadata["qkv_row_reconciliation_applied"],
        "quantized_layers": dict(metadata["quantized_layers"]),
        "authorization_receipt_sha256": fields["receipt_sha256"],
        "source_sha256": fields["source_sha256"],
        "qkv_receipt": dict(qkv_receipt),
    }


def _validate_surrounding_root(checkpoint_root: Path) -> dict[str, Any]:
    model_index_path = checkpoint_root / "model_index.json"
    _require_file(model_index_path, "surrounding checkpoint model_index.json")
    model_index = _read_json(model_index_path, "surrounding checkpoint model_index.json")
    _require(model_index.get("_class_name") == "MiniMaxH3Pipeline", "surrounding checkpoint is not MiniMaxH3Pipeline")
    _require(model_index.get("text_encoder") == ["transformers", "MiniMaxH3Qwen3VLHFEncoder"], "surrounding checkpoint text encoder contract is not Canonical Qwen3-VL")
    _require(model_index.get("tokenizer") == BETA_TOKENIZER_MODEL_INDEX, "surrounding checkpoint tokenizer contract is not Qwen2TokenizerFast")
    _require(model_index.get("video_vae") == ["diffusers", "MiniMaxH3VideoVAE"], "surrounding checkpoint video VAE contract is not accepted")
    _require(model_index.get("audio_vae") == ["diffusers", "MiniMaxH3AudioVAE"], "surrounding checkpoint audio VAE contract is not accepted")
    _require(model_index.get("scheduler") is None, "surrounding checkpoint scheduler component is not the accepted implicit contract")
    _require(model_index.get("processor") == BETA_PROCESSOR_MODEL_INDEX, "surrounding checkpoint processor contract is not Qwen3VLProcessor")

    meta = model_index.get("_minimax_h3")
    _require(isinstance(meta, dict), "surrounding checkpoint _minimax_h3 metadata is missing")
    _require(meta.get("schema_version") == 1, "surrounding checkpoint schema version is not 1")
    _require(meta.get("partition") == "fl2va", "surrounding checkpoint partition is not fl2va")
    _require(meta.get("tasks") == ["t2va", "fl2va"], "surrounding checkpoint task contract is not accepted")
    shifts = meta.get("sigma_shift_scales")
    _require(shifts == {"video": BETA_VIDEO_SCHEDULER_SHIFT, "audio": BETA_AUDIO_SCHEDULER_SHIFT}, "surrounding checkpoint scheduler shifts are not accepted")

    qwen_root = _resolved(checkpoint_root / "text_encoder")
    video_root = _resolved(checkpoint_root / "video_vae")
    audio_root = _resolved(checkpoint_root / "audio_vae")
    tokenizer_root = _resolved(checkpoint_root / "tokenizer")
    processor_root = _resolved(checkpoint_root / "processor")
    _require_directory(qwen_root, "Qwen text encoder root")
    _require_directory(video_root, "video VAE root")
    _require_directory(audio_root, "audio VAE root")
    tokenizer_identity = _validate_tokenizer_root(tokenizer_root)
    processor_identity = _validate_processor_root(processor_root)

    qwen_config_path = qwen_root / "config.json"
    _require_file(qwen_config_path, "Qwen config")
    qwen_config = _read_json(qwen_config_path, "Qwen config")
    _require(qwen_config.get("model_type") == BETA_QWEN_MODEL_TYPE, "Qwen model type is not qwen3_vl")
    _require(qwen_config.get("architectures") == [BETA_QWEN_ARCHITECTURE], "Qwen architecture is not accepted")
    text_config = qwen_config.get("text_config")
    _require(isinstance(text_config, dict), "Qwen text_config is missing")
    _require(text_config.get("hidden_size") == BETA_QWEN_TEXT_HIDDEN_SIZE, "Qwen text hidden size is not 5120")
    _require(text_config.get("num_hidden_layers") == BETA_QWEN_TEXT_LAYER_COUNT, "Qwen text layer count is not 64")
    quantization = qwen_config.get("quantization")
    quantization_config = qwen_config.get("quantization_config")
    expected_qwen_quant = {"bits": BETA_QWEN_QUANT_BITS, "group_size": BETA_QWEN_QUANT_GROUP_SIZE, "mode": BETA_QWEN_QUANT_MODE}
    _require(quantization == expected_qwen_quant and quantization_config == expected_qwen_quant, "Qwen quantization identity is not accepted")

    video_config_path = video_root / "config.json"
    video_source_config_path = video_root / "source" / "config.json"
    _require_file(video_config_path, "video VAE config")
    _require_file(video_source_config_path, "video VAE source config")
    video_config = _read_json(video_config_path, "video VAE config")
    video_source_config = _read_json(video_source_config_path, "video VAE source config")
    _require(video_config.get("_class_name") == "MiniMaxH3VideoVAE", "video VAE class identity is not accepted")
    _require(video_config.get("latent_channels") == BETA_VIDEO_LATENT_CHANNELS, "video VAE latent channel count is not 24")
    _require(video_source_config.get("z_channels") == BETA_VIDEO_LATENT_CHANNELS, "video VAE source latent channel count is not 24")
    _require(video_source_config.get("vae_ratio") == BETA_VIDEO_SPATIAL_RATIO, "video VAE spatial ratio is not 16")
    _require(video_source_config.get("vae_ratio_t") == BETA_VIDEO_TEMPORAL_RATIO, "video VAE temporal ratio is not 4")

    audio_config_path = audio_root / "config.json"
    audio_metadata_path = audio_root / "metadata.json"
    _require_file(audio_config_path, "audio VAE config")
    _require_file(audio_metadata_path, "audio VAE metadata")
    audio_config = _read_json(audio_config_path, "audio VAE config")
    audio_metadata = _read_json(audio_metadata_path, "audio VAE metadata")
    _require(audio_config.get("_class_name") == "MiniMaxH3AudioVAE", "audio VAE class identity is not accepted")
    _require(audio_config.get("sample_rate") == BETA_AUDIO_SAMPLE_RATE, "audio VAE sample rate is not 32 kHz")
    _require(audio_config.get("latent_channels") == BETA_AUDIO_LATENT_CHANNELS, "audio VAE latent channel count is not 32")
    audio_meta = audio_metadata.get("metadata")
    _require(isinstance(audio_meta, dict), "audio VAE metadata kwargs container is missing")
    audio_kwargs = audio_meta.get("kwargs")
    _require(isinstance(audio_kwargs, dict), "audio VAE metadata kwargs are missing")
    _require(audio_kwargs.get("sample_rate") == BETA_AUDIO_SAMPLE_RATE, "audio VAE metadata sample rate is not 32 kHz")
    _require(audio_kwargs.get("vae_latent_channels") == BETA_AUDIO_LATENT_CHANNELS, "audio VAE metadata latent channel count is not 32")

    return {
        "model_index_path": str(model_index_path.resolve()),
        "model_index_sha256": _sha256_file(model_index_path, "surrounding checkpoint model_index.json"),
        "qwen": {
            "path": str(qwen_root),
            "config_path": str(qwen_config_path.resolve()),
            "config_sha256": _sha256_file(qwen_config_path, "Qwen config"),
            "model_type": BETA_QWEN_MODEL_TYPE,
            "architecture": BETA_QWEN_ARCHITECTURE,
            "text_hidden_size": BETA_QWEN_TEXT_HIDDEN_SIZE,
            "text_layer_count": BETA_QWEN_TEXT_LAYER_COUNT,
            "quantization": dict(expected_qwen_quant),
        },
        "tokenizer": tokenizer_identity,
        "processor": processor_identity,
        "video_vae": {
            "path": str(video_root),
            "config_path": str(video_config_path.resolve()),
            "config_sha256": _sha256_file(video_config_path, "video VAE config"),
            "source_config_path": str(video_source_config_path.resolve()),
            "source_config_sha256": _sha256_file(video_source_config_path, "video VAE source config"),
            "latent_channels": BETA_VIDEO_LATENT_CHANNELS,
            "spatial_ratio": BETA_VIDEO_SPATIAL_RATIO,
            "temporal_ratio": BETA_VIDEO_TEMPORAL_RATIO,
        },
        "audio_vae": {
            "path": str(audio_root),
            "config_path": str(audio_config_path.resolve()),
            "config_sha256": _sha256_file(audio_config_path, "audio VAE config"),
            "metadata_path": str(audio_metadata_path.resolve()),
            "metadata_sha256": _sha256_file(audio_metadata_path, "audio VAE metadata"),
            "sample_rate": BETA_AUDIO_SAMPLE_RATE,
            "latent_channels": BETA_AUDIO_LATENT_CHANNELS,
        },
        "scheduler": {
            "identity": "model_index._minimax_h3.sigma_shift_scales",
            "component": None,
            "video_shift": BETA_VIDEO_SCHEDULER_SHIFT,
            "audio_shift": BETA_AUDIO_SCHEDULER_SHIFT,
        },
    }


def _validate_streamed_transformer(
    transformer_root: Path,
    conventional_identity: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = transformer_root / "conversion_manifest.json"
    quant_config_path = transformer_root / "quant_config.json"
    base_index_path = transformer_root / "base" / "model.safetensors.index.json"
    adaln_manifest_path = transformer_root / "adaln" / "manifest.json"
    for path, label in (
        (manifest_path, "streamed conversion manifest"),
        (quant_config_path, "streamed quantization config"),
        (base_index_path, "streamed base index"),
        (adaln_manifest_path, "streamed AdaLN manifest"),
    ):
        _require_file(path, label)

    manifest = _read_json(manifest_path, "streamed conversion manifest")
    _require(manifest.get("format_identifier") == BETA_STREAMED_FORMAT, "streamed transformer format is not minimax-h3-mlx-streamed-adaln-v1")
    _require(manifest.get("schema_version") == BETA_STREAMED_SCHEMA_VERSION, "streamed transformer schema version is not 1")
    _require(manifest.get("bounded") is False, "bounded streamed checkpoint cannot be selected as beta runtime")
    _require(manifest.get("verification_status") == "verified", "streamed transformer is not verified")
    _require(manifest.get("selected_blocks") == list(range(BETA_BLOCK_COUNT)), "streamed transformer does not contain all 50 selected blocks")
    source_checkpoint = manifest.get("source_checkpoint")
    _require(isinstance(source_checkpoint, dict), "streamed transformer source checkpoint linkage is missing")
    _require(source_checkpoint.get("logical_identity") == conventional_identity["index_sha256"], "streamed transformer source/index logical linkage is not the accepted conventional index")
    _require(manifest.get("source_safetensors_index_checksum") == conventional_identity["index_sha256"], "streamed transformer source safetensors index checksum is not linked to the accepted conventional index")
    _require(manifest.get("source_configuration_checksum") == conventional_identity["config_sha256"], "streamed transformer source config checksum is not linked to the accepted conventional config")
    _require(manifest.get("source_quantization_configuration_checksum") == conventional_identity["quant_config_sha256"], "streamed transformer source quantization checksum is not linked to the accepted conventional recipe")
    _require(manifest.get("source_tensor_count") == BETA_LOGICAL_TENSOR_COUNT, "streamed transformer source tensor count is not 1,050")
    _require(manifest.get("derived_base_tensor_count") == BETA_RESIDENT_TENSOR_COUNT, "streamed transformer resident tensor count is not 850")
    _require(manifest.get("sidecar_count") == BETA_SIDECAR_COUNT, "streamed transformer sidecar count is not 50")
    _require(manifest.get("sidecar_tensor_count") == BETA_SIDECAR_TENSOR_COUNT, "streamed transformer sidecar tensor count is not 200")
    _require(manifest.get("total_logical_tensor_count") == BETA_LOGICAL_TENSOR_COUNT, "streamed transformer logical tensor count is not 1,050")
    _require(manifest.get("original_checkpoint_modified") is False, "streamed transformer records that its source checkpoint was modified")

    quant_config = _read_json(quant_config_path, "streamed quantization config")
    _require(quant_config.get("bits") == BETA_CORE_BITS, "streamed transformer core quantization is not Q6")
    _require(quant_config.get("adaln_bits") == BETA_ADALN_BITS, "streamed transformer AdaLN quantization is not Q8")
    _require(quant_config.get("group_size") == BETA_GROUP_SIZE, "streamed transformer group size is not 64")
    _require(quant_config.get("quantize_adaln") is True, "streamed transformer does not enable AdaLN quantization")
    _require(quant_config.get("quantized_layers") == BETA_QUANTIZED_LAYER_COUNTS, "streamed transformer quantized-layer counts are not Q6 core/Q8 AdaLN")

    base_index = _read_json(base_index_path, "streamed base index")
    weight_map = base_index.get("weight_map")
    _require(isinstance(weight_map, dict), "streamed base index weight_map is missing")
    _require(len(weight_map) == BETA_RESIDENT_TENSOR_COUNT, "streamed base index resident count is not 850")
    _require(all(isinstance(key, str) for key in weight_map), "streamed base index tensor keys must be strings")
    _require(all(isinstance(value, str) for value in weight_map.values()), "streamed base index shard names must be strings")
    resident_block_adaln = sorted(key for key in weight_map if key.startswith("blocks.") and ".adaln_proj." in key)
    _require(not resident_block_adaln, "streamed base index contains resident block-AdaLN tensors")
    _require("final_layer.adaln_proj.linear.weight" in weight_map and "final_layer.adaln_proj.linear.bias" in weight_map, "streamed base index is missing final-layer AdaLN tensors")
    base_shards = sorted(set(weight_map.values()))
    _require(base_shards == [f"model-{index:05d}-of-00005.safetensors" for index in range(1, 6)], "streamed base shard topology is not the accepted five-shard layout")
    for shard in base_shards:
        _require_file(transformer_root / "base" / shard, f"streamed base shard {shard}")
    actual_base_payloads = sorted(path.name for path in (transformer_root / "base").glob("*.safetensors") if path.is_file())
    _require(actual_base_payloads == base_shards, "streamed base payload file set does not match its index")

    adaln_manifest = _read_json(adaln_manifest_path, "streamed AdaLN manifest")
    _require(adaln_manifest.get("format_identifier") == BETA_STREAMED_FORMAT, "streamed AdaLN manifest format is not accepted")
    _require(adaln_manifest.get("schema_version") == BETA_STREAMED_SCHEMA_VERSION, "streamed AdaLN manifest schema version is not 1")
    _require(adaln_manifest.get("bounded") is False, "bounded AdaLN manifest cannot be selected as beta runtime")
    blocks = adaln_manifest.get("blocks")
    _require(isinstance(blocks, dict) and set(blocks) == {str(index) for index in range(BETA_BLOCK_COUNT)}, "streamed AdaLN manifest does not describe exactly 50 blocks")
    expected_projection = {
        "quantization_bits": BETA_ADALN_BITS,
        "quantization_group_size": BETA_GROUP_SIZE,
        "logical_input_features": 2688,
        "logical_output_features": 96768,
        "packed_weight_shape": [96768, 672],
        "scales_shape": [96768, 42],
        "quantization_biases_shape": [96768, 42],
        "learned_bias_shape": [96768],
    }
    expected_roles = {
        "bias": ("learned_bias", "BF16", "unquantized", None, None),
        "biases": ("quantization_biases", "BF16", "affine", BETA_ADALN_BITS, BETA_GROUP_SIZE),
        "scales": ("scales", "BF16", "affine", BETA_ADALN_BITS, BETA_GROUP_SIZE),
        "weight": ("packed_weight", "U32", "affine", BETA_ADALN_BITS, BETA_GROUP_SIZE),
    }
    for index in range(BETA_BLOCK_COUNT):
        entry = blocks[str(index)]
        _require(isinstance(entry, dict), f"streamed AdaLN block {index} entry is malformed")
        _require(entry.get("block_index") == index, f"streamed AdaLN block {index} index is reordered")
        filename = f"block-{index:03d}.safetensors"
        _require(entry.get("sidecar_filename") == filename, f"streamed AdaLN block {index} sidecar filename is not canonical")
        _require_file(transformer_root / "adaln" / filename, f"streamed AdaLN sidecar {filename}")
        _require(entry.get("projection") == expected_projection, f"streamed AdaLN block {index} projection policy is not Q8/group-64")
        tensors = entry.get("tensors")
        _require(isinstance(tensors, list) and len(tensors) == 4, f"streamed AdaLN block {index} must contain four sidecar tensors")
        actual_keys = [item.get("tensor_key") for item in tensors if isinstance(item, dict)]
        expected_keys = [f"blocks.{index}.adaln_proj.linear.{suffix}" for suffix in expected_roles]
        _require(actual_keys == sorted(expected_keys), f"streamed AdaLN block {index} tensor keys are not canonical")
        for item in tensors:
            key = item.get("tensor_key")
            prefix = f"blocks.{index}.adaln_proj.linear."
            _require(isinstance(key, str) and key.startswith(prefix), f"streamed AdaLN block {index} has an unexpected tensor key")
            suffix = key[len(prefix):]
            _require(suffix in expected_roles, f"streamed AdaLN block {index} has an unexpected tensor key")
            role, dtype, quant_format, bits, group_size = expected_roles[suffix]
            _require(item.get("tensor_role") == role, f"streamed AdaLN block {index} role does not match its key")
            _require(item.get("source_dtype") == dtype, f"streamed AdaLN block {index} dtype is not accepted")
            _require(item.get("quantization_format") == quant_format, f"streamed AdaLN block {index} quantization format is not accepted")
            _require(item.get("quantization_bits") == bits and item.get("group_size") == group_size, f"streamed AdaLN block {index} quantization policy is not accepted")
    actual_sidecars = sorted(path.name for path in (transformer_root / "adaln").glob("*.safetensors") if path.is_file())
    _require(actual_sidecars == [f"block-{index:03d}.safetensors" for index in range(BETA_BLOCK_COUNT)], "streamed AdaLN sidecar payload file set does not match the manifest")

    return {
        "format": BETA_STREAMED_FORMAT,
        "schema_version": BETA_STREAMED_SCHEMA_VERSION,
        "conversion_manifest_path": str(manifest_path.resolve()),
        "conversion_manifest_sha256": _sha256_file(manifest_path, "streamed conversion manifest"),
        "source_index_sha256": conventional_identity["index_sha256"],
        "source_config_sha256": conventional_identity["config_sha256"],
        "source_quant_config_sha256": conventional_identity["quant_config_sha256"],
        "source_sha256": BETA_SOURCE_SHA256,
        "source_bytes": BETA_SOURCE_BYTES,
        "qkv_source_layout": BETA_QKV_SOURCE_LAYOUT,
        "qkv_canonical_layout": BETA_QKV_CANONICAL_LAYOUT,
        "qkv_tensors_reconciled": BETA_QKV_TENSOR_COUNT,
        "qkv_authorization_schema": BETA_QKV_RECEIPT_SCHEMA,
        "qkv_authorization_receipt_sha256": conventional_identity["authorization_receipt_sha256"],
        "core_bits": BETA_CORE_BITS,
        "adaln_bits": BETA_ADALN_BITS,
        "group_size": BETA_GROUP_SIZE,
        "logical_tensor_count": BETA_LOGICAL_TENSOR_COUNT,
        "resident_tensor_count": BETA_RESIDENT_TENSOR_COUNT,
        "resident_block_adaln_tensor_count": BETA_RESIDENT_BLOCK_ADALN_TENSOR_COUNT,
        "sidecar_count": BETA_SIDECAR_COUNT,
        "sidecar_tensor_count": BETA_SIDECAR_TENSOR_COUNT,
        "selected_blocks": list(range(BETA_BLOCK_COUNT)),
    }


def resolve_runtime(runtime_id: str, assets_root: str | Path | None) -> ResolvedRuntime:
    """Resolve and validate one explicit named runtime before model loading."""
    if runtime_id != BETA_RUNTIME_ID:
        raise RuntimeSelectionError("UNKNOWN_RUNTIME", f"unknown runtime identifier: {runtime_id!r}")
    if assets_root is None:
        raise RuntimeSelectionError(
            "RUNTIME_ASSETS_REQUIRED",
            f"{BETA_RUNTIME_ID} requires --runtime-assets or MINIMAX_H3_RUNTIME_ASSETS",
        )

    assets_root_path = _resolved(Path(assets_root))
    profile_root = _resolved(assets_root_path / BETA_RUNTIME_ASSET_DIR)
    _require_directory(profile_root, f"runtime profile directory {BETA_RUNTIME_ASSET_DIR!r}")
    checkpoint_root = _resolve_link(profile_root, BETA_CHECKPOINT_LINK)
    transformer_root = _resolve_link(profile_root, BETA_TRANSFORMER_LINK)
    conventional_root = _resolve_link(profile_root, BETA_CONVENTIONAL_LINK)

    qkv_receipt = _validate_qkv_authorization_receipt()
    conventional_identity = _validate_conventional_root(conventional_root, qkv_receipt)
    transformer_identity = _validate_streamed_transformer(transformer_root, conventional_identity)
    transformer_identity = {
        **transformer_identity,
        "root": str(transformer_root),
        **_validate_transformer_config(transformer_root),
    }
    surrounding_identity = _validate_surrounding_root(checkpoint_root)

    return ResolvedRuntime(
        runtime_id=BETA_RUNTIME_ID,
        asset_profile_root=profile_root,
        checkpoint_root=checkpoint_root,
        transformer_root=transformer_root,
        conventional_root=conventional_root,
        qwen_root=_resolved(checkpoint_root / "text_encoder"),
        video_vae_root=_resolved(checkpoint_root / "video_vae"),
        audio_vae_root=_resolved(checkpoint_root / "audio_vae"),
        transformer_identity=transformer_identity,
        surrounding_identity=surrounding_identity,
    )

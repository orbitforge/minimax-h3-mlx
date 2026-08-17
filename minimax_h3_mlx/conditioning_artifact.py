"""Canonical, CPU-readable MiniMax-H3 text-conditioning artifacts.

The H3 transformer consumes the unnormalized Qwen hidden state after layer 50
and one modality tag per conditioning row.  This module keeps that boundary
independent from MLX so artifacts can be validated and round-tripped without
constructing Qwen, the transformer, or a VAE.

The payload is intentionally stored as raw BF16 bits in a NumPy ``uint16``
array.  NumPy has no portable native BF16 dtype on all supported paths, and a
float32 ``.npz`` payload would make the serialized dtype ambiguous.  Replaying
the payload converts those exact BF16 values to float32 only as an
interchange step before explicitly casting to MLX BF16.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


ARTIFACT_SCHEMA_ID = "minimax-h3-mlx-conditioning-artifact"
ARTIFACT_SCHEMA_VERSION = 1
CONDITIONING_DTYPE = "bfloat16"
CONDITIONING_WIDTH = 5120
SELECTED_HIDDEN_STATE = "hidden_states[50]"

_ARRAY_KEYS = frozenset({
    "metadata_json",
    "text_conditioning_bf16_bits",
    "text_token_tags",
    "token_ids",
})
_WEIGHT_SUFFIXES = frozenset({".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".gguf"})


class ConditioningArtifactError(ValueError):
    """Raised when a conditioning artifact cannot be trusted for replay."""


def _canonical_json(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ConditioningArtifactError(f"artifact metadata is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _dtype_name(value: Any) -> str:
    return str(value).removeprefix("mlx.core.")


def _prompt_digest(prompt: str) -> tuple[str, int]:
    if not isinstance(prompt, str):
        raise ConditioningArtifactError(f"prompt must be a string, got {type(prompt).__name__}")
    encoded = prompt.encode("utf-8")
    return _sha256(encoded), len(encoded)


def _little_endian_uint16(value: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=np.uint16)).astype("<u2", copy=False)


def _array_checksum(value: np.ndarray, logical_dtype: str) -> str:
    array = np.ascontiguousarray(value)
    descriptor = _canonical_json({
        "shape": [int(item) for item in array.shape],
        "dtype": logical_dtype,
    })
    raw = array.tobytes(order="C")
    return _sha256(descriptor + b"\0" + raw)


def _bf16_bits_checksum(bits: np.ndarray) -> str:
    return _array_checksum(_little_endian_uint16(bits), CONDITIONING_DTYPE)


def bfloat16_bits_from_float32(value: Any) -> np.ndarray:
    """Return exact BF16 bit patterns from already-BF16-representable float32 values.

    This helper refuses implicit narrowing.  A real MLX BF16 array is first
    materialized as float32; because its values are already BF16 values, every
    lower sixteen float32 mantissa bits must be zero.  A caller holding an
    arbitrary float32 array must explicitly quantize it before using this
    boundary.
    """
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if not np.all(np.isfinite(array)):
        raise ConditioningArtifactError("conditioning payload contains non-finite values")
    words = array.view(np.uint32)
    if np.any(words & np.uint32(0xFFFF)):
        raise ConditioningArtifactError(
            "conditioning payload is float32 but not exactly BF16-representable; refusing implicit narrowing"
        )
    return np.ascontiguousarray((words >> np.uint32(16)).astype("<u2", copy=False))


def bfloat16_bits_to_float32(bits: Any) -> np.ndarray:
    """Decode raw BF16 bit patterns to exact float32 values."""
    array = np.ascontiguousarray(np.asarray(bits, dtype=np.uint16)).astype("<u2", copy=False)
    words = array.astype(np.uint32, copy=False) << np.uint32(16)
    return np.ascontiguousarray(words.view(np.float32))


def _mlx_to_float32(value: Any) -> np.ndarray:
    module = value.__class__.__module__
    is_mlx = bool(getattr(value, "__mlx_array__", False) or module.startswith("mlx."))
    if is_mlx:
        try:
            import mlx.core as mx

            if _dtype_name(value.dtype) != CONDITIONING_DTYPE:
                raise ConditioningArtifactError(
                    f"conditioning payload must be MLX {CONDITIONING_DTYPE}, got {_dtype_name(value.dtype)}"
                )
            value = value.astype(mx.float32)
            mx.eval(value)
        except ConditioningArtifactError:
            raise
        except Exception as exc:
            raise ConditioningArtifactError(f"could not materialize the MLX conditioning payload: {exc}") from exc
    return np.array(value, dtype=np.float32, copy=True)


def _conditioning_bits_from_value(value: Any) -> np.ndarray:
    dtype = _dtype_name(getattr(value, "dtype", ""))
    if dtype != CONDITIONING_DTYPE:
        raise ConditioningArtifactError(
            f"conditioning payload must have logical dtype {CONDITIONING_DTYPE}, got {dtype or 'unknown'}"
        )
    return bfloat16_bits_from_float32(_mlx_to_float32(value))


def _normalise_token_ids(value: Any) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    if array.ndim != 2 or array.shape[0] != 1:
        raise ConditioningArtifactError(f"token_ids must have shape (1, token_count), got {array.shape}")
    return array


def _normalise_token_tags(value: Any, token_count: int) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.int32))
    if array.shape != (token_count,):
        raise ConditioningArtifactError(
            f"text_token_tags must have shape ({token_count},), got {array.shape}"
        )
    if np.any(~np.isin(array, np.asarray((0, 1), dtype=np.int32))):
        raise ConditioningArtifactError("text_token_tags may contain only H3 text/video tags 0 or 1")
    if np.any(array != 1):
        raise ConditioningArtifactError("text-only conditioning artifacts require text_token_tags value 1")
    return array


def _manifest_digest(directory: Path) -> dict[str, Any]:
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        return {"present": False, "file_count": 0, "sha256": None}
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == ".DS_Store" or any(part == "__pycache__" for part in path.parts):
            continue
        if path.suffix.lower() in _WEIGHT_SUFFIXES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({
            "path": path.relative_to(directory).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": digest.hexdigest(),
        })
    return {
        "present": True,
        "file_count": len(entries),
        "sha256": _sha256(_canonical_json(entries)),
    }


def _weights_manifest_digest(directory: Path) -> dict[str, Any]:
    """Hash encoder weight files without importing or constructing the encoder."""
    directory = directory.expanduser().resolve()
    if not directory.is_dir():
        return {"present": False, "file_count": 0, "sha256": None}
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append({
            "path": path.relative_to(directory).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": digest.hexdigest(),
        })
    return {
        "present": bool(entries),
        "file_count": len(entries),
        "sha256": _sha256(_canonical_json(entries)) if entries else None,
    }


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_encoder_provenance(
    checkpoint_root: str | Path,
    *,
    selected_layer: int = 50,
    logical_dtype: str = CONDITIONING_DTYPE,
) -> dict[str, Any]:
    """Build replay identity from checkpoint metadata without loading Qwen."""
    root = Path(checkpoint_root).expanduser().resolve()
    model_dir = root / "text_encoder"
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise ConditioningArtifactError(f"text encoder config is missing: {config_path}")
    try:
        raw = json.loads(config_path.read_text())
    except (OSError, ValueError) as exc:
        raise ConditioningArtifactError(f"could not read text encoder config {config_path}: {exc}") from exc
    text_config = raw.get("text_config", {})
    tokenizer_dir = root / "tokenizer" if (root / "tokenizer").is_dir() else model_dir
    processor_dir = root / "processor" if (root / "processor").is_dir() else model_dir
    return {
        "family": raw.get("model_type", "qwen3_vl"),
        "model_index_sha256": _file_sha256(root / "model_index.json"),
        "config": {
            "sha256": _file_sha256(config_path),
            "model_type": raw.get("model_type", "qwen3_vl"),
            "full_decoder_layers": int(text_config.get("num_hidden_layers", -1)),
            "hidden_size": int(text_config.get("hidden_size", -1)),
        },
        "selected_state": {
            "hidden_state": f"hidden_states[{int(selected_layer)}]",
            "normalization": "unnormalized-pre-final-norm",
            "selected_decoder_layer": int(selected_layer),
            "logical_dtype": logical_dtype,
        },
        "weights": {
            "source": "text_encoder",
            "manifest": _weights_manifest_digest(model_dir),
        },
        "tokenizer": {
            "source": "checkpoint/tokenizer" if tokenizer_dir == root / "tokenizer" else "text_encoder",
            "manifest": _manifest_digest(tokenizer_dir),
        },
        "processor": {
            "source": "checkpoint/processor" if processor_dir == root / "processor" else "text_encoder",
            "manifest": _manifest_digest(processor_dir),
        },
    }


def _validate_canonical_encoder_provenance(encoder: Mapping[str, Any]) -> None:
    if not isinstance(encoder, Mapping):
        raise ConditioningArtifactError("artifact is missing encoder identity metadata")
    config = encoder.get("config")
    if not isinstance(config, Mapping):
        raise ConditioningArtifactError("artifact is missing encoder configuration identity")
    try:
        hidden_size = int(config.get("hidden_size", -1))
        full_decoder_layers = int(config.get("full_decoder_layers", -1))
    except (TypeError, ValueError) as exc:
        raise ConditioningArtifactError("artifact encoder configuration identity is invalid") from exc
    if hidden_size != CONDITIONING_WIDTH:
        raise ConditioningArtifactError("artifact encoder hidden size is incompatible with H3")
    if full_decoder_layers <= 50:
        raise ConditioningArtifactError("artifact encoder does not prove a layer-50 hidden state")
    selected = encoder.get("selected_state")
    if not isinstance(selected, Mapping):
        raise ConditioningArtifactError("artifact is missing selected Qwen hidden-state metadata")
    if (
        selected.get("hidden_state") != SELECTED_HIDDEN_STATE
        or selected.get("selected_decoder_layer") != 50
        or selected.get("logical_dtype") != CONDITIONING_DTYPE
        or selected.get("normalization") != "unnormalized-pre-final-norm"
    ):
        raise ConditioningArtifactError("artifact Qwen state is not the production layer-50 BF16 boundary")
    for name in ("weights", "tokenizer", "processor"):
        identity = encoder.get(name)
        if not isinstance(identity, Mapping) or not isinstance(identity.get("manifest"), Mapping):
            raise ConditioningArtifactError(f"artifact is missing {name} identity metadata")



def _validate_encoder_provenance(encoder: Mapping[str, Any]) -> None:
    _validate_canonical_encoder_provenance(encoder)


def _artifact_identity(metadata: Mapping[str, Any]) -> str:
    material = dict(metadata)
    material.pop("artifact_identity", None)
    return _sha256(_canonical_json(material))


def _required_metadata(metadata: Mapping[str, Any]) -> None:
    required = {
        "schema",
        "artifact_identity",
        "request",
        "encoder",
        "h3_boundary",
        "conditioning",
        "token_metadata",
        "postprocessing",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ConditioningArtifactError(f"artifact metadata is missing required fields: {missing}")
    schema = metadata["schema"]
    if not isinstance(schema, Mapping) or schema.get("id") != ARTIFACT_SCHEMA_ID:
        raise ConditioningArtifactError("unsupported conditioning artifact schema")
    if schema.get("version") != ARTIFACT_SCHEMA_VERSION:
        raise ConditioningArtifactError(
            f"unsupported conditioning artifact schema version: {schema.get('version')!r}"
        )
    if not isinstance(metadata["request"], Mapping) or metadata["request"].get("mode") != "text-only":
        raise ConditioningArtifactError("only text-only conditioning artifacts are replayable in this slice")
    _validate_encoder_provenance(metadata["encoder"])
    boundary = metadata["h3_boundary"]
    if not isinstance(boundary, Mapping):
        raise ConditioningArtifactError("artifact is missing the H3 conditioning boundary metadata")
    if (
        boundary.get("required_payload") != ["text_conditioning", "text_token_tags"]
        or boundary.get("text_input") != "prompt_embeds"
        or boundary.get("text_conditioning_consumed_by") != "DiT.condition_proj"
        or boundary.get("packing_input") != "text_token_tags"
        or boundary.get("attention_mask_required_by_h3") is not False
        or boundary.get("prompt_derived_state_outside_payload") != []
    ):
        raise ConditioningArtifactError("artifact H3 boundary does not identify the required replay payload")
    conditioning = metadata["conditioning"]
    if not isinstance(conditioning, Mapping):
        raise ConditioningArtifactError("artifact is missing conditioning metadata")
    if (
        conditioning.get("array_key") != "text_conditioning_bf16_bits"
        or conditioning.get("representation") != "raw-bfloat16-bits-in-uint16"
        or conditioning.get("dtype") != CONDITIONING_DTYPE
    ):
        raise ConditioningArtifactError("conditioning artifact dtype is not bfloat16")
    if not isinstance(metadata["token_metadata"], Mapping):
        raise ConditioningArtifactError("artifact is missing token metadata")
    token_metadata = metadata["token_metadata"]
    if (
        token_metadata.get("token_ids_array_key") != "token_ids"
        or token_metadata.get("token_ids_dtype") != "int32"
        or token_metadata.get("text_token_tags_array_key") != "text_token_tags"
        or token_metadata.get("text_token_tags_dtype") != "int32"
        or token_metadata.get("tokenizer_call") != "tokenizer(prompt, add_special_tokens=False)"
        or token_metadata.get("chat_template") is not None
        or token_metadata.get("truncation") != "disabled"
        or token_metadata.get("max_length") is not None
        or token_metadata.get("token_presence_mask") is not None
    ):
        raise ConditioningArtifactError("artifact tokenizer presentation metadata is incompatible")
    postprocessing = metadata["postprocessing"]
    expected_selected_state = SELECTED_HIDDEN_STATE
    expected_projection = "none before H3 condition_proj"
    if (
        not isinstance(postprocessing, Mapping)
        or postprocessing.get("selected_state") != expected_selected_state
        or postprocessing.get("normalization") != "none"
        or postprocessing.get("projection") != expected_projection
        or postprocessing.get("serialization") != "exact raw BF16 bits"
    ):
        raise ConditioningArtifactError("artifact is missing postprocessing metadata")


@dataclass(frozen=True)
class LoadedConditioningArtifact:
    """Validated artifact payload and immutable metadata."""

    path: Path
    metadata: dict[str, Any]
    conditioning_bits: np.ndarray
    text_token_tags: np.ndarray
    token_ids: np.ndarray

    @property
    def artifact_identity(self) -> str:
        return str(self.metadata["artifact_identity"])

    @property
    def tensor_checksum(self) -> str:
        return str(self.metadata["conditioning"]["tensor_checksum"])

    @property
    def token_count(self) -> int:
        return int(self.conditioning_bits.shape[1])

    @property
    def conditioning_shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.conditioning_bits.shape)

    def conditioning_float32(self) -> np.ndarray:
        return bfloat16_bits_to_float32(self.conditioning_bits)

    def conditioning_mlx(self, mx: Any) -> Any:
        """Convert exact decoded BF16 values to an explicit MLX BF16 array."""
        result = mx.array(self.conditioning_float32(), dtype=mx.float32).astype(mx.bfloat16)
        mx.eval(result)
        return result


def _validate_arrays(
    metadata: Mapping[str, Any],
    conditioning_bits: np.ndarray,
    text_token_tags: np.ndarray,
    token_ids: np.ndarray,
) -> None:
    conditioning = metadata["conditioning"]
    expected_shape = tuple(int(item) for item in conditioning.get("shape", ()))
    if conditioning_bits.dtype != np.dtype("uint16"):
        raise ConditioningArtifactError(
            f"serialized BF16 payload has dtype {conditioning_bits.dtype}, expected uint16 bit storage"
        )
    if conditioning_bits.ndim != 3 or conditioning_bits.shape != expected_shape:
        raise ConditioningArtifactError(
            f"conditioning shape mismatch: payload {conditioning_bits.shape}, metadata {expected_shape}"
        )
    if conditioning_bits.shape[0] != 1 or conditioning_bits.shape[2] != CONDITIONING_WIDTH:
        raise ConditioningArtifactError(
            f"conditioning shape must be (1, token_count, {CONDITIONING_WIDTH}), got {conditioning_bits.shape}"
        )
    if not np.all(np.isfinite(bfloat16_bits_to_float32(conditioning_bits))):
        raise ConditioningArtifactError("conditioning payload contains non-finite BF16 values")
    if conditioning.get("token_count") != int(conditioning_bits.shape[1]):
        raise ConditioningArtifactError("conditioning token count does not match its shape")
    if text_token_tags.dtype != np.dtype("int32") or text_token_tags.shape != (conditioning_bits.shape[1],):
        raise ConditioningArtifactError("text_token_tags shape or dtype is incompatible with conditioning")
    if np.any(~np.isin(text_token_tags, np.asarray((0, 1), dtype=np.int32))):
        raise ConditioningArtifactError("text_token_tags contain unsupported modality values")
    if np.any(text_token_tags != 1):
        raise ConditioningArtifactError("text-only conditioning artifacts require text_token_tags value 1")
    if token_ids.dtype != np.dtype("int32") or token_ids.shape != (1, conditioning_bits.shape[1]):
        raise ConditioningArtifactError("token_ids shape or dtype is incompatible with conditioning")
    if _bf16_bits_checksum(conditioning_bits) != conditioning.get("tensor_checksum"):
        raise ConditioningArtifactError("conditioning tensor checksum mismatch")
    token_metadata = metadata["token_metadata"]
    if _array_checksum(text_token_tags, "int32") != token_metadata.get("text_token_tags_checksum"):
        raise ConditioningArtifactError("text_token_tags checksum mismatch")
    if _array_checksum(token_ids, "int32") != token_metadata.get("token_ids_checksum"):
        raise ConditioningArtifactError("token_ids checksum mismatch")


def _load_npz(path: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            keys = set(loaded.files)
            if keys != _ARRAY_KEYS:
                missing = sorted(_ARRAY_KEYS - keys)
                extra = sorted(keys - _ARRAY_KEYS)
                raise ConditioningArtifactError(
                    f"conditioning artifact key set mismatch (missing={missing}, extra={extra})"
                )
            raw_metadata = loaded["metadata_json"]
            if raw_metadata.shape != ():
                raise ConditioningArtifactError("artifact metadata_json must be a scalar")
            try:
                metadata = json.loads(str(raw_metadata.item()))
            except (TypeError, ValueError) as exc:
                raise ConditioningArtifactError(f"artifact metadata_json is invalid: {exc}") from exc
            bits = np.array(loaded["text_conditioning_bf16_bits"], copy=True)
            tags = np.array(loaded["text_token_tags"], copy=True)
            token_ids = np.array(loaded["token_ids"], copy=True)
    except ConditioningArtifactError:
        raise
    except Exception as exc:
        raise ConditioningArtifactError(f"could not read conditioning artifact {path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ConditioningArtifactError("artifact metadata_json must contain an object")
    _required_metadata(metadata)
    if metadata.get("artifact_identity") != _artifact_identity(metadata):
        raise ConditioningArtifactError("conditioning artifact identity hash mismatch")
    _validate_arrays(metadata, bits, tags, token_ids)
    return metadata, bits, tags, token_ids


def load_conditioning_artifact(path: str | Path) -> LoadedConditioningArtifact:
    """Read and validate the self-contained artifact without loading MLX."""
    artifact_path = Path(path).expanduser().resolve()
    if not artifact_path.is_file():
        raise ConditioningArtifactError(f"conditioning artifact does not exist: {artifact_path}")
    metadata, bits, tags, token_ids = _load_npz(artifact_path)
    return LoadedConditioningArtifact(artifact_path, metadata, bits, tags, token_ids)


def validate_conditioning_artifact(
    artifact: LoadedConditioningArtifact,
    *,
    checkpoint_root: str | Path | None = None,
    prompt: str | None = None,
    text_dim: int = CONDITIONING_WIDTH,
) -> LoadedConditioningArtifact:
    """Apply the current checkpoint, prompt, and H3 consumer compatibility gates."""
    _required_metadata(artifact.metadata)
    if artifact.metadata.get("artifact_identity") != _artifact_identity(artifact.metadata):
        raise ConditioningArtifactError("conditioning artifact identity hash mismatch")
    if int(text_dim) != CONDITIONING_WIDTH:
        raise ConditioningArtifactError(
            f"this artifact contract requires H3 text_dim {CONDITIONING_WIDTH}, got {text_dim}"
        )
    if tuple(artifact.conditioning_bits.shape) != tuple(artifact.metadata["conditioning"]["shape"]):
        raise ConditioningArtifactError("conditioning shape changed after artifact load")
    _validate_arrays(
        artifact.metadata,
        artifact.conditioning_bits,
        artifact.text_token_tags,
        artifact.token_ids,
    )
    request = artifact.metadata["request"]
    if prompt is not None:
        prompt_sha256, byte_count = _prompt_digest(prompt)
        if prompt_sha256 != request.get("prompt_sha256") or byte_count != request.get("prompt_utf8_byte_count"):
            raise ConditioningArtifactError("prompt identity does not match the conditioning artifact")
    if checkpoint_root is not None:
        encoder = artifact.metadata["encoder"]
        identity_to_compare = encoder
        if not isinstance(identity_to_compare, Mapping):
            raise ConditioningArtifactError("artifact is missing the canonical H3 encoder identity")
        selected = identity_to_compare.get("selected_state", {})
        expected = build_encoder_provenance(
            checkpoint_root,
            selected_layer=int(selected.get("selected_decoder_layer", 50)),
            logical_dtype=str(selected.get("logical_dtype", CONDITIONING_DTYPE)),
        )
        if expected != identity_to_compare:
            raise ConditioningArtifactError("text-encoder/tokenizer/processor identity does not match the artifact")
    return artifact


def _metadata_for_artifact(
    *,
    prompt: str,
    conditioning_bits: np.ndarray,
    text_token_tags: np.ndarray,
    token_ids: np.ndarray,
    encoder_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_sha256, byte_count = _prompt_digest(prompt)
    token_count = int(conditioning_bits.shape[1])
    selected_state = SELECTED_HIDDEN_STATE
    projection = "none before H3 condition_proj"
    metadata: dict[str, Any] = {
        "schema": {"id": ARTIFACT_SCHEMA_ID, "version": ARTIFACT_SCHEMA_VERSION},
        "request": {
            "mode": "text-only",
            "prompt_sha256": prompt_sha256,
            "prompt_utf8_byte_count": byte_count,
        },
        "encoder": json.loads(json.dumps(dict(encoder_provenance), allow_nan=False)),
        "h3_boundary": {
            "required_payload": ["text_conditioning", "text_token_tags"],
            "text_input": "prompt_embeds",
            "text_conditioning_consumed_by": "DiT.condition_proj",
            "packing_input": "text_token_tags",
            "attention_mask_required_by_h3": False,
            "prompt_derived_state_outside_payload": [],
        },
        "conditioning": {
            "array_key": "text_conditioning_bf16_bits",
            "representation": "raw-bfloat16-bits-in-uint16",
            "shape": [int(item) for item in conditioning_bits.shape],
            "dtype": CONDITIONING_DTYPE,
            "token_count": token_count,
            "tensor_checksum": _bf16_bits_checksum(conditioning_bits),
        },
        "token_metadata": {
            "token_ids_array_key": "token_ids",
            "token_ids_shape": [int(item) for item in token_ids.shape],
            "token_ids_dtype": "int32",
            "token_ids_checksum": _array_checksum(token_ids, "int32"),
            "text_token_tags_array_key": "text_token_tags",
            "text_token_tags_shape": [int(item) for item in text_token_tags.shape],
            "text_token_tags_dtype": "int32",
            "text_token_tags_checksum": _array_checksum(text_token_tags, "int32"),
            "tokenizer_call": "tokenizer(prompt, add_special_tokens=False)",
            "chat_template": None,
            "truncation": "disabled",
            "max_length": None,
            "token_presence_mask": None,
        },
        "postprocessing": {
            "selected_state": selected_state,
            "normalization": "none",
            "projection": projection,
            "serialization": "exact raw BF16 bits",
        },
    }
    metadata["artifact_identity"] = _artifact_identity(metadata)
    return metadata


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing conditioning artifact: {path}")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _create_from_bits(
    path: str | Path,
    *,
    prompt: str,
    conditioning_bits: Any,
    text_token_tags: Any,
    token_ids: Any,
    encoder_provenance: Mapping[str, Any],
) -> LoadedConditioningArtifact:
    artifact_path = Path(path).expanduser().resolve()
    bits = np.ascontiguousarray(np.asarray(conditioning_bits, dtype=np.uint16)).astype("<u2", copy=False)
    if bits.ndim != 3 or bits.shape[0] != 1 or bits.shape[2] != CONDITIONING_WIDTH:
        raise ConditioningArtifactError(
            f"conditioning payload must have shape (1, token_count, {CONDITIONING_WIDTH}), got {bits.shape}"
        )
    ids = _normalise_token_ids(token_ids)
    if ids.shape[1] != bits.shape[1]:
        raise ConditioningArtifactError("token_ids token count does not match conditioning")
    tags = _normalise_token_tags(text_token_tags, int(bits.shape[1]))
    metadata = _metadata_for_artifact(
        prompt=prompt,
        conditioning_bits=bits,
        text_token_tags=tags,
        token_ids=ids,
        encoder_provenance=encoder_provenance,
    )
    _write_npz(
        artifact_path,
        {
            "metadata_json": np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
            "text_conditioning_bf16_bits": bits,
            "text_token_tags": tags,
            "token_ids": ids,
        },
    )
    return load_conditioning_artifact(artifact_path)


def create_conditioning_artifact(
    path: str | Path,
    *,
    prompt: str,
    conditioning: Any,
    text_token_tags: Any,
    token_ids: Any,
    encoder_provenance: Mapping[str, Any],
) -> LoadedConditioningArtifact:
    """Serialize a materialized MLX BF16 conditioning result atomically."""
    bits = _conditioning_bits_from_value(conditioning)
    return _create_from_bits(
        path,
        prompt=prompt,
        conditioning_bits=bits,
        text_token_tags=text_token_tags,
        token_ids=token_ids,
        encoder_provenance=encoder_provenance,
    )


def create_conditioning_artifact_from_bits(
    path: str | Path,
    *,
    prompt: str,
    conditioning_bits: Any,
    text_token_tags: Any,
    token_ids: Any,
    encoder_provenance: Mapping[str, Any],
) -> LoadedConditioningArtifact:
    """Test and tooling entry point for already canonical raw BF16 bits."""
    return _create_from_bits(
        path,
        prompt=prompt,
        conditioning_bits=conditioning_bits,
        text_token_tags=text_token_tags,
        token_ids=token_ids,
        encoder_provenance=encoder_provenance,
    )


__all__ = [
    "ARTIFACT_SCHEMA_ID",
    "ARTIFACT_SCHEMA_VERSION",
    "CONDITIONING_DTYPE",
    "CONDITIONING_WIDTH",
    "ConditioningArtifactError",
    "LoadedConditioningArtifact",
    "bfloat16_bits_from_float32",
    "bfloat16_bits_to_float32",
    "build_encoder_provenance",
    "create_conditioning_artifact",
    "create_conditioning_artifact_from_bits",
    "load_conditioning_artifact",
    "validate_conditioning_artifact",
]

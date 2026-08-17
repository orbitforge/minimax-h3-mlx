"""MLX-free LightX task routing and static ``transformer_ref`` audit.

This module owns only admission, discovery, and metadata inspection.  It never imports MLX,
constructs a transformer, opens a safetensors payload, or performs a forward pass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


TASK_FL2VA_T2VA = "FL2VA/T2VA"
TASK_REF2VA = "Ref2VA"
ORDINARY_TRANSFORMER_PARTITION = "transformer"
TRANSFORMER_REF_PARTITION = "transformer_ref"

STATIC_COMPATIBLE = "STATIC_COMPATIBLE"
STATIC_COMPATIBLE_WITH_ADAPTATION = "STATIC_COMPATIBLE_WITH_ADAPTATION"
STATIC_INCOMPATIBLE = "STATIC_INCOMPATIBLE"
UNKNOWN = "UNKNOWN"

DISCOVERY_ABSENT = "ABSENT"
DISCOVERY_PRESENT = "PRESENT"
DISCOVERY_INVALID = "INVALID"

TRANSFORMER_REF_UNAVAILABLE = "TRANSFORMER_REF_UNAVAILABLE"
TRANSFORMER_REF_STATIC_INCOMPATIBLE = "TRANSFORMER_REF_STATIC_INCOMPATIBLE"
TRANSFORMER_REF_PATH_MISMATCH = "TRANSFORMER_REF_PATH_MISMATCH"
REF2VA_REFERENCE_INPUT_NOT_IMPLEMENTED = "REF2VA_REFERENCE_INPUT_NOT_IMPLEMENTED"

_CONFIG_METADATA_FIELDS = {"_class_name", "_diffusers_version"}
_EXPECTED_CONFIG_VALUES: Mapping[str, Any] = {
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
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "final_norm_eps": 1e-5,
}
# The MLX config supplies this field's released default when the Diffusers config omits it.
_OPTIONAL_DEFAULT_CONFIG_VALUES: Mapping[str, Any] = {"rope_theta": 10000.0}

_EXPECTED_TOP_LEVEL_FAMILIES = {
    "audio_patch_proj",
    "video_patch_proj",
    "condition_proj",
    "time_embedder",
    "token_refiner",
    "blocks",
    "final_layer",
}
_QUANTIZED_WEIGHT_SUFFIXES = (".scales", ".biases")


class TransformerRoutingError(RuntimeError):
    """Fail-closed task routing error with a stable operator-facing code."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True)
class TransformerRefStaticAudit:
    """Header/config-only compatibility result for one transformer partition."""

    checkpoint_path: Path
    verdict: str
    config_path: Path | None
    weight_index_path: Path | None
    weight_shards: tuple[Path, ...]
    config_fields: tuple[str, ...] = ()
    missing_config_fields: tuple[str, ...] = ()
    extra_config_fields: tuple[str, ...] = ()
    mismatched_config_fields: tuple[str, ...] = ()
    missing_tensor_names: tuple[str, ...] = ()
    extra_tensor_names: tuple[str, ...] = ()
    missing_tensor_families: tuple[str, ...] = ()
    extra_tensor_families: tuple[str, ...] = ()
    indexed_tensor_count: int = 0
    reason: str = ""

    @property
    def validated(self) -> bool:
        return self.verdict == STATIC_COMPATIBLE


@dataclass(frozen=True)
class TransformerRefDiscovery:
    """Resolved paths and metadata for the exact ``<checkpoint>/transformer_ref`` candidate."""

    checkpoint_root: Path
    model_index_path: Path | None
    transformer_ref_path: Path
    config_path: Path | None
    weight_index_path: Path | None
    weight_shards: tuple[Path, ...]
    status: str
    model_index_partition: str | None
    model_index_tasks: tuple[str, ...]
    reason: str
    audit: TransformerRefStaticAudit

    @property
    def available(self) -> bool:
        return self.status == DISCOVERY_PRESENT and self.audit.validated


@dataclass(frozen=True)
class TransformerRoute:
    """A task-selected partition path; Ref2VA routes never alias ordinary ``transformer``."""

    task: str
    partition: str
    path: Path
    discovery: TransformerRefDiscovery | None = None


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _same_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, tuple):
        return tuple(actual) == expected if isinstance(actual, (list, tuple)) else False
    return actual == expected


def _base_tensor_name(name: str) -> str:
    for suffix in _QUANTIZED_WEIGHT_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)] + ".weight"
    return name


def _expected_tensor_names() -> tuple[str, ...]:
    names: set[str] = {
        "audio_patch_proj.bias",
        "audio_patch_proj.weight",
        "video_patch_proj.bias",
        "video_patch_proj.weight",
        "condition_proj.bias",
        "condition_proj.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_out.bias",
        "time_embedder.proj_out.weight",
        "token_refiner.final_norm.weight",
        "final_layer.adaln_proj.linear.bias",
        "final_layer.adaln_proj.linear.weight",
        "final_layer.audio_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.norm.weight",
        "final_layer.video_out.bias",
        "final_layer.video_out.weight",
    }
    token_suffixes = (
        "attn.k_norm.weight",
        "attn.out_proj.weight",
        "attn.q_norm.weight",
        "attn.qkv_proj.weight",
        "mlp.fc1.weight",
        "mlp.fc2.weight",
        "norm1.weight",
        "norm2.weight",
    )
    for index in range(2):
        names.update(f"token_refiner.blocks.{index}.{suffix}" for suffix in token_suffixes)
    block_suffixes = (
        "adaln_proj.linear.bias",
        "adaln_proj.linear.weight",
        *token_suffixes,
    )
    for index in range(50):
        names.update(f"blocks.{index}.{suffix}" for suffix in block_suffixes)
    return tuple(sorted(names))


def _quantized_extra_names() -> set[str]:
    names = set()
    for base in _expected_tensor_names():
        if base.endswith(".weight") and (
            ".qkv_proj." in base
            or ".out_proj." in base
            or ".fc1." in base
            or ".fc2." in base
            or ".adaln_proj.linear." in base
        ):
            stem = base[: -len(".weight")]
            names.update(stem + suffix for suffix in _QUANTIZED_WEIGHT_SUFFIXES)
    return names


def _top_level_family(name: str) -> str:
    return name.split(".", 1)[0]


def audit_transformer_ref(transformer_path: str | Path) -> TransformerRefStaticAudit:
    """Compare a candidate partition to the current MLX DiT loader contract.

    Only JSON config/index metadata is read.  A missing candidate is ``UNKNOWN``; an existing but
    incomplete or structurally divergent candidate is ``STATIC_INCOMPATIBLE``.  The audit is not a
    runtime or numerical-forward proof.
    """
    checkpoint_path = _resolved(Path(transformer_path))
    config_path = checkpoint_path / "config.json"
    weight_index_path = checkpoint_path / "model.safetensors.index.json"
    if not checkpoint_path.is_dir():
        return TransformerRefStaticAudit(
            checkpoint_path,
            UNKNOWN,
            _resolved(config_path),
            _resolved(weight_index_path),
            (),
            reason="transformer_ref directory is absent",
        )
    if not config_path.is_file() or not weight_index_path.is_file():
        missing = [str(path.name) for path in (config_path, weight_index_path) if not path.is_file()]
        return TransformerRefStaticAudit(
            checkpoint_path,
            STATIC_INCOMPATIBLE,
            _resolved(config_path) if config_path.exists() else None,
            _resolved(weight_index_path) if weight_index_path.exists() else None,
            (),
            reason="missing required metadata: " + ", ".join(missing),
        )

    try:
        config = _read_json(config_path, "transformer_ref config")
        index = _read_json(weight_index_path, "transformer_ref weight index")
    except ValueError as exc:
        return TransformerRefStaticAudit(
            checkpoint_path,
            STATIC_INCOMPATIBLE,
            _resolved(config_path),
            _resolved(weight_index_path),
            (),
            reason=str(exc),
        )

    config_fields = tuple(sorted(config))
    missing_required = tuple(sorted(set(_EXPECTED_CONFIG_VALUES) - set(config)))
    missing_optional = tuple(sorted(set(_OPTIONAL_DEFAULT_CONFIG_VALUES) - set(config)))
    extra_config = tuple(
        sorted(set(config) - set(_EXPECTED_CONFIG_VALUES) - set(_OPTIONAL_DEFAULT_CONFIG_VALUES) - _CONFIG_METADATA_FIELDS)
    )
    mismatched = tuple(
        sorted(
            key
            for key, expected in _EXPECTED_CONFIG_VALUES.items()
            if key in config and not _same_value(config[key], expected)
        )
    )

    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in weight_map.items()
    ):
        return TransformerRefStaticAudit(
            checkpoint_path,
            STATIC_INCOMPATIBLE,
            _resolved(config_path),
            _resolved(weight_index_path),
            (),
            config_fields=config_fields,
            missing_config_fields=tuple(sorted(set(missing_required) | set(missing_optional))),
            extra_config_fields=extra_config,
            mismatched_config_fields=mismatched,
            reason="weight index must contain a non-empty string weight_map",
        )

    indexed_names = set(weight_map)
    expected_names = set(_expected_tensor_names())
    quantized_names = _quantized_extra_names()
    allowed_names = expected_names | quantized_names
    missing_names = tuple(sorted(expected_names - indexed_names))
    extra_names = tuple(sorted(indexed_names - allowed_names))
    expected_families = {_top_level_family(name) for name in expected_names}
    actual_families = {_top_level_family(name) for name in indexed_names}
    missing_families = tuple(sorted(expected_families - actual_families))
    extra_families = tuple(sorted(actual_families - _EXPECTED_TOP_LEVEL_FAMILIES))

    shard_names = tuple(sorted(set(weight_map.values())))
    invalid_shards = tuple(
        sorted(
            name
            for name in shard_names
            if Path(name).name != name or not name.endswith(".safetensors") or not (checkpoint_path / name).is_file()
        )
    )
    shards = tuple(_resolved(checkpoint_path / name) for name in shard_names if name not in invalid_shards)

    hard_reasons: list[str] = []
    if missing_required:
        hard_reasons.append("missing required config fields: " + ", ".join(missing_required))
    if mismatched:
        hard_reasons.append("mismatched config fields: " + ", ".join(mismatched))
    if missing_names:
        hard_reasons.append("missing MLX tensor names: " + ", ".join(missing_names[:4]))
    if extra_names:
        hard_reasons.append("unexpected tensor names: " + ", ".join(extra_names[:4]))
    if invalid_shards:
        hard_reasons.append("invalid or missing shard files: " + ", ".join(invalid_shards[:4]))

    adaptation_reasons: list[str] = []
    if missing_optional:
        adaptation_reasons.append("MLX defaulted config fields: " + ", ".join(missing_optional))
    if extra_config:
        adaptation_reasons.append("unconsumed config fields: " + ", ".join(extra_config))

    if hard_reasons:
        verdict = STATIC_INCOMPATIBLE
    elif adaptation_reasons:
        verdict = STATIC_COMPATIBLE_WITH_ADAPTATION
    else:
        verdict = STATIC_COMPATIBLE
    reasons = hard_reasons + adaptation_reasons
    if not reasons:
        reasons.append("config, namespaces, and shard index match the current MLX DiT loader contract")
    return TransformerRefStaticAudit(
        checkpoint_path,
        verdict,
        _resolved(config_path),
        _resolved(weight_index_path),
        shards,
        config_fields=config_fields,
        missing_config_fields=tuple(sorted(set(missing_required) | set(missing_optional))),
        extra_config_fields=extra_config,
        mismatched_config_fields=mismatched,
        missing_tensor_names=missing_names,
        extra_tensor_names=extra_names,
        missing_tensor_families=missing_families,
        extra_tensor_families=extra_families,
        indexed_tensor_count=len(indexed_names),
        reason="; ".join(reasons),
    )


def discover_transformer_ref(checkpoint_root: str | Path) -> TransformerRefDiscovery:
    """Resolve and statically validate only ``<checkpoint_root>/transformer_ref``.

    The ordinary ``transformer`` directory is intentionally never considered a fallback.
    """
    root = _resolved(Path(checkpoint_root))
    model_index_candidate = root / "model_index.json"
    model_index_path = _resolved(model_index_candidate) if model_index_candidate.is_file() else None
    model_index_partition: str | None = None
    model_index_tasks: tuple[str, ...] = ()
    model_index_error: str | None = None
    model_index: dict[str, Any] | None = None
    if model_index_path is not None:
        try:
            model_index = _read_json(model_index_path, "checkpoint model_index.json")
        except ValueError as exc:
            model_index_error = str(exc)
        else:
            meta = model_index.get("_minimax_h3", {})
            if isinstance(meta, dict):
                raw_partition = meta.get("partition")
                model_index_partition = raw_partition if isinstance(raw_partition, str) else None
                raw_tasks = meta.get("tasks", ())
                if isinstance(raw_tasks, list) and all(isinstance(task, str) for task in raw_tasks):
                    model_index_tasks = tuple(raw_tasks)

    candidate = root / TRANSFORMER_REF_PARTITION
    candidate_resolved = _resolved(candidate)
    empty_audit = TransformerRefStaticAudit(candidate_resolved, UNKNOWN, None, None, ())
    if not candidate.exists():
        reason = f"no {TRANSFORMER_REF_PARTITION!r} directory at {candidate_resolved}"
        if model_index_error:
            reason += f"; {model_index_error}"
        return TransformerRefDiscovery(
            root,
            model_index_path,
            candidate_resolved,
            None,
            None,
            (),
            DISCOVERY_ABSENT,
            model_index_partition,
            model_index_tasks,
            reason,
            empty_audit,
        )
    if model_index_error:
        return TransformerRefDiscovery(
            root,
            model_index_path,
            candidate_resolved,
            None,
            None,
            (),
            DISCOVERY_INVALID,
            model_index_partition,
            model_index_tasks,
            model_index_error,
            empty_audit,
        )
    if model_index is not None and isinstance(model_index.get("_minimax_h3"), dict):
        meta = model_index["_minimax_h3"]
        declared_ref = "ref2va" in {task.lower() for task in model_index_tasks}
        if not declared_ref and meta.get("partition") != TRANSFORMER_REF_PARTITION:
            reason = (
                f"model_index does not declare ref2va: partition={model_index_partition!r}, "
                f"tasks={list(model_index_tasks)!r}"
            )
            return TransformerRefDiscovery(
                root,
                model_index_path,
                candidate_resolved,
                None,
                None,
                (),
                DISCOVERY_INVALID,
                model_index_partition,
                model_index_tasks,
                reason,
                empty_audit,
            )

    audit = audit_transformer_ref(candidate_resolved)
    status = DISCOVERY_PRESENT if audit.verdict == STATIC_COMPATIBLE else DISCOVERY_INVALID
    return TransformerRefDiscovery(
        root,
        model_index_path,
        candidate_resolved,
        audit.config_path,
        audit.weight_index_path,
        audit.weight_shards,
        status,
        model_index_partition,
        model_index_tasks,
        audit.reason,
        audit,
    )


def resolve_transformer_partition(
    task: str,
    checkpoint_root: str | Path,
    explicit_transformer_dir: str | Path | None = None,
) -> TransformerRoute:
    """Route an explicit task to its partition, failing closed for Ref2VA."""
    if task == TASK_REF2VA:
        discovery = discover_transformer_ref(checkpoint_root)
        if discovery.status == DISCOVERY_ABSENT:
            raise TransformerRoutingError(TRANSFORMER_REF_UNAVAILABLE, discovery.reason)
        if discovery.status != DISCOVERY_PRESENT or not discovery.audit.validated:
            raise TransformerRoutingError(
                TRANSFORMER_REF_STATIC_INCOMPATIBLE,
                f"{discovery.reason}; resolved candidate={discovery.transformer_ref_path}",
            )
        if explicit_transformer_dir is not None:
            explicit = _resolved(Path(explicit_transformer_dir))
            if explicit != discovery.transformer_ref_path:
                raise TransformerRoutingError(
                    TRANSFORMER_REF_PATH_MISMATCH,
                    f"Ref2VA requires {discovery.transformer_ref_path}, got {explicit}",
                )
        return TransformerRoute(
            task=task,
            partition=TRANSFORMER_REF_PARTITION,
            path=discovery.transformer_ref_path,
            discovery=discovery,
        )
    if task == TASK_FL2VA_T2VA:
        path = _resolved(Path(explicit_transformer_dir)) if explicit_transformer_dir else _resolved(Path(checkpoint_root) / ORDINARY_TRANSFORMER_PARTITION)
        return TransformerRoute(task, ORDINARY_TRANSFORMER_PARTITION, path)
    raise TransformerRoutingError("UNSUPPORTED_TASK", f"no transformer route for task {task!r}")


def resolve_manifest_transformer(
    manifest: Any,
    checkpoint_root: str | Path,
    explicit_transformer_dir: str | Path | None = None,
) -> TransformerRoute:
    """Route a selected LightX manifest without inferring identity from adapter topology."""
    task = getattr(manifest, "task", None)
    if not isinstance(task, str):
        raise TransformerRoutingError("MANIFEST_TASK_REQUIRED", "selected manifest has no explicit task identity")
    return resolve_transformer_partition(task, checkpoint_root, explicit_transformer_dir)

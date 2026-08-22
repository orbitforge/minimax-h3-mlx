"""MLX-free orchestration and evidence contracts for the local H3 Render Lab.

This module never constructs an H3 model.  It validates an operator request, reserves an immutable
run namespace, invokes the existing ``scripts/generate.py`` CLI in a child process, preserves raw
stdout/stderr, and extracts only metrics that the child actually emits.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .resolutions import (
    INDEPENDENT_DIMENSION_RULE_ID,
    INDEPENDENT_DIMENSION_SOURCE_ID,
    PROJECT_RESOLUTION_SOURCE_ID,
    RUNTIME_RESOLUTION_RULE_ID,
    MAX_RESOLUTION,
    MIN_RESOLUTION,
    RESOLUTION_STEP,
    ResolutionPreset,
    explicit_resolution_preset,
    preset_by_id,
    preset_payload,
    validate_preset_against_runtime,
)
from .encoder_catalog import (
    CANONICAL_ENCODER_ID,
    EncoderAssetError,
    HERETIC_ENCODER_ID,
    probe_heretic_assets,
    text_encoder_payload,
    validate_text_encoder_selection,
)
from minimax_h3_mlx.checkpoint_format import CACHE_ONLY_CONSTRUCTION, inspect_checkpoint_format
from minimax_h3_mlx.lora import LoRAError, LoRAStack
from .turbo_presets import (
    HOST_ASSET_MANIFEST_NOTE,
    HOST_ASSET_MANIFEST_STATUS,
    REFERENCE_TURBO_PRESET_ID,
    TurboPreset,
    turbo_preset_by_id,
    turbo_preset_payload,
)
from minimax_h3_mlx.conditioning_artifact import (
    ConditioningArtifactError,
    load_conditioning_artifact,
    validate_conditioning_artifact,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate.py"
HERETIC_ENCODER_SCRIPT = REPO_ROOT / "tools" / "render_lab" / "heretic_encoder.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out" / "render-lab"
DEFAULT_OUTPUT_NAME = "render.mp4"
CANONICAL_TRANSFORMER_NAME = "minimax-h3-mlx-6bit-streamed-adaln"
CANONICAL_TRANSFORMER_MODE = "streamed-adaln-q6"
CURRENT_MODEL_ID = "current"
BETA_MODEL_ID = "beta-0.6"
DEFAULT_MODEL_ID = BETA_MODEL_ID
BETA_TRANSFORMER_NAME = "minimax-h3-mlx-beta-0.6-q6-q8-corrected-slice-025-streamed-adaln"
MODEL_CHOICES = (
    (CURRENT_MODEL_ID, "Current", CANONICAL_TRANSFORMER_NAME),
    (BETA_MODEL_ID, "Beta 0.6", BETA_TRANSFORMER_NAME),
)
FORBIDDEN_TRANSFORMER_NAME = "minimax-h3-mlx-6bit"

T2V = "T2V"
I2V = "I2V"
FIRST_LAST = "FIRST_LAST"
SINGLE_RENDER_WORKFLOW = "SINGLE_RENDER"
FL2V_STORYBOARD_WORKFLOW = "FL2V_STORYBOARD"
FL2V_STORYBOARD_SEGMENT_WORKFLOW = "FL2V_STORYBOARD_SEGMENT"
MODE_LABELS = {
    T2V: "T2V — prompt only",
    I2V: "I2V — one image, first-frame anchor",
    FIRST_LAST: "First + last frame — two images",
}
WORKFLOW_LABELS = {
    SINGLE_RENDER_WORKFLOW: "Single render",
    FL2V_STORYBOARD_WORKFLOW: "FL2V storyboard",
}

MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 15.0
MIN_INFERENCE_STEPS = 2
MAX_INFERENCE_STEPS = 40
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_TURBO_PRESET_ID = "lightx-4step-v01"
_DEFAULT_TURBO_PRESET = turbo_preset_by_id(DEFAULT_TURBO_PRESET_ID)
if _DEFAULT_TURBO_PRESET is None:
    raise RuntimeError(f"Render Lab default Turbo preset is unavailable: {DEFAULT_TURBO_PRESET_ID}")
DEFAULT_TURBO_PRESET_NFE = _DEFAULT_TURBO_PRESET.nfe
DEFAULT_INFERENCE_STEPS = DEFAULT_TURBO_PRESET_NFE
DEFAULT_SEED = 0
DEFAULT_RESOLUTION_ID = "canonical-128-square-v05d"

# Display-only geometry derived from the current H3 VAE/DiT source contract.  Resolution
# admission itself is delegated to ``resolve_canvas_size`` through resolutions.py.
H3_SPATIAL_COMPRESSION_RATIO = 16
H3_DIT_PATCH_SIZE = (1, 2, 2)

RUN_SCHEMA_VERSION = 4
BENCHMARK_SCHEMA_VERSION = 1
STATUS_SCHEMA_VERSION = 1
RUN_ID_PATTERN = re.compile(r"^run-[0-9]{8}T[0-9]{6}Z-[0-9a-f]{10}$")
SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.mp4$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
MAX_LOG_TAIL_BYTES = 60_000


class RenderValidationError(ValueError):
    """An operator request failed before H3 was launched."""


class RenderBusyError(RuntimeError):
    """Another Render Lab process owns the output-root admission lock."""


@dataclass(frozen=True)
class AdditionalLoRA:
    """One ordered, non-scheduling LoRA row submitted by Render Lab."""

    path: str | Path | None
    scale: object = 1.0


@dataclass(frozen=True)
class ConditioningArtifactEvidence:
    """Immutable replay identity retained after Render Lab admission."""

    path: Path
    artifact_identity: str
    token_count: int
    conditioning_shape: tuple[int, ...]
    tensor_checksum: str


def parse_additional_loras_payload(value: object) -> tuple[AdditionalLoRA, ...]:
    """Decode the browser/API collection without applying runtime admission policy."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RenderValidationError("Additional LoRAs must be a JSON array") from exc
    if not isinstance(value, (list, tuple)):
        raise RenderValidationError("Additional LoRAs must be a JSON array")

    entries: list[AdditionalLoRA] = []
    for index, item in enumerate(value):
        if isinstance(item, AdditionalLoRA):
            entries.append(item)
            continue
        if not isinstance(item, Mapping):
            raise RenderValidationError(f"Additional LoRA row {index + 1} must be an object")
        path = item.get("path")
        if path is not None and not isinstance(path, (str, Path)):
            raise RenderValidationError(f"Additional LoRA row {index + 1} path must be a string")
        entries.append(AdditionalLoRA(path=path, scale=item.get("scale", 1.0)))
    return tuple(entries)


def parse_storyboard_card_paths(value: object) -> tuple[str | Path, ...]:
    """Decode ordered storyboard card paths without applying filesystem admission policy."""
    if value is None:
        return ()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return ()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RenderValidationError("Storyboard cards must be a JSON array") from exc
    if not isinstance(value, (list, tuple)):
        raise RenderValidationError("Storyboard cards must be a JSON array")
    cards: list[str | Path] = []
    for index, item in enumerate(value):
        if not isinstance(item, (str, Path)):
            raise RenderValidationError(f"Storyboard card {index + 1} path must be a string")
        cards.append(item)
    return tuple(cards)


@dataclass(frozen=True)
class RenderRequest:
    mode: str
    prompt: str
    resolution_id: str | None = DEFAULT_RESOLUTION_ID
    steps: int = DEFAULT_INFERENCE_STEPS
    duration_seconds: float = DEFAULT_DURATION_SECONDS
    seed: int = DEFAULT_SEED
    output_root: str | Path = DEFAULT_OUTPUT_ROOT
    output_name: str = DEFAULT_OUTPUT_NAME
    image_paths: tuple[str | Path, ...] = ()
    checkpoint_root: str | Path = field(default_factory=lambda: default_checkpoint_root())
    transformer_path: str | Path | None = None
    width: int | str | None = None
    height: int | str | None = None
    lora_enabled: bool = False
    lora_path: str | Path | None = None
    lora_scale: float = 1.0
    additional_loras: tuple[AdditionalLoRA, ...] = ()
    turbo_enabled: bool = False
    turbo_steps: int | str | None = None
    turbo_preset_id: str | None = DEFAULT_TURBO_PRESET_ID
    text_encoder_id: str = CANONICAL_ENCODER_ID
    conditioning_artifact_path: str | Path | None = None
    workflow: str = SINGLE_RENDER_WORKFLOW
    storyboard_card_paths: tuple[str | Path, ...] = ()
    model_id: str | None = DEFAULT_MODEL_ID

    def normalized(self) -> "RenderRequest":
        return replace(
            self,
            mode=str(self.mode).upper(),
            prompt=str(self.prompt),
            workflow=(str(self.workflow or SINGLE_RENDER_WORKFLOW).strip().upper()),
            model_id=(
                DEFAULT_MODEL_ID
                if self.model_id is None or not str(self.model_id).strip()
                else str(self.model_id).strip()
            ),
            resolution_id=(
                None
                if self.resolution_id is None or not str(self.resolution_id).strip()
                else str(self.resolution_id)
            ),
            output_root=Path(self.output_root).expanduser(),
            output_name=str(self.output_name),
            image_paths=tuple(Path(value).expanduser() for value in self.image_paths),
            checkpoint_root=Path(self.checkpoint_root).expanduser(),
            transformer_path=(
                None if self.transformer_path is None else Path(self.transformer_path).expanduser()
            ),
            lora_path=(
                None
                if self.lora_path is None or not str(self.lora_path).strip()
                else Path(self.lora_path).expanduser()
            ),
            additional_loras=tuple(
                replace(
                    entry,
                    path=(
                        None
                        if entry.path is None or not str(entry.path).strip()
                        else Path(entry.path).expanduser()
                    ),
                )
                for entry in parse_additional_loras_payload(self.additional_loras)
            ),
            turbo_preset_id=(
                None
                if self.turbo_preset_id is None or not str(self.turbo_preset_id).strip()
                else str(self.turbo_preset_id)
            ),
            text_encoder_id=(
                CANONICAL_ENCODER_ID
                if self.text_encoder_id is None or not str(self.text_encoder_id).strip()
                else str(self.text_encoder_id).strip()
            ),
            conditioning_artifact_path=(
                None
                if self.conditioning_artifact_path is None or not str(self.conditioning_artifact_path).strip()
                else Path(str(self.conditioning_artifact_path).strip()).expanduser()
            ),
            storyboard_card_paths=tuple(
                Path(value).expanduser() if isinstance(value, (str, Path)) else value
                for value in parse_storyboard_card_paths(self.storyboard_card_paths)
            ),
        )


@dataclass(frozen=True)
class ValidatedRequest:
    request: RenderRequest
    preset: ResolutionPreset
    height: int
    width: int
    anchors: tuple[str, ...]
    output_root: Path
    image_paths: tuple[Path, ...]
    checkpoint_root: Path
    transformer_path: Path | None
    turbo_preset: TurboPreset | None
    scheduling_adapter_path: Path | None
    additional_loras: tuple[AdditionalLoRA, ...]
    heretic_assets: Any | None = None
    conditioning_artifact_path: Path | None = None
    conditioning_artifact_evidence: ConditioningArtifactEvidence | None = None


@dataclass(frozen=True)
class ValidatedStoryboardRequest:
    """Validated global settings plus the ordered cards for a phase-one storyboard."""

    request: RenderRequest
    card_paths: tuple[Path, ...]
    shared: ValidatedRequest | None = None
    card_source_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class StoryboardSegmentJob:
    """One sequential first/last-frame child job derived from adjacent cards."""

    segment_index: int
    start_card_index: int
    end_card_index: int
    start_path: Path
    end_path: Path
    request: RenderRequest


@dataclass(frozen=True)
class RunNamespace:
    run_id: str
    run_dir: Path
    output_path: Path
    created_at: str
    artifact_prefix: str | None = None

    def _artifact_path(self, prefixed_name: str, ordinary_name: str) -> Path:
        return self.run_dir / (
            f"{self.artifact_prefix}.{prefixed_name}" if self.artifact_prefix else ordinary_name
        )

    @property
    def config_path(self) -> Path:
        return self._artifact_path("config.json", "render-config.json")

    @property
    def benchmark_path(self) -> Path:
        return self._artifact_path("benchmark.json", "benchmark.json")

    @property
    def status_path(self) -> Path:
        return self._artifact_path("status.json", "run-status.json")

    @property
    def stdout_path(self) -> Path:
        return self._artifact_path("stdout.log", "stdout.log")

    @property
    def stderr_path(self) -> Path:
        return self._artifact_path("stderr.log", "stderr.log")

    @property
    def telemetry_dir(self) -> Path:
        return self._artifact_path("telemetry", "telemetry")

    @property
    def conditioning_artifact_path(self) -> Path:
        return self._artifact_path("conditioning-artifact.npz", "conditioning-artifact.npz")

    @property
    def encoder_evidence_path(self) -> Path:
        return self._artifact_path("heretic-encoder-evidence.json", "heretic-encoder-evidence.json")

    @property
    def encoder_release_path(self) -> Path:
        return self._artifact_path("heretic-release-evidence.json", "heretic-release-evidence.json")


@dataclass(frozen=True)
class UploadedImage:
    filename: str
    data: bytes


@dataclass(frozen=True)
class RunResult:
    namespace: RunNamespace
    exit_code: int | None
    success: bool
    output_artifact: Path | None
    benchmark: dict[str, Any]


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def iso_timestamp(value: _dt.datetime | None = None) -> str:
    value = value or _utc_now()
    return value.astimezone(_dt.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _first_existing(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_checkpoint_root(repo_root: Path = REPO_ROOT) -> Path:
    env_value = os.environ.get("H3_CHECKPOINT_ROOT")
    local_checkpoint = repo_root.parent / "checkpoints" / "minimax-h3-fl2va"
    candidates = [Path(env_value).expanduser()] if env_value else []
    candidates.extend([local_checkpoint, repo_root.parent / "models" / "minimax-h3-fl2va"])
    return _first_existing(candidates) or candidates[0]


def _normalize_model_id(value: object) -> str:
    normalized = str(value).strip() if value is not None else DEFAULT_MODEL_ID
    if not normalized:
        normalized = DEFAULT_MODEL_ID
    if normalized not in {item[0] for item in MODEL_CHOICES}:
        choices = ", ".join(item[0] for item in MODEL_CHOICES)
        raise RenderValidationError(f"Unknown Render Lab model {value!r}; choose one of: {choices}")
    return normalized


def model_transformer_path(model_id: object, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve one logical Render Lab model to its exact admitted streamed transformer path."""
    normalized = _normalize_model_id(model_id)
    transformer_name = next(name for identifier, _label, name in MODEL_CHOICES if identifier == normalized)
    return (Path(repo_root).expanduser().resolve().parent / "models" / transformer_name).resolve(strict=False)


def model_label(model_id: object) -> str:
    normalized = _normalize_model_id(model_id)
    return next(label for identifier, label, _name in MODEL_CHOICES if identifier == normalized)


def streamed_transformer_asset_available(path: str | Path) -> tuple[bool, str | None]:
    """Check one streamed-AdaLN asset through the shared metadata-only checkpoint contract."""
    candidate = Path(path).expanduser().resolve(strict=False)
    if not candidate.is_dir():
        return False, "the streamed transformer directory is missing"
    try:
        format_info = inspect_checkpoint_format(candidate)
    except (OSError, TypeError, ValueError) as exc:
        return False, f"the streamed-AdaLN checkpoint metadata is invalid: {exc}"
    if format_info.checkpoint_format != "derived" or format_info.construction_mode != CACHE_ONLY_CONSTRUCTION:
        return False, "the directory does not satisfy the complete derived streamed-AdaLN format contract"
    if not (candidate / "config.json").is_file():
        return False, "the derived streamed-AdaLN config.json is missing"
    if not (candidate / "quant_config.json").is_file():
        return False, "the derived streamed-AdaLN quant_config.json is missing"
    return True, None


def model_selection_payload(repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    """Serialize only the bounded logical choices; filesystem paths stay server-side."""
    payload = []
    for identifier, label, _transformer_name in MODEL_CHOICES:
        transformer = model_transformer_path(identifier, repo_root)
        available, reason = streamed_transformer_asset_available(transformer)
        payload.append({
            "id": identifier,
            "label": label,
            "available": available,
            "transformer_mode": CANONICAL_TRANSFORMER_MODE,
            "disabled_reason": None if available else reason,
        })
    return payload


def default_transformer_path(repo_root: Path = REPO_ROOT) -> Path:
    return model_transformer_path(DEFAULT_MODEL_ID, repo_root)


def _resolve_model_selection(request: RenderRequest, repo_root: Path) -> tuple[str, Path]:
    model_id = _normalize_model_id(request.model_id)
    selected = model_transformer_path(model_id, repo_root)
    if request.transformer_path is not None:
        supplied = Path(request.transformer_path).expanduser().resolve(strict=False)
        if supplied != selected:
            raise RenderValidationError(
                f"Model {model_id!r} owns transformer {selected.name}; arbitrary transformer overrides are rejected"
            )
    return model_id, selected


def expected_image_count(mode: str) -> int:
    normalized = str(mode).upper()
    if normalized == T2V:
        return 0
    if normalized == I2V:
        return 1
    if normalized == FIRST_LAST:
        return 2
    raise RenderValidationError(f"Unknown generation mode {mode!r}")


def anchors_for_mode(mode: str, image_count: int) -> tuple[str, ...]:
    expected = expected_image_count(mode)
    if image_count != expected:
        if expected == 0:
            raise RenderValidationError("T2V rejects image inputs")
        raise RenderValidationError(
            f"{mode} requires exactly {expected} readable image(s); received {image_count}"
        )
    if str(mode).upper() == T2V:
        return ()
    if str(mode).upper() == I2V:
        return ("first",)
    return ("first", "last")


def _coerce_bool(value: object, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "on", "yes"}:
            return True
        if normalized in {"0", "false", "off", "no", ""}:
            return False
    raise RenderValidationError(f"{label} must be a boolean")


def _coerce_integer(value: object, label: str, *, allow_none: bool = True) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise RenderValidationError(f"{label} must be an integer")
    if allow_none and isinstance(value, str) and not value.strip():
        return None
    if isinstance(value, bool):
        raise RenderValidationError(f"{label} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value.strip())
    raise RenderValidationError(f"{label} must be an integer")


def _validate_dimension(value: object, label: str) -> int:
    dimension = _coerce_integer(value, label, allow_none=False)
    assert dimension is not None
    if dimension <= 0:
        raise RenderValidationError(f"{label} must be positive; got {dimension}")
    if not MIN_RESOLUTION <= dimension <= MAX_RESOLUTION:
        raise RenderValidationError(
            f"{label} must be between {MIN_RESOLUTION} and {MAX_RESOLUTION} pixels; got {dimension}"
        )
    if dimension % RESOLUTION_STEP:
        raise RenderValidationError(
            f"{label} must be divisible by {RESOLUTION_STEP} pixels; got {dimension}"
        )
    return dimension


def _validate_optional_dimensions(request: RenderRequest) -> tuple[int | None, int | None]:
    width = _coerce_integer(request.width, "Width")
    height = _coerce_integer(request.height, "Height")
    if (width is None) != (height is None):
        raise RenderValidationError("Width and height must be provided together")
    if width is None or height is None:
        return None, None
    return _validate_dimension(width, "Width"), _validate_dimension(height, "Height")


def _validate_lora_controls(
    request: RenderRequest,
    *,
    check_runtime_paths: bool,
    turbo_preset_selected: bool,
) -> tuple[bool, Path | None, float, bool, int | None]:
    lora_enabled = _coerce_bool(request.lora_enabled, "LoRA enabled")
    turbo_enabled = _coerce_bool(request.turbo_enabled, "Turbo enabled")
    if not lora_enabled:
        if turbo_enabled and not turbo_preset_selected:
            raise RenderValidationError("Turbo mode requires LoRA to be enabled with an adapter path")
        return False, None, 1.0, turbo_enabled, None

    if request.lora_path is None or not str(request.lora_path).strip():
        raise RenderValidationError("LoRA is enabled but no adapter path was provided")
    try:
        lora_path = Path(request.lora_path).expanduser().resolve(strict=False)
    except (OSError, ValueError) as exc:
        raise RenderValidationError("LoRA adapter path is not a valid filesystem path") from exc
    if check_runtime_paths and (not lora_path.is_file() or not os.access(lora_path, os.R_OK)):
        raise RenderValidationError(f"LoRA adapter path is not a readable file: {lora_path}")

    if isinstance(request.lora_scale, bool):
        raise RenderValidationError("LoRA scale must be a finite nonnegative number")
    try:
        lora_scale = float(request.lora_scale)
    except (TypeError, ValueError) as exc:
        raise RenderValidationError("LoRA scale must be a finite nonnegative number") from exc
    if not math.isfinite(lora_scale) or lora_scale < 0:
        raise RenderValidationError("LoRA scale must be a finite nonnegative number")

    turbo_steps = _coerce_integer(request.turbo_steps, "Turbo steps") if turbo_enabled else None
    if turbo_enabled and not turbo_preset_selected:
        if turbo_steps is not None and not MIN_INFERENCE_STEPS <= turbo_steps <= MAX_INFERENCE_STEPS:
            raise RenderValidationError(
                f"Turbo steps must be between {MIN_INFERENCE_STEPS} and {MAX_INFERENCE_STEPS}"
            )
        if turbo_steps is not None and turbo_steps != request.steps:
            raise RenderValidationError(
                "Turbo mode requires ordinary inference steps and Turbo steps to agree"
            )
    return True, lora_path, lora_scale, turbo_enabled, turbo_steps


def _resolve_lora_path(value: object, label: str, *, check_runtime_paths: bool) -> Path:
    if value is None or not str(value).strip():
        raise RenderValidationError(f"{label} is enabled but no adapter path was provided")
    try:
        resolved = Path(value).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise RenderValidationError(f"{label} path is not a valid filesystem path") from exc
    if check_runtime_paths and (not resolved.is_file() or not os.access(resolved, os.R_OK)):
        raise RenderValidationError(f"{label} path is not a readable file: {resolved}")
    return resolved


def _coerce_lora_scale(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise RenderValidationError(f"{label} must be a finite nonnegative number")
    try:
        scale = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RenderValidationError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(scale) or scale < 0:
        raise RenderValidationError(f"{label} must be a finite nonnegative number")
    return scale


def _validate_additional_loras(
    entries: Sequence[AdditionalLoRA],
    *,
    scheduling_path: Path | None,
    check_runtime_paths: bool,
) -> tuple[AdditionalLoRA, ...]:
    validated: list[AdditionalLoRA] = []
    for index, entry in enumerate(entries):
        path = _resolve_lora_path(
            entry.path,
            f"Additional LoRA row {index + 1}",
            check_runtime_paths=check_runtime_paths,
        )
        scale = _coerce_lora_scale(entry.scale, f"Additional LoRA row {index + 1} scale")
        validated.append(AdditionalLoRA(path=path, scale=scale))

    try:
        LoRAStack.from_sources(
            scheduling_path=scheduling_path,
            auxiliary_paths=tuple(entry.path for entry in validated),
            auxiliary_scales=tuple(float(entry.scale) for entry in validated),
        )
    except (LoRAError, ValueError, TypeError) as exc:
        raise RenderValidationError(str(exc)) from exc
    return tuple(validated)


def _validate_turbo_preset(
    request: RenderRequest,
    *,
    repo_root: Path,
    check_runtime_paths: bool,
) -> tuple[TurboPreset | None, Path | None]:
    try:
        turbo_preset = turbo_preset_by_id(request.turbo_preset_id)
    except ValueError as exc:
        raise RenderValidationError(str(exc)) from exc
    if turbo_preset is None:
        return None, None

    if request.steps != turbo_preset.nfe:
        raise RenderValidationError(
            f"Turbo preset {turbo_preset.label!r} owns {turbo_preset.nfe} NFE; "
            f"manual inference steps were {request.steps}"
        )
    requested_turbo_steps = _coerce_integer(request.turbo_steps, "Turbo steps")
    if requested_turbo_steps is not None and requested_turbo_steps != turbo_preset.nfe:
        raise RenderValidationError(
            f"Turbo preset {turbo_preset.label!r} owns {turbo_preset.nfe} NFE; "
            f"manual Turbo steps were {requested_turbo_steps}"
        )
    if isinstance(request.lora_scale, bool):
        raise RenderValidationError("Turbo preset scale is fixed at its validated runtime default")
    try:
        requested_scale = float(request.lora_scale)
    except (TypeError, ValueError) as exc:
        raise RenderValidationError("Turbo preset scale is fixed at its validated runtime default") from exc
    if not math.isfinite(requested_scale) or not math.isclose(
        requested_scale,
        turbo_preset.default_scale,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RenderValidationError(
            f"Turbo preset {turbo_preset.label!r} fixes LoRA scale at "
            f"{turbo_preset.default_scale:g}"
        )

    asset_path = turbo_preset.resolve_asset_path(repo_root)
    if check_runtime_paths and (not asset_path.is_file() or not os.access(asset_path, os.R_OK)):
        raise RenderValidationError(
            f"Turbo preset asset is not a readable file: {asset_path} "
            f"(logical asset {turbo_preset.logical_asset})"
        )
    return turbo_preset, asset_path


def _transformer_mode(transformer_path: Path | None) -> str:
    if transformer_path is not None and transformer_path.name in {
        CANONICAL_TRANSFORMER_NAME,
        BETA_TRANSFORMER_NAME,
    }:
        return CANONICAL_TRANSFORMER_MODE
    return "unspecified-or-custom"


def _validate_transformer_safety(
    transformer_path: Path | None,
    *,
    repo_root: Path = REPO_ROOT,
    model_id: str | None = None,
    check_runtime_paths: bool,
) -> None:
    if transformer_path is None:
        raise RenderValidationError("Render Lab requires a selected streamed transformer")
    resolved = transformer_path.expanduser().resolve(strict=False)
    if Path("/Volumes/models") == resolved or Path("/Volumes/models") in resolved.parents:
        raise RenderValidationError(
            "Render Lab rejects stale /Volumes/models transformer paths; configure the local streamed-AdaLN Q6 asset"
        )
    if transformer_path.name == FORBIDDEN_TRANSFORMER_NAME:
        raise RenderValidationError(
            f"Render Lab rejects the non-streamed {FORBIDDEN_TRANSFORMER_NAME} transformer; "
            f"use one of the admitted streamed choices"
        )
    admitted_paths = {
        model_transformer_path(identifier, repo_root)
        for identifier, _label, _transformer_name in MODEL_CHOICES
    }
    if resolved not in admitted_paths:
        raise RenderValidationError(
            "Render Lab admits only the Current or Beta 0.6 streamed transformers; "
            f"got {transformer_path.name}"
        )
    if model_id is not None and resolved != model_transformer_path(model_id, repo_root):
        raise RenderValidationError("Selected model and resolved transformer do not agree")
    if check_runtime_paths:
        available, reason = streamed_transformer_asset_available(resolved)
        if not available:
            selection_label = model_id if model_id is not None else "selection"
            raise RenderValidationError(
                f"Model {selection_label!r} is unavailable at {resolved}: {reason}"
            )


def _validate_number_fields(request: RenderRequest) -> None:
    if not request.prompt.strip():
        raise RenderValidationError("Prompt must not be empty")
    if isinstance(request.steps, bool) or not isinstance(request.steps, int):
        raise RenderValidationError("Inference steps must be an integer")
    if not MIN_INFERENCE_STEPS <= request.steps <= MAX_INFERENCE_STEPS:
        raise RenderValidationError(
            f"Inference steps must be between {MIN_INFERENCE_STEPS} and {MAX_INFERENCE_STEPS}"
        )
    if isinstance(request.duration_seconds, bool):
        raise RenderValidationError("Duration must be a number")
    try:
        duration = float(request.duration_seconds)
    except (TypeError, ValueError) as exc:
        raise RenderValidationError("Duration must be a number") from exc
    if not math.isfinite(duration) or not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise RenderValidationError(
            f"Duration must be between {MIN_DURATION_SECONDS:g} and {MAX_DURATION_SECONDS:g} seconds"
        )
    if isinstance(request.seed, bool) or not isinstance(request.seed, int):
        raise RenderValidationError("Seed must be an integer")


def resolve_output_root(value: str | Path, repo_root: Path = REPO_ROOT) -> Path:
    raw = str(value).strip()
    if not raw or "\x00" in raw:
        raise RenderValidationError("Output root must be a non-empty filesystem path")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError as exc:
        raise RenderValidationError(f"Could not resolve output root {raw!r}") from exc
    broad_namespaces = {
        Path(resolved.anchor),
        Path.home().resolve(),
        Path("/tmp").resolve(),
        Path("/private/tmp").resolve(),
        Path("/var").resolve(),
        Path("/private/var").resolve(),
        repo_root.resolve(),
    }
    if resolved in broad_namespaces:
        raise RenderValidationError("Output root is an unsafe broad namespace")
    # Resolve the complete path before applying the broad-namespace checks. macOS commonly exposes
    # safe system aliases such as /var -> /private/var; rejecting every ancestor symlink would make
    # ordinary temporary-directory validation fail. The resolved target is the namespace we admit.
    if candidate.exists() and candidate.is_symlink():
        raise RenderValidationError("Output root may not itself be a symlink")
    for ancestor in (resolved, *resolved.parents):
        if ancestor.name.startswith("run-") and (ancestor / "render-config.json").is_file():
            raise RenderValidationError("Output root points inside an existing immutable run namespace")
    if resolved.exists() and not resolved.is_dir():
        raise RenderValidationError("Output root exists but is not a directory")
    return resolved


def validate_output_name(value: str) -> str:
    name = str(value).strip()
    if not SAFE_OUTPUT_NAME.fullmatch(name) or Path(name).name != name:
        raise RenderValidationError("Output name must be a simple .mp4 filename without path components")
    return name


def validate_image_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RenderValidationError(f"Image is not readable: {value}") from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise RenderValidationError(f"Image is not a readable file: {value}")
    try:
        from PIL import Image

        with Image.open(resolved) as image:
            image.verify()
    except Exception as exc:
        raise RenderValidationError(f"Image is unreadable or not a supported image: {value}") from exc
    return resolved


def _validate_conditioning_artifact(
    request: RenderRequest,
    checkpoint_root: Path,
) -> tuple[Path | None, ConditioningArtifactEvidence | None]:
    """Admit one external canonical artifact without constructing Qwen or MLX."""
    raw_path = request.conditioning_artifact_path
    if raw_path is None:
        return None, None
    try:
        resolved = Path(raw_path).expanduser().resolve(strict=False)
    except (OSError, TypeError, ValueError) as exc:
        raise RenderValidationError("Conditioning artifact path is not a valid filesystem path") from exc
    if not resolved.is_file() or not os.access(resolved, os.R_OK):
        raise RenderValidationError(f"Conditioning artifact is not a readable file: {resolved}")

    try:
        artifact = load_conditioning_artifact(resolved)
        encoder = artifact.metadata.get("encoder", {})
        if (
            not isinstance(encoder, Mapping)
            or encoder.get("family") != "qwen3_vl"
            or encoder.get("encoder_id") is not None
            or encoder.get("experimental") is True
        ):
            raise ConditioningArtifactError(
                "external conditioning replay requires Canonical Qwen3-VL artifact provenance"
            )
        validate_conditioning_artifact(
            artifact,
            checkpoint_root=checkpoint_root,
            prompt=request.prompt,
        )
    except (ConditioningArtifactError, OSError, TypeError, ValueError) as exc:
        raise RenderValidationError(str(exc)) from exc

    return resolved, ConditioningArtifactEvidence(
        path=resolved,
        artifact_identity=artifact.artifact_identity,
        token_count=artifact.token_count,
        conditioning_shape=artifact.conditioning_shape,
        tensor_checksum=artifact.tensor_checksum,
    )


def validate_render_request(
    request: RenderRequest,
    *,
    repo_root: Path = REPO_ROOT,
    check_runtime_paths: bool = False,
    check_images: bool = True,
    verify_runtime_geometry: bool = True,
) -> ValidatedRequest:
    request = request.normalized()
    if request.workflow != SINGLE_RENDER_WORKFLOW:
        raise RenderValidationError(
            "Storyboard requests must use validate_storyboard_request before child admission"
        )
    if request.storyboard_card_paths:
        raise RenderValidationError("Single renders do not accept storyboard card paths")
    if request.conditioning_artifact_path is not None:
        if request.mode != T2V:
            raise RenderValidationError("External conditioning artifact replay is currently T2V-only")
        if request.text_encoder_id != CANONICAL_ENCODER_ID:
            raise RenderValidationError(
                "External conditioning artifact replay requires Canonical Qwen3-VL"
            )
    try:
        heretic_assets = validate_text_encoder_selection(
            request.text_encoder_id,
            request.mode,
            repo_root=repo_root,
            check_runtime_paths=check_runtime_paths,
        )
    except EncoderAssetError as exc:
        raise RenderValidationError(str(exc)) from exc
    _validate_number_fields(request)
    width, height = _validate_optional_dimensions(request)
    turbo_preset, turbo_asset_path = _validate_turbo_preset(
        request,
        repo_root=repo_root,
        check_runtime_paths=check_runtime_paths,
    )
    lora_enabled, lora_path, lora_scale, legacy_turbo_enabled, legacy_turbo_steps = _validate_lora_controls(
        request,
        check_runtime_paths=check_runtime_paths,
        turbo_preset_selected=turbo_preset is not None,
    )
    legacy_lora_is_scheduling = turbo_preset is None and legacy_turbo_enabled
    scheduling_adapter_path = turbo_asset_path if turbo_preset is not None else (
        lora_path if legacy_lora_is_scheduling else None
    )
    additional_inputs = list(request.additional_loras)
    if lora_enabled and not legacy_lora_is_scheduling:
        # The old single ordinary-LoRA fields remain accepted as a compatibility bridge.  A
        # selected Turbo preset turns that bridge into an auxiliary entry; the preset path above
        # remains the sole scheduling owner.
        additional_inputs.insert(0, AdditionalLoRA(path=lora_path, scale=lora_scale))
    additional_loras = _validate_additional_loras(
        additional_inputs,
        scheduling_path=scheduling_adapter_path,
        check_runtime_paths=check_runtime_paths,
    )
    turbo_enabled = turbo_preset is not None or legacy_turbo_enabled
    turbo_steps = turbo_preset.nfe if turbo_preset is not None else legacy_turbo_steps
    image_paths = tuple(Path(value) for value in request.image_paths)
    anchors = anchors_for_mode(request.mode, len(image_paths))
    explicit_dimensions = width is not None and height is not None
    if explicit_dimensions:
        preset = explicit_resolution_preset(width, height)
        runtime_height, runtime_width = height, width
    else:
        if request.resolution_id is None:
            raise RenderValidationError("Choose a resolution preset or provide both width and height")
        try:
            preset = (
                validate_preset_against_runtime(request.resolution_id)
                if verify_runtime_geometry
                else preset_by_id(request.resolution_id)
            )
        except RenderValidationError:
            raise
        except Exception as exc:
            raise RenderValidationError(str(exc)) from exc
        runtime_height, runtime_width = preset.runtime_dimensions
    output_root = resolve_output_root(request.output_root, repo_root)
    output_name = validate_output_name(request.output_name)
    if check_images:
        image_paths = tuple(validate_image_path(value) for value in image_paths)
    checkpoint_root = Path(request.checkpoint_root).expanduser().resolve(strict=False)
    model_id, transformer_path = _resolve_model_selection(request, repo_root)
    _validate_transformer_safety(
        transformer_path,
        repo_root=repo_root,
        model_id=model_id,
        check_runtime_paths=check_runtime_paths,
    )
    if check_runtime_paths:
        if not checkpoint_root.is_dir() or not (checkpoint_root / "model_index.json").is_file():
            raise RenderValidationError(
                f"Checkpoint root is not a usable H3 release directory: {checkpoint_root}"
            )
        effective_transformer = transformer_path
        assert effective_transformer is not None
        transformer_path = effective_transformer
    conditioning_artifact_path, conditioning_artifact_evidence = _validate_conditioning_artifact(
        request,
        checkpoint_root,
    )
    normalized_request = replace(
        request,
        model_id=model_id,
        resolution_id=preset.preset_id if explicit_dimensions else request.resolution_id,
        width=runtime_width,
        height=runtime_height,
        lora_enabled=lora_enabled,
        lora_path=lora_path,
        lora_scale=lora_scale,
        additional_loras=additional_loras,
        turbo_enabled=turbo_enabled,
        turbo_steps=turbo_steps,
        turbo_preset_id=turbo_preset.preset_id if turbo_preset is not None else None,
        conditioning_artifact_path=conditioning_artifact_path,
    )
    return ValidatedRequest(
        request=replace(normalized_request, output_name=output_name),
        preset=preset,
        height=runtime_height,
        width=runtime_width,
        anchors=anchors,
        output_root=output_root,
        image_paths=image_paths,
        checkpoint_root=checkpoint_root,
        transformer_path=transformer_path,
        turbo_preset=turbo_preset,
        scheduling_adapter_path=scheduling_adapter_path,
        additional_loras=additional_loras,
        heretic_assets=heretic_assets,
        conditioning_artifact_path=conditioning_artifact_path,
        conditioning_artifact_evidence=conditioning_artifact_evidence,
    )


def _normalize_storyboard_card_paths(value: object) -> tuple[Path, ...]:
    cards: list[Path] = []
    raw_cards = parse_storyboard_card_paths(value)
    if len(raw_cards) < 2:
        raise RenderValidationError("FL2V storyboard requires at least 2 cards")
    for index, raw in enumerate(raw_cards, start=1):
        if not str(raw).strip() or "\x00" in str(raw):
            raise RenderValidationError(f"Storyboard card {index} path must be non-empty")
        try:
            cards.append(Path(raw).expanduser().resolve(strict=False))
        except (OSError, TypeError, ValueError) as exc:
            raise RenderValidationError(f"Storyboard card {index} path is not valid") from exc
    return tuple(cards)


def validate_storyboard_request(
    request: RenderRequest,
    card_paths: object | None = None,
    *,
    card_source_names: Sequence[str] | None = None,
    repo_root: Path = REPO_ROOT,
    check_runtime_paths: bool = False,
    check_images: bool = True,
    verify_runtime_geometry: bool = True,
) -> ValidatedStoryboardRequest:
    """Validate a storyboard before any segment child is launched."""
    normalized = request.normalized()
    if normalized.workflow != FL2V_STORYBOARD_WORKFLOW:
        raise RenderValidationError(
            f"Storyboard validation requires workflow {FL2V_STORYBOARD_WORKFLOW!r}"
        )
    if normalized.mode != FIRST_LAST:
        raise RenderValidationError("FL2V storyboard segments require FIRST_LAST mode")
    if normalized.image_paths:
        raise RenderValidationError("Storyboard cards must be supplied as an ordered card list")
    cards = _normalize_storyboard_card_paths(
        normalized.storyboard_card_paths if card_paths is None else card_paths
    )
    source_names = tuple(str(name).strip() for name in (card_source_names or ()))
    if source_names and len(source_names) != len(cards):
        raise RenderValidationError("Storyboard card source-name count must match card count")
    if any(not name for name in source_names):
        raise RenderValidationError("Storyboard card source names must be non-empty")
    if check_images:
        cards = tuple(validate_image_path(path) for path in cards)

    storyboard_request = replace(
        normalized,
        image_paths=(),
        storyboard_card_paths=cards,
    )
    shared_request = replace(
        storyboard_request,
        workflow=SINGLE_RENDER_WORKFLOW,
        image_paths=(cards[0], cards[1]),
        storyboard_card_paths=(),
    )
    shared = validate_render_request(
        shared_request,
        repo_root=repo_root,
        check_runtime_paths=check_runtime_paths,
        check_images=check_images,
        verify_runtime_geometry=verify_runtime_geometry,
    )
    return ValidatedStoryboardRequest(
        request=storyboard_request,
        card_paths=cards,
        shared=shared,
        card_source_names=source_names,
    )


def build_storyboard_segment_jobs(
    request: RenderRequest | ValidatedStoryboardRequest,
    card_paths: object | None = None,
) -> tuple[StoryboardSegmentJob, ...]:
    """Build exactly one ordinary FIRST_LAST child request for every adjacent card pair."""
    if isinstance(request, ValidatedStoryboardRequest):
        validated = request
    else:
        normalized = request.normalized()
        if normalized.workflow != FL2V_STORYBOARD_WORKFLOW:
            raise RenderValidationError(
                f"Storyboard job construction requires workflow {FL2V_STORYBOARD_WORKFLOW!r}"
            )
        cards = _normalize_storyboard_card_paths(
            normalized.storyboard_card_paths if card_paths is None else card_paths
        )
        validated = ValidatedStoryboardRequest(
            request=replace(normalized, image_paths=(), storyboard_card_paths=cards),
            card_paths=cards,
        )
    base_request = validated.request
    if base_request.mode != FIRST_LAST:
        raise RenderValidationError("FL2V storyboard segments require FIRST_LAST mode")
    return tuple(
        StoryboardSegmentJob(
            segment_index=index,
            start_card_index=index,
            end_card_index=index + 1,
            start_path=start_path,
            end_path=end_path,
            request=replace(
                base_request,
                workflow=SINGLE_RENDER_WORKFLOW,
                image_paths=(start_path, end_path),
                storyboard_card_paths=(),
            ),
        )
        for index, (start_path, end_path) in enumerate(
            zip(validated.card_paths, validated.card_paths[1:]),
            start=1,
        )
    )


def _format_float(value: float) -> str:
    return format(float(value), ".6g")


def build_generation_command(
    validated: ValidatedRequest,
    *,
    python: str | Path = sys.executable,
    conditioning_artifact: Path | None = None,
) -> list[str]:
    """Construct the only H3 command the Render Lab is allowed to launch."""
    request = validated.request
    canonical_artifact_replay = (
        request.text_encoder_id == CANONICAL_ENCODER_ID
        and validated.conditioning_artifact_path is not None
    )
    command = [
        str(python),
        "-u",
        str(GENERATOR),
    ]
    if request.text_encoder_id != HERETIC_ENCODER_ID and not canonical_artifact_replay:
        command.append(request.prompt)
    command.extend([
        "--checkpoint",
        str(validated.checkpoint_root),
        "--duration",
        _format_float(request.duration_seconds),
        "--steps",
        str(request.steps),
        "--seed",
        str(request.seed),
        "--height",
        str(validated.height),
        "--width",
        str(validated.width),
        "--output",
        str(validated.output_root / "<run-directory>" / request.output_name),
    ])
    if request.text_encoder_id == HERETIC_ENCODER_ID:
        artifact_path = conditioning_artifact or validated.output_root / "<run-directory>" / "conditioning-artifact.npz"
        command.extend(["--conditioning-artifact", str(artifact_path)])
    elif canonical_artifact_replay:
        command.extend(["--conditioning-artifact", str(validated.conditioning_artifact_path)])
    if validated.transformer_path is not None:
        command.extend(["--transformer", str(validated.transformer_path)])
    for image_path, anchor in zip(validated.image_paths, validated.anchors):
        command.extend(["--image", str(image_path), "--anchor", anchor])
    if validated.turbo_preset is not None:
        preset = validated.turbo_preset
        if validated.scheduling_adapter_path is None:
            raise RenderValidationError("Turbo preset has no scheduling adapter path")
        command.extend([
            preset.adapter_flag,
            str(validated.scheduling_adapter_path),
            "--lora-scale",
            _format_float(preset.default_scale),
        ])
        if preset.runtime_variant is not None:
            command.extend(["--lightx-variant", preset.runtime_variant])
        command.extend(["--turbo-steps", str(preset.nfe)])
    elif request.turbo_enabled:
        if not request.lora_enabled or validated.scheduling_adapter_path is None:
            raise RenderValidationError("Turbo mode has no scheduling adapter path")
        command.extend([
            "--lora",
            str(validated.scheduling_adapter_path),
            "--lora-scale",
            _format_float(request.lora_scale),
        ])
        command.append("--turbo")
        if request.turbo_steps is not None:
            command.extend(["--turbo-steps", str(request.turbo_steps)])
    elif request.lora_enabled:
        # Preserve the pre-021B direct ordinary-LoRA route for callers that still construct the
        # legacy RenderRequest fields.  Browser-created rows use the repeated auxiliary route.
        command.extend([
            "--lora",
            str(request.lora_path),
            "--lora-scale",
            _format_float(request.lora_scale),
        ])

    auxiliary_loras = validated.additional_loras
    if request.lora_enabled and validated.turbo_preset is None and not request.turbo_enabled:
        auxiliary_loras = auxiliary_loras[1:]
    for entry in auxiliary_loras:
        command.extend([
            "--additional-lora",
            str(entry.path),
            "--additional-lora-scale",
            _format_float(float(entry.scale)),
        ])
    return command


def build_generation_command_for_namespace(validated: ValidatedRequest, namespace: RunNamespace) -> list[str]:
    command = build_generation_command(
        validated,
        conditioning_artifact=namespace.conditioning_artifact_path,
    )
    output_marker = str(validated.output_root / "<run-directory>" / validated.request.output_name)
    command[command.index(output_marker)] = str(namespace.output_path)
    return command


def build_heretic_encoder_command(
    validated: ValidatedRequest,
    namespace: RunNamespace,
    *,
    python: str | Path = sys.executable,
) -> list[str]:
    if validated.request.text_encoder_id != HERETIC_ENCODER_ID:
        raise RenderValidationError("Heretic encoder command requested for a canonical encoder run")
    assets = validated.heretic_assets or probe_heretic_assets(REPO_ROOT)
    return [
        str(python),
        "-u",
        str(HERETIC_ENCODER_SCRIPT),
        "--prompt",
        validated.request.prompt,
        "--checkpoint",
        str(validated.checkpoint_root),
        "--model",
        str(assets.model_path),
        "--bridge",
        str(assets.bridge_path),
        "--artifact",
        str(namespace.conditioning_artifact_path),
        "--evidence",
        str(namespace.encoder_evidence_path),
        "--release-evidence",
        str(namespace.encoder_release_path),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _git_identity(repo_root: Path) -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip()

    return {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
    }


def _file_identity(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        try:
            result["sha256"] = sha256_file(path)
            result["size_bytes"] = path.stat().st_size
        except OSError as exc:
            result["error"] = str(exc)
    return result


def runtime_identity(validated: ValidatedRequest, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    transformer = validated.transformer_path or validated.checkpoint_root / "transformer"
    explicit_dimensions = validated.preset.evidence_class == "explicit-independent-dimensions"
    return {
        "model_id": validated.request.model_id,
        "model_label": model_label(validated.request.model_id),
        "resolved_transformer_path": str(validated.transformer_path) if validated.transformer_path else None,
        "generation_entrypoint": _file_identity(GENERATOR),
        "checkpoint_model_index": _file_identity(validated.checkpoint_root / "model_index.json"),
        "transformer_config": _file_identity(transformer / "config.json"),
        "transformer_quant_config": _file_identity(transformer / "quant_config.json"),
        "transformer_mode": _transformer_mode(validated.transformer_path),
        "transformer_name": transformer.name,
        "resolution_source_id": INDEPENDENT_DIMENSION_SOURCE_ID if explicit_dimensions else PROJECT_RESOLUTION_SOURCE_ID,
        "runtime_resolution_rule_id": INDEPENDENT_DIMENSION_RULE_ID if explicit_dimensions else RUNTIME_RESOLUTION_RULE_ID,
        "host_asset_manifest": {
            "status": HOST_ASSET_MANIFEST_STATUS,
            "note": HOST_ASSET_MANIFEST_NOTE,
        },
        "runtime_branch_and_head": _git_identity(repo_root),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_json(path: Path, default: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {} if default is None else default
    return value if isinstance(value, dict) else ({} if default is None else default)


def _additional_lora_evidence(validated: ValidatedRequest) -> list[dict[str, Any]]:
    return [
        {
            "order": index,
            "enabled": True,
            "path": str(entry.path),
            "scale": float(entry.scale),
            "role": "auxiliary-model-delta",
        }
        for index, entry in enumerate(validated.additional_loras)
    ]


def _model_evidence(validated: ValidatedRequest) -> dict[str, Any]:
    model_id = validated.request.model_id
    return {
        "id": model_id,
        "label": model_label(model_id),
        "transformer_path": str(validated.transformer_path) if validated.transformer_path else None,
        "transformer_mode": _transformer_mode(validated.transformer_path),
    }


def _turbo_evidence(validated: ValidatedRequest) -> dict[str, Any]:
    request = validated.request
    preset = validated.turbo_preset
    if preset is not None:
        if validated.scheduling_adapter_path is None:
            raise RenderValidationError("Turbo preset has no scheduling adapter evidence")
        return {
            "selected": True,
            "preset_id": preset.preset_id,
            "label": preset.label,
            "role": preset.role,
            "family": preset.family,
            "logical_asset": preset.logical_asset,
            "adapter_asset": {
                "flag": preset.adapter_flag,
                "path": str(validated.scheduling_adapter_path),
                "logical_asset": preset.logical_asset,
            },
            "scheduling_owner": "turbo-preset",
            "effective_nfe": preset.nfe,
            "runtime_variant": preset.runtime_variant,
            "runtime_contract": preset.runtime_contract,
            "effective_scale": preset.default_scale,
            "effective_scheduler": {
                "video_shift": preset.video_shift,
                "audio_shift": preset.audio_shift,
                "source": "LightX2V manifest" if preset.family == "LightX2V" else "H3 PipelineConfig defaults",
            },
            "recommended_geometry": preset.recommended_geometry,
            "asset_manifest": {
                "status": HOST_ASSET_MANIFEST_STATUS,
                "note": HOST_ASSET_MANIFEST_NOTE,
            },
        }

    effective_nfe = request.turbo_steps if request.turbo_enabled and request.turbo_steps is not None else request.steps
    if request.turbo_enabled and validated.scheduling_adapter_path is None:
        raise RenderValidationError("Legacy Turbo has no scheduling adapter evidence")
    legacy_turbo = bool(request.turbo_enabled)
    return {
        "selected": False,
        "preset_id": None,
        "label": "None / Reference",
        "role": "Reference",
        "family": None,
        "logical_asset": None,
        "adapter_asset": (
            {
                "flag": "--lora",
                "path": str(validated.scheduling_adapter_path),
                "logical_asset": None,
            }
            if legacy_turbo
            else None
        ),
        "scheduling_owner": "legacy-turbo-adapter" if legacy_turbo else "none-reference",
        "effective_nfe": effective_nfe,
        "runtime_variant": None,
        "runtime_contract": (
            "legacy manual Turbo scheduling behavior"
            if legacy_turbo
            else "existing non-Turbo / manual LoRA behavior"
        ),
        "effective_scale": float(request.lora_scale) if legacy_turbo else None,
        "effective_scheduler": {
            "video_shift": 12.0,
            "audio_shift": 3.0,
            "source": "H3 PipelineConfig defaults",
        },
        "recommended_geometry": None,
        "asset_manifest": {
            "status": HOST_ASSET_MANIFEST_STATUS,
            "note": HOST_ASSET_MANIFEST_NOTE,
        },
    }


def _conditioning_source(validated: ValidatedRequest) -> str:
    if validated.request.text_encoder_id == CANONICAL_ENCODER_ID:
        return "artifact-replay" if validated.conditioning_artifact_evidence is not None else "live-encoder"
    return "heretic-internal-artifact"


def _conditioning_artifact_payload(validated: ValidatedRequest) -> dict[str, Any] | None:
    evidence = validated.conditioning_artifact_evidence
    if evidence is None:
        return None
    return {
        "path": str(evidence.path),
        "artifact_identity": evidence.artifact_identity,
        "token_count": evidence.token_count,
        "conditioning_shape": list(evidence.conditioning_shape),
        "tensor_checksum": evidence.tensor_checksum,
    }


def _text_encoder_evidence(validated: ValidatedRequest, namespace: RunNamespace) -> dict[str, Any]:
    request = validated.request
    if request.text_encoder_id == CANONICAL_ENCODER_ID:
        return {
            "id": CANONICAL_ENCODER_ID,
            "label": "Canonical Qwen3-VL",
            "experimental": False,
            "mode_contract": "T2V, I2V, FIRST_LAST",
            "conditioning_source": _conditioning_source(validated),
            "conditioning_artifact": _conditioning_artifact_payload(validated),
            "h3_launch_after_encoder_exit": None,
        }
    assets = validated.heretic_assets or probe_heretic_assets(REPO_ROOT)
    return {
        "id": HERETIC_ENCODER_ID,
        "label": "Heretic 35B-A3B · Experimental",
        "experimental": True,
        "mode_contract": "T2V only",
        "conditioning_source": _conditioning_source(validated),
        "hint": "Experimental text-only encoder using state 28 + learned H3 conditioning bridge.",
        "source_model": {
            "path": str(assets.model_path),
            "config_sha256": assets.model_config_sha256,
            "selected_state": "hidden_states[28]",
            "maximum_executed_state": 28,
            "layers_29_through_40_executed": False,
            "source_width": 2048,
            "full_decoder_layers": 40,
        },
        "target_width": 5120,
        "bridge": {
            "path": str(assets.bridge_path),
            "sha256": assets.bridge_sha256 or "8dc5dabb7da0d69dfe7ec0d5d80f684a50768d500b46bf70c03cec557141068e",
            "expected_sha256": "8dc5dabb7da0d69dfe7ec0d5d80f684a50768d500b46bf70c03cec557141068e",
            "shapes": {
                "input_mean": [2048],
                "input_scale": [2048],
                "target_mean": [5120],
                "weights": [2048, 5120],
            },
        },
        "token_alignment": "pending exact canonical-piece check",
        "canonical_token_count": None,
        "heretic_token_count": None,
        "conditioning_artifact": str(namespace.conditioning_artifact_path),
        "artifact_identity": None,
        "tensor_checksum": None,
        "timing_seconds": None,
        "peak_memory_bytes": None,
        "active_memory_after_release_bytes": None,
        "cache_memory_after_release_bytes": None,
        "release_gate": "pending",
        "encoder_process_exit_code": None,
        "h3_launched_before_encoder_exit": False,
        "h3_launched_after_encoder_exit": False,
    }


def _status_record(namespace: RunNamespace, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "run_id": namespace.run_id,
        "status": status,
        "run_directory": str(namespace.run_dir),
        "updated_at": iso_timestamp(),
        **extra,
    }


def write_status(namespace: RunNamespace, status: str, **extra: Any) -> dict[str, Any]:
    value = _status_record(namespace, status, **extra)
    _write_json(namespace.status_path, value)
    return value


def reserve_run_namespace(validated: ValidatedRequest) -> RunNamespace:
    """Reserve a never-before-used directory and create the required evidence placeholders."""
    validated.output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        now = _utc_now()
        run_id = f"run-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
        run_dir = validated.output_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        namespace = RunNamespace(
            run_id=run_id,
            run_dir=run_dir,
            output_path=run_dir / validated.request.output_name,
            created_at=iso_timestamp(now),
        )
        (run_dir / "inputs").mkdir()
        namespace.telemetry_dir.mkdir()
        namespace.stdout_path.touch()
        namespace.stderr_path.touch()
        _write_json(namespace.benchmark_path, {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": namespace.run_id,
            "status": "pending",
        })
        write_status(namespace, "reserved")
        return namespace
    raise RuntimeError("Could not reserve a unique immutable Render Lab run directory")


def reserve_storyboard_namespace(output_root: Path) -> RunNamespace:
    """Reserve a normal run-shaped parent namespace whose artifact is the storyboard manifest."""
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(20):
        now = _utc_now()
        run_id = f"run-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:10]}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        namespace = RunNamespace(
            run_id=run_id,
            run_dir=run_dir,
            output_path=run_dir / "storyboard-manifest.json",
            created_at=iso_timestamp(now),
        )
        (run_dir / "inputs").mkdir()
        namespace.telemetry_dir.mkdir()
        namespace.stdout_path.touch()
        namespace.stderr_path.touch()
        _write_json(namespace.benchmark_path, {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": namespace.run_id,
            "workflow": FL2V_STORYBOARD_WORKFLOW,
            "status": "pending",
        })
        write_status(namespace, "reserved", workflow=FL2V_STORYBOARD_WORKFLOW)
        return namespace
    raise RuntimeError("Could not reserve a unique Render Lab storyboard namespace")


def storyboard_segment_output_name(output_name: str, segment_index: int) -> str:
    """Derive one collision-safe, ordered segment filename from the requested base name."""
    if segment_index < 1:
        raise RenderValidationError("Storyboard segment index must be positive")
    name = validate_output_name(output_name)
    path = Path(name)
    return f"{path.stem}-{segment_index:02d}{path.suffix}"


def reserve_storyboard_segment_namespace(
    parent: RunNamespace,
    validated: ValidatedRequest,
    segment_index: int,
) -> RunNamespace:
    """Reserve a uniquely identified child namespace without creating a child directory."""
    output_name = storyboard_segment_output_name(validated.request.output_name, segment_index)
    artifact_prefix = Path(output_name).stem
    namespace = RunNamespace(
        run_id=f"{parent.run_id}-segment-{segment_index:02d}",
        run_dir=parent.run_dir,
        output_path=parent.run_dir / output_name,
        created_at=iso_timestamp(),
        artifact_prefix=artifact_prefix,
    )
    paths = (
        namespace.output_path,
        namespace.config_path,
        namespace.benchmark_path,
        namespace.status_path,
        namespace.stdout_path,
        namespace.stderr_path,
        namespace.telemetry_dir,
    )
    if any(path.exists() for path in paths):
        raise RuntimeError(f"Storyboard segment artifact collision: {output_name}")
    namespace.telemetry_dir.mkdir()
    namespace.stdout_path.touch()
    namespace.stderr_path.touch()
    _write_json(namespace.benchmark_path, {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": namespace.run_id,
        "workflow": FL2V_STORYBOARD_SEGMENT_WORKFLOW,
        "parent_run_id": parent.run_id,
        "segment_index": segment_index,
        "status": "pending",
    })
    write_status(
        namespace,
        "reserved",
        workflow=FL2V_STORYBOARD_SEGMENT_WORKFLOW,
        parent_run_id=parent.run_id,
        segment_index=segment_index,
    )
    return namespace


def _stage_uploads(namespace: RunNamespace, uploads: Sequence[UploadedImage]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, upload in enumerate(uploads, start=1):
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".img"
        path = namespace.run_dir / "inputs" / f"image-{index:02d}{suffix}"
        with path.open("xb") as handle:
            handle.write(upload.data)
        paths.append(validate_image_path(path))
    return tuple(paths)


def _stage_storyboard_uploads(
    namespace: RunNamespace,
    uploads: Sequence[UploadedImage],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, upload in enumerate(uploads, start=1):
        filename = str(upload.filename).strip()
        if not filename:
            raise RenderValidationError(f"Storyboard card {index} has no filename")
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".img"
        path = namespace.run_dir / "inputs" / f"card-{index:02d}{suffix}"
        with path.open("xb") as handle:
            handle.write(upload.data)
        paths.append(validate_image_path(path))
    return tuple(paths)


def _stage_storyboard_paths(
    namespace: RunNamespace,
    card_paths: Sequence[str | Path],
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index, raw_path in enumerate(card_paths, start=1):
        source = validate_image_path(raw_path)
        suffix = source.suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            suffix = ".img"
        path = namespace.run_dir / "inputs" / f"card-{index:02d}{suffix}"
        with source.open("rb") as source_handle, path.open("xb") as target_handle:
            shutil.copyfileobj(source_handle, target_handle)
        paths.append(validate_image_path(path))
    return tuple(paths)


def build_render_config(
    validated: ValidatedRequest,
    namespace: RunNamespace,
    command: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
    encoder_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    request = validated.request
    images = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "anchor": anchor,
        }
        for path, anchor in zip(validated.image_paths, validated.anchors)
    ]
    additional_loras = _additional_lora_evidence(validated)
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_identifier": namespace.run_id,
        "timestamp": namespace.created_at,
        "workflow": request.workflow,
        "generation_mode": request.mode,
        "generation_mode_label": MODE_LABELS.get(request.mode, request.mode),
        "prompt": request.prompt,
        "seed": request.seed,
        "duration_seconds_requested": float(request.duration_seconds),
        "inference_steps_requested": request.steps,
        "resolution_id": request.resolution_id,
        "width": validated.width,
        "height": validated.height,
        "requested_width": request.width,
        "requested_height": request.height,
        "lora_enabled": request.lora_enabled,
        "lora_path": str(request.lora_path) if request.lora_enabled and request.lora_path else None,
        "lora_scale": float(request.lora_scale) if request.lora_enabled else None,
        "additional_loras": additional_loras,
        "turbo_enabled": request.turbo_enabled,
        "turbo_steps": request.turbo_steps,
        "turbo_preset_id": request.turbo_preset_id,
        "lora": {
            "enabled": request.lora_enabled,
            "path": str(request.lora_path) if request.lora_enabled and request.lora_path else None,
            "scale": float(request.lora_scale) if request.lora_enabled else None,
            "compatibility_surface": "legacy-single-adapter-fields",
        },
        "auxiliary_lora_stack": {
            "ordered": additional_loras,
            "scheduling_owner": (
                "turbo-preset"
                if validated.turbo_preset is not None
                else ("legacy-turbo-adapter" if request.turbo_enabled else "none-reference")
            ),
        },
        "turbo": {
            **_turbo_evidence(validated),
            "enabled": request.turbo_enabled,
            "steps": request.turbo_steps,
        },
        "resolution": {
            "preset_id": validated.preset.preset_id,
            "preset_label": validated.preset.label,
            "width": validated.width,
            "height": validated.height,
            "requested_width": validated.preset.expected_width,
            "requested_height": validated.preset.expected_height,
            "runtime_width": validated.width,
            "runtime_height": validated.height,
            "runtime_validated": True,
            "project_approved": validated.preset.project_approved,
            "runtime_rule_id": (
                INDEPENDENT_DIMENSION_RULE_ID
                if validated.preset.evidence_class == "explicit-independent-dimensions"
                else RUNTIME_RESOLUTION_RULE_ID
            ),
            "project_source_id": (
                INDEPENDENT_DIMENSION_SOURCE_ID
                if validated.preset.evidence_class == "explicit-independent-dimensions"
                else PROJECT_RESOLUTION_SOURCE_ID
            ),
            "evidence_class": validated.preset.evidence_class,
            "evidence_reference": validated.preset.evidence_reference,
        },
        "input_image_paths": [item["path"] for item in images],
        "input_images": images,
        "image_anchors": list(validated.anchors),
        "output_root": str(validated.output_root),
        "output_name": request.output_name,
        "output_path": str(namespace.output_path),
        "checkpoint_root": str(validated.checkpoint_root),
        "transformer_path": str(validated.transformer_path) if validated.transformer_path else None,
        "transformer_mode": _transformer_mode(validated.transformer_path),
        "model_id": request.model_id,
        "model": _model_evidence(validated),
        "text_encoder_id": request.text_encoder_id,
        "text_encoder": _text_encoder_evidence(validated, namespace),
        "conditioning_source": _conditioning_source(validated),
        "conditioning_artifact": _conditioning_artifact_payload(validated),
        "conditioning_artifact_path": (
            str(validated.conditioning_artifact_path)
            if validated.conditioning_artifact_path is not None
            else (
                str(namespace.conditioning_artifact_path)
                if request.text_encoder_id == HERETIC_ENCODER_ID
                else None
            )
        ),
        "encoder_command": list(encoder_command) if encoder_command is not None else None,
        "git": _git_identity(repo_root),
        "runtime_identity": runtime_identity(validated, repo_root),
        "command": list(command),
    }


def _storyboard_shared_settings(validated: ValidatedRequest) -> dict[str, Any]:
    """Serialize only settings shared by every adjacent-card child run."""
    request = validated.request
    return {
        "mode": request.mode,
        "prompt": request.prompt,
        "resolution_id": request.resolution_id,
        "width": validated.width,
        "height": validated.height,
        "duration_seconds": float(request.duration_seconds),
        "seed": request.seed,
        "seed_behavior": "same global seed applied to each segment",
        "inference_steps": request.steps,
        "text_encoder_id": request.text_encoder_id,
        "conditioning_artifact_path": (
            str(validated.conditioning_artifact_path)
            if validated.conditioning_artifact_path is not None
            else None
        ),
        "turbo_preset_id": request.turbo_preset_id,
        "turbo": _turbo_evidence(validated),
        "lora_enabled": request.lora_enabled,
        "lora_path": str(request.lora_path) if request.lora_enabled and request.lora_path else None,
        "lora_scale": float(request.lora_scale) if request.lora_enabled else None,
        "additional_loras": _additional_lora_evidence(validated),
        "output_root": str(validated.output_root),
        "output_name": request.output_name,
        "model_id": request.model_id,
        "model": _model_evidence(validated),
        "transformer_path": str(validated.transformer_path) if validated.transformer_path else None,
        "transformer_mode": _transformer_mode(validated.transformer_path),
    }


def _storyboard_card_evidence(
    card_paths: Sequence[Path],
    source_names: Sequence[str] = (),
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, path in enumerate(card_paths, start=1):
        checksum = sha256_file(path) if path.is_file() else None
        evidence.append({
            "card_index": index,
            "source_path": str(path),
            "source_name": source_names[index - 1] if source_names else path.name,
            "sha256": checksum,
            "identity": checksum,
        })
    return evidence


def build_storyboard_config(
    validated: ValidatedStoryboardRequest,
    namespace: RunNamespace,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build the parent run config for a sequential FL2V storyboard."""
    if validated.shared is None:
        raise RenderValidationError("Storyboard parent config requires validated global settings")
    shared = validated.shared
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_identifier": namespace.run_id,
        "timestamp": namespace.created_at,
        "workflow": FL2V_STORYBOARD_WORKFLOW,
        "generation_mode": FIRST_LAST,
        "generation_mode_label": "FL2V storyboard — sequential first/last-frame segments",
        "output_root": str(shared.output_root),
        "output_name": shared.request.output_name,
        "output_path": str(namespace.output_path),
        "model_id": shared.request.model_id,
        "model": _model_evidence(shared),
        "storyboard": {
            "manifest_path": str(namespace.output_path),
            "card_count": len(validated.card_paths),
            "segment_count": len(validated.card_paths) - 1,
            "cards": _storyboard_card_evidence(validated.card_paths, validated.card_source_names),
            "shared_global_settings": _storyboard_shared_settings(shared),
            "execution": {
                "isolation": "one child render process per segment",
                "ordering": "sequential adjacent card pairs",
                "stop_on_failure": True,
                "parallel": False,
            },
        },
        "git": _git_identity(repo_root),
        "runtime_identity": runtime_identity(shared, repo_root),
    }


def _storyboard_manifest(
    validated: ValidatedStoryboardRequest,
    namespace: RunNamespace,
    segments: Sequence[dict[str, Any]],
    status: str,
    *,
    failure_segment_index: int | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    if validated.shared is None:
        raise RenderValidationError("Storyboard manifest requires validated global settings")
    return {
        "schema_version": 1,
        "run_id": namespace.run_id,
        "workflow": FL2V_STORYBOARD_WORKFLOW,
        "mode": FIRST_LAST,
        "card_count": len(validated.card_paths),
        "segment_count": len(validated.card_paths) - 1,
        "cards": _storyboard_card_evidence(validated.card_paths, validated.card_source_names),
        "shared_global_settings": _storyboard_shared_settings(validated.shared),
        "segments": list(segments),
        "per_segment_run_ids": [item.get("child_run_id") for item in segments],
        "per_segment_output_paths": [item.get("output_path") for item in segments],
        "overall_status": status,
        "status": status,
        "completed_segment_count": sum(1 for item in segments if item.get("success")),
        "failure_segment_index": failure_segment_index,
        "failure_reason": failure_reason,
        "execution": {
            "isolation": "one child render process per segment",
            "ordering": "sequential adjacent card pairs",
            "stop_on_failure": True,
            "parallel": False,
        },
    }


def write_storyboard_manifest(
    validated: ValidatedStoryboardRequest,
    namespace: RunNamespace,
    segments: Sequence[dict[str, Any]],
    status: str,
    *,
    failure_segment_index: int | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    manifest = _storyboard_manifest(
        validated,
        namespace,
        segments,
        status,
        failure_segment_index=failure_segment_index,
        failure_reason=failure_reason,
    )
    _write_json(namespace.output_path, manifest)
    return manifest


def initialize_run(namespace: RunNamespace, config: dict[str, Any]) -> None:
    """Write immutable config exactly once for a reserved namespace."""
    if namespace.config_path.exists():
        raise FileExistsError(f"Run config already exists: {namespace.config_path}")
    _write_json(namespace.config_path, config)


def _tail_text(path: Path, limit: int = MAX_LOG_TAIL_BYTES) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def read_run_snapshot(namespace: RunNamespace) -> dict[str, Any]:
    status = _read_json(namespace.status_path)
    benchmark = _read_json(namespace.benchmark_path)
    config = _read_json(namespace.config_path)
    snapshot = {
        "run_id": namespace.run_id,
        "run_directory": str(namespace.run_dir),
        "output_root": config.get("output_root"),
        "status": status.get("status", "unknown"),
        "status_record": status,
        "benchmark": benchmark,
        "stdout": _tail_text(namespace.stdout_path),
        "stderr": _tail_text(namespace.stderr_path),
    }
    if config.get("workflow") == FL2V_STORYBOARD_WORKFLOW:
        manifest = _read_json(namespace.output_path)
        snapshot["storyboard"] = manifest
        segments = manifest.get("segments") if isinstance(manifest, dict) else None
        if isinstance(segments, list) and segments:
            latest = segments[-1]
            stdout_path = latest.get("stdout_path")
            stderr_path = latest.get("stderr_path")
            if stdout_path and stderr_path:
                snapshot["stdout"] = _tail_text(Path(str(stdout_path)))
                snapshot["stderr"] = _tail_text(Path(str(stderr_path)))
            else:
                child_directory = latest.get("child_run_directory")
                if child_directory:
                    child_dir = Path(str(child_directory))
                    snapshot["stdout"] = _tail_text(child_dir / "stdout.log")
                    snapshot["stderr"] = _tail_text(child_dir / "stderr.log")
    return snapshot


def _copy_pipe(pipe: Any, path: Path) -> None:
    with path.open("ab") as handle:
        while True:
            chunk = pipe.readline()
            if not chunk:
                break
            handle.write(chunk)
            handle.flush()


def _run_command_text(command: Sequence[str], cwd: Path, namespace: RunNamespace) -> tuple[int, float, str, str]:
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(cwd) + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        with namespace.stderr_path.open("ab") as handle:
            handle.write(f"Render Lab could not launch child process: {exc}\n".encode("utf-8"))
        return 127, time.perf_counter() - started, _tail_text(namespace.stdout_path), _tail_text(namespace.stderr_path)
    assert process.stdout is not None and process.stderr is not None
    stdout_thread = threading.Thread(target=_copy_pipe, args=(process.stdout, namespace.stdout_path), daemon=True)
    stderr_thread = threading.Thread(target=_copy_pipe, args=(process.stderr, namespace.stderr_path), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    process.stdout.close()
    process.stderr.close()
    elapsed = time.perf_counter() - started
    return return_code, elapsed, _tail_text(namespace.stdout_path, 2_000_000), _tail_text(namespace.stderr_path, 2_000_000)


def _run_command_text_with_updates(
    command: Sequence[str],
    cwd: Path,
    namespace: RunNamespace,
    on_update: Callable[[dict[str, Any]], None] | None,
) -> tuple[int, float, str, str]:
    # The reader threads preserve both raw streams.  The polling loop only updates the local UI and
    # is deliberately not part of the child process or the benchmark timing.
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(cwd) + os.pathsep + environment.get("PYTHONPATH", "")
    started = time.perf_counter()
    try:
        process = subprocess.Popen(
            list(command),
            cwd=str(cwd),
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
    except OSError as exc:
        with namespace.stderr_path.open("ab") as handle:
            handle.write(f"Render Lab could not launch child process: {exc}\n".encode("utf-8"))
        return 127, time.perf_counter() - started, _tail_text(namespace.stdout_path), _tail_text(namespace.stderr_path)
    assert process.stdout is not None and process.stderr is not None
    stdout_thread = threading.Thread(target=_copy_pipe, args=(process.stdout, namespace.stdout_path), daemon=True)
    stderr_thread = threading.Thread(target=_copy_pipe, args=(process.stderr, namespace.stderr_path), daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    while process.poll() is None:
        if on_update is not None:
            on_update(read_run_snapshot(namespace))
        time.sleep(0.5)
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    process.stdout.close()
    process.stderr.close()
    if on_update is not None:
        on_update(read_run_snapshot(namespace))
    elapsed = time.perf_counter() - started
    return return_code, elapsed, _tail_text(namespace.stdout_path, 2_000_000), _tail_text(namespace.stderr_path, 2_000_000)


def _command_output(command: Sequence[str], timeout: float = 4.0) -> dict[str, Any]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
        return {
            "command": list(command),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": list(command), "error": str(exc)}


def capture_host_telemetry(output_root: Path) -> dict[str, Any]:
    """Capture lightweight host evidence without importing MLX or touching model files."""
    telemetry: dict[str, Any] = {"captured_at": iso_timestamp()}
    try:
        usage = shutil.disk_usage(output_root)
        telemetry["filesystem_free_bytes"] = usage.free
        telemetry["filesystem_total_bytes"] = usage.total
    except OSError as exc:
        telemetry["filesystem_error"] = str(exc)
    telemetry["vm_stat"] = _command_output(["vm_stat"])
    ps = _command_output(["ps", "-axo", "pid=,ppid=,rss=,%mem=,comm="])
    if "stdout" in ps:
        rows = []
        for line in ps["stdout"].splitlines():
            fields = line.strip().split(None, 4)
            if len(fields) == 5:
                try:
                    rows.append({
                        "pid": int(fields[0]),
                        "ppid": int(fields[1]),
                        "rss_kb": int(fields[2]),
                        "percent_memory": fields[3],
                        "command": fields[4],
                    })
                except ValueError:
                    continue
        telemetry["top_memory_processes"] = sorted(rows, key=lambda item: item["rss_kb"], reverse=True)[:12]
    telemetry["ps_command"] = ps
    if shutil.which("iostat"):
        telemetry["iostat"] = _command_output(["iostat", "-c", "1", "-w", "1"], timeout=3.0)
    else:
        telemetry["iostat"] = {"available": False}
    return telemetry


def _write_telemetry(namespace: RunNamespace, phase: str, telemetry: dict[str, Any]) -> None:
    _write_json(namespace.telemetry_dir / f"{phase}.json", telemetry)


_STEP_RE = re.compile(r"^\s*step\s+(\d+)/(\d+)\s+([0-9]+(?:\.[0-9]+)?)s", re.IGNORECASE | re.MULTILINE)
_FINAL_TIMING_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)s\s+per\s+step,\s+([0-9]+(?:\.[0-9]+)?)\s+min\s+total",
    re.IGNORECASE,
)
_MEMORY_RE = re.compile(r"mlx_peak=([0-9]+(?:\.[0-9]+)?)(B|KB|MB|GB|TB)", re.IGNORECASE)
_STAGE_RE = re.compile(r"^\s{2}(?!\[memory\])([^:\n]+):([^\n]*)", re.MULTILINE)
_SECONDS_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)s")


def _human_bytes_to_int(value: str, unit: str) -> int:
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit.upper()]
    return int(float(value) * scale)


def parse_runtime_metrics(stdout: str, stderr: str = "") -> dict[str, Any]:
    combined = "\n".join(part for part in (stdout, stderr) if part)
    steps = [
        {"step": int(match.group(1)), "total_steps": int(match.group(2)), "seconds": float(match.group(3))}
        for match in _STEP_RE.finditer(combined)
    ]
    stage_timings: dict[str, list[float]] = {}
    for match in _STAGE_RE.finditer(combined):
        label = match.group(1).strip()
        if label.lower().startswith("step") or label.lower().startswith("first denoising step"):
            continue
        values = [float(value) for value in _SECONDS_RE.findall(match.group(2))]
        if values:
            stage_timings.setdefault(label, []).append(values[-1])
    peaks = [
        _human_bytes_to_int(match.group(1), match.group(2))
        for match in _MEMORY_RE.finditer(combined)
    ]
    final_timing = _FINAL_TIMING_RE.search(combined)
    allocator_lines = [
        line.strip() for line in combined.splitlines()
        if "allocator" in line.lower() or "clear_cache" in line.lower()
    ][-50:]
    release_lines = [
        line.strip() for line in combined.splitlines()
        if "release" in line.lower()
    ][-50:]
    result: dict[str, Any] = {
        "actual_transformer_forward_count": len(steps) if steps else None,
        "observed_step_timings": steps,
        "seconds_per_forward": (
            sum(item["seconds"] for item in steps) / len(steps) if steps else None
        ),
        "stage_timings_seconds": stage_timings,
        "peak_mlx_memory_bytes": max(peaks) if peaks else None,
        "allocator_release_evidence": {
            "allocator_lines": allocator_lines,
            "release_lines": release_lines,
        },
    }
    if final_timing:
        result["runtime_reported_seconds_per_step"] = float(final_timing.group(1))
        result["runtime_reported_total_seconds"] = float(final_timing.group(2)) * 60.0
    else:
        result["runtime_reported_seconds_per_step"] = None
        result["runtime_reported_total_seconds"] = None
    return result


def recognize_output_artifact(output_path: Path) -> Path | None:
    if output_path.is_file() and output_path.stat().st_size > 0:
        return output_path
    frame_dir = output_path.with_suffix("")
    if frame_dir.is_dir() and any(frame_dir.glob("*.png")):
        return frame_dir
    wav_path = output_path.with_suffix(".wav")
    if wav_path.is_file() and wav_path.stat().st_size > 0:
        return wav_path
    return None


def _artifact_record(artifact: Path | None) -> dict[str, Any] | None:
    if artifact is None:
        return None
    record: dict[str, Any] = {
        "path": str(artifact),
        "kind": "directory" if artifact.is_dir() else artifact.suffix.lstrip(".") or "file",
    }
    if artifact.is_file():
        try:
            record["size_bytes"] = artifact.stat().st_size
            record["sha256"] = sha256_file(artifact)
        except OSError as exc:
            record["error"] = str(exc)
    else:
        files = [path for path in artifact.rglob("*") if path.is_file()]
        record["file_count"] = len(files)
        record["size_bytes"] = sum(path.stat().st_size for path in files)
    return record


def _storyboard_child_artifact_paths(namespace: RunNamespace) -> dict[str, Any]:
    return {
        "output_path": str(namespace.output_path),
        "config_path": str(namespace.config_path),
        "status_path": str(namespace.status_path),
        "stdout_path": str(namespace.stdout_path),
        "stderr_path": str(namespace.stderr_path),
        "benchmark_path": str(namespace.benchmark_path),
        "telemetry_dir": str(namespace.telemetry_dir),
        "telemetry_before_path": str(namespace.telemetry_dir / "before.json"),
        "telemetry_after_path": str(namespace.telemetry_dir / "after.json"),
    }


def _storyboard_segment_record(
    job: StoryboardSegmentJob,
    result: RunResult | None,
    *,
    child_namespace: RunNamespace | None = None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    child_namespace = child_namespace or (result.namespace if result is not None else None)
    benchmark = result.benchmark if result is not None else {}
    artifact = _artifact_record(result.output_artifact) if result is not None else None
    child_status = (
        _read_json(child_namespace.status_path).get("status")
        if child_namespace is not None
        else None
    )
    artifact_paths = (
        _storyboard_child_artifact_paths(child_namespace)
        if child_namespace is not None
        else {}
    )
    return {
        "segment_index": job.segment_index,
        "start_card_index": job.start_card_index,
        "end_card_index": job.end_card_index,
        "model_id": job.request.model_id,
        "start_card_path": str(job.start_path),
        "end_card_path": str(job.end_path),
        "child_run_id": child_namespace.run_id if child_namespace is not None else None,
        "child_run_directory": str(child_namespace.run_dir) if child_namespace is not None else None,
        "child_exit_code": result.exit_code if result is not None else None,
        "exit_code": result.exit_code if result is not None else None,
        "child_status": child_status,
        "success": bool(result is not None and result.success),
        "output_path": (
            artifact_paths.get("output_path")
            if artifact_paths
            else (artifact.get("path") if artifact else None)
        ),
        "output_checksum": artifact.get("sha256") if artifact else None,
        "config_path": artifact_paths.get("config_path"),
        "status_path": artifact_paths.get("status_path"),
        "stdout_path": artifact_paths.get("stdout_path"),
        "stderr_path": artifact_paths.get("stderr_path"),
        "benchmark_path": artifact_paths.get("benchmark_path"),
        "telemetry_dir": artifact_paths.get("telemetry_dir"),
        "telemetry_before_path": artifact_paths.get("telemetry_before_path"),
        "telemetry_after_path": artifact_paths.get("telemetry_after_path"),
        "artifact_paths": artifact_paths,
        "timing": {
            "total_elapsed_seconds": benchmark.get("total_elapsed_seconds"),
            "child_process_elapsed_seconds": benchmark.get("child_process_elapsed_seconds"),
            "wall_clock_start": benchmark.get("wall_clock_start"),
            "wall_clock_end": benchmark.get("wall_clock_end"),
        },
        "memory": {
            "peak_mlx_memory_bytes": benchmark.get("peak_mlx_memory_bytes"),
            "telemetry": benchmark.get("telemetry"),
            "allocator_release_evidence": benchmark.get("allocator_release_evidence"),
        },
        "failure_reason": failure_reason or benchmark.get("failure_reason"),
    }


def _run_encoder_child(
    namespace: RunNamespace,
    command: Sequence[str],
    *,
    repo_root: Path,
    on_update: Callable[[dict[str, Any]], None] | None,
    command_runner: Callable[..., tuple[int, float, str, str]] | None,
) -> tuple[int, float, str, str]:
    runner = command_runner or _run_command_text_with_updates
    try:
        if runner is _run_command_text_with_updates:
            return runner(command, repo_root, namespace, on_update)
        return runner(command, repo_root, namespace)
    except Exception as exc:
        with namespace.stderr_path.open("ab") as handle:
            handle.write(f"Heretic encoder runner failed: {exc}\n".encode("utf-8"))
        return 127, 0.0, _tail_text(namespace.stdout_path, 2_000_000), _tail_text(namespace.stderr_path, 2_000_000)


def execute_heretic_run(
    namespace: RunNamespace,
    encoder_command: Sequence[str],
    h3_command: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    command_runner: Callable[..., tuple[int, float, str, str]] | None = None,
) -> RunResult:
    """Run Heretic, validate/release its artifact, then launch H3 in a new process."""
    output_root = output_root or namespace.run_dir.parent
    write_status(
        namespace,
        "running",
        phase="heretic-encoder",
        encoder_command=list(encoder_command),
        h3_launched_before_encoder_exit=False,
        started_at=iso_timestamp(),
    )
    encoder_exit, encoder_elapsed, stdout, stderr = _run_encoder_child(
        namespace,
        encoder_command,
        repo_root=repo_root,
        on_update=on_update,
        command_runner=command_runner,
    )
    evidence = _read_json(namespace.encoder_evidence_path)
    release = _read_json(namespace.encoder_release_path)
    encoder_error: str | None = None
    loaded_artifact: Any | None = None
    if encoder_exit != 0:
        encoder_error = "Heretic encoder child exited nonzero"
    elif not evidence or evidence.get("status") != "complete":
        encoder_error = "Heretic encoder did not publish complete evidence"
    elif evidence.get("h3_launched_before_encoder_exit") is not False:
        encoder_error = "Heretic evidence does not prove H3 waited for encoder process exit"
    elif evidence.get("release_gate") is not True or release.get("clean") is not True:
        encoder_error = "Heretic encoder release gate did not pass"
    else:
        try:
            from minimax_h3_mlx.conditioning_artifact import (
                load_conditioning_artifact,
                validate_conditioning_artifact,
            )

            loaded_artifact = load_conditioning_artifact(namespace.conditioning_artifact_path)
            config = _read_json(namespace.config_path)
            validate_conditioning_artifact(
                loaded_artifact,
                checkpoint_root=config.get("checkpoint_root"),
                prompt=config.get("prompt"),
            )
            artifact_evidence = evidence.get("conditioning_artifact", {})
            if (
                artifact_evidence.get("path") != str(namespace.conditioning_artifact_path)
                or artifact_evidence.get("identity") != loaded_artifact.artifact_identity
                or artifact_evidence.get("tensor_checksum") != loaded_artifact.tensor_checksum
            ):
                encoder_error = "Heretic evidence does not match the validated conditioning artifact"
        except Exception as exc:
            encoder_error = f"Heretic conditioning artifact failed replay validation: {exc}"
    if encoder_error is not None:
        benchmark = {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "run_id": namespace.run_id,
            "success": False,
            "process_exit_code": encoder_exit,
            "encoder_process_exit_code": encoder_exit,
            "encoder_process_elapsed_seconds": encoder_elapsed,
            "encoder_evidence": evidence or None,
            "encoder_release_evidence": release or None,
            "h3_launched_before_encoder_exit": False,
            "h3_launched_after_encoder_exit": False,
            "failure_reason": encoder_error,
        }
        _write_json(namespace.benchmark_path, benchmark)
        write_status(
            namespace,
            "failed",
            success=False,
            phase="heretic-encoder",
            exit_code=encoder_exit,
            encoder_process_exit_code=encoder_exit,
            h3_launched_before_encoder_exit=False,
            failure_reason=encoder_error,
            finished_at=iso_timestamp(),
        )
        return RunResult(namespace, encoder_exit, False, None, benchmark)

    h3_result = execute_run(
        namespace,
        h3_command,
        repo_root=repo_root,
        output_root=output_root,
        on_update=on_update,
        command_runner=command_runner,
    )
    benchmark = dict(h3_result.benchmark)
    benchmark.update({
        "encoder_process_exit_code": encoder_exit,
        "encoder_process_elapsed_seconds": encoder_elapsed,
        "encoder_evidence": evidence,
        "encoder_release_evidence": release,
        "h3_launched_before_encoder_exit": False,
        "h3_launched_after_encoder_exit": True,
        "h3_process_created_after_encoder_exit": True,
        "conditioning_artifact": {
            "path": str(loaded_artifact.path),
            "identity": loaded_artifact.artifact_identity,
            "tensor_checksum": loaded_artifact.tensor_checksum,
        },
    })
    _write_json(namespace.benchmark_path, benchmark)
    write_status(
        namespace,
        "succeeded" if h3_result.success else "failed",
        exit_code=h3_result.exit_code,
        success=h3_result.success,
        phase="h3",
        encoder_process_exit_code=encoder_exit,
        h3_launched_before_encoder_exit=False,
        h3_launched_after_encoder_exit=True,
        finished_at=benchmark.get("wall_clock_end", iso_timestamp()),
        output_artifact=_artifact_record(h3_result.output_artifact),
    )
    return RunResult(namespace, h3_result.exit_code, h3_result.success, h3_result.output_artifact, benchmark)


def execute_run(
    namespace: RunNamespace,
    command: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    on_update: Callable[[dict[str, Any]], None] | None = None,
    telemetry: Callable[[Path], dict[str, Any]] = capture_host_telemetry,
    command_runner: Callable[..., tuple[int, float, str, str]] | None = None,
) -> RunResult:
    output_root = output_root or namespace.run_dir.parent
    started_wall = _utc_now()
    started_monotonic = time.perf_counter()
    write_status(namespace, "running", pid=None, command=list(command), started_at=iso_timestamp(started_wall))
    try:
        before = telemetry(output_root)
    except Exception as exc:  # telemetry is nonfatal by contract
        before = {"captured_at": iso_timestamp(), "error": str(exc)}
    _write_telemetry(namespace, "before", before)
    runner = command_runner or _run_command_text_with_updates
    try:
        if runner is _run_command_text_with_updates:
            exit_code, child_elapsed, stdout, stderr = runner(
                command, repo_root, namespace, on_update
            )
        else:
            exit_code, child_elapsed, stdout, stderr = runner(command, repo_root, namespace)
    except Exception as exc:
        exit_code = 127
        child_elapsed = time.perf_counter() - started_monotonic
        with namespace.stderr_path.open("ab") as handle:
            handle.write(f"Render Lab runner failed: {exc}\n".encode("utf-8"))
        stdout = _tail_text(namespace.stdout_path, 2_000_000)
        stderr = _tail_text(namespace.stderr_path, 2_000_000)
    try:
        after = telemetry(output_root)
    except Exception as exc:
        after = {"captured_at": iso_timestamp(), "error": str(exc)}
    _write_telemetry(namespace, "after", after)
    finished_wall = _utc_now()
    elapsed = time.perf_counter() - started_monotonic
    artifact = recognize_output_artifact(namespace.output_path)
    metrics = parse_runtime_metrics(stdout, stderr)
    success = exit_code == 0 and artifact is not None
    benchmark: dict[str, Any] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": namespace.run_id,
        "wall_clock_start": iso_timestamp(started_wall),
        "wall_clock_end": iso_timestamp(finished_wall),
        "total_elapsed_seconds": elapsed,
        "child_process_elapsed_seconds": child_elapsed,
        "process_exit_code": exit_code,
        "success": success,
        "configured_steps": _read_json(namespace.config_path).get("inference_steps_requested"),
        "configured_inference_steps": _read_json(namespace.config_path).get("inference_steps_requested"),
        "output_file_size_bytes": (
            (_artifact_record(artifact) or {}).get("size_bytes") if artifact and artifact.is_file() else None
        ),
        "output_artifact": _artifact_record(artifact),
        "telemetry": {
            "before_path": str(namespace.telemetry_dir / "before.json"),
            "after_path": str(namespace.telemetry_dir / "after.json"),
        },
        **metrics,
    }
    if not success:
        benchmark["failure_reason"] = (
            "child process exited nonzero" if exit_code != 0 else "child exited successfully without a media artifact"
        )
    _write_json(namespace.benchmark_path, benchmark)
    write_status(
        namespace,
        "succeeded" if success else "failed",
        exit_code=exit_code,
        success=success,
        finished_at=benchmark["wall_clock_end"],
        output_artifact=_artifact_record(artifact),
    )
    return RunResult(namespace, exit_code, success, artifact, benchmark)


def execute_storyboard(
    namespace: RunNamespace,
    validated: ValidatedStoryboardRequest,
    *,
    repo_root: Path = REPO_ROOT,
    output_root: Path | None = None,
    telemetry: Callable[[Path], dict[str, Any]] = capture_host_telemetry,
    command_runner: Callable[..., tuple[int, float, str, str]] | None = None,
    check_runtime_paths: bool = True,
) -> RunResult:
    """Run adjacent FL2V pairs in isolated child processes, stopping at the first failure."""
    if validated.shared is None:
        raise RenderValidationError("Storyboard execution requires validated global settings")
    output_root = output_root or validated.shared.output_root
    started_wall = _utc_now()
    started_monotonic = time.perf_counter()
    jobs = build_storyboard_segment_jobs(validated)
    segments: list[dict[str, Any]] = []
    write_status(
        namespace,
        "running",
        workflow=FL2V_STORYBOARD_WORKFLOW,
        segment_count=len(jobs),
        completed_segment_count=0,
    )
    write_storyboard_manifest(validated, namespace, segments, "running")

    failure_segment_index: int | None = None
    failure_reason: str | None = None
    failure_exit_code: int | None = None
    for job in jobs:
        write_status(
            namespace,
            "running",
            workflow=FL2V_STORYBOARD_WORKFLOW,
            segment_index=job.segment_index,
            start_card_index=job.start_card_index,
            end_card_index=job.end_card_index,
            segment_count=len(jobs),
            completed_segment_count=sum(1 for item in segments if item.get("success")),
        )
        child_namespace: RunNamespace | None = None
        child_result: RunResult | None = None
        try:
            assert validated.shared is not None
            child_namespace = reserve_storyboard_segment_namespace(
                namespace,
                validated.shared,
                job.segment_index,
            )
            child_request = replace(
                job.request,
                output_name=child_namespace.output_path.name,
            )
            child_validated = validate_render_request(
                child_request,
                repo_root=repo_root,
                check_runtime_paths=check_runtime_paths,
                check_images=True,
                verify_runtime_geometry=False,
            )
            child_command = build_generation_command_for_namespace(child_validated, child_namespace)
            child_config = build_render_config(
                child_validated,
                child_namespace,
                child_command,
                repo_root=repo_root,
            )
            child_config.update({
                "workflow": FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                "storyboard": {
                    "parent_run_id": namespace.run_id,
                    "segment_index": job.segment_index,
                    "start_card_index": job.start_card_index,
                    "end_card_index": job.end_card_index,
                    "start_card_path": str(job.start_path),
                    "end_card_path": str(job.end_path),
                    "child_run_id": child_namespace.run_id,
                    "artifact_paths": _storyboard_child_artifact_paths(child_namespace),
                },
            })
            initialize_run(child_namespace, child_config)
            child_result = execute_run(
                child_namespace,
                child_command,
                repo_root=repo_root,
                output_root=output_root,
                telemetry=telemetry,
                command_runner=command_runner,
            )
            child_benchmark = dict(child_result.benchmark)
            child_benchmark.update({
                "workflow": FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                "parent_run_id": namespace.run_id,
                "segment_index": job.segment_index,
                "start_card_index": job.start_card_index,
                "end_card_index": job.end_card_index,
                "artifact_paths": _storyboard_child_artifact_paths(child_namespace),
            })
            _write_json(child_namespace.benchmark_path, child_benchmark)
            write_status(
                child_namespace,
                "succeeded" if child_result.success else "failed",
                workflow=FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                parent_run_id=namespace.run_id,
                segment_index=job.segment_index,
                start_card_index=job.start_card_index,
                end_card_index=job.end_card_index,
                success=child_result.success,
                exit_code=child_result.exit_code,
                finished_at=child_benchmark.get("wall_clock_end", iso_timestamp()),
                output_artifact=_artifact_record(child_result.output_artifact),
                artifact_paths=_storyboard_child_artifact_paths(child_namespace),
            )
        except Exception as exc:
            failure_reason = str(exc)
            if child_namespace is not None:
                if not child_namespace.config_path.exists():
                    _write_json(child_namespace.config_path, {
                        "schema_version": RUN_SCHEMA_VERSION,
                        "run_identifier": child_namespace.run_id,
                        "timestamp": child_namespace.created_at,
                        "workflow": FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                        "generation_mode": FIRST_LAST,
                        "output_root": str(output_root),
                        "output_path": str(child_namespace.output_path),
                        "storyboard": {
                            "parent_run_id": namespace.run_id,
                            "segment_index": job.segment_index,
                            "child_run_id": child_namespace.run_id,
                            "artifact_paths": _storyboard_child_artifact_paths(child_namespace),
                        },
                        "admission_failure": failure_reason,
                    })
                _write_json(child_namespace.benchmark_path, {
                    "schema_version": BENCHMARK_SCHEMA_VERSION,
                    "run_id": child_namespace.run_id,
                    "workflow": FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                    "parent_run_id": namespace.run_id,
                    "segment_index": job.segment_index,
                    "success": False,
                    "process_exit_code": None,
                    "failure_reason": failure_reason,
                    "artifact_paths": _storyboard_child_artifact_paths(child_namespace),
                })
                write_status(
                    child_namespace,
                    "failed",
                    workflow=FL2V_STORYBOARD_SEGMENT_WORKFLOW,
                    parent_run_id=namespace.run_id,
                    segment_index=job.segment_index,
                    success=False,
                    exit_code=None,
                    failure_reason=failure_reason,
                    artifact_paths=_storyboard_child_artifact_paths(child_namespace),
                )
            child_result = None

        record = _storyboard_segment_record(
            job,
            child_result,
            child_namespace=child_namespace,
            failure_reason=failure_reason if child_result is None else None,
        )
        segments.append(record)
        if not record["success"]:
            failure_segment_index = job.segment_index
            failure_reason = record.get("failure_reason") or "segment child failed"
            failure_exit_code = record.get("child_exit_code")
            write_storyboard_manifest(
                validated,
                namespace,
                segments,
                "failed",
                failure_segment_index=failure_segment_index,
                failure_reason=failure_reason,
            )
            break
        write_storyboard_manifest(validated, namespace, segments, "running")

    success = failure_segment_index is None and len(segments) == len(jobs)
    if success:
        final_status = "succeeded"
        final_failure_reason = None
        final_exit_code = 0
    else:
        final_status = "failed"
        final_failure_reason = failure_reason or "storyboard did not complete"
        final_exit_code = failure_exit_code if failure_exit_code is not None else 1
    manifest = write_storyboard_manifest(
        validated,
        namespace,
        segments,
        final_status,
        failure_segment_index=failure_segment_index,
        failure_reason=final_failure_reason,
    )
    elapsed = time.perf_counter() - started_monotonic
    benchmark = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_id": namespace.run_id,
        "workflow": FL2V_STORYBOARD_WORKFLOW,
        "wall_clock_start": iso_timestamp(started_wall),
        "wall_clock_end": iso_timestamp(),
        "total_elapsed_seconds": elapsed,
        "process_exit_code": final_exit_code,
        "success": success,
        "segment_count": len(jobs),
        "completed_segment_count": sum(1 for item in segments if item.get("success")),
        "segments": segments,
        "manifest_path": str(namespace.output_path),
        "output_artifact": None,
    }
    if not success:
        benchmark["failure_reason"] = final_failure_reason
        benchmark["failure_segment_index"] = failure_segment_index
    _write_json(namespace.benchmark_path, benchmark)
    write_status(
        namespace,
        final_status,
        workflow=FL2V_STORYBOARD_WORKFLOW,
        success=success,
        exit_code=final_exit_code,
        segment_count=len(jobs),
        completed_segment_count=benchmark["completed_segment_count"],
        failure_segment_index=failure_segment_index,
        failure_reason=final_failure_reason,
        manifest_path=str(namespace.output_path),
        finished_at=benchmark["wall_clock_end"],
    )
    return RunResult(namespace, final_exit_code, success, None, benchmark | {"manifest": manifest})


class RenderFileLock:
    """A process-level output-root lock so a second render cannot be admitted."""

    _fallback_lock = threading.Lock()

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.lock_path = output_root / ".render-lab.lock"
        self._handle: Any = None
        self._fallback_acquired = False

    def acquire(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        try:
            import fcntl
        except ImportError:  # pragma: no cover - macOS has fcntl
            if not self._fallback_lock.acquire(blocking=False):
                raise RenderBusyError("A render is already active")
            self._fallback_acquired = True
            return
        handle = self.lock_path.open("a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RenderBusyError("A render is already active for this output root") from exc
        self._handle = handle

    def release(self) -> None:
        if self._handle is not None:
            try:
                import fcntl

                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None
        if self._fallback_acquired:
            self._fallback_lock.release()
            self._fallback_acquired = False

    def __enter__(self) -> "RenderFileLock":
        self.acquire()
        return self

    def __exit__(self, *_args: Any) -> None:
        self.release()


def history_rows(output_root: str | Path, *, limit: int = 25) -> list[dict[str, Any]]:
    root = Path(output_root).expanduser()
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not RUN_ID_PATTERN.fullmatch(run_dir.name):
            continue
        config = _read_json(run_dir / "render-config.json")
        status = _read_json(run_dir / "run-status.json")
        benchmark = _read_json(run_dir / "benchmark.json")
        if not config:
            continue
        shared = (config.get("storyboard") or {}).get("shared_global_settings", {})
        artifact = benchmark.get("output_artifact") or {}
        artifact_path = artifact.get("path")
        workflow = config.get("workflow")
        display_mode = (
            workflow
            if workflow in {FL2V_STORYBOARD_WORKFLOW, FL2V_STORYBOARD_SEGMENT_WORKFLOW}
            else config.get("generation_mode")
        )
        rows.append({
            "run_id": run_dir.name,
            "timestamp": config.get("timestamp"),
            "mode": display_mode,
            "model_id": config.get("model_id", shared.get("model_id")),
            "resolution": f"{config.get('width', shared.get('width'))} × {config.get('height', shared.get('height'))}",
            "width": config.get("width", shared.get("width")),
            "height": config.get("height", shared.get("height")),
            "steps": config.get("inference_steps_requested", shared.get("inference_steps")),
            "seed": config.get("seed", shared.get("seed")),
            "elapsed_seconds": benchmark.get("total_elapsed_seconds"),
            "status": status.get("status", "unknown"),
            "success": benchmark.get("success"),
            "output_artifact": artifact_path,
            "artifact_name": Path(artifact_path).name if artifact_path else None,
            "output_root": config.get("output_root"),
            "run_directory": str(run_dir),
        })
    rows.sort(key=lambda row: str(row.get("timestamp") or ""), reverse=True)
    return rows[:limit]


class RenderController:
    """Small in-process controller used by the standard-library browser surface."""

    def __init__(self, repo_root: Path = REPO_ROOT):
        self.repo_root = repo_root.resolve()
        self._state_lock = threading.RLock()
        self._active_namespace: RunNamespace | None = None
        self._active_thread: threading.Thread | None = None
        self._active_lock: RenderFileLock | None = None
        self._last_result: RunResult | None = None

    def start(
        self,
        request: RenderRequest,
        *,
        uploads: Sequence[UploadedImage] = (),
    ) -> RunNamespace:
        normalized_request = request.normalized()
        if normalized_request.workflow == FL2V_STORYBOARD_WORKFLOW:
            return self.start_storyboard(normalized_request, uploads=uploads)
        with self._state_lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RenderBusyError("A render is already active")
            request = normalized_request
            # Count validation happens before any child process admission. Uploaded bytes are
            # staged only after a fresh namespace has been reserved.
            submission_count = len(uploads) if uploads else len(request.image_paths)
            anchors_for_mode(request.mode, submission_count)
            preflight_paths = tuple(Path(f"__uploaded_image_{index}") for index in range(submission_count))
            validated = validate_render_request(
                replace(request, image_paths=preflight_paths),
                repo_root=self.repo_root,
                check_runtime_paths=True,
                check_images=False,
                verify_runtime_geometry=True,
            )
            output_lock = RenderFileLock(validated.output_root)
            output_lock.acquire()
            namespace: RunNamespace | None = None
            try:
                namespace = reserve_run_namespace(validated)
                staged_paths = (
                    _stage_uploads(namespace, uploads)
                    if uploads
                    else tuple(validate_image_path(path) for path in request.image_paths)
                )
                final_request = replace(request, image_paths=staged_paths)
                validated = validate_render_request(
                    final_request,
                    repo_root=self.repo_root,
                    check_runtime_paths=True,
                    check_images=True,
                    verify_runtime_geometry=False,
                )
                command = build_generation_command_for_namespace(validated, namespace)
                encoder_command = (
                    build_heretic_encoder_command(validated, namespace)
                    if validated.request.text_encoder_id == HERETIC_ENCODER_ID
                    else None
                )
                initialize_run(
                    namespace,
                    build_render_config(
                        validated,
                        namespace,
                        command,
                        repo_root=self.repo_root,
                        encoder_command=encoder_command,
                    ),
                )
            except Exception as exc:
                if namespace is not None:
                    if not namespace.config_path.exists():
                        _write_json(namespace.config_path, {
                            "schema_version": RUN_SCHEMA_VERSION,
                            "run_identifier": namespace.run_id,
                            "timestamp": namespace.created_at,
                            "generation_mode": request.mode,
                            "prompt": request.prompt,
                            "seed": request.seed,
                            "duration_seconds_requested": float(request.duration_seconds),
                            "inference_steps_requested": request.steps,
                            "resolution_id": request.resolution_id,
                            "width": validated.width,
                            "height": validated.height,
                            "requested_width": validated.width,
                            "requested_height": validated.height,
                            "lora_enabled": validated.request.lora_enabled,
                            "lora_path": (
                                str(validated.request.lora_path)
                                if validated.request.lora_enabled and validated.request.lora_path
                                else None
                            ),
                            "lora_scale": (
                                float(validated.request.lora_scale)
                                if validated.request.lora_enabled
                                else None
                            ),
                            "additional_loras": _additional_lora_evidence(validated),
                            "turbo_enabled": validated.request.turbo_enabled,
                            "turbo_steps": validated.request.turbo_steps,
                            "turbo_preset_id": validated.request.turbo_preset_id,
                            "turbo": {
                                **_turbo_evidence(validated),
                                "enabled": validated.request.turbo_enabled,
                                "steps": validated.request.turbo_steps,
                            },
                            "auxiliary_lora_stack": {
                                "ordered": _additional_lora_evidence(validated),
                                "scheduling_owner": (
                                    "turbo-preset"
                                    if validated.turbo_preset is not None
                                    else (
                                        "legacy-turbo-adapter"
                                        if validated.request.turbo_enabled
                                        else "none-reference"
                                    )
                                ),
                            },
                            "resolution": {
                                "preset_id": validated.preset.preset_id,
                                "width": validated.width,
                                "height": validated.height,
                                "runtime_validated": True,
                                "project_approved": validated.preset.project_approved,
                            },
                            "input_image_paths": [str(path) for path in request.image_paths],
                            "input_image_uploads": [
                                {"filename": item.filename, "sha256": sha256_bytes(item.data)}
                                for item in uploads
                            ],
                            "image_anchors": list(validated.anchors),
                            "output_root": str(validated.output_root),
                            "output_name": request.output_name,
                            "output_path": str(namespace.output_path),
                            "checkpoint_root": str(validated.checkpoint_root),
                            "transformer_path": str(validated.transformer_path) if validated.transformer_path else None,
                            "transformer_mode": _transformer_mode(validated.transformer_path),
                            "model_id": validated.request.model_id,
                            "model": _model_evidence(validated),
                            "text_encoder_id": request.text_encoder_id,
                            "text_encoder": _text_encoder_evidence(validated, namespace),
                            "conditioning_source": _conditioning_source(validated),
                            "conditioning_artifact": _conditioning_artifact_payload(validated),
                            "conditioning_artifact_path": (
                                str(validated.conditioning_artifact_path)
                                if validated.conditioning_artifact_path is not None
                                else (
                                    str(namespace.conditioning_artifact_path)
                                    if request.text_encoder_id == HERETIC_ENCODER_ID
                                    else None
                                )
                            ),
                            "git": _git_identity(self.repo_root),
                            "admission_failure": str(exc),
                        })
                    with namespace.stderr_path.open("ab") as handle:
                        handle.write(f"Render Lab admission failed: {exc}\n".encode("utf-8"))
                    _write_json(namespace.benchmark_path, {
                        "schema_version": BENCHMARK_SCHEMA_VERSION,
                        "run_id": namespace.run_id,
                        "success": False,
                        "process_exit_code": None,
                        "failure_reason": str(exc),
                    })
                    write_status(namespace, "failed", success=False, failure_reason=str(exc))
                output_lock.release()
                raise
            self._active_namespace = namespace
            self._active_lock = output_lock

            def worker() -> None:
                try:
                    if validated.request.text_encoder_id == HERETIC_ENCODER_ID:
                        self._last_result = execute_heretic_run(
                            namespace,
                            encoder_command or (),
                            command,
                            repo_root=self.repo_root,
                            output_root=validated.output_root,
                        )
                    else:
                        self._last_result = execute_run(
                            namespace,
                            command,
                            repo_root=self.repo_root,
                            output_root=validated.output_root,
                        )
                finally:
                    output_lock.release()

            thread = threading.Thread(target=worker, name=f"render-lab-{namespace.run_id}", daemon=True)
            self._active_thread = thread
            thread.start()
            return namespace

    def start_storyboard(
        self,
        request: RenderRequest,
        *,
        uploads: Sequence[UploadedImage] = (),
    ) -> RunNamespace:
        """Admit a storyboard parent, then execute its ordinary child runs sequentially."""
        with self._state_lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RenderBusyError("A render is already active")
            request = request.normalized()
            if request.workflow != FL2V_STORYBOARD_WORKFLOW:
                raise RenderValidationError(
                    f"Storyboard admission requires workflow {FL2V_STORYBOARD_WORKFLOW!r}"
                )
            if uploads and request.storyboard_card_paths:
                raise RenderValidationError("Storyboard cards must be supplied as uploads or paths, not both")
            if uploads:
                if len(uploads) < 2:
                    raise RenderValidationError("FL2V storyboard requires at least 2 cards")
                for index, upload in enumerate(uploads, start=1):
                    if not str(upload.filename).strip():
                        raise RenderValidationError(f"Storyboard card {index} has no filename")
                    if not upload.data:
                        raise RenderValidationError(f"Storyboard card {index} is empty")
                preflight_cards = tuple(Path(f"__uploaded_storyboard_card_{index}") for index in range(len(uploads)))
            else:
                preflight_cards = request.storyboard_card_paths
            card_source_names = (
                tuple(upload.filename for upload in uploads)
                if uploads
                else tuple(Path(path).name for path in request.storyboard_card_paths)
            )
            validated = validate_storyboard_request(
                request,
                preflight_cards,
                card_source_names=card_source_names,
                repo_root=self.repo_root,
                check_runtime_paths=True,
                check_images=False,
                verify_runtime_geometry=True,
            )
            assert validated.shared is not None
            output_lock = RenderFileLock(validated.shared.output_root)
            output_lock.acquire()
            namespace: RunNamespace | None = None
            try:
                namespace = reserve_storyboard_namespace(validated.shared.output_root)
                staged_paths = (
                    _stage_storyboard_uploads(namespace, uploads)
                    if uploads
                    else _stage_storyboard_paths(namespace, request.storyboard_card_paths)
                )
                validated = validate_storyboard_request(
                    request,
                    staged_paths,
                    card_source_names=card_source_names,
                    repo_root=self.repo_root,
                    check_runtime_paths=True,
                    check_images=True,
                    verify_runtime_geometry=False,
                )
                initialize_run(
                    namespace,
                    build_storyboard_config(validated, namespace, repo_root=self.repo_root),
                )
                write_storyboard_manifest(validated, namespace, (), "reserved")
            except Exception as exc:
                if namespace is not None:
                    if not namespace.config_path.exists():
                        _write_json(namespace.config_path, {
                            "schema_version": RUN_SCHEMA_VERSION,
                            "run_identifier": namespace.run_id,
                            "timestamp": namespace.created_at,
                            "workflow": FL2V_STORYBOARD_WORKFLOW,
                            "generation_mode": FIRST_LAST,
                            "output_root": str(validated.shared.output_root if validated.shared else request.output_root),
                            "output_path": str(namespace.output_path),
                            "admission_failure": str(exc),
                        })
                    _write_json(namespace.output_path, {
                        "schema_version": 1,
                        "run_id": namespace.run_id,
                        "workflow": FL2V_STORYBOARD_WORKFLOW,
                        "overall_status": "failed",
                        "status": "failed",
                        "card_count": len(preflight_cards),
                        "segment_count": max(0, len(preflight_cards) - 1),
                        "segments": [],
                        "failure_reason": str(exc),
                    })
                    _write_json(namespace.benchmark_path, {
                        "schema_version": BENCHMARK_SCHEMA_VERSION,
                        "run_id": namespace.run_id,
                        "workflow": FL2V_STORYBOARD_WORKFLOW,
                        "success": False,
                        "process_exit_code": None,
                        "failure_reason": str(exc),
                    })
                    write_status(
                        namespace,
                        "failed",
                        workflow=FL2V_STORYBOARD_WORKFLOW,
                        success=False,
                        failure_reason=str(exc),
                    )
                output_lock.release()
                raise
            self._active_namespace = namespace
            self._active_lock = output_lock
            self._last_result = None

            def worker() -> None:
                try:
                    self._last_result = execute_storyboard(
                        namespace,
                        validated,
                        repo_root=self.repo_root,
                        output_root=validated.shared.output_root if validated.shared else None,
                    )
                except Exception as exc:
                    _write_json(namespace.benchmark_path, {
                        "schema_version": BENCHMARK_SCHEMA_VERSION,
                        "run_id": namespace.run_id,
                        "workflow": FL2V_STORYBOARD_WORKFLOW,
                        "success": False,
                        "process_exit_code": None,
                        "failure_reason": str(exc),
                    })
                    write_status(
                        namespace,
                        "failed",
                        workflow=FL2V_STORYBOARD_WORKFLOW,
                        success=False,
                        failure_reason=str(exc),
                    )
                    self._last_result = RunResult(
                        namespace,
                        None,
                        False,
                        None,
                        _read_json(namespace.benchmark_path),
                    )
                finally:
                    output_lock.release()

            thread = threading.Thread(target=worker, name=f"render-lab-storyboard-{namespace.run_id}", daemon=True)
            self._active_thread = thread
            thread.start()
            return namespace

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            if self._active_namespace is None:
                return {"running": False, "status": "idle"}
            snapshot = read_run_snapshot(self._active_namespace)
            thread = self._active_thread
            snapshot["running"] = bool(thread and thread.is_alive())
            snapshot["last_result"] = self._last_result.benchmark if self._last_result else None
            return snapshot

    def config_payload(self) -> dict[str, Any]:
        checkpoint = default_checkpoint_root(self.repo_root)
        transformer = default_transformer_path(self.repo_root)
        models = model_selection_payload(self.repo_root)
        model_status = {item["id"]: item for item in models}
        default_model_status = model_status[DEFAULT_MODEL_ID]
        return {
            "models": models,
            "modes": [{"id": mode, "label": MODE_LABELS[mode], "image_count": expected_image_count(mode)} for mode in (T2V, I2V, FIRST_LAST)],
            "workflows": [
                {"id": workflow, "label": WORKFLOW_LABELS[workflow]}
                for workflow in (SINGLE_RENDER_WORKFLOW, FL2V_STORYBOARD_WORKFLOW)
            ],
            "text_encoders": text_encoder_payload(self.repo_root),
            "resolutions": preset_payload(),
            "turbo_presets": turbo_preset_payload(self.repo_root),
            "defaults": {
                "workflow": SINGLE_RENDER_WORKFLOW,
                "model_id": DEFAULT_MODEL_ID,
                "output_root": str(DEFAULT_OUTPUT_ROOT.relative_to(self.repo_root)),
                "output_name": DEFAULT_OUTPUT_NAME,
                "duration_seconds": DEFAULT_DURATION_SECONDS,
                "steps": DEFAULT_INFERENCE_STEPS,
                "seed": DEFAULT_SEED,
                "resolution_id": DEFAULT_RESOLUTION_ID,
                "width": 128,
                "height": 128,
                "lora_enabled": False,
                "lora_path": "",
                "lora_scale": 1.0,
                "additional_loras": [],
                "turbo_enabled": False,
                "turbo_steps": DEFAULT_TURBO_PRESET_NFE,
                "turbo_preset_id": DEFAULT_TURBO_PRESET_ID,
                "text_encoder_id": CANONICAL_ENCODER_ID,
                "conditioning_artifact_path": "",
            },
            "storyboard_contract": {
                "workflow": FL2V_STORYBOARD_WORKFLOW,
                "mode": FIRST_LAST,
                "minimum_cards": 2,
                "reordering": "append-order-only",
                "parallel": False,
                "child_process_isolation": True,
            },
            "resolution_contract": {
                "min_dimension": MIN_RESOLUTION,
                "max_dimension": MAX_RESOLUTION,
                "step": RESOLUTION_STEP,
                "positive": True,
                "source": "tools/render_lab/resolutions.py:independent-dimensions-v1",
            },
            "runtime": {
                "default_model_id": DEFAULT_MODEL_ID,
                "models": {
                    item["id"]: {
                        "available": item["available"],
                        "transformer_mode": item["transformer_mode"],
                    }
                    for item in models
                },
                "checkpoint_root": str(checkpoint),
                "transformer_path": str(transformer) if transformer else None,
                "transformer_name": transformer.name if transformer else None,
                "transformer_mode": _transformer_mode(transformer),
                "transformer_required_mode": CANONICAL_TRANSFORMER_MODE,
                "checkpoint_exists": checkpoint.is_dir() and (checkpoint / "model_index.json").is_file(),
                "transformer_exists": default_model_status["available"],
                "generator": str(GENERATOR),
                "host_asset_manifest": {
                    "status": HOST_ASSET_MANIFEST_STATUS,
                    "note": HOST_ASSET_MANIFEST_NOTE,
                },
            },
            "geometry_contract": {
                "spatial_compression_ratio": H3_SPATIAL_COMPRESSION_RATIO,
                "dit_patch_size": list(H3_DIT_PATCH_SIZE),
                "source": "minimax_h3_mlx/video_vae.py + config.py H3 defaults",
            },
        }

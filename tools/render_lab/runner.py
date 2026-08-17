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
from .turbo_presets import (
    HOST_ASSET_MANIFEST_NOTE,
    HOST_ASSET_MANIFEST_STATUS,
    REFERENCE_TURBO_PRESET_ID,
    TurboPreset,
    turbo_preset_by_id,
    turbo_preset_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate.py"
HERETIC_ENCODER_SCRIPT = REPO_ROOT / "tools" / "render_lab" / "heretic_encoder.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "out" / "render-lab"
DEFAULT_OUTPUT_NAME = "render.mp4"
CANONICAL_TRANSFORMER_NAME = "minimax-h3-mlx-6bit-streamed-adaln"
CANONICAL_TRANSFORMER_MODE = "streamed-adaln-q6"
FORBIDDEN_TRANSFORMER_NAME = "minimax-h3-mlx-6bit"

T2V = "T2V"
I2V = "I2V"
FIRST_LAST = "FIRST_LAST"
MODE_LABELS = {
    T2V: "T2V — prompt only",
    I2V: "I2V — one image, first-frame anchor",
    FIRST_LAST: "First + last frame — two images",
}

MIN_DURATION_SECONDS = 5.0
MAX_DURATION_SECONDS = 15.0
MIN_INFERENCE_STEPS = 2
MAX_INFERENCE_STEPS = 40
DEFAULT_DURATION_SECONDS = 5.0
DEFAULT_INFERENCE_STEPS = 16
DEFAULT_SEED = 0
DEFAULT_RESOLUTION_ID = "canonical-128-square-v05d"

# Display-only geometry derived from the current H3 VAE/DiT source contract.  Resolution
# admission itself is delegated to ``resolve_canvas_size`` through resolutions.py.
H3_SPATIAL_COMPRESSION_RATIO = 16
H3_DIT_PATCH_SIZE = (1, 2, 2)

RUN_SCHEMA_VERSION = 3
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
    transformer_path: str | Path | None = field(default_factory=lambda: default_transformer_path())
    width: int | str | None = None
    height: int | str | None = None
    lora_enabled: bool = False
    lora_path: str | Path | None = None
    lora_scale: float = 1.0
    turbo_enabled: bool = False
    turbo_steps: int | str | None = None
    turbo_preset_id: str | None = None
    text_encoder_id: str = CANONICAL_ENCODER_ID

    def normalized(self) -> "RenderRequest":
        return replace(
            self,
            mode=str(self.mode).upper(),
            prompt=str(self.prompt),
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
    heretic_assets: Any | None = None


@dataclass(frozen=True)
class RunNamespace:
    run_id: str
    run_dir: Path
    output_path: Path
    created_at: str

    @property
    def config_path(self) -> Path:
        return self.run_dir / "render-config.json"

    @property
    def benchmark_path(self) -> Path:
        return self.run_dir / "benchmark.json"

    @property
    def status_path(self) -> Path:
        return self.run_dir / "run-status.json"

    @property
    def stdout_path(self) -> Path:
        return self.run_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.run_dir / "stderr.log"

    @property
    def telemetry_dir(self) -> Path:
        return self.run_dir / "telemetry"

    @property
    def conditioning_artifact_path(self) -> Path:
        return self.run_dir / "conditioning-artifact.npz"

    @property
    def encoder_evidence_path(self) -> Path:
        return self.run_dir / "heretic-encoder-evidence.json"

    @property
    def encoder_release_path(self) -> Path:
        return self.run_dir / "heretic-release-evidence.json"


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


def default_transformer_path(repo_root: Path = REPO_ROOT) -> Path | None:
    env_value = os.environ.get("H3_TRANSFORMER")
    candidates = [
        Path(env_value).expanduser()
        if env_value
        else repo_root.parent / "models" / CANONICAL_TRANSFORMER_NAME,
    ]
    return _first_existing(candidates) or candidates[0]


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
) -> tuple[bool, Path | None, float, bool, int | None]:
    lora_enabled = _coerce_bool(request.lora_enabled, "LoRA enabled")
    turbo_enabled = _coerce_bool(request.turbo_enabled, "Turbo enabled")
    if not lora_enabled:
        if turbo_enabled:
            raise RenderValidationError("Turbo mode requires LoRA to be enabled with an adapter path")
        return False, None, 1.0, False, None

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
    if turbo_enabled:
        if turbo_steps is not None and not MIN_INFERENCE_STEPS <= turbo_steps <= MAX_INFERENCE_STEPS:
            raise RenderValidationError(
                f"Turbo steps must be between {MIN_INFERENCE_STEPS} and {MAX_INFERENCE_STEPS}"
            )
        if turbo_steps is not None and turbo_steps != request.steps:
            raise RenderValidationError(
                "Turbo mode requires ordinary inference steps and Turbo steps to agree"
            )
    return True, lora_path, lora_scale, turbo_enabled, turbo_steps


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
    if transformer_path is not None and transformer_path.name == CANONICAL_TRANSFORMER_NAME:
        return CANONICAL_TRANSFORMER_MODE
    return "unspecified-or-custom"


def _validate_transformer_safety(
    transformer_path: Path | None,
    *,
    check_runtime_paths: bool,
) -> None:
    if transformer_path is None:
        if check_runtime_paths:
            raise RenderValidationError(
                f"Render Lab requires the canonical {CANONICAL_TRANSFORMER_NAME} transformer"
            )
        return
    resolved = transformer_path.expanduser().resolve(strict=False)
    if Path("/Volumes/models") == resolved or Path("/Volumes/models") in resolved.parents:
        raise RenderValidationError(
            "Render Lab rejects stale /Volumes/models transformer paths; configure the local streamed-AdaLN Q6 asset"
        )
    if transformer_path.name == FORBIDDEN_TRANSFORMER_NAME:
        raise RenderValidationError(
            f"Render Lab rejects the non-streamed {FORBIDDEN_TRANSFORMER_NAME} transformer; "
            f"use {CANONICAL_TRANSFORMER_NAME}"
        )
    if check_runtime_paths and transformer_path.name != CANONICAL_TRANSFORMER_NAME:
        raise RenderValidationError(
            f"Render Lab requires the canonical {CANONICAL_TRANSFORMER_NAME} transformer; "
            f"got {transformer_path.name}"
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


def validate_render_request(
    request: RenderRequest,
    *,
    repo_root: Path = REPO_ROOT,
    check_runtime_paths: bool = False,
    check_images: bool = True,
    verify_runtime_geometry: bool = True,
) -> ValidatedRequest:
    request = request.normalized()
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
    if turbo_preset is None:
        lora_enabled, lora_path, lora_scale, turbo_enabled, turbo_steps = _validate_lora_controls(
            request,
            check_runtime_paths=check_runtime_paths,
        )
    else:
        lora_enabled = True
        lora_path = turbo_asset_path
        lora_scale = turbo_preset.default_scale
        turbo_enabled = True
        turbo_steps = turbo_preset.nfe
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
    transformer_path = (
        None if request.transformer_path is None else Path(request.transformer_path).expanduser().resolve(strict=False)
    )
    _validate_transformer_safety(transformer_path, check_runtime_paths=check_runtime_paths)
    if check_runtime_paths:
        if not checkpoint_root.is_dir() or not (checkpoint_root / "model_index.json").is_file():
            raise RenderValidationError(
                f"Checkpoint root is not a usable H3 release directory: {checkpoint_root}"
            )
        effective_transformer = transformer_path
        assert effective_transformer is not None
        if not effective_transformer.is_dir() or not (effective_transformer / "config.json").is_file():
            raise RenderValidationError(
                f"Transformer path is not a usable H3 transformer directory: {effective_transformer}"
            )
        transformer_path = effective_transformer
    normalized_request = replace(
        request,
        resolution_id=preset.preset_id if explicit_dimensions else request.resolution_id,
        width=runtime_width,
        height=runtime_height,
        lora_enabled=lora_enabled,
        lora_path=lora_path,
        lora_scale=lora_scale,
        turbo_enabled=turbo_enabled,
        turbo_steps=turbo_steps,
        turbo_preset_id=turbo_preset.preset_id if turbo_preset is not None else None,
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
        heretic_assets=heretic_assets,
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
    command = [
        str(python),
        "-u",
        str(GENERATOR),
    ]
    if request.text_encoder_id != HERETIC_ENCODER_ID:
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
    if validated.transformer_path is not None:
        command.extend(["--transformer", str(validated.transformer_path)])
    for image_path, anchor in zip(validated.image_paths, validated.anchors):
        command.extend(["--image", str(image_path), "--anchor", anchor])
    if validated.turbo_preset is not None:
        preset = validated.turbo_preset
        command.extend([
            preset.adapter_flag,
            str(request.lora_path),
            "--lora-scale",
            _format_float(preset.default_scale),
        ])
        if preset.runtime_variant is not None:
            command.extend(["--lightx-variant", preset.runtime_variant])
        command.extend(["--turbo-steps", str(preset.nfe)])
    elif request.lora_enabled:
        command.extend([
            "--lora",
            str(request.lora_path),
            "--lora-scale",
            _format_float(request.lora_scale),
        ])
        if request.turbo_enabled:
            command.append("--turbo")
            if request.turbo_steps is not None:
                command.extend(["--turbo-steps", str(request.turbo_steps)])
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


def _turbo_evidence(validated: ValidatedRequest) -> dict[str, Any]:
    request = validated.request
    preset = validated.turbo_preset
    if preset is not None:
        return {
            "selected": True,
            "preset_id": preset.preset_id,
            "label": preset.label,
            "role": preset.role,
            "family": preset.family,
            "logical_asset": preset.logical_asset,
            "adapter_asset": {
                "flag": preset.adapter_flag,
                "path": str(request.lora_path),
                "logical_asset": preset.logical_asset,
            },
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
    return {
        "selected": False,
        "preset_id": None,
        "label": "None / Reference",
        "role": "Reference",
        "family": None,
        "logical_asset": None,
        "adapter_asset": {
            "flag": "--lora" if request.lora_enabled else None,
            "path": str(request.lora_path) if request.lora_enabled and request.lora_path else None,
            "logical_asset": None,
        },
        "effective_nfe": effective_nfe,
        "runtime_variant": None,
        "runtime_contract": "existing non-Turbo / manual LoRA behavior",
        "effective_scale": request.lora_scale if request.lora_enabled else None,
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


def _text_encoder_evidence(validated: ValidatedRequest, namespace: RunNamespace) -> dict[str, Any]:
    request = validated.request
    if request.text_encoder_id == CANONICAL_ENCODER_ID:
        return {
            "id": CANONICAL_ENCODER_ID,
            "label": "Canonical Qwen3-VL",
            "experimental": False,
            "mode_contract": "T2V, I2V, FIRST_LAST",
            "conditioning_artifact": None,
            "h3_launch_after_encoder_exit": None,
        }
    assets = validated.heretic_assets or probe_heretic_assets(REPO_ROOT)
    return {
        "id": HERETIC_ENCODER_ID,
        "label": "Heretic 35B-A3B · Experimental",
        "experimental": True,
        "mode_contract": "T2V only",
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
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_identifier": namespace.run_id,
        "timestamp": namespace.created_at,
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
        "turbo_enabled": request.turbo_enabled,
        "turbo_steps": request.turbo_steps,
        "turbo_preset_id": request.turbo_preset_id,
        "lora": {
            "enabled": request.lora_enabled,
            "path": str(request.lora_path) if request.lora_enabled and request.lora_path else None,
            "scale": float(request.lora_scale) if request.lora_enabled else None,
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
        "text_encoder_id": request.text_encoder_id,
        "text_encoder": _text_encoder_evidence(validated, namespace),
        "conditioning_artifact_path": (
            str(namespace.conditioning_artifact_path)
            if request.text_encoder_id == HERETIC_ENCODER_ID
            else None
        ),
        "encoder_command": list(encoder_command) if encoder_command is not None else None,
        "git": _git_identity(repo_root),
        "runtime_identity": runtime_identity(validated, repo_root),
        "command": list(command),
    }


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
    return {
        "run_id": namespace.run_id,
        "run_directory": str(namespace.run_dir),
        "output_root": config.get("output_root"),
        "status": status.get("status", "unknown"),
        "status_record": status,
        "benchmark": benchmark,
        "stdout": _tail_text(namespace.stdout_path),
        "stderr": _tail_text(namespace.stderr_path),
    }


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
        artifact = benchmark.get("output_artifact") or {}
        artifact_path = artifact.get("path")
        rows.append({
            "run_id": run_dir.name,
            "timestamp": config.get("timestamp"),
            "mode": config.get("generation_mode"),
            "resolution": f"{config.get('width')} × {config.get('height')}",
            "width": config.get("width"),
            "height": config.get("height"),
            "steps": config.get("inference_steps_requested"),
            "seed": config.get("seed"),
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
        with self._state_lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RenderBusyError("A render is already active")
            request = request.normalized()
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
                            "turbo_enabled": validated.request.turbo_enabled,
                            "turbo_steps": validated.request.turbo_steps,
                            "turbo_preset_id": validated.request.turbo_preset_id,
                            "turbo": {
                                **_turbo_evidence(validated),
                                "enabled": validated.request.turbo_enabled,
                                "steps": validated.request.turbo_steps,
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
                            "text_encoder_id": request.text_encoder_id,
                            "text_encoder": _text_encoder_evidence(validated, namespace),
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
        return {
            "modes": [{"id": mode, "label": MODE_LABELS[mode], "image_count": expected_image_count(mode)} for mode in (T2V, I2V, FIRST_LAST)],
            "text_encoders": text_encoder_payload(self.repo_root),
            "resolutions": preset_payload(),
            "turbo_presets": turbo_preset_payload(self.repo_root),
            "defaults": {
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
                "turbo_enabled": False,
                "turbo_steps": 8,
                "turbo_preset_id": REFERENCE_TURBO_PRESET_ID,
                "text_encoder_id": CANONICAL_ENCODER_ID,
            },
            "resolution_contract": {
                "min_dimension": MIN_RESOLUTION,
                "max_dimension": MAX_RESOLUTION,
                "step": RESOLUTION_STEP,
                "positive": True,
                "source": "tools/render_lab/resolutions.py:independent-dimensions-v1",
            },
            "runtime": {
                "checkpoint_root": str(checkpoint),
                "transformer_path": str(transformer) if transformer else None,
                "transformer_name": transformer.name if transformer else None,
                "transformer_mode": _transformer_mode(transformer),
                "transformer_required_mode": CANONICAL_TRANSFORMER_MODE,
                "checkpoint_exists": checkpoint.is_dir() and (checkpoint / "model_index.json").is_file(),
                "transformer_exists": bool(
                    transformer and transformer.is_dir() and (transformer / "config.json").is_file()
                ),
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

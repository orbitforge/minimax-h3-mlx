"""MLX-free text-encoder catalog and Heretic asset admission gates.

The Render Lab owns the user-facing choice here, while the actual experimental
encoder lives in a separate child process.  This module deliberately performs
only filesystem, JSON, SHA-256, and NumPy bridge checks; it never imports MLX,
Qwen, or the H3 pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CANONICAL_ENCODER_ID = "canonical-qwen3-vl"
HERETIC_ENCODER_ID = "heretic-35b-a3b-state28"
HERETIC_ENCODER_LABEL = "Heretic 35B-A3B · Experimental"
HERETIC_ENCODER_HINT = "Experimental text-only encoder using state 28 + learned H3 conditioning bridge."
HERETIC_IMAGE_MODE_MESSAGE = (
    "Heretic is currently text-only; image-conditioned modes require Canonical Qwen3-VL."
)
HERETIC_MODEL_ENV = "H3_HERETIC_MODEL"
HERETIC_BRIDGE_ENV = "H3_HERETIC_BRIDGE"
DEFAULT_HERETIC_MODEL = Path(
    "/Users/elbancol/AI/MLX-Models/Jundot/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit"
)
HERETIC_BRIDGE_FILENAME = "qwen3.6-35b-a3b-heretic-state28-bridge.npz"
HERETIC_BRIDGE_SHA256 = "8dc5dabb7da0d69dfe7ec0d5d80f684a50768d500b46bf70c03cec557141068e"
HERETIC_SOURCE_WIDTH = 2048
HERETIC_TARGET_WIDTH = 5120
HERETIC_STATE = 28
HERETIC_FULL_LAYERS = 40
HERETIC_BRIDGE_KEYS = frozenset({"input_mean", "input_scale", "target_mean", "weights"})


class EncoderAssetError(ValueError):
    """A selected encoder cannot be admitted for this Render Lab run."""


@dataclass(frozen=True)
class HereticAssets:
    model_path: Path
    bridge_path: Path
    available: bool
    reason: str | None
    model_config_sha256: str | None = None
    bridge_sha256: str | None = None
    bridge_shapes: Mapping[str, tuple[int, ...]] | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_volatile_path(path: Path) -> bool:
    resolved = path.expanduser().resolve(strict=False)
    return resolved == Path("/tmp") or Path("/private/tmp") in resolved.parents or Path("/tmp") in resolved.parents


def heretic_model_path(repo_root: Path) -> Path:
    value = os.environ.get(HERETIC_MODEL_ENV)
    return (Path(value).expanduser() if value else DEFAULT_HERETIC_MODEL).resolve(strict=False)


def heretic_bridge_path(repo_root: Path) -> Path:
    value = os.environ.get(HERETIC_BRIDGE_ENV)
    if value:
        return Path(value).expanduser().resolve(strict=False)
    return (repo_root.resolve().parent / "models" / HERETIC_BRIDGE_FILENAME).resolve(strict=False)


def _model_identity(path: Path) -> tuple[dict[str, Any] | None, str | None, str | None]:
    config_path = path / "config.json"
    if not path.is_dir():
        return None, None, f"Heretic source model is unavailable: {path}"
    if not config_path.is_file() or not os.access(config_path, os.R_OK):
        return None, None, f"Heretic source model config is unavailable: {config_path}"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, None, f"Heretic source model config is unreadable: {config_path} ({exc})"
    text_config = config.get("text_config", {}) if isinstance(config, Mapping) else {}
    model_type = config.get("model_type") if isinstance(config, Mapping) else None
    hidden_size = text_config.get("hidden_size") if isinstance(text_config, Mapping) else None
    layer_count = text_config.get("num_hidden_layers") if isinstance(text_config, Mapping) else None
    if model_type != "qwen3_5_moe":
        return None, None, f"Heretic source model is not Qwen3.5 MoE: model_type={model_type!r}"
    try:
        parsed_hidden_size = int(hidden_size or -1)
        parsed_layer_count = int(layer_count or -1)
    except (TypeError, ValueError):
        return None, None, "Heretic source model text geometry is not numeric"
    if parsed_hidden_size != HERETIC_SOURCE_WIDTH or parsed_layer_count != HERETIC_FULL_LAYERS:
        return None, None, (
            "Heretic source model has incompatible text geometry: "
            f"hidden_size={hidden_size!r}, layers={layer_count!r}; "
            f"expected {HERETIC_SOURCE_WIDTH}/{HERETIC_FULL_LAYERS}"
        )
    return config, _sha256_file(config_path), None


def _bridge_identity(path: Path) -> tuple[str | None, dict[str, tuple[int, ...]] | None, str | None]:
    if _is_volatile_path(path):
        return None, None, f"Heretic bridge must not use volatile /tmp storage: {path}"
    if not path.is_file() or not os.access(path, os.R_OK):
        return None, None, f"Heretic state-28 bridge is unavailable: {path}"
    digest = _sha256_file(path)
    if digest != HERETIC_BRIDGE_SHA256:
        return digest, None, (
            "Heretic state-28 bridge SHA-256 mismatch: "
            f"got {digest}, expected {HERETIC_BRIDGE_SHA256}"
        )
    try:
        with np.load(path, allow_pickle=False) as loaded:
            keys = frozenset(loaded.files)
            if keys != HERETIC_BRIDGE_KEYS:
                return digest, None, f"Heretic bridge key set mismatch: {sorted(keys)}"
            expected = {
                "input_mean": (HERETIC_SOURCE_WIDTH,),
                "input_scale": (HERETIC_SOURCE_WIDTH,),
                "target_mean": (HERETIC_TARGET_WIDTH,),
                "weights": (HERETIC_SOURCE_WIDTH, HERETIC_TARGET_WIDTH),
            }
            shapes: dict[str, tuple[int, ...]] = {}
            for name, shape in expected.items():
                array = loaded[name]
                shapes[name] = tuple(int(item) for item in array.shape)
                if shapes[name] != shape or str(array.dtype) != "float32":
                    return digest, shapes, (
                        f"Heretic bridge {name} has shape/dtype {shapes[name]}/{array.dtype}; "
                        f"expected {shape}/float32"
                    )
    except Exception as exc:
        return digest, None, f"Heretic state-28 bridge is unreadable: {path} ({exc})"
    return digest, shapes, None


def probe_heretic_assets(repo_root: Path) -> HereticAssets:
    model_path = heretic_model_path(repo_root)
    bridge_path = heretic_bridge_path(repo_root)
    model_config, model_digest, model_error = _model_identity(model_path)
    if model_error:
        return HereticAssets(model_path, bridge_path, False, model_error)
    bridge_digest, bridge_shapes, bridge_error = _bridge_identity(bridge_path)
    if bridge_error:
        return HereticAssets(
            model_path,
            bridge_path,
            False,
            bridge_error,
            model_config_sha256=model_digest,
            bridge_sha256=bridge_digest,
            bridge_shapes=bridge_shapes,
        )
    assert model_config is not None
    return HereticAssets(
        model_path,
        bridge_path,
        True,
        None,
        model_config_sha256=model_digest,
        bridge_sha256=bridge_digest,
        bridge_shapes=bridge_shapes,
    )


def validate_text_encoder_selection(
    encoder_id: str,
    mode: str,
    *,
    repo_root: Path,
    check_runtime_paths: bool,
) -> HereticAssets | None:
    normalized = str(encoder_id or CANONICAL_ENCODER_ID).strip()
    if normalized == CANONICAL_ENCODER_ID:
        return None
    if normalized != HERETIC_ENCODER_ID:
        raise EncoderAssetError(f"Unknown text encoder choice: {normalized!r}")
    if str(mode).upper() != "T2V":
        raise EncoderAssetError(HERETIC_IMAGE_MODE_MESSAGE)
    assets = probe_heretic_assets(repo_root)
    if check_runtime_paths and not assets.available:
        raise EncoderAssetError(assets.reason or "Heretic text encoder assets are unavailable")
    return assets


def text_encoder_payload(repo_root: Path) -> list[dict[str, Any]]:
    assets = probe_heretic_assets(repo_root)
    return [
        {
            "id": CANONICAL_ENCODER_ID,
            "label": "Canonical Qwen3-VL",
            "default": True,
            "experimental": False,
            "available": True,
            "allowed_modes": ["T2V", "I2V", "FIRST_LAST"],
            "hint": "Existing H3 Qwen3-VL layer-50 conditioning path.",
        },
        {
            "id": HERETIC_ENCODER_ID,
            "label": HERETIC_ENCODER_LABEL,
            "default": False,
            "experimental": True,
            "available": assets.available,
            "allowed_modes": ["T2V"],
            "hint": HERETIC_ENCODER_HINT,
            "disabled_reason": assets.reason,
            "source_model_path": str(assets.model_path),
            "source_state": HERETIC_STATE,
            "source_width": HERETIC_SOURCE_WIDTH,
            "target_width": HERETIC_TARGET_WIDTH,
            "bridge_path": str(assets.bridge_path),
            "bridge_sha256": HERETIC_BRIDGE_SHA256,
        },
    ]


__all__ = [
    "CANONICAL_ENCODER_ID",
    "DEFAULT_HERETIC_MODEL",
    "EncoderAssetError",
    "HERETIC_BRIDGE_FILENAME",
    "HERETIC_BRIDGE_SHA256",
    "HERETIC_ENCODER_HINT",
    "HERETIC_ENCODER_ID",
    "HERETIC_ENCODER_LABEL",
    "HERETIC_FULL_LAYERS",
    "HERETIC_IMAGE_MODE_MESSAGE",
    "HERETIC_SOURCE_WIDTH",
    "HERETIC_STATE",
    "HERETIC_TARGET_WIDTH",
    "HereticAssets",
    "heretic_bridge_path",
    "heretic_model_path",
    "probe_heretic_assets",
    "text_encoder_payload",
    "validate_text_encoder_selection",
]

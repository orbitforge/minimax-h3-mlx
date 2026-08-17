"""The bounded production Turbo catalog for the local Render Lab.

This is a catalog, not a second adapter runtime.  The adapter flag and LightX variant names are
the argument contracts already owned by ``scripts/generate.py`` and the validated H3 runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minimax_h3_mlx.lora import LIGHTX_VARIANTS


REFERENCE_TURBO_PRESET_ID = "none"
REFERENCE_TURBO_LABEL = "None / Reference"

# No repository-owned host asset manifest is present in this checkout.  Keep this explicit and
# narrow instead of introducing an asset-management subsystem for five known local files.
HOST_ASSET_MANIFEST_STATUS = "missing"
HOST_ASSET_MANIFEST_NOTE = (
    "No repository-owned canonical host asset manifest is present; curated assets resolve under "
    "the checkout's sibling work/models directory."
)


@dataclass(frozen=True)
class TurboPreset:
    """One validated production adapter configuration exposed by the operator surface."""

    preset_id: str
    label: str
    role: str
    family: str
    logical_asset: str
    nfe: int
    adapter_flag: str
    asset_relative_path: str
    runtime_variant: str | None
    video_shift: float
    audio_shift: float
    runtime_contract: str
    default_scale: float = 1.0
    recommended_width: int | None = None
    recommended_height: int | None = None

    def __post_init__(self) -> None:
        if not self.preset_id or not self.label or not self.role or not self.family:
            raise ValueError("Turbo preset identity fields must be non-empty")
        if self.adapter_flag not in {"--turbo-lora", "--lightx-lora"}:
            raise ValueError(f"unsupported Turbo adapter flag {self.adapter_flag!r}")
        if isinstance(self.nfe, bool) or not isinstance(self.nfe, int) or self.nfe < 2:
            raise ValueError(f"Turbo preset NFE must be an integer at least 2, got {self.nfe!r}")
        if not self.asset_relative_path or Path(self.asset_relative_path).is_absolute():
            raise ValueError("Turbo preset asset path must be a non-empty relative path")
        if self.adapter_flag == "--lightx-lora":
            if self.runtime_variant is None or self.runtime_variant not in LIGHTX_VARIANTS:
                raise ValueError("LightX Turbo presets require a validated runtime variant")
            manifest = LIGHTX_VARIANTS[self.runtime_variant]
            if manifest.family != self.family or manifest.nfe != self.nfe:
                raise ValueError("LightX Turbo preset metadata does not match its runtime manifest")
            if manifest.video_shift != self.video_shift or manifest.audio_shift != self.audio_shift:
                raise ValueError("LightX Turbo preset scheduler metadata does not match its manifest")
            if manifest.runtime_scale_default != self.default_scale:
                raise ValueError("LightX Turbo preset scale does not match its runtime manifest")
        elif self.runtime_variant is not None:
            raise ValueError("generic Larry/Turbo presets do not accept a LightX runtime variant")
        if (self.recommended_width is None) != (self.recommended_height is None):
            raise ValueError("recommended Turbo geometry must provide both width and height")

    def resolve_asset_path(self, repo_root: str | Path) -> Path:
        """Resolve the known asset beneath the checkout's sibling ``work/models`` root."""
        root = Path(repo_root).expanduser().resolve()
        return (root.parent / "models" / self.asset_relative_path).resolve(strict=False)

    @property
    def recommended_geometry(self) -> str | None:
        if self.recommended_width is None:
            return None
        return f"{self.recommended_width} × {self.recommended_height} native/recommended"

    @property
    def summary(self) -> str:
        return f"{self.role} · {self.nfe} NFE · {self.family}"

    def payload(self, repo_root: str | Path) -> dict[str, Any]:
        asset_path = self.resolve_asset_path(repo_root)
        return {
            "id": self.preset_id,
            "label": self.label,
            "role": self.role,
            "family": self.family,
            "logical_asset": self.logical_asset,
            "nfe": self.nfe,
            "adapter_flag": self.adapter_flag,
            "adapter_asset_path": str(asset_path),
            "asset_exists": asset_path.is_file(),
            "runtime_variant": self.runtime_variant,
            "runtime_contract": self.runtime_contract,
            "default_scale": self.default_scale,
            "scheduler": {
                "video_shift": self.video_shift,
                "audio_shift": self.audio_shift,
                "source": "LightX2V manifest" if self.family == "LightX2V" else "H3 PipelineConfig defaults",
            },
            "recommended_geometry": self.recommended_geometry,
            "asset_manifest_status": HOST_ASSET_MANIFEST_STATUS,
            "asset_manifest_note": HOST_ASSET_MANIFEST_NOTE,
            "summary": self.summary,
        }


def _lightx(
    *,
    preset_id: str,
    label: str,
    role: str,
    logical_asset: str,
    asset_relative_path: str,
    runtime_variant: str,
    recommended_width: int | None = None,
    recommended_height: int | None = None,
) -> TurboPreset:
    manifest = LIGHTX_VARIANTS[runtime_variant]
    return TurboPreset(
        preset_id=preset_id,
        label=label,
        role=role,
        family=manifest.family,
        logical_asset=logical_asset,
        nfe=manifest.nfe,
        adapter_flag="--lightx-lora",
        asset_relative_path=asset_relative_path,
        runtime_variant=runtime_variant,
        video_shift=manifest.video_shift,
        audio_shift=manifest.audio_shift,
        runtime_contract="native LightX2V manifest-bound split-Q/K/V",
        default_scale=manifest.runtime_scale_default,
        recommended_width=recommended_width,
        recommended_height=recommended_height,
    )


TURBO_PRESETS: tuple[TurboPreset, ...] = (
    _lightx(
        preset_id="lightx-4step-v01",
        label="LightX 4-Step v0.1",
        role="Fast",
        logical_asset="lightx_v01_4step",
        asset_relative_path=(
            "minimax-h3-turbo/lightx2v/Minimax-h3-Turbo/"
            "minimax_h3_fl2v_turbo_4step_v0.1.safetensors"
        ),
        runtime_variant="fl2va-turbo-4step-v0.1",
    ),
    _lightx(
        preset_id="lightx-8step-v10",
        label="LightX 8-Step v1.0",
        role="Quality / General",
        logical_asset="lightx_v10_8step",
        asset_relative_path=(
            "minimax-h3-turbo/lightx2v/Minimax-h3-Turbo/"
            "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
        ),
        runtime_variant="fl2va-turbo-8step-v1.0",
    ),
    _lightx(
        preset_id="lightx-4step-v10-768p",
        label="LightX 4-Step v1.0 768p",
        role="High Resolution",
        logical_asset="lightx_v10_768p",
        asset_relative_path=(
            "minimax-h3-turbo/lightx2v/Minimax-h3-Turbo/"
            "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
        ),
        runtime_variant="fl2va-turbo-4step-v1.0-768p",
        recommended_width=1344,
        recommended_height=768,
    ),
    TurboPreset(
        preset_id="larry-v4-step600",
        label="Larry v4 Step 600",
        role="Quality / General",
        family="Larry",
        logical_asset="larry_v4",
        nfe=8,
        adapter_flag="--turbo-lora",
        asset_relative_path="minimax-h3-loras/larryvrh/minimax_h3_turbo_v4_step600_ema.safetensors",
        runtime_variant=None,
        video_shift=12.0,
        audio_shift=3.0,
        runtime_contract="existing generic Turbo LoRA",
    ),
    TurboPreset(
        preset_id="larry-850",
        label="Larry 850",
        role="Fast Motion",
        family="Larry",
        logical_asset="larry_850",
        nfe=4,
        adapter_flag="--turbo-lora",
        asset_relative_path="minimax-h3-loras/larryvrh/minimax_h3_turbo_4step_ema_ckpt850.safetensors",
        runtime_variant=None,
        video_shift=12.0,
        audio_shift=3.0,
        runtime_contract="existing generic Turbo LoRA",
    ),
)


def turbo_preset_by_id(preset_id: str | None) -> TurboPreset | None:
    normalized = "" if preset_id is None else str(preset_id).strip()
    if not normalized or normalized == REFERENCE_TURBO_PRESET_ID:
        return None
    for preset in TURBO_PRESETS:
        if preset.preset_id == normalized:
            return preset
    choices = ", ".join(preset.preset_id for preset in TURBO_PRESETS)
    raise ValueError(f"Unknown Turbo preset {preset_id!r}; choose one of: {choices}")


def turbo_preset_payload(repo_root: str | Path) -> list[dict[str, Any]]:
    """Serialize the reference option plus the exact five production presets."""
    reference = {
        "id": REFERENCE_TURBO_PRESET_ID,
        "label": REFERENCE_TURBO_LABEL,
        "role": "Reference",
        "family": None,
        "logical_asset": None,
        "nfe": None,
        "adapter_flag": None,
        "adapter_asset_path": None,
        "asset_exists": None,
        "runtime_variant": None,
        "runtime_contract": "existing non-Turbo / manual LoRA behavior",
        "default_scale": 1.0,
        "scheduler": {"video_shift": 12.0, "audio_shift": 3.0, "source": "H3 PipelineConfig defaults"},
        "recommended_geometry": None,
        "asset_manifest_status": HOST_ASSET_MANIFEST_STATUS,
        "asset_manifest_note": HOST_ASSET_MANIFEST_NOTE,
        "summary": "Existing normal Render Lab behavior",
    }
    return [reference, *(preset.payload(repo_root) for preset in TURBO_PRESETS)]

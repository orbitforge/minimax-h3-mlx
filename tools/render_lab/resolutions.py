"""The Render Lab's single resolution preset and validation source.

The runtime accepts explicit canvas dimensions, but the operator surface deliberately exposes a
small approved set.  Each preset is expressed in the same aspect/megapixel form accepted by the
authoritative ``minimax_h3_mlx.packing.resolve_canvas_size`` helper.  The curated catalog
dimensions are recorded here so the UI can render without importing MLX; render admission
re-resolves the preset through the runtime helper before launching the child process.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from math import ceil
import sys
import types
from pathlib import Path
from typing import Callable


RUNTIME_RESOLUTION_RULE_ID = "minimax_h3_mlx.packing.resolve_canvas_size:v1"
PROJECT_RESOLUTION_SOURCE_ID = "render-lab-approved-resolutions:v2"

# The independent-dimension surface is bounded by the smallest and largest canvases already
# admitted by the Render Lab catalog.  The runtime itself requires 32-pixel alignment.
RESOLUTION_STEP = 32
MIN_RESOLUTION = 128
MAX_RESOLUTION = 1344
INDEPENDENT_DIMENSION_SOURCE_ID = "render-lab-independent-dimensions:v1"
INDEPENDENT_DIMENSION_RULE_ID = "render-lab.independent-dimensions:v1"


@dataclass(frozen=True)
class ResolutionPreset:
    preset_id: str
    label: str
    aspect_width: int
    aspect_height: int
    megapixels: float | None
    expected_height: int
    expected_width: int
    evidence_class: str
    evidence_reference: str
    approval_note: str
    project_approved: bool = True
    runtime_height: int | None = None
    runtime_width: int | None = None

    @property
    def dimensions(self) -> tuple[int, int]:
        """Return H3 launch dimensions as ``(height, width)`` in the runtime convention."""
        return self.runtime_dimensions

    @property
    def catalog_dimensions(self) -> tuple[int, int]:
        """Return the requested catalog dimensions as ``(height, width)``."""
        return self.expected_height, self.expected_width

    @property
    def runtime_dimensions(self) -> tuple[int, int]:
        """Return the H3-aligned launch dimensions as ``(height, width)``.

        A few common display presets (for example 640x360) are not multiples of the H3
        runtime's 32-pixel canvas rule.  They remain curated catalog entries, while their
        deterministic aligned launch canvas is recorded explicitly instead of admitting a
        freeform size.
        """
        if (self.runtime_height is None) != (self.runtime_width is None):
            raise ValueError(f"Resolution preset {self.preset_id!r} has incomplete runtime dimensions")
        return (
            self.expected_height if self.runtime_height is None else self.runtime_height,
            self.expected_width if self.runtime_width is None else self.runtime_width,
        )

    @property
    def orientation(self) -> str:
        if self.expected_width == self.expected_height:
            return "square"
        return "landscape" if self.expected_width > self.expected_height else "portrait"


def _curated_preset(width: int, height: int) -> ResolutionPreset:
    """Build one approved catalog entry from a fixed display dimension."""
    runtime_dimensions = None
    if width % 32 or height % 32:
        runtime_dimensions = (ceil(height / 32) * 32, ceil(width / 32) * 32)
    orientation = "square" if width == height else "landscape" if width > height else "portrait"
    return ResolutionPreset(
        preset_id=f"curated-{width}x{height}",
        label=f"{width} × {height} — curated {orientation}",
        aspect_width=width,
        aspect_height=height,
        megapixels=(width * height) / 1_000_000,
        expected_height=height,
        expected_width=width,
        evidence_class="curated-approved-preset",
        evidence_reference="tools/render_lab/resolutions.py:curated-catalog-v2",
        approval_note="Approved Render Lab catalog entry; launch geometry is checked against the H3 resolver.",
        runtime_height=None if runtime_dimensions is None else runtime_dimensions[0],
        runtime_width=None if runtime_dimensions is None else runtime_dimensions[1],
    )


# These are intentionally the only dimensions the UI can submit.  The catalog is grouped by
# display orientation so the flat selector remains easy to scan without introducing freeform
# width/height controls.
#
# * 128x128 is the preserved v0.5d canonical full-schedule MP4 proof.
# * 256x256 is the preserved v0.5e quality/resource MP4 proof.
# * 608x352 is the documented current CLI 0.2 MP default and a practical smoke target.
# * 1344x768 is the released 768-pixel-short-edge canvas with a successful real 5 s render.
RESOLUTION_PRESETS: tuple[ResolutionPreset, ...] = (
    ResolutionPreset(
        preset_id="canonical-128-square-v05d",
        label="128 × 128 — v0.5d canonical proof",
        aspect_width=1,
        aspect_height=1,
        megapixels=0.016384,
        expected_height=128,
        expected_width=128,
        evidence_class="successful-render-evidence",
        evidence_reference="README.md:v0.5d-derived-full-schedule-functional-proof",
        approval_note="Preserved successful MP4 functional proof; frozen canonical baseline geometry.",
    ),
    ResolutionPreset(
        preset_id="quality-256-square-v05e",
        label="256 × 256 — v0.5e quality proof",
        aspect_width=1,
        aspect_height=1,
        megapixels=0.065536,
        expected_height=256,
        expected_width=256,
        evidence_class="successful-render-evidence",
        evidence_reference="README.md:v0.5e-Slice-3B6B-milestone-closeout",
        approval_note="Preserved successful 256×256 MP4 quality/resource proof; not a timing baseline.",
    ),
    _curated_preset(320, 320),
    _curated_preset(384, 384),
    _curated_preset(448, 448),
    _curated_preset(512, 512),
    _curated_preset(384, 256),
    _curated_preset(448, 256),
    _curated_preset(512, 288),
    _curated_preset(512, 384),
    ResolutionPreset(
        preset_id="cli-default-02mp-16x9",
        label="608 × 352 — CLI default 0.2 MP",
        aspect_width=16,
        aspect_height=9,
        megapixels=0.2,
        expected_height=352,
        expected_width=608,
        evidence_class="documented-runtime-default",
        evidence_reference="README.md:344-346",
        approval_note="Documented current CLI default; runtime-valid smoke configuration, not a frozen proof.",
    ),
    _curated_preset(640, 360),
    _curated_preset(640, 384),
    _curated_preset(768, 432),
    _curated_preset(768, 512),
    _curated_preset(896, 512),
    _curated_preset(1024, 576),
    ResolutionPreset(
        preset_id="released-768-short-edge",
        label="1344 × 768 — released 768-pixel canvas",
        aspect_width=16,
        aspect_height=9,
        megapixels=None,
        expected_height=768,
        expected_width=1344,
        evidence_class="successful-render-evidence",
        evidence_reference="README.md:89-93",
        approval_note="Successful real 5 s full-pipeline render; highest-cost operator preset.",
    ),
    _curated_preset(256, 384),
    _curated_preset(256, 448),
    _curated_preset(288, 512),
    _curated_preset(384, 512),
    _curated_preset(360, 640),
    _curated_preset(384, 640),
    _curated_preset(432, 768),
    _curated_preset(512, 768),
    _curated_preset(512, 896),
    _curated_preset(576, 1024),
    _curated_preset(768, 1344),
)


def explicit_resolution_preset(width: int, height: int) -> ResolutionPreset:
    """Describe a validated explicit canvas while retaining the legacy preset contract."""
    return ResolutionPreset(
        preset_id=f"explicit-{width}x{height}",
        label=f"{width} × {height} — independent dimensions",
        aspect_width=width,
        aspect_height=height,
        megapixels=(width * height) / 1_000_000,
        expected_height=height,
        expected_width=width,
        evidence_class="explicit-independent-dimensions",
        evidence_reference=INDEPENDENT_DIMENSION_SOURCE_ID,
        approval_note="Explicit width and height passed the Render Lab dimension contract.",
        runtime_height=height,
        runtime_width=width,
    )


def preset_by_id(preset_id: str) -> ResolutionPreset:
    for preset in RESOLUTION_PRESETS:
        if preset.preset_id == preset_id:
            return preset
    choices = ", ".join(p.preset_id for p in RESOLUTION_PRESETS)
    raise ValueError(f"Unknown resolution preset {preset_id!r}; choose one of: {choices}")


def authoritative_dimensions(
    preset: ResolutionPreset,
    resolver: Callable[[float, float, float | None], tuple[int, int]] | None = None,
) -> tuple[int, int]:
    """Resolve a preset through the current runtime helper.

    Importing the runtime lazily keeps contract tests and the browser's initial page load MLX-free.
    The actual Render admission path calls this function before spawning H3.
    """
    if resolver is None:
        previous_mlx = None
        previous_mlx_core = None
        try:
            # ``packing.py`` is the authoritative source, but importing its normal module would
            # initialize MLX in a process that only needs geometry validation. Execute that exact
            # source file with a tiny annotation-only MLX stub; resolve_canvas_size itself uses
            # only Python math and NumPy. This keeps Render admission and contract tests MLX-free.
            packing_path = Path(__file__).resolve().parents[2] / "minimax_h3_mlx" / "packing.py"
            module_name = "minimax_h3_mlx._render_lab_packing_contract"
            previous_mlx = sys.modules.get("mlx")
            previous_mlx_core = sys.modules.get("mlx.core")
            stub_mlx = types.ModuleType("mlx")
            stub_core = types.ModuleType("mlx.core")
            stub_mlx.core = stub_core  # type: ignore[attr-defined]
            sys.modules["mlx"] = stub_mlx
            sys.modules["mlx.core"] = stub_core
            spec = importlib.util.spec_from_file_location(module_name, packing_path)
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load authoritative runtime source: {packing_path}")
            module = importlib.util.module_from_spec(spec)
            module.__package__ = "minimax_h3_mlx"
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            resolve_canvas_size = module.resolve_canvas_size
        except Exception as exc:  # pragma: no cover - depends on the host environment
            raise RuntimeError("Could not load the authoritative H3 canvas resolver") from exc
        finally:
            if previous_mlx is None:
                sys.modules.pop("mlx", None)
            else:
                sys.modules["mlx"] = previous_mlx
            if previous_mlx_core is None:
                sys.modules.pop("mlx.core", None)
            else:
                sys.modules["mlx.core"] = previous_mlx_core
            sys.modules.pop("minimax_h3_mlx._render_lab_packing_contract", None)
        resolver = resolve_canvas_size
    return tuple(int(value) for value in resolver(
        preset.aspect_width,
        preset.aspect_height,
        megapixels=preset.megapixels,
    ))


def validate_preset_against_runtime(preset_id: str) -> ResolutionPreset:
    preset = preset_by_id(preset_id)
    actual = authoritative_dimensions(preset)
    if actual != preset.dimensions:
        raise ValueError(
            f"Resolution preset {preset_id!r} drifted: expected runtime {preset.runtime_dimensions}, "
            f"runtime returned {actual}"
        )
    if not preset.project_approved:
        raise ValueError(f"Resolution preset {preset_id!r} is not project-approved")
    return preset


def preset_payload() -> list[dict[str, object]]:
    """Serialize the source used by both the browser selector and run config."""
    payload: list[dict[str, object]] = []
    for preset in RESOLUTION_PRESETS:
        if not preset.project_approved:
            continue
        runtime_height, runtime_width = preset.runtime_dimensions
        payload.append(
            {
                "id": preset.preset_id,
                "label": preset.label,
                "width": preset.expected_width,
                "height": preset.expected_height,
                "dimensions": f"{preset.expected_width} × {preset.expected_height}",
                "orientation": preset.orientation,
                "runtime_width": runtime_width,
                "runtime_height": runtime_height,
                "runtime_dimensions": f"{runtime_width} × {runtime_height}",
                "evidence_class": preset.evidence_class,
                "evidence_reference": preset.evidence_reference,
                "approval_note": preset.approval_note,
                "project_approved": preset.project_approved,
                "source_id": PROJECT_RESOLUTION_SOURCE_ID,
                "runtime_rule_id": RUNTIME_RESOLUTION_RULE_ID,
            }
        )
    return payload

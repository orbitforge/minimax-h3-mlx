"""H3-specific LoRA admission without constructing the H3 runtime.

The generic registry deliberately knows nothing about a model tree.  This module owns the
small H3 target inventory used at the production pipeline admission boundary, so a syntactically
valid generic adapter cannot reach denoising with zero callable H3 targets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .lora import (
    LoRAError,
    LoRARegistry,
    canonical_target,
    is_core_projection_target,
    is_final_adaln_target,
    is_streamed_adaln_target,
)


H3_MAIN_BLOCK_COUNT = 50
H3_TOKEN_REFINER_BLOCK_COUNT = 2

_H3_BLOCK_TARGET_PREFIX = re.compile(r"^(blocks|token_refiner\.blocks)\.(\d+)\.")
_H3_DIRECT_TARGETS = frozenset(
    {
        "video_patch_proj",
        "audio_patch_proj",
        "condition_proj",
        "time_embedder.proj_in",
        "time_embedder.proj_out",
        "final_layer.video_out",
        "final_layer.audio_out",
    }
)


class H3LoRACompatibilityError(LoRAError):
    """Raised when a generic adapter has no target callable by the H3 runtime."""


@dataclass(frozen=True)
class H3LoRACompatibilityReport:
    """Header-level H3 target admission evidence."""

    adapter_path: str
    registered_adapter_count: int
    registered_target_count: int
    compatible_targets: tuple[str, ...]
    incompatible_targets: tuple[str, ...]
    topology_counts: dict[str, int]

    @property
    def compatible_target_count(self) -> int:
        return len(self.compatible_targets)

    @property
    def incompatible_target_count(self) -> int:
        return len(self.incompatible_targets)

    @property
    def topology(self) -> dict[str, int]:
        """Return the registry's existing semantic topology for the closeout receipt."""
        return dict(self.topology_counts)


def is_h3_compatible_target(target: str) -> bool:
    """Return whether one normalized target is callable by the production H3 module tree."""
    normalized = canonical_target(target)
    if normalized in _H3_DIRECT_TARGETS:
        return True
    if is_streamed_adaln_target(normalized) or is_final_adaln_target(normalized):
        return True
    if not is_core_projection_target(normalized):
        return False
    match = _H3_BLOCK_TARGET_PREFIX.match(normalized)
    if match is None:
        return False
    index = int(match.group(2))
    return (
        (match.group(1) == "blocks" and index < H3_MAIN_BLOCK_COUNT)
        or (match.group(1) == "token_refiner.blocks" and index < H3_TOKEN_REFINER_BLOCK_COUNT)
    )


def _adapter_path_label(registry: LoRARegistry, adapter_path: str | Path | None) -> str:
    if adapter_path is not None:
        return str(adapter_path)
    sources = registry.sources
    if len(sources) == 1:
        return str(sources[0].path)
    if sources:
        return ", ".join(str(source.path) for source in sources)
    return "<in-memory registry>"


def validate_h3_lora_compatibility(
    registry: LoRARegistry,
    *,
    adapter_path: str | Path | None = None,
) -> H3LoRACompatibilityReport:
    """Admit a generic registry only when at least one H3 target can be called.

    Partial compatibility is intentional: existing generic adapters may carry targets for another
    supported surface while still containing a usable H3 target.  Those targets are reported to
    the caller; a registry with zero compatible targets fails closed before denoising.
    """
    if not isinstance(registry, LoRARegistry):
        raise TypeError(f"expected LoRARegistry, got {type(registry).__name__}")

    targets = tuple(canonical_target(target) for target in registry.targets)
    compatible = tuple(target for target in targets if is_h3_compatible_target(target))
    incompatible = tuple(target for target in targets if not is_h3_compatible_target(target))
    report = H3LoRACompatibilityReport(
        adapter_path=_adapter_path_label(registry, adapter_path),
        registered_adapter_count=registry.adapter_count,
        registered_target_count=len(targets),
        compatible_targets=compatible,
        incompatible_targets=incompatible,
        topology_counts=registry.topology_counts,
    )
    if report.compatible_target_count == 0:
        examples = ", ".join(repr(target) for target in report.incompatible_targets[:4]) or "<none>"
        raise H3LoRACompatibilityError(
            "H3 LoRA admission rejected before denoising: "
            f"adapter={report.adapter_path}; "
            f"registered targets={report.registered_target_count}; "
            f"registered adapters={report.registered_adapter_count}; "
            "compatible H3 targets=0; "
            f"incompatible targets={report.incompatible_target_count}; "
            f"examples={examples}"
        )
    return report

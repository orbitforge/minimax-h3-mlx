"""Explicit Turbo-LoRA schedule selection."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurboSchedule:
    """The reduced-step schedule used when a Turbo adapter is active.

    The base H3 scheduler remains authoritative for sigma arithmetic; this
    object selects its validated number of transformer evaluations (NFE).
    Adapter metadata can advertise another supported NFE count, and a CLI/API
    override can make that choice explicit.
    """

    steps: int = 8
    name: str = "turbo-8"
    sigmas: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or self.steps < 2:
            raise ValueError(f"Turbo schedule requires an integer NFE count at least 2, got {self.steps!r}")
        if self.sigmas is not None:
            expected_points = self.steps + 1
            if len(self.sigmas) != expected_points:
                raise ValueError(
                    f"Turbo schedule has {self.steps} NFE but {len(self.sigmas)} sigma values; "
                    f"expected {expected_points}"
                )
            if any(not math.isfinite(float(value)) for value in self.sigmas):
                raise ValueError("Turbo schedule sigmas must be finite")
            if any(right >= left for left, right in zip(self.sigmas, self.sigmas[1:])):
                raise ValueError("Turbo schedule sigmas must be strictly decreasing")
            if self.sigmas[-1] != 0.0:
                raise ValueError("Turbo schedule sigmas must end at 0.0")

    @classmethod
    def from_registry(cls, registry: Any = None, steps: int | None = None) -> "TurboSchedule":
        advertised = None if registry is None else getattr(registry, "turbo_steps", None)
        if registry is None:
            metadata = {}
        elif hasattr(registry, "scheduling_metadata"):
            # A composed registry may carry many auxiliary sources.  Only the explicitly bound
            # scheduling owner is allowed to advertise NFE or a custom sigma grid.
            metadata = getattr(registry, "scheduling_metadata") or {}
        else:
            metadata = getattr(registry, "metadata", {})
        raw_sigmas = metadata.get("turbo_sigmas", metadata.get("sigmas"))
        sigmas = None
        if raw_sigmas is not None:
            if isinstance(raw_sigmas, str):
                try:
                    raw_sigmas = json.loads(raw_sigmas)
                except json.JSONDecodeError as exc:
                    raise ValueError("Turbo LoRA sigma metadata is not valid JSON") from exc
            if not isinstance(raw_sigmas, (list, tuple)):
                raise ValueError("Turbo LoRA sigma metadata must be a list")
            sigmas = tuple(float(value) for value in raw_sigmas)
        selected = steps if steps is not None else (
            advertised or (len(sigmas) - 1 if sigmas else cls.steps)
        )
        # An explicit count selects the base H3 sigma construction and does
        # not silently reuse a metadata list with a different cardinality.
        selected_sigmas = None if steps is not None else sigmas
        schedule = cls(int(selected), name=f"turbo-{int(selected)}", sigmas=selected_sigmas)
        return schedule

    def configure(self, video_scheduler: Any, audio_scheduler: Any) -> None:
        """Configure both H3 modality schedulers with one shared NFE count.

        The scalar scheduler accepts a sigma-point count, so one terminal zero
        must be added to the public Turbo NFE count.
        """
        if self.sigmas is None:
            sigma_points = self.steps + 1
            video_scheduler.set_timesteps(num_inference_steps=sigma_points)
            audio_scheduler.set_timesteps(num_inference_steps=sigma_points)
        else:
            video_scheduler.set_timesteps(sigmas=list(self.sigmas))
            audio_scheduler.set_timesteps(sigmas=list(self.sigmas))

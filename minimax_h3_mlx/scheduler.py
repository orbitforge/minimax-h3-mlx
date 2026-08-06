"""MLX port of ``MiniMaxH3Scheduler`` — rectified-flow Euler with an exponential sigma shift.

The released checkpoints use ``shift = 12.0`` for video latents and ``3.0`` for audio, so a run
drives two independent schedules over one shared transformer forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _mx() -> Any:
    """Import MLX only when a production scheduler array is actually requested."""
    import mlx.core as mx
    return mx


def _float32(value: Any) -> Any:
    """Promote MLX values without forcing MLX onto NumPy scheduler fakes."""
    if getattr(value, "__mlx_array__", False) or value.__class__.__module__.startswith("mlx."):
        return value.astype(_mx().float32)
    return np.asarray(value, dtype=np.float32)


def _linspace_1_to_0(n: int) -> np.ndarray:
    """``linspace(1, 0, n)`` in float32, bit-identical to ``torch.linspace``.

    The rectified-flow sigma grid is collapsed with a consecutive-duplicate check, so a one-ulp
    difference can change how many sigmas survive and therefore the number of model evaluations.
    Reproducing torch's grid exactly keeps the step count identical to the reference.

    ATen's kernel takes a **float32** step, splits the range at the halfway point to keep the two
    ends symmetric, and evaluates ``start + step*i`` with a fused multiply-add — a single rounding.
    Computing in float64 from the float32 step and rounding once reproduces that FMA exactly
    (verified over n = 2..399).
    """
    start, end = 1.0, 0.0
    step = float(np.float32((end - start) / np.float32(n - 1)))
    half = n // 2
    i = np.arange(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[:half] = start + step * i[:half]
    out[half:] = end - step * (n - 1 - i[half:])
    return out.astype(np.float32)


class MiniMaxH3Scheduler:
    """Rectified-flow Euler scheduler (``eta = 0``) with exponential sigma shift.

    ``sigma' = s*sigma / (1 + (s-1)*sigma)`` over a ``linspace(1, 0, N)`` grid. The terminal ``0``
    is part of the grid (the shift maps ``0`` to exactly ``0``), so ``N`` sigmas drive ``N - 1``
    model evaluations and ``timesteps = 1 - sigmas[:-1]``.
    """

    order = 1

    def __init__(self, shift: float = 12.0):
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)
        self.sigmas: Any | None = None
        self.timesteps: Any | None = None
        self.num_inference_steps: int | None = None
        self._step_index: int | None = None

    @property
    def shift(self) -> float:
        return self._shift

    @property
    def step_index(self) -> int | None:
        return self._step_index

    def set_shift(self, shift: float) -> None:
        """Override the sigma shift; call before :meth:`set_timesteps`."""
        if shift <= 0:
            raise ValueError(f"`shift` must be positive, got {shift}.")
        self._shift = float(shift)

    def set_timesteps(
        self,
        num_inference_steps: int | None = None,
        sigmas: list[float] | Any | None = None,
    ) -> None:
        """Build the sigma / timestep schedule.

        The grid is always built in float32 on the host so the schedule never depends on the
        accelerator, matching the reference.
        """
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError(
                    "`set_timesteps` requires either an explicit `sigmas` schedule or "
                    f"`num_inference_steps` >= 2, got {num_inference_steps}."
                )
            base = _linspace_1_to_0(int(num_inference_steps))
            shift32 = np.float32(self._shift)
            shifted = (shift32 * base) / (np.float32(1.0) + np.float32(self._shift - 1.0) * base)
            # The shift compresses the grid near sigma = 1; collapse float32 collisions it creates.
            values: list[float] = []
            for v in shifted.tolist():
                if not values or v != values[-1]:
                    values.append(v)
        else:
            values = [float(v) for v in (sigmas.tolist() if hasattr(sigmas, "tolist") else sigmas)]
            decreasing = all(b < a for a, b in zip(values, values[1:]))
            if len(values) < 2 or not decreasing or values[-1] != 0.0:
                raise ValueError(
                    "`sigmas` must hold at least two strictly decreasing values ending at 0.0."
                )

        mx = _mx()
        self.sigmas = mx.array(values, dtype=mx.float32)
        self.timesteps = mx.array([1.0 - s for s in values[:-1]], dtype=mx.float32)
        self.num_inference_steps = len(values) - 1
        self._step_index = None

    def index_for_timestep(self, timestep: float) -> int:
        """Map a timestep value to its schedule index. The schedule is strictly increasing in t."""
        target = float(timestep)
        for i, t in enumerate(self.timesteps.tolist()):
            if t == target:
                return i
        raise ValueError(
            "Passed `timestep` is not in `self.timesteps`. Use values from `scheduler.timesteps`."
        )

    def scale_noise(self, sample: Any, timestep: float, noise: Any) -> Any:
        """Rectified-flow forward process in MiniMax-H3's ``t`` convention: ``x_t = t*x0 + (1-t)*noise``.

        Used to noise conditioning anchors, where ``t`` is a ``noise_aug`` level rather than a
        schedule entry, so it is taken at face value and never looked up in ``self.timesteps``.
        """
        t32 = np.float32(timestep)
        return float(t32) * sample + float(np.float32(1.0) - t32) * noise

    def step(self, model_output: Any, timestep: float, sample: Any) -> Any:
        """One Euler (``eta = 0``) step.

        The model output is a **data-ward** velocity, so the denoised estimate is
        ``x0 = x_t + (1 - t) * v`` — note the ``+``, the opposite of the usual flow-match sign.
        The update is the blend ``x_next = r*x_t + (1 - r)*x0`` with ``r = sigma_next / sigma``,
        evaluated in float32.
        """
        if isinstance(timestep, int):
            raise ValueError(
                "Passing integer indices as timesteps is not supported; pass one of the "
                "`scheduler.timesteps` values."
            )
        requested_index = self.index_for_timestep(timestep)
        if self._step_index is None:
            self._step_index = requested_index
        elif self._step_index != requested_index:
            raise ValueError(
                f"scheduler cursor mismatch: cursor={self._step_index}, requested={requested_index}"
            )
        if self._step_index >= self.num_inference_steps:
            raise IndexError("scheduler has no remaining transition")

        # The sigma used for x0 is recovered from the *timestep* the transformer was conditioned
        # on, while the Euler ratio below uses the sigma grid: for sigma < 0.5 the float32 round
        # trip `1 - (1 - sigma)` is not exact, and the reference keeps the two sources apart.
        # Every scalar here is evaluated in float32, matching the reference's tensor arithmetic —
        # doing it in Python floats would round twice and drift by an ulp per step.
        sigma_from_timestep = float(np.float32(1.0) - np.float32(timestep))
        denoised = sample + sigma_from_timestep * model_output

        sigma = np.float32(self.sigmas[self._step_index].item())
        sigma_next = np.float32(self.sigmas[self._step_index + 1].item())
        ratio = sigma_next / sigma
        one_minus_ratio = float(np.float32(1.0) - ratio)

        mx = _mx()
        prev = float(ratio) * sample.astype(mx.float32) + one_minus_ratio * denoised.astype(mx.float32)
        self._step_index += 1
        return prev.astype(sample.dtype)

    def transition(self, step_index: int) -> "MiniMaxH3ScheduleTransition":
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise ValueError(f"step_index must be an integer, got {step_index!r}")
        if self.sigmas is None or self.timesteps is None or self.num_inference_steps is None:
            raise ValueError("scheduler has no configured timestep schedule")
        if step_index < 0 or step_index >= self.num_inference_steps:
            raise IndexError(
                f"step_index {step_index} is outside the valid range [0, {self.num_inference_steps})"
            )
        return MiniMaxH3ScheduleTransition(
            step_index=step_index,
            current_timestep=float(self.timesteps[step_index].item()),
            next_timestep=1.0 - float(self.sigmas[step_index + 1].item()),
            current_sigma=float(self.sigmas[step_index].item()),
            next_sigma=float(self.sigmas[step_index + 1].item()),
        )


@dataclass(frozen=True)
class MiniMaxH3ScheduleTransition:
    """The canonical, observable transition consumed by one-step denoising."""

    step_index: int
    current_timestep: float
    next_timestep: float
    current_sigma: float
    next_sigma: float


@dataclass(frozen=True)
class MiniMaxH3MultimodalScheduleTransition:
    """The complete state for one adjacent video/audio transition."""

    step_index: int
    video_current_timestep: float
    video_next_timestep: float
    video_current_sigma: float
    video_next_sigma: float
    audio_current_timestep: float
    audio_next_timestep: float
    audio_current_sigma: float
    audio_next_sigma: float


class MiniMaxH3MultimodalScheduler:
    """One production update over MiniMax-H3's video and audio schedules.

    H3 has one transformer forward but two rectified-flow schedules.  Keeping this adapter next to
    the scalar scheduler makes that fact explicit while giving the production denoising seam one
    update operation to call.  ``step`` is intentionally not a loop: it performs exactly one video
    update and one audio update for the selected adjacent transition.
    """

    prediction_parameterization = "velocity"
    input_scaling = "identity"
    update_method = "rectified-flow-euler-data-ward-velocity-v1"

    def __init__(self, video: MiniMaxH3Scheduler, audio: MiniMaxH3Scheduler):
        self.video = video
        self.audio = audio
        if video.timesteps is None or audio.timesteps is None:
            raise ValueError("both modality schedulers must have a canonical schedule before construction")
        if video.num_inference_steps != audio.num_inference_steps:
            raise ValueError("video and audio schedules must have the same number of inference steps")

    @property
    def num_inference_steps(self) -> int:
        return int(self.video.num_inference_steps or 0)

    @property
    def timesteps(self) -> Any:
        return self.video.timesteps

    def transition(self, step_index: int) -> MiniMaxH3MultimodalScheduleTransition:
        if not isinstance(step_index, int) or isinstance(step_index, bool):
            raise ValueError(f"step_index must be an integer, got {step_index!r}")
        if step_index < 0 or step_index >= self.num_inference_steps:
            raise IndexError(
                f"step_index {step_index} is outside the valid range [0, {self.num_inference_steps})"
            )
        video = self.video.transition(step_index)
        audio = self.audio.transition(step_index)
        if video.step_index != audio.step_index:
            raise ValueError("video and audio scalar transitions disagree on step index")
        if not (0.0 <= video.next_sigma < video.current_sigma and 0.0 <= audio.next_sigma < audio.current_sigma):
            raise ValueError("both modality next sigmas must be nonnegative and below current sigma")
        return MiniMaxH3MultimodalScheduleTransition(
            step_index=step_index,
            video_current_timestep=video.current_timestep,
            video_next_timestep=video.next_timestep,
            video_current_sigma=video.current_sigma,
            video_next_sigma=video.next_sigma,
            audio_current_timestep=audio.current_timestep,
            audio_next_timestep=audio.next_timestep,
            audio_current_sigma=audio.current_sigma,
            audio_next_sigma=audio.next_sigma,
        )

    def prepare_model_input(
        self, video_sample: Any, audio_sample: Any, step_index: int
    ) -> tuple[Any, Any]:
        """Return scheduler-prepared inputs; H3's production contract is identity scaling."""
        self.transition(step_index)
        return video_sample, audio_sample

    def step(
        self,
        video_prediction: Any,
        audio_prediction: Any,
        video_sample: Any,
        audio_sample: Any,
        step_index: int,
    ) -> tuple[Any, Any]:
        transition = self.transition(step_index)
        expected_cursor = step_index
        for label, scalar in (("video", self.video), ("audio", self.audio)):
            if scalar.step_index is not None and scalar.step_index != expected_cursor:
                raise ValueError(
                    f"{label} scheduler cursor mismatch: cursor={scalar.step_index}, expected={expected_cursor}"
                )
        # The production pipeline promotes each transformer prediction to float32 before the
        # scalar Euler update. Preserve that boundary here as well; the samples remain in their
        # production bfloat16 storage dtype and the scalar schedulers own the exact float32 math.
        video_next = self.video.step(
            _float32(video_prediction[0]), transition.video_current_timestep, video_sample[0]
        )[None]
        audio_next = self.audio.step(
            _float32(audio_prediction[0]), transition.audio_current_timestep, audio_sample[0]
        )[None]
        expected_next_cursor = step_index + 1
        if self.video.step_index != expected_next_cursor or self.audio.step_index != expected_next_cursor:
            raise ValueError("video and audio scalar cursors did not advance to the same next index")
        return video_next, audio_next

    def configuration(self) -> dict[str, object]:
        return {
            "identity": "MiniMaxH3MultimodalScheduler",
            "video": {"identity": "MiniMaxH3Scheduler", "shift": self.video.shift},
            "audio": {"identity": "MiniMaxH3Scheduler", "shift": self.audio.shift},
            "num_inference_steps": self.num_inference_steps,
            "prediction_parameterization": self.prediction_parameterization,
            "input_scaling": self.input_scaling,
            "update_method": self.update_method,
        }

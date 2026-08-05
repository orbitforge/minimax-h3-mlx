"""Dependency-injected production one-step denoising seam for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


CANONICAL_VIDEO_SHAPE = (1, 1, 96)
CANONICAL_AUDIO_SHAPE = (1, 2, 32)
CANONICAL_TEXT_SHAPE = (1, 1, 5120)
CANONICAL_VIDEO_DTYPE = "bfloat16"
CANONICAL_AUDIO_DTYPE = "bfloat16"
CANONICAL_TEXT_DTYPE = "bfloat16"
CANONICAL_PREDICTION_DTYPE = "float32"


@dataclass(frozen=True)
class OneStepResult:
    """All externally meaningful values from exactly one production update."""

    video_prediction: Any
    audio_prediction: Any
    updated_video_latent: Any
    updated_audio_latent: Any
    step_index: int
    video_current_timestep: float
    video_next_timestep: float
    video_current_sigma: float
    video_next_sigma: float
    audio_current_timestep: float
    audio_next_timestep: float
    audio_current_sigma: float
    audio_next_sigma: float


def _dtype_name(value: Any) -> str:
    name = str(value)
    return name.removeprefix("mlx.core.")


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": [int(item) for item in value.shape], "dtype": _dtype_name(value.dtype)}


def _copy(value: Any) -> Any:
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    if _mlx_core_for(value) is not None:
        mx = _mlx_core_for(value)
        return mx.array(value)
    return np.array(value, copy=True)


def _mlx_core_for(value: Any) -> Any | None:
    """Return MLX dispatch for an array without using dtype text as type detection.

    The marker is useful for MLX-like test doubles; real MLX arrays are identified by their
    owning module.  Dtypes are only validated separately and never select this path.
    """
    marker = getattr(value, "__mlx_array__", False)
    if marker:
        supplied = getattr(value, "__mlx_core__", None)
        if supplied is not None:
            return supplied
        try:
            import mlx.core as mx
            return mx
        except ImportError:
            return None
    module = value.__class__.__module__
    if module == "mlx.core" or module.startswith("mlx."):
        import mlx.core as mx
        return mx
    return None


def _scalar(value: Any) -> Any:
    return value.item() if hasattr(value, "item") else value


def _finite(value: Any, label: str) -> None:
    mx = _mlx_core_for(value)
    if mx is not None:
        result = mx.all(mx.isfinite(value))
        mx.eval(result)
        is_finite = bool(_scalar(result))
    else:
        try:
            is_finite = bool(np.all(np.isfinite(np.asarray(value))))
        except Exception as exc:
            raise ValueError(f"{label} finite-value validation failed") from exc
    if not is_finite:
        raise ValueError(f"{label} contains nonfinite values")


def _exact_equal(left: Any, right: Any) -> bool:
    mx = _mlx_core_for(left) or _mlx_core_for(right)
    if mx is not None:
        result = mx.all(left == right)
        mx.eval(result)
        return bool(_scalar(result))
    try:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    except Exception as exc:
        raise ValueError("exact equality validation failed") from exc


def _require_array(value: Any, *, label: str, shape: tuple[int, ...], dtype: str) -> None:
    actual_shape = tuple(int(item) for item in getattr(value, "shape", ()))
    actual_dtype = _dtype_name(getattr(value, "dtype", None))
    if actual_shape != tuple(shape):
        raise ValueError(f"{label} shape mismatch: got {actual_shape}, expected {tuple(shape)}")
    if actual_dtype != dtype:
        raise ValueError(f"{label} dtype mismatch: got {actual_dtype!r}, expected {dtype!r}")
    _finite(value, label)


def _require_vector(value: Any, *, label: str, dtype: str, nonempty: bool = True) -> None:
    shape = tuple(int(item) for item in getattr(value, "shape", ()))
    if len(shape) != 1 or (nonempty and shape[0] == 0):
        raise ValueError(f"{label} shape mismatch: got {shape}, expected a nonempty vector")
    if _dtype_name(getattr(value, "dtype", None)) != dtype:
        raise ValueError(f"{label} dtype mismatch: got {_dtype_name(getattr(value, 'dtype', None))!r}, expected {dtype!r}")
    _finite(value, label)


def _transition(scheduler: Any, step_index: int) -> Mapping[str, Any]:
    transition = getattr(scheduler, "transition", None)
    if not callable(transition):
        raise ValueError("scheduler contract is incomplete: transition(step_index) is required")
    value = transition(step_index)
    if hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise ValueError("scheduler transition must be a mapping or dataclass-like object")
    required = {
        "step_index", "video_current_timestep", "video_next_timestep", "video_current_sigma",
        "video_next_sigma", "audio_current_timestep", "audio_next_timestep", "audio_current_sigma",
        "audio_next_sigma",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"scheduler transition is missing required fields: {missing}")
    if int(value["step_index"]) != step_index:
        raise ValueError("scheduler transition step index does not match the requested step")
    return value


def one_step_denoise(
    transformer: Any,
    scheduler: Any,
    *,
    video_latent: Any,
    audio_latent: Any,
    text_embedding: Any,
    timestep: Any,
    timestep_indices: Any,
    token_tags: Any,
    position_ids: Any,
    video_indices: Any,
    audio_indices: Any,
    text_indices: Any,
    step_index: int,
    modulation_cache: Any = None,
    expected_video_shape: tuple[int, ...] = CANONICAL_VIDEO_SHAPE,
    expected_audio_shape: tuple[int, ...] = CANONICAL_AUDIO_SHAPE,
    expected_text_shape: tuple[int, ...] = CANONICAL_TEXT_SHAPE,
    expected_prediction_dtype: str = CANONICAL_PREDICTION_DTYPE,
) -> OneStepResult:
    """Run one forward and one scheduler update, with no hidden iteration or mutation.

    The scheduler dependency must expose ``transition``, ``prepare_model_input`` and ``step``.
    Its ``step`` accepts `(video_prediction, audio_prediction, video_sample, audio_sample,
    step_index)` and returns the two updated batched latents.  This keeps scheduler semantics out of
    probes and makes call counts directly testable without importing MLX.
    """
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        raise ValueError(f"step_index must be an integer, got {step_index!r}")
    transition = _transition(scheduler, step_index)
    prepare = getattr(scheduler, "prepare_model_input", None)
    update = getattr(scheduler, "step", None)
    if not callable(prepare) or not callable(update):
        raise ValueError("scheduler contract is incomplete: prepare_model_input and step are required")
    parameterization = getattr(scheduler, "prediction_parameterization", None)
    if parameterization != "velocity":
        raise ValueError(
            f"unsupported prediction parameterization: {parameterization!r}; expected 'velocity'"
        )
    if getattr(scheduler, "input_scaling", None) != "identity":
        raise ValueError("unsupported model-input scaling contract; expected scheduler input_scaling='identity'")

    _require_array(video_latent, label="video latent", shape=expected_video_shape, dtype=CANONICAL_VIDEO_DTYPE)
    _require_array(audio_latent, label="audio latent", shape=expected_audio_shape, dtype=CANONICAL_AUDIO_DTYPE)
    _require_array(text_embedding, label="text embedding", shape=expected_text_shape, dtype=CANONICAL_TEXT_DTYPE)
    _require_vector(timestep, label="timestep", dtype="float32")
    _require_vector(timestep_indices, label="timestep indices", dtype="int32")
    _require_vector(token_tags, label="token tags", dtype="int32")
    _require_array(position_ids, label="position IDs", shape=(int(position_ids.shape[0]), 3), dtype="float32")
    _require_vector(video_indices, label="video indices", dtype="int32")
    _require_vector(audio_indices, label="audio indices", dtype="int32")
    _require_vector(text_indices, label="text indices", dtype="int32")
    if timestep_indices.shape != token_tags.shape or position_ids.shape[0] != token_tags.shape[0]:
        raise ValueError("packed timetable arrays must agree on sequence length")
    mx = _mlx_core_for(timestep)
    current_values = timestep if mx is not None else np.asarray(timestep, dtype=np.float32).reshape(-1)
    if mx is not None:
        required_timesteps = (
            transition["video_current_timestep"], transition["audio_current_timestep"])
        for value in required_timesteps:
            present = mx.any(current_values == np.float32(value))
            mx.eval(present)
            if not bool(_scalar(present)):
                raise ValueError("a modality current timestep is not present in the scheduler input timestep tensor")
    else:
        for value in (transition["video_current_timestep"], transition["audio_current_timestep"]):
            if not np.any(current_values == np.float32(value)):
                raise ValueError("a modality current timestep is not present in the scheduler input timestep tensor")

    video_input = _copy(video_latent)
    audio_input = _copy(audio_latent)
    prepared_video, prepared_audio = prepare(video_input, audio_input, step_index)
    _require_array(prepared_video, label="prepared video model input", shape=expected_video_shape, dtype=CANONICAL_VIDEO_DTYPE)
    _require_array(prepared_audio, label="prepared audio model input", shape=expected_audio_shape, dtype=CANONICAL_AUDIO_DTYPE)

    kwargs = {
        "modulation_cache": modulation_cache,
    }
    if modulation_cache is None:
        kwargs.pop("modulation_cache")
    # This is deliberately the only transformer invocation in the seam.
    prediction = transformer(
        prepared_video,
        prepared_audio,
        text_embedding,
        timestep,
        timestep_indices,
        token_tags,
        position_ids,
        video_indices,
        audio_indices,
        text_indices,
        **kwargs,
    )
    if not isinstance(prediction, (tuple, list)) or len(prediction) != 2:
        raise ValueError("transformer must return separate video and audio predictions")
    video_prediction, audio_prediction = prediction
    _require_array(video_prediction, label="video transformer prediction", shape=expected_video_shape, dtype=expected_prediction_dtype)
    _require_array(audio_prediction, label="audio transformer prediction", shape=expected_audio_shape, dtype=expected_prediction_dtype)

    updated = update(video_prediction, audio_prediction, video_input, audio_input, step_index)
    if not isinstance(updated, (tuple, list)) or len(updated) != 2:
        raise ValueError("scheduler step must return separate video and audio updated latents")
    updated_video, updated_audio = updated
    _require_array(updated_video, label="updated video latent", shape=expected_video_shape, dtype=CANONICAL_VIDEO_DTYPE)
    _require_array(updated_audio, label="updated audio latent", shape=expected_audio_shape, dtype=CANONICAL_AUDIO_DTYPE)
    if _exact_equal(video_latent, updated_video):
        raise ValueError("production denoising step left the video latent unchanged")
    if _exact_equal(audio_latent, updated_audio):
        raise ValueError("production denoising step left the audio latent unchanged")

    return OneStepResult(
        video_prediction=video_prediction,
        audio_prediction=audio_prediction,
        updated_video_latent=updated_video,
        updated_audio_latent=updated_audio,
        step_index=step_index,
        video_current_timestep=float(transition["video_current_timestep"]),
        video_next_timestep=float(transition["video_next_timestep"]),
        video_current_sigma=float(transition["video_current_sigma"]),
        video_next_sigma=float(transition["video_next_sigma"]),
        audio_current_timestep=float(transition["audio_current_timestep"]),
        audio_next_timestep=float(transition["audio_next_timestep"]),
        audio_current_sigma=float(transition["audio_current_sigma"]),
        audio_next_sigma=float(transition["audio_next_sigma"]),
    )

"""Dependency-injected production one-step denoising seam for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
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


@dataclass(frozen=True)
class DenoiseStepReceipt:
    """The complete observable result of one loop transition."""

    step_index: int
    timestep: Any
    timestep_indices: Any
    video_current_timestep: float
    video_next_timestep: float
    video_current_sigma: float
    video_next_sigma: float
    audio_current_timestep: float
    audio_next_timestep: float
    audio_current_sigma: float
    audio_next_sigma: float
    input_video_latent: Any
    input_audio_latent: Any
    video_prediction: Any
    audio_prediction: Any
    updated_video_latent: Any
    updated_audio_latent: Any


@dataclass(frozen=True)
class DenoiseLoopResult:
    """Result of a bounded production denoising loop."""

    final_video_latent: Any
    final_audio_latent: Any
    completed_steps: int
    step_receipts: tuple[DenoiseStepReceipt, ...]
    transformer_calls: int
    scheduler_updates: int
    cache_acquisitions: int
    cache_releases: int


def _call_step_provider(provider: Any, step_index: int, transition: Mapping[str, Any]) -> Any:
    """Call a timestep/cache provider without hiding provider failures."""
    if callable(provider):
        try:
            return provider(step_index, transition)
        except TypeError as exc:
            # A one-argument provider is a useful small seam for tests and resident callers.  Only
            # fall back when signature inspection proves that the provider does not accept two args;
            # a TypeError raised inside a two-argument provider must remain visible.
            import inspect
            try:
                signature = inspect.signature(provider)
                signature.bind(step_index, transition)
            except (TypeError, ValueError):
                return provider(step_index)
            raise exc
    if isinstance(provider, Mapping):
        try:
            return provider[step_index]
        except KeyError as exc:
            raise ValueError(f"provider has no value for step {step_index}") from exc
    if isinstance(provider, Sequence) and not isinstance(provider, (str, bytes, bytearray)):
        try:
            return provider[step_index]
        except IndexError as exc:
            raise ValueError(f"provider has no value for step {step_index}") from exc
    raise TypeError("step provider must be callable, a mapping, or a sequence")


def _resolve_step_tensors(
    provider: Any,
    step_index: int,
    transition: Mapping[str, Any],
    default_indices: Any,
) -> tuple[Any, Any]:
    value = _call_step_provider(provider, step_index, transition)
    if isinstance(value, Mapping):
        if "timestep" not in value:
            raise ValueError(f"timestep provider result for step {step_index} is missing 'timestep'")
        timestep = value["timestep"]
        indices = value.get("timestep_indices", default_indices)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        timestep, indices = value
    else:
        timestep, indices = value, default_indices
    if indices is None:
        raise ValueError("timestep_indices are required for every denoising step")
    return timestep, indices


def _release_step_cache(provider: Any, step_index: int, cache: Any) -> None:
    if provider is not None:
        release = getattr(provider, "release_step", None)
        if callable(release):
            release(step_index, cache)
            return
    release = getattr(cache, "release", None)
    if callable(release):
        release()


def _provider_cache(provider: Any, step_index: int, timestep: Any, transition: Mapping[str, Any]) -> Any:
    if provider is None:
        return None
    method = getattr(provider, "cache_for_step", None)
    if callable(method):
        try:
            return method(step_index, timestep)
        except TypeError as exc:
            import inspect
            try:
                inspect.signature(method).bind(step_index, timestep)
            except (TypeError, ValueError):
                return method(step_index, timestep, transition)
            raise exc
    return _call_step_provider(provider, step_index, transition)


def _validate_loop_steps(scheduler: Any, step_indices: Sequence[int], transition_count: int | None) -> tuple[int, ...]:
    requested = tuple(step_indices)
    if not requested:
        raise ValueError("denoise loop requires at least one step")
    if any(not isinstance(step, int) or isinstance(step, bool) for step in requested):
        raise ValueError("denoise loop step indices must be integers")
    available = getattr(scheduler, "num_inference_steps", None)
    if available is None:
        raise ValueError("scheduler must expose num_inference_steps")
    count = len(requested) if transition_count is None else transition_count
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("transition_count must be a positive integer")
    if count != len(requested):
        raise ValueError("transition_count must equal the number of requested step indices")
    if count > int(available):
        raise ValueError(f"requested {count} steps but scheduler exposes only {available} transitions")
    expected = tuple(range(count))
    if requested != expected:
        raise ValueError(f"step ordering must be exactly {list(expected)}, got {list(requested)}")
    return requested


def denoise_loop(
    transformer: Any,
    scheduler: Any,
    *,
    initial_video_latent: Any,
    initial_audio_latent: Any,
    text_embedding: Any,
    timestep_provider: Any,
    timestep_indices: Any = None,
    token_tags: Any,
    position_ids: Any,
    video_indices: Any,
    audio_indices: Any,
    text_indices: Any,
    modulation_cache_provider: Any = None,
    step_indices: Sequence[int] = (0, 1),
    transition_count: int | None = None,
    expected_video_shape: tuple[int, ...] = CANONICAL_VIDEO_SHAPE,
    expected_audio_shape: tuple[int, ...] = CANONICAL_AUDIO_SHAPE,
    expected_text_shape: tuple[int, ...] = CANONICAL_TEXT_SHAPE,
    expected_prediction_dtype: str = CANONICAL_PREDICTION_DTYPE,
) -> DenoiseLoopResult:
    """Run a strictly ordered, dependency-injected production denoising loop.

    ``timestep_provider`` returns either a timestep tensor, ``(timestep, indices)``, or a mapping
    with those two keys.  A cache provider exposes ``cache_for_step(step_index, timestep)`` and may
    expose ``release_step(step_index, cache)``.  Each cache is released in a ``finally`` block
    before the next step can acquire one; resident execution can pass ``None`` for the provider.
    """
    steps = _validate_loop_steps(scheduler, step_indices, transition_count)
    video_latent = _copy(initial_video_latent)
    audio_latent = _copy(initial_audio_latent)
    receipts: list[DenoiseStepReceipt] = []
    acquired_tokens: set[Any] = set()
    previous_timestep: Any | None = None
    transformer_calls = 0
    scheduler_updates = 0
    cache_acquisitions = 0
    cache_releases = 0

    try:
        for step_index in steps:
            transition = _transition(scheduler, step_index)
            cursor = getattr(scheduler, "step_index", None)
            if cursor is not None and cursor != step_index:
                raise ValueError(f"scheduler cursor mismatch: cursor={cursor}, requested={step_index}")
            timestep, step_timestep_indices = _resolve_step_tensors(
                timestep_provider, step_index, transition, timestep_indices
            )
            if previous_timestep is not None and _exact_equal(previous_timestep, timestep):
                raise ValueError("timestep provider returned the same timestep tensor for two derived steps")
            previous_timestep = timestep
            step_video_input = _copy(video_latent)
            step_audio_input = _copy(audio_latent)
            cache = None
            if modulation_cache_provider is not None:
                cache = _provider_cache(modulation_cache_provider, step_index, timestep, transition)
                cache_acquisitions += 1
            try:
                if modulation_cache_provider is not None:
                    token = getattr(cache, "session_token", None)
                    if token is None:
                        token = getattr(modulation_cache_provider, "last_session_token", None)
                    if token is not None:
                        if token in acquired_tokens:
                            raise ValueError("modulation cache session token was reused across timesteps")
                        acquired_tokens.add(token)
                # This is the orchestration boundary: the counter does not depend on an optional
                # transformer attribute and remains truthful when the call raises.
                transformer_calls += 1
                result = one_step_denoise(
                    transformer,
                    scheduler,
                    video_latent=video_latent,
                    audio_latent=audio_latent,
                    text_embedding=text_embedding,
                    timestep=timestep,
                    timestep_indices=step_timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    video_indices=video_indices,
                    audio_indices=audio_indices,
                    text_indices=text_indices,
                    step_index=step_index,
                    modulation_cache=cache,
                    expected_video_shape=expected_video_shape,
                    expected_audio_shape=expected_audio_shape,
                    expected_text_shape=expected_text_shape,
                    expected_prediction_dtype=expected_prediction_dtype,
                )
                # one_step_denoise includes exactly one scheduler.step call.  Count it only after
                # the complete one-step operation has returned successfully.
                scheduler_updates += 1
            finally:
                if modulation_cache_provider is not None:
                    _release_step_cache(modulation_cache_provider, step_index, cache)
                    cache = None
                    cache_releases += 1
            receipts.append(DenoiseStepReceipt(
                step_index=step_index,
                timestep=timestep,
                timestep_indices=step_timestep_indices,
                video_current_timestep=result.video_current_timestep,
                video_next_timestep=result.video_next_timestep,
                video_current_sigma=result.video_current_sigma,
                video_next_sigma=result.video_next_sigma,
                audio_current_timestep=result.audio_current_timestep,
                audio_next_timestep=result.audio_next_timestep,
                audio_current_sigma=result.audio_current_sigma,
                audio_next_sigma=result.audio_next_sigma,
                input_video_latent=step_video_input,
                input_audio_latent=step_audio_input,
                video_prediction=result.video_prediction,
                audio_prediction=result.audio_prediction,
                updated_video_latent=result.updated_video_latent,
                updated_audio_latent=result.updated_audio_latent,
            ))
            video_latent, audio_latent = result.updated_video_latent, result.updated_audio_latent

        if len(receipts) != len(steps):
            raise ValueError("completed step receipt count does not match requested transitions")
        if cache_acquisitions != (len(steps) if modulation_cache_provider is not None else 0):
            raise ValueError("cache acquisition count does not match requested transitions")
        if cache_releases != (len(steps) if modulation_cache_provider is not None else 0):
            raise ValueError("cache release count does not match requested transitions")
        return DenoiseLoopResult(
            video_latent, audio_latent, len(receipts), tuple(receipts), transformer_calls,
            scheduler_updates, cache_acquisitions, cache_releases,
        )
    except BaseException as exc:
        # Preserve completed-step evidence for a caller that must write a truthful failure receipt.
        # The original exception remains the failure; these attributes are diagnostic only.
        exc.denoise_step_receipts = tuple(receipts)
        exc.denoise_completed_steps = len(receipts)
        exc.denoise_transformer_calls = transformer_calls
        exc.denoise_scheduler_updates = scheduler_updates
        exc.denoise_cache_acquisitions = cache_acquisitions
        exc.denoise_cache_releases = cache_releases
        raise
    finally:
        # Drop local orchestration references on both success and failure.  The returned result owns
        # the receipt references; no cache is retained here.
        cache = None


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

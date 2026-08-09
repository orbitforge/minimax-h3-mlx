"""Dependency-injected production one-step denoising seam for MiniMax-H3."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Callable, Mapping

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
class ValidatedForwardResult:
    """Validated inputs and predictions from exactly one transformer invocation."""

    video_prediction: Any
    audio_prediction: Any
    input_video_latent: Any
    input_audio_latent: Any
    prepared_video: Any
    prepared_audio: Any
    transition: Mapping[str, Any]


def materialize_predictions(video_prediction: Any, audio_prediction: Any) -> None:
    """Materialize both predictions without converting MLX arrays through the host.

    MLX arrays use one explicit ``mx.eval`` boundary.  Small deferred test doubles may expose a
    ``materialize`` method; that method models the same lazy-execution boundary without importing
    MLX in the test process.
    """
    values = (video_prediction, audio_prediction)
    for value in values:
        materialize = getattr(value, "materialize", None)
        if callable(materialize):
            materialize()
    mlx_values = [value for value in values if _mlx_core_for(value) is not None]
    if mlx_values:
        mx = _mlx_core_for(mlx_values[0])
        mx.eval(*mlx_values)


@dataclass(frozen=True)
class StreamedTransitionResult:
    """The scheduler-safe result of one streamed transformer transition.

    The cache is intentionally absent from this result.  A caller receives this value only after
    the session has materialized the predictions and completed cache release, so the value can be
    passed to a scheduler without retaining streamed AdaLN state.
    """

    step_index: int
    forward: ValidatedForwardResult


def _default_streamed_cache_builder(transformer: Any, timestep: Any) -> Any:
    """Build one cache through the established derived-checkpoint sidecar seam."""
    # Keep the MLX import behind the runtime boundary so the lifecycle contract remains unit
    # testable with injected fakes and importing denoise.py remains MLX-free.
    from .streamed_adaln import build_streamed_modulation_cache

    return build_streamed_modulation_cache(transformer, timestep)


def _split_streamed_cache_builder_result(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("streamed cache builder must return a cache or (cache, statistics)")
        return value[0]
    return value


def release_streamed_modulation_cache(cache: Any) -> None:
    """Release one streamed cache through a bounded runtime-owned reference boundary.

    ``ModulationCache`` is deliberately reusable for the resident path and therefore has no
    resident-specific release method.  Derived transition state is released by clearing its
    retained tables and timetable here, or by honoring a cache-provided release method when one
    exists.  The caller's local reference is dropped by :class:`StreamedTransitionSession` after
    this function returns.
    """
    if cache is None:
        raise RuntimeError("cannot release a missing streamed modulation cache")

    release = getattr(cache, "release", None)
    if callable(release):
        release()
        tables = getattr(cache, "tables", None)
        if isinstance(tables, list) and tables:
            raise RuntimeError("streamed cache release returned with retained modulation tables")
        return

    tables = getattr(cache, "tables", None)
    if not isinstance(tables, list):
        raise RuntimeError("streamed modulation cache has no bounded release path")
    tables.clear()
    try:
        setattr(cache, "tables", [])
        if hasattr(cache, "timesteps"):
            setattr(cache, "timesteps", None)
    except BaseException as exc:
        raise RuntimeError("streamed modulation cache references could not be dropped") from exc
    if getattr(cache, "tables", None):
        raise RuntimeError("streamed modulation cache tables remained live after release")


def _mark_streamed_transition_cleanup(
    error: BaseException,
    *,
    started: bool,
    succeeded: bool,
    cleanup_error: BaseException | None,
) -> None:
    """Attach distinct cleanup state without replacing the primary runtime exception."""
    for name, value in (
        ("streamed_transition_cleanup_started", started),
        ("streamed_transition_cleanup_succeeded", succeeded),
        ("streamed_transition_cleanup_error", cleanup_error),
    ):
        try:
            setattr(error, name, value)
        except BaseException:
            # Exception annotation is diagnostic only; never mask the runtime failure.
            pass


class StreamedTransitionSession:
    """Own one derived streamed-AdaLN transition from cache build through release.

    ``run`` performs one validated transformer forward and returns only after the streamed cache
    has been released.  It deliberately does not call the scheduler: the returned predictions and
    copied input latents are the scheduler-facing boundary.  Packed token metadata and modality
    indices are passed through unchanged, including fixed keyframe or reference rows; this class
    does not assume a text/audio/video-only layout or slice target rows on the caller's behalf.

    The instance may be reused for sequential transitions, but it rejects reentrant use while a
    cache is active.  The cache builder and releaser are injectable for MLX-free contract tests;
    the defaults use the established streamed sidecar builder and the bounded runtime release
    helper above.  An optional observer receives runtime-neutral phase events for proof or
    diagnostics; it never owns the cache or changes the scheduler boundary.
    """

    def __init__(
        self,
        transformer: Any,
        *,
        cache_builder: Callable[[Any, Any], Any] | None = None,
        cache_releaser: Callable[[Any], None] | None = None,
        observer: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self._transformer = transformer
        self._cache_builder = cache_builder or _default_streamed_cache_builder
        self._cache_releaser = cache_releaser or release_streamed_modulation_cache
        self._observer = observer
        self._active = False

    @property
    def active(self) -> bool:
        """Whether this session currently owns an unreleased streamed cache."""
        return self._active

    def run(
        self,
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
        expected_video_shape: tuple[int, ...] | None = CANONICAL_VIDEO_SHAPE,
        expected_audio_shape: tuple[int, ...] | None = CANONICAL_AUDIO_SHAPE,
        expected_text_shape: tuple[int, ...] | None = CANONICAL_TEXT_SHAPE,
        expected_video_dtype: str = CANONICAL_VIDEO_DTYPE,
        expected_audio_dtype: str = CANONICAL_AUDIO_DTYPE,
        expected_text_dtype: str = CANONICAL_TEXT_DTYPE,
        expected_prediction_dtype: str = CANONICAL_PREDICTION_DTYPE,
        prediction_materializer: Callable[[Any, Any], None] | None = None,
    ) -> StreamedTransitionResult:
        """Build, forward, materialize, release, and return one transition result.

        The existing :func:`validated_transformer_forward` owns the explicit prediction
        materialization boundary.  Any failure before return prevents a scheduler-facing result
        from being produced; a cache acquired before that failure is released in the surrounding
        cleanup boundary.  If both the forward and release fail, the forward exception remains the
        raised exception and carries the release exception as
        ``streamed_transition_cleanup_error``.

        ``prediction_materializer`` is an optional dependency-injection seam for proof doubles;
        the session still controls when that callback runs and when cache release follows it.
        """
        if self._active:
            raise RuntimeError("streamed transition session already owns an active cache")
        self._active = True
        cache = None
        primary_error: BaseException | None = None
        release_error: BaseException | None = None

        def observe(event: str, **details: Any) -> None:
            if self._observer is not None:
                self._observer(event, {"step_index": step_index, **details})

        def materialize(video_prediction: Any, audio_prediction: Any) -> None:
            # The transformer call has returned at this point, while the predictions are still
            # deferred. Keeping these callbacks here makes the runtime ownership boundary visible
            # to proof observers without moving proof terminology into the runtime seam.
            observe("forward-complete")
            observe("materialize-start")
            if prediction_materializer is None:
                materialize_predictions(video_prediction, audio_prediction)
            else:
                prediction_materializer(video_prediction, audio_prediction)
            observe("materialize-complete")

        try:
            try:
                observe("transition-start")
                observe("cache-build-start")
                built = self._cache_builder(self._transformer, timestep)
                cache = _split_streamed_cache_builder_result(built)
                if cache is None:
                    raise ValueError("streamed cache builder returned no cache")
                observe("cache-build-complete")
                observe("forward-start")
                forward = validated_transformer_forward(
                    self._transformer,
                    scheduler,
                    video_latent=video_latent,
                    audio_latent=audio_latent,
                    text_embedding=text_embedding,
                    timestep=timestep,
                    timestep_indices=timestep_indices,
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
                    expected_video_dtype=expected_video_dtype,
                    expected_audio_dtype=expected_audio_dtype,
                    expected_text_dtype=expected_text_dtype,
                    expected_prediction_dtype=expected_prediction_dtype,
                    materialize=materialize,
                )
                observe("transition-succeeded")
                result = StreamedTransitionResult(
                    step_index=step_index,
                    forward=forward,
                )
                return result
            except BaseException as exc:
                primary_error = exc
                try:
                    observe("transition-failed", cache_acquired=cache is not None)
                except BaseException as observer_error:
                    # Observer failures are diagnostic and must not replace the runtime failure.
                    try:
                        setattr(exc, "streamed_transition_observer_error", observer_error)
                    except BaseException:
                        pass
                raise
        finally:
            try:
                if cache is not None:
                    release_observer_error: BaseException | None = None
                    try:
                        observe("cache-release-start", forward_succeeded=primary_error is None)
                    except BaseException as exc:
                        # Always attempt the actual release even if an observer is faulty.
                        release_observer_error = exc
                    try:
                        self._cache_releaser(cache)
                    except BaseException as exc:
                        release_error = exc
                    else:
                        try:
                            observe("cache-release-complete", forward_succeeded=primary_error is None)
                        except BaseException as exc:
                            release_error = exc
                    if release_error is None and release_observer_error is not None:
                        release_error = release_observer_error
                    if release_error is not None:
                        if primary_error is not None:
                            _mark_streamed_transition_cleanup(
                                primary_error,
                                started=True,
                                succeeded=False,
                                cleanup_error=release_error,
                            )
                        else:
                            _mark_streamed_transition_cleanup(
                                release_error,
                                started=True,
                                succeeded=False,
                                cleanup_error=None,
                            )
                    elif primary_error is not None:
                        _mark_streamed_transition_cleanup(
                            primary_error,
                            started=True,
                            succeeded=True,
                            cleanup_error=None,
                        )
                elif primary_error is not None and not hasattr(
                    primary_error, "streamed_transition_cleanup_started"
                ):
                    # A builder failure owns any partial-build cleanup internally; no complete
                    # cache was returned to this session for a second release.
                    _mark_streamed_transition_cleanup(
                        primary_error,
                        started=False,
                        succeeded=False,
                        cleanup_error=None,
                    )
            finally:
                # This is the ownership boundary: neither the reusable session nor the returned
                # result retains the cache after a transition, including failure paths.
                cache = None
                self._active = False
            if release_error is not None and primary_error is None:
                raise release_error


def apply_target_scheduler_updates(
    video_scheduler: Any,
    audio_scheduler: Any,
    *,
    video_prediction: Any,
    audio_prediction: Any,
    video_rows: Any,
    audio_rows: Any,
    video_timestep: float,
    audio_timestep: float,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    prediction_cast: Callable[[Any], Any],
    concatenate: Callable[[list[Any]], Any],
) -> tuple[Any, Any]:
    """Apply the established target-only video/audio scheduler update once.

    Both resident and derived pipeline modes use this seam so the scheduler math, modality
    slicing, and conditioning-row rebind remain one implementation. The caller controls when this
    function runs; derived mode calls it only after its streamed session has released the cache.
    """
    stepped_video = video_scheduler.step(
        prediction_cast(video_prediction[0, num_condition_video_rows:]),
        video_timestep,
        video_rows[num_condition_video_rows:],
    )
    stepped_audio = audio_scheduler.step(
        prediction_cast(audio_prediction[0, num_condition_audio_rows:]),
        audio_timestep,
        audio_rows[num_condition_audio_rows:],
    )
    video_rows = (
        concatenate([video_rows[:num_condition_video_rows], stepped_video])
        if num_condition_video_rows
        else stepped_video
    )
    audio_rows = (
        concatenate([audio_rows[:num_condition_audio_rows], stepped_audio])
        if num_condition_audio_rows
        else stepped_audio
    )
    return video_rows, audio_rows


def run_streamed_transition(
    transformer: Any,
    transition_scheduler: Any,
    video_scheduler: Any,
    audio_scheduler: Any,
    *,
    video_model_input: Any,
    audio_model_input: Any,
    text_embedding: Any,
    timestep: Any,
    timestep_indices: Any,
    layout: Any,
    step_index: int,
    video_timestep: float,
    audio_timestep: float,
    num_condition_video_rows: int,
    num_condition_audio_rows: int,
    prediction_cast: Callable[[Any], Any],
    concatenate: Callable[[list[Any]], Any],
    session_factory: Callable[[Any], Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Run one derived transition and update only generated rows after cache release.

    The model inputs are prepared by the caller so this orchestration seam remains independent of
    MLX array construction in tests. ``StreamedTransitionSession`` returns only after prediction
    materialization and streamed-cache release; scheduler updates therefore happen strictly after
    that ownership boundary. A session factory is injectable for MLX-free integration contracts.
    """
    session = (session_factory or StreamedTransitionSession)(transformer)
    result = session.run(
        transition_scheduler,
        video_latent=video_model_input,
        audio_latent=audio_model_input,
        text_embedding=text_embedding,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=layout.token_tags,
        position_ids=layout.position_ids,
        video_indices=layout.video_indices,
        audio_indices=layout.audio_indices,
        text_indices=layout.text_indices,
        step_index=step_index,
        expected_video_shape=tuple(int(value) for value in video_model_input.shape),
        expected_audio_shape=tuple(int(value) for value in audio_model_input.shape),
        expected_text_shape=tuple(int(value) for value in text_embedding.shape),
    )
    video_prediction = result.forward.video_prediction
    audio_prediction = result.forward.audio_prediction
    scheduler_video_rows = result.forward.input_video_latent[0]
    scheduler_audio_rows = result.forward.input_audio_latent[0]

    # The session has already released its cache. Apply the shared target-only scheduler update
    # and rebind complete row buffers so conditioning rows cannot be overwritten.
    video_rows, audio_rows = apply_target_scheduler_updates(
        video_scheduler,
        audio_scheduler,
        video_prediction=video_prediction,
        audio_prediction=audio_prediction,
        video_rows=scheduler_video_rows,
        audio_rows=scheduler_audio_rows,
        video_timestep=video_timestep,
        audio_timestep=audio_timestep,
        num_condition_video_rows=num_condition_video_rows,
        num_condition_audio_rows=num_condition_audio_rows,
        prediction_cast=prediction_cast,
        concatenate=concatenate,
    )
    return video_rows, audio_rows, video_prediction, audio_prediction


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
            cache_acquired = False
            step_primary_error: BaseException | None = None
            step_cleanup_error: BaseException | None = None
            step_cleanup_attempted = False
            step_cleanup_succeeded = False
            try:
                try:
                    if modulation_cache_provider is not None:
                        cache = _provider_cache(modulation_cache_provider, step_index, timestep, transition)
                        cache_acquired = True
                        cache_acquisitions += 1
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
                except BaseException as exc:
                    step_primary_error = exc
                    raise
            finally:
                if modulation_cache_provider is not None:
                    step_cleanup_attempted = True
                    try:
                        if cache_acquired:
                            _release_step_cache(modulation_cache_provider, step_index, cache)
                            cache_releases += 1
                        else:
                            # A provider may have opened a session before its builder failed.  Give
                            # it one explicit failure-cleanup seam when it supplies one; providers
                            # without that seam own their cleanup inside cache_for_step().
                            cleanup_failed_step = getattr(modulation_cache_provider, "cleanup_failed_step", None)
                            if callable(cleanup_failed_step):
                                cleanup_failed_step(step_index, cache)
                        step_cleanup_succeeded = True
                    except BaseException as cleanup_exc:
                        step_cleanup_error = cleanup_exc
                        step_cleanup_succeeded = False
                        if step_primary_error is None:
                            raise
                        # Keep the computational exception as the active exception.  The outer
                        # evidence block below receives the cleanup exception as a diagnostic.
                        setattr(step_primary_error, "denoise_cleanup_error", cleanup_exc)
                    finally:
                        if step_primary_error is not None and not cache_acquired:
                            existing_cleanup = getattr(step_primary_error, "cache_cleanup_error", None)
                            if existing_cleanup is not None:
                                step_cleanup_error = existing_cleanup
                                step_cleanup_succeeded = False
                            step_cleanup_attempted = bool(
                                getattr(step_primary_error, "cache_cleanup_attempted", step_cleanup_attempted)
                            )
                            step_cleanup_succeeded = bool(
                                getattr(step_primary_error, "cache_cleanup_succeeded", step_cleanup_succeeded)
                            )
                        if step_primary_error is not None:
                            setattr(step_primary_error, "denoise_cleanup_attempted", step_cleanup_attempted)
                            setattr(step_primary_error, "denoise_cleanup_succeeded", step_cleanup_succeeded)
                        if step_cleanup_error is not None and step_primary_error is not None:
                            setattr(step_primary_error, "denoise_cleanup_error", step_cleanup_error)
                        cache = None
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
        exc.denoise_primary_error = exc
        exc.denoise_cleanup_attempted = getattr(exc, "denoise_cleanup_attempted", False)
        exc.denoise_cleanup_succeeded = getattr(exc, "denoise_cleanup_succeeded", False)
        exc.denoise_cleanup_error = getattr(exc, "denoise_cleanup_error", None)
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


def copy_runtime_array(value: Any) -> Any:
    """Copy a runtime array without routing an MLX value through NumPy.

    ``mx.asarray(..., copy=True)`` is the explicit MLX-native copy boundary.  Evaluating the
    result here makes the copy's ownership and materialization observable before the caller can
    release or mutate the source.  Non-MLX values retain their existing object/NumPy copy
    semantics, including the small typed fakes used by the contract tests.
    """
    mx = _mlx_core_for(value)
    if mx is not None:
        asarray = getattr(mx, "asarray", None)
        evaluate = getattr(mx, "eval", None)
        if not callable(asarray) or not callable(evaluate):
            raise RuntimeError("MLX runtime copy requires mlx.core.asarray and mlx.core.eval")
        copied = asarray(value, copy=True)
        evaluate(copied)
        return copied
    copier = getattr(value, "copy", None)
    if callable(copier):
        return copier()
    return np.array(value, copy=True)


def _copy(value: Any) -> Any:
    return copy_runtime_array(value)


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


def validated_transformer_forward(
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
    expected_video_shape: tuple[int, ...] | None = CANONICAL_VIDEO_SHAPE,
    expected_audio_shape: tuple[int, ...] | None = CANONICAL_AUDIO_SHAPE,
    expected_text_shape: tuple[int, ...] | None = CANONICAL_TEXT_SHAPE,
    expected_video_dtype: str = CANONICAL_VIDEO_DTYPE,
    expected_audio_dtype: str = CANONICAL_AUDIO_DTYPE,
    expected_text_dtype: str = CANONICAL_TEXT_DTYPE,
    expected_prediction_dtype: str = CANONICAL_PREDICTION_DTYPE,
    materialize: Callable[[Any, Any], None] | None = None,
) -> ValidatedForwardResult:
    """Validate the production forward contract and materialize both predictions.

    This is intentionally the only transformer-forward contract used by the production one-step
    path and the v0.5d full-schedule proof.  The caller decides when to perform the scheduler
    update; in particular, v0.5d releases its streamed cache and drops its local reference before
    calling the scheduler.
    """
    if not isinstance(step_index, int) or isinstance(step_index, bool):
        raise ValueError(f"step_index must be an integer, got {step_index!r}")
    transition = _transition(scheduler, step_index)
    cursor = getattr(scheduler, "step_index", None)
    if cursor is not None and cursor != step_index:
        raise ValueError(f"scheduler cursor mismatch: cursor={cursor}, requested={step_index}")
    prepare = getattr(scheduler, "prepare_model_input", None)
    if not callable(prepare):
        raise ValueError("scheduler contract is incomplete: prepare_model_input is required")
    parameterization = getattr(scheduler, "prediction_parameterization", None)
    if parameterization != "velocity":
        raise ValueError(
            f"unsupported prediction parameterization: {parameterization!r}; expected 'velocity'"
        )
    if getattr(scheduler, "input_scaling", None) != "identity":
        raise ValueError("unsupported model-input scaling contract; expected scheduler input_scaling='identity'")

    if expected_video_shape is not None:
        _require_array(video_latent, label="video latent", shape=expected_video_shape, dtype=expected_video_dtype)
    if expected_audio_shape is not None:
        _require_array(audio_latent, label="audio latent", shape=expected_audio_shape, dtype=expected_audio_dtype)
    if text_embedding is None:
        if expected_text_shape is not None:
            raise ValueError("text embedding is required by the transformer forward contract")
    elif expected_text_shape is not None:
        _require_array(text_embedding, label="text embedding", shape=expected_text_shape, dtype=expected_text_dtype)

    _require_vector(timestep, label="timestep", dtype="float32")
    _require_vector(timestep_indices, label="timestep indices", dtype="int32")
    _require_vector(token_tags, label="token tags", dtype="int32")
    position_shape = tuple(int(item) for item in getattr(position_ids, "shape", ()))
    if len(position_shape) != 2 or position_shape[1] != 3:
        raise ValueError(f"position IDs shape mismatch: got {position_shape}, expected (sequence_length, 3)")
    if _dtype_name(getattr(position_ids, "dtype", None)) != "float32":
        raise ValueError(
            f"position IDs dtype mismatch: got {_dtype_name(getattr(position_ids, 'dtype', None))!r}, expected 'float32'"
        )
    _finite(position_ids, "position IDs")
    _require_vector(video_indices, label="video indices", dtype="int32")
    _require_vector(audio_indices, label="audio indices", dtype="int32")
    _require_vector(text_indices, label="text indices", dtype="int32")
    sequence_length = int(token_tags.shape[0])
    if (
        timestep_indices.shape != token_tags.shape
        or position_shape[0] != sequence_length
    ):
        raise ValueError("packed timetable arrays must agree on sequence length")
    current_values = timestep if _mlx_core_for(timestep) is not None else np.asarray(timestep, dtype=np.float32).reshape(-1)
    mx = _mlx_core_for(timestep)
    for value in (transition["video_current_timestep"], transition["audio_current_timestep"]):
        if mx is not None:
            present = mx.any(current_values == np.float32(value))
            mx.eval(present)
            is_present = bool(_scalar(present))
        else:
            is_present = bool(np.any(current_values == np.float32(value)))
        if not is_present:
            raise ValueError("a modality current timestep is not present in the scheduler input timestep tensor")

    video_input = _copy(video_latent)
    audio_input = _copy(audio_latent)
    prepared_video, prepared_audio = prepare(video_input, audio_input, step_index)
    if expected_video_shape is not None:
        _require_array(
            prepared_video,
            label="prepared video model input",
            shape=expected_video_shape,
            dtype=expected_video_dtype,
        )
    if expected_audio_shape is not None:
        _require_array(
            prepared_audio,
            label="prepared audio model input",
            shape=expected_audio_shape,
            dtype=expected_audio_dtype,
        )

    kwargs = {"modulation_cache": modulation_cache}
    if modulation_cache is None:
        kwargs.pop("modulation_cache")
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
    if materialize is None:
        materialize_predictions(video_prediction, audio_prediction)
    else:
        materialize(video_prediction, audio_prediction)
    if expected_video_shape is not None:
        _require_array(
            video_prediction,
            label="video transformer prediction",
            shape=expected_video_shape,
            dtype=expected_prediction_dtype,
        )
    if expected_audio_shape is not None:
        _require_array(
            audio_prediction,
            label="audio transformer prediction",
            shape=expected_audio_shape,
            dtype=expected_prediction_dtype,
        )
    return ValidatedForwardResult(
        video_prediction=video_prediction,
        audio_prediction=audio_prediction,
        input_video_latent=video_input,
        input_audio_latent=audio_input,
        prepared_video=prepared_video,
        prepared_audio=prepared_audio,
        transition=transition,
    )


def validate_updated_latents(
    video_latent: Any,
    audio_latent: Any,
    *,
    expected_video_shape: tuple[int, ...],
    expected_audio_shape: tuple[int, ...],
    expected_video_dtype: str = CANONICAL_VIDEO_DTYPE,
    expected_audio_dtype: str = CANONICAL_AUDIO_DTYPE,
) -> tuple[Any, Any]:
    """Materialize and validate scheduler-updated video and audio rows."""
    materialize_predictions(video_latent, audio_latent)
    _require_array(video_latent, label="updated video latent", shape=expected_video_shape, dtype=expected_video_dtype)
    _require_array(audio_latent, label="updated audio latent", shape=expected_audio_shape, dtype=expected_audio_dtype)
    return video_latent, audio_latent


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
    update = getattr(scheduler, "step", None)
    if not callable(update):
        raise ValueError("scheduler contract is incomplete: prepare_model_input and step are required")
    forward = validated_transformer_forward(
        transformer,
        scheduler,
        video_latent=video_latent,
        audio_latent=audio_latent,
        text_embedding=text_embedding,
        timestep=timestep,
        timestep_indices=timestep_indices,
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        step_index=step_index,
        modulation_cache=modulation_cache,
        expected_video_shape=expected_video_shape,
        expected_audio_shape=expected_audio_shape,
        expected_text_shape=expected_text_shape,
        expected_prediction_dtype=expected_prediction_dtype,
    )
    transition = forward.transition
    video_prediction = forward.video_prediction
    audio_prediction = forward.audio_prediction
    video_input = forward.input_video_latent
    audio_input = forward.input_audio_latent

    updated = update(video_prediction, audio_prediction, video_input, audio_input, step_index)
    if not isinstance(updated, (tuple, list)) or len(updated) != 2:
        raise ValueError("scheduler step must return separate video and audio updated latents")
    updated_video, updated_audio = updated
    validate_updated_latents(
        updated_video,
        updated_audio,
        expected_video_shape=expected_video_shape,
        expected_audio_shape=expected_audio_shape,
    )
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

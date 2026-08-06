"""The MiniMax-H3 text/keyframe -> video+audio pipeline in MLX.

One packed sequence carries text, keyframe conditioning, audio and video rows at once, and a single
transformer forward per step predicts the velocity of every row — video and audio are denoised
*jointly*, on two schedules with different sigma shifts (12.0 and 3.0). The checkpoint is
CFG-distilled, so there is no unconditional pass and no guidance scale.

Conditioning rows are re-imposed by construction rather than by masking: only the generated rows are
ever written back, so keyframe anchors survive the whole loop untouched.

The AdaLN modulation cache is built once over the union of every timestep the run will present, and
the 13B of `adaln_proj` is then dropped — see :mod:`minimax_h3_mlx.adaln`.
"""

from __future__ import annotations

import gc
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mlx.core as mx
import numpy as np

from .adaln import ModulationCache, drop_adaln_weights
from .config import DiTConfig, PipelineConfig
from .packing import (
    AUDIO_CHANNELS,
    FPS,
    KEYFRAME_NOISE_AUG,
    PIXEL_MEAN,
    PIXEL_STD,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    pack_audio_latents,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from .scheduler import MiniMaxH3Scheduler


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "n/a"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return "n/a"


def _memory_snapshot() -> str:
    """Best-effort memory telemetry without adding a runtime dependency."""
    parts: list[str] = []
    try:
        import psutil

        parts.append(f"rss={_format_bytes(psutil.Process().memory_info().rss)}")
    except Exception:
        try:
            # macOS reports ru_maxrss in bytes; this fallback is a peak RSS label, not total
            # unified memory and not current resident memory.
            parts.append(f"rss_peak={_format_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}")
        except Exception:
            pass

    for label, name in (("mlx_active", "get_active_memory"), ("mlx_cache", "get_cache_memory"),
                        ("mlx_peak", "get_peak_memory")):
        try:
            getter = getattr(mx, name, None)
            if callable(getter):
                parts.append(f"{label}={_format_bytes(getter())}")
        except Exception:
            continue
    return " ".join(parts) or "unavailable"


@dataclass
class GenerationResult:
    video: np.ndarray  # (frames, height, width, 3) uint8
    audio: np.ndarray  # (2, samples) float32, in [-1, 1]
    sample_rate: int
    fps: int = FPS
    seconds_per_step: float = 0.0
    total_seconds: float = 0.0


class MiniMaxH3Pipeline:
    """Joint video + audio generation."""

    def __init__(
        self,
        dit,
        text_encoder,
        video_vae,
        audio_vae,
        config: PipelineConfig | None = None,
        unload_text_encoder: bool = False,
        generation_loaders: dict[str, Callable[[], object]] | None = None,
        dit_config: DiTConfig | None = None,
        video_vae_config=None,
        audio_vae_config=None,
    ):
        self.dit = dit
        self.text_encoder = text_encoder
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.config = config or PipelineConfig()
        self.unload_text_encoder = unload_text_encoder
        self._generation_loaders = generation_loaders or {}
        self._dit_config = dit_config or getattr(dit, "config", None)
        self._video_vae_config = video_vae_config or getattr(video_vae, "config", None)
        self._audio_vae_config = audio_vae_config or getattr(audio_vae, "config", None)
        self._cache: ModulationCache | None = None
        self._cache_timesteps: tuple[float, ...] | None = None
        self._adaln_purge_attempts = 0
        self._adaln_purge_status = "not-run"

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str | Path,
        transformer_dir: str | Path | None = None,
        dtype: mx.Dtype = mx.bfloat16,
        load_vision: bool = False,
        unload_text_encoder: bool = False,
        keep_adaln: bool = False,
        verbose: bool = True,
    ) -> "MiniMaxH3Pipeline":
        """Load a released ``FL2VA/`` (or ``Ref2VA/``) directory.

        Args:
            checkpoint_dir: the upstream release, which supplies the VAEs and the text encoder.
            transformer_dir: load the DiT from here instead of ``<checkpoint_dir>/transformer``.
                This is how a published quant is used: the quantized repository holds only the
                transformer, and everything else still comes from upstream. ``load_dit`` picks up
                the recorded recipe from its ``quant_config.json`` automatically.
        """
        from .load import (
            load_audio_vae,
            load_audio_vae_config,
            load_dit,
            load_video_vae,
            load_video_vae_config,
            inspect_checkpoint_format,
        )
        from .text_encoder import MiniMaxH3TextEncoder

        root = Path(checkpoint_dir)
        dit_path = Path(transformer_dir) if transformer_dir else root / "transformer"
        format_info = inspect_checkpoint_format(dit_path)
        if format_info.checkpoint_format == "derived" and keep_adaln:
            raise ValueError(
                "--keep-adaln is not supported for derived checkpoints: block AdaLN weights live in "
                "sidecars and resident derived loading is not implemented"
            )
        config = PipelineConfig.from_model_index(root / "model_index.json")
        dit_config = DiTConfig.from_json(dit_path / "config.json")
        video_vae_config = load_video_vae_config(root / "video_vae")
        audio_vae_config = load_audio_vae_config(root / "audio_vae")

        def step(label, fn):
            started = time.perf_counter()
            if verbose:
                print(f"  [memory] before {label}: {_memory_snapshot()}", flush=True)
            try:
                out = fn()
            except Exception as exc:
                raise RuntimeError(f"Stage 1 failed while loading the {label}.") from exc
            if verbose:
                print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)
                print(f"  [memory] after {label}: {_memory_snapshot()}", flush=True)
            return out

        if verbose:
            print(f"loading MiniMax-H3 from {root}")
        text_encoder = step(
            "text encoder", lambda: MiniMaxH3TextEncoder(root / "text_encoder", dtype=dtype, load_vision=load_vision)
        )

        # Stage 2 is deliberately represented as factories.  Constructing the pipeline must not
        # map or evaluate any generation component before the request has been encoded and the
        # large conditioner has been released.
        generation_loaders = {
            "dit": lambda: load_dit(dit_path, keep_adaln=keep_adaln),
            "video_vae": lambda: load_video_vae(root / "video_vae"),
            "audio_vae": lambda: load_audio_vae(root / "audio_vae"),
        }
        return cls(
            None,
            text_encoder,
            None,
            None,
            config,
            unload_text_encoder=unload_text_encoder,
            generation_loaders=generation_loaders,
            dit_config=dit_config,
            video_vae_config=video_vae_config,
            audio_vae_config=audio_vae_config,
        )

    def _timed_stage(self, label: str, fn: Callable[[], object], verbose: bool):
        started = time.perf_counter()
        if verbose:
            print(f"  [memory] before {label}: {_memory_snapshot()}", flush=True)
        result = fn()
        if verbose:
            print(f"  {label}: {time.perf_counter() - started:.1f}s", flush=True)
            print(f"  [memory] after {label}: {_memory_snapshot()}", flush=True)
        return result

    def _release_text_encoder(self, prompt_embeds: mx.array, verbose: bool) -> None:
        """Materialize conditioning, then drop every pipeline-owned encoder reference."""
        started = time.perf_counter()
        mx.eval(prompt_embeds)
        if verbose:
            print(f"  [memory] before text encoder release: {_memory_snapshot()}", flush=True)
        encoder = self.text_encoder
        self.text_encoder = None
        del encoder
        self._clear_runtime_memory("text-encoder-release", verbose)
        if verbose:
            print(f"  text encoder release: {time.perf_counter() - started:.1f}s", flush=True)
            print(f"  [memory] after text encoder release: {_memory_snapshot()}", flush=True)

    def _prepare_conditioning(self, prompt: str, images: list | None, verbose: bool):
        if self.text_encoder is None:
            raise RuntimeError(
                "This staged pipeline has already released its text encoder and is one-shot. "
                "Use --keep-text-encoder when repeated generation is required."
            )
        started = time.perf_counter()
        prompt_embeds, text_token_tags = self.text_encoder.encode(prompt, images)
        # `encode` evaluates today, but this boundary is intentional: it protects the lifetime
        # contract if the encoder implementation becomes lazier in the future.
        mx.eval(prompt_embeds)
        if verbose:
            print(f"  prompt encoding: {time.perf_counter() - started:.1f}s", flush=True)
        if self.unload_text_encoder:
            self._release_text_encoder(prompt_embeds, verbose)
        return prompt_embeds, text_token_tags

    def _load_component(self, attr: str, label: str, verbose: bool):
        current = getattr(self, attr)
        if current is not None:
            return current
        loader = self._generation_loaders.get(attr)
        if loader is None:
            raise RuntimeError(f"No loader factory is available for the {label}.")
        try:
            value = self._timed_stage(f"{label} loading", loader, verbose)
        except Exception as exc:
            self._clear_runtime_memory("failure-cleanup", verbose)
            raise RuntimeError(f"Stage failed while loading the {label}; no render was started.") from exc
        setattr(self, attr, value)
        if attr == "dit":
            self._dit_config = getattr(value, "config", self._dit_config)
        elif attr == "video_vae":
            self._video_vae_config = getattr(value, "config", self._video_vae_config)
        elif attr == "audio_vae":
            self._audio_vae_config = getattr(value, "config", self._audio_vae_config)
        return value

    def _clear_runtime_memory(self, purge_reason: str, verbose: bool) -> str:
        gc.collect()
        if verbose:
            print(f"  [memory] immediately after Python object reclamation ({purge_reason}): "
                  f"{_memory_snapshot()}", flush=True)
        return self._purge_allocator_cache(purge_reason, verbose)

    def _purge_allocator_cache(self, reason: str, verbose: bool) -> str:
        """Best-effort allocator purge with reasoned, nonfatal telemetry."""
        started = time.perf_counter()
        display_name = "AdaLN allocator purge" if reason == "adaln-pre-denoise" else "allocator purge"
        if verbose:
            print(f"  [memory] immediately before {display_name} ({reason}): "
                  f"{_memory_snapshot()}", flush=True)
        clear_cache = getattr(mx, "clear_cache", None)
        if not callable(clear_cache):
            status = "unavailable"
        else:
            try:
                clear_cache()
            except Exception as exc:
                status = "failed"
                if verbose:
                    print(f"  {display_name} failed nonfatally ({reason}): {exc}", flush=True)
            else:
                status = "success"
        if verbose:
            elapsed = time.perf_counter() - started
            print(f"  {display_name}: {status} in {elapsed:.3f}s (reason={reason})", flush=True)
            print(f"  [memory] immediately after {display_name} ({reason}): "
                  f"{_memory_snapshot()}", flush=True)
        return status

    def _purge_adaln_allocator_cache(self, verbose: bool) -> str:
        """Purge MLX's allocator cache once, without making it a generation failure."""
        if self._adaln_purge_attempts:
            return self._adaln_purge_status
        self._adaln_purge_attempts += 1
        status = self._purge_allocator_cache("adaln-pre-denoise", verbose)
        self._adaln_purge_status = status
        return status

    def _release_component(
        self,
        attr: str,
        label: str,
        verbose: bool,
        purge_reason: str | None = None,
    ) -> None:
        component = getattr(self, attr)
        if component is None:
            return
        started = time.perf_counter()
        if verbose:
            print(f"  [memory] before {label} release: {_memory_snapshot()}", flush=True)
        setattr(self, attr, None)
        del component
        if attr == "dit":
            self._cache = None
            self._cache_timesteps = None
        if purge_reason is None:
            purge_reason = {
                "dit": "transformer-release",
                "video_vae": "video-vae-release",
                "audio_vae": "audio-vae-release",
            }[attr]
        self._clear_runtime_memory(purge_reason, verbose)
        if verbose:
            print(f"  {label} release: {time.perf_counter() - started:.1f}s", flush=True)
            print(f"  [memory] after {label} release: {_memory_snapshot()}", flush=True)

    def _require_configs(self):
        missing = [name for name, value in (("transformer", self._dit_config),
                                             ("video VAE", self._video_vae_config),
                                             ("audio VAE", self._audio_vae_config)) if value is None]
        if missing:
            raise RuntimeError(
                "Static component configuration is unavailable for: " + ", ".join(missing)
            )

    def _load_transformer(self, verbose: bool):
        return self._load_component("dit", "transformer", verbose)

    def _load_video_vae(self, verbose: bool):
        return self._load_component("video_vae", "video VAE", verbose)

    def _load_audio_vae(self, verbose: bool):
        return self._load_component("audio_vae", "audio VAE", verbose)

    def _load_generation_components(self, verbose: bool) -> None:
        """Compatibility helper for older callers; the normal path never uses this bulk load."""
        self._load_transformer(verbose)
        self._load_video_vae(verbose)
        self._load_audio_vae(verbose)

    # -- schedule -----------------------------------------------------------------------------

    def _build_schedules(self, num_inference_steps: int):
        video = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video)
        audio = MiniMaxH3Scheduler(shift=self.config.sigma_shift_audio)
        video.set_timesteps(num_inference_steps)
        audio.set_timesteps(num_inference_steps)
        return video, audio

    def _row_timestep_plan(self, layout, video_timesteps, audio_timesteps):
        """Per-step ``(timestep_indices,)`` against one global timestep table.

        The transformer is handed the same table at every step, so a single
        :class:`ModulationCache` covers the whole run. Conditioning video rows sit at
        ``max(t, 0.999)`` and reference audio rows at ``1.0``, matching the reference.
        """
        per_step = []
        for t, at in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
            distinct, inverse = build_row_timesteps(
                layout, float(t), float(at), max(float(t), KEYFRAME_NOISE_AUG), 1.0
            )
            per_step.append((np.array(distinct), np.array(inverse)))

        table = sorted({float(v) for distinct, _ in per_step for v in distinct})
        lookup = {v: i for i, v in enumerate(table)}
        plan = []
        for distinct, inverse in per_step:
            remap = np.array([lookup[float(v)] for v in distinct], dtype=np.int32)
            plan.append(mx.array(remap[inverse].astype(np.int32)))
        return mx.array(np.array(table, dtype=np.float32)), plan

    def _ensure_cache(self, timesteps: mx.array, drop_adaln: bool, verbose: bool):
        key = tuple(round(float(v), 9) for v in timesteps.tolist())
        if self._cache is not None and self._cache_timesteps == key:
            return
        if self.dit is None:
            raise RuntimeError("Cannot build the AdaLN cache before the transformer is loaded.")
        if getattr(self.dit, "construction_mode", "resident") == "cache_only":
            if not drop_adaln:
                raise RuntimeError(
                    "--keep-adaln is not supported for derived checkpoints: block AdaLN weights "
                    "are stored in sidecars and resident loading is not implemented"
                )
            raise RuntimeError(
                "generation from a derived checkpoint is not supported in v0.3c: streamed AdaLN "
                "sidecar cache construction has not occurred"
            )
        started = time.perf_counter()
        if verbose:
            print(f"  [memory] before AdaLN cache construction: {_memory_snapshot()}", flush=True)
        self._cache = ModulationCache.build(self.dit, timesteps, dtype=mx.bfloat16)
        self._cache_timesteps = key
        self._cache.materialize()
        if verbose:
            print(f"  adaln cache: {len(key)} timesteps, {self._cache.nbytes() / 1e6:.0f} MB "
                  f"in {time.perf_counter() - started:.1f}s")
            print(f"  [memory] after AdaLN cache construction: {_memory_snapshot()}", flush=True)
        if drop_adaln:
            if verbose:
                print(f"  [memory] before AdaLN projection removal: {_memory_snapshot()}", flush=True)
            freed = drop_adaln_weights(self.dit)
            mx.eval(self.dit.parameters())
            self._cache.materialize()
            if verbose:
                print(f"  dropped adaln projections, freeing {freed / 1e9:.1f} GB")
                print(f"  [memory] after AdaLN projection removal: {_memory_snapshot()}", flush=True)
            self._purge_adaln_allocator_cache(verbose)

    # -- keyframe conditioning ----------------------------------------------------------------

    def _encode_keyframes(self, images: list, height: int, width: int, patch_size: tuple[int, int, int]) -> mx.array:
        """Encode ``fl2va`` keyframes into packed conditioning rows.

        Keyframes are single frames, so they go through the video VAE's **spatial** encoder only —
        none of its 17-frame temporal chunking applies. Two details of the reference are load-bearing
        and easy to miss:

        * the posterior is **sampled**, not taken at its mode, under a generator seeded with 42
          independently of the request seed;
        * the sampled latent is **rounded through float16** before normalization, which is about 11
          bits of every conditioning latent — the released model's conditioning cannot be reproduced
          without it.

        MLX's RNG differs from torch's, so the seed-42 draw is not bit-identical to the reference's;
        the distribution and every other step are.
        """
        from .packing import KEYFRAME_ENCODE_SEED, prepare_keyframe_image

        if self.video_vae is None:
            raise RuntimeError("Cannot encode keyframes without a loaded video VAE.")
        cfg = self._video_vae_config or self.video_vae.config
        latents_mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        latents_std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)

        mx.random.seed(KEYFRAME_ENCODE_SEED)
        rows = []
        for index, image in enumerate(images):
            prepared = prepare_keyframe_image(image, height, width, stretch=index == 0)
            pixels = np.asarray(prepared, dtype=np.float32).transpose(2, 0, 1)[None, :, None]
            pixels = (pixels / 255.0 - pixel_mean) / pixel_std

            # (1, 3, 1, H, W) -> channels-last for the spatial encoder.
            moments = self.video_vae._encode_clip(mx.array(pixels).transpose(0, 2, 3, 4, 1))
            channels = cfg.latent_channels
            mean, logvar = moments[..., :channels], moments[..., channels:]
            logvar = mx.clip(logvar, -30.0, 20.0)
            std = mx.exp(0.5 * logvar)
            latent = mean + std * mx.random.normal(mean.shape)
            # -> (1, C, 1, H', W'), then the float16 round trip the reference relies on.
            latent = latent.transpose(0, 4, 1, 2, 3).astype(mx.float16).astype(mx.float32)
            normalized = (latent - latents_mean) / latents_std
            rows.append(patchify_video_latents(normalized, patch_size))
        retained = mx.concatenate(rows)
        mx.eval(retained)
        return retained

    # -- generation ---------------------------------------------------------------------------

    def __call__(
        self,
        prompt: str,
        duration_seconds: float = 5.0,
        aspect: tuple[int, int] = (16, 9),
        megapixels: float | None = None,
        num_inference_steps: int = 16,
        seed: int = 0,
        images: list | None = None,
        keyframe_anchors: tuple[str, ...] = (),
        height: int | None = None,
        width: int | None = None,
        drop_adaln: bool = True,
        verbose: bool = True,
    ) -> GenerationResult:
        """Generate a clip.

        Args:
            duration_seconds: 5 to 15; snapped up to the ``17n + 5`` frame grid the VAE encodes.
            num_inference_steps: the weights are CFG-distilled, so each step is one forward.
            keyframe_anchors: ``"first"`` / ``"last"`` per conditioning keyframe, in packed order.
            height, width: override the canvas ``aspect`` would resolve to. Both must be multiples
                of 32. H3 was released for a 768-pixel short edge only, so anything else is
                off-distribution — useful for exercising the pipeline, not for quality.
        """
        run_started = time.perf_counter()
        self._adaln_purge_attempts = 0
        self._adaln_purge_status = "not-run"

        # 1. Stage-one conditioning. Keyframe vision blocks come back tagged as *video* rows.
        prompt_embeds, text_token_tags = self._prepare_conditioning(prompt, images, verbose)

        self._require_configs()
        dit_config = self._dit_config
        video_config = self._video_vae_config
        audio_config = self._audio_vae_config

        # 2. Geometry comes from lightweight configs, not resident VAE objects.
        if height is None or width is None:
            height, width = resolve_canvas_size(*aspect, megapixels=megapixels)
        elif height % 32 or width % 32:
            raise ValueError(f"`height` and `width` must be multiples of 32, got {height}x{width}.")
        num_frames = align_num_frames(int(round(duration_seconds * FPS)))
        num_latent_frames = video_latent_num_frames(num_frames)
        ratio = video_config.spatial_compression_ratio
        latent_height, latent_width = height // ratio, width // ratio
        num_audio_latents = audio_latent_num_frames(num_frames)
        patch_size = dit_config.patch_size

        layout = build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            keyframe_anchors,
        )
        if verbose:
            print(f"canvas {width}x{height}, {num_frames} frames ({num_latent_frames} latent), "
                  f"{num_audio_latents} audio latents")
            print(f"packed sequence: {layout.sequence_length:,} rows "
                  f"({len(text_token_tags):,} text, {layout.num_condition_video_rows:,} condition)")

        # 3. Image conditioning is the one case that needs a VAE before the transformer. The
        # resulting rows are evaluated before the VAE is released and the transformer is loaded.
        condition_rows = None
        if images:
            try:
                self._load_video_vae(verbose)
                condition_rows = self._encode_keyframes(images, height, width, patch_size)
            except Exception as exc:
                self._release_component("video_vae", "video VAE", verbose, "failure-cleanup")
                raise RuntimeError("Image-conditioning phase failed; no denoising was started.") from exc
            self._release_component("video_vae", "video VAE", verbose)

        # 4. Transformer-only denoising. No decoder is loaded in this phase.
        step_times = []
        try:
            self._load_transformer(verbose)

            # Initial noise draw order matches the reference — conditioning noise first, then
            # video, then audio — so a seed reproduces the same run.
            mx.random.seed(seed)
            if condition_rows is not None:
                condition_noise = mx.random.normal(condition_rows.shape).astype(mx.float32)
                condition_rows = MiniMaxH3Scheduler(shift=self.config.sigma_shift_video).scale_noise(
                    condition_rows, KEYFRAME_NOISE_AUG, condition_noise
                )
                mx.eval(condition_rows)

            latents = mx.random.normal(
                (1, video_config.latent_channels, num_latent_frames, latent_height, latent_width)
            ).astype(mx.float32)
            video_rows = patchify_video_latents(latents, patch_size)
            audio_latents = mx.random.normal(
                (AUDIO_CHANNELS, audio_config.latent_channels, num_audio_latents)
            ).astype(mx.float32)
            audio_rows = pack_audio_latents(audio_latents)
            if condition_rows is not None:
                video_rows = mx.concatenate([condition_rows, video_rows])

            video_sched, audio_sched = self._build_schedules(num_inference_steps)
            timestep_table, plan = self._row_timestep_plan(
                layout, video_sched.timesteps, audio_sched.timesteps
            )
            self._ensure_cache(timestep_table, drop_adaln, verbose)

            n_cond_v = layout.num_condition_video_rows
            n_cond_a = layout.num_condition_audio_rows
            embeds = prompt_embeds.astype(mx.bfloat16)

            denoise_started = time.perf_counter()
            if verbose:
                print(f"  [memory] before denoising: {_memory_snapshot()}", flush=True)
            for i, t in enumerate(video_sched.timesteps.tolist()):
                started = time.perf_counter()
                video_pred, audio_pred = self.dit(
                    video_rows[None].astype(mx.bfloat16),
                    audio_rows[None].astype(mx.bfloat16),
                    embeds,
                    timestep_table,
                    plan[i],
                    layout.token_tags,
                    layout.position_ids,
                    layout.video_indices,
                    layout.audio_indices,
                    layout.text_indices,
                    modulation_cache=self._cache,
                )
                # Rebind rather than assign into a slice: the stepped result is a lazy graph reading
                # the very rows it would overwrite, and the conditioning rows must stay distinct.
                stepped_video = video_sched.step(
                    video_pred[0, n_cond_v:].astype(mx.float32), float(t), video_rows[n_cond_v:]
                )
                stepped_audio = audio_sched.step(
                    audio_pred[0, n_cond_a:].astype(mx.float32),
                    float(audio_sched.timesteps[i].item()),
                    audio_rows[n_cond_a:],
                )
                video_rows = (
                    mx.concatenate([video_rows[:n_cond_v], stepped_video]) if n_cond_v else stepped_video
                )
                audio_rows = (
                    mx.concatenate([audio_rows[:n_cond_a], stepped_audio]) if n_cond_a else stepped_audio
                )
                mx.eval(video_rows, audio_rows)
                step_times.append(time.perf_counter() - started)
                if verbose:
                    done = i + 1
                    mean = sum(step_times) / len(step_times)
                    eta = mean * (len(video_sched.timesteps) - done)
                    print(f"  step {done}/{len(video_sched.timesteps)}  "
                          f"{step_times[-1]:.1f}s  eta {eta / 60:.1f} min", flush=True)
                    if done == 1:
                        print(f"  first denoising step: {step_times[-1]:.1f}s", flush=True)
                        print(f"  [memory] MLX cache after first denoising step: {_memory_snapshot()}", flush=True)
            mx.eval(video_rows, audio_rows)
            if verbose:
                print(f"  denoising: {time.perf_counter() - denoise_started:.1f}s", flush=True)
                later_steps = step_times[1:]
                if later_steps:
                    print(f"  later denoising step mean: {sum(later_steps) / len(later_steps):.1f}s", flush=True)
                print(f"  [memory] after denoising: {_memory_snapshot()}", flush=True)
        except Exception as exc:
            self._release_component("dit", "transformer", verbose, "failure-cleanup")
            raise RuntimeError("Transformer/AdaLN/denoising phase failed; decoders were not loaded.") from exc

        # These arrays are the only denoising outputs needed by the decoder phases. They have been
        # explicitly evaluated while the transformer was still resident.
        final_video_rows = video_rows[n_cond_v:]
        final_audio_rows = audio_rows[n_cond_a:]
        mx.eval(final_video_rows, final_audio_rows)
        # Drop local lazy-graph handles before releasing the module. The retained final arrays have
        # already been evaluated and no longer need prediction temporaries or the prompt cast.
        video_pred = audio_pred = stepped_video = stepped_audio = embeds = None
        self._release_component("dit", "transformer", verbose)

        # 5. Decode video in isolation, then release its VAE before audio loading.
        try:
            self._load_video_vae(verbose)
            video_started = time.perf_counter()
            if verbose:
                print(f"  [memory] before video decoding: {_memory_snapshot()}", flush=True)
            video = self._decode_video(
                final_video_rows, num_latent_frames, latent_height, latent_width, patch_size
            )
            if verbose:
                print(f"  video decoding: {time.perf_counter() - video_started:.1f}s", flush=True)
                print(f"  [memory] after video decoding: {_memory_snapshot()}", flush=True)
        except Exception as exc:
            self._release_component("video_vae", "video VAE", verbose, "failure-cleanup")
            raise RuntimeError("Video decoding phase failed; audio decoding was not started.") from exc
        self._release_component("video_vae", "video VAE", verbose)

        # 6. Decode audio in isolation and retain only the materialized NumPy result.
        try:
            self._load_audio_vae(verbose)
            audio_started = time.perf_counter()
            if verbose:
                print(f"  [memory] before audio decoding: {_memory_snapshot()}", flush=True)
            audio = self._decode_audio(final_audio_rows, num_audio_latents)
            if verbose:
                print(f"  audio decoding: {time.perf_counter() - audio_started:.1f}s", flush=True)
                print(f"  [memory] after audio decoding: {_memory_snapshot()}", flush=True)
        except Exception as exc:
            self._release_component("audio_vae", "audio VAE", verbose, "failure-cleanup")
            raise RuntimeError("Audio decoding phase failed.") from exc
        self._release_component("audio_vae", "audio VAE", verbose)

        total = time.perf_counter() - run_started
        return GenerationResult(
            video=video,
            audio=audio,
            sample_rate=audio_config.sampling_rate,
            seconds_per_step=sum(step_times) / max(len(step_times), 1),
            total_seconds=total,
        )

    # -- decoding -----------------------------------------------------------------------------

    def _decode_video(self, rows, num_latent_frames, latent_height, latent_width, patch_size) -> np.ndarray:
        if self.video_vae is None:
            raise RuntimeError("Cannot decode video without a loaded video VAE.")
        cfg = self._video_vae_config or self.video_vae.config
        latents = unpatchify_video_tokens(
            rows, num_latent_frames, latent_height, latent_width, cfg.latent_channels, patch_size
        )
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1, 1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1, 1, 1)
        latents = latents * std + mean

        frames = np.array(self.video_vae.decode(latents.astype(mx.float32)))
        # The VAE decodes into ImageNet-normalized RGB over a [0, 1] base range.
        pixel_mean = np.array(PIXEL_MEAN, np.float32).reshape(1, 3, 1, 1, 1)
        pixel_std = np.array(PIXEL_STD, np.float32).reshape(1, 3, 1, 1, 1)
        frames = frames * pixel_std + pixel_mean
        frames = np.clip(frames, 0.0, 1.0)[0].transpose(1, 2, 3, 0)  # -> (F, H, W, 3)
        return (frames * 255.0 + 0.5).astype(np.uint8)

    def _decode_audio(self, rows, num_audio_latents) -> np.ndarray:
        if self.audio_vae is None:
            raise RuntimeError("Cannot decode audio without a loaded audio VAE.")
        cfg = self._audio_vae_config or self.audio_vae.config
        latents = unpack_audio_tokens(rows, num_audio_latents)
        mean = mx.array(np.array(cfg.latents_mean, np.float32)).reshape(1, -1, 1)
        std = mx.array(np.array(cfg.latents_std, np.float32)).reshape(1, -1, 1)
        latents = latents * std + mean
        waveform = np.array(self.audio_vae.decode(latents.astype(mx.float32)))
        return waveform[:, 0, :].astype(np.float32)  # (2, samples), one row per stereo channel

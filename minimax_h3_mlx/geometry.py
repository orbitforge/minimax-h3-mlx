"""MLX-free production multimodal geometry for the decoder bridge."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import prod
from numbers import Integral
from typing import Any

from .video_decode_layout import VideoDecodeLayout


@dataclass(frozen=True)
class ProductionMultimodalGeometry:
    """The one canonical v0.5b geometry contract."""

    video_width: int
    video_height: int
    video_fps: int
    video_frames: int
    video_duration: Fraction
    video_latent_shape: tuple[int, int, int, int, int]
    audio_sample_rate: int
    audio_latent_rate: int
    audio_samples: int
    audio_duration: Fraction
    audio_latent_shape: tuple[int, int, int]
    video_patch_size: tuple[int, int, int]
    video_patch_width: int
    audio_patch_width: int
    video_token_count: int
    audio_token_count: int
    total_token_count: int
    video_output_shape: tuple[int, int, int, int, int]
    rgb_media_shape: tuple[int, int, int, int]
    audio_raw_shape: tuple[int, int, int]
    waveform_shape: tuple[int, int]
    alignment_evidence: tuple[str, ...]
    # The production v0.5b bridge has no text rows in its standalone packing contract.  The
    # v0.5d/e core contract supplies the locked 103 text rows explicitly while reusing this same
    # geometry object rather than introducing a second geometry type.
    text_token_count: int = 0

    @classmethod
    def canonical(
        cls,
        video_config: Any,
        audio_config: Any,
        dit_config: Any,
        video_layout: VideoDecodeLayout,
        *,
        width: int = 128,
        height: int = 128,
        text_token_count: int = 0,
    ) -> "ProductionMultimodalGeometry":
        if isinstance(width, bool) or not isinstance(width, Integral):
            raise ValueError("canonical video width must be an integer")
        if isinstance(height, bool) or not isinstance(height, Integral):
            raise ValueError("canonical video height must be an integer")
        if isinstance(text_token_count, bool) or not isinstance(text_token_count, Integral):
            raise ValueError("canonical text row count must be an integer")
        width = int(width)
        height = int(height)
        text_token_count = int(text_token_count)
        if width != height:
            raise ValueError(f"canonical video geometry must be square, got {width}x{height}")
        if width < 128 or height < 128:
            raise ValueError("canonical video geometry must be at least 128x128")
        if text_token_count < 0:
            raise ValueError("canonical text row count must be nonnegative")
        fps = 24
        frames = 30
        video_channels = int(video_config.latent_channels)
        spatial_ratio = int(video_config.spatial_compression_ratio)
        if spatial_ratio <= 0:
            raise ValueError("video spatial compression ratio must be positive")
        if width % spatial_ratio or height % spatial_ratio:
            raise ValueError(
                f"canonical video dimensions {width}x{height} are not divisible by the VAE "
                f"spatial compression ratio {spatial_ratio}"
            )
        latent_height = height // spatial_ratio
        latent_width = width // spatial_ratio
        audio_rate = int(audio_config.sampling_rate)
        audio_hop = int(audio_config.hop_length)
        latent_rate = audio_rate // audio_hop
        audio_latents = 50
        audio_samples = audio_latents * audio_hop
        audio_channels = int(audio_config.latent_channels)
        patch = tuple(int(value) for value in dit_config.patch_size)
        rows_per_frame = (latent_height // patch[1]) * (latent_width // patch[2])
        latent_frames = 9
        video_tokens = latent_frames
        video_tokens *= rows_per_frame
        audio_tokens = 2 * audio_latents
        geometry = cls(
            video_width=width,
            video_height=height,
            video_fps=fps,
            video_frames=frames,
            video_duration=Fraction(frames, fps),
            video_latent_shape=(1, video_channels, latent_frames, latent_height, latent_width),
            audio_sample_rate=audio_rate,
            audio_latent_rate=latent_rate,
            audio_samples=audio_samples,
            audio_duration=Fraction(audio_samples, audio_rate),
            audio_latent_shape=(2, audio_channels, audio_latents),
            video_patch_size=patch,
            video_patch_width=video_channels * prod(patch),
            audio_patch_width=audio_channels,
            video_token_count=video_tokens,
            audio_token_count=audio_tokens,
            total_token_count=video_tokens + audio_tokens,
            video_output_shape=(1, int(video_config.out_channels), frames, height, width),
            rgb_media_shape=(frames, height, width, int(video_config.out_channels)),
            audio_raw_shape=(2, 1, audio_samples),
            waveform_shape=(2, audio_samples),
            alignment_evidence=(
                f"{frames}/{fps} = {Fraction(frames, fps)} seconds",
                f"{audio_samples}/{audio_rate} = {Fraction(audio_samples, audio_rate)} seconds",
                f"{audio_latents}/{latent_rate} = {Fraction(audio_latents, latent_rate)} seconds",
                f"video decoder F={latent_frames} -> {video_decode_frame_count(latent_frames, video_layout)} frames",
                f"spatial {latent_height}x{latent_width} * {spatial_ratio} = {height}x{width}",
            ),
            text_token_count=text_token_count,
        )
        geometry.validate(video_config, audio_config, video_layout)
        return geometry

    @property
    def duration_seconds(self) -> float:
        return float(self.video_duration)

    @property
    def text_row_range(self) -> tuple[int, int]:
        return (0, self.text_token_count)

    @property
    def audio_row_range(self) -> tuple[int, int]:
        start = self.text_token_count
        return (start, start + self.audio_token_count)

    @property
    def video_row_range(self) -> tuple[int, int]:
        start = self.text_token_count + self.audio_token_count
        return (start, start + self.video_token_count)

    @property
    def total_packed_rows(self) -> int:
        return self.text_token_count + self.audio_token_count + self.video_token_count

    @property
    def video_rows(self) -> int:
        return self.video_token_count

    @property
    def audio_rows(self) -> int:
        return self.audio_token_count

    @property
    def position_ids_shape(self) -> tuple[int, int]:
        return (self.total_packed_rows, 3)

    @property
    def token_tags_shape(self) -> tuple[int]:
        return (self.total_packed_rows,)

    @property
    def video_indices_shape(self) -> tuple[int]:
        return (self.video_rows,)

    @property
    def audio_indices_shape(self) -> tuple[int]:
        return (self.audio_rows,)

    @property
    def text_indices_shape(self) -> tuple[int]:
        return (self.text_token_count,)

    def validate(self, video_config: Any, audio_config: Any, layout: VideoDecodeLayout) -> None:
        if self.video_duration != self.audio_duration:
            raise ValueError("video and audio durations are not exactly aligned")
        if self.video_width != self.video_height or self.video_width < 128:
            raise ValueError("canonical video geometry must be square and at least 128 pixels")
        if self.text_token_count < 0:
            raise ValueError("canonical text row count must be nonnegative")
        spatial_ratio = int(video_config.spatial_compression_ratio)
        if self.video_width % spatial_ratio or self.video_height % spatial_ratio:
            raise ValueError("canonical video dimensions are not divisible by the VAE spatial compression ratio")
        if self.video_latent_shape[2] < layout.minimum_latent_frames:
            raise ValueError("canonical video latent frame count is below the decoder minimum")
        pt, ph, pw = self.video_patch_size
        if any(value % patch for value, patch in zip(self.video_latent_shape[2:], (pt, ph, pw))):
            raise ValueError("canonical video latent geometry is not divisible by the DiT patch")
        if self.audio_latent_shape[0] != 2:
            raise ValueError("audio must remain two mono batch items")
        if int(audio_config.sampling_rate) % int(audio_config.hop_length):
            raise ValueError("audio sample rate must divide evenly by the native hop length")
        if prod(tuple(int(rate) for rate in audio_config.decoder_rates)) != int(audio_config.hop_length):
            raise ValueError("audio decoder rates do not equal the native hop length")
        if self.video_patch_width != int(video_config.latent_channels) * prod(self.video_patch_size):
            raise ValueError("video packed feature width does not match the DiT source contract")
        if self.audio_patch_width != int(audio_config.latent_channels):
            raise ValueError("audio packed feature width does not match the DiT source contract")
        if self.video_frames != video_decode_frame_count(self.video_latent_shape[2], layout):
            raise ValueError("canonical video latent count does not decode to 30 frames")


def video_decode_frame_count(latent_frames: int, layout: VideoDecodeLayout) -> int:
    """Mirror the executable VideoVAE.decode chunk/padding/tail-trim arithmetic."""
    if latent_frames <= 0:
        raise ValueError("video latent frame count must be positive")
    num_tokens = latent_frames + layout.token_drop
    pad_tokens = (-num_tokens) % layout.tokens_chunk_size
    num_chunks = (num_tokens + pad_tokens) // layout.tokens_chunk_size - int(layout.token_drop > 0)
    if num_chunks < 1:
        raise ValueError("video latent frame count is below the decoder minimum")
    frames = num_chunks * (layout.chunk_num_frames - layout.frame_pre_padding) + layout.frame_overlap
    if pad_tokens:
        before_pad = latent_frames
        frames -= sum(
            layout.tail_trim_remainder
            if layout.tail_trim_remainder and (before_pad + offset) % layout.tokens_chunk_size == 0
            else layout.temporal_compression_ratio
            for offset in range(pad_tokens)
        )
    return int(frames)

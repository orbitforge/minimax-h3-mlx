"""MLX-free temporal layout derived by the MiniMax-H3 video decoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VideoDecodeLayout:
    """Immutable temporal decode geometry shared by config-only contracts and the decoder."""

    clip_length: int
    temporal_compression_ratio: int
    tokens_chunk_size: int
    token_drop: int
    token_overlap: int
    frame_pre_padding: int
    frame_overlap: int
    chunk_num_frames: int
    tail_trim_remainder: int
    minimum_latent_frames: int


def _required_int(config: Any, field_name: str) -> int:
    try:
        value = getattr(config, field_name)
    except AttributeError as exc:
        raise ValueError(
            f"VideoVAEConfig is missing required source field: {field_name}"
        ) from exc
    if isinstance(value, bool):
        raise ValueError(f"VideoVAEConfig source field {field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"VideoVAEConfig source field {field_name} must be an integer"
        ) from exc


def resolve_video_decode_layout(config: Any) -> VideoDecodeLayout:
    """Resolve the decoder's temporal formulas from authentic config fields only.

    The formulas mirror ``VideoVAE.__init__`` and ``VideoVAE.decode``. In particular, the
    values returned here are not expected to be present on ``VideoVAEConfig`` itself.
    """

    clip_length = _required_int(config, "clip_length")
    ratio_t = _required_int(config, "temporal_compression_ratio")
    token_drop = _required_int(config, "token_drop")
    if clip_length <= 0:
        raise ValueError("VideoVAEConfig source field clip_length must be positive")
    if ratio_t <= 0:
        raise ValueError(
            "VideoVAEConfig source field temporal_compression_ratio must be positive"
        )
    if token_drop < 0:
        raise ValueError("VideoVAEConfig source field token_drop must be nonnegative")

    tokens_chunk_size = math.ceil(clip_length / ratio_t)
    frame_pre_padding = (-clip_length) % ratio_t
    token_overlap = (-token_drop) % tokens_chunk_size
    frame_overlap = max(token_overlap * ratio_t - frame_pre_padding, 0)
    chunk_num_frames = tokens_chunk_size * ratio_t
    tail_trim_remainder = clip_length % ratio_t
    minimum_latent_frames = 2 * tokens_chunk_size - token_drop
    if minimum_latent_frames <= 0:
        raise ValueError(
            "VideoVAEConfig source field token_drop leaves no positive decoder minimum"
        )

    return VideoDecodeLayout(
        clip_length=clip_length,
        temporal_compression_ratio=ratio_t,
        tokens_chunk_size=tokens_chunk_size,
        token_drop=token_drop,
        token_overlap=token_overlap,
        frame_pre_padding=frame_pre_padding,
        frame_overlap=frame_overlap,
        chunk_num_frames=chunk_num_frames,
        tail_trim_remainder=tail_trim_remainder,
        minimum_latent_frames=minimum_latent_frames,
    )

"""Precomputed AdaLN modulation.

~13B of MiniMax-H3's 33B parameters live in the per-block ``adaln_proj.linear`` projections
(50 blocks x ``[96768, 2688]``). Their input is the timestep embedding alone — it does not depend
on the packed sequence — so for a fixed sampler schedule every modulation tensor the whole
denoising run will ever need can be computed once, up front, and the projections then dropped.
The model card notes this explicitly: *"Because the AdaLN modulation outputs can be precomputed
and cached, these parameters do not need to be loaded for inference-only deployment."*

The saving is large and lopsided. For a 40-step schedule the cache holds

    50 blocks x 6 tensors x (40 timesteps x 3 modalities) x 5376  =  193M values

which is ~387 MB in bfloat16, against ~26 GB for the bfloat16 projections it replaces — a ~67x
reduction, and the resident DiT drops from 33B to ~20B parameters.

Precision note: the cache is built from the float32 timestep embedding through the *unquantized*
projections. Every block reads the same ``temb``, so an error introduced here biases every block's
modulation identically at every sampling step and accumulates coherently along the denoising
trajectory. Building the table before quantization keeps that path exact.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .config import MODALITY_NUM
from .dit import timestep_embedding


def _stable_timestep_values(values: mx.array) -> list[float]:
    return [round(float(value), 9) for value in values.tolist()]


def _is_integer_dtype(dtype: mx.Dtype) -> bool:
    return dtype in {
        getattr(mx, name)
        for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")
        if hasattr(mx, name)
    }


def _is_modulation_dtype(dtype: mx.Dtype) -> bool:
    return dtype in {
        getattr(mx, name)
        for name in ("float16", "bfloat16", "float32")
        if hasattr(mx, name)
    }


class ModulationCache:
    """Per-block AdaLN modulation, precomputed for a fixed set of distinct timesteps.

    ``timesteps`` must be the union of every distinct noise level the schedule will present to the
    transformer — target video, target audio and any conditioning levels. During sampling the
    packed sequence's ``timestep_indices`` index into exactly this array.
    """

    def __init__(self, tables: list[tuple[mx.array, ...]], timesteps: mx.array):
        self.tables = tables
        self.timesteps = timesteps

    def get(self, block_index: int) -> tuple[mx.array, ...]:
        return self.tables[block_index]

    def validate_for(
        self,
        config,
        requested_timesteps: mx.array,
        timestep_indices: mx.array,
        packed_sequence_length: int,
    ) -> None:
        """Validate the complete cache boundary before a cache-only forward.

        This intentionally checks structure and host-visible metadata only. It does not evaluate
        the retained arrays, and v0.3c never constructs or reads sidecar files.
        """
        if not isinstance(self.timesteps, mx.array) or self.timesteps.ndim != 1:
            raise ValueError("modulation cache timesteps must be a one-dimensional MLX array")
        if self.num_timesteps == 0:
            raise ValueError("modulation cache must contain at least one timestep")
        if not isinstance(requested_timesteps, mx.array) or requested_timesteps.ndim != 1:
            raise ValueError("requested transformer timesteps must be a one-dimensional MLX array")
        if not isinstance(timestep_indices, mx.array) or timestep_indices.ndim != 1:
            raise ValueError("transformer timestep_indices must be a one-dimensional MLX array")
        if not _is_integer_dtype(timestep_indices.dtype):
            raise ValueError("transformer timestep_indices must have an integer dtype")
        if timestep_indices.shape[0] != packed_sequence_length:
            raise ValueError(
                "transformer timestep_indices count must equal the packed sequence length: "
                f"got {timestep_indices.shape[0]}, expected {packed_sequence_length}"
            )

        if len(self.tables) != config.num_layers:
            raise ValueError(
                f"modulation cache is incomplete: expected {config.num_layers} block entries, "
                f"got {len(self.tables)}"
            )
        expected_shape = (self.num_timesteps * MODALITY_NUM, config.hidden_size)
        modulation_dtype: mx.Dtype | None = None
        for block_index, table in enumerate(self.tables):
            if not isinstance(table, tuple) or len(table) != 6:
                raise ValueError(
                    f"modulation cache block {block_index} must contain six modulation tensors"
                )
            for tensor_index, tensor in enumerate(table):
                if not isinstance(tensor, mx.array) or tensor.shape != expected_shape:
                    actual = getattr(tensor, "shape", None)
                    raise ValueError(
                        f"modulation cache block {block_index} tensor {tensor_index} has shape "
                        f"{actual}; expected {expected_shape}"
                    )
                if not _is_modulation_dtype(tensor.dtype):
                    raise ValueError(
                        f"modulation cache block {block_index} tensor {tensor_index} has dtype "
                        f"{tensor.dtype}; expected float16, bfloat16, or float32"
                    )
                if modulation_dtype is None:
                    modulation_dtype = tensor.dtype
                elif tensor.dtype != modulation_dtype:
                    raise ValueError("modulation cache arrays must use one consistent floating dtype")

        try:
            cached = _stable_timestep_values(self.timesteps)
            requested = _stable_timestep_values(requested_timesteps)
            indices = timestep_indices.tolist()
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("modulation cache metadata is not host-readable") from exc
        if len(requested) != len(cached):
            raise ValueError(
                "requested timetable count must equal cached timetable count: "
                f"got {len(requested)}, expected {len(cached)}"
            )
        if len(set(cached)) != len(cached):
            raise ValueError("modulation cache timetable contains duplicate entries")
        if requested != cached:
            raise ValueError("requested timetable must exactly match cached timetable in order")
        if any(index < 0 or index >= self.num_timesteps for index in indices):
            raise ValueError("modulation cache does not contain every requested timestep index")

    @property
    def num_timesteps(self) -> int:
        return int(self.timesteps.shape[0])

    def nbytes(self) -> int:
        return sum(t.nbytes for table in self.tables for t in table)

    def materialize(self) -> None:
        """Evaluate every array retained by the denoising cache."""
        arrays = [self.timesteps]
        arrays.extend(tensor for table in self.tables for tensor in table)
        mx.eval(*arrays)

    @classmethod
    def build(
        cls,
        dit,
        timesteps: mx.array,
        dtype: mx.Dtype = mx.bfloat16,
    ) -> "ModulationCache":
        """Precompute the modulation table for every block.

        Args:
            dit: a ``MiniMaxH3DiT`` whose ``adaln_proj`` weights are still loaded.
            timesteps: ``(T,)`` the distinct noise levels, unscaled in ``[0, 1]``.
            dtype: storage dtype of the cache. bfloat16 halves the footprint and matches the
                precision the modulation is consumed at inside the block stack.
        """
        temb = dit.time_embedder(timestep_embedding(timesteps, dit.config.timestep_input_dim))
        mx.eval(temb)

        tables: list[tuple[mx.array, ...]] = []
        for block in dit.blocks:
            table = tuple(t.astype(dtype) for t in block.adaln_proj(temb))
            mx.eval(table)
            tables.append(table)
        return cls(tables, timesteps)


def final_layer_modulation(dit, timesteps: mx.array, dtype: mx.Dtype = mx.bfloat16):
    """Precompute the output layer's ``shift``/``scale`` table.

    ``final_layer.adaln_proj`` is only ``[10752, 2688]`` (~29M params), so caching it is about
    avoiding a redundant projection per step rather than about memory.
    """
    temb = dit.time_embedder(timestep_embedding(timesteps, dit.config.timestep_input_dim))
    linear = dit.final_layer.adaln_proj.linear
    h = linear(nn.silu(temb).astype(linear.weight.dtype))
    hidden = dit.config.hidden_size
    shift, scale = h[..., :hidden], h[..., hidden:]
    shift, scale = shift.astype(dtype), scale.astype(dtype)
    mx.eval(shift, scale)
    return shift, scale


def drop_adaln_weights(dit) -> int:
    """Delete the per-block ``adaln_proj`` projections after a cache has been built.

    Returns the number of **bytes** freed. Only safe once a :class:`ModulationCache` covering the
    whole schedule exists — the block stack will raise if asked to modulate without one.

    A quantized projection carries ``scales`` and ``biases`` alongside its packed ``weight``;
    dropping only ``weight`` and ``bias`` would strand them and under-report the saving, so every
    array the layer holds is removed.
    """
    freed = 0
    for block in dit.blocks:
        linear = getattr(block.adaln_proj, "linear", None)
        if linear is None:
            continue
        for name in ("weight", "bias", "scales", "biases"):
            param = getattr(linear, name, None)
            if isinstance(param, mx.array):
                freed += param.nbytes
                delattr(linear, name)
    return freed


def schedule_timesteps(sigmas: mx.array, conditioning_level: float = 0.0) -> mx.array:
    """Build the union of distinct timesteps a full denoising run will present.

    Conditioning rows (reference image, first/last frame, clean audio) sit at a fixed clean noise
    level, so the union is the schedule's own sigmas plus that level.
    """
    values = mx.concatenate([mx.array([conditioning_level], dtype=sigmas.dtype), sigmas])
    # `mx.unique` is not available; sort and de-duplicate on the host, the array is tiny.
    host = sorted({round(float(v), 9) for v in values.tolist()})
    return mx.array(host, dtype=mx.float32)


def modality_row(timestep_index: int, tag: int) -> int:
    """Row of the modulation table addressed by a ``(timestep, modality)`` pair."""
    return timestep_index * MODALITY_NUM + max(tag, 0)

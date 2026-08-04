"""Build the complete v0.3d AdaLN cache and print sequential-release telemetry.

Run this script from an external Terminal.  It loads only the derived transformer base and AdaLN
sidecars; it does not load Qwen or either VAE and never enters denoising, decoding, or rendering.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mlx.core as mx
import numpy as np

from minimax_h3_mlx.config import MODALITY_NUM
from minimax_h3_mlx.load import load_dit
from minimax_h3_mlx.packing import KEYFRAME_NOISE_AUG, build_packed_sequence, build_row_timesteps
from minimax_h3_mlx.scheduler import MiniMaxH3Scheduler
from minimax_h3_mlx.streamed_adaln import build_streamed_modulation_cache


DEFAULT_CHECKPOINT = "/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln"
EXPECTED_TIMESTEPS = 77
EXPECTED_CACHE_BYTES = 745_113_600


def snapshot() -> dict[str, int | None]:
    result: dict[str, int | None] = {}
    for label, name in (
        ("active_memory", "get_active_memory"),
        ("allocator_cache", "get_cache_memory"),
        ("peak_memory", "get_peak_memory"),
    ):
        getter = getattr(mx, name, None)
        result[label] = int(getter()) if callable(getter) else None
    return result


def canonical_timetable(video_timesteps: mx.array, audio_timesteps: mx.array) -> mx.array:
    # This is the same per-step row assignment and sorted global union as Pipeline._row_timestep_plan.
    # One synthetic keyframe row supplies the 0.999 conditioning level required by that runtime path;
    # it does not load or execute a VAE.
    layout = build_packed_sequence(
        np.array([1], dtype=np.int64),
        num_latent_frames=1,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=(1, 1, 1),
        keyframe_anchors=("first",),
    )
    per_step = []
    for video_timestep, audio_timestep in zip(video_timesteps.tolist(), audio_timesteps.tolist()):
        distinct, _ = build_row_timesteps(
            layout,
            float(video_timestep),
            float(audio_timestep),
            max(float(video_timestep), KEYFRAME_NOISE_AUG),
            1.0,
        )
        per_step.append(distinct)
    values = sorted({float(value) for distinct in per_step for value in distinct.tolist()})
    return mx.array(values, dtype=mx.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    started = time.perf_counter()
    reset_peak = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak):
        reset_peak()
    print(f"before_derived_base_load: elapsed=0.000s memory={snapshot()}", flush=True)

    def base_telemetry(stage: str, _model, _format_info) -> None:
        print(f"base:{stage}: elapsed={time.perf_counter() - started:.3f}s memory={snapshot()}", flush=True)

    dit = load_dit(args.checkpoint, verbose=True, telemetry=base_telemetry)
    print(f"after_derived_base_evaluation: elapsed={time.perf_counter() - started:.3f}s memory={snapshot()}", flush=True)

    video_schedule = MiniMaxH3Scheduler(shift=12.0)
    audio_schedule = MiniMaxH3Scheduler(shift=3.0)
    video_schedule.set_timesteps(40)
    audio_schedule.set_timesteps(40)
    timetable = canonical_timetable(video_schedule.timesteps, audio_schedule.timesteps)
    if timetable.shape[0] != EXPECTED_TIMESTEPS:
        raise AssertionError(f"expected {EXPECTED_TIMESTEPS} distinct timetable entries, got {timetable.shape[0]}")
    print(f"video_schedule_entries={len(video_schedule.timesteps)}", flush=True)
    print(f"audio_schedule_entries={len(audio_schedule.timesteps)}", flush=True)
    print(f"global_timetable={timetable.tolist()}", flush=True)
    print(f"global_timetable_count={timetable.shape[0]}", flush=True)

    def cache_telemetry(event: str, details: dict) -> None:
        memory = details.get("memory", snapshot())
        if event == "shared_timestep_embedding_materialized":
            print(f"after_shared_timestep_embedding: elapsed={time.perf_counter() - started:.3f}s memory={memory}", flush=True)
        elif event == "sidecar_opening":
            print(f"sidecar_opening block={details['block_index']} path={details['path']} memory={memory}", flush=True)
        elif event == "sidecar_released":
            print(f"sidecar_released block={details['block_index']} purge_succeeded={details['purge_succeeded']} memory={memory}", flush=True)

    cache, stats = build_streamed_modulation_cache(dit, timetable, telemetry=cache_telemetry)
    if stats.final_cache_bytes != EXPECTED_CACHE_BYTES:
        raise AssertionError(f"expected final cache size {EXPECTED_CACHE_BYTES}, got {stats.final_cache_bytes}")
    if stats.timetable_count != EXPECTED_TIMESTEPS or stats.blocks_completed != 50:
        raise AssertionError(f"unexpected cache dimensions: {stats}")
    if stats.dense_temporary_projection_created is not False:
        raise AssertionError("built-in packed projection did not report dense_temporary_projection_created=False")

    for block in stats.per_block:
        print(
            "block_telemetry="
            f"{{'block_index': {block.block_index}, 'sidecar_filename': {block.sidecar_filename!r}, "
            f"'sidecar_logical_bytes': {block.sidecar_logical_bytes}, "
            f"'active_before_load': {block.active_before_load}, "
            f"'allocator_cache_before_load': {block.allocator_cache_before_load}, "
            f"'active_after_sidecar_materialization': {block.active_after_sidecar_materialization}, "
            f"'allocator_cache_after_sidecar_materialization': {block.allocator_cache_after_sidecar_materialization}, "
            f"'active_after_modulation_table_materialization': {block.active_after_modulation_materialization}, "
            f"'allocator_cache_after_modulation_table_materialization': {block.allocator_cache_after_modulation_materialization}, "
            f"'cumulative_retained_cache_bytes': {block.cumulative_retained_cache_bytes}, "
            f"'active_after_sidecar_release_and_purge': {block.active_after_release_and_purge}, "
            f"'allocator_cache_after_sidecar_release_and_purge': {block.allocator_cache_after_release_and_purge}, "
            f"'elapsed_seconds': {block.elapsed_seconds:.3f}}}",
            flush=True,
        )

    print(f"completed_block_count={stats.blocks_completed}", flush=True)
    print(f"sidecar_files_opened={stats.sidecar_files_opened}", flush=True)
    print(f"unique_sidecar_files_opened={stats.unique_sidecar_files_opened}", flush=True)
    print(f"final_cache_bytes={stats.final_cache_bytes}", flush=True)
    print(f"timetable_count={stats.timetable_count}", flush=True)
    print(f"table_count={len(cache.tables)}", flush=True)
    print(f"tensors_per_table={len(cache.tables[0])}", flush=True)
    print(f"table_shape={cache.tables[0][0].shape}", flush=True)
    print(f"storage_dtype={stats.storage_dtype}", flush=True)
    print(f"peak_active_mlx_memory={stats.peak_mlx_active_memory}", flush=True)
    print(f"peak_allocator_cache={stats.peak_allocator_cache}", flush=True)
    print(f"total_sidecar_bytes_processed={stats.sidecar_logical_bytes_processed}", flush=True)
    print(f"successful_payload_opens={stats.successful_payload_opens}", flush=True)
    print(f"completed_payload_releases={stats.completed_payload_releases}", flush=True)
    print(f"sidecar_overlap_observed={stats.sidecar_overlap_observed}", flush=True)
    print(f"next_sidecar_opened_before_previous_release={stats.next_sidecar_opened_before_previous_release}", flush=True)
    print(f"maximum_one_block_active_memory_increase={stats.maximum_one_block_active_memory_increase}", flush=True)
    print(f"every_sidecar_released_before_next_opened={stats.every_sidecar_released_before_next_opened}", flush=True)
    print(f"dense_temporary_projection_created={stats.dense_temporary_projection_created}", flush=True)
    print(f"build_elapsed_seconds={stats.elapsed_total_seconds:.3f}", flush=True)

    release_started = time.perf_counter()
    del cache
    del dit
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    gc.collect()
    print(f"after_cache_and_transformer_release: elapsed={time.perf_counter() - release_started:.3f}s memory={snapshot()}", flush=True)
    print(f"total_elapsed_seconds={time.perf_counter() - started:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load only the complete v0.3c derived transformer base and print Metal telemetry.

This probe never constructs Qwen or either VAE, never builds a modulation cache, and never enters
the denoising or decoding path.
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.load import CheckpointFormatInfo, load_dit


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default="/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln",
    )
    args = parser.parse_args()
    started = time.perf_counter()
    events: list[tuple[str, float, dict[str, int | None]]] = []
    opened: list[str] = []

    def record(stage: str, model, _info: CheckpointFormatInfo) -> None:
        events.append((stage, time.perf_counter() - started, snapshot()))
        print(f"{stage}: elapsed={events[-1][1]:.3f}s memory={events[-1][2]}", flush=True)

    def load_base_shard(path: str):
        opened.append(path)
        if "/adaln/" in path or path.endswith("/adaln"):
            raise AssertionError(f"sidecar payload was opened: {path}")
        return mx.load(path)

    mx.reset_peak_memory()
    print(f"before_model_construction: elapsed=0.000s memory={snapshot()}", flush=True)
    model = load_dit(
        args.checkpoint,
        verbose=True,
        telemetry=record,
        tensor_loader=load_base_shard,
    )
    stats = model.load_stats
    validation_started = time.perf_counter()
    keys = {key for key, _ in tree_flatten(model.parameters())}
    block_adaln_keys = sorted(key for key in keys if key.startswith("blocks.") and ".adaln_proj." in key)
    final_weight = "final_layer.adaln_proj.linear.weight" in keys
    final_bias = "final_layer.adaln_proj.linear.bias" in keys
    ordinary_proof = all(
        any(key.startswith(prefix) for prefix in ("blocks.", "token_refiner.", "video_patch_proj.", "audio_patch_proj.", "condition_proj.", "time_embedder.", "final_layer."))
        for key in keys
    )
    format_info = model.checkpoint_format_info
    if format_info.checkpoint_format != "derived":
        raise AssertionError(f"expected derived checkpoint, got {format_info.checkpoint_format!r}")
    attention_presence = [
        any(key.startswith(f"blocks.{index}.attn.") for key in keys)
        for index in range(len(model.blocks))
    ]
    feed_forward_presence = [
        any(key.startswith(f"blocks.{index}.mlp.") for key in keys)
        for index in range(len(model.blocks))
    ]
    block_adaln_absence = not any(
        key.startswith("blocks.") and ".adaln_proj." in key for key in keys
    )
    if not all(attention_presence):
        raise AssertionError("every transformer block must retain attention parameters")
    if not all(feed_forward_presence):
        raise AssertionError("every transformer block must retain feed-forward parameters")
    if not block_adaln_absence:
        raise AssertionError("cache-only base must not retain block-level AdaLN parameters")
    if not final_weight or not final_bias:
        raise AssertionError("final-layer AdaLN weight and bias must remain resident")
    if len(opened) != 5:
        raise AssertionError(f"expected exactly five base payload files, opened {len(opened)}")
    if any("/adaln/" in path for path in opened):
        raise AssertionError("an AdaLN sidecar payload was opened")
    if stats.loaded_tensor_count != 850:
        raise AssertionError(f"expected 850 loaded tensors, got {stats.loaded_tensor_count}")
    if stats.loaded_logical_bytes != 16_464_048_640:
        raise AssertionError(
            f"expected 16,464,048,640 loaded logical bytes, got {stats.loaded_logical_bytes}"
        )

    print(f"transformer_blocks={len(model.blocks)}", flush=True)
    print(f"block_attention_parameters_present={all(attention_presence)}", flush=True)
    print(f"block_feed_forward_parameters_present={all(feed_forward_presence)}", flush=True)
    print(f"block_adaln_parameter_keys={block_adaln_keys}", flush=True)
    print(f"block_adaln_parameters_absent={block_adaln_absence}", flush=True)
    print(f"final_layer_adaln_weight_present={final_weight}", flush=True)
    print(f"final_layer_adaln_bias_present={final_bias}", flush=True)
    print(f"ordinary_parameter_tree_present={ordinary_proof}", flush=True)
    print(f"base_payload_paths_opened={opened}", flush=True)
    print(f"base_payload_files_opened={len(opened)}", flush=True)
    print(f"sidecar_payload_opened={any('/adaln/' in path for path in opened)}", flush=True)
    print(f"sidecar_payload_files_opened={sum('/adaln/' in path for path in opened)}", flush=True)

    params = tree_flatten(model.parameters())
    print(f"loaded_tensor_count={stats.loaded_tensor_count}", flush=True)
    print(f"loaded_logical_bytes={stats.loaded_logical_bytes}", flush=True)
    print(f"parameter_tensor_count={len(params)}", flush=True)
    print(f"parameter_element_count={sum(value.size for _, value in params)}", flush=True)
    event_times = {stage: elapsed for stage, elapsed, _ in events}
    # The assertions above are deliberately performed after evaluation and before release. This
    # interval is the actual model-tree validation phase, not a cumulative process timestamp.
    validation_elapsed = time.perf_counter() - validation_started
    phase_durations = {
        "format_inspection": event_times["format_inspected"],
        "model_construction": event_times["model_constructed"] - event_times["format_inspected"],
        "quantized_structure_setup": event_times["quantized_structure_ready"] - event_times["model_constructed"],
        "base_shard_loading_and_attachment": event_times["base_weights_attached"] - event_times["quantized_structure_ready"],
        "parameter_evaluation": event_times["base_parameters_evaluated"] - event_times["base_weights_attached"],
        "model_tree_validation": validation_elapsed,
    }
    print(f"peak_memory_before_release={snapshot()['peak_memory']}", flush=True)

    release_started = time.perf_counter()
    del params
    del model
    gc.collect()
    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
    gc.collect()
    release_elapsed = time.perf_counter() - release_started
    total_elapsed = time.perf_counter() - started
    phase_durations["release_and_purge"] = release_elapsed
    phase_durations["total"] = total_elapsed
    print(f"phase_durations={phase_durations}", flush=True)
    print(f"after_model_release_gc_allocator_purge: elapsed={total_elapsed:.3f}s memory={snapshot()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

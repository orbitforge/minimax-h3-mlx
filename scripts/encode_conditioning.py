"""Encode one text-only prompt into a replayable MiniMax-H3 artifact.

This is intentionally a conditioner-only process.  It constructs Qwen, writes
the canonical CPU-readable artifact, releases Qwen, purges the MLX allocator,
and exits without importing the H3 transformer or either VAE loader.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.conditioning_artifact import (  # noqa: E402
    create_conditioning_artifact,
    build_encoder_provenance,
)


DEFAULT_CHECKPOINT = os.environ.get("H3_CHECKPOINT_ROOT", "/Volumes/models/MiniMax-H3/FL2VA")


def _memory_snapshot(mx) -> str:
    parts: list[str] = []
    for label, name in (("active", "get_active_memory"), ("cache", "get_cache_memory"), ("peak", "get_peak_memory")):
        getter = getattr(mx, name, None)
        try:
            if callable(getter):
                parts.append(f"{label}={int(getter())}")
        except Exception:
            continue
    return " ".join(parts) or "unavailable"


def _purge_allocator(mx) -> str:
    clear_cache = getattr(mx, "clear_cache", None)
    if not callable(clear_cache):
        return "unavailable"
    try:
        clear_cache()
    except Exception:
        return "failed"
    return "success"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="the exact text prompt to encode")
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="the H3 checkpoint root containing text_encoder/ (or H3_CHECKPOINT_ROOT)",
    )
    parser.add_argument("--artifact", required=True, help="new conditioning-artifact.npz path")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    artifact_path = Path(args.artifact).expanduser().resolve()
    started = time.perf_counter()
    mx = None
    encoder = None
    conditioning = None
    input_ids = token_tags = encoded_tags = vision_inputs = None
    primary: BaseException | None = None
    artifact = None

    try:
        import mlx.core as mx_module

        mx = mx_module
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

        print("conditioning source: live Qwen", flush=True)
        print(f"text encoder load: {checkpoint / 'text_encoder'}", flush=True)
        encoder = MiniMaxH3TextEncoder(
            checkpoint / "text_encoder",
            dtype=mx.bfloat16,
            load_vision=False,
            verbose=args.verbose,
        )
        print("text encoder loaded: Qwen3-VL layer-50 truncated path", flush=True)

        input_ids, token_tags, vision_inputs = encoder.build_request(args.prompt, None)
        if vision_inputs is not None:
            raise RuntimeError("encode-only conditioning accepts text-only requests; image state is outside this artifact")
        conditioning, encoded_tags = encoder.encode(args.prompt, None)
        if not np.array_equal(np.asarray(token_tags), np.asarray(encoded_tags)):
            raise RuntimeError("token tags changed between request construction and Qwen encoding")
        mx.eval(conditioning, input_ids)
        print("prompt encoding: complete", flush=True)

        provenance = build_encoder_provenance(checkpoint)
        artifact = create_conditioning_artifact(
            artifact_path,
            prompt=args.prompt,
            conditioning=conditioning,
            text_token_tags=np.asarray(token_tags, dtype=np.int32),
            token_ids=np.asarray(input_ids, dtype=np.int32),
            encoder_provenance=provenance,
        )
        print(f"artifact path: {artifact.path}", flush=True)
        print(
            "artifact tensor: "
            f"shape={artifact.metadata['conditioning']['shape']} "
            f"dtype={artifact.metadata['conditioning']['dtype']} "
            f"checksum={artifact.tensor_checksum}",
            flush=True,
        )
        print(f"artifact identity: {artifact.artifact_identity}", flush=True)
        print(
            "text encoder provenance: "
            + json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    except BaseException as exc:
        primary = exc
        print(f"encode-only failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    finally:
        release_status = "complete" if encoder is not None else "not-loaded"
        input_ids = token_tags = encoded_tags = vision_inputs = conditioning = None
        encoder = None
        gc.collect()
        print(f"text encoder release: {release_status}", flush=True)
        if mx is not None:
            print(f"allocator purge: {_purge_allocator(mx)}", flush=True)
            print(f"allocator telemetry after purge: {_memory_snapshot(mx)}", flush=True)

    if primary is not None:
        print("clean process exit: failed", file=sys.stderr, flush=True)
        return 1
    print(f"clean process exit: success ({time.perf_counter() - started:.3f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

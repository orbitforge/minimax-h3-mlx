"""Bounded forensic probe for MiniMax-H3 still-image conditioning.

This probe loads only the Qwen text/vision conditioner, using the exact image and
prompt recorded by an immutable Render Lab run.  It never loads the H3 DiT,
denoises, loads either VAE, decodes media, or publishes a render.

The output namespace is immutable: an existing output directory is rejected.
The probe records structural metadata and small fingerprints only; it does not
serialize model activations or tensor payloads.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import json
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    return _sha256_bytes(array.tobytes())


def _shape_dtype(value: Any) -> dict[str, Any]:
    return {"shape": list(value.shape), "dtype": str(value.dtype)}


def _ranges(indices: Any) -> list[list[int]]:
    values = [int(item) for item in indices]
    if not values:
        return []
    result: list[list[int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append([start, previous])
            start = value
        previous = value
    result.append([start, previous])
    return result


def _first_last(values: Any, count: int = 8) -> dict[str, list[int]]:
    items = [int(item) for item in values]
    return {"first": items[:count], "last": items[-count:]}


def _safe_scalar(value: Any) -> Any:
    try:
        return value.item()
    except AttributeError:
        return value


def _json_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _ForbiddenImportGuard:
    """Prove the bounded path does not import torch or torchvision."""

    def __init__(self) -> None:
        self.original = builtins.__import__
        self.attempts: list[str] = []

    def __call__(self, name: str, *args: Any, **kwargs: Any) -> Any:
        root_name = name.split(".", 1)[0]
        if root_name in {"torch", "torchvision"}:
            self.attempts.append(name)
            raise ModuleNotFoundError(f"probe-blocked import: {name}")
        return self.original(name, *args, **kwargs)

    def install(self) -> None:
        builtins.__import__ = self

    def uninstall(self) -> None:
        builtins.__import__ = self.original


def _load_run(run_dir: Path) -> tuple[dict[str, Any], Path, Path, str]:
    config_path = run_dir / "render-config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing immutable Render Lab config: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    entries = config.get("input_images")
    if not isinstance(entries, list) or len(entries) != 1:
        raise ValueError(f"expected exactly one recorded input image, got {entries!r}")
    entry = entries[0]
    image_path = Path(entry["path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"recorded source image is unavailable: {image_path}")
    actual_hash = _sha256_bytes(image_path.read_bytes())
    if actual_hash != entry.get("sha256"):
        raise ValueError(
            f"recorded source image hash mismatch: expected {entry.get('sha256')}, got {actual_hash}"
        )
    checkpoint_root = Path(config["checkpoint_root"])
    if not checkpoint_root.is_dir():
        raise FileNotFoundError(f"recorded checkpoint root is unavailable: {checkpoint_root}")
    return config, image_path, checkpoint_root, actual_hash


def _build_token_metadata(encoder: Any, input_ids: Any, token_tags: Any, grid_np: Any) -> dict[str, Any]:
    import numpy as np

    ids_np = np.asarray(input_ids.tolist(), dtype=np.int64)
    tags_np = np.asarray(token_tags, dtype=np.int64)
    if ids_np.ndim != 2 or ids_np.shape[0] != 1:
        raise ValueError(f"bounded probe expects one input row, got input_ids shape {ids_np.shape}")
    image_mask_np = ids_np == encoder.image_token_id
    visual_positions = np.flatnonzero(image_mask_np[0])
    counts = Counter(int(value) for value in ids_np[0].tolist())
    tag_counts = Counter(int(value) for value in tags_np.tolist())
    return {
        "input_ids": {
            "shape": list(ids_np.shape),
            "dtype": str(ids_np.dtype),
            "total_token_count": int(ids_np.size),
            "sha256": _sha256_array(ids_np),
            "first_ids": [int(value) for value in ids_np[0, :12].tolist()],
            "last_ids": [int(value) for value in ids_np[0, -12:].tolist()],
        },
        "token_ids": {
            "image_token_id": int(encoder.image_token_id),
            "video_token_id": int(encoder.model_config.video_token_id),
            "vision_start_token_id": int(encoder.vision_start_token_id),
            "vision_end_token_id": int(encoder.vision_end_token_id),
            "count_by_relevant_id": {
                "image_pad": int(counts.get(int(encoder.image_token_id), 0)),
                "video_pad": int(counts.get(int(encoder.model_config.video_token_id), 0)),
                "vision_start": int(counts.get(int(encoder.vision_start_token_id), 0)),
                "vision_end": int(counts.get(int(encoder.vision_end_token_id), 0)),
            },
        },
        "image_mask": {
            "shape": list(image_mask_np.shape),
            "sum": int(image_mask_np.sum()),
            "selected_visual_indices": _first_last(visual_positions),
            "selected_visual_ranges_in_full_sequence": _ranges(visual_positions),
            "is_contiguous": len(_ranges(visual_positions)) <= 1,
        },
        "token_tags": {
            "shape": list(tags_np.shape),
            "dtype": str(tags_np.dtype),
            "sha256": _sha256_array(tags_np),
            "count_by_tag": {str(key): int(value) for key, value in sorted(tag_counts.items())},
        },
        "image_grid_thw": {
            "shape": list(grid_np.shape),
            "dtype": str(grid_np.dtype),
            "values": grid_np.tolist(),
            "sha256": _sha256_array(grid_np),
            "merge_size": int(encoder.merge_size),
            "per_image_visual_rows": [
                int(np.prod(row)) // int(encoder.merge_size) ** 2 for row in grid_np
            ],
        },
    }


def _two_image_tokenization(encoder: Any, prompt: str, image: Any) -> dict[str, Any]:
    """Exercise multiple image-grid presentation without a second vision forward."""

    import numpy as np

    input_ids, token_tags, vision_inputs = encoder.build_request(prompt, [image, image.copy()])
    pixel_values, grid_np = vision_inputs
    ids_np = np.asarray(input_ids.tolist(), dtype=np.int64)
    mask_np = ids_np == encoder.image_token_id
    positions = np.flatnonzero(mask_np[0])
    per_image_rows = [int(np.prod(row)) // int(encoder.merge_size) ** 2 for row in grid_np]
    ranges: list[list[int]] = []
    offset = 0
    for rows in per_image_rows:
        image_positions = positions[offset : offset + rows]
        ranges.append([int(image_positions[0]), int(image_positions[-1])])
        offset += rows
    return {
        "image_count": 2,
        "uses_exact_source_image_twice_for_structure_only": True,
        "pixel_values": _shape_dtype(np.asarray(pixel_values)),
        "image_grid_thw": grid_np.tolist(),
        "per_image_visual_rows": per_image_rows,
        "total_visual_rows": int(positions.size),
        "input_ids_shape": list(ids_np.shape),
        "total_token_count": int(ids_np.size),
        "image_mask_sum": int(mask_np.sum()),
        "visual_ranges_in_full_sequence": ranges,
        "token_tags_shape": list(np.asarray(token_tags).shape),
        "ordering_contract": "image 1 rows then image 2 rows, matching image_grid_thw order",
    }


def _probe(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    config, image_path, checkpoint_root, image_hash = _load_run(Path(args.run_dir).resolve())
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    source_size = list(image.size)

    guard = _ForbiddenImportGuard()
    guard.install()
    try:
        import mlx.core as mx

        from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel
        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

        encoder = MiniMaxH3TextEncoder(
            checkpoint_root / "text_encoder",
            dtype=mx.bfloat16,
            load_vision=True,
            verbose=True,
        )
        prompt = config["prompt"]
        input_ids, token_tags, vision_inputs = encoder.build_request(prompt, [image])
        if vision_inputs is None:
            raise ValueError("exact failed I2V request unexpectedly produced no vision inputs")
        pixel_values, grid_np = vision_inputs
        grid_np = np.asarray(grid_np, dtype=np.int32)
        token_metadata = _build_token_metadata(encoder, input_ids, token_tags, grid_np)
        mask_np = np.asarray(input_ids.tolist(), dtype=np.int64) == encoder.image_token_id
        image_mask = input_ids == encoder.image_token_id

        grid_thw = mx.array(grid_np)
        pixel_values_mx = mx.array(pixel_values).astype(encoder.dtype)
        hidden, deepstack_embeds = encoder.vision(
            pixel_values_mx,
            grid_thw,
            output_hidden_states=True,
        )
        deepstack_embeds = list(deepstack_embeds)
        mx.eval(hidden, *deepstack_embeds)

        vision_rows = int(hidden.shape[0])
        visual_mask_rows = int(mask_np.sum())
        deepstack_shapes = [_shape_dtype(value) for value in deepstack_embeds]
        deepstack_rows = [int(value.shape[0]) for value in deepstack_embeds]
        deepstack_widths = [int(value.shape[1]) for value in deepstack_embeds]

        base = encoder.language.embed_tokens(input_ids)
        mx.eval(base)
        hidden_for_merge = hidden.astype(base.dtype)

        current_merge: dict[str, Any]
        try:
            current_candidate = mx.where(image_mask[..., None], hidden_for_merge[None], base)
            mx.eval(current_candidate)
            current_merge = {"status": "succeeded", "shape": list(current_candidate.shape)}
        except Exception as exc:  # noqa: BLE001 - the exception is evidence
            current_merge = {
                "status": "failed",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
            }

        reference_merged, reference_mask_expanded = Qwen3VLModel.merge_input_ids_with_image_features(
            hidden_for_merge,
            base,
            input_ids,
            encoder.image_token_id,
            encoder.model_config.video_token_id,
        )
        mx.eval(reference_merged, reference_mask_expanded)

        selected_indices = np.flatnonzero(mask_np[0])
        selected_mx = mx.array(selected_indices.astype(np.uint32))
        merged_visual = reference_merged[0][selected_mx]
        visual_delta = mx.max(mx.abs(merged_visual - hidden_for_merge))
        expanded_mask = mx.broadcast_to(image_mask[..., None], base.shape)
        non_visual_delta = mx.max(
            mx.where(expanded_mask, mx.zeros_like(base), mx.abs(reference_merged - base))
        )
        mx.eval(visual_delta, non_visual_delta)

        report = {
            "schema_version": 1,
            "probe": {
                "name": "i2v-conditioning-merge",
                "mode": "bounded_actual_checkpoint_text_vision_only",
                "run_directory": str(Path(args.run_dir).resolve()),
                "output_namespace": str(output_root),
                "did_not_load": ["H3 DiT transformer", "video VAE", "audio VAE"],
                "did_not_run": ["denoising", "media decode", "ffmpeg", "full H3 render"],
            },
            "source": {
                "image_path": str(image_path),
                "sha256": image_hash,
                "width": source_size[0],
                "height": source_size[1],
                "mode": image.mode,
                "requested_output_width": int(config["width"]),
                "requested_output_height": int(config["height"]),
                "generation_mode": config.get("generation_mode"),
                "anchor": config["image_anchors"][0],
            },
            "checkpoint": {
                "root": str(checkpoint_root),
                "text_encoder": str(checkpoint_root / "text_encoder"),
                "transformer_from_run": config.get("transformer_path"),
            },
            "environment": {
                "transformers_version": "5.14.1",
                "mlx_vlm_version": "0.6.9",
                "torch_or_torchvision_import_attempts": guard.attempts,
                "torch_or_torchvision_unused": not guard.attempts,
                "loaded_python_modules_include_dit_or_vae": any(
                    name.startswith(("diffusers", "minimax_h3_mlx.dit", "minimax_h3_mlx.video_vae", "minimax_h3_mlx.audio_vae"))
                    for name in sys.modules
                ),
            },
            "request": token_metadata,
            "vision": {
                "pixel_values": _shape_dtype(np.asarray(pixel_values)),
                "pixel_values_sha256": _sha256_array(pixel_values),
                "raw_patch_row_count": int(np.asarray(pixel_values).shape[0]),
                "merged_hidden": _shape_dtype(hidden),
                "merged_hidden_row_count": vision_rows,
                "merged_hidden_width": int(hidden.shape[1]),
                "merged_hidden_finite": bool(_safe_scalar(mx.all(mx.isfinite(hidden)))),
                "deepstack_count": len(deepstack_embeds),
                "deepstack": deepstack_shapes,
                "deepstack_row_counts": deepstack_rows,
                "deepstack_widths": deepstack_widths,
                "deepstack_rows_match_initial_visual_rows": all(row == vision_rows for row in deepstack_rows),
                "deepstack_ordering_contract": "each deepstack merger consumes the same flattened vision row order as the initial merger",
            },
            "merge": {
                "full_inputs_embeds": _shape_dtype(base),
                "visual_mask_shape": list(mask_np.shape),
                "visual_mask_selected_row_count": visual_mask_rows,
                "visual_hidden_row_count": vision_rows,
                "visual_mask_equals_hidden_rows": visual_mask_rows == vision_rows,
                "visual_mask_selected_indices": _first_last(selected_indices),
                "visual_mask_ranges": _ranges(selected_indices),
                "current_mx_where": current_merge,
                "authoritative_mlx_qwen3_vl_operation": "flatten expanded mask and embeddings, scatter compact visual rows into true positions, reshape",
                "authoritative_reference_merge_status": "succeeded",
                "authoritative_reference_merged_shape": list(reference_merged.shape),
                "authoritative_reference_mask_shape": list(reference_mask_expanded.shape),
                "reference_visual_rows_equal_hidden": bool(_safe_scalar(visual_delta) == 0),
                "reference_non_visual_rows_preserved": bool(_safe_scalar(non_visual_delta) == 0),
                "reference_visual_max_abs_delta": float(_safe_scalar(visual_delta)),
                "reference_non_visual_max_abs_delta": float(_safe_scalar(non_visual_delta)),
            },
            "multi_image_structural_extension": _two_image_tokenization(encoder, prompt, image),
            "classification": (
                "MERGE_OPERATION_BUG"
                if visual_mask_rows == vision_rows
                and current_merge.get("status") == "failed"
                and bool(_safe_scalar(visual_delta) == 0)
                and bool(_safe_scalar(non_visual_delta) == 0)
                else "INSUFFICIENT_EVIDENCE"
            ),
        }
        return report
    finally:
        guard.uninstall()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=str(ROOT / "out/render-lab/run-20260813T134202Z-808ada8668"),
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "out/v0.6/i2v-conditioning-inspection-01"),
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing immutable output namespace: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    status_path = output_root / "probe-status.json"
    _json_write(status_path, {"status": "started", "run_dir": str(Path(args.run_dir).resolve())})
    try:
        report = _probe(args, output_root)
        _json_write(output_root / "conditioning-report.json", report)
        _json_write(status_path, {"status": "succeeded", "classification": report["classification"]})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:  # preserve a diagnostic receipt for ordinary Python failures
        failure = {
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
        }
        _json_write(output_root / "failure-report.json", failure)
        _json_write(status_path, {"status": "failed", "exception_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

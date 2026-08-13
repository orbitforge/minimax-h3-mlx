"""One-shot bounded actual-checkpoint H3 text/vision encode smoke."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.probe_i2v_conditioning_merge import (
    ROOT,
    _ForbiddenImportGuard,
    _json_write,
    _load_run,
    _sha256_array,
)


def _scalar(value: Any) -> Any:
    try:
        return value.item()
    except AttributeError:
        return value


def _main(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    import numpy as np

    config, image_path, checkpoint_root, image_hash = _load_run(Path(args.run_dir).resolve())
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")

    guard = _ForbiddenImportGuard()
    guard.install()
    try:
        import mlx.core as mx

        from minimax_h3_mlx.text_encoder import MiniMaxH3TextEncoder

        encoder = MiniMaxH3TextEncoder(
            checkpoint_root / "text_encoder",
            dtype=mx.bfloat16,
            load_vision=True,
            verbose=True,
        )
        prompt = config["prompt"]

        # Build once for structural receipt fields.  This is preprocessing/tokenization only;
        # the actual vision and layer-50 forward happen exactly once inside encode().
        input_ids, token_tags, vision_inputs = encoder.build_request(prompt, [image])
        if vision_inputs is None:
            raise ValueError("exact failed I2V request unexpectedly produced no vision inputs")
        pixel_values, grid_np = vision_inputs
        grid_np = np.asarray(grid_np, dtype=np.int32)
        ids_np = np.asarray(input_ids.tolist(), dtype=np.int64)
        tags_np = np.asarray(token_tags, dtype=np.int64)
        image_mask_np = ids_np == encoder.image_token_id
        expected_visual_rows = sum(
            int(np.prod(row)) // int(encoder.merge_size) ** 2 for row in grid_np
        )

        conditioning, encoded_tags = encoder.encode(prompt, [image])
        mx.eval(conditioning)
        encoded_tags_np = np.asarray(encoded_tags, dtype=np.int64)
        if not np.array_equal(tags_np, encoded_tags_np):
            raise ValueError("encode returned token tags different from the request presentation")

        try:
            peak_memory = _scalar(mx.get_peak_memory())
        except AttributeError:
            peak_memory = None

        report = {
            "schema_version": 1,
            "probe": {
                "name": "i2v-conditioning-encode-smoke",
                "mode": "bounded_actual_checkpoint_text_vision_layer50_only",
                "run_directory": str(Path(args.run_dir).resolve()),
                "output_namespace": str(output_root),
                "conditioning_output": "MiniMaxH3TextEncoder.encode layer-50 pre-final-norm output",
                "did_not_load": ["H3 DiT transformer", "video VAE", "audio VAE"],
                "did_not_run": ["denoising", "media decode", "ffmpeg", "full H3 render"],
            },
            "source": {
                "image_path": str(image_path),
                "sha256": image_hash,
                "width": image.width,
                "height": image.height,
                "requested_output_width": int(config["width"]),
                "requested_output_height": int(config["height"]),
                "anchor": config["image_anchors"][0],
            },
            "checkpoint": {
                "root": str(checkpoint_root),
                "text_encoder": str(checkpoint_root / "text_encoder"),
            },
            "environment": {
                "torch_or_torchvision_import_attempts": guard.attempts,
                "torch_or_torchvision_unused": not guard.attempts,
                "loaded_python_modules_include_dit_or_vae": any(
                    name.startswith(("diffusers", "minimax_h3_mlx.dit", "minimax_h3_mlx.video_vae", "minimax_h3_mlx.audio_vae"))
                    for name in sys.modules
                ),
                "peak_mlx_memory_bytes": peak_memory,
            },
            "request": {
                "input_ids_shape": list(ids_np.shape),
                "input_ids_total_token_count": int(ids_np.size),
                "input_ids_sha256": _sha256_array(ids_np),
                "image_grid_thw": grid_np.tolist(),
                "image_grid_thw_sha256": _sha256_array(grid_np),
                "pixel_values_shape": list(np.asarray(pixel_values).shape),
                "pixel_values_dtype": str(np.asarray(pixel_values).dtype),
                "image_mask_shape": list(image_mask_np.shape),
                "image_mask_sum": int(image_mask_np.sum()),
                "visual_hidden_row_count_expected_from_grid": expected_visual_rows,
                "token_tags_shape": list(tags_np.shape),
                "token_tags_sha256": _sha256_array(tags_np),
                "token_tag_counts": {str(k): int(v) for k, v in sorted(Counter(tags_np.tolist()).items())},
            },
            "conditioning": {
                "output_shape": list(conditioning.shape),
                "output_dtype": str(conditioning.dtype),
                "output_finite": bool(_scalar(mx.all(mx.isfinite(conditioning)))),
                "merge_exception": None,
                "proof_layer50_conditioning_returned": True,
            },
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
        default=str(ROOT / "out/v0.6/i2v-conditioning-encode-smoke-01"),
    )
    args = parser.parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise SystemExit(f"refusing to overwrite existing immutable output namespace: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    status_path = output_root / "probe-status.json"
    _json_write(status_path, {"status": "started", "run_dir": str(Path(args.run_dir).resolve())})
    try:
        report = _main(args, output_root)
        _json_write(output_root / "conditioning-encode-report.json", report)
        _json_write(status_path, {"status": "succeeded"})
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except BaseException as exc:
        _json_write(
            output_root / "failure-report.json",
            {"status": "failed", "exception_type": type(exc).__name__, "exception": str(exc)},
        )
        _json_write(status_path, {"status": "failed", "exception_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate a MiniMax-H3 clip on Apple Silicon.

    ./.venv/bin/python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4

The CLI defaults to a 0.2 MP canvas (608x352 at 16:9), which is a safer starting point on smaller
Apple Silicon machines. Read the performance section of the README before selecting the released
768-pixel canvas: a step there is ~8.8 minutes for a 5 s clip on an M3 Ultra because MiniMax has
not released its sparse-attention implementation. `--megapixels` or `--height/--width` can select
larger canvases explicitly; anything below the 768-pixel training canvas is off-distribution.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.media import FFmpegUnavailableError, save_frames, save_mp4, save_wav
from minimax_h3_mlx.lora import (
    LIGHTX_DEFAULT_VARIANT,
    LIGHTX_TASK_REF2VA,
    LIGHTX_VARIANTS,
)
from minimax_h3_mlx.pipeline import MiniMaxH3Pipeline
from minimax_h3_mlx.transformer_routing import (
    REF2VA_REFERENCE_INPUT_NOT_IMPLEMENTED,
    TransformerRoutingError,
    resolve_manifest_transformer,
)

DEFAULT_CHECKPOINT = "/Volumes/models/MiniMax-H3/FL2VA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="the live prompt; omit it when --conditioning-artifact supplies the prompt identity",
    )
    parser.add_argument("-o", "--output", default="out.mp4")
    parser.add_argument("-c", "--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="the upstream release; supplies the VAEs and text encoder")
    parser.add_argument("-t", "--transformer", default=None,
                        help="a quantized transformer directory to use instead of the release's")
    parser.add_argument("-d", "--duration", type=float, default=5.0, help="seconds, 5 to 15")
    parser.add_argument("-s", "--steps", type=int, default=None,
                        help="transformer evaluations (NFE); normal default 16, Turbo default 8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aspect", type=int, nargs=2, default=(16, 9))
    parser.add_argument("--megapixels", type=float, default=0.2,
                        help="default canvas area in megapixels (default: 0.2; ignored with height/width)")
    parser.add_argument("--height", type=int, default=None, help="canvas override, multiple of 32")
    parser.add_argument("--width", type=int, default=None, help="canvas override, multiple of 32")
    parser.add_argument("--image", action="append", default=None, help="keyframe image (repeatable)")
    parser.add_argument("--anchor", action="append", default=None, choices=["first", "last"],
                        help="anchor for each --image, in order")
    parser.add_argument("--keep-adaln", action="store_true",
                        help="keep the 13B adaln_proj resident instead of caching and dropping it")
    parser.add_argument("--keep-text-encoder", action="store_true",
                        help="keep the Qwen conditioner resident after prompt encoding")
    parser.add_argument(
        "--conditioning-artifact",
        default=None,
        help="validated text-only conditioning-artifact.npz; skips Qwen construction and encoding",
    )
    parser.add_argument("--lora", dest="lora_path", default=None,
                        help="generic LoRA safetensors adapter to apply around H3 projections")
    parser.add_argument("--turbo-lora", dest="turbo_lora_path", default=None,
                        help="Turbo LoRA safetensors adapter; enables the reduced-step schedule")
    parser.add_argument(
        "--lightx-lora",
        "--lightx",
        dest="lightx_path",
        default=None,
        help="native LightX2V adapter; binds its explicit manifest configuration",
    )
    parser.add_argument(
        "--lightx-variant",
        choices=tuple(LIGHTX_VARIANTS),
        default=None,
        help=(
            "explicit native LightX2V variant (default: "
            f"{LIGHTX_DEFAULT_VARIANT}; requires --lightx-lora)"
        ),
    )
    parser.add_argument("--lora-scale", type=float, default=1.0,
                        help="global LoRA multiplier (default: 1.0)")
    parser.add_argument("--turbo", action="store_true",
                        help="use the Turbo schedule; requires an attached Turbo or LightX adapter")
    parser.add_argument("--turbo-steps", type=int, default=None,
                        help="override the adapter Turbo NFE (default: adapter metadata or 8)")
    args = parser.parse_args()

    if args.conditioning_artifact is None and args.prompt is None:
        parser.error("prompt is required unless --conditioning-artifact is supplied")
    if args.conditioning_artifact is not None and args.image:
        parser.error("--conditioning-artifact is text-only; image/VAE conditioning cannot be replayed")
    if args.conditioning_artifact is not None and args.keep_text_encoder:
        parser.error("--keep-text-encoder is incompatible with --conditioning-artifact")

    selected_adapters = [
        name
        for name, value in (
            ("--lora", args.lora_path),
            ("--turbo-lora", args.turbo_lora_path),
            ("--lightx-lora", args.lightx_path),
        )
        if value
    ]
    if len(selected_adapters) > 1:
        parser.error("use exactly one adapter path: " + ", ".join(selected_adapters))
    lightx_manifest = None
    if args.lightx_path:
        variant_name = args.lightx_variant or LIGHTX_DEFAULT_VARIANT
        lightx_manifest = LIGHTX_VARIANTS[variant_name]
        if args.lora_scale != lightx_manifest.runtime_scale_default:
            parser.error(
                "native LightX2V production requires --lora-scale "
                f"{lightx_manifest.runtime_scale_default:g}"
            )
        if args.steps not in (None, lightx_manifest.nfe):
            parser.error(
                f"native LightX2V {variant_name} requires --steps {lightx_manifest.nfe}"
            )
        if args.turbo_steps not in (None, lightx_manifest.nfe):
            parser.error(
                f"native LightX2V {variant_name} requires --turbo-steps {lightx_manifest.nfe}"
            )
    elif args.lightx_variant:
        parser.error("--lightx-variant requires --lightx-lora")
    turbo = args.turbo or bool(args.turbo_lora_path) or bool(args.lightx_path)
    lora_path = args.turbo_lora_path or args.lora_path
    if turbo and lora_path is None and args.lightx_path is None:
        parser.error("--turbo requires --turbo-lora (or a generic --lora adapter)")

    if lightx_manifest is not None and lightx_manifest.task == LIGHTX_TASK_REF2VA:
        try:
            resolve_manifest_transformer(
                lightx_manifest,
                args.checkpoint,
                explicit_transformer_dir=args.transformer,
            )
        except TransformerRoutingError as exc:
            parser.error(str(exc))
        # The route is deliberately checked before this stop.  No reference conditioning,
        # transformer construction, or ordinary-transformer fallback is admitted in this slice.
        parser.error(REF2VA_REFERENCE_INPUT_NOT_IMPLEMENTED)

    images = None
    if args.image:
        from PIL import Image, ImageOps

        images = [ImageOps.exif_transpose(Image.open(p).convert("RGB")) for p in args.image]
    anchors = tuple(args.anchor or ())
    if images and len(anchors) != len(images):
        parser.error(f"--anchor must be given once per --image ({len(images)} images, {len(anchors)} anchors)")

    pipe = MiniMaxH3Pipeline.from_pretrained(
        args.checkpoint,
        transformer_dir=args.transformer,
        load_vision=bool(images),
        unload_text_encoder=not args.keep_text_encoder,
        keep_adaln=args.keep_adaln,
        lora_path=lora_path,
        lightx_path=args.lightx_path,
        lightx_manifest=lightx_manifest,
        lora_scale=args.lora_scale,
        turbo=turbo,
        turbo_steps=args.turbo_steps,
        conditioning_artifact=args.conditioning_artifact,
    )
    result = pipe(
        args.prompt,
        duration_seconds=args.duration,
        aspect=tuple(args.aspect),
        megapixels=args.megapixels,
        num_inference_steps=args.steps,
        seed=args.seed,
        images=images,
        keyframe_anchors=anchors,
        height=args.height,
        width=args.width,
        drop_adaln=not args.keep_adaln,
        turbo=turbo,
        turbo_steps=args.turbo_steps,
    )

    output = Path(args.output)
    try:
        save_mp4(output, result.video, result.fps, result.audio, result.sample_rate)
        print(f"\nwrote {output} ({result.video.shape[0]} frames, "
              f"{result.audio.shape[-1] / result.sample_rate:.2f}s audio)")
    except FFmpegUnavailableError as exc:
        print(f"\nffmpeg unavailable ({exc}); writing frames and wav instead")
        save_frames(output.with_suffix(""), result.video)
        save_wav(output.with_suffix(".wav"), result.audio, result.sample_rate)

    print(f"{result.seconds_per_step:.1f}s per step, {result.total_seconds / 60:.1f} min total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

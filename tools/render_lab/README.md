# MiniMax H3 Render Lab

This is a loopback-only operator surface around the existing `scripts/generate.py` CLI. It does not
contain model or generation mathematics. Each accepted Render action reserves a fresh
`out/render-lab/run-*` namespace, stages uploaded images there, launches one child process, and
keeps the raw logs plus structured config, benchmark, status, and telemetry records.

## Start

From the repository root:

```sh
./.venv/bin/python tools/render_lab/server.py
```

Add `--open` to open the browser automatically. For this checkout, the canonical runtime paths
are:

```sh
H3_CHECKPOINT_ROOT=/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va \
H3_TRANSFORMER=/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit \
./.venv/bin/python tools/render_lab/server.py --open
```

On another machine, set `H3_CHECKPOINT_ROOT` and `H3_TRANSFORMER` to readable equivalents. Do not
substitute the obsolete upstream FL2VA path recorded by one preserved failed run.

Streamlit is not installed in this checkout. The standard-library browser surface follows the
existing `scripts/minimax_h3_surface.py` pattern and keeps the runtime environment unchanged.

## Resolution source

The selector is populated only from `resolutions.py`. Render admission re-resolves the selected
aspect/megapixel specification through `minimax_h3_mlx.packing.resolve_canvas_size` before launching
H3 and rejects drift. The approved choices are:

| Preset | Basis |
|---|---|
| 128 × 128 | Preserved successful v0.5d full-schedule MP4 proof |
| 256 × 256 | Preserved successful v0.5e quality/resource MP4 proof |
| 608 × 352 | Current documented 0.2 MP CLI default; smoke target, not frozen proof |
| 1344 × 768 | Successful real 5 s render on the released 768-pixel canvas |

The UI also displays the H3 spatial latent grid and spatial token count using the current VAE
compression ratio and DiT patch contract when those values are available from the repository.

## Image-conditioning policy

I2V and first/last-frame inputs are resized directly to the selected conditioning canvas with
deterministic LANCZOS resampling. The runtime intentionally does not crop, letterbox, or add dead
space, so an aspect-mismatched source may be visibly distorted. Callers who need composition
preserved must supply an aspect-matched reference image.

Each accepted run retains its immutable `render-config.json`, input hashes, raw logs, benchmark,
status, telemetry, and published media under `out/render-lab/run-*`.

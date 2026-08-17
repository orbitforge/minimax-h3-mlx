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
H3_TRANSFORMER=/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln \
./.venv/bin/python tools/render_lab/server.py --open
```

On another machine, set `H3_CHECKPOINT_ROOT` and `H3_TRANSFORMER` to readable equivalents. Do not
substitute the obsolete upstream FL2VA path recorded by one preserved failed run.

Streamlit is not installed in this checkout. The standard-library browser surface follows the
existing `scripts/minimax_h3_surface.py` pattern and keeps the runtime environment unchanged.

## Independent resolution controls

The browser exposes independent width and height controls. Each slider moves in 32-pixel
increments, while the paired numeric fields accept exact values for validation. Render admission
requires both dimensions to be positive, between 128 and 1344 pixels, and divisible by 32. The
legacy `resolution_id` request field and curated preset catalog remain accepted for compatibility;
legacy non-aligned display presets continue to use their recorded runtime-aligned dimensions. The
legacy catalog includes:

| Preset | Basis |
|---|---|
| 128 × 128 | Preserved successful v0.5d full-schedule MP4 proof |
| 256 × 256 | Preserved successful v0.5e quality/resource MP4 proof |
| 608 × 352 | Current documented 0.2 MP CLI default; smoke target, not frozen proof |
| 1344 × 768 | Successful real 5 s render on the released 768-pixel canvas |

The UI also displays the H3 spatial latent grid and spatial token count using the current VAE
compression ratio and DiT patch contract when those values are available from the repository.

## Curated Turbo preset surface

The browser defaults to `None / Reference`, preserving normal generation and manual generic LoRA
behavior. The Turbo selector exposes exactly these validated production presets:

| Preset | Role | Family | NFE | Logical asset | Runtime command contract |
|---|---|---|---:|---|---|
| LightX 4-Step v0.1 | Fast | LightX2V | 4 | `lightx_v01_4step` | `--lightx-lora` + `--lightx-variant fl2va-turbo-4step-v0.1` |
| LightX 8-Step v1.0 | Quality / General | LightX2V | 8 | `lightx_v10_8step` | `--lightx-lora` + `--lightx-variant fl2va-turbo-8step-v1.0` |
| LightX 4-Step v1.0 768p | High Resolution | LightX2V | 4 | `lightx_v10_768p` | `--lightx-lora` + `--lightx-variant fl2va-turbo-4step-v1.0-768p` |
| Larry v4 Step 600 | Quality / General | Larry | 8 | `larry_v4` | `--turbo-lora` |
| Larry 850 | Fast Motion | Larry | 4 | `larry_850` | `--turbo-lora` |

The selected preset owns its adapter path, adapter family, runtime variant, fixed scale, scheduler
semantics, and NFE. The Render Lab emits the matching `--steps` and `--turbo-steps` values and
rejects mismatched manual requests. The 768p preset displays `1344 × 768 native/recommended` but
does not silently change geometry.

The normal Render Lab transformer is the canonical streamed-AdaLN Q6 directory
`minimax-h3-mlx-6bit-streamed-adaln`. The repository has no host asset manifest, so the five
curated adapter paths resolve narrowly under the checkout's sibling `work/models` directory and
the missing-manifest dependency is recorded in config/run evidence. No `/Volumes/models` fallback
is used by the Render Lab.

## Text-encoder policy

The selector exposes exactly two choices: `Canonical Qwen3-VL` (the default, available for T2V,
I2V, and first/last-frame generation) and `Heretic 35B-A3B · Experimental`. Heretic is T2V-only;
the UI disables it for image-conditioned modes with the message `Heretic is currently text-only;
image-conditioned modes require Canonical Qwen3-VL.`

The experimental path is admitted only when the local source model
`/Users/elbancol/AI/MLX-Models/Jundot/froggeric/Qwen3.6-35B-A3B-Uncensored-Heretic-MLX-4bit` and a
stable state-28 bridge are readable. Set `H3_HERETIC_MODEL` to override the source model and
`H3_HERETIC_BRIDGE` to point at the bridge. The default bridge location is the sibling work-models
directory, named `qwen3.6-35b-a3b-heretic-state28-bridge.npz`; its SHA-256 must be
`8dc5dabb7da0d69dfe7ec0d5d80f684a50768d500b46bf70c03cec557141068e`. Volatile `/tmp` bridge paths
are rejected and the missing stable bridge leaves the option unavailable.

When selected, Render Lab starts `heretic_encoder.py` in a child process. It checks exact canonical
and Heretic token-piece alignment, executes only the pre-final-norm state 28 (`[1,T,2048]`), applies
the standardize-plus-affine bridge to BF16 `[1,T,5120]`, writes the existing conditioning-artifact
replay format with canonical H3 token IDs, and proves the Heretic process release gate. Only after
that child exits successfully does Render Lab launch `scripts/generate.py` with
`--conditioning-artifact`; the canonical Qwen construction is skipped by the existing replay path.
The run evidence records source identity, bridge identity, alignment, artifact identity/checksum,
timing, memory, release, and the encoder-before-H3 process boundary.

## Image-conditioning policy

I2V and first/last-frame inputs are resized directly to the selected conditioning canvas with
deterministic LANCZOS resampling. The runtime intentionally does not crop, letterbox, or add dead
space, so an aspect-mismatched source may be visibly distorted. Callers who need composition
preserved must supply an aspect-matched reference image.

Each accepted run retains its immutable `render-config.json`, input hashes, raw logs, benchmark,
status, telemetry, and published media under `out/render-lab/run-*`.

# minimax-h3-mlx

MLX (Apple Silicon) port of [**MiniMaxAI/MiniMax-H3**](https://huggingface.co/MiniMaxAI/MiniMax-H3) —
MiniMax's omni-modal generative system for synchronized **video + audio** generation.

> Powered by MiniMax H3.

H3 is **not** a language model. It is a diffusers pipeline: a 33B diffusion transformer denoising
video and audio latents jointly, conditioned by a frozen Qwen3-VL-32B encoder, with separate video
and audio VAEs. There is no autoregressive decoding and no `mlx_lm.convert` path — this repository
is a from-scratch MLX implementation of the pipeline.

## Architecture

| Component | Class | Size | Notes |
|---|---|---|---|
| `transformer` | `MiniMaxH3DiTModel` | 33B / 66.3 GB | 50 blocks, hidden 5376, 56x128 heads (inner dim 7168 > hidden), SwiGLU ffn 14336, 3D MM-RoPE |
| `text_encoder` | Qwen3-VL-32B | 66.7 GB | frozen conditioner; H3 reads the **unnormalized** hidden state after layer 50 of 64 |
| `video_vae` | `MiniMaxH3VideoVAE` | 10.4 GB | ViT+CNN KL VAE, 16x spatial / 4x temporal, 24 latent channels, tiled |
| `audio_vae` | `MiniMaxH3AudioVAE` | 0.6 GB | DAC/BigVGAN stereo 32 kHz, 32 latent channels, 40 Hz latent rate |

Everything runs over **one packed 1-D sequence** holding every modality at once:

```
[ text (L) | keyframe conditions (C) | target audio (A) | target video (V) ]
```

Attention is full self-attention over that sequence — no cross-attention, no per-modality block
weights. Modality-specific behaviour comes only from the two input patch projections, a per-row
AdaLN modality tag, and the two output heads.

### The two checkpoints are the same weights

`FL2VA/` and `Ref2VA/` are **byte-identical except for `model_index.json`** — all 80 weight and
code files share LFS hashes. The 288 GB repository is 144 GB of unique weights published twice;
only the pipeline metadata (`partition`, `tasks`) differs. One conversion covers both tasks.

### AdaLN precompute: 13B of the 33B need not be resident

13B parameters live in the per-block `adaln_proj.linear` projections (50 x `[96768, 2688]`). Their
only input is the timestep embedding — nothing sequence-dependent — so for a fixed sampler schedule
every modulation tensor a run will ever need can be computed once up front and the projections then
dropped. `ModulationCache` builds it and `drop_adaln_weights` frees the originals; the cache is
verified bit-exact against the live projection.

Measured on the real checkpoint, a 40-step run (video and audio use different sigma shifts, 12.0 and
3.0, so their schedules only partly coincide — 77 distinct timesteps in all, not 40):

| | params | resident |
|---|---:|---:|
| DiT as shipped | 33.12B | 66.3 GB |
| `adaln_proj` dropped | -13.01B | -26.0 GB |
| **after** | **20.11B** | **40.3 GB + 745 MB cache** |

A **25.3 GB net saving**, with the cache 35x smaller than the weights it replaces and built in 0.7 s.

The table is built from the float32 timestep MLP through the *unquantized* projections. Every block
reads the same `temb`, so an error there biases all 50 blocks identically at every step and
accumulates coherently along the trajectory — building it before quantization keeps that path exact.


### Encoder truncation: 14 of 64 layers are never evaluated

H3 reads the **unnormalized** hidden state after the 50th of Qwen3-VL-32B's 64 decoder layers and
feeds it straight to the DiT's `condition_proj`. The language-model head, the final norm and layers
50-63 are never touched, so the port loads only what it reads:

| | params | resident |
|---|---:|---:|
| text encoder on disk | — | 66.7 GB |
| layers 0-49 only, no `lm_head`, no vision tower | 25.16B | **50.3 GB** |

506 tensors are skipped. Reading pre-norm is a real distinction, not a detail — the parity test
asserts the returned state *differs* from `last_hidden_state`, since silently returning the normed
output would still look plausible.

Together with the AdaLN precompute, the two structural savings take the resident pipeline from
144 GB to about **102 GB before any quantization**.

## Performance: read this before converting anything

MiniMax has **not** released its sparse-attention implementation ("the initial open-source release
provides inference with full attention only"), so a run does dense attention over tens of thousands
of rows. Measured on an **M3 Ultra (550 GB unified memory)**, bfloat16, one transformer block timed
and multiplied by 50 (the blocks are identical):

| Request | Packed rows | Per block | **Per denoising step** | Peak activations |
|---|---:|---:|---:|---:|
| 5 s, 1344x768 | 37,966 | 10.5 s | **8.8 min** | 9.3 GB |
| 15 s, 1344x768 | 109,318 | 74.9 s | **1.04 h** | 24.4 GB |

A full 5 s generation confirms the extrapolation rather than relying on it: a real step of the
complete pipeline measured **531.6 s (8.86 min)** against the 8.8 min predicted from one block.

Per-step cost is the measured, assumption-free number. The released weights are **CFG-distilled**
("guidance baked into the weights, so there is no guider, no `negative_prompt` and no
`guidance_scale`"), so a step is one forward, not two — but MiniMax does not publish a recommended
step count, and the reference marks `num_inference_steps` required rather than defaulting it. Total
wall-clock therefore scales directly:

| Steps | 5 s clip | 15 s clip |
|---:|---:|---:|
| 8 | 1.2 h | 8.3 h |
| 16 | 2.3 h | 16.6 h |
| 50 (generic diffusers default) | 7.3 h | 52 h |

Peak memory is modest — MLX's attention is flash-style and never materializes the score matrix — so
**memory is not the constraint. Compute is.** 5 s is the shortest clip H3 supports and 15 s at 2K is
its flagship capability; 2K is out of reach locally.

This also changes what quantization buys. The bottleneck is attention FLOPs, which quantization
does not reduce. At 5 s the linear layers are ~42% of the work, at 15 s ~20%, so a 4-bit DiT is
worth roughly 1.2-1.4x end-to-end — useful for *fitting* the model on a smaller Mac, not for making
generation quick.

## Published quants

Collection: [**pipenetwork/MiniMax-H3 MLX**](https://huggingface.co/collections/pipenetwork/minimax-h3-mlx-6a70c7ef3f7bfae7dc3d2e82)


| build | on disk | resident | PSNR vs bf16 | velocity rel-L2 |
|---|---:|---:|---:|---:|
| [f32](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-f32) | 132.5 GB | 80.5 GB | — | — |
| [bf16](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-bf16) | 66.3 GB | 40.3 GB | reference | reference |
| [8bit](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-8bit) | 35.3 GB | **21.5 GB** | 27.6 dB | 0.0329 |
| [6bit](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-6bit) | 30.3 GB | **16.5 GB** | — | 0.0611 |
| [4bit](https://huggingface.co/pipenetwork/MiniMax-H3-MLX-4bit) | 25.3 GB | **11.5 GB** | 22.0 dB | 0.1649 |

`bf16` is the faithful conversion: it preserves the release's **mixed** precision rather than
flattening it — MiniMax ships twelve tensors (the two patch projections, the timestep MLP, both
output heads) in float32 and the other 522 in bfloat16, and casting those twelve down would be a
downgrade from upstream. `f32` upcasts everything; since the source weights are bfloat16 that is a
lossless widening carrying **no additional information**, published for float32 fine-tuning rather
than for generating. `load_dit(dtype=mx.float32)` gives the same thing from the smaller download.

Each holds the **transformer only** — the VAEs and text encoder still come from the upstream
release, which the pipeline loads alongside:

```bash
python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4 \
  -c /path/to/MiniMax-H3/FL2VA \
  -t /path/to/MiniMax-H3-MLX-4bit
```

### Streamed AdaLN derived base (v0.3c)

The v0.3b forge can produce a complete derived checkpoint with format identifier
`minimax-h3-mlx-streamed-adaln-v1`. `load_dit` detects that format only from a validated
`conversion_manifest.json`; missing, bounded, pending, unsupported, or structurally incomplete
derived output fails before model construction. The normal original checkpoint path remains
resident-AdaLN compatible.

For a complete derived checkpoint, the loader uses `base/`, constructs the transformer in explicit
`cache_only` mode, and strictly requires the 848 ordinary transformer tensors plus the two
`final_layer.adaln_proj.linear` tensors. All 50 block-level AdaLN projection modules are absent from
the model tree: their packed weights, scales, quantization biases, and learned biases remain in the
50 `adaln/` sidecars. The final-layer AdaLN projection remains resident and loadable.

This slice does not open sidecars, build the modulation cache, run denoising, or support generation
from the derived checkpoint. A cache-only forward rejects missing, partial, malformed, or
timesteps-incomplete modulation data with an explicit error. `--keep-adaln` continues to mean the
existing resident behavior for original checkpoints, but is rejected for derived checkpoints because
resident sidecar loading is not implemented.

Proven by the real derived-base load probe: the complete derived base loads successfully; all block
attention and feed-forward parameters are present; block-level AdaLN parameters are absent; the
final-layer AdaLN weight and bias remain resident; exactly five base payloads are opened and no
AdaLN sidecar payload is opened; and the base contains 850 tensors totaling 16,464,048,640 logical
bytes. Active base MLX memory is approximately 16.466 GB, approximately 11.7 GB below the previous
full-transformer load.

The v0.3c base receipt does not prove streamed cache construction or generation behavior; those remain
separate validation lanes below.

### Sequential streamed AdaLN cache construction (v0.3d)

The bounded v0.3d real probe was run externally against the complete derived checkpoint. It opened 50
sidecars in strict block order, with 50 unique files and one builder-owned payload released before the
next sidecar opened. The 77-entry timetable produced six BF16 arrays per block, each with shape
`(231, 5376)`, and an exact retained-cache size of `745,113,600` bytes. Total sidecar logical bytes
processed were `13,828,147,200`; maximum one-block active-memory increase was `295,665,800` bytes.

The measured peak active MLX memory was `17,494,943,308` bytes. Completed derived base plus retained
cache active memory was `17,212,941,760` bytes. Cache construction took `2.926 s`, and the total probe
took `5.881 s`. After releasing both cache and transformer, the measured result was `948` active bytes
and zero allocator cache.

The probe loaded no Qwen or VAE, performed no transformer forward, and did no denoising, decoding, or
rendering. At v0.3d, this receipt therefore did not claim generation support; real original-resident
versus streamed-sidecar numerical parity, real transformer-forward parity, denoising, rendering, the
final generation peak, and swap-write reduction were unproven at that slice. Later v0.5d/v0.5e
receipts establish the derived full-schedule transformer-forward, denoising, decoding/rendering, and
lifecycle proofs; they do not claim cross-format numerical parity or swap-write reduction.

The external real-cache probe is:

```bash
./.venv/bin/python scripts/probe_streamed_adaln_cache.py \
  /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln
```

The bounded real load probe is:

```bash
./.venv/bin/python scripts/probe_derived_base_load.py \
  /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln
```

It loads only the five base shards and reports MLX active memory, allocator cache, peak memory,
timings, tensor counts, model-tree invariants, sidecar-open proof, and release telemetry. It does not
claim swap reduction or successful rendering.

The core is quantized at the named width; `adaln_proj` is held at 8-bit in every build, which costs
0.25% on the modulation table and takes 12.2 GB off each download.

**3-bit and 2-bit are not published.** 3-bit was built and rendered: at 16.3 dB PSNR the subject is
destroyed — no animal, no log, just a textured field. It does not fail by blurring, so a sharpness
check would have passed it: per-frame variance *rises* to 54.7 against bfloat16's 37.1 as structure
becomes high-frequency noise. Velocity error ranked the widths correctly but could not have located
that cliff; only generating found it.

## How the quants were chosen

Comparing independent generations is the obvious way to rank widths and it is the wrong one.
Diffusion trajectories diverge chaotically from the first step, so the difference between a bf16
clip and a 4-bit clip is dominated by divergence amplification rather than by quantization error —
and at ~9 minutes per step, a sample large enough to separate five widths would take days.

`eval_quant.py` uses **teacher forcing** instead. One bfloat16 trajectory is recorded, and every
variant re-predicts the velocity *at those same latents*. Both models see identical inputs at every
point, so what is measured is quantization error alone, and each `(prompt, seed, step)` becomes an
independent paired observation. Aggregation is a **paired bootstrap** resampling one shared index
set across variants, which removes the between-input variance that otherwise swamps the
between-variant gaps.

| bits | video rel-L2 [95% CI] | audio rel-L2 | video cosine |
|---:|---|---:|---:|
| 8 | 0.0329 [0.0277, 0.0381] | 0.0130 | 0.99941 |
| 6 | 0.0611 [0.0501, 0.0728] | 0.0274 | 0.99791 |
| 4 | 0.1649 [0.1324, 0.1971] | 0.1016 | 0.98456 |
| 3 | 0.2842 [0.2362, 0.3358] | 0.2341 | 0.95635 |

Every interval is disjoint from its neighbours on only 20 observations per variant — an unpaired
comparison at that sample size would have produced overlapping intervals and an unusable ranking.
Two results that were not predictable in advance: the steepest step is **6 to 4 bits** (2.7x), not
at the low end, so interpolating between 8 and 4 puts the knee in the wrong place; and **audio
degrades faster in relative terms** than video, its share of the error climbing 0.40x -> 0.82x
across the range, plausibly because audio is a small fraction of the packed rows and has less
redundancy to absorb it.

Velocity error is the right quantity to watch rather than final pixels, because the scheduler
*integrates* it — a bias that is small per step still accumulates along the trajectory. That same
property is why it cannot tell you where output stops being usable, which has to be generated. It
is: 4-bit keeps the subject at 22.0 dB, 3-bit destroys it at 16.3 dB.

### AdaLN: measure, do not reason

`adaln_proj` is 13B of the 33B and dominates every build's download. The intuitive argument for
keeping it at bfloat16 is that every block reads the same `temb`, so an error there biases all 50
blocks and compounds. That argument is wrong. The table is computed **once**, so quantization
perturbs it like slightly different modulation weights, with nothing to compound.

`eval_adaln_quant.py` measures it directly — no forward passes, just build the table from bfloat16
and from quantized weights and compare:

| adaln bits | table rel-L2 | worst tensor | core velocity rel-L2 at that width |
|---:|---:|---|---:|
| 8 | **0.0025** | shift_mlp 0.0035 | 0.0329 |
| 6 | 0.0031 | shift_mlp 0.0074 | 0.0611 |
| 4 | 0.0077 | shift_mlp 0.0282 | 0.1649 |

8-bit AdaLN moves the table an order of magnitude less than the core's own error and takes 12.2 GB
off every download, so every published build uses it. 4-bit AdaLN is 3x worse and is not used at any
core width. Per-tensor error is reported rather than one aggregate because the six modulation
tensors do not enter equally: `x * (1 + scale) + shift` for the norms, `x + gate * f(x)` for the
residual branches.

## Porting notes: MLX specifics worth knowing

Things that cost real debugging time here and generalize to other diffusion ports:

* **`QuantizedLinear.weight` is packed uint32.** The reference aligns each activation to its
  projection's parameter dtype, and `layer.weight.dtype` is the natural way to express that — but
  once a layer is quantized it truncates activations to integers. It fails *silently and
  identically at every bit width*, which is the tell. Read `layer.scales.dtype` instead
  (`dit.param_dtype`).
* **Metal command buffers have a deadline.** Bulk work at checkpoint scale — 10 GB of 5-D conv
  transposes, or casting a 33B stack to float32 — overruns it, and worse, only once something else
  is using the GPU. Both now run on the CPU stream, which has no such limit. The failure mode is
  ugly: it aborts a multi-hour run at the *last* component to load.
* **Convolutions are channels-last.** `(N, D, H, W, C)` with weights `(C_out, kD, kH, kW, C_in)`
  against torch's `(N, C, D, H, W)` / `(C_out, C_in, kD, kH, kW)`. Both VAEs run channels-last
  internally and transpose only at their public boundary.
* **There is no reflect padding.** `mx.pad` offers constant and edge only; reflect is done by
  gather (`video_vae.reflect_pad`).
* **`mx.linspace` does not match `torch.linspace`.** ATen takes a float32 step, splits the range at
  the halfway point, and evaluates with an FMA. Here that mattered: the sigma grid is collapsed by a
  consecutive-duplicate check, so a one-ulp difference changes how many sigmas survive and therefore
  *the number of model evaluations*.

## Status

| Piece | State |
|---|---|
| DiT (`MiniMaxH3DiT`) | **done** — matches diffusers reference to 4.8e-07 |
| Video VAE | **done** — encode + decode match to 1.2e-06, tiled and untiled |
| Audio VAE | **done** — encode 6.6e-07, decode 2.1e-08 |
| AdaLN precompute + drop | **done** — bit-exact; verified on the real 33B checkpoint |
| Scheduler | **done** — bit-exact sigmas, timesteps and 16-step trajectory |
| Packed-sequence geometry | **done** — bit-exact `(t, h, w)` grid, tags, indices |
| Text encoder | **done** — `hidden_states[50]` matches HF to 5.0e-08 |
| Checkpoint loaders | **done** — all four components load from the release, zero key mismatches |
| Pipeline / denoise loop | **done** — generates prompt-faithful video + synced audio |
| Derived full-schedule proof | **done** — v0.5e complete; the 128×128 canonical baseline is frozen and the 256×256 quality/resource proof passed |
| Quant set | **done** — f32 / bf16 / 8 / 6 / 4-bit published; 3-bit built but withheld |

All four components were loaded from the released checkpoint and exercised:

* **DiT** — 33.12B over 534 tensors, every key matched, mixed precision intact (12 float32 tensors
  for the patch projections, timestep MLP and output heads; 522 bfloat16).
* **Audio VAE** — 151.3M params, 40 latents/s at 32 kHz as specified. Round-tripping real signals
  gives **33.2 dB SNR** on a 440 Hz tone and 0.9995 correlation on a decaying note; phase-accurate
  reconstruction is what proves the convolution transposes and the folded weight norm are right.
* **Video VAE** — 2.60B params, 16x spatial / 4x temporal. A 22-frame 128x128 clip round-trips
  through a `(1, 48, 7, 8, 8)` latent at **0.962 correlation / 21.9 dB PSNR**, and the decoded frame
  counts match `video_latent_num_frames` exactly (7 latent -> 22 frames, 12 -> 39) — the packing
  geometry and the VAE's temporal chunking were ported separately and agree.
* **Text encoder** — 25.16B params over 50 layers; a prompt encodes to `(1, N, 5120)` in 0.9 s.

The keyframe (`fl2va`) path is implemented — conditioning frames are encoded through the VAE's
spatial encoder, noised to `t = 0.999` and prepended as rows — but only the text-only `t2va` path has
been run end to end. Two details of the reference are load-bearing there and easy to get wrong: the
keyframe posterior is **sampled** rather than taken at its mode, under a generator seeded with 42
*independently of the request seed*, and the sampled latent is **rounded through float16** before
normalization — about 11 bits of every conditioning latent. MLX's RNG differs from torch's, so that
draw is not bit-identical to the reference's, though everything around it is.

### End to end

```bash
./.venv/bin/python scripts/generate.py "a red fox leaps over a mossy log" -o fox.mp4
```

The CLI defaults to the 0.2 MP 16:9 canvas (608x352) to keep local Apple Silicon experiments
within a smaller memory envelope. Use `--megapixels`, or explicit `--height` and `--width`, when
you intentionally want to move toward the 768-pixel training canvas.

The CLI uses phase-isolated staged loading: Qwen is loaded first, the prompt (and any images) are
encoded, and the resulting conditioning is materialized before the transformer is loaded. The
transformer builds its AdaLN cache, denoises, and is released before either decoder VAE is loaded.
Video and audio decoding are also separate phases. Image-conditioned runs temporarily load the
video VAE to encode keyframes, release it for transformer denoising, then reload it for final video
decoding; that reload is the cost of keeping transformer and decoder residency disjoint.
The default lifecycle releases Qwen before generation; `--keep-text-encoder` intentionally retains
it after encoding for repeated in-process use and therefore increases memory overlap with the
transformer.
After AdaLN projections are dropped, the dedicated AdaLN allocator purge runs once before denoising;
ordinary component-release purges remain at the text-encoder, transformer, and VAE boundaries, and
none runs between denoising steps. This may reduce peak pressure and swap at the cost of some
first-step reallocation, and the actual benefit depends on the MLX allocator and must be measured.

### Derived streamed-AdaLN checkpoint forge

The repository includes a v0.3b converter that creates a new
`minimax-h3-mlx-streamed-adaln-v1` directory with an AdaLN-free base, one exact raw-byte sidecar per
transformer block, manifests, and verification receipts. v0.3c consumes the derived base in
cache-only mode, and v0.3d constructs the retained modulation cache sequentially. At those earlier
slices the original mixed-shard format remained the only generation-capable runtime format; the
v0.5d/v0.5e derived full-schedule receipts below close that gap for the proven path.
The derived base has loaded and evaluated successfully, with component-level active MLX memory of
approximately 16.466 GB—approximately 11.7 GB below the previous full-transformer load. Sidecar
real transformer-forward execution, denoising, render peak, swap-write reduction, and streamed
numerical parity were unproven at the v0.3c/v0.3d boundary. Later v0.5d/v0.5e evidence below proves
the derived full-schedule transformer, denoising, decoding/rendering, and lifecycle path; swap-write
reduction and cross-format numerical parity remain unclaimed.

#### v0.3b Metal consumer-compatibility receipt

Validated on 2026-08-04 with the local MLX consumer. Block 0 loaded successfully with `mx.load`:
lazy active memory was `0`, evaluated active memory was `276,594,688` bytes, and allocator cache
after evaluation was `0`. The sidecar contained exactly four tensors: `bias`, `biases`, and
`scales` as BF16, and `weight` as U32, with shapes `(96768,)`, `(96768, 42)`, `(96768, 42)`,
and `(96768, 672)` respectively. After release and `clear_cache`, active memory was `196,608`
bytes and allocator cache was `0`.

Block 5, reconstructed from a cross-source-shard group, produced the identical successful receipt:
lazy active memory `0`, evaluated active memory `276,594,688` bytes, allocator cache `0`, the same
four tensors, dtypes, and shapes, and after release active memory `196,608` bytes with allocator
cache `0`. The terminal result was `MLX SIDECAR COMPATIBILITY PASSED`. In the probe, on-disk
Safetensors labels `BF16` and `U32` are compared against the MLX constants
`mlx.core.bfloat16` and `mlx.core.uint32` through their string representations; this is the
intentional dtype-name normalization boundary.

The source transformer is approximately 30.3 GB on disk. The derived checkpoint is approximately
30.3 GB more, so keeping both requires approximately 60.6 GB decimal before filesystem overhead
(about 28.3 GiB each, 56.5 GiB together). The base contains all 848 ordinary tensors and the two
final-layer AdaLN tensors; only the 200 block-level AdaLN tensors move to sidecars. Sidecars copy
the original packed quantized values and BF16 bytes without dequantization or NumPy conversion.

Inspect topology and disk requirements without writing:

```sh
./.venv/bin/python scripts/build_streamed_adaln_checkpoint.py \
  --source /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit \
  --output /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln \
  --dry-run
```

For bounded development validation, write at most the selected sidecars and a complete base
classification manifest under `/tmp`; this does not write the 16.4 GB base payload:

```sh
./.venv/bin/python scripts/build_streamed_adaln_checkpoint.py \
  --source /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit \
  --output /tmp/minimax-h3-streamed-adaln-block-5 \
  --blocks 5
```

The complete conversion is explicit and is not run by the test suite:

```sh
./.venv/bin/python scripts/build_streamed_adaln_checkpoint.py \
  --source /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit \
  --output /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln
```

Verify an existing derived or bounded output without writing:

```sh
./.venv/bin/python scripts/build_streamed_adaln_checkpoint.py \
  --source /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit \
  --output /tmp/minimax-h3-streamed-adaln-block-5 \
  --verify
```

The forge refuses an existing destination unless `--force` is supplied. It stages output in an
incomplete sibling directory and verifies exact dtype/shape/payload checksums before publication.
First-time publication uses an atomic directory rename. `--force` uses exception-safe
backup-and-rollback: it moves the existing destination to a sibling backup, installs the verified
output, and restores the backup when a handled publication step fails; this replacement is not
crash-atomic. The completion record is intentionally excluded from `per_file_checksums` because
it authenticates the other artifacts. The progress report uses decimal GB and binary GiB separately
and includes conservative temporary and replacement-space accounting. A failure removes the
incomplete sibling and preserves the prior destination when rollback is possible.

A first run produces a misty forest with tall trunks, a mossy log in the foreground and an orange
fox form moving across the later frames — semantically faithful to the prompt, with audio muxed in.
Beyond eyeballing it, three properties are what say the wiring is right rather than merely plausible:

* **Temporal coherence** — adjacent frames differ less than distant ones (mean |Δ| 9.6 vs 12.7). A
  pipeline that packed the time axis wrongly would show no such gradient.
* **Stereo coherence** — the two audio channels correlate at **+0.947**: high, but not 1.0. That is
  what real stereo looks like, and it exercises the channel-major audio packing, where the two
  blocks of rows are pinned to opposite ends of the width grid.
* **Duration agreement** — video and audio land within 8 ms of each other (5.167 s vs 5.175 s), the
  residue of the 40 Hz audio latent grid, which is the shared rotary clock doing its job.

Output is written with no extra dependencies: stereo WAV through the standard library's `wave`
module, video piped as raw RGB into `ffmpeg` (with a PNG-sequence fallback if it is absent).

#### v0.5d derived full-schedule functional proof

`V0.5D_FUNCTIONAL_PROOF_PASSED` is recorded from the preserved real run at
`out/v0.5d/dodecahedron-seed-0-16pt-terminal-02` with `functional_success: true`.
`canonical_timing_eligible: false`. Reason: `No uncontended-host operator declaration/process snapshot.`
These timings are engineering evidence, not canonical benchmarks.

| Phase | Verified result |
|---|---|
| Conditioning | 103 tokens; `[1,103,5120]`; bfloat16; release succeeded with allocator cache `0` |
| Derived denoising | Cache-only streamed-AdaLN v1; 15 transitions; 15 transformer forwards; 15 cache sessions created/released; 750 sidecar opens/releases; 750 validated block pairs; maximum simultaneous sidecars `1`; overlap violations `0`; dense reconstruction `0` |
| Derived memory | Peak active memory `17,218,975,504` bytes; final derived release active/cache `256 / 0` |
| Denoising timing | Total `41.372048 s`; cache construction total `27.861136 s`; mean `1.857409 s`; fastest transition step 6 `2.575642 s`; slowest step 0 `3.864866 s` |
| Video | Raw `[1,3,30,128,128]`; RGB `[30,128,128,3]`; 30 frames at 128×128 and 24 fps; peak `11,161,857,908` bytes; release allocator cache `0` |
| Audio | Raw `[2,1,40000]`; waveform `[2,40000]`; stereo, 32 kHz, 40,000 samples/channel, 1.25 s; peak `2,678,107,956` bytes; release allocator cache `0` |
| MP4 | H.264/yuv420p video, 128×128, 24 fps, 30 frames; AAC stereo audio at 32 kHz for 1.25 s; 59,122 bytes; SHA-256 `89ba3ae8…b3388a3f`; atomic publication passed |
| Lifecycle | All workers terminated; all release gates passed; final allocator cache `0`; no retry, replacement worker, or suppressed failure |

This is the first successful MP4 path from the derived full schedule. Attempt 01 remains preserved
as a failed historical receipt (the derived worker hit a PEP 3118 `B` item-size mismatch); Slice 5C's
bfloat16 conditioning-boundary correction is carried forward and verified by the v0.5d conditioning
receipt. The preserved evidence does not independently emit or certify the exact 850-base-tensor
count; video and audio worker receipts do not split VAE load time from decode time; the staged
partial MP4 SHA was not retained; and host-process inspection was unavailable. None of these limits
invalidate the functional proof.

The v0.5e receipts below record `functional_success` independently from canonical timing eligibility.
`--operator-declared-uncontended` defaults false and is never inferred. Before conditioning, the
parent writes one bounded, read-only `ps` snapshot and classifies only narrow known MLX, model-server,
generation, MPS, or Metal-compute workloads; ordinary desktop compositing is not a conflict and the
scan does not claim absolute idleness. Eligibility is true exactly when functional success, the
operator declaration, and successful snapshot capture are all true and
`known_conflicting_processes` is empty. Process-inspection failure remains nonfatal but forces timing
ineligibility. Derived preflight also requires the conversion manifest and base index to agree on
exactly 850 base tensors without opening tensor payloads.

Real MLX execution is currently constrained by the Codex/macOS Metal sandbox and must be launched
from ordinary Terminal. The v0.5e closeout below preserves cache-construction attribution as deferred
optimization evidence; it does not begin optimization or benchmarking.

#### v0.5e Slice 3B6B milestone closeout

`V0.5E_COMPLETE` is recorded through `bd29ce3ec0610075247176d37a9eb795bbe4d427`
(`Fix cache attribution transition identity`). The successful end-to-end 256×256 proof is
`quality-256-02` with `functional_success: true` and `retry_count: 0`.

The evidence namespaces are intentionally distinct:

| Namespace | Meaning |
|---|---|
| `canonical-baseline-01` | Frozen 128×128 canonical timing baseline |
| `quality-256-01` | Preserved failed telemetry-contract attempt |
| `quality-256-02` | Successful 256×256 quality/resource proof; not a replacement canonical baseline |

The completed 256 contract is:

```text
output:                  256×256
frames:                  30
fps:                     24
duration:                1.25 s nominal

video latent:            [1,24,9,16,16]
audio latent:            [2,32,50]

text rows:               103
audio rows:              100
video rows:              576
total packed rows:       779

sigma points:            16
transitions:             15
transformer forwards:    15

video shift:             12.0
audio shift:              3.0
seed:                    0
```

Prompt SHA-256: `c7d57d0bf61aa78dfe79d3267c13fc74b91bc397e09f1d73c35d12f4179dd00a`.
The global RNG draw order is video noise before audio noise; changing spatial video geometry changes
the subsequent audio initial noise, so cross-resolution bit-identical audio noise is not claimed.

The streamed-AdaLN proof recorded 15 cache sessions with 50 blocks per session, 750 sidecar opens,
750 sidecar releases, 750 matched pairs, maximum simultaneous sidecars `1`, overlap violations `0`,
dense reconstruction `0`, and telemetry failures `0`. The transition-identity emitter defect exposed
by `quality-256-01` was repaired in `bd29ce3` and strict parent validation accepted the corrected real
event stream in `quality-256-02`.

Lifecycle proof from `quality-256-02`:

| Component | Verified result |
|---|---|
| Transformer | Final active memory `256` bytes; final allocator cache `0` bytes; release gate passed |
| Video decoder | Raw `[1,3,30,256,256]`; RGB `[30,256,256,3]`; 30 PNGs published; video VAE peak `12,023,902,068` bytes; release gate passed; allocator cache `0` after release |
| Audio decoder | Raw `[2,1,40000]`; waveform `[2,40000]`; stereo at 32,000 Hz; 40,000 samples/channel; audio VAE peak `2,595,794,740` bytes; release gate passed; allocator cache `0` after release |

Final media proof:

```text
codec:              H.264
pixel format:       yuv420p
resolution:         256×256
fps:                24
frames:             30

audio codec:        AAC
channels:           stereo
sample rate:        32 kHz

ffmpeg calls:       1
ffprobe calls:      1
retries:            0

duration:           1.250 s
size:               133,521 bytes
```

Final MP4 SHA-256: `da5da3ce010673e09d6ddcb025311c6b62fd1ed294d8c6ee6661fb96fd2a427b`.
The preserved manifest hashes are frame
`12e4c56308a432d1539716e728081a9ca5864e964837228a334007af1909e3e2`, audio
`9ca7b912a63c7b1a8de389c38a050e12b83d01856696e1600591948c1cc32231`, and MP4
`c64bfc1098b36b6c6815fd1225ed086caa3a0b3f2c0af1b7f75174c7a09460e4`.

Deferred cache-attribution evidence, not cross-run performance conclusions:

```text
cache wall total:                   31.10547550197225 s

materialization/evaluation:         27.516481075639604 s  (88.46185641462036%)
release/purge:                       2.836448738875333 s  (9.118808483398647%)
sidecar I/O/reconstruction:          0.10752325801877305 s  (0.3456730890095395%)
projection compute:                  0.01100156965549104 s  (0.03536859500763292%)
```

The first cache session was `2.4995043340022676 s`; the warm-session mean was
`2.043283654854999 s`. These values are retained as deferred optimization evidence only.

All 30 frames were reviewed at the qualitative/operator-review level. The 256 output was a
coherent centered rotating faceted object with substantially better temporal stability and material
separation than the previously observed 128 output. The generated object did not clearly satisfy
regular-dodecahedron topology. That semantic finding is not classified as a runtime defect and is
not a v0.5e blocker.

The following work is intentionally deferred: same-HEAD 128-versus-256 timing, multi-seed quality
characterization, resolution scaling, streamed-AdaLN optimization, and semantic prompt/topology
investigation. None is required for v0.5e completion.

The next engineering milestone was **v0.6 — production generation surface**. Its completed Render
Lab/image-conditioning slice is recorded in
`docs/v0.6-render-lab-image-conditioning-closeout.md`. The slice is currently uncommitted on
`experiment/h3-generation` at the production-good base `62fba38cc1737004fae570b31a2bdbae10835cea`;
the unrelated ConvRot/CR-5 research files remain separate and must not hitchhike into its future
transfer.

#### v0.6 Render Lab and image-conditioning current state

The local loopback Render Lab under `tools/render_lab/` wraps the existing `scripts/generate.py`
runtime and supports T2V, I2V, and first/last-frame conditioning with curated geometry validation,
immutable per-run evidence, live logs, benchmark/telemetry records, and terminal-state polling that
stops after `SUCCESS` or `FAILED`. The still-image path uses the direct
`Qwen2VLImageProcessorPil` PIL/NumPy processor with `local_files_only=True`; it does not construct
the composite video processor and does not require torch or torchvision for image conditioning.

The visual merge repair uses the MLX Qwen masked-scatter contract: compact visual rows are inserted
at the selected positions in the full sequence, with fail-closed row/width checks for the initial
and three deepstack tensors. The bounded receipt records 1,920 total tokens, 1,530 visual rows,
and finite layer-50 output. Real post-repair artifact history contains successful 512×512 I2V and
first/last-frame runs. Image inputs intentionally use deterministic resize with no crop or
letterbox; aspect-matched references remain the caller's responsibility.

The canonical local runtime paths are:

```text
H3_CHECKPOINT_ROOT=/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va
H3_TRANSFORMER=/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit
```

### Turbo LoRA adapters

The generic adapter path is additive and target-indexed. It accepts the common
`lora_A`/`lora_B` and `lora_down`/`lora_up` safetensors layouts, keeps adapter
weights outside the checkpoint parameter tree, and applies each delta around
the callable projection. That includes MLX quantized projections: the packed
base weight is left untouched and the low-rank path uses the projection's
activation dtype.

Resident H3 core and token-refiner targets use their local module paths, for
example `blocks.0.attn.qkv_proj` and
`token_refiner.blocks.0.mlp.fc2`. Block AdaLN adapters use
`blocks.<index>.adaln_proj.linear`; the same target is applied while a streamed
sidecar is reconstructed. `final_layer.adaln_proj.linear` is applied at the
final norm, so a single registry covers resident and streamed execution.

Turbo is opt-in and defaults to an eight-sigma schedule. A metadata-advertised
step count or explicit `--turbo-steps` override may select another validated
count. The CLI wiring is:

```bash
./.venv/bin/python scripts/generate.py \
  "a red fox leaps over a mossy log" \
  --turbo-lora /path/to/turbo.safetensors \
  --turbo-steps 8 \
  -o fox-turbo.mp4
```

The first real render remains an operator-run Apple-Silicon gate; this
implementation slice does not execute H3.

ConvRot research remains a separate, explicitly excluded workstream.

### Validation

Parity is checked against the `minimax-h3` branch of diffusers, not against a re-reading of it. The
MLX model is the source of truth and its parameters are pushed through the **official** conversion
script (`reorder_interleaved_qkv` + `convert_transformer_key`) into the reference module, so the
test exercises the two raw-checkpoint layout quirks the port handles by reshape rather than
assuming them:

* `attn.qkv_proj` rows are **per-head interleaved** — `[h0: q,k,v][h1: q,k,v]...` — so the
  projection output reshapes to `(..., heads, 3, head_dim)`.
* `mlp.fc1` is a fused **`[gate; value]`** SwiGLU projection; the reference computes
  `fc2(silu(gate) * value)`.

Both mean the released checkpoint loads **1:1 with no weight surgery**.

The video VAE is checked the same way, through `convert_video_vae_key`, on the reference's own tiny
CPU-parity config. Its `attn.to_qkv` is interleaved and its `ff.w1` fused exactly like the DiT's.

The audio VAE inverts the port's two departures instead: its MLX weights are converted back to
channels-first, to torch's transposed-conv axis order, and to a reconstructed `weight_g`/`weight_v`
pair (`v = w`, `g = ||w||`), which proves folding weight norm at load is equivalent rather than
assumed. Its recomputed Kaiser-sinc anti-aliasing filters are additionally checked against the ones
the released checkpoint actually ships, matching to 3.0e-08.

```bash
./.venv/bin/python tests/test_dit_parity.py        # 4.8e-07 vs reference
./.venv/bin/python tests/test_video_vae_parity.py  # 1.2e-06, tiled + untiled
./.venv/bin/python tests/test_audio_vae_parity.py  # 6.6e-07 encode, 2.1e-08 decode
./.venv/bin/python tests/test_text_encoder_parity.py  # 5.0e-08 vs transformers
./.venv/bin/python tests/test_packing_parity.py    # 81 checks, all bit-exact
python3 tests/test_dit_smoke.py                    # no torch needed
```

Two places needed care to stay bit-exact, both because a one-ulp difference is observable:

* **`linspace`** — ATen takes a float32 step, splits the range at the halfway point, and evaluates
  `start + step*i` with an FMA. The sigma grid is collapsed by a consecutive-duplicate check, so an
  ulp can change how many sigmas survive and therefore the *number of model evaluations*.
* **Scheduler scalars** — the reference does its arithmetic in float32 tensors. Computing the same
  expressions in Python floats rounds twice and drifts by an ulp per step.

The packed grid is built in NumPy float64 (as the reference does) because video and audio share one
40-units-per-second rotary clock, and that shared clock *is* the audio/video alignment. The
reference notes its temporal span must be summed pairwise, since sequential summation differs in
the last ulp from 16 latent frames onwards.

## Layout

```
minimax_h3_mlx/
  config.py      DiTConfig / PipelineConfig, original checkpoint field names
  dit.py         the 33B diffusion transformer
  lora.py        generic LoRA registry, safetensors loader, projection math
  turbo.py       explicit reduced-step Turbo schedule selection
  adaln.py       ModulationCache, drop_adaln_weights
  scheduler.py   rectified-flow Euler with exponential sigma shift
  packing.py     packed-sequence geometry, patchify/unpatchify, row timesteps
  load.py        checkpoint loading, mixed fp32/bf16 split preserved
  video_vae.py   causal 3D CNN encoder + 36-layer ViT decoder, tiled
  audio_vae.py   DAC encoder + attention projection + BigVGAN decoder
  text_encoder.py Qwen3-VL-32B conditioner, truncated to the 50 layers H3 reads
  pipeline.py    packing, the joint denoise loop, decoding
  media.py       mp4 / wav writing, dependency-free
reference/       upstream sources, vendored for validation only (see reference/README.md)

scripts/
  generate.py           the CLI: prompt (+ keyframes) -> mp4
  build_quant.py        quantized builds; several widths from one load
  build_unquantized.py  bf16 (native mixed) / f32 builds
  eval_quant.py         teacher-forced paired comparison across widths
  eval_adaln_quant.py   how far quantizing adaln_proj moves the modulation table
  bench_dit.py          per-block timing at realistic packed lengths
  upload.py             publish to the Hub; refuses to run without the upstream LICENSE
  make_collection.py    build/refresh the Hub collection
  run_tests.sh          all ten suites

tests/           parity vs the reference, quant round-trip, smoke
```

## License

The port is Apache-2.0. The **weights** are governed by the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
which is not an open-source licence: redistribution must carry a copy of the agreement, mark
modified files, and display "Powered by MiniMax H3"; commercial use above $20M yearly revenue needs
separate authorization; and the grant is **territorially limited** (worldwide excluding the
Excluded Territories). Any republished weights inherit these terms.

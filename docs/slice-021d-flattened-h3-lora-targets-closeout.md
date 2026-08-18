# Slice 021D — Flattened H3 LoRA Exporter Compatibility closeout

Closeout status: **Complete — host accepted (bounded)** on 2026-08-18.

This document records the existing Slice 021D implementation, its focused
offline admission proof, and the two existing real Render Lab host receipts.
It is a documentation closeout only. No production code or tests were changed
for this closeout, and no new model or render execution was performed.

## 1. Objective and verdict

Slice 021D added compatibility for one proven flattened MiniMax-H3 LoRA
exporter namespace. The bounded repair is usable in production generation:

- both motivating real H3MT files were admitted with all 188 registered
  targets compatible;
- both files completed successful host Render Lab generations through the
  repaired path;
- LightX 4-Step v0.1 retained scheduling ownership;
- each H3MT file remained an auxiliary model delta; and
- the streamed-AdaLN transformer route remained active.

The verdict is **COMPLETE — HOST ACCEPTED**, with the topology and scale
caveats in Section 7 preserved.

## 2. Git and change boundary

| Field | Recorded value |
|---|---|
| Closeout worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021d` |
| Branch | `slice/021d-flattened-h3-lora-targets` |
| HEAD | `86bac23c782d7f1c39f4ac12e2bc42b39e2e9be3` |
| Expected parent | `547d298f0b5918b4411c3c230298a04cad7703c5` |
| Implementation commit | `86bac23` — Add flattened H3 LoRA target compatibility |
| Pre-edit tracked/staged state | No tracked worktree diff and no staged diff |
| Pre-existing untracked state | `.DS_Store`, preserved untouched |

Commit `86bac23` changed exactly these implementation paths:

- `minimax_h3_mlx/lora.py` — bounded flattened-target matching and mapping at
  the existing `canonical_target` seam;
- `tests/test_style_lora_repair_contract.py` — focused contracts for accepted
  forms, malformed/out-of-range/unknown forms, loader/admission behavior,
  fail-closed behavior, and Render Lab propagation.

This closeout changes only:

- `docs/ROADMAP.md`; and
- `docs/slice-021d-flattened-h3-lora-targets-closeout.md`.

The intentionally dirty `experiment/h3-generation` worktree was not touched.

## 3. Exact implementation boundary

The accepted flattened grammar is exactly:

```text
lora_unet_blocks_<0-49>_attn_qkv_proj
lora_unet_blocks_<0-49>_attn_out_proj
lora_unet_blocks_<0-49>_mlp_fc1
lora_unet_blocks_<0-49>_mlp_fc2
```

The mappings are:

```text
lora_unet_blocks_<n>_attn_qkv_proj  -> blocks.<n>.attn.qkv_proj
lora_unet_blocks_<n>_attn_out_proj  -> blocks.<n>.attn.out_proj
lora_unet_blocks_<n>_mlp_fc1        -> blocks.<n>.mlp.fc1
lora_unet_blocks_<n>_mlp_fc2        -> blocks.<n>.mlp.fc2
```

The implementation remains bounded and fail-closed. It did not add:

- generic underscore rewriting;
- arbitrary `lora_unet_` prefix stripping;
- guessed token-refiner syntax;
- out-of-range block support;
- alternate spellings;
- LoRA math changes;
- scheduling changes;
- stacking changes; or
- UI changes.

## 4. Focused and offline validation

The implementation's focused contract coverage exercises the exact flattened
mapping, boundary indices, malformed and out-of-range rejection, unknown-wrapper
preservation, generic loader admission, the existing projection application
seam, zero-compatible fail-closed behavior, partial H3 admission, legacy
wrapper stability, and Render Lab path/scale propagation.

The real-file admission proof was host-side and MLX-free. It loaded neither an
H3 model, Qwen, a VAE, nor an MLX runtime, and it did not materialize any A/B
matrix payloads.

| File | Registered adapters / targets / compatible / incompatible | Topology | Rank / alpha | Scale / multiplier | Registration fetch / bytes | A/B refs |
|---|---:|---|---:|---:|---:|---|
| `/Users/elbancol/Downloads/H3MT_pruned_r128_v2.safetensors` | `188 / 188 / 188 / 0` | `188 total; 188 resident_core; 0 streamed_block_adaln; 0 resident_final_adaln; 0 other` | `128 / 128` | `1.0 / 1.0` | `188 / 752` | down lazy; up lazy |
| `/Users/elbancol/Downloads/H3MT-v2-rank256.safetensors` | `188 / 188 / 188 / 0` | identical to r128 | `256 / 256` | `1.0 / 1.0` | `188 / 752` | down lazy; up lazy |

Both normalized target sets were proven identical: blocks `3` through `49`
inclusive, with the four accepted projection targets per block. Therefore,
each file is a 188-target partial H3 LoRA, not a claim of coverage for all 208
known projection targets.

## 5. Host Render Lab acceptance

The two real host renders used the same controls:

- T2V;
- Canonical Qwen3-VL (`canonical-qwen3-vl`);
- 512 × 512;
- 5 seconds requested;
- seed `1701`;
- LightX 4-Step v0.1 (`lightx-4step-v01`, runtime variant
  `fl2va-turbo-4step-v0.1`);
- effective NFE `4`;
- auxiliary LoRA scale `1.0`; and
- the same sword-fight prompt:
  `A cinematic close-range sword fight between two skilled warriors in a rain-soaked industrial courtyard at night. Fast deliberate combat, visible footwork, parries, dodges, sparks from blade impacts, wet clothing, realistic body mechanics, dynamic camera movement, dramatic practical lighting, coherent anatomy and continuous motion.`

Both `render-config.json` files record branch/HEAD identity, the
`streamed-adaln-q6` transformer mode, `turbo-preset` as scheduling owner, and
one ordered auxiliary row at order `0` with role `auxiliary-model-delta`.
The local receipts are authoritative:

- `render-config.json` (schema 3) records configuration, command, identity,
  ordered adapter stack, and output path;
- `benchmark.json` (schema 1) records timing, forwards, memory, output hash,
  and process exit; and
- `run-status.json` (schema 1) records terminal success and exit status.

| Adapter | Run ID | Receipt result | Forwards | Denoising | Seconds / forward | Total elapsed | Peak MLX | Output artifact |
|---|---|---|---:|---:|---:|---:|---:|---|
| `H3MT_pruned_r128_v2.safetensors` | `run-20260818T201341Z-80b72aca3a` | `succeeded`, exit `0` | `4` | `76.5 s` | `19.15 s` | `107.57305241699214 s` (~`107.57 s`) | `24,696,061,952` bytes (~`24.70 GB`) | `render.mp4`, 2,401,122 bytes, SHA-256 `333bf787770ceb27c1c424633eecc84126f21ed94bcfb40be6df91de5d4bae86` |
| `H3MT-v2-rank256.safetensors` | `run-20260818T201553Z-bd34bac1ae` | `succeeded`, exit `0` | `4` | `82.0 s` | `20.50 s` | `112.06735883301008 s` (~`112.07 s`) | `26,950,919,782` bytes (~`26.95 GB`) | `render.mp4`, 2,455,517 bytes, SHA-256 `576290cfbb10d19ce8c3dd2a42c9092d7bfd11e61a560133d0071ba5fa798468` |

The exact receipt paths are:

```text
out/render-lab/run-20260818T201341Z-80b72aca3a/render-config.json
out/render-lab/run-20260818T201341Z-80b72aca3a/benchmark.json
out/render-lab/run-20260818T201341Z-80b72aca3a/run-status.json

out/render-lab/run-20260818T201553Z-bd34bac1ae/render-config.json
out/render-lab/run-20260818T201553Z-bd34bac1ae/benchmark.json
out/render-lab/run-20260818T201553Z-bd34bac1ae/run-status.json
```

## 6. Interpretation and observed timing/memory

The live proof establishes that both real H3MT files are usable through the
bounded flattened-target repair in production generation, not merely through
structural admission. It also establishes that:

- both files remain partial 188-target H3 LoRAs;
- LightX remains the sole scheduling owner;
- each H3MT file is an auxiliary model delta;
- streamed-AdaLN transformer routing remains in use; and
- the recorded allocator-release evidence completed successfully, with
  post-purge `mlx_cache=0.0B` lines for the text encoder, transformer, video
  VAE, and audio VAE release stages.

In these observed host runs, r128 completed with lower denoising time, total
elapsed time, and peak MLX memory than rank256. That is a diagnostic observation
only; it is not a scale, quality, or architecture recommendation.

## 7. Caveats and explicit non-claims

This closeout does not claim:

- a complete 208-target topology;
- an optimal LoRA scale;
- formal visual superiority of rank 128 or rank 256;
- mathematical equivalence between the two LoRAs;
- that rank 256 is required; or
- a completed scale-response curve.

Successful host generation is runtime acceptance evidence. It is not an
objective visual-quality metric, and no visual superiority claim is added here.
The 188-target partial topology is retained explicitly rather than promoted to
complete H3 coverage.

## 8. Next authorized slice

**Slice 021E — Render Lab Conditioning Artifact Replay** is the next bounded
target. Its initial scope is:

- optional manually entered conditioning-artifact `.npz` path;
- Canonical Qwen T2V only;
- validate the artifact against the prompt and checkpoint before H3 launch;
- pass `--conditioning-artifact` to `generate.py`;
- skip Qwen on valid replay; and
- record artifact identity in run evidence.

The following are future work and explicitly outside Slice 021E:

- managed conditioning-artifact directory;
- `index.json` catalog;
- dropdown selection;
- automatic caching; and
- artifact auto-generation.

Slice 021E is not started by this closeout. Slice 022 and Slice 023 remain
intact after this sequencing step.

## 9. Closeout boundary

- No production code changed.
- No test code changed.
- No H3, Qwen, VAE, MLX, Metal, or render was run during this closeout.
- No JSON was modified.
- Nothing was staged, committed, pushed, reset, stashed, cleaned, reverted,
  amended, or moved.

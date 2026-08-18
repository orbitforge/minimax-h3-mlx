# Slice 021C Host LoRA Stack Acceptance closeout

Closeout status: host accepted with bounded evidence on 2026-08-18.

This closeout records the three existing controlled Render Lab host runs for
Slice 021C, "Host LoRA Stack Acceptance." It separates durable runtime and
artifact receipts from the operator's qualitative visual judgment. It does not
authorize another render or begin Slice 022.

## 1. Slice objective and verdict

Slice 021C compared a Turbo-only baseline with one auxiliary LoRA and then a
two-auxiliary stack while holding the prompt, seed, geometry, text encoder,
Turbo preset, NFE, and base transformer constant.

The bounded verdict is **HOST ACCEPTED**:

- all three controlled runs completed successfully with four transformer
  forwards each;
- Turbo scheduling remained LightX-owned and unchanged across A/B/C;
- the MMH3 auxiliary was recorded and observed to add visible influence in B;
- Combat was recorded and observed to add further visible influence in C while
  retaining characteristics of the B result;
- ordered auxiliary identity and independent configured scales were preserved
  in immutable Render Lab evidence;
- the streamed-AdaLN Q6 transformer path remained active, with no non-streamed
  full-Q6 transformer path in any recorded command; and
- stage release and allocator evidence remained healthy in all three receipts.

This proves successful live coexistence and control/propagation of the stacked
auxiliaries. It does not claim a dedicated visual scale-response curve or a
formal numerical proof of no duplicate model residency.

## 2. Git and runtime boundary

| Field | Recorded value |
|---|---|
| Closeout worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021c` |
| Branch recorded by A/B/C | `slice/021c-host-lora-stack-acceptance` |
| HEAD recorded by A/B/C | `fc51894eda0d436eed48b42470e55a222470eb38` |
| Runtime transformer mode | `streamed-adaln-q6` |
| Runtime transformer path | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln` |
| Evidence boundary | Existing `out/render-lab/run-*` artifacts; not rewritten |

The closeout change is documentation-only. Production runtime code and tests
were not modified. This validation did not load H3, Qwen, a VAE, or any
transformer, did not invoke MLX, and did not perform another render. The
pre-existing untracked `.DS_Store` in the worktree is unrelated and was
preserved.

## 3. Frozen A/B/C controls

The following controls are equal in all three `render-config.json` files:

| Control | Frozen value |
|---|---|
| Mode | `T2V` |
| Prompt | Same artifact text in A/B/C; UTF-8 text SHA-256 `e7b00c29399a5f6b53814793f8862b6a3bf9424f75dde1a8afd43e55a67380c3` |
| Text encoder | `Canonical Qwen3-VL` / `canonical-qwen3-vl` |
| Geometry | Requested and runtime `512 × 512` |
| Duration | `5.0` seconds requested |
| Seed | `1701` |
| Turbo preset | `LightX 4-Step v0.1` / `lightx-4step-v01` |
| Runtime variant | `fl2va-turbo-4step-v0.1` |
| Effective NFE | `4` |
| Transformer mode | `streamed-adaln-q6` |
| Scheduling owner | `turbo-preset` |

The configs also record the same LightX scheduler identity and the same
streamed-AdaLN transformer identity. A/B/C command arrays all select the
streamed-AdaLN directory; none contains the non-streamed full-Q6 transformer
path.

## 4. A/B/C adapter-stack table

The scheduling adapter is separate from the ordered model-delta auxiliaries.
Paths and scales below are copied from the immutable run configs.

| Run | Scheduling owner | Ordered auxiliary stack |
|---|---|---|
| A — Turbo only | `turbo-preset` / LightX 4-Step v0.1 | None |
| B — Turbo + MMH3 | `turbo-preset` / LightX 4-Step v0.1 | Order 0: `/Users/elbancol/Downloads/MMH3-V1.safetensors` @ `0.8` |
| C — Turbo + MMH3 + Combat | `turbo-preset` / LightX 4-Step v0.1 | Order 0: `/Users/elbancol/Downloads/MMH3-V1.safetensors` @ `0.8`; order 1: `/Users/elbancol/Downloads/MiniMax_H3_Combat_LoRA.safetensors` @ `0.5` |

Each auxiliary is recorded with role `auxiliary-model-delta`. The LightX
source retained scheduling ownership; the auxiliaries did not replace or alter
the Turbo schedule.

## 5. Durable run IDs and output hashes

Each run directory contains parsed `render-config.json`, `benchmark.json`,
`run-status.json`, and `render.mp4`. The output hash was independently
re-read from the file and matched both structured receipts.

| Run | Durable run ID | Status / exit | Transformer forwards | Output size | Output SHA-256 |
|---|---|---:|---:|---:|---|
| A | `run-20260818T143124Z-7df8c791b8` | `succeeded` / `0` | `4` | `2,410,448` bytes | `58cb2afc5a2d2b67b6e205d7c10a55e59e8e3698eaa688e5b0562595aab09968` |
| B | `run-20260818T144457Z-015fbde672` | `succeeded` / `0` | `4` | `2,278,313` bytes | `db15fcfffdf95d740f9aff3f62a292a2fa6f7f87a5d380485aedfdb555fa1f0b` |
| C | `run-20260818T145551Z-646183ed1c` | `succeeded` / `0` | `4` | `2,420,992` bytes | `e5a68f2b70a96424d72ffd538567ff273d302dcbef0d55cc838af2fb2e487f06` |

The three run directories are:

```text
out/render-lab/run-20260818T143124Z-7df8c791b8
out/render-lab/run-20260818T144457Z-015fbde672
out/render-lab/run-20260818T145551Z-646183ed1c
```

## 6. Diagnostic timing and memory

These are diagnostic measurements from `benchmark.json`, not visual quality
scores.

| Run | Total elapsed | Seconds per forward | Peak MLX memory |
|---|---:|---:|---:|
| A | `97.992656667` s (about `97.99` s) | `16.875` s | `22,441,204,121` bytes |
| B | `101.074139999` s (about `101.07` s) | `17.8` s | `23,192,823,398` bytes |
| C | `106.590431875` s (about `106.59` s) | `19.15` s | `23,407,571,763` bytes |

The expected four-forward structure is present in each benchmark. The
increasing diagnostic cost in B and C is not itself a quality or residency
claim.

## 7. Scheduling-owner and ordered-auxiliary evidence

The immutable configs record `turbo.scheduling_owner` and
`auxiliary_lora_stack.scheduling_owner` as `turbo-preset` in all three runs.
They also retain the ordered auxiliary rows independently from the Turbo
adapter. The command receipts preserve:

- the LightX scheduling source and `--lightx-variant
  fl2va-turbo-4step-v0.1` in every run;
- no auxiliary flags in A;
- one `--additional-lora` / `--additional-lora-scale` pair for MMH3 at `0.8`
  in B; and
- the same MMH3 pair followed by Combat at `0.5` in C.

This is the live evidence for independent scale **control and propagation**:
the two distinct configured scales were carried through to the host runs while
the schedule remained unchanged.

## 8. Header and admission evidence

The following MLX-free/header-level evidence was completed before the live
renders and is recorded here without repeating adapter or model loading:

| Adapter / registry | Registered adapters | Registered targets | H3-compatible targets | Incompatible targets | Rank | Payload evidence |
|---|---:|---:|---:|---:|---:|---|
| MMH3 | `200` | `200` | `200` | `0` | `32` | `200` scalar alpha tensors; only `800` bytes of alpha-scalar payload read; no A/B matrix payload read |
| HM_V2 | `208` | `208` | `208` | `0` | `32` | No alpha payload read during registration |
| Combat | `208` | `208` | `208` | `0` | `16` | No alpha payload read during registration |

The composed registry counts were:

| Composition | Adapters / targets |
|---|---:|
| B: Turbo + MMH3 | `200 / 200` |
| C: Turbo + MMH3 + Combat | `408 / 208` |

The exact Render Lab A/B/C command-admission checks passed before the live
renders. That admission retained LightX Turbo as scheduling owner, kept
auxiliaries model-delta-only, and explicitly preserved the streamed-AdaLN
path. The existing MLX-free source contracts corroborating the admission
seams are `tests/test_style_lora_repair_contract.py::test_real_mmh3_header_admission_resolves_exactly_200_h3_core_targets`
and the ordered-stack contracts in
`tests/test_render_lab_additional_lora_contract.py`.

HM_V2 was **not** live rendered. Its result above is structural admission
only; it remains a future optional qualitative candidate.

## 9. Visual and operator assessment

Human judgment is kept separate from the diagnostic receipts:

- Run A was judged visually successful and "pretty rad".
- The operator noted that motion was excessive for the 4-step Turbo
  configuration.
- The visual feel was compared loosely with an early-2000s / PS2-era
  action-game aesthetic.
- That excessive motion was treated as a qualitative characteristic to revisit
  with another Turbo preset or more suitable schedule, not as a Slice 021C
  runtime failure.
- After the stacked runs, the operator expressed strong positive overall
  progress.

The conversation-side comparison recorded a visible change from A to B after
MMH3 and an additional visible change from B to C after Combat. C retained
characteristics of B rather than appearing to replace the first auxiliary
influence. These are qualitative visual observations, not numerical proof;
there is no durable numeric quality score or detailed B/C operator quote to
report.

## 10. Acceptance-goal disposition

| Acceptance goal | Disposition |
|---|---|
| Turbo scheduling remains unchanged | **Accepted.** Identical Turbo preset, runtime variant, effective NFE, scheduler identity, and four-forward structure are recorded across A/B/C. |
| MMH3 visibly affects output | **Accepted qualitatively.** B was observed to differ visibly from A; B's output also has a distinct immutable hash. |
| Second style LoRA adds influence | **Accepted qualitatively.** C was observed to differ further from B after Combat was appended. |
| Adapters coexist rather than replace one another | **Accepted within the bounded visual observation.** C retains characteristics from B, and the ordered C registry/configuration records both auxiliaries. |
| Independent scale changes behave sensibly | **Control/propagation proven; visual curve not claimed.** MMH3 `0.8` and Combat `0.5` were exercised live and preserved independently. No dedicated one-variable scale-sweep render was performed. |
| Streamed-AdaLN remains active | **Accepted.** All three configs and commands record `streamed-adaln-q6`. |
| No unintended full-Q6 load occurs | **Accepted at command/config evidence level.** No command selects the non-streamed full-Q6 transformer path. |
| No unnecessary duplicate model residency | **Lifecycle evidence healthy; formal numerical proof not claimed.** See the caveat below. |
| Stage release / allocator behavior | **Supported by receipts.** Each benchmark records four successful allocator purges and four post-purge lines with `mlx_cache=0.0B`. |

## 11. Explicit caveats and non-claims

- Independent scale **control and propagation** is proven by the live `0.8`
  and `0.5` values. A visual scale-response curve is not claimed because no
  dedicated scale sweep was run.
- The release evidence supports healthy stage cleanup. It is not a formal
  numerical proof that no duplicate model residency occurred.
- HM_V2 was header-admitted only and was not live rendered.
- Operator visual acceptance is qualitative. Hash differences, timing, and
  memory deltas do not by themselves prove visual influence.
- No numeric quality score or invented detailed operator quote is supplied for
  B or C.
- This closeout does not broaden Slice 021C to obtain optional additional
  renders.

## 12. Next slice

**Slice 022 — LoRA Render Lab UX and Evidence Polish** is the next development
slice. Its existing roadmap scope remains the authority; this closeout does
not begin or expand it.

## Evidence-validation receipt

The closeout validation was read-only and completed against the three local
run directories. It verified:

- the three required JSON files existed and parsed for every run;
- run IDs, success/status, zero exit codes, and four transformer forwards;
- equality of prompt, seed, width, height, requested duration, text encoder,
  Turbo preset, runtime variant, effective NFE, and transformer mode;
- exact A/B/C auxiliary cardinality, order, paths, roles, and scales;
- branch/HEAD and streamed-AdaLN identity in both top-level and runtime
  identity evidence;
- absence of the non-streamed full-Q6 transformer path from all commands;
- output existence, filesystem size, and SHA-256 equality with both benchmark
  and status evidence; and
- successful allocator-purge and post-purge cache evidence in each benchmark.

No run artifact was modified during validation. No full Python test suite was
run for this documentation-only closeout. No stage, commit, or push occurred.

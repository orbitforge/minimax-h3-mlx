# MiniMax H3 MLX Runtime Roadmap

**Last updated:** 2026-08-21

## Purpose

This document records the current medium-term development direction for the MiniMax H3 MLX runtime.

It is a planning guide, not an implementation-state authority.

If this roadmap conflicts with a current explicit slice prompt or higher-precedence canonical repository context, the more authoritative/current source wins.

---

## Current Production Baseline

The former `experiment/h3-generation` runtime has been promoted to `main`.

Current production capabilities include:

- native Apple Silicon MLX MiniMax H3 runtime;
- streamed-AdaLN Q6 transformer path;
- canonical Qwen conditioning and the existing text-only conditioning-artifact replay boundary;
- experimental Heretic state-28 text encoder path;
- generic LoRA support;
- LightX2V Turbo adapters;
- Larry Turbo adapters;
- repaired H3 style-LoRA target normalization;
- Render Lab Turbo presets;
- Render Lab text, image, and first/last-frame workflows.

The existing LoRA registry already supports additive multiple-adapter deltas internally, but the production CLI, pipeline, and Render Lab surfaces currently expose only one adapter source at a time.

---

# Near-Term Roadmap

## Slice 021A - Turbo + Auxiliary LoRA Stack Core

**Status:** Complete.

### Objective

Allow the production runtime to use:

- zero or one scheduling adapter;
- zero or more auxiliary/generic/style LoRAs;
- independent scales for each auxiliary LoRA.

### Scheduling rule

At most one adapter may own scheduling semantics.

A scheduling adapter may determine:

- Turbo enablement;
- NFE / step count;
- scheduler configuration;
- video/audio sigma shifts;
- LightX variant or manifest semantics.

Auxiliary LoRAs contribute model deltas only.

They must not alter scheduling behavior.

### Example

```text
Scheduling adapter:
LightX 4-Step v0.1

Auxiliary LoRAs:
MMH3-V1        scale 0.80
FilmStyle      scale 0.45
CharacterLoRA  scale 1.00
```

Conceptually:

```text
effective model
=
base H3
+ LightX delta
+ 0.80 * MMH3 delta
+ 0.45 * FilmStyle delta
+ 1.00 * CharacterLoRA delta
```

LightX alone controls the 4-step denoising schedule.

### Scope

Runtime, registry composition, pipeline, and CLI only.

No Render Lab UI changes.

No expensive host render required.

---

## Slice 021B - Render Lab Additional LoRAs

**Status:** Complete.

### Objective

Expose the runtime stacking capability cleanly in Render Lab.

Target UX:

```text
Acceleration / Schedule
[ LightX 4-Step v0.1 ]

Additional LoRAs
✓ MMH3-V1.safetensors       0.80
✓ FilmStyle.safetensors     0.45

[ + Add LoRA ]
```

### Important behavior

Selecting a Turbo preset must no longer consume or disable the ordinary/style-LoRA surface.

Turbo controls acceleration/scheduling.

Additional LoRAs independently influence model behavior.

With `None / Reference` selected as the Turbo preset, Additional LoRAs remain available.

No live acceptance render is required for this slice unless needed to resolve a specific implementation uncertainty.

---

## Slice 021C - Host LoRA Stack Acceptance

**Status:** Complete — host accepted. See [the Slice 021C closeout](slice-021c-host-lora-stack-acceptance-closeout.md).

### Objective

Prove actual stacked behavior using controlled host renders.

Use identical:

- prompt;
- seed;
- geometry;
- text encoder;
- Turbo preset;
- NFE;
- base transformer.

Compare:

```text
A. Turbo only
B. Turbo + MMH3-V1
C. Turbo + MMH3-V1 + second style LoRA
```

### Acceptance goals

Verify:

- Turbo scheduling remains unchanged;
- MMH3 visibly affects output;
- second style LoRA adds additional influence;
- adapters do not replace one another;
- independent scale changes behave sensibly;
- streamed-AdaLN remains active;
- no unintended full-Q6 load occurs;
- no unnecessary duplicate model residency occurs.

Visual quality is judged by the operator. Numerical metrics are diagnostic evidence only.

---

## Slice 021D - Flattened H3 LoRA Exporter Compatibility

**Status:** Complete — host accepted. See [the Slice 021D closeout](slice-021d-flattened-h3-lora-targets-closeout.md).

### Objective

Accept one proven, bounded flattened MiniMax-H3 LoRA exporter namespace at the
existing H3 target-normalization seam. The accepted forms map the four main
transformer projection targets for blocks `0..49` into the local `blocks.<n>`
namespace and remain fail-closed for unknown forms.

### Evidence

- both real 188-target H3MT files passed the MLX-free admission proof with
  `188/188` compatible targets and lazy A/B payload references;
- both files completed successful host Render Lab T2V runs with four
  transformer forwards; and
- LightX 4-Step v0.1 remained the scheduling owner while each H3MT file was an
  auxiliary model delta at scale `1.0` on the streamed-AdaLN path.

This slice does not claim a complete 208-target topology, optimal scale,
visual superiority, or mathematical equivalence between the two files.

---

## Slice 021E - Render Lab Conditioning Artifact Replay

**Status:** Complete — host accepted. See [the Slice 021E closeout](slice-021e-conditioning-artifact-replay-closeout.md).

### Delivered scope

- optional manually entered conditioning-artifact `.npz` path;
- Canonical Qwen T2V only;
- validate artifact, prompt, and checkpoint provenance before H3 launch;
- pass `--conditioning-artifact` to `generate.py`;
- skip Qwen and live prompt encoding on valid replay; and
- record artifact identity and tensor evidence in run evidence.

### Evidence

- focused MLX-free contracts passed `63/63`, with `py_compile`, embedded
  browser JavaScript `node --check`, and implementation `git diff --check`
  also passing;
- the canonical `hidden_states[50]` artifact was replayed with identity and
  tensor evidence preserved; and
- the bounded host replay succeeded with four transformer forwards and
  `SLICE_021E_HOST_ACCEPTANCE=PASS`.

The managed conditioning-artifact library is future work, not part of Slice
021E. It may later include a managed directory, `index.json` catalog,
dropdown selection, automatic caching, and artifact auto-generation.

---

## Slice 021F - FL2V Storyboard Chain + Simple Card UI

**Status:** Complete — host accepted. See [the Slice 021F closeout](slice-021f-fl2v-storyboard-closeout.md).

### Delivered scope

- Storyboard mode in Render Lab with ordered image cards;
- Finder/file-picker selection, drag/drop input, thumbnails, visible numbering,
  and card removal;
- exact adjacent-card derivation (`1 -> 2`, `2 -> 3`, `3 -> 4`, and so on);
- shared global render settings across all segments;
- sequential FL2V rendering with one independent child process per segment;
- parent-local segment videos and evidence with unique internal child run IDs;
- storyboard manifest, status, and simple overall progress evidence; and
- live Canonical Qwen image/text conditioning on the streamed-AdaLN Q6 path.

For `N` cards, the delivered workflow produces `N - 1` FL2V segments. Each
child exits and finalizes its evidence before the next adjacent pair launches.

Phase-one exclusions:

- per-transition duration;
- per-transition prompts;
- per-transition generation settings;
- sophisticated timeline editor;
- automatic final MP4 concatenation;
- managed conditioning-artifact library;
- artifact catalog/index implementation;
- SQLite/database work;
- automatic conditioning-artifact discovery; and
- automatic conditioning-artifact generation.

Card reordering was not part of the delivered phase-one contract.

### Evidence and boundaries

- Focused storyboard contracts passed `11/11`.
- Surrounding Render Lab validation passed; changed-Python compilation,
  embedded browser JavaScript syntax checking, and the implementation
  whitespace check also passed.
- The accepted host storyboard used live Canonical Qwen3-VL conditioning,
  three cards, two sequential FL2V segments, 512 × 512 geometry, five
  seconds per segment, seed `1701`, LightX FL2V Turbo 4-step, the streamed-
  AdaLN Q6 transformer, no additional LoRAs, and no external conditioning
  artifact replay.
- Both segments succeeded with exit code `0`; the recorded evidence showed
  four transformer forwards per segment and no observed overlap between the
  sequential child runs.
- Streamed-AdaLN Q6 remains canonical. A stale README path was reconciled to
  that production path. At Slice 021F closeout, the separate legacy local
  browser launcher still referenced the removed resident transformer; Slice
  021G subsequently repaired that legacy surface default to the canonical
  streamed-AdaLN Q6 transformer. The historical 021F closeout document
  remains unchanged.

Deferred progress/logging polish remains future work. This slice does not add
continuous live child-output streaming, explicit pending-segment records, or
richer per-segment progress.

---

## Slice 021G - Legacy Browser Surface Streamed-Transformer Safety

**Status:** Complete — implementation committed locally; no host render required.

### Delivered scope

- repaired the stale legacy browser surface transformer default;
- changed the transformer basename from `minimax-h3-mlx-6bit` to the canonical
  `minimax-h3-mlx-6bit-streamed-adaln` path;
- preserved the legacy `scripts/generate.py` command propagation through
  `--transformer <TRANSFORMER>`; and
- preserved explicit original/resident checkpoint loading for research and
  probe workflows.

### Evidence and boundaries

- focused 021G contract passed `3/3`;
- surrounding offline validation passed: Render Lab encoder safety `10/10`,
  Render Lab Turbo safety `5/5`, and LightX production-entry contract `7/7`;
- `py_compile` passed for both changed Python files and the implementation
  `git diff --check` passed; and
- no MLX, Metal, or H3 render was run, and no host render was required.

021G fixed only the stale legacy browser surface default. It did not globally
prohibit original/resident transformers, alter `scripts/generate.py`, alter
the H3 pipeline or loader, change transformer routing, change Render Lab,
change probes or research workflows, change `Launch MiniMax H3.command`, or
begin beta-0.6 conversion. Broader resident-transformer fail-closed safety
work and beta-0.6 conversion remain separate future work.

---

## Slice 022 - LoRA Render Lab UX and Evidence Polish

**Status:** Later development slice after Slice 021F.

Only after stacking is proven.

Candidate scope:

- per-LoRA enable/disable;
- clearer stack rows;
- remove/reorder controls where useful;
- compatibility diagnostics per adapter;
- ordered effective adapter-stack evidence;
- recorded scales;
- scheduling-owner identity;
- NFE and scheduler metadata;
- adapter source identities;
- reusable/saved LoRA stacks if actual workflow demonstrates value.

Do not introduce speculative architecture solely for hypothetical future needs.

---

## Slice 023 - Heretic Token Reconciliation

Resume the parked Heretic tokenizer-reconciliation work.

Current baseline:

- exact-piece token agreement proceeds normally;
- mismatch currently fails closed;
- UTF-8 span reconciliation research exists;
- overlap-weighted pooling and token-center interpolation have been explored;
- host state-28 probe artifacts exist;
- no winning reconciliation method has yet been accepted.

### Objective

Improve arbitrary-prompt compatibility for Heretic without weakening conditioning correctness or silently accepting ambiguous token mappings.

---

## Slice 024 - Monolithic BF16-to-MLX Q6/Q8 Conversion

**Status:** Complete for implementation and full host completion. The
subsequent beta acceptance operation used those structurally verified outputs:
the runtime and media pipeline structural gates passed, while human semantic
acceptance exposed the source-layout contract repaired by Slice 025. Slice
024's conversion and runtime structural contracts passed; Slice 024 itself is
not characterized as failed.
See the [historical Slice 024 closeout](slice-024-monolithic-q6-converter-closeout.md)
and the [host-completion addendum](slice-024-host-completion-addendum.md).

### Delivered scope

Slice 024 added a bounded-memory converter for the verified monolithic
MiniMax H3 beta-0.6 BF16 safetensors source. It provides header-first source
admission, strict topology classification, exact named-range reads, one
logical linear resident at a time, deterministic streamed sharded output, and
atomic publication into the repository's conventional MLX quantized
checkpoint format.

The locked recipe is Q6 core weights, Q8 block-AdaLN weights, group size `64`,
and `quantize_adaln=true`. The complete conventional output is expected to
contain `1,050` logical tensors.

### Evidence and boundaries

- the verified source identity and complete header bounds were recorded;
- a real host single-tensor Q6 proof passed with source immutability,
  bounded output verification, and peak-footprint telemetry;
- the focused monolithic contract passed `33/33`, checkpoint forge passed
  `30` with `2` skipped, and previously reported surrounding MLX-free suites
  passed `44/44`, `31/31`, and `5/5`; and
- `py_compile` for all six implementation paths and `git diff --check` passed.

- The subsequent real-host full conversion exited `0` after `111` seconds,
  produced six shards and `1,050` logical tensors, and was independently
  accepted as `CONVENTIONAL_BETA_Q6_VERIFIED`.
- The subsequent real-host streamed-AdaLN forge exited `0` after `135`
  seconds, produced an `850`-tensor resident base and `50` block sidecars,
  and was independently accepted as
  `BETA_STREAMED_ADALN_CHECKPOINT_VERIFIED`.
- The existing production streamed-AdaLN runtime remains unchanged. The
  subsequent pre-repair beta acceptance operation used live Canonical Qwen3-VL
  conditioning, loaded the streamed beta transformer, completed `15`
  denoising forwards, ran video and audio VAE paths, produced a structurally
  valid MP4, and reclaimed memory. Human media acceptance failed at both
  `128×128` and `512×512`: output was muddy/low-contrast with no recognizable
  prompt semantics. This was the semantic source-layout discovery that led to
  Slice 025, not a failure of Slice 024's conversion or runtime structural
  contracts.
- The host operation detected no source mutation and no modified tracked
  repository paths; detailed source, artifact, hash, and resource receipts
  are recorded in the host-completion addendum.

The original Slice 024 closeout is a historical snapshot of the pre-host
state and remains unchanged. The host completion did not run beta H3 runtime
execution, live Qwen conditioning, denoising/generation, VAE execution, or
video/audio render acceptance; that later pre-repair acceptance operation is
recorded in the Slice 025 closeout. Corrected semantic output remains
unproven. Continuation, latent masking, and PR 15375 research are not Slice
024 deliverables.

### Next authorized step

The next operation after the pre-repair acceptance finding was the separately
authorized Slice 025 beta QKV layout reconciliation. The corrected full
conversion, verification, forge, and semantic acceptance gate are recorded as
the next host operation in the Slice 025 closeout.

---

## Slice 025 - Beta QKV Layout Reconciliation

**Status:** Closeout ready — implementation committed at `26869fb`; Slice 026
subsequently hardened named beta runtime selection at `685c6f2`. Corrected
beta conversion, verification, forge, named-runtime generation, and semantic
acceptance remain unproven. See the [Slice 025 closeout](slice-025-beta-qkv-layout-reconciliation-closeout.md)
and the [Slice 026 closeout](slice-026-beta-runtime-hardening-closeout.md).

### Objective

Reconcile the accepted PinkCherry beta fused-QKV source row order with the
runtime-native MLX row order before the existing Q6 quantization path, while
keeping runtime attention semantics, quantization policy, and streamed-AdaLN
format unchanged.

### Delivered scope

- exact beta source admission as `grouped_qkv`, bound to the accepted source
  identity and checked-in authorization receipt;
- fail-closed rejection of unknown, ambiguous, or contradictory layouts, with
  runtime-native sources remaining a no-op;
- grouped `[Q_all;K_all;V_all]` to runtime-native
  `[head0:q,k,v][head1:q,k,v]...` reconciliation before Q6 quantization;
- exhaustive coverage of `50` main-block and `2` token-refiner fused-QKV
  weights, with zero fused-QKV biases; and
- separate planning and execution receipts for QKV reconciliation.

### Evidence and boundaries

- independent payload evidence covered all `52/52` fused-QKV weights;
- direct beta-versus-runtime comparison had relative L2 range
  `1.400078–1.410067`, mean `1.405500`, and cosine range
  `0.006202–0.020231`;
- grouped-to-runtime reconciliation had relative L2 range
  `0.022318–0.026479`, mean `0.023695`, and cosine range
  `0.999650–0.999751`;
- the final accepted validation recorded `55/55` monolithic tests,
  `32 PASS` with `2` MLX-gated skips for checkpoint forge, exact source SHA
  verification, `py_compile` pass, `git diff --check` pass, and an
  independent second review of `PASS_WITH_GAPS` with the original P1 and P2
  resolved; and
- no corrected full conversion, conventional verification, streamed-AdaLN
  forge, MLX/Metal generation, or human media acceptance was run for Slice
  025.

### Next authorized step

Slice 026 subsequently addressed the operational runtime-selection gap with an
explicit, fail-closed `beta-0.6` named runtime and metadata-only pre-load
admission. Slice 026 did not run the corrected host operation: full
BF16-to-conventional Q6/Q8 conversion, independent verification,
streamed-AdaLN forge, independent streamed-checkpoint verification, named
runtime generation, or semantic acceptance remain unproven. The earlier
manual-transformer host run remains pre-repair evidence and is not proof of
the named-runtime CLI path. See the [Slice 026 closeout](slice-026-beta-runtime-hardening-closeout.md).

---

## Slice 026 - Beta Runtime Hardening

**Status:** Closeout ready — implementation committed at `685c6f2`; metadata
admission and runtime-selection contracts are accepted with gaps. No
model-backed generation or Render Surface integration was run. See the
[Slice 026 closeout](slice-026-beta-runtime-hardening-closeout.md).

### Objective

Replace fragile caller-composed beta runtime selection with the explicit,
opt-in named runtime `beta-0.6`. The named runtime represents the complete
accepted combination: the surrounding checkpoint root, Canonical Qwen3-VL,
tokenizer, processor, video and audio VAEs, scheduler/config contract,
corrected conventional provenance, corrected streamed-AdaLN transformer,
transformer configuration, Slice 025 QKV reconciliation, Q6/Q8 policy, and
streamed topology. It is not the default. Legacy explicit
`--checkpoint`/`--transformer` selection remains available when no named
runtime is requested, while named selection rejects conflicting manual
overrides.

### Delivered scope

- `--runtime beta-0.6` with `--runtime-assets` and
  `MINIMAX_H3_RUNTIME_ASSETS`, with the CLI value taking precedence over the
  environment value;
- a host-local `<runtime-assets>/beta-0.6/` profile whose current Slice 026
  deployment contract requires symbolic links named `checkpoint`,
  `transformer`, and `conventional`;
- fail-closed pre-load admission for streamed provenance, transformer
  configuration, tokenizer/processor metadata, Qwen/VAE identity, and
  scheduler metadata before lazy pipeline import and model construction;
- semantic transformer-architecture admission with a resolved config
  SHA/architecture receipt;
- Slice 025 QKV source/layout/count, quantization, and streamed-topology
  provenance admission; and
- a resolved-runtime receipt that reports validated facts rather than only
  caller intent. Conventional metadata remains required as provenance
  hardening, but conventional tensor payloads are not loaded by named-runtime
  admission.

The current symbolic-link deployment rule is an operational Slice 026 host
contract, not an eternal requirement of the model format.

### Evidence and boundaries

- the runtime-selection suite passed `23/23`;
- real-host positive and negative probes were respectively
  `ACCEPTED_METADATA_ONLY` and `REJECTED_METADATA_ONLY`;
- invalid transformer configuration was rejected before pipeline import with
  zero safetensors payload bytes read and no Qwen, VAE, MLX, Metal, or model
  construction;
- the legacy surface passed `3/3`, conditioning contracts passed `6/6`,
  LightX metadata contracts passed, `generate.py --help` passed,
  `py_compile` passed, and `git diff --check` passed; and
- the second independent review returned `PASS_WITH_GAPS` after resolving
  the original admission blocker.

Slice 026 does not prove full MLX generation through
`--runtime beta-0.6`, Render Surface integration,
Render Surface-to-named-runtime corrected beta generation, or a Q6-versus-Q8
beta comparison.

### Next authorized step

The next engineering slice is **Render Surface integration**. The intended
path is:

```text
Render Surface
→ named beta-0.6 runtime
→ hardened pre-load admission
→ corrected streamed beta transformer
→ Canonical Qwen/VAE components
→ final media
```

The UI must consume the named runtime contract rather than reconstructing the
checkpoint root, transformer override, or Slice 025 artifact history. A full
named-runtime render may serve as part of post-integration host acceptance.
This roadmap update does not implement that work.

---

# Later Workflow Improvements

After LoRA stacking and Heretic reconciliation, reassess based on actual Render Lab usage.

Possible areas include:

- saved generation presets;
- saved LoRA stacks;
- improved run comparison;
- conditioning-artifact workflow improvements;
- first/last-frame workflow improvements;
- encoder selection and provenance UX;
- better evidence browsing;
- additional runtime performance work where measurements justify it.

These are candidates, not committed slices.

---

# Development Principles

- Keep slices bounded.
- Protect known-good production behavior.
- Prefer explicit scheduling ownership.
- Keep scheduling semantics separate from ordinary model deltas.
- Preserve lazy adapter loading.
- Preserve streamed-AdaLN architecture.
- Never merge LoRA deltas into quantized base weights.
- Never requantize the base merely to apply an adapter.
- Preserve failure artifacts and classify failures before changing architecture.
- Do not perform expensive host rendering when offline contracts can establish correctness.
- Do not reset, stash, clean, or revert unrelated dirty research for convenience.
- Stage, commit, and push remain separate authorization gates.

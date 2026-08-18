# MiniMax H3 MLX Runtime Roadmap

**Last updated:** 2026-08-18

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

**Status:** Next bounded target.

### Initial scope

- optional manually entered conditioning-artifact `.npz` path;
- Canonical Qwen T2V only;
- validate the artifact against the prompt and checkpoint before H3 launch;
- pass `--conditioning-artifact` to `generate.py`;
- skip Qwen on valid replay; and
- record artifact identity in run evidence.

The managed conditioning-artifact library is future work, not part of Slice
021E. It may later include a managed directory, `index.json` catalog,
dropdown selection, automatic caching, and artifact auto-generation.

---

## Slice 022 - LoRA Render Lab UX and Evidence Polish

**Status:** Next development slice after Slice 021E.

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

# Slice 029 — Render Lab Beta Selection closeout

This is the documentation closeout for the committed Slice 029 implementation.
The implementation is complete and independently reviewed; post-promotion
Render Lab UI/storyboard host acceptance remains required. No model generation,
Qwen/VAE execution, MLX/Metal execution, conversion, forge, or artifact change
was performed for this closeout.

## A. Closeout verdict

```text
SLICE_029_IMPLEMENTATION_REVIEW_COMPLETE_HOST_ACCEPTANCE_PENDING
```

Slice 029 restores the Render Lab as the canonical operator surface and adds a
logical model selector. It records the final committed implementation, not the
abandoned earlier Turbo-checkbox design.

## B. Git and implementation identity

| Field | Recorded value |
|---|---|
| Original base | `31c5e79abec0f3cdec4f1424c097ffb14ff4efa6` |
| Implementation commit | `97175e6777a5301a1f9c49c92005486b83d39533` |
| Implementation subject | `Add beta selection to Render Lab` |
| Branch | `slice/029-render-surface-turbo` |
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-029` |
| Canonical operator surface | Render Lab |

The implementation commit contains the Render Lab launcher, shared streamed
checkpoint metadata admission, model-selection runner/server/UI wiring, model
selection contracts, and the small compatibility updates required by that
contract. It does not contain a new model artifact.

## C. Recorded product contract

`Launch MiniMax H3.command` now requires the repository interpreter:

```text
<repository>/.venv/bin/python
```

It fails closed when that interpreter is unavailable and launches:

```text
tools/render_lab/server.py --open
```

The browser accepts logical model identity only. It exposes exactly these
choices and resolves each to its exact admitted streamed transformer:

| Logical choice | Internal ID | Exact streamed transformer |
|---|---|---|
| Current | `current` | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln` |
| Beta 0.6 | `beta-0.6` | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-beta-0.6-q6-q8-corrected-slice-025-streamed-adaln` |

The default logical model is `Beta 0.6`. Arbitrary transformer paths and a
path that does not match the selected logical model are rejected.

The selected model propagates globally through FL2V storyboard children. The
existing contract remains unchanged: `N` cards produce exactly `N - 1`
adjacent segments, executed sequentially, with one global model selection.
There is no per-segment model selection and no automatic storyboard
concatenation.

The default Turbo preset is `LightX 4-Step v0.1` with NFE `4`. Existing native
LightX command ownership is preserved: the curated preset supplies the native
LightX adapter, variant, scale, and matching step count, while the existing
generator/pipeline and LightX manifest retain scheduling ownership. Slice 029
adds no Turbo implementation and introduces no generator Turbo default.

## D. Streamed-transformer admission

Render Lab and the loader share an MLX-free, metadata-only streamed-AdaLN
admission contract. It fails closed on:

- required config and quantization metadata, including the accepted DiT
  architecture and Q6 core/Q8 AdaLN, group-size-64 recipe;
- derived format/schema, verified completion, exact topology, 850 base
  tensors, 1,050 total tensors, five base shards, 50 sidecars, and 200
  sidecar tensors;
- physical base-shard ownership, exact sidecar file set, block ordering,
  sidecar roles, shapes, dtypes, and quantization metadata; and
- per-tensor byte counts derived from shape × dtype width plus the base-index
  and sidecar aggregate byte counts.

The final independent byte-count receipt used BF16 = 2 bytes and U32 = 4 bytes,
rejected all tested omission/type/sign/shape/dtype/count mutations, and
accepted the real assets metadata-only with these aggregate counts:

```text
base:    16,464,048,640 bytes
sidecar: 13,828,147,200 bytes
```

Both the real Current transformer and the corrected Beta 0.6 transformer were
admitted through this metadata-only path. No tensor payload was opened or
hashed for that admission.

## E. Implementation and independent-review history

The pre-commit review cycle found and repaired the following defects before
`97175e6777a5301a1f9c49c92005486b83d39533`:

1. The first independent review found that merely existing `config.json` and
   `quant_config.json` files could pass. The repair extracted the established
   checkpoint-format validator and made config, quantization, topology, and
   sidecar admission semantic and fail-closed.
2. The next independent review found that a mismatched sidecar `byte_count`
   and zero root aggregates could pass. The repair bound every declared count
   to shape × dtype and bound both root aggregates to independently validated
   metadata totals.
3. The final independent byte-count acceptance was `PASS`. It independently
   rejected 9/9 per-tensor negative cases and 12/12 root negative cases,
   including the `byte_count=1` versus expected `193,536` case and the altered
   child-plus-aggregate case, while admitting both real assets without payload
   reads.

The final MLX-free validation receipts were:

| Check | Result |
|---|---|
| Slice 029 model-selection/byte-count contracts | `17/17 PASS` |
| Runtime-selection metadata contracts | `27/27 PASS` |
| Broader MLX-free regression recorded during review | `128 tests PASS` |
| Checkpoint-forge contracts | `32 PASS, 2 skipped` |
| Python compilation | `PASS` |
| Embedded browser JavaScript syntax | `PASS` |
| `git diff --check` | `PASS` |
| MLX-backed loader tests | Not available: `[metal::load_device] No Metal device available` |

No model generation was performed during implementation or review. The Metal
failure is an environment boundary, not runtime or media evidence.

## F. External host-evidence boundary

The following evidence predates Slice 029 and is recorded here only as
external host evidence:

- Current streamed transformer plus LightX FL2V Turbo 4-step had prior Render
  Lab host acceptance.
- The corrected streamed Beta 0.6 plus LightX FL2V Turbo 4-step was manually
  host-run successfully and visually accepted before this implementation.

Slice 029 itself did not perform either host render. Post-promotion acceptance
must still exercise the Render Lab UI and FL2V storyboard path with the logical
Current/Beta selector, preserve the existing sequential storyboard evidence,
and obtain operator media acceptance. Metadata admission, contract tests, and
the external host evidence above do not substitute for that gate.

## G. Explicit non-claims

Slice 029 introduces or claims none of the following:

- new Beta conversion;
- new QKV reconciliation;
- quantization change;
- scheduler change;
- LightX manifest or math change;
- streamed-AdaLN runtime change;
- new generator Turbo ownership;
- model artifact changes;
- full host Render Lab acceptance by Slice 029;
- automatic storyboard concatenation; or
- per-segment model selection.

## H. Publication boundary and next action

The documentation closeout changes only this closeout, the genuinely stale
Slice 029 roadmap state, and the README's normal operator-launch note. No
production or test code is part of the closeout, and the two pre-existing dirty
test paths remain outside it. No push is performed.

After this local closeout commit, the next action is:

```text
READY_FOR_SLICE_029_PROMOTION
```

Promotion must retain the post-promotion Render Lab UI/storyboard host
acceptance gate described above.

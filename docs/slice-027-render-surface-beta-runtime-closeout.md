# Slice 027 — Render Surface Beta Runtime Integration closeout

This is a documentation-only closeout for the already committed Slice 027
implementation. No production code, tests, runtime-selection implementation,
model artifacts, or runtime-assets symlinks were changed or created while
preparing this closeout. Generation, Qwen, VAE, MLX/Metal execution,
conversion, forge, and Q6-versus-Q8 comparison were not run. No Git staging,
commit, push, reset, stash, clean, revert, or repair was performed.

## 1. Closeout verdict and scope

```text
SLICE_027_CLOSEOUT_READY
```

Slice 027 integrates the accepted Slice 026 named runtime into the existing
legacy browser Render Surface. Before Slice 027, that surface could launch
only its existing Current/manual generation path. After Slice 027, its
user-facing runtime selector contains:

| Display | Internal identifier |
|---|---|
| `Current` | `current` |
| `Beta 0.6` | `beta-0.6` |

`Current` remains the default. Beta remains explicit and opt-in; it did not
become the default.

The implementation commit is:

```text
6d117cc444237e75eeae0771e27745b30c6f3da3 Add beta runtime to Render Surface
```

The implementation is committed. This closeout is local documentation state
only and remains uncommitted and unstaged.

## 2. Repository and Git boundary

The required boundary was established before editing:

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-027` |
| Branch | `slice/027-render-surface-beta-runtime` |
| HEAD | `6d117cc444237e75eeae0771e27745b30c6f3da3` |
| HEAD subject | `Add beta runtime to Render Surface` |
| Starting status | Clean |
| Starting staged paths | None |
| Starting modified tracked paths | None |
| Starting untracked paths | None |

The implementation commit changed exactly:

- `scripts/minimax_h3_surface.py`; and
- `tests/test_legacy_surface_transformer_contract.py`.

The documentation closeout changes exactly:

- `docs/ROADMAP.md`; and
- `docs/slice-027-render-surface-beta-runtime-closeout.md`.

README, historical closeouts, continuity JSON, runtime-selection
implementation, model/checkpoint artifacts, and all other production and test
paths remain outside this closeout.

## 3. Recorded surface architecture

The changed surface is the legacy browser Render Surface. Its authoritative
launch seam is `start_job()`, which launches `scripts/generate.py` as a child
process. The surface copies the existing process environment into that child
while adding its repository `PYTHONPATH`.

Render Lab and storyboard use separate contracts and were not modified. Slice
027 is not a universal Render Lab/storyboard integration.

## 4. Recorded Beta selector and named-runtime contract

The selector displays `Beta 0.6` and carries the internal identifier
`beta-0.6`. When Beta is selected, `start_job()` launches:

```text
--runtime beta-0.6
```

The UI does not manually assemble Beta using:

- `--checkpoint`;
- `--transformer`;
- Slice 025 corrected artifact paths;
- conventional artifact paths; or
- QKV row-layout details.

Slice 026 remains the single owner of Beta runtime admission and compatibility
validation. Current/manual selection retains the prior checkpoint/transformer
launch contract, and an omitted runtime selection retains `Current`.

## 5. Recorded runtime-assets contract

The legacy surface inherits the existing process environment when it launches
the child. Therefore an existing:

```text
MINIMAX_H3_RUNTIME_ASSETS
```

value is inherited by `scripts/generate.py`.

No host-specific runtime-assets path is embedded in reusable UI code. No new
asset registry or settings subsystem was introduced. Actual host profile
symlinks were not created during Slice 027 implementation or this closeout.

Post-promotion Beta acceptance requires the current Slice 026 profile:

```text
<runtime-assets>/beta-0.6/
  checkpoint    -> <accepted surrounding checkpoint>
  transformer   -> <accepted corrected streamed Slice 025 transformer>
  conventional  -> <accepted corrected conventional Slice 025 artifact>
```

All three entries are required to be symbolic links under the current Slice
026 profile contract. The links are host setup, not reusable UI state, and
were not created here.

## 6. Recorded parameter contract

The legacy browser surface exposes and preserves these parameters across
Current and Beta:

| UI parameter | Behavior |
|---|---|
| Prompt | Passed to the child generator |
| Size / megapixels preset | Preserves the existing size behavior |
| Duration | Passed to the child generator |
| Output path | Passed to the child generator |

Seed and steps are not exposed or explicitly passed by this legacy surface;
they remain generator defaults. Explicit width and height are likewise not
direct legacy-surface parameters; the surface uses its existing
size/megapixels behavior.

## 7. Recorded failure and no-fallback contract

The surface rejects unsupported runtime values before child launch. Invalid
Beta admission remains visible through the existing log/status path, and a
child nonzero exit status is surfaced.

Beta failure does not:

- retry without `--runtime`;
- substitute Current/manual checkpoint behavior; or
- silently fall back.

The independent failure-path process probe observed Beta admission failure
with exit code `2` and no retry or fallback:

```text
NO_FALLBACK_CONFIRMED
```

## 8. Recorded status, logging, and feature scope

Existing status/logging exposes enough information to identify:

- `Beta 0.6 (beta-0.6)`;
- the child command;
- running state;
- child exit code; and
- Slice 026 admission output/error.

The full detailed Slice 026 provenance receipt is not claimed as primary UI.

The legacy browser surface does not expose:

- LoRA;
- Turbo;
- Heretic;
- storyboard;
- image-conditioning; or
- manual transformer controls.

Slice 027 therefore did not broaden Beta compatibility to those features.
Render Lab and storyboard remain separate surfaces/contracts.

## 9. Recorded implementation review history

The implementation receipt is:

```text
SLICE_027_IMPLEMENTED_AND_VALIDATED
```

The independent review result is:

```text
PASS_WITH_GAPS
```

Review findings preserved by this closeout:

| Review area | Result |
|---|---|
| Surface architecture | `PASS` |
| Beta selector | `PASS` |
| Named-runtime launch | `PASS` |
| No manual Beta assembly | `PASS` |
| Current regression | `PASS` |
| Runtime-assets environment propagation | `PASS` |
| Failure/no-fallback | `NO_FALLBACK_CONFIRMED` |
| Parameter preservation | `PASS` — seed/steps are generator defaults |
| Feature compatibility | `PASS` |
| Status/logging | `PASS` |
| Invalid-runtime rejection | `PASS` |
| Diff/scope | `PASS` |

This review history does not characterize host generation or media as
accepted.

## 10. Recorded MLX-free validation

The accepted Slice 027 validation is:

| Check | Result |
|---|---|
| Legacy/focused surface | `10/10 PASS` |
| Slice 026 runtime-selection | `23/23 PASS` |
| Render Lab | `19/19 PASS` |
| Storyboard | `11/11 PASS` |
| Encoder | `10/10 PASS` |
| Turbo | `5/5 PASS` |
| Conditioning | `6/6 PASS` |
| Render Lab conditioning | `6/6 PASS` |
| Additional LoRA | `12/12 PASS` |
| `generate.py --help` | `PASS` |
| `py_compile` | `PASS` |
| Implementation `git diff --check` | `PASS` |
| Independent review | `PASS_WITH_GAPS` |

An additional LightX production-entry check reached the environment-only
Metal boundary:

```text
[metal::load_device] No Metal device available
```

That boundary is not a Slice 027 implementation failure. No generation,
Qwen/VAE execution, MLX/Metal runtime, or render acceptance was run as part
of this documentation closeout.

## 11. Recorded unproven and deferred host work

The following remain **NOT YET PROVEN**:

- positive real-host Beta profile admission;
- full legacy Render Surface → Beta 0.6 MLX generation;
- real Qwen execution through the UI path;
- corrected streamed transformer execution through the UI path;
- real VAE execution through the UI path;
- final media acceptance;
- human visual acceptance; and
- Q6-vs-Q8 comparison.

No positive metadata-only host probe was run because the runtime-assets
profile does not yet exist. The earlier Slice 025 manual-transformer fox run
is not proof of the new Slice 027 UI path.

## 12. Recorded required host setup

Before real UI acceptance, establish a runtime-assets root containing:

```text
<runtime-assets>/beta-0.6/checkpoint
```

symbolically linked to the accepted surrounding checkpoint;

```text
<runtime-assets>/beta-0.6/transformer
```

symbolically linked to the accepted corrected streamed Slice 025 transformer;
and:

```text
<runtime-assets>/beta-0.6/conventional
```

symbolically linked to the accepted corrected conventional Slice 025
artifact. All three links are required by the current Slice 026 profile
contract and were not created in this closeout.

The linked Slice 027 worktree does not contain:

```text
.venv/bin/python
```

The canonical host-launch interpreter arrangement must be inspected and
resolved before real UI acceptance. This closeout does not prescribe copying
a virtualenv; the correct solution must be established from repository/runtime
evidence.

## 13. Recorded post-promotion host acceptance plan

After Slice 027 is promoted/published, the intended sequence is:

1. Inspect and establish the canonical host interpreter path.
2. Create the runtime-assets Beta profile symlinks.
3. Set `MINIMAX_H3_RUNTIME_ASSETS`.
4. Perform a positive metadata-only admission probe if practical.
5. Launch the actual legacy Render Surface.
6. Select `Beta 0.6`.
7. Run a real generation.
8. Verify the path reaches named `beta-0.6`, hardened Slice 026 admission,
   the corrected streamed Slice 025 transformer, Canonical Qwen, video/audio
   VAEs, and final media.
9. Perform human media acceptance.

This host proof may jointly close the remaining runtime acceptance gaps from
Slices 026 and 027.

## 14. Q6 versus Q8 remains later work

The controlled Beta Q6-vs-Q8 comparison remains a later optimization study.
It is not part of Slice 027 closeout or immediate host acceptance. No Q8
quality or performance conclusion is implemented or characterized here.

## 15. Documentation validation and publication boundary

After the documentation edits:

- no continuity JSON file was changed, so no JSON continuity validation was
  required;
- the complete documentation diff was inspected;
- `git diff --check` was run;
- the changed-path allowlist contains only `docs/ROADMAP.md` and this
  closeout;
- no production, test, runtime-selection, or model-artifact path changed;
- no runtime-assets symlinks were created; and
- no checkpoint or other model artifact changed.

The final local publication state is intentionally:

```text
DOCS_ONLY=PASS
STAGED=NONE
COMMIT=NOT_PERFORMED
PUSH=NOT_PERFORMED
PRODUCTION_CHANGES=NONE
TEST_CHANGES=NONE
RUNTIME_SELECTION_CHANGES=NONE
MODEL_ARTIFACT_CHANGES=NONE
RUNTIME_ASSETS_SYMLINKS=NOT_CREATED
```

## 16. Final Git state and next action

The documentation-only closeout leaves these local changes for review:

- modified: `docs/ROADMAP.md`;
- untracked: `docs/slice-027-render-surface-beta-runtime-closeout.md`;
- staged paths: none; and
- commit/push: not performed.

The final next-action verdict is:

```text
READY_FOR_SLICE_027_CLOSEOUT_COMMIT
```

This closeout stops at validated, uncommitted documentation.

# Slice 021F — FL2V Storyboard Chain + Simple Card UI closeout

Closeout status: **Complete — host accepted (bounded).**

This document records the committed Slice 021F implementation, its existing
static and MLX-free validation, and the existing real host storyboard
acceptance. This is a documentation closeout only. No new H3, Qwen, VAE, MLX,
Metal, or render execution was performed while closing out the slice.

## 1. Objective and verdict

Slice 021F delivered a bounded Render Lab storyboard workflow for adjacent
first/last-frame image transitions. The workflow accepts an ordered set of
image cards, derives adjacent FL2V segments, applies one shared global render
configuration, runs one independent child process per segment, and preserves
per-segment output and evidence in one parent storyboard run folder.

The bounded verdict is:

```text
SLICE_021F_HOST_ACCEPTANCE=PASS
```

The acceptance proves the recorded three-card, two-segment storyboard on the
canonical streamed-AdaLN Q6 Render Lab path. Publication or promotion remains
a later explicit gate.

## 2. Git and change boundary

| Field | Recorded value |
|---|---|
| Closeout worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021f` |
| Branch | `slice/021f-fl2v-storyboard` |
| HEAD | `9e9974017a2f1366ee93f269a2ce6821525bd473` |
| Expected parent | `a7bbb9fdd50b22c92ab73d36836276e03882bdcc` |
| Implementation commit | `9e9974017a2f1366ee93f269a2ce6821525bd473` — Add FL2V storyboard rendering |
| Pre-edit tracked state | Clean |
| Pre-edit index | Clean; no staged paths |
| Pre-existing untracked state | `.DS_Store`, preserved untouched |

The implementation commit changed exactly:

- `tools/render_lab/runner.py`;
- `tools/render_lab/server.py`; and
- `tests/test_render_lab_storyboard_contract.py`.

This documentation reconciliation changes exactly:

- `README.md`;
- `docs/ROADMAP.md`; and
- `docs/slice-021f-fl2v-storyboard-closeout.md`.

The README correction changes only the stale canonical local transformer value
from the absent resident `minimax-h3-mlx-6bit` directory to
`minimax-h3-mlx-6bit-streamed-adaln`. Historical original-to-derived forge
examples retain the resident path where it is intentionally the source
checkpoint. No global replacement was performed.

A separate legacy local browser launcher remains stale:
`Launch MiniMax H3.command` still launches `scripts/minimax_h3_surface.py`,
which references the removed resident transformer directory. Those runtime
paths were not modified in this closeout. They are deferred legacy/runtime-
safety debt, not a 021F feature defect, and do not weaken the accepted Render
Lab host evidence.

The intentionally dirty `experiment/h3-generation` worktree was not touched.

## 3. Feature contract

The delivered phase-one contract includes:

- a minimum of two image cards;
- Finder/file-picker image selection;
- drag/drop image input;
- thumbnail previews;
- visible card numbering;
- card removal;
- append-order cards with exact adjacent pairing;
- simple storyboard progress/status; and
- one shared/global render configuration for all segments.

For `N` cards, the workflow creates `N - 1` FL2V segments:

```text
1 -> 2
2 -> 3
...
N-1 -> N
```

Phase-one exclusions are per-segment prompts, per-segment duration,
per-segment render settings, a timeline editor, automatic concatenation, and a
managed conditioning library.

## 4. Process-isolation design

Each storyboard segment is executed as its own completely separate
`scripts/generate.py` child process. The lifecycle is:

```text
segment admission
  -> launch child
  -> wait for exact child to exit
  -> finalize segment evidence
  -> launch next segment
```

The implementation preserves one independent child process per segment and
stops the storyboard after the first failed child. A separate process identity
is required; a separate directory identity is not. Process exit is the
deliberate memory-isolation boundary, so no long-lived H3 process serves
multiple segments and no transformer, VAE, or MLX objects are intentionally
retained between segment processes.

## 5. Storyboard artifact model

One storyboard uses one parent run folder. The accepted parent-local model is:

```text
inputs/
  card-01.png
  card-02.png
  card-03.png

render-01.mp4
render-01.config.json
render-01.status.json
render-01.stdout.log
render-01.stderr.log
render-01.benchmark.json
render-01.telemetry/
  before.json
  after.json

render-02.mp4
render-02.config.json
render-02.status.json
render-02.stdout.log
render-02.stderr.log
render-02.benchmark.json
render-02.telemetry/
  before.json
  after.json

storyboard-manifest.json
run-status.json
```

If the requested base output is `sword-fight.mp4`, segment outputs are
`sword-fight-01.mp4`, `sword-fight-02.mp4`, and so on. Segment filenames are
deterministically ordered and collision-safe.

Each segment retains a unique internal child run identity in its config,
status, benchmark, and parent manifest evidence even though its artifacts
share the parent directory. The parent manifest records card identities,
shared settings, local artifact paths, child run IDs, statuses, exit codes, and
failure information.

No automatic final MP4 concatenation was added.

## 6. Conditioning safety boundary

Slice 021E external conditioning-artifact replay remains Canonical Qwen T2V
only. Slice 021F is FL2V / `FIRST_LAST` image-conditioned generation and uses
live Canonical Qwen3-VL image/text conditioning.

Therefore, the storyboard path does not allow an external
`conditioning_artifact_path` to bypass image-aware conditioning, does not
enable Heretic storyboard FL2V, and does not alter the 021E T2V replay
semantics. The accepted storyboard evidence records:

```text
conditioning_source = live-encoder
```

## 7. Implementation validation receipt

The existing implementation validation was completed before this
documentation closeout:

| Check | Result |
|---|---|
| Focused storyboard contracts | `11/11` passed |
| Surrounding Render Lab validation | Passed; no fabricated count recorded here |
| Changed-Python compilation | `py_compile` passed |
| Embedded browser JavaScript syntax | `node --check` passed |
| Implementation whitespace check | `git diff --check` passed |
| H3, Qwen, VAE, MLX, Metal, and live render execution during implementation validation | Not run |

These receipts establish static and contract proof only. They are distinct
from the host acceptance recorded below.

## 8. Host acceptance configuration

The authoritative accepted storyboard configuration was:

| Run field | Recorded value |
|---|---|
| Storyboard cards | `3` |
| FL2V segments | `2` |
| Text conditioning | Live Canonical Qwen3-VL |
| Conditioning source | `live-encoder` |
| Width × height | `512 × 512` |
| Duration | `5` seconds per segment |
| Seed | `1701` |
| Scheduling adapter | LightX FL2V Turbo 4-step |
| Transformer | Streamed-AdaLN Q6 transformer |
| Transformer mode | `streamed-adaln-q6` |
| Additional LoRAs | None |
| External conditioning artifact replay | None |

The canonical paths are:

```text
Checkpoint root:
/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va

Transformer:
/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln
```

The old resident transformer directory is absent. The checkpoint transformer
symlink resolves to the streamed-AdaLN Q6 transformer, and Render Lab rejects
the old resident transformer name.

## 9. Host result and sequential-process evidence

The accepted host result was:

| Result field | Recorded value |
|---|---|
| Segments succeeded | `2 / 2` |
| Child exit codes | `0`, `0` |
| Transformer forwards per segment | `4` |
| Segment 1 total | approximately `124.14 s` |
| Segment 2 total | approximately `120.15 s` |
| Peak MLX per segment | `23,085,449,216` bytes |

Sequencing evidence recorded:

```text
segment 1 finalized:         approximately 14:33:25.478
segment 2 pre-run telemetry: approximately 14:33:25.544
difference:                  approximately 66 ms
```

The conservative interpretation is that child 1 exited and finalized before
child 2 began, with no segment overlap observed. This does not claim stronger
timing or memory isolation proof than the recorded evidence supports.

## 10. Limitations and explicit non-claims

This closeout does not claim:

- automatic final MP4 concatenation;
- a managed conditioning library;
- per-segment prompts;
- per-segment duration;
- per-segment render settings;
- a timeline editor;
- conditioning-artifact replay for storyboard FL2V;
- Heretic storyboard FL2V;
- that live output currently streams continuously;
- that full/resident transformer format has already been fail-closed as a
  general future safety slice;
- beta-0.6 conversion work;
- repair of the stale legacy browser launcher; or
- a new render performed during this documentation closeout.

The accepted host evidence is bounded to the recorded three-card, two-segment
configuration. It does not establish a general performance curve, visual
quality ranking, or universal adapter/conditioning behavior beyond the
accepted controls.

## 11. Deferred UX polish

Live process output currently appears to buffer during each child process and
dump at child completion rather than continuously streaming during the render.
This is accepted deferred UX/evidence polish and is not a 021F blocker.

Also deferred are explicit pending-segment records and richer live
per-segment progress. These items must not reopen the accepted 021F feature
scope.

## 12. Next-work boundary

The roadmap retains Slice 022 — LoRA Render Lab UX and Evidence Polish — as a
later development slice after 021F. Slice 023 — Heretic Token Reconciliation
remains a separate later workflow improvement. This closeout does not promote
either slice or invent a new ordering.

Beta-0.6 transformer conversion work and the future resident-transformer
fail-closed safety slice remain outside 021F. The stale legacy local browser
launcher (`Launch MiniMax H3.command` -> `scripts/minimax_h3_surface.py`) is
also deferred future runtime-safety debt. Neither legacy path was modified,
and that debt is not a 021F feature defect or a qualification against the
accepted Render Lab storyboard evidence.

## 13. Closeout boundary

- only `README.md`, `docs/ROADMAP.md`, and this closeout document changed;
- no production/runtime code changed;
- no tests changed;
- `scripts/minimax_h3_surface.py` was not modified;
- `Launch MiniMax H3.command` was not modified;
- no JSON or host artifact was modified;
- no H3, Qwen, VAE, MLX, Metal, or host render was run;
- `.DS_Store` was preserved untouched;
- nothing was staged, committed, or pushed; and
- the closeout stops after this documentation receipt.

# Slice 021E — Render Lab Conditioning Artifact Replay closeout

Closeout status: **Complete — host accepted (bounded).**

This document records the committed Slice 021E implementation, its offline
validation, the canonical reusable conditioning artifact, and the existing
host acceptance run. This is a documentation closeout only: no new H3, Qwen,
VAE, MLX, Metal, or render execution was performed while closing out the
slice.

## 1. Objective and verdict

Slice 021E added a manually selected conditioning-artifact replay path for
Canonical Qwen T2V in Render Lab. The bounded implementation validates the
artifact, prompt, and checkpoint provenance before launch, passes the
artifact to `generate.py`, skips live Qwen construction and prompt encoding
for a valid replay, and preserves artifact identity and tensor evidence in
the run record.

The host verdict is:

```text
SLICE_021E_HOST_ACCEPTANCE=PASS
```

The bounded acceptance proves that:

- the existing conditioning-artifact format can be selected manually in
  Render Lab;
- admission validates artifact/prompt/checkpoint provenance before launch;
- successful replay launches H3 without constructing Qwen;
- successful replay launches H3 without live prompt encoding;
- artifact identity and tensor evidence are preserved in run evidence; and
- blank-path behavior remains the live Canonical Qwen path by implementation
  contract.

## 2. Git and change boundary

| Field | Recorded value |
|---|---|
| Closeout worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021e` |
| Branch | `slice/021e-conditioning-artifact-replay` |
| HEAD | `6fcd7378b4dd2bccbfc960e5f4912d283c27ade6` |
| Expected parent | `e9fb546efc701bf303f24ada5f67a79be5d7ee69` |
| Implementation commit | `6fcd7378b4dd2bccbfc960e5f4912d283c27ade6` — Add Render Lab conditioning artifact replay |
| Pre-edit index | Clean |
| Pre-edit tracked state | Clean; no modified or deleted tracked paths |
| Pre-edit untracked state | No untracked files |
| Implementation diff stat | `405 insertions, 26 deletions` |

The implementation commit changed exactly:

- `tools/render_lab/runner.py`;
- `tools/render_lab/server.py`; and
- `tests/test_render_lab_conditioning_artifact_contract.py`.

This closeout changes only:

- `docs/ROADMAP.md`; and
- `docs/slice-021e-conditioning-artifact-replay-closeout.md`.

The intentionally dirty `experiment/h3-generation` worktree was not touched.

## 3. Implementation validation receipt

The implementation validation was MLX-free and completed before this
documentation closeout:

| Check | Result |
|---|---|
| Focused MLX-free contracts | `63/63` passed |
| Python compilation | `py_compile` passed |
| Embedded browser JavaScript syntax | `node --check` passed |
| Implementation whitespace check | `git diff --check` passed |
| H3, Qwen, VAE, MLX, Metal, and live render execution | Not run |

The implementation proof establishes the bounded Render Lab admission,
command propagation, replay-boundary, and evidence contracts. It does not
substitute for the host acceptance recorded below.

## 4. Canonical reusable artifact

The reusable artifact was created separately from the replay run at:

```text
/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/conditioning-library/artifacts/sword-fight-canonical-qwen.npz
```

| Artifact field | Recorded value |
|---|---|
| Encoder | Canonical Qwen3-VL |
| Selected hidden state | `hidden_states[50]` |
| Artifact identity | `82407294f3fe5c3da466b99bde9a26b08861e41b9c27c1d0a4eacd8c2a7b90e7` |
| Token count | `60` |
| Conditioning shape | `[1, 60, 5120]` |
| Logical dtype | `bfloat16` |
| Tensor checksum | `ec090baee3294eb601a136298ff696c320f97e623863ee63e5fc2bdd52454e11` |
| Artifact size | Approximately `482 KiB` |

Qwen was loaded exactly for artifact creation and then released. The allocator
purge after artifact creation succeeded. No H3 or VAE was loaded and no render
occurred during artifact creation.

The frozen prompt used for the artifact and replay was:

> A cinematic close-range sword fight between two skilled warriors in a
> rain-soaked industrial courtyard at night. Fast deliberate combat, visible
> footwork, parries, dodges, sparks from blade impacts, wet clothing, realistic
> body mechanics, dynamic camera movement, dramatic practical lighting,
> coherent anatomy and continuous motion.

## 5. Host acceptance configuration

The authoritative host acceptance run is:

```text
run-20260819T001659Z-dd0d7d3fcd
```

| Run field | Recorded value |
|---|---|
| Mode | `T2V` |
| Text encoder selection | Canonical Qwen3-VL |
| Conditioning source | `artifact-replay` |
| Checkpoint | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/checkpoints/minimax-h3-fl2va` |
| Transformer | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/models/minimax-h3-mlx-6bit-streamed-adaln` |
| Transformer mode | `streamed-adaln-q6` |
| Width × height | `512 × 512` |
| Duration | `5` seconds |
| Seed | `1701` |
| Scheduling owner | LightX 4-Step v0.1 |
| Configured steps / NFE | `4` / `4` |
| Additional auxiliary LoRAs | None |

The generated H3 child command contained:

```text
--conditioning-artifact
/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/conditioning-library/artifacts/sword-fight-canonical-qwen.npz
```

The child command did not use a positional live prompt as the conditioning
source.

## 6. Replay-boundary and evidence proof

The run-config evidence recorded:

| Evidence field | Recorded value |
|---|---|
| `conditioning_source` | `artifact-replay` |
| `conditioning_artifact.path` | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/conditioning-library/artifacts/sword-fight-canonical-qwen.npz` |
| `conditioning_artifact.artifact_identity` | `82407294f3fe5c3da466b99bde9a26b08861e41b9c27c1d0a4eacd8c2a7b90e7` |
| `conditioning_artifact.token_count` | `60` |
| `conditioning_artifact.conditioning_shape` | `[1, 60, 5120]` |
| `conditioning_artifact.tensor_checksum` | `ec090baee3294eb601a136298ff696c320f97e623863ee63e5fc2bdd52454e11` |
| `encoder_command` | `None` |

The child stdout proof recorded:

```text
conditioning source: artifact
artifact conditioning:
shape=(1, 60, 5120)
dtype=bfloat16
checksum=ec090baee3294eb601a136298ff696c320f97e623863ee63e5fc2bdd52454e11
```

The explicit replay boundary was:

```text
LIVE_QWEN_LOAD=NO
LIVE_PROMPT_ENCODING=NO
REPLAY_BOUNDARY=PASS
```

Together, `encoder_command = None`, the artifact child-command flag, and the
child stdout establish that this host replay reused the supplied conditioning
artifact rather than constructing Qwen or encoding the prompt live.

## 7. Host result and lifecycle evidence

| Result field | Recorded value |
|---|---|
| Status | `succeeded` |
| Exit code | `0` |
| Actual transformer forwards | `4` |
| Denoising | `64.6 s` |
| Seconds per forward | `16.175 s` |
| Total elapsed | `97.47983004200796 s` |
| Peak MLX memory | `22,441,204,121` bytes |
| Output bytes | `2,410,448` |
| Output SHA-256 | `58cb2afc5a2d2b67b6e205d7c10a55e59e8e3698eaa688e5b0562595aab09968` |
| Output | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021e/out/render-lab/run-20260819T001659Z-dd0d7d3fcd/render.mp4` |

Allocator lifecycle evidence records successful release and purge for the
transformer, video VAE, and audio VAE. The allocator cache was `0` after each
recorded purge.

## 8. Limitations and explicit non-claims

This acceptance is bounded to the recorded Canonical Qwen T2V artifact replay.
It does not claim:

- that artifact replay works for I2V or FIRST_LAST;
- that external replay works for Heretic;
- that prompt identity can be fuzzy matched;
- that a managed artifact library exists;
- that artifacts are automatically generated;
- that artifacts are automatically discovered;
- that prompt fields may be omitted;
- that replay improves H3 denoising performance;
- that the artifact changes visual quality; or
- that this host run proves all possible artifact/checkpoint combinations.

Exact artifact and prompt identity validation remains required. No fuzzy
matching or prompt-omission behavior was implemented.

## 9. UX follow-up note

When a managed artifact selector exists, selecting an artifact should ideally
surface or populate the artifact's authoritative prompt rather than requiring
the operator to manually reproduce the exact prompt text. This is a future
usability improvement only; it was not implemented in Slice 021E and must not
weaken exact artifact/prompt identity validation.

## 10. Next implementation slice

**Slice 021F — FL2V Storyboard Chain + Simple Card UI** is the next
implementation slice. Phase-one intent is:

- a new Storyboard mode in Render Lab;
- ordered image cards;
- Finder image selection using the existing image-selection pattern;
- drag/drop image input using the existing image-drop pattern;
- thumbnail previews and obvious sequential numbering;
- adjacent pair derivation: `1 -> 2`, `2 -> 3`, `3 -> 4`, and so on;
- shared generation settings across all segments;
- sequential FL2V rendering, one segment at a time;
- hard runtime unload/release between segments;
- per-segment outputs and evidence; and
- simple overall storyboard progress/status.

Phase-one exclusions are per-transition duration, per-transition prompts,
per-transition generation settings, a sophisticated timeline editor,
automatic final MP4 concatenation, a managed conditioning-artifact library,
artifact catalog/index or SQLite/database work, automatic artifact discovery,
and automatic artifact generation. Card reordering may be considered only if
it remains trivially bounded; it is not required for phase one.

Slice 021F was not started by this closeout.

## 11. Closeout boundary

- only documentation paths changed;
- no production code or tests changed;
- no JSON files were modified;
- no H3, Qwen, VAE, MLX, Metal, or render was run during this documentation
  closeout;
- nothing was staged, committed, pushed, reset, stashed, cleaned, reverted,
  amended, or moved; and
- the closeout stops after Slice 021E documentation and does not begin Slice
  021F implementation.

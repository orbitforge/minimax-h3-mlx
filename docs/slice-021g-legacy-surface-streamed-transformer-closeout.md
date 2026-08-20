# Slice 021G — Legacy Browser Surface Streamed-Transformer Safety closeout

This is a documentation-only closeout for the already committed Slice 021G
implementation. No production code or test was changed while preparing this
closeout.

## 1. Objective

Repair the stale transformer default used by the legacy browser surface so it
selects the canonical streamed-AdaLN Q6 transformer while preserving the
existing `scripts/generate.py` child-command propagation.

## 2. Starting state / implementation commit

The implementation was complete before this documentation closeout began. The
named worktree was clean at the expected implementation checkpoint.

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-021g` |
| Branch | `slice/021g-legacy-surface-streamed-transformer` |
| HEAD | `200ad36c377e909a20ffbacd128772c33587d155` |
| Implementation commit | `200ad36c377e909a20ffbacd128772c33587d155` — `Repair legacy surface transformer path` |
| Starting worktree state | Clean |

The implementation commit changed exactly:

- `scripts/minimax_h3_surface.py`; and
- `tests/test_legacy_surface_transformer_contract.py`.

## 3. Exact production change

In `scripts/minimax_h3_surface.py`, the legacy browser surface transformer
basename changed from:

```text
minimax-h3-mlx-6bit
```

to:

```text
minimax-h3-mlx-6bit-streamed-adaln
```

The legacy browser surface still invokes `scripts/generate.py` with:

```text
--transformer <TRANSFORMER>
```

The implementation contract proves that the canonical transformer path is
propagated to that child command.

## 4. Focused contract

The focused Slice 021G legacy-surface contract passed:

```text
3/3 PASS
```

It covers the streamed-AdaLN default, rejection of the stale resident basename,
and propagation of the transformer path into the generated child command.

## 5. Validation evidence

The established implementation evidence is:

| Check | Result |
|---|---|
| Focused 021G contract | `3/3 PASS` |
| Render Lab encoder safety | `10/10 PASS` |
| Render Lab Turbo safety | `5/5 PASS` |
| LightX production-entry contract | `7/7 PASS` |
| `py_compile` for both changed Python files | Passed |
| Implementation `git diff --check` | Passed |
| MLX, Metal, or H3 render | Not run |

This is offline/static and contract evidence. No host render was required for
Slice 021G.

## 6. Research/runtime boundary

021G fixed only the stale legacy browser surface transformer default. It did
not alter:

- `scripts/generate.py`;
- `minimax_h3_mlx/pipeline.py`;
- `minimax_h3_mlx/load.py`;
- transformer routing;
- Render Lab;
- probes or research workflows; or
- `Launch MiniMax H3.command`.

Original/resident checkpoint loading remains intentionally available for
explicit research and probe workflows.

## 7. General resident-transformer fail-closed work

General resident-transformer fail-closed safety work was **NOT implemented** by
Slice 021G. The slice did not globally prohibit original/resident transformers
and does not claim that broader safety boundary.

## 8. Beta-0.6 conversion

Beta-0.6 conversion was **NOT started** by Slice 021G. It remains separate
future work.

## 9. Changed-path boundary

The documentation closeout changes exactly these two authorized paths:

- `docs/ROADMAP.md`; and
- `docs/slice-021g-legacy-surface-streamed-transformer-closeout.md`.

It does not modify `README.md`,
`docs/slice-021f-fl2v-storyboard-closeout.md`, any production/runtime file,
any test, any JSON, or any other documentation.

The historical 021F closeout remains untouched. Its statement about the
legacy browser debt is preserved historically in the living roadmap as the
state that existed when 021F closed; the roadmap now records that 021G
subsequently repaired the default.

## 10. Next-work boundary

The broader resident-transformer fail-closed boundary remains future work.
Beta-0.6 conversion remains future work. Slice 022 and Slice 023 are not
promoted or reordered by this closeout.

## 11. Git/publication state

| Field | Recorded value |
|---|---|
| Local implementation commit | `200ad36c377e909a20ffbacd128772c33587d155` |
| Documentation closeout commit | Not yet committed |
| Staged paths | None |
| 021G pushed | No; nothing from 021G has been pushed |
| `main` promotion | `main` has not yet been promoted to 021G |
| Publication claims | None may be made yet |

This closeout stops at the prepared, locally validated documentation state.

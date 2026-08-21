# Slice 028 — Runtime Assets Environment Isolation closeout

This is a documentation-only closeout for the already committed Slice 028
implementation. No production code, tests, runtime-selection implementation,
Render Surface production code, model or checkpoint artifacts, or runtime-assets
profile links were changed while preparing this closeout. Generation,
Current-vs-Beta comparison, Qwen, VAE, MLX/Metal execution, and media
acceptance were not run. No Git staging, commit, push, reset, stash, clean,
revert, or repair was performed.

## 1. Closeout verdict and scope

```text
SLICE_028_CLOSEOUT_READY
```

Slice 028 closes a bounded environment-isolation repair discovered during
post-Slice 027 legacy Render Surface host acceptance. Ambient
`MINIMAX_H3_RUNTIME_ASSETS` configuration must not turn an existing
Current/manual launch into an explicit runtime-assets request. Named Beta
selection remains explicit and opt-in.

The implementation commit is:

```text
fc45f2874f225995dbaf238823ecfcac354ce437 Isolate runtime assets environment
```

The implementation is committed. This closeout is local documentation state
only and remains uncommitted and unstaged.

## 2. Repository and Git boundary

The required boundary was established before editing:

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-028` |
| Branch | `slice/028-runtime-assets-env-isolation` |
| HEAD | `fc45f2874f225995dbaf238823ecfcac354ce437` |
| HEAD subject | `Isolate runtime assets environment` |
| Starting status | Clean |
| Starting staged paths | None |
| Starting modified tracked paths | None |
| Starting untracked paths | None |

The implementation commit changed exactly:

- `scripts/generate.py`; and
- `tests/test_runtime_selection.py`.

The documentation closeout changes exactly:

- `docs/ROADMAP.md`; and
- `docs/slice-028-runtime-assets-env-isolation-closeout.md`.

README, historical closeouts, Render Surface production code,
`minimax_h3_mlx/runtime_selection.py`, model/checkpoint artifacts,
runtime-assets profile links, and all other production and test paths remain
outside this documentation closeout.

## 3. Recorded post-Slice 027 host-acceptance failure

After Slice 027 was published, the legacy Render Surface was launched with:

```text
MINIMAX_H3_RUNTIME_ASSETS=<valid beta profile root>
```

Selecting `Current` produced the existing manual child command with:

```text
--checkpoint <manual checkpoint>
--transformer <manual transformer>
```

and no `--runtime` option. Nevertheless, `generate.py` exited `2` with:

```text
--runtime-assets requires --runtime
```

This blocked the intended Current-vs-Beta Surface comparison. The failure was
observed at the CLI admission boundary; it was not a model, Qwen, VAE, MLX,
Metal, or media result.

## 4. Recorded root cause

At the Slice 027 baseline, `argparse` configured `--runtime-assets` with an
environment-derived default from `MINIMAX_H3_RUNTIME_ASSETS`. Ambient host
configuration therefore became indistinguishable from explicit CLI intent:

```text
environment
→ argparse default
→ args.runtime_assets
→ no --runtime
→ invalid explicit-runtime-assets guard
→ exit 2
```

The failure occurred before named-runtime resolution and before pipeline or
model loading.

## 5. Recorded Slice 028 correction

Slice 028 changes the boundary as follows:

- `--runtime-assets` now defaults to `None`;
- `args.runtime_assets` therefore represents explicit CLI input only;
- `MINIMAX_H3_RUNTIME_ASSETS` is consulted only when a named runtime has
  actually been selected;
- Slice 026 `resolve_runtime()` admission remains unchanged; and
- Render Surface production code was not modified.

This preserves the existing named-runtime admission owner while preventing
ambient environment state from altering the Current/manual path.

## 6. Final CLI and environment contract

| Invocation | Result |
|---|---|
| Manual / Current with ambient `MINIMAX_H3_RUNTIME_ASSETS` and no `--runtime` | Environment is ignored; existing manual checkpoint/transformer behavior is preserved. |
| `--runtime-assets X` without `--runtime` | Still rejected with exit `2`. |
| `--runtime beta-0.6` without explicit `--runtime-assets` | Consumes `MINIMAX_H3_RUNTIME_ASSETS`. |
| `--runtime beta-0.6` with `--runtime-assets /cli/root` and ambient `/env/root` | Explicit CLI root wins. |

The environment alone never selects Beta. No fallback was introduced.

The intended examples are:

```text
ambient MINIMAX_H3_RUNTIME_ASSETS
+ no --runtime
→ environment ignored
→ existing manual checkpoint/transformer behavior preserved
```

```text
--runtime beta-0.6
+ no explicit --runtime-assets
→ consume MINIMAX_H3_RUNTIME_ASSETS
```

## 7. Recorded regression proof

The independent model-free Surface-shaped regression result is:

```text
CURRENT_ENV_ISOLATION_PASS
```

It used:

- ambient runtime-assets;
- a manual Current checkpoint;
- a manual Current transformer;
- `--megapixels 0.2`; and
- `--duration 5`.

The regression proved:

- parsing succeeds;
- named-runtime resolution is not invoked;
- checkpoint and transformer remain manual;
- the fake pipeline seam is reached;
- the previous exit-2 failure is gone; and
- no media or model payload execution occurs.

The explicit-invalid-CLI, named-Beta environment-consumption, and CLI
precedence cases are included in the new `4/4` environment-regression result.

## 8. Recorded implementation and review history

The implementation verdict was:

```text
READY_FOR_SLICE_028_REVIEW
```

The independent review result was:

```text
PASS_WITH_GAPS
```

The review record is preserved below:

| Review area | Result |
|---|---|
| Root cause | `PASS` |
| Parser/CLI-environment separation | `PASS` |
| Current isolation | `CURRENT_ENV_ISOLATION_PASS` |
| Explicit invalid CLI | `PASS` |
| Beta environment consumption | `PASS` |
| CLI precedence | `PASS` |
| Environment-only default behavior | `PASS` |
| No fallback / Slice 026 ownership | `PASS` |
| Surface-shaped regression | `PASS` |
| Diff/scope | `PASS` |

The gaps are host-generation and media-acceptance gaps, not code blockers.

## 9. Recorded validation

The final accepted MLX-free validation is:

| Check | Result |
|---|---|
| Slice 026 runtime-selection | `23/23 PASS` |
| New Slice 028 environment regressions | `4/4 PASS` |
| Slice 027 Surface | `10/10 PASS` |
| Combined focused validation | `37/37 PASS` |
| `generate.py --help` | `PASS` |
| `py_compile` | `PASS` |
| `git diff --check` | `PASS` |
| Independent review | `PASS_WITH_GAPS` |

These results establish the CLI/environment contract and fake-pipeline seam.
They do not establish Beta generation, Current-vs-Beta media comparison,
Qwen/VAE execution, MLX/Metal execution, or human media acceptance.

## 10. Explicit scope boundary

The implementation paths were limited to:

- `scripts/generate.py`; and
- `tests/test_runtime_selection.py`.

No changes were made to:

- Render Surface production code;
- `minimax_h3_mlx/runtime_selection.py`;
- model or checkpoint artifacts; or
- runtime-assets profile links.

The correction preserves Slice 026 runtime-selection admission and does not
introduce an environment-only Beta default or any fallback behavior.

## 11. Unproven host work

The following remain **NOT YET PROVEN**:

- a successful legacy Render Surface Current launch under the corrected
  environment contract on the host;
- a successful legacy Render Surface Beta launch through named `beta-0.6`
  admission;
- a Current-vs-Beta Surface comparison;
- Qwen, VAE, MLX, or Metal execution through the UI path;
- final media generation and acceptance; and
- human visual acceptance.

No generation or comparison was run for this closeout. The model-free
`CURRENT_ENV_ISOLATION_PASS` result must not be interpreted as host-generation
or media proof.

## 12. Documentation validation and publication boundary

After the documentation edits:

- the complete documentation diff was inspected;
- `git diff --check` passed;
- the changed-path allowlist contains only `docs/ROADMAP.md` and this
  closeout;
- no production, test, runtime-selection, Render Surface, model-artifact, or
  runtime-assets path changed;
- no runtime-assets profile links were created or modified; and
- no staging, commit, or push was performed.

The final local publication state is intentionally:

```text
DOCS_ONLY=PASS
STAGED=NONE
COMMIT=NOT_PERFORMED
PUSH=NOT_PERFORMED
PRODUCTION_CHANGES=NONE
TEST_CHANGES=NONE
RUNTIME_SELECTION_CHANGES=NONE
RENDER_SURFACE_CHANGES=NONE
MODEL_ARTIFACT_CHANGES=NONE
RUNTIME_ASSETS_LINKS=NOT_MODIFIED
GENERATION=NOT_RUN
QWEN_VAE_MLX_METAL=NOT_RUN
CURRENT_VS_BETA_COMPARISON=NOT_RUN
```

## 13. Final Git state and next action

The documentation-only closeout leaves these local changes for review:

- modified: `docs/ROADMAP.md`;
- untracked: `docs/slice-028-runtime-assets-env-isolation-closeout.md`;
- staged paths: none; and
- commit/push: not performed.

The final next-action verdict is:

```text
READY_FOR_SLICE_028_CLOSEOUT_COMMIT
```

This closeout stops at validated, uncommitted documentation. Any host
Current-vs-Beta acceptance remains a separately authorized operation.

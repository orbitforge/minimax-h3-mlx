# Slice 025 — Beta QKV Layout Reconciliation closeout

This is a documentation-only closeout for the already committed Slice 025
implementation. No production code, tests, or the QKV authorization receipt
were changed while preparing this closeout. Conversion, forge, generation,
MLX/Metal execution, and corrected beta acceptance were not run. No Git
staging, commit, push, reset, stash, clean, or revert was performed.

## 1. Closeout verdict and scope

```text
SLICE_025_CLOSEOUT_READY
```

Slice 025 reconciles the accepted beta fused-QKV source row order with the
runtime-native MLX row order before the existing Q6 quantization path. It does
not change runtime attention semantics, the existing quantization policy, or
the streamed-AdaLN format.

The implementation commit is:

```text
26869fb Reconcile beta QKV row layout
```

The implementation is committed. This closeout is local documentation state
only and remains uncommitted and unstaged.

## 2. Repository and Git boundary

The required boundary was established before editing:

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-025` |
| Branch | `slice/025-beta-qkv-layout-reconciliation` |
| HEAD | `26869fb0eb539c8df767a9e1e8e94162f94b991d` |
| Starting status | Clean |
| Starting staged paths | None |
| Starting modified tracked paths | None |
| Starting untracked paths | None |

The implementation commit changed the QKV admission/conversion implementation,
the CLI receipt surface, the tests, and added
`minimax_h3_mlx/qkv_layout_authorization.json`. None of those implementation,
test, or evidence-artifact paths is part of this documentation closeout.

The closeout changes only:

- `docs/ROADMAP.md`; and
- `docs/slice-025-beta-qkv-layout-reconciliation-closeout.md`.

The checked-in authorization artifact remains unchanged and is referenced
read-only as evidence.

## 3. Problem Slice 025 actually solved

The accepted beta source is:

`/Users/elbancol/Downloads/PinkCherry-beta-redownload/beta-0.6-fl2va/PinkCherry_fl2va_MiniMax_H3_bf16_beta-0.6.safetensors`

| Source identity field | Recorded value |
|---|---:|
| Bytes | `66,288,818,760` |
| SHA-256 | `16f1950cc83bd686106d49588c8611281fbb5e9ae46f8cd1ae7945fd4e00357d` |

The preceding Slice 024 conversion and streamed-AdaLN forge were structurally
successful. Slice 024 itself is not characterized as failed: its conversion
and runtime structural contracts passed.

The subsequent real beta runtime acceptance operation established that:

- live Canonical Qwen3-VL conditioning worked;
- the streamed beta transformer loaded;
- `15` denoising forwards completed;
- video VAE worked;
- audio VAE worked;
- a structurally valid MP4 was produced; and
- memory reclamation worked.

Human media acceptance nevertheless failed at both `128×128` and `512×512`.
The output was muddy and low-contrast, with no recognizable prompt semantics.
That semantic failure exposed a previously unknown source-layout contract; it
was not evidence of general Q6 degradation.

The previous generated beta media remains evidence of the **pre-repair**
converter. It must not be reclassified as corrected output.

## 4. Forensic root cause and exhaustive QKV evidence

The accepted PinkCherry beta serialization is grouped:

```text
[Q_all; K_all; V_all]
```

The runtime-native MLX ordering is per-head interleaved:

```text
[head0:q,k,v][head1:q,k,v]...
```

The original Slice 024 converter preserved the beta rows without
canonicalization. The runtime therefore interpreted grouped Q/K/V rows as if
they were per-head-interleaved. The resulting transformer execution was
numerically finite and structurally valid, but its attention semantics were
invalid.

The complete admitted fused-QKV surface is:

| Surface | Count |
|---|---:|
| Main-block fused-QKV weights | `50` |
| Token-refiner fused-QKV weights | `2` |
| Total fused-QKV weights | `52` |
| Fused-QKV biases | `0` |

The independently reproduced payload comparison covered all `52/52` weights:

All `52` tensors independently support the same grouped-to-runtime
orientation. Quantization forensic checks found ordinary reconstruction error;
the consistent orientation signal is not general Q6 degradation.

| Orientation | Relative L2 range | Relative L2 mean | Cosine range |
|---|---:|---:|---:|
| Direct beta versus runtime-native reference | `1.400078–1.410067` | `1.405500` | `0.006202–0.020231` |
| Grouped to runtime reconciliation | `0.022318–0.026479` | `0.023695` | `0.999650–0.999751` |

The corrected-orientation worst case was:

```text
blocks.2.attn.qkv_proj.weight
```

The direct-orientation worst case was:

```text
blocks.9.attn.qkv_proj.weight
```

The comparison used the reference logical identity:

```text
beff3b611b6b7597c30d07dd050c50ea393d785c97b3e0417309ead226719abe
```

The authorization evidence artifact is:

```text
minimax_h3_mlx/qkv_layout_authorization.json
```

Its admitted schema is `slice025.qkv_layout_authorization.v1`; the artifact
was not modified by this closeout.

## 5. Implemented admission and canonicalization contract

### Admission

- The exact accepted beta source is authorized as `grouped_qkv`.
- Authorization is bound to the admitted source identity and payload evidence;
  metadata or shape alone cannot authorize a layout.
- Unknown, ambiguous, or contradictory source layouts fail closed.
- Explicit `runtime_interleaved` sources remain a no-op.

### Canonicalization

For an admitted grouped source, reconciliation is:

```text
grouped source: [Q_all;K_all;V_all]
reshape:        (3, heads, head_dim, input_features)
transpose:      (heads, 3, head_dim, input_features)
flatten:        runtime-native rows
```

The reconciliation occurs before the existing Q6 quantization operation sees
the weight. Runtime attention semantics remain unchanged. The quantization
policy remains:

```text
Q6 core
Q8 block-AdaLN
group size 64
```

No streamed-AdaLN format change occurred.

## 6. Receipt semantics

The committed receipt contract separates planning, authorization, and actual
execution:

| Field | Meaning |
|---|---|
| `qkv_tensors_reconciled` | Number of successfully executed QKV reconciliations |
| `qkv_row_reconciliation_applied` | `true` iff the actual reconciliation count is greater than zero |
| `qkv_tensors_planned` | Planning/dry-run information only |

Authorization and execution are separate facts. The superseded first
implementation reported planned reconciliation as if it were executed;
independent review caught that distinction and the committed repair reports
execution only after each transformation completes.

## 7. Review history

The review sequence is preserved because it changed the closeout readiness:

1. Initial implementation: `SLICE_025_IMPLEMENTED_AND_VALIDATED`.
2. First independent review: `NOT READY`.
   - `P1`: grouped-beta layout was not sufficiently independently and
     payload-bound authorized.
   - `P2`: receipt reconciliation counts described planned rather than actual
     execution.
3. Repair: `SLICE_025_REPAIRED_AND_VALIDATED`.
4. Second independent review: `PASS_WITH_GAPS`.
   - P1 resolved.
   - P2 resolved.
   - Reference-converter conflict resolved.
   - Exhaustive `52/52` proof reproduced independently.

The second review recorded these nonblocking gaps:

- the authorization loader does not recompute the full source SHA during every
  admission;
- comparison-reference receipt fields are not exhaustively runtime-validated;
  and
- exact absolute-path binding is operationally brittle/redundant.

These are recorded review limitations, not current closeout blockers.

## 8. Final accepted implementation validation

The final accepted validation before implementation commit recorded:

| Check | Result |
|---|---|
| Exhaustive QKV payload proof | `52/52` |
| Monolithic suite | `55/55 PASS` |
| Checkpoint-forge suite | `32 PASS`, `2 MLX-gated skips` |
| Exact beta SHA | Verified |
| `py_compile` | `PASS` |
| `git diff --check` | `PASS` |
| Independent second review | `PASS_WITH_GAPS` |
| Original P1 | Resolved |
| Original P2 | Resolved |

These are implementation validation receipts. The documentation closeout did
not rerun conversion, forge, generation, MLX, Metal, or the test suites.

## 9. Explicitly unproven work

Slice 025 does **not** prove:

- corrected full beta BF16 to conventional Q6/Q8 conversion;
- corrected conventional Q6/Q8 checkpoint verification;
- corrected streamed-AdaLN forge;
- corrected beta MLX/Metal generation;
- corrected semantic output; or
- human visual/audio acceptance.

The pre-repair generated beta media remains pre-repair evidence and is not
corrected-output evidence.

## 10. Next host operation

After Slice 025 promotion, the next host operation is:

1. Full corrected beta BF16 to conventional Q6/Q8 conversion using the
   committed Slice 025 logic.
2. Independent conventional verification.
3. Streamed-AdaLN forge.
4. Independent streamed-checkpoint verification.
5. The same controlled `512×512` beta acceptance render with:
   - prompt: `a red fox leaps over a mossy log`;
   - seed: `0`;
   - duration: `5`;
   - steps: `16`;
   - geometry: `512×512`;
   - live Canonical Qwen3-VL; and
   - no LoRAs.
6. Human comparison against the known pre-repair semantic failure.

None of these operations was performed in this documentation closeout.

## 11. Documentation validation and publication boundary

The documentation-only validation boundary is:

- no JSON continuity file was changed;
- the existing `minimax_h3_mlx/qkv_layout_authorization.json` remained
  unchanged and was read-only evidence;
- the complete documentation diff was inspected;
- `git diff --check` passed; and
- the changed-path allowlist contains only `docs/ROADMAP.md` and this closeout.

The final local publication state is intentionally:

```text
DOCS_ONLY=PASS
STAGED=NONE
COMMIT=NOT_PERFORMED
PUSH=NOT_PERFORMED
PRODUCTION_OR_TEST_CHANGES=NONE
EVIDENCE_ARTIFACT_CHANGES=NONE
```

This closeout stops at validated, uncommitted documentation. The next action
verdict is:

```text
READY_FOR_SLICE_025_CLOSEOUT_COMMIT
```

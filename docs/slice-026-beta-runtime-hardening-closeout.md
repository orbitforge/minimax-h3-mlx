# Slice 026 — Beta Runtime Hardening closeout

This is a documentation-only closeout for the already committed Slice 026
implementation. No production code, tests, runtime-selection implementation,
model artifacts, or README content was changed while preparing this closeout.
Generation, conversion, forge, MLX/Metal execution, and Render Surface
integration were not run. No Git staging, commit, push, reset, stash, clean,
revert, or repair was performed.

## 1. Closeout verdict and scope

```text
SLICE_026_CLOSEOUT_READY
```

Before Slice 026, accepted beta generation required callers to compose the
canonical surrounding checkpoint root, an explicit corrected streamed
transformer override, knowledge of which beta artifact was accepted, and
implicit knowledge of Qwen/VAE/scheduler compatibility. Slice 026 makes the
accepted beta runtime a named, explicit, fail-closed runtime selection:

```text
beta-0.6
```

The named runtime remains explicit and opt-in. It did not become the default.
The implementation commit is:

```text
685c6f2 Harden beta runtime selection
```

The implementation is committed. This closeout is local documentation state
only and remains uncommitted and unstaged.

## 2. Repository and Git boundary

The required boundary was established before editing:

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-026` |
| Branch | `slice/026-beta-runtime-hardening` |
| HEAD | `685c6f28edf59941495c982c2173265b1f193e26` |
| Starting status | Clean |
| Starting staged paths | None |
| Starting modified tracked paths | None |
| Starting untracked paths | None |

The implementation commit changed exactly:

- `README.md`;
- `minimax_h3_mlx/runtime_selection.py`;
- `scripts/generate.py`; and
- `tests/test_runtime_selection.py`.

The README change belongs to the implementation commit and was not repeated in
this closeout. The documentation closeout changes exactly:

- `docs/ROADMAP.md`; and
- `docs/slice-026-beta-runtime-hardening-closeout.md`.

No production, test, runtime-selection, model-artifact, or continuity-JSON
path is part of the documentation closeout.

## 3. Problem Slice 026 solved

The prior caller contract required manual composition of:

- the canonical surrounding checkpoint root;
- an explicit corrected streamed transformer override;
- the accepted beta artifact identity; and
- compatible Qwen, VAE, and scheduler choices.

That composition was operationally fragile even after Slice 025 corrected the
QKV conversion defect. Slice 026 names the accepted complete combination as
`beta-0.6` and resolves it fail-closed before heavy runtime loading.

The named runtime is explicit and opt-in; no documentation here characterizes
`beta-0.6` as a default.

## 4. Named-runtime contract

`beta-0.6` represents the complete accepted runtime combination:

- surrounding checkpoint root;
- Canonical Qwen3-VL;
- tokenizer;
- processor;
- video VAE;
- audio VAE;
- scheduler/config contract;
- corrected conventional provenance;
- corrected streamed-AdaLN transformer;
- transformer `config.json`;
- Slice 025 QKV reconciliation evidence;
- Q6/Q8 quantization contract; and
- streamed topology.

Legacy manual `--checkpoint` and `--transformer` workflows remain
available when no named runtime is selected. Named beta selection rejects
conflicting manual checkpoint or transformer overrides.

The resolved-runtime receipt is a statement of validated and resolved facts,
not merely a record of caller intent.

## 5. Asset-location contract

The named runtime accepts:

```text
--runtime-assets <runtime-assets>
MINIMAX_H3_RUNTIME_ASSETS
```

When both are supplied, the CLI option takes precedence over the environment
variable. The canonical host-local profile layout is:

```text
<runtime-assets>/beta-0.6/
  checkpoint    -> <accepted surrounding checkpoint>
  transformer   -> <accepted streamed beta transformer>
  conventional  -> <accepted conventional transformer metadata>
```

The three profile entries resolve to the accepted assets and are required to
be symbolic links for the current Slice 026 host deployment contract. This
symlink-only rule is an operational deployment contract for this slice, not an
eternal model-format requirement.

The portable runtime API is the profile shape and selector contract above; it
does not embed a user's workstation paths.

## 6. Pre-load admission boundary

Named-runtime loading follows this boundary:

```text
args
→ resolve runtime/profile assets
→ validate streamed provenance
→ validate transformer config
→ validate Qwen/VAE/tokenizer/processor/scheduler metadata
→ emit resolved-runtime receipt
→ lazy import pipeline
→ construct/load models
```

Invalid named profiles fail before heavy payload or model loading. The final
independent review verified invalid-config rejection with CLI exit before
pipeline import, zero safetensors payload reads, and no Qwen, VAE, MLX, Metal,
or model construction.

## 7. Transformer-config admission

The accepted real transformer configuration is:

| Field | Recorded value |
|---|---:|
| Bytes | `563` |
| SHA-256 | `bd97e5da656ee83da7cf4d83146a19c06521ec8b455bd62775521c27bdb08ebf` |

Its normalized accepted architecture is:

| Field | Value |
|---|---:|
| Layers | `50` |
| Hidden size | `5376` |
| Heads | `56` |
| Head dimension | `128` |
| FFN | `14336` |
| Video latent channels | `24` |
| Audio latent channels | `32` |
| Patch | `[1,2,2]` |
| Text width | `5120` |
| AdaLN | `96768` |
| Final AdaLN | `10752` |
| RoPE length | `16` |
| RoPE theta | `10000.0` |
| Inner dimension | `7168` |
| Video output | `96` |
| Audio output | `32` |
| Rotary dimension | `96` |

`rope_theta` is absent from the accepted file and normalizes to the
production default `10000.0`. Admission uses semantic architecture
validation and reports the config SHA; the full exact-JSON hash is not treated
as the sole semantic identity.

## 8. Tokenizer and processor admission

Slice 026 validates the tokenizer's:

- model-index mapping;
- `Qwen2TokenizerFast` identity;
- tokenizer configuration;
- required files;
- special-token IDs; and
- EOS/pad behavior.

It validates the processor's:

- model-index mapping;
- `Qwen3VLProcessor` identity;
- image/video processor configurations;
- processor classes and types;
- geometry-related metadata; and
- chat-template presence and metadata.

Large vocabulary and other payload files are presence-checked rather than
recursively hashed. This closeout does not claim stronger payload verification
than the implementation provides.

## 9. Streamed and provenance contract

Named beta admission validates the accepted Slice 025 artifact semantics.

### Source identity

| Field | Recorded value |
|---|---:|
| SHA-256 | `16f1950cc83bd686106d49588c8611281fbb5e9ae46f8cd1ae7945fd4e00357d` |
| Bytes | `66,288,818,760` |
| Topology | `535 tensors / 522 BF16 / 13 F32` |

### QKV and quantization

| Field | Recorded value |
|---|---|
| Source layout | `grouped_qkv` |
| Canonical layout | `runtime_interleaved` |
| Reconciled | `52` |
| Core quantization | Q6 |
| Block-AdaLN quantization | Q8 |
| Group size | `64` |

### Streamed topology

| Field | Recorded value |
|---|---:|
| Format | `minimax-h3-mlx-streamed-adaln-v1` |
| Logical tensors | `1,050` |
| Resident tensors | `850` |
| Resident block-AdaLN tensors | `0` |
| Sidecars | `50` |
| Sidecar tensors | `200` |
| Loader requirement | Existing five-base-shard loader format |

Conventional metadata remains required as provenance hardening. Named-runtime
admission does not load conventional tensor payloads.

## 10. Resolved-runtime receipt

The CLI reports validated and resolved facts including:

- runtime identifier;
- resolved checkpoint path;
- resolved transformer path;
- conventional provenance path;
- transformer config path, bytes, and SHA;
- normalized transformer architecture;
- source identity;
- QKV receipt, layout, and count;
- quantization contract;
- streamed topology;
- tokenizer and processor identity;
- Qwen and VAE identity;
- scheduler identity and shifts; and
- override state.

The receipt is resolved runtime fact, not merely caller intent.

## 11. Review history

The review sequence is preserved:

1. Initial implementation: `SLICE_026_IMPLEMENTED_AND_VALIDATED`.
2. First independent review: `FAIL`.
   - Named runtime admission did not validate the streamed transformer's
     actual `config.json`, even though later pipeline construction consumed
     that file.
   - A significant related gap was weaker tokenizer/processor identity
     validation than the runtime contract claimed.
3. Repair: `SLICE_026_REPAIRED_AND_VALIDATED`.
4. Second independent review: `PASS_WITH_GAPS`.
   - Original blocker: `RESOLVED`.
   - Transformer config: `PASS`.
   - Tokenizer/processor: `SUFFICIENTLY_ADMITTED_FOR_SLICE_026`.
   - Resolved receipt: `PASS`.
   - Pre-load/lazy import: `PASS`.

The first implementation is not characterized as accepted; the repaired
implementation and second-review result are the closeout evidence.

## 12. Final accepted implementation validation

| Check | Result |
|---|---|
| Runtime-selection suite | `23/23 PASS` |
| Real-host positive metadata probe | `ACCEPTED_METADATA_ONLY` |
| Real-host negative config probe | `REJECTED_METADATA_ONLY` |
| Invalid config boundary | Rejected before pipeline import |
| Safetensors payload bytes read during admission probes | `0` |
| Legacy surface | `3/3 PASS` |
| Conditioning contracts | `6/6 PASS` |
| LightX metadata contracts | `PASS` |
| `generate.py --help` | `PASS` |
| `py_compile` | `PASS` |
| `git diff --check` | `PASS` |
| Second independent review | `PASS_WITH_GAPS` |

No model-backed generation was part of Slice 026 implementation acceptance.
The validation above is contract, metadata, and pre-load admission evidence.

## 13. Nonblocking gaps

The following are recorded as nonblocking:

- repeated CLI selectors use argparse last-wins behavior;
- symlink-only profile deployment is operationally brittle;
- mutable inner-link TOCTOU remains under the normal local-file threat model;
- deep streamed payload integrity remains delegated to the existing
  loader/verifier; and
- tokenizer/vocabulary payload files are presence-checked rather than fully
  hashed.

These are not current Slice 026 blockers.

## 14. Explicitly unproven work

Slice 026 does not yet prove:

- full MLX generation invoked through `--runtime beta-0.6`;
- Render Surface integration;
- Render Surface → named-runtime → corrected beta generation; or
- Q6-versus-Q8 beta comparison.

The earlier manual-transformer Slice 025 host run is not proof of the new
named-runtime CLI path and is not repeated here as such.

## 15. Next engineering slice

The next planned engineering slice is **Render Surface integration**. Its
conceptual path is:

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
named-runtime render may be used as part of post-integration host acceptance.
No Render Surface change is implemented in this closeout.

## 16. Documentation validation and publication boundary

After the documentation edits:

- no continuity JSON file was changed, so no JSON continuity validation was
  required;
- the complete documentation diff was inspected;
- `git diff --check` passed;
- the changed-path allowlist contains only `docs/ROADMAP.md` and this
  closeout;
- no production, test, runtime-selection, or model-artifact path changed; and
- no checkpoint or other model artifact changed.

The final local publication state is intentionally:

```text
DOCS_ONLY=PASS
STAGED=NONE
COMMIT=NOT_PERFORMED
PUSH=NOT_PERFORMED
PRODUCTION_OR_TEST_CHANGES=NONE
RUNTIME_SELECTION_CHANGES=NONE
MODEL_ARTIFACT_CHANGES=NONE
```

This closeout stops at validated, uncommitted documentation.

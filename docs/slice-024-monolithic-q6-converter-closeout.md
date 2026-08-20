# Slice 024 — Monolithic BF16-to-MLX Q6/Q8 Converter closeout

This is a documentation-only closeout for the already committed Slice 024
implementation and its bounded real-host proof. No production code or test
was changed while preparing this closeout. The complete beta conversion was
not run, H3 generation was not run, and no Git staging, commit, push, reset,
stash, clean, or revert was performed.

## 1. Objective

Slice 024 implemented and proved a bounded-memory converter for the monolithic
MiniMax H3 beta-0.6 BF16 safetensors source into the repository's conventional
MLX quantized checkpoint format.

The full beta conversion was explicitly not part of the implementation slice.
It remains a later host operation using the completed converter.

## 2. Starting state and implementation checkpoint

| Field | Recorded value |
|---|---|
| Worktree | `/Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx-024` |
| Branch | `slice/024-beta-monolithic-q6-converter` |
| Base | `96b8b7f5b48ad21f4de80666a89fb9d95debec30` |
| Implementation commit | `d6823ea75d1758f3286817579f935343d4dae4da` — `Add bounded monolithic H3 quantization` |
| Starting worktree state | Clean at the implementation commit |
| Documentation closeout state | Local, uncommitted, and unstaged |

The implementation commit changed exactly these six paths:

- `minimax_h3_mlx/checkpoint_forge/tensor_io.py`;
- `minimax_h3_mlx/quantize.py`;
- `minimax_h3_mlx/monolithic_quant.py`;
- `minimax_h3_mlx/monolithic_source.py`;
- `scripts/build_monolithic_quant.py`; and
- `tests/test_monolithic_quant.py`.

The documentation closeout changes exactly these two paths:

- `docs/ROADMAP.md`; and
- `docs/slice-024-monolithic-q6-converter-closeout.md`.

No production or test path is part of the documentation closeout.

## 3. Authoritative source identity and admission

The authoritative source is the verified re-downloaded beta file:

`/Users/elbancol/Downloads/PinkCherry-beta-redownload/beta-0.6-fl2va/PinkCherry_fl2va_MiniMax_H3_bf16_beta-0.6.safetensors`

| Check | Verified value |
|---|---:|
| File bytes | `66288818760` |
| SHA-256 | `16f1950cc83bd686106d49588c8611281fbb5e9ae46f8cd1ae7945fd4e00357d` |
| Header bytes | `8388608` |
| Data start | `8388616` |
| Header tensor count | `535` |
| Dtype mix | `522 BF16 + 13 F32` |
| Declared complete bytes | `66288818760` |
| Missing bytes | `0` |
| Tensors past EOF | `0` |

The earlier Safari download was truncated and is not authoritative. Its
identity and admission result must not be transferred to the verified
PinkCherry source.

The converter admits the source header first, extracts only the exact embedded
`config.transformer` metadata, validates the existing H3 topology, and reads
payloads only through exact named ranges. It records source identity and
detects source mutation before and after payload reads.

## 4. Topology and locked policy

| Quantity | Value |
|---|---:|
| Source tensors | `535` |
| Recomputed/excluded | `rope.inv_freq = 1` |
| Stored source tensors | `534` |
| Q6 core linear weights | `208` |
| Q8 block-AdaLN weights | `50` |
| Learned biases | `50` |
| Ordinary tensors | `226` |
| Canonical complete conventional output | `1,050` logical tensors |

The locked quantization recipe is:

```text
core = Q6
block AdaLN = Q8
group_size = 64
quantize_adaln = true
```

The CLI intentionally has no implicit full-conversion scope. Bounded
`--tensor` mode is explicit, and complete conversion requires explicit
`--full`. Duplicate selectors are rejected. Dry-run planning reads zero
payload bytes.

## 5. Implemented safety and output contract

The implementation provides:

- header-first monolithic safetensors admission;
- exact embedded configuration extraction and strict topology classification;
- exact named-range payload reads;
- source identity and mutation detection;
- one-logical-linear-at-a-time MLX quantization;
- BF16-safe decoding without relying on native NumPy BF16 support;
- deterministic streamed sharded output and atomic publication;
- source/output overlap refusal;
- live and dangling symlink destination refusal;
- overwrite refusal;
- exact shard ownership and index verification;
- exact bounded/full topology verification;
- bounded `--tensor` mode;
- explicit `--full` requirement for complete conversion; and
- conventional `config.json`, `quant_config.json`, and
  `model.safetensors.index.json` output.

Adversarial review found and repaired incorrect shard-index ownership
acceptance, incomplete output masquerading as full, dangling destination
symlink acceptance, duplicate selector acceptance, accidental implicit full
conversion scope, and an EOF-whitespace issue.

## 6. Bounded real-host proof

The selected real tensor was:

`token_refiner.blocks.0.attn.out_proj.weight`

| Measurement | Verified value |
|---|---:|
| Source payload read | `77070336` bytes |
| Source range reads | `1` |
| Source logical shape | `[5376, 7168]` BF16 |
| Produced Q6 weight | U32 `[5376, 1344]` |
| Produced scales | BF16 `[5376, 112]` |
| Produced biases | BF16 `[5376, 112]` |
| Output logical payload | `31309824` bytes |
| MLX peak | `262537224` bytes |
| Process maximum RSS | `582746112` bytes |
| Peak memory footprint | `753239024` bytes |
| Conversion wall time | `0.29` seconds |

The generated bounded checkpoint verification was:

```text
VERIFIED
tensor_count = 3
total_size = 31309824
bounded = true
```

The source size, mtime, ctime, and inode remained unchanged. This is a real
host Q6 tensor proof, not a complete beta conversion or an H3 runtime/render
proof.

## 7. Review and test receipts

| Check | Result |
|---|---|
| Initial focused monolithic tests | `18/18 PASS` |
| Focused monolithic tests after adversarial repairs | `33/33 PASS` |
| Checkpoint forge | `30 passed, 2 skipped` |
| Previously reported v0.3f surrounding suite | `44/44 PASS` |
| Previously reported v0.4a surrounding suite | `31/31 PASS` |
| Conditioning replay suite | `5/5 PASS` |
| `py_compile` for all six changed Python paths | Passed |
| `git diff --check` | Passed |
| Independent rereview | `READY / no remaining scoped blocker` |

A later deep operational audit of committed `d6823ea` returned
`GO_FOR_HOST_FULL_BETA_CONVERSION`. That audit found no known unbounded
full-run memory path, expected approximately `30.3 GB` of conventional logical
output, recommended at least `80 GB` of free disk for conversion plus the
streamed-AdaLN forge, confirmed the `1,050`-tensor topology is compatible with
the existing forge, and confirmed that interruption leaves incomplete
temporary output rather than publishing a valid-looking final checkpoint.
The initial production forge should not use `--force`.

## 8. Completion status and explicit non-claims

Slice 024 is **COMPLETE** for implementation and bounded host proof.

The following are explicitly not claimed:

- the full beta conversion has run;
- a complete conventional beta checkpoint exists;
- a streamed-AdaLN beta checkpoint exists;
- a beta H3 render has been performed;
- a complete-conversion memory or wall-time receipt exists; or
- continuation, latent masking, or PR 15375 research was implemented.

The lack of Metal inside agent environments does not invalidate the bounded
real-host Q6 tensor proof, but it also does not constitute beta H3 runtime or
render evidence.

## 9. Next operational step

The next operation is a host execution of the completed converter, not a new
development slice:

1. Run the complete conversion with the verified source and explicit `--full`:

   ```text
   /Users/elbancol/Documents/Codex/2026-08-03/i-am/work/minimax-h3-mlx/.venv/bin/python scripts/build_monolithic_quant.py \
     --source /Users/elbancol/Downloads/PinkCherry-beta-redownload/beta-0.6-fl2va/PinkCherry_fl2va_MiniMax_H3_bf16_beta-0.6.safetensors \
     --output <new-conventional-checkpoint> \
     --full
   ```

2. Source-link verify the resulting conventional checkpoint with the same
   source and output. The CLI requires this `--verify` operation to be separate
   from `--full`.

3. Run the existing streamed-AdaLN forge from the verified conventional
   checkpoint into a new derived destination, then verify that derived
   checkpoint with the forge's existing `--verify` mode.

Do not use `--force` for the first production forge publication. Preserve the
verified source identity and record the full conversion, conventional-output,
forge, and derived-output receipts independently.

## 10. Publication state

| Field | Recorded value |
|---|---|
| Implementation commit | Local `d6823ea75d1758f3286817579f935343d4dae4da` |
| Documentation closeout commit | Not committed |
| Staged paths | None |
| Pushed paths | None |
| Production/test paths changed by closeout | None |
| Publication claim | None |

This closeout stops at the local, validated documentation state.

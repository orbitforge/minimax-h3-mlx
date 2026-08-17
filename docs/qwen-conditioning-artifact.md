# Qwen conditioning artifact boundary

This slice promotes a narrow, text-only replay boundary for MiniMax-H3. The
artifact is `conditioning-artifact.npz`, schema
`minimax-h3-mlx-conditioning-artifact` version `1`.

## Audited production boundary

The current source flow is:

```text
prompt
  -> MiniMaxH3TextEncoder.build_request
  -> tokenizer(prompt, add_special_tokens=False)
  -> truncated Qwen3-VL language stack
  -> hidden_states[50], before the final norm
  -> prompt_embeds / text_token_tags
  -> build_packed_sequence
  -> DiT.condition_proj and token refiner
```

For a text-only request, the replay payload is exactly:

| Value | Required by replay | Contract |
|---|---:|---|
| `text_conditioning` | Yes | `(1, token_count, 5120)`, logical `bfloat16`, unnormalized layer-50 state |
| `text_token_tags` | Yes | `(token_count,)`, H3 modality tags used by packing/AdaLN |
| `token_ids` | No | Stored as `int32` for tokenizer/prompt provenance validation |
| attention mask | No | Qwen creates its internal causal mask; H3 never receives it |
| position IDs | No | H3 packing derives its own media positions from token tags and geometry |

The artifact stores the conditioning tensor as raw BF16 bits in a `uint16`
array. It also records the exact tensor checksum, shape, dtype, token count,
prompt UTF-8 digest, encoder-weight/tokenizer/processor provenance, selected
hidden state, prompt presentation, postprocessing identity, and a canonical
artifact identity hash. Encoder weight files are hashed as bytes without
constructing Qwen. Corrupt, stale, incompatible, or incomplete artifacts fail
closed.

The prior v0.5c/v0.5d worker-separated artifact remains useful historical
machinery for process orchestration, release telemetry, and the proof that
layer-50 conditioning was released before the transformer worker. It is not
safe to promote as the canonical replay payload: it is a geometry/scheduler
proof artifact, requires a sidecar receipt, and serializes BF16 conditioning as
float32. This slice therefore reuses the production encoder and worker-lifecycle
ideas while giving the standalone conditioning boundary one self-validating
representation.

Image-conditioned requests are intentionally outside this artifact. Their
final H3 input also includes VAE-encoded keyframe rows and anchor metadata,
which do not live in `text_conditioning`. Replay rejects those requests rather
than dropping those rows or falling back to Qwen.

## Safe benchmark reuse

The classifications below apply to this text-only artifact. They are based on
the current source flow, where all listed settings enter after Qwen encoding.

| Setting | Classification | Reason |
|---|---|---|
| seed | `CONDITIONING_INDEPENDENT` | Controls downstream noise only |
| LoRA / Turbo adapter selection | `CONDITIONING_INDEPENDENT` | Adapters are attached to H3 projections/AdaLN, not Qwen |
| LoRA scale | `CONDITIONING_INDEPENDENT` | Scales downstream adapter deltas |
| NFE | `CONDITIONING_INDEPENDENT` | Changes scheduler points and transformer forwards |
| scheduler shifts | `CONDITIONING_INDEPENDENT` | Builds downstream video/audio schedules |
| target width/height | `CONDITIONING_INDEPENDENT` | Changes media geometry and packing after text encoding |
| duration/frame count | `CONDITIONING_INDEPENDENT` | Changes media latent geometry after text encoding |

Those classifications do not authorize a live H3 render. The encode-only and
artifact-replay entries are separately testable; the Apple-Silicon generation
gate remains operator-run.

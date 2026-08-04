#!/usr/bin/env bash
# Run the parity suite. The smoke test needs only MLX; the rest compare against the
# `minimax-h3` branch of diffusers and transformers in .venv (see requirements.txt).
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
fail=0
run() {
  echo
  echo "=== $1 ==="
  $PY "$1" 2>&1 | grep -vE "^(Modular|/opt/homebrew.*Warning|  WeightNorm)"
  statuses=("${PIPESTATUS[@]}")
  [ "${statuses[0]}" -eq 0 ] || fail=1
}

$PY tests/test_dit_smoke.py || fail=1
run tests/test_dit_parity.py
run tests/test_video_vae_parity.py
run tests/test_audio_vae_parity.py
run tests/test_text_encoder_parity.py
run tests/test_pipeline_staged_loading.py
run tests/test_quant_roundtrip.py
run tests/test_packing_parity.py
run tests/test_checkpoint_forge.py
echo
[ $fail -eq 0 ] && echo "ALL SUITES PASSED" || echo "SOME SUITES FAILED"
exit $fail

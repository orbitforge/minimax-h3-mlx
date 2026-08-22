#!/bin/zsh
set -e
repo_dir="${0:A:h}"
python_bin="$repo_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  print -u2 "Launch MiniMax H3: required repository interpreter is unavailable: $python_bin"
  exit 1
fi
cd "$repo_dir"
nohup "$python_bin" tools/render_lab/server.py --open \
  >/tmp/minimax-h3-render-lab.log 2>&1 &
disown

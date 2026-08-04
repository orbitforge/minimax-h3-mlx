#!/bin/zsh
cd "$(dirname "$0")"
nohup /usr/bin/python3 scripts/minimax_h3_surface.py --open \
  >/tmp/minimax-h3-surface.log 2>&1 &
disown

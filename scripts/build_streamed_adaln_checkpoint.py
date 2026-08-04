#!/usr/bin/env python3
"""Build or verify MiniMax H3's exact derived streamed-AdaLN checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from minimax_h3_mlx.checkpoint_forge.forge import ForgeOptions, forge_checkpoint
from minimax_h3_mlx.checkpoint_forge.topology import parse_block_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="inspect topology and disk requirements without writing")
    parser.add_argument("--verify", action="store_true", help="verify an existing derived checkpoint exactly")
    parser.add_argument("--force", action="store_true", help="replace an existing output after verified conversion using backup-and-rollback")
    parser.add_argument("--blocks", help="bounded block selection, for example 0 or 5 or 0,5 or 0-1")
    args = parser.parse_args()
    try:
        blocks = parse_block_selection(args.blocks)
        result = forge_checkpoint(ForgeOptions(args.source, args.output, args.dry_run, args.verify, args.force, blocks))
        print(result.message)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"checkpoint forge failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

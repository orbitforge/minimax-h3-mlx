#!/usr/bin/env python3
"""Build or verify a conventional MLX quantized checkpoint from one H3 safetensors file.

This CLI intentionally has no full-conversion default: callers must select bounded tensors or pass
``--full``, and the real beta source is never read beyond headers unless conversion is requested.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from minimax_h3_mlx.monolithic_quant import OUTPUT_RECIPE, build_conversion_plan, convert, verify_output
from minimax_h3_mlx.monolithic_source import MonolithicSafetensorsSource, MonolithicSourceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="one monolithic BF16/F32 MiniMax H3 safetensors file")
    parser.add_argument("--output", required=True, help="new output directory, or an existing directory with --verify")
    parser.add_argument("--bits", type=int, default=6)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--adaln-bits", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true", help="inspect and plan without writing output")
    parser.add_argument("--verify", action="store_true", help="verify an existing output directory")
    parser.add_argument(
        "--tensor",
        action="append",
        dest="tensors",
        help="bounded logical quantized weight selector; repeat for a few Q6 core weights",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="explicitly plan or run the complete 535-source/1050-output conversion",
    )
    parser.add_argument(
        "--shard-bytes",
        type=int,
        default=5 * 1024**3,
        help=argparse.SUPPRESS,
    )
    return parser


def _validate_recipe(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    requested = {
        "bits": args.bits,
        "group_size": args.group_size,
        "adaln_bits": args.adaln_bits,
    }
    expected = {key: OUTPUT_RECIPE[key] for key in requested}
    if requested != expected:
        parser.error(
            "Slice 024 is locked to --bits 6 --group-size 64 --adaln-bits 8; "
            f"got {requested}"
        )


def _plan_receipt(plan) -> dict[str, object]:
    return {
        "status": "DRY_RUN",
        "source": str(plan.source.path),
        "source_size": plan.source.source_size,
        "source_identity": plan.source.identity,
        "header_bytes_read": plan.source.header_bytes_read,
        "payload_bytes_read": plan.source.payload_bytes_read,
        "source_tensor_count": plan.source_tensor_count,
        "stored_source_tensor_count": plan.stored_source_tensor_count,
        "output_tensor_count": plan.output_tensor_count,
        "bounded": plan.bounded,
        "selected_quantized_weights": list(plan.selected_quantized_weights),
        "classification_counts": dict(plan.classification.counts),
        "quant_config": plan.quant_config,
        "qkv_source_layout": plan.classification.qkv_layout.source_layout,
        "qkv_canonical_layout": plan.classification.qkv_layout.canonical_layout,
        "qkv_row_reconciliation_applied": False,
        "qkv_tensors_reconciled": 0,
        "qkv_tensors_planned": plan.qkv_tensors_planned,
        "qkv_layout_source_identity": (
            plan.classification.qkv_layout.source_identity or plan.source.identity
        ),
        "qkv_layout_authorization": plan.classification.qkv_layout.authorization,
        "shards": [{"filename": shard.filename, "tensor_count": len(shard.tensors), "bytes": shard.nbytes} for shard in plan.shards()],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_recipe(args, parser)
    if args.verify:
        if args.dry_run or args.tensors or args.full:
            parser.error("--verify cannot be combined with --dry-run, --tensor, or --full")
    else:
        if bool(args.tensors) == args.full:
            parser.error("conversion requires exactly one scope: one or more --tensor values OR --full")
        if args.tensors:
            seen: set[str] = set()
            duplicates: set[str] = set()
            for name in args.tensors:
                if name in seen:
                    duplicates.add(name)
                seen.add(name)
            if duplicates:
                parser.error(f"duplicate --tensor selector(s): {sorted(duplicates)}")
    output = Path(args.output).expanduser()

    try:
        if args.verify:
            source = MonolithicSafetensorsSource(args.source) if args.source else None
            receipt = verify_output(output, source=source)
            print(json.dumps({"status": "VERIFIED", **receipt.__dict__}, indent=2, default=str))
            return 0

        if not args.source:
            parser.error("--source is required unless --verify is used")
        source = MonolithicSafetensorsSource(args.source)
        selected = None if args.full else tuple(args.tensors)
        plan = build_conversion_plan(source, selected_quantized_weights=selected)
        if selected:
            invalid = [
                name
                for name in plan.selected_quantized_weights
                if plan.classification.by_name[name].role != "q6_core_weight"
            ]
            if invalid:
                raise MonolithicSourceError(
                    "the bounded CLI selector is restricted to Q6 core weights; "
                    f"Q8 block AdaLN was not selected: {invalid}"
                )
        if args.dry_run:
            print(json.dumps(_plan_receipt(plan), indent=2, default=str))
            return 0

        receipt = convert(plan, output, target_shard_bytes=args.shard_bytes)
        print(
            json.dumps(
                {
                    "status": "CONVERTED",
                    "output": str(receipt.output),
                    "shards": list(receipt.shard_names),
                    "output_tensor_count": receipt.output_tensor_count,
                    "source_identity": receipt.source_identity,
                    "source_size": receipt.source_size,
                    "header_bytes_read": receipt.header_bytes_read,
                    "payload_bytes_read": receipt.payload_bytes_read,
                    "range_read_count": receipt.range_read_count,
                    "memory_snapshots": list(receipt.memory_snapshots),
                    "qkv_source_layout": receipt.qkv_source_layout,
                    "qkv_canonical_layout": receipt.qkv_canonical_layout,
                    "qkv_row_reconciliation_applied": receipt.qkv_row_reconciliation_applied,
                    "qkv_tensors_reconciled": receipt.qkv_tensors_reconciled,
                    "qkv_layout_source_identity": receipt.qkv_layout_source_identity,
                    "qkv_layout_authorization": receipt.qkv_layout_authorization,
                },
                indent=2,
                default=str,
            )
        )
        return 0
    except (MonolithicSourceError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

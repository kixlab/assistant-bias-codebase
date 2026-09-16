#!/usr/bin/env python3
"""Summarize turn-termination error by steering strength."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from utils import ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/01_turn_termination/output/predictions.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/01_turn_termination/output/summary.jsonl",
    )
    args = parser.parse_args()
    groups = defaultdict(list)
    for row in read_jsonl(args.input):
        groups[float(row["alpha"])].append(row)
    summary = []
    for alpha, rows in sorted(groups.items()):
        errors = [float(row["predicted_turn"]) - int(row["gt_turn"]) for row in rows]
        summary.append(
            {
                "alpha": alpha,
                "n": len(rows),
                "mean_signed_error": mean(errors),
                "mean_absolute_error": mean(abs(value) for value in errors),
                "exact_match_rate": mean(value == 0 for value in errors),
                "order_agreement_rate": mean(bool(row["orders_agree"]) for row in rows),
            }
        )
    write_jsonl(args.output, summary)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()

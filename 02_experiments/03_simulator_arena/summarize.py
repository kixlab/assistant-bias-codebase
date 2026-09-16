#!/usr/bin/env python3
"""Summarize all available conditions after averaging judge trials per example."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from utils import ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "02_experiments/03_simulator_arena/output")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/03_simulator_arena/output/summary.jsonl",
    )
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    filename = "evaluations_raw.jsonl" if args.raw else "evaluations.jsonl"
    paths = sorted(args.root.glob(f"*/a*/{filename}"))
    if not paths:
        raise FileNotFoundError(f"No {filename} files under {args.root}; run evaluate.py first")
    trials = defaultdict(list)
    for path in paths:
        for row in read_jsonl(path):
            key = (row["condition"], row["alpha"], row["sample_id"], row["mode"])
            trials[key].append(row["score"])
    groups = defaultdict(lambda: defaultdict(list))
    for (condition, alpha, sample_id, mode), values in trials.items():
        groups[(condition, alpha)][mode].append(mean(values))
    output = []
    print("condition | alpha | n | writing | interaction | combined")
    for (condition, alpha), modes in sorted(groups.items()):
        writing, interaction = mean(modes["writing"]), mean(modes["interaction"])
        row = {
            "condition": condition, "alpha": alpha, "n": len(modes["writing"]),
            "writing_score": writing, "interaction_score": interaction,
            "combined_score": mean([writing, interaction]),
        }
        output.append(row)
        print(
            f"{condition} | {alpha:.1f} | {row['n']} | {writing:.3f} | "
            f"{interaction:.3f} | {row['combined_score']:.3f}"
        )
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()

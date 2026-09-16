#!/usr/bin/env python3
"""Correlate unsteered SimulatorArena activation with mean judge-trial scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from utils import (
    CONDITIONS, ROOT, activation_directory, correlation_rows,
    read_jsonl, run_directory, write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activations", type=Path)
    parser.add_argument("--evaluations", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="base")
    parser.add_argument("--raw", action="store_true")
    args = parser.parse_args()
    activation_name = "activations_raw.jsonl" if args.raw else "activations.jsonl"
    evaluation_name = "evaluations_raw.jsonl" if args.raw else "evaluations.jsonl"
    activation_path = args.activations or activation_directory(args.condition) / activation_name
    evaluation_path = args.evaluations or run_directory(args.condition) / evaluation_name

    trials = defaultdict(list)
    for row in read_jsonl(evaluation_path):
        if float(row["alpha"]) != 0.0:
            raise ValueError("Activation correlation requires unsteered evaluations (alpha 0.0).")
        key = (row["condition"], str(row["sample_id"]), str(row["mode"]))
        trials[key].append(float(row["score"]))
    scores = {key: mean(values) for key, values in trials.items()}
    joined = []
    for row in read_jsonl(activation_path):
        if float(row["alpha"]) != 0.0:
            raise ValueError("Activation correlation requires unsteered activations (alpha 0.0).")
        sample_id = str(row["sample_id"])
        writing = scores.get((row["condition"], sample_id, "writing"))
        interaction = scores.get((row["condition"], sample_id, "interaction"))
        if writing is None or interaction is None:
            raise ValueError(f"Missing writing or interaction evaluations for {sample_id}")
        joined.append({
            **row,
            "writing_score": writing,
            "interaction_score": interaction,
            "combined_score": mean([writing, interaction]),
        })

    output = []
    for target in ("writing_score", "interaction_score", "combined_score"):
        correlations = correlation_rows(
            joined,
            group_key=lambda row: row["condition"],
            x_key="mean_activation",
            y_key=target,
        )
        output.extend({**row, "target": target} for row in correlations)
    filename = "correlations_raw.jsonl" if args.raw else "correlations.jsonl"
    output_path = args.output or (
        ROOT / "02_experiments/04_activation_correlation/output" / args.condition / filename
    )
    write_jsonl(output_path, output)
    for row in output:
        print(row)


if __name__ == "__main__":
    main()

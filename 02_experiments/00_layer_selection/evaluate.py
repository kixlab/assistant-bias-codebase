#!/usr/bin/env python3
"""Judge request style and summarize each layer/alpha condition."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

from utils import ROOT, call_json, read_jsonl, read_prompt, render_prompt, write_jsonl


METRICS = ("brevity", "informality", "information_pacing")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/requests.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/evaluations.jsonl",
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/summary.jsonl",
    )
    parser.add_argument("--judge-model", default="gpt-5-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    template = read_prompt(Path(__file__).parent / "prompts/judge.txt")

    def evaluate(row: dict) -> dict:
        scores = call_json(
            render_prompt(template, request_message=row["generated_request"]),
            model=args.judge_model,
        )
        parsed = {metric: int(scores[metric]) for metric in METRICS}
        if any(value not in range(1, 6) for value in parsed.values()):
            raise ValueError(f"Scores must be in [1, 5]: {parsed}")
        return {**row, "scores": parsed, "mean_score": mean(parsed.values())}

    evaluated = [evaluate(row) for row in rows]
    write_jsonl(args.output, evaluated)

    groups: dict[tuple[int, float], list[dict]] = defaultdict(list)
    for row in evaluated:
        groups[(row["layer"], row["alpha"])].append(row)
    summary = []
    for (layer, alpha), items in sorted(groups.items()):
        summary.append(
            {
                "layer": layer,
                "alpha": alpha,
                "n": len(items),
                **{metric: mean(item["scores"][metric] for item in items) for metric in METRICS},
                "mean_score": mean(item["mean_score"] for item in items),
            }
        )
    write_jsonl(args.summary, summary)


if __name__ == "__main__":
    main()

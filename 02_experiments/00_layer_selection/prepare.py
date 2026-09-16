#!/usr/bin/env python3
"""Generate the task goals used by the layer sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT, call_json, read_prompt, render_prompt, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/goals.jsonl",
    )
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()
    template = read_prompt(Path(__file__).parent / "prompts/goals.txt")
    payload = call_json(
        render_prompt(template, request_count=args.count + 20),
        model=args.judge_model,
        max_output_tokens=8192,
    )
    goals, seen = [], set()
    for item in payload["goals"]:
        goal = " ".join(item["task_goal"].split())
        if goal and goal.lower() not in seen:
            seen.add(goal.lower())
            goals.append({"item_id": f"item_{len(goals) + 1:04d}", "task_goal": goal})
        if len(goals) == args.count:
            break
    if len(goals) != args.count:
        raise ValueError(f"Expected {args.count} unique goals, received {len(goals)}")
    write_jsonl(args.output, goals)
    print(f"wrote {len(goals)} goals to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate requests across layers and user-direction steering strengths."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Qwen35
from utils import ROOT, read_jsonl, read_prompt, render_prompt, write_jsonl


def comma_values(raw: str, cast: type) -> list:
    return [cast(value.strip()) for value in raw.split(",") if value.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--goals", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/goals.jsonl",
        help="JSONL rows with item_id and task_goal.",
    )
    parser.add_argument("--vectors", type=Path, default=ROOT / "01_method/output/role_vectors")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/00_layer_selection/output/requests.jsonl",
    )
    parser.add_argument("--layers", default="1,5,9,13,17,21,25,29")
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3")
    parser.add_argument(
        "--steering-mode",
        choices=("all_tokens", "response_tokens", "first_response_token"),
        default="all_tokens",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    goals = read_jsonl(args.goals)
    template = read_prompt(Path(__file__).parent / "prompts/request.txt")
    layers = comma_values(args.layers, int)
    alphas = comma_values(args.alphas, float)
    model = Qwen35(device=args.device)
    rows = []
    for layer in layers:
        for alpha in alphas:
            for item in goals:
                prompt = render_prompt(template, task_goal=item["task_goal"])
                generated = model.generate(
                    [{"role": "user", "content": prompt}],
                    vector_root=args.vectors,
                    layer=layer,
                    alpha=alpha,
                    steering_mode=args.steering_mode,
                    max_new_tokens=args.max_new_tokens,
                )
                rows.append(
                    {
                        "item_id": item["item_id"],
                        "task_goal": item["task_goal"],
                        "layer": layer,
                        "alpha": alpha,
                        "generated_request": generated,
                    }
                )
            print(f"completed layer={layer} alpha={alpha}")
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()

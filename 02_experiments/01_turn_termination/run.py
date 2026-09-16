#!/usr/bin/env python3
"""Predict where users disengage, with balanced answer order."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Qwen35
from utils import ROOT, read_jsonl, read_prompt, render_prompt, write_jsonl


ORDERS = {
    "continue_a": {"A": "continue", "B": "disengage"},
    "disengage_a": {"A": "disengage", "B": "continue"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/01_turn_termination/output/dataset.jsonl",
    )
    parser.add_argument("--vectors", type=Path, default=ROOT / "01_method/output/role_vectors")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/01_turn_termination/output/predictions.jsonl",
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--alphas", default="0.0,0.1,0.2,0.3")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def prefix_text(turns: list[dict], end: int) -> str:
    return "\n\n".join(
        f"USER: {turn['user']}\nASSISTANT: {turn['assistant']}" for turn in turns[:end]
    )


def decision(text: str, options: dict[str, str]) -> str:
    choice = text.strip().upper()
    if choice not in options:
        raise ValueError(f"Could not parse decision: {text!r}")
    return options[choice]


def main() -> None:
    args = parse_args()
    template = read_prompt(Path(__file__).parent / "prompts/decision.txt")
    model = Qwen35(device=args.device)
    output = []
    for alpha in [float(value) for value in args.alphas.split(",")]:
        for row in read_jsonl(args.input):
            trials = {}
            for order_name, options in ORDERS.items():
                predicted_turn = 0
                steps = []
                for turn_number in range(1, len(row["turns"]) + 1):
                    prompt = render_prompt(
                        template,
                        user_intent=row.get("user_intent", ""),
                        conversation=prefix_text(row["turns"], turn_number),
                        option_a=options["A"],
                        option_b=options["B"],
                    )
                    generated = model.generate(
                        [{"role": "user", "content": prompt}],
                        vector_root=args.vectors,
                        layer=args.layer,
                        alpha=alpha,
                        max_new_tokens=args.max_new_tokens,
                    )
                    choice = decision(generated, options)
                    steps.append({"turn": turn_number, "decision": choice, "raw_response": generated})
                    if choice == "disengage":
                        predicted_turn = turn_number
                        break
                trials[order_name] = {"predicted_turn": predicted_turn, "steps": steps}
            predictions = [trial["predicted_turn"] for trial in trials.values()]
            output.append(
                {
                    "sample_id": row["sample_id"],
                    "alpha": alpha,
                    "gt_turn": int(row["gt_turn"]),
                    "predicted_turn": sum(predictions) / len(predictions),
                    "orders_agree": len(set(predictions)) == 1,
                    "trials": trials,
                }
            )
        print(f"completed alpha={alpha}")
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()

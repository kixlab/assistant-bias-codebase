#!/usr/bin/env python3
"""Teacher-force SimulatorArena user turns and measure role-vector activation."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Qwen35
from utils import (
    CONDITIONS, VECTORS, activation_directory, read_jsonl,
    run_directory, truncate_conversation, write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--vectors", type=Path, default=VECTORS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="base")
    parser.add_argument(
        "--raw", action="store_true", help="Measure without applying termination predictions."
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--centered", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    input_path = args.input or run_directory(args.condition) / "conversations.jsonl"
    rows = read_jsonl(input_path)
    if any(float(row["alpha"]) != 0.0 for row in rows):
        raise ValueError("Activation correlation requires unsteered source runs (alpha 0.0).")
    termination_path = input_path.parent / "terminations.jsonl"
    terminations = (
        {row["sample_id"]: row["ending_user_turn"] for row in read_jsonl(termination_path)}
        if termination_path.exists() and not args.raw else {}
    )
    model = Qwen35(device=args.device)
    output = []
    for row in rows:
        weighted_sum = 0.0
        token_total = 0
        message_scores = []
        conversation = row["conversation"]
        if terminations:
            conversation = truncate_conversation(conversation, terminations[row["sample_id"]])
        for message in conversation:
            if message.get("role") != "user":
                continue
            prompt = str(message.get("generation_prompt") or "")
            if not prompt:
                raise ValueError(f"{row['sample_id']} is missing a saved user generation prompt")
            response = str(message["content"])
            messages = [{"role": "user", "content": prompt}]
            score = model.project_response(
                messages,
                response,
                vector_root=args.vectors,
                layer=args.layer,
                pooling="full",
                centered=args.centered,
            )
            assert model.tokenizer is not None
            tokens = len(model.tokenizer.encode(response, add_special_tokens=False))
            weighted_sum += score * tokens
            token_total += tokens
            message_scores.append(score)
        output.append(
            {
                "sample_id": row["sample_id"],
                "alpha": row["alpha"],
                "condition": row["condition"],
                "message_count": len(message_scores),
                "response_token_count": token_total,
                "message_activations": message_scores,
                "mean_activation": weighted_sum / token_total if token_total else None,
                "centered": args.centered,
            }
        )
        print(f"measured {row['sample_id']}")
    filename = "activations_raw.jsonl" if args.raw else "activations.jsonl"
    write_jsonl(args.output or activation_directory(args.condition) / filename, output)


if __name__ == "__main__":
    main()

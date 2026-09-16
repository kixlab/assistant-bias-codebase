#!/usr/bin/env python3
"""Detect the final relevant user turn without modifying saved conversations."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    CONDITIONS, call_json, read_jsonl, read_prompt, render_prompt,
    run_directory, user_messages_text, write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="base")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()
    directory = run_directory(args.condition, args.alpha)
    rows = read_jsonl(args.input or directory / "conversations.jsonl")
    template = read_prompt(Path(__file__).parent / "prompts/termination.txt")

    def predict(row: dict) -> dict:
        annotation = row.get("annotation") or {}
        conversation = row["conversation"]
        actual = sum(message.get("role") == "user" for message in conversation)
        if actual == 0:
            return {"sample_id": row["sample_id"], "ending_user_turn": 0, "raw_response": None}
        prompt = render_prompt(
            template,
            problem=annotation.get("question") or annotation.get("math_problem") or "",
            user_messages=user_messages_text(conversation, numbered=True),
        )
        raw = call_json(prompt, model=args.judge_model, max_output_tokens=8192)
        reported = int(raw["Ending Turn Number"])
        return {
            "sample_id": row["sample_id"],
            "ending_user_turn": max(1, min(reported, actual)),
            "reported_ending_user_turn": reported,
            "termination_reason": raw.get("Termination Reason"),
            "raw_response": raw,
        }

    write_jsonl(args.output or directory / "terminations.jsonl", (predict(row) for row in rows))


if __name__ == "__main__":
    main()

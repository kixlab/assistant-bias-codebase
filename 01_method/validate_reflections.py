#!/usr/bin/env python3
"""Validate whether each generated reflection represents its assigned role."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT, call_text, read_jsonl, read_prompt, render_prompt, write_jsonl


VALID_LABELS = {"strongly_represented", "weakly_represented", "not_represented"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "01_method/output/reflections.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "01_method/output/reflections_validated.jsonl")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    prompt_root = Path(__file__).parent / "prompts"
    system_prompt = read_prompt(prompt_root / "validate_reflection_system.txt")
    template = read_prompt(prompt_root / "validate_reflection.txt")

    def validate(row: dict, reflection: dict) -> str:
        prompt = render_prompt(
            template,
            role=reflection["role"],
            dialogue=row["dialogue_text"],
            reflection=reflection["answer"],
        )
        messages = [
            {"role": "system", "content": system_prompt.format(role=reflection["role"])},
            {"role": "user", "content": prompt},
        ]
        label = call_text(messages, model=args.judge_model, max_output_tokens=4096).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(f"Unexpected validation label: {label!r}")
        return label

    count = 0
    for row in rows:
        for reflection in row["reflections"]:
            reflection["representation"] = validate(row, reflection)
            count += 1
    write_jsonl(args.output, rows)
    print(f"validated {count} reflections")


if __name__ == "__main__":
    main()

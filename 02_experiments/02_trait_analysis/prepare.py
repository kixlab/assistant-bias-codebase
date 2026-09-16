#!/usr/bin/env python3
"""Generate controlled instruction pairs and elicitation questions for each trait."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model import Qwen35
from traits import selected_traits
from utils import ROOT, read_prompt, render_prompt, write_jsonl


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Qwen did not return JSON: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/trait_data.jsonl",
    )
    parser.add_argument("--trait", action="append")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    args = parser.parse_args()
    template = read_prompt(Path(__file__).parent / "prompts/build_trait_data.txt")
    model = Qwen35(device=args.device)
    rows = []
    for trait, definition in selected_traits(args.trait).items():
        prompt = render_prompt(template, trait=trait, definition=definition)
        payload = parse_json(
            model.generate(
                [{"role": "system",
                  "content": "You generate controlled research data for trait-vector extraction."},
                 {"role": "user", "content": prompt}],
                max_new_tokens=args.max_new_tokens,
            )
        )
        pairs = payload["instruction"]
        questions = payload["questions"]
        if len(pairs) != 5 or len(questions) != 40:
            raise ValueError(f"{trait}: expected 5 pairs and 40 questions")
        rows.append(
            {
                "trait": trait,
                "definition": definition,
                "instruction_pairs": pairs,
                "questions": questions,
                "eval_prompt": payload["eval_prompt"],
            }
        )
        print(f"prepared {trait}")
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()

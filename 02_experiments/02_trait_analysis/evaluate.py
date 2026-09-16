#!/usr/bin/env python3
"""Score trait expression in generated responses."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import ROOT, call_text, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/responses.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/evaluations.jsonl",
    )
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()
    rows = read_jsonl(args.input)

    def evaluate(row: dict) -> dict:
        prompt = (
            row["eval_prompt"]
            .replace("{{question}}", row["question"]).replace("{{answer}}", row["response"])
            .replace("{question}", row["question"]).replace("{answer}", row["response"])
        )
        raw = call_text(prompt, model=args.judge_model, max_output_tokens=4096).strip()
        score = None if raw.upper() == "REFUSAL" else int(raw)
        if score is not None and score not in range(101):
            raise ValueError(f"Invalid trait score: {score}")
        return {**row, "trait_score": score}

    write_jsonl(args.output, (evaluate(row) for row in rows))


if __name__ == "__main__":
    main()

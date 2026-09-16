#!/usr/bin/env python3
"""Generate positive and negative Qwen responses for trait-vector extraction."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Qwen35
from utils import ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/trait_data.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/responses.jsonl",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    model = Qwen35(device=args.device)
    output = []
    for trait_row in read_jsonl(args.input):
        for pair_index, pair in enumerate(trait_row["instruction_pairs"]):
            for question_index, question in enumerate(trait_row["questions"]):
                for polarity in ("pos", "neg"):
                    messages = [
                        {"role": "system", "content": pair[polarity]},
                        {"role": "user", "content": question},
                    ]
                    response = model.generate(messages, max_new_tokens=args.max_new_tokens)
                    output.append(
                        {
                            "trait": trait_row["trait"],
                            "definition": trait_row["definition"],
                            "eval_prompt": trait_row["eval_prompt"],
                            "pair_id": pair_index,
                            "question_id": question_index,
                            "polarity": polarity,
                            "instruction": pair[polarity],
                            "question": question,
                            "response": response,
                        }
                    )
        print(f"completed {trait_row['trait']}")
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()

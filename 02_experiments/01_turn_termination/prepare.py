#!/usr/bin/env python3
"""Filter WildChat, code disengagement, and build a type-balanced prediction set."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

from utils import ROOT, call_json, read_jsonl, read_prompt, render_prompt, write_jsonl


TYPES = (
    "Normal Completion", "Forgetting", "Technical Failure", "Misinterpretation",
    "Expectation Mismatch", "Unsolicited Action", "Refusal", "Hallucination",
)


def load_dialogues(source: Path) -> dict[str, dict]:
    from datasets import load_dataset

    files = sorted(str(path) for path in source.rglob("train*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train*.parquet files under {source}. See this experiment's README.md.")
    rows = load_dataset("parquet", data_files=files, split="train", streaming=True)
    dialogues = {}
    for row in rows:
        if int(row["turn"]) != 8:
            continue
        messages = row["conversation"]
        if len(messages) != 16:
            continue
        if any(
            message["role"] != ("user" if index % 2 == 0 else "assistant")
            or not message["content"].strip()
            or len(message["content"].split()) > 1000
            for index, message in enumerate(messages)
        ):
            continue
        normalized = [{"role": item["role"], "content": item["content"]} for item in messages]
        cid = str(row["conversation_hash"])
        dialogues[cid] = {
            "conversation_id": cid,
            "turns": [
                {"user": normalized[index]["content"], "assistant": normalized[index + 1]["content"]}
                for index in range(0, 16, 2)
            ],
            "dialogue": normalized,
        }
    return dialogues


def transcript(row: dict) -> str:
    return "\n\n".join(
        f"TURN {index}:\nUser: {turn['user']}\nAssistant: {turn['assistant']}"
        for index, turn in enumerate(row["turns"], start=1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/WildChat")
    parser.add_argument(
        "--coded-input", type=Path, help="Optional original coded-results JSON with a results list."
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/01_turn_termination/output/dataset.jsonl",
    )
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--judge-model", default="gpt-5-mini")
    args = parser.parse_args()
    dialogues = load_dialogues(args.source)
    if not dialogues:
        raise ValueError("No alternating eight-turn dialogues with at most 1,000 words per message")

    prompts = Path(__file__).parent / "prompts"
    coding_template = read_prompt(prompts / "code_disengagement.txt")
    coded_path = args.output.parent / "coded.jsonl"
    coded_input = args.coded_input or args.source / "coded_results.json"
    if args.coded_input or coded_input.exists():
        coded = json.loads(coded_input.read_text(encoding="utf-8"))["results"]
    else:
        previous = read_jsonl(coded_path) if coded_path.exists() else []
        by_id = {str(row["conversation_id"]): row for row in previous}

        def code_rows():
            for cid, row in dialogues.items():
                if cid in by_id:
                    yield by_id[cid]
                    continue
                analysis = call_json(
                    render_prompt(coding_template, conversation=transcript(row)),
                    model=args.judge_model,
                    max_output_tokens=8192,
                )
                yield {"conversation_id": cid, "analysis": analysis}
                print(f"coded {cid}")

        write_jsonl(coded_path, code_rows())
        coded = read_jsonl(coded_path)

    grouped = defaultdict(list)
    for row in coded:
        analysis = row["analysis"]
        cid = str(row["conversation_id"])
        category = analysis["disengagement_type"]
        if category in TYPES and analysis["type_valid"] is True and cid in dialogues:
            grouped[category].append(cid)
    rng = random.Random(args.seed)
    selected = []
    for category in TYPES:
        ids = sorted(set(grouped[category]))
        rng.shuffle(ids)
        selected.extend((category, cid) for cid in ids[:args.per_type])
        print(f"{category}: {min(len(ids), args.per_type)} / {args.per_type}")
    if not selected:
        raise ValueError("No valid coded dialogues; inspect coded.jsonl")
    intent_template = read_prompt(prompts / "intent.txt")

    def records():
        for category, cid in selected:
            row = dialogues[cid]
            dialogue = "\n\n".join(
                f"{item['role'].upper()}: {item['content']}" for item in row["dialogue"]
            )
            metadata = call_json(
                render_prompt(intent_template, dialogue=dialogue),
                model=args.judge_model,
                max_output_tokens=8192,
            )
            yield {
                "sample_id": cid,
                "disengagement_type": category,
                "gt_turn": 8,
                "turns": row["turns"],
                "user_intent": metadata["user_intent"],
            }

    write_jsonl(args.output, records())
    print(f"wrote {len(selected)} examples to {args.output}")


if __name__ == "__main__":
    main()

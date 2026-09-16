#!/usr/bin/env python3
"""Filter, label, and stratify LMSYS-Chat-1M conversations."""

from __future__ import annotations

import argparse
from collections import defaultdict
import random
from pathlib import Path
from typing import Any

from utils import ROOT, call_text, read_jsonl, read_prompt, write_jsonl


LABELS = (
    "edit_or_critique_provided_text",
    "argument_or_summary_generation",
    "personal_writing_or_communication",
    "write_fiction",
    "how_to_advice",
    "creative_ideation",
    "tutoring_or_teaching",
    "translation",
    "mathematical_calculation",
    "computer_programming",
    "purchasable_products",
    "cooking_and_recipes",
    "health_fitness_beauty_or_self_care",
    "specific_info",
    "greetings_and_chitchat",
    "relationships_and_personal_reflection",
    "games_and_role_play",
    "asking_about_the_model",
    "create_an_image",
    "analyze_an_image",
    "generate_or_retrieve_other_media",
    "data_analysis",
    "unclear",
    "other",
)


def conversation(row: dict[str, Any]) -> list[dict[str, str]]:
    raw = row.get("conversation") or row.get("messages") or []
    return [
        {"role": str(item.get("role") or "").lower(), "content": str(item.get("content") or "").strip()}
        for item in raw
        if isinstance(item, dict)
    ]


def valid(row: dict[str, Any], *, min_messages: int, max_messages: int) -> bool:
    messages = conversation(row)
    return (
        min_messages <= len(messages) <= max_messages
        and all(item["role"] in {"user", "assistant"} and item["content"] for item in messages)
    )


def filter_rows(args: argparse.Namespace) -> None:
    from datasets import load_dataset

    files = sorted(str(path) for path in args.source.rglob("train*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train*.parquet files under {args.source}. See 00_sampling/README.md.")
    source = load_dataset("parquet", data_files=files, split="train", streaming=True)
    rng = random.Random(args.seed)
    reservoir: list[dict[str, Any]] = []
    accepted = 0
    for seen, raw in enumerate(source, start=1):
        if args.max_source_rows and seen > args.max_source_rows:
            break
        row = {"conversation_id": raw["conversation_id"], "conversation": conversation(raw)}
        if not valid(row, min_messages=args.min_messages, max_messages=args.max_messages):
            continue
        accepted += 1
        if len(reservoir) < args.count:
            reservoir.append(row)
        else:
            index = rng.randrange(accepted)
            if index < args.count:
                reservoir[index] = row
    write_jsonl(args.output, reservoir)
    print(f"wrote {len(reservoir)} filtered conversations to {args.output}")


def format_label_input(row: dict[str, Any], prompt: str, max_messages: int) -> str:
    messages = conversation(row)[-max_messages:]
    transcript = "\n".join(f"{item['role'].upper()}: {item['content']}" for item in messages)
    return f"Conversation transcript:\n{transcript}\n\n{prompt}"


def label_rows(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.input)
    prompt = read_prompt(args.prompt)

    def label(row: dict[str, Any]) -> dict[str, Any]:
        raw = call_text(
            format_label_input(row, prompt, args.max_context_messages),
            model=args.judge_model,
            max_output_tokens=4096,
        )
        topic = raw.strip().lower().replace("`", "")
        if topic not in LABELS:
            raise ValueError(f"Invalid topic label: {raw!r}")
        return {**row, "topic_label": topic}

    labeled = [label(row) for row in rows]
    write_jsonl(args.output, labeled)
    print(f"wrote {len(labeled)} labeled conversations to {args.output}")


def sample_rows(args: argparse.Namespace) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.input):
        label = str(row.get("topic_label") or "")
        if label in LABELS:
            grouped[label].append(row)
    rng = random.Random(args.seed)
    sampled = []
    for label in LABELS:
        rows = grouped[label]
        if len(rows) < args.per_label:
            raise ValueError(f"{label} has {len(rows)} rows; need {args.per_label}")
        sampled.extend(rng.sample(rows, args.per_label))
    rng.shuffle(sampled)
    output = [
        {
            "id": str(row.get("conversation_id") or row.get("id") or f"dialogue_{index:04d}"),
            "topic": row["topic_label"],
            "conversation": conversation(row),
        }
        for index, row in enumerate(sampled, start=1)
    ]
    write_jsonl(args.output, output)
    print(f"wrote {len(output)} balanced conversations to {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    filter_parser = subparsers.add_parser("filter", help="Reservoir-sample valid LMSYS conversations.")
    filter_parser.add_argument("--source", type=Path, default=ROOT / "data/LMSYS-Chat-1M")
    filter_parser.add_argument("--output", type=Path, default=ROOT / "00_sampling/output/filtered.jsonl")
    filter_parser.add_argument("--count", type=int, default=100_000)
    filter_parser.add_argument("--min-messages", type=int, default=2)
    filter_parser.add_argument("--max-messages", type=int, default=50)
    filter_parser.add_argument("--max-source-rows", type=int)
    filter_parser.add_argument("--seed", type=int, default=42)
    filter_parser.set_defaults(func=filter_rows)

    label_parser = subparsers.add_parser("label", help="Label the final user message topic.")
    label_parser.add_argument("--input", type=Path, default=ROOT / "00_sampling/output/filtered.jsonl")
    label_parser.add_argument("--output", type=Path, default=ROOT / "00_sampling/output/labeled.jsonl")
    label_parser.add_argument(
        "--prompt",
        type=Path,
        default=Path(__file__).parent / "prompts/topic_label.txt",
    )
    label_parser.add_argument("--judge-model", default="gpt-5-mini")
    label_parser.add_argument("--max-context-messages", type=int, default=10)
    label_parser.set_defaults(func=label_rows)

    sample_parser = subparsers.add_parser("sample", help="Sample an equal number per topic.")
    sample_parser.add_argument("--input", type=Path, default=ROOT / "00_sampling/output/labeled.jsonl")
    sample_parser.add_argument("--output", type=Path, default=ROOT / "00_sampling/output/dialogues.jsonl")
    sample_parser.add_argument("--per-label", type=int, default=30)
    sample_parser.add_argument("--seed", type=int, default=42)
    sample_parser.set_defaults(func=sample_rows)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.func(arguments)

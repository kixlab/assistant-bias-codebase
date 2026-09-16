#!/usr/bin/env python3
"""Run a math-tutoring condition with a steered Qwen user and an API tutor."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from model import Qwen35
from utils import (
    CONDITIONS, ROOT, VECTORS, call_text, dialogue_text, read_jsonl,
    read_prompt, render_prompt, run_directory, write_jsonl,
)


def visible_message(text: str) -> str:
    stripped = text.strip()
    if re.fullmatch(r"(?is)terminate conversation[.!]?", stripped):
        return "terminate conversation"
    matches = list(re.finditer(r"(?im)^\s*(?:Query|Response)\s*:\s*", stripped))
    return stripped[matches[-1].end():].strip() if matches else stripped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/03_simulator_arena/output/dataset.jsonl",
    )
    parser.add_argument("--vectors", type=Path, default=VECTORS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="base")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--assistant-model", default="gpt-5-mini")
    parser.add_argument("--max-turns", type=int, default=15)
    parser.add_argument("--user-max-tokens", type=int, default=512)
    parser.add_argument("--assistant-max-tokens", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    output_path = args.output or run_directory(args.condition, args.alpha) / "conversations.jsonl"
    root = Path(__file__).parent / "prompts"
    prefix = "base" if args.condition == "base" else "user_profile"
    initial_template = read_prompt(root / f"{prefix}_initial.txt")
    next_template = read_prompt(root / f"{prefix}_next.txt")
    assistant_template = read_prompt(root / "assistant.txt")
    user_model = Qwen35(device=args.device)

    def simulations():
        for source in read_jsonl(args.input):
            problem = source["annotation"]["question"]
            profile_name = "interaction_style" if args.condition == "base" else args.condition
            profile = source["profiles"][profile_name]
            system = render_prompt(assistant_template, math_problem=problem)
            history = []
            for turn in range(args.max_turns):
                user_prompt = render_prompt(
                    initial_template if turn == 0 else next_template,
                    user_profile=profile,
                    math_problem=problem,
                    conversation_history=dialogue_text(history) or "(empty)",
                    length_control=source["length_control"],
                )
                generated = user_model.generate(
                    [{"role": "user", "content": user_prompt}], vector_root=args.vectors,
                    layer=args.layer, alpha=args.alpha, max_new_tokens=args.user_max_tokens,
                )
                message = visible_message(generated)
                if message.lower().rstrip(".!") == "terminate conversation":
                    break
                history.append({"role": "user", "content": message, "generation_prompt": user_prompt})
                tutor_input = [{"role": "system", "content": system}]
                tutor_input.extend({"role": item["role"], "content": item["content"]} for item in history)
                reply = call_text(
                    tutor_input, model=args.assistant_model, max_output_tokens=args.assistant_max_tokens
                )
                history.append({"role": "assistant", "content": reply})
            yield {
                "sample_id": source["sample_id"], "condition": args.condition,
                "alpha": args.alpha, "layer": args.layer, "annotation": source["annotation"],
                "user_profile": profile, "user_profile_available": source["profile_available"][profile_name],
                "human_conversation": source["human_conversation"], "conversation": history,
            }
            print(f"completed {source['sample_id']}")

    write_jsonl(output_path, simulations())


if __name__ == "__main__":
    main()

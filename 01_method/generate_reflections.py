#!/usr/bin/env python3
"""Generate paired user- and assistant-perspective reflections with Qwen 3.5."""

from __future__ import annotations

import argparse
from pathlib import Path

from model import Qwen35
from utils import ROOT, read_jsonl, read_prompt, render_prompt, write_jsonl


ROLE_INTROS = {
    "transcript_analysis": "You are analyzing the following dialogue from the perspective of the {role}.",
    "role_simulation": "You are simulating the role of the {role} in the following dialogue.",
    "dialogue_participant": "You are the {role} in the following dialogue.",
}


def reflection_dialogue(messages: list[dict[str, str]]) -> str:
    return "\n".join(
        f"{'User' if message['role'] == 'user' else 'Assistant'}: {message['content'].strip()}"
        for message in messages
        if message.get("content", "").strip()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "00_sampling/output/dialogues.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "01_method/output/reflections.jsonl")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = read_prompt(Path(__file__).parent / "prompts/reflection.txt")
    model = Qwen35(device=args.device)
    output = []
    for row in read_jsonl(args.input):
        dialogue = reflection_dialogue(row["conversation"])
        reflections = []
        for variant, intro in ROLE_INTROS.items():
            for role in ("user", "assistant"):
                prompt = render_prompt(
                    template,
                    role_intro=intro.format(role=role),
                    dialogue=dialogue,
                    role=role,
                )
                generation = model.generate(
                    [{"role": "user", "content": prompt}],
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                )
                reflections.append(
                    {
                        "role": role,
                        "prompt_variant": variant,
                        "prompt": prompt,
                        "answer": generation,
                    }
                )
        output.append({**row, "dialogue_text": dialogue, "reflections": reflections})
        print(f"generated reflections for {row['id']}")
    write_jsonl(args.output, output)


if __name__ == "__main__":
    main()

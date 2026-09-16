#!/usr/bin/env python3
"""Load native math annotations and profiles, restoring problem text from MATH if needed."""

from __future__ import annotations

import argparse
from pathlib import Path
import random

from utils import ROOT, find_profile, read_json, write_jsonl


def format_profile(profile) -> str:
    if profile is None:
        return "No user profile is available."
    if isinstance(profile, str):
        return profile
    if isinstance(profile, dict):
        return "\n".join(f"- {key}: {value}" for key, value in profile.items())
    lines = []
    for item in profile:
        if isinstance(item, dict):
            name = (
                item.get("Feature Name") or item.get("Preference Name")
                or item.get("Concept Name") or "Profile item"
            )
            answer = (
                item.get("Feature Question Answer") or item.get("Preference Question Answer")
                or item.get("Status") or item
            )
            lines.append(f"- {name}: {answer}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data/SimulatorArena/data")
    parser.add_argument("--math-root", type=Path, default=ROOT / "data/MATH")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/03_simulator_arena/output/dataset.jsonl",
    )
    parser.add_argument("--count", type=int, help="Optional smaller sample; otherwise use all annotations.")
    parser.add_argument("--seed", type=int, default=2)
    args = parser.parse_args()
    annotation_path = args.source / "math_tutoring_annotations.json"
    if not annotation_path.exists():
        annotation_path = args.source / "math_tutoring_annotations_redacted.json"
    annotations = read_json(annotation_path)
    if args.count is not None:
        annotations = random.Random(args.seed).sample(annotations, args.count)
    profile_root = args.source / "user_simulator_profiles/math_tutoring"
    profiles = {
        name: read_json(profile_root / f"{name}.json") for name in ("interaction_style", "writing_style")
    }
    full_path = profile_root / "full_profile.json"
    knowledge_path = profile_root / "knowledge_state.json"
    profiles["full_profile"] = read_json(full_path) if full_path.exists() else {}
    knowledge = read_json(knowledge_path) if knowledge_path.exists() else {}
    output = []
    for raw in annotations:
        annotation = dict(raw)
        if annotation_path.name.endswith("_redacted.json"):
            original = read_json(args.math_root / annotation["question_location"])
            similar = read_json(args.math_root / annotation["similar_question_location"])
            annotation.update(
                question=original["problem"], solution=original["solution"],
                similar_question=similar["problem"], problem_2_gold_solution=similar["solution"],
            )
        selected = {name: find_profile(annotation, mapping) for name, mapping in profiles.items()}
        if not full_path.exists():
            components = [
                selected["interaction_style"], selected["writing_style"], find_profile(annotation, knowledge)
            ]
            selected["full_profile"] = (
                "\n\n".join(format_profile(item) for item in components)
                if all(item is not None for item in components) else None
            )
        conversation = []
        for index, query in enumerate(annotation["user_queries"]):
            conversation.append({"role": "user", "content": query})
            if index < len(annotation["ai_responses"]):
                conversation.append({"role": "assistant", "content": annotation["ai_responses"][index]})
        queries = annotation["user_queries"][:int(annotation["problem_1_turns"])]
        lengths = [len(query.split()) for query in queries if query.strip()]
        low = max(1, min(lengths) // 5 * 5)
        high = (max(lengths) + 4) // 5 * 5
        output.append({
            "sample_id": f"{annotation['problem_id']}::{annotation['workerId']}::{annotation['user_id']}",
            "annotation": annotation,
            "human_conversation": conversation,
            "profiles": {name: format_profile(value) for name, value in selected.items()},
            "profile_available": {name: value is not None for name, value in selected.items()},
            "length_control": f"between {low} and {high} words",
        })
    write_jsonl(args.output, output)
    print(f"wrote {len(output)} examples to {args.output}")


if __name__ == "__main__":
    main()

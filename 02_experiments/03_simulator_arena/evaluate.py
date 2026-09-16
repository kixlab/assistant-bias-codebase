#!/usr/bin/env python3
"""Evaluate writing and interaction similarity to the human dialogue."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import (
    CONDITIONS,
    ROOT,
    call_json,
    dialogue_text,
    feature_names_from_flat_profile,
    find_profile,
    format_feature_questions,
    read_json,
    read_jsonl,
    read_prompt,
    render_prompt,
    run_directory,
    user_messages_text,
    write_jsonl,
)


def human_problem_conversation(row: dict) -> list[dict]:
    """Limit the reference dialogue to the annotated first math problem."""
    conversation = list(row["human_conversation"])
    problem_turns = (row.get("annotation") or {}).get("problem_1_turns")
    if problem_turns is None:
        return conversation

    result = []
    user_turns = 0
    for message in conversation:
        if message.get("role") == "user":
            if user_turns >= int(problem_turns):
                break
            user_turns += 1
        result.append(message)
    return result


def truncate_after_user_turn(conversation: list[dict], ending_user_turn: int) -> list[dict]:
    """Keep messages through the assistant reply to the selected user turn."""
    if ending_user_turn <= 0:
        return []
    result = []
    user_turns = 0
    target_reached = False
    for message in conversation:
        if message.get("role") == "user":
            if target_reached:
                break
            user_turns += 1
            target_reached = user_turns == ending_user_turn
        result.append(message)
        if target_reached and message.get("role") == "assistant":
            break
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--condition", choices=CONDITIONS, default="base")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument(
        "--terminations",
        type=Path,
        help="Optional termination JSONL used to truncate simulated dialogues in memory.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--raw", action="store_true", help="Evaluate without applying termination predictions."
    )
    parser.add_argument("--source", type=Path, default=ROOT / "data/SimulatorArena/data")
    parser.add_argument("--judge-model", default="gpt-5-mini")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be positive")
    input_path = args.input or run_directory(args.condition, args.alpha) / "conversations.jsonl"
    rows = read_jsonl(input_path)
    termination_path = args.terminations or input_path.parent / "terminations.jsonl"
    terminations = {}
    if not args.raw and (args.terminations or termination_path.exists()):
        terminations = {
            str(row["sample_id"]): int(row["ending_user_turn"])
            for row in read_jsonl(termination_path)
        }
        missing = sorted(str(row["sample_id"]) for row in rows if str(row["sample_id"]) not in terminations)
        if missing:
            raise ValueError(f"Missing termination predictions for {len(missing)} samples")
    root = Path(__file__).parent / "prompts"
    templates = {
        "writing": read_prompt(root / "evaluate_writing.txt"),
        "interaction": read_prompt(root / "evaluate_interaction.txt"),
    }
    profile_root = args.source / "user_simulator_profiles/math_tutoring"
    profiles = {}
    for mode in templates:
        path = profile_root / f"{mode}_style.json"
        profiles[mode] = read_json(path) if path.exists() else {}

    def evaluate(row: dict, mode: str, trial: int) -> dict:
        annotation = row.get("annotation") or {}
        profile = find_profile(annotation, profiles[mode])
        features = (
            format_feature_questions(profile)
            or feature_names_from_flat_profile(row.get("user_profile"))
            or "No feature list is available."
        )
        human_conversation = human_problem_conversation(row)
        simulated_conversation = row["conversation"]
        if terminations:
            simulated_conversation = truncate_after_user_turn(
                simulated_conversation,
                terminations[str(row["sample_id"])],
            )
        values = {
            "task": "math tutoring",
            "document_type": "math tutoring",
            "intent": annotation.get("intent") or annotation.get("question") or "",
            "features": features,
            "real_conversation": dialogue_text(human_conversation),
            "simulated_conversation": dialogue_text(simulated_conversation),
            "real_user_queries": user_messages_text(human_conversation),
            "simulated_queries": user_messages_text(simulated_conversation),
        }
        result = call_json(render_prompt(templates[mode], **values), model=args.judge_model)
        score = int(result["similarity_score"])
        if score not in range(1, 6):
            raise ValueError(f"Invalid similarity score: {score}")
        return {
            "sample_id": row["sample_id"],
            "alpha": row["alpha"],
            "condition": row["condition"],
            "user_profile_available": row["user_profile_available"],
            "mode": mode,
            "trial": trial,
            "score": score,
        }

    output = []
    for row in rows:
        for mode in templates:
            for trial in range(1, args.trials + 1):
                output.append(evaluate(row, mode, trial))
    filename = "evaluations_raw.jsonl" if args.raw else "evaluations.jsonl"
    write_jsonl(args.output or input_path.parent / filename, output)


if __name__ == "__main__":
    main()

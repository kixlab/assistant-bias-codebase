#!/usr/bin/env python3
"""Extract trait directions and compare them with the user-assistant role axis."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from model import Qwen35, load_role_vectors
from utils import ROOT, read_jsonl, write_jsonl


def unit(value: torch.Tensor) -> torch.Tensor:
    return value.float() / torch.linalg.vector_norm(value.float()).clamp_min(1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/evaluations.jsonl",
    )
    parser.add_argument("--role-vectors", type=Path, default=ROOT / "01_method/output/role_vectors")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/vectors",
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "02_experiments/02_trait_analysis/output/alignment.jsonl",
    )
    parser.add_argument("--layer", type=int, default=11)
    parser.add_argument("--pooling", choices=("first_token", "full"), default="full")
    parser.add_argument("--score-gap", type=float, default=50.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    rows = read_jsonl(args.input)
    by_key = {(row["trait"], row["pair_id"], row["question_id"], row["polarity"]): row for row in rows}
    selected = []
    pair_keys = sorted({(trait, pair_id, question_id) for trait, pair_id, question_id, _ in by_key})
    for trait, pair_id, question_id in pair_keys:
        pos = by_key.get((trait, pair_id, question_id, "pos"))
        neg = by_key.get((trait, pair_id, question_id, "neg"))
        if not pos or not neg or pos["trait_score"] is None or neg["trait_score"] is None:
            continue
        if float(pos["trait_score"]) - float(neg["trait_score"]) >= args.score_gap:
            selected.append((pos, neg))

    model = Qwen35(device=args.device)
    vectors: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: {"positive": [], "negative": []})
    for pos, neg in selected:
        for label, row in (("positive", pos), ("negative", neg)):
            hidden = model.response_hidden(
                [
                    {"role": "system", "content": row["instruction"]},
                    {"role": "user", "content": row["question"]},
                ],
                row["response"],
                layers=[args.layer],
                pooling=args.pooling,
            )[args.layer]
            vectors[row["trait"]][label].append(hidden)

    user, assistant = load_role_vectors(args.role_vectors, args.layer)
    role_axis = unit(user - assistant)
    summary = []
    args.output.mkdir(parents=True, exist_ok=True)
    for trait, values in sorted(vectors.items()):
        positive = torch.stack(values["positive"]).mean(dim=0)
        negative = torch.stack(values["negative"]).mean(dim=0)
        direction = positive - negative
        trait_root = args.output / trait / f"layer_{args.layer}"
        trait_root.mkdir(parents=True, exist_ok=True)
        torch.save(positive, trait_root / "positive.pt")
        torch.save(negative, trait_root / "negative.pt")
        torch.save(direction, trait_root / "direction.pt")
        summary.append(
            {
                "trait": trait,
                "layer": args.layer,
                "selected_pairs": len(values["positive"]),
                "cosine_to_user_direction": float(unit(direction) @ role_axis),
            }
        )
    write_jsonl(args.summary, summary)


if __name__ == "__main__":
    main()

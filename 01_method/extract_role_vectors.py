#!/usr/bin/env python3
"""Extract per-layer user and assistant centroids from validated reflections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from model import Qwen35
from utils import ROOT, read_jsonl


USABLE_LABELS = {"weakly_represented", "strongly_represented"}


def parse_layers(value: str) -> list[int]:
    layers = []
    for part in value.split(","):
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "01_method/output/reflections_validated.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "01_method/output/role_vectors")
    parser.add_argument("--layers", default="1-31")
    parser.add_argument("--pooling", choices=("first_token", "full"), default="first_token")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layers = parse_layers(args.layers)
    sums = {layer: {role: None for role in ("user", "assistant")} for layer in layers}
    counts = {layer: {role: 0 for role in ("user", "assistant")} for layer in layers}
    model = Qwen35(device=args.device)

    for row in read_jsonl(args.input):
        by_variant = {
            (reflection["prompt_variant"], reflection["role"]): reflection
            for reflection in row["reflections"]
        }
        variants = sorted({variant for variant, _ in by_variant})
        for variant in variants:
            pair = [by_variant.get((variant, role)) for role in ("user", "assistant")]
            if any(
                reflection is None or reflection.get("representation") not in USABLE_LABELS
                for reflection in pair
            ):
                continue
            for reflection in pair:
                assert reflection is not None
                role = reflection["role"]
                hidden = model.response_hidden(
                    [{"role": "user", "content": reflection["prompt"]}],
                    reflection["answer"],
                    layers=layers,
                    pooling=args.pooling,
                )
                for layer, vector in hidden.items():
                    sums[layer][role] = (
                        vector if sums[layer][role] is None else sums[layer][role] + vector
                    )
                    counts[layer][role] += 1

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"model": "Qwen/Qwen3.5-9B", "pooling": args.pooling, "layers": {}}
    for layer in layers:
        layer_root = args.output / f"layer_{layer}"
        layer_root.mkdir(parents=True, exist_ok=True)
        for role in ("user", "assistant"):
            if counts[layer][role] == 0:
                raise RuntimeError(f"No usable {role} reflections for layer {layer}")
            torch.save(sums[layer][role] / counts[layer][role], layer_root / f"{role}.pt")
        summary["layers"][str(layer)] = counts[layer]
    (args.output / "metadata.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote role centroids for {len(layers)} layers to {args.output}")


if __name__ == "__main__":
    main()

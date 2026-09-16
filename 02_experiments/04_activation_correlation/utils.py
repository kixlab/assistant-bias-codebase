from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Callable, Iterable


ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "01_method/output/role_vectors"

CONDITIONS = ("base", "interaction_style", "writing_style", "full_profile")


def run_directory(condition: str) -> Path:
    return ROOT / "02_experiments/03_simulator_arena/output" / condition / "a00"


def activation_directory(condition: str) -> Path:
    return ROOT / "02_experiments/04_activation_correlation/output" / condition / "a00"


def truncate_conversation(messages: list[dict], ending_user_turn: int) -> list[dict]:
    if ending_user_turn <= 0:
        return []
    result, turns = [], 0
    for message in messages:
        if message["role"] == "user":
            turns += 1
            if turns > ending_user_turn:
                break
        result.append(message)
    return result


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def correlation_rows(
    rows: list[dict],
    *,
    group_key: Callable[[dict], str],
    x_key: str,
    y_key: str,
) -> list[dict]:
    import numpy as np
    from scipy.stats import pearsonr, spearmanr

    groups = defaultdict(list)
    for row in rows:
        if row.get(x_key) is not None and row.get(y_key) is not None:
            groups[group_key(row)].append(row)

    output = []
    for group, items in sorted(groups.items()):
        x = np.asarray([float(item[x_key]) for item in items])
        y = np.asarray([float(item[y_key]) for item in items])
        if len(items) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
            pearson = spearman = pearson_p = spearman_p = None
        else:
            pearson_result = pearsonr(x, y)
            spearman_result = spearmanr(x, y)
            pearson = float(pearson_result.statistic)
            pearson_p = float(pearson_result.pvalue)
            spearman = float(spearman_result.statistic)
            spearman_p = float(spearman_result.pvalue)
        output.append(
            {
                "group": group,
                "n": len(items),
                "pearson_r": pearson,
                "pearson_p": pearson_p,
                "spearman_rho": spearman,
                "spearman_p": spearman_p,
            }
        )
    return output

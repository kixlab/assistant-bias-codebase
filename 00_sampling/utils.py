from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def render_prompt(template: str, **values) -> str:
    return template.format_map(values)


def call_text(
    prompt: str | list[dict[str, str]],
    *,
    model: str = "gpt-5-mini",
    max_output_tokens: int = 4096,
) -> str:
    from openai import OpenAI

    response = OpenAI().responses.create(
        model=model,
        input=prompt,
        max_output_tokens=max_output_tokens,
        reasoning={"effort": "minimal"},
        store=False,
    )
    text = str(response.output_text or "").strip()
    if not text:
        raise RuntimeError(f"Model returned no text (status={response.status})")
    return text

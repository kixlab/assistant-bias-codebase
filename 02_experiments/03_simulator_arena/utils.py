from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "01_method/output/role_vectors"

CONDITIONS = ("base", "interaction_style", "writing_style", "full_profile")


def run_directory(condition: str, alpha: float) -> Path:
    return ROOT / "02_experiments/03_simulator_arena/output" / condition / f"a{round(alpha * 10):02d}"


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_profile(annotation: dict, profiles: dict):
    key = f"({annotation['problem_id']}, '{annotation['workerId']}', '{annotation['model']}')"
    if key in profiles:
        return profiles[key]
    for key, profile in profiles.items():
        if str(annotation["workerId"]) in key and str(annotation["model"]) in key:
            return profile
    return None


def feature_names_from_flat_profile(profile) -> str:
    if not profile:
        return ""
    parts = []
    for raw_line in str(profile).splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        name = line.split(":", 1)[0].strip()
        if name:
            parts.append(f"- {name}")
    return "\n".join(parts)


def feature_name(item: dict) -> str:
    return str(
        item.get("Feature Name") or item.get("Preference Name") or item.get("Concept Name") or "Profile item"
    )


def feature_question(item: dict) -> str:
    return str(
        item.get("Feature Question") or item.get("Preference Question") or item.get("Concept Question") or ""
    ).strip()


def format_feature_questions(profile) -> str:
    if profile is None:
        return ""
    if isinstance(profile, str):
        return feature_names_from_flat_profile(profile)
    if isinstance(profile, dict):
        parts = []
        for key, value in profile.items():
            question = feature_question(value) if isinstance(value, dict) else ""
            parts.append(f"- {key}: {question}" if question else f"- {key}")
        return "\n".join(parts)
    if isinstance(profile, list):
        parts = []
        for item in profile:
            if isinstance(item, dict):
                name, question = feature_name(item), feature_question(item)
                parts.append(f"- {name}: {question}" if question else f"- {name}")
            else:
                parts.append(f"- {item}")
        return "\n".join(parts)
    return str(profile)


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


def dialogue_text(messages: list[dict]) -> str:
    return "\n\n".join(
        f"{str(message['role']).upper()}: {str(message['content']).strip()}"
        for message in messages
        if str(message.get("content") or "").strip()
    )


def user_messages_text(messages: list[dict], *, numbered: bool = False) -> str:
    values = [str(message["content"]).strip() for message in messages if message.get("role") == "user"]
    if numbered:
        return "\n\n".join(f"Turn {index}: {text}" for index, text in enumerate(values, start=1))
    return "\n\n".join(values)


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


def call_json(prompt: str, **kwargs) -> dict:
    raw = call_text(prompt, **kwargs)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Model did not return a JSON object: {raw[:200]!r}")
    return json.loads(raw[start : end + 1])

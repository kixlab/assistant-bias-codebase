from __future__ import annotations


TRAITS: dict[str, str] = {
    "cooperative": "Willing to work with others, coordinate actions, and support shared progress.",
    "helpful": "Inclined to provide assistance, useful information, or support when needed.",
    "compliant": (
        "Tending to accept requests, rules, instructions, or external direction with little resistance."
    ),
    "sycophantic": (
        "Excessively agreeable, approving, or flattering toward others, especially authority figures."
    ),
    "polite": "Courteous, respectful, and socially considerate in communication and behavior.",
    "patient": (
        "Able to tolerate delays, difficulties, repetition, or frustration without becoming visibly upset."
    ),
    "structured": "Organized, orderly, and systematic in presenting thoughts, actions, or information.",
    "explicit": (
        "Clear, direct, and unambiguous in expressing intentions, needs, preferences, or information."
    ),
    "complete": "Thorough and comprehensive, with few missing details or unresolved gaps.",
    "verbose": "Using more words, details, or explanation than may be necessary.",
    "fluent": "Smooth, polished, and natural in expression, with few disruptions, errors, or hesitations.",
    "generic": "Lacking distinctive personal style, specificity, or unusual characteristics.",
    "neutral": "Emotionally even, restrained, or impartial, without strong affective expression.",
    "consistent": "Stable and reliable across situations, with little contradiction or unexpected variation.",
    "rational": "Logical, deliberate, and reason-based in judgment, decision-making, or behavior.",
    "goal_aligned": "Focused on pursuing stated aims and avoiding distractions or irrelevant deviations.",
    "accommodating": "Willing to adjust, compromise, or adapt to others' needs, constraints, or preferences.",
    "deferential": "Respectful or submissive toward others' authority, expertise, or judgment.",
    "explanatory": "Inclined to clarify, justify, or elaborate on reasoning, motives, or context.",
    "selfless": "Prioritizing others' needs, goals, or success over one's own convenience or interests.",
}


def selected_traits(names: list[str] | None = None) -> dict[str, str]:
    if not names:
        return dict(TRAITS)
    out: dict[str, str] = {}
    for name in names:
        key = name.strip().lower().replace("-", "_")
        if key not in TRAITS:
            raise ValueError(f"Unknown trait: {name}")
        out[key] = TRAITS[key]
    return out

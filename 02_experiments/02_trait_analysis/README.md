# Trait Analysis

Prerequisites: [role vectors](../../01_method/README.md) and `OPENAI_API_KEY`. Run from the repository root.

```bash
uv run python 02_experiments/02_trait_analysis/prepare.py
uv run python 02_experiments/02_trait_analysis/run.py
uv run python 02_experiments/02_trait_analysis/evaluate.py
uv run python 02_experiments/02_trait_analysis/extract_and_compare.py
```

Preparation uses the original controlled-data prompt to generate five positive/negative instruction pairs, 40 questions, and a trait-specific judge prompt for each of 20 assistant-related traits.

Qwen answers each question under both instructions. The API judge applies the generated trait-specific rubric and returns a score from 0 to 100, or REFUSAL.

Extraction keeps matched responses whose positive-minus-negative score is at least 50. It averages response-token activations, builds a positive-minus-negative trait vector at layer 11, and reports cosine similarity with the user role direction.

## Output

`02_experiments/02_trait_analysis/output/{trait_data,responses,evaluations,alignment}.jsonl`, plus tensors under `02_experiments/02_trait_analysis/output/vectors/<trait>/layer_11/`. A positive cosine means alignment with the user direction; negative means alignment with its opposite.

To prepare fewer traits, repeat `--trait`, for example `prepare.py --trait cooperative --trait verbose`.

## Note

This experiment was inspired by [Assistant Axis](https://github.com/safety-research/assistant-axis).

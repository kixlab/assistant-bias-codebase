# Layer Selection

Prerequisites: [role vectors](../../01_method/README.md) and `OPENAI_API_KEY`. Run from the repository root.

```bash
uv run python 02_experiments/00_layer_selection/prepare.py
uv run python 02_experiments/00_layer_selection/run.py
uv run python 02_experiments/00_layer_selection/evaluate.py
```

Preparation generates 100 unique task goals with the original goal prompt. No external goal file is required. Keep the generated `goals.jsonl` fixed when comparing conditions.

For simplicity, this workflow uses GPT-5-mini for goal generation; the original experiment used Gemini 3 Flash Preview. Newly generated goals therefore differ from the original set.

Generation uses Qwen and alphas 0.0, 0.1, 0.2, 0.3 with all-token steering. The default sweep visits layers 1, 5, 9, 13, 17, 21, 25, 29. To examine every extracted layer:

```bash
uv run python 02_experiments/00_layer_selection/run.py --layers "$(seq -s, 1 31)"
```

The original request-style rubric scores brevity, informality, and information pacing on 1-5 scales. Higher means more user-like. Evaluation summarizes each layer/alpha.

## Output

`02_experiments/00_layer_selection/output/{goals,requests,evaluations,summary}.jsonl`. All inputs and outputs connect by default. Select a layer using `summary.jsonl`; downstream experiments default to layer 11 but accept `--layer`.

# Role Vector Extraction

Run from the repository root after [dialogue sampling](../00_sampling/README.md).

```bash
uv run python 01_method/generate_reflections.py
uv run python 01_method/validate_reflections.py
uv run python 01_method/extract_role_vectors.py
```

1. Qwen generates three user-perspective and three assistant-perspective reflections per sampled dialogue, using the original role prompts.
2. The API judge labels each reflection as strongly, weakly, or not represented. Set `OPENAI_API_KEY`.
3. For each prompt variant, extraction keeps the pair only when both roles are strongly or weakly represented. It teacher-forces the reflections through Qwen and averages each role's first response-token hidden state at layers 1-31.

GPU stages are generation and extraction. For example, prefix a command with `CUDA_VISIBLE_DEVICES=0`.

## Files

Inputs default to `00_sampling/output/dialogues.jsonl`. Intermediate reflections are under `01_method/output/`; centroids are written to:

```text
01_method/output/role_vectors/
  metadata.json
  layer_1/user.pt
  layer_1/assistant.pt
  ...
  layer_31/user.pt
  layer_31/assistant.pt
```

The steering axis is `unit(user - assistant)`. The first response token means the hidden state at the first decoded content token. Use `--pooling full` to extract a mean over all reflection response tokens instead.

# Dialogue Sampling

Run from the repository root after `uv sync --locked`.

## Source

Place the native [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) train parquet files under `data/LMSYS-Chat-1M/`. Subdirectories are scanned recursively.

After accepting its Hugging Face agreement, download directly to that location:

```bash
export HF_TOKEN=your_token
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('lmsys/lmsys-chat-1m', repo_type='dataset', local_dir='data/LMSYS-Chat-1M', allow_patterns=['data/train*.parquet'])"
```

## Run

```bash
uv run python 00_sampling/run.py filter
uv run python 00_sampling/run.py label
uv run python 00_sampling/run.py sample
```

Filtering retains nonempty user/assistant conversations with 2-50 messages and reservoir-samples 100,000 examples with seed 42. Labeling uses `OPENAI_API_KEY` and the original 24-topic prompt, with the last ten messages as context. Sampling draws 30 per topic and fails explicitly if any topic has too few examples.

## Output

`00_sampling/output/{filtered,labeled,dialogues}.jsonl`. The final dialogue file feeds reflection generation automatically. Use `filter --count` and `sample --per-label` to change the sample size; reducing the reservoir may leave rare topics undersampled.

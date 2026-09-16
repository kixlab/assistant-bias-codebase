# Turn Termination

Prerequisites: [role vectors](../../01_method/README.md), a CUDA GPU, and `OPENAI_API_KEY`. Run from the repository root.

## Source

Place native [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) train parquet files under `data/WildChat/`:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('allenai/WildChat-1M', repo_type='dataset', local_dir='data/WildChat', allow_patterns=['data/train*.parquet'])"
```

## Run

```bash
uv run python 02_experiments/01_turn_termination/prepare.py
uv run python 02_experiments/01_turn_termination/run.py
uv run python 02_experiments/01_turn_termination/analyze.py
```

Preparation keeps alternating eight-turn dialogues with at most 1,000 words per message. It codes disengagement, excludes Unknown and unsupported labels, and samples up to 100 per each of eight types. It then annotates user intent with `prompts/intent.txt`.

Coding is cached in `coded.jsonl` and reused on rerun. The preparation automatically uses its `results` list rather than recoding.

Prediction uses the original no-reason prompt without profile information: user intent, the dialogue prefix, and two choices. The model returns `A` or `B`. It tests every dialogue prefix in two balanced A/B option orders, at layer 11 and alphas 0.0-0.3. A prediction of 0 means no disengagement was detected.

## Output

`02_experiments/01_turn_termination/output/{coded,dataset,predictions,summary}.jsonl`. The summary reports signed/absolute turn error, exact prediction rate, and option-order agreement.

# Guide

Codebase for reproducing the experiments in [Investigating Assistant Bias in LLM User Simulators Using a Role Vector](https://arxiv.org/abs/2609.00608) (EMNLP 2026 Findings).

LLMs asked to act as users often retain assistant-like habits: they are unusually cooperative, detailed, and persistent. We extract a **user role direction** from Qwen's user- and assistant-perspective reflections, then measure activation along that direction and apply steering in user simulations to examine how it relates to this bias.

## Setup

Install [uv](https://docs.astral.sh/uv/) and run these commands from this repository's root:

```bash
uv sync --locked
cp .env.example .env
```

Fill in `.env`, then load it into your shell before running scripts:

```bash
set -a
source .env
set +a
```

`.env` is ignored by Git. `OPENAI_API_KEY` is required for API stages; `HF_TOKEN` is optional for gated dataset downloads.

The local model is **Qwen/Qwen3.5-9B**, loaded through TransformerLens with thinking disabled and deterministic decoding. API stages use `gpt-5-mini` by default and incur charges. Judge outputs can vary between trials; this variability is reported in the paper's appendix.

## Dataset

Keep the original dataset files and directory names.

```text
data/
  LMSYS-Chat-1M/data/train-*.parquet
  WildChat/data/train-*.parquet
  SimulatorArena/data/
    math_tutoring_annotations_redacted.json
    user_simulator_profiles/math_tutoring/
      interaction_style.json
      writing_style.json
      knowledge_state.json
  MATH/
    train/<subject>/*.json
    test/<subject>/*.json
```

Use the datasets needed for your chosen experiments:

- [LMSYS-Chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m): reflection dialogues. Accept the dataset agreement before downloading.
- [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M): eight-turn conversations for disengagement prediction.
- [SimulatorArena](https://github.com/microsoft/SimulatorArena): math-tutoring annotations and profiles.
- MATH: obtain separately as described in [SimulatorArena's data instructions](https://github.com/microsoft/SimulatorArena/blob/main/data/README.md). Preparation automatically restores the required problem fields.

Downloaded data stays under `data/`. Each sampling, method, or experiment folder writes generated files into its own `output/` directory. Scripts resolve default paths relative to the repository root.

## Step 1: Sample Reflection Dialogues

After placing LMSYS parquet files in the location above:

```bash
uv run python 00_sampling/run.py filter
uv run python 00_sampling/run.py label
uv run python 00_sampling/run.py sample
```

Defaults: filter 100,000 valid dialogues, then sample 30 dialogues for each of 24 topics. See [sampling](00_sampling/README.md).

## Step 2: Extract Role Vectors

```bash
uv run python 01_method/generate_reflections.py
uv run python 01_method/validate_reflections.py
uv run python 01_method/extract_role_vectors.py
```

These scripts consume Step 1's output automatically and write `01_method/output/role_vectors/layer_N/{user,assistant}.pt`. See [method](01_method/README.md). All experiments default to these vectors.

## Step 3: Choose An Experiment

Each guide provides the commands for preparation, generation, evaluation, and analysis.

| Folder                                                                          | Experiment                                                                       |
| ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| [00_layer_selection](02_experiments/00_layer_selection/README.md)               | Generate goals, steer requests across layers, and judge request style            |
| [01_turn_termination](02_experiments/01_turn_termination/README.md)             | Build a balanced WildChat set and predict disengagement                          |
| [02_trait_analysis](02_experiments/02_trait_analysis/README.md)                 | Generate controlled trait data and compare trait vectors with the role direction |
| [03_simulator_arena](02_experiments/03_simulator_arena/README.md)               | Simulate and judge math tutoring across prompts and alphas                       |
| [04_activation_correlation](02_experiments/04_activation_correlation/README.md) | Measure activations and relate them to SimulatorArena judgments                  |

Paths and model settings are available as CLI options; `--help` lists them. Most stages replace their output when rerun, so use a different `--output` path to retain an earlier version. Newly generated goals and API annotations may differ from those used in the original experiment.

## Citation

```bibtex
@misc{jeong2026investigatingassistantbiasllm,
      title={Investigating Assistant Bias in LLM User Simulators Using a Role Vector},
      author={Daeheon Jeong and Yoonjoo Lee and Eugene Choi and Sinie van der Ben and Juho Kim},
      year={2026},
      eprint={2609.00608},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.00608},
}
```

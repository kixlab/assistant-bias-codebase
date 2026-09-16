# SimulatorArena Activation Correlation

Prerequisites: [SimulatorArena runs and evaluations](../03_simulator_arena/README.md) and role vectors from `01_method/`. This analysis uses only unsteered runs.

Run from the repository root:

```bash
for condition in base interaction_style writing_style full_profile; do
  uv run python 02_experiments/04_activation_correlation/measure.py --condition "$condition"
  uv run python 02_experiments/04_activation_correlation/analyze.py --condition "$condition"
done
```

Measurement reads the default SimulatorArena conversation files directly. It replays each saved user-generation prompt and visible response without steering, captures all response-token activations at layer 11, and computes the token-weighted user-direction projection.

The default activation is the uncentered projection onto `unit(user - assistant)`. Use `--centered` during measurement to subtract the user/assistant centroid midpoint.

When `terminations.jsonl` exists, measurement selects the same truncated messages as evaluation. To examine raw dialogues, run SimulatorArena evaluation with `--raw`, then add `--raw` to both commands above.

Analysis averages all judge trials per sample and metric, then reports Pearson r and Spearman rho with p-values for writing, interaction, and combined similarity. Each prompting condition is analyzed separately; there is no pooling across alphas or prompt conditions.

## Output

`02_experiments/04_activation_correlation/output/<condition>/a00/activations.jsonl`, plus `<condition>/correlations.jsonl`.

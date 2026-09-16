# SimulatorArena

Prerequisites: [role vectors](../../01_method/README.md) and `OPENAI_API_KEY`. Run from the repository root.

## Source

Clone the [SimulatorArena repository](https://github.com/microsoft/SimulatorArena) into the predefined location:

```bash
git clone https://github.com/microsoft/SimulatorArena.git data/SimulatorArena
```

Obtain MATH separately and place its `train/` and `test/` directories under `data/MATH/`, following the [SimulatorArena instructions](https://github.com/microsoft/SimulatorArena/blob/main/data/README.md).

Preparation reads the native redacted annotations and profile mappings, restores the problem fields from MATH, and builds the dataset. Full profiles are loaded from `full_profile.json` when supplied; otherwise, they are constructed by combining the interaction, writing, and knowledge profiles.

## Conditions

Select one of four setups with `run.py --condition`:

| Condition           | Profile in the generation prompt            | CoT | Length control |
| ------------------- | ------------------------------------------- | --- | -------------- |
| `base`              | None                                        | No  | No             |
| `interaction_style` | Interaction style only                      | Yes | Yes            |
| `writing_style`     | Writing style only                          | Yes | Yes            |
| `full_profile`      | Interaction, writing, and knowledge profiles | Yes | Yes            |

`base` uses `base_initial.txt` and `base_next.txt`. The other conditions share `user_profile_initial.txt` and `user_profile_next.txt`, with the selected profile inserted into the prompt. These prompts request a thought followed by a query or response; only the query or response is passed to the tutor.

Length control uses a word-count range derived from the reference dialogue's user messages. CoT and length control are fixed by the selected setup: comparing `base` with a profile condition changes the profile, CoT, and length control together.

Every condition accepts `--alpha` to set the steering strength; the commands below use 0.0, 0.1, 0.2, and 0.3. Missing profiles are marked with `user_profile_available: false`. For `base`, the saved `user_profile` and availability field refer to the interaction profile for evaluation bookkeeping.

## Run

```bash
uv run python 02_experiments/03_simulator_arena/prepare.py
for condition in base interaction_style writing_style full_profile; do
  for alpha in 0.0 0.1 0.2 0.3; do
    uv run python 02_experiments/03_simulator_arena/run.py --condition "$condition" --alpha "$alpha"
    uv run python 02_experiments/03_simulator_arena/terminate.py --condition "$condition" --alpha "$alpha"
    uv run python 02_experiments/03_simulator_arena/evaluate.py --condition "$condition" --alpha "$alpha"
  done
done
uv run python 02_experiments/03_simulator_arena/summarize.py
```

Defaults: all 450 math conversations, layer 11, all-token steering, a maximum of 15 user turns, and response limits of 512 tokens for the user and 4,096 tokens for the tutor. `prepare.py --count 50` selects a smaller fixed sample.

The tutor receives the original tutor system instruction and the alternating dialogue messages. Termination detection clamps predictions to the actual dialogue length.

Evaluation automatically applies a neighboring `terminations.jsonl` when present. Writing similarity uses only user utterances; interaction similarity uses the full dialogue. Human reference dialogues stop at the annotated first problem (the original SimulatorArena combines two consecutive problems into one human reference dialogue). Three trials per metric are saved together and averaged to improve score stability. Add `--raw` to evaluation and summarization to report untruncated scores.

Both judges load each example's feature names and questions from the dataset's `writing_style.json` or `interaction_style.json`. Missing profiles fall back to feature names in the saved user profile. Evaluation defaults to `data/SimulatorArena/data/`; use `--source` if your source data is elsewhere.

## Output

Preparation writes `02_experiments/03_simulator_arena/output/dataset.jsonl`. Results are saved under the same `output/` directory in `<condition>/a00/` through `a03/`, each containing `conversations.jsonl`, `terminations.jsonl`, and `evaluations.jsonl`. The combined score table is `02_experiments/03_simulator_arena/output/summary.jsonl`.

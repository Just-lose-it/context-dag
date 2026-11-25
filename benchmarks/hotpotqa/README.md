# HotpotQA distractor pilot

This folder compares four policies on the [HotpotQA distractor dev set](https://hotpotqa.github.io/): one model call with all 10 paragraphs (`single`), two fixed evidence workers plus an answer call (`fixed`), a free `DynamicDAG` with `JSONAgent` (`dynamic`), and a constrained dynamic DAG (`constrained`). The constrained policy reads the DAG, asks the model to divide document titles between two workers, spawns them, and runs a finalizer. All use `deepseek-flash`, temperature 0, non-thinking mode.

## Data

The official download was unavailable in this environment. A public [mirror of the original JSON](https://huggingface.co/datasets/RAGLAB/data/blob/main/eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json) was downloaded to `data/hotpot_dev_distractor_v1.json`; its SHA-256 is `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, matching the checksum in the [HotpotQA dataset metadata](https://huggingface.co/datasets/hotpotqa/hotpot_qa). Data and individual results are ignored by Git.

```powershell
python benchmarks/hotpotqa/run.py --count 3 --policies fixed constrained
```

The script writes one JSON line per question to `results/pilot.jsonl`. It records predictions, exact-match answer/support/joint scores, model calls, tokens, and dynamic DAG events. Scores are computed locally with the [official evaluator's](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py) normalization and set comparison for exact match. This is a small pilot, not a leaderboard submission.

## First three dev questions

| Policy | Answer EM | Support EM | Joint EM | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 2/3 | 1/3 | 1/3 | 3 | 5,043 |
| Fixed two workers | 2/3 | 1/3 | 1/3 | 9 | 7,316 |
| Free dynamic DAG | 2/3 | 0/3 | 0/3 | 3 | 5,401 |

The free dynamic agent returned a direct answer at the root on all three questions; `JSONAgent` treated that as `conclude`. It never called `read_dag` or `spawn`. Thus the first pilot did **not** test dynamic graph construction.

## Constrained dynamic rerun

On the same three questions, both conditions used `max_tokens=1200` per call and the same ten documents. The constrained condition completed `read_dag → spawn → worker yield → finalizer` on each item. Worker evidence titles were converted from the displayed `[title #index]` reference into canonical title/index fields before finalization. The final scorer still uses exact title and index matching.

| Policy | Answer EM | Support EM | Joint EM | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed two workers | 2/3 | 1/3 | 1/3 | 9 | 6,983 |
| Model-planned two workers | 2/3 | 1/3 | 1/3 | 12 | 7,888 |

This verifies the dynamic execution path but provides no evidence of accuracy improvement. The planner sees only question and document titles; it adds a model call while still making two workers read all ten documents in total. Results are exploratory because the same three dev items were used to debug prompts and parsing. One earlier run failed when a 600-token output cap truncated worker JSON; the equal-cap rerun completed all items. Infrastructure/format errors must be reported separately from wrong answers.

## Next controlled comparison

Let the planner choose **whether** to spawn and how many workers to use, based on evidence gaps or expected value rather than titles alone. Run a larger, preselected sample and report results by comparison/bridge type and difficulty, with paired item-level outcomes and total input/output tokens. Match total budgets and add a single-agent retry/sampling baseline. Do not interpret reduced calls alone as a gain if supporting facts are wrong. See [research direction](../../docs/research_direction.md).

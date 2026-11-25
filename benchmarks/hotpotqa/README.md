# HotpotQA distractor pilot

This folder compares five policies on the [HotpotQA distractor dev set](https://hotpotqa.github.io/): one model call with all 10 paragraphs (`single`), two fixed evidence workers plus an answer call (`fixed`), a free `DynamicDAG` with `JSONAgent` (`dynamic`), a model-planned two-worker DAG (`constrained`), and an `adaptive` policy with `read_docs` and an evidence-completeness gate. All use `deepseek-flash`, temperature 0, non-thinking mode.

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

### Adaptive action prototype (diagnostic runs)

`adaptive` first inspects the DAG, then can read up to five named documents per tool call, spawn workers, yield, or conclude. If the root proposes a conclusion with fewer than two distinct cited documents, a rule spawns a verifier. A separate model call first repairs malformed answer structure, so formatting failure is not itself evidence of a missing fact. Invalid document requests are returned as tool observations.

On dev items 6–10, **before** separating formatting from evidence checks, the gate spawned on all five. Joint EM was 2/5 for single, 1/5 for fixed, and 2/5 for adaptive; total calls were 5, 15, and 25. This did not show a gain over single and did not demonstrate selective spawning. On items 11–13, after the change, one run concluded directly, one spawned a verifier, and one stopped on an invalid document request. The invalid request was handled in a targeted retry. These items were used to debug the protocol, so their scores are not an independent estimate. Across runs, answer correctness often exceeded supporting-fact correctness.

### Next controlled comparison

Freeze a larger new sample **before** modifying the controller again. Compare against a single-agent retry/sampling baseline at the same total token cap, report errors separately, and record actions, document reads, support completeness, total tokens, and latency by item. The current 8000-token cap is enforced from returned usage; one final call may overshoot. Do not interpret fewer calls alone as a gain if supporting facts are wrong. See [research direction](../../docs/research_direction.md).

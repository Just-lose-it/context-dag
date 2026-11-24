# HotpotQA distractor pilot

This folder compares three policies on the [HotpotQA distractor dev set](https://hotpotqa.github.io/): one model call with all 10 paragraphs (`single`), two fixed evidence workers plus an answer call (`fixed`), and the current `DynamicDAG` with a `JSONAgent` (`dynamic`). All use `deepseek-flash`, temperature 0, non-thinking mode. The dynamic policy may answer directly or spawn up to two workers.

## Data

The official download was unavailable in this environment. A public [mirror of the original JSON](https://huggingface.co/datasets/RAGLAB/data/blob/main/eval_datasets/HotPotQA/hotpot_dev_distractor_v1.json) was downloaded to `data/hotpot_dev_distractor_v1.json`; its SHA-256 is `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea`, matching the checksum in the [HotpotQA dataset metadata](https://huggingface.co/datasets/hotpotqa/hotpot_qa). Data and individual results are ignored by Git.

```powershell
python benchmarks/hotpotqa/run.py --count 3
```

The script writes one JSON line per question to `results/pilot.jsonl`. It records predictions, exact-match answer/support/joint scores, model calls, tokens, and dynamic DAG events. Scores are computed locally with the [official evaluator's](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py) normalization and set comparison for exact match. This is a small pilot, not a leaderboard submission.

## First three dev questions

| Policy | Answer EM | Support EM | Joint EM | Calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single | 2/3 | 1/3 | 1/3 | 3 | 5,043 |
| Fixed two workers | 2/3 | 1/3 | 1/3 | 9 | 7,316 |
| Free dynamic DAG | 2/3 | 0/3 | 0/3 | 3 | 5,401 |

The dynamic agent returned a direct answer at the root on all three questions; `JSONAgent` treated that as `conclude`. It never called `read_dag` or `spawn`. Thus this pilot does **not** establish whether dynamic graph construction helps. It shows that the current prompt and action interface do not reliably elicit graph use on these questions. The fixed workflow cost more calls without improving joint EM in this tiny sample. Three items cannot support a performance claim.

## Next controlled comparison

Add a constrained dynamic condition that must inspect the DAG and may adaptively choose document partitions and worker count, while retaining the free condition. Run a larger, preselected sample and report results by comparison/bridge type and difficulty, with paired item-level outcomes and total input/output tokens. Keep the same model and document access for every policy. Do not interpret reduced calls alone as a gain if supporting facts are wrong.

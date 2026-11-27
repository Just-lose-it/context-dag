# MuSiQue answerable dev pilot

This is a second, more explicitly multi-hop check of the same `deepseek-flash` policies. The [MuSiQue dataset](https://github.com/stonybrooknlp/musique) supplies 20 paragraphs and 2–4-hop question decompositions. We use the answerable dev JSONL mirrored at [Hugging Face](https://huggingface.co/datasets/voidful/MuSiQue/tree/main); data and individual predictions stay local and are ignored by Git.

`selection.json` was committed before model calls. It selects the first eight IDs by SHA-256 hash separately from answerable 2-hop and 4-hop dev questions with 20 paragraphs. The script never reads gold answers or support flags to select items. Run with:

```powershell
python benchmarks/musique/run.py --group 2hop
python benchmarks/musique/run.py --group 4hop
python benchmarks/musique/summarize.py
```

All policies have a 12,000-token observed-usage cap per question and `max_tokens=1200` per model call. The single policy reads all 20 paragraphs once; fixed uses two 10-paragraph evidence workers and a final answer call; adaptive sees titles, reads up to five paragraphs per action, and may spawn evidence workers. All use the same model, temperature 0, and non-thinking mode. The cap is checked after each response, so a final call can overshoot. The policies do **not** have equal numbers of model calls, and this is not an official leaderboard submission.

We report answer exact match against the supplied answer and aliases, exact paragraph-support set match, joint exact match, and support recall. Paragraph references use `P<idx> <title>` and `#0` for the whole paragraph. The model sometimes emits a bare title, which is mapped to a paragraph only when that title is unique in the item. Ambiguous titles and invalid sentence indices receive no support credit. `summarize.py` recomputes these scores from saved predictions; stored JSONL scores from the first run used a stricter ID-only parser. These scores are local pilot metrics, not the [official MuSiQue evaluator](https://github.com/stonybrooknlp/musique/blob/main/evaluate_v1.0.py)'s answer/support F1.

| Group | Policy | Answer EM | Support EM | Joint EM | Errors | Calls | Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2-hop (8) | Single | 4 | 3 | 1 | 0 | 8 | 20,888 |
| 2-hop (8) | Fixed | 5 | 1 | 1 | 0 | 24 | 26,589 |
| 2-hop (8) | Adaptive | 4 | 3 | 2 | 2 | 49 | 49,000 |
| 4-hop (8) | Single | 4 | 2 | 2 | 0 | 8 | 22,301 |
| 4-hop (8) | Fixed | 4 | 0 | 0 | 0 | 24 | 30,060 |
| 4-hop (8) | Adaptive | 2 | 1 | 1 | 2 | 65 | 70,267 |

All four adaptive errors are observed-usage budget overruns. The apparent one-item joint gain for adaptive on 2-hop comes with two failures and more than twice the token use; the 4-hop result favors single. No reliable benefit is established. This result suggests the current action protocol and evidence handoff lose information or spend too many calls. MuSiQue tests multi-hop QA, but it does not directly test the original routing question of whether a long intermediate artifact should be sent to a specific successor or placed in shared storage for multiple consumers.

## Multi-level spawn diagnostic protocol

`selection_multilevel.json` freezes eight unused 4hop1 items (chain proxy) and eight unused 4hop3 items (branch proxy), with no reused component question ID within each group. The first two per group are development diagnostics; the remaining six are held-out evaluation. The decomposition labels and gold answers are used only for selection grouping and scoring, never passed to a policy.

`multilevel.py` compares four conditions on the same 20 paragraphs, `deepseek-flash`, temperature 0, JSON output, `max_tokens=1200` per call, and a 16,000-token observed-usage cap per question: single call with all paragraphs; single context with one review call; flat two-lead DAG; and a two-lead DAG whose child may spawn up to two grandchildren after inspecting documents. Flat and nested DAGs reuse the same model-generated initial two-lead plan per question; each condition's token accounting includes that plan once. The run records answer/support/joint exact match, calls, tokens, document reads, child spawns, and errors. The current scheduler is sequential, so this pilot tests quality and cost, not parallel speedup.

```powershell
python benchmarks/musique/multilevel.py --group chain_4hop1 --start 0 --limit 2
python benchmarks/musique/multilevel.py --group branch_4hop3 --start 0 --limit 2
```

After diagnostics, run `--start 2 --limit 6` for held-out evaluation. This is a limited proxy for the PPT's unknown-graph scenario: the main question and document titles may already expose some decomposition, while true deep research discovers new sources during execution. A positive result would require actual child spawns and improved held-out joint correctness at the same budget; merely creating more nodes does not count.

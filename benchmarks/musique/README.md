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

`multilevel.py` compares four conditions on the same 20 paragraphs, `deepseek-flash`, temperature 0, JSON output, `max_tokens=1200` per call, and a 24,000-token observed-usage cap per question: single call with all paragraphs; single context with one review call; flat two-lead DAG; and a two-lead DAG whose child may spawn up to two grandchildren when its lead contains distinct facts. Flat and nested DAGs reuse the same model-generated initial two-lead plan per question; each condition's token accounting includes that plan once. The run records answer/support/joint exact match, calls, tokens, document reads, child spawns, and errors. The current scheduler is sequential, so this pilot tests quality and cost, not parallel speedup.

```powershell
python benchmarks/musique/multilevel.py --group chain_4hop1 --start 0 --limit 2
python benchmarks/musique/multilevel.py --group branch_4hop3 --start 0 --limit 2
```

After diagnostics, run `--start 2 --limit 6` for held-out evaluation. This is a limited proxy for the PPT's unknown-graph scenario: the main question and document titles may already expose some decomposition, while true deep research discovers new sources during execution. A positive result would require actual child spawns and improved held-out joint correctness at the same budget; merely creating more nodes does not count.

The first diagnostic version allowed optional nested `spawn` but produced zero child spawns on four development items; one branching item exceeded the initial 16,000-token cap. The second diagnostic version gave an explicit conditional-spawn instruction and raised the common cap to 24,000 for all four policies. These development results remain local.

The second diagnostic version also produced zero child spawns on the same four development items. The third version adds an explicit model call at each first-level child to decide whether its lead contains two independent factual unknowns; a positive decision creates two grandchildren. That call is included in the nested policy's token budget. The first two versions remain development diagnostics; held-out items will use only version three.

On the four third-version development items, nested DAG produced child spawns on three items (two chain, one branch), so the multi-level execution path is exercised. It exceeded the 24,000-token cap on the other branch item. Nested had no joint-correct answer, while flat had one; these development items are diagnostic and do not estimate performance. The split decision also fired on chain items, so it does not yet discriminate branching structure well. No further prompt or threshold tuning will be done on the held-out six items per group.

### Held-out result and interruption

The six held-out 4hop3 branch-proxy items completed. `python benchmarks/musique/summarize_multilevel.py` reproduces the aggregate below. Errors count as failures; one model-generated initial plan lacked the required two branches, affecting both DAG conditions.

| Policy | Answer EM | Support EM | Joint EM | Errors | Child spawns | Model calls | Total tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single call | 3/6 | 1/6 | 1/6 | 0 | 0 | 6 | 15,377 |
| Single plus review | 2/6 | 1/6 | 0/6 | 0 | 0 | 12 | 31,183 |
| Flat two-lead DAG | 3/6 | 1/6 | 0/6 | 1 | 0 | 44 | 36,898 |
| Conditional nested DAG | 4/6 | 2/6 | 2/6 | 1 | 8 | 91 | 83,649 |

The nested policy gained one joint-correct item over the single call, but spent 5.4 times as many tokens and made 91 versus 6 model calls. This six-item result cannot establish a reliable benefit. In the two items where nested was jointly correct, the flat DAG had already read all four gold supporting paragraphs. Nested generated more complete final evidence and, on one item, corrected the answer; the trace therefore does **not** show that a new branch discovered an otherwise unread gold document. The split controller also spawned on both chain development items. Current evidence points to evidence integration and extra sampling as possible explanations, not a demonstrated task-structure-aware expansion rule.

The chain-proxy held-out run was interrupted by the DeepSeek API returning `HTTP 402 Payment Required`: the first item reached that error during nested execution, and the other five had no successful model calls. There are zero complete paired chain items, so no chain-versus-branch conclusion is available. The runner now stops on the first 402 and allows those rows to be rerun after the external account state changes; the summarizer excludes payment-blocked rows. Do not count 402 as an algorithmic failure or treat the partial chain outputs as a full comparison.

### Trace audit: did grandchildren find new required evidence?

`python benchmarks/musique/audit_multilevel.py` recomputes document-read coverage from saved `read_docs` events. It ignores requests that the document tool rejected (over five titles or unknown titles). Five of the six branch-proxy evaluation items have completed, paired flat and nested traces; the sixth failed in their shared initial planner.

| Measure across five paired items | Flat DAG | Nested DAG |
| --- | ---: | ---: |
| Gold supporting paragraphs read, out of 20 | 18 | 18 |
| Gold supporting paragraphs cited in final answer, out of 20 | 12 | 17 |
| Joint exact match | 0/5 | 2/5 |

Grandchildren read all 18 gold paragraphs reached by the nested policy; 11 of these were not read by their own first-level parent. **None** was new relative to the flat policy's reads. The nested policy therefore redistributed and repeated evidence work rather than discovering a required document missed by flat DAG. More complete final citation may reflect separate contexts, repeated sampling, different finalizer inputs, or model variability; this audit cannot distinguish those causes. Extra or incorrect final citations still count against support exact match. The high token and call costs in the table above remain part of the result.

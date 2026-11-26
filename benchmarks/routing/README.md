# Artifact fan-out microbenchmark

`python benchmarks/routing/run.py` exercises the existing fixed `ContextDAG` with a source and several successors. A consumer returns the full artifact only if its task needs it. `push` places the full artifact in every successor inbox; `store` writes it once and only consumers that need it call `read`. All required consumers receive identical content.

| Successors | Actual readers | Artifact chars | `push` consumer exposure | `store` consumer exposure | Store reads |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 8,192 | 65,536 | 8,192 | 1 |
| 8 | 4 | 8,192 | 65,536 | 32,768 | 4 |
| 8 | 8 | 8,192 | 65,536 | 65,536 | 8 |
| 2 | 1 | 2,048 | 4,096 | 2,048 | 1 |
| 2 | 2 | 2,048 | 4,096 | 4,096 | 2 |

Exposure is the sum of payload characters in successor inboxes and store read responses. The saved copy in storage, short key references, model tokenization, retrieval latency, and response quality are outside this measure. The result follows the simple relation `push = successors × payload` and `store = actual readers × payload`; it is a mechanism check, not an empirical LLM gain. A store read also adds an action, so when every successor needs the artifact, direct handoff is preferable on this measure.

The fixed executor's plain `push` routes an artifact to *all* outgoing successors. `RoutingDecision("push", targets=...)` now limits direct delivery to named successors. The live experiment below uses both forms.

## Live routing experiment protocol

`live_sample.json` freezes three previously unused answerable MuSiQue four-hop questions as shared-report tasks, plus 21 disjoint first-step questions from previously unused two-hop items as local tasks. Each of three batches has eight consumers: one needs the 20-paragraph shared report, and seven have their own task-specific paragraph. All policies use the same question, report, local passages, `deepseek-flash`, temperature 0, JSON answer format, and 300 output-token cap per consumer. The sample was fixed before model calls.

`python benchmarks/routing/live.py` compares:

- `broadcast`: put the shared report in all eight inboxes.
- `targeted`: directly put it only in the known reader's inbox using `RoutingDecision`.
- `store`: write it to `SharedStore`; the known reader calls `read` once.

The script records each consumer's actual API prompt tokens, total tokens, wall time, exact-match answer score, errors, and store reads. A task's need for the shared report is supplied by the benchmark, so `targeted` and `store` have an oracle reader set. In a deployed workflow the controller would have to infer that set. The store is in process memory and its read does not incur a network or model call. This experiment can measure how avoiding irrelevant context affects actual model usage and answers under these assumptions; it does not establish a general routing policy.

### Frozen 3-batch result

Aggregate with `python benchmarks/routing/summarize_live.py`. There are 24 distinct tasks per policy: three four-hop shared-report questions and 21 local first-step questions. Each report has 7,930–12,808 characters. Each policy made one `deepseek-flash` call per task; there were no API or parsing errors.

| Policy | Correct / 24 | Shared tasks correct / 3 | Prompt tokens | Total tokens | Sequential wall time | Store reads |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Broadcast `push` | 13 | 2 | 58,104 | 58,354 | 12.88 s | 0 |
| Targeted `push` | 13 | 2 | 11,232 | 11,463 | 11.52 s | 0 |
| `store` + read | 12 | 1 | 11,232 | 11,454 | 12.68 s | 3 |

Targeted handoff and shared read both reduced measured prompt tokens by **80.7%** versus broadcast while retaining all task material for the one consumer that needed the report. Their prompt-token totals match exactly. The one-answer score difference between targeted and store occurred on a shared task whose prompt was identical under both policies; repeated `temperature=0` calls can still return different answers. This sample is too small to establish an accuracy or latency difference. Since the reader was known in advance and the store was local memory, targeted handoff used the same model context with no read action. The experiment supports eliminating irrelevant broadcast, while leaving the harder research question—predicting readers when need is uncertain—open.

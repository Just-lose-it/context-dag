# BrowseComp-Plus candidate-pool pilot

This is a small controlled test of evidence-conditioned DAG expansion. The official [BrowseComp-Plus dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) has a fixed corpus of about 100,000 documents and human-labeled evidence. We initially use each selected query's evidence, gold, and hard-negative documents as a local candidate pool. **This is not the official full-corpus retrieval setting**; constructing the pool with benchmark labels makes retrieval easier. The labels are withheld from the agent and used only for evaluation.

`selection.json` freezes eight hash-selected row offsets at the listed dataset revision before reading query content. Two offsets are diagnostic; six are held out. `download.py` fetches the obfuscated rows from the official Hugging Face dataset server and decrypts them locally using the method in the [dataset card](https://huggingface.co/datasets/Tevatron/browsecomp-plus/blob/main/README.md). Decrypted data, model predictions, and traces are ignored by Git and must not be pasted into public reports, following the benchmark's anti-leakage guidance.

The comparisons share `deepseek-flash`, the same deterministic BM25 index over 200-word overlapping chunks, candidate documents, JSON answer format, a 900-output-token per-call cap, and a 32,000-token observed-usage cap per query. The first two diagnostic queries used a 16,000-token cap; most policies exceeded it before answering, so the cap was raised before touching the held-out queries. The policies are one iterative single context, an initial flat two-lead DAG, and a DAG that checks each first-level finding for a specific unsupported factual question before adding up to two grandchildren. Flat and dynamic reuse the same initial model-generated plan per query; its tokens count once in each arm. The root and children can search chunks and read their full text. The gap check uses the child's finding and cited document IDs, not gold labels.

We report strict normalized answer exact match, evidence and gold-document retrieval recall, actual token use, search/read calls, and whether a new branch retrieved a labeled evidence document absent from the flat DAG's reads. Strict exact match is a local pilot measure, not the benchmark's official LLM judge score. A positive result must improve more than the number of spawned nodes; the controller's extra calls and errors count against it. The current scheduler runs nodes sequentially, so this does not test parallel speedup.

```powershell
python benchmarks/browsecomp/download.py
python benchmarks/browsecomp/run.py --split diagnostic
python benchmarks/browsecomp/run.py --split evaluation
python benchmarks/browsecomp/audit.py
```

## Held-out candidate-pool pilot

The six frozen evaluation queries were run once with the 32,000-token cap. These are local strict exact-match results, **not official BrowseComp-Plus scores**:

| Policy | Exact match | Completed | Total observed tokens | Model calls | Searches | Reads | Child expansions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single context | 4/6 | 5/6 | 67,936 | 34 | 16 | 12 | 0 |
| Flat two-lead DAG | 4/6 | 5/6 | 151,743 | 82 | 32 | 26 | 0 |
| Evidence-gap dynamic DAG | 0/6 | 0/6 | 202,684 | 119 | 46 | 31 | 9 |

The single context had one JSON parse error; the flat DAG had one token-budget error. Every dynamic run exceeded the token budget before producing a final answer. The observed total can pass 32,000 because the client checks usage after each model response, not before it. The dynamic grandchildren read a labeled gold document absent from the flat DAG's reads on two of six questions, but neither run completed. Thus the current controller demonstrates occasional new evidence discovery, not a net answer-quality benefit. A flat DAG answered one question correctly that the single context missed, while the single context answered one that the flat DAG failed on; aggregate exact match was tied and the flat DAG cost more than twice as many tokens.

The diagnostic queries are excluded from this table. At the initial 16,000-token cap most policies failed before answering; increasing the cap to 32,000 was the only protocol change before the six evaluation queries. This small, curated candidate-pool pilot cannot establish statistical superiority or performance on the official full corpus. `audit.py` reports numeric summaries without printing decrypted benchmark content.

## New held-out sample: guarded expansion

`selection_v2.json` freezes eight previously unused offsets by a new SHA-256 ranking before their content was downloaded: two diagnostics and six new evaluation questions. The guarded policy retains the same two-lead plan and evidence-gap check, but permits at most one grandchild per question. It checks observed token usage before expanding and stops unfinished workers when fewer than 9,000 tokens remain, reserving capacity for the finalizer. A diagnostic run confirmed that it could expand and still finish both questions; no thresholds were changed before the six new evaluation questions.

| Policy | Strict exact match | Completed | Total observed tokens | Model calls | Child expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Single context | 3/6 | 5/6 | 82,018 | 36 | 0 |
| Flat two-lead DAG | 3/6 | 5/6 | 154,327 | 80 | 0 |
| Guarded dynamic DAG | 3/6 | 6/6 | 151,344 | 85 | 2 |

The guarded policy avoids the six budget failures of the earlier unconstrained dynamic policy on a **different** six-question sample. Its two expansions did not improve exact match over either baseline. On one question, a grandchild read a labeled gold document not read by the flat DAG, but both policies answered that question correctly. The guarded policy forced eight worker stops to preserve finalizer capacity, so its complete-answer rate must not be mistaken for complete evidence collection. The single context again used substantially fewer tokens. The flat policy had one budget failure and the single context one JSON parse error. Because the old and new samples differ, the two tables are not a paired estimate of the guard's effect; the new paired comparison is flat versus guarded within this table.

```powershell
python benchmarks/browsecomp/download.py --selection benchmarks/browsecomp/selection_v2.json
python benchmarks/browsecomp/run.py --split diagnostic --selection benchmarks/browsecomp/selection_v2.json --policies single flat guarded
python benchmarks/browsecomp/run.py --split evaluation --selection benchmarks/browsecomp/selection_v2.json --policies single flat guarded
python benchmarks/browsecomp/audit.py --selection benchmarks/browsecomp/selection_v2.json --policies single flat guarded
```

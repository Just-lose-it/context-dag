# BrowseComp-Plus candidate-pool pilot

This is a small controlled test of evidence-conditioned DAG expansion. The official [BrowseComp-Plus dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus) has a fixed corpus of about 100,000 documents and human-labeled evidence. We initially use each selected query's evidence, gold, and hard-negative documents as a local candidate pool. **This is not the official full-corpus retrieval setting**; constructing the pool with benchmark labels makes retrieval easier. The labels are withheld from the agent and used only for evaluation.

`selection.json` freezes eight hash-selected row offsets at the listed dataset revision before reading query content. Two offsets are diagnostic; six are held out. `download.py` fetches the obfuscated rows from the official Hugging Face dataset server and decrypts them locally using the method in the [dataset card](https://huggingface.co/datasets/Tevatron/browsecomp-plus/blob/main/README.md). Decrypted data, model predictions, and traces are ignored by Git and must not be pasted into public reports, following the benchmark's anti-leakage guidance.

The comparisons share `deepseek-flash`, the same deterministic BM25 index over 200-word overlapping chunks, candidate documents, JSON answer format, a 900-output-token per-call cap, and a 16,000-token observed-usage cap per query. The policies are one iterative single context, an initial flat two-lead DAG, and a DAG that checks each first-level finding for a specific unsupported factual question before adding up to two grandchildren. Flat and dynamic reuse the same initial model-generated plan per query; its tokens count once in each arm. The root and children can search chunks and read their full text. The gap check uses the child's finding and cited document IDs, not gold labels.

We report strict normalized answer exact match, evidence and gold-document retrieval recall, actual token use, search/read calls, and whether a new branch retrieved a labeled evidence document absent from the flat DAG's reads. Strict exact match is a local pilot measure, not the benchmark's official LLM judge score. A positive result must improve more than the number of spawned nodes; the controller's extra calls and errors count against it. The current scheduler runs nodes sequentially, so this does not test parallel speedup.

```powershell
python benchmarks/browsecomp/download.py
python benchmarks/browsecomp/run.py --split diagnostic
python benchmarks/browsecomp/run.py --split evaluation
```

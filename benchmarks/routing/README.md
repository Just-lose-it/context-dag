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

The fixed executor currently routes an artifact to *all* outgoing successors for `push`. A per-edge direct target is not implemented. The next end-to-end test should use an actual workflow with one costly shared artifact, several consumers with independently determined need, a selective direct route, and measured model tokens, wall time, and task correctness.

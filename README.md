# context-dag

A small Python framework for the **context DAG + global datastore** view of multi-agent workflows. It provides two paths: a fixed DAG executor for artifact routing experiments, and a dynamic context DAG that follows the *general DAG discussion* slides.

This repository implements the execution substrate, not an LLM agent, scheduler, or benchmark result. Node functions can call any model or tool themselves.

## Quick start

```powershell
python -m pip install -e .
python examples/basic.py
python examples/dynamic.py
python examples/parent_handoff.py
python -m unittest discover -s tests -v
```

## Example: direct handoff or shared read

```python
from context_dag import Artifact, ContextDAG, Node

dag = ContextDAG()
dag.add_node(Node("source", lambda ctx: Artifact("finding", "evidence", "source", readers=frozenset({"reader"}))))
dag.add_node(Node("reader", lambda ctx: ctx.read("finding").value))
dag.add_edge("source", "reader")
result = dag.execute(lambda artifact, successors: "store")
assert result.outputs["reader"] == "evidence"
```

The default routing action is `push`, so successors see an artifact in `ctx.inbox`. With `store`, the inbox stays empty and a node must call `ctx.read(key)`. `both` does both; `hold` exposes neither. Store reads require a verified artifact and honor its optional `readers` set. Rewriting a key requires a strictly larger version. Run results include outputs, artifacts and route/read/store events.

To deliver directly to selected outgoing successors, return `RoutingDecision("push", frozenset({"reader"}))` from the route callback. Plain `push` still delivers to every successor. A selected target must be an outgoing successor.

## Dynamic DAG and ReAct-style loop

`DynamicDAG` starts with a root prompt. A `decide(context) -> Action` callback runs repeatedly for each context node. Four actions follow the updated slides; `read_docs` is an additional task tool:

| Action | Effect |
| --- | --- |
| `read_dag(verbose=False)` | Observe node/edge IDs and statuses; `verbose=True` also shows prompts and results. The observation is passed to the next decision. |
| `read_docs(documents=(...))` | Read documents from a caller-supplied tool; the returned observation is passed to the next decision. |
| `read_result(key=...)` | Read an allowed node result from shared storage by its key. |
| `spawn(children=((id, prompt), ...))` | Add new child contexts for the next stage. The parent waits and resumes when all direct children yield. |
| `yield(result=...)` | Finish this node without ending the whole task. |
| `conclude(result=...)` | End the task immediately with a final result. |

When a child yields, its result is written to `SharedStore` under `node/<id>/result`. After all direct children yield, the parent resumes with a completion observation. By default, results of at most 256 characters appear directly in that observation; longer results appear as a `store_key` and require `read_result`. Set `handoff_limit` on `run` to change this threshold. The threshold is in characters, not model tokens. `read_dag(verbose=True)` does not reveal a long child result to bypass the read. When every stage has finished without `conclude`, a separate `finalizer(snapshot, store)` produces the answer. A yielded node can also resume after a later stage completes. `max_nodes` and `max_steps` bound recursive growth. See [examples/dynamic.py](examples/dynamic.py).

`JSONAgent` adapts a model function `complete(messages) -> JSON text` to these actions, keeping separate message histories for each context. This is an application-level tool loop: the model returns a JSON action, the framework executes it, and the next call receives the observation. `DeepSeekClient` supplies an optional `deepseek-flash` client using the official Chat Completions API; it reads `DEEPSEEK_API_KEY` from the environment (or Windows user variables) and never stores it in the repository. A live smoke test is in [examples/deepseek_dynamic.py](examples/deepseek_dynamic.py).

For `read_docs`, pass a `read_documents(ids, node_id)` callback to `DynamicDAG.run`. The callback should enforce document permissions and return errors as observations when a request can be corrected. `DeepSeekClient(total_token_budget=...)` stops further calls once observed usage reaches the cap; a call can exceed the cap because exact usage arrives with the response.

## Scope and limits

- The fixed executor is sequential in deterministic topological order; its graph cannot change during a run. Neither executor provides parallelism, retry, or persistence.
- The dynamic executor processes spawned nodes in stages. Parents resume when direct children yield; node decisions still run sequentially, with no concurrent tool waiting. Results are stored in memory for one run and are not durable across processes.
- In the fixed executor each node runs once and may return an `Artifact`, a list or tuple of artifacts, or an ordinary value.
- Plain `push` delivers to every successor; `RoutingDecision` can limit direct delivery to a selected set. Direct delivery does not enforce store reader permissions.
- Store versions protect future reads; an artifact already delivered to a node cannot be recalled.
- The framework does not choose a routing policy. The `route` callback is where experimental policies, including Idea 1, belong.

## Origin

The fixed executor follows the “context DAG and global datastore” abstraction in *MAS Taxonomy & Discussion*. The dynamic executor follows pages 1 and 3 of *general DAG discussion*: a context observes the DAG, decides to yield/spawn/conclude, and a separate finalizer handles the all-yield path. The current `read_dag` tool reads the in-memory graph directly; mirroring its state through the shared datastore remains future work.

## Benchmark pilot

The [HotpotQA pilot](benchmarks/hotpotqa/README.md) reports two frozen samples totaling 60 questions. A [MuSiQue pilot](benchmarks/musique/README.md) adds frozen 2-hop and 4-hop samples. Neither establishes an answer-quality gain for the current DAG policies. [Routing experiments](benchmarks/routing/README.md) include a fan-out mechanism check and a frozen live `deepseek-flash` comparison: targeted handoff and shared read each used 80.7% fewer prompt tokens than broadcast with a known single reader. Datasets and per-question predictions stay local.
The [research direction note](docs/research_direction.md) records the results and the remaining reader-prediction question.

## License

MIT. See [LICENSE](LICENSE).

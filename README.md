# context-dag

A small Python framework for the **context DAG + global datastore** view of multi-agent workflows. It provides two paths: a fixed DAG executor for artifact routing experiments, and a dynamic context DAG that follows the *general DAG discussion* slides.

This repository implements the execution substrate, not an LLM agent, scheduler, or benchmark result. Node functions can call any model or tool themselves.

## Quick start

```powershell
python -m pip install -e .
python examples/basic.py
python examples/dynamic.py
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

## Dynamic DAG and ReAct-style loop

`DynamicDAG` starts with a root prompt. A `decide(context) -> Action` callback runs repeatedly for each context node. Its four actions correspond to the updated slides:

| Action | Effect |
| --- | --- |
| `read_dag(verbose=False)` | Observe node/edge IDs and statuses; `verbose=True` also shows prompts and results. The observation is passed to the next decision. |
| `spawn(children=((id, prompt), ...))` | Add new child contexts for the next stage. The parent finishes its current stage. |
| `yield(result=...)` | Finish this node without ending the whole task. |
| `conclude(result=...)` | End the task immediately with a final result. |

When every stage has finished without `conclude`, a separate `finalizer(snapshot, store)` produces the answer. `max_nodes` and `max_steps` bound recursive growth. See [examples/dynamic.py](examples/dynamic.py).

`JSONAgent` adapts a model function `complete(messages) -> JSON text` to these actions, keeping separate message histories for each context. This is an application-level tool loop: the model returns a JSON action, the framework executes it, and the next call receives the observation. A model client and finalizer are supplied by the caller; no API key or provider is built into the package.

## Scope and limits

- The fixed executor is sequential in deterministic topological order; its graph cannot change during a run. Neither executor provides parallelism, retry, or persistence.
- The dynamic executor processes spawned nodes in stages. `yield` finishes a node's stage; it does not resume that node later. Node decisions run sequentially, and there is no concurrent tool waiting.
- In the fixed executor each node runs once and may return an `Artifact`, a list or tuple of artifacts, or an ordinary value.
- Direct edges currently deliver to every successor; per-edge artifact filtering is a planned extension. Direct delivery does not enforce store reader permissions.
- Store versions protect future reads; an artifact already delivered to a node cannot be recalled.
- The framework does not choose a routing policy. The `route` callback is where experimental policies, including Idea 1, belong.

## Origin

The fixed executor follows the “context DAG and global datastore” abstraction in *MAS Taxonomy & Discussion*. The dynamic executor follows pages 1 and 3 of *general DAG discussion*: a context observes the DAG, decides to yield/spawn/conclude, and a separate finalizer handles the all-yield path. The current `read_dag` tool reads the in-memory graph directly; mirroring its state through the shared datastore remains future work.

## License

MIT. See [LICENSE](LICENSE).

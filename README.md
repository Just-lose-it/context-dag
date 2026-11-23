# context-dag

A small Python framework for the **context DAG + global datastore** view of multi-agent workflows. Nodes are logical execution contexts. Edges deliver artifacts directly to successor inboxes. A shared store lets nodes read versioned artifacts when they need them. A routing function chooses `push`, `store`, `both`, or `hold` for each artifact.

This repository implements the execution substrate, not an LLM agent, scheduler, or benchmark result. Node functions can call any model or tool themselves.

## Quick start

```powershell
python -m pip install -e .
python examples/basic.py
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

## Scope and limits

- Execution is sequential in deterministic topological order. There is no parallelism, retry, persistence, or dynamic edge insertion during a run.
- Each node runs once. A node may return an `Artifact`, a list or tuple of artifacts, or an ordinary value.
- Direct edges currently deliver to every successor; per-edge artifact filtering is a planned extension. Direct delivery does not enforce store reader permissions.
- Store versions protect future reads; an artifact already delivered to a node cannot be recalled.
- The framework does not choose a routing policy. The `route` callback is where experimental policies, including Idea 1, belong.

## Origin

The design follows the “context DAG and global datastore” abstraction in the local *MAS Taxonomy & Discussion* slides: nodes are contexts, edges are direct communication, and storage supports information useful across nodes or without a clear direct recipient.

## License

MIT. See [LICENSE](LICENSE).

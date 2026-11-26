"""Deterministic fan-out check of direct context exposure versus on-demand reads.

This measures characters delivered to consumer contexts, not model tokens or latency.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from context_dag import Artifact, ContextDAG, Node


def run(policy, fanout, readers, payload_size):
    payload = "x" * payload_size
    dag = ContextDAG()
    dag.add_node(Node("source", lambda ctx: Artifact(
        "report", payload, "source", readers=frozenset(f"c{i}" for i in range(fanout)))))
    context_chars = 0
    read_chars = 0

    def consumer(index):
        def use(ctx):
            nonlocal context_chars, read_chars
            context_chars += sum(len(a.value) for a in ctx.inbox)
            if index not in readers:
                return None
            artifact = ctx.inbox[0] if ctx.inbox else ctx.read("report")
            if not ctx.inbox and artifact:
                read_chars += len(artifact.value)
            return artifact.value if artifact else None
        return use

    for index in range(fanout):
        dag.add_node(Node(f"c{index}", consumer(index)))
        dag.add_edge("source", f"c{index}")
    result = dag.execute(lambda artifact, successors: policy)
    assert all(result.outputs[f"c{i}"] == (payload if i in readers else None)
               for i in range(fanout))
    return context_chars + read_chars, sum(event["kind"] == "read" for event in result.events)


if __name__ == "__main__":
    print("fanout readers payload_chars push_exposure store_exposure store_reads")
    for fanout, count, size in ((8, 1, 8192), (8, 4, 8192), (8, 8, 8192),
                                (2, 1, 2048), (2, 2, 2048)):
        readers = frozenset(range(count))
        pushed, _ = run("push", fanout, readers, size)
        stored, reads = run("store", fanout, readers, size)
        print(fanout, count, size, pushed, stored, reads)

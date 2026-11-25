"""A child stores its result; the parent resumes and reads a long result."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import Action, DynamicDAG


dag = DynamicDAG("Collect one result")


def decide(ctx):
    if ctx.node_id == "child":
        return Action("yield", result="A result longer than the direct handoff limit")
    if ctx.observation is None:
        return Action("spawn", children=(("child", "Produce a result"),))
    if "completed_children" in ctx.observation:
        return Action("read_result", key=ctx.observation["completed_children"][0]["store_key"])
    return Action("conclude", result=ctx.observation["result"])


result = dag.run(decide, lambda snapshot, store: "unused", handoff_limit=8)
print(result.answer)
print([(event["node"], event["action"]) for event in result.events])

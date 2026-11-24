import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import Action, DynamicDAG


dag = DynamicDAG("Summarize two topics")


def decide(ctx):
    if ctx.node_id == "root":
        return Action("spawn", children=(("a", "Summarize topic A"), ("b", "Summarize topic B")))
    if ctx.observation is None:
        return Action("read_dag", verbose=False)
    return Action("yield", result=f"{ctx.prompt}: done")


def finalizer(snapshot, store):
    return " | ".join(n["result"] for n in snapshot["nodes"] if n["status"] == "yielded")


result = dag.run(decide, finalizer)
print(result.answer)
print(result.events)

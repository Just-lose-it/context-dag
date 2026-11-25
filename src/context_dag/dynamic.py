"""Stage-based dynamic context DAG with a small ReAct-style tool loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from .core import SharedStore

ActionName = Literal["read_dag", "read_docs", "spawn", "yield", "conclude"]


@dataclass(frozen=True)
class Action:
    name: ActionName
    verbose: bool = False
    children: tuple[tuple[str, str], ...] = ()  # (node_id, prompt)
    documents: tuple[str, ...] = ()
    result: str = ""


@dataclass
class ContextNode:
    id: str
    prompt: str
    parent: str | None = None
    stage: int = 0
    status: str = "pending"
    result: str = ""


@dataclass(frozen=True)
class DecisionContext:
    node_id: str
    prompt: str
    stage: int
    observation: dict[str, Any] | None
    store: SharedStore


@dataclass(frozen=True)
class DynamicResult:
    answer: str
    nodes: dict[str, ContextNode]
    events: tuple[dict[str, Any], ...]
    finalized: bool


class DynamicDAG:
    def __init__(self, root_prompt: str, root_id: str = "root") -> None:
        self.nodes = {root_id: ContextNode(root_id, root_prompt)}
        self.root_id = root_id
        self.store = SharedStore()
        self.events: list[dict[str, Any]] = []
        self._ran = False

    def snapshot(self, verbose: bool = False) -> dict[str, Any]:
        nodes = []
        for node in self.nodes.values():
            item = {"id": node.id, "parent": node.parent, "stage": node.stage, "status": node.status}
            if verbose:
                item.update(prompt=node.prompt, result=node.result)
            nodes.append(item)
        return {"nodes": nodes, "edges": [
            {"from": node.parent, "to": node.id} for node in self.nodes.values() if node.parent
        ]}

    def run(
        self,
        decide: Callable[[DecisionContext], Action],
        finalizer: Callable[[dict[str, Any], SharedStore], str],
        *,
        max_nodes: int = 64,
        max_steps: int = 256,
        read_documents: Callable[[tuple[str, ...], str], dict[str, Any]] | None = None,
    ) -> DynamicResult:
        """Run each stage to completion, then start its spawned children.

        read_dag is a nonterminal tool action. A yielded node resumes after a
        later stage completes, if one exists. conclude ends the task. If all
        work finishes without concluding, a separate finalizer runs.
        """
        if max_nodes < 1 or max_steps < 1:
            raise ValueError("limits must be positive")
        if self._ran:
            raise RuntimeError("a DynamicDAG instance can run only once")
        self._ran = True
        steps = 0
        stage = 0
        waiting: dict[str, int] = {}

        def run_node(node: ContextNode, current_stage: int, observation: dict[str, Any] | None = None) -> str | None:
            nonlocal steps
            node.status = "running"
            while True:
                if steps >= max_steps:
                    raise RuntimeError("maximum ReAct steps exceeded")
                steps += 1
                action = decide(DecisionContext(node.id, node.prompt, current_stage, observation, self.store))
                if action.name == "read_dag":
                    observation = self.snapshot(action.verbose)
                    self.events.append({"node": node.id, "action": "read_dag", "verbose": action.verbose})
                    continue
                if action.name == "read_docs":
                    if read_documents is None:
                        raise ValueError("read_docs requires a read_documents tool")
                    if not action.documents:
                        raise ValueError("read_docs requires document IDs")
                    observation = read_documents(action.documents, node.id)
                    self.events.append({"node": node.id, "action": "read_docs",
                                        "documents": list(action.documents)})
                    continue
                if action.name == "spawn":
                    if not action.children:
                        raise ValueError("spawn requires children")
                    ids = [child_id for child_id, _ in action.children]
                    if len(ids) != len(set(ids)) or any(child_id in self.nodes for child_id in ids):
                        raise ValueError("child IDs must be new and unique")
                    if len(self.nodes) + len(ids) > max_nodes:
                        raise RuntimeError("maximum node count exceeded")
                    if any(not child_id or not prompt for child_id, prompt in action.children):
                        raise ValueError("children need IDs and prompts")
                    for child_id, prompt in action.children:
                        self.nodes[child_id] = ContextNode(child_id, prompt, node.id, current_stage + 1)
                    node.status = "spawned"
                    waiting.pop(node.id, None)
                    self.events.append({"node": node.id, "action": "spawn", "children": ids})
                    return None
                if action.name == "yield":
                    node.status = "yielded"
                    node.result = action.result
                    waiting[node.id] = current_stage
                    self.events.append({"node": node.id, "action": "yield", "result": action.result})
                    return None
                if action.name == "conclude":
                    node.status = "concluded"
                    node.result = action.result
                    self.events.append({"node": node.id, "action": "conclude", "result": action.result})
                    return action.result
                raise ValueError(f"unknown action: {action.name}")

        while True:
            current = [node for node in self.nodes.values() if node.stage == stage and node.status == "pending"]
            if not current:
                # A yielded context waits until a later stage has finished.
                resumed = [self.nodes[node_id] for node_id, waited_at in waiting.items() if waited_at < stage - 1]
                if not resumed:
                    break
                for node in resumed:
                    answer = run_node(node, stage - 1, self.snapshot(verbose=True))
                    if answer is not None:
                        return DynamicResult(answer, self.nodes.copy(), tuple(self.events), False)
                continue
            for node in current:
                answer = run_node(node, stage)
                if answer is not None:
                    return DynamicResult(answer, self.nodes.copy(), tuple(self.events), False)
            stage += 1
        answer = finalizer(self.snapshot(verbose=True), self.store)
        self.events.append({"node": "finalizer", "action": "conclude", "result": answer})
        return DynamicResult(answer, self.nodes.copy(), tuple(self.events), True)

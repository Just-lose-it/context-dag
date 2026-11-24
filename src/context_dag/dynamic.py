"""Stage-based dynamic context DAG with a small ReAct-style tool loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from .core import SharedStore

ActionName = Literal["read_dag", "spawn", "yield", "conclude"]


@dataclass(frozen=True)
class Action:
    name: ActionName
    verbose: bool = False
    children: tuple[tuple[str, str], ...] = ()  # (node_id, prompt)
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
    ) -> DynamicResult:
        """Run each stage to completion, then start its spawned children.

        read_dag is a nonterminal tool action. spawn/yield finish a node's stage.
        conclude ends the task. If all nodes yield or spawn without concluding,
        a separate finalizer runs after the final stage.
        """
        if max_nodes < 1 or max_steps < 1:
            raise ValueError("limits must be positive")
        steps = 0
        stage = 0
        while True:
            current = [node for node in self.nodes.values() if node.stage == stage]
            if not current:
                break
            for node in current:
                node.status = "running"
                observation = None
                while True:
                    if steps >= max_steps:
                        raise RuntimeError("maximum ReAct steps exceeded")
                    steps += 1
                    action = decide(DecisionContext(node.id, node.prompt, stage, observation, self.store))
                    if action.name == "read_dag":
                        observation = self.snapshot(action.verbose)
                        self.events.append({"node": node.id, "action": "read_dag", "verbose": action.verbose})
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
                            self.nodes[child_id] = ContextNode(child_id, prompt, node.id, stage + 1)
                        node.status = "spawned"
                        self.events.append({"node": node.id, "action": "spawn", "children": ids})
                        break
                    if action.name == "yield":
                        node.status = "yielded"
                        node.result = action.result
                        self.events.append({"node": node.id, "action": "yield", "result": action.result})
                        break
                    if action.name == "conclude":
                        node.status = "concluded"
                        node.result = action.result
                        self.events.append({"node": node.id, "action": "conclude", "result": action.result})
                        return DynamicResult(action.result, self.nodes.copy(), tuple(self.events), False)
                    raise ValueError(f"unknown action: {action.name}")
            stage += 1
        answer = finalizer(self.snapshot(verbose=True), self.store)
        self.events.append({"node": "finalizer", "action": "conclude", "result": answer})
        return DynamicResult(answer, self.nodes.copy(), tuple(self.events), True)

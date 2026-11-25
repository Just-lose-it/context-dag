"""Model adapter: turn a text completion function into dynamic DAG actions."""

import json
from collections import defaultdict
from typing import Callable

from .dynamic import Action, DecisionContext


class JSONAgent:
    """Call `complete(messages) -> JSON text` once per ReAct decision.

    The caller supplies the model client. This adapter keeps separate histories
    for each context node and appends tool observations after read_dag.
    """

    def __init__(self, complete: Callable[[list[dict[str, str]]], str]) -> None:
        self.complete = complete
        self.histories: dict[str, list[dict[str, str]]] = defaultdict(list)

    def __call__(self, context: DecisionContext) -> Action:
        history = self.histories[context.node_id]
        if not history:
            history.append({"role": "system", "content": (
                "You control one node of a context DAG. Return one JSON object only. "
                "Actions: read_dag {verbose: bool}; read_docs {documents: [id,...]}; "
                "read_result {key: str}; "
                "spawn {children: [[id,prompt],...]}; "
                "yield {result: str}; conclude {result: str}. "
                "Use read_dag when the current graph matters. Spawn only well-scoped subtasks."
            )})
            history.append({"role": "user", "content": context.prompt})
        if context.observation is not None:
            history.append({"role": "user", "content": "Tool observation: " + json.dumps(context.observation)})
        raw = self.complete(history)
        data = json.loads(raw)
        if "action" not in data and "answer" in data:
            data = {"action": "conclude", "result": raw}
        name = data["action"]
        history.append({"role": "assistant", "content": raw})
        return Action(name=name, verbose=bool(data.get("verbose", False)),
                      children=tuple((str(x[0]), str(x[1])) for x in data.get("children", [])),
                      documents=tuple(str(x) for x in data.get("documents", [])),
                      key=str(data.get("key", "")),
                      result=str(data.get("result", "")))

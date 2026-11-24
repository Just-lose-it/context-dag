"""Small live example; requires DEEPSEEK_API_KEY and makes paid API calls."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from context_dag import DeepSeekClient, DynamicDAG, JSONAgent


dag = DynamicDAG("Use read_dag once, then conclude with the answer to 2 + 2. Keep it short.")
agent = JSONAgent(DeepSeekClient())
result = dag.run(agent, lambda snapshot, store: "No node concluded", max_nodes=4, max_steps=6)
print(result.answer)
print([(event["node"], event["action"]) for event in result.events])

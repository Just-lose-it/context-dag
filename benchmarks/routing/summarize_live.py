"""Aggregate the frozen live routing comparison from local JSONL results."""

import json
from collections import defaultdict
from pathlib import Path

path = Path(__file__).parent / "results/live.jsonl"
groups = defaultdict(list)
for line in path.open(encoding="utf-8"):
    row = json.loads(line)
    groups[row["policy"]].append(row)

for policy in ("broadcast", "targeted", "store"):
    rows = groups[policy]
    tasks = [task for row in rows for task in row["tasks"]]
    print(policy, "batches", len(rows), "tasks", len(tasks),
          "correct", sum(task["correct"] for task in tasks),
          "shared_correct", sum(row["tasks"][-1]["correct"] for row in rows),
          "errors", sum(task["error"] is not None for task in tasks),
          "prompt_tokens", sum(task["prompt_tokens"] for task in tasks),
          "total_tokens", sum(task["tokens"] for task in tasks),
          "seconds", round(sum(row["elapsed_seconds"] for row in rows), 2),
          "reads", sum(row["reads"] for row in rows))

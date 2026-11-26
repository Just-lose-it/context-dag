"""Recompute paragraph scores from saved predictions and print all frozen results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmarks.musique.run import score

HERE = Path(__file__).parent
lookup = {item["id"]: item for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")
          if (item := json.loads(line))["id"] in json.loads((HERE / "selection.json").read_text())["ids"]["2hop"]
          + json.loads((HERE / "selection.json").read_text())["ids"]["4hop"]}

for group in ("2hop", "4hop"):
    path = HERE / "results" / f"{group}.jsonl"
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    for policy in ("single", "fixed", "adaptive"):
        values = []
        for row in rows:
            result = row[policy]
            values.append({**result, **score(result["prediction"], lookup[row["id"]])}
                          if "prediction" in result else result)
        print(group, policy, "n", len(values),
              "answer_em", sum(v.get("answer_em", 0) for v in values),
              "support_em", sum(v.get("support_em", 0) for v in values),
              "joint_em", sum(v.get("joint_em", 0) for v in values),
              "support_recall", round(sum(v.get("support_recall", 0) for v in values) / len(values), 3),
              "errors", sum("error" in v for v in values),
              "calls", sum(v["calls"] for v in values),
              "tokens", sum(v["tokens"] for v in values))

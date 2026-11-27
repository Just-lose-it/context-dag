"""Aggregate held-out multi-level DAG pilot rows, counting errors as failures."""

import json
from pathlib import Path

HERE = Path(__file__).parent
for group in ("chain_4hop1", "branch_4hop3"):
    path = HERE / "results" / f"multilevel_{group}.jsonl"
    if not path.exists():
        continue
    latest = {row["id"]: row for line in path.open(encoding="utf-8")
              if (row := json.loads(line))["split"] == "evaluation"}
    blocked = [row for row in latest.values() if row.get("stopped_reason") == "payment_required"
               or any("HTTP Error 402" in str(row.get(p, {}).get("error", ""))
                      for p in ("single", "single_review", "flat", "nested"))]
    rows = [row for row in latest.values() if row not in blocked]
    print(group, "complete", len(rows), "payment_blocked", len(blocked))
    if not rows:
        continue
    for policy in ("single", "single_review", "flat", "nested"):
        values = [row[policy] for row in rows]
        print(group, policy, "n", len(values),
              "answer_em", sum(v.get("answer_em", 0) for v in values),
              "support_em", sum(v.get("support_em", 0) for v in values),
              "joint_em", sum(v.get("joint_em", 0) for v in values),
              "errors", sum("error" in v for v in values),
              "child_spawns", sum(v.get("child_spawns", 0) for v in values),
              "reads", sum(v.get("reads", 0) for v in values),
              "calls", sum(v.get("calls", 0) for v in values),
              "tokens", sum(v.get("tokens", 0) for v in values))

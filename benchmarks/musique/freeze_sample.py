"""Freeze a reproducible answerable MuSiQue dev sample before model calls."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
items = [json.loads(line) for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8")]
groups = {"2hop": [], "4hop": []}
for item in items:
    group = "2hop" if item["id"].startswith("2hop__") else "4hop" if item["id"].startswith("4hop") else None
    if group and item["answerable"] and len(item["paragraphs"]) == 20:
        digest = hashlib.sha256(("context-dag-musique-v1:" + item["id"]).encode()).hexdigest()
        groups[group].append((digest, item["id"]))
selection = {
    "method": "First eight SHA-256(context-dag-musique-v1:ID) IDs in each answerable dev hop group with 20 paragraphs; no answer or model output used.",
    "ids": {group: [item_id for _, item_id in sorted(rows)[:8]] for group, rows in groups.items()},
}
(HERE / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
print({group: len(ids) for group, ids in selection["ids"].items()})

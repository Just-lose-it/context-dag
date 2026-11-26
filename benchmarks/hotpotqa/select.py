"""Freeze 20 hard, long-context dev IDs before evaluation."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
data = json.loads((HERE / "data" / "hotpot_dev_distractor_v1.json").read_text(encoding="utf-8"))
records = []
for item in data[14:]:  # first 14 dev items were used during development
    chars = sum(len(sentence) for _, sentences in item["context"] for sentence in sentences)
    threshold = 5850 if item["type"] == "comparison" else 6455
    if item["level"] == "hard" and chars >= threshold:
        digest = hashlib.sha256(("context-dag-v1:" + item["_id"]).encode()).hexdigest()
        records.append((item["type"], digest, item["_id"]))

selection = {
    "method": "First 10 IDs per type by SHA-256(context-dag-v1:ID), among dev items after index 13 with context length at least the type-specific 75th-percentile threshold.",
    "threshold_chars": {"comparison": 5850, "bridge": 6455},
    "ids": {kind: [record[2] for record in sorted(records) if record[0] == kind][:10]
            for kind in ("comparison", "bridge")},
}
(HERE / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
print({kind: len(ids) for kind, ids in selection["ids"].items()})

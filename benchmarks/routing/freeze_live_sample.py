"""Freeze MuSiQue questions for the live routing experiment before model calls."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE.parent / "musique/data/musique_ans_v1.0_dev.jsonl"
used = json.loads((HERE.parent / "musique/selection.json").read_text(encoding="utf-8"))["ids"]
exclude = set(used["2hop"] + used["4hop"])
shared = []
local = []
for line in DATA.open(encoding="utf-8"):
    item = json.loads(line)
    if item["id"] in exclude or not item["answerable"] or len(item["paragraphs"]) != 20:
        continue
    digest = hashlib.sha256(("context-dag-routing-live-v1:" + item["id"]).encode()).hexdigest()
    if item["id"].startswith("4hop"):
        shared.append((digest, item["id"]))
    if item["id"].startswith("2hop"):
        first = item["question_decomposition"][0]
        if ("?" in first["question"] and "#" not in first["question"]
                and any(p["idx"] == first["paragraph_support_idx"] for p in item["paragraphs"])):
            local.append((digest, item["id"]))

sample = {
    "method": "Hash-order first three unused answerable 4-hop dev items with 20 paragraphs as shared-report questions; first 21 unused answerable 2-hop items with 20 paragraphs and a natural first decomposition question as seven disjoint local questions per batch. No model output or gold answer was used for selection.",
    "shared_ids": [item_id for _, item_id in sorted(shared)[:3]],
    "local_ids": [item_id for _, item_id in sorted(local)[:21]],
}
(HERE / "live_sample.json").write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")
print(sample)

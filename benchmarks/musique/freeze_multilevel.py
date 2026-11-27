"""Freeze unused chain and branching four-hop MuSiQue IDs before the pilot."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
used = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))["ids"]["4hop"]
used += json.loads((HERE.parent / "routing/live_sample.json").read_text(encoding="utf-8"))["shared_ids"]
groups = {"chain_4hop1": [], "branch_4hop3": []}
for line in (HERE / "data/musique_ans_v1.0_dev.jsonl").open(encoding="utf-8"):
    item = json.loads(line)
    group = "chain_4hop1" if item["id"].startswith("4hop1__") else (
        "branch_4hop3" if item["id"].startswith("4hop3__") else None)
    if group and item["id"] not in used and item["answerable"] and len(item["paragraphs"]) == 20:
        digest = hashlib.sha256(("context-dag-multilevel-v1:" + item["id"]).encode()).hexdigest()
        groups[group].append((digest, item["id"]))
def distinct_ids(rows):
    selected = []
    components = set()
    for _, item_id in sorted(rows):
        parts = set(item_id.split("__", 1)[1].split("_"))
        if not parts & components:
            selected.append(item_id)
            components.update(parts)
        if len(selected) == 8:
            break
    return selected


selection = {
    "method": "First eight hash-ordered, previously unused answerable dev IDs with 20 paragraphs in each MuSiQue 4hop1 and 4hop3 group, excluding reuse of any component question ID within a group. First two per group are development diagnostics; remaining six are held-out evaluation. The ID group is a structural proxy; no gold decomposition or answer is passed to policies.",
    "ids": {group: distinct_ids(rows) for group, rows in groups.items()},
}
(HERE / "selection_multilevel.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
print({group: len(ids) for group, ids in selection["ids"].items()})

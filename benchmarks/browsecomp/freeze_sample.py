"""Freeze eight BrowseComp-Plus row offsets without looking at query contents."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
offsets = sorted(range(830), key=lambda i: hashlib.sha256(
    f"context-dag-browsecomp-v1:{i}".encode()).hexdigest())[:8]
selection = {
    "dataset": "Tevatron/browsecomp-plus",
    "revision": "144cff8e35b5eaef7e526346aa60774a9deb941f",
    "split": "test",
    "method": "First eight row offsets ranked by SHA-256(context-dag-browsecomp-v1:offset), before any query content or model output was read. First two diagnostic, remaining six held out.",
    "diagnostic_offsets": offsets[:2],
    "evaluation_offsets": offsets[2:],
}
(HERE / "selection.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
print(selection["diagnostic_offsets"], selection["evaluation_offsets"])

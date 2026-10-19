"""Freeze the BFCL items used in Project Mew (PREREGISTRATION.md §6.2, §9).

Run in the scoring venv (bfcl-eval==2026.3.23). Writes one JSON line per item:
  id, category, scored, messages (first-turn chat messages), tools (OpenAI-style tool list).
  data/frozen/bfcl_items.jsonl      8 scored categories + 2 descriptive irrelevance categories
  data/frozen/bfcl_dev_items.jsonl  harness-development items (Java/JS simple; never scored or reported)
Generation on Kaggle reads only these files, so bfcl itself is never imported there.
"""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

assert version("bfcl-eval") == "2026.3.23", version("bfcl-eval")
SCORED = ["simple_python", "multiple", "parallel", "parallel_multiple",
          "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
DESCRIPTIVE = ["irrelevance", "live_irrelevance"]
DEV = {"simple_java": 20, "simple_javascript": 20}


def freeze(out: str, plan: list) -> dict:
    n = {}
    with open(out, "w") as fh:
        for cat, limit in plan:
            entries = load_dataset_entry(cat)[:limit]  # default: includes bfcl's language-specific hint
            if cat in SCORED:
                assert [g["id"] for g in load_ground_truth_entry(cat)] == [e["id"] for e in entries], cat
            for e in entries:
                assert len(e["question"]) == 1, (cat, e["id"])  # single-turn only
                # OpenAI-completions style gives OpenAPI types; it also rewrites "." to "_" in names,
                # which the Qwen chat template does not need, so the original names are restored.
                # Names then match ground truth directly (checker underscore_to_dot=False).
                tools = convert_to_tool(e["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)
                assert len(tools) == len(e["function"]), (cat, e["id"])
                for t, f in zip(tools, e["function"]):
                    t["function"]["name"] = f["name"]
                rec = {"id": e["id"], "category": cat, "scored": cat in SCORED,
                       "messages": e["question"][0], "tools": tools}
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            n[cat] = len(entries)
    print(out, json.dumps(n))
    print("  sha256", hashlib.sha256(Path(out).read_bytes()).hexdigest())
    return n


n = freeze("data/frozen/bfcl_items.jsonl", [(c, None) for c in SCORED + DESCRIPTIVE])
print("scored total", sum(n[c] for c in SCORED))
freeze("data/frozen/bfcl_dev_items.jsonl", list(DEV.items()))

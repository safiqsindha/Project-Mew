"""Generate raw completions for one checkpoint on one task. Scoring happens offline (mew/score.py).

  python scripts/run_eval.py --task justlogic --items primary --model Qwen/Qwen3.5-0.8B-Base \
      --revision dc7cdfe2... --out outputs/raw/baseline_logic
"""
import argparse
import json
import platform
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from mew import bfcl, generate, justlogic  # noqa: E402

GEN = {  # frozen generation configs (PREREGISTRATION.md §6)
    "justlogic": {"max_new_tokens": 8, "do_sample": False, "stop": "first newline (applied at parse)",
                  "prompt": "JustLogic eval/prompt_3_shot.txt @ cdcd470e, trailing whitespace stripped",
                  "options": justlogic.OPTIONS},
    "bfcl": {"max_new_tokens": 512, "do_sample": False, "stop": "eos",
             "prompt": "model chat template, tools=frozen item tools, enable_thinking=False"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["justlogic", "bfcl"], required=True)
    ap.add_argument("--items", default="primary", help="justlogic: primary|dev; bfcl: all|scored|dev")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="debug only: first N items")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/mew_cache"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    tok, model = generate.load(a.model, a.revision, a.dtype, a.adapter)
    if a.task == "justlogic":
        items = justlogic.load_items(a.cache, ROOT / f"data/manifests/justlogic_{a.items}.csv")
        template = justlogic.fetch_sources(a.cache)["prompt_3_shot.txt"].read_text()
        prompts = {it["uid"]: justlogic.build_prompt(template, it) for it in items}
        options = justlogic.OPTIONS
    else:
        src = "bfcl_dev_items.jsonl" if a.items == "dev" else "bfcl_items.jsonl"
        items = bfcl.load_items(ROOT / "data/frozen" / src, scored_only=(a.items == "scored"))
        prompts = {it["uid"]: bfcl.build_prompt(tok, it) for it in items}
        options = None
    if a.limit:
        keep = [it["uid"] for it in items[:a.limit]]
        prompts = {u: prompts[u] for u in keep}

    meta = {"task": a.task, "items": a.items, "n_items": len(prompts), "model": a.model, "revision": a.revision,
            "adapter": a.adapter, "dtype": a.dtype, "batch_size": a.batch_size, "generation": GEN[a.task],
            "model_class": type(model).__name__, "torch": torch.__version__,
            "transformers": version("transformers"), "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(json.dumps(meta, indent=1), flush=True)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))

    t0 = time.time()
    generate.run(tok, model, prompts, a.out / "completions.jsonl", GEN[a.task]["max_new_tokens"],
                 a.batch_size, options=options)
    meta["wall_clock_s"] = round(time.time() - t0, 1)
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print("done", meta["wall_clock_s"], "s", flush=True)


if __name__ == "__main__":
    main()

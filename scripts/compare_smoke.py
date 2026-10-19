"""Compare smoke runs (dev items only): precision and batch-size invariance, parse rates, speed."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mew.bfcl import decode_tool_calls, load_items  # noqa: E402
from mew.justlogic import parse_answer  # noqa: E402

root = Path(sys.argv[1])


def comp(name):
    p = root / name / "completions.jsonl"
    return {json.loads(l)["uid"]: json.loads(l) for l in open(p)} if p.exists() else {}


def meta(name):
    p = root / name / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


for task, runs in [("jl", ["smoke_jl_fp32_b16", "smoke_jl_fp16_b16", "smoke_jl_fp32_b1"]),
                   ("bfcl", ["smoke_bfcl_fp32_b8", "smoke_bfcl_fp16_b8", "smoke_bfcl_fp32_b1"])]:
    ref = comp(runs[0])
    print(f"== {task}")
    for r in runs:
        c, m = comp(r), meta(r)
        if not c:
            print(f"  {r}: MISSING (see {root / r / 'log.txt'})")
            continue
        same_text = sum(c[u]["completion"] == ref[u]["completion"] for u in c if u in ref)
        line = f"  {r}: n={len(c)} wall={m.get('wall_clock_s')}s class={m.get('model_class')} identical_to_ref={same_text}/{len(c)}"
        if task == "jl":
            line += f" parse_fail={sum(parse_answer(x['completion']) is None for x in c.values())}"
            if "option_logprobs" in next(iter(c.values())):
                agree = sum(max(c[u]['option_logprobs'], key=c[u]['option_logprobs'].get) ==
                            max(ref[u]['option_logprobs'], key=ref[u]['option_logprobs'].get) for u in c if u in ref)
                line += f" rank_agree={agree}/{len(c)}"
        else:
            items = {it["uid"]: it for it in load_items(Path(__file__).resolve().parents[1] / "data/frozen/bfcl_dev_items.jsonl")}
            ok = 0
            for u, x in c.items():
                try:
                    decode_tool_calls(x["completion"], items[u]["tools"]); ok += 1
                except ValueError:
                    pass
            line += f" parsed={ok}/{len(c)}"
        print(line)

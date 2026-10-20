"""EXPLORATORY logic diagnostics (docs/exploratory/2026-10-19_logic_diagnostics_plan.md). Not confirmatory.

Runs one prompt variant of the registered 3-shot prompt on the frozen diagnostic subsample (or the
content-free inputs) with the untuned Base model, using the unchanged mew/generate.py.

  python scripts/explore_logic.py --benchmark justlogic --variant swap_R1 --out <dir>
Variants: cf_na | cf_empty | swap_R0 | swap_R1 | swap_R2 | order_012 ... order_210
"""
import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mew import generate, justlogic, proofwriter  # noqa: E402

BASE = ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68")
MEANINGS = ("TRUE", "FALSE", "UNCERTAIN")
ROTATIONS = {  # symbol -> meaning
    "R0": {"A": "TRUE", "B": "FALSE", "C": "UNCERTAIN"},
    "R1": {"A": "FALSE", "B": "UNCERTAIN", "C": "TRUE"},
    "R2": {"A": "UNCERTAIN", "B": "TRUE", "C": "FALSE"},
}
ORDERS = ["order_" + "".join(map(str, p)) for p in itertools.permutations(range(3))]
VARIANTS = ["cf_na", "cf_empty"] + [f"swap_{r}" for r in ROTATIONS] + ORDERS


def template(benchmark, cache):
    if benchmark == "justlogic":
        return justlogic.fetch_sources(cache)["prompt_3_shot.txt"].read_text()
    return (ROOT / "data/frozen/proofwriter_prompt_3_shot.txt").read_text()


def variant_template(t, variant):
    """Returns (template, options {key: continuation}, key->meaning map)."""
    parts = t.split("-----\n")
    assert len(parts) == 5 and [re.search(r"Answer: (\w+)", p).group(1) for p in parts[1:4]] == list(MEANINGS)
    if variant.startswith("order_"):
        perm = [int(c) for c in variant[6:]]
        shots = [parts[1:4][i] for i in perm]
        return "-----\n".join([parts[0], *shots, parts[4]]), justlogic.OPTIONS, {m: m for m in MEANINGS}
    if variant.startswith("swap_"):
        sym = ROTATIONS[variant[5:]]
        inv = {m: s for s, m in sym.items()}
        head = parts[0]
        for m in MEANINGS:  # definition lines keep meaning order; only the label token changes
            head, n = re.subn(rf"(?m)^- {m}: ", f"- {inv[m]}: ", head)
            assert n == 1, (m, n)
        head, n = re.subn(r'Output only "TRUE", "FALSE", or "UNCERTAIN"\.', 'Output only "A", "B", or "C".', head)
        assert n == 1
        shots = [re.sub(r"Answer: (\w+)\n", lambda mm: f"Answer: {inv[mm.group(1)]}\n", p) for p in parts[1:4]]
        return "-----\n".join([head, *shots, parts[4]]), {s: " " + s for s in "ABC"}, sym
    return t, justlogic.OPTIONS, {m: m for m in MEANINGS}  # content-free: registered prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", choices=["justlogic", "proofwriter"], required=True)
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=Path("/tmp/mew_cache"))
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t, options, keymap = variant_template(template(a.benchmark, a.cache), a.variant)
    if a.variant.startswith("cf_"):
        filler = "N/A" if a.variant == "cf_na" else ""
        prompts = {a.variant: t.replace("{PARAGRAPH}", filler).replace("{STATEMENT}", filler).rstrip()}
    else:
        sub = [r for r in csv.DictReader(open(ROOT / "docs/exploratory/logic_diag_subsample.csv"))
               if r["benchmark"] == a.benchmark]
        mod = justlogic if a.benchmark == "justlogic" else proofwriter
        man = ROOT / f"data/manifests/{a.benchmark}_primary.csv"
        items = {it["uid"]: it for it in mod.load_items(a.cache, man)}
        prompts = {r["uid"]: mod.build_prompt(t, items[r["uid"]]) for r in sub}
    meta = {"EXPLORATORY": True, "benchmark": a.benchmark, "variant": a.variant, "options": options,
            "key_to_meaning": keymap, "n": len(prompts), "model": BASE[0], "revision": BASE[1], "dtype": "fp32",
            "batch_size": a.batch_size, "template_sha256": generate.sha(t)}
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1), flush=True)
    tok, model = generate.load(*BASE, "fp32")
    generate.run(tok, model, prompts, a.out / "completions.jsonl", 8, a.batch_size, options=options)
    print("done", flush=True)


if __name__ == "__main__":
    main()

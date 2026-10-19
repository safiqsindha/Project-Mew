"""Build the Treatment and Control SFT sets from the Ditto chains and check the §4.2 matching rules.

  python scripts/build_sft_data.py --chains "~/Desktop/Project Ditto/Ditto V1/chains" --tokenizer <tokenizer.json>

Treatment = chains/real/<id>.jsonl; Control = chains/shuffled/<id>_shuffled_42.jsonl (PREREGISTRATION.md §3-4).
Each example: prompt = header + steps 1..cutoff_k, target = steps cutoff_k+1..n (loss on target only).
Writes data/sft/{treatment,control}_{train,monitor}.jsonl (git-ignored) and the committed hash manifest
data/manifests/chains_manifest.csv. Aborts if any matching check fails.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
STEP = re.compile(r"(?m)^(?=Step \d+ \(turn=)")
STEP_HEAD = re.compile(r"^Step \d+ \(turn=\d+\):\n")
MAX_LEN = 1536
LOSS_TOKEN_TOLERANCE = 0.005


def split(rendered: str, k: int):
    parts = STEP.split(rendered)
    header, steps = parts[0], parts[1:]
    return header + "".join(steps[:k]), "".join(steps[k:]), steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chains", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True, help="Qwen3.5-0.8B-Base tokenizer.json @ dc7cdfe2")
    a = ap.parse_args()
    chains = a.chains.expanduser()
    tok = Tokenizer.from_file(str(a.tokenizer))
    ntok = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731

    out = ROOT / "data/sft"
    out.mkdir(parents=True, exist_ok=True)
    files = {(arm, part): open(out / f"{arm}_{part}.jsonl", "w")
             for arm in ("treatment", "control") for part in ("train", "monitor")}
    manifest = []
    tot = Counter()
    ids = sorted(p.name[:-6] for p in (chains / "real").glob("*.jsonl"))
    assert len(ids) == 4806, len(ids)
    for cid in ids:
        src = {"treatment": chains / "real" / f"{cid}.jsonl",
               "control": chains / "shuffled" / f"{cid}_shuffled_42.jsonl"}
        recs = {arm: json.loads(p.read_text().splitlines()[0]) for arm, p in src.items()}
        t, c = recs["treatment"], recs["control"]
        assert t["match_id"] == c["match_id"] and t["cutoff_k"] == c["cutoff_k"], cid
        part = "monitor" if int(hashlib.sha256(t["match_id"].encode()).hexdigest(), 16) % 20 == 0 else "train"
        ex = {}
        for arm, r in recs.items():
            prompt, target, steps = split(r["rendered"], r["cutoff_k"])
            assert prompt + target == r["rendered"], (cid, arm)
            ex[arm] = {"chain_id": cid, "match_id": r["match_id"], "prompt": prompt, "target": target,
                       "steps": [STEP_HEAD.sub("", s) for s in steps],
                       "n_tok": ntok(r["rendered"]), "n_target_tok": ntok(target)}
        # §4.2 checks, per chain
        assert sorted(ex["treatment"]["steps"]) == sorted(ex["control"]["steps"]), f"content multiset differs: {cid}"
        assert len(ex["treatment"]["steps"]) == len(ex["control"]["steps"]), cid
        assert ex["treatment"]["n_tok"] == ex["control"]["n_tok"], f"token count differs: {cid}"
        assert ex["treatment"]["n_tok"] < MAX_LEN, f"too long: {cid}"
        for arm in ("treatment", "control"):
            e = ex[arm]
            files[(arm, part)].write(json.dumps({k: e[k] for k in ("chain_id", "match_id", "prompt", "target")}) + "\n")
            tot[(arm, part, "examples")] += 1
            tot[(arm, part, "tokens")] += e["n_tok"]
            tot[(arm, part, "target_tokens")] += e["n_target_tok"]
        for arm, p in src.items():
            manifest.append({"arm": arm, "chain_id": cid, "split": part, "file": str(p.relative_to(chains)),
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    for f in files.values():
        f.close()

    for part in ("train", "monitor"):
        for k in ("examples", "tokens"):
            assert tot[("treatment", part, k)] == tot[("control", part, k)], (part, k)
        lt, lc = tot[("treatment", part, "target_tokens")], tot[("control", part, "target_tokens")]
        assert abs(lc - lt) / lt <= LOSS_TOKEN_TOLERANCE, (part, lt, lc)

    mpath = ROOT / "data/manifests/chains_manifest.csv"
    with open(mpath, "w", newline="") as fh:
        w = csv.DictWriter(fh, ["arm", "chain_id", "split", "file", "sha256"])
        w.writeheader()
        w.writerows(manifest)
    report = {f"{arm}/{part}/{k}": v for (arm, part, k), v in sorted(tot.items())}
    report["chains_manifest_sha256"] = hashlib.sha256(mpath.read_bytes()).hexdigest()
    for arm in ("treatment", "control"):
        for part in ("train", "monitor"):
            report[f"sft_{arm}_{part}_sha256"] = hashlib.sha256((out / f"{arm}_{part}.jsonl").read_bytes()).hexdigest()
    (ROOT / "data/manifests/sft_build_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("ALL §4.2 CHECKS PASSED")


if __name__ == "__main__":
    main()

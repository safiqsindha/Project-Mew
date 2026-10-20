"""EXPLORATORY logic diagnostics analysis (docs/exploratory/2026-10-19_logic_diagnostics_plan.md).

  python scripts/explore_logic_analysis.py [--explore-dir <downloaded kernel out dir>]
C1 and C5 need only committed files; C2-C4 also need the exploratory kernel outputs.
Writes docs/exploratory/logic_diagnostics_results.json and prints a summary.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
M = ("TRUE", "FALSE", "UNCERTAIN")
RNG_SEED, BOOT = 0, 2000


def auc(pos, neg):
    s = np.concatenate([pos, neg])
    r = rankdata(s)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def auc_ci(pos, neg, rng):
    pos, neg = np.asarray(pos), np.asarray(neg)
    bs = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(BOOT)]
    return [round(float(auc(pos, neg)), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def norm(lp):
    v = np.array([lp[m] for m in M])
    e = np.exp(v - v.max())
    return e / e.sum()


def load_full(bench):
    run = {"justlogic": "baseline_logic", "proofwriter": "baseline_logic_pw"}[bench]
    comp = {json.loads(l)["uid"]: json.loads(l) for l in open(ROOT / f"results/rung0/{run}/completions.jsonl")}
    man = ROOT / f"data/manifests/{bench}_primary.csv"
    rows = []
    for m in csv.DictReader(open(man)):
        uid = m["uid"] if bench == "proofwriter" else f"{m['split']}:{m['id']}"
        stratum = int(m["qdep"] if bench == "proofwriter" else m["depth"])
        rows.append({"uid": uid, "stratum": stratum, "label": m["label"], "p": norm(comp[uid]["option_logprobs"]),
                     "lp": comp[uid]["option_logprobs"]})
    return rows


def c1(rows, rng):
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[r["stratum"]].append(r)
        groups["pooled"].append(r)
    for g, rs in groups.items():
        tf = lambda lab: [r["lp"]["TRUE"] - r["lp"]["FALSE"] for r in rs if r["label"] == lab]  # noqa: E731
        un = [r["p"][2] for r in rs if r["label"] == "UNCERTAIN"]
        nu = [r["p"][2] for r in rs if r["label"] != "UNCERTAIN"]
        out[str(g)] = {"auc_true_vs_false": auc_ci(tf("TRUE"), tf("FALSE"), rng),
                       "auc_uncertain_vs_rest": auc_ci(un, nu, rng),
                       "n_true": len(tf("TRUE")), "n_false": len(tf("FALSE")), "n_uncertain": len(un)}
    return out


def metrics(labels, preds):
    labels, preds = list(labels), list(preds)
    n = len(labels)
    cm = {g: {p: sum(1 for a, b in zip(labels, preds) if a == g and b == p) for p in M} for g in M}
    rec = [cm[g][g] / max(1, sum(cm[g].values())) for g in M]
    return {"n": n, "acc": round(sum(a == b for a, b in zip(labels, preds)) / n, 4),
            "balanced_acc": round(float(np.mean(rec)), 4),
            "pred_dist": {m: round(sum(p == m for p in preds) / n, 4) for m in M}, "confusion[gold][pred]": cm}


def c2(rows, cf):
    """cf: {'cf_na': p, 'cf_empty': p} normalised content-free option probabilities."""
    res = {"content_free_probs": {k: dict(zip(M, map(lambda x: round(float(x), 4), v))) for k, v in cf.items()}}
    res["uncalibrated"] = metrics([r["label"] for r in rows], [M[int(np.argmax(r["p"]))] for r in rows])
    variants = {"cal_na": cf["cf_na"], "cal_empty": cf["cf_empty"], "cal_mean": (cf["cf_na"] + cf["cf_empty"]) / 2}
    for name, pcf in variants.items():
        preds = [M[int(np.argmax(r["p"] / pcf))] for r in rows]
        res[name] = metrics([r["label"] for r in rows], preds)
        per = defaultdict(lambda: ([], []))
        for r, p in zip(rows, preds):
            per[r["stratum"]][0].append(r["label"])
            per[r["stratum"]][1].append(p)
        res[name]["per_stratum_acc"] = {str(s): metrics(*v)["acc"] for s, v in sorted(per.items())}
    return res


def read_variant(d):
    meta = json.loads((d / "meta.json").read_text())
    comp = [json.loads(l) for l in open(d / "completions.jsonl")]
    return meta, comp


def c3_c4(explore, bench, sub):
    gold = {r["uid"]: r["label"] for r in sub if r["benchmark"] == bench}
    out = {}
    for d in sorted(explore.glob(f"explore_{bench}_*")):
        if not (d / "completions.jsonl").exists():
            out[d.name] = "MISSING"
            continue
        meta, comp = read_variant(d)
        v = meta["variant"]
        if v.startswith("cf_"):
            continue
        km = meta["key_to_meaning"]
        rank_keys = [max(c["option_logprobs"], key=c["option_logprobs"].get) for c in comp]
        greedy = []
        for c in comp:
            first = c["completion"].split("\n", 1)[0].strip().upper()
            tok = next((k for k in km if first.startswith(k)), None)
            greedy.append(tok)
        labels = [gold[c["uid"]] for c in comp]
        rank_m = [km[k] for k in rank_keys]
        greedy_m = [km[k] if k else None for k in greedy]
        rec = {"n": len(comp),
               "rank": metrics(labels, rank_m),
               "greedy_acc": round(sum(a == b for a, b in zip(labels, greedy_m)) / len(comp), 4),
               "greedy_parse_fail": round(sum(g is None for g in greedy) / len(comp), 4)}
        if v.startswith("swap_"):
            rec["chosen_symbol_dist"] = {k: round(sum(x == k for x in rank_keys) / len(comp), 4) for k in km}
            rec["symbol_to_meaning"] = km
        if v.startswith("order_"):
            last_label = ["TRUE", "FALSE", "UNCERTAIN"][int(v[-1])]
            rec["last_shot_label"] = last_label
            rec["frac_pred_last_shot_label"] = rec["rank"]["pred_dist"][last_label]
        out[v] = rec
    for fam in ("swap_", "order_"):
        accs = [r["rank"]["acc"] for k, r in out.items() if isinstance(r, dict) and k.startswith(fam)]
        if accs:
            out[f"{fam}summary"] = {"rank_acc_mean": round(float(np.mean(accs)), 4),
                                    "rank_acc_min": min(accs), "rank_acc_max": max(accs), "k": len(accs)}
    return out


def c1b(bench, rng):
    """Plan addendum 1: model-free lexical baselines on the same items (benchmark diagnostics)."""
    import re
    import sys
    sys.path.insert(0, str(ROOT))
    from mew import justlogic, proofwriter
    mod = justlogic if bench == "justlogic" else proofwriter
    items = mod.load_items(Path("/tmp/mew_cache"), ROOT / f"data/manifests/{bench}_primary.csv")
    words = lambda t: set(re.findall(r"[a-z]+", t.lower()))  # noqa: E731
    neg = re.compile(r"\b(not|no|false|incorrect|misconception|wrong|doesn't)\b|does not", re.I)
    groups = defaultdict(list)
    for it in items:
        st = it["depth"] if bench == "justlogic" else it["qdep"]
        w = words(it["question"])
        overlap = len(w & words(it["paragraph"])) / max(1, len(w))
        r = {"label": it["label"], "overlap": overlap, "neg": len(neg.findall(it["question"]))}
        groups[st].append(r)
        groups["pooled"].append(r)
    out = {}
    for g, rs in groups.items():
        out[str(g)] = {
            "auc_uncertain_vs_rest_by_low_overlap": auc_ci([-r["overlap"] for r in rs if r["label"] == "UNCERTAIN"],
                                                            [-r["overlap"] for r in rs if r["label"] != "UNCERTAIN"], rng),
            "auc_true_vs_false_by_no_negation": auc_ci([-r["neg"] for r in rs if r["label"] == "TRUE"],
                                                        [-r["neg"] for r in rs if r["label"] == "FALSE"], rng)}
    return out


def c5():
    pw = list(csv.DictReader(open(ROOT / "data/manifests/proofwriter_primary.csv")))
    t = (ROOT / "data/frozen/proofwriter_prompt_3_shot.txt").read_text()
    return {"qdep_counts": dict(Counter(r["qdep"] for r in pw)),
            "label_counts": dict(Counter(r["label"] for r in pw)),
            "source_label_map": {"True": "TRUE", "False": "FALSE", "Unknown": "UNCERTAIN"},
            "prompt_shot_answers": [s.split("Answer: ")[1].split("\n")[0] for s in t.split("-----\n")[1:4]],
            "uncertain_shot_source": "train:AttNoneg-OWA-D2-571:Q8 (gold answer Unknown in source)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-dir", type=Path, default=None)
    a = ap.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    res = {"EXPLORATORY": True, "plan": "docs/exploratory/2026-10-19_logic_diagnostics_plan.md"}
    full = {b: load_full(b) for b in ("justlogic", "proofwriter")}
    res["C1_discrimination"] = {b: c1(rows, rng) for b, rows in full.items()}
    res["C1b_lexical_baselines"] = {b: c1b(b, rng) for b in ("justlogic", "proofwriter")}
    res["C5_proofwriter_checks"] = c5()
    if a.explore_dir:
        sub = list(csv.DictReader(open(ROOT / "docs/exploratory/logic_diag_subsample.csv")))
        res["C2_calibration"], res["C3_C4_variants"] = {}, {}
        for b, rows in full.items():
            cf = {}
            for v in ("cf_na", "cf_empty"):
                d = a.explore_dir / f"explore_{b}_{v}"
                if (d / "completions.jsonl").exists():
                    cf[v] = norm(read_variant(d)[1][0]["option_logprobs"])
            res["C2_calibration"][b] = c2(rows, cf) if len(cf) == 2 else "MISSING content-free runs"
            res["C3_C4_variants"][b] = c3_c4(a.explore_dir, b, sub)
    out = ROOT / "docs/exploratory/logic_diagnostics_results.json"
    out.write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()

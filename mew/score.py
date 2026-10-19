"""Offline scoring from raw completions (no GPU). Run in the scoring venv (bfcl-eval==2026.3.23).

  python -m mew.score justlogic outputs/raw/<run>   [--manifest data/manifests/justlogic_primary.csv]
  python -m mew.score proofwriter outputs/raw/<run>   [--manifest data/manifests/proofwriter_dev.csv]
  python -m mew.score bfcl outputs/raw/<run>

Writes <run>/scored.jsonl (per item) and <run>/summary.json, and prints the summary.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(run: Path) -> dict:
    recs = {}
    for line in open(run / "completions.jsonl"):
        r = json.loads(line)
        recs[r["uid"]] = r  # later line wins (resume duplicates are identical by determinism)
    return recs


def score_justlogic(run: Path, manifest: Path) -> dict:
    from scipy.stats import binomtest

    from mew.justlogic import parse_answer

    recs = _read(run)
    man = list(csv.DictReader(open(manifest)))
    rows = []
    for m in man:
        uid = f"{m['split']}:{m['id']}"
        r = recs.get(uid)
        if r is None:
            continue
        pred = parse_answer(r["completion"])
        lp = r.get("option_logprobs")
        rank = max(lp, key=lp.get) if lp else None
        rows.append({"uid": uid, "split": m["split"], "depth": int(m["depth"]), "label": m["label"],
                     "pred": pred, "correct": int(pred == m["label"]), "parse_fail": int(pred is None),
                     "rank_pred": rank, "rank_correct": int(rank == m["label"]) if rank else None})
    with open(run / "scored.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    def block(rs):
        n = len(rs)
        maj = max(sum(r["label"] == lab for r in rs) for lab in ("TRUE", "FALSE", "UNCERTAIN")) / n
        out = {"n": n, "acc": sum(r["correct"] for r in rs) / n, "majority_rate": maj,
               "parse_fail_rate": sum(r["parse_fail"] for r in rs) / n,
               "pred_dist": {k: sum(r["pred"] == k for r in rs) / n for k in ("TRUE", "FALSE", "UNCERTAIN", None)}}
        if rs and rs[0]["rank_correct"] is not None:
            out["rank_acc"] = sum(r["rank_correct"] for r in rs) / n
        return out

    summary = {"n_scored": len(rows), "n_manifest": len(man), "complete": len(rows) == len(man)}
    for name, sel in [("primary", lambda r: True), ("test_only", lambda r: r["split"] == "test")]:
        rs = [r for r in rows if sel(r)]
        if not rs:
            continue
        by = defaultdict(list)
        for r in rs:
            by[r["depth"]].append(r)
        summary[name] = {"per_depth": {d: block(by[d]) for d in sorted(by)}}
    # Rung-0 gate G-Logic (PREREGISTRATION.md §7): pooled depths 1-2 of the primary set.
    d12 = [r for r in rows if r["depth"] in (1, 2)]
    if d12:
        b = block(d12)
        k = sum(r["correct"] for r in d12)
        p0 = round(b["majority_rate"], 3)
        p = binomtest(k, len(d12), p0, alternative="greater").pvalue
        summary["gate_logic"] = {"n": len(d12), "correct": k, "acc": b["acc"], "p0": p0,
                                 "threshold": round(p0 + 0.05, 3), "binom_p_one_sided": p,
                                 "pass": bool(b["acc"] >= p0 + 0.05 and p < 0.05)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def score_proofwriter(run: Path, manifest: Path) -> dict:
    """ProofWriter fallback (Amendment A1): same parser/scorers as JustLogic, strata = QDep."""
    from scipy.stats import binomtest

    from mew.justlogic import parse_answer

    recs = _read(run)
    man = list(csv.DictReader(open(manifest)))
    rows = []
    for m in man:
        r = recs.get(m["uid"])
        if r is None:
            continue
        pred = parse_answer(r["completion"])
        lp = r.get("option_logprobs")
        rank = max(lp, key=lp.get) if lp else None
        rows.append({"uid": m["uid"], "theory_id": m["theory_id"], "qdep": int(m["qdep"]), "label": m["label"],
                     "pred": pred, "correct": int(pred == m["label"]), "parse_fail": int(pred is None),
                     "rank_pred": rank, "rank_correct": int(rank == m["label"]) if rank else None})
    with open(run / "scored.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    def block(rs):
        n = len(rs)
        maj = max(sum(r["label"] == lab for r in rs) for lab in ("TRUE", "FALSE", "UNCERTAIN")) / n
        out = {"n": n, "acc": sum(r["correct"] for r in rs) / n, "majority_rate": maj,
               "parse_fail_rate": sum(r["parse_fail"] for r in rs) / n,
               "pred_dist": {k: sum(r["pred"] == k for r in rs) / n for k in ("TRUE", "FALSE", "UNCERTAIN", None)}}
        if rs and rs[0]["rank_correct"] is not None:
            out["rank_acc"] = sum(r["rank_correct"] for r in rs) / n
        return out

    summary = {"n_scored": len(rows), "n_manifest": len(man), "complete": len(rows) == len(man),
               "per_qdep": {q: block([r for r in rows if r["qdep"] == q]) for q in (1, 2)
                            if any(r["qdep"] == q for r in rows)}}
    if rows:
        b = block(rows)
        k = sum(r["correct"] for r in rows)
        p0 = round(b["majority_rate"], 3)
        p = binomtest(k, len(rows), p0, alternative="greater").pvalue
        summary["gate_logic_fallback"] = {"n": len(rows), "correct": k, "acc": b["acc"], "p0": p0,
                                          "threshold": round(p0 + 0.05, 3), "binom_p_one_sided": p,
                                          "pass": bool(b["acc"] >= p0 + 0.05 and p < 0.05)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def score_bfcl(run: Path) -> dict:
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker
    from bfcl_eval.utils import is_function_calling_format_output, load_dataset_entry, load_ground_truth_entry

    from mew.bfcl import decode_tool_calls, load_items

    recs = _read(run)
    items = {it["uid"]: it for it in load_items(ROOT / "data/frozen/bfcl_items.jsonl")}
    cats = sorted({it["category"] for it in items.values()})
    rows = []
    for cat in cats:
        entries = {e["id"]: e for e in load_dataset_entry(cat)}
        gts = {g["id"]: g["ground_truth"] for g in load_ground_truth_entry(cat)} if items[next(
            u for u in items if items[u]["category"] == cat)]["scored"] else {}
        for uid, it in items.items():
            if it["category"] != cat or uid not in recs:
                continue
            comp = recs[uid]["completion"]
            try:
                calls = decode_tool_calls(comp, it["tools"])
                parsed = is_function_calling_format_output(calls)
            except ValueError:
                calls, parsed = None, False
            if it["scored"]:
                if parsed:
                    res = ast_checker(entries[uid]["function"], calls, gts[uid], Language.PYTHON, cat,
                                      "Qwen_Qwen3-0.6B-FC")  # config used only for underscore_to_dot=False
                    ok = res["valid"]
                    err = None if ok else res.get("error_type")
                else:
                    ok, err = False, "ast_decoder:decoder_failed"
            else:  # irrelevance: correct iff no function call is produced
                ok, err = not parsed, None if not parsed else "irrelevance:called_function"
            rows.append({"uid": uid, "category": cat, "scored": it["scored"], "parsed": int(parsed),
                         "correct": int(ok), "error_type": err})
    with open(run / "scored.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    per = {}
    for cat in cats:
        rs = [r for r in rows if r["category"] == cat]
        if rs:
            per[cat] = {"n": len(rs), "acc": sum(r["correct"] for r in rs) / len(rs),
                        "parse_rate": sum(r["parsed"] for r in rs) / len(rs)}
    sc = [r for r in rows if r["scored"]]
    n_expected = sum(it["scored"] for it in items.values())
    summary = {"n_scored": len(sc), "n_expected": n_expected, "complete": len(sc) == n_expected,
               "per_category": per}
    if sc:
        acc = sum(r["correct"] for r in sc) / len(sc)
        pr = sum(r["parsed"] for r in sc) / len(sc)
        summary["pooled_ast_acc"] = acc
        summary["parse_rate"] = pr
        summary["gate_agentic"] = {"parse_rate": pr, "pooled_ast_acc": acc,
                                   "pass": bool(pr >= 0.10 and acc >= 0.05)}
    (run / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["justlogic", "proofwriter", "bfcl"])
    ap.add_argument("run", type=Path)
    ap.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/justlogic_primary.csv")
    a = ap.parse_args()
    if a.task == "justlogic":
        s = score_justlogic(a.run, a.manifest)
    elif a.task == "proofwriter":
        m = a.manifest if a.manifest.name.startswith("proofwriter") else ROOT / "data/manifests/proofwriter_primary.csv"
        s = score_proofwriter(a.run, m)
    else:
        s = score_bfcl(a.run)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

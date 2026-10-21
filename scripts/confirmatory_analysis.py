"""Confirmatory Rung-1 analysis, Agentic target (PREREGISTRATION.md §8, as amended by A2/A3).

Run ONCE after all seeds exist:
  python scripts/confirmatory_analysis.py --runs results/rung1 --out results/rung1/confirmatory.json

§8, as implemented:
- Unit: item × checkpoint correctness on the 2,351 scored BFCL items.
- Arm score: per item, mean correctness over that arm's seeds. Baseline is one deterministic run.
- Δ(A, B) = mean item score of A − B, in percentage points.
- Hierarchical paired bootstrap, 10,000 replicates, RNG seed 20261019. Each replicate resamples the
  3 seeds with replacement within each tuned arm, then resamples items with replacement within each
  BFCL category, then recomputes Δ. One-sided p = fraction of replicates with Δ <= 0.
- Holm–Bonferroni over {H1_Agentic, H2_Agentic} (Logic stopped at Rung 0), family-wise α = 0.05.
- H1 supported iff Δ(T, B) > 0 and Holm-adjusted p < .05.
- H2 supported iff Δ(T, C) > 0, Holm-adjusted p < .05, and T > C in >= 2 of 3 seed-matched pairs.
- Headline only if H1 and H2 are both supported.
- Null definitions: clean null if H2 is unsupported and the two-sided 90% CI of Δ(T, C) lies within ±1 pp;
  harm if Δ(T, C) < 0 with the 95% CI excluding 0.
Scoring: A3 (truncate at first <|im_end|>) is primary; the as-run scoring is computed identically and
reported alongside.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2)
B_REPS, RNG_SEED, ALPHA = 10_000, 20261019, 0.05


def load(path: Path) -> dict:
    return {r["uid"]: r for r in (json.loads(l) for l in open(path)) if r["scored"]}


def matrices(runs: Path, suffix: str):
    base = load(ROOT / f"results/rung0/baseline_agentic/scored{suffix}.jsonl")
    uids = sorted(base)
    cats = np.array([base[u]["category"] for u in uids])
    B = np.array([base[u]["correct"] for u in uids], dtype=float)
    arms = {}
    for arm in ("treatment", "control"):
        rows = []
        for s in SEEDS:
            sc = load(runs / f"interim_s{s}" / f"eval_agentic_{arm}_s{s}" / f"scored{suffix}.jsonl")
            assert sorted(sc) == uids, (arm, s)
            rows.append([sc[u]["correct"] for u in uids])
        arms[arm] = np.array(rows, dtype=float)  # seeds × items
    return uids, cats, B, arms["treatment"], arms["control"]


def analyse(runs: Path, suffix: str) -> dict:
    uids, cats, B, T, C = matrices(runs, suffix)
    rng = np.random.default_rng(RNG_SEED)
    strata = [np.where(cats == c)[0] for c in sorted(set(cats))]
    point = {"T_B": (T.mean(0) - B).mean() * 100, "C_B": (C.mean(0) - B).mean() * 100,
             "T_C": (T.mean(0) - C.mean(0)).mean() * 100}
    boots = defaultdict(list)
    for _ in range(B_REPS):
        ts = T[rng.integers(0, len(SEEDS), len(SEEDS))].mean(0)
        cs = C[rng.integers(0, len(SEEDS), len(SEEDS))].mean(0)
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata])
        boots["T_B"].append((ts[idx] - B[idx]).mean() * 100)
        boots["C_B"].append((cs[idx] - B[idx]).mean() * 100)
        boots["T_C"].append((ts[idx] - cs[idx]).mean() * 100)
    out = {"n_items": len(uids), "per_arm_acc": {
        "baseline": B.mean(), **{f"treatment_s{s}": T[i].mean() for i, s in enumerate(SEEDS)},
        **{f"control_s{s}": C[i].mean() for i, s in enumerate(SEEDS)},
        "treatment_mean": T.mean(), "control_mean": C.mean()}}
    for k, v in boots.items():
        v = np.array(v)
        out[k] = {"delta_pp": point[k], "p_one_sided": float((v <= 0).mean()),
                  "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
                  "ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))]}
    # Holm over the two registered tests
    ps = sorted([("H1", out["T_B"]["p_one_sided"]), ("H2", out["T_C"]["p_one_sided"])], key=lambda x: x[1])
    adj, running = {}, 0.0
    for i, (h, p) in enumerate(ps):
        running = max(running, min(1.0, p * (len(ps) - i)))
        adj[h] = running
    pairs = [bool(T[i].mean() > C[i].mean()) for i in range(len(SEEDS))]
    h1 = out["T_B"]["delta_pp"] > 0 and adj["H1"] < ALPHA
    h2 = out["T_C"]["delta_pp"] > 0 and adj["H2"] < ALPHA and sum(pairs) >= 2
    lo, hi = out["T_C"]["ci90"]
    out["decision"] = {
        "holm_adjusted_p": adj, "seed_pairs_T_gt_C": pairs,
        "H1_supported": bool(h1), "H2_supported": bool(h2), "headline_claim": bool(h1 and h2),
        "H2_clean_null": bool(not h2 and -1.0 <= lo and hi <= 1.0),
        "H2_harm": bool(out["T_C"]["delta_pp"] < 0 and out["T_C"]["ci95"][1] < 0),
        "H1_harm": bool(out["T_B"]["delta_pp"] < 0 and out["T_B"]["ci95"][1] < 0)}
    # descriptive per-category deltas (point estimates, seed-mean)
    out["per_category_delta_pp"] = {
        c: {"T_B": float((T.mean(0)[cats == c] - B[cats == c]).mean() * 100),
            "C_B": float((C.mean(0)[cats == c] - B[cats == c]).mean() * 100),
            "T_C": float((T.mean(0)[cats == c] - C.mean(0)[cats == c]).mean() * 100),
            "n": int((cats == c).sum())} for c in sorted(set(cats))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "results/rung1")
    ap.add_argument("--out", type=Path, default=ROOT / "results/rung1/confirmatory.json")
    a = ap.parse_args()
    res = {"primary_A3": analyse(a.runs, "_a3"), "as_run": analyse(a.runs, "")}
    a.out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()

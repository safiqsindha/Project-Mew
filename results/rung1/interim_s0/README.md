# INTERIM — seed 0 only (1 of 3 seeds per arm). Not a result.

Shown under the interim-look rule (RUN_LOG 2026-10-19). Seeds 1–2 run regardless. No significance is claimed. The confirmatory analysis is the registered §8 test on all seeds, scored under Amendment A3 (with the as-run scoring alongside).

| BFCL, 2,351 scored items | Baseline | Control s0 | Treatment s0 |
|---|---|---|---|
| **Pooled AST accuracy, A3 scoring (primary)** | .519 | .372 | .524 |
| Pooled AST accuracy, as-run scoring (stop-token bug) | .519 | .054 | .124 |
| Parse rate (A3) | .767 | .985 | .978 |
| Outputs that run past `<|im_end|>` | .009 | .516 | .885 |
| irrelevance / live_irrelevance accuracy (A3, descriptive) | .796 / .827 | .062 / .165 | .133 / .283 |

Item-level, A3:
- Treatment vs Baseline: 364 fixed, 351 broke (net +0.6 pp).
- Control vs Baseline: 241 fixed, 586 broke (−14.7 pp).
- Treatment vs Control: 497 fixed, 139 broke (+15.2 pp).

**Reading (interim):** at seed 0 the Treatment–Control gap comes almost entirely from Control dropping *below* Baseline. Treatment is roughly level with Baseline. This is the "Control hurt vs Treatment helped" pattern flagged in the pre-unblinding audit (A1), and it has to be reported as such. Both tuned arms almost stop declining to call tools (irrelevance accuracy collapses).

Adapter weights are not committed. Their SHA-256 hashes are in each `train_*` folder.

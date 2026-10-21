# Rung 1 — Agentic target, confirmatory result (3 seeds per arm)

Analysis: `scripts/confirmatory_analysis.py`, committed before its single run (c2d7492). It implements PREREGISTRATION §8 as amended by A2 and A3. Full output: `confirmatory.json`. Per-seed raw completions, scores and training metadata: `interim_s0/`, `interim_s1/`, `interim_s2/`.

Model: Qwen3.5-0.8B (post-trained). Benchmark: BFCL single-turn AST, 2,351 scored items. Test: hierarchical paired bootstrap (10,000 reps), Holm over {H1, H2}.

## Primary scoring (Amendment A3: outputs truncated at the first `<|im_end|>`)

| | Baseline | Control s0 / s1 / s2 (mean) | Treatment s0 / s1 / s2 (mean) |
|---|---|---|---|
| Pooled AST accuracy | .519 | .372 / .476 / .435 (.428) | .524 / .535 / .533 (.531) |

| Comparison | Δ (pp) | 95% CI | One-sided p | Holm p |
|---|---|---|---|---|
| **H1: Treatment − Baseline** | **+1.18** | [−0.96, +3.32] | .143 | .143 |
| **H2: Treatment − Control** | **+10.31** | [+5.47, +15.50] | < .0001 | < .0001 |
| Control − Baseline (descriptive) | −9.13 | [−14.46, −4.15] | — | — |

Treatment beats Control in 3/3 seed-matched pairs.

**Decision (registered rules):**
- **H1 is not supported.** The result is inconclusive, not harm: the CI spans 0.
- **H2 is supported.**
- **Headline claim ("structure helps, beyond game data"): NOT supported**, because it requires H1 and H2 together.

**Reading:**
- The Treatment–Control gap comes almost entirely from **Control falling below Baseline** (−9.1 pp, CI excludes 0). Treatment stays level with Baseline (+1.2 pp, n.s.).
- So the evidence says that *destroying* the order of the constraint chains damages function calling, and that *preserving* it does no damage. It does **not** show that ordered chains improve the model.
- This is the "Control hurt vs Treatment helped" pattern flagged in the pre-unblinding audit (A1). The shuffled Control is incoherent data (events carry turn labels that don't match them), not merely unordered data.

## As-run scoring (registered stop rule not applied; reported per A3)

| Comparison | Δ (pp) | 95% CI | One-sided p | Holm p |
|---|---|---|---|---|
| H1: T − B | −29.12 | [−38.83, −21.62] | 1.0 | 1.0 |
| H2: T − C | +7.78 | [−10.45, +22.96] | .199 | .397 |
| C − B | −36.91 | [−47.55, −19.18] | — | — |

Neither hypothesis is supported, and the harm rule is met for H1. Tuned models rarely stop at `<|im_end|>`: the share of outputs that run past it is .35–.95 for tuned models against .009 for Baseline. Under the as-run harness, the extra generated turns are scored as wrong.

## Descriptive outcomes (not confirmatory)

**Per category (A3, seed-mean Δ pp):**

| Category | n | T − B | C − B | T − C |
|---|---|---|---|---|
| simple_python | 400 | +1.2 | −1.3 | +2.5 |
| multiple | 200 | +8.5 | −6.0 | +14.5 |
| parallel | 200 | +22.2 | +20.3 | +1.8 |
| parallel_multiple | 200 | −0.7 | −9.8 | +9.2 |
| live_simple | 258 | −5.9 | −13.8 | +7.9 |
| live_multiple | 1,053 | −1.4 | −16.7 | +15.3 |
| live_parallel | 16 | −8.3 | −18.8 | +10.4 |
| live_parallel_multiple | 24 | −23.6 | −16.7 | −6.9 |

**Abstention collapse.** All six tuned models almost stop declining to call tools. irrelevance accuracy is .796 for Baseline against .05–.42 tuned, and live_irrelevance .827 against .11–.35. Parse rate rises from .767 to .96–.99. Fine-tuning on assistant-turn continuations, all of which "answer", erodes abstention in both arms.

**Training.** Final held-out chain loss: Treatment 0.053, Control 0.077, consistent across seeds. Ordered chains are easier to model.

## Caveats
- Amendments:
  - A1 (Logic fallback source) doesn't affect this target.
  - A2 (micro-batch 2 × 8) applies to all tuned runs.
  - A3 (stop at `<|im_end|>`) is results-informed: it was found after the interim seed-0 look and committed before corrected numbers were computed. Both scorings are reported in full.
- One base model (0.8B) and one benchmark. Effect sizes say nothing about larger models or other tasks.
- Consistent with the program's prior evidence, nothing here is evidence of capability *installation*.

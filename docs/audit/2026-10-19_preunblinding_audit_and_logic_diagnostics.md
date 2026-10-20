# Pre-unblinding audit and exploratory logic diagnostics (2026-10-19)

No tuned-model BFCL result had been produced or inspected when this was written. The seed-0 session `mew-rung1-at0-ac0` was still training.

---

## PART A — Confirmatory audit (before BFCL unblinding)

**A1. Shuffle integrity.** Ran a per-chain comparison of `data/sft/{treatment,control}_train.jsonl` (n = 4,560 pairs).
- **Identical in all 4,560 pairs:** the header line, the step count in prompt and in target, the multiset of step contents, the sequence of `turn=` labels, and the 1..n step numbering in Control.
- Every target starts with `Step` and ends with a newline. Chain IDs aren't in the text, and there are no answers in this data.
- So the shuffle moves only typed step bodies.
- **Flag:** the type mix differs by position, by construction. Treatment's first step is opening-game ToolAvailability/SubGoalTransition 69% of the time; Control's first step is ResourceBudget 50% of the time. The *target* (loss-bearing) mix also differs slightly:

  | Target share | ResourceBudget | ToolAvailability | SubGoalTransition | InformationState | OptimizationCriterion | CoordinationDependency |
  |---|---|---|---|---|---|---|
  | Treatment | .529 | .246 | .094 | .072 | .040 | .019 |
  | Control | .489 | .248 | .112 | .096 | .034 | .022 |

  - Final-position mixes: RB .584 / .572; SubGoal .091 / .061.
  - Control also pairs content with turn labels that no longer fit it (e.g. a turn-5 event labelled with a later turn).
  - The two arms are therefore not trained to predict identical things at the end of the sequence. That is the registered manipulation, but it means "Control" is *incoherent* data, not just *unordered* data.

**A2. Token counts, example counts, loss masking.**
- Same 4,560 train / 246 monitor examples per arm.
- Full-example tokens are identical per chain and in total (raw text: 5,196,671 per arm).
- Loss-bearing tokens under the chat template: Treatment 2,604,330, Control 2,607,274 (+0.11%; registered tolerance 0.5%).
- Masking is the same code path for both arms (`labels = −100 × prompt + target ids`).
- **Training loss:** not yet available. The 20-step smoke run gave Treatment 0.263 → 0.153 and Control 0.320 → 0.204, with held-out loss 0.136 vs 0.189. Treatment is easier to fit, as expected for ordered data.

**A3. Seeds and hyperparameters.**
- 3 seeds per arm (0, 1, 2). Only seed 0 is running so far.
- Identical for both arms:
  - LoRA: r 16, α 32, dropout 0.05, 150 modules (10,223,616 params)
  - Optimiser: AdamW, LR 2e-4, cosine schedule, 3% warmup, weight decay 0
  - Batch: micro-batch 2 × accumulation 8 = effective 16 (Amendment A2; registered 4 × 4 ran out of memory)
  - Schedule: 2 epochs = 570 optimiser steps; fp16 autocast on fp32 weights; gradient checkpointing
  - Checkpoint rule: final checkpoint only
- **No mismatch between arms.** The only departure from the original §5 is A2, which applies to both arms.

**A4. Registered test.** It is specified, so no stop is needed.
- PREREGISTRATION §8: hierarchical paired bootstrap, 10,000 replicates, RNG seed 20261019. It resamples seeds within arm and items within each BFCL category, with one-sided p = P(Δ ≤ 0).
- Holm–Bonferroni across the tests. With Logic stopped, the family is **{H1_Agentic, H2_Agentic}**.
- H2 also requires Treatment > Control in ≥ 2 of 3 seed-matched pairs.

**A5. BFCL denominator.** All 2,351 scored items (§6.2; `mew/score.py`). An item that fails to parse counts as incorrect. Accuracy conditional on parse will be reported as descriptive only.

**A6 (added). Adapter eval path.** Tuned arms are evaluated through a PEFT-wrapped model, and the baseline through the plain model. Check: an untrained (B = 0) adapter vs the plain model on the 40 BFCL dev items, in the same kernel environment (`mew-pathcheck`). **Result: pending.**

---

## PART B — BFCL reporting

Blocked: no tuned checkpoint has finished. It will be reported with the A4 test, plus the secondary descriptives (parse rate, accuracy conditional on parse, fixed/broke counts, all three pairwise gaps) once all 3 seeds per arm exist.

---

## PART C — EXPLORATORY logic diagnostics (not confirmatory; the Logic gate stays FAILED)

**Plan:** `docs/exploratory/2026-10-19_logic_diagnostics_plan.md`, committed before running, plus Addendum 1, committed before C1b ran.
- Every variant run is reported below.
- Full raw outputs: `results/exploratory/logic_diagnostics/`. Numbers: `docs/exploratory/logic_diagnostics_results.json`.
- **Harness check:** the identity shot order reproduces the Rung-0 baseline exactly (700/700 and 600/600 completions; max |Δ log-prob| 5e-5).

**C1. Discrimination** (full sets, saved log-probs; AUC [95% bootstrap CI])

| | T vs F: logP(T) − logP(F) | UNCERTAIN vs rest: P(U) |
|---|---|---|
| JustLogic pooled | .647 [.631, .663] | .795 [.784, .806] |
| JustLogic per depth 1–7 | .668 · .698 · .641 · .657 · .645 · .646 · .592 | .812 · .819 · .803 · .803 · .802 · .785 · .761 |
| ProofWriter QDep 1 / 2 | .471 / .473 | .496 / .475 |

→ On JustLogic the log-probs carry label information that argmax hides (UNCERTAIN is never the argmax). On ProofWriter there is none: the CIs sit just below 0.5.

**C1b. Model-free lexical baselines** (Addendum 1; diagnostics of the items, not the model)

| | UNCERTAIN vs rest, by low statement–paragraph word overlap | T vs F, by absence of negation words |
|---|---|---|
| JustLogic pooled | **.950** [.944, .956] | **.684** [.671, .697] |
| ProofWriter pooled | .541 [.531, .552] | **.902** [.895, .909] |

→ The model's JustLogic discrimination (.795 / .647) is **below** trivial surface cues (.950 / .684), so it is not evidence of deduction. On ProofWriter a negation cue alone separates T/F at .90, yet the model scores .47: it doesn't even exploit that cue.

**C2. Content-free prior and contextual calibration**

| | P(T, F, U), query = `N/A` | P(T, F, U), query = empty |
|---|---|---|
| JustLogic | .317, .144, .539 | .487, .390, .123 |
| ProofWriter | .141, .086, .773 | .467, .278, .256 |

→ The "prior" depends heavily on the filler, so calibration is ill-determined.

Calibrated results on the full sets (all variants):

| | Accuracy | Balanced accuracy | Predicted T / F / U |
|---|---|---|---|
| JustLogic, uncalibrated | .354 | .356 | .77 / .23 / .00 |
| JustLogic, cal N/A | .384 | .389 | .10 / .90 / .00 |
| JustLogic, cal empty | .384 | .387 | .56 / .44 / .004 |
| JustLogic, cal mean | .403 | .408 | .36 / .64 / .00 |
| ProofWriter, uncalibrated | .312 | .294 | .85 / .15 / .00 |
| ProofWriter, cal N/A / empty / mean | .334 / .338 / .337 | .315 / .319 / .318 | ~.53 / .47 / .00 |

Confusion matrix (JustLogic, cal mean): gold-U → pred T 406, pred F 1,948, pred U 0. ProofWriter's calibrated confusion is near-uniform between T and F.

→ Calibration trades a TRUE bias for a FALSE bias (+3–5 pp on JustLogic, +2–3 pp on ProofWriter). It **never** produces UNCERTAIN.

**C3. Label swap** (subsample; single-token symbols; rank-classification)

| Rotation | JustLogic acc / bal | Chosen meaning T / F / U | ProofWriter acc / bal | Chosen meaning T / F / U |
|---|---|---|---|---|
| R0 (A=T, B=F, C=U) | .396 / .401 | .52 / .41 / .07 | .402 / .345 | .61 / .39 / .00 |
| R1 (A=F, B=U, C=T) | .343 / .357 | .88 / .12 / .00 | .400 / .344 | .65 / .36 / .00 |
| R2 (A=U, B=T, C=F) | .350 / .354 | .45 / .55 / .00 | .473 / .404 | .30 / .70 / .00 |
| Mean (range) | .363 (.343–.396) | | .425 (.400–.473) | |

Greedy decoding gives identical accuracy, with 0 parse failures.

→ The symbol mapped to UNCERTAIN is almost never chosen (≤ 7%), even when it is a single token. So **UNCERTAIN avoidance is not a token-length artifact.** The TRUE/FALSE split moves with the rotation, so choices partly follow the symbol, not the meaning.

Caveat: the UNCERTAIN definition line is always listed third, so "avoids the meaning UNCERTAIN" and "avoids the third-listed option" can't be told apart here.

**C4. Shot order** (all 6 orderings; subsample)

| Order (labels) | JustLogic acc / pred T / pred U | ProofWriter acc / pred T / pred U |
|---|---|---|
| T F U (registered) | .354 / .79 / .00 | .353 / .85 / .00 |
| T U F | .336 / .83 / .00 | .360 / .88 / .00 |
| F T U | .376 / .59 / .00 | .390 / .91 / .00 |
| F U T | .397 / .62 / .00 | .403 / .87 / .00 |
| U T F | .363 / .65 / .001 | .377 / .91 / .00 |
| U F T | .404 / .66 / .00 | .375 / .88 / .00 |
| Mean (range) | .372 (.336–.404) | .376 (.353–.403) |

→ There is mild recency: TRUE predictions rise when the last shot is TRUE. UNCERTAIN is never predicted under any ordering, including when the UNCERTAIN shot is last. Order moves accuracy by ±3–4 pp, and the best of 6 is a selected maximum, not a result.

**C5. ProofWriter set checks.**
- The primary set is QDep 1: 6,079 and QDep 2: 4,079. No QDep 0 or deeper items.
- Gold: TRUE 3,588 / FALSE 3,588 / UNCERTAIN 2,982, mapped from True/False/Unknown.
- The prompt's UNCERTAIN shot is `AttNoneg-OWA-D2-571:Q8`, whose source answer is Unknown.

**Overall exploratory reading:**
- The Base model's logic failure holds up across label format, token length, shot order and calibration. Across every variant, accuracy stays between .34 and .47, and UNCERTAIN is essentially never chosen.
- The JustLogic signal that does exist doesn't beat surface heuristics.
- Nothing here argues for revisiting the gate. The Logic target stays stopped.

---

## Blocks to trusting a BFCL result

1. **Adapter-path equivalence (A6):** pending. If zero-adapter ≠ plain model on the dev items, H1 (Treatment vs untuned Baseline) is confounded by the eval path. H2 isn't, because both tuned arms use the same path.
2. **Control is "incoherent", not just "unordered" (A1):** step content no longer fits its turn labels, and the target type-mix shifts slightly. A Treatment > Control result could reflect Control *hurting* rather than structure *helping*. All three pairwise gaps (T−B, C−B, T−C) must be reported together.
3. **Only seed 0 is in progress.** H2 needs all 3 seeds per arm, and no confirmatory statement can be made before that.
4. **Environment drift:** the tuned-arm eval kernels install peft 0.21.2 and remove the image's torchao, which the Rung-0 baseline eval did not. A6 covers this for the dev items.
5. **Amendment A2** (micro-batch 2 × 8) applies to all tuned runs and has to be cited in any write-up.

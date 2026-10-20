# EXPLORATORY plan — Logic-target diagnostics (2026-10-19)

**Status: EXPLORATORY.**
- These analyses are not pre-registered and are not confirmatory.
- They do not re-adjudicate the Rung-0 Logic gates. Both failed (JustLogic .362 at depths 1–2; ProofWriter fallback .311), and the Logic target stays stopped.
- No new training. No LoRA checkpoint is evaluated on any logic task.
- Every variant listed here is run and reported, including uninformative ones.

This file is committed before any variant below is run.

**Question:** is the Base model's failure mode a real absence of discrimination, or partly a protocol artifact? The failure mode is 83–87% TRUE predictions and never UNCERTAIN. The candidate artifacts are label prior, label-token length (" UNCERTAIN" is 3 tokens, " TRUE"/" FALSE" are 1), and shot order.

**Fixed for every run:**
- Model: `Qwen/Qwen3.5-0.8B-Base` @ `dc7cdfe2…`.
- Settings: fp32, batch 16, greedy, max_new_tokens 8, Tesla T4.
- Generation and option-scoring code: `mew/generate.py` (unchanged).
- "Option probabilities" means the three options' summed log-probabilities, softmax-normalised over the three options.

## Item sets
- **Full sets** (C1, C2 calibration): the saved Rung-0 baseline outputs. No re-run.
  - `results/rung0/baseline_logic` (JustLogic, n = 6,927)
  - `results/rung0/baseline_logic_pw` (ProofWriter, n = 10,158)
- **Diagnostic subsample** (C3, C4; GPU re-runs), drawn with `random.Random(0)` from the primary manifests:
  - JustLogic: 100 items per depth, n = 700.
  - ProofWriter: 300 items per QDep, n = 600.
  - Frozen to `docs/exploratory/logic_diag_subsample.csv` before running.

## C1 — Discrimination (local, saved log-probs, full sets)
- On TRUE-gold vs FALSE-gold items: AUC of logP(TRUE) − logP(FALSE). Reported per depth (JustLogic 1–7) and per QDep (ProofWriter 1–2), plus pooled.
- 95% CI by item bootstrap, 2,000 replicates, seed 0.
- One-vs-rest AUC for UNCERTAIN, using normalised P(UNCERTAIN). Same strata and CI.

## C2 — Content-free prior and contextual calibration (Zhao et al. 2021)
- Run the exact registered 3-shot prompt of each benchmark with the query paragraph **and** statement both replaced by (i) `N/A` and (ii) the empty string. That's 4 prompts.
- Report normalised P(TRUE), P(FALSE), P(UNCERTAIN) for each.
- Calibrate the full-set option probabilities with W = diag(p_cf)⁻¹, then argmax. Three variants per benchmark, all reported: p_cf from `N/A`, from empty, and from their mean.
- For each variant: accuracy, balanced accuracy, prediction distribution, 3×3 confusion matrix. Pooled and per depth/QDep.

## C3 — Label-swap (subsample, GPU)
- Replace the label words with single-token symbols ` A`/` B`/` C` in three places: the option-definition lines, the "Output only …" line, and the three shot answers.
- The definition lines keep their meaning order (lead to / contradict / neither). The question sentence ("true, false, or uncertain?") is unchanged.
- Three rotations:
  - R0: A = TRUE, B = FALSE, C = UNCERTAIN
  - R1: A = FALSE, B = UNCERTAIN, C = TRUE
  - R2: A = UNCERTAIN, B = TRUE, C = FALSE
- Scorers: rank-classification over the three single-token symbols (primary for this diagnostic, since it avoids the length confound), plus greedy first-symbol parse.
- Report per rotation:
  - accuracy (symbol mapped back to meaning)
  - distribution of chosen *symbols* and of chosen *meanings*
  - mean and spread across rotations

  If the choices follow the symbol, the model has a position/symbol prior. If they follow the meaning, the model is reading the definitions.

## C4 — Shot order (subsample, GPU)
- Original shots per benchmark: TRUE, FALSE, UNCERTAIN (label-balanced, one each).
- Run all 6 permutations of the three shot blocks; the identity permutation is the reference.
- Labels and text otherwise unchanged; original label words; both scorers.
- Report per ordering: accuracy, prediction distribution, and whether the last shot's label dominates (recency). Also mean and spread across orderings.

## C5 — ProofWriter set checks (local)
- Confirm the OWA depth-2 primary set contains no QDep 0 items and none deeper than QDep 2.
- Confirm "Unknown" gold labels map to UNCERTAIN, and that the prompt's UNCERTAIN shot is an "Unknown" item.

## Not done (scope)
- No new prompts beyond those listed.
- No other models.
- No re-gating, and no use of any of this to restart the Logic target.

## Addendum 1 — 2026-10-19 (added after C1 results, before running it)

C1 showed JustLogic AUC(UNCERTAIN vs rest) = 0.79, against ~0.48 on ProofWriter. A surface heuristic could produce that: UNCERTAIN statements may share fewer content words with the paragraph. To check, add **C1b**, a model-free lexical baseline on the same full sets:
- Score each item by the fraction of the statement's lowercase alphabetic word types that also appear in the paragraph.
- Report AUC (UNCERTAIN vs rest, higher score = less UNCERTAIN, so the score is negated), with the same bootstrap and strata.
- Also report AUC (TRUE vs FALSE) using a negation-cue score: count of {not, no, false, incorrect, misconception, wrong, does not, doesn't} in the statement, negated. Report as is.

These are diagnostics of the benchmark items, not of the model. They are run once and reported regardless of outcome.

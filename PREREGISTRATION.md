# Project Mew — Pre-registration (Rung 0 & Rung 1)

**Status:** LOCKED. Any change goes in **Amendments** at the bottom, with a date and a reason. The text above Amendments is never edited.
**Drafted:** 2026-10-19
**Locked:** 2026-10-19. The lock is the git commit that first added this file; its hash is recorded in `RUN_LOG.md`.
**Author:** Safiq Sindha (drafted with Claude Opus 5.5)

No eval or training numbers exist for this project as of drafting.

---

## 1. Hypotheses

Fine-tuning a small LM on constraint-chain-structured traces from Pokémon decision data improves out-of-domain reasoning, and the *structure* is the active ingredient rather than exposure to game data alone.

For each target *t* ∈ {Logic, Agentic}:

- **H1_t (tuned > baseline):** the Treatment arm scores higher than the untuned Baseline on target *t*.
- **H2_t (tuned > control):** the Treatment arm scores higher than the Control arm on target *t*.

**H2 is the only publishable claim.** H1 alone says only that game data helps, which is already established (Game-RL / Code2Logic).

Throughout, "structure" means exactly one thing: **the chronological ordering of the typed constraint steps within a chain.** The Control keeps the constraint types, step content, step count, turn values and token count. It destroys only the order (§4.2). H2 therefore tests whether *temporally coherent* constraint sequences carry signal beyond the same constraints in random order. It does **not** test whether the six-type labelling itself matters, because both arms carry the labels.

**Framing constraint:** prior program evidence (Cohen's κ ≈ −0.001 across 4,252 chains, Ditto V5.3) supports *cheaper elicitation of latent capability*, not installation of new capability. No result here will be reported as the model "knowing more."

## 2. Models

| Target | Model | HF revision (pinned) |
|---|---|---|
| Logic | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` |
| Agentic | `Qwen/Qwen3.5-0.8B` (post-trained; Qwen ships no `-Instruct` suffix) | `2fc06364715b967f1860aea9cf38778875588b17` |

**The two targets use different base models on purpose, and every writeup must say so.** A base model has never learned a tool-call format, so it would score ~0 on BFCL for reasons unrelated to the hypothesis. Logic uses Base so the result isn't confounded by instruction tuning. Agentic uses the post-trained model.

Both are multimodal checkpoints (`Qwen3_5ForConditionalGeneration`). Only the language model is used: text-only inputs, vision tower untouched and not adapted. Qwen3.5-0.8B runs in non-thinking mode by default, and all runs use non-thinking mode (`enable_thinking=False`).

## 3. Training data

**Source:** `Ditto V1/chains/real/`, 4,806 chains, the same corpus as the Ditto V5.3 κ analysis. These are Gen 9 OU Pokémon Showdown matches with one chain per player perspective. Each chain has 20–40 typed steps (median 40), 186,645 steps in total.

Step-type counts: ResourceBudget 94,235 · ToolAvailability 46,485 · SubGoalTransition 19,215 · InformationState 15,542 · OptimizationCriterion 6,866 · CoordinationDependency 4,302.

`real_v2/` (4,860 chains) is not used.

**Integrity:** before any training, a SHA-256 manifest of every chain file used (both arms) is committed to `data/manifests/`. `RUN_LOG.md` records the dataset hash (SHA-256 of the sorted manifest).

**Train / monitor split:** a chain goes to the *monitor* set iff `int(sha256(match_id), 16) % 20 == 0`. That's about 5% of chains, used only for logging held-out training loss and never for checkpoint selection. All other chains are *train*. Splitting by `match_id` keeps both perspectives of a match on the same side. Both arms use the identical split.

**SFT objective (prefix → continuation):** each chain's `rendered` text is split at its pre-computed `cutoff_k` (k = 20 for 4,180 chains; ⌊n/2⌋ or nearby otherwise).
- *Prompt* = header line + steps 1..k.
- *Target* = steps k+1..n.
- Loss is computed on target tokens only.
- No packing. One chain is one example.

## 4. Arms

All arms in a target share the same base model, the same eval harness and prompts, the same generation config, and the same hardware type.

| Arm | Training data |
|---|---|
| Baseline | none (untuned base model for that target) |
| Control | the same 4,806 chains, step order destroyed (§4.2) |
| Treatment | the same 4,806 chains, original chronological order |

### 4.1 Treatment
`rendered` from `chains/real/<chain_id>.jsonl`, used as-is.

### 4.2 Control: structure-destruction procedure
`rendered` from `chains/shuffled/<chain_id>_shuffled_42.jsonl`. These files were produced by `src/shuffler.py::shuffle_chain(chain, seed=42)` (Project-Ditto) before this study:
1. Draw a uniform random permutation of the chain's constraint steps (`random.Random(42).shuffle`).
2. Reassign the original timestamps in sorted order to the permuted steps. Turn numbers therefore stay non-decreasing, so the shuffle can't be detected from timestamps.
3. Re-render with the same renderer. Step indices are renumbered 1..n.
4. Apply the same permutation to `active_pair_by_step`.

Shuffle seed 42 is fixed for all Control runs. Training seeds vary (§5), exactly as in Treatment, so both arms have the same variance structure.

**Measured match (Qwen3.5 tokenizer, all 4,806 pairs, 2026-10-19):**
- multiset of step contents (excluding the `Step i (turn=t):` header): identical for 4,806 / 4,806 pairs
- total tokens per chain: identical for every pair (Δ = 0); corpus 5,476,880 tokens per arm
- example count, source traces, sequence-length distribution: identical by construction
- loss-bearing (target) tokens: Treatment 2,740,223 vs Control 2,743,438 (+0.12%). The difference comes from which steps land after `cutoff_k`. **Pre-registered tolerance: ≤ 0.5% (met).**

`scripts/build_sft_data.py` (to be written) reproduces both arms from the chain files and asserts all of the above. It aborts if any check fails.

## 5. Training configuration

The same configuration applies to every tuned run in both arms and both targets.

**LoRA:**
- r = 16, α = 32, dropout = 0.05, no bias
- Target modules (regex): `^model\.language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$`
- Not adapted: embeddings, LM head, vision tower, MTP head

**Optimisation:**
- AdamW, lr 2e-4, cosine schedule, 3% warmup, weight decay 0
- Per-device batch 4 × gradient accumulation 4 (effective 16)
- 2 epochs, max sequence length 1,536 tokens. Longest chain is 1,297 tokens; the Agentic chat-template overhead must fit. Any truncation aborts the run.

**Precision:** fp16 mixed precision with fp32 LoRA weights, because T4 and P100 lack native bf16. If the pre-training smoke test shows non-finite loss, every run switches to fp32 and the change is logged as an amendment before any eval.

**Agentic target formatting:** the prompt goes in a single user turn and the target is the assistant turn, using the model's own chat template with `enable_thinking=False`. The Logic target uses raw text with no template.

**Seeds:** 3 training seeds per tuned arm: **0, 1, 2**. A seed sets LoRA init, data order and dropout through `transformers.set_seed`. The same seed list is used across arms, so seed *s* of Treatment and seed *s* of Control see examples in the same order.

**Checkpoint selection:** the **final checkpoint is evaluated. No early stopping, no checkpoint picked on any eval score.** Intermediate checkpoints are saved every 50 optimiser steps only so a dead Kaggle session can resume.

**Hardware and software:** Kaggle. All evals for a given target run on one GPU type (T4 preferred). Library versions are pinned in `requirements.lock` at the harness-freeze commit (§9).

**Run count:** 2 targets × 2 tuned arms × 3 seeds = 12 training runs. Evaluated checkpoints: 2 targets × 7 (1 baseline + 3 control + 3 treatment) = 14.

## 6. Benchmarks

### 6.1 Logic target: JustLogic (arXiv 2501.14851)
- **Source:** `github.com/michaelchen-lab/JustLogic`. The Hugging Face mirror ships only train and validate.
- **Test split:** `dataset/test_dataset.csv` recovered from commit `5d21f08f` (2024-09-29; deleted from the repo in `04fc9443`).
  - 1,050 items, 150 per depth 1–7
  - SHA-256 `5c666eb09b874efe7004ae82961c98ec32bb41872b2170eb2813acacf2e2de73`
  - Verified: the current train (4,900) and validate (1,050) files are unchanged since that commit, and test shares zero items with either. This is the original 70/15/15 split, seed 0.
- **Primary eval set (D1 resolved 2026-10-19, Option B):** no arm ever trains on JustLogic, so all 7,000 items (train + validate + test) are held out for this study. The primary set is all 7,000 minus:
  - the 3 items used as few-shot examples: train #860, validate #972, validate #994. #994 matches its few-shot copy except for a typo ("glas" vs "glass"), so the match is by prefix.
  - a 70-item dev slice: 10 per depth, sampled from validate with `random.Random(0)` over id-sorted pools.

  That leaves **n = 6,927** (train 4,899 / validate 978 / test 1,050; ~990 per depth). Frozen lists:
  - `data/manifests/justlogic_primary.csv` (sha256 `da56c67f8e94331323df0daaaa9a36be2003e4da745e8cd82fb9052105913f92`)
  - `data/manifests/justlogic_dev.csv` (sha256 `cad26f4c2d23520f6e86c8fafa7139e857497eecfb3e16f5a399fa08b641ded9`)

  Both are built by `scripts/build_justlogic_manifest.py`, which verifies the source-file hashes.

  This departs from the original brief ("test split only") to raise power. The author approved the change before lock.
- **Pre-registered secondary eval set:** the original 1,050-item test split alone, scored and reported the same way. It does not decide H1 or H2.
- **Prompt:** the authors' `eval/prompt_3_shot.txt` (commit `cdcd470e`), with `{PARAGRAPH}` and `{STATEMENT}` filled from `paragraph` and `question`. Its three examples come from train/val, not test. A base model needs the few-shot examples to learn the answer format.
- **Generation:** greedy (do_sample=False), max_new_tokens = 8, no system prompt, stop at newline.
- **Scoring:** strip the completion, upper-case it, take the first token matching `TRUE|FALSE|UNCERTAIN`. No match counts as **incorrect** and is also logged as a parse failure.
- **Secondary scorer (pre-registered):** rank-classification, i.e. argmax of summed log-probability over the three continuations " TRUE", " FALSE", " UNCERTAIN" after the same prompt. This scorer never has a parse failure. It's reported alongside the primary scorer but **does not decide H1 or H2.**

**Chance floor (per depth):** the per-depth majority-class rate of the eval set. Uniform chance is .333.

| Depth | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Primary set (n = 6,927) | .354 | .352 | .363 | .352 | .355 | .342 | .334 |
| Test-only secondary (n = 1,050) | .347 | .360 | .393 | .393 | .407 | .373 | .387 |

**Fallback (used only if the Rung-0 logic gate fails):** ProofWriter, the OWA meta-test file as redistributed in the JustLogic repo (`pki_eval/proofwriter-meta-test.jsonl`, commit `cdcd470e`).
- Restricted to questions with QDep ∈ {1, 2}. Labels are True/False/Unknown.
- The same prompt style is adapted, with the three examples drawn from ProofWriter QDep 1–2 *train* (never test). Results are reported per QDep.
- The exact item list is frozen and hashed before the fallback is run.

### 6.2 Agentic target: BFCL
- **Package:** `bfcl-eval==2026.3.23`, the BFCL v4 codebase, whose single-turn categories are the v3 AST suite.
- **Why BFCL over Local Agent Bench:** Qwen publishes a first-party BFCL score for this exact model (BFCL-V4, thinking mode: 25.3). That makes it the reference point at our scale.
- **Categories (AST-scored, single-turn):** `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`. That's about 2,350 items. The exact count is recorded at harness freeze.
- **Excluded** (infrastructure and cost; not part of this study): multi-turn, web_search, memory, format_sensitivity, java and javascript. `irrelevance` and `live_irrelevance` are run and reported as descriptive only.
- **Prompting:** the model's own tool-call chat template, non-thinking mode, greedy, max_new_tokens = 512. Function-call outputs are decoded and scored with bfcl-eval's AST checker. Generation may use our own HF Transformers loop instead of bfcl's vLLM handler, but the prompt construction, decoding and checker are frozen at the harness-freeze commit and are identical across arms.
- **Primary metric:** item-level AST accuracy pooled over all items in the 8 categories. Per-category accuracy is always reported alongside it.
- **Parse rate:** fraction of items whose output decodes into ≥ 1 well-formed function call.

## 7. Rung 0: gating rules

These are evaluated on the Baseline arm only, using the frozen harness.

**G-Logic:** pool depths 1–2 of the JustLogic primary set (n = 1,977; pooled majority rate 0.342). **Pass** iff Baseline primary-scorer accuracy ≥ **0.392** (majority + 5 pp) **and** a one-sided exact binomial test against p₀ = 0.342 gives p < 0.05.
- Pass: JustLogic is locked as target one.
- Fail: run the ProofWriter QDep 1–2 fallback with the analogous rule (p₀ = majority rate of that item set, threshold p₀ + 5 pp). If that also fails, the Logic target **does not proceed to Rung 1**, and that is reported as a result.
- All 7 depths are reported regardless.

**G-Agentic:** **Pass** iff parse rate ≥ **10%** **and** pooled AST accuracy ≥ **5%** over the 8 categories. Fail: the Agentic target does not proceed to Rung 1, and that is reported.

The gate outcome, the numbers behind it, and any fallback are recorded in `RUN_LOG.md` and in a dated amendment before Rung 1 starts.

## 8. Rung 1: analysis and decision rules

**Unit:** item × checkpoint correctness (0/1) on the target's primary eval set and primary scorer.

**Arm score:** for a tuned arm, an item's score is its mean correctness over that arm's 3 seeds. Baseline is a single deterministic run.

**Effect:** Δ(A, B) = mean item score of arm A − arm B, in percentage points, on the same items.

**Confirmatory test:** a hierarchical paired bootstrap with 10,000 replicates and RNG seed 20261019. Each replicate:
1. resamples the 3 seeds with replacement within each tuned arm;
2. resamples items with replacement **within each depth stratum** (Logic) or **within each category** (Agentic);
3. recomputes Δ.

The one-sided p-value = fraction of replicates with Δ ≤ 0.

**Multiplicity:** Holm–Bonferroni over the four tests {H1_Logic, H2_Logic, H1_Agentic, H2_Agentic}, family-wise α = 0.05. A target that didn't pass Rung 0 contributes no test, and the family shrinks accordingly.

**Decision rules:**
- **H1_t supported** iff Δ(Treatment, Baseline) > 0 and its Holm-adjusted p < 0.05.
- **H2_t supported** iff Δ(Treatment, Control) > 0 **and** its Holm-adjusted p < 0.05 **and** Treatment's seed-mean exceeds Control's seed-mean for at least 2 of the 3 seed-matched pairs (s = 0, 1, 2).
- **Headline claim** ("structure helps, beyond game data") is made for target *t* only if H1_t **and** H2_t are both supported.

**Null-result definitions:**
- *Clean null for H2_t:* H2_t not supported **and** the two-sided 90% bootstrap CI of Δ(Treatment, Control) lies entirely inside ±1.0 pp. Reported as "the ordering effect, if any, is bounded below 1 pp on this target."
- *Inconclusive:* H2_t not supported and the 90% CI is not inside ±1.0 pp. Reported as underpowered / inconclusive, **not** as evidence of no effect.
- *Game-data-only:* H1_t supported, H2_t not supported. Reported as "consistent with game-data exposure; no evidence that ordering matters."
- *Harm:* Δ(Treatment, Control) < 0 with its two-sided 95% CI excluding 0. Reported as such.

**Descriptive, never confirmatory:**
- per-depth (Logic) and per-category (Agentic) Δ with 95% bootstrap CIs and per-seed values
- secondary scorer results; parse-failure rates per arm
- irrelevance categories; held-out chain loss
- Δ(Control, Baseline), i.e. the "game data alone" effect

**Power (stated up front):** the expected effect is 0.3–1.4 pp. Assuming ~15% of items flip between paired arms:
- Logic primary set (n = 6,927): SE(Δ) ≈ 0.47 pp, MDE at 80% power ≈ 1.3 pp. Only the upper end of the expected range is detectable.
- Test-only secondary (n = 1,050): MDE ≈ 3 pp.
- Agentic (n ≈ 2,350): MDE ≈ 2.2 pp.

An inconclusive H2 is a likely outcome and will be reported plainly as such.

**Contamination:** every claim is a delta between arms that share one base model and one eval set. Any pretraining contamination of JustLogic or BFCL affects all arms equally and largely cancels. Frontier-model scores (GPT-4o, o1, R1, etc.) are never compared against; published numbers are backdrop only.

## 9. Reproducibility and integrity

- **Harness freeze:** the eval harness (prompts, parsing, generation config, scorer, item lists + hashes, `requirements.lock`) is committed and its hash recorded in `RUN_LOG.md` **before the first Rung-0 number is produced.** Harness edits after that point require a dated amendment, and the changed harness is re-run on **every** checkpoint of that target.
- **Raw completions:** every completion from every checkpoint is saved as JSONL (item id, prompt hash, completion, parsed answer, correctness) so results can be re-scored without a GPU.
- **Logged per run:** git commit, dataset manifest hash, full hyperparameters, seed, GPU type, library versions, wall-clock, full generation config.
- **Resumability:** checkpoints every 50 steps with optimizer, scheduler and RNG state. A resumed run must reproduce the same data order.
- **Development set:** harness debugging uses only the 70-item JustLogic dev slice (`justlogic_dev.csv`) and BFCL entries outside the 8 scored categories. It never touches primary-eval items, and no score from development is reported.

## 10. Out of scope

GRPO/RL; SpreadsheetBench, Cybench, physics QA; large-model fine-tunes; overseer architectures; context-compaction, retrieval or memory spin-offs; LegalBench, MedQA; motor-control data; constraint types beyond the six; any eval target not listed in §6.

---

## Amendments

### A1 — 2026-10-19 — Logic fallback source replaced (ProofWriter official OWA depth-2)

**Trigger:** Rung-0 gate G-Logic failed. On JustLogic depths 1–2 pooled (n = 1,977), Base accuracy was 0.362 against a threshold of 0.392 (one-sided binomial p = 0.035 against 0.342). Details in `results/rung0/`. This was decided by the pre-registered rule before the fallback was examined.

**Problem with the registered fallback:** §6.1 named `pki_eval/proofwriter-meta-test.jsonl` from the JustLogic repo (commit `cdcd470e`). On inspection it holds ProofWriter *depth-0* theories, and **all 3,205 of its QDep 1–2 questions are labelled Unknown**. The majority rate is therefore 1.0 and the gate is unpassable by construction. The file is also not byte-identical to the official depth-0 test file. It cannot serve as the fallback as written.

**Replacement (approved by the author before any fallback run):**
- **Source:** official AllenAI ProofWriter release V2020.12.3 (`https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip`, sha256 `bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26`). File `OWA/depth-2/meta-test.jsonl`, all questions with QDep ∈ {1, 2}.
- **Primary set:** n = 10,158 (QDep 1: 6,079, majority .410; QDep 2: 4,079, majority .440), from 1,794 theories. Frozen as `data/manifests/proofwriter_primary.csv` (sha256 `de81163b3f845c146cac924a302649dde6f7d0430b3edc48e55ab9c9cb1b13f8`). No theory overlaps the train split.
- **Dev slice:** 70 `meta-dev` items (35 per QDep, `random.Random(0)`), `data/manifests/proofwriter_dev.csv` (sha256 `f01f8168c131a662fb160a862055a61efd39f53a8350b54bf438beae248001c5`). Harness development only.
- **Prompt:** `data/frozen/proofwriter_prompt_3_shot.txt` (sha256 `6bd414dbdea7eb8a2b7f6936571088a9f266bd9f1616b31637e9046298925adb`). This is the JustLogic 3-shot wording with the propositional argument-form list removed, since it doesn't apply to rule theories.
  - Three shots, one per label, drawn with `random.Random(0)` from `meta-train` QDep 1–2, from distinct theories: `RelNeg-OWA-D2-882:Q5` (TRUE), `AttNoneg-OWA-D2-1444:Q4` (FALSE), `AttNoneg-OWA-D2-571:Q8` (UNCERTAIN).
  - Labels map True/False/Unknown → TRUE/FALSE/UNCERTAIN.
  - Generation config, parser, rank-classification scorer, fp32 and batch size are all unchanged from the frozen JustLogic harness. Builder: `scripts/build_proofwriter_manifest.py`.
- **Gate (replaces the §7 fallback sentence):** pooled QDep 1–2 (n = 10,158; pooled majority rate p₀ = 0.353). **Pass** iff Base primary-scorer accuracy ≥ **0.403** **and** a one-sided exact binomial test against 0.353 gives p < 0.05. Reported per QDep regardless. On fail, the Logic target does not proceed to Rung 1.

**If the gate passes, Rung-1 Logic target = this ProofWriter set** (JustLogic is not evaluated in Rung 1):
- Strata are QDep ∈ {1, 2}, replacing depth.
- Questions from the same theory are not independent (true/negated pairs share a theory). The §8 bootstrap therefore resamples **theories** with replacement within each QDep stratum, taking all of a resampled theory's items, instead of resampling items.
- All other §8 rules stand: seed resampling, Holm across the four tests, the H2 seed-pair condition, and the null definitions.
- Power is lower than item-level n suggests because of the clustering. It will be reported from the realised design effect.

**Unaffected:** everything for the Agentic target, which passed G-Agentic (pooled AST accuracy 0.519, parse rate 0.767).

### A2 — 2026-10-19 — Training micro-batch split (memory only)

**Trigger:** the 20-step training smoke test (commit `8026752`, before any real training run) ran out of memory on a 15 GB T4 at the registered per-device batch 4. The failing allocation (4.38 GiB) is the fp32 logits tensor: 4 sequences × ~1.15k tokens × the 248,320-token vocabulary.

**Change to §5:** per-device batch **2** × gradient accumulation **8**, replacing 4 × 4.
- The effective batch stays 16.
- Each optimizer step sees the same 16 examples in the same order. The sampler and its seed are unchanged, and batches are consecutive slices of the same permutation.
- transformers 5.17 normalises the loss by the total number of target tokens across all accumulated micro-batches (`num_items_in_batch`), so the gradient of each update is the same as under 4 × 4 up to floating-point summation order.

**Also recorded:**
- Gradient checkpointing (non-reentrant) is enabled for memory. It doesn't change the update.
- transformers 5.x replaced `warmup_ratio` with `warmup_steps` given as a ratio; 0.03 is the registered 3% warmup.

Both are applied identically to every run in both arms.

### A3 — 2026-10-20 — BFCL end-of-turn stop (harness bug found after the interim seed-0 look)

**Disclosure:** this was found by inspecting the interim seed-0 BFCL outputs, so it is results-informed. It is recorded here **before** any corrected number was computed.

**Bug:** §6.2 registers generation that stops at eos. For Qwen3.5-0.8B the tokenizer's eos, which is also the chat template's end-of-turn delimiter, is `<|im_end|>` (id 248046). The model repo ships no `generation_config.json`, so `generate()` used the model config's eos `<|endoftext|>` (248044) and did not stop at `<|im_end|>`.
- The untuned model almost always emits `<|endoftext|>` straight after `<|im_end|>`: only 31/3,475 outputs continue past it.
- The tuned seed-0 models usually do not. They keep writing invented user/tool turns and further `<tool_call>` blocks up to the 512-token cap, and the AST checker then fails these as `wrong_count`.
- The as-run metric is therefore dominated by a stop-token mismatch rather than by the registered construct.

**Correction (scoring only; generation unchanged):** every BFCL raw completion, for every checkpoint including Baseline, is truncated at the first `<|im_end|>` before tool calls are decoded. That is the registered "stop at eos" with the tokenizer's eos. It is applied offline to the saved raw completions (`mew/score.py --stop-at-im-end`). Generation for seeds 1–2 stays exactly as frozen, for comparability.

**How both are used:**
- **Primary:** the A3-corrected scoring, because it implements the registered stop rule.
- **Always reported in full alongside it:** the as-run scoring, flagged as affected by this bug.
- A **turn-termination failure rate** per arm (fraction of outputs with non-whitespace text after the first `<|im_end|>`) is reported as a descriptive outcome.
- Readers can weigh the results-informed timing. Every number under both scorings is published.


# RUN_LOG

Append-only record of everything executed: when, with what config, what came out.

---

## 2026-10-19 — Phase 0 inventory (no model runs)

- Repo: cloned `safiqsindha/Project-Mew` (README + LICENSE only).
- Chains located locally at `~/Desktop/Project Ditto/Ditto V1/chains/`:
  `real/` 4,806 · `real_v2/` 4,860 · `shuffled/` 14,418 (seeds 42/1337/7919) · `shuffled_v2/` 14,580.
- Real-vs-shuffled_42 match check (Qwen3.5-0.8B-Base tokenizer @ dc7cdfe2): content multiset identical 4,806/4,806; per-chain token delta 0 for all; corpus 5,476,880 tokens/arm; loss tokens 2,740,223 vs 2,743,438.
- JustLogic: HF mirror has train/validate only; original test split recovered from GitHub commit 5d21f08f, sha256 5c666eb0…2de73; zero overlap with train/val.
- BFCL: `bfcl-eval==2026.3.23`; Qwen reports BFCL-V4 = 25.3 for Qwen3.5-0.8B (thinking).
- No LoRA/SFT code or JustLogic/BFCL harness found in any prior repo.
- PREREGISTRATION.md drafted (not yet locked).

## 2026-10-19 — Pre-registration locked

- D1 resolved: Option B (JustLogic primary n=6,927; test-only secondary; 70-item dev slice).
- PREREGISTRATION.md locked at commit `e91e422b5631f653522728075725f69f56bc1f0b` (pushed: no).
- Manifests: justlogic_primary.csv sha256 da56c67f…913f92; justlogic_dev.csv sha256 cad26f4c…41ded9.
- No eval or training numbers produced.
- Pushed to origin/main 2026-10-19; GitHub records lock commit e91e422 at 2026-10-19T18:52:05Z.

## 2026-10-19 — Phase 1 harness build

- **Model-routing departure:** at the author's explicit instruction, Phase 1 (harness build and rung-0 runs) is executed by Claude Opus 5.5 rather than Sonnet as in the routing table. Affects cost only, not method.
- Harness candidate committed at a4629d5 (generation: transformers 5.17.0 on Kaggle; scoring: offline, bfcl-eval 2026.3.23, `requirements-score.lock`).
- Harness details fixed in code (not changes to the pre-registration):
  - JustLogic prompt: authors' template with trailing whitespace stripped (it ends `"Answer: "`), so answers are natural space-prefixed tokens (`ĠTRUE`, `ĠFALSE`, `ĠUNC|ERT|AIN`). Stop-at-newline applied at parse time (equivalent under greedy decoding).
  - BFCL: bfcl-eval has no Qwen3.5 handler (its QwenFC parser expects Qwen3 JSON calls). Prompts use the model's own chat template; qwen3_coder XML calls are decoded in `mew/bfcl.py` (vLLM-equivalent typing) and scored with bfcl's `ast_checker`. Tools converted with bfcl `convert_to_tool` (OpenAI style) with original dotted names restored.
  - Frozen items: `data/frozen/bfcl_items.jsonl` sha256 1fb73955…3eb8ab (2,351 scored + 1,124 irrelevance); dev `bfcl_dev_items.jsonl` sha256 9b0f8af1…ffee57 (Java/JS simple, never scored).
- Smoke kernel `mew-rung0-smoke` v1 launched on dev items only: JustLogic dev (fp32 b16, fp16 b16, fp32 b1×20) and BFCL dev (fp32 b8, fp16 b8, fp32 b1×10). Purpose: load path, speed, precision and batch-invariance checks.
- Smoke v1 result (commit a4629d5): BFCL dev OK — model class `Qwen3_5ForCausalLM`, torch 2.11.0+cu128, transformers 5.17.0, Python 3.13, Tesla T4; fp32 b1 vs b8 identical (10/10); fp16 differs from fp32 on 5/40 and is not faster (109 s vs 111 s / 40 items) → **eval precision fixed at fp32**; ~2.8 s/item; flash-linear-attention absent (PyTorch reference path, used for all arms). Parse 13/40 on Java/JS dev items; all 27 failures are genuine non-calls (prose/code), every `<tool_call>` output decoded. JustLogic jobs failed on the source-hash check: HF-mirror bytes had been hashed, kernel downloads GitHub bytes. Parsed rows are identical (5,950/5,950, all fields); hashes repinned to GitHub bytes (train bb11dacd…, validate 1e329b81…); manifests regenerate byte-identical (sha256 unchanged). Prompt file hash now pinned (8841aa3d…).
- Smoke v2 (`mew-rung0-smoke-jl`, commit b6e542b): JustLogic dev OK — fp32 b1 vs b16 identical (20/20 completions, 20/20 rank preds); 0/70 parse failures; fp16 differs on 2/70; fp32 0.70 s/item at b16. Dev accuracy not inspected.

## 2026-10-19 — Harness freeze

- **Harness frozen at the commit that adds this entry** (hash recorded in the next entry). From here on, any harness change needs a dated PREREGISTRATION amendment and a re-run on every checkpoint of the affected target.
- Frozen eval config: fp32; JustLogic batch 16, max_new_tokens 8; BFCL batch 8, max_new_tokens 512, enable_thinking=False; transformers 5.17.0 (pip) on Kaggle image sha256:2757e0c7…; PyTorch reference path for gated-delta-rule layers (flash-linear-attention not installed) for every arm; Tesla T4.
- Harness freeze commit: `cfc1ea1294a70b24d10421083466b5b402459aa6`.

## 2026-10-19 — Rung-0 baselines launched

- Kernel `mew-rung0-baseline` at freeze commit cfc1ea1294a70b24d10421083466b5b402459aa6: GPU0 `baseline_logic` (Qwen3.5-0.8B-Base @ dc7cdfe2, JustLogic primary n=6,927); GPU1 `baseline_agentic` (Qwen3.5-0.8B @ 2fc06364, BFCL 2,351 scored + 1,124 irrelevance).

## 2026-10-19 — Rung-0 baseline results (kernel `mew-rung0-baseline`, harness cfc1ea1)

- Both jobs completed with every item (logic 6,927; BFCL 3,475). Wall-clock 4,991 s / 8,993 s. Outputs in `results/rung0/`.
- **G-Logic: FAIL.** Depths 1-2 pooled n=1,977, accuracy 0.3617 (715 correct) vs threshold 0.392 (majority 0.342 + 0.05); one-sided exact binomial p=0.035. Above chance, below the pre-registered margin. Per-depth accuracy .351/.373/.334/.340/.346/.374/.357 for depths 1-7 (majority .334-.363). Model answers TRUE for 83% of depth-1 items, never UNCERTAIN; 0 parse failures.
- **G-Agentic: PASS.** Pooled AST accuracy 0.519, parse rate 0.767 (needs >=0.05 / >=0.10).
- Consequence per PREREGISTRATION.md §7: ProofWriter QDep 1-2 fallback to be run for the Logic target. Not yet started; its item list, few-shot examples and prompt must be frozen and hashed first (amendment to follow).
- Results not yet pushed to GitHub (awaiting author's decision).

## 2026-10-19 — Logic fallback: Amendment A1

- Registered fallback file (JustLogic-redistributed ProofWriter meta-test) found degenerate: depth-0 theories, all 3,205 QDep 1-2 questions labelled Unknown → unpassable gate. Author chose the official-release replacement (and set a standing instruction: default to Claude's recommendations, logging each).
- PREREGISTRATION Amendment A1 appended (locked text untouched): official ProofWriter V2020.12.3 OWA/depth-2 meta-test QDep 1-2, n=10,158; gate p0=0.353, threshold 0.403; Rung-1 bootstrap clusters by theory.
- Fallback harness added: `mew/proofwriter.py`, `scripts/build_proofwriter_manifest.py`, task `proofwriter` in `run_eval.py`/`score.py` (same generation config, parser and scorers as JustLogic). Frozen: primary manifest de81163b…1b13f8, dev f01f8168…01c5, prompt 6bd414db…925adb.
- Next: smoke on the 70 dev items (fp32 b16 vs b1), then the Base baseline on the primary set.
- Smoke (`mew-rung0-smoke-pw`, commit 0cc302f): ProofWriter dev OK. fp32 b1 vs b16 identical (20/20 completions, 20/20 rank preds), 0/70 parse failures, 0.60 s/item. Dev accuracy not inspected. **Fallback harness frozen at 0cc302f.**
- Launched `mew-rung0-baseline-pw` at 0cc302f: Qwen3.5-0.8B-Base @ dc7cdfe2 on the ProofWriter primary set (n=10,158), fp32, batch 16. Expected ~1.7 h.

## 2026-10-19 — Rung-1 preparation (Agentic target passed its gate)

- `scripts/build_sft_data.py` run: all §4.2 checks pass. Train 4,560 / monitor 246 chains per arm (split by sha256(match_id) % 20); tokens identical per chain and in total (train 5,196,671/arm); loss tokens Treatment 2,599,770 vs Control 2,602,714 (+0.11%, tolerance 0.5%). Hashes in `data/manifests/sft_build_report.json`; per-file chain hashes in `data/manifests/chains_manifest.csv` (sha256 40e0d350…dfc8).
- SFT jsonl not committed (chain-text provenance, as in the Ditto repos); uploaded as **private** Kaggle dataset `safiqsindha/mew-sft-v1`.
- `scripts/train_lora.py`: §5 config (LoRA r16/α32/0.05 on the registered module regex; AdamW 2e-4 cosine, 3% warmup, wd 0; batch 4×4; 2 epochs; fp32 master weights + fp16 autocast; final checkpoint; checkpoints every 50 steps with Trainer resume). Logic: raw text + EOS. Agentic: chat template, user=prefix, assistant=continuation, enable_thinking=False, prompt-prefix property asserted. peft 0.21.2.
- Kernel template now also runs training jobs and installs peft 0.21.2 (eval code path unchanged; baseline evals did not load peft).
- Training smoke (`mew-rung1-smoke`): 20 steps, agentic treatment s0 and control s0, to check speed, memory and fp16 loss finiteness (§5) before real runs.
- Training smoke v1 (7afac6b) failed before step 1: peft found no modules for the registered regex. Cause: the regex names checkpoint keys (`model.language_model.layers…`); the text-only `Qwen3_5ForCausalLM` exposes the same modules as `model.layers…`. Fix: `language_model.` made optional, plus an assertion that exactly 150 modules match (6×4 full-attn + 18×3 linear-attn + 24×3 MLP). Same module set as registered; not a design change.
- Training smoke v2 (0d0e39e) failed at import: the Kaggle image's torchao 0.10.0 is rejected by peft 0.21.2 (needs >0.16). torchao is unused by Mew, so the kernel now uninstalls it after installing transformers/peft.
- Training smoke v3 (de0b784) failed: transformers 5.17 removed `warmup_ratio`; replaced by `warmup_steps=0.03` (documented as a ratio of total steps in 5.x) — same 3% warmup. Also added gradient checkpointing (non-reentrant) since batch 4 × ~1.3k tokens with a 248k vocab would not fit a 15 GB T4 otherwise; it changes memory/compute only, not the update. Registered batch 4×4 kept.
- Local CPU code-path test (transformers 5.17.0, peft 0.21.2, torch 2.14.1): 1 step agentic/treatment/s0 completed — 150 LoRA modules, 10,223,616 trainable params, monitor eval and adapter save OK. Not a result.
- Training smoke v4 (8026752) OOM on T4 at micro-batch 4 (fp32 logits, 4.38 GiB). **Amendment A2:** micro-batch 2 × accumulation 8 (effective 16 unchanged; token-normalised loss → same update). Kernel also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- Training smoke v5 (5cccb17): 20 training steps OK on T4 (~24-25 s/step at 2×8 → ~4 h per 570-step run), then OOM in the monitor-loss eval (batch 8 logits, 9.17 GiB). Fix: monitor eval batch 2 with `prediction_loss_only=True` (monitoring only; no effect on training or evaluation). Smoke now runs end to end: train 20 steps → save → evaluate adapter on BFCL dev items.
- Training smoke v6 (623e594): **end-to-end OK.** 20 steps per arm, 150 LoRA modules, peak CUDA memory 10.83 GiB, fp16 losses finite (§5 smoke check passed → fp16 kept), ~25-30 s/step; adapters saved, reloaded and evaluated on 40 BFCL dev items (not scored). **Training harness frozen at 623e594.**

## 2026-10-19 — Rung 1, Agentic target: session 1

- Kernel `mew-rung1-at0-ac0` at 623e594: GPU0 agentic/treatment/seed 0, GPU1 agentic/control/seed 0; each lane trains 570 steps then evaluates its final adapter on the frozen BFCL set. Expected ~7-7.5 h.
- Note: Kaggle keeps outputs only when a session ends normally; a killed session is re-run from scratch with the same seed (in-session resume is via Trainer checkpoints).

## 2026-10-19 — Logic fallback result (kernel `mew-rung0-baseline-pw`, harness 0cc302f)

- Completed, 10,158/10,158 items, 6,174 s. Outputs in `results/rung0/baseline_logic_pw/`.
- **Fallback gate: FAIL.** Pooled accuracy 0.311 (3,164 correct) vs threshold 0.403 (p0 0.353); one-sided binomial p = 1.0, i.e. below chance. QDep 1: 0.258 (majority 0.410); QDep 2: 0.391 (majority 0.440). Predicts TRUE 83-87%, never UNCERTAIN; 0 parse failures.
- **Per §7/A1 the Logic target does not proceed to Rung 1.** Rung 1 continues with the Agentic target only (6 runs). Reported as a Rung-0 result: at 0.8B the base model shows no usable deductive traction on either logic benchmark under this protocol.

## 2026-10-19 — EXPLORATORY logic diagnostics (not confirmatory)

- Plan committed before any run: `docs/exploratory/2026-10-19_logic_diagnostics_plan.md`; frozen subsample `docs/exploratory/logic_diag_subsample.csv` (sha256 5eab0fb3…98c5ca; JustLogic 100/depth, ProofWriter 300/QDep, seed 0); runner `scripts/explore_logic.py` (unchanged `mew/generate.py`, Base model, fp32). Logic gates are not re-adjudicated.
- Kernel `mew-explore-logic`: 11 variants × 2 benchmarks (content-free N/A and empty; label-swap R0-R2; 6 shot orderings).

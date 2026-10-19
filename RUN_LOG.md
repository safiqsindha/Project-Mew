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

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

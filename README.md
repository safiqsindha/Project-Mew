# Project Mew

**Does fine-tuning a small language model on *constraint chains* extracted from Pokémon battles improve out-of-domain reasoning, and is the chains' *structure* what does it?**

This is a small, pre-registered fine-tuning study on `Qwen3.5-0.8B`. The expected effect is small (~0.3–1.4 pp). The value of the work is in the controls, not the effect size.

> **Status (2026-10-21): Rung 1 complete.** Agentic target, 3 seeds per arm: **headline claim not supported.** Ordered chains (Treatment) are level with the untuned model (+1.2 pp, n.s.). Shuffled chains (Control) degrade it (−9.1 pp), so Treatment > Control (+10.3 pp) holds, but because Control is damaged, not because Treatment helps. Logic target stopped at Rung 0. See [results/rung1](results/rung1/README.md).

---

## The question

Earlier work in this program ([Project Ditto](https://github.com/safiqsindha/Project-Ditto)) turned competitive Pokémon Showdown matches into **constraint chains**. These are typed, time-ordered sequences of decision constraints in six categories:

`ResourceBudget` · `ToolAvailability` · `SubGoalTransition` · `InformationState` · `CoordinationDependency` · `OptimizationCriterion`

Mew asks whether training on these chains helps a small model reason elsewhere: on formal logic and on tool calling.

Beating the untuned model alone would prove little, because "game data helps" is already established. **The publishable claim requires beating a control trained on the same chains with their order destroyed.** That control isolates *structure* from *exposure to game data*.

This program's earlier evidence (Cohen's κ ≈ −0.001 across 4,252 chains) supports *cheaper elicitation of latent capability*, not installing new capability. Results here are framed that way.

## Design

**Three arms, identical eval harness:**

| Arm | Training data |
|---|---|
| Baseline | none (untuned) |
| Control | the same 4,806 chains, steps randomly permuted (turn labels reassigned in order) |
| Treatment | the same 4,806 chains, original chronological order |

The control matches the treatment on content tokens (5,476,880 per arm, identical per chain), example count, source traces and length distribution. It differs **only in step order**. See [PREREGISTRATION.md §4.2](PREREGISTRATION.md).

**Two targets:**

| Target | Model | Benchmark | Primary metric |
|---|---|---|---|
| Logic | `Qwen3.5-0.8B-Base` | [JustLogic](https://arxiv.org/abs/2501.14851): 6,927 held-out items, depths 1–7 | accuracy **per reasoning depth** |
| Agentic | `Qwen3.5-0.8B` (post-trained) | BFCL single-turn AST: 2,351 items, 8 categories | pooled AST accuracy, plus per category |

The Base→post-trained swap between targets is deliberate. A base model has never learned a tool-call format and would score ~0 on BFCL for reasons unrelated to the hypothesis.

**Training:** LoRA (r = 16), prefix→continuation SFT on each chain, 3 seeds per tuned arm, the final checkpoint only (no checkpoint picked on eval scores).

**Decision rule:** hierarchical paired bootstrap, Holm-corrected across four tests. "Structure helps" is claimed for a target only if Treatment beats **both** Baseline **and** Control. Clean nulls and inconclusive results are defined in advance and will be reported as such.

## Roadmap

- [x] **Phase 0:** inventory and [pre-registration](PREREGISTRATION.md), locked at `e91e422` before any numbers existed
- [x] Eval harness built, smoke-tested on dev items only, frozen at `cfc1ea1`
- [x] [Amendment A1](PREREGISTRATION.md#amendments): the registered ProofWriter fallback file was degenerate (every QDep 1–2 answer was Unknown), so it was replaced by the official release before use
- [x] [Amendment A2](PREREGISTRATION.md#amendments): training micro-batch 2×8 (effective batch 16 unchanged) to fit a T4
- [x] **Rung 0:** baseline gates. Agentic **passed** (BFCL AST accuracy .519). Logic **failed** (JustLogic .362 at depths 1–2; ProofWriter fallback .311, below chance). Logic stops here.
- [x] [Amendment A3](PREREGISTRATION.md#amendments): BFCL scoring stops at `<|im_end|>` (harness bug found at the interim look; both scorings reported)
- [x] **Rung 1:** Agentic, 3 seeds per arm. H1 (Treatment > Baseline) not supported; H2 (Treatment > Control) supported; headline claim **not supported** ([results](results/rung1/README.md))
- [ ] **Phase 3:** analysis and write-up

Out of scope for this repo: RL/GRPO, larger models, other benchmarks, new constraint types.

## Repository layout

```
PREREGISTRATION.md        hypotheses, arms, metrics, decision rules (locked; amendments appended only)
RUN_LOG.md                append-only record of every run: config, commit, outputs
mew/                      harness: JustLogic + BFCL tasks, generation, offline scoring
scripts/                  run_eval.py, item-freeze and manifest builders, smoke comparison
kaggle/                   kernel template + renderer (all GPU work runs on Kaggle T4s)
data/manifests/           frozen JustLogic item lists (hashed)
data/frozen/              frozen BFCL items and tool schemas (hashed)
requirements-score.lock   offline scoring environment (bfcl-eval 2026.3.23)
```

## Reproducing

Generation needs a GPU. Scoring doesn't.

```bash
# Generate raw completions for one checkpoint (T4, fp32)
python scripts/run_eval.py --task justlogic --items primary \
  --model Qwen/Qwen3.5-0.8B-Base --revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 \
  --out outputs/baseline_logic

# Score offline from the raw completions
pip install -r requirements-score.lock
python -m mew.score justlogic outputs/baseline_logic
```

Every raw completion is kept, so any result can be re-scored without a GPU. Benchmark sources are pinned by commit and SHA-256, and the harness refuses to run if a hash doesn't match.

## Integrity notes

- The pre-registration is never edited after lock. Changes are appended as dated amendments with reasons.
- The harness is identical across arms. Any harness change after the freeze triggers a re-run on every checkpoint.
- Nothing is compared against frontier-model scores. The control arm is the evidence.
- Contamination largely cancels: every claim is a *difference* between arms that share one base model and one eval set.

## License

Apache-2.0 (see [LICENSE](LICENSE)). JustLogic is MIT-licensed. BFCL is used via `bfcl-eval`.

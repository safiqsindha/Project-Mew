# Rung 0 baseline results (untuned models, frozen harness cfc1ea1)

Generated on Kaggle (Tesla T4, fp32, transformers 5.17.0), scored offline with `python -m mew.score`.
Raw completions are kept so everything re-scores without a GPU.

## Logic: Qwen3.5-0.8B-Base on JustLogic (primary set, n = 6,927; 3-shot, greedy)

| Depth | n | Accuracy | Majority-class rate |
|---|---|---|---|
| 1 | 987 | .351 | .354 |
| 2 | 990 | .373 | .352 |
| 3 | 990 | .334 | .363 |
| 4 | 990 | .340 | .352 |
| 5 | 990 | .346 | .355 |
| 6 | 990 | .374 | .342 |
| 7 | 990 | .357 | .334 |

Parse failures: 0. The model answers TRUE for 83% of depth-1 items and never answers UNCERTAIN.

**Gate G-Logic (depths 1-2 pooled, n = 1,977): FAIL.** Accuracy .362 vs required .392 (majority .342 + 5 pp). The one-sided binomial test against .342 gives p = .035, so the result is above chance but below the pre-registered margin. Per PREREGISTRATION.md §7 the ProofWriter QDep 1-2 fallback is next.

## Agentic: Qwen3.5-0.8B on BFCL single-turn AST (n = 2,351 scored)

| Category | n | AST accuracy | Parse rate |
|---|---|---|---|
| simple_python | 400 | .693 | .835 |
| multiple | 200 | .575 | .725 |
| parallel | 200 | .350 | .550 |
| parallel_multiple | 200 | .475 | .670 |
| live_simple | 258 | .527 | .783 |
| live_multiple | 1,053 | .480 | .804 |
| live_parallel | 16 | .688 | .812 |
| live_parallel_multiple | 24 | .458 | .750 |
| *irrelevance (descriptive)* | 240 | .796 | .204 |
| *live_irrelevance (descriptive)* | 884 | .827 | .173 |

**Pooled AST accuracy .519, parse rate .767. Gate G-Agentic: PASS** (needs parse >= .10 and accuracy >= .05).

Note: the pooled parse rate is dominated by live_multiple (n = 1,053). live_parallel and live_parallel_multiple have n = 16 and 24, so their per-category numbers are very noisy.

## Integrity
- completions.jsonl sha256: logic `50b7dd03...f3f971`, agentic `63173b03...653eda`.
- Wall-clock: logic 4,991 s, agentic 8,993 s.
- smoke results (dev items only, not results) are in `../smoke/`.

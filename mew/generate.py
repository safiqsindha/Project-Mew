"""Model loading and deterministic batched generation. Identical for every arm (PREREGISTRATION.md §4, §9)."""
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_id: str, revision: str, dtype: str, adapter: str | None = None):
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch_dtype)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tok, model


def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _done_uids(out: Path) -> set:
    if not out.exists():
        return set()
    done = set()
    for line in open(out):
        try:
            done.add(json.loads(line)["uid"])
        except json.JSONDecodeError:
            pass  # truncated last line from a killed session; that item is redone
    return done


@torch.no_grad()
def run(tok, model, prompts: dict, out: Path, max_new_tokens: int, batch_size: int,
        options: dict | None = None, log_every: int = 20):
    """Greedy-decode every prompt; append one JSON line per item to `out` (resumable).

    prompts: {uid: prompt_text}. Items are processed longest-first (deterministic order) in fixed-size
    batches with left padding. If `options` ({label: continuation}) is given, also records the summed
    log-probability of each continuation after the prompt (rank-classification).
    """
    done = _done_uids(out)
    todo = sorted((u for u in prompts if u not in done), key=lambda u: (-len(prompts[u]), u))
    dev = model.device
    opt_ids = None
    if options:
        opt_ids = {lab: tok(cont, add_special_tokens=False).input_ids for lab, cont in options.items()}
    t0 = time.time()
    with open(out, "a") as fh:
        for b in range(0, len(todo), batch_size):
            uids = todo[b:b + batch_size]
            texts = [prompts[u] for u in uids]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                                 pad_token_id=tok.pad_token_id)
            new = gen[:, enc.input_ids.shape[1]:]
            scores = _option_logprobs(tok, model, texts, opt_ids) if opt_ids else [None] * len(uids)
            for u, row, sc in zip(uids, new, scores):
                ids = [i for i in row.tolist() if i != tok.pad_token_id]
                rec = {"uid": u, "prompt_sha256": sha(prompts[u]),
                       "completion": tok.decode(ids, skip_special_tokens=False), "n_new_tokens": len(ids)}
                if sc is not None:
                    rec["option_logprobs"] = sc
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n = b + len(uids)
            if (b // batch_size) % log_every == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)} items, {el:.0f}s elapsed, {el / n:.2f}s/item", flush=True)


def _option_logprobs(tok, model, texts, opt_ids):
    """Summed log P(continuation | prompt) for each option.

    All options share the prompt, so log P(first token) comes from the logits at the prompt's last
    position. One forward pass is run per multi-token option (to score its later tokens); single-token
    options are read from whichever pass runs first. Only the last positions' logits are materialised.
    """
    prompt_ids = [tok(t, add_special_tokens=False).input_ids for t in texts]
    for t, p in zip(texts, prompt_ids):
        for cont in opt_ids.values():
            full = tok(t + tok.decode(cont), add_special_tokens=False).input_ids
            assert full == p + cont, "prompt/option tokenization is not additive"
    multi = [lab for lab, c in opt_ids.items() if len(c) > 1]
    passes = multi or [None]  # None = prompt-only pass
    out = [dict() for _ in texts]
    first_lp = None
    for lab in passes:
        cont = opt_ids[lab] if lab else []
        seqs = [p + cont for p in prompt_ids]
        L = max(map(len, seqs))
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, L - len(s):] = torch.tensor(s)
            att[i, L - len(s):] = 1
        ids, att = ids.to(model.device), att.to(model.device)
        k = len(cont)
        logits = model(input_ids=ids, attention_mask=att, logits_to_keep=k + 1).logits.float()
        lp = torch.log_softmax(logits, dim=-1)  # positions: last prompt token .. last token
        if first_lp is None:
            first_lp = lp[:, 0]  # distribution right after the prompt
        if lab:
            tgt = ids[:, L - k:]
            tok_lp = lp[:, :k].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for i, v in enumerate(tok_lp.sum(-1).tolist()):
                out[i][lab] = v
    for lab, cont in opt_ids.items():
        if len(cont) == 1:
            for i, v in enumerate(first_lp[:, cont[0]].tolist()):
                out[i][lab] = v
    return out

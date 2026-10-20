"""LoRA SFT for one (target, arm, seed). Identical config for every run (PREREGISTRATION.md §5).

  python scripts/train_lora.py --target agentic --arm treatment --seed 0 --data <sft dir> --out <run dir>

Resumable: if <out> holds checkpoints, training resumes from the latest one (optimizer, scheduler, RNG
and data position restored by the HF Trainer). The final adapter is written to <out>/final_adapter.
"""
import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint

MODELS = {  # pinned revisions (PREREGISTRATION.md §2)
    "logic": ("Qwen/Qwen3.5-0.8B-Base", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"),
    "agentic": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
}
LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
            # Registered regex (PREREGISTRATION.md §5) names checkpoint keys "model.language_model.layers…";
            # the text-only Qwen3_5ForCausalLM exposes the same modules as "model.layers…", so the
            # "language_model." segment is optional. The matched set is asserted below (150 modules).
            target_modules=r"^model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj"
                           r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)$")
EXPECTED_LORA_MODULES = 6 * 4 + 18 * 3 + 24 * 3  # full-attn q/k/v/o + linear-attn qkv/z/out + mlp gate/up/down
# transformers 5.x: warmup_steps as a float in [0, 1) is the warmup ratio (3% of total steps).
HP = dict(learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.03, weight_decay=0.0,
          # Amendment A2: micro-batch 2 x accumulation 8 (registered 4 x 4 OOMs a T4 on the fp32 logits).
          # Same 16 examples per update; the Trainer normalises loss by target tokens across accumulation.
          per_device_train_batch_size=2, gradient_accumulation_steps=8, num_train_epochs=2)
assert HP["per_device_train_batch_size"] * HP["gradient_accumulation_steps"] == 16  # effective batch (§5)
MAX_LEN = 1536


def encode(tok, target_kind, prompt, target):
    """Token ids and labels; loss only on the target (continuation) tokens."""
    if target_kind == "logic":  # raw text, no template
        p = tok(prompt, add_special_tokens=False).input_ids
        t = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        ids = p + t
    else:  # chat template: prompt as the user turn, continuation as the assistant turn
        user = [{"role": "user", "content": prompt}]
        p = tok.apply_chat_template(user, tokenize=True, add_generation_prompt=True, enable_thinking=False)
        full = tok.apply_chat_template(user + [{"role": "assistant", "content": target}], tokenize=True,
                                       enable_thinking=False)
        p, ids = list(_ids(p)), list(_ids(full))
        assert ids[:len(p)] == p, "chat template prompt is not a prefix of the full conversation"
    assert len(ids) <= MAX_LEN, f"example exceeds {MAX_LEN} tokens ({len(ids)}); truncation is not allowed"
    return {"input_ids": ids, "labels": [-100] * len(p) + ids[len(p):]}


def _ids(x):
    return x["input_ids"] if isinstance(x, dict) or hasattr(x, "keys") else x


class Collator:
    def __init__(self, pad_id):
        self.pad = pad_id

    def __call__(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.pad, dtype=torch.long)
        lab = torch.full((len(batch), L), -100, dtype=torch.long)
        att = torch.zeros((len(batch), L), dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[i, :n] = torch.tensor(b["input_ids"])
            lab[i, :n] = torch.tensor(b["labels"])
            att[i, :n] = 1
        return {"input_ids": ids, "labels": lab, "attention_mask": att}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(MODELS), required=True)
    ap.add_argument("--arm", choices=["treatment", "control"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data", type=Path, required=True, help="dir with {arm}_{train,monitor}.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test only")
    ap.add_argument("--cpu", action="store_true", help="local code-path test only")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    set_seed(a.seed)

    model_id, rev = MODELS[a.target]
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = {part: [json.loads(l) for l in open(a.data / f"{a.arm}_{part}.jsonl")] for part in ("train", "monitor")}
    data = {part: [encode(tok, a.target, r["prompt"], r["target"]) for r in rs] for part, rs in rows.items()}

    # fp32 master weights; fp16 autocast for compute (T4 has no native bf16). LoRA weights stay fp32.
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=rev, dtype=torch.float32)
    import re
    matched = [n for n, _ in model.named_modules() if re.match(LORA["target_modules"], n)]
    assert len(matched) == EXPECTED_LORA_MODULES, (len(matched), matched[:5])
    model = get_peft_model(model, LoraConfig(**LORA))
    # Gradient checkpointing: recomputes activations to fit batch 4 x ~1.3k tokens on a 15 GB T4.
    # Changes memory/compute only, not the update (checkpointing restores the RNG state for dropout).
    model.enable_input_require_grads()
    n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)

    args = TrainingArguments(
        output_dir=str(a.out), seed=a.seed, data_seed=a.seed, **HP, max_steps=a.max_steps,
        fp16=(a.precision == "fp16"), logging_steps=10, eval_strategy="steps", eval_steps=50,
        save_strategy="steps", save_steps=50, save_total_limit=2, report_to=[], remove_unused_columns=False,
        dataloader_num_workers=0, per_device_eval_batch_size=8, use_cpu=a.cpu,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=args, train_dataset=data["train"], eval_dataset=data["monitor"],
                      data_collator=Collator(tok.pad_token_id))

    meta = {"target": a.target, "arm": a.arm, "seed": a.seed, "model": model_id, "revision": rev,
            "lora": LORA, "lora_modules_matched": len(matched), "hp": HP, "precision": a.precision, "max_len": MAX_LEN, "n_lora_params": n_lora,
            "n_train": len(data["train"]), "n_monitor": len(data["monitor"]),
            "n_train_target_tokens": sum(sum(l != -100 for l in d["labels"]) for d in data["train"]),
            "torch": torch.__version__, "transformers": version("transformers"), "peft": version("peft"),
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
    print(json.dumps(meta, indent=1), flush=True)
    last = get_last_checkpoint(str(a.out)) if any(a.out.glob("checkpoint-*")) else None
    meta["resumed_from"] = last
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=last)
    meta["train_wall_clock_s_this_session"] = round(time.time() - t0, 1)
    meta["train_metrics"] = result.metrics
    if torch.cuda.is_available():
        meta["peak_cuda_mem_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    meta["log_history"] = trainer.state.log_history
    if any(not torch.isfinite(torch.tensor(h["loss"])) for h in trainer.state.log_history if "loss" in h):
        meta["NONFINITE_LOSS"] = True
    model.save_pretrained(str(a.out / "final_adapter"))
    (a.out / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("done", json.dumps(result.metrics), flush=True)


if __name__ == "__main__":
    main()

"""Pre-unblinding check: save an untrained (B = 0) LoRA adapter with the registered config, so the
adapter eval path can be compared with the plain-model baseline path on dev items.

  python scripts/make_zero_adapter.py --target agentic --out <dir>
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import LORA, MODELS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--target", choices=list(MODELS), required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
torch.manual_seed(0)
model = AutoModelForCausalLM.from_pretrained(*MODELS[a.target][:1], revision=MODELS[a.target][1], dtype=torch.float32)
model = get_peft_model(model, LoraConfig(**LORA))
assert all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if "lora_B" in n)
model.save_pretrained(str(a.out / "final_adapter"))
print("saved zero adapter", a.out)

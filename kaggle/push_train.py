"""Render a Rung-1 Kaggle kernel: per GPU lane, train LoRA runs then evaluate their final adapters.

  python kaggle/push_train.py --mode smoke --commit <sha> --build /tmp/k
  python kaggle/push_train.py --mode runs --runs agentic:treatment:0,agentic:control:0 --commit <sha> --build /tmp/k
"""
import argparse
import json
from pathlib import Path

EVAL = {  # frozen eval settings (identical to rung 0 baselines)
    "agentic": ["--task", "bfcl", "--items", "all", "--model", "Qwen/Qwen3.5-0.8B",
                "--revision", "2fc06364715b967f1860aea9cf38778875588b17", "--dtype", "fp32", "--batch-size", "8"],
    "logic": ["--task", "proofwriter", "--items", "primary", "--model", "Qwen/Qwen3.5-0.8B-Base",
              "--revision", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68", "--dtype", "fp32", "--batch-size", "16"],
}


def run_jobs(target, arm, seed, max_steps=None, evaluate=True):
    name = f"{target}_{arm}_s{seed}"
    train = {"name": f"train_{name}", "script": "train_lora.py",
             "args": ["--target", target, "--arm", arm, "--seed", str(seed), "--data", "{SFT}",
                      *(["--max-steps", str(max_steps)] if max_steps else [])]}
    jobs = [train]
    if evaluate:
        jobs.append({"name": f"eval_{name}", "args": [*EVAL[target], "--adapter", f"{{OUT}}/train_{name}/final_adapter"]})
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "runs"], required=True)
    ap.add_argument("--runs", default="", help="comma list target:arm:seed; lanes alternate GPUs")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--slug", default=None)
    a = ap.parse_args()
    if a.mode == "smoke":
        # End-to-end check: 20 steps, save, then evaluate the adapter on BFCL dev items (never scored).
        lanes = []
        for arm in ("treatment", "control"):
            jobs = run_jobs("agentic", arm, 0, max_steps=20, evaluate=False)
            jobs.append({"name": f"eval_dev_agentic_{arm}_s0",
                         "args": [*[x if x != "all" else "dev" for x in EVAL["agentic"]],
                                  "--adapter", f"{{OUT}}/train_agentic_{arm}_s0/final_adapter"]})
            lanes.append(jobs)
        slug = a.slug or "mew-rung1-smoke"
    else:
        runs = [r.split(":") for r in a.runs.split(",")]
        lanes = [[], []]
        for i, (t, arm, s) in enumerate(runs):
            lanes[i % 2] += run_jobs(t, arm, int(s))
        slug = a.slug or "mew-rung1-" + "-".join(f"{t[0]}{arm[0]}{s}" for t, arm, s in runs)
    a.build.mkdir(parents=True, exist_ok=True)
    body = (Path(__file__).parent / "kernel_rung0.py").read_text()
    body = body.replace("__COMMIT__", a.commit).replace("__LANES__", json.dumps(lanes))
    (a.build / f"{slug}.py").write_text(body)
    (a.build / "kernel-metadata.json").write_text(json.dumps({
        "id": f"safiqsindha/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"], "dataset_sources": ["safiqsindha/mew-sft-v1"],
        "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4",
    }, indent=1))
    print(a.build, slug)


if __name__ == "__main__":
    main()

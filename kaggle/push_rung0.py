"""Render the rung-0 Kaggle kernel for a commit and mode, ready for `kaggle kernels push -p <build dir>`.

  python kaggle/push_rung0.py --mode smoke --commit <sha> --build /tmp/mew_kernel_smoke
"""
import argparse
import json
from pathlib import Path

BASE = ["--model", "Qwen/Qwen3.5-0.8B-Base", "--revision", "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"]
INSTRUCT = ["--model", "Qwen/Qwen3.5-0.8B", "--revision", "2fc06364715b967f1860aea9cf38778875588b17"]


def lanes(mode: str, dtype: str):
    if mode == "smoke":  # dev items only (PREREGISTRATION.md §9); nothing here is reported
        return [
            [{"name": "smoke_jl_fp32_b16", "args": ["--task", "justlogic", "--items", "dev", *BASE, "--dtype", "fp32", "--batch-size", "16"]},
             {"name": "smoke_jl_fp16_b16", "args": ["--task", "justlogic", "--items", "dev", *BASE, "--dtype", "fp16", "--batch-size", "16"]},
             {"name": "smoke_jl_fp32_b1", "args": ["--task", "justlogic", "--items", "dev", *BASE, "--dtype", "fp32", "--batch-size", "1", "--limit", "20"]}],
            [{"name": "smoke_bfcl_fp32_b8", "args": ["--task", "bfcl", "--items", "dev", *INSTRUCT, "--dtype", "fp32", "--batch-size", "8"]},
             {"name": "smoke_bfcl_fp16_b8", "args": ["--task", "bfcl", "--items", "dev", *INSTRUCT, "--dtype", "fp16", "--batch-size", "8"]},
             {"name": "smoke_bfcl_fp32_b1", "args": ["--task", "bfcl", "--items", "dev", *INSTRUCT, "--dtype", "fp32", "--batch-size", "1", "--limit", "10"]}],
        ]
    if mode == "smoke_jl":  # re-run of the JustLogic smoke jobs only
        jl = lanes("smoke", dtype)[0]
        return [[jl[0], jl[2]], [jl[1]]]
    if mode == "baseline":
        return [
            [{"name": "baseline_logic", "args": ["--task", "justlogic", "--items", "primary", *BASE, "--dtype", dtype, "--batch-size", "16"]}],
            [{"name": "baseline_agentic", "args": ["--task", "bfcl", "--items", "all", *INSTRUCT, "--dtype", dtype, "--batch-size", "8"]}],
        ]
    raise ValueError(mode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "smoke_jl", "baseline"], required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--build", type=Path, required=True)
    a = ap.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    slug = f"mew-rung0-{a.mode}".replace("_", "-")
    body = (Path(__file__).parent / "kernel_rung0.py").read_text()
    body = body.replace("__COMMIT__", a.commit).replace("__LANES__", json.dumps(lanes(a.mode, a.dtype)))
    (a.build / f"{slug}.py").write_text(body)
    (a.build / "kernel-metadata.json").write_text(json.dumps({
        "id": f"safiqsindha/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [],
        "competition_sources": [], "model_sources": [],
        # Same pinned image as the program's earlier kernels; all Mew evals use it.
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
        "machine_shape": "NvidiaTeslaT4",
    }, indent=1))
    print(a.build)


if __name__ == "__main__":
    main()

"""Render the EXPLORATORY logic-diagnostics kernel (one lane per benchmark). Not confirmatory."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
VARIANTS = ["cf_na", "cf_empty", "swap_R0", "swap_R1", "swap_R2",
            "order_012", "order_021", "order_102", "order_120", "order_201", "order_210"]

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True)
ap.add_argument("--build", type=Path, required=True)
a = ap.parse_args()
lanes = [[{"name": f"explore_{b}_{v}", "script": "explore_logic.py", "args": ["--benchmark", b, "--variant", v]}
          for v in VARIANTS] for b in ("justlogic", "proofwriter")]
slug = "mew-explore-logic"
a.build.mkdir(parents=True, exist_ok=True)
body = (Path(__file__).parent / "kernel_rung0.py").read_text()
(a.build / f"{slug}.py").write_text(body.replace("__COMMIT__", a.commit).replace("__LANES__", json.dumps(lanes)))
(a.build / "kernel-metadata.json").write_text(json.dumps({
    "id": f"safiqsindha/{slug}", "title": slug, "code_file": f"{slug}.py", "language": "python",
    "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "keywords": ["gpu"], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f",
    "machine_shape": "NvidiaTeslaT4"}, indent=1))
print(a.build, slug)

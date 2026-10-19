"""Kaggle kernel body for Project Mew rung 0. Rendered by kaggle/push_rung0.py; not run directly.

Holds no credential: the repo, JustLogic sources and Qwen3.5 weights are all public and ungated.
The top level never raises (failures become KERNEL_FAILED.txt). Jobs are grouped into one lane per GPU;
lanes run in parallel, jobs within a lane run in order. Every raw completion lands in /kaggle/working/out.
"""
import json
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path

COMMIT = "__COMMIT__"
LANES = json.loads('__LANES__')  # [[{"name":..., "args":[...]}, ...], ...] one list per GPU
DEADLINE_H = 11.0  # stop launching new jobs after this (Kaggle hard limit is 12h)

T0 = time.time()
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "kernel_started.txt").write_text(time.strftime("started %Y-%m-%d %H:%M:%S UTC\n", time.gmtime()))


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=True, **kw)


def lane(gpu, jobs, status):
    for job in jobs:
        if (time.time() - T0) / 3600 > DEADLINE_H:
            status[job["name"]] = "skipped_deadline"
            continue
        d = OUT / job["name"]
        d.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED="1")
        t = time.time()
        with open(d / "log.txt", "a") as log:
            rc = subprocess.run(["python", "/tmp/mew/scripts/run_eval.py", *job["args"], "--out", str(d)],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        status[job["name"]] = {"rc": rc, "wall_s": round(time.time() - t, 1), "gpu": gpu}
        print(job["name"], status[job["name"]], flush=True)


try:
    sh("nvidia-smi")
    sh("pip install -q transformers==5.17.0")
    sh(f"git clone -q https://github.com/safiqsindha/Project-Mew /tmp/mew && cd /tmp/mew && git checkout -q {COMMIT}")
    head = subprocess.run("cd /tmp/mew && git rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
    assert head.startswith(COMMIT), (head, COMMIT)
    env_info = subprocess.run("pip freeze", shell=True, capture_output=True, text=True).stdout
    (OUT / "pip_freeze.txt").write_text(env_info)
    status = {}
    threads = [threading.Thread(target=lane, args=(g, jobs, status)) for g, jobs in enumerate(LANES)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    (OUT / "status.json").write_text(json.dumps({"commit": head, "status": status,
                                                 "total_wall_s": round(time.time() - T0, 1)}, indent=1))
    print("ALL DONE", json.dumps(status), flush=True)
except Exception:
    (OUT / "KERNEL_FAILED.txt").write_text(traceback.format_exc())
    print(traceback.format_exc(), flush=True)

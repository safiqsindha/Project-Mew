"""Freeze the ProofWriter fallback items and prompt (PREREGISTRATION.md Amendment A1).

  python scripts/build_proofwriter_manifest.py [cache_dir]

Outputs:
  data/manifests/proofwriter_primary.csv   all OWA/depth-2 meta-test questions with QDep 1-2
  data/manifests/proofwriter_dev.csv       70 meta-dev items (35 per QDep, seed 0), harness dev only
  data/frozen/proofwriter_prompt_3_shot.txt  JustLogic-style 3-shot prompt; shots from meta-train
"""
import csv
import hashlib
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mew.proofwriter import fetch_zip, read_split  # noqa: E402

HEADER = """You are given a paragraph of facts and rules, followed by a statement. Perform logical reasoning on the paragraph to determine the truth value of the statement.

You must answer with either one of the 3 options:
- TRUE: When the premises in the paragraph lead to the statement
- FALSE: When the premises in the paragraph directly contradict the statement
- UNCERTAIN: When the premises in the paragraph neither support nor contradict the statement

Do not use your prior knowledge; your answer must be solely determined by the information within the paragraph. Assume that all premises in the paragraph are true.

Question: Is the statement true, false, or uncertain?

Output only "TRUE", "FALSE", or "UNCERTAIN". Do not generate anything else.
-----
"""
SHOT = "\nParagraph: {p}\n\nStatement: {s}\n\nAnswer: {a}\n-----\n"
TAIL = "\nParagraph: {PARAGRAPH}\n\nStatement: {STATEMENT}\n\nAnswer: "
COLS = ["uid", "split", "theory_id", "qdep", "label", "item_sha256"]


def write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(path, len(rows), hashlib.sha256(Path(path).read_bytes()).hexdigest())


z = fetch_zip(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/mew_cache"))
test, dev, train = read_split(z, "test"), read_split(z, "dev"), read_split(z, "train")
assert not ({i["theory_id"] for i in test} & {i["theory_id"] for i in train}), "train/test theory overlap"

rng = random.Random(0)
dev_slice = []
for q in (1, 2):
    dev_slice += rng.sample(sorted((i for i in dev if i["qdep"] == q), key=lambda i: i["uid"]), 35)

shots, used = [], set()
for lab in ("TRUE", "FALSE", "UNCERTAIN"):
    cands = sorted((i for i in train if i["label"] == lab and i["theory_id"] not in used), key=lambda i: i["uid"])
    s = rng.choice(cands)
    used.add(s["theory_id"])
    shots.append(s)

template = HEADER + "".join(SHOT.format(p=s["paragraph"], s=s["question"], a=s["label"]) for s in shots) + TAIL
tpath = ROOT / "data/frozen/proofwriter_prompt_3_shot.txt"
tpath.write_text(template)
print(tpath, hashlib.sha256(tpath.read_bytes()).hexdigest(), "shots:", [(s["uid"], s["label"]) for s in shots])
write(ROOT / "data/manifests/proofwriter_primary.csv", test)
write(ROOT / "data/manifests/proofwriter_dev.csv", dev_slice)

for name, rows in [("primary", test)]:
    for q in (1, 2):
        rs = [r for r in rows if r["qdep"] == q]
        print(name, "QDep", q, "n", len(rs), "majority", round(max(sum(r["label"] == l for r in rs) for l in ("TRUE", "FALSE", "UNCERTAIN")) / len(rs), 4))
    print(name, "pooled majority", round(max(sum(r["label"] == l for r in rows) for l in ("TRUE", "FALSE", "UNCERTAIN")) / len(rows), 4),
          "theories", len({r["theory_id"] for r in rows}))

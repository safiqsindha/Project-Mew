"""Build the frozen JustLogic item manifests for Project Mew (PREREGISTRATION.md §6.1, D1 = Option B).

Inputs (pinned): JustLogic GitHub commit 5d21f08f dataset/{train,validate,test}_dataset.csv (raw.githubusercontent bytes;
the HF mirror has identical rows but different bytes)
and eval/prompt_3_shot.txt @ cdcd470e.
Outputs: data/manifests/justlogic_{primary,dev}.csv with (split, id, depth, label, item_sha256).
"""
import csv, hashlib, random, re, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
EXPECTED = {
    "train_dataset.csv": "bb11dacd9bd7a377526062072d6a81512b5a52ed9de7952cf6ba4ac9def64a7c",
    "validate_dataset.csv": "1e329b810c21d3a3926e10867ee230d8b0c8de0dd8a83dc03bbd04cedcc3fff4",
    "test_dataset.csv": "5c666eb09b874efe7004ae82961c98ec32bb41872b2170eb2813acacf2e2de73",
}
for f, h in EXPECTED.items():
    assert hashlib.sha256((src / f).read_bytes()).hexdigest() == h, f"hash mismatch: {f}"

def item_hash(r):
    return hashlib.sha256((r["paragraph"] + "\x1f" + r["question"] + "\x1f" + r["label"]).encode()).hexdigest()

items = []
for split, f in [("train", "train_dataset.csv"), ("validate", "validate_dataset.csv"), ("test", "test_dataset.csv")]:
    for r in csv.DictReader(open(src / f)):
        items.append({"split": split, "id": r["id"], "depth": int(r["depth"]),
                      "label": r["label"].upper(), "paragraph": r["paragraph"], "item_sha256": item_hash(r)})
assert len(items) == 7000
assert len({i["item_sha256"] for i in items}) == 7000, "duplicate items across splits"

shots = [p.strip() for p in re.findall(r"Paragraph: (.+)", (src / "prompt_3_shot.txt").read_text()) if "{" not in p]
assert len(shots) == 3
# Prefix match: one few-shot example differs from its source item only by a typo ("glas" vs "glass").
is_shot = lambda i: any(s[:80] in i["paragraph"] for s in shots)
fewshot = [i for i in items if is_shot(i)]
assert sorted((i["split"], i["id"]) for i in fewshot) == [("train", "860"), ("validate", "972"), ("validate", "994")], fewshot

rng = random.Random(0)
dev = []
for d in range(1, 8):
    pool = sorted((i for i in items if i["split"] == "validate" and i["depth"] == d and not is_shot(i)),
                  key=lambda i: int(i["id"]))
    dev += rng.sample(pool, 10)
dev_keys = {i["item_sha256"] for i in dev}
primary = [i for i in items if not is_shot(i) and i["item_sha256"] not in dev_keys]

cols = ["split", "id", "depth", "label", "item_sha256"]
for name, rows in [("justlogic_primary.csv", primary), ("justlogic_dev.csv", dev)]:
    with open(out / name, "w", newline="") as fh:
        w = csv.DictWriter(fh, cols, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("few-shot excluded:", [(i["split"], i["id"]) for i in fewshot])
print("primary", len(primary), "dev", len(dev))

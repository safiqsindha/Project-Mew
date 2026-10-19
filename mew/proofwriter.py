"""ProofWriter fallback task for the Logic target (PREREGISTRATION.md Amendment A1).

Source: official AllenAI release V2020.12.3, OWA/depth-2, questions with QDep in {1, 2}.
Labels map True/False/Unknown -> TRUE/FALSE/UNCERTAIN so the JustLogic parser and scorers are reused.
"""
import csv
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip"
ZIP_SHA256 = "bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26"
MEMBER = "proofwriter-dataset-V2020.12.3/OWA/depth-2/meta-{split}.jsonl"
LABEL = {"True": "TRUE", "False": "FALSE", "Unknown": "UNCERTAIN"}
QDEPS = (1, 2)


def fetch_zip(cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    dst = cache / "proofwriter-dataset-V2020.12.3.zip"
    if not dst.exists():
        urllib.request.urlretrieve(ZIP_URL, dst)
    h = hashlib.sha256()
    with open(dst, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    assert h.hexdigest() == ZIP_SHA256, f"proofwriter zip sha256 {h.hexdigest()} != {ZIP_SHA256}"
    return dst


def item_hash(theory: str, question: str, label: str) -> str:
    return hashlib.sha256((theory + "\x1f" + question + "\x1f" + label).encode()).hexdigest()


def read_split(zip_path: Path, split: str) -> list[dict]:
    """All QDep 1-2 questions of one split as items, in file order."""
    items = []
    with zipfile.ZipFile(zip_path) as z, z.open(MEMBER.format(split=split)) as fh:
        for line in fh:
            r = json.loads(line)
            for qid, q in r["questions"].items():
                if q["QDep"] not in QDEPS:
                    continue
                lab = LABEL[str(q["answer"])]
                items.append({"uid": f"{split}:{r['id']}:{qid}", "split": split, "theory_id": r["id"],
                              "qdep": q["QDep"], "label": lab, "paragraph": r["theory"],
                              "question": q["question"],
                              "item_sha256": item_hash(r["theory"], q["question"], lab)})
    return items


def load_items(cache: Path, manifest: Path) -> list[dict]:
    """Items listed in a frozen manifest (proofwriter_primary.csv / proofwriter_dev.csv), in manifest order."""
    z = fetch_zip(cache)
    man = list(csv.DictReader(open(manifest)))
    pool = {}
    for split in sorted({m["split"] for m in man}):
        pool.update({it["uid"]: it for it in read_split(z, split)})
    items = []
    for m in man:
        it = pool[m["uid"]]
        assert it["item_sha256"] == m["item_sha256"], m["uid"]
        items.append(it)
    return items


def build_prompt(template: str, item: dict) -> str:
    # Same construction as JustLogic: fill, then strip trailing whitespace so the answer is " TRUE" etc.
    return template.replace("{PARAGRAPH}", item["paragraph"]).replace("{STATEMENT}", item["question"]).rstrip()

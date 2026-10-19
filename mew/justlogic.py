"""JustLogic task: pinned sources, prompt construction, answer parsing (PREREGISTRATION.md §6.1)."""
import csv
import hashlib
import re
import urllib.request
from pathlib import Path

REPO = "https://raw.githubusercontent.com/michaelchen-lab/JustLogic"
SOURCES = {
    # name: (commit, repo path, sha256)
    "train_dataset.csv": ("5d21f08f", "dataset/train_dataset.csv",
                          "bb11dacd9bd7a377526062072d6a81512b5a52ed9de7952cf6ba4ac9def64a7c"),
    "validate_dataset.csv": ("5d21f08f", "dataset/validate_dataset.csv",
                             "1e329b810c21d3a3926e10867ee230d8b0c8de0dd8a83dc03bbd04cedcc3fff4"),
    "test_dataset.csv": ("5d21f08f", "dataset/test_dataset.csv",
                         "5c666eb09b874efe7004ae82961c98ec32bb41872b2170eb2813acacf2e2de73"),
    "prompt_3_shot.txt": ("cdcd470e51c37edd83ba7df4210fabb473546439", "eval/prompt_3_shot.txt",
                          "8841aa3d813ed2274b308e1f92f397a74bed61e1cf54c41f7d626e5b3582b11f"),
}
SPLIT_FILE = {"train": "train_dataset.csv", "validate": "validate_dataset.csv", "test": "test_dataset.csv"}
LABELS = ("TRUE", "FALSE", "UNCERTAIN")
OPTIONS = {lab: " " + lab for lab in LABELS}  # continuations scored after a prompt ending in "Answer:"
ANSWER_RE = re.compile(r"\b(TRUE|FALSE|UNCERTAIN)\b")


def fetch_sources(cache: Path) -> dict:
    """Download pinned files into cache (once) and verify hashes. Returns {name: path}."""
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (commit, path, sha) in SOURCES.items():
        dst = cache / name
        if not dst.exists():
            urllib.request.urlretrieve(f"{REPO}/{commit}/{path}", dst)
        if sha is not None:
            got = hashlib.sha256(dst.read_bytes()).hexdigest()
            assert got == sha, f"{name}: sha256 {got} != {sha}"
        paths[name] = dst
    return paths


def load_items(cache: Path, manifest: Path) -> list[dict]:
    """Items listed in a frozen manifest (justlogic_primary.csv / justlogic_dev.csv), in manifest order."""
    paths = fetch_sources(cache)
    rows = {}
    for split, fname in SPLIT_FILE.items():
        for r in csv.DictReader(open(paths[fname])):
            rows[(split, r["id"])] = r
    items = []
    for m in csv.DictReader(open(manifest)):
        r = rows[(m["split"], m["id"])]
        h = hashlib.sha256((r["paragraph"] + "\x1f" + r["question"] + "\x1f" + r["label"]).encode()).hexdigest()
        assert h == m["item_sha256"], (m["split"], m["id"])
        items.append({"uid": f"{m['split']}:{m['id']}", "split": m["split"], "depth": int(m["depth"]),
                      "label": r["label"].upper(), "paragraph": r["paragraph"], "question": r["question"]})
    return items


def build_prompt(template: str, item: dict) -> str:
    # The authors' template ends "Answer: " (trailing space). Trailing whitespace is stripped so the
    # answer is produced as a natural space-prefixed token (" TRUE"), matching OPTIONS.
    return template.replace("{PARAGRAPH}", item["paragraph"]).replace("{STATEMENT}", item["question"]).rstrip()


def parse_answer(completion: str):
    """Primary scorer: generation is cut at the first newline (stop-at-newline), then the first
    TRUE|FALSE|UNCERTAIN in the upper-cased text is the answer. None = parse failure (scored incorrect)."""
    text = completion.split("\n", 1)[0].strip().upper()
    m = ANSWER_RE.search(text)
    return m.group(1) if m else None

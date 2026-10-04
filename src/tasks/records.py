"""Shared helpers for the public v0 record builders (session 4): downloads with checksums, JSON lines,
and the counts and checksums files.

A record is one question about one case: id, qid, state (a JSON object), label (option index), target
(one-hot list), and optionally group (fields kept outside the state for subgroup analysis only).
Records hold no tokens; each model tokenizes them with its own tokenizer (src/train/build_items.py).
"""
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "public"
OUT = ROOT / "data" / "processed" / "public"
RESULTS = ROOT / "results" / "public_v0"
SEED = 20260927
SPLITS = ("train", "validation", "test")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, dest, sha256):
    """Download url to dest unless dest already exists with the expected SHA-256; verify either way."""
    dest = Path(dest)
    if dest.exists() and sha256_file(dest) == sha256:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ed-decision-model/public_v0"})
    with urllib.request.urlopen(request, timeout=600) as response, open(tmp, "wb") as f:
        shutil.copyfileobj(response, f, 1 << 20)
    got = sha256_file(tmp)
    if got != sha256:
        raise RuntimeError(f"checksum mismatch for {url}: expected {sha256}, got {got} (kept as {tmp})")
    tmp.replace(dest)
    return dest


def one_hot(index, n):
    return [1.0 if i == index else 0.0 for i in range(n)]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def update_json(path, key, value):
    """Set one top-level key in a JSON object file, keeping the other keys (sorted on write)."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finish(qid, split_rows, counts, out_dir, results_dir):
    """Write the three record files for one question, then record counts and checksums."""
    out_dir, results_dir = Path(out_dir), Path(results_dir)
    checksums = {}
    for split in SPLITS:
        path = write_jsonl(out_dir / f"{qid}_{split}.jsonl", split_rows[split])
        checksums[path.name] = sha256_file(path)
        labels = {}
        for row in split_rows[split]:
            labels[str(row["label"])] = labels.get(str(row["label"]), 0) + 1
        counts.setdefault("rows_per_split", {})[split] = len(split_rows[split])
        counts.setdefault("labels_per_split", {})[split] = dict(sorted(labels.items()))
    update_json(out_dir / "counts.json", qid, counts)
    for name, digest in checksums.items():
        update_json(out_dir / "checksums.json", name, digest)
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(out_dir / "counts.json", results_dir / "counts.json")
    shutil.copy(out_dir / "checksums.json", results_dir / "checksums.json")
    return checksums

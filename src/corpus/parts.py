"""Part files for session 5's pretraining text (docs/prompt_session5.md, step 2), shared by the builders.

A part is gzipped JSON lines with the fields id and text, written deterministically (the gzip header carries no
file name and time 0), so the same documents always give the same SHA-256. Token counts use Laya's tokenizer without
special tokens, after replacing any special-token string that occurs in the text ([CLS], [SEP], [MASK] and the rest)
with a space, exactly as src/corpus/tokenize_blocks.py does before tokenizing, so a part's count is what training
reads. Progress files are written atomically, so a builder stopped at any point resumes from its last whole part.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, obj):
    """Write JSON atomically (a temporary file, then a rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def part_name(number):
    return f"part-{number:04d}.jsonl.gz"


def write_part(path, docs):
    """docs: (id, text) pairs. Returns the part's record: file, documents, bytes, sha256."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
            for doc_id, text in docs:
                gz.write((json.dumps({"id": str(doc_id), "text": text}, ensure_ascii=False) + "\n").encode("utf-8"))
    os.replace(tmp, path)
    return {"file": path.name, "documents": len(docs), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def read_part(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                yield row["id"], row["text"]


def load_tokenizer(path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(path))


def special_strings(tokenizer_json):
    """The text of every special token in a tokenizer.json (added tokens marked special)."""
    data = json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))
    return sorted({t["content"] for t in data.get("added_tokens", []) if t.get("special")}, key=len, reverse=True)


def strip_special(text, specials):
    for s in specials:
        if s in text:
            text = text.replace(s, " ")
    return text


def token_counts(tokenizer, texts, specials, batch=5000, max_chars=4_000_000):
    """Tokens per text with Laya's tokenizer, no special tokens added, special-token strings replaced first. Texts are
    encoded a batch at a time (at most `batch` texts and, past the first text, `max_chars` characters), because an
    encoding holds far more than its ids: 100,000 abstracts at once needed gigabytes. Each text is encoded on its own,
    so the batching does not change the counts."""
    counts, start = [], 0
    while start < len(texts):
        end, chars = start + 1, len(texts[start])
        while end < len(texts) and end - start < batch and chars + len(texts[end]) <= max_chars:
            chars += len(texts[end])
            end += 1
        encodings = tokenizer.encode_batch([strip_special(t, specials) for t in texts[start:end]], add_special_tokens=False)
        counts.extend(len(e.ids) for e in encodings)
        start = end
    return counts

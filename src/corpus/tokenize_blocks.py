"""Token blocks for session 5's MLM runs (docs/prompt_session5.md, READER METHOD and step 2), run on the pod.

For one source at a time, the part files (each checked against the SHA-256 in the source's manifest.json) are
tokenized with Laya's tokenizer without special tokens, after any special-token string in the text is replaced by a
space (src/corpus/parts.py). Held-out documents are chosen first, walking a seeded random order of the documents, until
their tokens fill the held-out blocks (1,000 per source, 200 for task); they are packed on their own, so no held-out
block shares a document with training. The other documents, in part order, are joined with [SEP] and cut into blocks of
1,024 tokens that start with [CLS] and end with [SEP] (1,022 tokens of text each); tokens after the last whole block
are dropped and counted. Writes <out>/<source>/:
- docs.u16: every document's tokens, concatenated (the intermediate the blocks are cut from);
- train.npy and heldout.npy: the blocks as uint16 NumPy arrays, shape (blocks, 1024);
- block_doc.npy: for each training block, the index of the document its first text token belongs to (document index,
  counting documents in part order);
- perm.npy: the seeded permutation of the training blocks that MLM runs walk, fixed here before the first MLM run;
- counts.json: documents, tokens, blocks, held-out documents and blocks, dropped tokens, the parts' SHA-256, and
  whether the tokens equal the manifest's count.
Also holds MixSampler, which src/train/pretrain_reader.py uses to draw blocks without replacement in a mix's token
shares by walking each source's permutation.

Usage on the pod: python src/corpus/tokenize_blocks.py --source abstracts --parts /root/eddm_s5/text/abstracts \
    --out /root/eddm_s5/blocks --tokenizer /root/eddm_s5/models/laya/tokenizer/tokenizer.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from parts import load_tokenizer, read_part, save_json, sha256_file, special_strings, strip_special  # noqa: E402

SEED = 20260927
BLOCK = 1024
CONTENT = BLOCK - 2
SOURCES = ("abstracts", "pmc", "case_reports", "web", "task")
HELDOUT = {"abstracts": 1000, "pmc": 1000, "case_reports": 1000, "web": 1000, "task": 200}


def special_id(tokenizer_json, content):
    for t in json.loads(Path(tokenizer_json).read_text(encoding="utf-8"))["added_tokens"]:
        if t["content"] == content:
            return int(t["id"])
    raise ValueError(f"{content} not in {tokenizer_json}")


def stream_length(lengths):
    """Tokens in the joined stream: each document followed by one [SEP]."""
    return int(np.sum(np.asarray(lengths, dtype=np.int64) + 1))


def choose_heldout(lengths, n_blocks, seed):
    """Document indices, in the order walked, whose joined tokens fill n_blocks blocks."""
    need = n_blocks * CONTENT
    order = np.random.default_rng(seed).permutation(len(lengths))
    total, chosen = 0, []
    for d in order:
        if total >= need:
            break
        chosen.append(int(d))
        total += int(lengths[d]) + 1
    if total < need:
        raise ValueError(f"only {total} tokens available for {n_blocks} held-out blocks")
    return chosen


def write_stream(dst, position, segment):
    """Write a 1-D run of text tokens at a stream position into blocks dst (rows of [CLS] text... [SEP]).
    Returns the new position; tokens beyond the last whole block are not written."""
    n, i = len(segment), 0
    while i < n:
        row, col = divmod(position, CONTENT)
        if row >= dst.shape[0]:
            return position + (n - i)
        take = min(CONTENT - col, n - i)
        dst[row, 1 + col : 1 + col + take] = segment[i : i + take]
        i += take
        position += take
    return position


def pack(doc_iter, n_blocks, dst, cls_id, sep_id, chunk_docs=20000):
    """Fill dst (n_blocks, BLOCK) from documents (1-D token arrays) joined with [SEP]; returns tokens not written."""
    dst[:, 0] = cls_id
    dst[:, BLOCK - 1] = sep_id
    position, pending = 0, []
    sep = np.array([sep_id], dtype=np.uint16)
    for tokens in doc_iter:
        pending.extend((np.asarray(tokens, dtype=np.uint16), sep))
        if len(pending) >= 2 * chunk_docs:
            position = write_stream(dst, position, np.concatenate(pending))
            pending = []
    if pending:
        position = write_stream(dst, position, np.concatenate(pending))
    return position - n_blocks * CONTENT


def first_documents(lengths, n_blocks):
    """For each block, the index (among these documents, in order) of the document of its first text token."""
    starts = np.concatenate([[0], np.cumsum(np.asarray(lengths, dtype=np.int64) + 1)[:-1]])
    return (np.searchsorted(starts, np.arange(n_blocks, dtype=np.int64) * CONTENT, side="right") - 1).astype(np.int32)


def tokenize_source(parts_dir, tokenizer_json, work, batch=10000):
    """Tokenize every document of a source into work/docs.u16 (concatenated) and return (lengths, parts info)."""
    manifest = json.loads((parts_dir / "manifest.json").read_text(encoding="utf-8"))
    tokenizer, specials = load_tokenizer(tokenizer_json), special_strings(tokenizer_json)
    lengths, parts = [], []
    with open(work / "docs.u16", "wb") as out:
        for part in manifest["parts"]:
            path = parts_dir / part["file"]
            if sha256_file(path) != part["sha256"]:
                raise SystemExit(f"{path}: SHA-256 differs from the manifest")
            parts.append({"file": part["file"], "sha256": part["sha256"]})
            texts = [t for _, t in read_part(path)]
            for start in range(0, len(texts), batch):
                encodings = tokenizer.encode_batch([strip_special(t, specials) for t in texts[start : start + batch]],
                                                   add_special_tokens=False)
                for e in encodings:
                    ids = np.asarray(e.ids, dtype=np.uint16)
                    out.write(ids.tobytes())
                    lengths.append(len(ids))
    return np.asarray(lengths, dtype=np.int64), parts, manifest


def build(source, parts_dir, out_root, tokenizer_json, n_heldout=None, seed=SEED):
    t0 = time.time()
    out = Path(out_root) / source
    out.mkdir(parents=True, exist_ok=True)
    n_heldout = HELDOUT[source] if n_heldout is None else n_heldout
    cls_id, sep_id = special_id(tokenizer_json, "[CLS]"), special_id(tokenizer_json, "[SEP]")
    lengths, parts, manifest = tokenize_source(Path(parts_dir), tokenizer_json, out)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    docs = np.memmap(out / "docs.u16", dtype=np.uint16, mode="r", shape=(int(offsets[-1]),))
    source_number = SOURCES.index(source)
    heldout_docs = choose_heldout(lengths, n_heldout, [seed, source_number, 1])
    held = set(heldout_docs)
    train_docs = [d for d in range(len(lengths)) if d not in held]

    heldout = np.lib.format.open_memmap(out / "heldout.npy", mode="w+", dtype=np.uint16, shape=(n_heldout, BLOCK))
    pack((docs[offsets[d] : offsets[d + 1]] for d in heldout_docs), n_heldout, heldout, cls_id, sep_id)
    heldout.flush()
    train_len = stream_length(lengths[train_docs])
    n_train = train_len // CONTENT
    train = np.lib.format.open_memmap(out / "train.npy", mode="w+", dtype=np.uint16, shape=(n_train, BLOCK))
    dropped = pack((docs[offsets[d] : offsets[d + 1]] for d in train_docs), n_train, train, cls_id, sep_id)
    train.flush()
    block_doc = np.asarray(train_docs, dtype=np.int64)[first_documents(lengths[train_docs], n_train)].astype(np.int32)
    np.save(out / "block_doc.npy", block_doc)
    perm = np.random.default_rng([seed, source_number, 2]).permutation(n_train)
    np.save(out / "perm.npy", perm)
    counts = {
        "source": source, "documents": int(len(lengths)), "tokens": int(lengths.sum()),
        "manifest_tokens": manifest.get("tokens"), "tokens_equal_manifest": int(lengths.sum()) == manifest.get("tokens"),
        "heldout_documents": len(heldout_docs), "heldout_blocks": n_heldout, "heldout_document_indices": heldout_docs,
        "train_documents": len(train_docs), "train_blocks": int(n_train), "train_tokens_in_blocks": int(n_train) * CONTENT,
        "dropped_tail_tokens": int(dropped), "block_tokens": BLOCK, "seed": seed, "parts": parts,
        "files": {"train": "train.npy", "heldout": "heldout.npy", "perm": "perm.npy", "block_doc": "block_doc.npy"},
        "seconds": round(time.time() - t0, 1),
    }
    save_json(out / "counts.json", counts)
    return counts


def open_source(blocks_root, source):
    """The training and held-out blocks of one source, memory-mapped, with its permutation and counts."""
    d = Path(blocks_root) / source
    return {"train": np.load(d / "train.npy", mmap_mode="r"), "heldout": np.load(d / "heldout.npy", mmap_mode="r"),
            "perm": np.load(d / "perm.npy"), "counts": json.loads((d / "counts.json").read_text(encoding="utf-8"))}


class MixSampler:
    """Draws blocks without replacement in a mix's token shares by walking each source's permutation.

    Every block holds 1,024 tokens, so token shares are block shares. Each draw goes to the source whose count is
    furthest below its share of all draws so far (ties to the source listed first), so the realized shares stay within
    one block of the target at every step and the order is deterministic. With passes above 1 (the task round), a
    source whose permutation is used up starts a fresh seeded permutation for its next pass; otherwise running out is
    an error. state() and the state argument carry the positions across a resume."""

    def __init__(self, shares, perms, passes=1, seed=SEED, state=None):
        total = float(sum(shares.values()))
        self.sources = list(shares)
        self.weight = {s: shares[s] / total for s in self.sources}
        self.base = perms
        self.passes, self.seed = passes, seed
        self.pos = {s: 0 for s in self.sources}
        self.drawn = {s: 0 for s in self.sources}
        self.pass_no = {s: 0 for s in self.sources}
        self.total = 0
        if state:
            self.pos, self.drawn, self.pass_no = dict(state["pos"]), dict(state["drawn"]), dict(state["pass_no"])
            self.total = int(state["total"])
        self._current = {s: self._perm(s) for s in self.sources}

    def _perm(self, s):
        if self.pass_no[s] == 0:
            return self.base[s]
        n = len(self.base[s])
        return np.random.default_rng([self.seed, SOURCES.index(s) if s in SOURCES else 99, 3, self.pass_no[s]]).permutation(n)

    def draw(self):
        s = max(self.sources, key=lambda x: (self.weight[x] * (self.total + 1) - self.drawn[x], -self.sources.index(x)))
        if self.pos[s] >= len(self._current[s]):
            if self.pass_no[s] + 1 >= self.passes:
                raise RuntimeError(f"source {s} has no training blocks left")
            self.pass_no[s] += 1
            self.pos[s] = 0
            self._current[s] = self._perm(s)
        block = int(self._current[s][self.pos[s]])
        self.pos[s] += 1
        self.drawn[s] += 1
        self.total += 1
        return s, block

    def draw_many(self, k):
        return [self.draw() for _ in range(k)]

    def state(self):
        return {"pos": dict(self.pos), "drawn": dict(self.drawn), "pass_no": dict(self.pass_no), "total": self.total}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", choices=SOURCES, required=True)
    parser.add_argument("--parts", type=Path, required=True, help="the source's part files and manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="blocks root; the source's folder is made inside it")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Laya's tokenizer.json")
    parser.add_argument("--heldout", type=int, default=None)
    args = parser.parse_args()
    counts = build(args.source, args.parts, args.out, args.tokenizer, args.heldout)
    print(json.dumps({k: v for k, v in counts.items() if k not in ("parts", "heldout_document_indices")}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

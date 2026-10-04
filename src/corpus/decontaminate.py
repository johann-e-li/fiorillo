"""The 13-word check for session 5's pretraining text (docs/prompt_session5.md, THE TEXT).

Text is lowercased, every character that is not a letter or digit becomes a space, and the text is split on
whitespace with empty strings dropped. A document is dropped if any 13 consecutive words in it also occur, in order,
in the contexts or long answer of any of the 1,000 pqa_labeled items. Each item's reference text is its contexts in
source order followed by its long answer, joined, so a run of 13 words that crosses from one section of the abstract
into the next also counts.

Shared by every builder: Checker(texts).matches(text) for one document, and ParallelChecker(reference_path) for many,
with multiprocessing (each worker builds its own reference set from the file, so nothing depends on string hashing
being the same across processes).

As a command it reads one part file (JSON lines, gzipped or not, fields id and text) and reports the documents that
match:
    python src/corpus/decontaminate.py --reference data/processed/corpus/session5/pubmedqa_labeled_text.jsonl \
        --input data/processed/corpus/session5/abstracts/part-0000.jsonl.gz [--processes 4]
"""
import argparse
import gzip
import json
import multiprocessing as mp
import re
import sys
from pathlib import Path

N_WORDS = 13
_NOT_LETTER_OR_DIGIT = re.compile(r"[\W_]+")  # \W is "not alphanumeric and not underscore"; the underscore goes too


def words(text):
    """Lowercase; every character that is not a letter or digit becomes a space; split on whitespace."""
    return _NOT_LETTER_OR_DIGIT.sub(" ", text.lower()).split()


def ngrams(tokens, n=N_WORDS):
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def reference_texts(path):
    """One text per pqa_labeled item: its contexts in order, then its long answer."""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                out.append(" ".join(list(item["contexts"]) + [item["long_answer"] or ""]))
    return out


class Checker:
    """The reference 13-grams and a fast test for one document.

    Only a run of at least 13 consecutive words that all occur somewhere in the reference can match, so the window is
    joined and looked up only inside such runs."""

    def __init__(self, texts, n=N_WORDS):
        self.n = n
        self.grams = set()
        for text in texts:
            self.grams |= ngrams(words(text), n)
        self.vocabulary = {w for gram in self.grams for w in gram.split(" ")}

    def matches(self, text):
        tokens, n, grams, vocabulary = words(text), self.n, self.grams, self.vocabulary
        run = 0
        for i, w in enumerate(tokens):
            if w in vocabulary:
                run += 1
                if run >= n and " ".join(tokens[i - n + 1 : i + 1]) in grams:
                    return True
            else:
                run = 0
        return False


_WORKER = None


def _init_worker(reference_path):
    global _WORKER
    _WORKER = Checker(reference_texts(reference_path))


def _check_chunk(texts):
    return [_WORKER.matches(t) for t in texts]


class ParallelChecker:
    """A pool of worker processes, each holding the reference; check(texts) returns one bool per text, in order."""

    def __init__(self, reference_path, processes=None):
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        self.pool = mp.get_context(method).Pool(processes, initializer=_init_worker, initargs=(str(reference_path),))

    def check(self, texts, chunk=500):
        texts = list(texts)
        chunks = [texts[i : i + chunk] for i in range(0, len(texts), chunk)]
        out = []
        for result in self.pool.map(_check_chunk, chunks):
            out.extend(result)
        return out

    def close(self):
        self.pool.close()
        self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_documents(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                yield row["id"], row["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--reference", type=Path, required=True, help="pubmedqa_labeled_text.jsonl")
    parser.add_argument("--input", type=Path, required=True, help="a part file: JSON lines with id and text")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    ids, texts = zip(*read_documents(args.input)) if args.input.stat().st_size else ((), ())
    with ParallelChecker(args.reference, args.processes) as checker:
        flags = checker.check(texts)
    hits = [i for i, hit in zip(ids, flags) if hit]
    print(json.dumps({"input": str(args.input), "documents": len(ids), "matches": len(hits), "matching_ids": hits[:50]}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

"""The task source for session 5 (docs/prompt_session5.md, THE TEXT and step 2).

Each distinct state among session 4's training records for both questions (pubmedqa_answer_train.jsonl and
readmit_window_train.jsonl; no validation or test records) becomes one document, serialized as
laya.common.build_sequence serializes a state (laya.common.serialize_state: json.dumps with ensure_ascii false),
without the question, options or label. States repeated in the records (the pqa_labeled items are written three
times) are kept once, at their first appearance. Writes part-0000.jsonl.gz (and more parts if needed, 100,000
documents each) with manifest.json: records read and distinct states per question, tokens, and the SHA-256 of each
records file, which must equal results/public_v0/checksums.json.

Run on the pod (step 8): python src/corpus/build_task_text.py --out /root/eddm_s5/text/task
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from parts import load_tokenizer, part_name, save_json, sha256_file, special_strings, token_counts, write_part  # noqa: E402

QIDS = ("pubmedqa_answer", "readmit_window")


def serialize(state):
    from laya.common import serialize_state

    return serialize_state(state)


def distinct_states(records_by_qid):
    """(id, text) for each distinct serialized state, in order of first appearance; and counts per question."""
    seen, docs, counts = set(), [], {}
    for qid, records in records_by_qid.items():
        n_read, n_new = 0, 0
        for r in records:
            n_read += 1
            text = serialize(r["state"])
            if text in seen:
                continue
            seen.add(text)
            docs.append((r["id"], text))
            n_new += 1
        counts[qid] = {"records": n_read, "distinct_states": n_new}
    return docs, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "processed" / "public")
    parser.add_argument("--checksums", type=Path, default=ROOT / "results" / "public_v0" / "checksums.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True, help="Laya's tokenizer.json")
    parser.add_argument("--per-part", type=int, default=100_000)
    args = parser.parse_args()
    expected = json.loads(args.checksums.read_text(encoding="utf-8"))
    records_by_qid, digests = {}, {}
    for qid in QIDS:
        path = args.records_dir / f"{qid}_train.jsonl"
        digests[path.name] = sha256_file(path)
        if digests[path.name] != expected.get(path.name):
            raise SystemExit(f"{path.name}: SHA-256 {digests[path.name]} does not equal results/public_v0/checksums.json")
        with open(path, encoding="utf-8") as f:
            records_by_qid[qid] = [json.loads(line) for line in f if line.strip()]
    docs, counts = distinct_states(records_by_qid)
    tokenizer, specials = load_tokenizer(args.tokenizer), special_strings(args.tokenizer)
    parts, tokens = [], 0
    for start in range(0, len(docs), args.per_part):
        chunk = docs[start : start + args.per_part]
        n = token_counts(tokenizer, [t for _, t in chunk], specials)
        info = write_part(args.out / part_name(len(parts)), chunk)
        info["tokens"] = int(sum(n))
        tokens += info["tokens"]
        parts.append(info)
    manifest = {"source": "session 4 training records, distinct states", "per_question": counts, "documents": len(docs),
                "tokens": tokens, "parts": parts, "records_sha256": digests}
    save_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "parts"}, indent=1))


if __name__ == "__main__":
    sys.exit(main())

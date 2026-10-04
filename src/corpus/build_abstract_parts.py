"""The abstracts source for session 5 (docs/prompt_session5.md, THE TEXT and step 2).

From corpus v1 (data/processed/corpus/pubmed_abstracts_v1/, 1,334 parquet shards), whole shards are taken in a seeded
random order (seed 20260927). In each shard, in its own row order, a record is dropped if its PMID is in
pubmedqa_pmids.txt or its text fails the 13-word check (src/corpus/decontaminate.py); otherwise its text, title then
abstract joined by a space (as src/corpus/token_estimate.py measured it), is written. Parts of 100,000 documents,
data/processed/corpus/session5/abstracts/part-<nnnn>.jsonl.gz (fields id, the PMID, and text), are written until
7,000,000 documents exist; the last shard is cut where the total is reached. Tokens are counted exactly with Laya's
tokenizer (src/corpus/parts.py). manifest.json records the parts (documents, tokens, SHA-256), the exact tokens, the
drops by reason, the shards in order with the rows read from each, and the input checksums; a copy goes to
results/reader_v1/text_manifests/abstracts_manifest.json.

Resumable: the linked shell ends every command after about three minutes, so each call works until --max-seconds and
stops after a whole part; progress.json in the output folder holds the position after the last part, and the next
call continues from there. Run on the Mac from the linked shell's python3 until it prints "complete":
    python3 src/corpus/build_abstract_parts.py [--max-seconds 150] [--processes 4]
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from decontaminate import ParallelChecker  # noqa: E402
from parts import (load_json, load_tokenizer, part_name, save_json, sha256_file, special_strings,  # noqa: E402
                   token_counts, write_part)

SEED = 20260927
SHARDS = ROOT / "data" / "processed" / "corpus" / "pubmed_abstracts_v1"
SESSION5 = ROOT / "data" / "processed" / "corpus" / "session5"
TOKENIZER = ROOT / "models" / "tokenizers" / "laya" / "tokenizer.json"
MANIFESTS = ROOT / "results" / "reader_v1" / "text_manifests"


def shard_order(names, seed=SEED):
    names = sorted(names)
    random.Random(seed).shuffle(names)
    return names


def record_text(title, abstract):
    return f"{title or ''} {abstract}".strip()


def load_exclusion(path):
    return {int(line) for line in Path(path).read_text(encoding="utf-8").split() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shards", type=Path, default=SHARDS)
    parser.add_argument("--out", type=Path, default=SESSION5 / "abstracts")
    parser.add_argument("--exclusion", type=Path, default=SESSION5 / "pubmedqa_pmids.txt")
    parser.add_argument("--reference", type=Path, default=SESSION5 / "pubmedqa_labeled_text.jsonl")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER)
    parser.add_argument("--manifests", type=Path, default=MANIFESTS)
    parser.add_argument("--documents", type=int, default=7_000_000)
    parser.add_argument("--per-part", type=int, default=100_000)
    parser.add_argument("--processes", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    t0 = time.time()
    progress_path = args.out / "progress.json"
    inputs = {"exclusion_sha256": sha256_file(args.exclusion), "reference_sha256": sha256_file(args.reference),
              "tokenizer_sha256": sha256_file(args.tokenizer)}
    state = load_json(progress_path)
    if state is None:
        state = {"seed": SEED, "target_documents": args.documents, "per_part": args.per_part, "inputs": inputs,
                 "order": shard_order([p.name for p in args.shards.glob("*.parquet")]),
                 "shard_pos": 0, "row": 0, "parts": [], "documents": 0, "tokens": 0,
                 "drops": {"pubmedqa_pmid": 0, "thirteen_word": 0}, "shards_used": [], "complete": False}
    elif state["inputs"] != inputs or state["target_documents"] != args.documents or state["per_part"] != args.per_part:
        raise SystemExit("progress.json was started with other inputs or settings; stop (nothing changed)")
    if state["complete"]:
        print("complete (nothing to do)")
        return
    print(f"resuming at shard {state['shard_pos']} ({state['order'][state['shard_pos']]}), row {state['row']}; "
          f"{state['documents']:,} documents in {len(state['parts'])} parts")

    exclusion = load_exclusion(args.exclusion)
    tokenizer, specials = load_tokenizer(args.tokenizer), special_strings(args.tokenizer)
    pos, row = state["shard_pos"], state["row"]
    drops = dict(state["drops"])
    shards_used = [dict(s) for s in state["shards_used"]]
    buffer = []
    last_part_seconds = 0.0
    with ParallelChecker(args.reference, args.processes) as checker:
        while state["documents"] + len(buffer) < args.documents:
            if pos >= len(state["order"]):
                raise SystemExit("corpus v1 ran out before the target; stop")
            name = state["order"][pos]
            table = pq.read_table(args.shards / name, columns=["pmid", "title", "abstract"]).to_pydict()
            n_rows = len(table["pmid"])
            texts = [record_text(t, a) for t, a in zip(table["title"], table["abstract"])]
            excluded = [int(p) in exclusion for p in table["pmid"]]
            candidates = [i for i in range(row, n_rows) if not excluded[i]]
            flags = dict(zip(candidates, checker.check([texts[i] for i in candidates])))
            if not shards_used or shards_used[-1]["shard"] != name:
                shards_used.append({"shard": name, "rows_read": 0, "documents_written": 0})
            part_start = time.time()
            while row < n_rows and state["documents"] + len(buffer) < args.documents:
                i = row
                row += 1
                shards_used[-1]["rows_read"] += 1
                if excluded[i]:
                    drops["pubmedqa_pmid"] += 1
                    continue
                if flags[i]:
                    drops["thirteen_word"] += 1
                    continue
                buffer.append((str(table["pmid"][i]), texts[i]))
                shards_used[-1]["documents_written"] += 1
                if len(buffer) == args.per_part or state["documents"] + len(buffer) == args.documents:
                    counts = token_counts(tokenizer, [t for _, t in buffer], specials)
                    info = write_part(args.out / part_name(len(state["parts"])), buffer)
                    info["tokens"] = int(sum(counts))
                    state["parts"].append(info)
                    state["documents"] += len(buffer)
                    state["tokens"] += info["tokens"]
                    at_end = row >= n_rows
                    state.update({"shard_pos": pos + 1 if at_end else pos, "row": 0 if at_end else row,
                                  "drops": dict(drops), "shards_used": [dict(s) for s in shards_used]})
                    save_json(progress_path, state)
                    buffer = []
                    last_part_seconds = time.time() - part_start
                    part_start = time.time()
                    print(f"{info['file']}: {info['documents']:,} documents, {info['tokens']:,} tokens; "
                          f"total {state['documents']:,} ({time.time() - t0:.0f} s)", flush=True)
                    if time.time() - t0 + 1.2 * last_part_seconds > args.max_seconds and state["documents"] < args.documents:
                        print("stopping here to stay inside the time limit; run again to continue")
                        return
            if row >= n_rows:
                pos, row = pos + 1, 0
    state["complete"] = True
    manifest = {
        "source": "corpus v1 (docs/corpus_manifest.md), title and abstract",
        "rules": "shards in a seeded random order; PubMedQA PMIDs and 13-word matches dropped; 7,000,000 documents",
        "seed": state["seed"], "documents": state["documents"], "tokens": state["tokens"],
        "tokens_per_document_mean": round(state["tokens"] / max(1, state["documents"]), 1),
        "drops": state["drops"], "parts": state["parts"], "shards_used": state["shards_used"], "inputs": state["inputs"],
        "tokenizer": str(args.tokenizer.relative_to(ROOT)) if args.tokenizer.is_relative_to(ROOT) else str(args.tokenizer),
    }
    save_json(args.out / "manifest.json", manifest)
    save_json(args.manifests / "abstracts_manifest.json", manifest)
    save_json(progress_path, state)
    print(f"complete: {state['documents']:,} documents, {state['tokens']:,} tokens, drops {state['drops']}")


if __name__ == "__main__":
    sys.exit(main())

"""Token estimate for the PubMed abstract corpus with the Laya tokenizer.

Draws a fixed random sample of records (seed 20260927) from the parquet shards, tokenizes title plus abstract
with the tokenizer of convaiinnovations/laya (models/tokenizers/laya/, pinned revision in REVISION.txt), and
writes results/corpus/pubmed_v1_tokens.json: sample size, mean and median tokens per record, the 95th and
99th percentiles, the share of records over 512 tokens, and the estimated total tokens for the whole corpus.

Usage: python src/corpus/token_estimate.py [--sample 20000] [--seed 20260927]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
SHARDS = ROOT / "data" / "processed" / "corpus" / "pubmed_abstracts_v1"
TOKENIZER = ROOT / "models" / "tokenizers" / "laya" / "tokenizer.json"
OUT = ROOT / "results" / "corpus" / "pubmed_v1_tokens.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    shards = sorted(SHARDS.glob("*.parquet"))
    rows_per_shard = np.array([pq.read_metadata(s).num_rows for s in shards])
    total = int(rows_per_shard.sum())
    picks = np.sort(rng.choice(total, size=args.sample, replace=False))
    offsets = np.concatenate([[0], np.cumsum(rows_per_shard)])
    tokenizer = Tokenizer.from_file(str(TOKENIZER))
    counts = []
    for i, shard in enumerate(shards):
        local = picks[(picks >= offsets[i]) & (picks < offsets[i + 1])] - offsets[i]
        if len(local) == 0:
            continue
        table = pq.read_table(shard, columns=["title", "abstract"]).take(local)
        texts = [f"{t or ''} {a}".strip() for t, a in zip(table["title"].to_pylist(), table["abstract"].to_pylist())]
        counts.extend(len(enc.ids) for enc in tokenizer.encode_batch(texts))
    counts = np.array(counts)
    result = {
        "tokenizer": "convaiinnovations/laya, revision " + (ROOT / "models" / "tokenizers" / "laya" / "REVISION.txt").read_text().split()[1],
        "text": "title plus abstract, no special tokens added",
        "corpus_records": total,
        "sample_size": int(len(counts)),
        "seed": args.seed,
        "tokens_mean": round(float(counts.mean()), 1),
        "tokens_median": float(np.median(counts)),
        "tokens_p95": float(np.percentile(counts, 95)),
        "tokens_p99": float(np.percentile(counts, 99)),
        "share_over_512_tokens": round(float((counts > 512).mean()), 4),
        "estimated_total_tokens": int(round(float(counts.mean()) * total)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

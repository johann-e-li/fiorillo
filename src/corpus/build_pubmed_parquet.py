"""PubMed extraction, pass 3: the abstract corpus, one parquet shard per baseline file.

For one baseline file's pass 1 output, keeps the records whose pmid is in that file's keep list (pass 2),
whose language is "eng", and whose abstract has at least 100 characters, and writes
data/processed/corpus/pubmed_abstracts_v1/<stem>.parquet (the same columns as the JSON) plus
<stem>.counts.json (read, dropped_superseded_or_deleted, dropped_non_english, dropped_short, written).
Outputs are written to a temporary name and renamed. A file whose two outputs exist is skipped.

Usage: python src/corpus/build_pubmed_parquet.py <stem> [<stem> ...]   (stem such as pubmed26n0001)
"""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus"
MIN_ABSTRACT_CHARS = 100
SCHEMA = pa.schema(
    [
        ("pmid", pa.int64()),
        ("version", pa.int32()),
        ("file_no", pa.int32()),
        ("year", pa.int32()),
        ("language", pa.string()),
        ("journal", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("pub_types", pa.list_(pa.string())),
        ("copyright", pa.bool_()),
        ("mesh_count", pa.int32()),
    ]
)


def build(stem, extract_dir, keep_dir, out_dir):
    out_path = out_dir / f"{stem}.parquet"
    counts_path = out_dir / f"{stem}.counts.json"
    if out_path.exists() and counts_path.exists():
        return stem, "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(pq.read_table(keep_dir / f"{stem}.parquet")["pmid"].to_pylist())
    counts = {"read": 0, "dropped_superseded_or_deleted": 0, "dropped_non_english": 0, "dropped_short": 0, "written": 0}
    rows = []
    with gzip.open(extract_dir / f"{stem}.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            record = json.loads(line)
            counts["read"] += 1
            if record["pmid"] not in keep:
                counts["dropped_superseded_or_deleted"] += 1
            elif record["language"] != "eng":
                counts["dropped_non_english"] += 1
            elif len(record["abstract"]) < MIN_ABSTRACT_CHARS:
                counts["dropped_short"] += 1
            else:
                rows.append(record)
                counts["written"] += 1
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    tmp = out_dir / f"{stem}.parquet.tmp{os.getpid()}"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out_path)
    tmp_counts = out_dir / f"{stem}.counts.json.tmp{os.getpid()}"
    tmp_counts.write_text(json.dumps(counts) + "\n", encoding="utf-8")
    os.replace(tmp_counts, counts_path)
    return stem, ", ".join(f"{k} {v}" for k, v in counts.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("stems", nargs="+")
    parser.add_argument("--extract", type=Path, default=CORPUS / "pubmed-extract")
    parser.add_argument("--keep", type=Path, default=CORPUS / "pubmed-keep")
    parser.add_argument("--out", type=Path, default=CORPUS / "pubmed_abstracts_v1")
    args = parser.parse_args()
    for stem in args.stems:
        name, note = build(stem, args.extract, args.keep, args.out)
        print(f"{name}: {note}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

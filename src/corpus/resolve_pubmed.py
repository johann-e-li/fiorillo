"""PubMed extraction, pass 2: decide which record of each PMID survives.

Reads every <stem>.index.parquet from pass 1. Every PMID named in a DeleteCitation is removed. For each
remaining PMID, the occurrence with the highest file_no is kept, because later baseline files supersede
earlier ones. Writes data/processed/corpus/pubmed-keep/<stem>.parquet (one column, pmid, the records to keep
from that file) and results/corpus/pubmed_v1_resolve.json with the counts.

Usage: python src/corpus/resolve_pubmed.py [--extract DIR] [--keep DIR] [--out FILE]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def resolve(extract_dir, keep_dir, out_file):
    index_files = sorted(extract_dir.glob("*.index.parquet"))
    if not index_files:
        raise SystemExit(f"no index files in {extract_dir}")
    table = pa.concat_tables([pq.read_table(f, columns=["pmid", "file_no", "deleted"]) for f in index_files])
    pmid = table["pmid"].to_numpy()
    file_no = table["file_no"].to_numpy()
    deleted = table["deleted"].to_numpy(zero_copy_only=False)
    articles_seen = int((~deleted).sum())
    deleted_pmids = np.unique(pmid[deleted])
    live_pmid, live_file = pmid[~deleted], file_no[~deleted]
    order = np.lexsort((live_file, live_pmid))              # sorted by pmid, then file_no
    live_pmid, live_file = live_pmid[order], live_file[order]
    last = np.ones(len(live_pmid), dtype=bool)
    last[:-1] = live_pmid[1:] != live_pmid[:-1]            # last occurrence of each pmid has the highest file_no
    distinct = int(last.sum())
    keep_pmid, keep_file = live_pmid[last], live_file[last]
    not_deleted = ~np.isin(keep_pmid, deleted_pmids)
    removed_by_deletion = int((~not_deleted).sum())
    keep_pmid, keep_file = keep_pmid[not_deleted], keep_file[not_deleted]
    keep_dir.mkdir(parents=True, exist_ok=True)
    stems = {int(f.name.split("n")[1].split(".")[0]): f.name[: -len(".index.parquet")] for f in index_files}
    for number, stem in stems.items():
        chosen = keep_pmid[keep_file == number]
        pq.write_table(pa.table({"pmid": pa.array(np.sort(chosen), pa.int64())}), keep_dir / f"{stem}.parquet", compression="zstd")
    counts = {
        "index_files": len(index_files),
        "articles_seen": articles_seen,
        "distinct_pmids": distinct,
        "superseded_duplicates": articles_seen - distinct,
        "delete_citations": int(len(deleted_pmids)),
        "removed_by_deletion": removed_by_deletion,
        "kept": int(len(keep_pmid)),
    }
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--extract", type=Path, default=ROOT / "data" / "processed" / "corpus" / "pubmed-extract")
    parser.add_argument("--keep", type=Path, default=ROOT / "data" / "processed" / "corpus" / "pubmed-keep")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "corpus" / "pubmed_v1_resolve.json")
    args = parser.parse_args()
    for key, value in resolve(args.extract, args.keep, args.out).items():
        print(f"{key}: {value:,}")


if __name__ == "__main__":
    main()

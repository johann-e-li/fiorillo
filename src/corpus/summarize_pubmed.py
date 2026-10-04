"""PubMed corpus summary: combine the pass 3 counts and the parquet shards into aggregate result tables.

Per shard (resumable; a shard with an existing stats file is skipped), writes <stem>.stats.json and
<stem>.lengths.npy under data/processed/corpus/pubmed-stats/. Once every shard has stats, writes:
  results/corpus/pubmed_v1_summary.json       totals from the counts files; abstracts; characters; mean and median
                                              abstract length; records with copyright notices
  results/corpus/pubmed_v1_year_counts.csv    year, records
  results/corpus/pubmed_v1_journal_top100.csv journal, records

Usage: python src/corpus/summarize_pubmed.py [--budget SECONDS]
"""
import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus"
RESULTS = ROOT / "results" / "corpus"


def shard_stats(shard, stats_dir):
    stem = shard.name[: -len(".parquet")]
    stats_path = stats_dir / f"{stem}.stats.json"
    if stats_path.exists():
        return False
    table = pq.read_table(shard, columns=["abstract", "year", "journal", "copyright"])
    lengths = pc.utf8_length(table["abstract"]).to_numpy().astype(np.int32)
    years = Counter(y for y in table["year"].to_pylist() if y is not None)
    journals = Counter(j for j in table["journal"].to_pylist() if j)
    stats = {
        "records": table.num_rows,
        "characters": int(lengths.sum()),
        "copyright": int(pc.sum(table["copyright"]).as_py() or 0),
        "year_missing": table.num_rows - sum(years.values()),
        "years": {str(k): v for k, v in years.items()},
        "journals": dict(journals),
    }
    np.save(stats_dir / f"{stem}.lengths.npy", lengths)
    stats_path.write_text(json.dumps(stats) + "\n", encoding="utf-8")
    return True


def finalize(shard_dir, stats_dir):
    counts_total = Counter()
    for counts_file in sorted(shard_dir.glob("*.counts.json")):
        counts_total.update(json.loads(counts_file.read_text(encoding="utf-8")))
    records = characters = copyright = year_missing = 0
    years, journals, lengths = Counter(), Counter(), []
    for stats_file in sorted(stats_dir.glob("*.stats.json")):
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        records += stats["records"]
        characters += stats["characters"]
        copyright += stats["copyright"]
        year_missing += stats["year_missing"]
        years.update({int(k): v for k, v in stats["years"].items()})
        journals.update(stats["journals"])
        lengths.append(np.load(stats_dir / stats_file.name.replace(".stats.json", ".lengths.npy")))
    lengths = np.concatenate(lengths)
    assert records == counts_total["written"] == len(lengths)
    summary = {
        "shards": len(list(shard_dir.glob("*.parquet"))),
        "pass1_records_with_abstract": counts_total["read"],
        "dropped_superseded_or_deleted": counts_total["dropped_superseded_or_deleted"],
        "dropped_non_english": counts_total["dropped_non_english"],
        "dropped_short_under_100_chars": counts_total["dropped_short"],
        "abstracts": records,
        "characters": characters,
        "abstract_chars_mean": round(float(lengths.mean()), 1),
        "abstract_chars_median": float(np.median(lengths)),
        "abstract_chars_p95": float(np.percentile(lengths, 95)),
        "records_with_copyright_notice": copyright,
        "records_without_year": year_missing,
        "distinct_journals": len(journals),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "pubmed_v1_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with open(RESULTS / "pubmed_v1_year_counts.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["year", "records"])
        writer.writerows(sorted(years.items()))
    with open(RESULTS / "pubmed_v1_journal_top100.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["journal", "records"])
        writer.writerows(journals.most_common(100))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--shards", type=Path, default=CORPUS / "pubmed_abstracts_v1")
    parser.add_argument("--stats", type=Path, default=CORPUS / "pubmed-stats")
    parser.add_argument("--budget", type=float, default=0, help="stop starting new shards after this many seconds")
    args = parser.parse_args()
    args.stats.mkdir(parents=True, exist_ok=True)
    shards = sorted(args.shards.glob("*.parquet"))
    started = time.time()
    done = 0
    for shard in shards:
        if args.budget and time.time() - started > args.budget:
            break
        done += shard_stats(shard, args.stats)
    have = len(list(args.stats.glob("*.stats.json")))
    print(f"shard stats: {have} of {len(shards)} (+{done})")
    if have == len(shards):
        for key, value in finalize(args.shards, args.stats).items():
            print(f"{key}: {value:,}" if isinstance(value, int) else f"{key}: {value}")


if __name__ == "__main__":
    main()

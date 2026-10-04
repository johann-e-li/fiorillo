"""Passes 2 and 3 of the PubMed extraction on the invented sample: superseded and deleted records, filters."""
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))

from src.corpus.build_pubmed_parquet import build  # noqa: E402
from src.corpus.extract_pubmed import extract  # noqa: E402
from src.corpus.resolve_pubmed import resolve  # noqa: E402

SAMPLE = SRC / "tests" / "fixtures" / "pubmed-sample"


def test_resolve_keeps_latest_file_and_drops_deleted(tmp_path):
    for name in ("pubmed00n0001.xml.gz", "pubmed00n0002.xml.gz"):
        extract(SAMPLE / name, tmp_path / "extract")
    counts = resolve(tmp_path / "extract", tmp_path / "keep", tmp_path / "resolve.json")
    assert counts == {"index_files": 2, "articles_seen": 8, "distinct_pmids": 7, "superseded_duplicates": 1,
                      "delete_citations": 1, "removed_by_deletion": 1, "kept": 6}
    assert pq.read_table(tmp_path / "keep" / "pubmed00n0001.parquet")["pmid"].to_pylist() == [1001, 1002, 1003, 1004]
    assert pq.read_table(tmp_path / "keep" / "pubmed00n0002.parquet")["pmid"].to_pylist() == [1005, 2001]


def test_build_applies_keep_list_language_and_length_filters(tmp_path):
    for name in ("pubmed00n0001.xml.gz", "pubmed00n0002.xml.gz"):
        extract(SAMPLE / name, tmp_path / "extract")
    resolve(tmp_path / "extract", tmp_path / "keep", tmp_path / "resolve.json")
    build("pubmed00n0001", tmp_path / "extract", tmp_path / "keep", tmp_path / "out")
    counts = json.loads((tmp_path / "out" / "pubmed00n0001.counts.json").read_text())
    # 1005 superseded by file 2 and 1006 deleted; 1003 is French; 1001 and 1004 are under 100 characters
    assert counts == {"read": 5, "dropped_superseded_or_deleted": 2, "dropped_non_english": 1, "dropped_short": 2, "written": 0}
    table = pq.read_table(tmp_path / "out" / "pubmed00n0001.parquet")
    assert table.num_rows == 0
    assert table.schema.names == ["pmid", "version", "file_no", "year", "language", "journal", "title", "abstract", "pub_types", "copyright", "mesh_count"]

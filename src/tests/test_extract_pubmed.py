"""Tests for src/corpus/extract_pubmed.py on the invented two-file sample in src/tests/fixtures/pubmed-sample/."""
import gzip
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))

from src.corpus.extract_pubmed import extract  # noqa: E402

SAMPLE = SRC / "tests" / "fixtures" / "pubmed-sample"


def run(tmp_path):
    for name in ("pubmed00n0001.xml.gz", "pubmed00n0002.xml.gz"):
        extract(SAMPLE / name, tmp_path)
    records = {}
    for name in ("pubmed00n0001", "pubmed00n0002"):
        with gzip.open(tmp_path / f"{name}.jsonl.gz", "rt", encoding="utf-8") as fh:
            records[name] = [json.loads(line) for line in fh]
    return records


def test_structured_abstract_is_joined_with_labels_and_cleaned(tmp_path):
    records = run(tmp_path)
    first = {r["pmid"]: r for r in records["pubmed00n0001"]}[1001]
    assert first["abstract"] == "BACKGROUND: Invented background sentence with a gene name. RESULTS: Invented results sentence."
    assert first["copyright"] is True
    assert first["mesh_count"] == 2
    assert first["pub_types"] == ["Journal Article"]
    assert first["journal"] == "Invent J Med"
    assert list(first) == ["pmid", "version", "file_no", "year", "language", "journal", "title", "abstract", "pub_types", "copyright", "mesh_count"]


def test_year_from_pubdate_year_or_medlinedate(tmp_path):
    records = run(tmp_path)
    by_pmid = {r["pmid"]: r for r in records["pubmed00n0001"]}
    assert by_pmid[1001]["year"] == 2020
    assert by_pmid[1004]["year"] == 1998


def test_only_records_with_abstracts_are_written_and_language_is_kept(tmp_path):
    records = run(tmp_path)
    pmids = sorted(r["pmid"] for r in records["pubmed00n0001"])
    assert pmids == [1001, 1003, 1004, 1005, 1006]
    assert {r["pmid"]: r["language"] for r in records["pubmed00n0001"]}[1003] == "fre"


def test_index_lists_every_article_and_deletion(tmp_path):
    run(tmp_path)
    index1 = pq.read_table(tmp_path / "pubmed00n0001.index.parquet").to_pylist()
    index2 = pq.read_table(tmp_path / "pubmed00n0002.index.parquet").to_pylist()
    assert sorted(r["pmid"] for r in index1) == [1001, 1002, 1003, 1004, 1005, 1006]
    assert all(r["deleted"] is False and r["file_no"] == 1 for r in index1)
    deleted = [r for r in index2 if r["deleted"]]
    assert [r["pmid"] for r in deleted] == [1006]
    assert {r["pmid"]: r["version"] for r in index2 if not r["deleted"]} == {1005: 2, 2001: 1}


def test_existing_outputs_are_skipped(tmp_path):
    run(tmp_path)
    assert extract(SAMPLE / "pubmed00n0001.xml.gz", tmp_path)[1] == "skipped"

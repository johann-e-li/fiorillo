"""PubMed extraction, pass 1: one baseline file in, two files out.

Reads one PubMed annual baseline file (pubmedNNnNNNN.xml.gz) with a streaming parser and writes, under
data/processed/corpus/pubmed-extract/:
  <stem>.jsonl.gz       one JSON line per PubmedArticle that has a non-empty abstract
  <stem>.index.parquet  every PubmedArticle's pmid and version, plus every PMID named in DeleteCitation,
                        with columns pmid, version, file_no, deleted
The JSON fields, in order: pmid, version, file_no, year, language, journal, title, abstract, pub_types,
copyright, mesh_count. The abstract joins every AbstractText element in order, prefixing labelled sections
with "LABEL: ", collapses whitespace, and is Unicode NFC. Outputs are written to a temporary name and renamed,
so a file is either complete or absent. A file whose two outputs exist is skipped.

Usage: python src/corpus/extract_pubmed.py <file.xml.gz> [<file.xml.gz> ...] [--out DIR]
"""
import argparse
import gzip
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from lxml import etree

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "data" / "processed" / "corpus" / "pubmed-extract"
WHITESPACE = re.compile(r"\s+")
FILE_NO = re.compile(r"n(\d+)\.xml\.gz$")


def clean(text):
    if text is None:
        return ""
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def element_text(element):
    """All text inside an element, including inline markup such as <i> or <sup>."""
    return clean("".join(element.itertext())) if element is not None else ""


def parse_year(article):
    pub_date = article.find("MedlineCitation/Article/Journal/JournalIssue/PubDate")
    if pub_date is None:
        return None
    year = pub_date.findtext("Year")
    if year and year.strip().isdigit():
        return int(year.strip())
    medline_date = pub_date.findtext("MedlineDate") or ""
    match = re.search(r"\d{4}", medline_date)
    return int(match.group()) if match else None


def parse_abstract(article):
    abstract = article.find("MedlineCitation/Article/Abstract")
    if abstract is None:
        return "", False
    parts = []
    for section in abstract.findall("AbstractText"):
        text = element_text(section)
        if not text:
            continue
        label = clean(section.get("Label"))
        parts.append(f"{label}: {text}" if label else text)
    has_copyright = abstract.find("CopyrightInformation") is not None
    return clean(" ".join(parts)), has_copyright


def record_for(article, file_no):
    citation = article.find("MedlineCitation")
    pmid_element = citation.find("PMID")
    pmid = int(pmid_element.text.strip())
    version = int(pmid_element.get("Version", "1"))
    abstract, has_copyright = parse_abstract(article)
    journal = clean(citation.findtext("Article/Journal/ISOAbbreviation")) or clean(
        citation.findtext("Article/Journal/Title")
    )
    return {
        "pmid": pmid,
        "version": version,
        "file_no": file_no,
        "year": parse_year(article),
        "language": clean(citation.findtext("Article/Language")) or None,
        "journal": journal or None,
        "title": element_text(citation.find("Article/ArticleTitle")) or None,
        "abstract": abstract,
        "pub_types": [clean(p.text) for p in citation.findall("Article/PublicationTypeList/PublicationType") if p.text],
        "copyright": has_copyright,
        "mesh_count": len(citation.findall("MeshHeadingList/MeshHeading")),
    }


def extract(source, out_dir):
    source = Path(source)
    match = FILE_NO.search(source.name)
    if not match:
        raise ValueError(f"{source.name} is not a PubMed baseline file name")
    file_no = int(match.group(1))
    stem = source.name[: -len(".xml.gz")]
    jsonl_path = out_dir / f"{stem}.jsonl.gz"
    index_path = out_dir / f"{stem}.index.parquet"
    if jsonl_path.exists() and index_path.exists():
        return stem, "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_jsonl = out_dir / f"{stem}.jsonl.gz.tmp{os.getpid()}"
    tmp_index = out_dir / f"{stem}.index.parquet.tmp{os.getpid()}"
    pmids, versions, deleted = [], [], []
    written = 0
    with gzip.open(source, "rb") as fh, gzip.open(tmp_jsonl, "wt", encoding="utf-8", compresslevel=6) as out:
        for _, element in etree.iterparse(fh, events=("end",), tag=("PubmedArticle", "DeleteCitation")):
            if element.tag == "DeleteCitation":
                for pmid in element.findall("PMID"):
                    if pmid.text and pmid.text.strip().isdigit():
                        pmids.append(int(pmid.text.strip()))
                        versions.append(int(pmid.get("Version", "1")))
                        deleted.append(True)
            else:
                record = record_for(element, file_no)
                pmids.append(record["pmid"])
                versions.append(record["version"])
                deleted.append(False)
                if record["abstract"]:
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
            element.clear()
            while element.getprevious() is not None:
                del element.getparent()[0]
    table = pa.table(
        {
            "pmid": pa.array(pmids, pa.int64()),
            "version": pa.array(versions, pa.int32()),
            "file_no": pa.array([file_no] * len(pmids), pa.int32()),
            "deleted": pa.array(deleted, pa.bool_()),
        }
    )
    pq.write_table(table, tmp_index, compression="zstd")
    os.replace(tmp_index, index_path)
    os.replace(tmp_jsonl, jsonl_path)
    return stem, f"{len(pmids)} articles and deletions, {written} with abstracts"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for source in args.files:
        stem, note = extract(source, args.out)
        print(f"{stem}: {note}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

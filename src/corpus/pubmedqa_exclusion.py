"""PubMedQA exclusion files for session 5 (docs/prompt_session5.md, step 2).

Downloads the three PubMedQA configs, pqa_labeled, pqa_artificial and pqa_unlabeled, from the Hugging Face dataset
qiaojin/PubMedQA at revision 9001f2853fb87cab8d220904e0de81ac6973b318 into data/raw/public_s5/, each file checked
against the SHA-256 that Hugging Face lists for it at that revision (pqa_labeled and pqa_artificial are the same files
session 4 used). Then writes:
- data/processed/corpus/session5/pubmedqa_pmids.txt: every pubid of the three configs, sorted as numbers, one per line
  (every builder drops a document whose PMID is on this list);
- data/processed/corpus/session5/pubmedqa_labeled_text.jsonl: the 1,000 pqa_labeled items, one per line, with pubid,
  contexts (in source order) and long_answer (the text of the 13-word check);
- results/reader_v1/text_manifests/pubmedqa_exclusion.json: counts per config and the SHA-256 of every file.

Run on the Mac from the linked shell's python3 (step 7): python3 src/corpus/pubmedqa_exclusion.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "corpus"))
import records as R  # noqa: E402
from parts import save_json, sha256_file  # noqa: E402

REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
FILES = {
    "pqa_labeled": ("pqa_labeled/train-00000-of-00001.parquet", "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c"),
    "pqa_artificial": ("pqa_artificial/train-00000-of-00001.parquet", "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05"),
    "pqa_unlabeled": ("pqa_unlabeled/train-00000-of-00001.parquet", "2f390ea5e6e3dc53b2736a52be2f594e05f5d8a599c3de754ad72f1068251e6e"),
}
RAW = ROOT / "data" / "raw" / "public_s5"
OUT = ROOT / "data" / "processed" / "corpus" / "session5"
MANIFESTS = ROOT / "results" / "reader_v1" / "text_manifests"


def url(path_in_repo):
    return f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{REVISION}/{path_in_repo}"


def labeled_items(rows):
    """pubid, contexts and long answer of each pqa_labeled row, sorted by pubid as a number."""
    items = [{"pubid": int(r["pubid"]), "contexts": list(r["context"]["contexts"]), "long_answer": r["long_answer"]} for r in rows]
    return sorted(items, key=lambda x: x["pubid"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifests", type=Path, default=MANIFESTS)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    pubids, counts, checksums = set(), {}, {}
    labeled = None
    for config, (path_in_repo, sha) in FILES.items():
        local = R.fetch(url(path_in_repo), args.raw / f"{config}.parquet", sha)
        checksums[f"{config}.parquet"] = sha
        ids = [int(x) for x in pq.read_table(local, columns=["pubid"]).column("pubid").to_pylist()]
        counts[config] = {"rows": len(ids), "distinct_pubids": len(set(ids))}
        pubids.update(ids)
        if config == "pqa_labeled":
            labeled = labeled_items(pq.read_table(local, columns=["pubid", "context", "long_answer"]).to_pylist())
    if labeled is None or len(labeled) != 1000:
        raise SystemExit(f"expected 1,000 pqa_labeled items, found {0 if labeled is None else len(labeled)}")

    args.out.mkdir(parents=True, exist_ok=True)
    pmid_path = args.out / "pubmedqa_pmids.txt"
    pmid_path.write_text("".join(f"{p}\n" for p in sorted(pubids)), encoding="utf-8")
    text_path = args.out / "pubmedqa_labeled_text.jsonl"
    with open(text_path, "w", encoding="utf-8", newline="\n") as f:
        for item in labeled:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "source": f"qiaojin/PubMedQA at revision {REVISION}",
        "configs": counts,
        "distinct_pubids_all_configs": len(pubids),
        "labeled_items": len(labeled),
        "downloads_sha256": checksums,
        "outputs_sha256": {pmid_path.name: sha256_file(pmid_path), text_path.name: sha256_file(text_path)},
    }
    save_json(args.manifests / "pubmedqa_exclusion.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())

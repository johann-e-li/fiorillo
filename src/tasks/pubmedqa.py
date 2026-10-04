"""Public v0 records for PubMedQA, question pubmedqa_answer (docs/prompt_session4.md, step 4).

Sources: Hugging Face dataset qiaojin/PubMedQA at a fixed revision (configs pqa_labeled, 1,000 expert-labeled
items, and pqa_artificial, 211,269 automatically labeled items), and the official 500-item test list
data/test_ground_truth.json from github.com/pubmedqa/pubmedqa at a fixed commit. MIT licence.

Splits (as corrected on 27 September 2026, before any run; see docs/prompt_session4.md, correction 1). Test:
the 500 official test items. Validation: 150 pqa_labeled items not in the test list (used for checkpoint selection
and the temperature; 50 in runs 1 and 2, changed by correction 7 because 50 items gave a noisy selection score and
temperature), plus 2,000 pqa_artificial items marked calibrate false (monitoring only). Train: the other
350 pqa_labeled items (450 in runs 1 and 2), each written 3 times (10 in run 1; changed by correction 4, because ten repeats let the model
memorize them), plus 20,000 pqa_artificial items. The pqa_artificial samples are
drawn by label so that yes and no appear in the same ratio as in the 500 non-test pqa_labeled items (pqa_artificial
has no maybe). Sampling by seed 20260927. pqa_artificial items whose PubMed ID is also in pqa_labeled are dropped
and counted, so no PubMed ID appears in two splits. State: the question and the abstract's contexts as a list of {section, text}, in source
order. Never in the state: long_answer, final_decision, reasoning_required_pred, reasoning_free_pred, MeSH terms.
Label: yes 0, no 1, maybe 2. group.source records pqa_labeled or pqa_artificial (evaluation only).

Usage: python src/tasks/pubmedqa.py [--raw data/raw/public/pubmedqa] [--out data/processed/public]
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import records as R  # noqa: E402

HF_REVISION = "9001f2853fb87cab8d220904e0de81ac6973b318"
GITHUB_COMMIT = "3e7616d288b52edaf38f5fd5c59821c7806bde86"
SOURCES = {
    "pqa_labeled.parquet": (
        f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{HF_REVISION}/pqa_labeled/train-00000-of-00001.parquet",
        "3d56bd1abc11884579ecc3aab9dd3cdfce8b7cf54715daeca93b8022b7be231c",
    ),
    "pqa_artificial.parquet": (
        f"https://huggingface.co/datasets/qiaojin/PubMedQA/resolve/{HF_REVISION}/pqa_artificial/train-00000-of-00001.parquet",
        "cdee2d3c01369f47d2ea7b89cd32ee64b142cc0debcb863bf39b3a81a3cece05",
    ),
    "test_ground_truth.json": (
        f"https://raw.githubusercontent.com/pubmedqa/pubmedqa/{GITHUB_COMMIT}/data/test_ground_truth.json",
        "939fe566f09017d13b1ca64d2ddfee0bc2374b366048152997669cccedc44d51",
    ),
}
QID = "pubmedqa_answer"
LABELS = ("yes", "no", "maybe")
N_LABELED_VALIDATION = 150  # correction 7 (27 September 2026); runs 1 and 2 used 50
N_ARTIFICIAL_VALIDATION = 2000
N_ARTIFICIAL_TRAIN = 20000
LABELED_TRAIN_REPEATS = 3  # correction 4 (27 September 2026); run 1 used 10


def state_of(row):
    """The question and the abstract's sections; nothing that states or predicts the answer."""
    context = row["context"]
    texts, sections = list(context["contexts"]), list(context["labels"])
    if len(texts) != len(sections):
        raise ValueError(f"PubMed ID {row['pubid']}: {len(texts)} contexts but {len(sections)} section labels")
    return {"question": row["question"], "contexts": [{"section": s, "text": t} for s, t in zip(sections, texts)]}


def record(row, source, calibrate=True):
    decision = row["final_decision"]
    label = LABELS.index(decision)
    out = {
        "id": f"pubmedqa-{row['pubid']}", "qid": QID, "state": state_of(row), "label": label,
        "target": R.one_hot(label, len(LABELS)), "group": {"source": source},
    }
    if not calibrate:
        out["calibrate"] = False
    return out


def stratified_sample(indices_by_label, n, yes_share, rng):
    """n indices with round(n * yes_share) yes and the rest no, drawn without replacement."""
    n_yes = round(n * yes_share)
    return rng.sample(indices_by_label["yes"], n_yes), rng.sample(indices_by_label["no"], n - n_yes)


def build(labeled_rows, artificial_table, test_ids, seed=R.SEED):
    """Pure split logic on already-loaded sources; returns ({split: records}, counts)."""
    test_ids = {str(i) for i in test_ids}
    labeled_ids = {str(r["pubid"]) for r in labeled_rows}
    missing = sorted(test_ids - labeled_ids)
    if missing:
        raise ValueError(f"{len(missing)} official test IDs are not in pqa_labeled, for example {missing[:3]}")
    labeled_rows = sorted(labeled_rows, key=lambda r: int(r["pubid"]))
    test_rows = [r for r in labeled_rows if str(r["pubid"]) in test_ids]
    rest = [r for r in labeled_rows if str(r["pubid"]) not in test_ids]
    random.Random(seed).shuffle(rest)
    val_labeled, train_labeled = rest[:N_LABELED_VALIDATION], rest[N_LABELED_VALIDATION:]

    non_test_yes = sum(1 for r in rest if r["final_decision"] == "yes")
    non_test_no = sum(1 for r in rest if r["final_decision"] == "no")
    yes_share = non_test_yes / (non_test_yes + non_test_no)

    art_ids = [str(i) for i in artificial_table["pubid"]]
    art_labels = list(artificial_table["final_decision"])
    in_test = sum(1 for i in art_ids if i in test_ids)
    in_other_labeled = sum(1 for i in art_ids if i in labeled_ids and i not in test_ids)
    eligible = [k for k, i in enumerate(art_ids) if i not in labeled_ids]
    if len(set(art_ids[k] for k in eligible)) != len(eligible):
        raise ValueError("pqa_artificial has repeated PubMed IDs")
    by_label = {"yes": [k for k in eligible if art_labels[k] == "yes"], "no": [k for k in eligible if art_labels[k] == "no"]}
    rng = random.Random(seed + 1)
    val_yes, val_no = stratified_sample(by_label, N_ARTIFICIAL_VALIDATION, yes_share, rng)
    taken = set(val_yes) | set(val_no)
    remaining = {lab: [k for k in ks if k not in taken] for lab, ks in by_label.items()}
    train_yes, train_no = stratified_sample(remaining, N_ARTIFICIAL_TRAIN, yes_share, rng)
    val_idx, train_idx = sorted(val_yes + val_no), sorted(train_yes + train_no)
    val_art, train_art = artificial_table["rows"](val_idx), artificial_table["rows"](train_idx)

    labeled_train_records = [record(r, "pqa_labeled") for r in train_labeled]
    splits = {
        "train": [rec for rec in labeled_train_records for _ in range(LABELED_TRAIN_REPEATS)]
        + [record(r, "pqa_artificial") for r in train_art],
        "validation": [record(r, "pqa_labeled") for r in val_labeled]
        + [record(r, "pqa_artificial", calibrate=False) for r in val_art],
        "test": [record(r, "pqa_labeled") for r in test_rows],
    }
    seen = {}
    for split, rows in splits.items():
        for r in rows:
            if seen.get(r["id"], split) != split:
                raise ValueError(f"{r['id']} appears in both {seen[r['id']]} and {split}")
            seen[r["id"]] = split
    counts = {
        "rows_read": {"pqa_labeled": len(labeled_rows), "pqa_artificial": len(art_ids)},
        "excluded": {
            "pqa_artificial_id_in_official_test": in_test,
            "pqa_artificial_id_in_other_pqa_labeled": in_other_labeled,
        },
        "pqa_labeled_non_test_yes_share": round(yes_share, 5),
        "pqa_labeled_train_repeats": LABELED_TRAIN_REPEATS,
        "sources_per_split": {
            "train": {"pqa_labeled": len(train_labeled), "pqa_artificial": len(train_art)},
            "validation": {"pqa_labeled": len(val_labeled), "pqa_artificial": len(val_art)},
            "test": {"pqa_labeled": len(test_rows), "pqa_artificial": 0},
        },
        "sources": {name: {"url": url, "sha256": sha} for name, (url, sha) in SOURCES.items()},
    }
    return splits, counts


def load_sources(raw_dir):
    import pyarrow.parquet as pq

    paths = {name: R.fetch(url, Path(raw_dir) / name, sha) for name, (url, sha) in SOURCES.items()}
    labeled = pq.read_table(paths["pqa_labeled.parquet"]).to_pylist()
    art = pq.read_table(paths["pqa_artificial.parquet"], columns=["pubid", "question", "context", "final_decision"])
    table = {
        "pubid": art.column("pubid").to_pylist(),
        "final_decision": art.column("final_decision").to_pylist(),
        "rows": lambda indices: art.take(indices).to_pylist(),
    }
    test_ids = json.loads(Path(paths["test_ground_truth.json"]).read_text(encoding="utf-8")).keys()
    return labeled, table, test_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=R.RAW / "pubmedqa")
    parser.add_argument("--out", type=Path, default=R.OUT)
    parser.add_argument("--results", type=Path, default=R.RESULTS)
    args = parser.parse_args()
    labeled, table, test_ids = load_sources(args.raw)
    splits, counts = build(labeled, table, test_ids)
    checksums = R.finish(QID, splits, counts, args.out, args.results)
    for split in R.SPLITS:
        print(f"{QID} {split}: {len(splits[split]):,} records")
    print(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()

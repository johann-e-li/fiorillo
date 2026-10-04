"""Session 7's PubMedQA records (docs/prompt_session7.md, THE RECORDS and step 2).

Reads PubMedQA's three parts as session 5 downloaded them (data/raw/public_s5/, qiaojin/PubMedQA at revision
9001f2853fb87cab8d220904e0de81ac6973b318; every file checked against the SHA-256 in src/corpus/pubmedqa_exclusion.py)
and session 4's PubMedQA training and validation records (data/processed/public/, checked against
results/public_v0/checksums.json; the test split is never read), and writes data/processed/session7/:
- pubmedqa_phase1_train.jsonl: every PQA-A item that passes the exclusions, as session 4's record (state: the question
  and the contexts; label yes or no; group source pqa_artificial), in PubMed ID order;
- pqa_u_unlabeled.jsonl: every PQA-U item that passes, with session 4's state and no label;
- labeler_pqa_a.jsonl, labeler_pqa_u.jsonl, labeler_pqa_l_train.jsonl and labeler_validation.jsonl: the labeler's
  records (question pubmedqa_conclusion; state {"question", "conclusion"} with the long answer as the conclusion, never
  the contexts; record IDs pubmedqa-<PubMed ID>), for every phase1 item, every pqa_u item (label -1 and an all-zero
  target), session 4's 350 PQA-L training items (once each) and its 150 PQA-L validation items;
- pqa_l_reference.jsonl: the 1,000 PQA-L items' PubMed ID, contexts and long answer, the reference of the 13-word check.
Exclusions, for PQA-A and PQA-U, in this order, each counted: the PubMed ID is one of the 1,000 PQA-L items
(pubid_in_pqa_labeled); the record ID is one of the 2,000 PQA-A items in session 4's validation file
(session4_validation_item); session 5's 13-word check (src/corpus/decontaminate.py, unchanged) against the PQA-L
reference matches the item's question, contexts and long answer joined (13_word_match).

The parquet files are read in batches of 20,000 rows. Each batch's output goes to data/processed/session7/parts/, with
a small JSON of its counts written last, so a run that the linked shell ends after about three minutes resumes where
it stopped; --max-seconds stops it from starting a new batch after that many seconds (each run builds at least one).
When every batch is done, the final files are joined from the parts (sorted by PubMed ID) and
results/fiorillo_v0_3/records_manifest.json is written with the counts, the drops by reason, and the SHA-256 of every
record file and of the three parquet files.

Run on the Mac from the linked shell's python3 (step 2), again until it prints "assembled":
    python3 src/tasks/pubmedqa_phases.py [--processes 4] [--max-seconds 140]
"""
import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "corpus"))
import records as R  # noqa: E402
from pubmedqa import LABELS, record, state_of  # noqa: E402

RAW = ROOT / "data" / "raw" / "public_s5"
PUBLIC = ROOT / "data" / "processed" / "public"
OUT = ROOT / "data" / "processed" / "session7"
MANIFEST = ROOT / "results" / "fiorillo_v0_3" / "records_manifest.json"
CHECKSUMS = ROOT / "results" / "public_v0" / "checksums.json"
BATCH_ROWS = 20000
N_PQA_L = 1000  # the expected counts; tests with made-up data change them
N_VALIDATION_PQA_A = 2000
N_L_TRAIN, N_L_VALIDATION = 350, 150
LABELER_QID = "pubmedqa_conclusion"
SOURCES = {"pqa_artificial": "phase1", "pqa_unlabeled": "pqa_u"}
DROP_REASONS = ("pubid_in_pqa_labeled", "session4_validation_item", "13_word_match")
FINAL = {
    "phase1": "pubmedqa_phase1_train.jsonl",
    "pqa_u": "pqa_u_unlabeled.jsonl",
    "labeler_pqa_a": "labeler_pqa_a.jsonl",
    "labeler_pqa_u": "labeler_pqa_u.jsonl",
    "labeler_pqa_l_train": "labeler_pqa_l_train.jsonl",
    "labeler_validation": "labeler_validation.jsonl",
}
_ID = re.compile(r'^\{"id": "pubmedqa-(\d+)"')


def parquet_files(raw=RAW):
    """The three parquet files with the SHA-256 values session 5 fixed for them (src/corpus/pubmedqa_exclusion.py)."""
    from pubmedqa_exclusion import FILES
    return {config: (Path(raw) / f"{config}.parquet", sha) for config, (_, sha) in FILES.items()}


def check_files(files):
    out = {}
    for config, (path, sha) in files.items():
        got = R.sha256_file(path)
        if got != sha:
            raise SystemExit(f"{path}: SHA-256 {got}, expected {sha}")
        out[path.name] = got
    return out


def session4_records(public=PUBLIC, checksums=CHECKSUMS):
    """Session 4's PubMedQA training and validation records, checked against their SHA-256 values (never the test)."""
    expected = json.loads(Path(checksums).read_text(encoding="utf-8"))
    out = {}
    for split in ("train", "validation"):
        name = f"pubmedqa_answer_{split}.jsonl"
        path = Path(public) / name
        got = R.sha256_file(path)
        if got != expected[name]:
            raise SystemExit(f"{path}: SHA-256 {got}, expected {expected[name]} (results/public_v0/checksums.json)")
        out[split] = R.read_jsonl(path)
    return out


def labeler_record(pubid, question, long_answer, label, source, calibrate=True):
    """A reasoning-free record: the question and the abstract's conclusion (its long answer), never its contexts."""
    target = [0.0] * len(LABELS) if label < 0 else R.one_hot(label, len(LABELS))
    out = {"id": f"pubmedqa-{pubid}", "qid": LABELER_QID, "state": {"question": question, "conclusion": long_answer or ""},
           "label": label, "target": target, "group": {"source": source}}
    if not calibrate:
        out["calibrate"] = False
    return out


def check_text(row):
    return " ".join([row["question"] or ""] + list(row["context"]["contexts"]) + [row["long_answer"] or ""])


def part_paths(parts, source, k):
    stem = parts / f"{source}_part-{k:03d}"
    return stem.with_suffix(".jsonl"), parts / f"{source}_labeler_part-{k:03d}.jsonl", stem.with_suffix(".json")


def build_batch(rows, source, labeled_ids, validation_ids, checker):
    """Records, labeler records and counts for one batch of rows of one source."""
    counts = {"rows": len(rows), "kept": 0, **{r: 0 for r in DROP_REASONS}}
    keep = []
    for row in rows:
        pubid = int(row["pubid"])
        if pubid in labeled_ids:
            counts["pubid_in_pqa_labeled"] += 1
        elif f"pubmedqa-{pubid}" in validation_ids:
            counts["session4_validation_item"] += 1
        else:
            keep.append(row)
    flags = checker([check_text(r) for r in keep]) if keep else []
    records, labeler = [], []
    for row, hit in zip(keep, flags):
        if hit:
            counts["13_word_match"] += 1
            continue
        pubid = int(row["pubid"])
        if source == "pqa_artificial":
            records.append(record(row, "pqa_artificial"))
            labeler.append(labeler_record(pubid, row["question"], row["long_answer"], LABELS.index(row["final_decision"]), source))
        else:
            records.append({"id": f"pubmedqa-{pubid}", "qid": "pubmedqa_answer", "state": state_of(row), "group": {"source": source}})
            labeler.append(labeler_record(pubid, row["question"], row["long_answer"], -1, source))
    counts["kept"] = len(records)
    return records, labeler, counts


def write_jsonl(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run_batches(source, path, parts, labeled_ids, validation_ids, checker, deadline, batch_rows=BATCH_ROWS):
    """Builds every batch of one source not yet built; returns (batches done, batches in all)."""
    import pyarrow.parquet as pq

    f = pq.ParquetFile(path)
    total = -(-f.metadata.num_rows // batch_rows)
    done = sum(1 for k in range(total) if part_paths(parts, source, k)[2].exists())
    if done == total:
        return done, total
    built_now = 0
    columns = ["pubid", "question", "context", "long_answer"] + (["final_decision"] if source == "pqa_artificial" else [])
    for k, batch in enumerate(f.iter_batches(batch_size=batch_rows, columns=columns)):
        rec_path, lab_path, count_path = part_paths(parts, source, k)
        if count_path.exists():
            continue
        if built_now and time.time() > deadline:
            break  # every run builds at least one batch, so each run makes progress
        records, labeler, counts = build_batch(batch.to_pylist(), source, labeled_ids, validation_ids, checker)
        write_jsonl(rec_path, records)
        write_jsonl(lab_path, labeler)
        count_path.write_text(json.dumps(counts) + "\n", encoding="utf-8")
        done += 1
        built_now += 1
        print(f"{source} batch {k + 1} of {total}: {counts}", flush=True)
    return done, total


def joined_lines(paths):
    """Every line of the part files, as (PubMed ID, line), sorted by PubMed ID; stops on a repeated ID."""
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                m = _ID.match(line)
                if not m:
                    raise SystemExit(f"{p}: a line without a leading pubmedqa id")
                rows.append((int(m.group(1)), line))
    rows.sort(key=lambda x: x[0])
    for (a, _), (b, _) in zip(rows, rows[1:]):
        if a == b:
            raise SystemExit(f"PubMed ID {a} appears twice")
    return rows


def write_lines(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for _, line in rows:
            f.write(line)
    tmp.replace(path)


def assemble(out, parts, totals, s4, labeled_rows, parquet_sha, reference_sha, manifest_path, public=PUBLIC):
    counts = {}
    for source, stem in SOURCES.items():
        n = totals[source]
        batch_counts = [json.loads(part_paths(parts, source, k)[2].read_text(encoding="utf-8")) for k in range(n)]
        counts[source] = {key: sum(c[key] for c in batch_counts) for key in batch_counts[0]}
        recs = joined_lines([part_paths(parts, source, k)[0] for k in range(n)])
        labs = joined_lines([part_paths(parts, source, k)[1] for k in range(n)])
        if [a for a, _ in recs] != [a for a, _ in labs]:
            raise SystemExit(f"{source}: records and labeler records differ")
        write_lines(out / FINAL[stem], recs)
        write_lines(out / FINAL["labeler_pqa_a" if source == "pqa_artificial" else "labeler_pqa_u"], labs)

    by_pubid = {int(r["pubid"]): r for r in labeled_rows}
    for split, name, calibrate in (("train", "labeler_pqa_l_train", True), ("validation", "labeler_validation", True)):
        seen, rows = set(), []
        for r in s4[split]:
            if (r.get("group") or {}).get("source") != "pqa_labeled" or r["id"] in seen:
                continue
            if split == "validation" and not r.get("calibrate", True):
                continue
            seen.add(r["id"])
            pubid = int(r["id"].split("-", 1)[1])
            src = by_pubid[pubid]
            label = LABELS.index(src["final_decision"])
            if label != int(r["label"]):
                raise SystemExit(f"{r['id']}: label {label} in pqa_labeled, {r['label']} in session 4's {split} records")
            rows.append((pubid, labeler_record(pubid, src["question"], src["long_answer"], label, "pqa_labeled", calibrate)))
        rows.sort(key=lambda x: x[0])
        write_jsonl(out / FINAL[name], [r for _, r in rows])
        counts[name] = {"records": len(rows)}
    if counts["labeler_pqa_l_train"]["records"] != N_L_TRAIN or counts["labeler_validation"]["records"] != N_L_VALIDATION:
        raise SystemExit(f"expected 350 PQA-L training and 150 validation items, got {counts}")

    manifest = {
        "source": "src/tasks/pubmedqa_phases.py",
        "plan": "docs/prompt_session7.md, THE RECORDS",
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "parquet_sha256": parquet_sha,
        "session4_records_sha256": {f"pubmedqa_answer_{s}.jsonl": R.sha256_file(Path(public) / f"pubmedqa_answer_{s}.jsonl") for s in ("train", "validation")},
        "reference": {"file": "pqa_l_reference.jsonl", "items": len(labeled_rows), "sha256": reference_sha},
        "drop_reasons": list(DROP_REASONS),
        "counts": counts,
        "files": {},
    }
    for key, name in FINAL.items():
        path = out / name
        with open(path, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        manifest["files"][name] = {"records": n, "bytes": path.stat().st_size, "sha256": R.sha256_file(path)}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--public", type=Path, default=PUBLIC)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--checksums", type=Path, default=CHECKSUMS)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=140.0)
    parser.add_argument("--batch-rows", type=int, default=BATCH_ROWS)
    args = parser.parse_args(argv)
    t0 = time.time()
    deadline = t0 + args.max_seconds
    import pyarrow.parquet as pq
    from decontaminate import ParallelChecker
    from pubmedqa_exclusion import labeled_items

    files = parquet_files(args.raw)
    parquet_sha = check_files(files)
    s4 = session4_records(args.public, args.checksums)
    validation_ids = {r["id"] for r in s4["validation"] if (r.get("group") or {}).get("source") == "pqa_artificial"}
    if len(validation_ids) != N_VALIDATION_PQA_A:
        raise SystemExit(f"expected {N_VALIDATION_PQA_A:,} PQA-A validation items in session 4's records, found {len(validation_ids)}")
    labeled_rows = pq.read_table(files["pqa_labeled"][0]).to_pylist()
    labeled_ids = {int(r["pubid"]) for r in labeled_rows}
    if len(labeled_ids) != N_PQA_L:
        raise SystemExit(f"expected {N_PQA_L:,} PQA-L items, found {len(labeled_ids)}")

    args.out.mkdir(parents=True, exist_ok=True)
    parts = args.out / "parts"
    parts.mkdir(exist_ok=True)
    reference = args.out / "pqa_l_reference.jsonl"
    if not reference.exists():
        write_jsonl(reference, labeled_items(labeled_rows))
    reference_sha = R.sha256_file(reference)

    totals = {}
    with ParallelChecker(reference, args.processes) as checker:
        for source in SOURCES:
            done, total = run_batches(source, files[source][0], parts, labeled_ids, validation_ids, checker.check, deadline,
                                      args.batch_rows)
            totals[source] = total
            if done < total:
                print(json.dumps({"status": "stopped; run again to resume", "source": source, "batches_done": done,
                                  "batches": total, "seconds": round(time.time() - t0, 1)}))
                return 0
    manifest = assemble(args.out, parts, totals, s4, labeled_rows, parquet_sha, reference_sha, args.manifest, args.public)
    print(json.dumps({"status": "assembled", "seconds": round(time.time() - t0, 1), "counts": manifest["counts"],
                      "files": {k: v["records"] for k, v in manifest["files"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

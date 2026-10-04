"""Subset the Synthea sample CSV export to the first N patients (default 100), sorted by patient Id.

Reads every CSV in the Synthea export and writes, for each, only the rows that belong to the kept
patients:
  - tables keyed by patient (a PATIENT or PATIENTID column; Id in patients.csv) are filtered on that key;
  - reference tables with no patient column (payers, providers, organizations) keep only the rows that
    the kept patients' records point to.
Output files are gzip-compressed CSV (.csv.gz, written with a fixed timestamp so reruns are
byte-identical). The full export's claims_transactions.csv is about 38 MB for 108 patients, over the
25 MB fixture limit when uncompressed; compression keeps every row of the kept patients.

Usage (from the project root):
  python src/tools/subset_synthea.py
  python src/tools/subset_synthea.py --source data/raw/synthea-download --dest src/tests/fixtures/synthea-sample-100 --n 100
"""
import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "data" / "raw" / "synthea-download"
DEFAULT_DEST = ROOT / "src" / "tests" / "fixtures" / "synthea-sample-100"
PATIENT_KEYS = ("PATIENT", "PATIENTID")
REFERENCE_TABLES = ("payers.csv", "providers.csv", "organizations.csv")  # processed in this order
REFERENCE_COLUMNS = (
    "ORGANIZATION", "PROVIDER", "PAYER", "SECONDARY_PAYER", "PROVIDERID", "SUPERVISINGPROVIDERID",
    "REFERRINGPROVIDERID", "PRIMARYPATIENTINSURANCEID", "SECONDARYPATIENTINSURANCEID",
)
LIMIT_BYTES = 25 * 1024 * 1024

csv.field_size_limit(sys.maxsize)


def find_export(source):
    hits = sorted(source.rglob("patients.csv"))
    if not hits:
        sys.exit(f"No patients.csv found under {source}")
    return hits[0].parent


def open_gzip_writer(path):
    raw = open(path, "wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return raw, gz, io.TextIOWrapper(gz, encoding="utf-8", newline="")


def filter_table(src_path, dest_path, key, keep, references=None):
    """Copy the header and the rows whose `key` value is in `keep`. Returns (kept, total)."""
    kept = total = 0
    raw, gz, text = open_gzip_writer(dest_path)
    try:
        with open(src_path, encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            writer = csv.writer(text)
            writer.writerow(header)
            index = header.index(key)
            ref_indexes = [i for i, name in enumerate(header) if name in REFERENCE_COLUMNS]
            for row in reader:
                total += 1
                if row[index] in keep:
                    writer.writerow(row)
                    kept += 1
                    if references is not None:
                        references.update(row[i] for i in ref_indexes if row[i])
    finally:
        text.close()
        gz.close()
        raw.close()
    return kept, total


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--n", type=int, default=100)
    args = parser.parse_args()

    export = find_export(args.source)
    args.dest.mkdir(parents=True, exist_ok=True)
    with open(export / "patients.csv", encoding="utf-8", newline="") as handle:
        all_ids = sorted(row["Id"] for row in csv.DictReader(handle))
    keep = set(all_ids[: args.n])
    print(f"Export: {export}")
    print(f"Patients in export: {len(all_ids):,}; kept: {len(keep):,} (first {args.n} by sorted Id)")

    references = set()
    report = []
    reference_files = []
    for src_path in sorted(export.glob("*.csv")):
        name = src_path.name
        if name in REFERENCE_TABLES:
            reference_files.append(src_path)
            continue
        with open(src_path, encoding="utf-8", newline="") as handle:
            header = next(csv.reader(handle))
        key = "Id" if name == "patients.csv" else next((k for k in PATIENT_KEYS if k in header), None)
        if key is None:
            report.append((name, None, None, None, "skipped: no patient column and not a reference table"))
            continue
        dest_path = args.dest / (name + ".gz")
        kept, total = filter_table(src_path, dest_path, key, keep, references)
        report.append((name, kept, total, dest_path.stat().st_size, f"filtered on {key}"))

    for table in REFERENCE_TABLES:
        src_path = export / table
        if src_path not in reference_files:
            continue
        dest_path = args.dest / (table + ".gz")
        new_refs = set() if table == "providers.csv" else None
        kept, total = filter_table(src_path, dest_path, "Id", references, new_refs)
        if new_refs:
            references.update(new_refs)
        report.append((table, kept, total, dest_path.stat().st_size, "reference table: rows the kept patients point to"))

    over = False
    print(f"{'file':32} {'rows kept':>10} {'rows in export':>15} {'bytes':>12}  note")
    for name, kept, total, size, note in report:
        if kept is None:
            print(f"{name:32} {'':>10} {'':>15} {'':>12}  {note}")
            continue
        flag = ""
        if size > LIMIT_BYTES:
            flag = "  OVER 25 MB"
            over = True
        print(f"{name + '.gz':32} {kept:>10,} {total:>15,} {size:>12,}  {note}{flag}")
    if over:
        sys.exit("At least one file is over the 25 MB fixture limit.")
    print("Every file is under 25 MB.")


if __name__ == "__main__":
    main()

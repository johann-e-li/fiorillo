"""TCGA's pathology reports with the GDC's labels, prepared as records for the next question (I-38).

Plan: docs/prompt_session9.md, THE NEXT QUESTION'S RECORDS (with DECISIONS 12; run in STEP 18). Preparation only:
nothing is trained, and no questions YAML is written.

Sources (fixed in step 0; docs/fiorillo_v0_5.md, Licenses, read on 1 October 2026):
- TCGA-Reports (Kefeli and Tatonetti, Patterns 2024; PMID 38487800): Mendeley Data, dataset hyg5xkznpx, version 1,
  doi 10.17632/hyg5xkznpx.1, CC BY 4.0; the file TCGA_Reports.csv.zip (10,913,820 bytes; its SHA-256 is ZIP_SHA256
  below). The paper reports 9,523 reports. A copy of the same zip is in github.com/tatonetti-lab/tnm-stage-classifier
  (MIT); the plan names Mendeley Data as the source, and if the linked shell cannot reach it, the file is downloaded in
  a browser and put in raw/, and download checks it.
- The GDC's cases endpoint (https://api.gdc.cancer.gov/cases): open-access fields only, no token.

Files (under data/processed/session9/tcga/, ignored by git, unless named otherwise):
  licenses.json                        licenses: Mendeley Data's record as read, the GDC's terms, the verdict
  raw/TCGA_Reports.csv.zip             download (parts/TCGA_Reports.csv.zip.part while it downloads)
  raw/gdc/cases_from_<from>.json       labels: each page of the cases endpoint, byte for byte as fetched
  raw/gdc/query.json                   labels: the query, and each page's URL, date, size and SHA-256
  <qid>_{train,validation,test}.jsonl  records, for qid tcga_t, tcga_n, tcga_stage and tcga_grade
  parts/                               saved progress: download_progress.json, labels_progress.json,
                                       records_build.json (the build's counts, which counts reads), report_tokens.jsonl
  results/tcga_prep/records_manifest.json  counts (committed)

Subcommands (every network step saves its progress under parts/ and resumes, since each shell command ends after
about 3 minutes; every subcommand except licenses refuses to run until licenses.json allows the build):
  licenses  reads Mendeley Data's public record of the dataset (its data_licence fields) and writes licenses.json with
            the time; the GDC's terms are constants summarized from docs/fiorillo_v0_5.md. The build is allowed only
            when the dataset's license is CC BY (any version), CC0 or the public domain, a deliberately narrow list: any
            other license, or none stated, prints "NOT BUILT: I-38's records, <reason>" and stops (exit code 1), and
            the other subcommands then refuse to run.
  download  the report zip from Mendeley Data into raw/, resuming a partial download with an HTTP range request; its
            SHA-256 must equal ZIP_SHA256, or the command fails and keeps what it got. A zip already in raw/ (put there
            by hand) is only checked.
  labels    every case of the TCGA program from the cases endpoint (the fields in GDC_FIELDS), in pages of
            --page-size cases through the API's size and from parameters, sorted by case_id; each page saved as
            fetched, and the query and the dates in raw/gdc/query.json once every page is in. It checks that every
            page reports the same total and that the pages hold that many distinct cases.
  records   the records (the rules below) and parts/records_build.json.
  counts    results/tcga_prep/records_manifest.json: counts per question, split, answer and project; drops by reason
            and the values left out by their raw text; each report's length in Qwen3-4B-Base's tokens (resumable); the
            label's own words on the training and validation splits; the SHA-256 of every file written and of the raw
            inputs.

The rules (records):
- Reports. The one CSV in the zip (TCGA_Reports.csv) names each report by a file name whose first 12 characters
  (case ignored) are the patient's TCGA barcode (TCGA-XX-XXXX). The columns expected are patient_filename and text;
  otherwise the one column whose name contains "filename" and the one named "text" (case ignored) are used, or the
  build stops and prints the header (records_build.json and the manifest record the header and the columns used). A
  row whose file name does not begin with a barcode is counted and left out. One report per patient: the first in
  the file's order; the others are counted.
- The join. A patient's report joins the GDC case whose submitter_id is the barcode. The report rows and the patients
  that join no case are counted, as are barcodes that name several cases and kept reports whose text is empty; all
  of these are left out.
- The primary diagnosis. For each case and field, the primary diagnoses are those whose classification_of_tumor is
  exactly "primary" (the manifest counts every value seen on the joined cases). Those that state no value for the
  field (the key absent, null or blank) are passed over. The case is left out of that question, and counted, when it
  has no primary diagnosis, when none states a value, when the stated values disagree after folding (a value outside
  the answer set counts as its raw text, so T2a and T2b agree, and T2 and TX disagree), or when the common value is
  outside the answer set (counted by its raw text).
- The questions. tcga_t from ajcc_pathologic_t, answers T1, T2, T3, T4; tcga_n from ajcc_pathologic_n, N0, N1, N2,
  N3; tcga_stage from ajcc_pathologic_stage, I, II, III, IV; tcga_grade from tumor_grade, G1, G2, G3, G4. The label is
  the answer's index in that order. A value is looked up in its question's table of accepted spellings (T_SPELLINGS,
  N_SPELLINGS, STAGE_SPELLINGS, GRADE_SPELLINGS) with its whitespace removed and its case ignored, which folds each
  subcategory into its number (T2a to T2, N1mi and N1a to N1, N0 (i+) to N0, Stage IIIA to III, Stage IS to I); grade
  accepts G1 to G4 only, since a grade spelled out (Grade II) may belong to another system. Anything not in a table is
  left out, for example Tis, T0 and TX; NX; Stage 0 and Stage X; GX, GB, High Grade, Low Grade, Grade II, Not Reported
  and Unknown. primary_diagnosis and laterality are fetched and counted only, by the same primary-diagnosis rule
  without folding.
- The splits. The joined patients are split once, by patient, and the split is shared by the four questions; it reads
  no label. Projects are taken in sorted order with one generator, random.Random(20260927); within each project the
  patients, sorted as text, are shuffled, and the shuffled list of n patients is cut at (80 n + 50) // 100 and
  (90 n + 50) // 100 (80 and 90 percent of n, halves rounded up): the first part is train, the second validation, the
  rest test. So 45 patients give 36, 5 and 4; 10 give 8, 1 and 1; 3 give 2, 1 and 0.
- The records, in session 4's format (src/tasks/records.py), sorted by ID within each file: id tcga-<qid>-<barcode>;
  qid; state {"report": the report's text, unchanged}; label; a one-hot target; "calibrate": true on validation
  records; group {"source": "tcga_reports", "patient": <barcode>, "project": <the GDC's project_id>}. A patient enters
  a question's records only with a label in its answer set.

The label's own words (LABEL_WORDS), counted on the training and validation splits only. A record counts when its
report matches one of its answer's patterns (regular expressions searched anywhere in the text):
- tcga_t, answer Tk: T or t followed by k and not by another digit, after at most two of the prefix letters p, c, y
  and r (either case), with no letter or digit just before: pT2, ypT3, T2a and PT4 count; T12 counts for none.
- tcga_n, answer Nk: the same with N or n (pN0, N1mi, ypN2), which may also follow a T category directly, as in
  pT2N0 or pT2aN1b.
- tcga_stage, answer I to IV: the word "stage" (any case), then optionally "group" or "grouping", a colon or a hyphen,
  then the Roman numeral in capitals or the Arabic digit, not followed by another numeral letter or digit: Stage IIA,
  stage: III and Stage 2B count; "the stage is" and Stage IV do not count for stage I.
- tcga_grade, answer Gk: a capital G followed by k and not by another digit, with no letter or digit just before (G3);
  or the word "grade" (any case), an optional colon or hyphen, then k or its Roman numeral, not followed by another
  numeral letter or digit (grade 3, Grade III, grade: 2). Words for differentiation (well, moderately or poorly
  differentiated) are not counted, and "Gleason grade 3" counts for G3, so the count is a rough one.
The manifest also counts the records whose report matches another answer's patterns, since a report that lists
several categories (a staging legend or a form) matches more than its own label.

The test split: its labels are written and counted (per answer and project), never read for anything else; the
label's own words are not counted on it. Its reports' token lengths are counted, which reads no label.

Usage (the linked shell, from the project folder, with python3; step 18):
    python3 src/tasks/tcga_reports.py licenses
    python3 src/tasks/tcga_reports.py download [--max-seconds 150]
    python3 src/tasks/tcga_reports.py labels [--max-seconds 150] [--page-size 500]
    python3 src/tasks/tcga_reports.py records
    python3 src/tasks/tcga_reports.py counts [--max-seconds 150] [--tokenizer-dir models/qwen3_4b_base]
Each takes --out-dir (default data/processed/session9/tcga); counts also takes --results (default results/tcga_prep).
Run download, labels and counts again until they print "finished": true.
"""
import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
from records import one_hot, sha256_file, write_jsonl  # noqa: E402

PLAN = "docs/prompt_session9.md, THE NEXT QUESTION'S RECORDS"
OUT = ROOT / "data" / "processed" / "session9" / "tcga"
RESULTS = ROOT / "results" / "tcga_prep"
QWEN_DIR = ROOT / "models" / "qwen3_4b_base"
QWEN_REVISION = "906bfd4b4dc7f14ee4320094d8b41684abff8539"
SEED = 20260927
SPLITS = ("train", "validation", "test")
WORD_SPLITS = ("train", "validation")  # the label's own words are never counted on the test split
CUTS_PERCENT = (80, 90)  # train is the first 80 percent of a project's shuffled patients, validation the next 10
SOURCE = "tcga_reports"
USER_AGENT = "ed-decision-model/session9"

MENDELEY_DATASET = "hyg5xkznpx"
MENDELEY_VERSION = 1
MENDELEY_DOI = "10.17632/hyg5xkznpx.1"
MENDELEY_API = "https://data.mendeley.com/public-api/datasets/hyg5xkznpx"
ZIP_NAME = "TCGA_Reports.csv.zip"
ZIP_BYTES = 10913820
ZIP_SHA256 = "bbbc0ee0c75d8feb0bb246b4fa40e8e29bba9775d4a9e4f41e0b73a7955e1c12"
ZIP_URL = ("https://data.mendeley.com/public-files/datasets/hyg5xkznpx/files/60abe141-9352-4a54-943c-3d015eabefea/"
           "file_downloaded")
ZIP_COPY = "github.com/tatonetti-lab/tnm-stage-classifier, TCGA_Pathology_Reports/TCGA_Reports.csv.zip (MIT)"
PAPER_REPORTS = 9523
PAPER = {"authors": "Kefeli and Tatonetti", "journal": "Patterns", "year": 2024, "pmid": "38487800",
         "doi": "10.1016/j.patter.2024.100933", "reports": PAPER_REPORTS,
         "license": "CC BY, as the plan states (THE NEXT QUESTION'S RECORDS); not read by this command"}

GDC_CASES = "https://api.gdc.cancer.gov/cases"
GDC_FIELDS = ("submitter_id", "case_id", "project.project_id", "diagnoses.classification_of_tumor",
              "diagnoses.ajcc_pathologic_t", "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_stage",
              "diagnoses.tumor_grade", "diagnoses.primary_diagnosis", "diagnoses.laterality")
GDC_FILTERS = {"op": "in", "content": {"field": "project.program.name", "value": ["TCGA"]}}
GDC_SORT = "case_id:asc"
GDC_PAGE_SIZE = 500
GDC_TERMS = {
    "source": "docs/fiorillo_v0_5.md, Licenses (read 1 October 2026 in step 0); summarized here as constants, not read "
              "by this command",
    "open_access": "The GDC's page on data access says that open-access data include most clinical data elements and "
                   "that no login is needed.",
    "policy": "Its page on data access policies, under the NIH Genomic Data Sharing Policy, asks users to acknowledge "
              "in their publications the datasets and the NIH-designated repository through which they reached the "
              "data, and not to attempt to identify individual participants.",
    "this_session": "Session 9 fetches open-access fields only, attempts no identification, and its documents and the "
                    "preprint name TCGA and the GDC as the sources.",
    "allows_research_use_with_publication": True,
}

# The tables of accepted spellings: each value is looked up with its whitespace removed and its case ignored.
T_SPELLINGS = {
    "T1": ("T1", "T1a", "T1a1", "T1a2", "T1b", "T1b1", "T1b2", "T1b3", "T1c", "T1c1", "T1c2", "T1c3", "T1mi"),
    "T2": ("T2", "T2a", "T2a1", "T2a2", "T2b", "T2c", "T2d"),
    "T3": ("T3", "T3a", "T3b", "T3c", "T3d", "T3e"),
    "T4": ("T4", "T4a", "T4b", "T4c", "T4d", "T4e"),
}
N_SPELLINGS = {
    "N0": ("N0", "N0 (i-)", "N0 (i+)", "N0 (mol-)", "N0 (mol+)", "N0a", "N0b"),
    "N1": ("N1", "N1a", "N1b", "N1bI", "N1bII", "N1bIII", "N1bIV", "N1c", "N1mi"),
    "N2": ("N2", "N2a", "N2b", "N2c"),
    "N3": ("N3", "N3a", "N3b", "N3c"),
}
STAGE_SPELLINGS = {
    "I": ("Stage I", "Stage IA", "Stage IA1", "Stage IA2", "Stage IA3", "Stage IB", "Stage IB1", "Stage IB2",
          "Stage IB3", "Stage IC", "Stage IC1", "Stage IC2", "Stage IC3", "Stage IS"),
    "II": ("Stage II", "Stage IIA", "Stage IIA1", "Stage IIA2", "Stage IIB", "Stage IIC"),
    "III": ("Stage III", "Stage IIIA", "Stage IIIA1", "Stage IIIA2", "Stage IIIB", "Stage IIIC", "Stage IIIC1",
            "Stage IIIC2", "Stage IIID"),
    "IV": ("Stage IV", "Stage IVA", "Stage IVB", "Stage IVC"),
}
GRADE_SPELLINGS = {"G1": ("G1",), "G2": ("G2",), "G3": ("G3",), "G4": ("G4",)}  # spelled-out grades may be another system
QUESTIONS = {
    "tcga_t": {"field": "ajcc_pathologic_t", "answers": ("T1", "T2", "T3", "T4"), "spellings": T_SPELLINGS},
    "tcga_n": {"field": "ajcc_pathologic_n", "answers": ("N0", "N1", "N2", "N3"), "spellings": N_SPELLINGS},
    "tcga_stage": {"field": "ajcc_pathologic_stage", "answers": ("I", "II", "III", "IV"), "spellings": STAGE_SPELLINGS},
    "tcga_grade": {"field": "tumor_grade", "answers": ("G1", "G2", "G3", "G4"), "spellings": GRADE_SPELLINGS},
}
QIDS = tuple(QUESTIONS)
COUNTED_ONLY = ("primary_diagnosis", "laterality")

PRIMARY = "primary"
NO_PRIMARY = "no diagnosis classified primary"
NO_VALUE = "no value on the primary diagnosis"
DISAGREE = "several primary diagnoses disagree"
OUTSIDE = "value outside the answer set"
REASONS = (NO_PRIMARY, NO_VALUE, DISAGREE, OUTSIDE)
SPLIT_RULE = ("by patient, shared by the four questions, reading no label: projects in sorted order with one "
              "generator random.Random(20260927); within each project the patients, sorted as text, are shuffled and "
              "the shuffled list of n is cut at (80 n + 50) // 100 and (90 n + 50) // 100: train, validation, test")
PRIMARY_RULE = ("the diagnoses whose classification_of_tumor is exactly 'primary'; those with no value for the field "
                "are passed over; left out when none is primary, none states a value, the stated values disagree after "
                "folding, or the common value is outside the answer set")

# The label's own words (module docstring): {qid: {answer: [patterns]}}.
ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}
_ALONE = r"(?<![A-Za-z0-9])"
_PREFIX = r"[pcyrPCYR]{0,2}"
_AFTER_T = (r"(?:(?<![A-Za-z0-9])|(?<=[Tt][0-4])|(?<=[Tt][0-4][a-e])|(?<=[Tt][0-4][a-e][0-9])|(?<=[Tt][0-4]mi)"
            r"|(?<=[Tt]is)|(?<=[Tt][Xx]))")
_NUMERAL_END = r"(?![IVXivx0-9])"
LABEL_WORDS = {
    "tcga_t": {f"T{k}": [_ALONE + _PREFIX + f"[Tt]{k}(?![0-9])"] for k in (1, 2, 3, 4)},
    "tcga_n": {f"N{k}": [_AFTER_T + _PREFIX + f"[Nn]{k}(?![0-9])"] for k in (0, 1, 2, 3)},
    "tcga_stage": {ROMAN[k]: [r"(?i:\bstage)\s*(?:(?i:group(?:ing)?)\s*)?[:\-]?\s*" + f"(?:{ROMAN[k]}|{k})" + _NUMERAL_END]
                   for k in (1, 2, 3, 4)},
    "tcga_grade": {f"G{k}": [_ALONE + f"G{k}(?![0-9])", r"(?i:\bgrade)\s*[:\-]?\s*" + f"(?:{ROMAN[k]}|{k})" + _NUMERAL_END]
                   for k in (1, 2, 3, 4)},
}
WORD_RES = {qid: {a: [re.compile(p) for p in pats] for a, pats in table.items()} for qid, table in LABEL_WORDS.items()}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def iter_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def get_bytes(url, opener=None, tries=4, timeout=60):
    """The body of a GET, tried up to tries times."""
    opener = opener or urllib.request.urlopen
    last = None
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with opener(req, timeout=timeout) as response:
                return response.read()
        except Exception as e:  # noqa: BLE001
            last = e
            if k + 1 < tries:
                time.sleep(min(1 + 2 * k, 8))
    raise RuntimeError(f"could not fetch {url}: {last}")


# ---------------------------------------------------------------- licenses

OPEN_LICENSE_URL = re.compile(r"https?://(?:www\.)?creativecommons\.org/(?:licenses/by/\d+(?:\.\d+)?|"
                              r"publicdomain/(?:zero|mark)/\d+(?:\.\d+)?)/?", re.I)
OPEN_LICENSE_NAME = re.compile(r"CC BY(?: \d+(?:\.\d+)?)?(?: INTERNATIONAL)?|CC0(?: \d+(?:\.\d+)?)?|"
                               r"PUBLIC DOMAIN(?: MARK)?(?: \d+(?:\.\d+)?)?|PDM(?: \d+(?:\.\d+)?)?")
RESTRICTING = re.compile(r"non-?commercial|no-?deriv|share-?alike", re.I)


def license_verdict(licence):
    """(allowed, reason) from Mendeley Data's data_licence object: allowed only for CC BY (any version), CC0 or the
    public domain, with the link and the short name (where given) both naming one of them and no restricting words in
    the full name."""
    if not isinstance(licence, dict):
        return False, "Mendeley Data's record of the dataset states no license"
    url = str(licence.get("url") or "").strip()
    name = str(licence.get("short_name") or "").strip()
    full = str(licence.get("full_name") or "").strip()
    if not (url or name or full):
        return False, "Mendeley Data's record of the dataset states no license"
    shown = name or full or url
    name_ok = not name or bool(OPEN_LICENSE_NAME.fullmatch(" ".join(re.sub(r"[-_]", " ", name).upper().split())))
    url_ok = not url or bool(OPEN_LICENSE_URL.fullmatch(url))
    if not (url or name) or not name_ok or not url_ok or RESTRICTING.search(full):
        return False, (f"Mendeley Data's record states the license {shown} ({url or 'no link'}), and this builder accepts "
                       "only CC BY, CC0 or the public domain as allowing research use with publication without a reading "
                       "by a person")
    return True, f"Mendeley Data's record states {shown} ({url or 'no link'}), which allows research use with publication"


def licenses(out_dir=OUT, fetch=None):
    """Reads Mendeley Data's record of the dataset and writes licenses.json; returns its contents."""
    out_dir = Path(out_dir)
    fetch = fetch or get_bytes
    try:
        record = json.loads(fetch(MENDELEY_API))
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"could not read Mendeley Data's record of the dataset ({MENDELEY_API}): {e}")
    read_utc = now()
    allowed, reason = license_verdict(record.get("data_licence"))
    if allowed and not GDC_TERMS["allows_research_use_with_publication"]:
        allowed, reason = False, "the GDC's terms do not allow research use with publication"
    doi = record.get("doi")
    found = next((f for f in record.get("files") or [] if f.get("filename") == ZIP_NAME), None)
    details = (found or {}).get("content_details") or {}
    zip_record = {"filename": (found or {}).get("filename"), "size": details.get("size", (found or {}).get("size")),
                  "sha256_hash": details.get("sha256_hash"), "download_url": details.get("download_url")}
    out = {
        "plan": PLAN, "written_utc": read_utc, "allowed": allowed, "reason": reason,
        "tcga_reports": {
            "source": MENDELEY_API, "read_utc": read_utc,
            "expected": {"dataset": MENDELEY_DATASET, "version": MENDELEY_VERSION, "doi": MENDELEY_DOI},
            "dataset": record.get("id"), "doi": doi.get("id") if isinstance(doi, dict) else doi,
            "version": record.get("version"),
            "versions": record.get("versions"), "publish_date": record.get("publish_date"),
            "data_licence": record.get("data_licence"),
            "file": zip_record,
            "file_matches_constants": (zip_record["size"] == ZIP_BYTES and zip_record["sha256_hash"] == ZIP_SHA256
                                       and zip_record["download_url"] == ZIP_URL),
            "copy_if_unreachable": ZIP_COPY},
        "paper": PAPER,
        "gdc": GDC_TERMS,
        "mendeley_api_response": record,
    }
    if not allowed:
        out["not_built"] = f"NOT BUILT: I-38's records, {reason}"
    save_json(out_dir / "licenses.json", out)
    return out


def require_licenses(out_dir):
    """licenses.json's contents, or a stop when it is missing or does not allow the build."""
    path = Path(out_dir) / "licenses.json"
    lic = load_json(path, None)
    if lic is None:
        raise SystemExit(f"run licenses first: {rel(path)} does not exist")
    if not lic.get("allowed"):
        raise SystemExit(lic.get("not_built") or f"NOT BUILT: I-38's records, {rel(path)} does not allow them")
    return lic


# ---------------------------------------------------------------- download

def download(out_dir=OUT, max_seconds=150.0, url=ZIP_URL, sha256=ZIP_SHA256, size=ZIP_BYTES, opener=None, block=1 << 20):
    """The report zip into raw/, resumable; a status dict ("finished" true or false)."""
    out_dir = Path(out_dir)
    require_licenses(out_dir)
    dest = out_dir / "raw" / ZIP_NAME
    part = out_dir / "parts" / (ZIP_NAME + ".part")
    progress_path = out_dir / "parts" / "download_progress.json"
    progress = load_json(progress_path, None) or {"plan": PLAN, "url": url, "expected": {"bytes": size, "sha256": sha256},
                                                   "complete": False, "runs": []}
    t0 = time.time()
    if dest.exists():
        got = sha256_file(dest)
        if got != sha256:
            raise SystemExit(f"{rel(dest)} has SHA-256 {got}, not {sha256}; it is left in place (move it aside to "
                             "download again)")
        if not progress.get("complete"):
            progress.update({"complete": True, "source": "found in raw/ (put there by hand) and checked here",
                             "finished_utc": now()})
        progress.update({"written_utc": now(), "checked_utc": now(), "bytes": dest.stat().st_size, "sha256": got,
                         "matches_constant": True})
        save_json(progress_path, progress)
        status = {"finished": True, "file": rel(dest), "bytes": dest.stat().st_size, "sha256": got, "matches_constant": True}
        print(json.dumps(status))
        return status
    have = part.stat().st_size if part.exists() else 0
    if have < size:
        run = {"started_utc": now(), "from_byte": have}
        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            response = (opener or urllib.request.urlopen)(urllib.request.Request(url, headers=headers), timeout=30)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"could not download {url}: {e}. If the linked shell cannot reach Mendeley Data, download "
                             f"the file in a browser, put it in {rel(dest.parent)}, and run download again to check it")
        with response:
            code = getattr(response, "status", None) or response.getcode()
            if have and code != 206:
                have = 0  # the server sent the whole file, so the partial file is started again
            part.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(part, "ab" if have else "wb") as f:
                    while True:
                        data = response.read(block)
                        if not data:
                            break
                        f.write(data)
                        have += len(data)
                        if time.time() - t0 > max_seconds:
                            break
            except OSError as e:  # a dropped connection: the next run resumes from what was written
                run["error"] = repr(e)
                have = part.stat().st_size
        run.update({"http_status": code, "to_byte": have, "ended_utc": now()})
        progress["runs"].append(run)
        progress["written_utc"] = now()
        save_json(progress_path, progress)
    if have < size:
        status = {"finished": False, "bytes": have, "of": size}
        print(json.dumps(status))
        return status
    got = sha256_file(part)
    if have != size or got != sha256:
        raise SystemExit(f"the download has {have} bytes and SHA-256 {got}, not {size} and {sha256}; it is kept as "
                         f"{rel(part)} (move it aside to download again)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    part.replace(dest)
    progress.update({"complete": True, "source": f"downloaded from {url}", "finished_utc": now(), "written_utc": now(),
                     "bytes": have, "sha256": got, "matches_constant": True})
    save_json(progress_path, progress)
    status = {"finished": True, "file": rel(dest), "bytes": have, "sha256": got, "matches_constant": True}
    print(json.dumps(status))
    return status


# ---------------------------------------------------------------- labels (the GDC)

def gdc_query(page_size=GDC_PAGE_SIZE):
    return {"endpoint": GDC_CASES, "filters": GDC_FILTERS, "fields": list(GDC_FIELDS), "sort": GDC_SORT,
            "format": "JSON", "size": page_size}


def gdc_url(query, start):
    params = {"filters": json.dumps(query["filters"], separators=(",", ":")), "fields": ",".join(query["fields"]),
              "format": query["format"], "size": str(query["size"]), "from": str(start), "sort": query["sort"]}
    return query["endpoint"] + "?" + urllib.parse.urlencode(params)


def page_starts(total, page_size):
    return [0] if not total else list(range(0, total, page_size))


def labels(out_dir=OUT, max_seconds=150.0, page_size=GDC_PAGE_SIZE, opener=None):
    """Every TCGA case from the cases endpoint, page by page, resumable; a status dict ("finished" true or false)."""
    out_dir = Path(out_dir)
    require_licenses(out_dir)
    gdc_dir = out_dir / "raw" / "gdc"
    progress_path = out_dir / "parts" / "labels_progress.json"
    query = gdc_query(page_size)
    progress = load_json(progress_path, None) or {"plan": PLAN, "query": query, "total": None, "pages": {}}
    if progress.get("query") != query:
        raise SystemExit(f"{rel(progress_path)} was written for another query; move it aside to fetch again")
    t0, fetched = time.time(), 0
    while True:
        total = progress["total"]
        todo = [s for s in ([0] if total is None else page_starts(total, page_size)) if str(s) not in progress["pages"]]
        if not todo or (fetched and time.time() - t0 > max_seconds):
            break
        start = todo[0]
        url = gdc_url(query, start)
        try:
            body = get_bytes(url, opener)
        except RuntimeError as e:
            raise SystemExit(f"{e}; the pages fetched so far are kept, so run labels again")
        data = json.loads(body)
        if data.get("warnings"):
            raise SystemExit(f"the GDC answered {url} with warnings: {data['warnings']}")
        pagination, hits = data["data"]["pagination"], data["data"]["hits"]
        if total is not None and pagination["total"] != total:
            raise SystemExit(f"the GDC's total changed during the fetch, from {total} to {pagination['total']}; move "
                             f"{rel(gdc_dir)} and {rel(progress_path)} aside and fetch again")
        name = f"cases_from_{start:05d}.json"
        write_bytes(gdc_dir / name, body)
        progress["pages"][str(start)] = {"from": start, "file": name, "url": url, "fetched_utc": now(), "bytes": len(body),
                                         "sha256": hashlib.sha256(body).hexdigest(), "hits": len(hits),
                                         "total": pagination["total"]}
        progress["total"] = pagination["total"]
        progress["written_utc"] = now()
        save_json(progress_path, progress)
        fetched += 1
    total = progress["total"]
    starts = [0] if total is None else page_starts(total, page_size)
    if any(str(s) not in progress["pages"] for s in starts):
        status = {"finished": False, "pages_done": len(progress["pages"]), "pages": len(starts), "fetched_now": fetched}
        print(json.dumps(status))
        return status
    pages = [progress["pages"][str(s)] for s in starts]
    seen, hits_n = set(), 0
    for page in pages:
        path = gdc_dir / page["file"]
        if sha256_file(path) != page["sha256"]:
            raise SystemExit(f"{rel(path)} differs from {rel(progress_path)}")
        for hit in json.loads(path.read_bytes())["data"]["hits"]:
            seen.add(hit.get("case_id"))
            hits_n += 1
    if hits_n != total or len(seen) != total:
        raise SystemExit(f"the pages hold {hits_n} cases ({len(seen)} distinct), not the {total} the GDC reported; move "
                         f"{rel(gdc_dir)} and {rel(progress_path)} aside and fetch again")
    query_path = gdc_dir / "query.json"
    old = load_json(query_path, {})
    if old.get("pages") != pages:
        save_json(query_path, {"plan": PLAN, "written_utc": now(), "program": "TCGA", "query": query,
                               "fetched_first_utc": min(p["fetched_utc"] for p in pages),
                               "fetched_last_utc": max(p["fetched_utc"] for p in pages),
                               "total": total, "cases": len(seen), "pages": pages})
    status = {"finished": True, "cases": len(seen), "pages": len(pages), "fetched_now": fetched}
    print(json.dumps(status))
    return status


def read_cases(out_dir):
    """The GDC's cases from the saved pages (each checked against query.json), and query.json's contents."""
    gdc_dir = Path(out_dir) / "raw" / "gdc"
    query = load_json(gdc_dir / "query.json", None)
    if query is None:
        raise SystemExit(f"run labels first: {rel(gdc_dir / 'query.json')} does not exist")
    cases = []
    for page in query["pages"]:
        path = gdc_dir / page["file"]
        if sha256_file(path) != page["sha256"]:
            raise SystemExit(f"{rel(path)} differs from query.json")
        cases.extend(json.loads(path.read_bytes())["data"]["hits"])
    return cases, query


# ---------------------------------------------------------------- the folding and the primary diagnosis

def normalize(text):
    """A value as the tables look it up: whitespace removed, case ignored."""
    return re.sub(r"\s+", "", str(text)).casefold()


def spelling_table(spellings):
    """{normalized spelling: answer} from {answer: [spellings]}."""
    table = {}
    for answer, names in spellings.items():
        for name in names:
            key = normalize(name)
            if key in table:
                raise ValueError(f"{name!r} is listed twice")
            table[key] = answer
    return table


TABLES = {qid: spelling_table(q["spellings"]) for qid, q in QUESTIONS.items()}


def fold(qid, raw):
    """The answer that a value folds to in question qid, or None when it is outside the answer set."""
    return None if raw is None else TABLES[qid].get(normalize(raw))


def fold_t(raw):
    return fold("tcga_t", raw)


def fold_n(raw):
    return fold("tcga_n", raw)


def fold_stage(raw):
    return fold("tcga_stage", raw)


def fold_grade(raw):
    return fold("tcga_grade", raw)


FOLDS = {"tcga_t": fold_t, "tcga_n": fold_n, "tcga_stage": fold_stage, "tcga_grade": fold_grade}


def stated(value):
    """A field's raw text, or None when the GDC gives no value (the key absent, null or blank)."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def primary_value(diagnoses, field, fold_value=None):
    """(answer, reason, raw) for one case and one field by the primary-diagnosis rule (module docstring).

    answer is the folded value (the stated value itself when fold_value is None) or None; reason is None or one of
    REASONS; raw is the stated values' distinct raw texts, sorted and joined by " | " (None without a stated value)."""
    primaries = [d for d in (diagnoses or []) if isinstance(d, dict) and d.get("classification_of_tumor") == PRIMARY]
    if not primaries:
        return None, NO_PRIMARY, None
    values = [v for v in (stated(d.get(field)) for d in primaries) if v is not None]
    if not values:
        return None, NO_VALUE, None
    raw = " | ".join(sorted(set(values)))
    results = set()
    for v in values:
        answer = fold_value(v) if fold_value else v
        results.add((True, answer) if answer is not None else (False, v))
    if len(results) > 1:
        return None, DISAGREE, raw
    in_set, value = results.pop()
    return (value, None, raw) if in_set else (None, OUTSIDE, raw)


# ---------------------------------------------------------------- the splits and the records

def cut_points(n):
    """Where a project's shuffled list of n patients is cut: 80 and 90 percent of n, halves rounded up."""
    return (CUTS_PERCENT[0] * n + 50) // 100, (CUTS_PERCENT[1] * n + 50) // 100


def split_patients(project_of, seed=SEED):
    """{patient: split} from {patient: project} alone, so no label is read (the rule is in the module docstring)."""
    rng = random.Random(seed)
    members = collections.defaultdict(list)
    for patient, project in project_of.items():
        members[project].append(patient)
    split_of = {}
    for project in sorted(members):
        patients = sorted(members[project])
        rng.shuffle(patients)
        a, b = cut_points(len(patients))
        for i, patient in enumerate(patients):
            split_of[patient] = SPLITS[0] if i < a else SPLITS[1] if i < b else SPLITS[2]
    return split_of


REPORT_FILE_COLUMN = "patient_filename"
REPORT_TEXT_COLUMN = "text"
BARCODE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
LATER_REPORTS = "later reports of a patient who already has one (left out)"
NOT_A_BARCODE = "rows whose file name does not begin with a patient barcode (left out)"


def report_columns(header):
    """The indices of the file-name column and the text column of TCGA_Reports.csv's header."""
    names = [h.strip() for h in header]
    if REPORT_FILE_COLUMN in names and REPORT_TEXT_COLUMN in names:
        return names.index(REPORT_FILE_COLUMN), names.index(REPORT_TEXT_COLUMN)
    files = [i for i, h in enumerate(names) if "filename" in h.lower()]
    texts = [i for i, h in enumerate(names) if h.lower() == "text"]
    if len(files) == 1 and len(texts) == 1:
        return files[0], texts[0]
    raise SystemExit(f"TCGA_Reports.csv's columns are {header}; expected a file-name column ({REPORT_FILE_COLUMN}) and "
                     f"a text column ({REPORT_TEXT_COLUMN})")


def read_reports(zip_path):
    """The report rows [(file name, text)] in the file's order, and what was read (member, header, columns)."""
    csv.field_size_limit(1 << 30)
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.lower().endswith(".csv") and not m.startswith("__MACOSX/")
                   and not Path(m).name.startswith("._")]
        if len(members) != 1:
            raise SystemExit(f"{Path(zip_path).name} holds {len(members)} CSV files ({members}); expected one")
        with z.open(members[0]) as f:
            reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8-sig", newline=""))
            header = next(reader)
            fi, ti = report_columns(header)
            rows = [(row[fi] if fi < len(row) else "", row[ti] if ti < len(row) else "") for row in reader if row]
    return rows, {"member": members[0], "header": header, "file_column": header[fi].strip(),
                  "text_column": header[ti].strip()}


def patient_of(name):
    """The patient's barcode from a report's file name: its first 12 characters, case ignored."""
    return name.strip()[:12].upper()


def select_reports(rows):
    """{patient: text}, one report per patient (the first in the file's order), and the counts."""
    reports = {}
    counts = {"rows in the file": len(rows), NOT_A_BARCODE: 0, LATER_REPORTS: 0}
    for name, text in rows:
        patient = patient_of(name)
        if not BARCODE_RE.fullmatch(patient):
            counts[NOT_A_BARCODE] += 1
        elif patient in reports:
            counts[LATER_REPORTS] += 1
        else:
            reports[patient] = text
    counts["patients with a report"] = len(reports)
    counts["reports by the paper"] = PAPER_REPORTS
    return reports, counts


def make_record(qid, patient, project, text, label, split):
    rec = {"id": f"tcga-{qid}-{patient}", "qid": qid, "state": {"report": text}, "label": label,
           "target": one_hot(label, len(QUESTIONS[qid]["answers"]))}
    if split == "validation":
        rec["calibrate"] = True
    rec["group"] = {"source": SOURCE, "patient": patient, "project": project}
    return rec


def make_records(rows, cases, seed=SEED):
    """({qid: {split: records}}, the build's counts) from the report rows [(file name, text)] in the file's order and
    the GDC's cases. The split is made before any label is read and reads none."""
    reports, report_counts = select_reports(rows)
    by_barcode = collections.defaultdict(list)
    for case in cases:
        by_barcode[case.get("submitter_id")].append(case)
    join = {"cases fetched from the GDC": len(cases),
            "report rows that join no case": sum(1 for name, _ in rows if BARCODE_RE.fullmatch(patient_of(name))
                                                 and not by_barcode.get(patient_of(name))),
            "patients whose report joins no case": 0, "patients whose barcode names several cases": 0,
            "patients whose kept report is empty": 0}
    joined = {}
    for patient, text in reports.items():
        found = by_barcode.get(patient, [])
        if not found:
            join["patients whose report joins no case"] += 1
        elif len(found) > 1:
            join["patients whose barcode names several cases"] += 1
        elif not text.strip():
            join["patients whose kept report is empty"] += 1
        else:
            joined[patient] = (found[0], text)
    join["patients joined"] = len(joined)
    project_of = {p: ((case.get("project") or {}).get("project_id") or "(none)") for p, (case, _) in joined.items()}
    split_of = split_patients(project_of, seed)
    by_project = collections.defaultdict(lambda: {s: 0 for s in SPLITS})
    for patient, split in split_of.items():
        by_project[project_of[patient]][split] += 1
    splits = {"rule": SPLIT_RULE, "seed": seed, "patients": {s: sum(1 for v in split_of.values() if v == s) for s in SPLITS},
              "by_project": dict(sorted(by_project.items()))}

    recs = {qid: {s: [] for s in SPLITS} for qid in QIDS}
    questions = {}
    for qid, q in QUESTIONS.items():
        drops = {r: 0 for r in REASONS}
        left = {OUTSIDE: collections.Counter(), DISAGREE: collections.Counter()}
        folded = {a: collections.Counter() for a in q["answers"]}
        for patient in sorted(joined):
            case, text = joined[patient]
            answer, reason, raw = primary_value(case.get("diagnoses"), q["field"], FOLDS[qid])
            if answer is None:
                drops[reason] += 1
                if reason in left:
                    left[reason][raw] += 1
                continue
            folded[answer][raw] += 1
            split = split_of[patient]
            recs[qid][split].append(make_record(qid, patient, project_of[patient], text, q["answers"].index(answer), split))
        questions[qid] = {"field": q["field"], "records": {s: len(recs[qid][s]) for s in SPLITS}, "drops": drops,
                          "left_out_values": {k: dict(sorted(v.items())) for k, v in left.items()},
                          "folded_from": {a: dict(sorted(c.items())) for a, c in folded.items()}}

    counted_only = {}
    for field in COUNTED_ONLY:
        values, left_out = collections.Counter(), {r: 0 for r in (NO_PRIMARY, NO_VALUE, DISAGREE)}
        for patient in sorted(joined):
            value, reason, _ = primary_value(joined[patient][0].get("diagnoses"), field)
            if value is None:
                left_out[reason] += 1
            else:
                values[value] += 1
        counted_only[field] = {"values": dict(sorted(values.items())), "left_out": left_out}
    classes, primaries = collections.Counter(), collections.Counter()
    for patient in joined:
        diagnoses = joined[patient][0].get("diagnoses") or []
        for d in diagnoses:
            classes[stated(d.get("classification_of_tumor")) or "(no value)"] += 1
        primaries[str(sum(1 for d in diagnoses if d.get("classification_of_tumor") == PRIMARY))] += 1
    summary = {"reports": report_counts, "join": join, "splits": splits, "questions": questions,
               "fetched_and_counted_only": counted_only,
               "classification_of_tumor_of_joined_cases": dict(sorted(classes.items())),
               "primary_diagnoses_per_joined_case": dict(sorted(primaries.items(), key=lambda kv: int(kv[0])))}
    return recs, summary


def records(out_dir=OUT, sha256=ZIP_SHA256, seed=SEED):
    """Writes the twelve record files and parts/records_build.json; returns the build's record."""
    out_dir = Path(out_dir)
    require_licenses(out_dir)
    zip_path = out_dir / "raw" / ZIP_NAME
    if not zip_path.exists():
        raise SystemExit(f"run download first: {rel(zip_path)} does not exist")
    zip_sha = sha256_file(zip_path)
    if zip_sha != sha256:
        raise SystemExit(f"{rel(zip_path)} has SHA-256 {zip_sha}, not {sha256}")
    cases, query = read_cases(out_dir)
    rows, csv_info = read_reports(zip_path)
    recs, summary = make_records(rows, cases, seed)
    written = {}
    for qid in QIDS:
        for split in SPLITS:
            path = write_jsonl(out_dir / f"{qid}_{split}.jsonl", recs[qid][split])
            written[path.name] = sha256_file(path)
    fetched = load_json(out_dir / "parts" / "download_progress.json", {})
    build = {"plan": PLAN, "written_utc": now(), "seed": seed,
             "inputs": {"reports": {"file": "raw/" + ZIP_NAME, "url": ZIP_URL, "doi": MENDELEY_DOI,
                                    "bytes": zip_path.stat().st_size, "sha256": zip_sha,
                                    "matches_constant": zip_sha == ZIP_SHA256,
                                    "obtained": fetched.get("source"), "obtained_utc": fetched.get("finished_utc"), **csv_info},
                        "gdc": {"file": "raw/gdc/query.json", "sha256": sha256_file(out_dir / "raw" / "gdc" / "query.json"),
                                "endpoint": query["query"]["endpoint"], "filters": query["query"]["filters"],
                                "fields": query["query"]["fields"], "sort": query["query"]["sort"],
                                "page_size": query["query"]["size"], "fetched_first_utc": query["fetched_first_utc"],
                                "fetched_last_utc": query["fetched_last_utc"], "total": query["total"],
                                "pages": len(query["pages"])}},
             **summary, "records_sha256": written}
    save_json(out_dir / "parts" / "records_build.json", build)
    print(json.dumps({"records": {qid: summary["questions"][qid]["records"] for qid in QIDS},
                      "patients": summary["splits"]["patients"], "join": summary["join"]}, indent=1))
    return build


# ---------------------------------------------------------------- counts

def percentile(values, q):
    """Session 8's rule (src/tasks/evidence_inference.py): the value at index int(q n) of the sorted values."""
    v = sorted(values)
    return v[min(len(v) - 1, int(q * len(v)))]


def length_summary(values):
    values = list(values)
    if not values:
        return {"n": 0, "mean": None, "p90": None, "largest": None}
    return {"n": len(values), "mean": sum(values) / len(values), "p90": percentile(values, 0.9), "largest": max(values)}


def load_tokenizer(tokenizer_dir):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


def report_tokens(texts, cache_path, tokenizer_dir=QWEN_DIR, max_seconds=150.0, tokenizer=None, batch=32):
    """{patient: the report's length in tokens}, or None when the time ran out; the counts are kept in cache_path
    (one JSON line per report, with the text's SHA-256), so the next run resumes. Every run counts at least one batch."""
    t0 = time.time()
    cache_path = Path(cache_path)
    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done[(r["patient"], r["text_sha256"])] = r["tokens"]
            except (ValueError, KeyError, TypeError):
                continue  # a line cut short when a run was stopped
    want = {p: hashlib.sha256(t.encode("utf-8")).hexdigest() for p, t in texts.items()}
    todo = [p for p in sorted(texts) if (p, want[p]) not in done]
    if todo:
        tok = tokenizer or load_tokenizer(tokenizer_dir)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "a", encoding="utf-8") as f:
            for k in range(0, len(todo), batch):
                if k and time.time() - t0 > max_seconds:
                    return None
                chunk = todo[k:k + batch]
                ids = tok([texts[p] for p in chunk], add_special_tokens=False)["input_ids"]
                for p, x in zip(chunk, ids):
                    f.write(json.dumps({"patient": p, "text_sha256": want[p], "tokens": len(x)}) + "\n")
                    done[(p, want[p])] = len(x)
                f.flush()
    return {p: done[(p, want[p])] for p in texts}


def label_words_in(qid, answer, text):
    return any(r.search(text) for r in WORD_RES[qid][answer])


def label_word_counts(qid, items):
    """How many of the (report, label) items state their label's own words, and another answer's (module
    docstring). Called with the training and validation splits only."""
    answers = QUESTIONS[qid]["answers"]
    out = {"records": 0, "label_words": 0, "share": None, "other_answer_words": 0, "other_share": None,
           "by_answer": {a: {"records": 0, "label_words": 0} for a in answers}}
    for text, label in items:
        answer = answers[label]
        own = label_words_in(qid, answer, text)
        other = any(label_words_in(qid, a, text) for a in answers if a != answer)
        out["records"] += 1
        out["label_words"] += own
        out["other_answer_words"] += other
        out["by_answer"][answer]["records"] += 1
        out["by_answer"][answer]["label_words"] += own
    if out["records"]:
        out["share"] = out["label_words"] / out["records"]
        out["other_share"] = out["other_answer_words"] / out["records"]
    return out


def file_hashes(out_dir):
    out_dir = Path(out_dir)
    block = {"records": {f"{qid}_{s}.jsonl": sha256_file(out_dir / f"{qid}_{s}.jsonl") for qid in QIDS for s in SPLITS},
             "licenses.json": sha256_file(out_dir / "licenses.json"), "parts": {}, "raw": {}}
    for p in sorted((out_dir / "parts").glob("*")):
        if p.is_file() and not p.name.endswith((".part", ".tmp")):
            block["parts"][p.name] = sha256_file(p)
    for p in sorted((out_dir / "raw").rglob("*")):
        if p.is_file() and not p.name.endswith((".part", ".tmp")):
            block["raw"][p.relative_to(out_dir / "raw").as_posix()] = sha256_file(p)
    return block


def counts(out_dir=OUT, results=RESULTS, tokenizer_dir=QWEN_DIR, max_seconds=150.0, tokenizer=None, batch=32):
    """Writes results/tcga_prep/records_manifest.json; a status dict ("finished" false while the token counts run)."""
    out_dir, results = Path(out_dir), Path(results)
    lic = require_licenses(out_dir)
    build = load_json(out_dir / "parts" / "records_build.json", None)
    if build is None:
        raise SystemExit("run records first: parts/records_build.json does not exist")
    for name, digest in build["records_sha256"].items():
        if sha256_file(out_dir / name) != digest:
            raise SystemExit(f"{name} differs from parts/records_build.json; run records again")
    texts, rows, word_items = {}, {}, {}
    for qid in QIDS:
        rows[qid], word_items[qid] = {}, {}
        for split in SPLITS:
            rows[qid][split] = []
            if split in WORD_SPLITS:
                word_items[qid][split] = []
            for rec in iter_jsonl(out_dir / f"{qid}_{split}.jsonl"):
                patient, text = rec["group"]["patient"], rec["state"]["report"]
                texts.setdefault(patient, text)
                rows[qid][split].append((patient, rec["group"]["project"], rec["label"]))  # the test labels: counted only
                if split in WORD_SPLITS:
                    word_items[qid][split].append((texts[patient], rec["label"]))  # one copy of each report in memory
    tokens = report_tokens(texts, out_dir / "parts" / "report_tokens.jsonl", tokenizer_dir, max_seconds, tokenizer, batch)
    if tokens is None:
        cached = (out_dir / "parts" / "report_tokens.jsonl").read_text(encoding="utf-8").splitlines()
        status = {"finished": False, "reports": len(texts), "counted": sum(1 for line in cached if line.strip())}
        print(json.dumps(status))
        return status

    count_block, token_block = {}, {"by_split": {}, "by_question": {}}
    for qid, q in QUESTIONS.items():
        answers = q["answers"]
        count_block[qid] = {"field": q["field"], "answers": list(answers)}
        token_block["by_question"][qid] = {}
        for split in SPLITS:
            these = rows[qid][split]
            projects = collections.defaultdict(collections.Counter)
            for _, project, label in these:
                projects[project][answers[label]] += 1
            tally = collections.Counter(answers[label] for _, _, label in these)
            count_block[qid][split] = {
                "records": len(these), "answers": {a: tally.get(a, 0) for a in answers},
                "projects": {p: {"records": sum(c.values()), "answers": {a: c.get(a, 0) for a in answers}}
                             for p, c in sorted(projects.items())}}
            token_block["by_question"][qid][split] = length_summary(tokens[p] for p, _, _ in these)
    for split in SPLITS:
        in_split = {p for qid in QIDS for p, _, _ in rows[qid][split]}
        token_block["by_split"][split] = length_summary(tokens[p] for p in sorted(in_split))
    revision_path = Path(tokenizer_dir) / "REVISION.txt"
    revision_file = revision_path.read_text(encoding="utf-8").strip() if revision_path.exists() else None
    token_block = {"tokenizer": rel(tokenizer_dir), "revision_expected": QWEN_REVISION, "revision_file": revision_file,
                   "revision_matches": bool(revision_file) and QWEN_REVISION in revision_file,
                   "counted": "each report's text as in the records' state, tokenized with add_special_tokens=False; a "
                              "report counted once however many questions it enters",
                   "all": length_summary(tokens[p] for p in sorted(texts)), **token_block}
    words = {"splits": list(WORD_SPLITS), "patterns": LABEL_WORDS,
             "rule": "a record counts when its report matches one of its answer's patterns (src/tasks/tcga_reports.py, "
                     "module docstring); other_answer_words counts the records whose report matches another answer's",
             "test": "not counted: the test split's labels are written and counted, never read for anything else"}
    for qid in QIDS:
        words[qid] = {split: label_word_counts(qid, word_items[qid][split]) for split in WORD_SPLITS}
    questions = build["questions"]
    manifest = {
        "plan": PLAN, "written_utc": now(), "trained": "nothing (I-38 is preparation only)",
        "licenses": {"allowed": lic["allowed"], "reason": lic["reason"],
                     "tcga_reports": (lic["tcga_reports"].get("data_licence") or {}).get("short_name"),
                     "read_utc": lic["tcga_reports"]["read_utc"], "gdc": GDC_TERMS["source"]},
        "inputs": build["inputs"],
        "questions": {qid: {"field": q["field"], "answers": list(q["answers"]),
                            "accepted_spellings": {a: list(s) for a, s in q["spellings"].items()}}
                      for qid, q in QUESTIONS.items()},
        "primary_rule": PRIMARY_RULE,
        "reports": build["reports"], "join": build["join"], "patients": build["splits"],
        "counts": count_block,
        "drops": {qid: questions[qid]["drops"] for qid in QIDS},
        "left_out_values": {qid: questions[qid]["left_out_values"] for qid in QIDS},
        "folded_from": {qid: questions[qid]["folded_from"] for qid in QIDS},
        "fetched_and_counted_only": build["fetched_and_counted_only"],
        "classification_of_tumor_of_joined_cases": build["classification_of_tumor_of_joined_cases"],
        "primary_diagnoses_per_joined_case": build["primary_diagnoses_per_joined_case"],
        "tokens": token_block,
        "label_words": words,
        "test_split": "its labels are written and counted (counts), never read for anything else; the label's own "
                      "words are not counted on it",
        "sha256": file_hashes(out_dir),
    }
    save_json(results / "records_manifest.json", manifest)
    status = {"finished": True, "manifest": rel(results / "records_manifest.json"),
              "records": {qid: {s: count_block[qid][s]["records"] for s in SPLITS} for qid in QIDS},
              "tokens": token_block["all"]}
    print(json.dumps(status, indent=1))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    lc = sub.add_parser("licenses")
    lc.add_argument("--out-dir", type=Path, default=OUT)
    d = sub.add_parser("download")
    d.add_argument("--out-dir", type=Path, default=OUT)
    d.add_argument("--max-seconds", type=float, default=150.0)
    lb = sub.add_parser("labels")
    lb.add_argument("--out-dir", type=Path, default=OUT)
    lb.add_argument("--max-seconds", type=float, default=150.0)
    lb.add_argument("--page-size", type=int, default=GDC_PAGE_SIZE)
    r = sub.add_parser("records")
    r.add_argument("--out-dir", type=Path, default=OUT)
    c = sub.add_parser("counts")
    c.add_argument("--out-dir", type=Path, default=OUT)
    c.add_argument("--results", type=Path, default=RESULTS)
    c.add_argument("--tokenizer-dir", type=Path, default=QWEN_DIR)
    c.add_argument("--max-seconds", type=float, default=150.0)
    args = parser.parse_args(argv)
    if args.cmd == "licenses":
        out = licenses(args.out_dir)
        if not out["allowed"]:
            print(out["not_built"])
            return 1
        print(json.dumps({"allowed": True, "reason": out["reason"],
                          "file_matches_constants": out["tcga_reports"]["file_matches_constants"]}))
    elif args.cmd == "download":
        download(args.out_dir, args.max_seconds)
    elif args.cmd == "labels":
        labels(args.out_dir, args.max_seconds, args.page_size)
    elif args.cmd == "records":
        records(args.out_dir)
    else:
        counts(args.out_dir, args.results, args.tokenizer_dir, args.max_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())

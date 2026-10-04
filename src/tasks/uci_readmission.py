"""Public v0 records for the UCI Diabetes 130-US hospitals data, question readmit_window
(docs/prompt_session4.md, step 4).

Source: UCI Machine Learning Repository dataset 296, "Diabetes 130-US Hospitals for Years 1999-2008", CC BY 4.0
(diabetic_data.csv and IDS_mapping.csv in the downloaded zip, fixed by SHA-256).

Cohort: the first encounter of each patient (lowest encounter_id per patient_nbr); then encounters whose
discharge_disposition_id is 11, 13, 14, 19, 20 or 21 (death or hospice) are dropped. Split by patient into train,
validation and test (80, 10, 10 percent; seed 20260927). State: age band as recorded, sex, admission type,
admission source and discharge disposition as text from IDS_mapping.csv, time_in_hospital, num_lab_procedures,
num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses,
diag_1 to diag_3 as code plus ICD-9 chapter name, max_glu_serum, A1Cresult, each medication whose value is not
"No" with its value, change, diabetesMed, medical_specialty. Values are as recorded; missing values ("?") are
left out. Never in the state: encounter_id, patient_nbr, race, payer_code, weight, readmitted.
Label: NO 0, >30 1, <30 2. group holds sex, age band and race, for subgroup analysis only.

Usage: python src/tasks/uci_readmission.py [--raw data/raw/public/uci] [--out data/processed/public]
"""
import argparse
import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import records as R  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"
SHA256 = "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
ZIP_NAME = "diabetes_130_us_hospitals_1999_2008.zip"
QID = "readmit_window"
LABEL = {"NO": 0, ">30": 1, "<30": 2}
DEATH_OR_HOSPICE = {"11", "13", "14", "19", "20", "21"}
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
    "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
    "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]
NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
           "number_emergency", "number_inpatient", "number_diagnoses"]
ICD9_CHAPTERS = [
    (1, 139, "Infectious and parasitic diseases"), (140, 239, "Neoplasms"),
    (240, 279, "Endocrine, nutritional and metabolic diseases, and immunity disorders"),
    (280, 289, "Diseases of the blood and blood-forming organs"), (290, 319, "Mental disorders"),
    (320, 389, "Diseases of the nervous system and sense organs"), (390, 459, "Diseases of the circulatory system"),
    (460, 519, "Diseases of the respiratory system"), (520, 579, "Diseases of the digestive system"),
    (580, 629, "Diseases of the genitourinary system"),
    (630, 679, "Complications of pregnancy, childbirth, and the puerperium"),
    (680, 709, "Diseases of the skin and subcutaneous tissue"),
    (710, 739, "Diseases of the musculoskeletal system and connective tissue"), (740, 759, "Congenital anomalies"),
    (760, 779, "Certain conditions originating in the perinatal period"),
    (780, 799, "Symptoms, signs, and ill-defined conditions"), (800, 999, "Injury and poisoning"),
]


def icd9_chapter(code):
    if code.startswith("V"):
        return "Supplementary classification of factors influencing health status and contact with health services"
    if code.startswith("E"):
        return "Supplementary classification of external causes of injury and poisoning"
    number = int(float(code))
    for low, high, name in ICD9_CHAPTERS:
        if low <= number <= high:
            return name
    raise ValueError(f"ICD-9 code outside every chapter: {code}")


def parse_mapping(text):
    """IDS_mapping.csv holds three id-to-description tables separated by blank lines."""
    tables, current = {}, None
    for row in csv.reader(io.StringIO(text)):
        if not row or not any(cell.strip() for cell in row):
            current = None
            continue
        if row[0].endswith("_id"):
            current = tables.setdefault(row[0], {})
            continue
        if current is not None:
            current[row[0].strip()] = row[1].strip()
    return tables


def state_of(row, mapping):
    state = {"age band": row["age"], "sex": row["gender"]}
    for key, column in (("admission type", "admission_type_id"), ("admission source", "admission_source_id"),
                        ("discharge disposition", "discharge_disposition_id")):
        state[key] = mapping[column][row[column]]
    for column in NUMERIC[:-1]:
        state[column] = int(row[column])
    state["number_diagnoses"] = int(row["number_diagnoses"])
    for column in ("diag_1", "diag_2", "diag_3"):
        if row[column] != "?":
            state[column] = f"{row[column]} ({icd9_chapter(row[column])})"
    for column in ("max_glu_serum", "A1Cresult"):
        if row[column] != "?":
            state[column] = row[column]
    medications = {m: row[m] for m in MEDICATIONS if row[m] not in ("No", "?")}
    if medications:
        state["medications"] = medications
    for column in ("change", "diabetesMed", "medical_specialty"):
        if row[column] != "?":
            state[column] = row[column]
    return state


def build(rows, mapping, seed=R.SEED):
    """Pure cohort and split logic on already-loaded rows; returns ({split: records}, counts)."""
    first = {}
    for row in rows:
        patient, encounter = row["patient_nbr"], int(row["encounter_id"])
        if patient not in first or encounter < int(first[patient]["encounter_id"]):
            first[patient] = row
    kept = [r for r in first.values() if r["discharge_disposition_id"] not in DEATH_OR_HOSPICE]
    patients = sorted(r["patient_nbr"] for r in kept)
    random.Random(seed).shuffle(patients)
    n = len(patients)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    split_of = {}
    for i, p in enumerate(patients):
        split_of[p] = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
    splits = {s: [] for s in R.SPLITS}
    for row in sorted(kept, key=lambda r: int(r["encounter_id"])):
        label = LABEL[row["readmitted"]]
        splits[split_of[row["patient_nbr"]]].append({
            "id": f"uci-{row['encounter_id']}", "qid": QID, "state": state_of(row, mapping), "label": label,
            "target": R.one_hot(label, 3),
            "group": {"sex": row["gender"], "age_band": row["age"], "race": row["race"] if row["race"] != "?" else "unknown"},
        })
    counts = {
        "rows_read": len(rows),
        "excluded": {
            "not_first_encounter_of_patient": len(rows) - len(first),
            "death_or_hospice_disposition": len(first) - len(kept),
        },
        "patients": n,
        "sources": {ZIP_NAME: {"url": URL, "sha256": SHA256}},
    }
    return splits, counts


def load_sources(raw_dir):
    path = R.fetch(URL, Path(raw_dir) / ZIP_NAME, SHA256)
    with zipfile.ZipFile(path) as z:
        names = {n.lower(): n for n in z.namelist()}
        data = z.read(names["diabetic_data.csv"]).decode("utf-8")
        mapping = z.read(names["ids_mapping.csv"]).decode("utf-8")
    return list(csv.DictReader(io.StringIO(data))), parse_mapping(mapping)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--raw", type=Path, default=R.RAW / "uci")
    parser.add_argument("--out", type=Path, default=R.OUT)
    parser.add_argument("--results", type=Path, default=R.RESULTS)
    args = parser.parse_args()
    rows, mapping = load_sources(args.raw)
    splits, counts = build(rows, mapping)
    checksums = R.finish(QID, splits, counts, args.out, args.results)
    for split in R.SPLITS:
        print(f"{QID} {split}: {len(splits[split]):,} records")
    print(json.dumps(checksums, indent=2))


if __name__ == "__main__":
    main()

"""Our public v0 records in Duan's state format and questions (docs/prompt_session4.md, step 9).

Duan's formats are taken from its demo (docs/prior_art_snapshots/2026-09-27_duan-demo-index.html), which shows one
real test case per task:
- PubMedQA, question duan_conclusion_supported (noul, options [false, true]): state {question, contexts as
  [{section, text}], year}. Only items labeled yes or no are kept (Duan's task is binary): yes becomes true, no
  becomes false. The year comes from the dataset's original file data/ori_pqal.json (fixed commit, SHA-256 below).
- UCI, question duan_readmit_window (score, the same three levels as ours): state with age_years (the middle of
  the recorded ten-year band), sex, race, admission_type_id, discharge_disposition_id, admission_source_id,
  time_in_hospital_days, the counts, number_diagnoses, a1c_result, diagnoses {primary, secondary, additional},
  medications {name: value} and diabetes_med, medication_change, in the demo's order and lower case where the demo
  is lower case. Not shown in the demo and so assumed: max_glu_serum under that name after a1c_result; values
  "None" (not measured) and "?" (missing) left out; medical_specialty left out. The raw fields come from the UCI
  file by encounter ID.
Record IDs, splits and calibrate flags are the same as ours, so every Duan prediction pairs with ours.
Writes data/processed/public/duan/<qid>_<split>.jsonl for the validation and test splits.

Usage: python src/eval/duan_adapter.py
"""
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
import records as R  # noqa: E402
import uci_readmission as U  # noqa: E402

ORI_PQAL = ("https://raw.githubusercontent.com/pubmedqa/pubmedqa/3e7616d288b52edaf38f5fd5c59821c7806bde86/data/ori_pqal.json",
            "8b3276be8942ebbd77f3ddcda12c1749bf0e490045a736fd8438ee40cf37a41d")
OUT = R.OUT / "duan"


def age_years(band):
    low, high = band.strip("[)").split("-")
    return (float(low) + float(high)) / 2


def uci_state(row):
    state = {"age_years": age_years(row["age"]), "sex": row["gender"].lower()}
    if row["race"] != "?":
        state["race"] = row["race"].lower()
    for column in ("admission_type_id", "discharge_disposition_id", "admission_source_id"):
        state[column] = int(row[column])
    state["time_in_hospital_days"] = int(row["time_in_hospital"])
    for column in ("num_lab_procedures", "num_procedures", "num_medications", "number_outpatient", "number_emergency",
                   "number_inpatient", "number_diagnoses"):
        state[column] = int(row[column])
    if row["A1Cresult"] not in ("None", "?"):
        state["a1c_result"] = row["A1Cresult"]
    if row["max_glu_serum"] not in ("None", "?"):
        state["max_glu_serum"] = row["max_glu_serum"]
    diagnoses = {name: row[col] for name, col in (("primary", "diag_1"), ("secondary", "diag_2"), ("additional", "diag_3")) if row[col] != "?"}
    if diagnoses:
        state["diagnoses"] = diagnoses
    medications = {m: row[m].lower() for m in U.MEDICATIONS if row[m] not in ("No", "?")}
    if medications:
        state["medications"] = medications
    state["diabetes_med"] = row["diabetesMed"].lower()
    state["medication_change"] = row["change"].lower()
    return state


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ori = json.loads(Path(R.fetch(*ORI_PQAL[:1], R.RAW / "pubmedqa" / "ori_pqal.json", ORI_PQAL[1])).read_text(encoding="utf-8"))
    uci_zip = R.fetch(U.URL, R.RAW / "uci" / U.ZIP_NAME, U.SHA256)
    with zipfile.ZipFile(uci_zip) as z:
        names = {n.lower(): n for n in z.namelist()}
        raw = {r["encounter_id"]: r for r in csv.DictReader(io.StringIO(z.read(names["diabetic_data.csv"]).decode("utf-8")))}
    counts = {}
    for split in ("validation", "test"):
        out = []
        for rec in R.read_jsonl(R.OUT / f"pubmedqa_answer_{split}.jsonl"):
            if rec["label"] == 2:
                continue
            pubid = rec["id"].split("-", 1)[1]
            state = dict(rec["state"])
            if pubid in ori:
                state["year"] = ori[pubid]["YEAR"]
            label = 1 if rec["label"] == 0 else 0  # yes -> true (index 1), no -> false (index 0)
            new = {"id": rec["id"], "qid": "duan_conclusion_supported", "state": state, "label": label,
                   "target": R.one_hot(label, 2), "group": rec.get("group", {})}
            if rec.get("calibrate") is False:
                new["calibrate"] = False
            out.append(new)
        R.write_jsonl(OUT / f"duan_conclusion_supported_{split}.jsonl", out)
        counts[f"duan_conclusion_supported_{split}"] = len(out)
        out = []
        for rec in R.read_jsonl(R.OUT / f"readmit_window_{split}.jsonl"):
            row = raw[rec["id"].split("-", 1)[1]]
            out.append({"id": rec["id"], "qid": "duan_readmit_window", "state": uci_state(row), "label": rec["label"],
                        "target": rec["target"], "group": rec.get("group", {})})
        R.write_jsonl(OUT / f"duan_readmit_window_{split}.jsonl", out)
        counts[f"duan_readmit_window_{split}"] = len(out)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

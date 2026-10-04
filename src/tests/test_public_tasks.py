"""Tests for the public v0 record builders (src/tasks/), on invented rows only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks"))
import pubmedqa  # noqa: E402
import uci_readmission as uci  # noqa: E402

MAPPING = {
    "admission_type_id": {"1": "Emergency", "3": "Elective"},
    "discharge_disposition_id": {"1": "Discharged to home", "11": "Expired", "13": "Hospice / home"},
    "admission_source_id": {"7": "Emergency Room", "1": "Physician Referral"},
}


def uci_row(encounter, patient, readmitted="NO", disposition="1", **extra):
    row = {
        "encounter_id": str(encounter), "patient_nbr": str(patient), "race": "Caucasian", "gender": "Female",
        "age": "[70-80)", "weight": "?", "admission_type_id": "1", "discharge_disposition_id": disposition,
        "admission_source_id": "7", "time_in_hospital": "3", "payer_code": "MC", "medical_specialty": "?",
        "num_lab_procedures": "40", "num_procedures": "1", "num_medications": "12", "number_outpatient": "0",
        "number_emergency": "0", "number_inpatient": "1", "diag_1": "428", "diag_2": "250.83", "diag_3": "?",
        "number_diagnoses": "5", "max_glu_serum": "None", "A1Cresult": ">8", "change": "Ch", "diabetesMed": "Yes",
        "readmitted": readmitted,
    }
    for m in uci.MEDICATIONS:
        row[m] = "No"
    row["insulin"] = "Up"
    row.update(extra)
    return row


def test_uci_first_encounter_exclusions_labels_and_state():
    rows = [
        uci_row(10, "A", readmitted="<30"), uci_row(5, "A", readmitted=">30"),  # first encounter of A is 5
        uci_row(7, "B", disposition="11"),  # death: dropped
        uci_row(8, "C", disposition="13"),  # hospice: dropped
    ] + [uci_row(100 + i, f"P{i}", readmitted=["NO", ">30", "<30"][i % 3]) for i in range(20)]
    splits, counts = uci.build(rows, MAPPING)
    records = [r for s in splits.values() for r in s]
    ids = [r["id"] for r in records]
    assert "uci-5" in ids and "uci-10" not in ids
    assert "uci-7" not in ids and "uci-8" not in ids
    assert counts["excluded"] == {"not_first_encounter_of_patient": 1, "death_or_hospice_disposition": 2}
    assert len(ids) == len(set(ids)) == 21
    a = next(r for r in records if r["id"] == "uci-5")
    assert a["label"] == 1 and a["target"] == [0.0, 1.0, 0.0]
    for r in records:
        assert r["label"] == uci.LABEL[next(x for x in rows if f"uci-{x['encounter_id']}" == r["id"])["readmitted"]]
    for forbidden in ("race", "payer_code", "weight", "encounter_id", "patient_nbr", "readmitted"):
        assert forbidden not in a["state"]
    assert "diag_3" not in a["state"] and "medical_specialty" not in a["state"]
    assert a["state"]["diag_1"] == "428 (Diseases of the circulatory system)"
    assert a["state"]["diag_2"].startswith("250.83 (Endocrine")
    assert a["state"]["medications"] == {"insulin": "Up"}
    assert a["state"]["admission type"] == "Emergency" and a["state"]["discharge disposition"] == "Discharged to home"
    assert a["group"] == {"sex": "Female", "age_band": "[70-80)", "race": "Caucasian"}
    split_of = {r["id"]: s for s, rs in splits.items() for r in rs}
    assert sum(1 for s in split_of.values() if s == "train") == int(21 * 0.8)


def test_icd9_chapters():
    assert uci.icd9_chapter("V45") .startswith("Supplementary classification of factors")
    assert uci.icd9_chapter("E888").startswith("Supplementary classification of external causes")
    assert uci.icd9_chapter("786") == "Symptoms, signs, and ill-defined conditions"
    with pytest.raises(ValueError):
        uci.icd9_chapter("1000")


def pqa_row(pubid, decision, n_sections=2):
    return {
        "pubid": pubid, "question": f"Question {pubid}?", "final_decision": decision,
        "long_answer": f"The answer is {decision}.",
        "context": {"contexts": [f"text {i}" for i in range(n_sections)], "labels": ["BACKGROUND", "RESULTS"][:n_sections],
                    "meshes": ["Humans"], "reasoning_required_pred": [decision], "reasoning_free_pred": [decision]},
    }


def test_pubmedqa_splits_state_and_calibrate_flags(monkeypatch):
    monkeypatch.setattr(pubmedqa, "N_LABELED_VALIDATION", 2)
    monkeypatch.setattr(pubmedqa, "N_ARTIFICIAL_VALIDATION", 4)
    monkeypatch.setattr(pubmedqa, "N_ARTIFICIAL_TRAIN", 10)
    monkeypatch.setattr(pubmedqa, "LABELED_TRAIN_REPEATS", 3)
    decisions = ["yes", "no", "maybe"]
    labeled = [pqa_row(i, decisions[i % 3]) for i in range(1, 13)]
    test_ids = [1, 2, 3, 4]
    art = [pqa_row(1000 + i, "yes" if i % 3 else "no") for i in range(40)] + [pqa_row(5, "yes")]  # 5 overlaps pqa_labeled
    table = {"pubid": [r["pubid"] for r in art], "final_decision": [r["final_decision"] for r in art],
             "rows": lambda idx: [art[k] for k in idx]}
    splits, counts = pubmedqa.build(labeled, table, test_ids)
    assert sorted(r["id"] for r in splits["test"]) == [f"pubmedqa-{i}" for i in test_ids]
    assert counts["excluded"]["pqa_artificial_id_in_other_pqa_labeled"] == 1
    split_of = {}
    for s, rows in splits.items():
        for r in rows:
            assert split_of.setdefault(r["id"], s) == s
            assert set(r["state"]) == {"question", "contexts"}
            assert "answer is" not in str(r["state"])
            assert r["target"][r["label"]] == 1.0
    labeled_train = [r for r in splits["train"] if r["group"]["source"] == "pqa_labeled"]
    assert len(labeled_train) == 3 * (12 - 4 - 2)
    art_val = [r for r in splits["validation"] if r["group"]["source"] == "pqa_artificial"]
    assert len(art_val) == 4 and all(r["calibrate"] is False for r in art_val)
    assert all("calibrate" not in r for r in splits["validation"] if r["group"]["source"] == "pqa_labeled")
    assert splits["test"][0]["state"]["contexts"][0] == {"section": "BACKGROUND", "text": "text 0"}

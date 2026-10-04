"""Session 9's evaluation tests (docs/prompt_session9.md, step 4), merged on 1 October 2026 from eleven test files.

The tests run on made-up inputs written under tmp_path; no weights are downloaded and no network is used. The file has
eleven parts, each under a comment header that names what it tests and keeps its file's own description:

1. src/tasks/tcga_reports.py (the TCGA-Reports question; TCGA is The Cancer Genome Atlas): the label folding, the
   primary-diagnosis rule, the splits by patient, the record format, the label-word counts, the license gate, and the
   resumable download and page fetch. Test names test_tcga_*.
2. src/eval/handoff_curves_session9.py: the per-item AURC, E-AURC and AUGRC (the areas under the risk-coverage curves)
   against a hand calculation, the risk-coverage curve, the random curve's exact expectation, the oracle's bound, the
   confidence rule against session 8's, I-35's rule on a planted signal, the resampling grouped by article, and the
   commands with their guards. Test names test_handoff_*.
3. src/eval/trees_session9.py: the recoded diagnosis fields and the out-of-fold target encoding without leakage,
   lightgbm_tuned's refit check, the codes members' refits, the pool over eight members recovering a planted member,
   the batched grid on torch against session 8's fit, the 0.002 rule, TabICL's stand-in through the same interface,
   and the pool end to end. Test names test_trees_*.
4. src/eval/biogpt_session9.py: BioGPT's input with the learned prompt's tokens in place, the answers' probabilities
   from made-up next-token logits (one-token and multi-token answers), correction 5's reader on made-up archives, the
   archive's address read from microsoft/BioGPT's README, and the commands end to end. Test names test_biogpt_*.
5. src/eval/reading_checks_session9.py: the reading checks' scores, shares, intervals and readings on made-up
   predictions, the comparator's regular expression, v0.5's guard and swap averaging, and the stop. Test names
   test_reading_checks_*.
6. src/eval/calibration_family_cv.py: the four calibration families (each recovers a planted distortion), the vector
   family's free parameters, the objectives and gradients, the tie rules, out_of_fold and apply_family, and the
   command's output unchanged. Test names test_calfam_*.
7. src/eval/compare_session9.py: the guard before the freeze, the bar's four criteria with either large model, K3,
   the reading of a difference whose interval includes zero, the one-against-rest calibration slope and intercept on a
   planted miscalibration, the cost per 1,000 questions, and the comparisons end to end. Test names test_compare9_*.
8. Session 9's pod tools: pilot-s8's --headroom-gib (src/train/train_qwen_head.py), upload_to_pod.py accepting session
   9's stages, and scripts/pod_session9.sh run off the pod with stand-ins, stopping at each of its stops. Test names
   test_pod_*.
9. src/eval/recipe_session9.py, weight_average_session9.py and calibrate_session9.py: THE RECIPE RULE on made-up logits
   (the margin and its floor, two runs for each candidate, lr1 and lr2 as one change, the combination, the winner, base
   when none passes, swap averaging), the cost rule and its stop, the weight average's joined update against the mean
   of the runs' updates, and calibrate_session9.py's fit, the input of the ECE (expected calibration error) limit, the
   freeze's guards and apply's guards. Test names test_core9_*.
10. scripts/build_model_card_v0_5.py, scripts/predict_v0_5.py and scripts/release_fiorillo_v0_5.sh: the model card on
   made-up results (every number traced to its file), predict_v0_5.py on a made-up release folder with a two-layer
   Qwen3 stand-in, and the release script with a stand-in hf command. Test names test_rel9_*.
11. src/eval/check_fiorillo_v0_5_figures.py: the figure check on a made-up document and results files (PASS, MISMATCH
   and NO SOURCE, the rounding rule, intervals, readings, and the command's self-test). Test names test_fig9_*.

Each part's module-level names (constants, helpers, stand-in classes and strings) carry a prefix of their own: tcga, hc,
trees, biogpt, rc, calfam, cmp9, pod, core9, rel9 and fig9. No two parts share a name. The project's modules keep the
aliases their files gave them, which all differ, and src/, its folders and scripts/ are put on sys.path once, before the
imports. One part runs with -k and its test names' start, for example: python3 -m pytest src/tests/test_session9_eval.py
-k test_pod_
"""
import argparse
import copy
import csv
import csv as core9_csv
import datetime as core9_dt
import datetime as dt
import hashlib
import hashlib as core9_hashlib
import io
import itertools
import json
import json as core9_json
import math
import math as core9_math
import os
import os as core9_os
import pickle
import random
import re
import shlex as core9_shlex
import shutil
import subprocess
import subprocess as core9_subprocess
import sys
import sys as core9_sys
import tarfile
import time
import types
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from pathlib import Path as core9_Path

import numpy as core9_np
import numpy as np
import pytest
import pytest as core9_pytest
import torch as core9_torch
from safetensors.torch import load_file as core9_load_file
from safetensors.torch import save_file as core9_save_file
from scipy.optimize import approx_fprime

# src/, its folders and scripts/, whose modules the parts import by name, first on sys.path, as the eleven files put them
_PROJECT_SRC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(p) for p in (_PROJECT_SRC / "tasks", _PROJECT_SRC / "eval", _PROJECT_SRC / "tools",
                                 _PROJECT_SRC / "train", _PROJECT_SRC, _PROJECT_SRC.parent,
                                 _PROJECT_SRC.parent / "scripts")]

import tcga_reports as TCGA  # noqa: E402
import handoff_curves_session9 as HC  # noqa: E402
import handoff_session8 as H8  # noqa: E402
import trees_session7 as T7  # noqa: E402
import trees_session8 as T8  # noqa: E402
import trees_session9 as T9  # noqa: E402
import biogpt_session9 as BG  # noqa: E402
import ei_session9 as EI9  # noqa: E402
import reading_checks_session9 as RCHK  # noqa: E402
import calibrate_session8 as CAL8  # noqa: E402
import calibration_family_cv as CF  # noqa: E402
import calibrate_session7 as CMP9_CAL7  # noqa: E402
import calibration_family_cv as CMP9_CFC  # noqa: E402
import compare_session9 as CMP9  # noqa: E402
import metrics as CMP9_M  # noqa: E402
import upload_to_pod as U  # noqa: E402
from eval import calibrate_session7 as core9_C7  # noqa: E402
from eval import calibrate_session9 as core9_C9  # noqa: E402
from eval import calibration_family_cv as core9_CFC  # noqa: E402
from eval import recipe_session9 as core9_R9  # noqa: E402
from eval import weight_average_session9 as core9_W9  # noqa: E402
import build_model_card_v0_5 as rel9_card  # noqa: E402
import predict_v0_5 as rel9_predict  # noqa: E402
import check_fiorillo_v0_5_figures as fig9_check  # noqa: E402


# ======================================================================================================================
# Part 1 of 11: src/tasks/tcga_reports.py
# From test_tcga_reports.py. Module-level names prefixed tcga; test names test_tcga_*.
#
# Tests for src/tasks/tcga_reports.py (docs/prompt_session9.md, THE NEXT QUESTION'S RECORDS, I-38, and step 4's
# "tcga_reports.py's label folding and splits by patient on made-up rows"), on made-up rows only, under tmp_path:
# - the label folding: each answer set's accepted spellings (case and whitespace ignored) and the values left out;
# - the primary-diagnosis rule: none, one, several agreeing, several disagreeing;
# - the splits by patient: the shares and the rounding rule, the stratification by project, determinism with the seed,
#   no label read, and a patient in the same split in all four questions;
# - the record format, from made-up report and GDC files run through licenses, download, labels, records and counts,
#   with stand-ins for the network and the tokenizer;
# - the label-word counts on made-up reports, and that the test split's labels are only written and counted;
# - the license gate, the resumable download and the resumable fetch of the GDC's pages.
# ======================================================================================================================

TCGA_HERE = Path(__file__).resolve()
TCGA_ROOT = TCGA_HERE.parents[2] if TCGA_HERE.parent.name == "tests" and TCGA_HERE.parents[1].name == "src" else Path.cwd()
TCGA_SRC = TCGA_ROOT / "src"
assert (TCGA_SRC / "tasks" / "tcga_reports.py").exists(), "run from the project folder"

TCGA_SEED = 20260927
TCGA_CC_BY = {"url": "http://creativecommons.org/licenses/by/4.0", "short_name": "CC BY 4.0",
              "full_name": "Creative Commons Attribution 4.0 International", "category": "Creative",
              "description": "You can share, copy and modify this dataset so long as you give appropriate credit."}


class _TcgaResponse:
    """A stand-in for urllib's response: read(n), status, getcode() and the context manager."""

    def __init__(self, body, status=200):
        self.body, self.status, self.pos = body, status, 0

    def read(self, n=-1):
        n = len(self.body) - self.pos if n is None or n < 0 else n
        out = self.body[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _TcgaWordTokenizer:
    """A stand-in for Qwen's tokenizer: one token per whitespace-separated word."""

    def __call__(self, texts, add_special_tokens=True):
        assert add_special_tokens is False
        return {"input_ids": [t.split() for t in texts]}


def _tcga_mendeley(licence):
    return {"id": "hyg5xkznpx", "doi": {"id": "10.17632/hyg5xkznpx.1"}, "version": 1,
            "versions": [{"version": 1, "available": True, "publish_date": "2024-01-03T17:05:58.12Z"}],
            "publish_date": "2024-01-03T17:05:58.12Z", "data_licence": licence,
            "files": [{"filename": "TCGA_Reports.csv.zip", "size": TCGA.ZIP_BYTES,
                       "content_details": {"size": TCGA.ZIP_BYTES, "sha256_hash": TCGA.ZIP_SHA256,
                                           "download_url": TCGA.ZIP_URL}}]}


def _tcga_licenses(out, licence=TCGA_CC_BY):
    return TCGA.licenses(out, fetch=lambda url: json.dumps(_tcga_mendeley(licence)).encode("utf-8"))


def _tcga_diag(cls="primary", **fields):
    d = {"classification_of_tumor": cls}
    d.update(fields)
    return d


def _tcga_case(barcode, project, diagnoses):
    case_id = hashlib.md5(barcode.encode()).hexdigest()
    return {"id": case_id, "case_id": case_id, "submitter_id": barcode, "project": {"project_id": project},
            "diagnoses": diagnoses}


def _tcga_gdc_opener(cases, served, total=None):
    """Serves the made-up cases as the cases endpoint would, sorted by case_id; keeps each body it serves."""
    ordered = sorted(cases, key=lambda c: c["case_id"])

    def opener(req, timeout=None):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(req.full_url).query)
        assert req.full_url.startswith(TCGA.GDC_CASES + "?")
        assert q["sort"] == ["case_id:asc"] and json.loads(q["filters"][0]) == TCGA.GDC_FILTERS
        assert q["fields"][0].split(",") == list(TCGA.GDC_FIELDS) and q["format"] == ["JSON"]
        start, size = int(q["from"][0]), int(q["size"][0])
        hits = ordered[start:start + size]
        body = json.dumps({"data": {"hits": hits, "pagination": {"count": len(hits), "total": total or len(ordered),
                                                                 "size": size, "from": start}},
                           "warnings": {}}).encode("utf-8")
        served[start] = body
        return _TcgaResponse(body)
    return opener


def _tcga_zip(path, rows, header=("patient_filename", "text")):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    w.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("TCGA_Reports.csv", buf.getvalue())
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tcga_world():
    """Made-up cases and report rows: 16 joined patients in two projects, with the special cases the rules name."""
    cases, rows, texts = [], [], {}
    ts, ns, stages, grades = ["T1", "T2a", "T3", "T4b"], ["N0", "N1mi", "N2", "N3a"], ["Stage I", "Stage IIB", "Stage IIIA", "Stage IV"], ["G1", "G2", "G3", "G4"]
    for i in range(1, 11):
        b = f"TCGA-A1-{i:04d}"
        k = i % 4
        d = [_tcga_diag("metastasis", ajcc_pathologic_t="T4"),
             _tcga_diag(ajcc_pathologic_t=ts[k], ajcc_pathologic_n=ns[k], ajcc_pathologic_stage=stages[k],
                        tumor_grade=grades[k], primary_diagnosis="Adenocarcinoma, NOS", laterality="Left")]
        if i == 3:  # two primary diagnoses: T disagrees, N agrees after folding, stage has one value
            d = [_tcga_diag(ajcc_pathologic_t="T2", ajcc_pathologic_n="N1", ajcc_pathologic_stage="Stage II",
                            tumor_grade="G2", primary_diagnosis="Adenocarcinoma, NOS", laterality="Left"),
                 _tcga_diag(ajcc_pathologic_t="T3", ajcc_pathologic_n="N1a", ajcc_pathologic_stage=None,
                            tumor_grade="G2", primary_diagnosis="Adenocarcinoma, NOS", laterality="Right")]
        if i == 5:  # values outside the answer sets
            d[1].update(ajcc_pathologic_t="TX", tumor_grade="High Grade")
        if i == 7:  # no primary diagnosis
            d = [_tcga_diag("metastasis", ajcc_pathologic_t="T2"), _tcga_diag("Synchronous primary", ajcc_pathologic_t="T2")]
        if i == 9:  # no value for grade
            d[1]["tumor_grade"] = None
        cases.append(_tcga_case(b, "TCGA-AAA", d))
        texts[b] = f"Report {b}. Diagnosis: pT{k + 1}N{k} . Stage {TCGA.ROMAN[k + 1]} . Grade {k + 1} ."
    for i in range(1, 7):
        b = f"TCGA-B2-{i:04d}"
        k = (i + 1) % 4
        cases.append(_tcga_case(b, "TCGA-BBB", [_tcga_diag(ajcc_pathologic_t=ts[k], ajcc_pathologic_n=ns[k],
                                                           ajcc_pathologic_stage=stages[k], tumor_grade=grades[k])]))
        texts[b] = f"Report {b} without staging words."
    cases.append(_tcga_case("TCGA-C3-0001", "TCGA-CCC", [_tcga_diag(ajcc_pathologic_t="T1")]))  # a case with no report
    for b, text in texts.items():
        rows.append((f"{b}.{hashlib.md5(b.encode()).hexdigest()}", text))
        if b == "TCGA-A1-0002":
            rows.append((f"{b}.second", "LATER REPORT of the same patient"))
    rows.insert(4, ("notabarcode.pdf", "a report with no barcode"))
    rows.append(("TCGA-ZZ-9999.abc", "a report whose patient has no case"))
    return cases, rows, texts


def _tcga_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_tcga_label_folding():
    for qid, q in TCGA.QUESTIONS.items():
        fold = TCGA.FOLDS[qid]
        assert set(q["spellings"]) == set(q["answers"])
        for answer, spellings in q["spellings"].items():
            for s in spellings:
                assert fold(s) == answer and fold(s.lower()) == answer and fold(s.upper()) == answer
                assert fold("  " + s.replace(" ", "  ") + " ") == answer
    assert TCGA.fold_t("T2a") == "T2" and TCGA.fold_t("T1mi") == "T1" and TCGA.fold_t("T4e") == "T4"
    assert TCGA.fold_n("N1mi") == "N1" and TCGA.fold_n("N1a") == "N1" and TCGA.fold_n("N0 (i+)") == "N0"
    assert TCGA.fold_stage("Stage IIIA") == "III" and TCGA.fold_stage("Stage IS") == "I" and TCGA.fold_stage("Stage IVB") == "IV"
    assert TCGA.fold_grade("G3") == "G3" and TCGA.fold_grade(" g 4") == "G4"
    left_out = {
        "tcga_t": ["Tis", "Tis (DCIS)", "T0", "TX", "Ta", "T5", "T12", "T2/T3", "Not Reported", "Unknown", "", None],
        "tcga_n": ["NX", "N4", "N1/N2", "Not Reported", "Unknown", "", None],
        "tcga_stage": ["Stage 0", "Stage 0a", "Stage 0is", "Stage X", "Stage Tis", "I/II NOS", "Not Reported", "Unknown", None],
        "tcga_grade": ["GX", "GB", "G2-G3", "G5", "Grade 2", "Grade II", "High Grade", "Low Grade", "Intermediate Grade",
                       "Well differentiated", "Not Reported", "Unknown", None],
    }
    for qid, values in left_out.items():
        assert [TCGA.FOLDS[qid](v) for v in values] == [None] * len(values), qid
    with pytest.raises(ValueError, match="listed twice"):
        TCGA.spelling_table({"T1": ("T1",), "T2": ("t 1",)})


def test_tcga_primary_rule():
    pv, f = TCGA.primary_value, TCGA.fold_t
    field = "ajcc_pathologic_t"
    # none: no diagnosis, or none classified exactly "primary"
    assert pv([], field, f) == (None, TCGA.NO_PRIMARY, None)
    assert pv(None, field, f) == (None, TCGA.NO_PRIMARY, None)
    assert pv([_tcga_diag("metastasis", ajcc_pathologic_t="T2"), _tcga_diag("Synchronous primary", ajcc_pathologic_t="T2"),
               _tcga_diag("Primary", ajcc_pathologic_t="T2")], field, f) == (None, TCGA.NO_PRIMARY, None)
    # one
    assert pv([_tcga_diag("recurrence", ajcc_pathologic_t="T1"), _tcga_diag(ajcc_pathologic_t="T3b")], field, f) == ("T3", None, "T3b")
    assert pv([_tcga_diag(ajcc_pathologic_t=None)], field, f) == (None, TCGA.NO_VALUE, None)
    assert pv([_tcga_diag()], field, f) == (None, TCGA.NO_VALUE, None)
    assert pv([_tcga_diag(ajcc_pathologic_t="  ")], field, f) == (None, TCGA.NO_VALUE, None)
    assert pv([_tcga_diag(ajcc_pathologic_t="TX")], field, f) == (None, TCGA.OUTSIDE, "TX")
    # several agreeing: after folding, and with one stating no value
    assert pv([_tcga_diag(ajcc_pathologic_t="T2a"), _tcga_diag(ajcc_pathologic_t="T2b")], field, f) == ("T2", None, "T2a | T2b")
    assert pv([_tcga_diag(ajcc_pathologic_t="T2"), _tcga_diag(ajcc_pathologic_t=None)], field, f) == ("T2", None, "T2")
    assert pv([_tcga_diag(ajcc_pathologic_t="TX"), _tcga_diag(ajcc_pathologic_t="TX")], field, f) == (None, TCGA.OUTSIDE, "TX")
    # several disagreeing
    assert pv([_tcga_diag(ajcc_pathologic_t="T2"), _tcga_diag(ajcc_pathologic_t="T3")], field, f) == (None, TCGA.DISAGREE, "T2 | T3")
    assert pv([_tcga_diag(ajcc_pathologic_t="T2"), _tcga_diag(ajcc_pathologic_t="TX")], field, f) == (None, TCGA.DISAGREE, "T2 | TX")
    # without folding (primary_diagnosis and laterality, counted only)
    assert pv([_tcga_diag(laterality="Left"), _tcga_diag(laterality="Left")], "laterality") == ("Left", None, "Left")
    assert pv([_tcga_diag(laterality="Left"), _tcga_diag(laterality="Right")], "laterality") == (None, TCGA.DISAGREE, "Left | Right")


def test_tcga_splits_by_patient():
    project_of = {f"TCGA-AA-{i:04d}": "TCGA-AAA" for i in range(45)}
    project_of.update({f"TCGA-BB-{i:04d}": "TCGA-BBB" for i in range(10)})
    project_of.update({f"TCGA-CC-{i:04d}": "TCGA-CCC" for i in range(3)})
    split_of = TCGA.split_patients(project_of)
    assert set(split_of) == set(project_of)
    sizes = {p: [sum(1 for b, s in split_of.items() if project_of[b] == p and s == x) for x in TCGA.SPLITS]
             for p in ("TCGA-AAA", "TCGA-BBB", "TCGA-CCC")}
    assert sizes == {"TCGA-AAA": [36, 5, 4], "TCGA-BBB": [8, 1, 1], "TCGA-CCC": [2, 1, 0]}  # (80 n + 50) // 100, (90 n + 50) // 100
    assert [TCGA.cut_points(n) for n in (0, 1, 2, 5, 44, 45, 1000)] == [(0, 0), (1, 1), (2, 2), (4, 5), (35, 40), (36, 41), (800, 900)]
    # determinism: the same seed gives the same splits whatever the input order; another seed differs
    assert TCGA.split_patients(dict(reversed(list(project_of.items())))) == split_of == TCGA.split_patients(project_of, TCGA_SEED)
    assert TCGA.split_patients(project_of, TCGA_SEED + 1) != split_of
    # the rule by hand: projects in sorted order, one generator, patients sorted as text then shuffled
    rng, expected = random.Random(TCGA_SEED), {}
    for p in sorted(set(project_of.values())):
        ids = sorted(b for b in project_of if project_of[b] == p)
        rng.shuffle(ids)
        a, b = (80 * len(ids) + 50) // 100, (90 * len(ids) + 50) // 100
        expected.update({x: "train" for x in ids[:a]} | {x: "validation" for x in ids[a:b]} | {x: "test" for x in ids[b:]})
    assert expected == split_of


def test_tcga_label_word_counts():
    cases = {
        "tcga_t": [("Final: pT2aN1b, margins clear", 1, True, False), ("T12 vertebra, no tumor", 0, False, False),
                   ("ypT3 (m)", 2, True, False), ("pt4 lowercase", 3, True, False),
                   ("Legend: T1, T2, T3, T4", 1, True, True)],
        "tcga_n": [("Final: pT2aN1b", 1, True, False), ("pT2N0M0", 0, True, False), ("N/A", 0, False, False),
                   ("pN3a (12/20)", 3, True, False)],
        "tcga_stage": [("AJCC Stage IIB", 1, True, False), ("Stage IV", 0, False, True), ("the stage is pending", 0, False, False),
                       ("stage: III", 2, True, False), ("AJCC stage 2A", 1, True, False), ("Stage IIIA", 1, False, True)],
        "tcga_grade": [("Grade 3", 2, True, False), ("G2", 1, True, False), ("Gleason grade 3+4", 2, True, False),
                       ("moderately differentiated", 1, False, False), ("Grade II-III", 1, True, False), ("G12", 0, False, False)],
    }
    for qid, rows in cases.items():
        for text, label, own, other in rows:
            answers = TCGA.QUESTIONS[qid]["answers"]
            assert TCGA.label_words_in(qid, answers[label], text) is own, (qid, text)
            assert any(TCGA.label_words_in(qid, a, text) for a in answers if a != answers[label]) is other, (qid, text)
        got = TCGA.label_word_counts(qid, [(t, l) for t, l, _, _ in rows])
        assert got["records"] == len(rows) and got["label_words"] == sum(r[2] for r in rows)
        assert got["other_answer_words"] == sum(r[3] for r in rows) and got["share"] == got["label_words"] / len(rows)
        for a in TCGA.QUESTIONS[qid]["answers"]:
            mine = [r for r in rows if TCGA.QUESTIONS[qid]["answers"][r[1]] == a]
            assert got["by_answer"][a] == {"records": len(mine), "label_words": sum(r[2] for r in mine)}
    assert TCGA.label_word_counts("tcga_t", [])["share"] is None


def test_tcga_report_columns_and_one_report_per_patient(tmp_path):
    assert TCGA.report_columns(["patient_filename", "text"]) == (0, 1)
    with pytest.raises(SystemExit, match="columns are"):
        TCGA.report_columns(["", "file_name", "Text"])  # no column name holds "filename"
    assert TCGA.report_columns(["Unnamed: 0", "Patient_Filename_x", "TEXT"]) == (1, 2)
    path = tmp_path / "reports.zip"
    _tcga_zip(path, [("TCGA-AB-0001.x", "first, with a comma\nand a new line"), ("TCGA-AB-0001.y", "second"),
                     ("bad", "no barcode"), ("TCGA-AB-0002.z", "other")])
    rows, info = TCGA.read_reports(path)
    assert info["file_column"] == "patient_filename" and info["text_column"] == "text" and info["member"] == "TCGA_Reports.csv"
    assert rows[0] == ("TCGA-AB-0001.x", "first, with a comma\nand a new line")
    reports, counts = TCGA.select_reports(rows)
    assert reports == {"TCGA-AB-0001": "first, with a comma\nand a new line", "TCGA-AB-0002": "other"}
    assert counts[TCGA.LATER_REPORTS] == 1 and counts[TCGA.NOT_A_BARCODE] == 1 and counts["patients with a report"] == 2


def test_tcga_license_gate(tmp_path, monkeypatch, capsys):
    ok = _tcga_licenses(tmp_path / "ok")
    assert ok["allowed"] is True and "not_built" not in ok and ok["plan"] == TCGA.PLAN and ok["written_utc"]
    assert ok["tcga_reports"]["data_licence"]["short_name"] == "CC BY 4.0" and ok["tcga_reports"]["file_matches_constants"]
    assert ok["gdc"]["allows_research_use_with_publication"] is True
    assert json.loads((tmp_path / "ok" / "licenses.json").read_text())["allowed"] is True
    v = TCGA.license_verdict
    assert v({"url": "https://creativecommons.org/publicdomain/zero/1.0/", "short_name": "CC0 1.0"})[0] is True
    assert v({"url": "http://creativecommons.org/licenses/by/3.0", "short_name": "CC-BY-3.0"})[0] is True
    assert v({"url": "http://creativecommons.org/licenses/by-sa/4.0", "short_name": "CC BY-SA 4.0"})[0] is False
    assert v({"url": "http://creativecommons.org/licenses/by/4.0", "short_name": "CC BY-NC 4.0"})[0] is False
    assert v({"short_name": "CC BY 4.0", "full_name": "Creative Commons Attribution-NonCommercial 4.0"})[0] is False
    assert v({"full_name": "Some license"})[0] is False
    nc = dict(TCGA_CC_BY, url="http://creativecommons.org/licenses/by-nc/4.0", short_name="CC BY-NC 4.0",
              full_name="Creative Commons Attribution-NonCommercial 4.0 International")
    for name, licence, words in (("nc", nc, "CC BY-NC 4.0"), ("none", None, "states no license"), ("empty", {}, "states no license")):
        out = tmp_path / name
        got = _tcga_licenses(out, licence)
        assert got["allowed"] is False and got["not_built"].startswith("NOT BUILT: I-38's records, ") and words in got["not_built"]
        for call in (lambda: TCGA.download(out, opener=lambda *a, **k: pytest.fail("no download")),
                     lambda: TCGA.labels(out, opener=lambda *a, **k: pytest.fail("no fetch")),
                     lambda: TCGA.records(out), lambda: TCGA.counts(out, results=tmp_path / "res")):
            with pytest.raises(SystemExit, match="NOT BUILT: I-38's records"):
                call()
    monkeypatch.setattr(TCGA, "get_bytes", lambda url, *a, **k: json.dumps(_tcga_mendeley(nc)).encode())
    assert TCGA.main(["licenses", "--out-dir", str(tmp_path / "main")]) == 1
    assert capsys.readouterr().out.strip().startswith("NOT BUILT: I-38's records, Mendeley Data's record states the license CC BY-NC 4.0")
    with pytest.raises(SystemExit, match="run licenses first"):
        TCGA.records(tmp_path / "nothing")
    assert not (tmp_path / "res").exists()


def test_tcga_download_resumes_and_checks(tmp_path):
    data = bytes(range(256)) * 3
    sha = hashlib.sha256(data).hexdigest()
    calls = []

    def opener(req, timeout=None):
        rng = req.get_header("Range")
        calls.append(rng)
        return _TcgaResponse(data[int(rng.split("=")[1].rstrip("-")):], 206) if rng else _TcgaResponse(data, 200)

    out = tmp_path / "tcga"
    _tcga_licenses(out)
    first = TCGA.download(out, max_seconds=-1, url="https://example.invalid/z", sha256=sha, size=len(data), opener=opener, block=100)
    assert first == {"finished": False, "bytes": 100, "of": len(data)} and (out / "parts" / "TCGA_Reports.csv.zip.part").stat().st_size == 100
    second = TCGA.download(out, max_seconds=1000, url="https://example.invalid/z", sha256=sha, size=len(data), opener=opener, block=100)
    assert second["finished"] is True and (out / "raw" / "TCGA_Reports.csv.zip").read_bytes() == data
    assert calls == [None, "bytes=100-"] and not (out / "parts" / "TCGA_Reports.csv.zip.part").exists()
    progress = json.loads((out / "parts" / "download_progress.json").read_text())
    assert progress["complete"] and progress["sha256"] == sha and progress["plan"] == TCGA.PLAN and len(progress["runs"]) == 2
    assert TCGA.download(out, sha256=sha, size=len(data), opener=lambda *a, **k: pytest.fail("checked only"))["finished"]
    # a server that ignores the range: the partial file is started again
    out2 = tmp_path / "tcga2"
    _tcga_licenses(out2)
    TCGA.download(out2, max_seconds=-1, url="https://example.invalid/z", sha256=sha, size=len(data), opener=opener, block=100)
    full = TCGA.download(out2, url="https://example.invalid/z", sha256=sha, size=len(data), opener=lambda req, timeout=None: _TcgaResponse(data, 200))
    assert full["finished"] and (out2 / "raw" / "TCGA_Reports.csv.zip").read_bytes() == data
    # a connection dropped part way: what was written is kept, and the next run resumes from it
    class _Dropped(_TcgaResponse):
        def read(self, n=-1):
            if self.pos >= 200:
                raise ConnectionResetError("made-up drop")
            return super().read(n)

    out4 = tmp_path / "tcga4"
    _tcga_licenses(out4)
    cut = TCGA.download(out4, url="https://example.invalid/z", sha256=sha, size=len(data), block=100,
                        opener=lambda req, timeout=None: _Dropped(data, 200))
    assert cut == {"finished": False, "bytes": 200, "of": len(data)}
    assert "made-up drop" in json.loads((out4 / "parts" / "download_progress.json").read_text())["runs"][0]["error"]
    assert TCGA.download(out4, url="https://example.invalid/z", sha256=sha, size=len(data), opener=opener)["finished"]
    assert (out4 / "raw" / "TCGA_Reports.csv.zip").read_bytes() == data
    # a different SHA-256 fails, and the file got is kept
    out3 = tmp_path / "tcga3"
    _tcga_licenses(out3)
    with pytest.raises(SystemExit, match="SHA-256"):
        TCGA.download(out3, url="https://example.invalid/z", sha256="0" * 64, size=len(data), opener=opener)
    assert (out3 / "parts" / "TCGA_Reports.csv.zip.part").read_bytes() == data and not (out3 / "raw" / "TCGA_Reports.csv.zip").exists()
    (out3 / "raw").mkdir()
    (out3 / "raw" / "TCGA_Reports.csv.zip").write_bytes(b"put there by hand")
    with pytest.raises(SystemExit, match="SHA-256"):
        TCGA.download(out3, url="https://example.invalid/z", sha256=sha, size=len(data), opener=opener)


def test_tcga_labels_paginate_and_resume(tmp_path):
    cases, _, _ = _tcga_world()
    out, served = tmp_path / "tcga", {}
    _tcga_licenses(out)
    first = TCGA.labels(out, max_seconds=-1, page_size=5, opener=_tcga_gdc_opener(cases, served))
    assert first == {"finished": False, "pages_done": 1, "pages": 4, "fetched_now": 1}
    assert not (out / "raw" / "gdc" / "query.json").exists()
    second = TCGA.labels(out, max_seconds=1000, page_size=5, opener=_tcga_gdc_opener(cases, served))
    assert second == {"finished": True, "cases": len(cases), "pages": 4, "fetched_now": 3}
    query = json.loads((out / "raw" / "gdc" / "query.json").read_text())
    assert query["plan"] == TCGA.PLAN and query["total"] == query["cases"] == len(cases) and query["fetched_first_utc"]
    assert query["query"]["fields"] == list(TCGA.GDC_FIELDS) and query["query"]["filters"] == TCGA.GDC_FILTERS
    for page in query["pages"]:  # saved as fetched, byte for byte
        assert (out / "raw" / "gdc" / page["file"]).read_bytes() == served[page["from"]]
    got, _ = TCGA.read_cases(out)
    assert sorted(c["submitter_id"] for c in got) == sorted(c["submitter_id"] for c in cases)
    with pytest.raises(SystemExit, match="another query"):
        TCGA.labels(out, page_size=7, opener=_tcga_gdc_opener(cases, {}))
    out2 = tmp_path / "tcga2"
    _tcga_licenses(out2)
    TCGA.labels(out2, max_seconds=-1, page_size=5, opener=_tcga_gdc_opener(cases, {}))
    with pytest.raises(SystemExit, match="total changed"):
        TCGA.labels(out2, page_size=5, opener=_tcga_gdc_opener(cases, {}, total=len(cases) + 1))


def _tcga_build(tmp_path):
    cases, rows, texts = _tcga_world()
    out = tmp_path / "tcga"
    _tcga_licenses(out)
    TCGA.labels(out, page_size=5, opener=_tcga_gdc_opener(cases, {}))
    zip_sha = _tcga_zip(out / "raw" / "TCGA_Reports.csv.zip", rows)  # as if put in raw/ by hand
    assert TCGA.download(out, sha256=zip_sha, opener=lambda *a, **k: pytest.fail("checked only"))["finished"]
    build = TCGA.records(out, sha256=zip_sha)
    return out, build, cases, rows, texts


def test_tcga_records_format_and_splits(tmp_path):
    out, build, cases, rows, texts = _tcga_build(tmp_path)
    project_of = {c["submitter_id"]: c["project"]["project_id"] for c in cases}
    joined = {b: project_of[b] for b in texts}
    split_of = TCGA.split_patients(joined)
    seen = {}
    for qid in TCGA.QIDS:
        answers = TCGA.QUESTIONS[qid]["answers"]
        for split in TCGA.SPLITS:
            recs = _tcga_read(out / f"{qid}_{split}.jsonl")
            assert [r["id"] for r in recs] == sorted(r["id"] for r in recs)
            for r in recs:
                patient = r["group"]["patient"]
                keys = ["id", "qid", "state", "label", "target"] + (["calibrate"] if split == "validation" else []) + ["group"]
                assert list(r) == keys and r["id"] == f"tcga-{qid}-{patient}" and r["qid"] == qid
                assert r["state"] == {"report": texts[patient]} and 0 <= r["label"] < len(answers)
                assert r["target"] == [1.0 if i == r["label"] else 0.0 for i in range(len(answers))]
                assert r.get("calibrate", True) is True
                assert r["group"] == {"source": "tcga_reports", "patient": patient, "project": project_of[patient]}
                assert split_of[patient] == split
                assert seen.setdefault(patient, split) == split  # the same split in all four questions
    assert "LATER REPORT" not in json.dumps([_tcga_read(out / f"tcga_n_{s}.jsonl") for s in TCGA.SPLITS])
    assert len(seen) == 15 and "TCGA-A1-0007" not in seen  # no primary diagnosis: in no question
    t3 = _tcga_read(out / f"tcga_t_{split_of['TCGA-A1-0004']}.jsonl")
    assert [r["label"] for r in t3 if r["group"]["patient"] == "TCGA-A1-0004"] == [0]  # T1 (i % 4 == 0)
    q = build["questions"]
    assert q["tcga_t"]["drops"] == {TCGA.NO_PRIMARY: 1, TCGA.NO_VALUE: 0, TCGA.DISAGREE: 1, TCGA.OUTSIDE: 1}
    assert q["tcga_t"]["left_out_values"] == {TCGA.OUTSIDE: {"TX": 1}, TCGA.DISAGREE: {"T2 | T3": 1}}
    assert q["tcga_grade"]["drops"] == {TCGA.NO_PRIMARY: 1, TCGA.NO_VALUE: 1, TCGA.DISAGREE: 0, TCGA.OUTSIDE: 1}
    assert q["tcga_grade"]["left_out_values"][TCGA.OUTSIDE] == {"High Grade": 1}
    assert q["tcga_n"]["folded_from"]["N1"]["N1 | N1a"] == 1 and q["tcga_stage"]["folded_from"]["II"]["Stage II"] == 1
    for qid in TCGA.QIDS:  # every joined patient is a record or a counted drop
        assert sum(q[qid]["records"].values()) + sum(q[qid]["drops"].values()) == 16
    assert build["join"] == {"cases fetched from the GDC": 17, "report rows that join no case": 1,
                             "patients whose report joins no case": 1, "patients whose barcode names several cases": 0,
                             "patients whose kept report is empty": 0, "patients joined": 16}
    assert build["reports"][TCGA.LATER_REPORTS] == 1 and build["reports"][TCGA.NOT_A_BARCODE] == 1
    assert build["splits"]["by_project"] == {"TCGA-AAA": {"train": 8, "validation": 1, "test": 1},
                                             "TCGA-BBB": {"train": 5, "validation": 0, "test": 1}}
    assert build["fetched_and_counted_only"]["laterality"]["left_out"][TCGA.DISAGREE] == 1
    assert build["primary_diagnoses_per_joined_case"] == {"0": 1, "1": 14, "2": 1}
    assert build["plan"] == TCGA.PLAN and build["inputs"]["reports"]["file_column"] == "patient_filename"
    # the split reads no label: other labels give the same patients in the same splits
    shifted = json.loads(json.dumps(cases))
    for c in shifted:
        for d in c["diagnoses"]:
            if d.get("ajcc_pathologic_t") == "T1":
                d["ajcc_pathologic_t"] = "T4"
    recs, _ = TCGA.make_records(rows, shifted)
    before, _ = TCGA.make_records(rows, cases)
    for split in TCGA.SPLITS:
        assert [r["group"]["patient"] for r in recs["tcga_t"][split]] == [r["group"]["patient"] for r in before["tcga_t"][split]]


def test_tcga_counts_manifest_and_test_split(tmp_path, monkeypatch):
    out, build, cases, rows, texts = _tcga_build(tmp_path)
    passed = []
    real = TCGA.label_word_counts

    def spy(qid, items):
        passed.extend(text for text, _ in items)
        return real(qid, items)

    monkeypatch.setattr(TCGA, "label_word_counts", spy)
    results, tokdir = tmp_path / "results", tmp_path / "tokenizer"
    tokdir.mkdir()
    (tokdir / "REVISION.txt").write_text(f"revision {TCGA.QWEN_REVISION}\n", encoding="utf-8")
    tok = _TcgaWordTokenizer()
    first = TCGA.counts(out, results, tokenizer_dir=tokdir, max_seconds=-1, tokenizer=tok, batch=4)
    assert first == {"finished": False, "reports": 15, "counted": 4} and not (results / "records_manifest.json").exists()
    done = TCGA.counts(out, results, tokenizer_dir=tokdir, max_seconds=1000, tokenizer=tok, batch=4)
    assert done["finished"] is True
    m = json.loads((results / "records_manifest.json").read_text())
    assert m["plan"] == TCGA.PLAN and m["written_utc"] and m["trained"].startswith("nothing")
    test_texts = set()
    for qid in TCGA.QIDS:
        answers = TCGA.QUESTIONS[qid]["answers"]
        for split in TCGA.SPLITS:
            recs = _tcga_read(out / f"{qid}_{split}.jsonl")
            block = m["counts"][qid][split]
            assert block["records"] == len(recs)  # the test split's labels too: written and counted
            assert block["answers"] == {a: sum(1 for r in recs if answers[r["label"]] == a) for a in answers}
            for project, p in block["projects"].items():
                assert p["records"] == sum(1 for r in recs if r["group"]["project"] == project)
            lengths = [len(r["state"]["report"].split()) for r in recs]
            if lengths:
                assert m["tokens"]["by_question"][qid][split]["largest"] == max(lengths)
                assert m["tokens"]["by_question"][qid][split]["mean"] == sum(lengths) / len(lengths)
            if split == "test":
                test_texts.update(r["state"]["report"] for r in recs)
            else:
                assert m["label_words"][qid][split] == real(qid, [(r["state"]["report"], r["label"]) for r in recs])
        assert sorted(m["label_words"][qid]) == ["train", "validation"]
    assert test_texts and not test_texts & set(passed)  # the label-word counts never saw a test report
    assert m["label_words"]["splits"] == ["train", "validation"] and "test" in m["label_words"]["test"]
    assert m["tokens"]["all"] == TCGA.length_summary(len(texts[p].split()) for p in sorted(texts) if p != "TCGA-A1-0007")
    assert m["tokens"]["all"]["n"] == 15 and m["tokens"]["revision_expected"] == TCGA.QWEN_REVISION
    assert m["tokens"]["revision_matches"] is True
    lines = (out / "parts" / "report_tokens.jsonl").read_text().splitlines()
    assert len(lines) == 15
    sha = m["sha256"]
    assert sha["records"]["tcga_t_train.jsonl"] == hashlib.sha256((out / "tcga_t_train.jsonl").read_bytes()).hexdigest()
    assert len(sha["records"]) == 12 and sha["licenses.json"] and "records_build.json" in sha["parts"]
    assert {"TCGA_Reports.csv.zip", "gdc/query.json", "gdc/cases_from_00000.json"} <= set(sha["raw"])
    assert m["drops"]["tcga_t"][TCGA.DISAGREE] == 1 and m["left_out_values"]["tcga_t"][TCGA.OUTSIDE] == {"TX": 1}
    assert m["patients"]["patients"] == {"train": 13, "validation": 1, "test": 2}
    assert m["questions"]["tcga_stage"]["answers"] == ["I", "II", "III", "IV"]
    assert m["licenses"]["allowed"] is True and m["licenses"]["tcga_reports"] == "CC BY 4.0"
    # a changed record file stops counts
    with open(out / "tcga_t_train.jsonl", "a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(SystemExit, match="differs"):
        TCGA.counts(out, results, tokenizer_dir=tokdir, tokenizer=tok)


# ======================================================================================================================
# Part 2 of 11: src/eval/handoff_curves_session9.py
# From test_handoff_curves.py. Module-level names prefixed hc; test names test_handoff_*.
#
# Tests for src/eval/handoff_curves_session9.py (docs/prompt_session9.md, HAND-OFF ANALYSES, I-34 and I-35; step 4), on
# made-up inputs: the per-item AURC, E-AURC and AUGRC against a hand calculation (with and without ties), the risk-coverage
# curve, the random curve's exact expectation, the oracle's bound, the confidence rule's order against session 8's
# (handoff_session8.py), I-35's rule on a planted signal, the resampling grouped by article, and the commands end to end
# with their guards.
# ======================================================================================================================

HC_SEED = 20260927


def hc_write_probs(path, ids, labels, probs, digits=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{x:.{digits}g}" for x in p])


def hc_write_records(path, ids, labels, articles=None, calibrate=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, (i, y) in enumerate(zip(ids, labels)):
            r = {"id": i, "label": int(y), "state": "made up"}
            if articles is not None:
                r["group"] = {"pmcid": articles[k]}
            if calibrate is not None and calibrate[k] is not None:
                r["calibrate"] = calibrate[k]
            f.write(json.dumps(r) + "\n")


def hc_one_hot_probs(answers, sure=0.8, k=3):
    p = np.full((len(answers), k), (1 - sure) / (k - 1))
    p[np.arange(len(answers)), answers] = sure
    return p


# the areas ----------------------------------------------------------------------------------------------------------------

def test_handoff_areas_against_a_hand_calculation():
    ids = ["e1", "e2", "e3", "e4", "e5"]
    conf = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    wrong = np.array([False, True, False, True, False])
    a = HC.areas(conf, wrong, H8.id_ranks(ids))
    # the most confident first: right, wrong, right, wrong, right; error rates 0, 1/2, 1/3, 2/4, 2/5
    assert a["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 2 / 4 + 2 / 5) / 5)
    # the perfect ranking: three right, then two wrong: 0, 0, 0, 1/4, 2/5
    assert a["aurc_perfect_ranking"] == pytest.approx((1 / 4 + 2 / 5) / 5)
    assert a["e_aurc"] == pytest.approx((1 / 2 + 1 / 3 + 2 / 4 + 2 / 5 - 1 / 4 - 2 / 5) / 5)
    # AUGRC: generalized risks 0, 1/5, 1/5, 2/5, 2/5 at coverages 1/5 to 5/5, closed at (0, 0); the trapezoid rule
    assert a["augrc"] == pytest.approx((0 + 0.1 + 0.2 + 0.3 + 0.4) / 5) == pytest.approx(0.2)
    # Traub et al. 2024, equation 7: acc 0.6; AUROC_f 3 of 6 pairs = 0.5
    closed, auroc_f = HC.augrc_closed_form(conf, wrong)
    assert auroc_f == pytest.approx(0.5) and closed == pytest.approx(0.5 * 0.6 * 0.4 + 0.5 * 0.4 ** 2) == pytest.approx(0.2)
    # the per-item right sum: (0 + 1 + 1 + 2 + 2) / 5 / 5, which is AUGRC + (1 - acc) / (2 n)
    assert HC.augrc_right_sum(conf, wrong, H8.id_ranks(ids)) == pytest.approx(6 / 25) == pytest.approx(0.2 + 0.4 / 10)


def test_handoff_areas_with_ties_against_a_hand_calculation():
    ids = ["a", "b", "c", "d"]
    conf = np.array([0.9, 0.7, 0.7, 0.5])
    wrong = np.array([False, True, False, True])
    a = HC.areas(conf, wrong, H8.id_ranks(ids))
    # session 8 hands off d, then b (the lower ID of the tie), then c, then a; kept, the most confident first: a, c, b, d
    assert a["aurc"] == pytest.approx((0 + 0 + 1 / 3 + 2 / 4) / 4)
    assert a["e_aurc"] == pytest.approx(0.0)
    # AUGRC: one point for the tied pair: (0, 0), (1/4, 0), (3/4, 1/4), (1, 2/4)
    by_hand = 0.25 * 0 + 0.5 * (0 + 0.25) / 2 + 0.25 * (0.25 + 0.5) / 2
    assert a["augrc"] == pytest.approx(by_hand) == pytest.approx(0.15625)
    closed, auroc_f = HC.augrc_closed_form(conf, wrong)  # the tied pair counts half: 3.5 of 4
    assert auroc_f == pytest.approx(0.875) and closed == pytest.approx(by_hand)
    # one answer only, and none right
    assert HC.augrc(np.array([0.6, 0.6, 0.6]), np.array([True, True, True])) == pytest.approx(0.5)
    assert HC.augrc_closed_form(np.array([0.6, 0.7]), np.array([True, True])) == (pytest.approx(0.5), None)
    assert HC.augrc(np.array([0.6, 0.7]), np.array([False, False])) == 0.0


def test_handoff_augrc_matches_equation_7_on_made_up_data():
    rng = np.random.default_rng(HC_SEED)
    for n in (7, 50, 333):
        conf = np.round(rng.uniform(0.34, 1.0, n), 2)  # rounded, so that there are ties
        wrong = rng.uniform(size=n) < 0.3
        wrong[0], wrong[1] = True, False
        closed, _ = HC.augrc_closed_form(conf, wrong)
        assert HC.augrc(conf, wrong) == pytest.approx(closed, abs=1e-12)


# the risk-coverage curve --------------------------------------------------------------------------------------------------

def test_handoff_risk_coverage_curve():
    rng = np.random.default_rng(HC_SEED)
    n = 23
    ids = [f"q{k:02d}" for k in rng.permutation(n)]
    conf = np.round(rng.uniform(0.4, 1.0, n), 1)  # ties
    wrong = rng.uniform(size=n) < 0.35
    rank = H8.id_ranks(ids)
    kept, wrong_kept, risk = HC.risk_coverage(conf, wrong, rank)
    assert kept.tolist() == [n - int(np.floor((100 - c) / 100 * n + 1e-9)) for c in HC.COVERAGES]
    assert kept[0] == 2 and kept[9] == 12 and kept[-1] == n  # 5 percent of 23: 23 - floor(21.85); 50 percent: 23 - 11
    labels = np.zeros(n, dtype=int)
    fio = np.column_stack([conf, (1 - conf) / 2, (1 - conf) / 2])  # its top probability is conf
    for c, k, w, r in zip(HC.COVERAGES, kept, wrong_kept, risk):
        # the kept answers are exactly those session 8's rule keeps at hand-off rate 100 - c
        _, handed = H8.combine(rank, labels, fio, fio, (100 - c) / 100)
        assert (~handed).sum() == k and int(wrong[~handed].sum()) == w and r == pytest.approx(w / k)
    assert risk[-1] == pytest.approx(wrong.mean())


# the deferral curves ------------------------------------------------------------------------------------------------------

def hc_made_up_pair(n, seed=HC_SEED, ties=False):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    fio = rng.dirichlet([1.0, 1.0, 1.0], n)
    big = rng.dirichlet([1.0, 1.0, 1.0], n)
    if ties:
        fio = np.round(fio, 1)
        fio = fio / fio.sum(1, keepdims=True)
        big = np.round(big, 1)
        big = big / big.sum(1, keepdims=True)
    ids = [f"id{k:03d}" for k in rng.permutation(n)]
    return ids, labels, fio, big


def test_handoff_random_curve_is_the_exact_expectation():
    ids, labels, fio, big = hc_made_up_pair(6)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    for h in range(7):
        accs = []
        for subset in itertools.combinations(range(6), h):
            handed = np.zeros(6, dtype=bool)
            handed[list(subset)] = True
            accs.append(np.where(handed, right_l, right_f).mean())
        assert HC.random_expected(right_f, right_l, [h])[0] == pytest.approx(np.mean(accs), abs=1e-12)
    counts, curves, _ = HC.statistics(labels, fio, big, H8.id_ranks(ids), HC.RATES)
    assert counts.tolist() == [r * 6 // 100 for r in HC.RATES]
    assert curves["random_expected"][0] == pytest.approx(right_f.mean()) and curves["random_expected"][-1] == pytest.approx(right_l.mean())


def test_handoff_the_oracle_bounds_every_rule():
    for seed in range(5):
        ids, labels, fio, big = hc_made_up_pair(8, seed=HC_SEED + seed, ties=seed % 2 == 1)
        right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
        rank = H8.id_ranks(ids)
        order = HC.oracle_order(right_f, right_l, fio.max(1), rank)
        gain = (right_l.astype(int) - right_f.astype(int))[order]
        assert list(gain) == sorted(gain, reverse=True)  # +1 first, then 0, then -1
        oracle = HC.accuracy_after(order, right_f, right_l, range(9))
        for h in range(9):
            best = max(np.where(np.isin(np.arange(8), s), right_l, right_f).mean() for s in itertools.combinations(range(8), h))
            assert oracle[h] == pytest.approx(best)
        _, curves, _ = HC.statistics(labels, fio, big, rank, HC.RATES)
        for name in ("confidence", "random_expected", "large_model_confidence"):
            assert np.all(curves[name] <= curves["oracle"] + 1e-12)
        rng = np.random.default_rng(seed)
        for _ in range(50):
            assert np.all(HC.accuracy_after(rng.permutation(8), right_f, right_l, range(9)) <= oracle + 1e-12)


def test_handoff_confidence_rule_matches_session_8():
    for n in list(range(1, 60)) + [123, 1237, 1218, 150, 500]:
        for r in HC.RATES:
            assert HC.handed(r, n) == int(np.floor(r / 100 * n + 1e-9))
    ids, labels, fio, big = hc_made_up_pair(40, ties=True)
    assert len(np.unique(fio.max(1))) < 40  # ties in confidence
    rank = H8.id_ranks(ids)
    order = HC.handoff_order(fio.max(1), rank)
    _, curves, _ = HC.statistics(labels, fio, big, rank, HC.RATES)
    for j, r in enumerate(HC.RATES):
        probs, handed = H8.combine(rank, labels, fio, big, r / 100)
        assert set(np.flatnonzero(handed)) == set(order[: HC.handed(r, 40)])
        assert curves["confidence"][j] == pytest.approx(float(np.mean(probs.argmax(1) == labels)), abs=1e-15)
    # the intervals: the same resamples and the same ranking inside each (items, with ties)
    mine = HC.analyze_pair(ids, labels, fio, big, None, resamples=60)
    theirs = H8.analyze(ids, labels, fio, big, None, resamples=60)
    for r in HC.REGISTERED:
        a, b = mine["registered_rates"][f"rate_{r}"], theirs[f"rate_{r}"]
        assert a["handed_off"] == b["handed_off"]
        for k in ("estimate", "low", "high"):
            assert a["confidence"][k] == pytest.approx(b["accuracy"][k], abs=1e-15)
    # articles, without ties
    ids, labels, fio, big = hc_made_up_pair(45, seed=HC_SEED + 7)
    clusters = [f"PMC{k % 11}" for k in range(45)]
    mine = HC.analyze_pair(ids, labels, fio, big, clusters, resamples=60)
    theirs = H8.analyze(ids, labels, fio, big, clusters, resamples=60)
    for r in HC.REGISTERED:
        for k in ("estimate", "low", "high"):
            assert mine["registered_rates"][f"rate_{r}"]["confidence"][k] == pytest.approx(theirs[f"rate_{r}"]["accuracy"][k], abs=1e-15)


def test_handoff_large_model_confidence_order():
    ids = ["b", "a", "c", "d"]
    big = np.array([[0.7, 0.2, 0.1], [0.1, 0.7, 0.2], [0.9, 0.05, 0.05], [0.4, 0.3, 0.3]])
    order = HC.highest_first(big.max(1), H8.id_ranks(ids))
    assert [ids[k] for k in order] == ["c", "a", "b", "d"]  # 0.9, then the tie at 0.7 by ID (a before b), then 0.4


# the resampling ---------------------------------------------------------------------------------------------------------

def test_handoff_resampling_grouped_by_article():
    clusters = ["PMC2", "PMC10", "PMC2", "PMC3", "PMC10", "PMC2"]
    draws = list(H8.resample_indices(6, clusters, 5, HC_SEED))
    rng = np.random.default_rng(HC_SEED)
    rows = {0: [1, 4], 1: [0, 2, 5], 2: [3]}  # PMC10, PMC2, PMC3: sorted as text
    for got in draws:
        assert got.tolist() == sum((rows[int(j)] for j in rng.integers(0, 3, 3)), [])
    # analyze_pair's intervals come from those draws, with the items ranked again inside each
    ids, labels, fio, big = hc_made_up_pair(30, seed=HC_SEED + 3)
    clusters = [f"PMC{k % 7}" for k in range(30)]
    out = HC.analyze_pair(ids, labels, fio, big, clusters, resamples=40)
    rank = H8.id_ranks(ids)
    aurc, oracle20 = [], []
    for idx in H8.resample_indices(30, clusters, 40, HC_SEED):
        conf, wrong = fio[idx].max(1), fio[idx].argmax(1) != labels[idx]
        key = [(conf[k], rank[idx][k], k) for k in range(len(idx))]  # least confident first, then the ID, then the place
        kept = [k for _, _, k in sorted(key)][::-1]
        aurc.append(np.mean(np.cumsum(wrong[kept]) / np.arange(1, len(idx) + 1)))
        right_f, right_l = ~wrong, big[idx].argmax(1) == labels[idx]
        h = 20 * len(idx) // 100
        up, down = int((~right_f & right_l).sum()), int((right_f & ~right_l).sum())
        oracle20.append((right_f.sum() + min(h, up) - max(0, h - (len(idx) - down))) / len(idx))
    assert out["areas"]["aurc"]["low"] == pytest.approx(np.percentile(aurc, 2.5))
    assert out["areas"]["aurc"]["high"] == pytest.approx(np.percentile(aurc, 97.5))
    assert out["registered_rates"]["rate_20"]["oracle"]["low"] == pytest.approx(np.percentile(oracle20, 2.5))
    assert out["registered_rates"]["rate_20"]["oracle"]["high"] == pytest.approx(np.percentile(oracle20, 97.5))
    assert out["articles"] == 7 and out["resampling"].startswith("articles")


# I-35 ---------------------------------------------------------------------------------------------------------------------

def hc_planted(n=600, articles=60, seed=HC_SEED):
    """Fiorillo always answers 0 with a confidence that carries no signal; the spread of its three runs (u) decides
    whether it is wrong where the large model is right."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.5, 0.95, n)
    fio = np.column_stack([t, (1 - t) * 0.6, (1 - t) * 0.4])
    u = rng.uniform(0, 1, n)
    d = 0.02 * u
    runs = [fio + np.column_stack([-d, d, 0 * d]), fio.copy(), fio + np.column_stack([d, -d, 0 * d])]
    target = rng.uniform(size=n) < 1 / (1 + np.exp(-(-7 + 9 * u)))
    labels = np.where(target, 1, 0)
    big = np.where(target[:, None], [[0.1, 0.8, 0.1]], np.where(rng.uniform(size=(n, 1)) < 0.5, [[0.8, 0.1, 0.1]], [[0.1, 0.8, 0.1]]))
    ids = [f"evidence-{k}" for k in range(n)]
    clusters = [f"PMC{k % articles}" for k in range(n)]
    return ids, labels, fio, runs, big, clusters, target, u


def hc_write_planted(folder, split="validation", **kwargs):
    ids, labels, fio, runs, big, clusters, target, u = hc_planted(**kwargs)
    paths = {"fiorillo": folder / f"fio_evidence_effect_{split}.csv", "large": folder / f"big_evidence_effect_{split}.csv",
             "records": folder / f"evidence_effect_{split}.jsonl",
             "runs": [folder / f"fio_{k}-raw_evidence_effect_{split}.csv" for k in (1, 2, 3)]}
    hc_write_probs(paths["fiorillo"], ids, labels, fio)
    hc_write_probs(paths["large"], ids, labels, big)
    for p, r in zip(paths["runs"], runs):
        hc_write_probs(p, ids, labels, r)
    hc_write_records(paths["records"], ids, labels, clusters, [True] * len(ids))
    return paths, target


def test_handoff_rule_features():
    cal = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2], [1.0, 0.0, 0.0]])
    runs = [np.array([[0.6, 0.3, 0.1], [0.5, 0.3, 0.2], [0.9, 0.1, 0.0]]),
            np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.8, 0.1, 0.1]]),
            np.array([[0.8, 0.1, 0.1], [0.2, 0.6, 0.2], [1.0, 0.0, 0.0]])]
    x, answer = HC.rule_features(cal, runs)
    assert answer.tolist() == [0, 1, 0]  # the means: 0.7; 0.3 against 0.5; 0.9
    assert x[:, 0].tolist() == pytest.approx([0.7, 0.4, 1.0]) and x[:, 1].tolist() == pytest.approx([0.5, 0.0, 1.0])
    assert x[0, 2] == pytest.approx(-(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1))) and x[2, 2] == 0.0
    assert x[:, 3].tolist() == pytest.approx([np.std([0.6, 0.7, 0.8], ddof=1), np.std([0.3, 0.6, 0.6], ddof=1),
                                              np.std([0.9, 0.8, 1.0], ddof=1)])


def test_handoff_rule_recovers_a_planted_signal(tmp_path):
    from sklearn.linear_model import LogisticRegression

    paths, target = hc_write_planted(tmp_path)
    rule = HC.fit_rule("fio", paths["fiorillo"], paths["runs"], "big", paths["large"], paths["records"])
    coef = rule["coefficients"]
    assert rule["positives"] == int(target.sum()) and rule["model"]["at_l2_optimum"]
    assert coef["run_sd"] > 1.0 and coef["run_sd"] > 3 * max(abs(coef[k]) for k in ("top_probability", "margin", "entropy"))
    # the frozen rule, as JSON, gives scikit-learn's probabilities
    rule = json.loads(json.dumps(rule))
    ids, labels, fio, big, clusters = HC.load_pair(HC.EI, paths["fiorillo"], paths["large"], paths["records"], "validation")
    runs = HC.load_runs(paths["runs"], ids, labels)
    x, _ = HC.rule_features(fio, runs)
    z = (x - x.mean(0)) / x.std(0)
    clf = LogisticRegression(C=1.0, tol=1e-10, max_iter=10000).fit(z, target.astype(int))
    assert HC.rule_probability(rule, x) == pytest.approx(clf.predict_proba(z)[:, 1], abs=1e-6)
    # the rule's 10 percent hand-off holds more of the planted cases than the confidence rule's
    out = HC.compare_rule(rule, ids, labels, fio, runs, big, clusters, "validation", resamples=30)
    r10 = out["registered_rates"]["rate_10"]
    assert r10["handed_off"] == 60 and out["in_sample"]
    assert r10["rule_hand_off"]["fiorillo_wrong_large_model_right"] > r10["confidence_hand_off"]["fiorillo_wrong_large_model_right"] + 15
    assert r10["rule_minus_confidence"]["estimate"] > 0 and r10["rule_minus_confidence"]["reading"] == "higher (better)"


def test_handoff_rule_comparison_is_paired_over_articles(tmp_path):
    ids, labels, fio, runs, big, clusters, target, u = hc_planted(n=90, articles=13)
    rule = {"system": "fio", "large_model": "big", "room": {"statement": "made up"}, "intercept": -1.0,
            "features": [{"name": k} for k in HC.FEATURES], "coefficients": {"top_probability": -0.5, "margin": 0.2, "entropy": 0.4, "run_sd": 1.0},
            "scaling": {"mean": {k: 0.0 for k in HC.FEATURES}, "scale": {k: 1.0 for k in HC.FEATURES}}}
    out = HC.compare_rule(rule, ids, labels, fio, runs, big, clusters, "test", resamples=25)
    assert not out["in_sample"] and out["room_on_validation"] == "made up"
    x, _ = HC.rule_features(fio, runs)
    prob = HC.sigmoid(-1.0 + x @ np.array([-0.5, 0.2, 0.4, 1.0]))
    rank = H8.id_ranks(ids)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    diffs = []
    for idx in H8.resample_indices(90, clusters, 25, HC_SEED):
        m = len(idx)
        key = rank[idx] * m + np.arange(m)
        h = 30 * m // 100
        a = sorted(range(m), key=lambda k: (-prob[idx][k], key[k]))[:h]
        b = sorted(range(m), key=lambda k: (fio[idx].max(1)[k], key[k]))[:h]
        acc = lambda handed: np.where(np.isin(np.arange(m), handed), right_l[idx], right_f[idx]).mean()  # noqa: E731
        diffs.append(acc(a) - acc(b))
    d = out["registered_rates"]["rate_30"]["rule_minus_confidence"]
    assert d["low"] == pytest.approx(np.percentile(diffs, 2.5)) and d["high"] == pytest.approx(np.percentile(diffs, 97.5))
    with pytest.raises(SystemExit, match="EI only"):
        HC.compare_rule(rule, ids, labels, fio, runs, big, None, "test", resamples=5)
    bad = {**rule, "features": [{"name": k} for k in reversed(HC.FEATURES)]}
    with pytest.raises(SystemExit, match="features"):
        HC.compare_rule(bad, ids, labels, fio, runs, big, clusters, "test", resamples=5)


# the commands ------------------------------------------------------------------------------------------------------------

def test_handoff_commands_end_to_end(tmp_path, capsys):
    folder = tmp_path / "files"
    paths, _ = hc_write_planted(folder, n=120, articles=17)
    # PubMedQA: 40 items marked calibrate (no key) and 10 not, resampled by item
    rng = np.random.default_rng(HC_SEED)
    qids = [f"pubmedqa-{k}" for k in range(50)]
    qlabels = rng.integers(0, 3, 50)
    hc_write_records(folder / "pubmedqa_answer_validation.jsonl", qids, qlabels, None, [None] * 40 + [False] * 10)
    hc_write_probs(folder / "s7_pubmedqa_answer_validation.csv", qids, qlabels, rng.dirichlet([1, 1, 1], 50))
    hc_write_probs(folder / "g_pubmedqa_answer_validation.csv", qids[:40], qlabels[:40], rng.dirichlet([1, 1, 1], 40))
    rule_path = tmp_path / "out" / "handoff_rule.json"
    assert HC.main(["fit-rule", "--fiorillo-name", "fio", "--fiorillo", str(paths["fiorillo"]), "--runs", *map(str, paths["runs"]),
                    "--large-name", "big", "--large", str(paths["large"]), "--records", str(paths["records"]), "--out", str(rule_path)]) == 0
    text = rule_path.read_text()
    assert text.startswith('{\n "plan": "docs/prompt_session9.md, HAND-OFF ANALYSES (I-35')
    rule = json.loads(text)
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rule["written_utc"]) and rule["n"] == 120 and rule["articles"] == 17
    assert "room of at most" in rule["room"]["statement"] and rule["files"]["runs"][2]["path"].endswith("fio_3-raw_evidence_effect_validation.csv")
    out_path = tmp_path / "out" / "curves.json"
    pairs = ["--pair", "evidence_effect", "fio", str(paths["fiorillo"]), "big", str(paths["large"]), str(paths["records"]),
             "--pair", "pubmedqa_answer", "s7", str(folder / "s7_pubmedqa_answer_validation.csv"), "g",
             str(folder / "g_pubmedqa_answer_validation.csv"), str(folder / "pubmedqa_answer_validation.jsonl")]
    assert HC.main(["curves", "--split", "validation", *pairs, "--rule", "fio", str(rule_path), *map(str, paths["runs"]),
                    "--resamples", "20", "--out", str(out_path)]) == 0
    out = json.loads(out_path.read_text())
    assert out["plan"].startswith("docs/prompt_session9.md, HAND-OFF ANALYSES") and out["resamples"] == 20 and out["written_utc"]
    ei, pqa = out["pairs"]
    assert ei["n"] == 120 and ei["articles"] == 17 and pqa["n"] == 40 and pqa["articles"] is None and pqa["resampling"] == "items"
    assert len(ei["deferral"]["confidence"]) == 21 and len(ei["risk_coverage"]["risk"]) == 20 and "i35_rule" in ei and "i35_rule" not in pqa
    assert set(ei["registered_rates"]) == {"rate_10", "rate_20", "rate_30", "rate_50"}
    for k in ("aurc", "e_aurc", "augrc"):
        assert ei["areas"][k]["low"] <= ei["areas"][k]["estimate"] <= ei["areas"][k]["high"]
    assert abs(ei["augrc_check"]["difference"]) < 1e-12
    assert ei["i35_rule"]["in_sample"] and ei["i35_rule"]["rule_file"]["sha256"] == HC.C8.sha256_file(rule_path)
    # apply-rule: the same comparison on its own
    apply_path = tmp_path / "out" / "apply.json"
    assert HC.main(["apply-rule", "--split", "validation", "--rule", str(rule_path), "--fiorillo", str(paths["fiorillo"]),
                    "--runs", *map(str, paths["runs"]), "--large", str(paths["large"]), "--records", str(paths["records"]),
                    "--resamples", "20", "--out", str(apply_path)]) == 0
    applied = json.loads(apply_path.read_text())
    assert applied["registered_rates"] == ei["i35_rule"]["registered_rates"]
    # the guards: test needs the freeze's two files; a file of the other split is refused; a rule of another system
    test_paths, _ = hc_write_planted(tmp_path / "test_files", split="test", n=120, articles=17)
    args = ["apply-rule", "--split", "test", "--rule", str(rule_path), "--fiorillo", str(test_paths["fiorillo"]), "--runs",
            *map(str, test_paths["runs"]), "--large", str(test_paths["large"]), "--records", str(test_paths["records"]),
            "--resamples", "10", "--guard-dir", str(tmp_path / "guard"), "--out", str(tmp_path / "out" / "apply_test.json")]
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        HC.main(args)
    (tmp_path / "guard").mkdir()
    (tmp_path / "guard" / "release_candidate.json").write_text("{}")
    with pytest.raises(SystemExit, match="testpreds_log.json does not exist"):
        HC.main(args)
    (tmp_path / "guard" / "testpreds_log.json").write_text("{}")
    assert HC.main(args) == 0 and not json.loads((tmp_path / "out" / "apply_test.json").read_text())["in_sample"]
    with pytest.raises(SystemExit, match="name the other split"):
        HC.main(["fit-rule", "--fiorillo-name", "fio", "--fiorillo", str(test_paths["fiorillo"]), "--runs", *map(str, paths["runs"]),
                 "--large-name", "big", "--large", str(paths["large"]), "--records", str(paths["records"]), "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit, match="is the rule of fio"):
        HC.main(["curves", "--split", "validation", *pairs, "--rule", "other", str(rule_path), *map(str, paths["runs"]),
                 "--resamples", "5", "--out", str(tmp_path / "y.json")])
    with pytest.raises(SystemExit, match="found 0"):
        HC.main(["curves", "--split", "validation", *pairs[7:], "--rule", "fio", str(rule_path), *map(str, paths["runs"]),
                 "--resamples", "5", "--out", str(tmp_path / "z.json")])
    assert not (tmp_path / "x.json").exists() and not (tmp_path / "y.json").exists() and not (tmp_path / "z.json").exists()
    capsys.readouterr()


def test_handoff_labels_must_agree(tmp_path):
    ids = ["a", "b", "c"]
    hc_write_records(tmp_path / "evidence_effect_validation.jsonl", ids, [0, 1, 2], ["PMC1", "PMC1", "PMC2"])
    hc_write_probs(tmp_path / "f_evidence_effect_validation.csv", ids, [0, 1, 2], hc_one_hot_probs([0, 1, 2]))
    hc_write_probs(tmp_path / "g_evidence_effect_validation.csv", ids, [0, 1, 1], hc_one_hot_probs([0, 1, 1]))
    with pytest.raises(SystemExit, match="labels differ"):
        HC.load_pair(HC.EI, tmp_path / "f_evidence_effect_validation.csv", tmp_path / "g_evidence_effect_validation.csv",
                     tmp_path / "evidence_effect_validation.jsonl", "validation")
    hc_write_records(tmp_path / "evidence_effect_validation.jsonl", ids, [0, 1, 2], None)
    hc_write_probs(tmp_path / "g_evidence_effect_validation.csv", ids, [0, 1, 2], hc_one_hot_probs([0, 1, 1]))
    with pytest.raises(SystemExit, match="no group.pmcid"):
        HC.load_pair(HC.EI, tmp_path / "f_evidence_effect_validation.csv", tmp_path / "g_evidence_effect_validation.csv",
                     tmp_path / "evidence_effect_validation.jsonl", "validation")


# ======================================================================================================================
# Part 3 of 11: src/eval/trees_session9.py
# From test_trees_session9.py. Module-level names prefixed trees; test names test_trees_*.
#
# Tests for src/eval/trees_session9.py (docs/prompt_session9.md, THE READMISSION POOL, VERSION 2, and step 4), on made-up
# inputs:
# - the recoded diagnosis fields: three-character codes, chapters and indicators; the out-of-fold target encoding without
#   leakage (an item's own label never enters its own encoding); recode's tables rebuilding the frames exactly (the
#   release's path);
# - lightgbm_tuned's refit check against a made-up file (a pass and a fail), and the lightgbm stage end to end with its
#   test split;
# - xgboost_codes and catboost_codes refitted with the recoded fields, and their test split after the freeze;
# - the pool over eight members recovering a planted member; the batched grid on torch against session 8's fit (log loss
#   within 1e-6); the grid sizes 19,448, 8,008 and 3,003; the 0.002 rule;
# - TabICL's stand-in (a small class with fit and predict_proba) through the same interface, with its saved state loaded
#   against a moved checkpoint;
# - the module importing, and its TabICL path running, with xgboost, catboost and lightgbm unavailable;
# - the pool end to end (members dropped, the choice, resuming), and the test subcommand refusing a member file whose
#   SHA-256 differs from the freeze's record.
# ======================================================================================================================

TREES_HERE = Path(__file__).resolve()
TREES_SRC = TREES_HERE.parents[1] if (TREES_HERE.parents[1] / "eval" / "trees_session9.py").exists() else Path.home() / "mnt" / "ed-decision-model" / "src"

TREES_SEED = 20260927
TREES_CKPT_FILE = "tabicl-classifier-v2-20260212.ckpt"
TREES_CKPT_REVISION = "4dcd344ece2c00be9e831fdd35bed57b5ad83e19"
TREES_TABICL_DEFAULTS = {"n_estimators": 8, "norm_methods": None, "feat_shuffle_method": "latin", "class_shuffle_method": "shift",
                   "outlier_threshold": 4.0, "softmax_temperature": 0.9, "average_logits": True, "support_many_classes": True,
                   "batch_size": 8, "kv_cache": False, "use_amp": "auto", "use_fa3": "auto", "offload_mode": "auto", "random_state": 42}
TREES_DIAGS = ["250.83 (Endocrine)", "250 (Endocrine)", "428 (Circulatory)", "38 (Infectious)", "8 (Infectious)", "V57 (Supplementary)",
         "E888 (External causes)", "414 (Circulatory)", "786 (Symptoms)", "996 (Injury)"]


def trees_write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def trees_write_probs(path, ids, labels, probs, digits=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{x:.{digits}g}" for x in p])


def trees_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trees_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def trees_change_one_digit(path):
    text = path.read_text()
    path.write_text(text[:-2] + ("1" if text[-2] != "1" else "2") + text[-1])


def trees_uci_rows(split, n, seed):
    from baselines import UCI_CATEGORICAL, UCI_NUMERIC
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        state = {c: int(rng.integers(0, 20)) for c in UCI_NUMERIC}
        state["time_in_hospital"], state["number_inpatient"] = int(rng.integers(1, 14)), int(rng.integers(0, 4))
        for c in UCI_CATEGORICAL:
            if not c.startswith("diag_"):
                state[c] = str(rng.choice([f"{c} a", f"{c} b"]))
        if rng.random() < 0.2:
            del state["medical_specialty"]  # a missing categorical value
        for f in T9.FIELDS:
            if rng.random() < 0.9:
                state[f] = str(rng.choice(TREES_DIAGS + ([] if split == "train" else ["999 (unseen)", "V99 (unseen)"])))
        state["medications"] = {"insulin": str(rng.choice(["No", "Steady", "Up"]))}
        score = (0.45 * state["number_inpatient"] + 0.08 * state["time_in_hospital"] + 0.4 * state.get("diag_1", "").startswith("428")
                 + rng.normal(0.0, 0.8))
        label = 2 if score > 1.6 else (1 if score > 0.9 else 0)
        rows.append({"id": f"uci-{split}-{i}", "qid": "readmit_window", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": state})
    return rows


def trees_records(folder):
    records = Path(folder) / "records"
    for split, n, seed in (("train", 200, 1), ("validation", 80, 2), ("test", 80, 3)):
        trees_write_jsonl(records / f"readmit_window_{split}.jsonl", trees_uci_rows(split, n, seed))
    return records


def trees_freeze(results, sha256):
    (results / "release_candidate.json").write_text(json.dumps({"sha256": sha256, "time_utc": "2026-10-01T00:00:00Z"}))
    (results / "testpreds_log.json").write_text(json.dumps({"release_candidate_sha256": trees_sha(results / "release_candidate.json")}))


# the recoded fields ---------------------------------------------------------------------------------------------------

def test_trees_codes_chapters_and_indicators():
    cases = {"38": "038", "8": "008", "250.83": "250", "250.4": "250", "V57": "V57", "V5": "V05", "V45.81": "V45", "E888": "E888",
             "E9": "E009", "E885.9": "E885", "001": "001", "999": "999", None: "missing"}
    for code, want in cases.items():
        assert T9.three_char(code) == want, code
    assert T9.diag_code("250.83 (Endocrine, nutritional and metabolic diseases)") == "250.83"
    assert T9.diag_code(None) is None and T9.diag_code("?") is None and T9.diag_code(float("nan")) is None
    for name, low, high in T9.CHAPTER_RANGES:
        assert T9.chapter(f"{low:03d}") == name == T9.chapter(f"{high:03d}")
    assert [T9.chapter(c) for c in ("V57", "E888", "missing")] == ["V", "E", "missing"]
    assert len(T9.CHAPTERS) == 20 == len(set(T9.CHAPTERS)) and len(T9.CHAPTER_RANGES) == 17
    with pytest.raises(ValueError):
        T9.chapter("000")
    rows = [{"state": {"diag_1": "250.83 (x)", "diag_3": "V57 (y)"}},
            {"state": {"diag_1": "38 (z)", "diag_2": "41 (z)", "diag_3": "E888 (w)"}},
            {"state": {}}]
    prior = [0.5, 0.3, 0.2]
    tables = {"code3_categories": {f: ["038", "041", "250", "E888", "V57", "missing"] for f in T9.FIELDS},
              "target_encoding": {"fields": {f: {"prior": prior, "shares": {"250": [0.1, 0.2, 0.7]}} for f in T9.FIELDS}}}
    cols = T9.recoded_columns(rows, tables)
    assert list(cols.columns) == T9.added_columns() and cols.shape == (3, 35)
    assert [cols[T9.code3_column(f)][0] for f in T9.FIELDS] == ["250", "missing", "V57"]
    assert [cols[T9.chapter_column(f)][0] for f in T9.FIELDS] == ["240-279", "missing", "V"]
    assert list(cols[T9.chapter_column("diag_1")].cat.categories) == list(T9.CHAPTERS)
    indicators = cols[[T9.any_column(c) for c in T9.CHAPTERS]]
    on = {r: {c for c in T9.CHAPTERS if indicators[T9.any_column(c)][r] == 1} for r in range(3)}
    assert on == {0: {"240-279", "missing", "V"}, 1: {"001-139", "E"}, 2: {"missing"}}
    te = cols[[T9.te_column("diag_1", a) for a in range(3)]].to_numpy()
    assert np.allclose(te[0], [0.1, 0.2, 0.7]) and np.allclose(te[1], prior) and np.allclose(te[2], prior)


def test_trees_out_of_fold_target_encoding_has_no_leakage():
    rng = np.random.default_rng(3)
    n = 300
    codes = list(rng.choice(["250", "428", "038", "V57", "missing", "E888"], size=n))
    y = rng.integers(0, 3, n)
    fold = T9.te_folds(y)
    assert sorted(set(fold.tolist())) == [0, 1, 2, 3, 4] and np.array_equal(fold, T9.te_folds(y))
    for label in range(3):
        per_fold = np.bincount(fold[y == label], minlength=5)
        assert per_fold.max() - per_fold.min() <= 1  # stratified
    enc = T9.oof_encoding(codes, y, fold)
    arr = np.array(codes)
    for i in range(n):  # each record from the other folds only: their counts and their prior
        other = fold != fold[i]
        prior = np.bincount(y[other], minlength=3) / other.sum()
        same = other & (arr == codes[i])
        want = (np.bincount(y[same], minlength=3) + 20 * prior) / (same.sum() + 20)
        assert np.allclose(enc[i], want, rtol=0, atol=1e-12)
    for i in (0, 17, 123, 299):  # an item's own label never enters its own encoding
        y2 = y.copy()
        y2[i] = (y[i] + 1) % 3
        enc2 = T9.oof_encoding(codes, y2, fold)
        assert np.array_equal(enc2[i], enc[i])
        reached = (fold != fold[i]) & (arr == codes[i])
        assert reached.any() and not np.allclose(enc2[reached], enc[reached])  # it does reach the other folds' records
    table = T9.te_table(codes, y)
    prior = np.bincount(y, minlength=3) / n
    for c in ("250", "missing"):
        mask = arr == c
        assert np.allclose(table["shares"][c], (np.bincount(y[mask], minlength=3) + 20 * prior) / (mask.sum() + 20))
    assert np.allclose(T9.te_lookup(["never seen"], table)[0], prior) and np.allclose(table["prior"], prior)


def test_trees_recode_tables_rebuild_the_frames(tmp_path):
    records = trees_records(tmp_path)
    results, models = tmp_path / "results" / "fiorillo_v0_5", tmp_path / "models"
    T9.recode(results, records, models)
    tables = json.loads((models / "recode_tables.json").read_text())
    assert tables["plan"] == T9.PLAN and "written_utc" in tables and tables["training_records"]["rows"] == 200
    data, built, plain = T9.build_frames(records, ("train", "validation", "test"))
    assert T9.same_tables(tables, built)
    for split in ("validation", "test"):  # the release's path: the saved tables alone give the same frames
        rows = data[split][2]
        assert T9.same_frame(T9.frame_from_tables(rows, tables), data[split][0]) == (True, None)
        assert T9.same_frame(T9.frame_from_tables(rows, tables, codes=False), plain[split][0]) == (True, None)
    x = data["train"][0]
    assert x.shape[1] == plain["train"][0].shape[1] + 35 and list(x.columns[-35:]) == T9.added_columns()
    xv, rows_v = data["validation"][0], data["validation"][2]
    unseen = [k for k, r in enumerate(rows_v) if r["state"].get("diag_1", "").startswith("999")]
    assert unseen and xv[T9.code3_column("diag_1")].iloc[unseen].isna().all()  # a code not seen in training is missing
    prior = tables["target_encoding"]["fields"]["diag_1"]["prior"]
    assert np.allclose(xv[[T9.te_column("diag_1", a) for a in range(3)]].iloc[unseen].to_numpy(), prior)
    trees = json.loads((results / "trees_v2.json").read_text())
    assert trees["plan"] == T9.PLAN and trees["recode"]["columns"]["added"] == 35 and trees["recode"]["tables_sha256"] == trees_sha(models / "recode_tables.json")
    assert json.loads((results / "trees_v2_progress.json").read_text())["recode"]["done"]
    T9.recode(results, records, models)  # a rerun on the same records keeps the tables
    rows = [json.loads(line) for line in open(records / "readmit_window_train.jsonl")]
    rows[0]["label"] = (rows[0]["label"] + 1) % 3
    trees_write_jsonl(records / "readmit_window_train.jsonl", rows)
    with pytest.raises(SystemExit, match="written once"):
        T9.recode(results, records, models)


# lightgbm_tuned's refit -------------------------------------------------------------------------------------------------

def test_trees_lightgbm_refit_check_against_a_made_up_file(tmp_path):
    rng = np.random.default_rng(4)
    n = 50
    ids, y, p = [f"uci-{i}" for i in range(n)], rng.integers(0, 3, n), rng.dirichlet([2, 2, 2], size=n)
    path = tmp_path / "lightgbm_tuned_readmit_window_validation.csv"
    q = p.copy()
    q[3] += [0.0, 0.0009, -0.0009]
    trees_write_probs(path, ids[::-1], y[::-1], q[::-1])  # another order
    check = T9.refit_check(ids, y, p, path)
    assert check["passed"] and 0.0008 < check["max_abs_difference"] <= 0.001 and "stand for" in check["consequence"]
    q[7] += [0.0011, -0.0011, 0.0]
    trees_write_probs(path, ids, y, q)
    check = T9.refit_check(ids, y, p, path)
    assert not check["passed"] and check["max_abs_difference"] > 0.001 and "left out" in check["consequence"]
    trees_write_probs(path, ids[:-1], y[:-1], q[:-1])
    with pytest.raises(SystemExit, match="different records"):
        T9.refit_check(ids, y, p, path)
    y2 = y.copy()
    y2[0] = (y2[0] + 1) % 3
    trees_write_probs(path, ids, y2, q)
    with pytest.raises(SystemExit, match="labels differ"):
        T9.refit_check(ids, y, p, path)


TREES_S7_SETTING = {"index": 0, "num_leaves": 15, "min_child_samples": 10, "learning_rate": 0.1, "feature_fraction": 1.0,
              "bagging_fraction": 1.0, "lambda_l2": 0, "n_estimators": 12}


def test_trees_lightgbm_stage_pass_and_fail_and_its_test_split(tmp_path):
    pytest.importorskip("lightgbm")
    records = trees_records(tmp_path)
    data = T7.frames(records, ("train", "validation"))
    xv, yv, rows = data["validation"]
    refit = np.asarray(T7.refit(TREES_S7_SETTING, data).predict_proba(xv))
    for shift, passed in ((0.0004, True), (0.002, False)):
        base = tmp_path / str(passed) / "results"
        results, models = base / "fiorillo_v0_5", tmp_path / str(passed) / "models"
        (base / "fiorillo_v0_3").mkdir(parents=True)
        (base / "fiorillo_v0_3" / "trees.json").write_text(json.dumps({"chosen": TREES_S7_SETTING}))
        s7_file = base / "fiorillo_v0_3" / "predictions" / "lightgbm_tuned_readmit_window_validation.csv"
        made_up = refit.copy()
        made_up[5] += [shift, -shift, 0.0]
        trees_write_probs(s7_file, [r["id"] for r in rows], yv, made_up)
        (base / "fiorillo_v0_4").mkdir(parents=True)
        (base / "fiorillo_v0_4" / "readmission_pool.json").write_text(json.dumps({"member_files": {"lightgbm_tuned": {"sha256": "0" * 64}}}))
        with pytest.raises(SystemExit, match="not the file session 8's pool read"):
            T9.lightgbm_refit(results, records, models)
        (base / "fiorillo_v0_4" / "readmission_pool.json").write_text(json.dumps({"member_files": {"lightgbm_tuned": {"sha256": trees_sha(s7_file)}}}))
        T9.recode(results, records, models)
        entry = T9.lightgbm_refit(results, records, models)
        assert entry["check"]["passed"] is passed and entry["saved_model_reproduces_validation_max_abs_difference"] <= 1e-8
        assert abs(entry["check"]["max_abs_difference"] - shift) < 1e-6 and (models / "lightgbm_tuned.txt").exists()
        assert json.loads((results / "trees_v2.json").read_text())["members"]["lightgbm_tuned"]["check"]["passed"] is passed
        val_file = T9.new_file("lightgbm_tuned", "validation", results)
        with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
            T9.trees_test(results, records, models, ("lightgbm_tuned",))
        trees_freeze(results, {T9.rel(val_file): trees_sha(val_file)})
        log = T9.trees_test(results, records, models, ("lightgbm_tuned",))
        test_file = T9.new_file("lightgbm_tuned", "test", results)
        if passed:
            assert log["lightgbm_tuned"]["validation_reproduced_max_abs_difference"] <= 1e-8 and log["lightgbm_tuned"]["test_rows"] == 80
            assert len(T7.M_read(records / "readmit_window_test.jsonl")) == len(open(test_file).read().splitlines()) - 1
        else:
            assert not test_file.exists() and log["lightgbm_tuned"]["written"] is False
        assert "lightgbm_tuned" in json.loads((results / "testpreds_log.json").read_text())["trees_v2"]


# xgboost_codes and catboost_codes ---------------------------------------------------------------------------------------

TREES_S8_SETTINGS = {"xgboost_tuned": {"index": 0, "max_depth": 3, "min_child_weight": 1, "eta": 0.1, "subsample": 1.0, "colsample_bytree": 1.0,
                                 "lambda": 1, "n_estimators": 15},
               "catboost_tuned": {"index": 0, "depth": 4, "learning_rate": 0.1, "l2_leaf_reg": 3, "bagging_temperature": 1,
                                  "border_count": 32, "n_estimators": 15}}


def test_trees_refit_codes_members_and_their_test_split(tmp_path, capsys):
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    records = trees_records(tmp_path)
    results, models = tmp_path / "results" / "fiorillo_v0_5", tmp_path / "models"
    (results.parent / "fiorillo_v0_4").mkdir(parents=True)
    (results.parent / "fiorillo_v0_4" / "trees.json").write_text(json.dumps({"members": {m: {"chosen": s} for m, s in TREES_S8_SETTINGS.items()}}))
    with pytest.raises(SystemExit, match="run recode first"):
        T9.refit(results, records, models, threads=1)
    T9.recode(results, records, models)
    trees = T9.refit(results, records, models, threads=1)
    for m in T9.CODES_BASE:
        e = trees["members"][m]
        assert e["trees"] == 15 and e["columns"]["added"] == 35 and e["validation"]["rows"] == 80 and e["setting"]["n_estimators"] == 15
        assert e["saved_model_reproduces_validation_max_abs_difference"] <= 1e-8 and (models / T9.MODEL_FILES[m]).exists()
    capsys.readouterr()
    T9.refit(results, records, models, threads=1)  # resumes: both are skipped
    assert capsys.readouterr().out.count("skipped") == 2
    files = {m: T9.new_file(m, "validation", results) for m in T9.CODES_BASE}
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        T9.trees_test(results, records, models, tuple(T9.CODES_BASE), threads=1)
    trees_freeze(results, {T9.rel(f): trees_sha(f) for f in files.values()})
    log = T9.trees_test(results, records, models, tuple(T9.CODES_BASE), threads=1)
    assert all(v["validation_reproduced_max_abs_difference"] <= 1e-8 and v["test_rows"] == 80 for v in log.values())
    assert set(json.loads((results / "testpreds_log.json").read_text())["trees_v2"]) == set(T9.CODES_BASE)
    trees_change_one_digit(files["xgboost_codes"])
    with pytest.raises(SystemExit, match="changed after the freeze"):
        T9.trees_test(results, records, models, ("xgboost_codes",), threads=1)


# the pool ---------------------------------------------------------------------------------------------------------------

def test_trees_pool_over_eight_members_recovers_a_planted_member():
    rng = np.random.default_rng(5)
    n = 1200
    p_true = trees_softmax(rng.normal(0, 1.5, (n, 3)))
    y = np.array([rng.choice(3, p=p) for p in p_true])
    logs = [np.log(rng.dirichlet([0.5] * 3, size=n)) for _ in range(7)]
    logs.insert(5, np.log(p_true))
    kept = T9.fit_pool_torch(np.stack(logs), y, device="cpu", chunk=4096)
    assert kept["weights"] == [0.0] * 5 + [1.0, 0.0, 0.0] and abs(kept["T"] - 1.0) < 0.15


def test_trees_batched_grid_on_torch_against_session8_fit():
    rng = np.random.default_rng(11)
    n = 400
    p_true = trees_softmax(rng.normal(0, 1.2, (n, 3)))
    y = np.array([rng.choice(3, p=p) for p in p_true])
    logs = np.stack([np.log(trees_softmax(np.log(p_true) * a + rng.normal(0, s, (n, 3))))
                     for a, s in ((0.7, 0.8), (1.3, 0.6), (1.0, 1.0), (0.5, 0.3), (1.6, 1.2))])
    s8 = T8.fit_pool(logs, y)
    s9 = T9.fit_pool_torch(logs, y, device="cpu", chunk=37)  # batches that do not divide the grid
    assert s9["grid_index"] == s8["grid_index"] and s9["weights"] == s8["weights"]
    assert abs(s9["validation_log_loss"] - s8["validation_log_loss"]) < 1e-6 and abs(s9["T"] - s8["T"]) < 1e-6
    grid = T8.weight_grid(5)
    loss8, beta8 = T8._fit_chunk(logs, y, grid)
    loss9, beta9 = T9.grid_losses_torch(logs, y, grid, chunk=100)
    assert np.max(np.abs(loss8 - loss9)) < 1e-6 and np.max(np.abs(beta8 - beta9)) < 1e-6  # at every grid point


def test_trees_grid_sizes():
    for m in range(1, 6):
        assert np.array_equal(T9.weight_grid(m), T8.weight_grid(m))  # session 8's grid and order
    for m, size in ((8, 19448), (7, 8008), (6, 3003), (5, 1001)):
        grid = T9.weight_grid(m)
        assert len(grid) == size == T9.grid_size(m) and np.allclose(grid.sum(axis=1), 1.0) and len({tuple(g) for g in grid}) == size
        assert np.all(np.isclose(grid * 10, np.round(grid * 10)))


def test_trees_the_0002_rule():
    assert T9.choose(0.8350, 0.8371) == ("new", pytest.approx(0.0021))
    assert T9.choose(0.8352, 0.8371)[0] == "session8"  # lower by 0.0019 only
    assert T9.choose(1.0, 1.002)[0] == "new"  # lower by 0.002: at least the margin
    assert T9.choose(0.84, 0.8371)[0] == "session8" and T9.choose(0.8371, 0.8371)[0] == "session8"
    assert T9.MARGIN == 0.002


# TabICL -----------------------------------------------------------------------------------------------------------------

class TreesStandInTabICL:
    """A stand-in for tabicl.TabICLClassifier (tabicl is not installed on the linked shell): TabICL's parameters with
    their defaults; a checkpoint read from model_path when fitted and again when unpickled, as TabICL rebuilds its model
    in __setstate__; and a nearest class profile on the numeric columns as its answer."""
    seen = {}
    _params = ("n_estimators", "norm_methods", "feat_shuffle_method", "class_shuffle_method", "outlier_threshold", "softmax_temperature",
               "average_logits", "support_many_classes", "batch_size", "kv_cache", "model_path", "allow_auto_download",
               "checkpoint_version", "device", "use_amp", "use_fa3", "offload_mode", "disk_offload_dir", "random_state", "n_jobs",
               "verbose", "inference_config")

    def __init__(self, n_estimators=8, norm_methods=None, feat_shuffle_method="latin", class_shuffle_method="shift", outlier_threshold=4.0,
                 softmax_temperature=0.9, average_logits=True, support_many_classes=True, batch_size=8, kv_cache=False, model_path=None,
                 allow_auto_download=True, checkpoint_version=TREES_CKPT_FILE, device=None, use_amp="auto", use_fa3="auto",
                 offload_mode="auto", disk_offload_dir=None, random_state=42, n_jobs=None, verbose=False, inference_config=None):
        values = dict(locals())
        for k in self._params:
            setattr(self, k, values[k])

    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in self._params}

    def _load_model(self):
        path = Path(self.model_path) if self.model_path else None
        if path is None or not path.exists():
            raise ValueError("Checkpoint not found and automatic download is disabled")
        self.model_ = path.read_bytes()

    def fit(self, X, y):
        import pandas as pd
        cats = [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]
        self.cols_ = [c for c in X.columns if c not in cats]
        type(self).seen = {"frame": isinstance(X, pd.DataFrame), "categorical": cats, "nan_in_categorical": bool(X[cats].isna().to_numpy().any()),
                           "numeric_float": all(str(X[c].dtype) == "float64" for c in self.cols_), "rows": len(X), "params": self.get_params()}
        self._load_model()
        self.classes_ = np.unique(y)
        num = X[self.cols_].to_numpy(dtype=float)
        self.mean_ = np.nanmean(num, axis=0)
        num = np.where(np.isnan(num), self.mean_, num)
        self.centers_ = np.stack([num[np.asarray(y) == c].mean(axis=0) for c in self.classes_])
        return self

    def predict_proba(self, X):
        num = X[self.cols_].to_numpy(dtype=float)
        num = np.where(np.isnan(num), self.mean_, num)
        d = ((num[:, None, :] - self.centers_[None]) ** 2).sum(-1)
        return trees_softmax(-d / d.mean())

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._load_model()

    def save(self, path, save_model_weights=False, save_training_data=True, save_kv_cache=True):
        type(self).saved_with = {"save_model_weights": save_model_weights, "save_training_data": save_training_data, "save_kv_cache": save_kv_cache}
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=5)


def trees_fake_tabicl(monkeypatch, version="2.2.0"):
    module = types.ModuleType("tabicl")
    cls = type("TabICLClassifier", (TreesStandInTabICL,), {"__module__": "tabicl"})
    module.TabICLClassifier, module.__version__ = cls, version
    monkeypatch.setitem(sys.modules, "tabicl", module)
    return cls


def trees_fake_hub(monkeypatch, path):
    calls = []
    hub = types.ModuleType("huggingface_hub")

    def hf_hub_download(repo_id, filename, revision=None, cache_dir=None, **kwargs):
        calls.append({"repo_id": repo_id, "filename": filename, "revision": revision})
        return str(path)

    hub.hf_hub_download = hf_hub_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    return calls


def trees_tabicl_setup(tmp_path, monkeypatch):
    records = trees_records(tmp_path)
    results, models = tmp_path / "results" / "fiorillo_v0_5", tmp_path / "models"
    ckpt = tmp_path / "hub" / TREES_CKPT_FILE
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"made-up TabICL weights")
    results.mkdir(parents=True)
    spec = {"version": "2.2.0", "defaults": dict(TREES_TABICL_DEFAULTS),
            "checkpoint": {"repo": "jingang/TabICL", "revision": TREES_CKPT_REVISION, "file": TREES_CKPT_FILE, "sha256": trees_sha(ckpt)}}
    (results / "fixed_versions.json").write_text(json.dumps({"tabicl": spec}))
    return records, results, models, ckpt, spec, trees_fake_tabicl(monkeypatch), trees_fake_hub(monkeypatch, ckpt)


def test_trees_tabicl_stand_in_through_the_same_interface(tmp_path, monkeypatch):
    records, results, models, ckpt, spec, cls, calls = trees_tabicl_setup(tmp_path, monkeypatch)
    record = T9.tabicl_run(results, records, models, device="cpu")
    assert calls == [{"repo_id": "jingang/TabICL", "filename": TREES_CKPT_FILE, "revision": TREES_CKPT_REVISION}]
    seen = cls.seen
    assert seen["frame"] and seen["nan_in_categorical"] and seen["numeric_float"] and seen["rows"] == 200
    assert {"medical_specialty", "diag_1", "med_insulin"} <= set(seen["categorical"])
    params = seen["params"]
    assert params["random_state"] == TREES_SEED and params["model_path"] == str(ckpt) and params["allow_auto_download"] is False
    assert params["device"] == "cpu" and all(params[k] == v for k, v in TREES_TABICL_DEFAULTS.items() if k != "random_state")
    assert cls.saved_with == {"save_model_weights": False, "save_training_data": True, "save_kv_cache": True}
    assert record["plan"] == T9.PLAN and record["context"]["records"] == 200 and record["validation"]["rows"] == 80
    ids, _, probs = T9.M.read_predictions(T9.new_file("tabicl", "validation", results))
    assert len(ids) == 80 and np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    state = json.loads((models / "tabicl_state.json").read_text())
    assert state["checkpoint"] == {**spec["checkpoint"], "download": state["checkpoint"]["download"]} and state["sha256"] == trees_sha(models / "tabicl.pkl")
    T9.tabicl_run(results, records, models, device="cpu")  # resumes: nothing is fitted again
    assert len(calls) == 1
    moved = tmp_path / "elsewhere" / TREES_CKPT_FILE
    moved.parent.mkdir()
    ckpt.rename(moved)
    with pytest.raises(ValueError, match="Checkpoint not found"):
        pickle.loads((models / "tabicl.pkl").read_bytes())  # as saved, it looks for the checkpoint where it was fitted
    est = T9.load_tabicl(models / "tabicl.pkl", moved, "cpu")
    assert type(est) is cls and est.model_path == str(moved) and est.allow_auto_download is False and est.device == "cpu"
    assert np.allclose(est.predict_proba(T7.frames(records, ("train", "validation"))["validation"][0]), probs, atol=1e-8)
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        T9.tabicl_run(results, records, models, split="test", device="cpu", checkpoint=moved)
    val_file = T9.new_file("tabicl", "validation", results)
    trees_freeze(results, {T9.rel(val_file): trees_sha(val_file)})
    entry = T9.tabicl_run(results, records, models, split="test", device="cpu", checkpoint=moved)
    assert entry["test_rows"] == 80 and entry["validation_reproduced_max_abs_difference"] <= 1e-8
    assert json.loads((results / "testpreds_log.json").read_text())["tabicl"]["test_rows"] == 80
    trees_change_one_digit(val_file)
    with pytest.raises(SystemExit, match="changed after the freeze"):
        T9.tabicl_run(results, records, models, split="test", device="cpu", checkpoint=moved)
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(b"other weights")
    with pytest.raises(SystemExit, match="SHA-256"):
        T9.fetch_checkpoint(spec, bad)
    wrong = copy.deepcopy(spec)
    wrong["defaults"]["n_estimators"] = 4
    with pytest.raises(SystemExit, match="defaults differ"):
        T9.tabicl_estimator(wrong, moved, "cpu")
    trees_fake_tabicl(monkeypatch, version="2.1.0")
    with pytest.raises(SystemExit, match="fixed version is 2.2.0"):
        T9.tabicl_estimator(spec, moved, "cpu")


def test_trees_module_imports_and_runs_tabicl_without_xgboost_catboost_lightgbm(tmp_path, monkeypatch):
    code = ("import sys\n"
            "for n in ('xgboost', 'catboost', 'lightgbm'):\n    sys.modules[n] = None\n"
            f"sys.path.insert(0, {str(TREES_SRC / 'eval')!r})\n"
            "import trees_session9\n"
            "print('imported', 'torch' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "imported False"  # torch is imported only by the functions that need it
    for name in ("xgboost", "catboost", "lightgbm"):
        monkeypatch.setitem(sys.modules, name, None)  # any import of them now fails
    records, results, models, ckpt, spec, cls, calls = trees_tabicl_setup(tmp_path, monkeypatch)
    record = T9.tabicl_run(results, records, models, device="cpu")
    assert record["validation"]["rows"] == 80 and calls
    with pytest.raises(ImportError):
        import xgboost  # noqa: F401


# the pool end to end, and the test subcommand ---------------------------------------------------------------------------

def trees_pool_layout(tmp_path, n=150, seed=8):
    base = tmp_path / "results"
    results = base / "fiorillo_v0_5"
    rng = np.random.default_rng(seed)
    p_true = trees_softmax(rng.normal(0, 1.5, (n, 3)))
    y = np.array([rng.choice(3, p=p) for p in p_true])
    ids = [f"uci-{i}" for i in range(n)]

    def noisy():
        return trees_softmax(np.log(p_true) + rng.normal(0, 1.0, (n, 3)))

    places = {"lightgbm_tuned": "fiorillo_v0_3", "xgboost_tuned": "fiorillo_v0_4", "catboost_tuned": "fiorillo_v0_4", "arm_laya": "public_v0",
              "s7_b3": "fiorillo_v0_3"}
    s8_files = {}
    for split in ("validation", "test"):
        for m, folder in places.items():
            path = base / folder / "predictions" / f"{m}_readmit_window_{split}.csv"
            trees_write_probs(path, ids, y, noisy())
            s8_files.setdefault(m, path)
        for m in ("xgboost_codes", "catboost_codes"):
            trees_write_probs(T9.new_file(m, split, results), ids, y, p_true if m == "xgboost_codes" else noisy())
    s8 = {"members": list(T9.S8_MEMBERS), "weights": {"lightgbm_tuned": 0.0, "xgboost_tuned": 0.4, "catboost_tuned": 0.6, "arm_laya": 0.0, "s7_b3": 0.0},
          "T": 1.0, "member_files": {m: {"file": str(p), "sha256": trees_sha(p)} for m, p in s8_files.items()},
          "validation_log_loss": 0.9, "out_of_fold_log_loss": {"mean": 0.91}}
    (base / "fiorillo_v0_4" / "readmission_pool.json").write_text(json.dumps(s8))
    trees = {"plan": T9.PLAN, "members": {m: {"predictions_sha256": trees_sha(T9.new_file(m, "validation", results))} for m in ("xgboost_codes", "catboost_codes")}}
    trees["members"]["lightgbm_tuned"] = {"check": {"passed": False, "max_abs_difference": 0.004}}
    (results / "trees_v2.json").write_text(json.dumps(trees))
    return results


def test_trees_pool_choice_resume_and_test_refusing_a_changed_member(tmp_path, monkeypatch):
    results = trees_pool_layout(tmp_path)
    markers = tmp_path / "not_built"
    markers.mkdir()
    (markers / "tabicl.txt").write_text("NOT BUILT: tabicl, a made-up reason\n")
    out = T9.pool(results, device="cpu", threads=1, repeats=2, not_built_dir=markers)
    new = out["pools"]["new"]
    assert new["members"] == ["xgboost_tuned", "catboost_tuned", "arm_laya", "s7_b3", "xgboost_codes", "catboost_codes"]
    assert new["grid_points"] == 3003 and out["pools"]["session8"]["grid_points"] == 1001
    dropped = {d["member"]: d for d in new["dropped"]}
    assert dropped["tabicl"] == {"member": "tabicl", "status": "NOT BUILT", "reason": "NOT BUILT: tabicl, a made-up reason"}
    assert dropped["lightgbm_tuned"]["status"] == "left out" and "0.004" in dropped["lightgbm_tuned"]["reason"]
    assert max(new["weights"], key=new["weights"].get) == "xgboost_codes" and new["weights"]["xgboost_codes"] >= 0.7
    assert out["chosen"] == "new" and out["choice"]["session8_minus_new"] >= 0.002 and out["members"] == new["members"]
    assert out["weights"] == new["weights"] and out["T"] == new["T"]
    assert out["nonzero_weight_members"] == [m for m in new["members"] if new["weights"][m] > 0]
    assert out["pools"]["session8"]["out_of_fold_log_loss"]["repeats"] == 2 and out["plan"] == T9.PLAN and "written_utc" in out
    pred = results / "predictions" / "readmission_pool_v2_readmit_window_validation.csv"
    assert pred.exists() and out["predictions_sha256"] == trees_sha(pred)
    calls = []
    real = T9.fit_pool_torch
    monkeypatch.setattr(T9, "fit_pool_torch", lambda *a, **k: calls.append(1) or real(*a, **k))
    again = T9.pool(results, device="cpu", threads=1, repeats=2, not_built_dir=markers)
    assert len(calls) == 2  # the fold fits were saved; only the two whole-validation fits run again
    assert again["pools"]["new"]["out_of_fold_log_loss"] == new["out_of_fold_log_loss"] and again["chosen"] == "new"
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        T9.pool_test(results)
    frozen = T9.frozen_inputs(results)
    assert T9.rel(results / "readmission_pool_v2.json") in frozen and len(frozen) == 1 + 7  # the two pools share four files
    trees_freeze(results, frozen)
    got = T9.pool_test(results)
    assert got["rows"] == 150 and got["chosen"] == "new" and got["members"] == new["members"]
    test_ids, _, test_probs = T9.M.read_predictions(results / "predictions" / "readmission_pool_v2_readmit_window_test.csv")
    _, _, logs = T9.read_aligned([(m, T9.twin_file(T9.resolve(v["file"], results))) for m, v in new["member_files"].items()])
    assert np.allclose(test_probs, T8.pool_probs(logs, [new["weights"][m] for m in new["members"]], new["T"]), atol=1e-8)
    listed = json.loads((results / "trees_v2_test.json").read_text())["files"]
    assert len(listed) == 7 and all(v["rows"] == 150 for v in listed.values())
    test_file = T9.new_file("xgboost_codes", "test", results)
    log = json.loads((results / "testpreds_log.json").read_text())
    log["trees_v2"] = {"xgboost_codes": {"test_predictions": T9.rel(test_file), "test_predictions_sha256": trees_sha(test_file)}}
    (results / "testpreds_log.json").write_text(json.dumps(log))
    T9.pool_test(results)  # the test file is the one the pod recorded
    trees_change_one_digit(test_file)
    with pytest.raises(SystemExit, match="differs from the test predictions that testpreds_log.json recorded"):
        T9.pool_test(results)
    trees_change_one_digit(results / "predictions" / "catboost_codes_readmit_window_validation.csv")
    with pytest.raises(SystemExit, match="changed after the freeze"):
        T9.pool_test(results)


def test_trees_command_line_end_to_end(tmp_path):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    records = trees_records(tmp_path)
    base = tmp_path / "results"
    results, models = base / "fiorillo_v0_5", tmp_path / "models"
    common = ["--results", str(results), "--records-dir", str(records), "--models-dir", str(models)]
    (base / "fiorillo_v0_4").mkdir(parents=True)
    (base / "fiorillo_v0_4" / "trees.json").write_text(json.dumps({"members": {m: {"chosen": s} for m, s in TREES_S8_SETTINGS.items()}}))
    (base / "fiorillo_v0_3").mkdir(parents=True)
    (base / "fiorillo_v0_3" / "trees.json").write_text(json.dumps({"chosen": TREES_S7_SETTING}))
    data = T7.frames(records, ("train", "validation"))
    rows_v, y_v = data["validation"][2], data["validation"][1]
    rows_t = T7.M_read(records / "readmit_window_test.jsonl")
    rng = np.random.default_rng(12)
    places = {"lightgbm_tuned": "fiorillo_v0_3", "xgboost_tuned": "fiorillo_v0_4", "catboost_tuned": "fiorillo_v0_4", "arm_laya": "public_v0",
              "s7_b3": "fiorillo_v0_3"}
    files = {}
    for m, folder in places.items():
        for split, rows in (("validation", rows_v), ("test", rows_t)):
            path = base / folder / "predictions" / f"{m}_readmit_window_{split}.csv"
            if (m, split) == ("lightgbm_tuned", "validation"):
                probs = np.asarray(T7.refit(TREES_S7_SETTING, data).predict_proba(data[split][0]))
            elif m == "xgboost_tuned":  # strong in both pools, so that the new pool cannot be lower by 0.002
                probs = np.full((len(rows), 3), 0.1) + 0.7 * np.eye(3)[[r["label"] for r in rows]]
            else:
                probs = rng.dirichlet([2, 2, 2], size=len(rows))
            trees_write_probs(path, [r["id"] for r in rows], [r["label"] for r in rows], probs)
            files[(m, split)] = path
    s8 = {"members": list(T9.S8_MEMBERS), "weights": {"lightgbm_tuned": 0.0, "xgboost_tuned": 0.4, "catboost_tuned": 0.6, "arm_laya": 0.0, "s7_b3": 0.0},
          "T": 1.0, "member_files": {m: {"file": str(files[(m, "validation")]), "sha256": trees_sha(files[(m, "validation")])} for m in places},
          "validation_log_loss": 1.0, "out_of_fold_log_loss": {"mean": 1.0}}
    (base / "fiorillo_v0_4" / "readmission_pool.json").write_text(json.dumps(s8))
    assert T9.main(["recode"] + common) == 0
    assert T9.main(["refit", "--members", "xgboost_codes", "--threads", "1"] + common) == 0
    assert T9.main(["lightgbm"] + common) == 0
    assert json.loads((results / "trees_v2.json").read_text())["members"]["lightgbm_tuned"]["check"]["passed"]
    assert T9.main(["pool", "--device", "cpu", "--repeats", "2", "--threads", "1"] + common) == 0
    kept = json.loads((results / "readmission_pool_v2.json").read_text())
    assert kept["pools"]["new"]["members"] == ["lightgbm_tuned", "xgboost_tuned", "catboost_tuned", "arm_laya", "s7_b3", "xgboost_codes"]
    assert kept["pools"]["new"]["grid_points"] == 3003 and {d["member"] for d in kept["pools"]["new"]["dropped"]} == {"catboost_codes", "tabicl"}
    assert kept["chosen"] in ("new", "session8") and kept["members"] == kept["pools"][kept["chosen"]]["members"]
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        T9.main(["refit", "--split", "test", "--members", "xgboost_codes"] + common)
    trees_freeze(results, T9.frozen_inputs(results))
    assert T9.main(["refit", "--split", "test", "--members", "xgboost_codes", "--threads", "1"] + common) == 0
    assert T9.main(["lightgbm", "--split", "test"] + common) == 0
    assert T9.main(["test"] + common) == 0
    got = json.loads((results / "readmission_pool_v2_test.json").read_text())
    assert got["rows"] == 80 and got["plan"] == T9.PLAN_TEST and got["members"] == kept["members"]
    listed = json.loads((results / "trees_v2_test.json").read_text())["files"]
    assert all(v["rows"] == 80 for v in listed.values()) and len(listed) == 7
    assert kept["chosen"] == "session8" and got["weights"] == s8["weights"] and got["T"] == s8["T"]  # session 8's pool is kept as it was


# ======================================================================================================================
# Part 4 of 11: src/eval/biogpt_session9.py
# From test_biogpt_session9.py. Module-level names prefixed biogpt; test names test_biogpt_*.
#
# Tests for src/eval/biogpt_session9.py (docs/prompt_session9.md, PUBMEDQA BESIDE BIOGPT-LARGE, correction 5 and step
# 4), on made-up inputs; no weights are downloaded and no network is used.
#
# - BioGPT's input: the source as rebuild_data.py writes it; the learned prompt's tokens in place and in order (after the
#   source, before the opening words); the source cut at max_source_positions with '...'; the leading </s> and no eos
#   after the source; prefix True's order; a manual prompt's words;
# - the answers' probabilities from made-up next-token logits: one-token answers, a made-up multi-token answer (the
#   first-token rule recorded), and the renormalization over the three;
# - next-token logits from right-padded batches against each item alone, on a tiny BioGptForCausalLM;
# - correction 5's reader on made-up archives with the real layout (a gzipped tar holding a torch checkpoint whose
#   data.pkl carries a cfg dict): the zip and legacy formats, GNU and pax long names, a directory and an AppleDouble
#   member passed over, omegaconf-like containers and a legacy argparse.Namespace, a value that differs from
#   fixed_versions.json's and changes the input, a network failure retried, and the fallback when the stream fails,
#   hits the read limit or the deadline;
# - the archive's address, which the module no longer holds (ARCHIVE_URL is None): archive_url_from_readme on made-up
#   README text, with a made-up fixed_versions.json, finds the address with its signature, and refuses a README whose
#   SHA-256 differs from the one fixed_versions.json records and a README without the address; url_for_record keeps a
#   made-up signature out of the record;
# - the validation, test and args commands end to end on made-up records with the tiny model and a stand-in tokenizer:
#   the files written, the fallback to fixed_versions.json's values, test refusing to run without release_candidate.json
#   or with another dtype or a changed model file, and test reading nothing from the network;
# - with BioGPT-Large-PubMedQA's own tokenizer files (skipped when absent; EDDM_BIOGPT_TOKENIZER, or
#   /tmp/eddm_s9/agent_biogpt/tok): yes, no and maybe after the opening words are 22134, 126 and 28139, learned1 to
#   learned9 are 57717 to 57725, '...' is 11774, and load_model reads that tokenizer beside a tiny model.
# ======================================================================================================================

_BIOGPT_PARENT = Path(__file__).resolve().parents[1]
BIOGPT_SRC = _BIOGPT_PARENT if (_BIOGPT_PARENT / "eval" / "biogpt_session9.py").exists() else Path.cwd() / "src"
BIOGPT_ROOT = BIOGPT_SRC.parent

BIOGPT_SEED = 20260927
BIOGPT_TOKENIZER_DIR = Path(os.environ.get("EDDM_BIOGPT_TOKENIZER", "/tmp/eddm_s9/agent_biogpt/tok"))
BIOGPT_FIXED_VERSIONS = BIOGPT_ROOT / "results" / "fiorillo_v0_5" / "fixed_versions.json"
BIOGPT_OPENING = "the answer to the question given the context is"
BIOGPT_WORDS = ("question", ":", "context", "?", ".", "...", "the", "answer", "to", "given", "is", "yes", "no", "maybe",
         "a", "b", "c", "d", "e", "f", "g", "h")
BIOGPT_SIGNED_URL = BG.ARCHIVE_BASE + "?sv=2020-01-01&sr=b&sp=r&sig=MADE-UP-FOR-A-TEST"  # a made-up read-only signature


class BioGptStandIn:
    """A stand-in for BioGptTokenizer: "...", ':', '.' and '?' split off, each word w the token "w</w>" (a word in
    splits becomes its pieces), learned1 to learned9 at the end of the vocabulary, and </s> (id 2) put first when
    special tokens are added (unless lead is False)."""
    sep_token_id = eos_token_id = 2
    pad_token_id = 1
    unk_token_id = 3

    def __init__(self, words=BIOGPT_WORDS, splits=None, extra=(), learned=9, lead=True):
        self.vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        for w in words:
            self.vocab.setdefault(f"{w}</w>", len(self.vocab))
        for t in extra:
            self.vocab.setdefault(t, len(self.vocab))
        for i in range(learned):
            self.vocab.setdefault(f"learned{i + 1}</w>", len(self.vocab))
        self.splits, self.lead = splits or {}, lead
        self.inv = {i: t for t, i in self.vocab.items()}

    def get_vocab(self):
        return dict(self.vocab)

    def __call__(self, text, add_special_tokens=True, split_special_tokens=False):
        ids = []
        for w in re.findall(r"\.\.\.|[:.?]|[^\s:.?]+", text):
            ids += [self.vocab[p] for p in self.splits[w]] if w in self.splits else [self.vocab.get(f"{w}</w>", 3)]
        return {"input_ids": ([2] if add_special_tokens and self.lead else []) + ids}

    def convert_ids_to_tokens(self, ids):
        return [self.inv[int(i)] for i in ids]


class BioGptMeter:
    """A class the reader does not import (pickled by reference, as fairseq's meters are)."""

    def __init__(self, value):
        self.value = value


class BioGptNodeLike:
    """Pickles as omegaconf's value nodes do (their state holds _val)."""

    def __init__(self, value):
        self._val, self._metadata, self._parent = value, None, None


class BioGptDictConfigLike:
    """Pickles as omegaconf's DictConfig does (its state holds _content)."""

    def __init__(self, content):
        self._content, self._metadata, self._parent = content, None, None


def biogpt_settings(**change):
    s = {"learned_prompt": 9, "learned_prompt_pattern": "learned", "prefix": False, "manual_prompt": None,
         "max_source_positions": 384, "max_target_positions": None, "tokens_per_sample": None}
    s.update(change)
    return s


def biogpt_entry(tok, max_source_positions=384, files=None):
    vocab = tok.get_vocab()
    return {"repo": "microsoft/BioGPT-Large-PubMedQA", "revision": "made-up", "files": files or {},
            "learned_prompt": {"tokens": [f"learned{i + 1}" for i in range(9)],
                               "ids": [vocab[f"learned{i + 1}</w>"] for i in range(9)], "count": 9},
            "order": "source tokens (without their eos), then learned1 to learned9, then the target (collate's make_sentence "
                     "with prefix False: torch.cat([source, prompt, target]))",
            "source_truncation": {"max_source_positions": max_source_positions},
            "target_opening_words": BIOGPT_OPENING,
            "answer_words": {w: {"token": f"{w}</w>", "id": vocab[f"{w}</w>"]} for w in ("yes", "no", "maybe")}}


def biogpt_checkpoint(cfg_task=None, args=None, cfg=True):
    import torch
    task = {"_name": "language_modeling_prompt", "learned_prompt": 9, "learned_prompt_pattern": "learned", "prefix": False,
            "manual_prompt": None, "max_source_positions": 384, "max_target_positions": 1024, "tokens_per_sample": 1024,
            "sep_token": "<seqsep>"}
    task.update(cfg_task or {})
    out = {"args": args, "model": OrderedDict([("decoder.embed_tokens.weight", torch.zeros(4, 2))]),
           "extra_state": {"metrics": {"train": BioGptMeter(3.0)}}, "optimizer_history": [{"num_updates": 7}]}
    out["cfg"] = {"common": {"fp16": True}, "task": task, "model": {"_name": "transformer_lm_prompt_biogpt_large"}} if cfg else None
    return out


def biogpt_archive(path, obj, legacy=False, fmt=tarfile.GNU_FORMAT, name="QA-PubMedQA-BioGPT-Large/checkpoint.pt", before=()):
    """A gzipped tar like QA-PubMedQA-BioGPT-Large.tgz: the members in before ((name, bytes), or (name, None) for a
    directory), then the checkpoint written by torch.save; returns its file URL."""
    import torch
    buf = io.BytesIO()
    torch.save(obj, buf, _use_new_zipfile_serialization=not legacy)
    with tarfile.open(path, "w:gz", format=fmt) as tar:
        for member, data in list(before) + [(name, buf.getvalue())]:
            info = tarfile.TarInfo(member)
            if data is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    return Path(path).as_uri()


def biogpt_tiny_model(vocab_size, positions=64):
    import torch
    from transformers import BioGptConfig, BioGptForCausalLM
    torch.manual_seed(BIOGPT_SEED)
    config = BioGptConfig(vocab_size=vocab_size, hidden_size=8, num_hidden_layers=2, num_attention_heads=2, intermediate_size=16,
                          max_position_embeddings=positions, pad_token_id=1, bos_token_id=0, eos_token_id=2)
    return BioGptForCausalLM(config).eval()


def biogpt_loader(model, tok):
    def load(model_dir, dtype, device):
        return model, tok, {"class": type(model).__name__, "tokenizer": type(tok).__name__, "dtype": dtype, "device": str(device),
                            "gpu": None, "max_position_embeddings": int(model.config.max_position_embeddings),
                            "vocab_size": int(model.config.vocab_size), "load_seconds": 0.0}
    return load


def biogpt_record(pmid, label, question, texts, calibrate=True):
    r = {"id": f"pubmedqa-{pmid}", "qid": "pubmedqa_answer", "label": label, "target": [float(k == label) for k in range(3)],
         "state": {"question": question, "contexts": [{"section": "S", "text": t} for t in texts]}, "group": {"source": "pqa_labeled"}}
    if not calibrate:
        r["calibrate"] = False
    return r


def biogpt_write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def biogpt_read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


# ---------------------------------------------------------------- the input

def test_biogpt_source_text_follows_rebuild_data():
    state = {"question": "  Is x better?\n", "contexts": [{"section": "A", "text": " First  line\nsecond. "},
                                                        {"section": "B", "text": "\tThird\t\tpart "}]}
    assert BG.source_text(state) == "question: Is x better? context: First line second. Third part "


def test_biogpt_input_has_the_learned_prompt_in_place_and_in_order():
    tok = BioGptStandIn()
    spec = BG.input_spec(tok, biogpt_settings(), biogpt_entry(tok))
    state = {"question": "a b?", "contexts": [{"section": "S", "text": "c d e."}]}
    seq, n, cut = BG.item_sequence(state, tok, spec)
    source = tok("question: a b? context: c d e. ", add_special_tokens=False)["input_ids"]
    learned = [tok.vocab[f"learned{i}</w>"] for i in range(1, 10)]
    opening = tok(BIOGPT_OPENING, add_special_tokens=False)["input_ids"]
    assert seq == [2] + source + learned + opening and n == len(source) and not cut
    assert tok.convert_ids_to_tokens(seq[1 + len(source):10 + len(source)]) == [f"learned{i}</w>" for i in range(1, 10)]
    assert spec["prompt_ids"] == learned and spec["prompt_ids_same_as_fixed_versions"]
    assert spec["order"] == "</s>, the source (cut), the prompt, the opening words"
    assert tok.convert_ids_to_tokens(seq[-len(opening):]) == [f"{w}</w>" for w in BIOGPT_OPENING.split()]


def test_biogpt_source_cut_at_max_source_positions_with_ellipsis():
    assert BG.cut_source(list(range(10, 20)), 5, 99) == ([10, 11, 12, 13, 99], True)
    assert BG.cut_source(list(range(10, 15)), 5, 99) == (list(range(10, 15)), False)  # exactly P tokens: not cut
    assert BG.cut_source(list(range(10, 16)), 5, 99) == ([10, 11, 12, 13, 99], True)
    tok = BioGptStandIn()
    spec = BG.input_spec(tok, biogpt_settings(max_source_positions=6), biogpt_entry(tok))
    state = {"question": "a b?", "contexts": [{"section": "S", "text": "c d e."}]}
    seq, n, cut = BG.item_sequence(state, tok, spec)
    source = tok("question: a b? context: c d e. ", add_special_tokens=False)["input_ids"]
    assert cut and n == len(source) == 11
    assert seq[1:7] == source[:5] + [tok.vocab["...</w>"]] and spec["ellipsis_id"] == tok.vocab["...</w>"]
    assert seq[7:16] == [tok.vocab[f"learned{i}</w>"] for i in range(1, 10)] and len(seq) == 1 + 6 + 9 + 9


def test_biogpt_eos_first_and_never_after_the_source():
    tok = BioGptStandIn()
    spec = BG.input_spec(tok, biogpt_settings(), biogpt_entry(tok))
    seq, _, _ = BG.item_sequence({"question": "a?", "contexts": [{"section": "S", "text": "b."}]}, tok, spec)
    assert seq[0] == 2 and seq.count(2) == 1  # the leading </s> only: the source's own eos is dropped, as make_sentence drops it
    no_lead = BioGptStandIn(lead=False)
    spec2 = BG.input_spec(no_lead, biogpt_settings(), biogpt_entry(no_lead))
    with pytest.raises(SystemExit, match="did not put </s> before the source"):
        BG.item_sequence({"question": "a?", "contexts": [{"section": "S", "text": "b."}]}, no_lead, spec2)
    other = BioGptStandIn()
    other.sep_token_id = 0
    with pytest.raises(SystemExit, match="sep_token is not </s>"):
        BG.input_spec(other, biogpt_settings(), biogpt_entry(other))


def test_biogpt_prefix_order_and_manual_prompt():
    tok = BioGptStandIn()
    with pytest.raises(SystemExit, match="no <seqsep>"):
        BG.input_spec(tok, biogpt_settings(prefix=True), biogpt_entry(tok))
    tok = BioGptStandIn(extra=("<seqsep>",))
    spec = BG.input_spec(tok, biogpt_settings(prefix=True), biogpt_entry(tok))
    seq, _, _ = BG.item_sequence({"question": "a?", "contexts": [{"section": "S", "text": "b."}]}, tok, spec)
    source = tok("question: a? context: b. ", add_special_tokens=False)["input_ids"]
    learned = [tok.vocab[f"learned{i}</w>"] for i in range(1, 10)]
    assert seq == [2] + learned + source + [tok.vocab["<seqsep>"]] + spec["opening_ids"]
    tok = BioGptStandIn(extra=("b",))
    spec = BG.input_spec(tok, biogpt_settings(learned_prompt=None, manual_prompt="a b@@ c"), biogpt_entry(tok))
    assert spec["prompt_kind"] == "manual" and spec["prompt_tokens"] == ["a</w>", "b", "c</w>"]
    with pytest.raises(SystemExit, match="not in the model's vocabulary"):
        BG.input_spec(tok, biogpt_settings(learned_prompt=12), biogpt_entry(tok))


# ---------------------------------------------------------------- the answers' probabilities

def test_biogpt_answer_probabilities_from_made_up_logits():
    z = np.array([[0.0, np.log(2.0), 0.0, np.log(6.0), np.log(2.0), 1.0],
                  [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    probs, mass, top = BG.answer_probabilities(z, [3, 1, 4])
    assert np.allclose(probs[0], [0.6, 0.2, 0.2]) and np.allclose(probs[1], [1 / 3, 1 / 3, 1 / 3])
    assert mass[0] == pytest.approx(10.0 / (12.0 + np.e)) and mass[1] == pytest.approx(3.0 / (np.exp(10.0) + 5.0))
    assert list(top) == [3, 0]


def test_biogpt_answer_tokens_one_token_and_the_first_token_rule():
    tok = BioGptStandIn()
    used, info = BG.answer_token_ids(tok, BIOGPT_OPENING, tok(BIOGPT_OPENING, add_special_tokens=False)["input_ids"])
    assert used == [tok.vocab["yes</w>"], tok.vocab["no</w>"], tok.vocab["maybe</w>"]] and info["order"] == ["yes", "no", "maybe"]
    assert not info["first_token_rule"] and info["multi_token_answers"] == []
    assert all(info["words"][w]["same_in_the_target_line"] for w in ("yes", "no", "maybe"))
    split = BioGptStandIn(splits={"maybe": ["may", "be</w>"]}, extra=("may", "be</w>"))
    spec = BG.input_spec(split, biogpt_settings(), biogpt_entry(split))
    assert spec["answer_ids"] == [split.vocab["yes</w>"], split.vocab["no</w>"], split.vocab["may"]]
    assert spec["answers"]["first_token_rule"] and spec["answers"]["multi_token_answers"] == ["maybe"]
    assert spec["answers"]["words"]["maybe"]["ids"] == [split.vocab["may"], split.vocab["be</w>"]]
    z = np.zeros((1, len(split.vocab)))
    z[0, split.vocab["may"]] = np.log(2.0)
    z[0, split.vocab["be</w>"]] = 5.0  # the second piece's logit is not read
    probs, _, _ = BG.answer_probabilities(z, spec["answer_ids"])
    assert np.allclose(probs[0], [0.25, 0.25, 0.5])
    clash = BioGptStandIn(splits={"maybe": ["no</w>", "be</w>"]}, extra=("be</w>",))
    with pytest.raises(SystemExit, match="share their first token"):
        BG.input_spec(clash, biogpt_settings(), biogpt_entry(clash))


def test_biogpt_renormalization_over_the_three_answers():
    rng = np.random.default_rng(BIOGPT_SEED)
    z = rng.normal(0.0, 3.0, size=(6, 50))
    ids = [7, 3, 11]
    probs, mass, _ = BG.answer_probabilities(z, ids)
    full = np.exp(z - z.max(1, keepdims=True))
    full /= full.sum(1, keepdims=True)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-12)
    assert np.allclose(mass, full[:, ids].sum(1)) and np.allclose(probs, full[:, ids] / mass[:, None])
    shifted, _, _ = BG.answer_probabilities(z + 100.0, ids)
    assert np.allclose(shifted, probs)


def test_biogpt_padded_batches_match_each_item_alone():
    import torch
    model = biogpt_tiny_model(40)
    rng = np.random.default_rng(BIOGPT_SEED)
    seqs = [[2] + [int(x) for x in rng.integers(4, 40, size=n)] for n in (4, 8, 2, 11, 6)]
    batched, seconds = BG.next_token_logits(model, seqs, 1, "cpu", 3)
    alone, _ = BG.next_token_logits(model, seqs, 1, "cpu", 1)
    with torch.no_grad():
        full = np.stack([model(torch.tensor([s])).logits[0, -1].numpy() for s in seqs])
    assert batched.shape == (5, 40) and seconds >= 0.0
    assert np.allclose(batched, alone, atol=1e-5) and np.allclose(alone, full, atol=1e-5)


# ---------------------------------------------------------------- correction 5's reader

def test_biogpt_reader_reads_a_zip_checkpoint_in_a_gzipped_tar(tmp_path):
    uri = biogpt_archive(tmp_path / "QA-PubMedQA-BioGPT-Large.tgz", biogpt_checkpoint(),
                       before=[("QA-PubMedQA-BioGPT-Large", None), ("QA-PubMedQA-BioGPT-Large/dict.txt", b"the 5\n" * 40)])
    r = BG.read_checkpoint_arguments(uri, seconds=60, limit_bytes=10 ** 7)
    assert r["status"] == "read" and r["missing"] == []
    assert r["values"] == {"learned_prompt": 9, "learned_prompt_pattern": "learned", "prefix": False, "manual_prompt": None,
                           "max_source_positions": 384, "max_target_positions": 1024, "tokens_per_sample": 1024}
    assert set(r["found_in"].values()) == {"cfg.task"} and r["context"]["cfg.common.fp16"] is True
    assert r["member"] == "QA-PubMedQA-BioGPT-Large/checkpoint.pt" and r["checkpoint_format"].startswith("zip")
    assert r["pickle_entry"].endswith("data.pkl") and [m["name"] for m in r["members_passed"]] == [
        "QA-PubMedQA-BioGPT-Large/", "QA-PubMedQA-BioGPT-Large/dict.txt"]
    assert 0 < r["bytes_read"] <= (tmp_path / "QA-PubMedQA-BioGPT-Large.tgz").stat().st_size
    assert "cfg" in r["checkpoint_keys"] and r["attempts"][0]["outcome"] == "read" and r["finished_utc"].endswith("Z")
    json.dumps(r)  # the reading is written to biogpt.json as it is


def test_biogpt_reader_legacy_format_long_names_and_appledouble(tmp_path):
    long_name = "QA-PubMedQA-BioGPT-Large/" + "a-folder-name-longer-than-a-ustar-name-field-allows/" * 3 + "checkpoint.pt"
    for k, fmt in enumerate((tarfile.GNU_FORMAT, tarfile.PAX_FORMAT)):
        uri = biogpt_archive(tmp_path / f"legacy_{k}.tgz", biogpt_checkpoint({"max_source_positions": 512}), legacy=True, fmt=fmt,
                           name=long_name, before=[("QA-PubMedQA-BioGPT-Large/._checkpoint.pt", b"\x00\x05\x16\x07 not a checkpoint")])
        r = BG.read_checkpoint_arguments(uri, seconds=60, limit_bytes=10 ** 7)
        assert r["status"] == "read" and r["member"] == long_name and r["checkpoint_format"].startswith("legacy")
        assert r["values"]["max_source_positions"] == 512 and r["pickle_entry"] is None
        assert r["members_passed"][0]["name"] == "QA-PubMedQA-BioGPT-Large/._checkpoint.pt"


def test_biogpt_reader_reads_omegaconf_like_containers_and_legacy_args(tmp_path):
    import torch
    cfg = BioGptDictConfigLike({"task": BioGptDictConfigLike({"learned_prompt": BioGptNodeLike(7), "learned_prompt_pattern": BioGptNodeLike("learned"),
                                                  "prefix": BioGptNodeLike(False), "manual_prompt": BioGptNodeLike(None),
                                                  "max_source_positions": BioGptNodeLike(300)}),
                          "model": BioGptDictConfigLike({"max_target_positions": BioGptNodeLike(2048), "tokens_per_sample": BioGptNodeLike(2048)})})
    obj = {"args": None, "cfg": cfg, "model": OrderedDict(w=torch.zeros(2))}
    r = BG.read_checkpoint_arguments(biogpt_archive(tmp_path / "omega.tgz", obj), seconds=60, limit_bytes=10 ** 7)
    assert r["status"] == "read" and r["values"]["learned_prompt"] == 7 and r["values"]["max_source_positions"] == 300
    assert r["found_in"]["max_target_positions"] == "cfg.model" and r["values"]["tokens_per_sample"] == 2048
    ns = argparse.Namespace(task="language_modeling_prompt", arch="transformer_lm_prompt_biogpt_large", learned_prompt=9,
                            learned_prompt_pattern="learned", prefix=False, manual_prompt=None, max_source_positions=384,
                            max_target_positions=1024, tokens_per_sample=1024, fp16=True)
    r = BG.read_checkpoint_arguments(biogpt_archive(tmp_path / "args.tgz", biogpt_checkpoint(args=ns, cfg=False)), seconds=60,
                                    limit_bytes=10 ** 7)
    assert r["status"] == "read" and set(r["found_in"].values()) == {"args"} and r["context"]["args.arch"] == "transformer_lm_prompt_biogpt_large"
    partial = biogpt_checkpoint()
    del partial["cfg"]["task"]["tokens_per_sample"]
    r = BG.read_checkpoint_arguments(biogpt_archive(tmp_path / "partial.tgz", partial), seconds=60, limit_bytes=10 ** 7)
    assert r["status"] == "read; not found in the checkpoint: tokens_per_sample" and r["missing"] == ["tokens_per_sample"]


def test_biogpt_checkpoint_values_that_differ_are_used_and_change_the_input(tmp_path):
    tok = BioGptStandIn()
    entry = biogpt_entry(tok)
    uri = biogpt_archive(tmp_path / "differ.tgz", biogpt_checkpoint({"learned_prompt": 5, "max_source_positions": 6, "max_target_positions": 40}))
    reading = BG.read_checkpoint_arguments(uri, seconds=60, limit_bytes=10 ** 7)
    settings, res = BG.resolve_settings(reading, BG.fixed_settings(entry))
    assert settings["learned_prompt"] == 5 and settings["max_source_positions"] == 6 and settings["max_target_positions"] == 40
    assert res["differences_from_fixed_versions"] == {"learned_prompt": {"checkpoint": 5, "fixed_versions": 9},
                                                      "max_source_positions": {"checkpoint": 6, "fixed_versions": 384}}
    assert res["origin"]["learned_prompt"] == "checkpoint (cfg.task)" and res["origin"]["prefix"] == "checkpoint (cfg.task)"
    spec = BG.input_spec(tok, settings, entry)
    seq, _, cut = BG.item_sequence({"question": "a b?", "contexts": [{"section": "S", "text": "c d e."}]}, tok, spec)
    source = tok("question: a b? context: c d e. ", add_special_tokens=False)["input_ids"]
    assert cut and seq == [2] + source[:5] + [tok.vocab["...</w>"]] + [tok.vocab[f"learned{i}</w>"] for i in range(1, 6)] + spec["opening_ids"]
    assert "prompt_ids_same_as_fixed_versions" not in spec  # five learned tokens are not fixed_versions.json's nine
    small = dict(settings, max_target_positions=20)  # 20 - 6 - 5 = 9 places, fewer than the target's 12 tokens
    with pytest.raises(SystemExit, match="would have been cut"):
        BG.input_spec(tok, small, entry)


def test_biogpt_reader_falls_back_when_the_stream_fails(tmp_path, monkeypatch):
    tok = BioGptStandIn()
    fixed = BG.fixed_settings(biogpt_entry(tok))
    missing = BG.read_checkpoint_arguments((tmp_path / "absent.tgz").as_uri(), seconds=60, limit_bytes=10 ** 7)
    assert missing["status"].startswith("not read") and len(missing["attempts"]) == 1
    settings, res = BG.resolve_settings(missing, fixed)
    assert settings == fixed and res["differences_from_fixed_versions"] == {} and "not been read" in res["used_from"]
    assert res["origin"]["max_source_positions"] == "fixed_versions.json"
    assert res["origin"]["tokens_per_sample"].startswith("not read")
    garbage = tmp_path / "garbage.tgz"
    import gzip
    garbage.write_bytes(gzip.compress(np.random.default_rng(BIOGPT_SEED).bytes(5000)))
    r = BG.read_checkpoint_arguments(garbage.as_uri(), seconds=60, limit_bytes=10 ** 7)
    assert r["status"].startswith("not read") and "checksum" in r["status"]
    uri = biogpt_archive(tmp_path / "ok.tgz", biogpt_checkpoint())
    r = BG.read_checkpoint_arguments(uri, seconds=60, limit_bytes=100)
    assert r["status"].startswith("not read") and "read limit of 100 bytes" in r["status"] and r["bytes_read"] <= 100

    class Slow:
        headers = {}

        def read(self, n=-1):
            time.sleep(0.2)
            return b"\0"

        def close(self):
            pass

    t0 = time.monotonic()
    r = BG.read_checkpoint_arguments("https://example.invalid/QA-PubMedQA-BioGPT-Large.tgz", seconds=1.0, limit_bytes=10 ** 7,
                                    opener=lambda url, timeout: Slow())
    assert r["status"].startswith("not read") and time.monotonic() - t0 < 3.0
    assert BG.resolve_settings(r, fixed)[0] == fixed

    def forbidden(url, timeout):
        raise urllib.error.HTTPError(url, 403, "Server failed to authenticate the request", {}, None)

    r = BG.read_checkpoint_arguments("https://example.invalid/x.tgz", seconds=60, limit_bytes=10 ** 7, opener=forbidden)
    assert r["status"] == "not read: ReadStop: HTTP 403 Server failed to authenticate the request" and len(r["attempts"]) == 1


def test_biogpt_reader_retries_a_network_failure(tmp_path, monkeypatch):
    uri = biogpt_archive(tmp_path / "retry.tgz", biogpt_checkpoint())
    calls = []

    def opener(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise ConnectionResetError("reset by the made-up peer")
        return urllib.request.urlopen(uri, timeout=timeout)

    monkeypatch.setattr(BG, "RETRY_WAIT_SECONDS", 0.05)
    r = BG.read_checkpoint_arguments("https://example.invalid/QA-PubMedQA-BioGPT-Large.tgz", seconds=60, limit_bytes=10 ** 7,
                                    opener=opener)
    assert r["status"] == "read" and len(calls) == 2 and r["attempts"][0]["outcome"].startswith("network failure")


def test_biogpt_reader_records_no_signature_and_keeps_fixed_values_it_cannot_use():
    # The module no longer holds the address (ARCHIVE_URL is None; archive_url_from_readme reads it at run time), so the
    # record is checked on a made-up signed address.
    assert "sig=" not in BG.url_for_record(BIOGPT_SIGNED_URL) and BG.url_for_record(BIOGPT_SIGNED_URL).startswith(
        "https://msralaphilly2.blob.core.windows.net/release/BioGPT/checkpoints/QA-PubMedQA-BioGPT-Large.tgz (with the read-only")
    assert BG.url_for_record("file:///tmp/x.tgz") == "file:///tmp/x.tgz"
    tok = BioGptStandIn()
    fixed = BG.fixed_settings(biogpt_entry(tok))
    reading = {"status": "read", "values": {"max_source_positions": None, "prefix": "False", "learned_prompt": "9"},
               "found_in": {"max_source_positions": "cfg.task", "prefix": "args", "learned_prompt": "args"}}
    settings, res = BG.resolve_settings(reading, fixed)
    assert settings == fixed and res["unreadable_values"] == {"max_source_positions": "None"}
    assert res["origin"]["max_source_positions"] == "fixed_versions.json" and res["origin"]["prefix"] == "checkpoint (args)"
    assert res["differences_from_fixed_versions"] == {}


# ---------------------------------------------------------------- the archive's address, from microsoft/BioGPT's README

def biogpt_readme_case(tmp_path, readme, recorded=None):
    """A made-up fixed_versions.json whose BioGPT entry records, at a made-up commit, the SHA-256 of recorded (by default
    the README itself), and an opener that serves the README's bytes and keeps the addresses it is asked for."""
    recorded = readme if recorded is None else recorded
    fixed_versions = tmp_path / "fixed_versions.json"
    fixed_versions.write_text(json.dumps({"plan": "made up", BG.FIXED_KEY: {"scripts": {
        "commit": "made-up-commit", "files_sha256": {"README.md": hashlib.sha256(recorded).hexdigest()}}}}), encoding="utf-8")
    asked = []

    def opener(url, timeout):
        asked.append(url)
        return io.BytesIO(readme)

    return fixed_versions, opener, asked


def test_biogpt_archive_url_found_in_the_readme(tmp_path):
    other = BG.ARCHIVE_BASE.replace("-Large.tgz", ".tgz") + "?sv=2020-01-01&sr=b&sp=r&sig=ANOTHER-MADE-UP-ONE"
    readme = ("# BioGPT\n\n| Model | Description | Link |\n|---|---|---|\n"
              f"| QA-PubMedQA-BioGPT | fine-tuned on PubMedQA | [link]({other}) |\n"
              f"| QA-PubMedQA-BioGPT-Large | fine-tuned on PubMedQA | [link]({BIOGPT_SIGNED_URL}) |\n").encode("utf-8")
    fixed_versions, opener, asked = biogpt_readme_case(tmp_path, readme)
    assert BG.archive_url_from_readme(fixed_versions, opener=opener) == BIOGPT_SIGNED_URL  # the link's closing ")" left out
    assert asked == [BG.README_URL.format(commit="made-up-commit")]  # the README at fixed_versions.json's commit


def test_biogpt_archive_url_refused_when_the_readme_sha256_differs(tmp_path):
    readme = f"| QA-PubMedQA-BioGPT-Large | [link]({BIOGPT_SIGNED_URL}) |\n".encode("utf-8")
    recorded = readme.replace(b"[link]", b"[Link]")  # fixed_versions.json records another README's SHA-256
    fixed_versions, opener, asked = biogpt_readme_case(tmp_path, readme, recorded)
    got, want = hashlib.sha256(readme).hexdigest(), hashlib.sha256(recorded).hexdigest()
    with pytest.raises(BG.ReadStop, match=f"README has SHA-256 {got}, not fixed_versions.json's {want}"):
        BG.archive_url_from_readme(fixed_versions, opener=opener)
    assert len(asked) == 1


def test_biogpt_archive_url_refused_when_the_readme_has_no_such_link(tmp_path):
    for readme in (b"# BioGPT\n\nNo checkpoint is linked here.\n",
                   f"| QA-PubMedQA-BioGPT-Large | [link]({BG.ARCHIVE_BASE}) |\n".encode("utf-8")):  # no signature
        fixed_versions, opener, asked = biogpt_readme_case(tmp_path, readme)
        with pytest.raises(BG.ReadStop, match="has no link to the archive"):
            BG.archive_url_from_readme(fixed_versions, opener=opener)
        assert len(asked) == 1


# ---------------------------------------------------------------- the commands end to end

def biogpt_end_to_end_setup(tmp_path, monkeypatch, max_source_positions_in_checkpoint=10):
    tok = BioGptStandIn()
    model = biogpt_tiny_model(len(tok.vocab))
    monkeypatch.setattr(BG, "load_model", biogpt_loader(model, tok))
    monkeypatch.setattr(BG, "EXPECTED_ITEMS", {"validation": 3, "test": 2})
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"made": "up"}', encoding="utf-8")
    data = (model_dir / "config.json").read_bytes()
    entry = biogpt_entry(tok, max_source_positions=8,
                          files={"config.json": {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}})
    fixed_versions = tmp_path / "fixed_versions.json"
    fixed_versions.write_text(json.dumps({"plan": "made up", BG.FIXED_KEY: entry}), encoding="utf-8")
    records = tmp_path / "records"
    biogpt_write_jsonl(records / "pubmedqa_answer_validation.jsonl", [
        biogpt_record(1, 0, "a b?", ["c d.", "e f."]), biogpt_record(2, 1, "a?", ["c."]),
        biogpt_record(9, 0, "h?", ["g."], calibrate=False), biogpt_record(3, 2, "a b c d e f g h?", ["a b c d e f g h a b c d."])])
    biogpt_write_jsonl(records / "pubmedqa_answer_test.jsonl", [biogpt_record(4, 1, "b?", ["d e."]), biogpt_record(5, 0, "c d?", ["f g h."])])
    uri = biogpt_archive(tmp_path / "ckpt.tgz", biogpt_checkpoint({"max_source_positions": max_source_positions_in_checkpoint}))
    results = tmp_path / "results"
    common = ["--model-dir", str(model_dir), "--results", str(results), "--records-public", str(records),
              "--fixed-versions", str(fixed_versions), "--device", "cpu", "--batch-size", "2"]
    return tok, model, entry, uri, results, records, model_dir, common


def test_biogpt_validation_and_test_end_to_end(tmp_path, monkeypatch):
    import torch
    tok, model, entry, uri, results, records, model_dir, common = biogpt_end_to_end_setup(tmp_path, monkeypatch)
    assert BG.main(["validation", *common, "--url", uri]) == 0
    rows = biogpt_read_csv(results / "predictions" / "biogpt_large_pqa-raw_pubmedqa_answer_validation.csv")
    assert rows[0] == ["id", "label", "p0", "p1", "p2"]
    assert [r[:2] for r in rows[1:]] == [["pubmedqa-1", "0"], ["pubmedqa-2", "1"], ["pubmedqa-3", "2"]]  # the 150-item rule: calibrate rows only
    p = np.array([[float(x) for x in r[2:]] for r in rows[1:]])
    assert np.allclose(p.sum(1), 1.0, atol=1e-8)
    data = json.loads((results / "biogpt.json").read_text(encoding="utf-8"))
    assert data["plan"] == BG.PLAN and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", data["written_utc"])
    assert data["checkpoint_arguments"]["status"] == "read" and data["input"]["settings"]["max_source_positions"] == 10
    assert data["input"]["differences_from_fixed_versions"] == {"max_source_positions": {"checkpoint": 10, "fixed_versions": 8}}
    v = data["validation"]
    assert v["items"] == 3 and v["sources_cut"] == 2 and v["items_by_label"] == {"yes": 1, "no": 1, "maybe": 1}
    assert v["forward_seconds"] >= 0.0 and v["example_input"]["first_tokens"] == ["</s>", "question</w>", ":</w>"]
    assert v["example_input"]["last_tokens"][-9:] == [f"{w}</w>" for w in BIOGPT_OPENING.split()]
    assert data["model"]["files"]["config.json"]["sha256"] == entry["files"]["config.json"]["sha256"]
    # the written probabilities are those of the input built here, item by item, with no padding
    spec = BG.input_spec(tok, data["input"]["settings"], entry)
    val = [r for r in (json.loads(x) for x in (records / "pubmedqa_answer_validation.jsonl").read_text().splitlines()) if r.get("calibrate", True)]
    seqs = [BG.item_sequence(r["state"], tok, spec)[0] for r in val]
    with torch.no_grad():
        logits = np.stack([model(torch.tensor([s])).logits[0, -1].numpy() for s in seqs])
    want, _, _ = BG.answer_probabilities(logits, spec["answer_ids"])
    assert np.allclose(p, want, rtol=1e-6, atol=1e-9)

    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        BG.main(["test", *common])
    assert not (results / "predictions" / "biogpt_large_pqa-raw_pubmedqa_answer_test.csv").exists()
    (results / "release_candidate.json").write_text('{"time_utc": "made up"}', encoding="utf-8")
    with pytest.raises(SystemExit, match="differs from validation's"):
        BG.main(["test", *common, "--dtype", "bfloat16"])

    def no_network(*a, **k):
        raise AssertionError("the test command must not read the archive")

    monkeypatch.setattr(BG, "read_checkpoint_arguments", no_network)
    assert BG.main(["test", *common]) == 0
    test_rows = biogpt_read_csv(results / "predictions" / "biogpt_large_pqa-raw_pubmedqa_answer_test.csv")
    assert [r[:2] for r in test_rows[1:]] == [["pubmedqa-4", "1"], ["pubmedqa-5", "0"]]  # labels from the records, as session 8 wrote them
    after = json.loads((results / "biogpt.json").read_text(encoding="utf-8"))
    assert after["test"]["items"] == 2 and after["validation"] == data["validation"] and after["input"] == data["input"]
    assert after["test"]["release_candidate_sha256"] == hashlib.sha256(b'{"time_utc": "made up"}').hexdigest()
    (model_dir / "config.json").write_text('{"made": "up", "changed": true}', encoding="utf-8")
    with pytest.raises(SystemExit, match="fixed_versions.json lists"):
        BG.main(["test", *common])


def test_biogpt_validation_falls_back_to_fixed_values_and_args_command(tmp_path, monkeypatch):
    tok, model, entry, uri, results, records, model_dir, common = biogpt_end_to_end_setup(tmp_path, monkeypatch)
    assert BG.main(["validation", *common, "--url", (tmp_path / "absent.tgz").as_uri()]) == 0
    data = json.loads((results / "biogpt.json").read_text(encoding="utf-8"))
    assert data["checkpoint_arguments"]["status"].startswith("not read")
    assert data["input"]["settings"]["max_source_positions"] == 8 and data["input"]["differences_from_fixed_versions"] == {}
    assert "not been read" in data["input"]["used_from"]
    with pytest.raises(SystemExit, match="holds the validation run's record"):
        BG.main(["args", "--results", str(results), "--fixed-versions", str(tmp_path / "fixed_versions.json"), "--url", uri])
    out = tmp_path / "reading.json"
    BG.main(["args", "--out", str(out), "--fixed-versions", str(tmp_path / "fixed_versions.json"), "--url", uri])
    reading = json.loads(out.read_text(encoding="utf-8"))
    assert reading["plan"] == BG.PLAN and reading["checkpoint_arguments"]["status"] == "read"
    assert reading["settings_if_scored_now"]["settings"]["max_source_positions"] == 10


def test_biogpt_model_files_checked_against_fixed_versions(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "pytorch_model.bin").write_bytes(b"made-up weights")
    sha = hashlib.sha256(b"made-up weights").hexdigest()
    entry = {"repo": "r", "revision": "v", "files": {"pytorch_model.bin": {"bytes": 15, "sha256": sha}}}
    assert BG.check_model_files(folder, entry)["pytorch_model.bin"] == {"bytes": 15, "sha256": sha}
    with pytest.raises(SystemExit, match="SHA-256"):
        BG.check_model_files(folder, {**entry, "files": {"pytorch_model.bin": {"bytes": 15, "sha256": "0" * 64}}})
    with pytest.raises(SystemExit, match="does not exist"):
        BG.check_model_files(folder, {**entry, "files": {"vocab.json": {"bytes": 1}}})
    with pytest.raises(SystemExit, match="download r at revision v"):
        BG.check_model_files(tmp_path / "absent", entry)


# ---------------------------------------------------------------- the real tokenizer (skipped when its files are absent)

def biogpt_real_tokenizer():
    if not (BIOGPT_TOKENIZER_DIR / "vocab.json").exists() or not (BIOGPT_TOKENIZER_DIR / "merges.txt").exists():
        pytest.skip(f"BioGPT-Large-PubMedQA's tokenizer files are not in {BIOGPT_TOKENIZER_DIR}")
    pytest.importorskip("sacremoses")
    from transformers import BioGptTokenizer
    return BioGptTokenizer.from_pretrained(str(BIOGPT_TOKENIZER_DIR))


def biogpt_real_entry():
    if not BIOGPT_FIXED_VERSIONS.exists():
        pytest.skip(f"{BIOGPT_FIXED_VERSIONS} is absent")
    return BG.fixed_entry(BIOGPT_FIXED_VERSIONS)


def test_biogpt_real_tokenizer_answer_prompt_and_ellipsis_ids():
    tok = biogpt_real_tokenizer()
    entry = biogpt_real_entry()
    spec = BG.input_spec(tok, BG.fixed_settings(entry), entry)
    assert spec["answer_ids"] == [22134, 126, 28139] and not spec["answers"]["first_token_rule"]
    assert all(spec["answers"]["words"][w]["same_in_the_target_line"] for w in ("yes", "no", "maybe"))
    assert spec["prompt_ids"] == list(range(57717, 57726)) and spec["prompt_tokens"] == [f"learned{i}</w>" for i in range(1, 10)]
    assert spec["ellipsis_id"] == 11774 and spec["lead_id"] == 2
    assert spec["opening_ids"] == [8, 9367, 20, 8, 2509, 785, 8, 1328, 34]
    assert spec["answer_ids_same_as_fixed_versions"] and spec["prompt_ids_same_as_fixed_versions"]
    for word, i in (("yes", 22134), ("no", 126), ("maybe", 28139)):  # as the training target "... is <answer>." gives them
        ids = tok(f"{BIOGPT_OPENING} {word}.", add_special_tokens=False)["input_ids"]
        assert ids[len(spec["opening_ids"])] == i
    state = {"question": "Is a </s> 5-year follow-up better?", "contexts": [{"section": "S", "text": "word " * 500 + "end."}]}
    seq, n, cut = BG.item_sequence(state, tok, spec)
    assert cut and n > 384 and seq[0] == 2 and seq.count(2) == 1  # the literal "</s>" in the question is text, not eos
    assert seq[384] == 11774 and seq[385:394] == list(range(57717, 57726)) and seq[394:] == spec["opening_ids"]
    tokens = tok.convert_ids_to_tokens(seq[:14])
    assert tokens[:3] == ["</s>", "question</w>", ":</w>"] and "&lt;</w>" in tokens and "@-@</w>" in tokens


def test_biogpt_load_model_reads_the_real_tokenizer_beside_a_tiny_model(tmp_path):
    tok = biogpt_real_tokenizer()
    entry = biogpt_real_entry()
    model_dir = tmp_path / "model"
    biogpt_tiny_model(57726, positions=512).save_pretrained(model_dir)
    for name in ("vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copy(BIOGPT_TOKENIZER_DIR / name, model_dir / name)
    model, tokenizer, about = BG.load_model(model_dir, "float32", "cpu")
    assert type(tokenizer).__name__ == "BioGptTokenizer" and about["max_position_embeddings"] == 512
    assert about["float32_matmul_precision"] == "highest" and about["dtype"] == "float32" and about["gpu"] is None
    spec = BG.input_spec(tokenizer, BG.fixed_settings(entry), entry)
    rows = [biogpt_record(1, 0, "Is it so?", ["Yes, it is. 20 patients were seen."]), biogpt_record(2, 2, "Does it work?", ["Maybe."])]
    probs, counts = BG.score_split(rows, model, tokenizer, spec, about, "cpu", 2)
    assert probs.shape == (2, 3) and np.allclose(probs.sum(1), 1.0) and counts["sources_cut"] == 0
    assert counts["example_input"]["last_tokens"][-18:-9] == [f"learned{i}</w>" for i in range(1, 10)]
    import torch
    if not torch.cuda.is_available():
        with pytest.raises(SystemExit, match="CUDA is not available"):
            BG.load_model(model_dir, "float32", "cuda")


# ======================================================================================================================
# Part 5 of 11: src/eval/reading_checks_session9.py
# From test_reading_checks.py. Module-level names prefixed rc; test names test_reading_checks_*.
#
# Tests for src/eval/reading_checks_session9.py (docs/prompt_session9.md, THE READING CHECKS, COMPARISONS K4, and
# corrections 3 and 6), on made-up predictions and records:
#
# - the runs' raw probabilities averaged (not their votes) and the calibration applied before the argmax: temperature and
#   temperature_intercepts against calibration_family_cv.calibrated and a hand calculation, vector and dirichlet_odir
#   through a stand-in apply_family, imported only when one of them is chosen;
# - each score and share (accuracy, macro-F1, agreement with the full input, the exchange and stay shares, the time-point
#   subset) against a hand calculation, with each interval against a hand resampling of the articles sorted as text;
# - the comparator's regular expression, read on the comparator as written (the exchanged record's intervention);
# - the paired difference in macro-F1, title minus none, on the prompts in both inputs, the same draws for both;
# - every reading's lines (at, just above and just below), the difference reading and what the interval excludes;
# - v0.5: the guard, the swap augmentation caveat, swap averaging (consistent by construction; none and title on the single
#   pass, with --single-pass-key's calibration), and no stop;
# - the stop (exit code 3, its JSON written first), the other failures' exit codes, the prior's reference, and combine.
# ======================================================================================================================

RC_SRC = Path(__file__).resolve().parents[1]
if not (RC_SRC / "eval" / "reading_checks_session9.py").exists():  # run from outside the repository, before the merge
    RC_SRC = Path.cwd() / "src"

RC_SEED = 20260927
RC_SCRIPT = RC_SRC / "eval" / "reading_checks_session9.py"
RC_EXCH = np.array([1, 0, 2])

# the made-up study: 10 prompts in 5 articles, whose PMCIDs sort as text (10, 2, 3, 4, 5)
RC_IDS = [f"evidence-{k}" for k in range(10)]
RC_PMCIDS = ["10", "10", "2", "2", "3", "3", "4", "4", "5", "5"]
RC_Y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 2])
RC_FULL = np.array([0, 1, 2, 0, 1, 2, 0, 0, 2, 1])  # the system's answers on its full input
RC_NONE = np.array([2, 2, 2, 0, 2, 2, 1, 2, 2, 2])
RC_TITLE_KEEP = list(range(9))  # evidence-9's article gives no title
RC_TITLE = np.array([0, 2, 2, 0, 1, 2, 0, 2, 2])
RC_SWAP_KEEP = [k for k in range(10) if k != 5]  # evidence-5's intervention and comparator are the same text
RC_SWAP = np.array([1, 1, 2, 1, 0, 0, 1, 0, 0])  # answers on the exchanged prompts, in RC_SWAP_KEEP's order
RC_INTERVENTIONS = ["drug A"] * 10
RC_COMPARATORS = ["placebo"] * 10
RC_COMPARATORS[3], RC_COMPARATORS[7], RC_COMPARATORS[8] = "Baseline", "pre-treatment values", "before surgery"
RC_INTERVENTIONS[6] = "baseline"  # the intervention as written names a baseline: not in the subset
RC_INTERVENTIONS[5], RC_COMPARATORS[5] = "Drug  B", "drug b"
RC_TIME_ROWS = [3, 6, 7]  # evidence-3, evidence-7 and evidence-8 in RC_SWAP_KEEP's order


def rc_write_jsonl(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def rc_write_probs(path, ids, labels, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p0", "p1", "p2"])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{x:.9g}" for x in p])


def rc_probs_for(answers, top=0.7):
    p = np.full((len(answers), 3), (1 - top) / 2)
    p[np.arange(len(answers)), answers] = top
    return p


def rc_record_line(k, article=""):
    return json.dumps({"id": RC_IDS[k], "qid": "evidence_effect", "label": int(RC_Y[k]), "target": [float(RC_Y[k] == j) for j in range(3)],
                       "state": {"intervention": RC_INTERVENTIONS[k], "comparator": RC_COMPARATORS[k], "outcome": "pain", "article": article},
                       "group": {"source": "evidence_inference_v2", "pmcid": RC_PMCIDS[k], "flag": "none",
                                 "accepted_labels": [int(RC_Y[k])]}}) + "\n"


def rc_f1_macro(y, pred):
    out = []
    for c in sorted(set(np.asarray(y).tolist()) | set(np.asarray(pred).tolist())):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        out.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(out))


def rc_frac(hit, among):
    return float(np.sum(hit & among) / np.sum(among)) if np.sum(among) else float("nan")


def rc_hand_interval(stat, clusters, resamples):
    """Articles sorted as text, drawn with default_rng(RC_SEED), every prompt of a drawn article kept; non-finite left out."""
    arts = sorted(set(clusters))
    rows = {a: [i for i, c in enumerate(clusters) if c == a] for a in arts}
    rng = np.random.default_rng(RC_SEED)
    vals = []
    for _ in range(resamples):
        idx = np.array([i for j in rng.integers(0, len(arts), len(arts)) for i in rows[arts[int(j)]]])
        vals.append(stat(idx))
    vals = np.array(vals, dtype=float)
    v = vals[np.isfinite(vals)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), int(len(v))


def rc_study(tmp_path, none=RC_NONE, title=RC_TITLE, swap=RC_SWAP, full=RC_FULL, family="temperature", params=(math.log(1.46),)):
    """Records, s8_ei's three runs on each input (at the checks_s8 stage's names), its full input, its calibration and
    session 8's comparison.json."""
    checks, res9, res8 = tmp_path / "checks", tmp_path / "results9", tmp_path / "results8"
    rc_write_jsonl(checks / "none" / "evidence_effect_test.jsonl", [rc_record_line(k) for k in range(10)])
    rc_write_jsonl(checks / "title" / "evidence_effect_test.jsonl", [rc_record_line(k, "A made-up title") for k in RC_TITLE_KEEP])
    swapped, same = EI9.swap_check_lines([rc_record_line(k, "the article") for k in range(10)])
    assert same == 1
    rc_write_jsonl(checks / "swap" / "evidence_effect_test.jsonl", swapped)
    answers = {"none": (list(range(10)), RC_Y, np.asarray(none)), "title": (RC_TITLE_KEEP, RC_Y[RC_TITLE_KEEP], np.asarray(title)),
               "swap": (RC_SWAP_KEEP, RC_EXCH[RC_Y[RC_SWAP_KEEP]], np.asarray(swap))}
    for name, (keep, labels, ans) in answers.items():
        base = rc_probs_for(ans)
        for r in range(3):
            p = base.copy()
            if name == "none" and int(ans[0]) == 2:  # evidence-0: two runs of three vote 0, but the mean's argmax is 2
                p[0] = [[0.45, 0.2, 0.35], [0.45, 0.2, 0.35], [0.0, 0.2, 0.8]][r]
            rc_write_probs(res9 / "checks_s8" / f"s8_ei_{r + 1}_{name}-raw_evidence_effect_test.csv", [RC_IDS[k] for k in keep], labels, p)
    full_path = res8 / "predictions" / "s8_ei-raw_evidence_effect_test.csv"
    rc_write_probs(full_path, RC_IDS[::-1], RC_Y[::-1], rc_probs_for(np.asarray(full))[::-1])  # rows in another order: aligned by ID
    cal = res8 / "calibration_s8.json"
    cal.write_text(json.dumps({"plan": "made up", "s8_ei": {"evidence_effect": {
        "chosen": family, "families": {family: {"params": list(params)}, "temperature": {"params": [math.log(1.46)]}}}}}))
    cmp_path = res8 / "comparison.json"
    cmp_path.write_text(json.dumps({"metrics": {"evidence_effect": {"prior": {"n": 1218, "macro_f1": {
        "estimate": 0.1975736568457539, "low": 0.18548019217842512, "high": 0.20953532588178017}}}}}))
    return {"checks": checks, "res9": res9, "res8": res8, "full": full_path, "cal": cal, "comparison": cmp_path}


def rc_s8_args(s, resamples=7, out=None):
    args = ["score", "--system", "s8_ei", "--results", str(s["res9"]), "--full", str(s["full"]), "--calibration", str(s["cal"]),
            "--comparison-s8", str(s["comparison"]), "--resamples", str(resamples)]
    for i in ("none", "title", "swap"):
        args += [f"--records-{i}", str(s["checks"] / i / "evidence_effect_test.jsonl")]
    return args + (["--out", str(out)] if out else [])


# the average and the calibration --------------------------------------------------------------------------------------

def test_reading_checks_runs_average_their_probabilities_then_the_calibration(tmp_path):
    rec = RCHK.load_records(rc_study(tmp_path)["checks"] / "none" / "evidence_effect_test.jsonl")
    paths = [tmp_path / "results9" / "checks_s8" / f"s8_ei_{r}_none-raw_evidence_effect_test.csv" for r in (1, 2, 3)]
    mean = RCHK.average_runs(paths, rec)
    assert mean[0] == pytest.approx([0.3, 0.2, 0.5], abs=1e-9)  # the mean of the three runs, not their vote (0)
    assert mean[1:] == pytest.approx(rc_probs_for(RC_NONE[1:]), abs=1e-9)
    t, b = 1.46, (0.0, -0.3, 0.4)
    cal_t = {"family": "temperature", "params": [math.log(t)]}
    cal_ti = {"family": "temperature_intercepts", "params": [math.log(t), b[1], b[2]]}
    z = np.log(mean) / t
    hand = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    assert RCHK.calibrated(mean, cal_t) == pytest.approx(hand, abs=1e-12)
    z2 = np.log(mean) / t + np.array(b)
    assert RCHK.calibrated(mean, cal_ti) == pytest.approx(np.exp(z2) / np.exp(z2).sum(1, keepdims=True), abs=1e-12)
    import calibration_family_cv as CFC  # the formula calibrate_session8.py applies

    logp = np.log(np.maximum(mean, 1e-300))
    assert RCHK.calibrated(mean, cal_ti) == pytest.approx(CFC.calibrated(logp, cal_ti["params"], 3), abs=1e-15)
    assert RCHK.calibrated(mean, cal_t) == pytest.approx(CFC.calibrated(logp, cal_t["params"], 3), abs=1e-15)
    clipped = RCHK.calibrated(mean, {"family": "temperature", "params": [math.log(50.0)]})  # T is clipped to 10
    assert clipped == pytest.approx(RCHK.calibrated(mean, {"family": "temperature", "params": [math.log(10.0)]}), abs=1e-15)
    with pytest.raises(SystemExit):
        RCHK.calibrated(mean, {"family": "temperature_intercepts", "params": [0.0]})
    # an intercept moves an answer: the calibration is applied before the argmax
    row = np.array([[0.40, 0.35, 0.25]])
    assert RCHK.calibrated(row, {"family": "temperature_intercepts", "params": [0.0, 0.0, 1.0]}).argmax(1)[0] == 2
    assert RCHK.calibrated(row, cal_t).argmax(1)[0] == 0


def test_reading_checks_vector_and_dirichlet_go_through_apply_family_imported_lazily(tmp_path, monkeypatch):
    code = ("import sys; sys.path.insert(0, %r); import reading_checks_session9; print('calibration_family_cv' in sys.modules)"
            % str(RC_SRC / "eval"))
    got = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path))
    assert got.returncode == 0 and got.stdout.strip() == "False", got.stderr
    import calibration_family_cv as CFC

    calls = []

    def stand_in(name, params, logp):
        calls.append((name, params, logp.copy()))
        z = 2.0 * logp
        return np.exp(z) / np.exp(z).sum(1, keepdims=True)

    monkeypatch.setattr(CFC, "apply_family", stand_in, raising=False)
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    for family in ("vector", "dirichlet_odir"):
        out = RCHK.calibrated(p, {"family": family, "params": {"made_up": [1, 2]}})
        assert calls[-1][0] == family and calls[-1][1] == {"made_up": [1, 2]}
        assert calls[-1][2] == pytest.approx(np.log(p)) and out == pytest.approx(p ** 2 / (p ** 2).sum(1, keepdims=True))
    monkeypatch.setattr(CFC, "apply_family", lambda name, params, logp: logp[:, :2], raising=False)
    with pytest.raises(SystemExit):
        RCHK.calibrated(p, {"family": "vector", "params": []})
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"x": {"evidence_effect": {"chosen": "beta", "families": {"beta": {"params": [1]}}}}}))
    with pytest.raises(SystemExit):
        RCHK.calibration_entry(cal, "x")
    with pytest.raises(SystemExit):
        RCHK.calibration_entry(cal, "missing")


def test_reading_checks_real_apply_family_and_the_log_floor(tmp_path):
    import calibration_family_cv as CFC

    if not hasattr(CFC, "apply_family"):
        pytest.skip("calibration_family_cv.apply_family is not written yet")
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.1, 0.8]])
    assert RCHK.calibrated(p, {"family": "vector", "params": [1.0, 1.0, 1.0, 0.0, 0.0]}) == pytest.approx(p)
    identity = {"W": np.eye(3).tolist(), "b": [0.0, 0.0, 0.0], "lambda_od": 0.01, "lambda_b": 0.01}
    assert RCHK.calibrated(p, {"family": "dirichlet_odir", "params": identity}) == pytest.approx(p)
    tiny = np.array([[1e-20, 0.5, 0.5]])
    doubled = {"family": "vector", "params": [2.0, 1.0, 1.0, 0.0, 0.0]}  # answer 0's log-probability doubled
    assert RCHK.calibrated(tiny, doubled)[0, 0] == pytest.approx(1e-40, rel=1e-6)
    assert RCHK.calibrated(tiny, doubled, log_floor=1e-12)[0, 0] == pytest.approx(1e-24, rel=1e-6)
    s = rc_study(tmp_path)
    assert RCHK.main(rc_s8_args(s) + ["--log-floor", "1e-12"]) == 0
    assert json.loads((s["res9"] / "checks_s8.json").read_text())["settings"]["log_floor"] == 1e-12
    with pytest.raises(SystemExit):
        RCHK.main(rc_s8_args(s) + ["--log-floor", "0"])


# the scores, the shares and their intervals ---------------------------------------------------------------------------

def test_reading_checks_scores_shares_and_intervals_against_a_hand_calculation(tmp_path):
    s = rc_study(tmp_path)
    assert RCHK.main(rc_s8_args(s, resamples=7)) == 0
    out = json.loads((s["res9"] / "checks_s8.json").read_text())
    assert out["plan"] == "docs/prompt_session9.md, THE READING CHECKS" and out["written_utc"].endswith("Z")
    assert out["stop"] is False and out["system"] == "s8_ei" and out["settings"]["seed"] == RC_SEED
    assert out["description"] == "s8_ei (its three runs' raw probabilities averaged, then session 8's calibration)"
    clusters = np.array(RC_PMCIDS)
    # none: every prompt
    none = out["inputs"]["none"]
    assert none["n"] == 10 and none["articles"] == 5
    assert none["accuracy"]["estimate"] == pytest.approx(0.5) and none["macro_f1"]["estimate"] == pytest.approx((0.5 + 0 + 2 / 3) / 3)
    assert none["agreement_with_full_input"]["estimate"] == pytest.approx(0.4) and none["agreement_with_full_input"]["count"] == 4
    for key, stat in (("accuracy", lambda i: np.mean(RC_NONE[i] == RC_Y[i])), ("macro_f1", lambda i: rc_f1_macro(RC_Y[i], RC_NONE[i])),
                      ("agreement_with_full_input", lambda i: np.mean(RC_NONE[i] == RC_FULL[i]))):
        low, high, _ = rc_hand_interval(stat, list(clusters), 7)
        assert none[key]["low"] == pytest.approx(low) and none[key]["high"] == pytest.approx(high)
    # title: the prompts whose article gives a title
    title, yt, ft = out["inputs"]["title"], RC_Y[RC_TITLE_KEEP], RC_FULL[RC_TITLE_KEEP]
    assert title["n"] == 9 and title["left_out_no_title"] == 1 and title["articles"] == 5
    assert title["accuracy"]["estimate"] == pytest.approx(7 / 9) and title["macro_f1"]["estimate"] == pytest.approx(0.75)
    assert title["agreement_with_full_input"]["estimate"] == pytest.approx(7 / 9)
    low, high, _ = rc_hand_interval(lambda i: rc_f1_macro(yt[i], RC_TITLE[i]), list(clusters[RC_TITLE_KEEP]), 7)
    assert title["macro_f1"]["low"] == pytest.approx(low) and title["macro_f1"]["high"] == pytest.approx(high)
    # swap: accuracy against the exchanged answers, and the shares against the answers on the full input
    swap, ys, fs = out["inputs"]["swap"], RC_EXCH[RC_Y[RC_SWAP_KEEP]], RC_FULL[RC_SWAP_KEEP]
    assert swap["n"] == 9 and swap["left_out_same_text"] == 1
    assert swap["accuracy"]["estimate"] == pytest.approx(4 / 9) and swap["macro_f1"]["estimate"] == pytest.approx(rc_f1_macro(ys, RC_SWAP))
    assert (swap["exchange_share"]["count"], swap["exchange_share"]["of"]) == (5, 7)
    assert swap["exchange_share"]["estimate"] == pytest.approx(5 / 7)
    assert (swap["stay_share"]["count"], swap["stay_share"]["of"]) == (1, 2) and swap["stay_share"]["estimate"] == pytest.approx(0.5)
    sc = list(clusters[RC_SWAP_KEEP])
    for key, stat in (("exchange_share", lambda i: rc_frac(RC_SWAP[i] == RC_EXCH[fs[i]], fs[i] != 2)),
                      ("stay_share", lambda i: rc_frac(RC_SWAP[i] == 2, fs[i] == 2))):
        low, high, used = rc_hand_interval(stat, sc, 7)
        assert swap[key]["low"] == pytest.approx(low) and swap[key]["high"] == pytest.approx(high)
        assert swap[key]["resamples_defined"] == used and swap[key]["resamples"] == 7
    # the time-point subset: the comparator as written, its own articles resampled
    tp = swap["time_point_or_baseline"]
    assert tp["n"] == 3 and tp["articles"] == 3 and tp["pattern"] == RCHK.TIME_POINT.pattern
    assert tp["exchange_share"]["estimate"] == pytest.approx(1.0) and tp["stay_share"]["estimate"] == pytest.approx(0.0)
    f_t, s_t, c_t = fs[RC_TIME_ROWS], RC_SWAP[RC_TIME_ROWS], [sc[r] for r in RC_TIME_ROWS]
    low, high, used = rc_hand_interval(lambda i: rc_frac(s_t[i] == 2, f_t[i] == 2), c_t, 7)
    assert tp["stay_share"]["low"] == pytest.approx(low) and tp["stay_share"]["resamples_defined"] == used
    # the references and the notes
    prior = out["references"]["prior"]
    assert prior["macro_f1"]["estimate"] == pytest.approx(0.1975736568457539) and "0.198" in prior["text"]
    assert out["references"]["windisch_2026"]["accuracy_low"] == 0.79 and out["references"]["windisch_2026"]["majority_class_accuracy"] == 0.584
    assert "EI's test split was read in session 8" in out["contamination"]
    assert [f["file"].endswith(f"s8_ei_{k}_none-raw_evidence_effect_test.csv") for k, f in
            zip((1, 2, 3), out["files"]["runs"]["none"])] == [True, True, True]


def test_reading_checks_comparator_regular_expression():
    match = ["Baseline", "at baseline", "post-baseline", "before surgery", "BEFORE", "pre-treatment", "Pre-operative values",
             "pre" + chr(0x2013) + "alvimopan", "pre" + chr(0x2010) + "dose", "pretreatment", "prior to randomization",
             "PRIOR  TO dosing", "start of therapy", "Started on day 1"]
    no_match = ["placebo", "preoperative", "previous treatment", "post-treatment", "usual care", "priority group", "pre treatment"]
    assert all(RCHK.TIME_POINT.search(t) for t in match)
    assert not any(RCHK.TIME_POINT.search(t) for t in no_match)
    assert RCHK.TIME_POINT.flags & __import__("re").IGNORECASE
    # the subset reads the comparator as written, which the exchanged record holds as its intervention
    rec = json.loads(rc_record_line(3))
    swapped = EI9.swapped(rec, suffix="")
    assert swapped["state"]["intervention"] == "Baseline" and RCHK.TIME_POINT.search(swapped["state"]["intervention"])
    rec6 = EI9.swapped(json.loads(rc_record_line(6)), suffix="")
    assert not RCHK.TIME_POINT.search(rec6["state"]["intervention"])  # its intervention as written is "baseline"


def test_reading_checks_title_minus_none_on_the_prompts_in_both(tmp_path):
    s = rc_study(tmp_path)
    assert RCHK.main(rc_s8_args(s, resamples=7)) == 0
    t = json.loads((s["res9"] / "checks_s8.json").read_text())["title_minus_none"]
    yc, nc = RC_Y[RC_TITLE_KEEP], RC_NONE[RC_TITLE_KEEP]
    assert t["n"] == 9 and t["articles"] == 5
    assert t["title_macro_f1"]["estimate"] == pytest.approx(0.75)
    assert t["none_macro_f1"]["estimate"] == pytest.approx((0.5 + 0 + 0.6) / 3)  # none on the common prompts only
    assert t["macro_f1_difference"]["estimate"] == pytest.approx(0.75 - 1.1 / 3)
    low, high, _ = rc_hand_interval(lambda i: rc_f1_macro(yc[i], RC_TITLE[i]) - rc_f1_macro(yc[i], nc[i]), RC_PMCIDS[:9], 7)
    assert t["macro_f1_difference"]["low"] == pytest.approx(low) and t["macro_f1_difference"]["high"] == pytest.approx(high)
    assert t["difference_reading"] == RCHK.difference_reading(t["macro_f1_difference"]["low"], t["macro_f1_difference"]["high"])
    assert "excludes" in t["excludes"]


# the readings ---------------------------------------------------------------------------------------------------------

def test_reading_checks_reading_lines_at_just_above_and_just_below():
    e = 1e-9
    assert RCHK.none_reading(0.60) == RCHK.NONE_DEPENDS and RCHK.none_reading(0.60 - e) == RCHK.NONE_DEPENDS
    assert RCHK.none_reading(0.60 + e) == "partly" and RCHK.none_reading(0.75) == "partly" and RCHK.none_reading(0.75 - e) == "partly"
    assert RCHK.none_reading(0.75 + e) == RCHK.NONE_PROMPT
    assert RCHK.NONE_DEPENDS == "its answers depend on the article's text"
    assert RCHK.NONE_PROMPT.startswith("the prompt alone carries more than a classifier trained on prompts alone (0.511 macro-F1")
    assert RCHK.title_reading(0.05 + e, 0.2) == RCHK.TITLE_ADDS and RCHK.title_reading(0.05, 0.2) == "not settled"
    assert RCHK.title_reading(0.05 - e, 0.2) == "not settled"
    assert RCHK.title_reading(-0.1, 0.05 - e) == RCHK.TITLE_NONE and RCHK.title_reading(-0.1, 0.05) == "not settled"
    assert RCHK.title_reading(-0.1, 0.05 + e) == "not settled" and RCHK.title_reading(0.01, 0.04) == RCHK.TITLE_NONE
    assert RCHK.TITLE_NONE == "no added knowledge of 0.05 or more was detected"
    assert RCHK.swap_reading(0.85) == RCHK.SWAP_FOLLOWS and RCHK.swap_reading(0.85 + e) == RCHK.SWAP_FOLLOWS
    assert RCHK.swap_reading(0.85 - e) == "partly" and RCHK.swap_reading(0.70) == "partly" and RCHK.swap_reading(0.70 + e) == "partly"
    assert RCHK.swap_reading(0.70 - e) == RCHK.SWAP_NOT
    assert RCHK.SWAP_FOLLOWS == "it follows the comparison as written"
    assert RCHK.SWAP_NOT == "it does not follow the exchange (recall of the trial or a bias toward one direction)"
    assert RCHK.stops(0.85) is False and RCHK.stops(0.85 - e) is False and RCHK.stops(0.85 + e) is True and RCHK.stops(None) is False
    assert RCHK.difference_reading(-0.01, 0.03) == "no difference was detected"
    assert RCHK.difference_reading(0.01, 0.03).startswith("higher") and RCHK.difference_reading(-0.03, -0.01).startswith("lower")
    text = RCHK.excludes_text(0.01, -0.012, 0.043)
    assert "-0.012 to 0.043" in text and "zero is inside it" in text and "it excludes 0.05 (above the interval)" in text
    assert "0.05 is inside it" in RCHK.excludes_text(0.06, 0.02, 0.09)


# v0.5 -----------------------------------------------------------------------------------------------------------------

def rc_v05_files(tmp_path, s, full_probs=None, swap_probs=None, none=None):
    """v0.5's single runs (the weight average) on each input and on its full input, its calibration and the freeze's files."""
    folder = tmp_path / "checks_v05"
    labels = {"none": RC_Y, "title": RC_Y[RC_TITLE_KEEP], "swap": RC_EXCH[RC_Y[RC_SWAP_KEEP]]}
    keep = {"none": list(range(10)), "title": RC_TITLE_KEEP, "swap": RC_SWAP_KEEP}
    probs = {"none": rc_probs_for(np.asarray(none if none is not None else RC_NONE)), "title": rc_probs_for(RC_TITLE),
             "swap": swap_probs if swap_probs is not None else rc_probs_for(RC_SWAP, 0.6)}
    runs = {}
    for i in ("none", "title", "swap"):
        runs[i] = folder / f"s9_ei_wavg_{i}-raw_evidence_effect_test.csv"
        rc_write_probs(runs[i], [RC_IDS[k] for k in keep[i]], labels[i], probs[i])
    full = tmp_path / "s9_ei_wavg-raw_evidence_effect_test.csv"
    rc_write_probs(full, RC_IDS, RC_Y, full_probs if full_probs is not None else rc_probs_for(RC_FULL, 0.8))
    cal = s["res9"] / "calibration_s9.json"
    cal.write_text(json.dumps({"s9_ei_wavg": {"evidence_effect": {"chosen": "temperature", "families": {"temperature": {"params": [0.2]}}}},
                               "s9_ei_wavg_single": {"evidence_effect": {"chosen": "temperature_intercepts", "families": {
                                   "temperature_intercepts": {"params": [0.0, 0.0, 5.0]}}}}}))
    (s["res9"] / "release_candidate.json").write_text("{}")
    (s["res9"] / "testpreds_log.json").write_text("{}")
    args = ["score", "--system", "v05", "--name", "s9_ei_wavg", "--results", str(s["res9"]), "--full", str(full),
            "--comparison-s8", str(s["comparison"]), "--resamples", "7"]
    for i in ("none", "title", "swap"):
        args += [f"--runs-{i}", str(runs[i]), f"--records-{i}", str(s["checks"] / i / "evidence_effect_test.jsonl")]
    return args


def test_reading_checks_v05_guard_caveats_and_no_stop(tmp_path):
    s = rc_study(tmp_path)
    args = rc_v05_files(tmp_path, s, none=RC_Y)  # every answer with no article right: macro-F1 1, but the stop is s8_ei's only
    moved = s["res9"] / "testpreds_log.json"
    moved.rename(s["res9"] / "testpreds_log.json.away")
    with pytest.raises(SystemExit):
        RCHK.main(args)
    (s["res9"] / "testpreds_log.json.away").rename(moved)
    assert RCHK.main(args) == 0
    out = json.loads((s["res9"] / "checks_v05.json").read_text())
    assert out["system"] == "v05" and out["name"] == "s9_ei_wavg" and out["stop"] is False
    assert out["inputs"]["none"]["macro_f1"]["estimate"] == pytest.approx(1.0) and out["readings"]["none"]["reading"] == RCHK.NONE_PROMPT
    assert all(not out["readings"][i]["caveats"] for i in ("none", "title", "swap"))
    assert out["readings"]["swap"]["reading"] == RCHK.swap_reading(out["inputs"]["swap"]["accuracy"]["estimate"])
    assert "its one run" in out["description"] and out["calibration"]["own"]["key"] == "s9_ei_wavg"
    assert RCHK.main(args + ["--swap-augmented", "--out", str(tmp_path / "aug.json")]) == 0
    aug = json.loads((tmp_path / "aug.json").read_text())
    assert aug["readings"]["swap"]["caveats"] == [RCHK.CAVEAT_AUGMENTED] and aug["flags"]["swap_augmented"] is True
    assert aug["readings"]["swap"]["reading"] == out["readings"]["swap"]["reading"]
    assert not aug["readings"]["none"]["caveats"] and not aug["readings"]["title"]["caveats"]
    with pytest.raises(SystemExit):  # an argument error: v0.5's flags on s8_ei
        RCHK.main(rc_s8_args(s) + ["--swap-augmented"])


def test_reading_checks_v05_swap_averaging_is_consistent_by_construction(tmp_path):
    s = rc_study(tmp_path)
    full = rc_probs_for(RC_FULL, 0.8)
    full[1] = [0.3, 0.4, 0.3]  # evidence-1: its single pass answers "decreased", and with swap averaging "no difference"
    swap = rc_probs_for(RC_SWAP, 0.6)
    swap[1] = [0.05, 0.05, 0.9]
    args = rc_v05_files(tmp_path, s, full_probs=full, swap_probs=swap)
    assert RCHK.main(args + ["--swap-averaged"]) == 0
    out = json.loads((s["res9"] / "checks_v05.json").read_text())
    sw = out["inputs"]["swap"]
    averaged = (full[RC_SWAP_KEEP] + swap[:, RC_EXCH]) / 2  # its answer to each prompt as written, before calibration
    f_avg = averaged.argmax(1)
    assert f_avg[1] == 2 and RC_FULL[1] == 1
    assert sw["exchange_share"]["estimate"] == 1.0 and sw["stay_share"]["estimate"] == 1.0
    assert sw["accuracy"]["estimate"] == pytest.approx(np.mean(RC_EXCH[f_avg] == RC_EXCH[RC_Y[RC_SWAP_KEEP]]))
    assert sw["full_input_answers"]["no_difference"] == int(np.sum(f_avg == 2))
    assert out["readings"]["swap"]["reading"] == RCHK.CONSISTENT and out["readings"]["swap"]["read_as_a_reading_check"] is False
    assert out["readings"]["none"]["caveats"] == [RCHK.CAVEAT_SINGLE_PASS] == out["readings"]["title"]["caveats"]
    # none and title against the single pass on the full input (evidence-1: "decreased", so the none answer 2 disagrees)
    assert out["inputs"]["none"]["agreement_with_full_input"]["count"] == int(np.sum(RC_NONE == RC_FULL))
    # the identity: the swap-averaged answer to an exchanged prompt is the exchange of the answer to the prompt
    cal = {"family": "temperature", "params": [0.2]}
    to_prompt = RCHK.calibrated(averaged, cal)
    to_exchanged = RCHK.calibrated((swap + full[RC_SWAP_KEEP][:, RC_EXCH]) / 2, cal)
    assert to_exchanged == pytest.approx(RCHK.exchange(to_prompt), abs=1e-12)
    # --single-pass-key: the single pass is calibrated by the single-pass form's calibration (b2 = 5: every answer 2)
    assert RCHK.main(args + ["--swap-averaged", "--single-pass-key", "s9_ei_wavg_single", "--out", str(tmp_path / "sp.json")]) == 0
    sp = json.loads((tmp_path / "sp.json").read_text())
    assert sp["calibration"]["single_pass"]["key"] == "s9_ei_wavg_single" and sp["calibration"]["applied_to"]["none"] == "single pass"
    assert sp["inputs"]["none"]["full_input_answers"]["no_difference"] == 10 and sp["inputs"]["none"]["answers"]["no_difference"] == 10
    assert sp["inputs"]["none"]["agreement_with_full_input"]["estimate"] == 1.0
    assert sp["inputs"]["swap"]["exchange_share"]["estimate"] == 1.0  # the swap check keeps v0.5's own calibration
    with pytest.raises(SystemExit):
        RCHK.main(args + ["--single-pass-key", "s9_ei_wavg_single"])  # only with --swap-averaged


# the stop, the exit codes, the references and combine ------------------------------------------------------------------

def test_reading_checks_the_stop_writes_its_json_then_exits_with_code_3(tmp_path):
    s = rc_study(tmp_path, none=RC_Y)  # every answer with no article right: macro-F1 1, above 0.85
    out_path = tmp_path / "checks_s8.json"
    assert RCHK.main(rc_s8_args(s, out=out_path)) == RCHK.STOP_EXIT == 3
    out = json.loads(out_path.read_text())
    assert out["stop"] is True and out["stop_numbers"]["none_macro_f1"]["estimate"] == pytest.approx(1.0)
    assert out["inputs"]["swap"]["n"] == 9  # the numbers are all there
    run = subprocess.run([sys.executable, str(RC_SCRIPT)] + rc_s8_args(s, out=tmp_path / "again.json"), capture_output=True, text=True)
    assert run.returncode == 3 and "STOP" in run.stdout and json.loads((tmp_path / "again.json").read_text())["stop"] is True
    # no stop: exit code 0
    s2 = rc_study(tmp_path / "b")
    run = subprocess.run([sys.executable, str(RC_SCRIPT)] + rc_s8_args(s2), capture_output=True, text=True)
    assert run.returncode == 0 and json.loads((s2["res9"] / "checks_s8.json").read_text())["stop"] is False
    # another failure: a run whose answers differ from the records (exit code 1), and an argument error (exit code 2)
    bad = s2["res9"] / "checks_s8" / "s8_ei_2_title-raw_evidence_effect_test.csv"
    rc_write_probs(bad, [RC_IDS[k] for k in RC_TITLE_KEEP], (RC_Y[RC_TITLE_KEEP] + 1) % 3, rc_probs_for(RC_TITLE))
    run = subprocess.run([sys.executable, str(RC_SCRIPT)] + rc_s8_args(s2, out=tmp_path / "bad.json"), capture_output=True, text=True)
    assert run.returncode not in (0, 3) and not (tmp_path / "bad.json").exists()
    run = subprocess.run([sys.executable, str(RC_SCRIPT), "score", "--system", "v05"], capture_output=True, text=True)
    assert run.returncode not in (0, 3)
    # a missing run file
    s3 = rc_study(tmp_path / "c")
    (s3["res9"] / "checks_s8" / "s8_ei_3_swap-raw_evidence_effect_test.csv").rename(tmp_path / "away.csv")
    with pytest.raises(SystemExit):
        RCHK.main(rc_s8_args(s3))
    # the stop reads the calibrated answers: an intercept that turns every answer into "no difference" leaves no stop
    s4 = rc_study(tmp_path / "d", none=RC_Y, family="temperature_intercepts", params=(0.0, 0.0, 5.0))
    assert RCHK.main(rc_s8_args(s4)) == 0
    out4 = json.loads((s4["res9"] / "checks_s8.json").read_text())
    assert out4["calibration"]["own"]["family"] == "temperature_intercepts" and out4["stop"] is False
    assert out4["inputs"]["none"]["answers"]["no_difference"] == 10 and out4["inputs"]["none"]["macro_f1"]["estimate"] < 0.85


def test_reading_checks_prior_reference_and_combine(tmp_path):
    s = rc_study(tmp_path)
    assert RCHK.main(rc_s8_args(s)) == 0
    args = rc_v05_files(tmp_path, s)
    assert RCHK.main(args + ["--swap-augmented"]) == 0
    out = RCHK.combine(s["res9"] / "checks_s8.json", s["res9"] / "checks_v05.json", s["comparison"], s["res9"] / "reading_checks.json")
    saved = json.loads((s["res9"] / "reading_checks.json").read_text())
    assert saved["plan"] == "docs/prompt_session9.md, THE READING CHECKS and COMPARISONS (K4)" and saved["written_utc"].endswith("Z")
    assert set(saved["systems"]) == {"s8_ei", "v05"} and saved["same_none_and_title_records"] == {"none": True, "title": True}
    assert saved["readings"]["v05"]["swap"]["caveats"] == [RCHK.CAVEAT_AUGMENTED]
    assert saved["references"]["prior"]["macro_f1"]["estimate"] == pytest.approx(0.1975736568457539)
    rows = [r for r in saved["table"] if r["system"] == "s8_ei" and r["input"] == "none" and r["statistic"] == "macro_f1"]
    assert len(rows) == 1 and rows[0]["estimate"] == pytest.approx(out["systems"]["s8_ei"]["inputs"]["none"]["macro_f1"]["estimate"])
    assert any(r["input"].startswith("swap, comparator") for r in saved["table"])
    # the prior's file absent (as on the pod): recorded as not read by score, read again by combine
    missing = tmp_path / "no_comparison.json"
    assert RCHK.main(rc_s8_args(s, out=tmp_path / "pod.json") + ["--comparison-s8", str(missing)]) == 0
    pod = json.loads((tmp_path / "pod.json").read_text())
    assert "not_read" in pod["references"]["prior"]
    again = RCHK.combine(tmp_path / "pod.json", s["res9"] / "checks_v05.json", s["comparison"], tmp_path / "rc.json")
    assert again["references"]["prior"]["macro_f1"]["estimate"] == pytest.approx(0.1975736568457539)
    with pytest.raises(SystemExit):
        RCHK.combine(s["res9"] / "checks_s8.json", tmp_path / "absent.json", s["comparison"], tmp_path / "x.json")
    with pytest.raises(SystemExit):  # the files the other way round
        RCHK.combine(s["res9"] / "checks_v05.json", s["res9"] / "checks_s8.json", s["comparison"], tmp_path / "y.json")
    assert RCHK.main(["combine", "--checks-s8", str(s["res9"] / "checks_s8.json"), "--checks-v05", str(s["res9"] / "checks_v05.json"),
                    "--comparison-s8", str(s["comparison"]), "--out", str(tmp_path / "z.json")]) == 0


# ======================================================================================================================
# Part 6 of 11: src/eval/calibration_family_cv.py (the four calibration families)
# From test_calibration_families.py. Module-level names prefixed calfam; test names test_calfam_*.
#
# Tests for the four calibration families of src/eval/calibration_family_cv.py (docs/prompt_session9.md, CALIBRATION
# and step 4), on made-up inputs.
#
# - Each family recovers a planted calibration map (a temperature, intercepts, a scale for each answer, a full matrix): the
#   model's log-probabilities are drawn first and the labels from known probabilities, the planted map applied to them, so
#   the model's log-probabilities are the known ones distorted by the map's inverse. The matching family's out-of-fold log
#   loss is the lowest or within 0.005 of it, the families that cannot express the map score at least 0.01 worse, and its
#   parameters fitted on all items are close to the planted ones.
# - The vector family's free parameters: adding the same constant to every intercept changes nothing, b_0 is fixed at 0
#   (a planted b_0 of 0.7 comes back as the other intercepts less 0.7), and it takes 2k - 1 numbers.
# - dirichlet_odir's objective has the stated form and scale, and both new families' gradients match finite differences.
# - The tie rules: within 1e-12 of the lowest is a tie, which goes to the first listed, for the families and for the penalty
#   pairs, on made-up scores and on items where every fit stays at its start.
# - out_of_fold() draws the fold sets main() draws, refits on each training part (dirichlet_odir with its own seeded inner
#   folds) and returns the probabilities of every repeat.
# - apply_family() for temperature and temperature_intercepts matches calibrate_session8.py's application of made-up
#   parameters stored as calibration_s8.json stores them.
# - out_of_fold() for those two gives exactly the numbers the file's own command writes on the same input.
# - The command's output (its JSON, its printout and its help) is unchanged against the file as it was before session 9.
# ======================================================================================================================

CALFAM_HERE = Path(__file__).resolve()
CALFAM_SRC = CALFAM_HERE.parents[1] if (CALFAM_HERE.parents[1] / "eval" / "calibration_family_cv.py").exists() else Path.cwd() / "src"
CALFAM_ROOT = CALFAM_SRC.parent

CALFAM_SEED = 20260927
# calibration_family_cv.py before session 9 changed it: its SHA-256 (as registered for v0.4, docs/osf_prereg_fiorillo_v0_4.md),
# the commit that last changed it before, and the copy saved before the change
CALFAM_ORIGINAL_SHA256 = "295e5f60a2f3ccdf10bec332ee2a7ca2c24bae3be9d2939a98932160398c122f"
CALFAM_ORIGINAL_COMMIT = "cacc61cf8027d14e7b9d1a4abeb079b59b0f1ff1"
CALFAM_ORIGINAL_COPY = Path("/tmp/eddm_s9/agent_families/calibration_family_cv.orig.py")
CALFAM_W_PLANT = [[1.3, 0.5, -0.4], [-0.6, 0.8, 0.4], [0.6, -0.5, 1.6]]  # each column's off-diagonal entries sum to 0
CALFAM_B_PLANT = [0.3, -0.5, 0.2]  # mean 0; with CALFAM_W_PLANT, the representative the penalties pick, so the fit should return these


def calfam_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calfam_log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def calfam_draw(q, rng):
    """One label for each row of probabilities q."""
    return np.minimum((rng.random(len(q))[:, None] > np.cumsum(q, axis=1)).sum(axis=1), q.shape[1] - 1)


def calfam_planted_map(family, params, logp):
    """The known probabilities: the planted map, written out here rather than through the code under test."""
    if family == "temperature":
        return calfam_softmax(logp / np.exp(params[0]))
    if family == "temperature_intercepts":
        return calfam_softmax(logp / np.exp(params[0]) + np.array([0.0] + list(params[1:])))
    if family == "vector":
        return calfam_softmax(logp * np.array(params[:3]) + np.array([0.0] + list(params[3:])))
    w, b = np.array(params["W"]), np.array(params["b"])
    z = np.column_stack([sum(w[j, i] * logp[:, i] for i in range(3)) + b[j] for j in range(3)])  # answer j: sum_i W[j][i] log p_i
    return calfam_softmax(z)


def calfam_planted(family, params, n, seed):
    rng = np.random.default_rng(seed)
    logp = calfam_log_softmax(rng.normal(size=(n, 3)) * 2.0)
    q = calfam_planted_map(family, params, logp)
    return logp, calfam_draw(q, rng), q


def calfam_flat(family, params):
    if family == "dirichlet_odir":
        return np.concatenate((np.ravel(params["W"]), params["b"]))
    return np.asarray(params, dtype=float)


def calfam_check_plant(family, params, tolerance, cannot):
    """Out-of-fold on 1,500 items (5 folds, 1 repeat) and the parameters fitted on 12,000 items of one planted map."""
    logp, y, q = calfam_planted(family, params, 12000, 41)
    result, probs = CF.choose(logp[:1500], y[:1500], repeats=1)
    oof = {f: result["families"][f]["out_of_fold_log_loss"]["mean"] for f in CF.FAMILIES}
    assert oof[family] <= min(oof.values()) + 0.005, oof
    assert oof[result["chosen"]] <= min(oof.values()) + 1e-12
    for other in cannot:
        assert oof[other] > oof[family] + 0.01, (other, oof)
    assert all(p.shape == (1, 1500, 3) for p in probs.values())
    fitted = CF.fit_family(family, logp, y)
    assert np.abs(calfam_flat(family, fitted) - calfam_flat(family, params)).max() < tolerance, (fitted, params)
    assert np.abs(CF.apply_family(family, fitted, logp) - q).mean() < 0.01
    return fitted


def test_calfam_temperature_recovers_a_planted_temperature():
    calfam_check_plant("temperature", [np.log(2.0)], 0.05, ())


def test_calfam_temperature_intercepts_recovers_planted_intercepts():
    calfam_check_plant("temperature_intercepts", [np.log(1.5), 0.8, -0.6], 0.1, ("temperature",))


def test_calfam_vector_recovers_planted_scales():
    calfam_check_plant("vector", [0.5, 1.0, 1.8, 0.4, -0.3], 0.2, ("temperature", "temperature_intercepts"))


def test_calfam_dirichlet_odir_recovers_a_planted_matrix():
    fitted = calfam_check_plant("dirichlet_odir", {"W": CALFAM_W_PLANT, "b": CALFAM_B_PLANT}, 0.25, ("temperature", "temperature_intercepts", "vector"))
    assert set(fitted) == {"W", "b", "lambda_od", "lambda_b"}
    assert fitted["lambda_od"] in CF.ODIR_GRID and fitted["lambda_b"] in CF.ODIR_GRID
    assert fitted["lambda_od"] < 1.0  # the off-diagonal entries carry the map, so the strongest penalty is not chosen
    w = np.array(fitted["W"])
    # the representative the penalties pick: each column's off-diagonal entries and the intercepts sum to 0 (to the optimizer's tolerance)
    assert np.allclose([w[1, 0] + w[2, 0], w[0, 1] + w[2, 1], w[0, 2] + w[1, 2]], 0.0, atol=1e-3)
    assert abs(sum(fitted["b"])) < 1e-3
    json.dumps(fitted)  # stored as plain numbers


def test_calfam_vector_free_parameters():
    rng = np.random.default_rng(8)
    logp = calfam_log_softmax(rng.normal(size=(30, 3)) * 2.0)
    params = [0.7, 1.2, 1.9, 0.5, -0.8]
    got = CF.apply_family("vector", params, logp)
    for shift in (0.0, 3.7, -12.5):  # the same constant added to every intercept, b_0 included, changes nothing
        assert np.allclose(got, calfam_softmax(logp * np.array(params[:3]) + np.array([0.0, 0.5, -0.8]) + shift), rtol=0, atol=1e-14)
    # b_0 is fixed at 0: a map with intercepts (0.7, 0.2, -0.4) is the map with (0, -0.5, -1.1), and is fitted as that
    rng = np.random.default_rng(9)
    logp = calfam_log_softmax(rng.normal(size=(50000, 3)) * 2.0)
    y = calfam_draw(calfam_softmax(logp * np.array([0.6, 1.0, 1.5]) + np.array([0.7, 0.2, -0.4])), rng)
    fitted = CF.fit_family("vector", logp, y)
    assert len(fitted) == 5 and all(isinstance(v, float) for v in fitted)
    assert np.abs(np.array(fitted) - [0.6, 1.0, 1.5, -0.5, -1.1]).max() < 0.15, fitted
    logp4 = calfam_log_softmax(rng.normal(size=(400, 4)))
    assert len(CF.fit_family("vector", logp4, rng.integers(0, 4, 400))) == 7  # 2k - 1 for four answers
    with pytest.raises(ValueError):
        CF.apply_family("vector", params + [0.1], logp[:5])


def test_calfam_objectives_and_gradients():
    rng = np.random.default_rng(12)
    logp = calfam_log_softmax(rng.normal(size=(50, 3)) * 2.0)
    y = rng.integers(0, 3, 50)
    w = np.array([[1.2, 0.3, -0.2], [0.1, 0.9, 0.4], [-0.5, 0.2, 1.4]])
    b = np.array([0.2, -0.3, 0.6])
    theta = np.concatenate((w.ravel(), b))
    for lambda_od, lambda_b in ((0.01, 0.1), (1.0, 0.0001)):
        value, grad = CF.odir_objective(theta, logp, y, lambda_od, lambda_b)
        q = calfam_softmax(logp @ w.T + b)
        off = [w[i, j] for i in range(3) for j in range(3) if i != j]
        # the mean negative log-likelihood + lambda_od * mean of the 6 squared off-diagonal entries + lambda_b * mean of the 3 squared intercepts
        want = -np.log(q[np.arange(50), y]).mean() + lambda_od * np.mean(np.square(off)) + lambda_b * np.mean(b ** 2)
        assert value == pytest.approx(want, rel=1e-12)
        numeric = approx_fprime(theta, lambda t: CF.odir_objective(t, logp, y, lambda_od, lambda_b)[0], 1e-7)
        assert np.abs(grad - numeric).max() < 1e-5
    vtheta = np.array([0.7, 1.3, 2.1, 0.4, -0.6])
    rows, onehot = (np.arange(50), y), np.eye(3)[y]
    value, grad = CF._vector_objective(vtheta, logp, rows, onehot)
    q = calfam_softmax(logp * vtheta[:3] + np.array([0.0, 0.4, -0.6]))
    assert value == pytest.approx(-np.log(q[np.arange(50), y]).mean(), rel=1e-12)
    numeric = approx_fprime(vtheta, lambda t: CF._vector_objective(t, logp, rows, onehot)[0], 1e-7)
    assert np.abs(grad - numeric).max() < 1e-5


def test_calfam_tie_rules():
    assert CF.lowest([0.5, 0.5]) == 0
    assert CF.lowest([0.5, 0.5 - 1e-13]) == 0  # within 1e-12 of the lowest: a tie, which goes to the first
    assert CF.lowest([0.5, 0.5 - 1e-11]) == 1
    assert CF.lowest([0.3, 0.3 - 0.6e-12, 0.3 - 1.2e-12]) == 1  # the tie is with the lowest, not with the first
    with pytest.raises(ValueError):
        CF.lowest([0.5, float("nan")])
    # uniform log-probabilities and 25 items of each answer: every training part, outer and inner, holds as many items of
    # each answer, so every fit stays at its start and every family and every penalty pair gives the same probabilities
    y = np.repeat(np.arange(3), 25)
    np.random.default_rng(3).shuffle(y)
    logp = np.full((75, 3), np.log(1.0 / 3.0))
    orders = (CF.FAMILIES, tuple(reversed(CF.FAMILIES)), ("vector", "dirichlet_odir", "temperature", "temperature_intercepts"))
    for order in orders:
        result, _ = CF.choose(logp, y, families=order, repeats=2)
        assert len({result["families"][f]["out_of_fold_log_loss"]["mean"] for f in order}) == 1  # an exact tie
        assert result["chosen"] == order[0]
        assert result["params"] == result["families"][order[0]]["params"]
    pair, inner = CF.odir_select(logp, y)
    assert len(inner["pairs"]) == 25 and inner["pairs"] == sorted(inner["pairs"])  # the order (lambda_od, lambda_b) ascending
    assert len(set(inner["inner_log_loss"])) == 1  # every pair ties
    assert pair == (0.0001, 0.0001)
    fitted = CF.fit_family("dirichlet_odir", logp, y)
    assert (fitted["lambda_od"], fitted["lambda_b"]) == (0.0001, 0.0001)
    assert fitted["W"] == np.eye(3).tolist() and fitted["b"] == [0.0, 0.0, 0.0]  # the start, W = I and b = 0
    with pytest.raises(ValueError):
        CF.choose(logp, y, families=("temperature", "temperature"))


def test_calfam_out_of_fold_folds_and_inner_folds():
    logp, y, _ = calfam_planted("dirichlet_odir", {"W": CALFAM_W_PLANT, "b": CALFAM_B_PLANT}, 100, 5)
    summary, probs = CF.out_of_fold("dirichlet_odir", logp, y, folds=5, repeats=2)
    assert probs.shape == (2, 100, 3) and np.allclose(probs.sum(axis=2), 1.0)
    rng = np.random.default_rng(CALFAM_SEED)  # the fold sets main() draws
    for r, fold in enumerate([CF.folds_for(y, 5, rng) for _ in range(2)]):
        for f in range(5):
            test = fold == f
            # refitted on the training part, with its inner folds drawn from their own generator with the same seed
            params = CF.fit_family("dirichlet_odir", logp[~test], y[~test], CALFAM_SEED)
            assert np.array_equal(probs[r][test], CF.apply_family("dirichlet_odir", params, logp[test]))
    losses = [-np.log(np.clip(p[np.arange(100), y], 1e-12, None)).mean() for p in probs]
    assert summary["out_of_fold_log_loss"] == {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses))}


def test_calfam_apply_family_matches_session8(tmp_path):
    rng = np.random.default_rng(5)
    logp = calfam_log_softmax(rng.normal(size=(40, 3)) * 2.5)
    labels = rng.integers(0, 3, 40)
    ids = [f"r{i}" for i in range(40)]
    cases = (("temperature_intercepts", [float(np.log(1.7))], [float(np.log(1.3)), 0.4, -0.25]),
             ("temperature", [float(np.log(0.6))], [float(np.log(2.2)), -1.1, 0.3]),
             ("temperature", [float(np.log(25.0))], [float(np.log(0.05)), 0.2, 0.1]))  # outside [0.1, 10]: calibrated() clips T
    for chosen, t_params, ti_params in cases:
        entry = {"chosen": chosen, "families": {"temperature": {"params": t_params}, "temperature_intercepts": {"params": ti_params}}}
        out = tmp_path / chosen
        got = CAL8.write_calibrated(out, "made_up", "evidence_effect", "validation", ids, labels, logp, entry)
        assert np.array_equal(CF.apply_family(chosen, entry["families"][chosen]["params"], logp), got)
        for family in ("temperature", "temperature_intercepts"):
            params = entry["families"][family]["params"]
            assert np.array_equal(CF.apply_family(family, params, logp), CAL8.calibrated(logp, params, 3))
        _, _, written = CF.M.read_predictions(out / "made_up_temperature_evidence_effect_validation.csv")  # 9 significant digits
        assert np.allclose(written, CF.apply_family("temperature", t_params, logp), rtol=1e-8, atol=1e-12)
    with pytest.raises(ValueError):
        CF.apply_family("temperature", [0.1, 0.2, 0.3], logp)  # calibrated() alone would read the extra numbers as intercepts
    with pytest.raises(ValueError):
        CF.apply_family("temperature_intercepts", [0.1], logp)


def calfam_command_inputs(folder, n, seed=11):
    """A made-up raw predictions CSV (9 significant digits, as session 4 writes them) and its validation records, some
    marked calibrate false and some without the key."""
    rng = np.random.default_rng(seed)
    logp = calfam_log_softmax(rng.normal(size=(n, 3)) * 2.0)
    y = calfam_draw(calfam_planted_map("temperature_intercepts", [np.log(1.6), 0.5, -0.4], logp), rng)
    ids = [f"item{i:03d}" for i in range(n)]
    folder.mkdir(parents=True, exist_ok=True)
    predictions, records = folder / "made_up-raw_evidence_effect_validation.csv", folder / "evidence_effect_validation.jsonl"
    with open(predictions, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "p0", "p1", "p2"])
        for i, label, row in zip(ids, y, np.exp(logp)):
            w.writerow([i, int(label)] + [f"{x:.9g}" for x in row])
    with open(records, "w", encoding="utf-8") as f:
        for k, i in enumerate(ids):
            f.write(json.dumps({"id": i} if k % 5 == 0 else {"id": i, "calibrate": k % 13 != 4}) + "\n")
    return predictions, records


def calfam_command_items(predictions, records):
    """y and log p as main() reads them."""
    keep = {r["id"] for r in map(json.loads, records.read_text(encoding="utf-8").splitlines()) if r.get("calibrate", True)}
    ids, labels, probs = CF.M.read_predictions(predictions)
    rows = np.array([i for i, x in enumerate(ids) if x in keep])
    return np.asarray(labels)[rows], np.log(np.clip(np.asarray(probs)[rows], 1e-12, None))


def calfam_run_command(script, args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(CALFAM_SRC)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return subprocess.run([sys.executable, str(script)] + [str(a) for a in args], check=True, capture_output=True, text=True, env=env)


def calfam_original_command(tmp_path):
    """The file as it was before session 9, under its own name (so that its help names the same program): the copy saved
    before the change or, if that copy is gone, the file at the commit that last changed it before."""
    if CALFAM_ORIGINAL_COPY.exists():
        data = CALFAM_ORIGINAL_COPY.read_bytes()
    else:
        data = subprocess.run(["git", "-C", str(CALFAM_ROOT), "show", f"{CALFAM_ORIGINAL_COMMIT}:src/eval/calibration_family_cv.py"],
                              check=True, capture_output=True).stdout
    assert hashlib.sha256(data).hexdigest() == CALFAM_ORIGINAL_SHA256
    path = tmp_path / "original" / "calibration_family_cv.py"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    return path


def test_calfam_out_of_fold_gives_the_commands_numbers(tmp_path):
    predictions, records = calfam_command_inputs(tmp_path / "inputs", 150)
    out = tmp_path / "command.json"
    calfam_run_command(CALFAM_SRC / "eval" / "calibration_family_cv.py", ["--predictions", predictions, "--records", records, "--repeats", 4,
                                                            "--bootstrap", 20, "--out", out])
    data = json.loads(out.read_text(encoding="utf-8"))
    y, logp = calfam_command_items(predictions, records)
    assert data["n"] == len(y) < 150
    for family in ("temperature", "temperature_intercepts"):
        summary, probs = CF.out_of_fold(family, logp, y, folds=5, repeats=4)
        for key in ("out_of_fold_log_loss", "out_of_fold_ece", "out_of_fold_log_loss_by_label"):
            assert summary[key] == data["families"][family][key], (family, key)
        assert probs.shape == (4, len(y), 3)
        params = CF.fit_family(family, logp, y)
        assert float(np.clip(np.exp(params[0]), CF.T_MIN, CF.T_MAX)) == data["families"][family]["fitted_on_all"]["temperature"]
        assert [0.0] + params[1:] == data["families"][family]["fitted_on_all"]["intercepts"]
    # THE RECIPE RULE's score, session 8's rule through choose(): the lower of the two families' means
    result, _ = CF.choose(logp, y, families=("temperature", "temperature_intercepts"), repeats=4)
    means = [data["families"][f]["out_of_fold_log_loss"]["mean"] for f in ("temperature", "temperature_intercepts")]
    assert result["families"][result["chosen"]]["out_of_fold_log_loss"]["mean"] == min(means)


def test_calfam_command_unchanged(tmp_path):
    predictions, records = calfam_command_inputs(tmp_path / "inputs", 90)
    scripts = {"original": calfam_original_command(tmp_path), "new": CALFAM_SRC / "eval" / "calibration_family_cv.py"}
    assert hashlib.sha256(scripts["new"].read_bytes()).hexdigest() != CALFAM_ORIGINAL_SHA256  # the new file is the extended one
    for extra in ([], ["--folds", 4, "--repeats", 3, "--bootstrap", 50]):  # the defaults (5 folds, 20 repeats, 1,000 resamples), then others
        outputs = {}
        for name, script in scripts.items():
            out = tmp_path / f"{name}_{len(extra)}.json"
            printed = calfam_run_command(script, ["--predictions", predictions, "--records", records, "--out", out] + extra).stdout
            outputs[name] = (printed, out.read_bytes())
        assert outputs["new"] == outputs["original"]
        assert json.loads(outputs["new"][1])["families"].keys() == {"temperature", "temperature_intercepts"}
    assert calfam_run_command(scripts["new"], ["--help"]).stdout == calfam_run_command(scripts["original"], ["--help"]).stdout


# ======================================================================================================================
# Part 7 of 11: src/eval/compare_session9.py
# From test_compare_session9.py. Module-level names prefixed cmp9; test names test_compare9_*.
#
# Tests for src/eval/compare_session9.py (docs/prompt_session9.md, COMPARISONS and POWER AND THE BAR, with corrections 2
# and 3), on made-up results folders, records and predictions written under tmp_path:
#
# - the guard: nothing runs before release_candidate.json and testpreds_log.json exist;
# - the bar's four criteria with either large model of criterion 3 (gemma4_31b or gemma4_31b_long), each passing and
#   failing, with the reason after "v0.5 release bar not met:";
# - K3: the two-run average of the runs' softmaxed validation logits, session 8's rule out of fold
#   (calibration_family_cv.out_of_fold averaged over the repeats, the family calibrate_session7.choose picks) and K3's pairs
#   from recipe_selection.json (two runs, one run, dropped, the combination);
# - the reading of a difference whose interval includes zero, with the estimate, the interval and what the interval
#   excludes, never equivalence;
# - the one-against-rest calibration slope and intercept recovering a planted miscalibration;
# - the cost per 1,000 questions (with and without swap averaging, a missing run summary, the rate's fallback to
#   cost_rule.json, gemma4_31b_long's timing);
# - end to end on a made-up step 15: K8, K9 and K10 read the right files, K1 to K7 and the sensitivity, systems not built
#   (tabicl's marker, gemma4_31b_long not run, s9_ei_wavg not built) reported and their comparisons listed as not made,
#   zeros.json, comparison.csv, and a run stopped after one block resuming to the same comparison.
# ======================================================================================================================

CMP9_SEED = 20260927
CMP9_FAMILIES = ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")


def cmp9_write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def cmp9_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def cmp9_write_probs(path, ids, labels, probs, order=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    order = range(len(ids)) if order is None else order
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i in order:
            w.writerow([ids[i], int(labels[i])] + [f"{x:.9g}" for x in probs[i]])


def cmp9_write_logits(path, ids, labels, z, order):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"z{j}" for j in range(z.shape[1])])
        for i in order:
            w.writerow([ids[i], int(labels[i])] + [repr(float(v)) for v in z[i]])


def cmp9_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cmp9_aligned(path, ids):
    f_ids, labels, probs = CMP9_M.read_predictions(path)
    pos = {i: k for k, i in enumerate(f_ids)}
    order = [pos[i] for i in ids]
    return labels[order], probs[order]


# the guard ------------------------------------------------------------------------------------------------------------

def test_compare9_guard_refuses_before_the_freeze(tmp_path):
    results = tmp_path / "results" / "fiorillo_v0_5"
    results.mkdir(parents=True)
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        CMP9.run(results=results, cache_dir=tmp_path / "cache")
    (results / "release_candidate.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="testpreds_log.json does not exist"):
        CMP9.run(results=results, cache_dir=tmp_path / "cache")
    assert not (results / "comparison.json").exists() and not (tmp_path / "cache").exists()
    (results / "testpreds_log.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not name v0.5's EI specialist"):  # past the guard, an empty freeze stops
        CMP9.Study(CMP9.Paths(results=results, cache=tmp_path / "cache")).run()


# the bar --------------------------------------------------------------------------------------------------------------

def cmp9_bar_out(large, ece=0.03, prior_iv=(-0.30, -0.20), large_iv=(-0.06, -0.01), f1=(0.80, 0.78), v05="s9_ei_sa"):
    return {"metrics": {"evidence_effect": {v05: {"ece": {"estimate": ece}, "macro_f1": {"estimate": f1[0]}},
                                             large: {"macro_f1": {"estimate": f1[1]}}}},
            "bar_intervals": {"v05_minus_prior": {"estimate": prior_iv[0], "low": prior_iv[0] - 0.05, "high": prior_iv[1]},
                              "v05_minus_large_model": {"estimate": large_iv[0], "low": large_iv[0] - 0.05,
                                                        "high": large_iv[1]}}}


@pytest.mark.parametrize("large", ["gemma4_31b", "gemma4_31b_long"])
def test_compare9_bar_criteria_with_either_large_model(large):
    rc = {"ece_limits": {"evidence_effect": {"limit": 0.05}}}
    bar = CMP9.release_bar(cmp9_bar_out(large), rc, "s9_ei_sa", large)
    assert bar["overall"] == "PASS" and all(c["result"] == "PASS" for c in bar["criteria"].values())
    assert bar["large_model"] == large and bar["specialist"] == "s9_ei_sa"
    assert f"{large}'s" in bar["criteria"]["3"]["rule"] and f"{large}'s point estimate" in bar["criteria"]["4"]["rule"]
    assert bar["outcome"].startswith("every criterion passes")
    cases = {"1": (cmp9_bar_out(large, ece=0.051), "its ECE, 0.0510, is above the limit, 0.0500"),
             "2": (cmp9_bar_out(large, prior_iv=(-0.10, 0.01)),
                   "s9_ei_sa's log loss minus the prior's is -0.1000, with a 95 percent interval of -0.1500 to 0.0100"),
             "3": (cmp9_bar_out(large, large_iv=(-0.03, 0.002)), f"s9_ei_sa's log loss minus {large}'s is -0.0300"),
             "4": (cmp9_bar_out(large, f1=(0.779, 0.78)), f"its macro-F1, 0.7790, is below {large}'s, 0.7800")}
    for k, (out, why) in cases.items():
        bar = CMP9.release_bar(out, rc, "s9_ei_sa", large)
        failed = [j for j, c in bar["criteria"].items() if c["result"] == "FAIL"]
        assert failed == [k] and bar["overall"] == "FAIL", k
        assert bar["outcome"] == f"v0.5 release bar not met: criterion {k} ({bar['criteria'][k]['reason']})"
        assert why in bar["criteria"][k]["reason"], k
    assert CMP9.release_bar(cmp9_bar_out(large, large_iv=(0.01, 0.05)), rc, "s9_ei_sa", large)["criteria"]["3"]["result"] == "FAIL"
    assert CMP9.release_bar(cmp9_bar_out(large, ece=0.05), rc, "s9_ei_sa", large)["criteria"]["1"]["result"] == "PASS"
    assert CMP9.release_bar(cmp9_bar_out(large, f1=(0.78, 0.78)), rc, "s9_ei_sa", large)["criteria"]["4"]["result"] == "PASS"
    two = CMP9.release_bar(cmp9_bar_out(large, ece=0.06, f1=(0.70, 0.78)), rc, "s9_ei_sa", large)
    assert two["outcome"].startswith("v0.5 release bar not met: criterion 1 (its ECE") and "; criterion 4 (its macro-F1" in two["outcome"]
    missing = CMP9.release_bar({"metrics": {}, "bar_intervals": {}}, {}, "s9_ei_sa", large)
    assert missing["overall"] == "FAIL" and all(c["reason"] for c in missing["criteria"].values())
    assert "holds no ECE limit" in missing["criteria"]["1"]["reason"]
    assert f"s9_ei_sa's or {large}'s test predictions are missing" in missing["criteria"]["3"]["reason"]


# K3 -------------------------------------------------------------------------------------------------------------------

def test_compare9_k3_two_run_averages_and_out_of_fold_probabilities(tmp_path):
    rng = np.random.default_rng(11)
    results = tmp_path / "results"
    clean_ids = [f"v-{k}" for k in range(30)]
    y = np.array([k % 3 for k in range(30)])
    all_ids = clean_ids + [f"other-{k}" for k in range(6)]  # each run predicts every validation prompt
    all_y = np.concatenate([y, rng.integers(0, 3, 6)])
    z = {}
    for k, run in enumerate(("s9_x_1", "s9_x_2")):
        z[run] = np.eye(3)[all_y] * (1.0 + k) + rng.normal(0, 1, (36, 3))
        cmp9_write_logits(results / "logits" / f"{run}_evidence_effect_validation_logits.csv", all_ids, all_y, z[run],
                          rng.permutation(36))
    avg = CMP9.average_runs(results, ["s9_x_1", "s9_x_2"], clean_ids, y)
    by_hand = (cmp9_softmax(z["s9_x_1"][:30]) + cmp9_softmax(z["s9_x_2"][:30])) / 2
    assert np.allclose(avg, by_hand, rtol=0, atol=1e-12)
    logp = np.log(np.maximum(avg, 1e-300))
    chosen, fams, oof = CMP9.oof_session8_rule(logp, y, repeats=3)
    expected = {}
    for fam in ("temperature", "temperature_intercepts"):
        summary, probs = CMP9_CFC.out_of_fold(fam, logp, y, 5, 3, CMP9_SEED)
        assert fams[fam]["out_of_fold_log_loss"]["mean"] == pytest.approx(summary["out_of_fold_log_loss"]["mean"], abs=1e-12)
        assert probs.shape == (3, 30, 3)
        expected[fam] = probs.mean(axis=0)  # each prompt's out-of-fold probabilities averaged over the repeats
    a, b = fams["temperature"]["out_of_fold_log_loss"]["mean"], fams["temperature_intercepts"]["out_of_fold_log_loss"]["mean"]
    assert chosen == ("temperature_intercepts" if b < a else "temperature")
    assert np.allclose(oof, expected[chosen], rtol=0, atol=1e-12) and np.allclose(oof.sum(axis=1), 1.0)
    family7, fams7 = CMP9_CAL7.choose(logp, y, repeats=3)  # session 8's rule as the recipe rule's scores apply it
    assert family7 == chosen and fams7["temperature"]["out_of_fold_log_loss"]["mean"] == pytest.approx(a, abs=1e-12)
    assert fams7["temperature_intercepts"]["out_of_fold_log_loss"]["mean"] == pytest.approx(b, abs=1e-12)
    block = CMP9.k3_block(results, ["s9_x_1", "s9_x_2"], clean_ids, y, repeats=3)
    assert block["chosen"] == chosen and np.allclose(np.array(block["probs"]), oof, rtol=0, atol=1e-15)
    json.dumps(block, allow_nan=False)
    with pytest.raises(SystemExit, match="lack 1 of the clean prompts"):
        CMP9.average_runs(results, ["s9_x_1"], clean_ids + ["v-missing"], np.append(y, 0))
    with pytest.raises(SystemExit, match="labels differ"):
        CMP9.average_runs(results, ["s9_x_1"], clean_ids, (y + 1) % 3)
    sel = {"candidates": {"swap": {"runs": ["s9_swap_1", "s9_swap_2"], "passes": True}, "long": {"status": "dropped", "passes": False},
                          "lr1": {"runs": ["s9_lr1_1"], "passes": False}, "lr2": {"status": "not run", "passes": False}},
           "combination": {"called": True, "parts": ["swap", "lr1"], "runs": ["s9_combo_1", "s9_combo_2"]}}
    plan, skipped = CMP9.k3_plan(sel)
    assert [p[:4] for p in plan] == [("s9_swap_1_2", ["s9_swap_1", "s9_swap_2"], "s9_b_1_2", ["s9_b_1", "s9_b_2"]),
                                     ("s9_lr1_1", ["s9_lr1_1"], "s9_b_1", ["s9_b_1"]),
                                     ("s9_combo_1_2", ["s9_combo_1", "s9_combo_2"], "s9_b_1_2", ["s9_b_1", "s9_b_2"])]
    assert skipped == [{"candidate": "long", "reason": "dropped"}, {"candidate": "lr2", "reason": "not run"}]
    _, skipped = CMP9.k3_plan({"candidates": {}, "combination": {"called": False}})
    assert skipped[-1] == {"candidate": "combo", "reason": "not called: fewer than two of the three changes passed"}


# the readings ---------------------------------------------------------------------------------------------------------

def test_compare9_reading_of_a_difference_whose_interval_includes_zero():
    text = CMP9.excludes_text("log_loss", 0.0031, -0.012, 0.015)
    assert text.startswith("no difference was detected: the estimate is 0.0031, with a 95 percent interval of -0.0120 to 0.0150")
    assert "excludes the system being higher, that is worse, than the reference by more than 0.0150" in text
    assert "lower, that is better, by more than 0.0120" in text
    assert text.endswith("it is not a finding of equivalence") and "equivalent" not in text
    f1 = CMP9.excludes_text("macro_f1", -0.01, -0.03, 0.02)
    assert "higher, that is better, than the reference by more than 0.0200" in f1 and "lower, that is worse, by more than 0.0300" in f1
    slope = CMP9.excludes_text("calibration_slope_30d", 0.0, -0.03, 0.02)
    assert "higher than the reference by more than 0.0200, and lower by more than 0.0300" in slope
    assert CMP9.excludes_text("log_loss", -0.2, -0.3, -0.1) == "the interval, -0.3000 to -0.1000, lies wholly below zero"
    assert CMP9.excludes_text("accuracy", 0.2, 0.1, 0.3) == "the interval, 0.1000 to 0.3000, lies wholly above zero"
    assert CMP9.excludes_text("accuracy", None, None, None) == "not computed"
    assert CMP9.excludes_text("ece", 0.0, 0.0, 0.0).startswith("no difference was detected: the difference is 0 in every resample")
    assert CMP9.CMP.reading("log_loss", -0.012, 0.015) == "no difference was detected"
    assert CMP9.CMP.reading("log_loss", -0.3, -0.1) == "lower (better)" and CMP9.CMP.reading("macro_f1", -0.3, -0.1) == "lower (worse)"


# the calibration slope and intercept ----------------------------------------------------------------------------------

def cmp9_planted(a, b, reps=2000):
    """Answer 0 stated with p on a grid, and true with chance 1 / (1 + exp(-(a + b logit p))): reps rows a grid point, the
    share of answer 0 rounded to the nearest row; the other rows split between answers 1 and 2."""
    grid = np.linspace(0.05, 0.95, 19)
    q = 1 / (1 + np.exp(-(a + b * np.log(grid / (1 - grid)))))
    labels, p0 = [], []
    for g, qq in zip(grid, q):
        k = int(round(qq * reps))
        rest = reps - k
        labels += [0] * k + [1] * (rest // 2) + [2] * (rest - rest // 2)
        p0 += [g] * reps
    p0 = np.array(p0)
    return np.array(labels), np.column_stack([p0, (1 - p0) / 2, (1 - p0) / 2])


def test_compare9_one_against_rest_recovers_a_planted_miscalibration():
    labels, probs = cmp9_planted(0.4, 0.6)  # overconfident: logit P(answer 0) = 0.4 + 0.6 logit p
    out = CMP9.one_against_rest(labels, probs)
    assert out.shape == (9,)
    intercept, slope, joint = out[0:3]
    assert slope == pytest.approx(0.6, abs=0.01) and joint == pytest.approx(0.4, abs=0.01)
    assert slope == pytest.approx(CMP9_M.calibration_slope(labels == 0, probs[:, 0]), abs=1e-9)
    assert intercept == pytest.approx(CMP9_M.calibration_intercept(labels == 0, probs[:, 0]), abs=1e-9)
    labels, probs = cmp9_planted(0.5, 1.0)  # a shift alone: the intercept with the logit as an offset recovers it
    intercept, slope, joint = CMP9.one_against_rest(labels, probs)[0:3]
    assert intercept == pytest.approx(0.5, abs=0.01) and slope == pytest.approx(1.0, abs=0.01) and joint == pytest.approx(0.5, abs=0.01)
    labels, probs = cmp9_planted(0.0, 1.0)  # calibrated
    intercept, slope, joint = CMP9.one_against_rest(labels, probs)[0:3]
    assert abs(intercept) < 0.01 and slope == pytest.approx(1.0, abs=0.01) and abs(joint) < 0.01
    flat = CMP9.one_against_rest(np.array([0, 1, 2, 0]), np.full((4, 3), 1 / 3))  # one probability for every row
    assert np.isnan(flat[1]) and np.isnan(flat[2])


# the cost per 1,000 questions -----------------------------------------------------------------------------------------

def test_compare9_cost_per_1000_questions(tmp_path):
    r5, r4 = tmp_path / "r5", tmp_path / "r4"
    for k in (1, 2, 3):
        cmp9_write_json(r5 / "run_summaries" / f"s9_b_{k}_test_run_summary.json",
                        {"timing_test": {"items": 100, "forward_seconds": 20.0 + k, "seconds_per_item": (20.0 + k) / 100}})
        swap_summary = (r5 / "checks_v05" / f"s9_b_{k}_swap_run_summary.json" if k < 3 else  # the driver's name, and another
                        r5 / "checks_v05" / "run_summaries" / f"s9_b_{k}_swap_test_run_summary.json")
        cmp9_write_json(swap_summary, {"timing_test": {"items": 90, "forward_seconds": 18.0, "seconds_per_item": 0.2}})
    cmp9_write_json(r4 / "run_summaries" / "s7_b3_timing_test_run_summary.json", {"timing_test": {"seconds_per_item": 0.02}})
    cmp9_write_json(r4 / "large_model" / "gemma4_31b_timing.json", {"evidence_effect": {"test": {"gpu_seconds_per_item": 1.6}},
                                                                    "pubmedqa_answer": {"test": {"gpu_seconds_per_item": 0.8}}})
    (r5 / "pod_rate.txt").write_text("created_utc 2026-10-01T10:00:00Z\nhourly_usd 3.52\n", encoding="utf-8")
    paths = CMP9.Paths(results=r5, results8=r4)
    names = {"v05": "s9_ei_sa", "single_pass_systems": ["s9_b_1", "s9_b_2", "s9_b_3"], "swap_averaging": True,
             "large": "gemma4_31b"}
    out = CMP9.cost_section(paths, names, 100)
    seconds = (21 + 22 + 23 + 3 * 18) / 100  # three passes over the prompts and three over the exchanged prompts
    ei = out["specialists"]["evidence_effect"]
    assert out["hourly_usd"] == 3.52 and "hourly_usd" in out["rate_source"]
    assert [p["system"] for p in ei["passes"]] == ["s9_b_1", "s9_b_2", "s9_b_3", "s9_b_1_swap", "s9_b_2_swap", "s9_b_3_swap"]
    assert ei["passes"][0]["run_summary"].endswith("run_summaries/s9_b_1_test_run_summary.json")
    assert ei["passes"][3]["run_summary"].endswith("checks_v05/s9_b_1_swap_run_summary.json")
    assert ei["passes"][5]["run_summary"].endswith("checks_v05/run_summaries/s9_b_3_swap_test_run_summary.json")
    assert ei["seconds_per_question"] == pytest.approx(seconds) and ei["usd"] == pytest.approx(1000 * seconds / 3600 * 3.52)
    assert out["specialists"]["pubmedqa_answer"]["usd"] == pytest.approx(1000 * 0.02 / 3600 * 3.52)
    assert out["large_model"]["evidence_effect"]["usd"] == pytest.approx(1000 * 1.6 / 3600 * 3.52)
    assert out["large_model"]["pubmedqa_answer"]["usd"] == pytest.approx(1000 * 0.8 / 3600 * 3.52)
    assert out["specialists"]["readmit_window"]["usd"] is None and out["specialists"]["readmit_window"]["not_computed"]
    plain = CMP9.cost_section(paths, dict(names, v05="s9_ei", swap_averaging=False), 100)["specialists"]["evidence_effect"]
    assert plain["seconds_per_question"] == pytest.approx(0.21 + 0.22 + 0.23)  # session 8's sum of the runs' seconds per item
    gap = CMP9.cost_section(paths, dict(names, single_pass_systems=["s9_b_1", "s9_b_4"]), 100)["specialists"]["evidence_effect"]
    none = CMP9.cost_section(paths, dict(names, single_pass_systems=[]), 100)["specialists"]["evidence_effect"]
    assert none["usd"] is None and "no single-pass system" in none["not_computed"]
    assert gap["usd"] is None and gap["not_computed"].endswith("for s9_b_4, s9_b_4_swap")
    cmp9_write_json(tmp_path / "r5b" / "cost_rule.json", {"pod": {"hourly_usd": 3.49}})
    rate, source = CMP9.hourly_rate(CMP9.Paths(results=tmp_path / "r5b"))
    assert rate == 3.49 and "cost_rule.json" in source
    assert CMP9.hourly_rate(CMP9.Paths(results=tmp_path / "none"))[0] is None
    assert CMP9.cost_section(CMP9.Paths(results=tmp_path / "none", results8=r4), names, 100)["large_model"]["evidence_effect"]["usd"] is None
    cmp9_write_json(r5 / "large_model" / "gemma4_31b_long_timing.json", {"evidence_effect": {"test": {"gpu_seconds_per_item": 2.4}}})
    long_ = CMP9.cost_section(paths, dict(names, large="gemma4_31b_long"), 100)["large_model"]
    assert long_["evidence_effect"]["system"] == "gemma4_31b_long"
    assert long_["evidence_effect"]["usd"] == pytest.approx(1000 * 2.4 / 3600 * 3.52)
    assert long_["pubmedqa_answer"]["system"] == "gemma4_31b"


# end to end -----------------------------------------------------------------------------------------------------------

def cmp9_study(tmp_path, large="gemma4_31b", wavg=True):
    """A made-up step 15: records, the systems' test predictions, K3's validation logits and the JSON files the comparisons
    read. v0.5's EI specialist is s9_ei_sa (swap averaging kept on s9_ei, the swap recipe's three runs); tabicl is NOT
    BUILT (its marker in not_built.json) and lightgbm_tuned's refit has no file."""
    rng = np.random.default_rng(7)
    base = tmp_path / "repo"
    r5, r4, r3, rp = (base / "results" / d for d in ("fiorillo_v0_5", "fiorillo_v0_4", "fiorillo_v0_3", "public_v0"))
    records_ei, records9, public = base / "records8", base / "records9", base / "public"
    ei = []
    for a in range(13):  # 13 articles of 3 prompts; the prompts of articles 0, 5 and 10 flagged
        for k in range(3):
            y = int(rng.integers(0, 3))
            ei.append({"id": f"ev-{3 * a + k}", "qid": "evidence_effect", "label": y,
                       "group": {"pmcid": str(100 + a), "flag": "Questionable" if a % 5 == 0 else "none", "accepted_labels": [y]}})
    cmp9_write_jsonl(records_ei / "evidence_effect_test.jsonl", ei)
    cmp9_write_jsonl(records_ei / "evidence_effect_train.jsonl", [{"id": f"t{i}", "label": i % 3} for i in range(30)])
    ids, y = [r["id"] for r in ei], np.array([r["label"] for r in ei])

    def probs(strength, labels):
        return cmp9_softmax(np.eye(3)[labels] * strength + rng.normal(0, 1, (len(labels), 3)))

    p5, p4, p3, pp = (r / "predictions" for r in (r5, r4, r3, rp))
    shuffled = rng.permutation(len(ids))
    good = probs(2.5, y)
    s9_b = probs(1.5, y)
    s9_b[0, y[0]] = 0.0  # a true answer that prints as 0
    s9_b[0] /= s9_b[0].sum()
    ei_files = {(p5, "s9_ei_sa"): good, (p5, "s9_ei_sa_s8rule"): good, (p5, "s9_ei_sa-raw"): probs(2.5, y),
                (p5, "s9_ei"): probs(2.0, y), (p5, "s9_b"): s9_b, (p5, "s9_swap_1"): probs(2.0, y),
                (p4, "s8_ei"): probs(2.6, y), (p4, "gemma4_31b"): probs(0.8, y)}
    if wavg:
        ei_files[(p5, "s9_ei_wavg")] = probs(2.0, y)
    if large == "gemma4_31b_long":
        ei_files[(p5, "gemma4_31b_long")] = probs(0.9, y)
    for (folder, name), p in ei_files.items():
        cmp9_write_probs(folder / f"{name}_evidence_effect_test.csv", ids, y, p, shuffled)
    cmp9_write_probs(p5 / "s9_ei_sa_evidence_effect_validation.csv", ids[:10], y[:10], good[:10])
    pq_rows = [{"id": f"pq-{i}", "label": int(rng.integers(0, 3))} for i in range(30)]
    cmp9_write_jsonl(public / "pubmedqa_answer_test.jsonl", pq_rows)
    pq_ids, pq_y = [r["id"] for r in pq_rows], np.array([r["label"] for r in pq_rows])
    for folder, name, s in ((p3, "s7_b3", 2.0), (p5, "biogpt_large_pqa-raw", 1.0), (p5, "biogpt_large_pqa", 1.2),
                            (p4, "gemma4_31b", 1.0)):
        cmp9_write_probs(folder / f"{name}_pubmedqa_answer_test.csv", pq_ids, pq_y, probs(s, pq_y))
    cmp9_write_probs(pp / "prior_pubmedqa_answer_test.csv", pq_ids, pq_y, np.tile([0.5, 0.3, 0.2], (30, 1)))
    for split in ("test", "validation"):
        rows = [{"id": f"rw-{split}-{i}", "label": int(rng.integers(0, 3))} for i in range(40)]
        cmp9_write_jsonl(public / f"readmit_window_{split}.jsonl", rows)
        rw_ids, rw_y = [r["id"] for r in rows], np.array([r["label"] for r in rows])
        for folder, name, s in ((p5, "readmission_pool_v2", 1.6), (p4, "readmission_pool", 1.5), (p5, "xgboost_codes", 1.3),
                                (p4, "xgboost_tuned", 1.2), (p5, "catboost_codes", 1.3), (p4, "catboost_tuned", 1.2),
                                (p3, "lightgbm_tuned", 1.1)):
            cmp9_write_probs(folder / f"{name}_readmit_window_{split}.csv", rw_ids, rw_y, probs(s, rw_y))
        if split == "test":
            cmp9_write_probs(pp / "prior_readmit_window_test.csv", rw_ids, rw_y, np.tile([0.6, 0.3, 0.1], (40, 1)))
    clean_rows = [{"id": f"v-{k}", "qid": "evidence_effect", "label": k % 3, "calibrate": True,
                   "group": {"pmcid": f"c{k // 3}", "flag": "none", "accepted_labels": [k % 3]}} for k in range(30)]
    cmp9_write_jsonl(records9 / "clean" / "evidence_effect_validation.jsonl", clean_rows)
    v_ids = [r["id"] for r in clean_rows] + [f"other-{k}" for k in range(6)]
    v_y = np.array([r["label"] for r in clean_rows] + [int(v) for v in rng.integers(0, 3, 6)])
    for run, s in (("s9_b_1", 1.0), ("s9_b_2", 1.1), ("s9_swap_1", 1.6), ("s9_swap_2", 1.5), ("s9_lr1_1", 1.2)):
        z = np.eye(3)[v_y] * s + rng.normal(0, 1, (len(v_ids), 3))
        cmp9_write_logits(r5 / "logits" / f"{run}_evidence_effect_validation_logits.csv", v_ids, v_y, z, rng.permutation(len(v_ids)))
    cmp9_write_probs(p5 / "s9_b_1-raw_evidence_effect_validation.csv", v_ids, v_y, np.full((len(v_ids), 3), 1 / 3))
    runs = ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    v05_ei = {"system": "s9_ei_sa", "base_system": "s9_ei", "swap_averaging": True, "recipe": "swap", "runs": runs,
              "single_pass_systems": runs, "max_len": 6144 if large == "gemma4_31b_long" else 4096, "other_form": "s9_ei"}
    cmp9_write_json(r5 / "release_candidate.json", {
        "specialists": {"evidence_effect": {**{k: v for k, v in v05_ei.items() if k != "other_form"}, "family": "vector"}},
        "reference": {"criterion_3_large_model": large}, "ece_limits": {"evidence_effect": {"limit": 0.3}},
        "time_utc": "2026-10-01T12:00:00Z"})
    cmp9_write_json(r5 / "testpreds_log.json", {"release_candidate_sha256": "made up", "start_utc": "2026-10-01T12:05:00Z"})
    cmp9_write_json(r5 / "recipe_selection.json", {
        "margin": {"m": 0.005}, "candidates": {"swap": {"runs": ["s9_swap_1", "s9_swap_2"], "score": 0.50, "passes": True},
                                              "long": {"status": "dropped", "passes": False},
                                              "lr1": {"runs": ["s9_lr1_1"], "score": 0.60, "passes": False},
                                              "lr2": {"status": "not run", "passes": False}},
        "combination": {"called": False, "parts": ["swap"]}, "winner": {"recipe": "swap", "runs": runs, "settings": {"x": 1}},
        "weight_average": {"built": wavg}, "swap_averaging": {"system": "s9_ei", "kept": True}, "v05_ei": v05_ei,
        "decisions": {"winner": {"value": "swap", "time_utc": "2026-10-01T11:00:00Z"}}})
    table = {f: {"mean": 0.5 - 0.01 * k, "min": 0.49 - 0.01 * k, "max": 0.51 - 0.01 * k} for k, f in enumerate(CMP9_FAMILIES)}
    cmp9_write_json(r5 / "calibration_s9.json", {
        "v05_ei": {"system": "s9_ei_sa"}, "s9_ei_sa": {"evidence_effect": {"chosen": "vector", "n_calibrate": 30}},
        "s9_ei_sa_s8rule": {"evidence_effect": {"chosen": "temperature"}}, "s9_ei": {"evidence_effect": {"chosen": "vector"}},
        **({"s9_ei_wavg": {"evidence_effect": {"chosen": "vector"}}} if wavg else {}),
        "families_table_v05": {"system": "s9_ei_sa", "table": table, "chosen": "dirichlet_odir"}})
    cmp9_write_json(r5 / "not_built.json", {"tabicl": "NOT BUILT: tabicl, a made-up reason"})
    cmp9_write_json(r5 / "readmission_pool_v2.json", {"chosen": "new", "choice": {"chosen": "new", "new_out_of_fold_log_loss": 0.80,
                                                                                  "session8_out_of_fold_log_loss": 0.81}})
    cmp9_write_json(r5 / "reading_checks.json", {
        "written_utc": "2026-10-01T13:00:00Z",
        "readings": {"s8_ei": {"none": {"reading": "partly", "caveats": []}},
                     "v05": {"none": {"reading": "its answers depend on the article's text", "caveats": ["single pass"]}}},
        "table": [{"system": "s8_ei", "name": "s8_ei", "input": "none", "statistic": "macro_f1", "estimate": 0.55, "low": 0.5,
                   "high": 0.6, "n": 39, "articles": 13},
                  {"system": "v05", "name": "s9_ei", "input": "none", "statistic": "reading",
                   "reading": "its answers depend on the article's text", "caveats": ["single pass"]}],
        "references": {"prior": {"text": "made up"}}, "notes": ["made up"], "same_none_and_title_records": {"none": True}})
    cell = {"estimate": 0.8, "low": 0.7, "high": 0.9}
    cmp9_write_json(r5 / "handoff_curves.json", {"split": "test", "written_utc": "2026-10-01T13:10:00Z", "pairs": [{
        "question": "evidence_effect", "fiorillo": "s9_ei_sa", "large_model": large, "n": 39, "articles": 13,
        "areas": {a: {"estimate": 0.1, "low": 0.05, "high": 0.15} for a in ("aurc", "e_aurc", "augrc")},
        "registered_rates": {"rate_10": {"handed_off": 3, **{c: cell for c in ("confidence", "random_expected", "oracle",
                                                                               "large_model_confidence")}}},
        "i35_rule": {"split": "test", "system": "s9_ei_sa", "large_model": large, "n": 39, "registered_rates": {
            "rate_10": {"rule_minus_confidence": {"estimate": 0.0, "low": -0.05, "high": 0.05,
                                                  "reading": "no difference was detected"}}}}}]})
    for k, run in enumerate(runs, start=1):
        cmp9_write_json(r5 / "run_summaries" / f"{run}_test_run_summary.json", {"timing_test": {"items": 39, "forward_seconds": 7.8}})
        cmp9_write_json(r5 / "checks_v05" / f"{run}_swap_run_summary.json", {"timing_test": {"items": 36, "forward_seconds": 7.2}})
    cmp9_write_json(r4 / "run_summaries" / "s7_b3_timing_test_run_summary.json", {"timing_test": {"seconds_per_item": 0.02}})
    timing = {"evidence_effect": {"test": {"gpu_seconds_per_item": 1.6}}, "pubmedqa_answer": {"test": {"gpu_seconds_per_item": 0.8}}}
    cmp9_write_json(r4 / "large_model" / "gemma4_31b_timing.json", timing)
    cmp9_write_json(r5 / "large_model" / "gemma4_31b_long_timing.json", {"evidence_effect": {"test": {"gpu_seconds_per_item": 2.4}}})
    (r5 / "pod_rate.txt").write_text("created_utc 2026-10-01T10:00:00Z\nhourly_usd 3.52\n", encoding="utf-8")
    return {"kwargs": {"results": r5, "results8": r4, "results7": r3, "results_public": rp, "records_public": public,
                       "records_ei": records_ei, "records9": records9},
            "ids": ids, "y": y}


def cmp9_log_loss(folder, system, ids):
    return CMP9_M.log_loss(*cmp9_aligned(folder / "predictions" / f"{system}_evidence_effect_test.csv", ids))


def test_compare9_end_to_end_k1_to_k10_zeros_and_systems_not_built(tmp_path):
    study = cmp9_study(tmp_path)
    kw, ids = study["kwargs"], study["ids"]
    r5, r4 = kw["results"], kw["results8"]
    out = CMP9.run(**kw, cache_dir=tmp_path / "cache", resamples=40, resamples_bar=200, k3_repeats=2)
    assert out is not None and json.loads((r5 / "comparison.json").read_text())["plan"] == "docs/prompt_session9.md, COMPARISONS"

    def comps(section, metric="log_loss"):
        return [c for c in out["comparisons"] if c["section"] == section and c["metric"] == metric]

    k1 = comps("K1")
    assert [(c["system"], c["reference"]) for c in k1] == [("s9_ei_sa", "prior"), ("s9_ei_sa", "gemma4_31b")]
    assert k1[1]["difference"]["estimate"] == pytest.approx(out["bar_intervals"]["v05_minus_large_model"]["estimate"], abs=1e-9)
    assert out["bar_intervals"]["v05_minus_prior"]["resamples"] == 200 and out["bar_intervals"]["v05_minus_prior"]["articles"] == 13
    bar = out["release_bar"]
    assert bar["specialist"] == "s9_ei_sa" and bar["large_model"] == "gemma4_31b" and bar["criteria"]["2"]["result"] == "PASS"
    assert bar["criteria"]["1"]["values"]["limit"] == 0.3 and set(bar["criteria"]) == {"1", "2", "3", "4"}
    assert [(c["system"], c["reference"]) for c in comps("K2")] == [("s9_b", "s8_ei"), ("s9_ei_sa", "s8_ei")]
    assert [(c["system"], c["reference"]) for c in comps("K3_test")] == [("s9_ei_sa", "s9_b")]
    # K8, K9 and K10 read the files calibrate_session9.py apply writes
    (k8,) = comps("K8")
    assert (k8["system"], k8["reference"]) == ("s9_ei_wavg", "s9_ei")
    assert k8["difference"]["estimate"] == pytest.approx(cmp9_log_loss(r5, "s9_ei_wavg", ids) - cmp9_log_loss(r5, "s9_ei", ids),
                                                         abs=1e-9)
    assert out["k8"]["families"] == {"s9_ei_wavg": "vector", "s9_ei": "vector"} and out["k8"]["v05_family"] == "vector"
    k9 = [c for c in out["comparisons"] if c["section"] == "K9"]
    assert {(c["system"], c["reference"]) for c in k9} == {("s9_ei_sa", "s9_ei_sa_s8rule")} and len(k9) == 8
    assert all(c["reading"] == "no difference was detected" and c["excludes"].startswith("no difference was detected: the difference is 0")
               for c in k9)  # the two files are the same here
    assert out["k9"]["families_table"]["chosen"] == "dirichlet_odir" and out["k9"]["session8_rule_family"] == "temperature"
    (k10,) = comps("K10")
    assert (k10["system"], k10["reference"]) == ("s9_ei_sa", "s9_ei") and out["k10"]["kept"] is True
    assert k10["difference"]["estimate"] == pytest.approx(cmp9_log_loss(r5, "s9_ei_sa", ids) - cmp9_log_loss(r5, "s9_ei", ids),
                                                          abs=1e-9)
    # K2 against session 8's file; the prior is v0.4's (session 8's training records)
    (k2,) = [c for c in comps("K2") if c["system"] == "s9_b"]
    assert k2["reference_estimate"] == pytest.approx(cmp9_log_loss(r4, "s8_ei", ids), abs=1e-9)
    assert out["ei_prior"]["counts"] == [10, 10, 10]
    # K3 on the clean validation prompts, beside recipe_selection.json
    k3 = comps("K3")
    assert [(c["system"], c["reference"], c["split"], c["n"]) for c in k3] == [("s9_swap_1_2", "s9_b_1_2", "validation", 30),
                                                                              ("s9_lr1_1", "s9_b_1", "validation", 30)]
    assert [s["candidate"] for s in out["k3"]["not_made"]] == ["long", "lr2", "combo"]
    assert set(out["k3"]["systems"]) == {"s9_b_1_2", "s9_swap_1_2", "s9_b_1", "s9_lr1_1"}
    assert out["k3"]["systems"]["s9_swap_1_2"]["session8_rule_family"] in ("temperature", "temperature_intercepts")
    assert out["k3"]["metrics"]["s9_swap_1_2"]["log_loss"]["low"] is not None and out["k3"]["articles"] == 10
    assert out["k3"]["recipe_selection"]["winner"] == {"recipe": "swap", "runs": ["s9_swap_1", "s9_swap_2", "s9_swap_3"]}
    # K4 to K7
    assert out["k4"]["readings"]["v05"]["none"]["reading"] == "its answers depend on the article's text"
    assert len([c for c in out["comparisons"] if c["section"] == "K5"]) == 4 * 5
    assert [(c["system"], c["reference"]) for c in comps("K5")][0] == ("s7_b3", "biogpt_large_pqa-raw")
    assert [(c["system"], c["reference"]) for c in comps("K6")] == [("readmission_pool_v2", "readmission_pool"),
                                                                   ("xgboost_codes", "xgboost_tuned"),
                                                                   ("catboost_codes", "catboost_tuned")]
    assert comps("K6", "macro_f1_tuned")[0]["difference"]["estimate"] is not None
    assert out["k6"]["chosen"] == "new" and out["k7"]["pairs"][0]["areas"]["aurc"]["estimate"] == 0.1
    # systems not built, and the comparisons they stop
    not_built = {nb["system"]: nb["reason"] for nb in out["not_built"]}
    assert "a made-up reason" in not_built["tabicl"] and not_built["gemma4_31b_long"].startswith("not run")
    assert "does not exist" in not_built["lightgbm_tuned_refit"]
    (tab,) = [c for c in out["comparisons_not_made"] if c["section"] == "K6"]
    assert tab["system"] == "tabicl" and "a made-up reason" in tab["reason"]
    # the sensitivity: articles 0, 5 and 10 are flagged
    assert out["sensitivity"]["flagged_test_prompts"] == 9 and out["sensitivity"]["flags"] == {"Questionable": 9}
    sens = comps("K1_sensitivity")
    assert [(c["n"], c["subset"], c["reference"]) for c in sens] == [(30, "unflagged", "prior"), (30, "unflagged", "gemma4_31b")]
    unflagged = out["sensitivity"]["metrics"]
    assert set(unflagged) == {"s9_ei_sa", "prior", "gemma4_31b"} and unflagged["s9_ei_sa"]["n"] == 30
    assert unflagged["s9_ei_sa"]["log_loss"]["low"] is not None
    assert sens[1]["difference"]["estimate"] == pytest.approx(unflagged["s9_ei_sa"]["log_loss"]["estimate"]
                                                              - unflagged["gemma4_31b"]["log_loss"]["estimate"], abs=1e-9)
    # the slope and intercept of v0.5's specialists; the costs
    cal = out["calibration_slope_intercept"]
    assert set(cal["evidence_effect"]["answers"]) == {"increased", "decreased", "no_difference"}
    assert set(cal["evidence_effect"]["answers"]["increased"]) >= {"intercept", "slope", "intercept_with_slope"}
    assert cal["pubmedqa_answer"]["system"] == "s7_b3" and set(cal["pubmedqa_answer"]["answers"]) == {"yes", "no", "maybe"}
    assert cal["readmit_window"]["slope_30d"] == out["metrics"]["readmit_window"]["readmission_pool_v2"]["calibration_slope_30d"]
    cost = out["cost_per_1000_questions"]
    assert cost["specialists"]["evidence_effect"]["seconds_per_question"] == pytest.approx((3 * 7.8 + 3 * 7.2) / 39)
    assert cost["large_model"]["evidence_effect"]["usd"] == pytest.approx(1000 * 1.6 / 3600 * 3.52)
    assert "second time" in out["contamination"]
    # zeros.json, comparison.csv
    zeros = json.loads((r5 / "zeros.json").read_text())["rows_whose_true_answer_prints_as_0"]
    assert zeros["s9_b"]["test"]["evidence_effect"] == {"rows": 39, "true_answer_printed_zero": 1}
    assert zeros["s9_ei_sa"]["test"]["evidence_effect"]["true_answer_printed_zero"] == 0
    assert zeros["s9_ei_sa"]["validation"]["evidence_effect"]["rows"] == 10
    assert zeros["s9_b_1-raw"]["validation"]["evidence_effect"]["rows"] == 36
    assert zeros["readmission_pool_v2"]["validation"]["readmit_window"]["rows"] == 40
    with open(r5 / "comparison.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert tuple(rows[0]) == CMP9.C.CSV_COLUMNS
    sections = {r["section"] for r in rows}
    assert {"metrics", "K1", "K2", "K3", "K3_metrics", "K3_test", "K4", "K5", "K6", "K7", "K8", "K9", "K10", "K1_sensitivity",
            "calibration_slope_intercept", "cost_per_1000_questions", "release_bar", "note"} <= sections
    assert any(r["section"] == "K9" and "the difference is 0 in every resample" in r["note"] for r in rows)
    (decided,) = [r for r in rows if r["section"] == "K3" and r["metric"] == "recipe_selection"]
    assert decided["note"] == ("recipe_selection.json's decisions: passing candidates swap; the combination not called; the "
                               "winning recipe swap; the weight average used None; swap averaging kept True")
    text = (r5 / "comparison.json").read_text(encoding="utf-8") + (r5 / "comparison.csv").read_text(encoding="utf-8")
    assert chr(0x2013) not in text and chr(0x2014) not in text  # no en or em dash in what the comparisons write


def test_compare9_a_stopped_run_resumes_to_the_same_comparison(tmp_path):
    kw = cmp9_study(tmp_path)["kwargs"]
    whole = CMP9.run(**kw, cache_dir=tmp_path / "cache_whole", resamples=20, resamples_bar=50, k3_repeats=2)
    for _ in range(3):  # each run computes one block, then stops without writing
        assert CMP9.run(**kw, cache_dir=tmp_path / "cache_parts", max_seconds=0.0, resamples=20, resamples_bar=50,
                        k3_repeats=2) is None
    assert len(list((tmp_path / "cache_parts").glob("*.json"))) == 3
    again = CMP9.run(**kw, cache_dir=tmp_path / "cache_parts", resamples=20, resamples_bar=50, k3_repeats=2)
    for key in ("metrics", "comparisons", "bar_intervals", "release_bar", "k3", "calibration_slope_intercept", "not_built"):
        assert again[key] == whole[key], key


def test_compare9_the_long_large_model_and_a_weight_average_not_built(tmp_path):
    kw = cmp9_study(tmp_path, large="gemma4_31b_long", wavg=False)["kwargs"]
    out = CMP9.run(**kw, cache_dir=tmp_path / "cache", resamples=20, resamples_bar=50, k3_repeats=2)
    assert out["bar_intervals"]["v05_minus_large_model"]["reference"] == "gemma4_31b_long"
    k1 = [c for c in out["comparisons"] if c["section"] == "K1" and c["metric"] == "log_loss"]
    assert [c["reference"] for c in k1] == ["prior", "gemma4_31b_long"] and out["release_bar"]["large_model"] == "gemma4_31b_long"
    assert "gemma4_31b_long" in out["release_bar"]["criteria"]["3"]["rule"]
    assert not any(nb["system"] == "gemma4_31b_long" for nb in out["not_built"])
    assert out["k8"]["not_built"].startswith("not built: recipe_selection.json records no weight average")
    (k8,) = [c for c in out["comparisons_not_made"] if c["section"] == "K8"]
    assert k8["system"] == "s9_ei_wavg" and "no weight average" in k8["reason"]
    assert not any(c["section"] == "K8" for c in out["comparisons"])
    assert out["cost_per_1000_questions"]["large_model"]["evidence_effect"]["system"] == "gemma4_31b_long"
    assert out["cost_per_1000_questions"]["large_model"]["evidence_effect"]["usd"] == pytest.approx(1000 * 2.4 / 3600 * 3.52)
    cal = json.loads((kw["results"] / "calibration_s9.json").read_text())
    cmp9_write_json(kw["results"] / "calibration_s9.json", dict(cal, v05_ei={"system": "s9_ei"}))
    with pytest.raises(SystemExit, match="calibration_s9.json names s9_ei as v0.5's EI specialist"):
        CMP9.run(**kw, cache_dir=tmp_path / "cache", resamples=20, resamples_bar=50, k3_repeats=2)


# ======================================================================================================================
# Part 8 of 11: session 9's pod tools (src/train/train_qwen_head.py's pilot-s8, src/tools/upload_to_pod.py and
# scripts/pod_session9.sh)
# From test_pod_session9.py. Module-level names prefixed pod; test names test_pod_*.
#
# Tests for session 9's pod tools (docs/prompt_session9.md, step 4), on made-up inputs:
#
# - pilot-s8's --headroom-gib: the default, 16, gives session 8's choice; 0 lets a setting with a higher peak pass, through
#   the rule's functions and through the command line (pilot.json records the headroom);
# - upload_to_pod.py accepting session 9's stages (and its run command passing them to the pod), the driver's syntax and
#   stages, and step 9's upload list;
# - scripts/pod_session9.sh run off the pod, with EDDM_POD_BASE set to a folder under tmp_path and every python, GPU and
#   install command replaced by stand-ins on PATH (python runs the project's modules as stand-ins that write the files the
#   modules write, at the sizes the driver checks, and runs the driver's own inline checks on the real interpreter with
#   stand-in torch, huggingface_hub and tabicl modules; nvidia-smi, pip and uv are stand-ins; nothing is downloaded and no
#   real uv runs): all stopping at each of its stops (osf_filed.txt missing, the reading checks' stop with exit 3, the cost
#   rule's stop with exit 4, and the first failure), and continuing after Johann's decision; the whole chain to the
#   package, its check passing, with session 8's fallback to gradient checkpointing, a combination that includes long (so
#   the large stages run), and tabicl and biogpt NOT BUILT; prepare's checks; and a stage detaching itself.
# ======================================================================================================================

POD_HERE = Path(__file__).resolve()
POD_SRC = POD_HERE.parents[1] if (POD_HERE.parents[1] / "train" / "train_qwen_head.py").exists() else Path.home() / "mnt" / "ed-decision-model" / "src"
POD_ROOT = POD_SRC.parent

POD_DRIVER = POD_ROOT / "scripts" / "pod_session9.sh"
POD_S9_DRIVER = "/root/eddm_s9/code/scripts/pod_session9.sh"
POD_S9_STAGES = ("prepare", "checks_s8", "pilot_long", "train", "trees", "tabicl", "prepare_large", "smoke_large", "large_validation",
             "biogpt", "calibrate", "freeze", "testpreds", "package", "all")


# pilot-s8's --headroom-gib -------------------------------------------------------------------------------------------------

def pod_attempt(name, rate, peak, oom=False):
    return {"setting": name, "micro_batch": 1, "grad_accum": 32, "gradient_checkpointing": name != "a", "train_items_per_second": rate,
            "peak_gpu_memory_gib": peak, "out_of_memory": oom}


def test_pod_pilot_s8_headroom_gib_in_the_rule():
    import train_qwen_head as Q
    a, b = pod_attempt("a", 2.0, 61.0), pod_attempt("b", 1.5, 30.0)
    assert Q.S8_ROOM_GIB == 16.0 and Q.S8_LIMIT_GIB == 76.0
    assert Q.choose_s8([a, b])["setting"] == "b" and Q.choose_s8([a, b], 16)["setting"] == "b"   # 61 + 16 = 77 > 76: session 8's choice
    assert Q.choose_s8([a, b], 0)["setting"] == "a"                                             # 61 + 0 <= 76: the higher peak passes
    assert Q.s8_passes(pod_attempt("a", 2.0, 76.0), 0) and not Q.s8_passes(pod_attempt("a", 2.0, 76.5), 0)
    assert Q.s8_passes(pod_attempt("a", 2.0, 60.0)) and not Q.s8_passes(pod_attempt("a", 2.0, 60.5))   # 60 + 16 = 76 passes, 60.5 does not
    assert not Q.s8_passes(pod_attempt("a", None, 40.0, oom=True), 0)                                # out of memory never passes
    assert Q.choose_s8([pod_attempt("a", None, 79.0, oom=True), pod_attempt("b", 1.5, 77.0)], 0) is None
    assert Q.s8_passes(pod_attempt("a", 2.0, None), 0)                                              # no peak on a CPU (tests only)


def test_pod_pilot_s8_headroom_gib_on_the_command_line(tmp_path, monkeypatch):
    """pilot-s8 through main(): the settings measured by a stand-in, nothing loaded; the default keeps session 8's rule and
    its text, and --headroom-gib 0 chooses the faster setting whose peak plus 16 GiB was above 76."""
    import torch
    import transformers
    import train_qwen_head as Q

    peaks, rates = {"a": 61.0, "b": 30.0, "c": 20.0}, {"a": 2.0, "b": 1.5, "c": 1.0}

    def measured(args, name, micro_batch, grad_accum, checkpointing, train_items, pad_id, device, ac):
        out = {"setting": name, "micro_batch": micro_batch, "grad_accum": grad_accum, "gradient_checkpointing": checkpointing,
               "steps": args.steps, "train_items_per_second": rates[name], "peak_gpu_memory_gib": peaks[name], "out_of_memory": False}
        out["passes"] = Q.s8_passes(out, getattr(args, "headroom_gib", Q.S8_ROOM_GIB))
        return out

    monkeypatch.setattr(Q, "s8_attempt", measured)
    monkeypatch.setattr(Q, "load_base", lambda *a, **k: None)
    monkeypatch.setattr(Q, "build_model", lambda *a, **k: (torch.nn.Linear(1, 1), None))
    monkeypatch.setattr(Q, "timed_run", lambda model, items, pad_id, device, batch_size, ac: ([], 1.0))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: types.SimpleNamespace(pad_token_id=0))
    torch.save([{"ids": [1]}] * 3, tmp_path / "train.pt")
    torch.save([{"ids": [1]}] * 2, tmp_path / "val.pt")
    common = ["pilot-s8", "--base-dir", str(tmp_path), "--train", str(tmp_path / "train.pt"), "--val", str(tmp_path / "val.pt"),
              "--device", "cpu"]
    s8 = Q.main(common + ["--out-json", str(tmp_path / "pilot.json")])
    s9 = Q.main(common + ["--headroom-gib", "0", "--out-json", str(tmp_path / "pilot_long.json")])
    assert s8["headroom_gib"] == 16.0 and s8["chosen"]["setting"] == "b" and not s8["stop_and_ask"]
    assert [s["passes"] for s in s8["settings"]] == [False, True]
    assert s8["plan"] == "docs/prompt_session8.md, THE PILOT AND THE COST RULE"
    assert "its peak GPU memory plus 16 GiB (room for arm A beside it) is at most 76 GiB" in s8["rule"]
    assert s9["headroom_gib"] == 0.0 and s9["chosen"]["setting"] == "a" and [s["passes"] for s in s9["settings"]] == [True, True]
    assert "plus 0 GiB" in s9["rule"] and "docs/prompt_session9.md" in s9["plan"]
    saved = json.loads((tmp_path / "pilot_long.json").read_text())
    assert saved["chosen"]["setting"] == "a" and saved["headroom_gib"] == 0.0 and saved["validation_items"] == 2
    peaks["b"] = 61.0  # neither passes at 16 GiB: setting c is run, as session 8's rule says
    s8c = Q.main(common + ["--out-json", str(tmp_path / "pilot_c.json")])
    assert [s["setting"] for s in s8c["settings"]] == ["a", "b", "c"] and s8c["chosen"]["setting"] == "c"


# upload_to_pod.py, the driver's stages and step 9's upload list ----------------------------------------------------------

def test_pod_upload_accepts_session9_stages_and_runs_them(monkeypatch):
    assert all(s in U.DRIVER_COMMANDS for s in POD_S9_STAGES)
    assert "upload_list" not in U.DRIVER_COMMANDS  # printed on the linked shell, never run on the pod
    assert U.log_dir(POD_S9_DRIVER) == "/root/eddm_s9/logs" and U.manifest_path(POD_S9_DRIVER) == "/root/eddm_s9/uploads_manifest.json"
    calls = []

    class FakePod:
        def __init__(self, url, token, timeout=300):
            pass

        def python(self, code, timeout=600):
            calls.append(code)
            return "started all (process 1); log /root/eddm_s9/logs/all.log"

        def close(self):
            pass

    monkeypatch.setattr(U, "Pod", FakePod)
    argv = ["upload_to_pod.py", "--url", "https://pod-8888.proxy.runpod.net", "--token", "made-up", "--driver", POD_S9_DRIVER]
    for stage in ("all", "checks_s8", "pilot_long", "train", "tabicl", "biogpt"):
        monkeypatch.setattr(sys, "argv", argv + ["run", stage])
        assert U.main() == 0
        assert repr(POD_S9_DRIVER) in calls[-1] and repr([stage]) in calls[-1]
    monkeypatch.setattr(sys, "argv", argv + ["log", "all"])
    assert U.main() == 0 and "/root/eddm_s9/logs" in calls[-1] and "all" in calls[-1]
    monkeypatch.setattr(sys, "argv", argv + ["run", "upload_list"])
    with pytest.raises(SystemExit):
        U.main()


def test_pod_driver_syntax_stages_and_usage():
    assert subprocess.run(["bash", "-n", str(POD_DRIVER)]).returncode == 0
    text = POD_DRIVER.read_text(encoding="utf-8")
    for s in POD_S9_STAGES:
        assert f"\n  {s})\n" in text, s
    out = subprocess.run(["bash", str(POD_DRIVER)], capture_output=True, text=True)
    assert out.returncode == 1 and "Stages:" in out.stdout and "upload_list" in out.stdout
    out = subprocess.run(["bash", str(POD_DRIVER), "no_such_stage"], capture_output=True, text=True)
    assert out.returncode == 1 and "unknown stage no_such_stage" in out.stdout
    assert all(ord(c) < 128 for c in text)  # plain ASCII: no en or em dashes
    for fixed in ("--headroom-gib 0", "pod_rate.txt", "osf_filed.txt", "uploads_manifest.json", "johann_decision", "nice -n 10 python",
                  "--system-site-packages", "--no-deps", "39c04fb31dfc8094b0f04c7bb7973cf612606c0c", "906bfd4b4dc7f14ee4320094d8b41684abff8539",
                  '"xgboost==3.2.0" "catboost==1.2.10"', "tabicl==2.2.0", "sacremoses==0.2.0", "import vllm._C_stable_libtorch",
                  "torchcodec==0.17.0+cpu", "records_manifest_s4_pod.json", "records_manifest_s8_pod.json", "records_manifest_pod.json"):
        assert fixed in text, fixed


def test_pod_upload_list_prints_step_9_uploads():
    out = subprocess.run(["bash", str(POD_DRIVER), "upload_list", "code"], capture_output=True, text=True, cwd="/")
    paths = [p for p in out.stdout.splitlines() if p]
    assert len(paths) == len(set(paths))
    for p in ("scripts/pod_session9.sh", "src/tools/upload_to_pod.py", "src/train/train_qwen_head.py", "src/questions/session8.yaml",
              "results/fiorillo_v0_5/fixed_versions.json", "results/fiorillo_v0_5/records_manifest.json",
              "results/fiorillo_v0_5/license_recheck.csv", "results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json",
              "results/fiorillo_v0_4/pilot.json", "results/fiorillo_v0_4/comparison.json", "results/fiorillo_v0_4/calibration_s8.json",
              "results/fiorillo_v0_4/predictions/s8_ei-raw_evidence_effect_test.csv",
              "results/fiorillo_v0_4/predictions/gemma4_31b_evidence_effect_validation.csv",
              "results/fiorillo_v0_3/predictions/lightgbm_tuned_readmit_window_validation.csv",
              "results/public_v0/predictions/arm_laya_readmit_window_validation.csv",
              "results/fiorillo_v0_3/predictions/s7_b3_readmit_window_validation.csv", "results/fiorillo_v0_3/calibration_s7.json",
              "results/public_v0/checksums.json", "data/processed/session8/article_selection.jsonl",
              "data/processed/session8/bm25_stats.json", "data/processed/session9/long/article_selection_long.jsonl"):
        assert p in paths, p
    assert not [p for p in paths if p.startswith("src/tests/") or "__pycache__" in p]
    out = subprocess.run(["bash", str(POD_DRIVER), "upload_list"], capture_output=True, text=True, cwd="/")
    assert out.returncode == 0, out.stderr
    assert f"--driver {POD_S9_DRIVER} upload --local-root . --remote-root /root/eddm_s9/code --list" in out.stdout
    assert "--local-root models/fiorillo_v0_3/arm_b --remote-root /root/eddm_s9/models/s7_b3 adapter.safetensors" in out.stdout
    for k in (1, 2, 3):
        assert f"--local-root models/fiorillo_v0_4/s8_ei_{k} --remote-root /root/eddm_s9/models/s8_ei_{k} adapter.safetensors" in out.stdout
    assert "COMMIT.txt" in out.stdout and "run prepare" in out.stdout and "pod_token.txt" in out.stdout


# scripts/pod_session9.sh off the pod: the stand-ins ----------------------------------------------------------------------

POD_PYTHON_STUB = r'''#!/bin/bash
# python for a test of scripts/pod_session9.sh: the project's modules (src/...) run as stand-ins (fake_modules.py); the
# driver's own inline code runs on the real interpreter with stand-in torch, huggingface_hub, tabicl and the like first on
# its path; vLLM's and Triton's checks are skipped
printf 'python %s\n' "$*" >> "$STUB_LOG"
if [ -n "${STUB_REAL_RECIPE:-}" ] && [ "${1:-}" = src/eval/recipe_session9.py ] && [ "${2:-}" = next ]; then
  shift  # the real rule (THE RECIPE RULE), on the made-up pod's records and results
  exec "$REAL_PY" "$STUB_REPO/src/eval/recipe_session9.py" "$@" --records9 "$PWD/data/processed/session9"
fi
case "${1:-}" in
  src/*) exec "$REAL_PY" "$STUB_DIR/fake_modules.py" "$@" ;;
esac
export PYTHONPATH="$STUB_DIR/fakemods"
if [ "${1:-}" = "-" ]; then
  code=$(cat)
  case "$code" in *"import vllm"*|*"cuda.bindings"*|*"import triton"*) echo "[stand-in] a vLLM check skipped"; exit 0 ;; esac
  printf '%s\n' "$code" | "$REAL_PY" "$@"
  exit $?
fi
if [ -f "${1:-}" ] && grep -q "import vllm" "$1"; then echo "[stand-in] a vLLM check skipped"; exit 0; fi
exec "$REAL_PY" "$@"
'''

POD_NVIDIA_SMI_STUB = r'''#!/bin/bash
printf 'nvidia-smi %s\n' "$*" >> "$STUB_LOG"
case "$*" in
  *memory.used*) echo "${STUB_GPU_MIB:-0}" ;;
  *) echo "| NVIDIA-SMI 570.211.01   Driver Version: 570.211.01   CUDA Version: 12.8 |"; echo "| H100 80GB HBM3 (stand-in) |" ;;
esac
'''

POD_PIP_STUB = r'''#!/bin/bash
printf 'pip %s\n' "$*" >> "$STUB_LOG"
'''

POD_UV_STUB = r'''#!/bin/bash
# uv for a test: no package is installed; "uv venv ... DIR" makes DIR/bin/python, the python stand-in
printf 'uv %s\n' "$*" >> "$STUB_LOG"
if [ "${1:-}" = venv ]; then
  dir="${*: -1}"
  mkdir -p "$dir/bin" && cp "$STUB_DIR/python" "$dir/bin/python" && chmod +x "$dir/bin/python"
fi
'''

POD_FAKE_PACKAGES = ("transformers", "tokenizers", "laya", "numpy", "scipy", "sklearn", "pandas", "xgboost", "catboost", "lightgbm")
POD_FAKE_DISTRIBUTIONS = {"tabicl": "2.2.0", "einops": "0.8.1", "psutil": "7.2.2", "tqdm": "4.70.1"}

POD_FAKE_TORCH = '''"""torch's stand-in for the driver's inline checks."""
__version__ = "2.8.0+cu128"


class _Cuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_name(index=0):
        return "NVIDIA H100 80GB HBM3 (stand-in)"


cuda = _Cuda()


class version:
    cuda = "12.8"
'''

POD_FAKE_HUB = '''"""huggingface_hub's stand-in: snapshot_download writes the scenario's files for the repository and downloads nothing."""
import json
import os
from pathlib import Path

__version__ = "1.33.0"


def snapshot_download(repo_id, revision=None, local_dir=None, allow_patterns=None, **kwargs):
    out = Path(local_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = json.loads(Path(os.environ["STUB_SCENARIO"]).read_text()).get("downloads", {}).get(repo_id, {"README.md": "stand-in"})
    for name, content in files.items():
        (out / name).write_text(content)
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(f"snapshot_download {repo_id} {revision}\\n")
    return str(out)
'''

POD_FAKE_MODULES = r'''"""Stand-ins for the project's modules, so that scripts/pod_session9.sh runs off the pod in a test: no model, no GPU and no
network. Each writes the files its module writes, at the sizes the driver checks, and checks the inputs the driver must
give it; the scenario (STUB_SCENARIO, a JSON file) makes chosen calls fail or stop."""
import datetime
import hashlib
import json
import math
import os
import random
import shlex
import sys
from pathlib import Path

SCEN = json.loads(Path(os.environ["STUB_SCENARIO"]).read_text())
CODE = Path.cwd()  # the driver runs every module from the code root
RES = CODE / "results" / "fiorillo_v0_5"
REC9 = CODE / "data" / "processed" / "session9"
EI = "evidence_effect"
N_VAL, N_TEST = 1237, 1218
SEEDS = {1: 20260927, 2: 20260928, 3: 20260929}
LR = {"b": 5e-5, "swap": 5e-5, "long": 5e-5, "lr1": 1e-4, "lr2": 2e-4}


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def say(message):
    print(f"[stand-in] {message}", flush=True)


def opt(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def opts(args, name):
    out = []
    if name in args:
        for a in args[args.index(name) + 1:]:
            if a.startswith("--"):
                break
            out.append(a)
    return out


def count(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def rows(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def table(path, header, n, row="0,0.5,0.3,0.2"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.writelines(f"r{i},{row}\n" for i in range(n))


def probs(path, n):
    table(path, "id,label,p0,p1,p2", n)


def logits(path, n):
    table(path, "id,label,z0,z1,z2", n, "0,1.0,0.5,0.1")


def fails(key, value=True):
    got = SCEN.get(key)
    if got is True or (isinstance(got, list) and value in got):
        say(f"{key}: {value} fails (the scenario)")
        sys.exit(1)


def need(*paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        say(f"missing inputs: {missing}")
        sys.exit(1)


def train_qwen_head(args):
    cmd = args[0]
    if cmd == "predict":
        system, split, qid = opt(args, "--system"), opt(args, "--splits"), opt(args, "--qids")
        fails("predict_fail", system)
        need(Path(opt(args, "--phase-dir")) / "adapter.safetensors")
        path = Path(opt(args, "--records-dir")) / f"{qid}_{split}.jsonl"
        records = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        n, strength, rng = len(records), SCEN.get("strength", {}).get(system, 2.0), random.Random(system)
        rows_out = []
        for i, r in enumerate(records):  # the label's logit raised by the scenario's strength, with noise seeded by the system's name
            y = int(r.get("label", i % 3))
            z = [rng.gauss(0.0, 1.0) + (strength if j == y else 0.0) for j in range(3)]
            e = [math.exp(v - max(z)) for v in z]
            rows_out.append((r.get("id", f"r{i}"), y, z, [v / sum(e) for v in e]))
        for target, cells in ((Path(opt(args, "--out")) / f"{system}-raw_{qid}_{split}.csv", 3),
                              (Path(opt(args, "--logits-dir")) / f"{system}_{qid}_{split}_logits.csv", 2)):
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write("id,label," + ",".join(f"{'p' if cells == 3 else 'z'}{j}" for j in range(3)) + "\n")
                f.writelines(f"{i},{y}," + ",".join(f"{v:.9g}" if cells == 3 else repr(v) for v in (p if cells == 3 else z)) + "\n"
                             for i, y, z, p in rows_out)
        save(opt(args, "--run-summary"), {"system": system, f"items_{split}": {"records": n, "items": n, "skipped": 0, "truncated": 0},
                                          f"timing_{split}": {"items": n, "forward_seconds": 1.0}})
    elif cmd == "train":
        name = opt(args, "--system-name")
        fails("train_fail", name)
        if name in SCEN.get("train_oom", []) and "--gradient-checkpointing" not in args:
            print("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB", flush=True)
            sys.exit(1)
        need(opt(args, "--train"), opt(args, "--val"), Path(opt(args, "--init")) / "adapter.safetensors")
        out = Path(opt(args, "--out"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter.safetensors").write_text(f"stand-in adapter {name}")
        save(out / "config.json", {"system": name, "max_len": int(opt(args, "--max-len")), "args": args})
        table(opt(args, "--log"), "step,loss", 2)
        save(opt(args, "--metrics"), {"system": name, "args": args})
    elif cmd == "pilot-s8":
        fails("pilot_crash")
        need(opt(args, "--train"), opt(args, "--val"))
        passes = SCEN.get("pilot_long_passes", True)
        setting = {"setting": "a", "micro_batch": 1, "grad_accum": 32, "gradient_checkpointing": False, "train_items_per_second": 2.0,
                   "peak_gpu_memory_gib": 70.0 if passes else 79.5, "out_of_memory": not passes, "passes": passes}
        save(opt(args, "--out-json"), {"headroom_gib": float(opt(args, "--headroom-gib", 16)), "settings": [setting],
                                       "chosen": setting if passes else None, "stop_and_ask": not passes, "written_utc": now()})


def qwen_items(args):
    out, files = Path(opt(args, "--out")), opts(args, "--records")
    n = sum(count(f) for f in files)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("stand-in items")
    cut = 1 if out.stem in SCEN.get("items_cut", []) else 0
    save(out.with_suffix(".summary.json"), {"records": n, "items": n, "skipped": 0, "truncated": cut, "max_len": int(opt(args, "--max-len")),
                                             "files": files})


def settings(recipe, parts=()):
    """recipe_session9.py's recipe_settings, for the stand-in of next."""
    if recipe == "combo":
        long_, swap = "long" in parts, "swap" in parts
        where = "long" if long_ else "clean"
        return {"recipe": "combo", "parts": sorted(parts), "lr_choice": None,
                "train_records": ("data/processed/session9/combo/evidence_effect_train.jsonl" if swap else
                                  f"data/processed/session9/{where}/evidence_effect_train.jsonl"),
                "monitor_records": f"data/processed/session9/{where}/evidence_effect_validation_monitor.jsonl",
                "train_items": "combo_train" if swap else f"{where}_train", "monitor_items": f"{where}_monitor",
                "epochs": 1 if swap else 2, "lr_adapters": 5e-5, "max_len": 6144 if long_ else 4096,
                "memory_setting": "long" if long_ else "base",
                "validation_records_dir": "data/processed/session9/long" if long_ else "data/processed/session8",
                "exchanged_validation_dir": "data/processed/session9/checks/swap_long" if long_ else "data/processed/session9/checks/swap"}
    out = {"recipe": recipe, "lr_adapters": LR[recipe], "epochs": 2, "max_len": 4096, "memory_setting": "base",
           "train_records": "data/processed/session9/clean/evidence_effect_train.jsonl",
           "monitor_records": "data/processed/session9/clean/evidence_effect_validation_monitor.jsonl",
           "train_items": "clean_train", "monitor_items": "clean_monitor", "validation_records_dir": "data/processed/session8",
           "exchanged_validation_dir": "data/processed/session9/checks/swap"}
    if recipe == "swap":
        out.update({"train_records": "data/processed/session9/swap/evidence_effect_train.jsonl", "train_items": "swap_train", "epochs": 1})
    elif recipe == "long":
        out.update({"train_records": "data/processed/session9/long/evidence_effect_train.jsonl", "train_items": "long_train",
                    "monitor_records": "data/processed/session9/long/evidence_effect_validation_monitor.jsonl",
                    "monitor_items": "long_monitor", "max_len": 6144, "memory_setting": "long",
                    "validation_records_dir": "data/processed/session9/long",
                    "exchanged_validation_dir": "data/processed/session9/checks/swap_long"})
    return out


def has_run(system, n=N_VAL):
    lp, rp = RES / "logits" / f"{system}_{EI}_validation_logits.csv", RES / "predictions" / f"{system}-raw_{EI}_validation.csv"
    return lp.exists() and rp.exists() and rows(lp) == n and rows(rp) == n


def next_action():
    plan = SCEN.get("recipe", {})
    parts = plan.get("combo_parts", [])
    for run in plan.get("runs", ["s9_b_1", "s9_b_2", "s9_b_3"]):
        recipe, k = run.split("_")[1], int(run.split("_")[2])
        if recipe == "combo" and "swap" in parts and not (REC9 / "combo" / "combo.json").exists():
            return {"action": "combo_records", "parts": [p for p in ("swap", "long") if p in parts]}
        if not has_run(run):
            return {"action": "train", "name": run, "recipe": recipe, "seed": SEEDS[k], **settings(recipe, parts)}
    winner, runs = plan.get("winner", "b"), plan.get("winner_runs", ["s9_b_1", "s9_b_2", "s9_b_3"])
    st = settings(winner, parts)
    if not has_run("s9_ei_wavg"):
        return {"action": "wavg", "name": "s9_ei_wavg", "source_runs": runs, "max_len": st["max_len"],
                "validation_records_dir": st["validation_records_dir"]}
    uses_wavg, keep_sa = plan.get("uses_wavg", False), plan.get("swap_averaging", False)
    singles = ["s9_ei_wavg"] if uses_wavg else runs
    n_exch = count(CODE / st["exchanged_validation_dir"] / f"{EI}_validation.jsonl")
    for s in singles:
        if not has_run(f"{s}_exch", n_exch):
            return {"action": "exch", "name": f"{s}_exch", "phase_system": s, "records_dir": st["exchanged_validation_dir"], "rows": n_exch}
    x = "s9_ei_wavg" if uses_wavg else "s9_ei"
    for name in ("s9_b", "s9_ei", f"{x}_sa"):
        probs(RES / "predictions" / f"{name}-raw_{EI}_validation.csv", N_VAL)
    save(RES / "recipe_selection.json", {"plan": "stand-in", "v05_ei": {
        "system": f"{x}_sa" if keep_sa else x, "base_system": x, "swap_averaging": keep_sa, "recipe": winner, "runs": runs,
        "single_pass_systems": singles, "max_len": st["max_len"], "validation_records_dir": st["validation_records_dir"],
        "test_records_dir": st["validation_records_dir"], "checks_swap_dir": st["exchanged_validation_dir"],
        "other_form": x if keep_sa else f"{x}_sa", "swap_augmented": winner == "combo" and "swap" in parts}})
    return {"action": "done"}


def recipe(args):
    if args[0] == "cost":
        need(opt(args, "--pod-rate"), RES / "pilot_long.json")
        stops = bool(SCEN.get("cost_stops"))
        save(RES / "cost_rule.json", {"stops": stops, "maintenance_before_the_end": False, "projected_total_usd": 130.0 if stops else 80.0,
                                      "stop_line_usd": 115.0, "options_in_order": [], "dropped": []})
        return 4 if stops else 0
    action = next_action()
    if SCEN.get("next_chatter"):
        print("a line that is not an assignment")
    for key, value in action.items():
        if isinstance(value, (list, tuple)):
            value = " ".join(str(v) for v in value)
        print(f"EDDM_{key.upper()}={shlex.quote(str(value))}")
    return 0


def reading_checks(args):
    for k in (1, 2, 3):
        for inp in ("none", "title", "swap"):
            need(RES / "checks_s8" / f"s8_ei_{k}_{inp}-raw_{EI}_test.csv", RES / "checks_s8" / "logits" / f"s8_ei_{k}_{inp}_{EI}_test_logits.csv")
    stop = bool(SCEN.get("checks_stop"))
    save(RES / "checks_s8.json", {"system": "s8_ei", "stop": stop})
    return 3 if stop else 0


def ei_session9(args):
    if args[0] == "combo":
        parts, out = opts(args, "--parts"), Path(opt(args, "--out"))
        lines = (out / ("long" if "long" in parts else "clean") / f"{EI}_train.jsonl").read_text().splitlines(keepends=True)
        (out / "combo").mkdir(parents=True, exist_ok=True)
        (out / "combo" / f"{EI}_train.jsonl").write_text("".join(lines * (2 if "swap" in parts else 1)))
        save(out / "combo" / "combo.json", {"parts": sorted(parts)})
    elif args[0] == "rebuild":
        need(opt(args, "--selection-long"), opt(args, "--licenses"), opt(args, "--manifest"))
        save(opt(args, "--check-out"), {"all_equal": True})


def evidence_inference(args):
    if args[0] == "download":
        print(json.dumps({"complete": True}))
    elif args[0] == "from-selection":
        need(opt(args, "--selection"), opt(args, "--manifest"))
        save(opt(args, "--check-out"), {"all_equal": True})


def weight_average(args):
    runs = opts(args, "--runs")
    if len(runs) != 3:
        sys.exit(2)
    need(*[Path(r) / "adapter.safetensors" for r in runs])
    out = Path(opt(args, "--out"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "adapter.safetensors").write_text("stand-in weight average")
    save(out / "config.json", {"system": "s9_ei_wavg", "sources": runs})
    save(Path(opt(args, "--results")) / "weight_average.json", {"sources": runs, "passed": True})


def trees(args):
    cmd, split = args[0], opt(args, "--split", "validation")
    models = Path(opt(args, "--models-dir"))
    models.mkdir(parents=True, exist_ok=True)
    if split == "test":
        need(RES / "release_candidate.json", RES / "testpreds_log.json")

    def pred(m, s):
        probs(RES / "predictions" / f"{m}_readmit_window_{s}.csv", 6997 if s == "validation" else 6998)

    if cmd == "tabicl":
        fails("tabicl_fail")
        if split == "validation":
            (models / "tabicl.pkl").write_text("stand-in")
            save(models / "tabicl_state.json", {})
            save(RES / "tabicl.json", {})
        pred("tabicl", split)
        return 0
    path = RES / "trees_v2.json"
    data = json.loads(path.read_text()) if path.exists() else {"members": {}}
    if cmd == "recode":
        save(models / "recode_tables.json", {})
    elif cmd == "refit":
        fails("trees_fail")
        for m, f in (("xgboost_codes", "xgboost_codes.json"), ("catboost_codes", "catboost_codes.cbm")):
            if split == "validation":
                (models / f).write_text("stand-in")
                data["members"][m] = {}
            pred(m, split)
    elif cmd == "lightgbm":
        passed = SCEN.get("lightgbm_passes", True)
        if split == "validation":
            (models / "lightgbm_tuned.txt").write_text("stand-in")
            data["members"]["lightgbm_tuned"] = {"check": {"passed": passed}}
            pred("lightgbm_tuned", split)
        elif passed:
            pred("lightgbm_tuned", split)
    elif cmd == "pool":
        need(RES / "predictions" / "xgboost_codes_readmit_window_validation.csv")
        save(RES / "readmission_pool_v2.json", {"chosen": "v2"})
        pred("readmission_pool_v2", "validation")
    save(path, data)
    save(RES / "trees_v2_progress.json", {"last": cmd})


def biogpt(args):
    need(Path(opt(args, "--model-dir")) / "DOWNLOAD_DONE.txt")
    if args[0] == "validation":
        fails("biogpt_fail")
        save(RES / "biogpt.json", {"validation": True})
        probs(RES / "predictions" / "biogpt_large_pqa-raw_pubmedqa_answer_validation.csv", 150)
    else:
        need(RES / "release_candidate.json")
        probs(RES / "predictions" / "biogpt_large_pqa-raw_pubmedqa_answer_test.csv", 500)


def calibrate(args):
    cmd = args[0]
    sel = json.loads((RES / "recipe_selection.json").read_text())["v05_ei"]
    if cmd == "fit":
        save(RES / "calibration_s9.json", {"v05_ei": {"system": sel["system"]}})
        probs(RES / "predictions" / f"{sel['system']}_{EI}_validation.csv", N_VAL)
        if sel["max_len"] == 6144:
            need(RES / "large_model" / f"gemma4_31b_long-raw_{EI}_validation.csv")
            probs(RES / "predictions" / f"gemma4_31b_long_{EI}_validation.csv", N_VAL)
    elif cmd == "ece":
        save(RES / "ece_limit_evidence_effect.json", {"limit": 0.05})
    elif cmd == "freeze":
        base = Path(opt(args, "--base"))
        need(base / "osf_filed.txt")
        for stage in ("trees", "tabicl", "biogpt", "calibrate"):
            if not ((base / "stage_done" / f"{stage}.txt").exists() or (base / "not_built" / f"{stage}.txt").exists()):
                say(f"the freeze waits for {stage}")
                sys.exit(1)
        if (RES / "release_candidate.json").exists():
            sys.exit("release_candidate.json exists; it is written once")
        keys = ["results/fiorillo_v0_5/recipe_selection.json", "results/fiorillo_v0_5/calibration_s9.json", "results/fiorillo_v0_3/calibration_s7.json"]
        save(RES / "release_candidate.json", {"time_utc": now(), "sha256": {k: hashlib.sha256((CODE / k).read_bytes()).hexdigest() for k in keys}})


def handoff(args):
    if args[0] == "curves":
        pair = opts(args, "--pair")
        need(pair[2], pair[4], pair[5])
    else:
        runs = opts(args, "--runs")
        if len(runs) != 3:
            sys.exit(2)
        need(opt(args, "--fiorillo"), opt(args, "--large"), opt(args, "--records"), *runs)
    save(opt(args, "--out"), {"command": args[0], "args": args})


def power(args):
    need(opt(args, "--records"), opt(args, "--system"), opt(args, "--reference"))
    save(opt(args, "--out"), {"args": args})


def large_model(args):
    cmd, splits, qids = args[0], opts(args, "--splits"), opts(args, "--qids") or ["evidence_effect", "pubmedqa_answer"]
    items, results = Path(opt(args, "--items-dir")), Path(opt(args, "--results", RES))
    if "test" in splits:
        need(results / "release_candidate.json", results / "testpreds_log.json")
    if cmd == "prompts":
        items.mkdir(parents=True, exist_ok=True)
        for q in qids:
            for s in splits:
                n = count(Path(opt(args, "--records-ei")) / f"{EI}_{s}.jsonl") if q == EI else 150
                (items / f"prompts_{q}_{s}.jsonl").write_text("".join('{"id": "x"}\n' for _ in range(n)))
        return 0
    if qids != [EI]:
        sys.exit("answer and second take --qids evidence_effect only here")
    out = results / "large_model"
    out.mkdir(parents=True, exist_ok=True)
    letters_path, timing_path = out / "gemma4_31b_letters.json", out / "gemma4_31b_timing.json"
    letters = json.loads(letters_path.read_text()) if letters_path.exists() else {}
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    for s in splits:
        n = count(items / f"prompts_{EI}_{s}.jsonl")
        probs(out / f"gemma4_31b-raw_{EI}_{s}.csv", n)
        table(out / f"gemma4_31b_logprobs_{EI}_{s}.csv", "id,label,method,letter_mass,logp_A,logp_B,logp_C,top_token", n,
              "0,top20,0.9,-0.1,-3,-4,A")
        letters.setdefault(EI, {})[s] = {"items": n, "letters_summed_probability": {"median": 0.9}, "answered_second_way": 0}
        timing.setdefault(EI, {})[s] = {"items": n, "wall_seconds": 1.0}
    save(letters_path, letters)
    save(timing_path, timing)


def smoke(args):
    for flag in ("--script", "--tokenizer-dir", "--model-dir", "--items-dir", "--out-dir"):
        if flag not in args or "eddm_s8" in opt(args, flag):
            sys.exit(f"{flag} is not session 9's (smoke_large_session8.py's defaults name /root/eddm_s8)")
    need(Path(opt(args, "--items-dir")) / f"prompts_{EI}_validation.jsonl", Path(opt(args, "--items-dir")) / "prompts_pubmedqa_answer_validation.jsonl")


HANDLERS = {"train_qwen_head.py": train_qwen_head, "qwen_items.py": qwen_items, "recipe_session9.py": recipe,
            "reading_checks_session9.py": reading_checks, "ei_session9.py": ei_session9, "evidence_inference.py": evidence_inference,
            "weight_average_session9.py": weight_average, "trees_session9.py": trees, "biogpt_session9.py": biogpt,
            "calibrate_session9.py": calibrate, "handoff_curves_session9.py": handoff, "power_session9.py": power,
            "large_model_session8.py": large_model, "smoke_large_session8.py": smoke}


if __name__ == "__main__":
    handler = HANDLERS.get(Path(sys.argv[1]).name)
    sys.exit((handler(sys.argv[2:]) or 0) if handler else 0)
'''


class PodPod:
    """A made-up pod: /root/eddm_s9 under tmp_path, its code folder holding only the records and results files the stages
    read, the stand-ins on PATH, and a scenario."""

    def __init__(self, tmp_path, scenario, osf=True, prepared=True):
        self.tmp = tmp_path
        self.base = tmp_path / "pod"
        self.code = self.base / "code"
        self.res = self.code / "results" / "fiorillo_v0_5"
        self.models = self.base / "models"
        self.stubs = tmp_path / "stubs"
        self.log = tmp_path / "stub.log"
        self.scenario_path = tmp_path / "scenario.json"
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.scenario(**scenario)
        self.stubs.mkdir(parents=True)
        for name, text in (("python", POD_PYTHON_STUB), ("nvidia-smi", POD_NVIDIA_SMI_STUB), ("pip", POD_PIP_STUB), ("uv", POD_UV_STUB)):
            (self.stubs / name).write_text(text)
            (self.stubs / name).chmod(0o755)
        (self.stubs / "fake_modules.py").write_text(POD_FAKE_MODULES)
        mods = self.stubs / "fakemods"
        for name, text in [("torch", POD_FAKE_TORCH), ("huggingface_hub", POD_FAKE_HUB), ("tabicl", "class TabICLClassifier:\n    pass\n")] + \
                [(p, f'__version__ = "stand-in"\n') for p in POD_FAKE_PACKAGES]:
            (mods / name).mkdir(parents=True)
            (mods / name / "__init__.py").write_text(text)
        for name, version in POD_FAKE_DISTRIBUTIONS.items():
            (mods / f"{name}-{version}.dist-info").mkdir()
            (mods / f"{name}-{version}.dist-info" / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        for rel, n in (("data/processed/session8/evidence_effect_validation.jsonl", 1237), ("data/processed/session8/evidence_effect_test.jsonl", 1218),
                       ("data/processed/session9/checks/none/evidence_effect_test.jsonl", 1218),
                       ("data/processed/session9/checks/title/evidence_effect_test.jsonl", 1209),
                       ("data/processed/session9/checks/swap/evidence_effect_test.jsonl", 1218),
                       ("data/processed/session9/checks/swap/evidence_effect_validation.jsonl", 1237),
                       ("data/processed/session9/checks/swap_long/evidence_effect_test.jsonl", 1218),
                       ("data/processed/session9/checks/swap_long/evidence_effect_validation.jsonl", 1237),
                       ("data/processed/session9/long/evidence_effect_validation.jsonl", 1237),
                       ("data/processed/session9/long/evidence_effect_test.jsonl", 1218),
                       ("data/processed/session9/clean/evidence_effect_train.jsonl", 6),
                       ("data/processed/session9/clean/evidence_effect_validation.jsonl", 654),
                       ("data/processed/session9/clean/evidence_effect_validation_monitor.jsonl", 3),
                       ("data/processed/session9/long/evidence_effect_train.jsonl", 6),
                       ("data/processed/session9/long/evidence_effect_validation_monitor.jsonl", 3),
                       ("data/processed/session9/swap/evidence_effect_train.jsonl", 12)):
            self.jsonl(rel, n)
        self.json("results/fiorillo_v0_4/pilot.json", {"stop_and_ask": False, "chosen": {
            "setting": "a", "micro_batch": 1, "grad_accum": 32, "gradient_checkpointing": False, "train_items_per_second": 2.86,
            "peak_gpu_memory_gib": 36.4}})
        self.json("results/fiorillo_v0_3/calibration_s7.json", {"s7_b3": {}})
        self.json("results/fiorillo_v0_5/fixed_versions.json", {"gemma_4_31b_it": {"revision": "842da37", "weights_bytes": 10}})
        (self.code / "results/fiorillo_v0_4/predictions").mkdir(parents=True, exist_ok=True)
        (self.code / "results/fiorillo_v0_4/predictions/gemma4_31b_evidence_effect_validation.csv").write_text(
            "id,label,p0,p1,p2\n" + "".join(f"v{i},0,0.5,0.3,0.2\n" for i in range(1237)))
        for m in ("s7_b3", "s8_ei_1", "s8_ei_2", "s8_ei_3"):
            (self.models / m).mkdir(parents=True)
            (self.models / m / "adapter.safetensors").write_text(f"made-up {m}")
        created = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        (self.base / "pod_rate.txt").write_text(f"created_utc {created}\nhourly_usd 3.52\nmaintenance_utc none\n")
        if osf:
            self.osf()
        if prepared:  # what prepare leaves: its marker and the items
            (self.base / "stage_done").mkdir(parents=True, exist_ok=True)
            (self.base / "stage_done" / "prepare.txt").write_text("2026-10-01T12:30:00Z\n")
            for name in ("records_manifest_s4_pod.json", "records_manifest_s8_pod.json", "records_manifest_pod.json"):
                self.json(f"results/fiorillo_v0_5/{name}", {"all_equal": True})
            rec9 = self.code / "data" / "processed" / "session9"
            for key, length, rel in (("clean_train", 4096, "clean/evidence_effect_train.jsonl"), ("swap_train", 4096, "swap/evidence_effect_train.jsonl"),
                                     ("long_train", 6144, "long/evidence_effect_train.jsonl"),
                                     ("clean_monitor", 4096, "clean/evidence_effect_validation_monitor.jsonl"),
                                     ("long_monitor", 6144, "long/evidence_effect_validation_monitor.jsonl")):
                (self.base / "items").mkdir(parents=True, exist_ok=True)
                (self.base / "items" / f"{key}.pt").write_text("made-up items")
                n = len((rec9 / rel).read_text().splitlines())
                (self.base / "items" / f"{key}.summary.json").write_text(json.dumps(
                    {"records": n, "items": n, "skipped": 0, "truncated": 0, "max_len": length, "files": [str(rec9 / rel)]}))
        self.env = {"PATH": f"{self.stubs}:/usr/bin:/bin", "HOME": os.environ.get("HOME", str(tmp_path)), "LANG": "C.UTF-8",
                    "TMPDIR": str(tmp_path), "EDDM_POD_BASE": str(self.base), "EDDM_DETACHED": "1", "STUB_DIR": str(self.stubs),
                    "STUB_LOG": str(self.log), "STUB_SCENARIO": str(self.scenario_path), "REAL_PY": sys.executable, "STUB_REPO": str(POD_ROOT)}

    def scenario(self, **values):
        old = json.loads(self.scenario_path.read_text()) if self.scenario_path.exists() else {}
        downloads = {"google/gemma-4-31B-it": {"model.safetensors": "0123456789", "tokenizer.json": "{}", "chat_template.jinja": "t"},
                     "google/gemma-4-E2B-it": {"model.safetensors": "e2b", "tokenizer.json": "{}"},
                     "microsoft/BioGPT-Large-PubMedQA": {"config.json": "{}"}}
        self.scenario_path.write_text(json.dumps({"downloads": downloads, **old, **values}))

    def osf(self):
        (self.base / "osf_filed.txt").write_text("link https://osf.io/made-up\nfiled_utc 2026-10-01T12:20:00Z\n")

    def jsonl(self, rel, n):
        """n made-up records: validation and test files share their split's IDs (val-<i>, test-<i>), as the inputs of one
        prompt do; labels cycle through the three answers."""
        path = self.code / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        prefix = "val" if "validation." in path.name else ("test" if "_test." in path.name else path.parent.name)
        path.write_text("".join(json.dumps({"id": f"{prefix}-{i:04d}", "label": i % 3}) + "\n" for i in range(n)))

    def json(self, rel, data):
        path = self.code / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))

    def drive(self, stage, timeout=240, detached=True):
        env = dict(self.env)
        if not detached:
            env.pop("EDDM_DETACHED")
        return subprocess.run(["bash", str(POD_DRIVER), stage], env=env, capture_output=True, text=True, timeout=timeout)

    def stage_log(self, stage):
        path = self.base / "logs" / f"{stage}.log"
        return path.read_text() if path.exists() else ""

    def calls(self, needle):
        return [line for line in self.log.read_text().splitlines() if needle in line] if self.log.exists() else []

    def wait_for(self, path, seconds=60):
        t0 = time.time()
        while not Path(path).exists():
            assert time.time() - t0 < seconds, f"{path} did not appear"
            time.sleep(0.2)

    def wait_for_trees(self):
        """all leaves the trees stage beside train to finish when it stops; the test waits for it before it looks further."""
        self.wait_for(self.base / "stage_done" / "trees.txt")
        lock = self.base / "locks" / "trees.lock"
        if shutil.which("flock"):
            subprocess.run(["flock", str(lock), "true"], timeout=60, check=True)

    def started(self, out):
        return [line.split("all: ")[1].split(" started")[0] for line in out.stdout.splitlines() if " started (log " in line]


def pod_why(pod, out):
    return out.stdout[-3000:] + out.stderr[-2000:] + "".join(f"\n--- {p.name}\n" + p.read_text()[-1500:] for p in sorted((pod.base / "logs").glob("*.log")))


def test_pod_all_stops_while_osf_filed_is_missing(tmp_path):
    pod = PodPod(tmp_path, {}, osf=False)
    out = pod.drive("all")
    assert out.returncode == 5, pod_why(pod, out)
    assert "osf_filed.txt is missing" in out.stdout and not pod.started(out)
    assert not (pod.base / "logs" / "checks_s8.log").exists() and not pod.calls("train_qwen_head.py")
    for stage in ("checks_s8", "freeze"):  # the two stages that read it refuse by themselves too
        out = pod.drive(stage)
        assert out.returncode == 5 and "osf_filed.txt is missing" in out.stdout, pod_why(pod, out)
    assert not (pod.res / "release_candidate.json").exists() and not pod.calls("train_qwen_head.py")
    pod.osf()
    (pod.base / "stage_done" / "prepare.txt").rename(pod.base / "prepare_moved.txt")
    out = pod.drive("all")
    assert out.returncode == 1 and "prepare has not finished" in out.stdout and not pod.started(out)


def test_pod_all_stops_at_the_reading_checks_stop_and_continues_after_johanns_decision(tmp_path):
    pod = PodPod(tmp_path, {"checks_stop": True})
    out = pod.drive("all")
    assert out.returncode == 3, pod_why(pod, out)
    assert "STOPPED at checks_s8: THE READING CHECKS' STOP" in out.stdout and pod.started(out) == ["checks_s8"]
    assert "THE READING CHECKS' STOP" in pod.stage_log("checks_s8")
    assert json.loads((pod.res / "checks_s8.json").read_text())["stop"] is True
    for k in (1, 2, 3):
        for inp, n in (("none", 1218), ("title", 1209), ("swap", 1218)):
            raw = pod.res / "checks_s8" / f"s8_ei_{k}_{inp}-raw_evidence_effect_test.csv"
            assert len(raw.read_text().splitlines()) == n + 1
            assert (pod.res / "checks_s8" / "logits" / f"s8_ei_{k}_{inp}_evidence_effect_test_logits.csv").exists()
    # the reading checks' predictions are never in predictions/, logits/ or the staging folder (the freeze's guard)
    assert not list((pod.res / "predictions").glob("*")) and not list((pod.res / "logits").glob("*"))
    assert not [p for p in (pod.base / "predictions_staging").rglob("*") if p.is_file()]
    assert not (pod.base / "stage_done" / "checks_s8.txt").exists() and not (pod.base / "logs" / "pilot_long.log").exists()
    out = pod.drive("all")  # run again with no decision: the same stop, the predictions not made again
    assert out.returncode == 3 and len(pod.calls("--system s8_ei_1_none")) == 1, pod_why(pod, out)
    data = json.loads((pod.res / "checks_s8.json").read_text())
    data["johann_decision"] = "made-up: continue (2026-10-01T15:00:00Z)"
    (pod.res / "checks_s8.json").write_text(json.dumps(data))
    pod.scenario(pilot_crash=True)  # so that the chain stops at its next stage
    out = pod.drive("all")
    assert out.returncode == 1 and "STOPPED at pilot_long: the first failure" in out.stdout, pod_why(pod, out)
    assert "Johann's decision" in pod.stage_log("checks_s8") and (pod.base / "stage_done" / "checks_s8.txt").exists()
    assert len(pod.calls("reading_checks_session9.py score")) == 2 and pod.started(out) == ["checks_s8", "pilot_long"]
    assert not (pod.base / "logs" / "train.log").exists() and not (pod.base / "logs" / "trees.log").exists()


def test_pod_all_stops_at_the_first_failure(tmp_path):
    pod = PodPod(tmp_path, {"train_fail": ["s9_b_2"]})
    out = pod.drive("all")
    assert out.returncode == 1, pod_why(pod, out)
    assert "STOPPED at train: the first failure" in out.stdout and pod.started(out) == ["checks_s8", "pilot_long", "train"]
    assert "trees started beside train" in out.stdout and "left to finish on its own" in out.stdout
    assert len((pod.res / "predictions" / "s9_b_1-raw_evidence_effect_validation.csv").read_text().splitlines()) == 1238
    assert not (pod.models / "s9_b_2" / "adapter.safetensors").exists() and not (pod.base / "logs" / "tabicl.log").exists()
    assert json.loads((pod.res / "pilot_long.json").read_text())["headroom_gib"] == 0.0
    pod.wait_for_trees()
    assert not (pod.base / "stage_done" / "train.txt").exists() and (pod.base / "stage_done" / "pilot_long.txt").exists()


def test_pod_all_stops_at_the_cost_rule_then_runs_the_chain_to_the_package(tmp_path):
    pod = PodPod(tmp_path, {"cost_stops": True, "train_oom": ["s9_b_1"], "next_chatter": True})
    out = pod.drive("all")
    assert out.returncode == 4, pod_why(pod, out)
    assert "STOPPED at train: THE COST RULE'S STOP" in out.stdout and "THE COST RULE'S STOP" in pod.stage_log("train")
    assert not pod.calls("train_qwen_head.py train") and not (pod.models / "s9_b_1").exists()
    pod.wait_for_trees()
    out = pod.drive("all")  # no decision yet: cost_rule.json stops again, and the cost rule is not run again
    assert out.returncode == 4 and len(pod.calls("recipe_session9.py cost")) == 1, pod_why(pod, out)
    cost = json.loads((pod.res / "cost_rule.json").read_text())
    cost["johann_decision"] = "made-up: continue as projected (2026-10-01T16:00:00Z)"
    (pod.res / "cost_rule.json").write_text(json.dumps(cost))
    out = pod.drive("all", timeout=600)
    assert out.returncode == 0, pod_why(pod, out)
    assert pod.started(out) == ["train", "tabicl", "biogpt", "calibrate", "freeze", "testpreds", "package"]
    assert "GEMMA ON THE LONGER INPUT does not apply" in out.stdout and "the chain finished" in out.stdout
    assert "PACKAGE CHECK PASSED" in pod.stage_log("package")
    # session 8's fallback: s9_b_1 ran out of memory, restarted with gradient checkpointing, kept on for the base setting
    trains = pod.calls("train_qwen_head.py train")
    assert [("s9_b_1" in c, "--gradient-checkpointing" in c) for c in trains[:2]] == [(True, False), (True, True)]
    assert all("--gradient-checkpointing" in c for c in trains[2:]) and len(trains) == 4
    assert (pod.base / "checkpointing_on_base.txt").exists()
    assert all("--init " + str(pod.models / "s7_b3") in c and "--keep last --freeze-steps 0 --eval-every 80" in c for c in trains)
    assert "--lr-adapters 5e-05 --max-len 4096" in trains[0] and "--epochs 2" in trains[0]
    assert "next also printed (not read): a line that is not an assignment" in pod.stage_log("train")
    # the weight average, the exchanged validation records and the test predictions
    assert pod.calls("weight_average_session9.py build --runs") and len(pod.calls("_exch --records-dir")) == 3
    for s in ("s9_b_1", "s9_b_2", "s9_b_3", "s9_ei_wavg"):
        assert len((pod.res / "predictions" / f"{s}-raw_evidence_effect_test.csv").read_text().splitlines()) == 1219
        assert (pod.res / "run_summaries" / f"{s}_test_run_summary.json").exists()
    for s in ("s9_b_1", "s9_b_2", "s9_b_3"):
        for inp, n in (("none", 1218), ("title", 1209), ("swap", 1218)):
            assert len((pod.res / "checks_v05" / f"{s}_{inp}-raw_evidence_effect_test.csv").read_text().splitlines()) == n + 1
    assert "data/processed/session9/checks/swap --qids" in " ".join(pod.calls("--system s9_b_1_swap"))
    log = json.loads((pod.res / "testpreds_log.json").read_text())
    rc = json.loads((pod.res / "release_candidate.json").read_text())
    assert log["testpreds_start_utc"] > rc["time_utc"]
    for split in ("validation", "test"):
        assert (pod.res / "predictions" / f"tabicl_readmit_window_{split}.csv").exists()
        assert (pod.res / "predictions" / f"biogpt_large_pqa-raw_pubmedqa_answer_{split}.csv").exists()
    assert pod.calls("--no-deps tabicl==2.2.0") and pod.calls("venv --python") and pod.calls("--system-site-packages")
    assert (pod.base / "tabicl_env" / "bin" / "python").exists()
    assert pod.calls("refit --members xgboost_codes catboost_codes --threads 12") and pod.calls("pool --device cuda --not-built-dir")
    assert json.loads((pod.res / "not_built.json").read_text())["stages"] == {}
    sums = (pod.base / "outputs" / "SHA256SUMS.txt").read_text().splitlines()
    assert sorted(line.split()[2] for line in sums) == ["fiorillo_v0_5_ei.zip", "fiorillo_v0_5_logs.zip", "fiorillo_v0_5_results.zip",
                                                        "fiorillo_v0_5_trees.zip"]
    fit_rule = json.loads((pod.res / "handoff_rule_v05.json").read_text())["args"]
    assert fit_rule[fit_rule.index("--large-name") + 1] == "gemma4_31b"
    for stage in ("checks_s8", "pilot_long", "train", "trees", "tabicl", "biogpt", "calibrate", "freeze", "testpreds", "package"):
        assert (pod.base / "stage_done" / f"{stage}.txt").exists(), stage
    out = pod.drive("all")  # every stage finished: all runs nothing again
    assert out.returncode == 0 and not pod.started(out), pod_why(pod, out)


def test_pod_all_with_a_long_combination_and_stages_not_built(tmp_path):
    runs = ["s9_b_1", "s9_b_2", "s9_b_3", "s9_swap_1", "s9_long_1", "s9_swap_2", "s9_long_2", "s9_combo_1", "s9_combo_2", "s9_combo_3"]
    pod = PodPod(tmp_path, {"recipe": {"runs": runs, "combo_parts": ["swap", "long"], "winner": "combo",
                                    "winner_runs": ["s9_combo_1", "s9_combo_2", "s9_combo_3"], "uses_wavg": True, "swap_averaging": True},
                         "tabicl_fail": True, "biogpt_fail": True, "lightgbm_passes": False})
    out = pod.drive("all", timeout=600)
    assert out.returncode == 0, pod_why(pod, out)
    assert pod.started(out) == ["checks_s8", "pilot_long", "train", "tabicl", "prepare_large", "smoke_large", "large_validation", "biogpt",
                                "calibrate", "freeze", "testpreds", "package"]
    assert "PACKAGE CHECK PASSED" in pod.stage_log("package") and "SMOKE TEST PASSED" in pod.stage_log("smoke_large")
    markers = json.loads((pod.res / "not_built.json").read_text())["stages"]
    assert sorted(markers) == ["biogpt", "tabicl"] and markers["tabicl"].startswith("NOT BUILT: tabicl, trees_session9.py tabicl failed")
    # the combination's records and items at 6,144 tokens, and its runs on long's memory setting
    assert json.loads((pod.base / "items" / "combo_train.summary.json").read_text())["max_len"] == 6144
    combo = [c for c in pod.calls("train_qwen_head.py train") if "s9_combo_" in c]
    assert len(combo) == 3 and all("combo_train.pt" in c and "--max-len 6144" in c and "--epochs 1" in c for c in combo)
    assert [c for c in pod.calls("train_qwen_head.py train") if "s9_swap_1" in c][0].count("swap_train.pt") == 1
    # GEMMA ON THE LONGER INPUT: the gemma4_31b files copied to gemma4_31b_long, the originals kept
    for split, n in (("validation", 1237), ("test", 1218)):
        for name in ("gemma4_31b", "gemma4_31b_long"):
            assert len((pod.res / "large_model" / f"{name}-raw_evidence_effect_{split}.csv").read_text().splitlines()) == n + 1
    assert all("--qids evidence_effect pubmedqa_answer --records-ei" in c for c in pod.calls("large_model_session8.py prompts --splits validation"))
    assert all("--qids evidence_effect --model-dir" in c for c in pod.calls("large_model_session8.py answer"))
    assert json.loads((pod.res / "handoff_rule_v05.json").read_text())["args"][1:3] == ["--fiorillo-name", "s9_ei_wavg_sa"]
    assert "single pass; no swap-averaged file of this run exists" in pod.stage_log("calibrate")
    # v0.5's single-pass system is the weight average: its reading checks, swap from the 6,144-token inputs
    for inp in ("none", "title", "swap"):
        assert (pod.res / "checks_v05" / f"s9_ei_wavg_{inp}-raw_evidence_effect_test.csv").exists()
    assert "data/processed/session9/checks/swap_long --qids" in " ".join(pod.calls("--system s9_ei_wavg_swap"))
    assert "data/processed/session9/long --qids" in " ".join(pod.calls("--system s9_combo_1 --records-dir"))
    assert not (pod.res / "predictions" / "lightgbm_tuned_readmit_window_test.csv").exists()
    assert not (pod.res / "predictions" / "tabicl_readmit_window_test.csv").exists()


def test_pod_prepare_checks_its_inputs_and_builds_the_items(tmp_path):
    import hashlib

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def uploaded_inputs(code):  # the uploads that prepare's rebuilds read, and the records the stand-ins of the builders read
        for rel in ("data/processed/public/pubmedqa_answer_validation.jsonl", "data/raw/evidence_inference/a.txt",
                    "data/processed/session8/article_selection.jsonl", "data/processed/session9/long/article_selection_long.jsonl",
                    "results/fiorillo_v0_5/license_recheck.csv"):
            (code / rel).parent.mkdir(parents=True, exist_ok=True)
            (code / rel).write_text("made-up\n")

    pod = PodPod(tmp_path, {"downloads": {"Qwen/Qwen3-4B-Base": {"model.safetensors": "made-up weights"}}}, prepared=False)
    code = pod.code
    uploaded_inputs(code)
    pod.json("results/public_v0/checksums.json", {"pubmedqa_answer_validation.jsonl": sha(code / "data/processed/public/pubmedqa_answer_validation.jsonl")})
    pod.json("results/fiorillo_v0_4/records_manifest.json", {"sha256": {"raw": {"a.txt": sha(code / "data/raw/evidence_inference/a.txt")}}})
    pod.json("results/fiorillo_v0_5/records_manifest.json", {"models": {
        **{m: {"adapter.safetensors": sha(pod.models / m / "adapter.safetensors")} for m in ("s7_b3", "s8_ei_1", "s8_ei_2", "s8_ei_3")},
        "s7_b3_adapter_expected": sha(pod.models / "s7_b3" / "adapter.safetensors")}})
    pod.json("results/fiorillo_v0_5/fixed_versions.json", {"qwen3_4b_base": {"revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539",
                                                                               "weights": {"model.safetensors": {"bytes": 15, "sha256": hashlib.sha256(b"made-up weights").hexdigest()}}}})
    manifest = {"roots": {str(code): {"results/fiorillo_v0_5/records_manifest.json": sha(code / "results/fiorillo_v0_5/records_manifest.json")}}}
    (pod.base / "uploads_manifest.json").write_text(json.dumps(manifest))
    out = pod.drive("prepare")
    assert out.returncode == 0, pod_why(pod, out)
    for key, n, length in (("clean_train", 6, 4096), ("swap_train", 12, 4096), ("long_train", 6, 6144), ("clean_monitor", 3, 4096),
                           ("long_monitor", 3, 6144)):
        summary = json.loads((pod.base / "items" / f"{key}.summary.json").read_text())
        assert (summary["records"], summary["max_len"]) == (n, length), key
    for key in ("check_none_test", "check_title_test", "check_swap_test", "check_swap_validation", "check_swap_long_test",
                "check_swap_long_validation", "check_long_validation", "check_long_test"):
        assert (pod.base / "items_check" / f"{key}.summary.json").exists(), key
    for name in ("records_manifest_s4_pod.json", "records_manifest_s8_pod.json", "records_manifest_pod.json"):
        assert (pod.base / name).exists() and (pod.res / name).exists(), name
    assert (pod.base / "stage_done" / "prepare.txt").exists()
    assert "revision 906bfd4b4dc7f14ee4320094d8b41684abff8539" in (pod.models / "qwen3_4b_base" / "REVISION.txt").read_text()
    assert pod.calls("snapshot_download Qwen/Qwen3-4B-Base 906bfd4b4dc7f14ee4320094d8b41684abff8539")
    n_items = len(pod.calls("qwen_items.py"))
    out = pod.drive("prepare")  # run again: the items are not built again
    assert out.returncode == 0 and len(pod.calls("qwen_items.py")) == n_items, pod_why(pod, out)
    # a changed upload stops prepare before its installs; so does a model folder that differs from the manifest
    (code / "results/fiorillo_v0_5/records_manifest.json").write_text("{}")
    out = pod.drive("prepare")
    assert out.returncode != 0 and "uploaded files missing or changed" in out.stdout + out.stderr
    pod2 = PodPod(tmp_path / "second", {"items_cut": ["long_train"], "downloads": {"Qwen/Qwen3-4B-Base": {"model.safetensors": "made-up weights"}}}, prepared=False)
    for rel in ("results/public_v0/checksums.json", "results/fiorillo_v0_4/records_manifest.json", "results/fiorillo_v0_5/fixed_versions.json"):
        (pod2.code / rel).parent.mkdir(parents=True, exist_ok=True)
        (pod2.code / rel).write_bytes((code / rel).read_bytes())
    uploaded_inputs(pod2.code)
    pod2.json("results/fiorillo_v0_5/records_manifest.json", {"models": {m: {"adapter.safetensors": sha(pod2.models / m / "adapter.safetensors")}
                                                                       for m in ("s7_b3", "s8_ei_1", "s8_ei_2", "s8_ei_3")}})
    (pod2.base / "uploads_manifest.json").write_text(json.dumps({"roots": {}}))
    out = pod2.drive("prepare")
    assert out.returncode != 0 and "long_train.summary.json: records were cut or skipped" in out.stdout, pod_why(pod2, out)
    assert not (pod2.base / "stage_done" / "prepare.txt").exists()
    (pod2.models / "s8_ei_2" / "adapter.safetensors").write_text("changed")
    out = pod2.drive("prepare")
    assert out.returncode != 0 and "differ from results/fiorillo_v0_5/records_manifest.json's models digests" in out.stdout + out.stderr


def test_pod_a_stage_detaches_itself_and_logs(tmp_path):
    pod = PodPod(tmp_path, {}, osf=False)
    out = pod.drive("freeze", detached=False)
    assert out.returncode == 0 and out.stdout.startswith("started freeze (process "), out.stdout + out.stderr
    pod.wait_for(pod.base / "logs" / "freeze.log")
    t0 = time.time()
    while "osf_filed.txt is missing" not in pod.stage_log("freeze"):
        assert time.time() - t0 < 30, pod.stage_log("freeze")
        time.sleep(0.2)
    assert not (pod.base / "stage_done" / "freeze.txt").exists()


def test_pod_train_follows_the_real_recipe_rule(tmp_path):
    """The train stage's loop with the real recipe_session9.py next (THE RECIPE RULE with correction 2, on made-up logits:
    long's runs better than base's, lr1's worse; lr2, swap, the combination and lr1's second run dropped by Johann's
    decision under the cost rule), so that the driver and the rule agree on every action: train, the weight average,
    the exchanged validation records and done; v0.5's recipe then reads 6,144 tokens, and the large stages run."""
    strength = {"s9_long_1": 3.5, "s9_long_2": 3.5, "s9_long_3": 3.5, "s9_lr1_1": 1.0}
    pod = PodPod(tmp_path, {"strength": strength})
    pod.env["STUB_REAL_RECIPE"] = "1"
    pod.json("results/fiorillo_v0_5/cost_rule.json", {
        "stops": True, "maintenance_before_the_end": False, "projected_total_usd": 120.0, "stop_line_usd": 115.0,
        "dropped": ["lr2", "swap", "combo", "lr1_2"], "johann_decision": "made-up: drop lr2, swap, the combination and lr1's second run"})
    out = pod.drive("all", timeout=900)
    assert out.returncode == 0, pod_why(pod, out)
    sel = json.loads((pod.res / "recipe_selection.json").read_text())
    v = sel["v05_ei"]
    assert sel["winner"]["recipe"] == "long" and v["recipe"] == "long" and v["max_len"] == 6144
    assert v["runs"] == ["s9_long_1", "s9_long_2", "s9_long_3"] and v["checks_swap_dir"] == "data/processed/session9/checks/swap_long"
    assert sel["candidates"]["long"]["passes"] and not sel["candidates"]["lr1"]["passes"] and "one run" in sel["candidates"]["lr1"]["rule"]
    assert sel["candidates"]["swap"]["status"] == "dropped" and sel["candidates"]["lr2"]["status"] == "dropped"
    assert not sel["combination"]["called"] and sel["dropped"] == ["combo", "lr1_2", "lr2", "swap"]
    trained = [c.split("--system-name ")[1].split()[0] for c in pod.calls("train_qwen_head.py train")]
    assert trained == ["s9_b_1", "s9_b_2", "s9_b_3", "s9_long_1", "s9_lr1_1", "s9_long_2", "s9_long_3"]
    assert all("--max-len 6144" in c and "long_train.pt" in c for c in pod.calls("train_qwen_head.py train") if "s9_long_" in c)
    assert all("--lr-adapters 0.0001" in c for c in pod.calls("train_qwen_head.py train") if "s9_lr1_" in c)
    assert pod.started(out) == ["checks_s8", "pilot_long", "train", "tabicl", "prepare_large", "smoke_large", "large_validation", "biogpt",
                                "calibrate", "freeze", "testpreds", "package"]
    assert "PACKAGE CHECK PASSED" in pod.stage_log("package")
    singles = v["single_pass_systems"]
    for s in singles:
        assert len((pod.res / "predictions" / f"{s}_exch-raw_evidence_effect_validation.csv").read_text().splitlines()) == 1238
        assert (pod.res / "checks_v05" / f"{s}_swap-raw_evidence_effect_test.csv").exists()
    for s in ("s9_long_1", "s9_long_2", "s9_long_3"):
        assert "data/processed/session9/long --qids evidence_effect --splits test" in " ".join(pod.calls(f"--system {s} --records-dir"))


# ======================================================================================================================
# Part 9 of 11: session 9's recipe rule, cost rule, weight average and calibration (src/eval/recipe_session9.py,
# weight_average_session9.py and calibrate_session9.py)
# From test_core_session9.py. Module-level names prefixed core9; test names test_core9_*.
#
# Tests for session 9's recipe rule, cost rule, weight average and calibration (docs/prompt_session9.md: THE RECIPES,
# THE RECIPE RULE, THE PILOT AND THE COST RULE, CALIBRATION, the freeze, testpreds and apply stages of step 3, step 4's
# list for this file, and corrections 1 to 7), on made-up files in tmp_path.
# Every module-level name starts with core9_ and every test with test_core9_.
#
# - THE RECIPE RULE, driven through recipe_session9.py's Decider and its next command (--format env) on made-up logits and
#   raw prediction files. One test checks that a run's score is session 8's rule on the 654 clean prompts. The others
#   replace that rule's fits with the plain log loss, so that each made-up run's score is exactly -log of the probability
#   planted on its clean prompts' answers. They cover the margin and its floor of 0.005, two runs for each candidate
#   (correction 2), passing and failing, lr1 and lr2 as one change, no combination when only the learning rate passes, the
#   combination, the winner, base when none passes, a candidate left with one run, the winner's third run, the weight
#   average's action with three runs and with two (correction 7), I-39's decision, and swap averaging's exchanged-input
#   actions and decision (correction 3).
# - The cost command: cost_rule.json, exit status 4 above the $115 stop under the $125 cap (correction 1) and before an
#   announced maintenance, and next continuing once Johann's decision is recorded in cost_rule.json.
# - The weight average on made-up adapters: the joined update against the mean of the runs' updates (three runs, rank 48
#   and alpha 96; two runs, rank 32 and alpha 64), its refusals, and the files it writes.
# - calibrate_session9.py: fit's calibrated validation files, the ECE limit's input, the freeze's guards, its NOT BUILT
#   markers, checks_s8's files, its SHA-256 table, and apply's guards.
# ======================================================================================================================

core9_HERE = core9_Path(__file__).resolve()
core9_SRC = (core9_HERE.parents[1] if (core9_HERE.parents[1] / "eval" / "recipe_session9.py").exists()
             else core9_Path.cwd() / "src")  # src/tests/ once merged; otherwise run from the repository

core9_approx = core9_pytest.approx
core9_EI, core9_PQA = "evidence_effect", "pubmedqa_answer"
core9_N_VAL, core9_N_CLEAN, core9_N_TEST = 1237, 654, 1218
core9_FAMILIES = ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")
core9_SWAP = core9_np.array([1, 0, 2])  # "significantly increased" and "significantly decreased" exchanged, the third kept

# The made-up validation split: 1,237 prompts, 654 of them clean, and 7 whose intervention and comparator are the same
# text (4 of them clean), which have no exchanged record.
core9_rng = core9_np.random.default_rng(20261001)
core9_IDS = [f"evidence-{i:05d}" for i in range(core9_N_VAL)]
core9_POS = {i: k for k, i in enumerate(core9_IDS)}
core9_LABELS = core9_rng.integers(0, 3, core9_N_VAL)
core9_IS_CLEAN = core9_np.zeros(core9_N_VAL, dtype=bool)
core9_IS_CLEAN[core9_rng.choice(core9_N_VAL, core9_N_CLEAN, replace=False)] = True
core9_CLEAN = [i for i, c in zip(core9_IDS, core9_IS_CLEAN) if c]
core9_SAME_TEXT = set(core9_CLEAN[5:8] + core9_CLEAN[400:401] + [i for i, c in zip(core9_IDS, core9_IS_CLEAN) if not c][:3])
core9_EXCH = [i for i in core9_IDS if i not in core9_SAME_TEXT]
core9_EXCH_ROWS = core9_np.array([core9_POS[i] for i in core9_EXCH])
# A made-up test split for apply, written only after the freeze
core9_TEST_IDS = [f"evidence-test-{i:05d}" for i in range(core9_N_TEST)]
core9_TEST_LABELS = core9_np.random.default_rng(20261002).integers(0, 3, core9_N_TEST)


def core9_softmax(z):
    z = core9_np.asarray(z, dtype=core9_np.float64)
    e = core9_np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def core9_write_text(path, text):
    path = core9_Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def core9_write_json(path, data):
    return core9_write_text(path, core9_json.dumps(data, indent=1) + "\n")


def core9_read_json(path):
    return core9_json.loads(core9_Path(path).read_text(encoding="utf-8"))


def core9_write_jsonl(path, rows):
    return core9_write_text(path, "".join(core9_json.dumps(r) + "\n" for r in rows))


def core9_write_csv(path, header, rows):
    path = core9_Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = core9_csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def core9_read_probs(path):
    """(ids, labels, probabilities) of a predictions CSV (id, label, p0, p1, p2)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(core9_csv.DictReader(f))
    return ([r["id"] for r in rows], core9_np.array([int(r["label"]) for r in rows]),
            core9_np.array([[float(r[f"p{j}"]) for j in range(3)] for r in rows]))


def core9_sha256(path):
    return core9_hashlib.sha256(core9_Path(path).read_bytes()).hexdigest()


def core9_utc(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def core9_write_probs(path, ids, labels, probs):
    """Probabilities as session 4's predictions format prints them (9 significant digits)."""
    return core9_write_csv(path, ["id", "label", "p0", "p1", "p2"],
                           [[i, int(y)] + [f"{v:.9g}" for v in row] for i, y, row in zip(ids, labels, probs)])


def core9_write_system(results, system, z, ids=None, labels=None, split="validation"):
    """A system's logits (every digit) and raw predictions in results, as train_qwen_head.py predict writes them:
    logits/<system>_evidence_effect_<split>_logits.csv and predictions/<system>-raw_evidence_effect_<split>.csv."""
    ids = core9_IDS if ids is None else ids
    labels = core9_LABELS if labels is None else labels
    results = core9_Path(results)
    core9_write_csv(results / "logits" / f"{system}_{core9_EI}_{split}_logits.csv", ["id", "label", "z0", "z1", "z2"],
                    [[i, int(y)] + [repr(float(v)) for v in row] for i, y, row in zip(ids, labels, z)])
    core9_write_probs(results / "predictions" / f"{system}-raw_{core9_EI}_{split}.csv", ids, labels, core9_softmax(z))
    return z


def core9_planted_logits(labels, q):
    """Logits whose softmax gives each row's label the probability q (one per row) and each other answer (1 - q) / 2."""
    labels = core9_np.asarray(labels)
    q = core9_np.broadcast_to(core9_np.asarray(q, dtype=core9_np.float64), labels.shape)
    z = core9_np.repeat(core9_np.log((1.0 - q) / 2.0)[:, None], 3, axis=1)
    z[core9_np.arange(len(labels)), labels] = core9_np.log(q)
    return z


def core9_write_run(results, system, score, other=0.2):
    """A made-up run (or weight average) whose clean prompts' answers all have the probability exp(-score), so that its
    plain log loss on the clean prompts is score. The other prompts' answers get `other`, which no score may read."""
    q = core9_np.where(core9_IS_CLEAN, core9_math.exp(-score), other)
    return core9_write_system(results, system, core9_planted_logits(core9_LABELS, q))


def core9_write_exch(results, system, r, other=0.3):
    """A made-up answer to the exchanged validation records (IDs kept, the two directional answers exchanged in the
    labels): on each clean prompt the exchanged answer gets r, so that exchanged back the prompt's own label has r."""
    labels = core9_SWAP[core9_LABELS[core9_EXCH_ROWS]]
    q = core9_np.where(core9_IS_CLEAN[core9_EXCH_ROWS], r, other)
    return core9_write_system(results, system, core9_planted_logits(labels, q), ids=core9_EXCH, labels=labels)


def core9_noisy(seed, labels, shift):
    """Made-up logits with normal noise and the label's logit raised by shift."""
    rng = core9_np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, (len(labels), 3))
    z[core9_np.arange(len(labels)), labels] += shift
    return z


def core9_layout(tmp_path):
    """The folders recipe_session9.py reads, under tmp_path as under the repository: results/fiorillo_v0_5/ and
    data/processed/session9/ (the 654 clean validation records, and the 1,230 exchanged validation records at 4,096 and
    6,144 tokens)."""
    results = tmp_path / "results" / "fiorillo_v0_5"
    records9 = tmp_path / "data" / "processed" / "session9"
    results.mkdir(parents=True)
    core9_write_jsonl(records9 / "clean" / f"{core9_EI}_validation.jsonl",
                      [{"id": i, "label": int(y)} for i, y, c in zip(core9_IDS, core9_LABELS, core9_IS_CLEAN) if c])
    exch = [{"id": i, "label": int(core9_SWAP[core9_LABELS[core9_POS[i]]])} for i in core9_EXCH]
    for folder in ("swap", "swap_long"):
        core9_write_jsonl(records9 / "checks" / folder / f"{core9_EI}_validation.jsonl", exch)
    return results, records9


def core9_selection(results):
    return core9_read_json(core9_Path(results) / "recipe_selection.json")


def core9_avg_score(*scores):
    """The plain log loss, on the clean prompts, of the probability average of made-up runs with these scores."""
    return -core9_math.log(sum(core9_math.exp(-s) for s in scores) / len(scores))


@core9_pytest.fixture
def core9_plain_rule(monkeypatch):
    """Session 8's rule (calibrate_session7.choose, as recipe_session9.py imported it) replaced by the plain log loss of
    the log-probabilities it is given, reported as the temperature family's out-of-fold mean, the other family worse.
    Returns the number of prompts of every call."""
    sizes = []

    def plain(logp, y, *args, **kwargs):
        logp, y = core9_np.asarray(logp), core9_np.asarray(y)
        sizes.append(len(y))
        loss = float(core9_np.mean(-logp[core9_np.arange(len(y)), y]))
        return "temperature", {"temperature": {"out_of_fold_log_loss": {"mean": loss}},
                               "temperature_intercepts": {"out_of_fold_log_loss": {"mean": loss + 0.25}}}

    monkeypatch.setattr(core9_R9, "choose", plain)
    return sizes


def core9_next(results, records9, capsys, not_built=None):
    """recipe_session9.py next --format env, run in process as the train stage runs it; returns its assignments, each
    value read as one shell word (shlex), as the stage's eval reads them."""
    argv = ["next", "--format", "env", "--results", str(results), "--records9", str(records9)]
    if not_built is not None:
        argv += ["--not-built-dir", str(not_built)]
    capsys.readouterr()
    assert core9_R9.main(argv) == 0
    env = {}
    for line in capsys.readouterr().out.splitlines():
        key, sep, value = line.partition("=")
        assert sep and key.startswith("EDDM_"), line
        words = core9_shlex.split(value)
        assert len(words) == 1, line
        env[key] = words[0]
    return env


def core9_drive(results, records9, capsys, scores, not_built=None):
    """Calls next and writes each run it asks for with its planned score, until it asks for something else or for a run
    with no planned score. Returns the runs trained, in order, with their seeds, and the last assignments."""
    trained = []
    for _ in range(40):
        env = core9_next(results, records9, capsys, not_built)
        if env["EDDM_ACTION"] != "train" or env["EDDM_NAME"] not in scores:
            return trained, env
        core9_write_run(results, env["EDDM_NAME"], scores[env["EDDM_NAME"]])
        trained.append((env["EDDM_NAME"], int(env["EDDM_SEED"])))
    raise AssertionError("next kept asking for runs")


# Made-up scores. Base's spread is 0.002, so m is the floor, 0.005, and a candidate's mean must be below
# (0.300 + 0.302) / 2 - 0.005 = 0.296, with its first run below 0.300 and its second below 0.302.
core9_BASE = {"s9_b_1": 0.300, "s9_b_2": 0.302, "s9_b_3": 0.301}
core9_FAILING = {"s9_long_1": 0.2990, "s9_long_2": 0.3000, "s9_lr1_1": 0.2990, "s9_lr1_2": 0.3010,
                 "s9_lr2_1": 0.3000, "s9_lr2_2": 0.3010}  # means 0.2995, 0.3000 and 0.3005: none passes
core9_SWAP_RUNS = {"s9_swap_1": 0.2900, "s9_swap_2": 0.2910, "s9_swap_3": 0.2890}
core9_SWAP_WINS = {**core9_BASE, **core9_FAILING, **core9_SWAP_RUNS}  # swap passes alone (mean 0.2905) and wins
core9_FIRST = ["s9_swap_1", "s9_long_1", "s9_lr1_1", "s9_lr2_1"]
core9_SECOND = ["s9_swap_2", "s9_long_2", "s9_lr1_2", "s9_lr2_2"]


# ---------------------------------------------------------------- THE RECIPE RULE (recipe_session9.py)

def test_core9_next_command_as_the_train_stage_reads_it(tmp_path):
    """The next command, run as the pod's train stage runs it (its own python process, --format env), with nothing yet
    trained: it asks for s9_b_1 with seed 20260927 and base's settings, and bash's eval of its lines, as the stage
    evaluates them, reads back every value the Decider gives."""
    results, records9 = core9_layout(tmp_path)
    not_built = tmp_path / "pod" / "not_built"
    out = core9_subprocess.run(
        [core9_sys.executable, str(core9_SRC / "eval" / "recipe_session9.py"), "next", "--format", "env", "--results",
         str(results), "--records9", str(records9), "--not-built-dir", str(not_built)],
        capture_output=True, text=True, check=True).stdout
    action, _ = core9_R9.Decider(results, records9, not_built).state()
    script = ("eval \"$(printf '%s\\n' \"$CORE9_OUT\" | grep '^EDDM_[A-Z0-9_]*=')\"; "
              + "; ".join(f"printf '%s\\n' \"$EDDM_{k.upper()}\"" for k in action))
    shown = core9_subprocess.run(["bash", "-c", script], env={**core9_os.environ, "CORE9_OUT": out}, capture_output=True,
                                 text=True, check=True).stdout.splitlines()
    assert shown == [str(v) for v in action.values()]
    assert action == {"action": "train", "name": "s9_b_1", "recipe": "b", "seed": 20260927, "lr_adapters": 5e-5,
                      "epochs": 2, "max_len": 4096, "memory_setting": "base",
                      "train_records": "data/processed/session9/clean/evidence_effect_train.jsonl",
                      "monitor_records": "data/processed/session9/clean/evidence_effect_validation_monitor.jsonl",
                      "train_items": "clean_train", "monitor_items": "clean_monitor",
                      "validation_records_dir": "data/processed/session8",
                      "exchanged_validation_dir": "data/processed/session9/checks/swap"}
    sel = core9_selection(results)
    assert (sel["n_clean_validation"], sel["folds"], sel["repeats"], sel["seed"], sel["dropped"]) == (654, 5, 20, 20260927, [])


def test_core9_rule_score_is_session8_rule_on_the_clean_prompts(tmp_path):
    """A run's score is session 8's calibration rule (the lower of the temperature and temperature_intercepts families'
    out-of-fold log losses, 5 folds, 20 repeats, the seed) on the 654 clean prompts, and s9_b's is the same on the base
    runs' averaged probabilities; m is the base runs' spread, at least 0.005. Every other prompt carries a confident
    answer that would change the scores if the rule read it."""
    results, records9 = core9_layout(tmp_path)
    rng = core9_np.random.default_rng(7)
    probs = []
    for k in (1, 2, 3):
        z = rng.normal(0.0, 1.0, (core9_N_VAL, 3))
        z[core9_np.arange(core9_N_VAL), core9_LABELS] += 0.6 + 0.4 * k
        z[~core9_IS_CLEAN] = [6.0, -6.0, -6.0]
        core9_write_system(results, f"s9_b_{k}", z)
        probs.append(core9_softmax(z))
    action, sel = core9_R9.Decider(results, records9).state()
    assert (action["action"], action["name"], action["seed"]) == ("train", "s9_swap_1", 20260927)

    def rule(p, keep=core9_IS_CLEAN):
        _, fams = core9_C7.choose(core9_np.log(p[keep]), core9_LABELS[keep])
        return min(f["out_of_fold_log_loss"]["mean"] for f in fams.values())

    runs = {f"s9_b_{k}": rule(p) for k, p in zip((1, 2, 3), probs)}
    spread = max(runs.values()) - min(runs.values())
    assert sel["margin"]["base_scores"] == core9_approx(runs, abs=1e-9)
    assert sel["margin"]["spread"] == core9_approx(spread, abs=1e-9)
    assert sel["margin"]["m"] == core9_approx(max(spread, 0.005), abs=1e-9)
    assert sel["s9_b"]["score"] == core9_approx(rule(core9_np.mean(probs, axis=0)), abs=1e-9)
    assert sel == core9_selection(results)
    assert rule(probs[0], core9_np.ones(core9_N_VAL, dtype=bool)) > runs["s9_b_1"] + 0.1


@core9_pytest.mark.parametrize("base, spread, m", [
    ((0.300, 0.302, 0.301), 0.002, 0.005),     # the spread is below the floor: m is 0.005
    ((0.300, 0.3049, 0.302), 0.0049, 0.005),
    ((0.300, 0.310, 0.305), 0.010, 0.010),     # above the floor: m is the spread
    ((0.3123, 0.3020, 0.3071), 0.0103, 0.0103),
])
def test_core9_rule_margin_and_its_floor(tmp_path, capsys, core9_plain_rule, base, spread, m):
    """m is the largest minus the smallest of s9_b_1's to s9_b_3's scores, and at least 0.005; it is recorded with the
    time of its decision. Every score reads the 654 clean prompts only."""
    results, records9 = core9_layout(tmp_path)
    for k, s in enumerate(base, start=1):
        core9_write_run(results, f"s9_b_{k}", s)
    env = core9_next(results, records9, capsys)
    assert (env["EDDM_ACTION"], env["EDDM_NAME"], env["EDDM_SEED"]) == ("train", "s9_swap_1", "20260927")
    sel = core9_selection(results)
    assert sel["margin"]["spread"] == core9_approx(spread, abs=1e-12)
    assert sel["margin"]["m"] == core9_approx(m, abs=1e-12) and sel["margin"]["floor"] == 0.005
    assert sel["decisions"]["margin"]["value"] == core9_approx(m, abs=1e-11) and sel["decisions"]["margin"]["time_utc"]
    assert sel["s9_b"]["score"] == core9_approx(core9_avg_score(*base), abs=1e-12)
    assert set(core9_plain_rule) == {core9_N_CLEAN}


def test_core9_rule_two_runs_for_each_candidate(tmp_path, capsys, core9_plain_rule):
    """Correction 2: each candidate is trained twice (seeds 20260927 and 20260928, every first run before any second run)
    and passes if its mean is below the mean of s9_b_1's and s9_b_2's scores by more than m and each run is below the base
    run with the same seed. swap passes; long passes on its mean but fails on its second seed; lr1 and lr2 have each run
    below its base run but a mean within m. swap, the only passing candidate, wins, and its third run (seed 20260929) is
    asked for with swap's settings."""
    results, records9 = core9_layout(tmp_path)
    scores = {**core9_BASE, "s9_swap_1": 0.2900, "s9_swap_2": 0.2910, "s9_long_1": 0.2800, "s9_long_2": 0.3035,
              "s9_lr1_1": 0.2955, "s9_lr1_2": 0.2975, "s9_lr2_1": 0.2992, "s9_lr2_2": 0.3005}
    trained, env = core9_drive(results, records9, capsys, scores)
    assert trained == ([("s9_b_1", 20260927), ("s9_b_2", 20260928), ("s9_b_3", 20260929)]
                       + [(r, 20260927) for r in core9_FIRST] + [(r, 20260928) for r in core9_SECOND])
    assert (env["EDDM_ACTION"], env["EDDM_NAME"], env["EDDM_SEED"], env["EDDM_RECIPE"]) == ("train", "s9_swap_3", "20260929", "swap")
    assert (env["EDDM_EPOCHS"], env["EDDM_MAX_LEN"], float(env["EDDM_LR_ADAPTERS"])) == ("1", "4096", 5e-5)
    assert env["EDDM_TRAIN_RECORDS"] == "data/processed/session9/swap/evidence_effect_train.jsonl"
    sel = core9_selection(results)
    cand = sel["candidates"]
    for c in ("swap", "long", "lr1", "lr2"):
        assert cand[c]["rule"] == "two runs (correction 2)" and cand[c]["runs"] == [f"s9_{c}_1", f"s9_{c}_2"]
        assert cand[c]["base_mean_12"] == core9_approx(0.301, abs=1e-12) and cand[c]["m"] == core9_approx(0.005, abs=1e-12)
        assert cand[c]["score"] == core9_approx((scores[f"s9_{c}_1"] + scores[f"s9_{c}_2"]) / 2, abs=1e-12)
        assert sel["decisions"][f"candidate_{c}"]["value"] is (c == "swap")
    assert [cand["swap"][k] for k in ("below_base_mean_by_more_than_m", "run_1_below_s9_b_1", "run_2_below_s9_b_2", "passes")] == [True] * 4
    assert [cand["long"][k] for k in ("below_base_mean_by_more_than_m", "run_1_below_s9_b_1", "run_2_below_s9_b_2", "passes")] == [True, True, False, False]
    for c in ("lr1", "lr2"):
        assert [cand[c][k] for k in ("below_base_mean_by_more_than_m", "run_1_below_s9_b_1", "run_2_below_s9_b_2", "passes")] == [False, True, True, False]
    assert sel["passing_changes"] == ["swap"] and sel["combination"]["called"] is False
    assert sel["winner"]["recipe"] == "swap" and sel["decisions"]["winner"]["value"] == "swap"


@core9_pytest.mark.parametrize("lr1, lr2, chosen", [((0.2900, 0.2920), (0.2850, 0.2880), "lr2"),
                                                    ((0.2850, 0.2870), (0.2900, 0.2910), "lr1")])
def test_core9_rule_learning_rate_is_one_change(tmp_path, capsys, core9_plain_rule, lr1, lr2, chosen):
    """lr1 and lr2 count as one change, the learning rate: both pass here, the one with the lower mean is the learning
    rate's, and since no other change passes there is no combination. The winner is the passing candidate with the lowest
    score, and its third run is asked for with its learning rate."""
    results, records9 = core9_layout(tmp_path)
    scores = {**core9_BASE, "s9_swap_1": 0.2990, "s9_swap_2": 0.3000, "s9_long_1": 0.2980, "s9_long_2": 0.3010,
              "s9_lr1_1": lr1[0], "s9_lr1_2": lr1[1], "s9_lr2_1": lr2[0], "s9_lr2_2": lr2[1]}
    for name, s in scores.items():
        core9_write_run(results, name, s)
    env = core9_next(results, records9, capsys)
    assert (env["EDDM_ACTION"], env["EDDM_NAME"], env["EDDM_SEED"], env["EDDM_RECIPE"]) == ("train", f"s9_{chosen}_3", "20260929", chosen)
    assert float(env["EDDM_LR_ADAPTERS"]) == {"lr1": 1e-4, "lr2": 2e-4}[chosen]
    sel = core9_selection(results)
    assert [sel["candidates"][c]["passes"] for c in ("swap", "long", "lr1", "lr2")] == [False, False, True, True]
    assert sel["learning_rate"]["passing"] == ["lr1", "lr2"] and sel["learning_rate"]["chosen"] == chosen
    assert sel["passing_changes"] == [chosen]
    assert sel["combination"]["called"] is False and "settings" not in sel["combination"]
    assert "combination_called" not in sel["decisions"] and not (records9 / "combo").exists()
    assert sel["winner"]["recipe"] == chosen


core9_COMBO_SCENARIOS = {
    # swap, long, lr1 and lr2 all pass; lr2 has the lower mean, so the changes are swap, long and lr2, and lr2 is the best
    # passing candidate (0.289)
    "swap_long_lr": {**core9_BASE, "s9_swap_1": 0.2900, "s9_swap_2": 0.2910, "s9_long_1": 0.2930, "s9_long_2": 0.2950,
                     "s9_lr1_1": 0.2930, "s9_lr1_2": 0.2940, "s9_lr2_1": 0.2880, "s9_lr2_2": 0.2900},
    # long and lr1 pass, swap and lr2 fail: two changes, without swap, so no records are built for the combination; lr1
    # is the best passing candidate (0.291)
    "long_lr1": {**core9_BASE, "s9_swap_1": 0.2990, "s9_swap_2": 0.3000, "s9_long_1": 0.2930, "s9_long_2": 0.2950,
                 "s9_lr1_1": 0.2900, "s9_lr1_2": 0.2920, "s9_lr2_1": 0.2990, "s9_lr2_2": 0.3010},
    # swap and lr1 pass, long and lr2 fail: two changes at 4,096 tokens; swap is the best passing candidate (0.285)
    "swap_lr1": {**core9_BASE, "s9_swap_1": 0.2840, "s9_swap_2": 0.2860, "s9_long_1": 0.2990, "s9_long_2": 0.3000,
                 "s9_lr1_1": 0.2900, "s9_lr1_2": 0.2920, "s9_lr2_1": 0.2990, "s9_lr2_2": 0.3010},
}
core9_COMBO_PARTS = {"swap_long_lr": ("lr2", ["swap", "long", "lr2"]), "long_lr1": ("lr1", ["long", "lr1"]),
                     "swap_lr1": ("swap", ["swap", "lr1"])}  # (the best passing candidate, the passing changes)
core9_COMBO_SETTINGS = {  # THE RECIPES: every passing change at once
    "swap_long_lr": {"recipe": "combo", "parts": ["long", "lr2", "swap"], "lr_choice": "lr2",
                     "train_records": "data/processed/session9/combo/evidence_effect_train.jsonl",
                     "monitor_records": "data/processed/session9/long/evidence_effect_validation_monitor.jsonl",
                     "train_items": "combo_train", "monitor_items": "long_monitor", "epochs": 1, "lr_adapters": 2e-4,
                     "max_len": 6144, "memory_setting": "long", "validation_records_dir": "data/processed/session9/long",
                     "exchanged_validation_dir": "data/processed/session9/checks/swap_long"},
    "long_lr1": {"recipe": "combo", "parts": ["long", "lr1"], "lr_choice": "lr1",
                 "train_records": "data/processed/session9/long/evidence_effect_train.jsonl",
                 "monitor_records": "data/processed/session9/long/evidence_effect_validation_monitor.jsonl",
                 "train_items": "long_train", "monitor_items": "long_monitor", "epochs": 2, "lr_adapters": 1e-4,
                 "max_len": 6144, "memory_setting": "long", "validation_records_dir": "data/processed/session9/long",
                 "exchanged_validation_dir": "data/processed/session9/checks/swap_long"},
    "swap_lr1": {"recipe": "combo", "parts": ["lr1", "swap"], "lr_choice": "lr1",
                 "train_records": "data/processed/session9/combo/evidence_effect_train.jsonl",
                 "monitor_records": "data/processed/session9/clean/evidence_effect_validation_monitor.jsonl",
                 "train_items": "combo_train", "monitor_items": "clean_monitor", "epochs": 1, "lr_adapters": 1e-4,
                 "max_len": 4096, "memory_setting": "base", "validation_records_dir": "data/processed/session8",
                 "exchanged_validation_dir": "data/processed/session9/checks/swap"},
}


@core9_pytest.mark.parametrize("scenario, combo, passes", [
    ("swap_long_lr", (0.2800, 0.2820), True),   # mean 0.281: below lr2's 0.289 by more than m, each run below lr2's
    ("swap_long_lr", (0.2845, 0.2850), False),  # mean 0.28475: not below 0.289 by more than m
    ("swap_long_lr", (0.2700, 0.2905), False),  # mean 0.28025, but its second run is above lr2's second, 0.2900
    ("long_lr1", (0.2800, 0.2820), True),       # mean 0.281: below lr1's 0.291 by more than m, each run below lr1's
    ("swap_lr1", (0.2820, 0.2830), False),      # mean 0.2825: not below swap's 0.285 by more than m, so swap wins
])
def test_core9_rule_combination(tmp_path, capsys, core9_plain_rule, scenario, combo, passes):
    """When two or more of the three changes pass (lr1 and lr2 counting as one), the combination of every passing change
    is trained twice (seeds 20260927 and 20260928), after its records when swap is a part. It passes if its mean is below
    the best passing candidate's by more than m and each run is below that candidate's run with the same seed (correction
    2). The winner is the combination if it passed, otherwise the passing candidate with the lowest score; its third run
    (seed 20260929) is asked for."""
    results, records9 = core9_layout(tmp_path)
    cand = core9_COMBO_SCENARIOS[scenario]
    for name, s in cand.items():
        core9_write_run(results, name, s)
    best, changes = core9_COMBO_PARTS[scenario]
    settings = core9_COMBO_SETTINGS[scenario]
    env = core9_next(results, records9, capsys)
    if "swap" in changes:
        records_parts = [c for c in changes if c in ("swap", "long")]
        assert env == {"EDDM_ACTION": "combo_records", "EDDM_PARTS": " ".join(records_parts)}
        assert core9_selection(results)["combination"]["called"] is True
        core9_write_json(records9 / "combo" / "combo.json", {"parts": sorted(records_parts)})  # as the pod's combo step writes it
        env = core9_next(results, records9, capsys)
    assert env == {"EDDM_ACTION": "train", "EDDM_NAME": "s9_combo_1", "EDDM_SEED": "20260927",
                   **{f"EDDM_{k.upper()}": " ".join(v) if isinstance(v, list) else str(v) for k, v in settings.items()}}
    core9_write_run(results, "s9_combo_1", combo[0])
    env = core9_next(results, records9, capsys)
    assert (env["EDDM_ACTION"], env["EDDM_NAME"], env["EDDM_SEED"]) == ("train", "s9_combo_2", "20260928")
    core9_write_run(results, "s9_combo_2", combo[1])
    env = core9_next(results, records9, capsys)
    sel = core9_selection(results)
    if scenario == "swap_long_lr":
        assert sel["learning_rate"]["passing"] == ["lr1", "lr2"] and sel["learning_rate"]["chosen"] == "lr2"
    best_score = (cand[f"s9_{best}_1"] + cand[f"s9_{best}_2"]) / 2
    c = sel["combination"]
    assert sel["passing_changes"] == changes and c["called"] is True and c["parts"] == changes
    assert c["settings"] == settings and sel["decisions"]["combination_called"]["value"] == changes
    assert c["best_candidate"] == best and c["best_score"] == core9_approx(best_score, abs=1e-12)
    assert c["score"] == core9_approx(sum(combo) / 2, abs=1e-12)
    assert c["below_best_by_more_than_m"] is (sum(combo) / 2 < best_score - 0.005)
    assert c["each_run_below_its_seed"] == [combo[0] < cand[f"s9_{best}_1"], combo[1] < cand[f"s9_{best}_2"]]
    assert c["passes"] is passes and sel["decisions"]["combination_passes"]["value"] is passes
    winner = "combo" if passes else best
    assert sel["winner"]["recipe"] == winner
    assert (env["EDDM_ACTION"], env["EDDM_NAME"], env["EDDM_SEED"]) == ("train", f"s9_{winner}_3", "20260929")
    if passes:
        assert env == {"EDDM_ACTION": "train", "EDDM_NAME": "s9_combo_3", "EDDM_SEED": "20260929",
                       **{f"EDDM_{k.upper()}": " ".join(v) if isinstance(v, list) else str(v) for k, v in settings.items()}}
    else:
        assert env["EDDM_RECIPE"] == best and float(env["EDDM_LR_ADAPTERS"]) == {"lr1": 1e-4, "lr2": 2e-4, "swap": 5e-5}[best]


def test_core9_rule_base_when_none_passes(tmp_path, capsys, core9_plain_rule):
    """No candidate passes, so base wins: no third run is asked for, s9_ei is the base runs' average (s9_b), and the weight
    average is asked for from s9_b_1 to s9_b_3 at 4,096 tokens."""
    results, records9 = core9_layout(tmp_path)
    for name, s in {**core9_BASE, **core9_FAILING, "s9_swap_1": 0.2990, "s9_swap_2": 0.3000}.items():
        core9_write_run(results, name, s)
    env = core9_next(results, records9, capsys)
    assert env == {"EDDM_ACTION": "wavg", "EDDM_NAME": "s9_ei_wavg", "EDDM_SOURCE_RUNS": "s9_b_1 s9_b_2 s9_b_3",
                   "EDDM_MAX_LEN": "4096", "EDDM_VALIDATION_RECORDS_DIR": "data/processed/session8"}
    sel = core9_selection(results)
    assert not any(c["passes"] for c in sel["candidates"].values())
    assert sel["passing_changes"] == [] and sel["combination"]["called"] is False
    assert sel["winner"]["recipe"] == "b" and sel["winner"]["runs"] == ["s9_b_1", "s9_b_2", "s9_b_3"]
    assert sel["s9_ei"]["runs"] == ["s9_b_1", "s9_b_2", "s9_b_3"]
    assert sel["s9_ei"]["score"] == core9_approx(core9_avg_score(*core9_BASE.values()), abs=1e-12)


@core9_pytest.mark.parametrize("long_1, passes", [(0.2945, True), (0.2955, False)])
def test_core9_rule_candidate_left_with_one_run(tmp_path, capsys, core9_plain_rule, long_1, passes):
    """Correction 2: when Johann's decision under the cost rule drops lr2 and then long's second run (cost_rule.json),
    neither is asked for, and long, left with one run, is judged by the rule as first written: its run below s9_b_1's
    score by more than m (0.300 - 0.005 = 0.295). 0.2955 is below s9_b_1 but not by more than m. A passing long wins and its
    third run reads 6,144 tokens."""
    results, records9 = core9_layout(tmp_path)
    core9_write_json(results / "cost_rule.json", {"stops": True, "dropped": ["lr2", "long_2"],
                                                  "johann_decision": "made up: drop lr2, then long's second run"})
    scores = {**core9_BASE, "s9_swap_1": 0.2990, "s9_swap_2": 0.3000, "s9_long_1": long_1, "s9_lr1_1": 0.2990,
              "s9_lr1_2": 0.3010}
    trained, env = core9_drive(results, records9, capsys, scores)
    assert [r for r, _ in trained] == ["s9_b_1", "s9_b_2", "s9_b_3", "s9_swap_1", "s9_long_1", "s9_lr1_1", "s9_swap_2", "s9_lr1_2"]
    sel = core9_selection(results)
    assert sel["dropped"] == ["long_2", "lr2"] and sel["candidates"]["lr2"] == {"status": "dropped", "passes": False}
    long = sel["candidates"]["long"]
    assert long["runs"] == ["s9_long_1"] and long["score"] == core9_approx(long_1, abs=1e-12) and "one run" in long["rule"]
    assert long["passes"] is passes and sel["decisions"]["candidate_long"]["value"] is passes
    if passes:
        assert sel["winner"]["recipe"] == "long"
        assert {k: env[k] for k in ("EDDM_ACTION", "EDDM_NAME", "EDDM_SEED", "EDDM_MAX_LEN", "EDDM_MEMORY_SETTING",
                                    "EDDM_TRAIN_ITEMS", "EDDM_VALIDATION_RECORDS_DIR")} == {
            "EDDM_ACTION": "train", "EDDM_NAME": "s9_long_3", "EDDM_SEED": "20260929", "EDDM_MAX_LEN": "6144",
            "EDDM_MEMORY_SETTING": "long", "EDDM_TRAIN_ITEMS": "long_train",
            "EDDM_VALIDATION_RECORDS_DIR": "data/processed/session9/long"}
    else:
        assert sel["winner"]["recipe"] == "b" and env["EDDM_ACTION"] == "wavg"


@core9_pytest.mark.parametrize("winner", ["swap", "long"])
def test_core9_rule_weight_average_action_with_three_and_two_runs(tmp_path, capsys, core9_plain_rule, winner):
    """I-39's weight average is asked for from the winning recipe's runs: swap's three; or long's first and third, when the
    cost rule dropped its second (correction 7), at 6,144 tokens. s9_ei is the probability average of the same runs. Once
    s9_ei_wavg's validation predictions exist (here as good as s9_ei's, so it is used), it predicts the exchanged
    validation records at its recipe's length (correction 3)."""
    results, records9 = core9_layout(tmp_path)
    if winner == "swap":
        scores, runs = dict(core9_SWAP_WINS), ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    else:
        core9_write_json(results / "cost_rule.json", {"dropped": ["lr2", "long_2"], "johann_decision": "made up"})
        scores = {**core9_BASE, "s9_swap_1": 0.2990, "s9_swap_2": 0.3000, "s9_long_1": 0.2900, "s9_long_3": 0.2880,
                  "s9_lr1_1": 0.2990, "s9_lr1_2": 0.3010}
        runs = ["s9_long_1", "s9_long_3"]
    trained, env = core9_drive(results, records9, capsys, scores)
    assert trained[-1] == (runs[-1], 20260929)
    length, records_dir = ("4096", "data/processed/session8") if winner == "swap" else ("6144", "data/processed/session9/long")
    assert env == {"EDDM_ACTION": "wavg", "EDDM_NAME": "s9_ei_wavg", "EDDM_SOURCE_RUNS": " ".join(runs),
                   "EDDM_MAX_LEN": length, "EDDM_VALIDATION_RECORDS_DIR": records_dir}
    sel = core9_selection(results)
    s9_ei = core9_avg_score(*[scores[r] for r in runs])
    assert sel["winner"]["recipe"] == winner and sel["winner"]["runs"] == runs and sel["s9_ei"]["runs"] == runs
    assert sel["s9_ei"]["score"] == core9_approx(s9_ei, abs=1e-12)
    core9_write_run(results, "s9_ei_wavg", s9_ei)
    env = core9_next(results, records9, capsys)
    folder = "swap" if winner == "swap" else "swap_long"
    assert env == {"EDDM_ACTION": "exch", "EDDM_NAME": "s9_ei_wavg_exch", "EDDM_PHASE_SYSTEM": "s9_ei_wavg",
                   "EDDM_RECORDS_DIR": f"data/processed/session9/checks/{folder}", "EDDM_ROWS": str(len(core9_EXCH))}
    assert core9_selection(results)["weight_average"]["uses_wavg"] is True


@core9_pytest.mark.parametrize("delta, uses", [(-0.0020, True), (0.0040, True), (0.0060, False)])
def test_core9_rule_weight_average_decision(tmp_path, core9_plain_rule, delta, uses):
    """I-39: v0.5 uses s9_ei_wavg if its score is not above s9_ei's by more than m (0.005). Below s9_ei's, or above it by
    0.004, it is used; above it by 0.006, it is not. The system chosen then predicts the exchanged validation records:
    s9_ei_wavg itself, or the first of s9_ei's runs (correction 3)."""
    results, records9 = core9_layout(tmp_path)
    for name, s in core9_SWAP_WINS.items():
        core9_write_run(results, name, s)
    s9_ei = core9_avg_score(*core9_SWAP_RUNS.values())
    core9_write_run(results, "s9_ei_wavg", s9_ei + delta)
    action, sel = core9_R9.Decider(results, records9).state()
    wa = sel["weight_average"]
    assert wa["score"] == core9_approx(s9_ei + delta, abs=1e-12) and wa["s9_ei_score"] == core9_approx(s9_ei, abs=1e-12)
    assert wa["m"] == core9_approx(0.005, abs=1e-12) and wa["built"] is True
    assert wa["uses_wavg"] is uses and sel["decisions"]["uses_weight_average"]["value"] is uses
    first = "s9_ei_wavg" if uses else "s9_swap_1"
    assert action == {"action": "exch", "name": f"{first}_exch", "phase_system": first,
                      "records_dir": "data/processed/session9/checks/swap", "rows": len(core9_EXCH)}
    assert sel == core9_selection(results)


@core9_pytest.mark.parametrize("gain, kept", [(0.0055, True), (0.0045, False), (0.0, False)])
@core9_pytest.mark.parametrize("uses_wavg", [True, False])
def test_core9_rule_swap_averaging(tmp_path, capsys, core9_plain_rule, uses_wavg, gain, kept):
    """Correction 3: the system v0.5 would use (s9_ei_wavg, or s9_ei when the weight average is not used) predicts the
    exchanged validation records (s9_ei_wavg itself, or each of s9_ei's runs in turn). Its swap-averaged answer to a
    prompt is the mean of its answer and the exchange of its answer to the exchanged prompt; a prompt with no exchanged
    record keeps its answer. Swap averaging is kept if its score is below the plain one by more than m. The exchanged
    answers are planted so that the swap-averaged score is below the plain one by `gain`."""
    results, records9 = core9_layout(tmp_path)
    z = {name: core9_write_run(results, name, s) for name, s in core9_SWAP_WINS.items()}
    runs = ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    s9_ei = core9_avg_score(*core9_SWAP_RUNS.values())
    z["s9_ei_wavg"] = core9_write_run(results, "s9_ei_wavg", s9_ei + (-0.001 if uses_wavg else 0.006))
    x, x_score, singles = ("s9_ei_wavg", s9_ei - 0.001, ["s9_ei_wavg"]) if uses_wavg else ("s9_ei", s9_ei, runs)
    q = core9_math.exp(-x_score)
    n_exch = sum(1 for i in core9_CLEAN if i not in core9_SAME_TEXT)  # 650 clean prompts have an exchanged record
    r = q * (2.0 * core9_math.exp(core9_N_CLEAN * gain / n_exch) - 1.0)
    asked, e = [], []
    env = core9_next(results, records9, capsys)
    while env["EDDM_ACTION"] == "exch":
        asked.append((env["EDDM_NAME"], env["EDDM_PHASE_SYSTEM"], env["EDDM_RECORDS_DIR"], int(env["EDDM_ROWS"])))
        e.append(core9_softmax(core9_write_exch(results, env["EDDM_NAME"], r)))
        env = core9_next(results, records9, capsys)
    assert asked == [(f"{s}_exch", s, "data/processed/session9/checks/swap", len(core9_EXCH)) for s in singles]
    assert env == {"EDDM_ACTION": "done"}
    p_x = core9_np.mean([core9_softmax(z[s]) for s in singles], axis=0)
    p_sa = p_x.copy()
    p_sa[core9_EXCH_ROWS] = 0.5 * (p_x[core9_EXCH_ROWS] + core9_np.mean(e, axis=0)[:, core9_SWAP])
    clean_rows = core9_np.flatnonzero(core9_IS_CLEAN)
    sa_score = float(core9_np.mean(-core9_np.log(p_sa[clean_rows, core9_LABELS[clean_rows]])))
    assert sa_score == core9_approx(x_score - gain, abs=1e-12)
    sel = core9_selection(results)
    sa = sel["swap_averaging"]
    assert sa["system"] == x and sa["plain_score"] == core9_approx(x_score, abs=1e-12)
    assert sa["swap_averaged_score"] == core9_approx(sa_score, abs=1e-12) and sa["m"] == core9_approx(0.005, abs=1e-12)
    assert sa["kept"] is kept and sel["decisions"]["swap_averaging"]["value"] is kept
    assert (sa["exchanged_records"], sa["exchanged_prompts"]) == ("data/processed/session9/checks/swap", len(core9_EXCH))
    ids, labels, written = core9_read_probs(results / "predictions" / f"{x}_sa-raw_{core9_EI}_validation.csv")
    assert ids == core9_IDS and (labels == core9_LABELS).all()
    core9_np.testing.assert_allclose(written, p_sa, rtol=0, atol=1e-8)
    v = sel["v05_ei"]
    assert (v["system"], v["base_system"], v["other_form"]) == ((f"{x}_sa", x, x) if kept else (x, x, f"{x}_sa"))
    assert v["swap_averaging"] is kept and v["runs"] == runs and v["single_pass_systems"] == singles
    assert (v["recipe"], v["max_len"], v["checks_swap_dir"]) == ("swap", 4096, "data/processed/session9/checks/swap")


# ---------------------------------------------------------------- the cost rule (recipe_session9.py cost)

core9_NOW = core9_dt.datetime(2026, 10, 1, 15, 0, 0, tzinfo=core9_dt.timezone.utc)
core9_DROP_ORDER = ["lr2", "long_2", "combo_2", "lr1_2", "swap_2", "long", "combo", "swap"]  # correction 2's order


def core9_cost_inputs(tmp_path, created, maintenance=None, long_chosen=True, hourly=3.52):
    """pod_rate.txt as Claude writes it in step 9 (created_utc, hourly_usd and an optional maintenance_utc), and made-up
    pilot files: session 8's pilot.json (2.86 training items a second) and pilot_long.json (2.0, or no setting passed)."""
    lines = [f"created_utc {core9_utc(created)}", f"hourly_usd {hourly}"]
    if maintenance is not None:
        lines.append(f"maintenance_utc {maintenance}")
    rate = core9_write_text(tmp_path / "pod" / "pod_rate.txt", "\n".join(lines) + "\n")
    pilot8 = core9_write_json(tmp_path / "results" / "fiorillo_v0_4" / "pilot.json",
                              {"chosen": {"setting": "a", "train_items_per_second": 2.86}})
    pilot_long = core9_write_json(tmp_path / "results" / "fiorillo_v0_5" / "pilot_long.json",
                                  {"chosen": {"setting": "b", "train_items_per_second": 2.0}} if long_chosen else
                                  {"chosen": None, "not_built": "NOT BUILT: long, made-up reason"})
    return rate, pilot8, pilot_long, tmp_path / "results" / "fiorillo_v0_5"


def core9_cost_command(rate, pilot8, pilot_long, results, capsys):
    """recipe_session9.py cost as the train stage runs it: (exit status, cost_rule.json)."""
    capsys.readouterr()
    status = core9_R9.main(["cost", "--pod-rate", str(rate), "--pilot-s8", str(pilot8), "--pilot-long", str(pilot_long),
                            "--results", str(results)])
    capsys.readouterr()
    return status, core9_read_json(core9_Path(results) / "cost_rule.json")


@core9_pytest.mark.parametrize("long_chosen", [True, False])
def test_core9_cost_rule_written(tmp_path, long_chosen):
    """cost_rule.json: the spend so far (hours since the pod was created times the hourly rate) plus the longest path still
    possible, with correction 2's second runs (three base runs, two runs of each candidate and of the combination, the
    winner's third run), the fixed items and, when pilot_long chose a setting, Gemma on the longer input. A clean run is
    10,558 items at session 8's 2.70 items a second (the pilot's rate times 2.70 / 2.86) plus its 4-minute validation
    pass. The stop line is $115 under the $125 cap (correction 1), and the options follow correction 2's order."""
    rate, pilot8, pilot_long, results = core9_cost_inputs(tmp_path, core9_NOW - core9_dt.timedelta(hours=2),
                                                          long_chosen=long_chosen)
    out = core9_R9.cost(rate, pilot8, pilot_long, results, now_utc=core9_NOW)
    saved = core9_read_json(results / "cost_rule.json")
    assert saved == core9_json.loads(core9_json.dumps(out))
    assert (saved["stop_line_usd"], saved["cap_usd"]) == (115.0, 125.0)
    assert saved["pod"]["hours_so_far"] == core9_approx(2.0) and saved["pod"]["spend_so_far_usd"] == core9_approx(7.04)
    cands = ["swap", "long", "lr1", "lr2"] if long_chosen else ["swap", "lr1", "lr2"]
    path = saved["longest_path_hours"]
    assert list(path) == (["s9_b_1", "s9_b_2", "s9_b_3"] + [f"s9_{c}_1" for c in cands] + [f"s9_{c}_2" for c in cands]
                          + ["s9_combo_1", "s9_combo_2", "the winning recipe's third run"])
    clean_run = 10558 / 2.70 / 3600 + 4 / 60
    assert path["s9_b_1"] == core9_approx(clean_run, rel=1e-12) and path["s9_lr2_2"] == core9_approx(clean_run, rel=1e-12)
    gemma = "Gemma 4 31B-it on the longer input"
    if long_chosen:
        long_run = 10558 / (2.0 * 2.70 / 2.86) / 3600 + 6 / 60
        assert path["s9_long_2"] == core9_approx(long_run, rel=1e-12)
        assert path["the winning recipe's third run"] == core9_approx(long_run, rel=1e-12)
        assert saved["fixed_hours"][gemma] == 2.5
    else:
        assert path["the winning recipe's third run"] == core9_approx(clean_run, rel=1e-12) and gemma not in saved["fixed_hours"]
    remaining = sum(path.values()) + sum(saved["fixed_hours"].values())
    assert saved["remaining_hours"] == core9_approx(remaining, rel=1e-12)
    assert saved["projected_total_usd"] == core9_approx((2.0 + remaining) * 3.52, rel=1e-12)
    assert saved["stops"] is (saved["projected_total_usd"] > 115.0) and saved["maintenance_before_the_end"] is False
    assert saved["projected_copy_checked_utc"] == core9_utc(core9_NOW + core9_dt.timedelta(hours=remaining))
    options = saved["options_in_order"]
    assert [o["drop"] for o in options] == core9_DROP_ORDER
    assert options[0]["runs_removed"] == ["s9_lr2_1", "s9_lr2_2"]
    totals = [saved["projected_total_usd"]] + [o["projected_total_usd"] for o in options]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(totals, totals[1:]))
    assert saved["dropped"] == []


def test_core9_cost_stop_at_115_under_the_125_cap(tmp_path, capsys):
    """Correction 1: the stop line is $115 under the session's $125 cap (the plan as first written stopped above $90). A
    projection just under $115 does not stop and one just over does; the command exits 0 at $100 (which the first stop
    line would have stopped), and 4 at $120 (under the cap) and at $130."""
    rate, pilot8, pilot_long, results = core9_cost_inputs(tmp_path, core9_NOW)
    remaining = core9_R9.cost(rate, pilot8, pilot_long, results, now_utc=core9_NOW)["remaining_hours"]
    for total, stops in ((114.9, False), (115.1, True)):
        created = core9_NOW - core9_dt.timedelta(seconds=round((total / 3.52 - remaining) * 3600))
        core9_cost_inputs(tmp_path, created)
        out = core9_R9.cost(rate, pilot8, pilot_long, results, now_utc=core9_NOW)
        assert out["projected_total_usd"] == core9_approx(total, abs=0.001) and out["stops"] is stops
    for total, status in ((100.0, 0), (120.0, 4), (130.0, 4)):
        created = core9_dt.datetime.now(core9_dt.timezone.utc) - core9_dt.timedelta(hours=total / 3.52 - remaining)
        core9_cost_inputs(tmp_path, created)
        got, saved = core9_cost_command(rate, pilot8, pilot_long, results, capsys)
        assert got == status and saved["stops"] is (status == 4) and saved["maintenance_before_the_end"] is False
        assert saved["projected_total_usd"] == core9_approx(total, abs=0.02)
        assert (saved["stop_line_usd"], saved["cap_usd"]) == (115.0, 125.0)


@core9_pytest.mark.parametrize("maintenance_hours, status", [(1.0, 4), (72.0, 0), (None, 0)])
def test_core9_cost_maintenance_before_the_end(tmp_path, capsys, maintenance_hours, status):
    """The schedule: if the projected end of the copy is after the start of a maintenance announced for the pod's server,
    the command exits 4 although the projection (about $88 here) is under the stop line; a maintenance after the end, or
    none, lets it pass."""
    now = core9_dt.datetime.now(core9_dt.timezone.utc)
    maintenance = "none" if maintenance_hours is None else core9_utc(now + core9_dt.timedelta(hours=maintenance_hours))
    rate, pilot8, pilot_long, results = core9_cost_inputs(tmp_path, now - core9_dt.timedelta(hours=1), maintenance)
    got, saved = core9_cost_command(rate, pilot8, pilot_long, results, capsys)
    assert got == status and saved["stops"] is False and saved["projected_total_usd"] < 115.0
    assert saved["maintenance_before_the_end"] is (status == 4) and saved["pod"]["maintenance_utc"] == maintenance


def test_core9_cost_decision_recorded_then_next_continues(tmp_path, capsys, core9_plain_rule):
    """After the cost rule's stop (exit 4), Claude records Johann's decision in cost_rule.json: its "dropped" list (here
    the first option, lr2) and the decision. The module's next then continues through every run not dropped and asks for
    none of lr2's. (The cost command does not read the decision; the train stage skips it once cost_rule.json exists.)"""
    results, records9 = core9_layout(tmp_path)
    rate, pilot8, pilot_long, _ = core9_cost_inputs(tmp_path, core9_dt.datetime.now(core9_dt.timezone.utc)
                                                    - core9_dt.timedelta(hours=12))
    status, saved = core9_cost_command(rate, pilot8, pilot_long, results, capsys)
    assert status == 4 and saved["stops"] is True and saved["options_in_order"][0]["drop"] == "lr2"
    saved.update({"johann_decision": "made up: drop lr2 (both runs)", "dropped": ["lr2"],
                  "dropped_utc": core9_utc(core9_dt.datetime.now(core9_dt.timezone.utc))})
    core9_write_json(results / "cost_rule.json", saved)
    scores = {k: v for k, v in core9_SWAP_WINS.items() if not k.startswith("s9_lr2_")}
    trained, env = core9_drive(results, records9, capsys, scores)
    assert [r for r, _ in trained] == ["s9_b_1", "s9_b_2", "s9_b_3", "s9_swap_1", "s9_long_1", "s9_lr1_1", "s9_swap_2",
                                       "s9_long_2", "s9_lr1_2", "s9_swap_3"]
    assert env["EDDM_ACTION"] == "wavg" and env["EDDM_SOURCE_RUNS"] == "s9_swap_1 s9_swap_2 s9_swap_3"
    sel = core9_selection(results)
    assert sel["dropped"] == ["lr2"] and sel["candidates"]["lr2"] == {"status": "dropped", "passes": False}
    assert sel["winner"]["recipe"] == "swap"


# ---------------------------------------------------------------- the weight average (weight_average_session9.py)

def core9_adapter_state(seed, r=16):
    """A made-up phase's trainable state, named as train_qwen_head.py saves it (float32): LoRA A (r by in) and B (out by
    r) for two targets in each of two decoder layers, and the projection's and the head's weights."""
    g = core9_torch.Generator().manual_seed(seed)
    state = {}
    for layer in (0, 1):
        for target, d_in, d_out in (("self_attn.q_proj", 12, 10), ("mlp.down_proj", 20, 12)):
            name = f"encoder.base.layers.{layer}.{target}"
            state[f"{name}.lora_A.weight"] = core9_torch.randn(r, d_in, generator=g) * 0.5
            state[f"{name}.lora_B.weight"] = core9_torch.randn(d_out, r, generator=g) * 0.5
    for name, shape in (("encoder.proj.weight", (8, 12)), ("encoder.proj.bias", (8,)), ("head.layers.0.linear1.weight", (32, 8)),
                        ("scorer.3.weight", (1, 8)), ("type_emb.weight", (3, 8))):
        state[name] = core9_torch.randn(*shape, generator=g)
    return state


def core9_adapter_run(folder, k, **config_changes):
    """A made-up run folder in train_qwen_head.py's phase layout: adapter.safetensors, config.json (rank 16, alpha 32, as
    every session 9 run) and a tokenizer file."""
    folder = core9_Path(folder)
    folder.mkdir(parents=True)
    state = core9_adapter_state(100 + k)
    core9_save_file(state, str(folder / "adapter.safetensors"))
    config = {"arm": "B", "plan": "docs/prompt_session7.md, ARM B", "base_dir": "/root/eddm_s9/models/qwen3_4b_base",
              "base_revision": "made-up-revision", "base_config_sha256": "made-up",
              "adapters": {"targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           "r": 16, "alpha": 32, "dropout": 0.05},
              "projection": 1024, "head_layers": 2, "n_act": 2, "max_len": 4096, "item_format": "made-up item format",
              "project_commit": "made-up", "seed": 20260926 + k, "system": folder.name, "written_utc": f"2026-10-01T0{k}:00:00Z"}
    config.update(config_changes)
    core9_write_json(folder / "config.json", config)
    core9_write_text(folder / "tokenizer.json", core9_json.dumps({"made-up tokenizer": k}))
    return state


@core9_pytest.mark.parametrize("ks, rank, alpha", [((1, 2, 3), 48, 96), ((1, 3), 32, 64)])
def test_core9_weight_average_joined_update_is_the_mean(tmp_path, ks, rank, alpha, capsys):
    """weight_average_session9.py build: from three runs (rank 48, alpha 96) or, under correction 7, two (rank 32, alpha
    64), the joined adapter's update (alpha / rank times B A, from the saved file) equals the mean of the runs' updates
    within 1e-5 on every adapted layer; every other weight is the elementwise mean. config.json (the joined rank and
    alpha) and weight_average.json (the check, the sources' SHA-256) are written."""
    runs = [tmp_path / "models" / f"s9_swap_{k}" for k in ks]
    states = [core9_adapter_run(p, k) for p, k in zip(runs, ks)]
    out, results = tmp_path / "models" / "s9_ei_wavg", tmp_path / "results" / "fiorillo_v0_5"
    assert core9_W9.main(["build", "--runs", *map(str, runs), "--out", str(out), "--results", str(results)]) == 0
    joined = core9_load_file(str(out / "adapter.safetensors"))
    config = core9_read_json(out / "config.json")
    assert (config["adapters"]["r"], config["adapters"]["alpha"]) == (rank, alpha) and set(joined) == set(states[0])
    scale = config["adapters"]["alpha"] / config["adapters"]["r"]
    adapted = sorted(k for k in joined if ".lora_A." in k)
    assert len(adapted) == 4
    for a in adapted:
        b = a.replace(".lora_A.", ".lora_B.")
        assert joined[a].shape[0] == rank and joined[b].shape[1] == rank
        got = scale * (joined[b].double() @ joined[a].double())
        want = sum(2.0 * (s[b].double() @ s[a].double()) for s in states) / len(states)  # each run: alpha 32 / rank 16
        assert float((got - want).abs().max()) <= 1e-5, a
        assert float(want.abs().max()) > 1.0
    for key in joined:
        if ".lora_" not in key:
            want = sum(s[key] for s in states) / len(states)
            assert core9_torch.allclose(joined[key], want, rtol=0, atol=1e-6), key
    first = core9_read_json(runs[0] / "config.json")
    assert config["system"] == "s9_ei_wavg" and config["weight_average"]["sources"] == [str(p) for p in runs]
    same = [k for k in first if k not in ("adapters", "system", "written_utc")]
    assert {k: config[k] for k in same} == {k: first[k] for k in same}
    assert (out / "tokenizer.json").read_text(encoding="utf-8") == (runs[0] / "tokenizer.json").read_text(encoding="utf-8")
    record = core9_read_json(results / "weight_average.json")
    assert (record["rank"], record["alpha"], record["scale"], record["adapted_layers"]) == (rank, alpha, 2.0, 4)
    assert record["passed"] is True and record["tolerance"] == 1e-5 and 0.0 <= record["largest_difference"] <= 1e-5
    assert {k: v["adapter_sha256"] for k, v in record["sources"].items()} == {
        str(p): core9_sha256(p / "adapter.safetensors") for p in runs}
    assert record["adapter_sha256"] == core9_sha256(out / "adapter.safetensors")


def test_core9_weight_average_refuses_one_run(tmp_path):
    """The weight average takes the winner's two or three runs; one run is refused and nothing is written."""
    run = tmp_path / "models" / "s9_swap_1"
    core9_adapter_run(run, 1)
    out, results = tmp_path / "models" / "s9_ei_wavg", tmp_path / "results" / "fiorillo_v0_5"
    with core9_pytest.raises(SystemExit, match="at least two"):
        core9_W9.main(["build", "--runs", str(run), "--out", str(out), "--results", str(results)])
    assert not out.exists() and not (results / "weight_average.json").exists()


@core9_pytest.mark.parametrize("key, value", [
    ("adapters", {"targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], "r": 16,
                  "alpha": 64, "dropout": 0.05}),
    ("projection", 512), ("head_layers", 1), ("n_act", 3), ("max_len", 6144), ("base_revision", "another-revision"),
    ("item_format", "another item format"),
])
def test_core9_weight_average_refuses_runs_whose_configs_differ(tmp_path, key, value):
    """Runs whose config.json differ in what the joined model depends on (the adapters' settings, the projection, the head,
    the input length, the base's revision or the item format) are refused, and nothing is written."""
    runs = [tmp_path / "models" / f"s9_swap_{k}" for k in (1, 2, 3)]
    for k, p in zip((1, 2, 3), runs):
        core9_adapter_run(p, k, **({key: value} if k == 3 else {}))
    out, results = tmp_path / "models" / "s9_ei_wavg", tmp_path / "results" / "fiorillo_v0_5"
    with core9_pytest.raises(SystemExit, match=f"differ in {key}"):
        core9_W9.main(["build", "--runs", *map(str, runs), "--out", str(out), "--results", str(results)])
    assert not out.exists() and not (results / "weight_average.json").exists()


# ---------------------------------------------------------------- calibrate_session9.py: fit and the ECE limit

def core9_assert_params(got, want, what=""):
    """Calibration parameters (a list, or dirichlet_odir's dict) equal within 1e-6."""
    if isinstance(want, dict):
        assert (got["lambda_od"], got["lambda_b"]) == (want["lambda_od"], want["lambda_b"]), what
        for key in ("W", "b"):
            core9_np.testing.assert_allclose(core9_np.asarray(got[key]), core9_np.asarray(want[key]), rtol=0, atol=1e-6,
                                             err_msg=what)
    else:
        core9_np.testing.assert_allclose(core9_np.asarray(got, dtype=float), core9_np.asarray(want, dtype=float), rtol=0,
                                         atol=1e-6, err_msg=what)


def core9_fit_layout(tmp_path):
    """Made-up inputs of calibrate_session9.py fit: the runs' validation logits; s9_ei_wavg's and its answers to the
    exchanged records; recipe_selection.json naming s9_ei_wavg_sa as v0.5's EI specialist (swap won with three runs, the
    weight average was used and swap averaging kept); gemma4_31b_long's raw validation predictions with session 8's
    validation records; BioGPT-Large's on 180 PubMedQA items, 150 marked calibrate; and sessions 7's and 8's calibration
    files."""
    results, records9 = core9_layout(tmp_path)
    names = ["s9_b_1", "s9_b_2", "s9_b_3", "s9_swap_1", "s9_swap_2", "s9_swap_3", "s9_lr1_1", "s9_lr1_2", "s9_ei_wavg"]
    z = {name: core9_write_system(results, name, core9_noisy(100 + k, core9_LABELS, 0.9 + 0.1 * k))
         for k, name in enumerate(names)}
    exch_labels = core9_SWAP[core9_LABELS[core9_EXCH_ROWS]]
    z["s9_ei_wavg_exch"] = core9_write_system(results, "s9_ei_wavg_exch", core9_noisy(120, exch_labels, 1.4),
                                              ids=core9_EXCH, labels=exch_labels)
    swap_runs = ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    core9_write_json(results / "recipe_selection.json", {
        "s9_ei": {"runs": swap_runs, "score": 0.3},
        "scores": {name: 0.3 for name in names + ["s9_b", "s9_ei", "s9_ei_wavg_sa"]},
        "v05_ei": {"system": "s9_ei_wavg_sa", "base_system": "s9_ei_wavg", "swap_averaging": True, "recipe": "swap",
                   "runs": swap_runs, "single_pass_systems": ["s9_ei_wavg"], "max_len": 4096, "other_form": "s9_ei_wavg",
                   "validation_records_dir": "data/processed/session8", "test_records_dir": "data/processed/session8",
                   "checks_swap_dir": "data/processed/session9/checks/swap", "swap_augmented": True}})
    records8 = tmp_path / "data" / "processed" / "session8"
    core9_write_jsonl(records8 / f"{core9_EI}_validation.jsonl",
                      [{"id": i, "label": int(y)} for i, y in zip(core9_IDS, core9_LABELS)])
    core9_write_probs(results / "large_model" / f"gemma4_31b_long-raw_{core9_EI}_validation.csv", core9_IDS, core9_LABELS,
                      core9_softmax(core9_noisy(130, core9_LABELS, 1.1)))
    records_public = tmp_path / "data" / "processed" / "public"
    pqa_ids = [f"pqa-{i:03d}" for i in range(180)]
    pqa_labels = core9_np.random.default_rng(140).integers(0, 3, 180)
    core9_write_jsonl(records_public / f"{core9_PQA}_validation.jsonl",
                      [{"id": i, "label": int(y), "calibrate": k % 6 != 0} for k, (i, y) in enumerate(zip(pqa_ids, pqa_labels))])
    core9_write_probs(results / "predictions" / f"biogpt_large_pqa-raw_{core9_PQA}_validation.csv", pqa_ids, pqa_labels,
                      core9_softmax(core9_noisy(150, pqa_labels, 0.8)))
    results8, results7 = tmp_path / "results" / "fiorillo_v0_4", tmp_path / "results" / "fiorillo_v0_3"
    core9_write_json(results8 / "calibration_s8.json", {"s8_ei": {core9_EI: {"chosen": "temperature"}}})
    core9_write_json(results7 / "calibration_s7.json", {"s7_b3": {core9_PQA: {"chosen": "temperature"}}})
    return {"results": results, "records9": records9, "records8": records8, "records_public": records_public,
            "results8": results8, "results7": results7, "z": z, "pqa_ids": pqa_ids, "pqa_labels": pqa_labels}


def test_core9_calibrate_fit_writes_each_systems_validation_files(tmp_path):
    """fit writes, for every system, its calibrated validation predictions on every row of its file
    (predictions/<system>_<qid>_validation.csv), with the calibration fitted on its calibrate items alone: v0.5's EI
    specialist (s9_ei_wavg_sa) by the lowest of the four families' out-of-fold log losses (5 folds, 20 repeats, the seed)
    on the 654 clean prompts; its other form (s9_ei_wavg) and the unused one of s9_ei and s9_ei_wavg (s9_ei) by the same
    family on their own clean prompts; v0.5's specialist also by session 8's rule (s9_ei_wavg_sa_s8rule); s9_b and each
    run by session 8's rule on the clean prompts; gemma4_31b_long by session 8's rule on all 1,237 prompts;
    biogpt_large_pqa by session 8's rule on PubMedQA's 150 calibrate items."""
    lay = core9_fit_layout(tmp_path)
    results = lay["results"]
    core9_C9.fit(results, lay["records9"], lay["records8"], lay["records_public"], lay["results8"], lay["results7"])
    cal = core9_read_json(results / "calibration_s9.json")
    p = {name: core9_softmax(z) for name, z in lay["z"].items()}
    swap_runs = ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    sa = p["s9_ei_wavg"].copy()
    sa[core9_EXCH_ROWS] = 0.5 * (sa[core9_EXCH_ROWS] + p["s9_ei_wavg_exch"][:, core9_SWAP])
    ei = {"s9_ei_wavg_sa": sa, "s9_ei_wavg_sa_s8rule": sa, "s9_ei_wavg": p["s9_ei_wavg"],
          "s9_ei": core9_np.mean([p[r] for r in swap_runs], axis=0),
          "s9_b": core9_np.mean([p[f"s9_b_{k}"] for k in (1, 2, 3)], axis=0),
          **{r: p[r] for r in ["s9_b_1", "s9_b_2", "s9_b_3", *swap_runs, "s9_lr1_1", "s9_lr1_2"]}}
    meta = {"plan", "v05_ei", "kept_from_earlier_sessions", "families_table_v05", "time_utc"}
    assert set(cal) - meta == set(ei) | {"gemma4_31b_long", "biogpt_large_pqa"}
    assert cal["v05_ei"]["system"] == "s9_ei_wavg_sa" and cal["v05_ei"]["other_form"] == "s9_ei_wavg"
    four = cal["s9_ei_wavg_sa"][core9_EI]
    assert set(four["families"]) == set(core9_FAMILIES) and (four["folds"], four["repeats"], four["seed"]) == (5, 20, 20260927)
    means = [four["families"][f]["out_of_fold_log_loss"]["mean"] for f in core9_FAMILIES]
    family = core9_FAMILIES[next(k for k, v in enumerate(means) if v <= min(means) + 1e-12)]
    assert four["chosen"] == family and cal["families_table_v05"]["chosen"] == family
    every = core9_np.ones(core9_N_VAL, dtype=bool)
    _, _, p_gemma = core9_read_probs(results / "large_model" / f"gemma4_31b_long-raw_{core9_EI}_validation.csv")
    _, _, p_bio = core9_read_probs(results / "predictions" / f"biogpt_large_pqa-raw_{core9_PQA}_validation.csv")
    cases = [(name, core9_EI, core9_IDS, core9_LABELS, core9_np.log(core9_np.maximum(probs, 1e-300)), core9_IS_CLEAN)
             for name, probs in ei.items()]
    cases += [("gemma4_31b_long", core9_EI, core9_IDS, core9_LABELS, core9_np.log(core9_np.maximum(p_gemma, 1e-12)), every),
              ("biogpt_large_pqa", core9_PQA, lay["pqa_ids"], lay["pqa_labels"], core9_np.log(core9_np.maximum(p_bio, 1e-12)),
               core9_np.array([k % 6 != 0 for k in range(180)]))]
    for name, qid, ids, labels, logp, keep in cases:
        entry = cal[name][qid]
        chosen = entry["chosen"]
        assert entry["n_calibrate"] == int(keep.sum()), name
        if name in ("s9_ei_wavg_sa", "s9_ei_wavg", "s9_ei"):
            assert chosen == family, name
            refit = core9_CFC.fit_family(chosen, logp[keep], labels[keep])
        else:
            chosen8, fams = core9_C7.choose(logp[keep], labels[keep])
            assert chosen == chosen8, name
            refit = fams[chosen]["params"]
        params = entry["families"][chosen]["params"]
        core9_assert_params(params, refit, name)
        got_ids, got_labels, got = core9_read_probs(results / "predictions" / f"{name}_{qid}_validation.csv")
        assert got_ids == list(ids) and (got_labels == labels).all(), name
        core9_np.testing.assert_allclose(got, core9_CFC.apply_family(chosen, params, logp), rtol=0, atol=2e-9, err_msg=name)


def test_core9_calibrate_ece_limit_input(tmp_path, monkeypatch):
    """The ECE limit's input (CALIBRATION): ece_noise_floor.py, unchanged, with 5,000 draws and the seed, at the test size
    of 1,218, on v0.5's EI specialist's calibrated validation predictions (the chosen family's file, not its raw or
    temperature-only file, nor another system's) for the 654 clean prompts only; the limit is the larger of 0.05 and the
    95th percentile. The same draws on a file holding only the clean rows give the same numbers."""
    results, records9 = core9_layout(tmp_path)
    v05 = "s9_ei_sa"
    core9_write_json(results / "calibration_s9.json", {"v05_ei": {"system": v05, "base_system": "s9_ei", "swap_averaging": True}})
    probs = core9_softmax(core9_noisy(170, core9_LABELS, 1.0))
    probs[~core9_IS_CLEAN] = [0.98, 0.01, 0.01]  # rows the limit must not read
    pred = core9_write_probs(results / "predictions" / f"{v05}_{core9_EI}_validation.csv", core9_IDS, core9_LABELS, probs)
    for k, name in enumerate((f"{v05}_temperature", "s9_ei", f"{v05}-raw")):  # files the limit must not read
        core9_write_probs(results / "predictions" / f"{name}_{core9_EI}_validation.csv", core9_IDS, core9_LABELS,
                          core9_softmax(core9_noisy(180 + k, core9_LABELS, 3.0)))
    calls = []
    real_run = core9_subprocess.run

    def recording_run(cmd, *args, **kwargs):
        calls.append([str(c) for c in cmd])
        return real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(core9_C9.subprocess, "run", recording_run)
    core9_C9.ece(results, records9)
    assert len(calls) == 1 and core9_Path(calls[0][1]).name == "ece_noise_floor.py"
    out = results / "ece_limit_evidence_effect.json"
    assert dict(zip(calls[0][2::2], calls[0][3::2])) == {
        "--predictions": str(pred), "--records": str(records9 / "clean" / f"{core9_EI}_validation.jsonl"),
        "--n": "1218", "--draws": "5000", "--out": str(out)}
    saved = core9_read_json(out)
    assert (saved["system"], saved["validation_rows"], saved["n"], saved["draws"]) == (v05, 654, 1218, 5000)
    assert saved["limit"] == max(0.05, saved["ece_if_calibrated"]["p95"])
    clean_rows = core9_np.flatnonzero(core9_IS_CLEAN)
    ref_pred = core9_write_probs(tmp_path / "reference" / "clean_rows.csv", [core9_IDS[k] for k in clean_rows],
                                 core9_LABELS[clean_rows], probs[clean_rows])
    ref_records = core9_write_jsonl(tmp_path / "reference" / "records.jsonl", [{"id": core9_IDS[k]} for k in clean_rows])
    ref_out = tmp_path / "reference" / "ece.json"
    real_run([core9_sys.executable, str(core9_SRC / "eval" / "ece_noise_floor.py"), "--predictions", str(ref_pred),
              "--records", str(ref_records), "--n", "1218", "--draws", "5000", "--out", str(ref_out)],
             check=True, capture_output=True)
    assert saved["ece_if_calibrated"] == core9_read_json(ref_out)["ece_if_calibrated"]


# ---------------------------------------------------------------- calibrate_session9.py: the freeze and apply

core9_V05 = "s9_ei_wavg"  # v0.5's EI specialist in the freeze's made-up inputs (swap won at 4,096 tokens, no swap averaging)
core9_OSF = {"link": "https://osf.io/madeup/", "filed_utc": "2026-10-01T12:34:56Z"}
core9_STAGE_OUTPUTS = {
    "trees": ["results/fiorillo_v0_5/trees_v2.json"],
    "tabicl": ["results/fiorillo_v0_5/tabicl.json"],
    "biogpt": ["results/fiorillo_v0_5/biogpt.json", f"results/fiorillo_v0_5/predictions/biogpt_large_pqa-raw_{core9_PQA}_validation.csv"],
    "calibrate": ["results/fiorillo_v0_5/calibration_s9.json", "results/fiorillo_v0_5/ece_limit_evidence_effect.json",
                  "results/fiorillo_v0_5/readmission_pool_v2.json", "results/fiorillo_v0_5/handoff_rule_v05.json",
                  "results/fiorillo_v0_5/handoff_validation.json", "results/fiorillo_v0_5/power_validation.json"]}
core9_FROZEN = ["results/fiorillo_v0_5/recipe_selection.json", "results/fiorillo_v0_5/weight_average.json",
                "results/fiorillo_v0_3/calibration_s7.json", "results/fiorillo_v0_5/calibration_s9.json",
                "results/fiorillo_v0_5/ece_limit_evidence_effect.json",
                "results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json", "results/fiorillo_v0_5/handoff_rule_v05.json",
                "results/fiorillo_v0_5/readmission_pool_v2.json",
                "results/fiorillo_v0_5/predictions/xgboost_codes_readmit_window_validation.csv",
                "results/fiorillo_v0_3/predictions/s7_b3_readmit_window_validation.csv"]


def core9_family(chosen, params, loss=0.3):
    return {"chosen": chosen, "families": {chosen: {"params": params, "out_of_fold_log_loss": {"mean": loss}}}}


def core9_freeze_layout(tmp_path, omit=(), not_built=()):
    """Everything the freeze reads, made up, laid out under tmp_path as in the repository and on the pod:
    results/fiorillo_v0_5, results/fiorillo_v0_4 and results/fiorillo_v0_3, and pod/ for /root/eddm_s9 (stage_done/,
    not_built/ and osf_filed.txt). s8_ei's reading checks are in checks_s8/, with "_test" in their names, as on the pod
    before training. Paths in omit (relative to tmp_path) are not written; a stage in not_built has a NOT BUILT marker
    instead of its done marker and outputs."""
    root = core9_Path(tmp_path)
    skipped = set(omit) | {f for stage in not_built for f in core9_STAGE_OUTPUTS[stage]}

    def put(rel, data):
        if rel in skipped:
            return None
        return core9_write_text(root / rel, data) if isinstance(data, str) else core9_write_json(root / rel, data)

    r5, runs = "results/fiorillo_v0_5", ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    put(f"{r5}/recipe_selection.json", {"s9_ei": {"runs": runs}, "v05_ei": {
        "system": core9_V05, "base_system": core9_V05, "swap_averaging": False, "other_form": f"{core9_V05}_sa",
        "recipe": "swap", "runs": runs, "single_pass_systems": [core9_V05], "max_len": 4096}})
    bio = ({"not_built": "NOT BUILT: biogpt_large_pqa's calibration, its raw validation predictions are missing"}
           if "biogpt" in not_built else {core9_PQA: core9_family("temperature", [0.1], 0.9)})
    put(f"{r5}/calibration_s9.json", {"v05_ei": {"system": core9_V05, "base_system": core9_V05, "swap_averaging": False},
                                      core9_V05: {core9_EI: core9_family("temperature", [0.2])}, "biogpt_large_pqa": bio})
    put("results/fiorillo_v0_3/calibration_s7.json", {"s7_b3": {core9_PQA: core9_family("temperature_intercepts", [0.1, 0.2, -0.3], 0.8)}})
    put("results/fiorillo_v0_4/calibration_s8.json", {"gemma4_31b": {core9_EI: core9_family("temperature", [0.3], 0.5),
                                                                     core9_PQA: core9_family("temperature", [0.4], 0.9)}})
    put(f"{r5}/ece_limit_evidence_effect.json", {"system": core9_V05, "n": 1218, "draws": 5000, "limit": 0.05,
                                                 "ece_if_calibrated": {"median": 0.021, "p05": 0.015, "p95": 0.031}})
    put(f"{r5}/validation_analyses/handoff_rule_s8.json", {"made_up": "handoff_rule_s8"})
    for name in ("handoff_rule_v05.json", "handoff_validation.json", "power_validation.json", "trees_v2.json", "tabicl.json",
                 "biogpt.json", "weight_average.json", "checks_s8.json"):
        put(f"{r5}/{name}", {"made_up": name})
    member_files = {}
    for member, rel in (("xgboost_codes", f"{r5}/predictions/xgboost_codes_readmit_window_validation.csv"),
                        ("s7_b3", "results/fiorillo_v0_3/predictions/s7_b3_readmit_window_validation.csv")):
        path = put(rel, "id,label,p0,p1,p2\nreadmit-1,0,0.7,0.2,0.1\nreadmit-2,2,0.2,0.3,0.5\n")
        member_files[member] = {"file": rel, "sha256": core9_sha256(path)}
    put(f"{r5}/readmission_pool_v2.json", {"pools": {"v2": {"member_files": member_files}}, "chosen": "v2",
                                           "members": list(member_files), "weights": {"xgboost_codes": 0.6, "s7_b3": 0.4},
                                           "T": 1.1, "validation_log_loss": 0.62})
    put(f"{r5}/predictions/biogpt_large_pqa-raw_{core9_PQA}_validation.csv", "id,label,p0,p1,p2\npqa-1,0,0.6,0.3,0.1\n")
    for rel in (f"{r5}/checks_s8/predictions/s8_ei_1_none-raw_{core9_EI}_test.csv",
                f"{r5}/checks_s8/logits/s8_ei_2_title_{core9_EI}_test_logits.csv",
                f"{r5}/checks_s8/s8_ei_3_swap-raw_{core9_EI}_test.csv"):
        put(rel, "id,label,p0,p1,p2\nevidence-test-1,1,0.2,0.7,0.1\n")
    for stage in ("trees", "tabicl", "biogpt", "calibrate"):
        if stage in not_built:
            put(f"pod/not_built/{stage}.txt", f"NOT BUILT: {stage}, made-up reason\n")
        else:
            put(f"pod/stage_done/{stage}.txt", "2026-10-01T12:00:00Z\n")
    put("pod/osf_filed.txt", f"link {core9_OSF['link']}\nfiled_utc {core9_OSF['filed_utc']}\n")
    return {"results": root / r5, "results8": root / "results" / "fiorillo_v0_4", "results7": root / "results" / "fiorillo_v0_3",
            "base": root / "pod"}


def core9_freeze(paths, staging=None):
    return core9_C9.freeze(paths["results"], paths["results8"], paths["results7"], base=paths["base"], staging=staging)


def test_core9_freeze_refuses_before_osf_filed(tmp_path):
    """The freeze refuses to run before the pod's osf_filed.txt (the OSF update's link and filing time) exists."""
    paths = core9_freeze_layout(tmp_path, omit=("pod/osf_filed.txt",))
    with core9_pytest.raises(SystemExit, match="osf_filed.txt does not exist"):
        core9_freeze(paths)
    assert not (paths["results"] / "release_candidate.json").exists()


core9_FREEZE_INPUTS = ([(f"pod/stage_done/{s}.txt", s) for s in ("trees", "tabicl", "biogpt", "calibrate")]
                       + [(f, s) for s, files in core9_STAGE_OUTPUTS.items() for f in files]
                       + [("results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json", None),
                          ("results/fiorillo_v0_3/calibration_s7.json", None)])


@core9_pytest.mark.parametrize("missing, stage", core9_FREEZE_INPUTS)
def test_core9_freeze_refuses_before_its_inputs(tmp_path, missing, stage):
    """The freeze refuses to run before trees, tabicl, biogpt and calibrate have finished (their done marker and their
    outputs) or written a NOT BUILT marker, and before the frozen files that no stage writes exist."""
    paths = core9_freeze_layout(tmp_path, omit=(missing,))
    match = rf"waits for \['{stage}'\]" if stage else rf"{core9_Path(missing).name} does not exist"
    with core9_pytest.raises(SystemExit, match=match):
        core9_freeze(paths)
    assert not (paths["results"] / "release_candidate.json").exists()


@core9_pytest.mark.parametrize("stage", ["tabicl", "biogpt"])
def test_core9_freeze_accepts_a_not_built_marker(tmp_path, stage):
    """A stage that wrote a NOT BUILT marker (not_built/<stage>.txt) counts as finished without its done marker or its
    outputs, and the freeze records the marker's reason."""
    paths = core9_freeze_layout(tmp_path, not_built=(stage,))
    assert not (paths["base"] / "stage_done" / f"{stage}.txt").exists()
    assert not any((tmp_path / f).exists() for f in core9_STAGE_OUTPUTS[stage])
    rc = core9_freeze(paths)
    assert rc["stages"] == {s: f"not built: NOT BUILT: {s}, made-up reason" if s == stage else "finished"
                            for s in ("trees", "tabicl", "biogpt", "calibrate")}
    if stage == "biogpt":
        assert rc["biogpt_large_pqa"] == core9_read_json(paths["results"] / "calibration_s9.json")["biogpt_large_pqa"]


@core9_pytest.mark.parametrize("folder", [None, "predictions", "logits", "large_model", "checks_v05", "staging"])
def test_core9_freeze_does_not_count_checks_s8(tmp_path, folder):
    """checks_s8/ holds s8_ei's reading checks on the test split and does not count: with only those, the freeze runs.
    A v0.5 test prediction in predictions/, logits/, large_model/, checks_v05/ or the pod's staging folder stops it."""
    paths = core9_freeze_layout(tmp_path)
    assert sum("_test" in p.name for p in (paths["results"] / "checks_s8").rglob("*")) == 3
    staging = paths["base"] / "predictions_staging"
    staging.mkdir(parents=True)
    files = {"predictions": paths["results"] / "predictions" / f"s9_swap_1-raw_{core9_EI}_test.csv",
             "logits": paths["results"] / "logits" / f"s9_swap_1_{core9_EI}_test_logits.csv",
             "large_model": paths["results"] / "large_model" / f"gemma4_31b_long-raw_{core9_EI}_test.csv",
             "checks_v05": paths["results"] / "checks_v05" / "logits" / f"{core9_V05}_none_{core9_EI}_test_logits.csv",
             "staging": staging / f"{core9_V05}-raw_{core9_EI}_test.csv"}
    if folder is None:
        rc = core9_freeze(paths, staging)
        assert (paths["results"] / "release_candidate.json").exists() and rc["system"] == "fiorillo_v0_5"
    else:
        core9_write_text(files[folder], "id,label,p0,p1,p2\nevidence-test-1,1,0.2,0.7,0.1\n")
        with core9_pytest.raises(SystemExit, match="a test prediction of a v0.5 system already exists"):
            core9_freeze(paths, staging)
        assert not (paths["results"] / "release_candidate.json").exists()


def test_core9_freeze_records_sha256(tmp_path):
    """release_candidate.json records the SHA-256 of recipe_selection.json, weight_average.json, calibration_s7.json,
    calibration_s9.json, ece_limit_evidence_effect.json, handoff_rule_s8.json, handoff_rule_v05.json,
    readmission_pool_v2.json and each pool member's validation file, and of nothing else; with the OSF update's link and
    filing time, and each stage's state."""
    paths = core9_freeze_layout(tmp_path)
    rc = core9_freeze(paths)
    assert core9_read_json(paths["results"] / "release_candidate.json") == rc
    table = {str(core9_Path(k).resolve()): v for k, v in rc["sha256"].items()}
    assert table == {str((tmp_path / f).resolve()): core9_sha256(tmp_path / f) for f in core9_FROZEN}
    assert rc["osf_update"] == core9_OSF and rc["stages"] == {s: "finished" for s in ("trees", "tabicl", "biogpt", "calibrate")}
    assert rc["specialists"][core9_EI]["system"] == core9_V05 and rc["ece_limits"][core9_EI]["limit"] == 0.05


def core9_after_freeze(tmp_path, log=True):
    """The freeze; then, as the pod's testpreds stage writes them, testpreds_log.json and v0.5's made-up raw test logits."""
    paths = core9_freeze_layout(tmp_path)
    core9_freeze(paths)
    if log:
        core9_write_json(paths["results"] / "testpreds_log.json", {
            "release_candidate_sha256": core9_sha256(paths["results"] / "release_candidate.json"),
            "started_utc": "2026-10-01T13:00:00Z"})
    z = core9_write_system(paths["results"], core9_V05, core9_noisy(160, core9_TEST_LABELS, 1.0), ids=core9_TEST_IDS,
                           labels=core9_TEST_LABELS, split="test")
    return paths, z


@core9_pytest.mark.parametrize("frozen, log", [(False, False), (False, True), (True, False)])
def test_core9_apply_refuses_without_release_candidate_or_testpreds_log(tmp_path, frozen, log):
    """apply refuses to run on the test split unless release_candidate.json and testpreds_log.json both exist, and writes
    nothing."""
    if frozen:
        paths, _ = core9_after_freeze(tmp_path, log=log)
    else:
        paths = core9_freeze_layout(tmp_path)
        core9_write_system(paths["results"], core9_V05, core9_noisy(160, core9_TEST_LABELS, 1.0), ids=core9_TEST_IDS,
                           labels=core9_TEST_LABELS, split="test")
        if log:
            core9_write_json(paths["results"] / "testpreds_log.json", {"started_utc": "2026-10-01T13:00:00Z"})
    missing = "testpreds_log.json" if frozen else "release_candidate.json"
    with core9_pytest.raises(SystemExit, match=f"{missing} does not exist"):
        core9_C9.apply(paths["results"], paths["results7"])
    assert not (paths["results"] / "predictions" / f"{core9_V05}_{core9_EI}_test.csv").exists()


@core9_pytest.mark.parametrize("changed", [None, "results/fiorillo_v0_5/calibration_s9.json",
                                           "results/fiorillo_v0_5/recipe_selection.json",
                                           "results/fiorillo_v0_3/calibration_s7.json"])
def test_core9_apply_refuses_a_changed_frozen_file(tmp_path, changed):
    """apply refuses to run when a file it reads (calibration_s9.json, recipe_selection.json, calibration_s7.json) no
    longer has the SHA-256 the freeze recorded, and writes nothing; with every file as frozen, it writes v0.5's
    calibrated test predictions by the frozen family and parameters."""
    paths, z = core9_after_freeze(tmp_path)
    out = paths["results"] / "predictions" / f"{core9_V05}_{core9_EI}_test.csv"
    if changed:
        path = tmp_path / changed
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with core9_pytest.raises(SystemExit, match="changed after the freeze"):
            core9_C9.apply(paths["results"], paths["results7"])
        assert not out.exists()
    else:
        assert core9_C9.apply(paths["results"], paths["results7"]) == {core9_V05: core9_N_TEST}
        ids, labels, got = core9_read_probs(out)
        assert ids == core9_TEST_IDS and (labels == core9_TEST_LABELS).all()
        want = core9_CFC.apply_family("temperature", [0.2], core9_np.log(core9_softmax(z)))
        core9_np.testing.assert_allclose(got, want, rtol=0, atol=2e-9)


# ======================================================================================================================
# Part 10 of 11: Fiorillo v0.5's release files (scripts/build_model_card_v0_5.py, scripts/predict_v0_5.py and
# scripts/release_fiorillo_v0_5.sh)
# From test_release_session9.py. Module-level names prefixed rel9; test names test_rel9_*.
#
# Tests for Fiorillo v0.5's release files (docs/prompt_session9.md, THE RELEASE, and step 4: "the model card builder on
# made-up results" and "predict_v0_5.py on a made-up release folder with the two-layer stand-in"), on made-up inputs in
# tmp_path:
# - scripts/build_model_card_v0_5.py: every number in the card traced to its file and key (or to the numbers it was derived
#   from), and no digit in the prose that no file gave, names, versions and citations aside; a changed number in a file
#   changes the card; NOT BUILT text, and no number, for a missing file; no card when the bar was not met; swap averaging's
#   two passes and their cost; correction 7 (f)'s statement; README.md and docs/model_card_v0_5.md the same text;
# - scripts/predict_v0_5.py: its questions, calibration families, readmission frames, passage selection loader and TabICL
#   loader equal the project's; on a made-up release folder with a two-layer Qwen3 stand-in, its answers to one record of
#   each question (and to a prompt whose intervention and comparator are the same text) equal the project's own pipeline:
#   the article cut by the shipped selection, the readers, swap averaging, the weight average, each specialist's calibration
#   and the readmission pool;
# - scripts/release_fiorillo_v0_5.sh: bash -n; the refusal when SHA256SUMS.txt does not match, before any hf call; with a
#   stand-in hf on PATH, no call that would make a repository public, and the refusal of an existing public repository.
# Nothing here contacts Hugging Face: the stand-in hf answers every call, HOME is a folder under tmp_path, HF_HUB_OFFLINE
# is set, and a stand-in uv fails if it is ever called.
# ======================================================================================================================

rel9_HERE = Path(__file__).resolve()
rel9_SRC = (rel9_HERE.parents[1] if (rel9_HERE.parents[1] / "eval" / "trees_session9.py").exists()
            else Path.home() / "mnt" / "ed-decision-model" / "src")
rel9_ROOT = rel9_SRC.parent

rel9_SCRIPT = rel9_ROOT / "scripts" / "release_fiorillo_v0_5.sh"
rel9_LAYA_TOKENIZER = rel9_ROOT / "models" / "laya-base" / "tokenizer"
rel9_needs_laya = pytest.mark.skipif(not (rel9_LAYA_TOKENIZER / "tokenizer.json").exists(),
                                     reason="models/laya-base/tokenizer is not present")
rel9_NOW = dt.datetime(2026, 10, 2, 18, 5, tzinfo=dt.timezone.utc)
rel9_TOKEN_RE = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\w\-])")


def rel9_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return path


def rel9_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- made-up results for the card builder

class rel9_Values:
    """Distinct made-up numbers, so that a number in the card can be matched to the one place it came from."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def p(self, low=0.05, high=0.95):
        return round(float(self.rng.uniform(low, high)), 6)

    def ival(self, low=0.05, high=0.95):
        e = self.p(low, high)
        return {"estimate": e, "low": round(e - self.p(0.005, 0.03), 6), "high": round(e + self.p(0.005, 0.03), 6)}

    def metrics(self, n):
        return {"n": n, **{k: self.ival() for k in ("log_loss", "brier", "ece", "accuracy", "macro_f1")}}


def rel9_results(folder, swap=True, wavg=True, overall="PASS", members=None, seed=20261002):
    """Every input of the card builder, made up, in folder; returns {name: path}."""
    folder = Path(folder)
    v = rel9_Values(seed)
    singles = ["s9_ei_wavg"] if wavg else ["s9_swap_1", "s9_swap_2", "s9_swap_3"]
    base = "s9_ei_wavg" if wavg else "s9_ei"
    v05 = base + ("_sa" if swap else "")
    weights = members or {"lightgbm_tuned": 0.1, "xgboost_tuned": 0.0, "catboost_tuned": 0.0, "arm_laya": 0.0, "s7_b3": 0.2,
                          "xgboost_codes": 0.4, "catboost_codes": 0.0, "tabicl": 0.3}
    rc = {"plan": "made up", "system": "fiorillo_v0_5", "time_utc": "2026-10-01T21:14:09Z",
          "specialists": {
              "pubmedqa_answer": {"system": "s7_b3", "family": "temperature_intercepts", "params": [v.p(), -v.p(), -v.p()]},
              "evidence_effect": {"system": v05, "base_system": base, "swap_averaging": swap, "recipe": "swap",
                                  "runs": ["s9_swap_1", "s9_swap_2", "s9_swap_3"], "single_pass_systems": singles, "max_len": 4096,
                                  "family": "vector", "params": [v.p(), v.p(), v.p(), v.p(), v.p()]},
              "readmit_window": {"system": "readmission_pool_v2", "chosen": "new", "members": list(weights), "weights": weights,
                                 "T": 1 + v.p(0.01, 0.2)}},
          "reference": {"criterion_3_large_model": "gemma4_31b"},
          "ece_limits": {"evidence_effect": {"limit": v.p(0.05, 0.07), "floor": 0.05, "p95": v.p(0.02, 0.07)}},
          "osf_update": {"link": "https://osf.io/abcde/", "filed_utc": "2026-10-01T09:41:00Z"}}
    metrics = {
        "evidence_effect": {v05: v.metrics(1218), "gemma4_31b": v.metrics(1218), "s8_ei": v.metrics(1218), "prior": v.metrics(1218)},
        "pubmedqa_answer": {"s7_b3": v.metrics(500), "biogpt_large_pqa-raw": v.metrics(500), "gemma4_31b": v.metrics(500),
                            "prior": v.metrics(500)},
        "readmit_window": {"readmission_pool_v2": v.metrics(6998), "readmission_pool": v.metrics(6998),
                           "lightgbm_tuned": v.metrics(6998), "prior": v.metrics(6998)}}

    def slopes(answers, n):
        return {"system": "x", "n": n, "answers": {a: {"intercept": v.ival(-0.5, 0.5), "slope": v.ival(0.6, 1.4),
                                                        "intercept_with_slope": v.ival(-0.5, 0.5),
                                                        "rows_with_this_answer": int(v.rng.integers(50, 400))} for a in answers}}

    def rates():
        return {f"rate_{r}": {"handed_off": int(r * 12.18), **{c: v.ival() for c in ("confidence", "random_expected", "oracle",
                                                                                      "large_model_confidence")}}
                for r in (10, 20, 30, 50)}

    def pair(question, fio, n):
        return {"question": question, "fiorillo": fio, "large_model": "gemma4_31b", "n": n,
                "fiorillo_alone": {"accuracy": v.p(), "wrong": 70}, "large_model_alone": {"accuracy": v.p(), "wrong": 90},
                "registered_rates": rates(), "areas": {k: v.ival(0.01, 0.2) for k in ("aurc", "e_aurc", "augrc")}}

    table = []
    for system in ("v05", "s8_ei"):
        for inp in ("none", "title", "swap"):
            stats = ["accuracy", "macro_f1"] + (["agreement_with_full_input"] if inp != "swap" else ["exchange_share", "stay_share"])
            for s in stats:
                table.append({"system": system, "name": v05 if system == "v05" else "s8_ei", "input": inp, "statistic": s,
                              **v.ival(), "n": 1218, "articles": 333})
        table.append({"system": system, "input": "title minus none (prompts in both)", "statistic": "macro_f1_difference",
                      **v.ival(-0.1, 0.2), "n": 1209, "articles": 331})
        for inp in ("none", "title", "swap"):
            table.append({"system": system, "input": inp, "statistic": "reading", "reading": f"a made-up reading of {inp}",
                          "caveats": []})
    readings = {s: {i: {"reading": f"a made-up reading of {i}", "caveats": []} for i in ("none", "title", "swap")}
                for s in ("v05", "s8_ei")}
    readings["v05"]["swap"]["caveats"] = ["consistent by construction (swap averaging kept)"] if swap else []
    passes = [{"system": s, "forward_seconds": v.p(100, 300)} for s in singles] * (2 if swap else 1)
    comparison = {
        "plan": "made up", "written_utc": "2026-10-02T03:20:44Z",
        "settings": {"interval": "95 percent, percentile, paired", "resamples": 1000, "resamples_bar": 10000, "ece_bins": 15},
        "contamination": "Known contamination: a made-up note.",
        "metrics": metrics,
        "calibration_slope_intercept": {
            "evidence_effect": slopes(("increased", "decreased", "no_difference"), 1218),
            "pubmedqa_answer": slopes(("yes", "no", "maybe"), 500),
            "readmit_window": {"system": "readmission_pool_v2", "intercept_30d": v.ival(-0.3, 0.3), "slope_30d": v.ival(0.7, 1.3)}},
        "release_bar": {"overall": overall, "outcome": "made up", "specialist": v05, "large_model": "gemma4_31b",
                        "criteria": {"1": {"result": "PASS", "values": {"ece": v.p(0.01, 0.04), "limit": v.p(0.05, 0.06)}},
                                     "2": {"result": "PASS", "values": {"system": v05, "reference": "prior", **v.ival(-0.9, -0.5)}},
                                     "3": {"result": "PASS", "values": {"system": v05, "reference": "gemma4_31b",
                                                                         **v.ival(-0.3, -0.1)}},
                                     "4": {"result": "PASS" if overall == "PASS" else "FAIL",
                                           "values": {"specialist": v.p(0.85, 0.95), "large_model": v.p(0.7, 0.84)}}}},
        "k4": {"readings": readings, "table": table,
               "references": {"prior": {"text": "the training split's answer shares score a macro-F1 of 0.198 on the test split"},
                              "windisch_2026": {"text": "frontier models scored an accuracy of 0.79 to 0.88 from titles",
                                                "source": "Windisch et al. 2026, JMIR AI"}}},
        "k5": {"caveat": "BioGPT-Large-PubMedQA was fine-tuned on the validation items"},
        "k7": {"pairs": [pair("evidence_effect", "s8_ei", 1218), pair("evidence_effect", v05, 1218),
                         pair("pubmedqa_answer", "s7_b3", 500)]},
        "cost_per_1000_questions": {"hourly_usd": 3.52, "specialists": {"evidence_effect": {
            "system": v05, "passes": passes, "seconds_per_question": v.p(0.1, 0.9), "usd": v.p(0.1, 0.9)}}}}
    candidates = {}
    for c in ("swap", "long", "lr1"):
        candidates[c] = {"runs": [f"s9_{c}_1", f"s9_{c}_2"], "scores": {}, "score": v.p(0.15, 0.3), "passes": c == "swap"}
    candidates["lr2"] = {"status": "dropped", "passes": False}
    recipe = {"plan": "made up", "n_clean_validation": 654,
              "margin": {"m": v.p(0.005, 0.01), "spread": v.p(0.001, 0.004), "floor": 0.005,
                         "base_scores": {"s9_b_1": v.p(0.2, 0.3), "s9_b_2": v.p(0.2, 0.3), "s9_b_3": v.p(0.2, 0.3)}},
              "candidates": candidates, "combination": {"called": False, "parts": ["swap"]},
              "winner": {"recipe": "swap", "settings": {"recipe": "swap", "epochs": 1, "max_len": 4096, "lr_adapters": 5e-5},
                         "runs": ["s9_swap_1", "s9_swap_2", "s9_swap_3"]},
              "weight_average": {"built": True, "score": v.p(0.15, 0.2), "s9_ei_score": v.p(0.15, 0.2), "m": v.p(0.005, 0.01),
                                 "uses_wavg": wavg},
              "swap_averaging": {"system": base, "plain_score": v.p(0.15, 0.2), "swap_averaged_score": v.p(0.15, 0.2),
                                 "m": v.p(0.005, 0.01), "kept": swap}}
    calibration = {"families_table_v05": {"system": v05, "chosen": "vector",
                                          "table": {f: {"mean": v.p(0.15, 0.3), "min": v.p(0.1, 0.15), "max": v.p(0.3, 0.4)}
                                                    for f in ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")}}}
    pool = {"plan": "made up", "rows": 6997, "chosen": "new",
            "choice": {"chosen": "new", "new_out_of_fold_log_loss": v.p(0.8, 0.85), "session8_out_of_fold_log_loss": v.p(0.85, 0.9),
                       "session8_minus_new": v.p(0.002, 0.01), "margin": 0.002, "time_utc": "2026-10-01T20:02:31Z"}}
    licenses = {"written_utc": "2026-10-01T06:36:16Z", "read_utc": {"first": "2026-10-01T06:30:15Z", "last": "2026-10-01T06:36:09Z"},
                "by_split": {"train": {"articles": 2657, "prompts": 10046, "clean_articles": 1431, "clean_prompts": 5279},
                             "validation": {"articles": 340, "prompts": 1237, "clean_articles": 179, "clean_prompts": 654}}}
    fixed = {"qwen3_4b_base": {"repo": "Qwen/Qwen3-4B-Base", "revision": "906bfd4b4dc7f14ee4320094d8b41684abff8539"},
             "tabicl": {"version": "2.2.0", "checkpoint": {"repo": "jingang/TabICL", "revision": "4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
                                                           "file": "tabicl-classifier-v2-20260212.ckpt", "sha256": "ab" * 32}}}
    coverage = {"max_len": 6144, "coverage": {"train": {"coverage": v.p(0.97, 0.995), "covered": 9754, "prompts_with_usable_offsets": 9850}},
                "at_4096_session8": {"train": {"coverage": v.p(0.95, 0.97), "covered": 9487, "prompts_with_usable_offsets": 9850}}}
    selection8 = {"max_len": 4096, "bm25": {"N": 44280, "avgL": 100 + v.p(10, 60), "articles": 2657, "vocabulary": 115286, "k1": 1.2,
                                            "b": 0.75}}
    calibration7 = {"s7_b3": {"pubmedqa_answer": {"n_calibrate": 150, "time_utc": "2026-09-30T06:20:33Z", "chosen": "temperature_intercepts"}}}
    files = {"comparison": ("v05/comparison.json", comparison), "release_candidate": ("v05/release_candidate.json", rc),
             "calibration": ("v05/calibration_s9.json", calibration), "recipe": ("v05/recipe_selection.json", recipe),
             "pool": ("v05/readmission_pool_v2.json", pool), "licenses": ("v05/license_recheck.json", licenses),
             "fixed": ("v05/fixed_versions.json", fixed), "coverage": ("v05/coverage_long.json", coverage),
             "selection8": ("v04/article_selection.json", selection8), "calibration7": ("v03/calibration_s7.json", calibration7),
             "comparison8": ("v04/comparison.json", {"time_utc": "2026-10-01T02:53:16Z",
                                                      "metrics": {"evidence_effect": {}, "pubmedqa_answer": {}, "readmit_window": {}}}),
             "comparison7": ("v03/comparison.json", {"time_utc": "2026-09-30T06:45:22Z", "metrics": {"pubmedqa_answer": {}, "readmit_window": {}}}),
             "comparison5": ("v02/comparison.json", {"created": "2026-09-28", "metrics": {"pubmedqa_answer": {}, "readmit_window": {}}}),
             "comparison4": ("v01/comparison.json", {"created": "2026-09-27", "metrics": {"pubmedqa_answer": {}, "readmit_window": {}}})}
    paths = {name: rel9_write_json(folder / rel, data) for name, (rel, data) in files.items()}
    prior = folder / "docs" / "prior_art.md"
    prior.parent.mkdir(parents=True, exist_ok=True)
    prior.write_text("# Prior art\n\n### Session 9 check, 1 October 2026\n\nmade up\n\n### Release-day check, 2 October 2026 (v0.5)\n\n"
                     "made up\n", encoding="utf-8")
    paths["prior_art"] = prior
    return paths


def rel9_load(path):
    path = Path(path)
    return path.read_text(encoding="utf-8") if path.suffix == ".md" else json.loads(path.read_text(encoding="utf-8"))


def rel9_follow(data, keys):
    for k in keys:
        data = data[k]
    return data


def rel9_recompute(source, paths):
    """A derived number again, from its inputs' files."""
    vals = [rel9_follow(rel9_load(paths[name]), keys) for name, *keys in source["names"]]
    how = source["how"]
    if how == "difference":
        return vals[0] - sum(vals[1:])
    if how == "sum":
        return sum(vals)
    if how == "mean":
        return sum(vals) / len(vals)
    if how == "complement":
        return 1 - vals[0]
    if how == "length":
        return len(vals[0])
    if how == "passes":
        return len(vals[0]) * (2 if vals[1] else 1)
    if how == "readers":
        return sum(1 for m in vals[0] if m in rel9_card.READERS and float(vals[1][m]) > 0)
    raise AssertionError(f"unknown derivation {how}")


def rel9_prose(text):
    return re.sub(r"```.*?```", "", text, flags=re.S)


# ---------------------------------------------------------------- the model card builder

def rel9_build(tmp_path, **kwargs):
    paths = rel9_results(tmp_path / "results", **kwargs)
    text, sources, not_built = rel9_card.build(paths, now=rel9_NOW)
    return paths, text, sources, not_built


def test_rel9_card_every_number_traced_to_its_file(tmp_path):
    paths, text, sources, not_built = rel9_build(tmp_path)
    assert not_built == [] and "Status: complete" in text
    by_label = {rel9_card.rel(p): name for name, p in paths.items()}
    checked = {"file": 0, "derived": 0, "clock": 0}
    for s in sources:
        assert s["text"] in text, s
        if s["file"] == "clock":
            checked["clock"] += 1
            continue
        if s["file"] == "derived":
            assert abs(float(rel9_recompute(s, paths)) - float(s["value"])) < 1e-12, s
            checked["derived"] += 1
            continue
        data = rel9_load(paths[by_label[s["file"]]])
        if s["format"] == "rate":
            assert s["keys"][-1] == s["value"] and isinstance(rel9_follow(data, s["keys"]), dict), s
        elif by_label[s["file"]] == "prior_art":
            assert s["keys"][1] in data.splitlines(), s
        else:
            assert rel9_follow(data, s["keys"]) == s["value"], s
            assert s["text"] == rel9_card.FORMATS[s["format"]](s["value"]), s
        checked["file"] += 1
    assert checked["file"] > 200 and checked["derived"] >= 6 and checked["clock"] == 2
    # no digit in the prose that a file did not give, names, versions and citations aside
    allowed = {t for s in sources for t in rel9_TOKEN_RE.findall(s["text"])}
    allowed |= {t for lit in rel9_card.LITERALS for t in rel9_TOKEN_RE.findall(lit)}
    untracked = sorted({t for t in rel9_TOKEN_RE.findall(rel9_prose(text))} - allowed)
    assert untracked == [], untracked
    # planted numbers, read where they belong
    rc, comp = rel9_load(paths["release_candidate"]), rel9_load(paths["comparison"])
    ll = comp["metrics"]["evidence_effect"]["s9_ei_wavg_sa"]["log_loss"]
    assert f"| **v0.5's EI specialist (s9_ei_wavg_sa)** | 1,218 | {ll['estimate']:.4f} ({ll['low']:.4f} to {ll['high']:.4f}) |" in text
    assert f"T is {rc['specialists']['readmit_window']['T']:.4f}." in text
    assert "1,226 of them (derived)" in text and "1,610 in all (derived)" in text
    assert "Prepared: 2 October 2026" in text and "Hugging Face, 2026." in text
    assert "The prior-art check of the release day, 2 October 2026," in text


def test_rel9_card_follows_its_files(tmp_path):
    paths, text, _, _ = rel9_build(tmp_path)
    comp = rel9_load(paths["comparison"])
    entry = comp["metrics"]["pubmedqa_answer"]["s7_b3"]["macro_f1"]
    old = f"{entry['estimate']:.4f} ({entry['low']:.4f} to {entry['high']:.4f})"
    assert old in text
    entry.update({"estimate": 0.612345, "low": 0.598765, "high": 0.634567})
    rel9_write_json(paths["comparison"], comp)
    text2, _, _ = rel9_card.build(paths, now=rel9_NOW)
    assert "0.6123 (0.5988 to 0.6346)" in text2 and old not in text2
    lic = rel9_load(paths["licenses"])
    lic["by_split"]["train"]["clean_articles"] = 1400
    rel9_write_json(paths["licenses"], lic)
    text3, _, _ = rel9_card.build(paths, now=rel9_NOW)
    assert "1,257 of them (derived)" in text3 and "1,579 in all (derived)" in text3


def test_rel9_card_not_built_for_a_missing_file(tmp_path):
    paths = rel9_results(tmp_path / "results")
    full, full_sources, _ = rel9_card.build(paths, now=rel9_NOW)
    pool_numbers = {s["text"] for s in full_sources if s["file"] == rel9_card.rel(paths["pool"])}
    gone = dict(paths, pool=tmp_path / "nowhere" / "readmission_pool_v2.json")
    text, sources, not_built = rel9_card.build(gone, now=rel9_NOW)
    assert "NOT BUILT: the readmission pool, " in text and "readmission_pool_v2.json does not exist" in text
    assert "NOT BUILT: readmission splits, " in text and "NOT BUILT: readmission training data, " in text
    assert any(x.startswith("the readmission pool:") for x in not_built)
    assert "Status: incomplete: three parts are marked NOT BUILT" in text
    assert not any(s["file"] == rel9_card.rel(gone["pool"]) for s in sources)
    printed = set(rel9_TOKEN_RE.findall(rel9_prose(text)))
    leaked = [x for x in pool_numbers if x in printed and not any(s["text"] == x for s in sources)]
    assert leaked == []
    # every other part is still built
    assert "## Question card: evidence_effect" in text and "| **v0.5's readmission pool** |" in text
    # with no comparison.json, the test results are NOT BUILT, and none of its numbers is printed
    comp_numbers = [s["text"] for s in full_sources if s["file"] == rel9_card.rel(paths["comparison"]) and len(s["text"]) > 5]
    gone = dict(paths, comparison=tmp_path / "nowhere" / "comparison.json")
    text, sources, not_built = rel9_card.build(gone, now=rel9_NOW)
    for part in ("Evidence Inference test metrics", "the release bar", "the reading checks", "PubMedQA test metrics",
                 "readmission test metrics", "Evidence Inference hand-off"):
        assert f"NOT BUILT: {part}, " in text and "comparison.json does not exist" in text
    printed = set(rel9_TOKEN_RE.findall(rel9_prose(text)))
    leaked = [n for n in comp_numbers if n in printed and not any(s["text"] == n for s in sources)]
    assert leaked == []
    assert not any(s["file"] == rel9_card.rel(gone["comparison"]) for s in sources)


def test_rel9_card_written_twice_the_same_and_refused_when_the_bar_fails(tmp_path):
    paths = rel9_results(tmp_path / "results")
    readme, doc = tmp_path / "release" / "README.md", tmp_path / "docs" / "model_card_v0_5.md"
    args = [x for name, option, _ in rel9_card.INPUTS for x in (option, str(paths[name]))]
    assert rel9_card.main(args + ["--readme", str(readme), "--doc", str(doc)]) == 0
    assert readme.read_bytes() == doc.read_bytes() and readme.read_text().startswith("---\nlicense: apache-2.0\n")
    failed = rel9_results(tmp_path / "failed", overall="FAIL")
    out = tmp_path / "failed_out"
    with pytest.raises(SystemExit, match="bar was not met"):
        rel9_card.main([x for name, option, _ in rel9_card.INPUTS for x in (option, str(failed[name]))]
                       + ["--readme", str(out / "README.md"), "--doc", str(out / "card.md")])
    assert not out.exists()


def test_rel9_card_swap_averaging_takes_two_passes_with_its_cost(tmp_path):
    paths, text, _, _ = rel9_build(tmp_path / "kept", swap=True, wavg=True)
    comp = rel9_load(paths["comparison"])
    cost = comp["cost_per_1000_questions"]["specialists"]["evidence_effect"]
    assert "So each question takes two passes of each reader" in text
    assert (f"Cost: two reader passes a question. In the test predictions, run alone on a rented GPU (an NVIDIA H100), they "
            f"took {cost['seconds_per_question']:.3f} GPU seconds a question, which at its rate of 3.52 dollars an hour is "
            f"{cost['usd']:.3f} dollars per thousand questions.") in text
    assert "| evidence_effect | s9_ei_wavg_sa: one model whose adapters, projection and head average its recipe's three runs" in text
    assert "and answers both orders of the comparison (swap averaging)" in text
    paths, text, _, _ = rel9_build(tmp_path / "not_kept", swap=False, wavg=False)
    assert "Swap averaging is not kept" in text and "question takes one pass of each reader" in text
    assert "Cost: three reader passes a question." in text and "two passes of each reader" not in text
    assert "the probability average of its recipe's three runs (s9_ei)" in text


def test_rel9_card_one_pass_and_the_cards_wording(tmp_path):
    """Step 17: one reader pass is written as one; the large model's criterion reads "below that of"; comparison.json's
    statements lose the clause on models v0.5 does not use and the plan's session numbers and headings; TabICL is named
    only when it is a member; and the ECE interval's note."""
    paths, text, sources, _ = rel9_build(tmp_path, swap=False, wavg=True)
    assert "Cost: one reader pass a question. In the test predictions, run alone on a rented GPU (an NVIDIA H100), it took " in text
    # step 17's review: what was known at the freeze, why nothing is handed off, every date a split was read, the families,
    # and each question's own contamination statement
    assert "before any test prediction of v0.5's new Evidence Inference specialist existed" in text
    assert "so their test results were known before the freeze" in text
    assert "The rate was fixed at zero before v0.5 was trained" in text
    assert text.count("also read on 29 September 2026, for a comparison that this card does not report") == 2
    assert "The families: temperature, one temperature for all the answers" in text and "(Kull et al. 2019)" in text
    assert rel9_card.PQA_CONTAMINATION in text and rel9_card.RW_CONTAMINATION in text
    assert "Its reader is Qwen3-4B-Base" in text and "takes one pass of its reader" in text
    assert "the levels 0 (no readmission), 1 (after more than 30 days) and 2 (within 30 days)" in text
    assert "- config.json (a short description of the release's parts, which predict_v0_5.py does not read" in text  # correction 9
    comp0 = rel9_load(paths["comparison"])
    sub = [r for r in comp0["k4"]["table"] if r.get("input") == rel9_card.SUBSET]
    if sub:  # the swap check on comparators that are a time point or baseline, apart
        assert "On the exchanged prompts whose comparator is a time point or baseline" in text
        assert "- Comparators that are a time point or baseline:" in text
    assert "| Log loss below that of " in text and "'s (calibrated on validation)" not in text
    assert "A percentile interval of ECE can lie above its estimate" in text
    members = [m for m, w in rel9_load(paths["release_candidate"])["specialists"]["readmit_window"]["weights"].items() if w > 0]
    expect = "the tree members and TabICL need none." if "tabicl" in members else "the tree members need none."
    assert expect in text
    comp = rel9_load(paths["comparison"])
    comp["contamination"] = ("Known contamination: a made-up note; Laya's, ModernBERT's and reader v1's notes and session 4's "
                             "notes on Duan stand. EI's test split was read in session 8, for v0.4's bar, its comparisons.")
    comp["k5"]["caveat"] = ("so its calibration (session 8's rule on those items) was fitted on items it was trained on "
                            "(PUBMEDQA BESIDE BIOGPT-LARGE)")
    rel9_write_json(paths["comparison"], comp)
    text2, sources2, _ = rel9_card.build(paths, now=rel9_NOW)
    assert "- Known contamination: a made-up note. EI's test split was read before, for v0.4's bar, its comparisons." in text2
    assert ("- About the BioGPT-Large-PubMedQA row: so its calibration (v0.4's calibration rule on those items) was fitted "
            "on items it was trained on (docs/fiorillo_v0_5.md, the project's report).") in text2
    for gone in ("Laya's, ModernBERT's", "session 8", "session 4", "PUBMEDQA BESIDE"):
        assert gone not in text2, gone
    card = [s for s in sources2 if s["format"] == "card"]
    assert len(card) >= 2 and all(s["text"] == rel9_card.card_words(s["value"]) and s["text"] in text2 for s in card)


def test_rel9_card_pubmedqa_statement_and_house_style(tmp_path):
    paths, text, _, _ = rel9_build(tmp_path)
    for sentence in ("Every Evidence Inference article that v0.5's EI specialist learned from, or was calibrated and chosen on, is "
                     "CC BY, CC0 or the public domain.",
                     "s7_b3 learned from PubMedQA's abstracts (its automatically labeled items, PQA-A; its unlabeled items, "
                     "PQA-U; and the training part of its expert-labeled items, PQA-L) under PubMedQA's MIT License.",
                     "those abstracts' own licenses were not checked one by one, and most PubMed abstracts are the publishers'",
                     "s7_b3's labels for PQA-U came from this project's own labeler, not from an outside language model.",
                     "s7_b3 is v0.5's PubMedQA specialist and is a member of the readmission pool, with weight 0.2."):
        assert sentence in text, sentence
    assert "v0.4's EI specialist, s8_ei, learned from all 2,657 articles" in text and "UNVERIFIABLE" in text
    assert "hands nothing off" in text and "jev" not in text.lower() and "session 6" not in text.lower()
    assert chr(0x2014) not in text and chr(0x2013) not in text
    with pytest.raises(SystemExit, match="to our knowledge"):
        rel9_card.check_text("It is the first openly licensed model of its kind.")
    rel9_card.check_text("To our knowledge, it is the first openly licensed model of its kind.")
    # s7_b3 with no weight: the statement says so
    w = {"lightgbm_tuned": 0.0, "xgboost_tuned": 0.6, "catboost_tuned": 0.4, "arm_laya": 0.0, "s7_b3": 0.0,
         "xgboost_codes": 0.0, "catboost_codes": 0.0, "tabicl": 0.0}
    _, text, _, _ = rel9_build(tmp_path / "no_s7", members=w)
    assert "is in the readmission pool with no weight, so it does not answer readmission." in text


# ---------------------------------------------------------------- predict_v0_5.py: its parts against the project's

rel9_DIAGS = ["250.83 (Endocrine)", "250 (Endocrine)", "428 (Circulatory)", "38 (Infectious)", "V57 (Supplementary)",
              "E888 (External causes)", "414 (Circulatory)", "786 (Symptoms)"]


def rel9_uci_rows(split, n, seed):
    """Made-up readmission records in session 4's format (some fields missing, some codes unseen in training)."""
    from baselines import UCI_CATEGORICAL, UCI_NUMERIC
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        state = {c: int(rng.integers(0, 20)) for c in UCI_NUMERIC}
        state["time_in_hospital"], state["number_inpatient"] = int(rng.integers(1, 14)), int(rng.integers(0, 4))
        for c in UCI_CATEGORICAL:
            if not c.startswith("diag_"):
                state[c] = str(rng.choice([f"{c} a", f"{c} b"]))
        if rng.random() < 0.2:
            del state["medical_specialty"]
        for f in ("diag_1", "diag_2", "diag_3"):
            if rng.random() < 0.9:
                state[f] = str(rng.choice(rel9_DIAGS + ([] if split == "train" else ["999 (unseen)", "V99 (unseen)"])))
        if rng.random() < 0.7:
            state["medications"] = {"insulin": str(rng.choice(["Steady", "Up"]))}
        score = 0.45 * state["number_inpatient"] + 0.08 * state["time_in_hospital"] + rng.normal(0.0, 0.8)
        label = 2 if score > 1.6 else (1 if score > 0.9 else 0)
        rows.append({"id": f"uci-{split}-{i}", "qid": "readmit_window", "label": label,
                     "target": [float(label == k) for k in range(3)], "state": state})
    return rows


def rel9_uci_records(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for split, n, seed in (("train", 160, 1), ("validation", 60, 2)):
        with open(folder / f"readmit_window_{split}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rel9_uci_rows(split, n, seed))
    return folder


def test_rel9_predict_questions_and_families_are_the_projects():
    import calibration_family_cv as CFC
    from build_items import load_questions
    q8 = load_questions(rel9_SRC / "questions" / "session8.yaml")
    q4 = load_questions(rel9_SRC / "questions" / "public_v0.yaml")
    for qid, bank in (("evidence_effect", q8), ("pubmedqa_answer", q4), ("readmit_window", q4)):
        assert rel9_predict.QUESTIONS[qid] == bank[qid]
        assert list(rel9_predict.QUESTIONS[qid]["crit"]) == list(bank[qid]["crit"])
    rng = np.random.default_rng(7)
    z = rng.normal(0, 2, size=(80, 3))
    logp = z - np.log(np.exp(z).sum(1, keepdims=True))
    y = rng.integers(0, 3, size=80)
    for fam in ("temperature", "temperature_intercepts", "vector"):
        params = CFC.fit_family(fam, logp, y)
        assert np.allclose(rel9_predict.apply_family(fam, params, logp), CFC.apply_family(fam, params, logp), atol=1e-15, rtol=0)
    params = {"W": rng.normal(1, 0.3, size=(3, 3)).tolist(), "b": rng.normal(0, 0.3, size=3).tolist(), "lambda_od": 0.01, "lambda_b": 0.01}
    assert np.allclose(rel9_predict.apply_family("dirichlet_odir", params, logp), CFC.apply_family("dirichlet_odir", params, logp),
                       atol=1e-15, rtol=0)
    entry = {"chosen": "temperature_intercepts", "families": {"temperature_intercepts": {"params": [0.2, -0.1, 0.3]}}}
    assert rel9_predict.entry_family(entry) == ("temperature_intercepts", [0.2, -0.1, 0.3])
    assert rel9_predict.entry_family({"family": "vector", "params": [1, 1, 1, 0, 0]}) == ("vector", [1, 1, 1, 0, 0])
    with pytest.raises(SystemExit):
        rel9_predict.apply_family("vector", [1.0, 1.0], logp)


def test_rel9_predict_frames_are_frame_from_tables(tmp_path):
    import pandas as pd
    import trees_session9 as T9
    records = rel9_uci_records(tmp_path / "records")
    _, tables, _ = T9.build_frames(records, ("train", "validation"))
    tables = json.loads(json.dumps(tables))  # as recode_tables.json holds them
    rows = rel9_uci_rows("validation", 40, 9)
    plain, coded = rel9_predict.frames(rows, tables)
    pd.testing.assert_frame_equal(plain, T9.frame_from_tables(rows, tables, codes=False), check_exact=True)
    pd.testing.assert_frame_equal(coded, T9.frame_from_tables(rows, tables, codes=True), check_exact=True)
    import trees_session8 as T8
    pd.testing.assert_frame_equal(rel9_predict.cat_frame(plain), T8.cat_frame(plain)[0], check_exact=True)


def test_rel9_predict_loads_the_shipped_selection_and_restores_the_path(tmp_path):
    shipped = tmp_path / "passage_selection" / "evidence_inference.py"
    shipped.parent.mkdir(parents=True)
    shutil.copyfile(rel9_SRC / "tasks" / "evidence_inference.py", shipped)
    before_path, before_records = list(sys.path), sys.modules.get("records")
    sel = rel9_predict.load_selection(shipped)
    assert sys.path == before_path and sys.modules.get("records") is before_records
    assert sel.SPARE == 16 and sel.__file__ == str(shipped)
    import evidence_inference as EI8
    text = "TITLE: a trial\n\nABSTRACT.RESULTS: pain fell\n\nBODY.RESULTS:\npain was lower with aspirin\n\nmore text here"
    assert sel.article_parts(text) == EI8.article_parts(text)
    with pytest.raises(SystemExit):
        rel9_predict.load_selection(tmp_path / "none.py")


class rel9_StandInTabICL:
    """A stand-in for tabicl.TabICLClassifier: it reads its checkpoint from model_path when unpickled, as TabICL rebuilds
    its model in __setstate__."""

    def __init__(self, model_path=None, allow_auto_download=True, device=None):
        self.model_path, self.allow_auto_download, self.device = model_path, allow_auto_download, device
        self.weights = None

    def fit(self, x, y):
        self.counts = np.bincount(np.asarray(y), minlength=3).astype(float) + 1.0
        self.weights = Path(self.model_path).read_bytes()
        return self

    def __getstate__(self):
        state = dict(self.__dict__)
        state.pop("weights", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.weights = Path(self.model_path).read_bytes()

    def predict_proba(self, x):
        p = self.counts / self.counts.sum() * (1 + len(self.weights) / 1000.0)
        return np.tile(p / p.sum(), (len(x), 1))


def test_rel9_predict_tabicl_loader_reads_the_checked_checkpoint(tmp_path, monkeypatch):
    module = types.ModuleType("tabicl")
    cls = type("TabICLClassifier", (rel9_StandInTabICL,), {"__module__": "tabicl"})
    module.TabICLClassifier = cls
    monkeypatch.setitem(sys.modules, "tabicl", module)
    ckpt = tmp_path / "fitted_here" / "tabicl.ckpt"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"made-up TabICL weights")
    est = cls(model_path=str(ckpt), device="cpu").fit(np.zeros((9, 2)), [0, 1, 2, 2, 1, 0, 2, 2, 2])
    est.model_path = str(tmp_path / "gone" / "tabicl.ckpt")  # where it was fitted no longer holds the checkpoint
    folder = tmp_path / "readmit_window"
    folder.mkdir()
    (folder / "tabicl.pkl").write_bytes(pickle.dumps(est, protocol=5))
    rel9_write_json(folder / "tabicl_state.json", {"sha256": rel9_sha(folder / "tabicl.pkl"),
                                                   "checkpoint": {"repo": "jingang/TabICL", "revision": "abc", "file": "tabicl.ckpt",
                                                                  "sha256": rel9_sha(ckpt)}})
    with pytest.raises(FileNotFoundError):
        pickle.loads((folder / "tabicl.pkl").read_bytes())
    frame = np.zeros((4, 2))
    got = rel9_predict.tabicl_probs(folder, frame, "cpu", checkpoint=ckpt)
    assert got.shape == (4, 3) and np.allclose(got, est.predict_proba(frame))
    loaded = rel9_predict.load_tabicl(folder / "tabicl.pkl", ckpt, "cpu")
    assert type(loaded) is cls and loaded.model_path == str(ckpt) and loaded.allow_auto_download is False
    other = tmp_path / "other.ckpt"
    other.write_bytes(b"other weights")
    with pytest.raises(SystemExit, match="SHA-256"):
        rel9_predict.tabicl_probs(folder, frame, "cpu", checkpoint=other)


# ---------------------------------------------------------------- predict_v0_5.py on a made-up release folder

rel9_WORDS = ("aspirin placebo mortality pain score trial patients reduced increased significant difference weeks dose outcome "
              "baseline randomized control blood pressure fell rose events").split()


def rel9_article(seed):
    rng = np.random.default_rng(seed)

    def para(n):
        return " ".join(str(w) for w in rng.choice(rel9_WORDS, n))

    return "\n".join(["TITLE: A randomized trial of aspirin against placebo for pain", "", "ABSTRACT.BACKGROUND: " + para(25), "",
                      "ABSTRACT.RESULTS: " + para(25), "", "BODY.METHODS:", para(45), "", para(45), "", "BODY.RESULTS:", para(45), "",
                      para(45), "", "BODY.DISCUSSION:", para(45), "", para(45)])


def rel9_stats(article):
    import collections
    import evidence_inference as EI8
    _, passages = EI8.article_parts(article)
    df, total = collections.Counter(), 0
    for _, _, t, _ in passages:
        w = EI8.words(t)
        df.update(set(w))
        total += len(w)
    return {"N": len(passages), "avgL": total / len(passages), "df": dict(sorted(df.items())), "articles": 1, "k1": 1.2, "b": 0.75}


def rel9_tiny_base(folder):
    """A two-layer Qwen3 stand-in with Laya's tokenizer, saved as a model folder (src/tests/test_qwen_head.py's)."""
    import torch
    from transformers import AutoModel, AutoTokenizer, Qwen3Config
    tok = AutoTokenizer.from_pretrained(str(rel9_LAYA_TOKENIZER))
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, pad_token_id=tok.pad_token_id)
    AutoModel.from_config(cfg).save_pretrained(str(folder))
    tok.save_pretrained(str(folder))
    return folder


def rel9_phase(base_folder, out, name, seed, max_len):
    """A phase folder as train_qwen_head.py writes one, its adapters, projection and head moved off their start."""
    import torch
    import train_qwen_head as Q
    model, _ = Q.build_model(Q.load_base(base_folder, torch.float32), seed)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.requires_grad:
                p.add_(0.05 * torch.randn(p.shape, generator=g))
    args = types.SimpleNamespace(lora_r=16, lora_alpha=32.0, lora_dropout=0.05, max_len=max_len, seed=seed)
    Q.save_phase(out, model, args, base_folder, {"system": name, "keep": "last"})
    return out


def rel9_project_logits(base_folder, phase, records, questions, max_len):
    """The project's own logits: train_qwen_head.py's model, qwen_items.py's items and train_laya.py's run()."""
    import torch
    import train_qwen_head as Q
    from build_items import load_questions
    from qwen_items import build_qwen_items
    from train_laya import run
    from transformers import AutoTokenizer
    cfg = json.loads((Path(phase) / "config.json").read_text())
    a = cfg["adapters"]
    model, _ = Q.build_model(Q.load_base(base_folder, torch.float32), cfg["seed"], a["r"], a["alpha"], a["dropout"])
    Q.load_trainable(model, Path(phase) / "adapter.safetensors")
    tok = AutoTokenizer.from_pretrained(str(phase))
    items, summary = build_qwen_items(records, load_questions(questions), tok, max_len)
    assert summary["skipped"] == 0
    rows = run(model, items, tok.pad_token_id, torch.device("cpu"), 1, None)
    return {r["id"]: np.array(r["logits"], dtype=float) for r in rows}


def rel9_release(tmp_path, variant):
    """A made-up release folder for predict_v0_5.py: the stand-in readers, the shipped selection, the calibrations, and a
    readmission pool of xgboost_codes, catboost_tuned and s7_b3 with weight (the members without weight have no file)."""
    import catboost as cb
    import trees_session9 as T9
    import weight_average_session9 as WA
    import xgboost as xgb
    base = rel9_tiny_base(tmp_path / "base")
    work, rel = tmp_path / "work", tmp_path / "release"
    ei = rel / "evidence_effect"
    runs = [rel9_phase(base, work / f"s9_swap_{k}", f"s9_swap_{k}", 20260926 + k, 320) for k in (1, 2)]
    if variant == "wavg_swap":
        WA.build(runs, ei / "s9_ei_wavg", results=work / "wa_results")
        singles, swap, system, family = ["s9_ei_wavg"], True, "s9_ei_wavg_sa", "dirichlet_odir"
        params = {"W": [[1.1, 0.1, -0.2], [0.05, 0.9, 0.1], [-0.1, 0.2, 1.2]], "b": [0.1, -0.2, 0.05], "lambda_od": 0.01,
                  "lambda_b": 0.001}
    else:
        for r in runs:
            shutil.copytree(r, ei / r.name)
        singles, swap, system, family, params = [r.name for r in runs], False, "s9_ei", "temperature", [0.3]
    rel9_write_json(ei / "calibration.json", {"system": system, "base_system": system.replace("_sa", ""), "swap_averaging": swap,
                                              "recipe": "swap", "runs": [r.name for r in runs], "single_pass_systems": singles,
                                              "max_len": 320, "family": family, "params": params})
    (ei / "passage_selection").mkdir(parents=True)
    shutil.copyfile(rel9_SRC / "tasks" / "evidence_inference.py", ei / "passage_selection" / "evidence_inference.py")
    article = rel9_article(5)
    rel9_write_json(ei / "passage_selection" / "bm25_stats.json", rel9_stats(article))
    pqa = rel / "pubmedqa_answer"
    rel9_phase(base, pqa / "s7_b3", "s7_b3", 20260930, 256)
    rel9_write_json(pqa / "calibration.json", {"system": "s7_b3", "family": "temperature_intercepts", "params": [0.4, -0.3, -0.9]})
    rw = rel / "readmit_window"
    records = rel9_uci_records(work / "records")
    data, tables, plain = T9.build_frames(records, ("train", "validation"))
    rel9_write_json(rw / "recode_tables.json", tables)
    x, y, _ = data["train"]
    model = xgb.XGBClassifier(objective="multi:softprob", n_estimators=6, max_depth=3, learning_rate=0.3, tree_method="hist",
                              enable_categorical=True, random_state=20260927, n_jobs=1)
    model.fit(x, y)
    model.save_model(str(rw / "xgboost_codes.json"))
    import trees_session8 as T8
    xc, cols = T8.cat_frame(plain["train"][0])
    cbm = cb.CatBoostClassifier(loss_function="MultiClass", iterations=8, depth=3, random_seed=20260927, verbose=False,
                                allow_writing_files=False, thread_count=1)
    cbm.fit(xc, plain["train"][1], cat_features=cols)
    cbm.save_model(str(rw / "catboost_tuned.cbm"))
    weights = {"lightgbm_tuned": 0.0, "xgboost_tuned": 0.0, "catboost_tuned": 0.3, "arm_laya": 0.0, "s7_b3": 0.2,
               "xgboost_codes": 0.5, "catboost_codes": 0.0, "tabicl": 0.0}
    rel9_write_json(rw / "pool.json", {"system": "readmission_pool_v2", "chosen": "new", "members": list(weights), "weights": weights,
                                       "T": 0.93})
    rel9_write_json(rw / "s7_b3_calibration.json", {"chosen": "vector", "families": {"vector": {"params": [1.1, 0.9, 1.2, 0.1, -0.2]}}})
    return base, rel, article, tables, {"singles": singles, "swap": swap, "family": family, "params": params}


@rel9_needs_laya
@pytest.mark.parametrize("variant", ["wavg_swap", "runs_plain"])
def test_rel9_predict_answers_one_record_of_each_question_as_the_project_does(tmp_path, variant):
    import calibrate_session7 as C7
    import calibration_family_cv as CFC
    import catboost as cb
    import ei_session9 as E9
    import evidence_inference as EI8
    import recipe_session9 as R9
    import trees_session8 as T8
    import trees_session9 as T9
    from transformers import AutoTokenizer
    base, rel, article, tables, ei = rel9_release(tmp_path, variant)
    state = {"intervention": "aspirin", "comparator": "placebo", "outcome": "pain score", "article": article}
    same = {"intervention": "Placebo ", "comparator": " placebo", "outcome": "blood pressure", "article": article}
    pqa_state = {"question": "Does aspirin reduce pain after surgery?", "contexts": [{"section": "RESULTS", "text": "pain fell " * 20}]}
    rw_row = rel9_uci_rows("validation", 1, 11)[0]
    records = [{"id": "ei-1", "qid": "evidence_effect", "state": state}, {"id": "ei-same", "qid": "evidence_effect", "state": same},
               {"id": "pqa-1", "qid": "pubmedqa_answer", "state": pqa_state},
               {"id": "rw-1", "qid": "readmit_window", "state": rw_row["state"]}]
    path = tmp_path / "records.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "answers.jsonl"
    assert rel9_predict.main(["--records", str(path), "--release", str(rel), "--base", str(base), "--device", "cpu",
                              "--out", str(out)]) == 0
    got = {a["id"]: a for a in map(json.loads, out.read_text().splitlines())}

    # Evidence Inference: the article cut by the project's selection at 320 tokens, the project's readers, swap averaging
    folder = rel / "evidence_effect" / ei["singles"][0]
    tok = AutoTokenizer.from_pretrained(str(folder))
    stats = json.loads((rel / "evidence_effect" / "passage_selection" / "bm25_stats.json").read_text())
    budget = EI8.Budget(tok, rel9_SRC / "questions" / "session8.yaml", max_len=320)
    cuts = {}
    for rid, st in (("ei-1", state), ("ei-same", same)):
        prompt = {k: st[k] for k in ("intervention", "comparator", "outcome")}
        prefix, passages = EI8.article_parts(st["article"])
        enc = tok([json.dumps(p[2], ensure_ascii=False)[1:-1] for p in passages] or [""], add_special_tokens=False)["input_ids"]
        sel = EI8.select_one(prompt, prefix, passages, [len(e) for e in enc][: len(passages)], budget.count_escaped(prefix), stats,
                             budget)
        cuts[rid] = EI8.assemble(prefix, sel[0], passages, sel[1])
        assert len(cuts[rid]) < len(article)
        mine = rel9_predict.cut_article(st, tok, rel9_predict.load_selection(rel / "evidence_effect" / "passage_selection" /
                                                                            "evidence_inference.py"), stats,
                                        rel9_predict.SelectionBudget(tok, rel9_predict.QwenFormat(tok, rel9_predict.QUESTIONS["evidence_effect"]),
                                                                     EI8, 320))
        assert mine == cuts[rid]
    ei_records = []
    for rid, st in (("ei-1", state), ("ei-same", same)):
        rec = {"id": rid, "qid": "evidence_effect", "state": dict(st, article=cuts[rid]), "label": 0, "target": [1.0, 0.0, 0.0]}
        ei_records.append(rec)
        exch = E9.swapped(rec, suffix="-exchanged")
        if exch is not None:
            ei_records.append(exch)
    assert [r["id"] for r in ei_records] == ["ei-1", "ei-1-exchanged", "ei-same"]
    probs, single = {"ei-1": [], "ei-same": []}, []
    for s in ei["singles"]:
        z = rel9_project_logits(base, rel / "evidence_effect" / s, ei_records, rel9_SRC / "questions" / "session8.yaml", 320)
        p1 = R9.softmax(z["ei-1"][None])
        probs["ei-1"].append(R9.swap_average(p1, R9.softmax(z["ei-1-exchanged"][None]))[0] if ei["swap"] else p1[0])
        probs["ei-same"].append(R9.softmax(z["ei-same"][None])[0])
        single.append(p1[0])
    if ei["swap"]:  # the swap-averaged answer differs from the single pass, so the comparison below can tell them apart
        assert np.abs(np.mean(probs["ei-1"], axis=0) - np.mean(single, axis=0)).max() > 1e-3
    if len(ei["singles"]) > 1:  # the runs differ, so the average is not one run's answer
        assert np.abs(probs["ei-1"][0] - probs["ei-1"][1]).max() > 1e-3
    for rid in ("ei-1", "ei-same"):
        logp = np.log(np.maximum(np.mean(probs[rid], axis=0), 1e-300))
        want = CFC.apply_family(ei["family"], ei["params"], logp[None])[0]
        have = np.array(list(got[rid]["probabilities"].values()))
        assert list(got[rid]["probabilities"]) == ["significantly increased", "significantly decreased", "no significant difference"]
        assert np.allclose(have, want, atol=1e-6, rtol=0), (rid, have, want)
        assert got[rid]["answer"] == list(got[rid]["probabilities"])[int(np.argmax(want))]
    assert got["ei-1"]["passes"] == len(ei["singles"]) * (2 if ei["swap"] else 1)
    assert got["ei-same"]["passes"] == len(ei["singles"])
    if ei["swap"]:
        assert any("same text" in n for n in got["ei-same"]["notes"]) and any("swap averaging" in n for n in got["ei-1"]["notes"])

    # PubMedQA: s7_b3's logits and session 7's calibration
    s7 = rel / "pubmedqa_answer" / "s7_b3"
    pq = {"id": "pqa-1", "qid": "pubmedqa_answer", "state": pqa_state, "label": 0, "target": [1.0, 0.0, 0.0]}
    z = rel9_project_logits(base, s7, [pq], rel9_SRC / "questions" / "public_v0.yaml", 256)["pqa-1"]
    want = CFC.calibrated(C7.log_softmax(z[None]), [0.4, -0.3, -0.9], 3)[0]
    assert np.allclose(list(got["pqa-1"]["probabilities"].values()), want, atol=1e-6, rtol=0)
    assert list(got["pqa-1"]["probabilities"]) == ["yes", "no", "maybe"] and got["pqa-1"]["system"] == "s7_b3"

    # readmission: the members with weight, from the saved tables, and the pool
    rw = rel / "readmit_window"
    row = [dict(rw_row)]
    p_xgb = T9.predict_tree("xgboost_codes", T9.load_tree("xgboost_codes", rw), T9.frame_from_tables(row, tables, codes=True))
    cbm = cb.CatBoostClassifier()
    cbm.load_model(str(rw / "catboost_tuned.cbm"))
    p_cat = T8.predict_member("catboost_tuned", cbm, T9.frame_from_tables(row, tables, codes=False))
    z = rel9_project_logits(base, s7, row, rel9_SRC / "questions" / "public_v0.yaml", 256)[rw_row["id"]]
    p_s7 = CFC.apply_family("vector", [1.1, 0.9, 1.2, 0.1, -0.2], C7.log_softmax(z[None]))
    logs = np.log(np.clip(np.stack([p_xgb, p_cat, p_s7]), 1e-12, None))
    want = T8.pool_probs(logs, [0.5, 0.3, 0.2], 0.93)[0]
    assert np.allclose(list(got["rw-1"]["probabilities"].values()), want, atol=1e-9, rtol=0)
    assert list(got["rw-1"]["probabilities"]) == ["0", "1", "2"] and got["rw-1"]["passes"] == 1
    for a in got.values():
        assert abs(sum(a["probabilities"].values()) - 1) < 1e-9


@rel9_needs_laya
def test_rel9_predict_refuses_a_reader_trained_at_another_length(tmp_path):
    base, rel, article, _, ei = rel9_release(tmp_path, "runs_plain")
    cal = json.loads((rel / "evidence_effect" / "calibration.json").read_text())
    cal["max_len"] = 512
    rel9_write_json(rel / "evidence_effect" / "calibration.json", cal)
    records = [{"id": "ei-1", "qid": "evidence_effect", "state": {"intervention": "a", "comparator": "b", "outcome": "c", "article": article}}]
    with pytest.raises(SystemExit, match="trained at 320 tokens"):
        rel9_predict.predict(records, rel, base, "cpu")
    out = rel9_predict.predict([{"id": "x-1", "qid": "other", "state": {}}], rel, base, "cpu")
    assert out == [{"id": "x-1", "qid": "other", "error": "unknown question other; this release answers "
                    "['evidence_effect', 'pubmedqa_answer', 'readmit_window']"}]


# ---------------------------------------------------------------- scripts/release_fiorillo_v0_5.sh

rel9_STUB_HF = r'''#!/usr/bin/env python3
"""A stand-in for the hf command: it logs every call and answers from a state file; it never contacts Hugging Face."""
import hashlib, json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ["REL9_STUB_LOG"], "a", encoding="utf-8") as f:
    f.write(json.dumps(args) + "\n")
state_path = Path(os.environ["REL9_STUB_STATE"])
state = json.loads(state_path.read_text(encoding="utf-8"))
def save():
    state_path.write_text(json.dumps(state), encoding="utf-8")
if args[:2] == ["auth", "whoami"]:
    print(json.dumps({"user": state["user"], "orgs": None}))
elif args[:2] == ["repos", "create"]:
    if not state["exists"]:
        state.update({"exists": True, "private": "--private" in args, "files": {}})
        save()
    print("Repo created")
elif args[:2] == ["models", "info"]:
    if not state["exists"]:
        print("Model not found.", file=sys.stderr)
        sys.exit(1)
    print(json.dumps({"id": args[2], "private": state["private"], "sha": state.get("sha", "0" * 40),
                      "siblings": [{"rfilename": n} for n in sorted(state["files"])]}))
elif args[:2] == ["models", "ls"]:
    print(json.dumps([{"path": n, **v} for n, v in sorted(state["files"].items())], indent=2))
elif args[:1] == ["upload"]:
    if not state["exists"]:
        state.update({"exists": True, "private": "--private" in args, "files": {}})
    folder = Path(args[2])
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            data = p.read_bytes()
            entry = {"size": len(data), "blob_id": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()}
            if len(data) > 200 or os.environ.get("REL9_STUB_CORRUPT") == p.name:
                entry["lfs"] = {"size": len(data), "sha256": hashlib.sha256(data + (b"x" if os.environ.get("REL9_STUB_CORRUPT") == p.name else b"")).hexdigest(),
                                "pointer_size": 134}
            state["files"][p.relative_to(folder).as_posix()] = entry
    state["sha"] = hashlib.sha1(json.dumps(state["files"], sort_keys=True).encode()).hexdigest()
    save()
    print("uploaded")
else:
    print(f"unexpected hf call: {args}", file=sys.stderr)
    sys.exit(2)
'''
rel9_STUB_UV = "#!/bin/sh\necho \"uv $*\" >> \"$REL9_STUB_LOG.uv\"\nexit 97\n"


def rel9_release_folder(folder, sums=True):
    folder = Path(folder)
    (folder / "evidence_effect" / "s9_ei_wavg").mkdir(parents=True)
    files = {"README.md": "---\nlicense: apache-2.0\n---\n# Fiorillo v0.5\n", "LICENSE": "Apache License 2.0 (made up)\n",
             "NOTICE": "made-up notice\n", "evidence_effect/calibration.json": json.dumps({"family": "vector"}),
             "evidence_effect/s9_ei_wavg/adapter.safetensors": "w" * 500}
    for name, text in files.items():
        (folder / name).write_text(text, encoding="utf-8")
    if sums:
        (folder / "SHA256SUMS.txt").write_text("".join(f"{rel9_sha(folder / n)}  {n}\n" for n in sorted(files)), encoding="utf-8")
    return folder


def rel9_run_release(tmp_path, folder, state=None, extra_env=None):
    stub = tmp_path / "stub_bin"
    stub.mkdir(exist_ok=True)
    (stub / "hf").write_text(rel9_STUB_HF, encoding="utf-8")
    (stub / "uv").write_text(rel9_STUB_UV, encoding="utf-8")
    for name in ("hf", "uv"):
        (stub / name).chmod(0o755)
    run = len(list(tmp_path.glob("hf_calls_*.jsonl"))) + len(list(tmp_path.glob("hf_state_*.json")))
    log, state_path = tmp_path / f"hf_calls_{run}.jsonl", tmp_path / f"hf_state_{run}.json"  # each run its own log and state
    rel9_write_json(state_path, {"user": "johanneli", "exists": False, "private": False, "files": {}, **(state or {})})
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    env = {"PATH": f"{stub}:/usr/local/bin:/usr/bin:/bin", "HOME": str(home), "EDDM_HF": "hf", "EDDM_PYTHON": sys.executable,
           "REL9_STUB_LOG": str(log), "REL9_STUB_STATE": str(state_path), "HF_HUB_OFFLINE": "1", "TMPDIR": "/tmp/eddm_tmp",
           "PYTHONDONTWRITEBYTECODE": "1", **(extra_env or {})}
    proc = subprocess.run(["bash", str(rel9_SCRIPT), str(folder)], env=env, capture_output=True, text=True, timeout=120)
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    assert not Path(str(log) + ".uv").exists(), "the release script called uv"
    return proc, calls, json.loads(state_path.read_text())


def test_rel9_release_script_parses_and_reads_no_token():
    assert subprocess.run(["bash", "-n", str(rel9_SCRIPT)], capture_output=True).returncode == 0
    code = [line for line in rel9_SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]
    for pattern in ("--token", "HF_TOKEN", "huggingface/token", "--public", "settings", "visibility", "delete", "rm "):
        assert not any(pattern in line for line in code), pattern


def test_rel9_release_refuses_a_folder_that_does_not_match_its_sums(tmp_path):
    folder = rel9_release_folder(tmp_path / "changed")
    (folder / "NOTICE").write_text("changed after the sums were written\n", encoding="utf-8")
    proc, calls, _ = rel9_run_release(tmp_path, folder)
    assert proc.returncode != 0 and "Stop: these files differ from SHA256SUMS.txt: ['NOTICE']" in proc.stdout + proc.stderr
    assert calls == []
    folder = rel9_release_folder(tmp_path / "extra")
    (folder / ".DS_Store").write_bytes(b"\0")
    proc, calls, _ = rel9_run_release(tmp_path, folder)
    assert proc.returncode != 0 and "hidden files ['.DS_Store']" in proc.stdout + proc.stderr and calls == []
    folder = rel9_release_folder(tmp_path / "unlisted")
    (folder / "evidence_effect" / "notes.txt").write_text("not listed\n", encoding="utf-8")
    proc, calls, _ = rel9_run_release(tmp_path, folder)
    assert proc.returncode != 0 and "does not list: ['evidence_effect/notes.txt']" in proc.stdout + proc.stderr and calls == []
    folder = rel9_release_folder(tmp_path / "nosums", sums=False)
    proc, calls, _ = rel9_run_release(tmp_path, folder)
    assert proc.returncode != 0 and "SHA256SUMS.txt does not exist" in proc.stdout and calls == []


def rel9_assert_never_public(calls):
    allowed = (["auth", "whoami"], ["repos", "create"], ["models", "info"], ["models", "ls"], ["upload"])
    for call in calls:
        assert any(call[:len(a)] == a for a in allowed), call
        assert "--public" not in call and "--no-private" not in call and "--token" not in call and "settings" not in call, call
        if call[:2] == ["repos", "create"] or call[:1] == ["upload"]:
            assert "--private" in call, call


def test_rel9_release_with_a_stand_in_hf_uploads_privately_and_never_makes_it_public(tmp_path):
    folder = rel9_release_folder(tmp_path / "release")
    proc, calls, state = rel9_run_release(tmp_path, folder)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    rel9_assert_never_public(calls)
    assert [c[:2] for c in calls] == [["auth", "whoami"], ["repos", "create"], ["models", "info"], ["upload", "johanneli/fiorillo-v0.5"],
                                      ["models", "info"], ["models", "ls"]]
    assert calls[1][2] == "johanneli/fiorillo-v0.5" and calls[3][2:4] == [str(folder), "."]
    assert state["private"] is True and sorted(state["files"]) == sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                                                                         if p.is_file())
    assert f"On Hugging Face at revision {state['sha']}" in proc.stdout
    assert f"Uploaded to the private repository https://huggingface.co/johanneli/fiorillo-v0.5 (revision {state['sha']})." in proc.stdout
    # run again into the same private repository: allowed, and still private
    proc, calls, state = rel9_run_release(tmp_path, folder, state=state)
    assert proc.returncode == 0 and state["private"] is True
    rel9_assert_never_public(calls)


def test_rel9_release_refuses_a_public_or_foreign_repository_and_a_bad_upload(tmp_path):
    folder = rel9_release_folder(tmp_path / "release")
    proc, calls, state = rel9_run_release(tmp_path, folder, state={"exists": True, "private": False, "files": {}})
    assert proc.returncode != 0 and "exists and is not private; nothing uploaded" in proc.stdout + proc.stderr
    assert not any(c[:1] == ["upload"] for c in calls) and state["private"] is False
    rel9_assert_never_public(calls)
    proc, calls, _ = rel9_run_release(tmp_path, folder, state={"exists": True, "private": True,
                                                               "files": {"old_weights.bin": {"size": 3, "blob_id": "x"}}})
    assert proc.returncode != 0 and "already holds files that are not in the release folder (['old_weights.bin'])" in proc.stdout + proc.stderr
    assert not any(c[:1] == ["upload"] for c in calls)
    proc, calls, _ = rel9_run_release(tmp_path, folder, state={"user": "someone-else"})
    assert proc.returncode != 0 and "signed in as 'someone-else'" in proc.stdout and calls == [["auth", "whoami", "--format", "json"]]
    proc, calls, _ = rel9_run_release(tmp_path, folder, extra_env={"REL9_STUB_CORRUPT": "LICENSE"})
    assert proc.returncode != 0 and "Check failed on Hugging Face" in proc.stderr and "LICENSE: SHA-256" in proc.stderr


# ======================================================================================================================
# Part 11 of 11: src/eval/check_fiorillo_v0_5_figures.py
# From test_figcheck_session9.py. Module-level names prefixed fig9; test names test_fig9_*.
#
# Tests for src/eval/check_fiorillo_v0_5_figures.py (docs/prompt_session9.md, step 16), on a made-up document and
# made-up results files written under tmp_path, in the layouts session 9's modules write (compare_session9.py,
# recipe_session9.py next and cost, reading_checks_session9.py combine, handoff_curves_session9.py curves,
# power_session9.py plan, ei_session9.py manifest and ei_license_recheck.py):
#
# - a number that matches its source (PASS, with the file and key);
# - a number that does not (MISMATCH, with the source its context names and that source's value as printed);
# - a number with no source (NO SOURCE);
# - the rounding rule, as the documents print numbers (decimals, percentages, thousands separators, -0.0000, dollars,
#   times);
# - an interval two of whose numbers are one source's; readings; and the command, whose self-test plants errors and which
#   writes the check document's three lists.
# ======================================================================================================================

def fig9_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def fig9_iv(estimate, low, high):
    return {"estimate": estimate, "low": low, "high": high}


def fig9_results(root):
    """Made-up session 9 results files under root/results/fiorillo_v0_5/."""
    r = root / "results" / "fiorillo_v0_5"
    ei = "evidence_effect"
    fig9_write(r / "comparison.json", {
        "plan": "docs/prompt_session9.md, COMPARISONS", "written_utc": "2026-10-02T03:14:15Z",
        "settings": {"resamples": 1000, "resamples_bar": 10000, "seed": 20260927},
        "metrics": {
            ei: {"s9_ei": {"n": 1218, "log_loss": fig9_iv(0.212349, 0.17561, 0.25502),
                           "macro_f1": fig9_iv(0.93011, 0.91004, 0.94798), "ece": fig9_iv(0.01234, 0.01301, 0.03312)},
                 "prior": {"n": 1218, "log_loss": fig9_iv(1.07672, 1.05118, 1.10231),
                           "macro_f1": fig9_iv(0.19761, 0.18877, 0.20613), "ece": fig9_iv(0.03051, 0.02012, 0.04477)},
                 "gemma4_31b": {"n": 1218, "log_loss": fig9_iv(0.40781, 0.36114, 0.45663),
                                "macro_f1": fig9_iv(0.86262, 0.84013, 0.88327), "ece": fig9_iv(0.02833, 0.01911, 0.04012)}},
            "pubmedqa_answer": {"s7_b3": {"n": 500, "log_loss": fig9_iv(0.60231, 0.55102, 0.65519),
                                          "accuracy": fig9_iv(0.75201, 0.71399, 0.78802)}}},
        "comparisons": [
            {"section": "K1", "evaluation": ei, "split": "test", "system": "s9_ei", "reference": "prior",
             "metric": "log_loss", "difference": fig9_iv(-0.86437, -0.91013, -0.81662), "reading": "lower (better)",
             "excludes": "the interval, -0.9101 to -0.8166, lies wholly below zero", "ranked": True, "n": 1218,
             "system_estimate": 0.212349, "reference_estimate": 1.07672, "note": ""},
            {"section": "K1", "evaluation": ei, "split": "test", "system": "s9_ei", "reference": "prior", "metric": "ece",
             "difference": fig9_iv(-0.01861, -0.04612, 0.01873), "reading": "no difference was detected",
             "excludes": "no difference was detected", "ranked": True, "n": 1218, "system_estimate": 0.01234,
             "reference_estimate": 0.03051, "note": ""},
            {"section": "K1", "evaluation": ei, "split": "test", "system": "s9_ei", "reference": "gemma4_31b",
             "metric": "log_loss", "difference": fig9_iv(-0.19546, -0.25109, -0.14422), "reading": "lower (better)",
             "excludes": "", "ranked": True, "n": 1218, "system_estimate": 0.212349, "reference_estimate": 0.40781,
             "note": ""}],
        "bar_intervals": {
            "v05_minus_prior": {"system": "s9_ei", "reference": "prior", "estimate": -0.86437, "low": -0.92251,
                                "high": -0.82811, "resamples": 10000, "n": 1218, "articles": 333},
            "v05_minus_large_model": {"system": "s9_ei", "reference": "gemma4_31b", "estimate": -0.19546, "low": -0.27199,
                                      "high": -0.15051, "resamples": 10000, "n": 1218, "articles": 333}},
        "release_bar": {
            "specialist": "s9_ei", "large_model": "gemma4_31b",
            "criteria": {"1": {"rule": "ECE at most the limit", "values": {"ece": 0.01234, "limit": 0.05},
                               "result": "PASS", "reason": ""},
                         "2": {"rule": "log loss below the prior's", "values": {"system": "s9_ei", "reference": "prior",
                                                                               "estimate": -0.86437, "low": -0.92251,
                                                                               "high": -0.82811},
                               "result": "PASS", "reason": ""}},
            "overall": "PASS", "outcome": "every criterion passes: v0.5's bar is met, and THE RELEASE follows (step 17)"},
        "cost_per_1000_questions": {
            "hourly_usd": 3.52,
            "specialists": {ei: {"system": "s9_ei", "questions": 1218, "seconds_per_question": 0.44012, "usd": 0.43033}},
            "large_model": {ei: {"system": "gemma4_31b", "seconds_per_question": 1.93612, "usd": 1.89310}}}})
    fig9_write(r / "recipe_selection.json", {
        "plan": "docs/prompt_session9.md, THE RECIPE RULE", "written_utc": "2026-10-01T22:10:05Z",
        "n_clean_validation": 654, "folds": 5, "repeats": 20, "seed": 20260927,
        "margin": {"m": 0.00612, "spread": 0.00612, "floor": 0.005,
                   "base_scores": {"s9_b_1": 0.23121, "s9_b_2": 0.22901, "s9_b_3": 0.22509}},
        "s9_b": {"runs": ["s9_b_1", "s9_b_2", "s9_b_3"], "score": 0.22102},
        "candidates": {"swap": {"runs": ["s9_swap_1", "s9_swap_2"], "scores": {"s9_swap_1": 0.22411, "s9_swap_2": 0.22687},
                                "score": 0.22549, "passes": False, "rule": "two runs (correction 2)"}},
        "winner": {"recipe": "b", "runs": ["s9_b_1", "s9_b_2", "s9_b_3"],
                   "settings": {"lr_adapters": 5e-05, "epochs": 2, "max_len": 4096}},
        "weight_average": {"built": True, "score": 0.22051, "s9_ei_score": 0.22102, "m": 0.00612, "uses_wavg": True},
        "scores": {"s9_b": 0.22102, "s9_b_1": 0.23121, "s9_b_2": 0.22901, "s9_b_3": 0.22509, "s9_ei_wavg": 0.22051},
        "decisions": {"winner": {"value": "b", "time_utc": "2026-10-01T21:59:41Z"}}})
    fig9_write(r / "cost_rule.json", {
        "plan": "docs/prompt_session9.md, THE PILOT AND THE COST RULE", "written_utc": "2026-10-01T14:02:11Z",
        "pod": {"created_utc": "2026-10-01T13:37:21Z", "hourly_usd": 3.52, "hours_so_far": 0.41389,
                "spend_so_far_usd": 1.45689},
        "projected_total_usd": 88.4121, "stop_line_usd": 115.0, "cap_usd": 125.0,
        "projected_copy_checked_utc": "2026-10-02T09:12:00Z", "dropped": []})
    fig9_write(r / "records_manifest.json", {
        "plan": "docs/prompt_session9.md, EI'S LICENSE-CLEAN RECORDS", "written_utc": "2026-10-01T09:44:02Z",
        "expected": {"train": {"prompts": 5279, "articles": 1431}, "validation": {"prompts": 654, "articles": 179}},
        "counts": {"clean/evidence_effect_train.jsonl": {"records": 5279, "articles": 1431,
                                                        "answers": {"0": 1906, "1": 1622, "2": 1751}},
                   "clean/evidence_effect_validation.jsonl": {"records": 654, "articles": 179,
                                                             "answers": {"0": 230, "1": 199, "2": 225}}},
        "tokens_per_record": {"clean/evidence_effect_train.jsonl": {"mean": 3760.4, "p90": 4071, "largest": 4080,
                                                                   "n": 5279}},
        "long": {"max_len": 6144, "coverage_train": 0.96314}})
    fig9_write(r / "license_recheck.json", {
        "written_utc": "2026-10-01T06:36:16Z", "articles": 3348,
        "by_split": {"train": {"articles": 2657, "prompts": 10046, "clean_articles": 1431, "clean_prompts": 5279},
                     "test": {"articles": 333, "prompts": 1218, "clean_articles": 160, "clean_prompts": 600,
                              "clean_by_europe_pmc_not_now": ["PMC1", "PMC2", "PMC3", "PMC4", "PMC5"]}}})
    fig9_write(r / "power.json", {
        "source": "src/eval/power_session9.py plan", "se": 0.01503, "pass_line": -0.02946,
        "real_gain_for_80_percent": 0.04215,
        "chance_by_real_gain": [{"real_gain": 0.0, "chance": 0.025}, {"real_gain": 0.05, "chance": 0.91123}],
        "if_worse_than_s8_ei": [{"worse_than_s8_ei_by": 0.1, "real_gain": 0.1084, "chance": 0.99702}]})
    fig9_write(r / "handoff_curves.json", {
        "plan": "docs/prompt_session9.md, HAND-OFF ANALYSES", "split": "test",
        "registered_rates_percent": [10, 20, 30, 50],
        "pairs": [{"question": ei, "fiorillo": "s9_ei", "large_model": "gemma4_31b", "n": 1218, "articles": 333,
                   "fiorillo_alone": {"accuracy": 0.93432, "wrong": 80},
                   "areas": {"aurc": fig9_iv(0.01712, 0.01205, 0.02341), "e_aurc": fig9_iv(0.01499, 0.01033, 0.02086)},
                   "registered_rates": {"rate_10": {"handed_off": 121,
                                                    "confidence": fig9_iv(0.93021, 0.91106, 0.94662)}}}]})
    fig9_write(r / "reading_checks.json", {
        "plan": "docs/prompt_session9.md, THE READING CHECKS (K4)",
        "readings": {"s8_ei": {"none": {"reading": "its answers depend on the article's text", "caveats": []},
                               "swap": {"reading": "it follows the comparison as written", "caveats": []}}},
        "systems": {"s8_ei": {"system": "s8_ei",
                              "inputs": {"none": {"n": 1218, "articles": 333, "accuracy": fig9_iv(0.43021, 0.40112, 0.45988),
                                                  "macro_f1": fig9_iv(0.41233, 0.38511, 0.43872)},
                                         "swap": {"n": 1101, "articles": 330, "accuracy": fig9_iv(0.90517, 0.88013, 0.92744)}},
                              "readings": {"none": {"reading": "its answers depend on the article's text", "value": 0.41233,
                                                    "rule": "macro-F1 at or below 0.60, its answers depend on the article's "
                                                            "text; above 0.75, the prompt alone carries more"},
                                           "swap": {"reading": "it follows the comparison as written", "value": 0.90517,
                                                    "rule": "accuracy at or above 0.85, it follows the comparison"}}}}})
    return r


fig9_doc_text = """# Fiorillo v0.5

## Licenses (read 1 October 2026)

- Clean: 1,431 of the 2,657 training articles; this section is not checked, so 9,999 is not reported.

## How each stage was run

Everything on the GPU ran on one Runpod pod with a 200 GB container disk, at $3.52 an hour. Its container started at 13:37:21 UTC on 1 October.

The cost rule's projected total was $88.41, under the $115 stop line (cost_rule.json).

## Results (test split)

EI's test split has 1,218 prompts. v0.5's EI specialist, s9_ei, learned from 5,279 clean training records in 1,431 articles, with a mean of 3,760 tokens a record (records_manifest.json).

s9_ei's macro-F1 is 0.9301 (0.9100 to 0.9480) and its log loss 0.2123 (0.1756 to 0.2550); Gemma 4 31B-it's are 0.8626 (0.8401 to 0.8833) and 0.4078 (0.3611 to 0.4566).

### Table 5. Evidence Inference, every system (test split)

| System | Log loss | Macro-F1 |
|---|---|---|
| `s9_ei` | 0.2123 (0.1756 to 0.2550) | 0.9301 (0.9100 to 0.9480) |
| `prior` | 1.0767 (1.0512 to 1.1023) | 0.1976 (0.1888 to 0.2061) |
| `gemma4_31b` | 0.4078 (0.3611 to 0.4566) | 0.8626 (0.8401 to 0.8833) |

### Table 7. K1: s9_ei minus the prior

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_ei` minus `prior` | Log loss | 0.2123 | 1.0767 | -0.8644 (-0.9101 to -0.8166) | s9_ei better |
| `s9_ei` minus `prior` | ECE | 0.0123 | 0.0305 | -0.0186 (-0.0461 to 0.0187) | no difference was detected; excludes below -0.0461 and above 0.0187 |

### Table 8. The release bar

| Criterion | Measure | s9_ei, or the difference | Limit, 95 percent interval, or Gemma 4 31B-it | Result |
|---|---|---|---|---|
| 1 | ECE | 0.0123 | 0.0500 | PASS |
| 2 | Log loss, s9_ei minus the prior | -0.8644 | -0.9225 to -0.8281 | PASS |

The recipe rule (recipe_selection.json): the margin m was 0.0061, above the floor of 0.005; the weight average's score was 0.2205 against s9_ei's 0.2210.

A thousand EI questions cost $0.43 of GPU time with no hand-off.

At a real gain of 0.05, criterion 3's chance of passing was 0.91 (power.json).

The selection covered 96.31 percent of the clean training prompts.

On EI the combined answers' accuracy was 0.9302 (0.9111 to 0.9466) with 10 percent handed off, against Fiorillo's 0.9343 alone.

The reading checks: s8_ei's macro-F1 with no article was 0.4123 (0.3851 to 0.4387), so its reading is "its answers depend on the article's text".
"""


def fig9_setup(tmp_path, text=fig9_doc_text):
    fig9_results(tmp_path)
    (tmp_path / "docs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "docs" / "fiorillo_v0_5.md").write_text(text, encoding="utf-8")
    return fig9_check.Sources(tmp_path / "results" / "fiorillo_v0_5")


def fig9_rows(check, as_written, where=""):
    """The number rows printed as as_written whose where holds the given text."""
    return [r for r in check.numbers if r["as_written"] == as_written and where in r["where"]]


def fig9_one(check, as_written, where=""):
    rows = fig9_rows(check, as_written, where)
    assert len(rows) == 1, (as_written, where, [r["where"] for r in rows])
    return rows[0]


# ---------------------------------------------------------------- the four cases the task names

def test_fig9_a_number_that_matches(tmp_path):
    src = fig9_setup(tmp_path)
    check = fig9_check.Check(src, fig9_doc_text)
    want = {("0.2123", "Table 5, `s9_ei`, Log loss, estimate"): ("comparison.json", "metrics.evidence_effect.s9_ei.log_loss.estimate"),
            ("5,279", "line 15"): ("records_manifest.json", "counts.clean/evidence_effect_train.jsonl.records"),
            ("1,218", "line 15"): ("license_recheck.json", "by_split.test.prompts"),
            ("$3.52", "line 9"): ("cost_rule.json", "pod.hourly_usd"),
            ("13:37:21", "line 9"): ("cost_rule.json", "pod.created_utc"),
            ("0.0061", "line 41"): ("recipe_selection.json", "margin.m"),
            ("96.31", "line 47"): ("records_manifest.json", "long.coverage_train"),
            ("-0.9225", "Table 8"): ("comparison.json", "bar_intervals.v05_minus_prior.low")}
    for (text, where), (file, key) in want.items():
        row = fig9_one(check, text, where)
        assert row["check"] == "PASS", row
        assert (row["source"]["file"], row["source"]["key"]) == (file, key), row
    assert fig9_one(check, "13:37:21")["source"]["printed"] == "13:37:21"  # the time of day of 2026-10-01T13:37:21Z
    assert not fig9_rows(check, "9,999")  # the Licenses section is not checked
    assert not [r for r in check.numbers if r["as_written"] in ("1", "2")]  # the Criterion column holds labels


def test_fig9_a_number_that_does_not_match(tmp_path):
    src = fig9_setup(tmp_path)
    text = (fig9_doc_text.replace("the margin m was 0.0061", "the margin m was 0.0016")
            .replace("learned from 5,279 clean", "learned from 5,297 clean").replace("at $3.52 an hour", "at $3.25 an hour"))
    check = fig9_check.Check(src, text)
    for written, file, key, printed in (("0.0016", "recipe_selection.json", "margin.m", "0.0061"),
                                        ("5,297", "records_manifest.json", "counts.clean/evidence_effect_train.jsonl.records",
                                         "5,279"),
                                        ("$3.25", "cost_rule.json", "pod.hourly_usd", "$3.52")):
        row = fig9_one(check, written)
        assert row["check"] == "MISMATCH", row
        assert row["source"] is None
        assert (row["expected"]["file"], row["expected"]["key"], row["expected"]["printed"]) == (file, key, printed), row
    assert sum(r["check"] != "PASS" for r in check.numbers) == 4  # the three above and the disk's 200 (no source)


def test_fig9_a_number_with_no_source(tmp_path):
    src = fig9_setup(tmp_path)
    text = fig9_doc_text.replace(
        "The selection covered",
        "About 15 of its 78 validation errors were on those prompts (derived). The pilot trained 2.86 items a second "
        "(pilot_long.json). The selection covered")
    check = fig9_check.Check(src, text)
    disk = fig9_one(check, "200", "line 9")
    assert (disk["check"], disk["source"], disk["expected"]) == ("NO SOURCE", None, None), disk
    for written in ("15", "78"):
        row = fig9_one(check, written)
        assert row["check"] == "NO SOURCE", row
        assert "derived" in row["note"], row
    pilot = fig9_one(check, "2.86")
    assert pilot["check"] == "NO SOURCE", pilot
    assert "pilot_long.json, which is not among the sources" in pilot["note"], pilot


def test_fig9_rounding_as_the_documents_print_numbers(tmp_path):
    r = tmp_path / "results" / "fiorillo_v0_5"
    fig9_write(r / "power.json", {"standard_error": 0.123449, "coverage": 0.96314, "prompts": 1218, "difference": -0.00002,
                                  "mean_tokens": 3760.4, "floor": 0.05, "cost_usd": 0.43033,
                                  "started_utc": "2026-10-01T13:37:21Z"})
    src = fig9_check.Sources(r)
    cases = [("The standard error was 0.1234.", "PASS", None),
             ("The standard error was 0.123.", "PASS", None),
             ("The standard error was 0.1235.", "MISMATCH", "0.1234"),
             ("The coverage was 96.31 percent.", "PASS", None),
             ("The coverage was 96.32 percent.", "MISMATCH", "96.31"),
             ("The prompts were 1,218.", "PASS", None),
             ("The prompts were 1218.", "MISMATCH", "1,218"),
             ("The difference was 0.0000.", "PASS", None),
             ("The difference was -0.0000.", "MISMATCH", "0.0000"),
             ("The mean tokens were 3,760.", "PASS", None),
             ("The mean tokens were 3,761.", "MISMATCH", "3,760"),
             ("The floor was 0.0500.", "PASS", None),
             ("The cost was $0.43 in usd.", "PASS", None),
             ("The cost was $0.44 in usd.", "MISMATCH", "$0.43"),
             ("It started at 13:37:21 UTC.", "PASS", None),
             ("It started at 13:37 UTC.", "PASS", None),
             ("It started at 13:38 UTC.", "MISMATCH", "13:37")]
    for sentence, want, printed in cases:
        check = fig9_check.Check(src, f"# D\n\n## Results\n\n{sentence}\n")
        assert len(check.numbers) == 1, (sentence, check.numbers)
        row = check.numbers[0]
        assert row["check"] == want, (sentence, row)
        if printed:
            assert row["expected"]["printed"] == printed, (sentence, row)
    sep = fig9_check.Check(src, "# D\n\n## Results\n\nThe prompts were 1218.\n").numbers[0]
    assert "thousands separators" in sep["note"], sep
    neg = fig9_check.Check(src, "# D\n\n## Results\n\nThe difference was -0.0000.\n").numbers[0]
    assert "negative zero" in neg["note"], neg
    f = fig9_check.Fig("-1,234.50", 0, 9)
    assert (f.value, f.decimals, f.negative, f.separators_ok) == (-1234.5, 2, True, True)
    assert not fig9_check.Fig("1234", 0, 4).separators_ok and fig9_check.Fig("999", 0, 3).separators_ok
    assert fig9_check.printed_like(-0.00002, fig9_check.Fig("-0.0000", 0, 7)) == "0.0000"


# ---------------------------------------------------------------- intervals, readings and the command

def test_fig9_an_interval_two_of_whose_numbers_are_one_source(tmp_path):
    src = fig9_setup(tmp_path)
    check = fig9_check.Check(src, fig9_doc_text.replace("| 0.9301 (0.9100 to 0.9480) |", "| 0.9301 (0.9100 to 0.9840) |"))
    parts = {r["where"].rsplit(", ", 1)[1]: r for r in check.numbers if "Table 5, `s9_ei`, Macro-F1" in r["where"]}
    assert [parts[p]["check"] for p in ("estimate", "low", "high")] == ["PASS", "PASS", "MISMATCH"]
    assert parts["high"]["expected"]["key"] == "metrics.evidence_effect.s9_ei.macro_f1.high"
    assert parts["high"]["expected"]["printed"] == "0.9480"


def test_fig9_readings_are_checked_against_their_sources(tmp_path):
    src = fig9_setup(tmp_path)
    check = fig9_check.Check(src, fig9_doc_text)
    assert [(r["as_written"], r["check"]) for r in check.readings] == [
        ("s9_ei better", "PASS"), ("PASS", "PASS"), ("PASS", "PASS"), ("its answers depend on the article's text", "PASS")]
    assert check.readings[0]["source"]["key"].endswith("(K1, evidence_effect, test, s9_ei minus prior, log_loss).reading")
    assert check.readings[2]["source"]["key"] == "release_bar.criteria.2.result"
    text = (fig9_doc_text.replace("| s9_ei better |", "| the prior better |").replace("| 0.0500 | PASS |", "| 0.0500 | FAIL |")
            .replace("\"its answers depend on the article's text\"", "\"partly\""))
    changed = fig9_check.Check(src, text)
    assert [r["check"] for r in changed.readings] == ["MISMATCH", "MISMATCH", "PASS", "MISMATCH"]


def test_fig9_a_document_without_the_checked_sections_stops(tmp_path):
    src = fig9_setup(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        fig9_check.Check(src, "# Fiorillo v0.5\n\n## Licenses (read 1 October 2026)\n\n- 1,431 of 2,657 articles.\n")
    assert stopped.value.code == 2


def test_fig9_the_command_plants_errors_and_writes_the_three_lists(tmp_path):
    fig9_setup(tmp_path, fig9_doc_text.replace("the margin m was 0.0061", "the margin m was 0.0016"))
    assert fig9_check.main(["--root", str(tmp_path)]) == 1
    data = json.loads((tmp_path / "results" / "fiorillo_v0_5" / "figure_sources.json").read_text(encoding="utf-8"))
    assert len(data["self_test"]) >= 10
    assert all(p["reported_as"] in ("MISMATCH", "NO SOURCE") for p in data["self_test"])
    assert {p["what"] for p in data["self_test"]} == {"number", "reading"}
    assert data["counts"]["numbers"] == {"PASS": 70, "MISMATCH": 1, "NO SOURCE": 1}
    doc = (tmp_path / "docs" / "fiorillo_v0_5_figure_check.md").read_text(encoding="utf-8")
    heads = [line for line in doc.splitlines() if line.startswith("## ")]
    assert heads == ["## How the check was made", "## Result", "## Every number", "## Numbers that do not match",
                     "## Numbers with no source", "## Every reading"]
    mismatch = doc.split("## Numbers that do not match")[1].split("## Numbers with no source")[0]
    assert "| 0.0016 | recipe_selection.json margin.m | 0.0061 |" in mismatch
    nosource = doc.split("## Numbers with no source")[1].split("## Every reading")[0]
    assert "| 200 |" in nosource
    every = doc.split("## Every number")[1].split("## Numbers that do not match")[0]
    assert "| 5,279 | records_manifest.json counts.clean/evidence_effect_train.jsonl.records | 5,279 | PASS |" in every
    assert chr(0x2014) not in doc and chr(0x2013) not in doc  # no em or en dashes



def test_fig9_stage_logs_are_sources(tmp_path):
    """Step 16 added the pod's stage logs: a time-stamped line gives its time of day, and every number in a line is a
    source under the line's words; a sentence that names a log is searched in that log."""
    fig9_results(tmp_path)
    logs = tmp_path / "results" / "fiorillo_v0_5" / "logs" / "pod"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "testpreds.log").write_text(
        "2026-10-02T13:40:31Z  stage testpreds started\n"
        "Loading weights: 100%|##########| 398/398 [00:00<00:00, 3028.72it/s]\n"
        "2026-10-02T13:40:32Z  s9_b_1 predicts test from data/processed/session8 (1218 rows)\n"
        "s9_b_1 test: 1218 rows, 221.2 forward seconds\n"
        "2026-10-02T15:18:19Z  stage testpreds finished\n", encoding="utf-8")
    text = fig9_doc_text.replace(
        "The cost rule's projected total",
        "testpreds ran from 13:40:31 to 15:18:19 UTC, and s9_b_1's test predictions took 221.2 forward seconds "
        "(testpreds.log). The cost rule's projected total")
    src = fig9_check.Sources(tmp_path / "results" / "fiorillo_v0_5")
    assert "logs/pod/testpreds.log" in src.files
    check = fig9_check.Check(src, text)
    for written in ("13:40:31", "15:18:19", "221.2"):
        row = fig9_one(check, written)
        assert row["check"] == "PASS", row
        assert row["source"]["file"] == "logs/pod/testpreds.log", row
    assert not fig9_rows(check, "3028.72")  # a progress bar's line is not a source, and the document never printed it
    changed = text.replace("221.2 forward seconds", "212.2 forward seconds")
    row = fig9_one(fig9_check.Check(src, changed), "212.2")
    assert row["check"] == "MISMATCH", row


def test_fig9_numbers_the_document_attributes_do_not_keep_it_a_draft(tmp_path):
    """A number with no source is attributed when the document labels it derived or its sentence names only files that
    are not sources (docs/log.md and the like); with no mismatch and only attributed numbers without a source, the
    check is complete and the command returns 0. An unattributed one (the disk's 200) keeps it a draft."""
    text = (fig9_doc_text.replace("with a 200 GB container disk, at $3.52 an hour.", "at $3.52 an hour.")
            .replace("The cost rule's projected total",
                     "The copy took 15:23:53 to 15:27:24 UTC (docs/log.md). The pod ran 23.49 hours (derived). "
                     "The cost rule's projected total"))
    fig9_setup(tmp_path, text)
    assert fig9_check.main(["--root", str(tmp_path)]) == 0
    data = json.loads((tmp_path / "results" / "fiorillo_v0_5" / "figure_sources.json").read_text(encoding="utf-8"))
    att = {r["as_written"]: r["attributed"] for r in fig9_check.from_table(data["numbers"]) if r["check"] == "NO SOURCE"}
    assert att == {"15:23:53": "outside", "15:27:24": "outside", "23.49": "derived"}, att
    assert data["counts"]["numbers"]["MISMATCH"] == 0
    doc = (tmp_path / "docs" / "fiorillo_v0_5_figure_check.md").read_text(encoding="utf-8")
    assert "Status: complete (with 2 numbers whose sources the document names outside the files checked and 1 it labels derived" in doc
    fig9_setup(tmp_path, fig9_doc_text)  # the disk's 200 has no source and no attribution
    assert fig9_check.main(["--root", str(tmp_path)]) == 1

def test_fig9_figure_sources_is_two_tables_that_read_back(tmp_path):
    """Step 16: figure_sources.json holds the numbers and the readings as tables (the field names once, one row a line),
    so that the full document's file stays under the data guard's 1 MB limit for results/; from_table() gives back the
    rows the check made, each source a dictionary again, and a row with no source keeps null."""
    fig9_setup(tmp_path, fig9_doc_text.replace("the margin m was 0.0061", "the margin m was 0.0016"))
    src = fig9_check.Sources(tmp_path / "results" / "fiorillo_v0_5")
    check = fig9_check.Check(src, (tmp_path / "docs" / "fiorillo_v0_5.md").read_text(encoding="utf-8"))
    assert fig9_check.main(["--root", str(tmp_path)]) == 1
    path = tmp_path / "results" / "fiorillo_v0_5" / "figure_sources.json"
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    for key, made in (("numbers", check.numbers), ("readings", check.readings)):
        table = data[key]
        assert set(table) == {"fields", "dict_fields", "rows"}
        assert "source" in table["dict_fields"] and all(isinstance(r, list) for r in table["rows"])
        assert fig9_check.from_table(table) == [fig9_check.public(r) for r in made]
    rows = fig9_check.from_table(data["numbers"])
    assert sum(data["counts"]["numbers"].values()) == len(rows)
    assert any(r["source"] is None for r in rows) and any(isinstance(r["source"], dict) for r in rows)
    assert len(text.splitlines()) > len(rows)  # one row a line
    made = [{"a": 1, "source": {"file": "x.json", "key": "k", "value": 0.5}, "note": ""},
            {"a": 2, "source": None, "note": "no source"}]
    assert fig9_check.from_table(json.loads(json.dumps(fig9_check.as_table(made)))) == made


def test_handoff_rule_two_runs_correction_8(tmp_path):
    """Correction 8: I-35's rule is fitted with two runs when v0.5's winner has two, and a frozen rule is applied only with
    as many runs as it was fitted on."""
    paths, _ = hc_write_planted(tmp_path)
    two = paths["runs"][:2]
    rule = HC.fit_rule("fio", paths["fiorillo"], two, "big", paths["large"], paths["records"])
    assert len(rule["files"]["runs"]) == 2
    rule_path = tmp_path / "rule_two_runs.json"
    rule_path.write_text(json.dumps(rule), encoding="utf-8")
    HC.check_rule_runs(rule, rule_path, two)
    with pytest.raises(SystemExit):
        HC.check_rule_runs(rule, rule_path, paths["runs"])  # fitted on two runs, applied with three
    with pytest.raises(SystemExit):
        HC.main(["fit-rule", "--fiorillo-name", "fio", "--fiorillo", str(paths["fiorillo"]), "--runs", str(paths["runs"][0]),
                 "--large-name", "big", "--large", str(paths["large"]), "--records", str(paths["records"]),
                 "--out", str(tmp_path / "one_run.json")])

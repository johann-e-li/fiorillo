"""Tests for session 7's records and the labeler's rules (docs/prompt_session7.md, THE RECORDS, THE LABELER and step 4),
on made-up inputs: the exclusions and their counts; a planted 13-word match is dropped; the labeler records hold no
contexts; a builder run stopped part way resumes and gives the same files; the subset rule's N, its cap, its shares and its
tie order; and the NOT BUILT cases of the gate and of N under 1,000."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))
sys.path.insert(0, str(SRC / "tasks"))
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "corpus"))

import label_pqa_u as L  # noqa: E402
import pubmedqa_phases as P  # noqa: E402

REFERENCE_TEXT = ("the randomized trial found that early mobilization after hip fracture surgery shortened hospital stay "
                  "and reduced delirium in older adults without raising the fall rate")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(pubid, question, text, long_answer, decision=None):
    r = {"pubid": pubid, "question": question, "context": {"contexts": [text], "labels": ["RESULTS"], "meshes": []}, "long_answer": long_answer}
    if decision is not None:
        r["final_decision"] = decision
    return r


def write_parquet(path, rows):
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), str(path))


def build_world(tmp):
    """Made-up PubMedQA parts and session 4 records; returns the paths and the expected counts."""
    raw, public = tmp / "raw", tmp / "public"
    raw.mkdir()
    public.mkdir()
    labeled = [row(1, "Does A help?", REFERENCE_TEXT, "Yes, A helps.", "yes"),
               row(2, "Does B help?", "B did nothing measurable in this cohort of patients.", "No.", "no"),
               row(3, "Does C help?", "The effect of C was unclear in a small sample.", "Maybe.", "maybe"),
               row(4, "Does D help?", "D improved outcomes in a large trial.", "Yes.", "yes")]
    artificial = [row(10 + i, f"Question {i}?", f"Abstract text number {i} about a treatment.", f"Conclusion {i}.",
                      "yes" if i % 3 else "no") for i in range(9)]
    artificial.append(row(1, "Duplicate of a labeled item?", "Some text.", "Some conclusion.", "yes"))  # pubid in PQA-L
    artificial.append(row(50, "Copied abstract?", "Earlier: " + REFERENCE_TEXT + " (reprinted)", "Copied.", "yes"))  # 13-word match
    unlabeled = [row(100 + i, f"Unlabeled question {i}?", f"Unlabeled abstract {i}.", f"Unlabeled conclusion {i}.") for i in range(5)]
    unlabeled.append(row(2, "Unlabeled copy of a labeled item?", "Text.", "Conclusion."))
    for name, rows in (("pqa_labeled", labeled), ("pqa_artificial", artificial), ("pqa_unlabeled", unlabeled)):
        write_parquet(raw / f"{name}.parquet", rows)

    def s4(pubid, label, source, calibrate=True):
        r = {"id": f"pubmedqa-{pubid}", "qid": "pubmedqa_answer", "state": {"question": "q", "contexts": []}, "label": label,
             "target": [float(label == k) for k in range(3)], "group": {"source": source}}
        if not calibrate:
            r["calibrate"] = False
        return r

    train = [s4(1, 0, "pqa_labeled"), s4(1, 0, "pqa_labeled"), s4(2, 1, "pqa_labeled"), s4(11, 0, "pqa_artificial")]
    validation = [s4(3, 2, "pqa_labeled"), s4(12, 0, "pqa_artificial", calibrate=False)]
    for name, rows in (("pubmedqa_answer_train.jsonl", train), ("pubmedqa_answer_validation.jsonl", validation)):
        with open(public / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    checksums = tmp / "checksums.json"
    checksums.write_text(json.dumps({n: sha(public / n) for n in ("pubmedqa_answer_train.jsonl", "pubmedqa_answer_validation.jsonl")}))
    return raw, public, checksums


@pytest.fixture
def world(tmp_path, monkeypatch):
    raw, public, checksums = build_world(tmp_path)
    monkeypatch.setattr(P, "parquet_files", lambda raw_dir=raw: {c: (Path(raw_dir) / f"{c}.parquet", sha(Path(raw_dir) / f"{c}.parquet"))
                                                                  for c in ("pqa_labeled", "pqa_artificial", "pqa_unlabeled")})
    monkeypatch.setattr(P, "N_PQA_L", 4)
    monkeypatch.setattr(P, "N_VALIDATION_PQA_A", 1)
    monkeypatch.setattr(P, "N_L_TRAIN", 2)
    monkeypatch.setattr(P, "N_L_VALIDATION", 1)
    return tmp_path, raw, public, checksums


def run_builder(tmp, raw, public, checksums, out_name, max_seconds=1000.0):
    out = tmp / out_name
    args = ["--raw", str(raw), "--public", str(public), "--checksums", str(checksums), "--out", str(out),
            "--manifest", str(out / "manifest.json"), "--processes", "1", "--batch-rows", "4", "--max-seconds", str(max_seconds)]
    P.main(args)
    return out


def read_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def test_exclusions_counts_and_labeler_records(world):
    tmp, raw, public, checksums = world
    out = run_builder(tmp, raw, public, checksums, "out")
    manifest = json.loads((out / "manifest.json").read_text())
    a, u = manifest["counts"]["pqa_artificial"], manifest["counts"]["pqa_unlabeled"]
    assert a == {"rows": 11, "kept": 8, "pubid_in_pqa_labeled": 1, "session4_validation_item": 1, "13_word_match": 1}
    assert u == {"rows": 6, "kept": 5, "pubid_in_pqa_labeled": 1, "session4_validation_item": 0, "13_word_match": 0}
    phase1 = read_jsonl(out / "pubmedqa_phase1_train.jsonl")
    ids = [r["id"] for r in phase1]
    assert ids == sorted(ids, key=lambda x: int(x.split("-")[1])) and "pubmedqa-50" not in ids and "pubmedqa-12" not in ids
    assert all(set(r["state"]) == {"question", "contexts"} and r["group"]["source"] == "pqa_artificial" for r in phase1)
    for name in ("labeler_pqa_a.jsonl", "labeler_pqa_u.jsonl", "labeler_pqa_l_train.jsonl", "labeler_validation.jsonl"):
        rows = read_jsonl(out / name)
        assert rows and all(set(r["state"]) == {"question", "conclusion"} and r["qid"] == "pubmedqa_conclusion" for r in rows)
    assert all(r["label"] == -1 and r["target"] == [0.0, 0.0, 0.0] for r in read_jsonl(out / "labeler_pqa_u.jsonl"))
    assert [r["id"] for r in read_jsonl(out / "labeler_pqa_l_train.jsonl")] == ["pubmedqa-1", "pubmedqa-2"]
    assert [r["id"] for r in read_jsonl(out / "labeler_validation.jsonl")] == ["pubmedqa-3"]
    assert all("label" not in r for r in read_jsonl(out / "pqa_u_unlabeled.jsonl"))
    for name, entry in manifest["files"].items():
        assert entry["sha256"] == sha(out / name)


def test_a_stopped_run_resumes_to_the_same_files(world):
    tmp, raw, public, checksums = world
    whole = run_builder(tmp, raw, public, checksums, "whole")
    parts_run = tmp / "parts_run"
    for _ in range(10):  # --max-seconds 0: each run builds exactly one batch
        run_builder(tmp, raw, public, checksums, "parts_run", max_seconds=0.0)
        if (parts_run / "manifest.json").exists():
            break
    assert (parts_run / "manifest.json").exists()
    for name in P.FINAL.values():
        assert sha(whole / name) == sha(parts_run / name)


def test_subset_rule_cap_shares_and_ties():
    rng = np.random.default_rng(0)
    n = 4000
    probs = rng.dirichlet([1, 1, 1], size=n)
    ids = [f"pubmedqa-{1000 + i}" for i in range(n)]
    taken, d = L.subset(ids, probs)
    top = probs.argmax(axis=1)
    n_c = [int((top == c).sum()) for c in range(3)]
    expected_n = min(30600, int(min(n_c[0] / 0.552, n_c[1] / 0.338, n_c[2] / 0.110)))
    assert d["N"] == expected_n
    assert d["taken"] == {"yes": int(expected_n * 0.552), "no": int(expected_n * 0.338), "maybe": int(expected_n * 0.110)}
    for c, name in enumerate(("yes", "no", "maybe")):
        chosen = [i for i, lab, _ in taken if lab == c]
        assert len(chosen) == d["taken"][name]
        pool = sorted([i for i in range(n) if top[i] == c], key=lambda i: -probs[i, c])
        assert {ids[i] for i in pool[: d["taken"][name]]} == set(chosen)
    # the cap: many items of every answer
    big = np.tile(np.eye(3) * 0.8 + 0.1 / 1.0, (40000, 1))
    big = big / big.sum(axis=1, keepdims=True)
    _, d2 = L.subset([f"pubmedqa-{i}" for i in range(len(big))], big)
    assert d2["N"] == 30600
    # ties are taken in PubMed ID order
    tied = np.array([[0.9, 0.05, 0.05]] * 10 + [[0.05, 0.9, 0.05]] * 10 + [[0.05, 0.05, 0.9]] * 10)
    tie_ids = [f"pubmedqa-{i}" for i in (30, 12, 44, 7, 99, 3, 58, 21, 70, 5)] * 3
    tie_ids = [f"{x}{k}" for k, x in enumerate(tie_ids)]
    taken3, d3 = L.subset(tie_ids, tied, cap=10)
    got_yes = [i for i, lab, _ in taken3 if lab == 0]
    yes_pool = sorted(tie_ids[:10], key=L.pubid)
    assert got_yes == sorted(yes_pool[: d3["taken"]["yes"]], key=L.pubid)


def test_gate_and_not_built_cases(tmp_path):
    v01 = tmp_path / "v01.csv"
    v01.write_text("id,label,p0,p1,p2\npubmedqa-1,0,0.9,0.05,0.05\npubmedqa-2,1,0.2,0.7,0.1\npubmedqa-3,2,0.5,0.3,0.2\n")
    better = [{"id": "pubmedqa-1", "label": 0, "probs": [0.8, 0.1, 0.1]}, {"id": "pubmedqa-2", "label": 1, "probs": [0.1, 0.8, 0.1]},
              {"id": "pubmedqa-3", "label": 2, "probs": [0.1, 0.1, 0.8]}]
    same = [dict(r, probs=[0.8, 0.1, 0.1] if r["label"] != 1 else [0.1, 0.8, 0.1]) for r in better]
    ids = {"pubmedqa-1", "pubmedqa-2", "pubmedqa-3"}
    g1 = L.gate(better, v01, ids)
    assert g1["labeler_correct"] == 3 and g1["v0_1_correct"] == 2 and g1["passed"]
    g2 = L.gate(same, v01, ids)
    assert g2["labeler_correct"] == 2 and not g2["passed"]
    assert L.phase2_decision(g2) == "the labeler did not beat v0.1 on validation"
    assert L.phase2_decision(g1, {"N": 999}) == "fewer than 1,000 items could be labeled in PQA-L's proportions"
    assert L.phase2_decision(g1, {"N": 1000}) is None
    L.write_not_built(tmp_path, "made-up reason")
    data = json.loads((tmp_path / "labeler.json").read_text())
    assert data["phase2_built"] is False and data["not_built"] == "NOT BUILT: phase 2, made-up reason" and "before_the_failure" not in data
    (tmp_path / "labeler.json").write_text(json.dumps({"plan": "p", "phase2_built": True, "gate": g1, "subset": {"N": 1234}}))
    later = L.write_not_built(tmp_path, "phase 2's items for arm B could not be built")  # a failure after labeler.json was written
    assert later["phase2_built"] is False and later["before_the_failure"]["subset"] == {"N": 1234}
    assert json.loads((tmp_path / "labeler.json").read_text()) == later

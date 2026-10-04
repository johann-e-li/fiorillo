"""Tests for session 5's comparison (src/eval/compare.py --session5; docs/prompt_session5.md, step 14) on made-up records
and predictions: the output files, the direction of D1 (v0.2 minus v0.1), and the four release criteria passing, and
criterion 4 failing when v0.1 is better on PubMedQA."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "eval"))
import compare  # noqa: E402

CAND = "s5_final_fresh_head"
N_PQ, N_RW = 60, 80
FLAT_DUAN = [0.4, 0.3, 0.3]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def write_preds(path, ids, labels, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{v:.6f}" for v in p])


def calibrated(labels, k=3):
    """0.9 on the true label for nine rows in ten and on the next label for the tenth: accuracy and confidence both 0.9."""
    probs = np.full((len(labels), k), 0.05)
    wrong = np.arange(len(labels)) % 10 == 9
    probs[np.arange(len(labels)), np.where(wrong, (labels + 1) % k, labels)] = 0.9
    return probs


def on_true(value):
    def make(labels, k=3):
        probs = np.full((len(labels), k), (1 - value) / (k - 1))
        probs[np.arange(len(labels)), labels] = value
        return probs
    return make


def build(tmp, v01):
    rng = np.random.default_rng(1)
    pq_labels = rng.choice(3, size=N_PQ, p=[0.55, 0.34, 0.11])
    rw_labels = rng.choice(3, size=N_RW, p=[0.6, 0.3, 0.1])
    rwv_labels = rng.choice(3, size=N_RW, p=[0.6, 0.3, 0.1])
    pq_ids = [f"pubmedqa-{i}" for i in range(N_PQ)]
    rw_ids = [f"uci-{i}" for i in range(N_RW)]
    rwv_ids = [f"uci-v{i}" for i in range(N_RW)]
    rec, s5, v0 = tmp / "records", tmp / "s5", tmp / "v0"
    write_jsonl(rec / "pubmedqa_answer_test.jsonl", [{"id": i, "label": int(y)} for i, y in zip(pq_ids, pq_labels)])
    write_jsonl(rec / "readmit_window_test.jsonl",
                [{"id": i, "label": int(y), "group": {"sex": "Female"}} for i, y in zip(rw_ids, rw_labels)])
    write_jsonl(rec / "readmit_window_validation.jsonl", [{"id": i, "label": int(y)} for i, y in zip(rwv_ids, rwv_labels)])
    for qid, ids, labels in (("pubmedqa_answer", pq_ids, pq_labels), ("readmit_window", rw_ids, rw_labels)):
        for name in (CAND, CAND + "-raw"):
            write_preds(s5 / f"{name}_{qid}_test.csv", ids, labels, calibrated(labels))
        for name in ("arm_laya", "arm_laya-raw"):
            write_preds(v0 / f"{name}_{qid}_test.csv", ids, labels, v01(labels))
        freq = np.bincount(labels, minlength=3) / len(labels)
        write_preds(v0 / f"prior_{qid}_test.csv", ids, labels, np.tile(freq, (len(labels), 1)))
        for name in ("duan_ourwording", "duan_ourwording-raw"):
            write_preds(v0 / f"{name}_{qid}_test.csv", ids, labels, np.tile(FLAT_DUAN, (len(labels), 1)))
    write_preds(v0 / "lightgbm_readmit_window_test.csv", rw_ids, rw_labels, on_true(0.45)(rw_labels))
    keep = pq_labels < 2
    yes_no_ids = [i for i, k in zip(pq_ids, keep) if k]
    for name in ("duan", "duan-raw"):
        write_preds(v0 / f"{name}_duan_readmit_window_test.csv", rw_ids, rw_labels, np.tile(FLAT_DUAN, (N_RW, 1)))
        write_preds(v0 / f"{name}_duan_conclusion_supported_test.csv", yes_no_ids, 1 - pq_labels[keep],
                    np.tile([0.5, 0.5], (len(yes_no_ids), 1)))
    flat_val = np.tile(FLAT_DUAN, (N_RW, 1))
    for name, directory, probs in ((CAND, s5, calibrated(rwv_labels)), (CAND + "-raw", s5, calibrated(rwv_labels)),
                                   ("arm_laya", v0, v01(rwv_labels)), ("arm_laya-raw", v0, v01(rwv_labels)),
                                   ("prior", v0, np.tile([0.6, 0.3, 0.1], (N_RW, 1))), ("lightgbm", v0, on_true(0.45)(rwv_labels)),
                                   ("duan", v0, flat_val), ("duan-raw", v0, flat_val),
                                   ("duan_ourwording", v0, flat_val), ("duan_ourwording-raw", v0, flat_val)):
        qid = "duan_readmit_window" if name in ("duan", "duan-raw") else "readmit_window"
        write_preds(directory / f"{name}_{qid}_validation.csv", rwv_ids, rwv_labels, probs)
    rc = tmp / "release_candidate.json"
    rc.write_text(json.dumps({"system": CAND, "reader": "reader v1", "recipe": "--maybe-weight none", "head": "fresh head",
                              "selection": {"task_rule": {"task_round_kept": False}}, "time_utc": "2026-09-28T23:00:00Z",
                              "ece_limits": {"pubmedqa_answer": {"p95_if_calibrated": 0.09, "limit": 0.09},
                                             "readmit_window": {"p95_if_calibrated": 0.06, "limit": 0.06}}}), encoding="utf-8")
    return ["--session5", "--predictions-dir", str(v0), "--s5-predictions-dir", str(s5), "--records-dir", str(rec),
            "--release-candidate", str(rc), "--out-dir", str(tmp / "out")]


def d1(out, ev, metric="log_loss"):
    return next(c for c in out["comparisons"] if c["section"] == "D1" and c["evaluation"] == ev and c["metric"] == metric)


def test_release_bar_passes_when_the_candidate_beats_everything(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESAMPLES", 200)
    out = compare.main(build(tmp_path, on_true(0.5)))
    assert (tmp_path / "out" / "comparison.json").exists() and (tmp_path / "out" / "comparison.csv").exists()
    saved = json.loads((tmp_path / "out" / "comparison.json").read_text())
    assert saved["candidate"]["system"] == CAND and saved["release_bar"]["system"] == CAND
    c = d1(out, "pubmedqa_answer")
    assert c["system"] == CAND and c["reference"] == "arm_laya"
    assert c["difference"]["estimate"] == compare.num(c["system_estimate"] - c["reference_estimate"])
    assert c["difference"]["estimate"] < 0 and c["difference"]["high"] < 0
    assert {k: v["result"] for k, v in out["release_bar"]["criteria"].items()} == {"1": "PASS", "2": "PASS", "3": "PASS", "4": "PASS"}
    assert out["release_bar"]["overall"] == "PASS"
    assert {c["section"] for c in out["comparisons"]} >= {"D1", "D2", "D4", "D5"}
    assert "was not kept" in out["contamination"]


def test_criterion_4_fails_when_v01_is_better_on_pubmedqa(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESAMPLES", 200)
    out = compare.main(build(tmp_path, on_true(0.95)))
    assert d1(out, "pubmedqa_answer")["difference"]["estimate"] > 0
    assert out["release_bar"]["criteria"]["4"]["result"] == "FAIL" and out["release_bar"]["overall"] == "FAIL"


def test_criterion_3_fails_when_one_of_duans_variants_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESAMPLES", 200)
    args = build(tmp_path, on_true(0.5))
    (tmp_path / "v0" / "duan-raw_duan_conclusion_supported_test.csv").rename(tmp_path / "v0" / "moved_aside.csv")
    out = compare.main(args)
    c3 = out["release_bar"]["criteria"]["3"]
    assert c3["result"] == "FAIL" and "duan-raw" in c3["reason"] and out["release_bar"]["overall"] == "FAIL"


def test_d3_uses_the_baseline_the_maybe_rule_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(compare, "RESAMPLES", 50)
    args = build(tmp_path, on_true(0.5))
    rc = tmp_path / "release_candidate.json"
    data = json.loads(rc.read_text())
    data["selection"]["maybe_rule"] = {"baseline": "s5_laya_w"}
    rc.write_text(json.dumps(data))
    labels = {"pubmedqa_answer": None, "readmit_window": None}
    for qid in labels:
        rows = list(csv.DictReader(open(tmp_path / "s5" / f"{CAND}_{qid}_test.csv")))
        ids, ys = [r["id"] for r in rows], np.array([int(r["label"]) for r in rows])
        for name in ("s5_mix_a", "s5_laya_w", "s5_laya"):
            write_preds(tmp_path / "s5" / f"{name}_{qid}_test.csv", ids, ys, on_true(0.6)(ys))
    out = compare.main(args)
    d3 = {(c["system"], c["reference"]) for c in out["comparisons"] if c["section"] == "D3"}
    assert ("s5_mix_a", "s5_laya_w") in d3 and not any(ref == "s5_laya" for _, ref in d3)
    assert out["d3_baseline"] == "s5_laya_w"


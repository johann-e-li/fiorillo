"""Tests for session 8's analyses and tools (docs/prompt_session8.md, step 4, with corrections 3 and 4), on made-up inputs.

- The article-level resampling against a hand calculation on a tiny set, and the same draws in the comparisons, the bar's
  intervals and the hand-off;
- the bar's four criteria on made-up numbers, each passing and failing, and compare_session8.py end to end on made-up
  files, with a run stopped after every block resuming to the same comparison;
- the average of three seeds and its log-probabilities;
- the calibration rule on a planted bias (temperature_intercepts chosen) and on none (temperature chosen); the freeze's
  guards;
- the readmission pool recovers a planted member and gives the others no weight, and T is the maximum-likelihood T;
- trees_session8.py: the search, a search stopped part way that resumes to the same fits, validation, the pool, and the
  test stages refusing to run without release_candidate.json and testpreds_log.json or with a changed member file
  (correction 4: no fit keeps its own saved state after correction 3, so there is no test of one);
- the hand-off numbers against a hand calculation, with a second model's probabilities;
- the large model's option probabilities from made-up top-20 log-probabilities (a stand-in for vLLM): letter variants
  summed and renormalized, an item missing a letter sent the second way, and the files written after both passes;
- the plain-text prompts for EI and PubMedQA;
- power_session8.py plan and validation against hand calculations;
- pilot-s8's choice between the settings on made-up measurements, and pilot-s8 and predict's run summary (seconds per
  item) on the two-layer stand-in;
- label_pqa_u.py rebuild against a made-up SHA-256;
- upload_to_pod.py accepting session 8's stages, and the pod script's syntax and stages."""
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "tasks"))
sys.path.insert(0, str(SRC / "train"))
sys.path.insert(0, str(SRC / "tools"))

import calibrate_session8 as CAL  # noqa: E402
import compare_session8 as CMP  # noqa: E402
import handoff_session8 as H  # noqa: E402
import large_model_session8 as L  # noqa: E402
import power_session8 as PW  # noqa: E402
import upload_to_pod as U  # noqa: E402

LAYA_TOKENIZER = ROOT / "models" / "laya-base" / "tokenizer"
SEED = 20260927


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


def write_probs(path, ids, labels, probs, digits=9):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{x:.{digits}g}" for x in p])


def write_logits(path, ids, labels, z):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"z{j}" for j in range(z.shape[1])])
        for i, y, row in zip(ids, labels, z):
            w.writerow([i, int(y)] + [repr(float(x)) for x in row])


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# the article-level resampling ----------------------------------------------------------------------------------------

def test_article_resampling_against_a_hand_calculation():
    clusters = ["PMC2", "PMC10", "PMC2", "PMC3"]
    rows_of = CMP.article_rows(clusters)
    assert [r.tolist() for r in rows_of] == [[1], [0, 2], [3]]  # PMC10, PMC2, PMC3: sorted as text
    values = np.array([1.0, 2.0, 3.0, 4.0])
    est, low, high = CMP.cluster_bootstrap(lambda idx: values[idx].mean(), rows_of, 4, resamples=7, seed=SEED)
    rng = np.random.default_rng(SEED)
    by_hand = []
    for _ in range(7):
        rows = []
        for j in rng.integers(0, 3, 3):
            rows += {0: [1], 1: [0, 2], 2: [3]}[int(j)]
        by_hand.append(values[rows].mean())
    assert est[0] == 2.5 and low[0] == pytest.approx(np.percentile(by_hand, 2.5)) and high[0] == pytest.approx(np.percentile(by_hand, 97.5))
    # the bar's intervals draw the same articles
    labels = np.array([0, 1, 2, 0])
    a = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.1, 0.8], [0.5, 0.3, 0.2]])
    b = np.full((4, 3), 1 / 3)
    iv = CMP.bar_interval(labels, a, b, rows_of, resamples=7, seed=SEED)
    d = -np.log(a[np.arange(4), labels]) + np.log(b[np.arange(4), labels])
    rng = np.random.default_rng(SEED)
    hand = [d[[i for j in rng.integers(0, 3, 3) for i in ([1], [0, 2], [3])[int(j)]]].mean() for _ in range(7)]
    assert iv["estimate"] == pytest.approx(d.mean()) and iv["low"] == pytest.approx(np.percentile(hand, 2.5))
    assert iv["high"] == pytest.approx(np.percentile(hand, 97.5)) and iv["articles"] == 3
    # the hand-off draws the same articles too
    got = [idx.tolist() for idx in H.resample_indices(4, clusters, 3, SEED)]
    rng = np.random.default_rng(SEED)
    assert got == [[i for j in rng.integers(0, 3, 3) for i in ([1], [0, 2], [3])[int(j)]] for _ in range(3)]


# the bar ---------------------------------------------------------------------------------------------------------------

def bar_out(ece=0.03, prior_iv=(-0.30, -0.20), gemma_iv=(-0.06, -0.01), f1=(0.80, 0.78)):
    return {"metrics": {"evidence_effect": {"s8_ei": {"ece": {"estimate": ece}, "macro_f1": {"estimate": f1[0]}},
                                             "gemma4_31b": {"macro_f1": {"estimate": f1[1]}}}},
            "bar_intervals": {"s8_ei_minus_prior": {"estimate": prior_iv[0], "low": prior_iv[0] - 0.05, "high": prior_iv[1]},
                              "s8_ei_minus_gemma4_31b": {"estimate": gemma_iv[0], "low": gemma_iv[0] - 0.05, "high": gemma_iv[1]}}}


def test_the_bar_criteria_pass_and_fail():
    rc = {"ece_limits": {"evidence_effect": {"limit": 0.05}}}
    bar = CMP.release_bar(bar_out(), rc)
    assert bar["overall"] == "PASS" and all(c["result"] == "PASS" for c in bar["criteria"].values())
    assert bar["outcome"].startswith("every criterion passes")
    cases = {"1": bar_out(ece=0.051), "2": bar_out(prior_iv=(-0.10, 0.01)), "3": bar_out(gemma_iv=(-0.03, 0.002)),
             "4": bar_out(f1=(0.779, 0.78))}
    for k, out in cases.items():
        bar = CMP.release_bar(out, rc)
        failed = [j for j, c in bar["criteria"].items() if c["result"] == "FAIL"]
        assert failed == [k] and bar["overall"] == "FAIL" and bar["outcome"] == f"v0.4 release bar not met: criterion {k}"
    assert CMP.release_bar(bar_out(gemma_iv=(0.01, 0.05)), rc)["criteria"]["3"]["result"] == "FAIL"  # worse than the reference
    missing = CMP.release_bar({"metrics": {}, "bar_intervals": {}}, rc)
    assert missing["overall"] == "FAIL" and all(c["reason"] for c in missing["criteria"].values())
    assert CMP.reading("log_loss", -0.2, -0.1) == "lower (better)" and CMP.reading("macro_f1", -0.2, -0.1) == "lower (worse)"
    assert CMP.reading("ece", -0.01, 0.02) == "no difference was detected" and CMP.excludes(0.7, 0.8, 0.851) == "excluded: above the interval"


def made_up_study(tmp_path, n_articles=40, per_article=3):
    """Records and test predictions for EI, PubMedQA and readmission, for compare_session8.py."""
    rng = np.random.default_rng(3)
    results = tmp_path / "results" / "fiorillo_v0_4"
    records_ei, records_public = tmp_path / "records8", tmp_path / "public"
    ei = []
    for a in range(n_articles):
        for k in range(per_article):
            y = int(rng.integers(0, 3))
            ei.append({"id": f"evidence-{a * per_article + k}", "qid": "evidence_effect", "label": y, "target": [float(y == j) for j in range(3)],
                       "state": {}, "group": {"pmcid": str(1000 + a), "flag": "Questionable" if a % 7 == 0 else "none",
                                              "accepted_labels": sorted({y, int(rng.integers(0, 3))})}})
    write_jsonl(records_ei / "evidence_effect_test.jsonl", ei)
    write_jsonl(records_ei / "evidence_effect_train.jsonl", [{"id": f"t{i}", "label": i % 3} for i in range(30)])
    y_ei = np.array([r["label"] for r in ei])
    ids_ei = [r["id"] for r in ei]
    good = softmax(np.eye(3)[y_ei] * 2.0 + rng.normal(0, 1, (len(ei), 3)))
    weak = softmax(np.eye(3)[y_ei] * 0.8 + rng.normal(0, 1, (len(ei), 3)))
    for name, p in (("s8_ei", good), ("s8_ei_temperature", good), ("s8_ei-raw", good), ("s8_ei_1", good), ("s8_ei_2", weak),
                    ("gemma4_31b", weak)):
        write_probs(results / "predictions" / f"{name}_evidence_effect_test.csv", ids_ei[::-1], y_ei[::-1], p[::-1])
    for q, n in (("pubmedqa_answer", 30), ("readmit_window", 40)):
        rows = [{"id": f"{q}-{i}", "label": int(rng.integers(0, 3))} for i in range(n)]
        write_jsonl(records_public / f"{q}_test.jsonl", rows)
        if q == "readmit_window":
            write_jsonl(records_public / f"{q}_validation.jsonl", rows)
        y = np.array([r["label"] for r in rows])
        p = softmax(np.eye(3)[y] + rng.normal(0, 1, (n, 3)))
        write_probs(results / "predictions" / f"s8_a3_{q}_test.csv", [r["id"] for r in rows], y, p)
        if q == "readmit_window":
            write_probs(results / "predictions" / f"s8_a3_{q}_validation.csv", [r["id"] for r in rows], y, p)
    (results / "release_candidate.json").write_text(json.dumps({"ece_limits": {"evidence_effect": {"limit": 0.2}}, "time_utc": "t"}))
    (results / "testpreds_log.json").write_text("{}")
    return results, records_public, records_ei


def test_compare_end_to_end_and_resume(tmp_path):
    results, public, records_ei = made_up_study(tmp_path)
    whole = CMP.run(results, public, records_ei, tmp_path / "cache_whole", None, resamples=40, resamples_bar=200)
    assert whole is not None and whole["release_bar"]["criteria"]["2"]["result"] == "PASS"
    assert whole["release_bar"]["criteria"]["3"]["values"]["articles"] == 40
    g1 = [c for c in whole["comparisons"] if c["section"] == "G1" and c["reference"] == "gemma4_31b" and c["metric"] == "log_loss"][0]
    assert g1["difference"]["estimate"] == pytest.approx(whole["bar_intervals"]["s8_ei_minus_gemma4_31b"]["estimate"], abs=1e-9)
    assert whole["sensitivity"]["flagged_test_prompts"] == 18 and whole["g3"]["lenient_macro_f1"]["estimate"] >= whole["g3"]["macro_f1"]["estimate"]
    assert any("lightgbm_tuned" in m for m in whole["missing_predictions"])
    json.loads((results / "comparison.json").read_text())
    zeros = json.loads((results / "zeros.json").read_text())["rows_whose_true_answer_prints_as_0"]
    assert zeros["s8_ei"]["test"]["evidence_effect"]["rows"] == 120
    first = (results / "comparison.json").read_text()
    stopped = 0
    for _ in range(200):  # each run computes one block, then stops
        out = CMP.run(results, public, records_ei, tmp_path / "cache_parts", 0.0, resamples=40, resamples_bar=200)
        if out is not None:
            break
        stopped += 1
    assert stopped > 5
    again = json.loads((results / "comparison.json").read_text())
    ref = json.loads(first)
    for key in ("metrics", "comparisons", "bar_intervals", "release_bar", "g3"):
        assert again[key] == ref[key], key


# the seeds' average and the calibration rule --------------------------------------------------------------------------

def seed_files(results, ids, labels, zs, split="validation"):
    for k, z in enumerate(zs, start=1):
        order = np.random.default_rng(k).permutation(len(ids))
        write_logits(results / "logits" / f"s8_ei_{k}_evidence_effect_{split}_logits.csv", [ids[i] for i in order], labels[order], z[order])


def test_seed_average_and_its_log_probabilities(tmp_path):
    results = tmp_path / "results"
    ids = [f"evidence-{i}" for i in range(6)]
    labels = np.array([0, 1, 2, 0, 1, 2])
    rng = np.random.default_rng(0)
    zs = [rng.normal(0, 2, (6, 3)) for _ in range(3)]
    seed_files(results, ids, labels, zs)
    got_ids, got_labels, mean = CAL.seed_average("validation", results)
    assert sorted(got_ids) == sorted(ids)
    order = [ids.index(i) for i in got_ids]
    want = np.mean([softmax(z[order]) for z in zs], axis=0)
    assert np.abs(mean - want).max() < 1e-12 and np.array_equal(got_labels, labels[order])
    ids2, _, logp, source = CAL.load_logp("s8_ei", "evidence_effect", "validation", results)
    assert np.abs(logp - np.log(want)).max() < 1e-12 and source["seeds"] == list(CAL.SEEDS)
    written = np.loadtxt(results / "predictions" / "s8_ei-raw_evidence_effect_validation.csv", delimiter=",", skiprows=1,
                         usecols=(2, 3, 4))
    assert np.abs(written - want).max() < 1e-8
    write_logits(results / "logits" / "s8_ei_3_evidence_effect_validation_logits.csv", ids[:5], labels[:5], zs[2][:5])
    with pytest.raises(SystemExit, match="different prompts"):
        CAL.seed_average("validation", results)


def calibration_case(tmp_path, bias):
    results, records = tmp_path / "results", tmp_path / "records8"
    rng = np.random.default_rng(11)
    n = 1500
    z = rng.normal(0, 1.5, (n, 3))
    labels = np.array([rng.choice(3, p=p) for p in softmax(z)])
    ids = [f"evidence-{i}" for i in range(n)]
    write_jsonl(records / "evidence_effect_validation.jsonl", [{"id": i, "label": int(y), "calibrate": True} for i, y in zip(ids, labels)])
    write_logits(results / "logits" / "s8_ei_1_evidence_effect_validation_logits.csv", ids, labels, 0.7 * z + np.array([0.0, 0.0, bias]))
    return results, records


@pytest.mark.parametrize("bias,chosen", [(1.5, "temperature_intercepts"), (0.0, "temperature")])
def test_calibration_rule_on_a_planted_bias_and_on_none(tmp_path, bias, chosen):
    results, records = calibration_case(tmp_path, bias)
    data = CAL.fit(("s8_ei_1",), results, tmp_path / "public", records)
    entry = data["s8_ei_1"]["evidence_effect"]
    assert entry["chosen"] == chosen and entry["n_calibrate"] == 1500
    for name in ("s8_ei_1_evidence_effect_validation.csv", "s8_ei_1_temperature_evidence_effect_validation.csv"):
        assert (results / "predictions" / name).exists()
    assert entry["validation_calibrate_items"]["log_loss"] < entry["validation_calibrate_items"]["raw_log_loss"]
    data = CAL.fit(("s8_a3", "s8_ei_1"), results, tmp_path / "public", records)  # a system without logits is NOT BUILT, the next goes on
    assert data["s8_a3"]["not_built"].startswith("NOT BUILT: s8_a3's calibration") and data["s8_ei_1"]["evidence_effect"]["chosen"] == chosen


def test_freeze_guards_and_what_it_records(tmp_path, monkeypatch):
    results, results7 = tmp_path / "results" / "fiorillo_v0_4", tmp_path / "results" / "fiorillo_v0_3"
    fam = {"params": [0.1], "temperature": 1.1, "intercepts": [0.0, 0.0, 0.0], "out_of_fold_log_loss": {"mean": 0.5}}
    entry = {"families": {"temperature": fam, "temperature_intercepts": fam}, "chosen": "temperature"}
    (results7).mkdir(parents=True)
    (results7 / "calibration_s7.json").write_text(json.dumps({"s7_b3": {"pubmedqa_answer": entry}}))
    results.mkdir(parents=True)
    (results / "calibration_s8.json").write_text(json.dumps({"s8_ei": {"evidence_effect": entry}, "gemma4_31b": {"evidence_effect": entry}}))
    (results / "ece_limit_evidence_effect.json").write_text(json.dumps({"n": 1218, "draws": 5000, "ece_if_calibrated": {"p95": 0.061}}))
    member = results / "predictions" / "xgboost_tuned_readmit_window_validation.csv"
    write_probs(member, ["u1"], np.array([0]), np.array([[0.5, 0.3, 0.2]]))
    (results / "readmission_pool.json").write_text(json.dumps({"members": ["xgboost_tuned"], "weights": {"xgboost_tuned": 1.0}, "T": 1.0,
                                                               "validation_log_loss": 0.7,
                                                               "member_files": {"xgboost_tuned": {"file": str(member), "sha256": sha(member)}}}))
    staging = tmp_path / "staging"
    (staging / "test").mkdir(parents=True)
    (staging / "test" / "s8_ei_1-raw_evidence_effect_test.csv").write_text("id,label\n")
    with pytest.raises(SystemExit, match="a test prediction already exists"):
        CAL.freeze(results, results7, staging)
    (staging / "test" / "s8_ei_1-raw_evidence_effect_test.csv").rename(staging / "moved.csv")
    (results / "readmission_pool.json").rename(results / "pool_aside.json")
    with pytest.raises(SystemExit, match="waits for the trees stage"):  # correction 5: no freeze without the pool
        CAL.freeze(results, results7, staging)
    (results / "pool_aside.json").rename(results / "readmission_pool.json")
    rc = CAL.freeze(results, results7, staging)
    assert rc["ece_limits"]["evidence_effect"]["limit"] == 0.061 and rc["specialists"]["readmit_window"]["weights"] == {"xgboost_tuned": 1.0}
    assert rc["sha256"][CAL.rel(member)] == sha(member) and rc["sha256"][CAL.rel(results / "calibration_s8.json")] == sha(results / "calibration_s8.json")
    with pytest.raises(SystemExit, match="written once"):
        CAL.freeze(results, results7, staging)
    (results / "testpreds_log.json").write_text("{}")
    (results / "calibration_s8.json").write_text("{}")
    with pytest.raises(SystemExit, match="changed after the freeze"):
        CAL.apply(("s8_ei",), results, results7)


# the readmission pool and the trees -----------------------------------------------------------------------------------

def test_pool_recovers_a_planted_member_and_fits_t():
    import trees_session8 as T8
    rng = np.random.default_rng(5)
    n = 1200
    p_true = softmax(rng.normal(0, 1.5, (n, 3)))
    y = np.array([rng.choice(3, p=p) for p in p_true])
    logs = np.stack([np.log(rng.dirichlet([1, 1, 1], size=n)), np.log(p_true), np.log(rng.dirichlet([1, 1, 1], size=n))])
    kept = T8.fit_pool(logs, y)
    assert kept["weights"] == [0.0, 1.0, 0.0] and abs(kept["T"] - 1.0) < 0.15
    grid = T8.weight_grid(5)
    assert len(grid) == 1001 and np.allclose(grid.sum(axis=1), 1.0) and len(T8.weight_grid(3)) == 66
    w = np.array([0.3, 0.5, 0.2])
    ts = np.exp(np.linspace(np.log(0.1), np.log(10.0), 3001))
    losses = [np.mean(-np.log(T8.pool_probs(logs, w, t)[np.arange(n), y])) for t in ts]
    loss, beta = T8._fit_chunk(logs, y, w[None, :])
    assert abs(1 / beta[0] - ts[int(np.argmin(losses))]) < 0.01 and loss[0] <= min(losses) + 1e-9


def uci_rows(split, n, seed):
    from baselines import UCI_CATEGORICAL, UCI_NUMERIC
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        state = {c: int(rng.integers(0, 20)) for c in UCI_NUMERIC}
        state["time_in_hospital"], state["number_inpatient"] = int(rng.integers(1, 14)), int(rng.integers(0, 4))
        for c in UCI_CATEGORICAL:
            state[c] = str(rng.choice([f"{c} a", f"{c} b"]))
        state["diag_1"] = str(rng.choice(["250 (d)", "428 (c)"] + ([] if split == "train" else ["999 (unseen)"])))
        state["medications"] = {"insulin": str(rng.choice(["No", "Steady", "Up"]))}
        score = 0.45 * state["number_inpatient"] + 0.08 * state["time_in_hospital"] + rng.normal(0.0, 0.8)
        label = 2 if score > 1.6 else (1 if score > 0.9 else 0)
        rows.append({"id": f"uci-{split}-{i}", "qid": "readmit_window", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": state})
    return rows


def test_trees_search_resumes_validation_pool_and_guards(tmp_path, monkeypatch):
    pytest.importorskip("xgboost")
    pytest.importorskip("catboost")
    import trees_session8 as T8
    monkeypatch.setattr(T8, "MAX_TREES", 60)
    monkeypatch.setattr(T8, "EARLY_STOP", 10)
    records = tmp_path / "records"
    for split, n, seed in (("train", 200, 1), ("validation", 80, 2), ("test", 80, 3)):
        write_jsonl(records / f"readmit_window_{split}.jsonl", uci_rows(split, n, seed))
    base = tmp_path / "results"
    whole = T8.search(base / "whole" / "fiorillo_v0_4", records, threads=1, n_settings=1)
    parts_dir = base / "parts" / "fiorillo_v0_4"
    for calls in range(1, 30):
        progress = T8.search(parts_dir, records, threads=1, max_seconds=0.0, n_settings=1)  # one fit a call
        if all(progress["members"].get(m, {}).get("complete") for m in T8.SEARCHED):
            break
    assert calls >= 2 * T8.FOLDS
    for m in T8.SEARCHED:
        for key, fit in whole["members"][m]["fits"].items():
            assert progress["members"][m]["fits"][key]["best_iteration"] == fit["best_iteration"]
            assert progress["members"][m]["fits"][key]["log_loss"] == pytest.approx(fit["log_loss"], rel=1e-9)
    grid, n_combos = T8.settings("xgboost_tuned")
    assert n_combos == 1620 and len(grid) == 50 and len({s["combination"] for s in grid}) == 50
    grid, n_combos = T8.settings("catboost_tuned")
    assert n_combos == 144 and len(grid) == 30
    out = T8.validation(parts_dir, records, tmp_path / "models", threads=1, n_settings=1)
    for m in T8.SEARCHED:
        assert out["members"][m]["validation"]["rows"] == 80 and (parts_dir / "predictions" / f"{m}_readmit_window_validation.csv").exists()
    rng = np.random.default_rng(9)
    for split in ("validation", "test"):
        rows = [json.loads(line) for line in open(records / f"readmit_window_{split}.jsonl")]
        ids, y = [r["id"] for r in rows], np.array([r["label"] for r in rows])
        for m in ("lightgbm_tuned", "arm_laya", "s7_b3"):
            write_probs(T8.member_file(m, split, parts_dir), ids, y, rng.dirichlet([1, 1, 1], size=len(ids)))
    pool = T8.pool(parts_dir, processes=1, repeats=2)
    assert pool["members"] == list(T8.MEMBERS) and abs(sum(pool["weights"].values()) - 1.0) < 1e-9
    assert pool["out_of_fold_log_loss"]["repeats"] == 2 and (parts_dir / "predictions" / "readmission_pool_readmit_window_validation.csv").exists()
    for fn in (lambda: T8.testpreds(parts_dir, records, tmp_path / "models", threads=1), lambda: T8.pool_test(parts_dir)):
        with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
            fn()
    rc = {"sha256": {T8.rel(parts_dir / "readmission_pool.json"): sha(parts_dir / "readmission_pool.json"),
                     **{v["file"]: v["sha256"] for v in pool["member_files"].values()}}}
    (parts_dir / "release_candidate.json").write_text(json.dumps(rc))
    with pytest.raises(SystemExit, match="testpreds_log.json does not exist"):
        T8.pool_test(parts_dir)
    (parts_dir / "testpreds_log.json").write_text(json.dumps({"release_candidate_sha256": "x"}))
    log = T8.testpreds(parts_dir, records, tmp_path / "models", threads=1)
    assert all(v["validation_reproduced_max_abs_difference"] <= 1e-8 for v in log.values())
    assert "trees" in json.loads((parts_dir / "testpreds_log.json").read_text())
    got = T8.pool_test(parts_dir)
    assert got["rows"] == 80 and json.loads((parts_dir / "trees_test.json").read_text())["members"]["s7_b3"]["rows"] == 80
    member = T8.member_file("arm_laya", "validation", parts_dir)
    text = member.read_text()
    member.write_text(text[:-2] + ("1" if text[-2] != "1" else "2") + text[-1])  # one digit changed
    with pytest.raises(SystemExit, match="changed after the freeze"):
        T8.pool_test(parts_dir)


# the hand-off -----------------------------------------------------------------------------------------------------------

def test_handoff_against_a_hand_calculation():
    ids = ["e5", "e1", "e3", "e2", "e4"]
    labels = np.array([0, 1, 2, 0, 1])
    fio = np.array([[0.9, 0.05, 0.05],    # e5: sure, right
                    [0.5, 0.4, 0.1],      # e1: 0.5, wrong
                    [0.2, 0.2, 0.6],      # e3: 0.6, right
                    [0.4, 0.35, 0.25],    # e2: 0.4, the least sure, right
                    [0.25, 0.5, 0.25]])   # e4: 0.5, right; ties with e1, which comes first by ID
    big = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]])  # always answers 1
    rank = H.id_ranks(ids)
    est = H.point(rank, labels, fio, big)
    assert est["rate_0"]["handed_off"] == 0 and est["rate_0"]["accuracy"] == pytest.approx(4 / 5)
    # 20 percent: floor(0.2 x 5) = 1, the least confident, e2 (label 0): the large model answers 1, wrong
    assert est["rate_20"]["handed_off"] == 1 and est["rate_20"]["accuracy"] == pytest.approx(3 / 5)
    assert est["rate_20"]["handed_off_fiorillo_accuracy"] == 1.0 and est["rate_20"]["handed_off_large_model_accuracy"] == 0.0
    # 50 percent: floor(2.5) = 2: e2, then e1 (0.5, before e4 by ID; label 1: the large model is right)
    assert est["rate_50"]["handed_off"] == 2 and est["rate_50"]["accuracy"] == pytest.approx(4 / 5)
    assert est["rate_50"]["handed_off_fiorillo_accuracy"] == 0.5 and est["rate_50"]["handed_off_large_model_accuracy"] == 0.5
    combined, handed = H.combine(rank, labels, fio, big, 0.5)
    assert handed.tolist() == [False, True, False, True, False]
    ll = -np.mean(np.log([0.9, 0.8, 0.6, 0.1, 0.5]))
    assert est["rate_50"]["log_loss"] == pytest.approx(ll)
    assert est["large_model_alone"]["accuracy"] == pytest.approx(2 / 5)
    out = H.analyze(ids, labels, fio, big, clusters=["a", "a", "b", "c", "c"], resamples=30)
    assert out["rate_20"]["accuracy"]["estimate"] == pytest.approx(3 / 5) and out["rate_20"]["handed_off"] == 1
    assert out["rate_20"]["accuracy"]["low"] <= out["rate_20"]["accuracy"]["high"]
    costs = H.costs(0.3, 2.0, 3.6)
    assert costs["rate_0"] == pytest.approx(1000 * 0.3 / 3600 * 3.6) and costs["rate_50"] == pytest.approx(1000 * (0.3 + 1.0) / 3600 * 3.6)
    assert H.costs(None, 2.0, 3.6)["rate_10"] is None


# the large model --------------------------------------------------------------------------------------------------------

LETTER_IDS = {"A": [65, 1065], "B": [66, 1066], "C": [67]}  # made up: C's spaced variant is not one token


def test_option_probabilities_from_made_up_top_20():
    top = {65: math.log(0.6), 1065: math.log(0.1), 66: math.log(0.15), 67: math.log(0.05), 999: math.log(0.1)}
    probs, mass, per_letter, missing = L.option_probabilities(top, LETTER_IDS)
    assert missing == [] and mass == pytest.approx(0.9) and per_letter == pytest.approx([0.7, 0.15, 0.05])
    assert probs == pytest.approx([0.7 / 0.9, 0.15 / 0.9, 0.05 / 0.9])
    probs, mass, _, missing = L.option_probabilities({65: math.log(0.9), 66: math.log(0.05)}, LETTER_IDS)
    assert probs is None and missing == ["C"]  # sent the second way
    probs, mass, per_letter = L.from_prompt_logprobs({65: math.log(0.5), 1065: math.log(0.1), 66: math.log(0.2), 1066: math.log(0.1),
                                                      67: math.log(0.05)}, LETTER_IDS)
    assert mass == pytest.approx(0.95) and probs == pytest.approx([0.6 / 0.95, 0.3 / 0.95, 0.05 / 0.95])

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            table = {"A": [65], " A": [1065], "B": [66], " B": [1066], "C": [67], " C": [1, 67]}
            return {"input_ids": table[text]}

    ids, left_out = L.letter_ids(Tok())
    assert ids == LETTER_IDS and left_out == [" C"]


def test_large_model_files_after_both_passes(tmp_path):
    items, results = tmp_path / "items", tmp_path / "results"
    first = [{"id": "e1", "label": 0, "prompt_tokens": 100, "method": "top20", "probs": [0.7, 0.2, 0.1], "letter_mass": 0.9,
              "per_letter": [0.63, 0.18, 0.09], "missing": [], "top_token": "A"},
             {"id": "e2", "label": 2, "prompt_tokens": 120, "method": "pending", "probs": None, "letter_mass": 0.4,
              "per_letter": [0.4, 0.0, 0.0], "missing": ["B", "C"], "top_token": "A"}]
    L.write_jsonl(items / "first_evidence_effect_validation.jsonl", first)
    L.save_json(items / "timing_first.json", {"vllm": "made up", "passes": {"evidence_effect_validation": {"first_items": 2, "first_seconds": 4.0,
                                                                                                         "prompt_tokens_mean": 110.0}}})
    with pytest.raises(SystemExit, match="no answer yet"):
        L.finish(["evidence_effect"], ["validation"], items, results)
    second = [{**first[1], "method": "prompt", "probs": [0.5, 0.3, 0.2], "letter_mass": 0.8, "per_letter": [0.4, 0.24, 0.16]}]
    L.write_jsonl(items / "second_evidence_effect_validation.jsonl", second)
    timing = json.loads((items / "timing_first.json").read_text())
    timing["passes"]["evidence_effect_validation"].update({"second_items": 1, "second_seconds": 2.0})
    L.save_json(items / "timing_first.json", timing)
    letters = L.finish(["evidence_effect"], ["validation"], items, results)
    v = letters["evidence_effect"]["validation"]
    assert v["answered_second_way"] == 1 and v["letters_summed_probability"]["median"] == pytest.approx(0.85)
    assert v["letters_summed_probability"]["share_below_0_5"] == 0.0
    rows = list(csv.reader(open(results / "large_model" / "gemma4_31b-raw_evidence_effect_validation.csv")))
    assert rows[0] == ["id", "label", "p0", "p1", "p2"] and rows[2] == ["e2", "2", "0.5", "0.3", "0.2"]
    t = json.loads((results / "large_model" / "gemma4_31b_timing.json").read_text())["evidence_effect"]["validation"]
    assert t["wall_seconds"] == 6.0 and t["gpu_seconds_per_item"] == 3.0 and t["prompt_tokens_per_item"] == 110.0


def test_plain_text_prompts_for_ei_and_pubmedqa():
    from build_items import load_questions
    q8 = load_questions(ROOT / "src" / "questions" / "session8.yaml")["evidence_effect"]
    state = {"intervention": "drug X", "comparator": "placebo", "outcome": "pain", "article": "TITLE: \nT\n\nABSTRACT: a\n\nBODY.RESULTS:\nr"}
    assert L.user_message("evidence_effect", state, q8) == (
        "Intervention: drug X\nComparator: placebo\nOutcome: pain\n\nArticle:\nTITLE: \nT\n\nABSTRACT: a\n\nBODY.RESULTS:\nr\n\n"
        "Based only on this article, how did the outcome differ with the intervention compared with the comparator?\n\n"
        "A. significantly increased: the outcome was significantly higher with the intervention than with the comparator\n"
        "B. significantly decreased: the outcome was significantly lower with the intervention than with the comparator\n"
        "C. no significant difference: the article reports no significant difference in the outcome between the intervention and the comparator\n"
        "Answer with the letter of one option only.")
    q4 = load_questions(ROOT / "src" / "questions" / "public_v0.yaml")["pubmedqa_answer"]
    state = {"question": "Does Y work?", "contexts": [{"section": "BACKGROUND", "text": "b"}, {"section": "RESULTS", "text": "r"}]}
    assert L.user_message("pubmedqa_answer", state, q4) == (
        "Question: Does Y work?\n\nAbstract:\nBACKGROUND: b\n\nRESULTS: r\n\n"
        "Based only on this abstract, what is the answer to the research question?\n\n"
        "A. yes: the reported results support yes\nB. no: the reported results support no\nC. maybe: the reported results are inconclusive\n"
        "Answer with the letter of one option only.")
    assert "{" not in L.user_message("evidence_effect", {"intervention": "a", "comparator": "b", "outcome": "c", "article": "d"}, q8)


# power ------------------------------------------------------------------------------------------------------------------

def test_power_plan_and_validation_against_hand_calculations(tmp_path):
    plan = PW.plan()
    case = next(c for c in plan["cases"] if c["s"] == 0.8 and c["D"] == 1.5)
    se = 0.8 * math.sqrt(1.5 / 1218)
    assert case["se"] == pytest.approx(se) and case["pass_line"] == pytest.approx(-1.959963985 * se, rel=1e-8)
    assert case["real_gain_for_80_percent"] == pytest.approx((1.959963985 + 0.841621234) * se, rel=1e-8)
    at = next(r for r in case["chance_by_real_gain"] if r["real_gain"] == 0.08)
    assert at["chance"] == pytest.approx(0.5 * (1 + math.erf((0.08 / se - 1.959963985) / math.sqrt(2))), rel=1e-7)
    assert len(plan["cases"]) == 9 and len(case["chance_by_real_gain"]) == 31
    records, ids, labels, clusters = [], [], [], []
    rng = np.random.default_rng(2)
    for i in range(12):
        ids.append(f"e{i}")
        labels.append(i % 3)
        clusters.append(f"PMC{i // 3}")
        records.append({"id": ids[-1], "label": labels[-1], "group": {"pmcid": clusters[-1]}})
    write_jsonl(tmp_path / "rec.jsonl", records)
    a, b = softmax(rng.normal(0, 1, (12, 3))), softmax(rng.normal(0, 1, (12, 3)))
    y = np.array(labels)
    write_probs(tmp_path / "a.csv", ids, y, a)
    write_probs(tmp_path / "b.csv", ids, y, b)
    got = PW.validation(tmp_path / "rec.jsonl", tmp_path / "a.csv", tmp_path / "b.csv")
    pa, pb = np.array([[float(f"{x:.9g}") for x in r] for r in a]), np.array([[float(f"{x:.9g}") for x in r] for r in b])
    diff = -np.log(pa[np.arange(12), y]) + np.log(pb[np.arange(12), y])
    assert got["s"] == pytest.approx(np.std(diff, ddof=1)) and got["validation_mean_difference"] == pytest.approx(diff.mean())
    rng = np.random.default_rng(PW.SEED)
    item = [diff[rng.integers(0, 12, 12)].mean() for _ in range(PW.RESAMPLES)]
    rng = np.random.default_rng(PW.SEED)
    rows_of = {c: [i for i in range(12) if clusters[i] == c] for c in sorted(set(clusters))}
    order = sorted(rows_of)
    cluster = [diff[[i for j in rng.integers(0, 4, 4) for i in rows_of[order[j]]]].mean() for _ in range(PW.RESAMPLES)]
    assert got["D"] == pytest.approx(np.var(cluster, ddof=1) / np.var(item, ddof=1))
    assert got["se_test"] == pytest.approx(got["s"] * math.sqrt(got["D"] / 1218))


# pilot-s8, predict's run summary and the labeler's rebuild --------------------------------------------------------------

def attempt(name, rate, peak, oom=False):
    return {"setting": name, "micro_batch": 1, "grad_accum": 32, "gradient_checkpointing": name != "a", "train_items_per_second": rate,
            "peak_gpu_memory_gib": peak, "out_of_memory": oom}


def test_pilot_s8_choice_on_made_up_measurements():
    import train_qwen_head as Q
    assert Q.choose_s8([attempt("a", 2.0, 40.0), attempt("b", 1.5, 30.0)])["setting"] == "a"
    assert Q.choose_s8([attempt("a", 2.0, 61.0), attempt("b", 1.5, 30.0)])["setting"] == "b"   # 61 + 16 = 77 > 76
    assert Q.choose_s8([attempt("a", 2.0, 60.0), attempt("b", 2.5, 60.0)])["setting"] == "b"   # 60 + 16 = 76: both pass
    assert Q.choose_s8([attempt("a", None, None, oom=True), attempt("b", 1.5, 30.0)])["setting"] == "b"
    assert Q.choose_s8([attempt("a", None, 79.0, oom=True), attempt("b", 1.5, 62.0)]) is None
    assert Q.choose_s8([attempt("a", 2.0, 40.0), attempt("b", 2.0, 30.0)])["setting"] == "a"   # an exact tie: the first listed


@pytest.mark.skipif(not (LAYA_TOKENIZER / "tokenizer.json").exists(), reason="models/laya-base/tokenizer not present")
def test_pilot_s8_and_predict_run_summary_on_the_stand_in(tmp_path, monkeypatch):
    import torch
    import train_qwen_head as Q
    from build_items import load_questions
    from qwen_items import build_qwen_items
    from test_qwen_head import made_up_records, tiny_base

    folder, tok = tiny_base(tmp_path / "tiny_qwen")
    questions = load_questions()
    train, _ = build_qwen_items(made_up_records(24, 1), questions, tok, 256)
    val, _ = build_qwen_items(made_up_records(6, 2), questions, tok, 256)
    torch.save(train, tmp_path / "train.pt")
    torch.save(val, tmp_path / "val.pt")
    monkeypatch.setattr(Q, "S8_SETTINGS", (("a", 1, 4, False), ("b", 2, 2, True)))
    out = Q.main(["pilot-s8", "--base-dir", str(folder), "--train", str(tmp_path / "train.pt"), "--val", str(tmp_path / "val.pt"),
                  "--steps", "4", "--time-from", "1", "--device", "cpu", "--out-json", str(tmp_path / "pilot.json")])
    saved = json.loads((tmp_path / "pilot.json").read_text())
    assert [s["setting"] for s in saved["settings"]] == ["a", "b"] and all(s["steps"] == 4 for s in saved["settings"])
    assert all(s["train_items_per_second"] > 0 and s["passes"] for s in saved["settings"])
    assert saved["chosen"]["setting"] == Q.choose_s8(saved["settings"])["setting"] and not saved["stop_and_ask"]
    assert saved["validation_items"] == len(val) and saved["validation_items_per_second"] > 0 and out["validation_batch_size"] == 4
    phase = tmp_path / "phase"
    model, _ = Q.build_model(Q.load_base(folder, torch.float32), 20260927)
    args = type("A", (), {"lora_r": 16, "lora_alpha": 32.0, "lora_dropout": 0.05, "max_len": 256, "seed": 20260927})()
    Q.save_phase(phase, model, args, folder, {"system": "stand_in", "keep": "last"})
    records = tmp_path / "records"
    write_jsonl(records / "pubmedqa_answer_validation.jsonl", [r for r in made_up_records(5, 3) if r["qid"] == "pubmedqa_answer"])
    summary_path = tmp_path / "summaries" / "stand_in_validation_run_summary.json"
    Q.main(["predict", "--base-dir", str(folder), "--phase-dir", str(phase), "--system", "stand_in", "--records-dir", str(records),
            "--qids", "pubmedqa_answer", "--splits", "validation", "--out", str(tmp_path / "preds"), "--device", "cpu",
            "--batch-size", "2", "--run-summary", str(summary_path)])
    summary = json.loads(summary_path.read_text())
    t = summary["timing_validation"]
    assert t["items"] == 5 and t["forward_seconds"] > 0 and t["seconds_per_item"] == pytest.approx(t["forward_seconds"] / 5)
    assert t["items_per_second"] == pytest.approx(5 / t["forward_seconds"]) and summary["batch_size"] == 2
    assert (tmp_path / "preds" / "stand_in-raw_pubmedqa_answer_validation.csv").exists()


def test_label_pqa_u_rebuild_against_a_made_up_sha256(tmp_path):
    import label_pqa_u as LAB
    results, records = tmp_path / "results", tmp_path / "records"
    unlabeled = [{"id": f"pubmedqa-{i}", "state": {"question": f"q{i}", "contexts": [{"section": "RESULTS", "text": f"t{i}"}]}}
                 for i in (30, 4, 1000, 7)]
    write_jsonl(records / "pqa_u_unlabeled.jsonl", unlabeled)
    results.mkdir(parents=True)
    with open(results / "phase2_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "probability"])
        w.writerows([["pubmedqa-4", 0, "0.9"], ["pubmedqa-30", 2, "0.8"], ["pubmedqa-1000", 1, "0.95"]])
    LAB.write_phase2(records, {"pubmedqa-4": 0, "pubmedqa-30": 2, "pubmedqa-1000": 1}, tmp_path / "first.jsonl")
    digest = sha(tmp_path / "first.jsonl")
    lines = [json.loads(line) for line in open(tmp_path / "first.jsonl")]
    assert [r["id"] for r in lines] == ["pubmedqa-4", "pubmedqa-30", "pubmedqa-1000"]  # PubMed ID order
    assert lines[1]["label"] == 2 and lines[1]["target"] == [0.0, 0.0, 1.0] and lines[0]["group"] == {"source": "pqa_unlabeled_labeler"}
    info = {"phase2_built": True, "phase2": {"file": "pubmedqa_phase2_train.jsonl", "records": 3, "sha256": digest,
                                             "labels_files": ["phase2_labels.csv"]}}
    (results / "labeler.json").write_text(json.dumps(info))
    got = LAB.rebuild(results, records)
    assert got == {"records": 3, "sha256": digest} and sha(records / "pubmedqa_phase2_train.jsonl") == digest
    with pytest.raises(SystemExit, match="NOT BUILT: arm A, phase 2's records differ from session 7's"):
        LAB.main(["rebuild", "--results", str(results), "--records-dir", str(records), "--expected-sha256", "0" * 64])
    (results / "labeler.json").write_text(json.dumps({"phase2_built": False}))
    with pytest.raises(SystemExit, match="NOT BUILT: arm A"):
        LAB.rebuild(results, records)


# the pod tools -----------------------------------------------------------------------------------------------------------

def test_upload_accepts_session8_stages_and_the_pod_script():
    stages = ("prepare", "prepare_a", "pilot", "ei", "arm_a", "trees", "prepare_large", "large_validation", "calibrate", "freeze",
              "testpreds", "package")
    assert all(s in U.DRIVER_COMMANDS for s in stages)
    assert U.log_dir("/root/eddm_s8/code/scripts/pod_session8.sh") == "/root/eddm_s8/logs"
    script = ROOT / "scripts" / "pod_session8.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    text = script.read_text()
    for s in stages:
        assert f"\n  {s})\n" in text, s
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert out.returncode == 1 and "Stages:" in out.stdout
    assert "uploads_manifest.json" in text and "memory.used" in text and "pgrep -f \"trees_session8.py\"" in text  # correction 5
    assert "nice -n 10 python src/eval/trees_session8.py" in text and "import vllm._C" in text


def test_upload_records_a_manifest_on_the_pod(tmp_path, monkeypatch):
    """The uploads manifest: {roots: {remote root: {relative path: sha256}}}, merged on the pod after each call."""
    calls = []

    class FakePod:
        def python(self, code, timeout=600):
            calls.append(code)
            return json.dumps({"manifest": "x", "root": "r", "files": 2})

    out = U.record_manifest(FakePod(), "/root/eddm_s8/code/scripts/pod_session8.sh", "/root/eddm_s8/code/", {"a.py": "1" * 64, "b/c.json": "2" * 64})
    assert json.loads(out)["files"] == 2 and len(calls) == 1
    code = calls[0]
    assert "'/root/eddm_s8/uploads_manifest.json'" in code and "'/root/eddm_s8/code'" in code and '"a.py"' in code
    assert U.record_manifest(FakePod(), "/root/eddm_s8/code/scripts/pod_session8.sh", "/root/eddm_s8/code", {}) is None
    # the merge, run here as the pod would run it
    scope = {}
    exec(U.MANIFEST.replace("PATH", repr(str(tmp_path / "m.json"))).replace("ROOT", repr("/root/eddm_s8/code"))
         .replace("ENTRIES", repr(json.dumps({"a.py": "1" * 64}))), scope)
    exec(U.MANIFEST.replace("PATH", repr(str(tmp_path / "m.json"))).replace("ROOT", repr("/root/eddm_s8/code"))
         .replace("ENTRIES", repr(json.dumps({"b.py": "2" * 64}))), scope)
    exec(U.MANIFEST.replace("PATH", repr(str(tmp_path / "m.json"))).replace("ROOT", repr("/root/eddm_s8/models/s7_b3"))
         .replace("ENTRIES", repr(json.dumps({"adapter.safetensors": "3" * 64}))), scope)
    got = json.loads((tmp_path / "m.json").read_text())
    assert got == {"roots": {"/root/eddm_s8/code": {"a.py": "1" * 64, "b.py": "2" * 64}, "/root/eddm_s8/models/s7_b3": {"adapter.safetensors": "3" * 64}}}


def test_large_model_engine_settings_and_the_top_1000_pass(tmp_path, monkeypatch):
    """Correction 5: the first engine allows 1,000 log-probabilities; an item missing a letter in the top 20 is asked again
    with the top 1,000 and, if still missing, left for the second process. Correction 9: every modality is limited to 0 in
    one engine (vLLM 0.30.0 accepts a limit for any modality name), and the outputs are checked to come back in order.
    Correction 12: each process shuts its engine down; the second process answers the item left pending from the prompt
    log-probabilities of its prompt followed by each letter variant."""
    made, closed = [], []
    second_probs = {65: 0.6, 165: 0.1, 66: 0.2, 166: 0.05, 67: 0.04, 167: 0.01}

    class FakeOut:
        def __init__(self, top, prompt_token_ids):
            self.outputs = [type("O", (), {"logprobs": [top]})()]
            self.prompt_token_ids = prompt_token_ids

    class FakeCore:
        def shutdown(self, timeout=None):
            closed.append(timeout)

    class FakeLLM:
        def __init__(self, **kwargs):
            made.append(kwargs)
            self.llm_engine = type("E", (), {"engine_core": FakeCore()})()

        def generate(self, requests, params, use_tqdm=False):
            outs = []
            for r in requests:
                n = len(r["prompt_token_ids"])
                if getattr(params, "prompt_logprobs", None) is not None:  # the second process: the appended letter's entry
                    v = r["prompt_token_ids"][-1]
                    outs.append(type("P", (), {"prompt_token_ids": r["prompt_token_ids"],
                                               "prompt_logprobs": [None] * (n - 1) + [{v: math.log(second_probs[v])}]})())
                    continue
                if params.logprobs == 20:
                    top = {65: math.log(0.5), 66: math.log(0.3)} if n % 2 == 0 else {65: math.log(0.5), 66: math.log(0.3), 67: math.log(0.1)}
                else:  # the top 1,000 finds C for the even prompt of length 4, not for the one of length 6
                    top = {65: math.log(0.5), 66: math.log(0.3), 67: math.log(0.05)} if n == 4 else {65: math.log(0.5), 66: math.log(0.3)}
                outs.append(FakeOut(top, r["prompt_token_ids"]))
            return outs

    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.logprobs = kw.get("logprobs")

    class Tok:
        bos_token_id = 2

        def __call__(self, text, add_special_tokens=False):
            table = {"A": [65], " A": [165], "B": [66], " B": [166], "C": [67], " C": [167]}
            if text in table:
                return {"input_ids": table[text]}
            return {"input_ids": [2] + [9] * (len(text) - 1)}

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
            return "x" * len(messages[0]["content"]) + "<|channel>thought\n<channel|>"

        def convert_ids_to_tokens(self, i):
            return str(i)

    import types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM, fake_vllm.SamplingParams, fake_vllm.__version__ = FakeLLM, Params, "fake"
    fake_tr = types.ModuleType("transformers")
    fake_tr.AutoTokenizer = type("AT", (), {"from_pretrained": staticmethod(lambda d: Tok())})
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)
    monkeypatch.setattr(L, "wait_for_gpu", lambda *a, **k: 0)
    items = tmp_path / "items"
    tok = Tok()
    # prompts whose templated length is 4 + 28 = 32 (even), 33 (odd) and 6 + 28 = 34 (even) tokens: text length n - 28 - 1
    L.write_jsonl(items / "prompts_evidence_effect_validation.jsonl",
                  [{"id": "e1", "label": 0, "text": "t" * 5}, {"id": "e2", "label": 1, "text": "t" * 6}, {"id": "e3", "label": 2, "text": "t" * 7}])
    monkeypatch.setattr(L, "templated_ids", lambda tokenizer, text: [2] + [9] * ({5: 3, 6: 4, 7: 5}[len(text)]))  # lengths 4, 5 and 6
    pending = L.answer(["evidence_effect"], ["validation"], tmp_path / "model", items, tmp_path / "results")
    assert made[0]["max_logprobs"] == 1000 and made[0]["enforce_eager"] is True and made[0]["limit_mm_per_prompt"] == {"image": 0, "audio": 0, "video": 0} and len(made) == 1
    first = {r["id"]: r for r in L.read_jsonl(items / "first_evidence_effect_validation.jsonl")}
    assert first["e2"]["method"] == "top20" and first["e1"]["method"] == "top1000" and first["e3"]["method"] == "pending"
    assert pending == 1 and first["e1"]["probs"] == pytest.approx([0.5 / 0.85, 0.3 / 0.85, 0.05 / 0.85])
    timing = json.loads((items / "timing_first.json").read_text())
    assert timing["passes"]["evidence_effect_validation"]["top1000_items"] == 2 and timing["max_logprobs"] == 1000
    assert closed == [30]
    assert L.second(["evidence_effect"], ["validation"], tmp_path / "model", items, tmp_path / "results") == 1
    assert len(made) == 2 and made[1]["max_num_seqs"] == 1 and made[1]["max_num_batched_tokens"] == 1024
    assert made[1]["gpu_memory_utilization"] == 0.90 and made[1]["max_logprobs"] == 20 and closed == [30, 30]
    out = tmp_path / "results" / "large_model"
    rows = {r[0]: r for r in csv.reader(open(out / "gemma4_31b-raw_evidence_effect_validation.csv"))}
    assert [float(x) for x in rows["e3"][2:]] == pytest.approx([0.7, 0.25, 0.05]) and len(rows) == 4
    letters = json.loads((out / "gemma4_31b_letters.json").read_text())
    assert letters["evidence_effect"]["validation"]["by_method"] == {"top20": 1, "top1000": 1, "prompt": 1}


def test_large_model_engine_shut_down_when_a_pass_fails(tmp_path, monkeypatch):
    """Correction 12: the engine is shut down on every way out of answer and second, also when a pass stops with an
    error; the error then goes on up."""
    closed = []

    class FakeCore:
        def shutdown(self, timeout=None):
            closed.append(timeout)

    class FakeLLM:
        def __init__(self, **kwargs):
            self.llm_engine = type("E", (), {"engine_core": FakeCore()})()

        def generate(self, requests, params, use_tqdm=False):
            raise RuntimeError("the engine died")

    class Params:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": {"A": [65], " A": [165], "B": [66], " B": [166], "C": [67], " C": [167]}[text]}

    import types
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM, fake_vllm.SamplingParams, fake_vllm.__version__ = FakeLLM, Params, "fake"
    fake_tr = types.ModuleType("transformers")
    fake_tr.AutoTokenizer = type("AT", (), {"from_pretrained": staticmethod(lambda d: Tok())})
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "transformers", fake_tr)
    monkeypatch.setattr(L, "wait_for_gpu", lambda *a, **k: 0)
    monkeypatch.setattr(L, "templated_ids", lambda tokenizer, text: [2, 9, 9])
    items = tmp_path / "items"
    L.write_jsonl(items / "prompts_evidence_effect_validation.jsonl", [{"id": "e1", "label": 0, "text": "t"}])
    with pytest.raises(RuntimeError, match="the engine died"):
        L.answer(["evidence_effect"], ["validation"], tmp_path / "model", items, tmp_path / "results")
    assert closed == [30] and not (items / "first_evidence_effect_validation.jsonl").exists()
    L.write_jsonl(items / "first_evidence_effect_validation.jsonl", [{"id": "e1", "label": 0, "method": "pending"}])
    L.save_json(items / "timing_first.json", {"passes": {"evidence_effect_validation": {"first_seconds": 1.0, "prompt_tokens_mean": 3.0}}})
    with pytest.raises(RuntimeError, match="the engine died"):
        L.second(["evidence_effect"], ["validation"], tmp_path / "model", items, tmp_path / "results")
    assert closed == [30, 30]


def test_large_model_close_llm_reports_a_failed_shutdown(capsys):
    """Correction 12: a shutdown that fails is printed, not raised, so the process goes on to exit."""
    class Core:
        def shutdown(self, timeout=None):
            raise OSError("no engine")

    L.close_llm(type("F", (), {"llm_engine": type("E", (), {"engine_core": Core()})()})())
    assert "vLLM engine shutdown failed" in capsys.readouterr().out


def test_large_model_close_llm_stops_a_child_left_running():
    """Correction 12: when vLLM's shutdown fails, a child process still running is stopped, so the process can exit."""
    import multiprocessing
    import time as _time

    class Core:
        def shutdown(self, timeout=None):
            raise OSError("no engine")

    child = multiprocessing.Process(target=_time.sleep, args=(60,))
    child.start()
    L.close_llm(type("F", (), {"llm_engine": type("E", (), {"engine_core": Core()})()})(), timeout=5)
    assert not child.is_alive() and not multiprocessing.active_children()

"""Tests for session 7's training options, rules and analyses (docs/prompt_session7.md, step 4), on made-up inputs.

- train_laya.py --keep last on a tiny Laya-format stand-in keeps the final weights and saves no best checkpoint, and
  --logits-dir writes logits whose softmax is the raw predictions, which run_laya_model.py --logits-dir reproduces;
- the calibration rule chooses temperature_intercepts when a bias is planted and temperature when it is not, and writes
  both families' predictions;
- the candidate rule, its NOT BUILT case, and the freeze's guards;
- trees_session7.py: search and validation on made-up rows, a search stopped part way that resumes to the same fits, and
  the test subcommand refusing to run without release_candidate.json and testpreds_log.json;
- the blend: T at a fixed w against a direct search, and w = 1 recovered when p_T is noise;
- the hand-off numbers against a hand calculation;
- criterion 4's two routes on made-up predictions against a hand calculation, and compare_session7.py end to end, with a
  run stopped after every block resuming to the same comparison;
- power_session7.py against retry_check.json's own pass lines and gains;
- upload_to_pod.py's resume logic on a local fake of the Jupyter calls;
- mac_training_check.py after correction 6: one process per try with the memory cap, the ladder after out of memory,
  after a stopped try and after any other error, a try's process stopped when no step finishes in time, and a run of
  one arm keeping the other arm's result."""
import base64
import contextlib
import csv
import io
import json
import math
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC / "eval"))
sys.path.insert(0, str(SRC / "tasks"))
sys.path.insert(0, str(SRC / "train"))
sys.path.insert(0, str(SRC / "tools"))

import calibrate_session7 as CAL  # noqa: E402
import compare_session7 as CMP  # noqa: E402
import handoff_session7 as H  # noqa: E402
import power_session7 as PW  # noqa: E402
import select_session7 as SEL  # noqa: E402
import trees_session7 as TR  # noqa: E402
import upload_to_pod as U  # noqa: E402
import mac_training_check as MAC  # noqa: E402

LAYA_TOKENIZERS = [ROOT / "models" / "tokenizers" / "laya", ROOT / "models" / "laya-base" / "tokenizer"]
LAYA_TOKENIZER = next((p for p in LAYA_TOKENIZERS if (p / "tokenizer.json").exists()), None)
RETRY = ROOT / "results" / "fiorillo_v0_2" / "retry_check.json"


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


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# train_laya.py --keep last and --logits-dir, and run_laya_model.py --logits-dir ------------------------------------

def tiny_laya_folder(path):
    """A tiny Laya-format model (two-layer ModernBERT reader, Laya's decision head and tokenizer), seeded; as in
    test_public_training.py."""
    import torch
    from laya.common import build_model
    from safetensors.torch import save_file
    from transformers import ModernBertConfig

    (path / "encoder").mkdir(parents=True)
    ModernBertConfig(vocab_size=50368, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                     max_position_embeddings=512, global_attn_every_n_layers=2, local_attention=64, pad_token_id=50283,
                     cls_token_id=50281, sep_token_id=50282, bos_token_id=50281, eos_token_id=50282).save_pretrained(str(path / "encoder"))
    config = {"encoder": "answerdotai/ModernBERT-large", "head_layers": 1, "max_len": 384, "head_max_len": 192,
              "act_costs": {"escalate": 0.5}, "temperature": [1.0, 1.0, 1.0]}
    torch.manual_seed(0)
    model = build_model(config, encoder_dir=str(path / "encoder"), pretrained=False)
    save_file({k: v.to(torch.float16).contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    shutil.copytree(LAYA_TOKENIZER, path / "tokenizer")
    (path / "rl_agent_config.json").write_text(json.dumps(config))
    return path


def made_up_records(split, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 3))
        rows.append({"id": f"p{split}{i}", "qid": "pubmedqa_answer", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": {"question": f"Does treatment {i % 4} help?", "contexts": [{"section": "RESULTS", "text": f"word{label} " * 10}]},
                     "group": {"source": "pqa_artificial"}})
        label2 = int(rng.integers(0, 3))
        rows.append({"id": f"u{split}{i}", "qid": "readmit_window", "label": label2, "target": [float(label2 == k) for k in range(3)],
                     "state": {"age": f"[{50 + 10 * label2}-{60 + 10 * label2})", "time_in_hospital": int(rng.integers(1, 9))}})
    return rows


@pytest.mark.skipif(LAYA_TOKENIZER is None, reason="no copy of Laya's tokenizer")
def test_keep_last_and_logits_on_a_tiny_stand_in(tmp_path, monkeypatch):
    import torch
    from transformers import AutoTokenizer

    import build_items
    from src.eval import run_laya_model
    from src.train import train_laya

    model_dir = tiny_laya_folder(tmp_path / "tiny")
    records_dir = tmp_path / "records"
    splits = {"train": made_up_records("train", 12, 1), "validation": made_up_records("validation", 10, 2)}
    for split, rows in splits.items():
        for qid in ("pubmedqa_answer", "readmit_window"):
            write_jsonl(records_dir / f"{qid}_{split}.jsonl", [r for r in rows if r["qid"] == qid])
    write_jsonl(records_dir / "pubmedqa_answer_test.jsonl", [])
    write_jsonl(records_dir / "readmit_window_test.jsonl", [])
    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    questions = build_items.load_questions()
    for split in ("train", "validation"):
        items, _ = build_items.build_items(splits[split], questions, tok, max_len=384, head_max_len=192)
        torch.save(items, tmp_path / f"{split}.pt")

    saved = []
    real_save = train_laya.save_file
    monkeypatch.setattr(train_laya, "save_file", lambda tensors, path: (saved.append(Path(path)), real_save(tensors, path))[1])
    out, logits_dir, preds = tmp_path / "s7_a1", tmp_path / "logits", tmp_path / "preds"
    monkeypatch.setattr(sys, "argv", ["train_laya.py", "--train", str(tmp_path / "train.pt"), "--val", str(tmp_path / "validation.pt"),
                                      "--no-test", "--model-dir", str(model_dir), "--out", str(out), "--device", "cpu", "--epochs", "2",
                                      "--micro-batch", "4", "--grad-accum", "1", "--max-len", "384", "--batching", "by-question",
                                      "--eval-every", "2", "--lr-head", "3e-3", "--lr-encoder", "1e-3", "--keep", "last",
                                      "--predictions-dir", str(preds), "--logits-dir", str(logits_dir), "--system-name", "tiny_a1",
                                      "--metrics", str(tmp_path / "metrics.json"), "--log", str(tmp_path / "log.csv")])
    train_laya.main()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    history = metrics["validation_history"]
    assert metrics["keep"] == "last" and len(history) >= 3
    assert round(metrics["kept_validation_selection_log_loss_exact"], 4) == history[-1]["selection_log_loss"]
    assert saved == [out / "model.safetensors"]  # no best checkpoint was ever written; the final weights were saved once

    # the saved final weights reproduce the last validation's score, and the logits file gives the raw predictions
    for qid in ("pubmedqa_answer", "readmit_window"):
        z = read_csv(logits_dir / f"tiny_a1_{qid}_validation_logits.csv")
        p = read_csv(preds / f"tiny_a1-raw_{qid}_validation.csv")
        assert z[0] == ["id", "label", "z0", "z1", "z2"] and [r[:2] for r in z][1:] == [r[:2] for r in p][1:]
        probs = softmax(np.array([[float(x) for x in r[2:]] for r in z[1:]]))
        assert np.abs(probs - np.array([[float(x) for x in r[2:]] for r in p[1:]])).max() < 1e-6
    monkeypatch.setattr(sys, "argv", ["run_laya_model.py", "--model-dir", str(out), "--system", "tiny_a1_rerun", "--records-dir",
                                      str(records_dir), "--qids", "pubmedqa_answer", "readmit_window", "--out", str(tmp_path / "preds2"),
                                      "--max-len", "384", "--head-max-len", "192", "--device", "cpu", "--splits", "validation",
                                      "--calibration", str(out / "calibration.json"), "--logits-dir", str(tmp_path / "logits2")])
    run_laya_model.main()
    losses = []
    for qid in ("pubmedqa_answer", "readmit_window"):
        a = read_csv(logits_dir / f"tiny_a1_{qid}_validation_logits.csv")
        b = read_csv(tmp_path / "logits2" / f"tiny_a1_rerun_{qid}_validation_logits.csv")
        za, zb = (np.array([[float(x) for x in r[2:]] for r in rows[1:]]) for rows in (a, b))
        assert [r[:2] for r in a] == [r[:2] for r in b] and np.abs(softmax(za) - softmax(zb)).max() < 1e-3  # float16 weights
        labels = np.array([int(r[1]) for r in b[1:]])
        losses.append(float(np.mean(-np.log(softmax(zb)[np.arange(len(labels)), labels]))))
    assert abs(np.mean(losses) - metrics["kept_validation_selection_log_loss_exact"]) < 2e-3


# the calibration rule --------------------------------------------------------------------------------------------------

def calibration_case(n, bias, temperature, seed):
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.6, size=(n, 3))
    truth = softmax(z / temperature + np.asarray(bias))
    y = np.array([rng.choice(3, p=p) for p in truth])
    return z, y


def symmetric_case(n, temperature, seed):
    """No bias by construction: every item comes with its two cyclic relabelings, so the intercepts fitted on all the
    items are exactly zero, and in the folds they only fit noise."""
    z, y = calibration_case(n, [0.0, 0.0, 0.0], temperature, seed)
    return np.concatenate([np.roll(z, s, axis=1) for s in range(3)]), np.concatenate([(y + s) % 3 for s in range(3)])


def test_calibration_rule_chooses_by_out_of_fold_log_loss(tmp_path):
    z, y = calibration_case(600, [0.0, 0.9, -0.9], 1.0, 3)
    chosen, fams = CAL.choose(CAL.log_softmax(z), y)
    assert chosen == "temperature_intercepts"
    assert fams["temperature_intercepts"]["out_of_fold_log_loss"]["mean"] < fams["temperature"]["out_of_fold_log_loss"]["mean"] - 0.02
    b = fams["temperature_intercepts"]["intercepts"]
    assert b[0] == 0.0 and abs(b[1] - 0.9) < 0.35 and abs(b[2] + 0.9) < 0.35
    z, y = symmetric_case(200, 1.5, 4)
    chosen, fams = CAL.choose(CAL.log_softmax(z), y)
    assert chosen == "temperature" and fams["temperature"]["temperature"] > 1.2
    assert max(abs(v) for v in fams["temperature_intercepts"]["intercepts"]) < 1e-4

    # fit_system: only calibrate items are used; both files are written with 9 significant digits
    ids = [f"r{i}" for i in range(len(y))]
    logits_dir, records = tmp_path / "logits", tmp_path / "records"
    logits_dir.mkdir()
    with open(logits_dir / "s7_a3_pubmedqa_answer_validation_logits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "label", "z0", "z1", "z2"])
        for i, label, row in zip(ids, y, z):
            w.writerow([i, int(label)] + [repr(float(x)) for x in row])
    write_jsonl(records / "pubmedqa_answer_validation.jsonl",
                [{"id": i, **({"calibrate": False} if k % 4 == 0 else {})} for k, i in enumerate(ids)])
    entry = CAL.fit_system("s7_a3", "s7_a3", ["pubmedqa_answer"], logits_dir, None, tmp_path / "cal.json", tmp_path / "preds", records)
    keep = np.array([k % 4 != 0 for k in range(len(ids))])
    chosen, fams = CAL.choose(CAL.log_softmax(z[keep]), y[keep])
    assert entry["pubmedqa_answer"]["n_calibrate"] == int(keep.sum()) and entry["pubmedqa_answer"]["chosen"] == chosen
    t = fams["temperature"]["temperature"]
    rows = read_csv(tmp_path / "preds" / "s7_a3_temperature_pubmedqa_answer_validation.csv")
    assert len(rows) == len(ids) + 1 and all(len(x.replace(".", "").replace("-", "").lstrip("0").split("e")[0]) <= 9 for x in rows[1][2:])
    got = np.array([[float(x) for x in r[2:]] for r in rows[1:]])
    assert np.abs(got - softmax(CAL.log_softmax(z) / t)).max() < 1e-8
    assert (tmp_path / "preds" / "s7_a3_pubmedqa_answer_validation.csv").exists()
    CAL.apply_system("s7_a3", "s7_a3", ["pubmedqa_answer"], "validation", logits_dir, None, tmp_path / "cal.json", tmp_path / "again")
    assert read_csv(tmp_path / "again" / "s7_a3_pubmedqa_answer_validation.csv") == read_csv(tmp_path / "preds" / "s7_a3_pubmedqa_answer_validation.csv")
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        CAL.apply_system("s7_a3", "s7_a3", ["pubmedqa_answer"], "test", logits_dir, None, tmp_path / "cal.json", tmp_path / "again", tmp_path)


# the candidate rule and the freeze -------------------------------------------------------------------------------------

def test_candidate_rule_and_not_built():
    v02 = 0.7828
    assert SEL.candidate_rule({"s7_a3": 0.75, "s7_b3": 0.76}, v02)[0] == "s7_a3"
    assert SEL.candidate_rule({"s7_a3": 0.75, "s7_b3": 0.74}, v02)[0] == "s7_b3"
    assert SEL.candidate_rule({"s7_a3": 0.75, "s7_b3": 0.75}, v02)[0] == "s7_a3"  # an exact tie goes to s7_a3
    assert SEL.candidate_rule({"s7_a3": None, "s7_b3": 0.70}, v02)[0] == "s7_b3"
    cand, steps, winner, not_built = SEL.candidate_rule({"s7_a3": 0.79, "s7_b3": 0.7828}, v02)
    assert cand is None and winner == "s7_b3" and not_built == "NOT BUILT: v0.3 candidate, no arm beat the v0.2 candidate on validation"
    cand, _, _, not_built = SEL.candidate_rule({"s7_a3": None, "s7_b3": None}, v02)
    assert cand is None and not_built.startswith("NOT BUILT: v0.3 candidate")


def test_select_and_freeze_guards(tmp_path):
    results = tmp_path / "results"
    (results / "metrics").mkdir(parents=True)
    (results / "metrics" / "s7_a3_metrics.json").write_text(json.dumps({"kept_validation_selection_log_loss_exact": 0.7701}))
    (results / "metrics" / "s7_b3_metrics.json").write_text(json.dumps({"kept_validation_selection_log_loss_exact": 0.7702}))
    v02 = tmp_path / "v02.json"
    v02.write_text(json.dumps({"head_rule": {"candidate": "s5_final_fresh_head", "fresh_head_score": 0.7827999999999999}}))
    out = SEL.select(results, v02)
    assert out["candidate_rule"]["candidate"] == "s7_a3" and out["labeler"] == "labeler.json missing"
    for qid in SEL.QIDS:
        (results / f"ece_limit_{qid}.json").write_text(json.dumps({"predictions": f"s7_a3_temperature_{qid}_validation.csv",
                                                                   "n": SEL.TEST_ROWS[qid], "draws": 5000,
                                                                   "ece_if_calibrated": {"p95": 0.07 if qid == "pubmedqa_answer" else 0.01}}))
    with pytest.raises(SystemExit, match="the freeze needs"):  # computed on the temperature-only file, not the candidate's
        SEL.freeze(results, tmp_path / "empty")
    for qid in SEL.QIDS:
        (results / f"ece_limit_{qid}.json").write_text(json.dumps({"predictions": f"s7_a3_{qid}_validation.csv",
                                                                   "n": SEL.TEST_ROWS[qid], "draws": 5000,
                                                                   "ece_if_calibrated": {"p95": 0.07 if qid == "pubmedqa_answer" else 0.01}}))
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "s7_b3-raw_readmit_window_test.csv").write_text("id,label,p0,p1,p2\n")
    with pytest.raises(SystemExit, match="test predictions already exist"):
        SEL.freeze(results, staging)
    frozen = SEL.freeze(results, tmp_path / "empty")
    assert frozen["system"] == "s7_a3" and frozen["ece_limits"]["pubmedqa_answer"]["limit"] == 0.07
    assert frozen["ece_limits"]["readmit_window"]["limit"] == 0.05
    with pytest.raises(SystemExit, match="written once"):
        SEL.freeze(results)
    (results / "metrics" / "s7_a3_metrics.json").write_text(json.dumps({"kept_validation_selection_log_loss_exact": 0.80}))
    (results / "metrics" / "s7_b3_metrics.json").unlink()
    assert SEL.select(results, v02)["candidate_rule"]["not_built"].startswith("NOT BUILT")


# the readmission trees and the blend ----------------------------------------------------------------------------------

def uci_rows(split, n, seed):
    from baselines import UCI_NUMERIC

    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        state = {c: int(rng.integers(0, 20)) for c in UCI_NUMERIC}
        state["time_in_hospital"], state["number_inpatient"] = int(rng.integers(1, 14)), int(rng.integers(0, 4))
        state["age band"] = str(rng.choice(["[50-60)", "[60-70)", "[70-80)"]))
        state["sex"] = str(rng.choice(["Female", "Male"]))
        state["medications"] = {"insulin": str(rng.choice(["No", "Steady", "Up"]))}
        score = 0.45 * state["number_inpatient"] + 0.08 * state["time_in_hospital"] + rng.normal(0.0, 0.8)
        label = 2 if score > 1.6 else (1 if score > 0.9 else 0)
        rows.append({"id": f"uci-{split}-{i}", "qid": "readmit_window", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": state})
    return rows


@pytest.fixture
def uci_records(tmp_path):
    records = tmp_path / "records"
    for split, n, seed in (("train", 300, 1), ("validation", 120, 2), ("test", 120, 3)):
        write_jsonl(records / f"readmit_window_{split}.jsonl", uci_rows(split, n, seed))
    return records


def test_trees_search_resumes_and_test_is_guarded(tmp_path, uci_records):
    whole = TR.search(tmp_path / "whole", uci_records, max_seconds=1e9, n_settings=2)
    assert whole["complete"] and len(whole["fits"]) == 2 * TR.FOLDS
    parts = tmp_path / "parts"
    for calls in range(1, 20):
        progress = TR.search(parts, uci_records, max_seconds=0.0, n_settings=2)  # each call fits once, then stops
        if progress.get("complete"):
            break
    assert calls == 2 * TR.FOLDS + 1 or calls == 2 * TR.FOLDS
    for key, fit in whole["fits"].items():  # LightGBM's threaded metric sum can differ in the last bit
        assert progress["fits"][key]["log_loss"] == pytest.approx(fit["log_loss"], rel=1e-12, abs=0)
        assert progress["fits"][key]["best_iteration"] == fit["best_iteration"]
    out = TR.validation(parts, uci_records, n_settings=2)
    table = out["settings"]
    assert out["chosen"] == min(table, key=lambda r: (r["mean_out_of_fold_log_loss"], r["index"]))
    assert out["validation"]["rows"] == 120 and (parts / "predictions" / "lightgbm_tuned_readmit_window_validation.csv").exists()
    assert out["settings_drawn"] == {"from": 1620, "drawn": 2, "without_replacement": True}
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        TR.test(parts, uci_records)
    (parts / "release_candidate.json").write_text("{}")
    with pytest.raises(SystemExit, match="testpreds_log.json does not exist"):
        TR.test(parts, uci_records)
    (parts / "testpreds_log.json").write_text("{}")
    test_out = TR.test(parts, uci_records)
    assert test_out["rows"] == 120 and test_out["validation_reproduced_max_abs_difference"] <= 1e-8
    with pytest.raises(SystemExit, match="release_candidate.json"):
        TR.blend_test("blend_x", parts / "nothing.csv", tmp_path / "whole")


def test_settings_are_fifty_distinct_draws():
    grid, n = TR.settings()
    assert n == 4 * 5 * 3 * 3 * 3 * 3 == 1620 and len(grid) == 50 and len({s["combination"] for s in grid}) == 50
    assert TR.lgb_params({**grid[0], "bagging_fraction": 1.0})["subsample_freq"] == 0


def blend_case(n, seed):
    rng = np.random.default_rng(seed)
    p_f = softmax(rng.normal(0.0, 1.5, size=(n, 3)))
    y = np.array([rng.choice(3, p=p) for p in p_f])
    p_t = rng.dirichlet([1.0, 1.0, 1.0], size=n)  # noise: unrelated to y
    return y, np.log(p_f), np.log(p_t)


def test_blend_finds_t_by_maximum_likelihood_and_w_one_for_noise(tmp_path, monkeypatch):
    y, lf, lt = blend_case(1500, 5)
    kept = TR.fit_blend(lf, lt, y)
    assert kept["w"] == 1.0 and abs(kept["T"] - 1.0) < 0.1  # p_T is noise, so the pool keeps p_F alone
    grid = np.exp(np.linspace(np.log(0.1), np.log(10.0), 4001))
    for w in (0.3, 0.8):  # T at a fixed w, against a direct search over T
        monkeypatch.setattr(TR, "W_GRID", [w])
        fitted = TR.fit_blend(lf, lt, y)
        losses = np.array([np.mean(-np.log(TR.pool(lf, lt, w, t)[np.arange(len(y)), y])) for t in grid])
        assert abs(fitted["T"] - grid[int(np.argmin(losses))]) < 0.01 and fitted["validation_log_loss"] <= losses.min() + 1e-9
    monkeypatch.undo()

    # blend(): files, the kept pair and the out-of-fold score (two repeats here, to keep the test short)
    monkeypatch.setattr(TR, "BLEND_REPEATS", 2)
    results = tmp_path / "results"
    ids = [f"u{i}" for i in range(len(y))]
    write_probs(results / "predictions" / "lightgbm_tuned_readmit_window_validation.csv", ids[::-1], y[::-1], np.exp(lt)[::-1])
    write_probs(tmp_path / "fiorillo.csv", ids, y, np.exp(lf))
    out = TR.blend("blend_test", tmp_path / "fiorillo.csv", results, processes=1)
    assert out["w"] == 1.0 and out["rows"] == len(y) and out["out_of_fold_log_loss"]["repeats"] == 2
    rows = read_csv(results / "predictions" / "blend_test_readmit_window_validation.csv")
    got = np.array([[float(x) for x in r[2:]] for r in rows[1:]])
    assert [r[0] for r in rows[1:]] == ids and np.abs(got - softmax(lf / out["T"])).max() < 1e-7
    for name in ("release_candidate.json", "testpreds_log.json"):
        (results / name).write_text("{}")
    with pytest.raises(SystemExit, match="not the same system's test file"):
        TR.blend_test("blend_test", tmp_path / "other_readmit_window_test.csv", results)


# the hand-off analysis ------------------------------------------------------------------------------------------------

def test_handoff_against_a_hand_calculation(tmp_path):
    ids = ["r4", "r1", "r3", "r2"]
    labels = np.array([0, 1, 2, 0])
    probs = np.array([[0.9, 0.05, 0.05],   # r4: sure, right
                      [0.6, 0.3, 0.1],     # r1: 0.6, wrong
                      [0.2, 0.2, 0.6],     # r3: 0.6, right; ties with r1, which comes first by ID
                      [0.4, 0.35, 0.25]])  # r2: least sure, right
    rank = np.array([3, 0, 2, 1])
    c = H.curve(rank, labels, probs)
    # order r4, r1, r3, r2; correct 1, 0, 1, 1
    assert c["answered_accuracy_100"] == 0.75 and c["handed_off_accuracy_100"] is None and c["answered_100"] == 4
    assert c["answered_accuracy_75"] == pytest.approx(2 / 3) and c["handed_off_accuracy_75"] == 1.0 and c["answered_75"] == 3
    assert c["answered_accuracy_50"] == 0.5 and c["handed_off_accuracy_50"] == 1.0 and c["answered_50"] == 2
    assert c["aurc"] == pytest.approx((0 + 1 / 2 + 1 / 3 + 1 / 4) / 4)
    out = H.analyze(ids, labels, probs, resamples=200)
    assert out["n"] == 4 and out["aurc"]["estimate"] == pytest.approx(c["aurc"])
    assert 0.0 <= out["aurc"]["low"] <= out["aurc"]["high"] <= 1.0

    # main(): validation keeps only the calibrate items
    rng = np.random.default_rng(0)
    n = 40
    p = rng.dirichlet([2, 2, 2], size=n)
    y = rng.integers(0, 3, size=n)
    names = [f"x{i:02d}" for i in range(n)]
    write_probs(tmp_path / "preds" / "sys_pubmedqa_answer_validation.csv", names, y, p)
    write_jsonl(tmp_path / "records" / "pubmedqa_answer_validation.jsonl",
                [{"id": x, **({"calibrate": False} if i < 10 else {})} for i, x in enumerate(names)])
    H.main(["--split", "validation", "--systems", "sys", "--predictions-dir", str(tmp_path / "preds"), "--records-dir",
            str(tmp_path / "records"), "--qids", "pubmedqa_answer", "--out", str(tmp_path / "h.json"), "--resamples", "50"])
    got = json.loads((tmp_path / "h.json").read_text())["systems"]["sys"]["pubmedqa_answer"]
    with pytest.raises(SystemExit, match="release_candidate.json does not exist"):
        H.main(["--split", "test", "--systems", "sys", "--predictions-dir", str(tmp_path / "preds"), "--records-dir",
                str(tmp_path / "records"), "--qids", "pubmedqa_answer", "--out", str(tmp_path / "t.json"), "--results", str(tmp_path)])
    keep = slice(10, n)
    assert got["n"] == 30 and got["aurc"]["estimate"] == pytest.approx(H.analyze(names[keep], y[keep], p[keep], 50)["aurc"]["estimate"])


# criterion 4 and the comparisons --------------------------------------------------------------------------------------

def test_criterion_4_intervals_and_routes_by_hand():
    labels = np.array([0, 1, 2, 0, 1, 2])
    a = np.full((6, 3), 0.25)
    b = np.full((6, 3), 0.125)
    a[np.arange(6), labels], b[np.arange(6), labels] = 0.5, 0.75
    ci = CMP.c4_interval(labels, a, b, resamples=500)
    same = math.log(0.75) - math.log(0.5)  # every row's difference: -log 0.5 - (-log 0.75)
    assert all(abs(ci[k] - same) < 1e-12 for k in ("estimate", "low_95", "high_95", "low_97_5", "high_97_5"))

    rng = np.random.default_rng(1)
    n = 50
    labels = rng.integers(0, 3, size=n)
    a, b = rng.dirichlet([1, 1, 1], size=n), rng.dirichlet([1, 1, 1], size=n)
    ci = CMP.c4_interval(labels, a, b, resamples=400, seed=7)
    d = -np.log(a[np.arange(n), labels]) + np.log(b[np.arange(n), labels])
    draws = np.random.default_rng(7)
    means = np.array([d[draws.integers(0, n, n)].mean() for _ in range(400)])
    assert ci["estimate"] == pytest.approx(d.mean())
    assert ci["low_95"] == pytest.approx(np.percentile(means, 2.5)) and ci["high_95"] == pytest.approx(np.percentile(means, 97.5))
    assert ci["low_97_5"] == pytest.approx(np.percentile(means, 1.25)) and ci["high_97_5"] == pytest.approx(np.percentile(means, 98.75))
    assert ci["low_97_5"] <= ci["low_95"] <= ci["high_95"] <= ci["high_97_5"]

    def iv(high_97_5, low_95):
        return {"high_97_5": high_97_5, "low_95": low_95}

    r = CMP.criterion4_routes({"pubmedqa_answer": iv(-0.01, -0.05), "readmit_window": iv(0.004, -0.0002)})
    assert r["a"]["passes"] and not r["b"]["passes"]
    r = CMP.criterion4_routes({"pubmedqa_answer": iv(-0.01, -0.05), "readmit_window": iv(0.004, 0.0003)})
    assert not r["a"]["passes"] and not r["b"]["passes"]  # readmission significantly worse
    r = CMP.criterion4_routes({"pubmedqa_answer": iv(0.02, -0.02), "readmit_window": iv(-0.001, -0.004)})
    assert not r["a"]["passes"] and r["b"]["passes"]
    r = CMP.criterion4_routes({"pubmedqa_answer": iv(0.0, -0.02), "readmit_window": iv(0.001, 0.0)})
    assert not r["a"]["own"] and r["a"]["other_not_worse"] and r["b"]["other_not_worse"]  # at zero: not below; at or below zero: not worse


def comparison_world(tmp):
    """Made-up test records and predictions for v0.1, the prior and a candidate s7_a3 (the other systems missing)."""
    rng = np.random.default_rng(11)
    records, s4, s7, results = tmp / "records", tmp / "s4", tmp / "s7", tmp / "results"
    files = {}
    for qid, n in (("pubmedqa_answer", 90), ("readmit_window", 120)):
        for split in ("test", "validation"):
            y = rng.integers(0, 3, size=n)
            ids = [f"{qid[:3]}-{split}-{i}" for i in range(n)]
            write_jsonl(records / f"{qid}_{split}.jsonl", [{"id": i, "label": int(v)} for i, v in zip(ids, y)])
            onehot = np.eye(3)[y]
            prior = np.tile(np.bincount(y, minlength=3) / n, (n, 1))
            for system, folder, strength in (("arm_laya", s4, 0.35), ("prior", s4, None), ("s7_a3", s7, 0.55)):
                probs = prior if strength is None else (1 - strength) * rng.dirichlet([1, 1, 1], size=n) + strength * onehot
                write_probs(folder / f"{system}_{qid}_{split}.csv", ids, y, probs)
                files[(system, qid, split)] = probs
    results.mkdir()
    (results / "testpreds_log.json").write_text(json.dumps({"release_candidate_sha256": "made up", "testpreds_start_utc": "made up"}))
    (results / "release_candidate.json").write_text(json.dumps({
        "system": "s7_a3", "not_built": None,
        "ece_limits": {"pubmedqa_answer": {"limit": 0.5}, "readmit_window": {"limit": 0.5}}}))
    return records, s4, s7, results, files


def test_compare_end_to_end_and_resume(tmp_path, monkeypatch):
    records, s4, s7, results, files = comparison_world(tmp_path)
    monkeypatch.setattr(CMP, "S4", s4)
    monkeypatch.setattr(CMP, "S5", tmp_path / "s5_missing")
    monkeypatch.setattr(CMP, "S7", s7)
    args = ["--records-dir", str(records), "--results", str(results), "--resamples", "40", "--resamples-c4", "300"]
    (results / "testpreds_log.json").rename(results / "held_back.json")
    with pytest.raises(SystemExit, match="testpreds_log.json does not exist"):
        CMP.main(args + ["--cache", str(tmp_path / "cache_whole")])
    (results / "held_back.json").rename(results / "testpreds_log.json")
    out = CMP.main(args + ["--cache", str(tmp_path / "cache_whole"), "--max-seconds", "1e9"])
    whole = json.loads((results / "comparison.json").read_text())
    assert out is not None and whole["candidate"] == "s7_a3" and whole["release_bar"]["overall"] == "FAIL"
    assert whole["release_bar"]["outcome"].startswith("v0.3 release bar not met: ")
    bar = whole["release_bar"]["criteria"]
    assert bar["3"]["result"] == "FAIL" and "Duan" in bar["3"]["reason"]  # Duan's predictions are missing
    ci = whole["criterion_4_intervals"]
    for qid in ("pubmedqa_answer", "readmit_window"):
        labels = np.array([json.loads(x)["label"] for x in open(records / f"{qid}_test.jsonl")])
        by_hand = CMP.c4_interval(labels, files[("s7_a3", qid, "test")], files[("arm_laya", qid, "test")], 300)
        assert ci[qid]["estimate"] == pytest.approx(by_hand["estimate"], abs=1e-8)  # probabilities printed to 9 digits
        assert ci[qid]["high_97_5"] == pytest.approx(by_hand["high_97_5"], abs=1e-6)
    routes = CMP.criterion4_routes(ci)
    assert bar["4"]["values"]["routes"] == json.loads(json.dumps(routes)) and bar["4"]["result"] == ("PASS" if any(
        r["passes"] for r in routes.values()) else "FAIL")
    assert any(c["section"] == "E1" and c["metric"] == "log_loss" for c in whole["comparisons"])
    zeros = json.loads((results / "zeros.json").read_text())["rows_whose_true_answer_prints_as_0"]
    assert zeros["s7_a3"]["test"]["pubmedqa_answer"] == {"rows": 90, "true_answer_printed_zero": 0}

    for calls in range(1, 200):  # every call computes one block and stops
        done = CMP.main(args + ["--cache", str(tmp_path / "cache_parts"), "--max-seconds", "0"])
        if done is not None:
            break
    assert calls > 5
    resumed = json.loads((results / "comparison.json").read_text())
    for d in (whole, resumed):
        d.pop("time_utc")
        d.pop("created")
    assert resumed == whole


# power -----------------------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not RETRY.exists(), reason="results/fiorillo_v0_2/retry_check.json not present")
def test_power_against_retry_check():
    retry = json.loads(RETRY.read_text())
    out = PW.compute(retry)
    lines = retry["pubmedqa_lines"]
    v02 = out["v0_2_criterion_4"]
    assert v02["matches_retry_check"]
    # retry_check_session5.py used z values rounded to six decimals (1.959964, 2.241403, 0.841621), hence 1e-7
    assert v02["pass_line"] == pytest.approx(lines["pass_line_95"], abs=1e-7)
    assert v02["real_gain_for_80_percent_alone"] == pytest.approx(lines["true_gain_for_80_percent_95"], abs=1e-7)
    a = out["routes"]["route_a"]["by_se_scale"]["1.0"]
    assert a["pass_line"] == pytest.approx(lines["pass_line_97_5_second_look"], abs=1e-7)  # the same 97.5 percent line
    assert a["real_gain_for_80_percent_alone"] == pytest.approx(lines["true_gain_for_80_percent_97_5"], abs=1e-7)
    se_p = retry["candidate_minus_v0_1"]["pubmedqa_answer"]["se_bootstrap"]
    assert PW.own(a["real_gain_for_80_percent_alone"], se_p) == pytest.approx(0.80, abs=1e-9)
    assert 0.02 < out["false_pass_rate_no_real_gain"] < 0.025
    bad = json.loads(json.dumps(retry))
    bad["pubmedqa_lines"]["pass_line_95"] = -0.03
    with pytest.raises(SystemExit, match="disagrees"):
        PW.compute(bad)


# the upload's resume logic ----------------------------------------------------------------------------------------------

class FakePod(U.Pod):
    """The Jupyter calls upload() makes, served from a local folder: the Contents API's chunked PUT (chunk 1 creates,
    later chunks append, -1 is the last) and a kernel that runs the code here. fail_after: raise after that many
    chunks, as a lost connection would."""

    def __init__(self, root, fail_after=None):
        super().__init__("http://fake-pod", "made-up-token")
        self.root, self.fail_after, self.chunks = Path(root), fail_after, []
        self.session = SimpleNamespace(get=self._get)

    def _get(self, url, params=None, timeout=None):
        return SimpleNamespace(status_code=200 if (self.root / url.split("/api/contents/", 1)[1]).exists() else 404)

    def rest(self, method, path, json=None, **kw):
        assert method == "PUT" and path.startswith("/api/contents/")
        target = self.root / path[len("/api/contents/"):]
        if json.get("type") == "directory":
            target.mkdir(parents=True, exist_ok=True)
            return {}
        chunk = json.get("chunk")
        if chunk is not None and self.fail_after is not None and len(self.chunks) >= self.fail_after:
            raise RuntimeError("the connection was lost (made up)")
        data = base64.b64decode(json["content"]) if json.get("format") == "base64" else json["content"].encode()
        if chunk is None or chunk == 1:
            target.write_bytes(data)
        else:
            with open(target, "ab") as f:
                f.write(data)
        if chunk is not None:
            self.chunks.append((target.name, chunk, len(data)))
        return {}

    def python(self, code, timeout=600):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            exec(code, {})
        return out.getvalue()

    def close(self):
        pass


def test_upload_resumes_part_way_and_recovers_a_bad_copy(tmp_path, monkeypatch):
    server = tmp_path / "server"
    server.mkdir()
    monkeypatch.chdir(server)  # the kernel finds the server's root folder from its working folder
    local = tmp_path / "local"
    rng = np.random.default_rng(0)
    (local / "sub").mkdir(parents=True)
    big = rng.integers(0, 256, size=300_000, dtype=np.uint8).tobytes()
    (local / "sub" / "big.bin").write_bytes(big)
    (local / "small.txt").write_text("small file\n")
    remote = tmp_path / "remote"
    chunk_mb = 65536 / 2**20
    rel = ["sub/big.bin", "small.txt"]

    first = FakePod(server, fail_after=2)
    with pytest.raises(RuntimeError, match="lost"):
        U.upload(first, local, str(remote), rel, chunk_mb, 1e9)
    assert sum(n for _, _, n in first.chunks) == 2 * 65536 and not (remote / "sub" / "big.bin").exists()

    second = FakePod(server)
    assert U.upload(second, local, str(remote), rel, chunk_mb, 1e9)
    assert second.chunks[0][1] == 2 and sum(n for _, _, n in second.chunks) == len(big) - 2 * 65536  # nothing sent twice
    assert (remote / "sub" / "big.bin").read_bytes() == big and (remote / "small.txt").read_text() == "small file\n"
    assert not [p for p in (server / U.STAGING).iterdir() if not p.name.startswith("probe-")]

    third = FakePod(server)
    assert U.upload(third, local, str(remote), rel, chunk_mb, 1e9) and third.chunks == []  # already in place

    # a deadline stops each call after one chunk; the calls together send the file once
    (local / "sub" / "big2.bin").write_bytes(big[::-1])
    sent = 0
    for calls in range(1, 20):
        pod = FakePod(server)
        ok = U.upload(pod, local, str(remote), ["sub/big2.bin"], chunk_mb, 0.0)
        sent += sum(n for _, _, n in pod.chunks)
        if ok:
            break
    assert calls == math.ceil(len(big) / 65536) and sent == len(big) and (remote / "sub" / "big2.bin").read_bytes() == big[::-1]

    # a staged copy of the right size but the wrong content is set aside, and the next call sends the file again
    (local / "c.bin").write_bytes(big[:100_000])
    digest = U.sha256_file(local / "c.bin")
    staged = server / U.STAGING / f"{digest[:16]}-c.bin"
    staged.write_bytes(bytes(100_000))
    assert not U.upload(FakePod(server), local, str(remote), ["c.bin"], chunk_mb, 1e9)
    assert (server / U.STAGING / f"{digest[:16]}-c.bin.mismatch").exists() and not (remote / "c.bin").exists()
    assert U.upload(FakePod(server), local, str(remote), ["c.bin"], chunk_mb, 1e9)
    assert (remote / "c.bin").read_bytes() == big[:100_000]


def test_kernel_connection_closed_is_opened_again(monkeypatch):
    """The proxy closes an idle websocket; python() opens it again to the same kernel and runs the code once more."""
    pod = U.Pod("http://fake-pod", "made-up-token")
    pod.kernel_id, pod.ws = "k1", SimpleNamespace(close=lambda: None)
    calls = []

    def once(code, timeout):
        calls.append((code, pod.ws is None))
        if len(calls) == 1:
            raise EOFError("the websocket was closed")
        return "done\n"

    monkeypatch.setattr(pod, "_python_once", once)
    assert pod.python("print('done')") == "done\n" and len(calls) == 2 and calls[1][1] is True and pod.kernel_id == "k1"
    monkeypatch.setattr(pod, "_python_once", lambda code, timeout: (_ for _ in ()).throw(TimeoutError("slow")))
    with pytest.raises(TimeoutError):
        pod.python("x")


def test_mac_check_ladder_one_process_per_try(monkeypatch):
    """Correction 6: every try is its own process, run with the memory cap; out of memory and a stopped try move the
    ladder on; any other error moves arm B to float16; arm A never adds the checkpointing try."""
    args = SimpleNamespace(steps=20, records_dir=Path("r"), laya_dir=Path("l"), qwen_dir=Path("q"), step_limit=600)
    monkeypatch.setattr(MAC, "swap_used_gib", lambda: 1.5)
    calls = []
    outcomes = iter([
        ("error", {"error": "RuntimeError: MPS backend out of memory (MPS allocated: 15 GB)", "oom": True,
                   "recommended_gib": 16.0, "cap_gib": 16.0}),
        ("stopped", {"error": "stopped: 600 s passed without an optimizer step finishing", "oom": True}),
        ("error", {"error": "RuntimeError: MPS backend out of memory", "oom": True}),
        ("result", {"seconds_per_step": 90.0, "peak_memory_gib": 10.5, "precision": "bfloat16",
                    "recommended_gib": 16.0, "cap_gib": 16.0}),
    ])

    def run(cmd, env):
        calls.append((cmd, env))
        return next(outcomes)

    res = MAC.check("b", args, run)
    assert res["fit"] and res["micro_batch"] == 2 and res["gradient_checkpointing"] is True and res["accumulation"] == 16
    assert [a["micro_batch"] for a in res["attempts"]] == [8, 4, 2, 2]
    assert [a["fit"] for a in res["attempts"]] == [False, False, False, True]
    assert res["hours_per_100000_records"] == pytest.approx(3125 * 90 / 3600)
    assert res["attempts"][0]["swap_used_gib_before"] == 1.5 and res["attempts"][0]["cap_gib"] == 16.0
    for cmd, env in calls:
        assert cmd[:3] == [sys.executable, str(Path(MAC.__file__).resolve()), "--one"]
        assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.0" and env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.8"
    assert "--checkpointing" in calls[3][0] and all("--checkpointing" not in c for c, _ in calls[:3])

    outcomes = iter([("error", {"error": "TypeError: bfloat16 is not supported", "oom": False}),
                     ("result", {"seconds_per_step": 80.0, "peak_memory_gib": 12.0, "precision": "float16"})])
    calls.clear()
    res = MAC.check("b", args, run)
    assert res["fit"] and res["precision"] == "float16" and res["micro_batch"] == 8
    assert calls[1][0][calls[1][0].index("--dtype") + 1] == "float16"

    outcomes = iter([("error", {"error": "out of memory", "oom": True})] * 3)
    calls.clear()
    res = MAC.check("a", args, run)
    assert not res["fit"] and [a["micro_batch"] for a in res["attempts"]] == [8, 4, 2]
    assert all(a["gradient_checkpointing"] for a in res["attempts"])


def test_mac_check_run_try_stops_a_silent_try():
    """Correction 6: a try whose next step does not finish within the limit is stopped and counted as out of memory;
    RESULT and ERROR lines are read; a try that ends with no result is an error that is not out of memory."""
    py = sys.executable
    shown = []
    t0 = time.monotonic()
    status, payload = MAC.run_try([py, "-c", "import time; print('step 1: 0.10 s', flush=True); time.sleep(60)"],
                                  None, step_limit=1.0, echo=shown.append)
    assert status == "stopped" and payload["oom"] is True and time.monotonic() - t0 < 20
    assert shown == ["step 1: 0.10 s"]
    status, payload = MAC.run_try([py, "-c", "print('step 1: 1.00 s'); print('RESULT ' + '{\"seconds_per_step\": 1.5, "
                                   "\"peak_memory_gib\": 2.0, \"precision\": \"float32\"}')"], None, step_limit=30)
    assert status == "result" and payload["seconds_per_step"] == 1.5
    status, payload = MAC.run_try([py, "-c", "print('ERROR ' + '{\"error\": \"RuntimeError: MPS backend out of memory\", "
                                   "\"oom\": true}')"], None, step_limit=30)
    assert status == "error" and payload["oom"] is True
    status, payload = MAC.run_try([py, "-c", "import sys; sys.exit(1)"], None, step_limit=30)
    assert status == "error" and payload["oom"] is False and "code 1" in payload["error"]


def test_mac_check_one_arm_keeps_the_other(tmp_path, monkeypatch):
    """Correction 6: a run of arm B alone keeps arm A's result from an earlier run on the same Mac, with its time; on
    another Mac nothing is kept."""
    machine = {"chip": "Apple M5", "memory_gib": 24.0, "macos": "26.4.1", "python": "3.11.16", "torch": "2.14.0",
               "mps_available": True}
    monkeypatch.setattr(MAC, "machine", lambda: dict(machine))
    rec = tmp_path / "records"
    for qid in ("pubmedqa_answer", "readmit_window"):
        write_jsonl(rec / f"{qid}_train.jsonl", [{"id": f"{qid}-{k}"} for k in range(3)])
    out = tmp_path / "mac_training_check.json"
    arm_a = {"attempts": [{"micro_batch": 8, "fit": True}], "micro_batch": 8, "seconds_per_step": 15.65, "fit": True}
    out.write_text(json.dumps({"machine": machine, "arms": {"a": arm_a}, "time_utc": "2026-09-30T08:06:14Z"}),
                   encoding="utf-8")
    monkeypatch.setattr(MAC, "check", lambda arm, args, run=None: {"attempts": [], "fit": True, "micro_batch": 2})
    MAC.main(["--arms", "b", "--out", str(out), "--records-dir", str(rec)])
    got = json.loads(out.read_text(encoding="utf-8"))
    assert got["arms"]["a"]["seconds_per_step"] == 15.65 and got["arms"]["a"]["time_utc"] == "2026-09-30T08:06:14Z"
    assert got["arms"]["a"]["tries_run"] == "in one process, before correction 6"
    assert got["arms"]["b"]["micro_batch"] == 2 and got["arms"]["b"]["tries_run"] == "one process per try (correction 6)"
    assert [r.get("arms_recorded", r.get("arms")) for r in got["runs"]] == [["a"], ["b"]]
    monkeypatch.setattr(MAC, "machine", lambda: {**machine, "chip": "Apple M4"})
    MAC.main(["--arms", "b", "--out", str(out), "--records-dir", str(rec)])
    assert set(json.loads(out.read_text(encoding="utf-8"))["arms"]) == {"b"}

"""Tests for session 4 training code: build_items, the trainer's per-question logic, and assemble_reader."""
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC.parent))
sys.path.insert(0, str(SRC / "train"))

from src.train import assemble_reader, build_items  # noqa: E402
from src.train.train_laya import (fit_temperatures, item_key, per_question, probabilities, selection_score,  # noqa: E402
                                  summarize, write_predictions)

TOKENIZER = SRC.parent / "models" / "tokenizers" / "laya"


def row(key, label, logits, calibrate=True, rid="x"):
    target = [0.0] * len(logits)
    target[label] = 1.0
    return {"key": key, "id": rid, "label": label, "logits": logits, "target": target, "calibrate": calibrate}


def test_item_key_uses_qid_then_type():
    assert item_key({"qid": "readmit_window", "qtype": 1}) == "readmit_window"
    assert item_key({"qtype": 2}) == "noul"


def test_selection_score_weights_questions_equally_and_skips_monitoring_rows():
    rows = [row("a", 0, [2.0, 0.0])] * 10 + [row("b", 1, [0.0, 0.0])] + [row("b", 0, [-9.0, 9.0], calibrate=False)] * 50
    probs = probabilities(rows)
    loss_a = -np.log(np.exp(2.0) / (np.exp(2.0) + 1.0))
    loss_b = -np.log(0.5)
    assert abs(selection_score(rows, probs) - (loss_a + loss_b) / 2) < 1e-4
    assert per_question(rows, probs, calibrate_only=True)["b"]["n"] == 1
    assert per_question(rows, probs)["b"]["n"] == 51


def test_temperatures_are_fitted_per_question_on_calibrate_rows():
    rows = []
    for i in range(200):
        gold = i % 2
        z = [0.0, 0.0]
        z[gold] = 6.0
        if i % 4 == 0:
            z = [z[1], z[0]]
        rows.append(row("over", gold, z))
        rows.append(row("fine", gold, [0.4 if gold == 0 else -0.4, -0.4 if gold == 0 else 0.4]))
    t = fit_temperatures(rows)
    assert set(t) == {"fine", "over"} and t["over"] > 1.0
    scaled = probabilities(rows, t)
    assert summarize(rows, scaled, lambda r: r["key"] == "over")["log_loss"] < summarize(rows, probabilities(rows), lambda r: r["key"] == "over")["log_loss"]


def test_write_predictions_format(tmp_path):
    rows = [row("q1", 1, [0.0, 1.0, 0.0], rid="r1"), row("q2", 0, [1.0, 0.0], rid="r2")]
    paths = write_predictions(tmp_path, "sys", "test", rows, probabilities(rows))
    assert sorted(p.name for p in paths) == ["sys_q1_test.csv", "sys_q2_test.csv"]
    with open(tmp_path / "sys_q1_test.csv") as f:
        lines = list(csv.reader(f))
    assert lines[0] == ["id", "label", "p0", "p1", "p2"] and lines[1][:2] == ["r1", "1"]
    assert abs(sum(float(x) for x in lines[1][2:]) - 1.0) < 1e-5


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="Laya tokenizer copy not present")
def test_build_items_markers_flags_and_truncation():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(TOKENIZER))
    questions = build_items.load_questions()
    records = [
        {"id": "u1", "qid": "readmit_window", "state": {"age band": "[70-80)", "time_in_hospital": 3}, "label": 2, "target": [0.0, 0.0, 1.0]},
        {"id": "p1", "qid": "pubmedqa_answer", "state": {"question": "Q?", "contexts": [{"section": "RESULTS", "text": "word " * 900}]},
         "label": 0, "target": [1.0, 0.0, 0.0], "calibrate": False},
    ]
    items, summary = build_items.build_items(records, questions, tok, max_len=256, head_max_len=192)
    assert summary == {"records": 2, "items": 2, "skipped": 0, "truncated": 1}
    assert [len(it["markers"]) for it in items] == [3, 3]
    assert items[0]["qtype"] == 1 and items[1]["qtype"] == 0
    assert items[1]["calibrate"] is False and items[0]["calibrate"] is True
    assert all(len(it["ids"]) <= 256 for it in items)
    bad = [dict(records[0], target=[1.0, 0.0])]
    with pytest.raises(ValueError):
        build_items.build_items(bad, questions, tok, 256, 192)


def make_folder(path, tensors, vocab):
    path.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path / "model.safetensors"))
    return path


def test_assemble_reader_replaces_encoder_and_checks(tmp_path):
    vocab = {"type": "BPE", "vocab": {"a": 0, "b": 1}}
    laya = make_folder(tmp_path / "laya", {
        "encoder.w": torch.zeros(2, 2, dtype=torch.float16), "encoder.v": torch.zeros(3, dtype=torch.float16),
        "head.x": torch.ones(2), "temperature": torch.ones(3)}, vocab)
    for folder in ("encoder", "tokenizer"):
        (laya / folder).mkdir()
    (laya / "encoder" / "config.json").write_text("{}")
    (laya / "tokenizer" / "tokenizer.json").write_text(json.dumps({"model": vocab}))
    (laya / "rl_agent_config.json").write_text(json.dumps({"head_layers": 2}))
    reader = make_folder(tmp_path / "reader", {
        "model.w": torch.full((2, 2), 0.5), "model.v": torch.full((3,), 0.25), "head.dense.weight": torch.ones(1),
        "decoder.bias": torch.ones(1)}, vocab)
    (reader / "tokenizer.json").write_text(json.dumps({"model": vocab}))
    n = assemble_reader.assemble(laya, reader, tmp_path / "out", "L", "R")
    out = load_file(str(tmp_path / "out" / "model.safetensors"))
    assert n == 2 and out["encoder.w"].dtype == torch.float16 and float(out["encoder.w"][0, 0]) == 0.5
    assert torch.equal(out["head.x"], torch.ones(2)) and "decoder.bias" not in out
    (reader / "tokenizer.json").write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}}}))
    with pytest.raises(ValueError):
        assemble_reader.assemble(laya, reader, tmp_path / "out2")
    (reader / "tokenizer.json").write_text(json.dumps({"model": vocab}))
    save_file({"model.w": torch.zeros(3, 3), "model.v": torch.zeros(3)}, str(reader / "model.safetensors"))
    with pytest.raises(ValueError):
        assemble_reader.assemble(laya, reader, tmp_path / "out3")


def tiny_laya_folder(path):
    """A tiny Laya-format model (two-layer ModernBERT reader, Laya's decision head and tokenizer), seeded."""
    import shutil

    from laya.common import build_model
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
    shutil.copytree(TOKENIZER, path / "tokenizer")
    (path / "rl_agent_config.json").write_text(json.dumps(config))
    return path


def made_up_records(split, n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 3))
        rows.append({"id": f"p{split}{i}", "qid": "pubmedqa_answer", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": {"question": f"Does treatment {i % 4} help?", "contexts": [{"section": "RESULTS", "text": f"word{label} " * 10}]},
                     "group": {"source": "pqa_labeled" if i % 3 else "pqa_artificial"}})
        label2 = int(rng.integers(0, 3))
        rows.append({"id": f"u{split}{i}", "qid": "readmit_window", "label": label2, "target": [float(label2 == k) for k in range(3)],
                     "state": {"age": f"[{50 + 10 * label2}-{60 + 10 * label2})", "time_in_hospital": int(rng.integers(1, 9))}})
    return rows


@pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="Laya tokenizer copy not present")
def test_no_test_and_calibration_on_a_tiny_stand_in(tmp_path, monkeypatch):
    """--maybe-weight auto trains; --no-test reads no test split and writes no test file; run_laya_model.py
    --calibration with the run's calibration.json reproduces the validation probabilities train_laya.py wrote."""
    from transformers import AutoTokenizer

    from src.eval import run_laya_model
    from src.train import train_laya

    model_dir = tiny_laya_folder(tmp_path / "tiny")
    records_dir = tmp_path / "records"
    records_dir.mkdir()
    splits = {"train": made_up_records("train", 12, 1), "validation": made_up_records("validation", 12, 2), "test": made_up_records("test", 4, 3)}
    for split, rows in splits.items():
        for qid in ("pubmedqa_answer", "readmit_window"):
            with open(records_dir / f"{qid}_{split}.jsonl", "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in rows if r["qid"] == qid)
    tok = AutoTokenizer.from_pretrained(str(model_dir / "tokenizer"))
    questions = build_items.load_questions()
    for split in ("train", "validation"):
        items, _ = build_items.build_items(splits[split], questions, tok, max_len=384, head_max_len=192)
        torch.save(items, tmp_path / f"{split}.pt")
    out, preds = tmp_path / "out", tmp_path / "preds"
    monkeypatch.setattr(sys, "argv", ["train_laya.py", "--train", str(tmp_path / "train.pt"), "--val", str(tmp_path / "validation.pt"),
                                      "--test", str(tmp_path / "no_such_file.pt"), "--no-test", "--model-dir", str(model_dir),
                                      "--out", str(out), "--device", "cpu", "--epochs", "1", "--micro-batch", "4", "--grad-accum", "1",
                                      "--max-len", "384", "--batching", "by-question", "--reinit-head", "--freeze-encoder-steps", "1",
                                      "--maybe-weight", "auto", "--predictions-dir", str(preds), "--system-name", "tiny",
                                      "--metrics", str(tmp_path / "metrics.json"), "--log", str(tmp_path / "log.csv")])
    train_laya.main()
    metrics = json.loads((tmp_path / "metrics.json").read_text())
    labeled = [r for r in splits["train"] if r["qid"] == "pubmedqa_answer"]
    other = sum(1 for r in labeled if r["label"] != 2)
    labeled_not_maybe = sum(1 for r in labeled if r["label"] != 2 and r["group"]["source"] == "pqa_labeled")
    assert metrics["test"] == "not run (--no-test)" and metrics["maybe_weight"]["w"] == other / labeled_not_maybe
    assert not list(preds.glob("*_test.csv")) and len(list(preds.glob("*_validation.csv"))) == 4
    monkeypatch.setattr(sys, "argv", ["run_laya_model.py", "--model-dir", str(out), "--system", "tiny_rerun", "--records-dir", str(records_dir),
                                      "--qids", "pubmedqa_answer", "readmit_window", "--out", str(tmp_path / "preds2"),
                                      "--max-len", "384", "--head-max-len", "192", "--device", "cpu", "--calibration", str(out / "calibration.json")])
    run_laya_model.main()
    for qid in ("pubmedqa_answer", "readmit_window"):
        with open(preds / f"tiny_{qid}_validation.csv") as f:
            a = list(csv.reader(f))
        with open(tmp_path / "preds2" / f"tiny_rerun_{qid}_validation.csv") as f:
            b = list(csv.reader(f))
        assert [r[:2] for r in a] == [r[:2] for r in b]
        diff = max(abs(float(x) - float(y)) for ra, rb in zip(a[1:], b[1:]) for x, y in zip(ra[2:], rb[2:]))
        assert diff < 1e-4

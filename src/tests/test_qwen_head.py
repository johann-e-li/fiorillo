"""Tests for arm B, Laya's decision head on Qwen3 with LoRA (docs/prompt_session7.md, ARM B and step 4).

On a two-layer Qwen3 stand-in built from a small Qwen3Config with random weights (head dimension set explicitly) and Laya's
tokenizer from models/laya-base: at the start the adapted model's output equals the base's; only the adapters, the
projection and the head can learn; in the freeze steps the adapters do not change and the projection does; the learning
rates follow the name groups; a saved and reloaded phase folder gives the same logits; a short training run lowers the loss;
predict writes session 4's format and the logits file. Then with Qwen's own tokenizer.json and tokenizer_config.json at the
fixed revision, saved in models/qwen3_4b_base/: each marker is ID 198 and sits after its option, and every option comes after
the state; no special token is added and the pad ID is 151643; a state that contains "[MASK]" builds unchanged; the state is
cut from its end and the options are never cut."""
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1]
ROOT = SRC.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC / "train"))

import train_qwen_head as Q  # noqa: E402
from build_items import load_questions  # noqa: E402
from lora import LoRALinear, is_adapter  # noqa: E402
from qwen_items import build_qwen_items, build_qwen_sequence  # noqa: E402
from train_laya import collate, log_loss  # noqa: E402

LAYA_TOKENIZER = ROOT / "models" / "laya-base" / "tokenizer"
QWEN_TOKENIZER = ROOT / "models" / "qwen3_4b_base"
needs_laya = pytest.mark.skipif(not (LAYA_TOKENIZER / "tokenizer.json").exists(), reason="models/laya-base/tokenizer not present")
needs_qwen = pytest.mark.skipif(not (QWEN_TOKENIZER / "tokenizer.json").exists(), reason="Qwen's tokenizer not in models/qwen3_4b_base")


def tiny_base(folder):
    """A two-layer Qwen3 stand-in with Laya's tokenizer, saved as a model folder."""
    from transformers import AutoModel, AutoTokenizer, Qwen3Config
    tok = AutoTokenizer.from_pretrained(str(LAYA_TOKENIZER))
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16, max_position_embeddings=1024, pad_token_id=tok.pad_token_id)
    AutoModel.from_config(cfg).save_pretrained(str(folder))
    tok.save_pretrained(str(folder))
    return folder, tok


def made_up_records(n, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 3))
        rows.append({"id": f"p{seed}-{i}", "qid": "pubmedqa_answer", "label": label, "target": [float(label == k) for k in range(3)],
                     "state": {"question": f"Does treatment {i % 4} help?", "contexts": [{"section": "RESULTS", "text": f"word{label} " * 8}]}})
        label2 = int(rng.integers(0, 3))
        rows.append({"id": f"u{seed}-{i}", "qid": "readmit_window", "label": label2, "target": [float(label2 == k) for k in range(3)],
                     "state": {"age": f"[{50 + 10 * label2}-{60 + 10 * label2})", "time_in_hospital": int(rng.integers(1, 9))}})
    return rows


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory):
    if not (LAYA_TOKENIZER / "tokenizer.json").exists():
        pytest.skip("models/laya-base/tokenizer not present")
    folder, tok = tiny_base(tmp_path_factory.mktemp("base") / "tiny_qwen")
    questions = load_questions()
    train, _ = build_qwen_items(made_up_records(16, 1), questions, tok, 256)
    val, _ = build_qwen_items(made_up_records(8, 2), questions, tok, 256)
    return folder, tok, train, val


def forward_loss(model, tok, items):
    b = collate(items, tok.pad_token_id)
    logits, act = model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])
    return log_loss(logits.float(), b["target"], b["marker_mask"]) + 0.0 * act.float().sum(), logits


@needs_laya
def test_start_equals_base_and_only_new_parts_learn(stand_in):
    folder, tok, train, _ = stand_in
    base = Q.load_base(folder, torch.float32)
    reference = copy.deepcopy(base).eval()
    model, wrapped = Q.build_model(base, 20260927)
    model.eval()
    assert wrapped == 2 * 7
    ids = torch.tensor([train[0]["ids"][:40]])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a = reference(input_ids=ids, attention_mask=mask).last_hidden_state
        b = model.encoder.base(input_ids=ids, attention_mask=mask).last_hidden_state
    assert torch.allclose(a, b, atol=1e-6)
    for name, p in model.named_parameters():
        if name.startswith("encoder.base.") and not is_adapter(name):
            assert not p.requires_grad, name
        else:
            assert p.requires_grad, name
    assert all(isinstance(m, LoRALinear) for n, m in model.named_modules() if n.endswith(("q_proj", "down_proj")))
    groups = Q.param_groups(model, 5e-5, 1e-4)
    assert groups[0]["lr"] == 5e-5 and groups[1]["lr"] == 1e-4
    names = {id(p): n for n, p in model.named_parameters()}
    assert all(is_adapter(names[id(p)]) for p in groups[0]["params"])
    assert not any(is_adapter(names[id(p)]) for p in groups[1]["params"])
    assert any(names[id(p)].startswith("encoder.proj.") for p in groups[1]["params"])


@needs_laya
def test_freeze_steps_train_projection_and_head_only(stand_in):
    folder, tok, train, _ = stand_in
    model, _ = Q.build_model(Q.load_base(folder, torch.float32), 20260927)
    model.train()
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in Q.param_groups(model, 1e-3, 1e-3)], weight_decay=0.01)
    adapters = {n: p.detach().clone() for n, p in model.named_parameters() if is_adapter(n)}
    proj = model.encoder.proj.weight.detach().clone()
    model.encoder.freeze_base = True
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, tok, train[:4])
        loss.backward()
        opt.step()
    now = dict(model.named_parameters())
    assert all(torch.equal(adapters[n], now[n]) for n in adapters)
    assert not torch.equal(proj, model.encoder.proj.weight)
    model.encoder.freeze_base = False
    opt.zero_grad(set_to_none=True)
    loss, _ = forward_loss(model, tok, train[:4])
    loss.backward()
    opt.step()
    assert any(not torch.equal(adapters[n], now[n]) for n in adapters if ".lora_B." in n)


@needs_laya
def test_save_reload_same_logits_and_training_lowers_loss(stand_in, tmp_path):
    folder, tok, train, val = stand_in
    torch.manual_seed(1)
    model, _ = Q.build_model(Q.load_base(folder, torch.float32), 20260927)
    opt = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in Q.param_groups(model, 1e-3, 1e-3)], weight_decay=0.0)
    batch = train[:4]
    losses = []
    model.train()
    for _ in range(15):
        opt.zero_grad(set_to_none=True)
        loss, _ = forward_loss(model, tok, batch)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert np.mean(losses[-3:]) < np.mean(losses[:3]) - 0.3
    args = type("A", (), {"lora_r": 16, "lora_alpha": 32.0, "lora_dropout": 0.05, "max_len": 256, "seed": 20260927})
    Q.save_phase(tmp_path / "phase", model, args, folder, {"system": "tiny"})
    model.eval()
    with torch.no_grad():
        _, before = forward_loss(model, tok, val[:6])
    other, _ = Q.build_model(Q.load_base(folder, torch.float32), 7)
    Q.load_trainable(other, tmp_path / "phase" / "adapter.safetensors")
    other.eval()
    with torch.no_grad():
        _, after = forward_loss(other, tok, val[:6])
    assert torch.allclose(before, after, atol=1e-5)
    config = json.loads((tmp_path / "phase" / "config.json").read_text())
    assert config["adapters"]["r"] == 16 and config["projection"] == 1024 and (tmp_path / "phase" / "tokenizer.json").exists()


@needs_laya
def test_train_and_predict_commands(stand_in, tmp_path):
    folder, tok, train, val = stand_in
    torch.save(train, tmp_path / "train.pt")
    torch.save(val, tmp_path / "val.pt")
    out = tmp_path / "s_b1"
    Q.main(["train", "--base-dir", str(folder), "--train", str(tmp_path / "train.pt"), "--val", str(tmp_path / "val.pt"),
            "--out", str(out), "--system-name", "tiny_b1", "--freeze-steps", "1", "--keep", "best", "--eval-every", "2",
            "--micro-batch", "4", "--grad-accum", "2", "--max-len", "256", "--device", "cpu",
            "--predictions-dir", str(tmp_path / "preds"), "--logits-dir", str(tmp_path / "logits"),
            "--metrics", str(tmp_path / "m.json"), "--log", str(tmp_path / "log.csv")])
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["keep"] == "best" and m["lr_adapters"] == 5e-5 and not (out / "checkpoint_best").exists()
    assert (out / "adapter.safetensors").exists() and (out / "calibration.json").exists()
    records = tmp_path / "records"
    records.mkdir()
    rows = made_up_records(5, 3)
    for qid in ("pubmedqa_answer", "readmit_window"):
        with open(records / f"{qid}_test.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows if r["qid"] == qid)
    Q.main(["predict", "--base-dir", str(folder), "--phase-dir", str(out), "--system", "tiny_b1", "--records-dir", str(records),
            "--qids", "pubmedqa_answer", "readmit_window", "--splits", "test", "--out", str(tmp_path / "test_preds"),
            "--logits-dir", str(tmp_path / "test_logits"), "--device", "cpu"])
    for qid in ("pubmedqa_answer", "readmit_window"):
        with open(tmp_path / "test_preds" / f"tiny_b1-raw_{qid}_test.csv") as f:
            p = list(csv.reader(f))
        with open(tmp_path / "test_logits" / f"tiny_b1_{qid}_test_logits.csv") as f:
            z = list(csv.reader(f))
        assert p[0] == ["id", "label", "p0", "p1", "p2"] and z[0] == ["id", "label", "z0", "z1", "z2"] and len(p) == 6
        for rp, rz in zip(p[1:], z[1:]):
            logits = np.array([float(x) for x in rz[2:]])
            probs = np.exp(logits - logits.max())
            probs /= probs.sum()
            assert rp[:2] == rz[:2] and np.allclose(probs, [float(x) for x in rp[2:]], atol=2e-6)


@needs_qwen
def test_qwen_item_format():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(QWEN_TOKENIZER))
    assert tok.pad_token_id == 151643
    q = load_questions()["pubmedqa_answer"]
    state = {"question": "Is [MASK] safe?", "contexts": [{"section": "RESULTS", "text": "It was safe in 40 of 42 patients."}]}
    ids, markers, cut = build_qwen_sequence(tok, state, q, 768)
    assert not cut and len(markers) == 3 and all(ids[m] == 198 for m in markers)
    state_ids = tok(json.dumps(state, ensure_ascii=False), add_special_tokens=False)["input_ids"]
    assert ids[: len(state_ids)] == state_ids and "[MASK]" in tok.decode(ids[: len(state_ids)])
    assert markers[0] > len(state_ids) and markers == sorted(markers)
    text = tok.decode(ids[len(state_ids):])
    assert text.startswith("\n\nchoice question: ") and text.endswith("maybe: the reported results are inconclusive\n")
    for m, option in zip(markers, ("yes", "no", "maybe")):
        assert tok.decode(ids[m - 12: m]).rstrip().split(":")[0].split()[-1] in (option, "yes", "no", "maybe")
    assert tok.bos_token_id is None or ids[0] != tok.bos_token_id
    long_state = {"question": "Long?", "contexts": [{"section": "RESULTS", "text": "word " * 2000}]}
    ids2, markers2, cut2 = build_qwen_sequence(tok, long_state, q, 768)
    long_ids = tok(json.dumps(long_state, ensure_ascii=False), add_special_tokens=False)["input_ids"]
    tail = ids[len(state_ids):]
    assert cut2 and len(ids2) == 768 and ids2[-len(tail):] == tail and ids2[: 768 - len(tail)] == long_ids[: 768 - len(tail)]
    assert all(ids2[m] == 198 for m in markers2)

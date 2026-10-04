"""Tests for session 5's reader training (docs/prompt_session5.md, step 3) on a tiny ModernBERT stand-in (two layers,
Laya's tokenizer) with a matching tiny Laya-format folder: build_mlm_init's checks and tensors; 20 steps on CPU make
the loss fall; the checkpoints load; src/train/assemble_reader.py accepts the output; a run resumed from resume.pt gives
the same losses for its next 5 steps as the run that was not interrupted; the schedule and the masking."""
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "train"))
sys.path.insert(0, str(SRC / "corpus"))
import assemble_reader  # noqa: E402
import build_mlm_init as B  # noqa: E402
import pretrain_reader as PR  # noqa: E402

TOKENIZER = SRC.parent / "models" / "tokenizers" / "laya"
needs_tokenizer = pytest.mark.skipif(not (TOKENIZER / "tokenizer.json").exists(), reason="Laya tokenizer copy not present")
CLS, SEP, PAD, MASK = 50281, 50282, 50283, 50284


def tiny_config(**changes):
    from transformers import ModernBertConfig

    base = dict(vocab_size=50368, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
                max_position_embeddings=1024, global_attn_every_n_layers=2, local_attention=64, pad_token_id=PAD,
                cls_token_id=CLS, sep_token_id=SEP, bos_token_id=CLS, eos_token_id=SEP)
    base.update(changes)
    return ModernBertConfig(**base)


def make_tiny_models(tmp):
    """A tiny ModernBERT-large stand-in and a matching tiny Laya-format folder (encoder in float16, as Laya stores it)."""
    from safetensors.torch import save_file
    from transformers import AutoModel, ModernBertForMaskedLM

    torch.manual_seed(0)
    cfg = tiny_config()
    mb_dir = tmp / "modernbert"
    ModernBertForMaskedLM(cfg).save_pretrained(str(mb_dir))
    shutil.copy(TOKENIZER / "tokenizer.json", mb_dir / "tokenizer.json")
    laya_dir = tmp / "laya"
    (laya_dir / "tokenizer").mkdir(parents=True)
    encoder = AutoModel.from_config(cfg)
    encoder.config.save_pretrained(str(laya_dir / "encoder"))
    tensors = {"encoder." + k: v.to(torch.float16).contiguous() for k, v in encoder.state_dict().items()}
    tensors.update({"head.layers.0.linear1.weight": torch.zeros(4, 4), "temperature": torch.ones(3)})
    save_file(tensors, str(laya_dir / "model.safetensors"))
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(TOKENIZER / name, laya_dir / "tokenizer" / name)
    (laya_dir / "rl_agent_config.json").write_text(json.dumps({"head_layers": 2}))
    return mb_dir, laya_dir


def make_blocks(root, sources=("abstracts",), n_train=200, n_heldout=4, seed=0):
    """Learnable made-up blocks: [CLS], a repeating cycle of 10 token ids from a random start, [SEP]."""
    rng = np.random.default_rng(seed)
    for k, name in enumerate(sources):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        for split, n in (("train", n_train), ("heldout", n_heldout)):
            starts = rng.integers(0, 10, size=n)
            body = (1000 + 20 * k + (starts[:, None] + np.arange(1022)[None, :]) % 10).astype(np.uint16)
            blocks = np.concatenate([np.full((n, 1), CLS, np.uint16), body, np.full((n, 1), SEP, np.uint16)], axis=1)
            np.save(d / f"{split}.npy", blocks)
        np.save(d / "perm.npy", rng.permutation(n_train))
        (d / "counts.json").write_text(json.dumps({"source": name, "train_blocks": n_train}))


def write_mixes(path, tokens):
    path.write_text(yaml.safe_dump({"mixes": {"A": {"shares": {"abstracts": 100}, "tokens": tokens}}}), encoding="utf-8")
    return path


def run_args(init, blocks, mixes, out, **extra):
    args = ["--mix", "A", "--blocks", str(blocks), "--mixes", str(mixes), "--out", str(out), "--log", str(out / "log.csv"),
            "--micro-batch", "2", "--grad-accum", "2", "--peak-lr", "3e-3", "--device", "cpu", "--eval-every", "10",
            "--log-every", "1"]
    if init:
        args += ["--init", str(init)]
    for key, value in extra.items():
        args += [f"--{key.replace('_', '-')}", str(value)]
    return args


def test_schedule_numbers_for_the_real_runs():
    tps = 256 * 1024
    assert PR.schedule(tps, run_tokens=500_000_000) == (1908, 57, 1717)
    assert PR.schedule(tps, run_tokens=500_000_000, total_tokens=2_000_000_000) == (7630, 57, 6867)
    assert PR.schedule(tps, total_tokens=2_000_000_000, warmup=57, steps=1908) == (7630, 57, 6867)
    assert PR.schedule(tps, steps=math.ceil(3 * 19_000 / 256)) == (223, 6, 200)
    s, w, d = 1908, 57, 1717
    assert PR.learning_rate(0, 1e-4, s, w, d) == pytest.approx(1e-4 / 57) and PR.learning_rate(56, 1e-4, s, w, d) == pytest.approx(1e-4)
    assert PR.learning_rate(1716, 1e-4, s, w, d) == 1e-4 and PR.learning_rate(1717, 1e-4, s, w, d) == pytest.approx(1e-4)
    assert PR.learning_rate(1907, 1e-4, s, w, d) == pytest.approx(1e-4 / 191)


def test_masking_rates_and_determinism():
    ids = torch.full((8, 1024), 1234, dtype=torch.long)
    ids[:, 0], ids[:, -1] = CLS, SEP
    a, la = PR.mask_blocks(ids, step=3, micro=1, seed=7, special_ids=[CLS, SEP, PAD], mask_id=MASK, vocab_size=50368)
    b, lb = PR.mask_blocks(ids, step=3, micro=1, seed=7, special_ids=[CLS, SEP, PAD], mask_id=MASK, vocab_size=50368)
    assert torch.equal(a, b) and torch.equal(la, lb)
    selected = la != -100
    assert not selected[:, 0].any() and not selected[:, -1].any()
    assert abs(float(selected.float().mean()) * 1024 / 1022 - 0.30) < 0.01
    masked = a[selected] == MASK
    assert abs(float(masked.float().mean()) - 0.8) < 0.02
    c, _ = PR.mask_blocks(ids, step=4, micro=1, seed=7, special_ids=[CLS, SEP, PAD], mask_id=MASK, vocab_size=50368)
    assert not torch.equal(a, c)


@needs_tokenizer
def test_build_mlm_init_fills_and_checks(tmp_path):
    from safetensors import safe_open
    from transformers import ModernBertForMaskedLM

    mb_dir, laya_dir = make_tiny_models(tmp_path)
    report = B.build(laya_dir, mb_dir, tmp_path / "init")
    assert report["encoder_tensors_from_laya"] == 14
    model = ModernBertForMaskedLM.from_pretrained(str(tmp_path / "init"))
    with safe_open(str(laya_dir / "model.safetensors"), framework="pt") as laya, safe_open(str(mb_dir / "model.safetensors"), framework="pt") as mb:
        state = model.state_dict()
        for key in [k for k in state if k.startswith("model.")]:
            assert torch.equal(state[key], laya.get_tensor("encoder." + key[len("model."):]).to(torch.float32))
        for key in B.HEAD_TENSORS:
            assert torch.equal(state[key], mb.get_tensor(key).to(torch.float32))
    assert model.decoder.weight.data_ptr() == model.model.embeddings.tok_embeddings.weight.data_ptr()
    assert model.config.sparse_prediction is True
    bad = json.loads((laya_dir / "encoder" / "config.json").read_text())
    bad["intermediate_size"] = 96
    (laya_dir / "encoder" / "config.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        B.build(laya_dir, mb_dir, tmp_path / "init_bad_config")
    make_tiny_models(tmp_path / "second")
    tok = json.loads((tmp_path / "second" / "modernbert" / "tokenizer.json").read_text())
    tok["model"]["vocab"]["zzz_extra_token"] = 99999
    (tmp_path / "second" / "modernbert" / "tokenizer.json").write_text(json.dumps(tok))
    with pytest.raises(ValueError):
        B.build(tmp_path / "second" / "laya", tmp_path / "second" / "modernbert", tmp_path / "init_bad_vocab")
    # the same vocabulary with its merges written in the other JSON form ("a b" against ["a", "b"]) is accepted
    make_tiny_models(tmp_path / "third")
    tok = json.loads((tmp_path / "third" / "modernbert" / "tokenizer.json").read_text())
    merges = tok["model"]["merges"]
    tok["model"]["merges"] = [" ".join(m) for m in merges] if isinstance(merges[0], list) else [m.split(" ", 1) for m in merges]
    (tmp_path / "third" / "modernbert" / "tokenizer.json").write_text(json.dumps(tok))
    assert B.vocabulary_digest(tmp_path / "third" / "modernbert" / "tokenizer.json") == B.vocabulary_digest(
        tmp_path / "third" / "laya" / "tokenizer" / "tokenizer.json")
    assert B.build(tmp_path / "third" / "laya", tmp_path / "third" / "modernbert", tmp_path / "init_other_merges_form")[
        "encoder_tensors_from_laya"] == 14


@needs_tokenizer
def test_twenty_steps_on_cpu_loss_falls_and_outputs_load(tmp_path):
    from transformers import ModernBertForMaskedLM

    mb_dir, laya_dir = make_tiny_models(tmp_path)
    B.build(laya_dir, mb_dir, tmp_path / "init")
    make_blocks(tmp_path / "blocks")
    mixes = write_mixes(tmp_path / "mixes.yaml", tokens=20 * 4 * 1024)
    out = tmp_path / "run"
    losses = PR.run(run_args(tmp_path / "init", tmp_path / "blocks", mixes, out))
    assert sorted(losses) == list(range(20))
    assert np.mean([losses[i] for i in range(17, 20)]) < np.mean([losses[i] for i in range(3)]) - 0.5
    for folder in ("final", "decay_start"):
        ModernBertForMaskedLM.from_pretrained(str(out / folder))
    assert (out / "decay_start" / "resume.pt").exists() and json.loads((out / "decay_start" / "checkpoint.json").read_text())["step"] == 18
    assert assemble_reader.assemble(laya_dir, out / "final", tmp_path / "assembled") == 14
    held = json.loads((out / "heldout_losses.json").read_text())
    assert [r["step"] for r in held] == [10, 20] and held[1]["loss"]["abstracts"] < held[0]["loss"]["abstracts"] + 1.0
    with open(out / "log.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0][:4] == ["step", "tokens", "lr", "loss"] and len(rows) == 21


@needs_tokenizer
def test_resumed_run_matches_the_uninterrupted_run(tmp_path):
    mb_dir, laya_dir = make_tiny_models(tmp_path)
    B.build(laya_dir, mb_dir, tmp_path / "init")
    make_blocks(tmp_path / "blocks")
    t1, t2 = 20 * 4 * 1024, 30 * 4 * 1024
    mixes = write_mixes(tmp_path / "mixes.yaml", tokens=t1)
    whole = PR.run(run_args(tmp_path / "init", tmp_path / "blocks", mixes, tmp_path / "whole", total_tokens=t2))
    first = PR.run(run_args(tmp_path / "init", tmp_path / "blocks", mixes, tmp_path / "first"))
    resumed = PR.run(run_args(None, tmp_path / "blocks", mixes, tmp_path / "resumed", total_tokens=t2,
                              resume=tmp_path / "first" / "decay_start" / "resume.pt"))
    assert sorted(whole) == list(range(30)) and sorted(resumed) == list(range(18, 30))
    for step in range(18):
        assert first[step] == pytest.approx(whole[step], abs=1e-6)
    for step in range(18, 23):
        assert resumed[step] == pytest.approx(whole[step], abs=1e-6)


@needs_tokenizer
def test_task_round_draws_every_block_of_every_pass_with_a_short_last_step(tmp_path):
    """Correction 5: three passes over 7 task blocks are 21 blocks; at 4 blocks a step that is 6 steps (rounded up), the
    last taking the 1 block that remains, and the tokens reported are the tokens drawn."""
    mb_dir, laya_dir = make_tiny_models(tmp_path)
    B.build(laya_dir, mb_dir, tmp_path / "init")
    make_blocks(tmp_path / "blocks", sources=("task",), n_train=7, n_heldout=2)
    mixes = tmp_path / "mixes.yaml"
    mixes.write_text(yaml.safe_dump({"mixes": {"T": {"shares": {"task": 100}, "passes": 3}}}), encoding="utf-8")
    out = tmp_path / "task"
    args = run_args(tmp_path / "init", tmp_path / "blocks", mixes, out)
    args[args.index("--mix") + 1] = "T"
    losses = PR.run(args)
    assert sorted(losses) == list(range(6)) and all(math.isfinite(v) for v in losses.values())
    throughput = json.loads((out / "throughput.json").read_text())
    assert throughput["blocks_drawn"] == {"task": 21} and throughput["tokens_run"] == 21 * 1024
    with open(out / "log.csv") as f:
        rows = list(csv.reader(f))
    assert rows[-1][0] == "6" and rows[-1][1] == str(21 * 1024)
    assert json.loads((out / "final" / "checkpoint.json").read_text())["tokens"] == 21 * 1024

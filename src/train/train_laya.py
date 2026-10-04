"""Supervised fine-tuning of Laya's typed decision head and encoder on one device.

Method (fixed for this project): soft cross-entropy, the log loss, against the target distribution over
options, a strictly proper scoring rule. AdamW (encoder 2.5e-5, head 1.0e-4, weight decay 0.01), cosine
schedule, gradient clipping at 1.0, gradient checkpointing, float16 autocast with a loss scaler on CUDA and
float32 on MPS and CPU.

Checkpoint selection: the checkpoint with the lowest mean of the per-question validation log losses is kept
(each question weighted equally; only validation items marked calibrate, which is every item unless the item says
calibrate false). A question is the item's qid; items without a qid (the session 3 practice items) use their
question type.

Calibration: after training, one temperature per question is fitted on the calibrate validation items (LBFGS on
the log loss, clamped to [0.1, 10]). --calibration-mode separate (the default; every model after the practice run)
writes the temperatures to <out>/calibration.json and leaves temperature 1.0 in rl_agent_config.json and in the
weights. --calibration-mode config (the session 3 practice model only) writes them into rl_agent_config.json.

Batching: --batching mixed (default) shuffles all items together; --batching by-question shuffles each question's
items, cuts them into micro-batches of one question each, and shuffles the micro-batches, so that short and long
sequences are not padded together (docs/prompt_session4.md, correction 2).

Outputs: <out>/ (model.safetensors in float16, encoder/config.json, tokenizer/, rl_agent_config.json, and in
separate mode calibration.json); the metrics JSON; the per-step log CSV; and with --predictions-dir, one CSV per
question and split (validation, test): id, label, p0, p1, ... for <system> (calibrated) and <system>-raw.

Session 5 options (docs/prompt_session5.md, DECISION-LAYER METHOD; both default to session 4's behavior):
--maybe-weight auto: each PubMedQA training record labeled maybe gets loss weight w and every other record 1, and a
micro-batch's loss is the weighted sum divided by the number of records in the micro-batch. w makes maybe's weighted
share of the PubMedQA training records equal its share among the pqa_labeled training records: w is the number of
PubMedQA training records that are not maybe divided by the number of pqa_labeled training records that are not maybe
(20,924 / 924 = 22.65 with session 4's records). It is computed from the items (which carry each record's source; see
src/train/build_items.py) and written to the metrics JSON. --maybe-weight none (the default) is session 4's loss.
--no-test: the test split is never read, and no test prediction or test metric is written.

Session 7 options (docs/prompt_session7.md, ARM A; both default to the earlier behavior):
--keep best|last: best (the default) keeps the checkpoint with the lowest selection score, as before; last keeps the
final weights, saves no best checkpoint, and still logs each validation (the phases before the last one, whose
checkpoints are never chosen on validation).
--logits-dir DIR: also writes each prediction row's raw logits, before any temperature, at full precision, to
DIR/<system>_<qid>_<split>_logits.csv (id, label, z0, z1, ...), for the splits whose predictions are written.

Usage: python src/train/train_laya.py --train data/processed/practice/emotion_train.pt \
    --val data/processed/practice/emotion_validation.pt --test data/processed/practice/emotion_test.pt
"""
import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from laya.common import build_model, ece_score
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
PRACTICE = ROOT / "data" / "processed" / "practice"
PASS_MARK = 0.573  # Laya's reported zero-shot accuracy on DAIR Emotion (practice model only)
TYPE_NAMES = ("choice", "score", "noul")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def item_key(item):
    """The question an item belongs to: its qid, or its question type for items without one."""
    return item.get("qid") or TYPE_NAMES[int(item["qtype"])]


def collate(items, pad_id):
    """Pad a batch of items (ids, markers, qtype, target) into tensors."""
    n, length = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    att = torch.zeros((n, length), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]), "label": torch.tensor([it["label"] for it in items]),
    }


def log_loss(logits, target, mask):
    """Soft cross-entropy per row, averaged: -sum_k target_k log softmax(logits)_k over the valid options."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
    return -(target * logp).sum(-1).mean()


MAYBE = 2  # PubMedQA's option index for maybe (src/tasks/pubmedqa.py: yes 0, no 1, maybe 2)
PUBMEDQA = "pubmedqa_answer"


def maybe_weight(items):
    """w for --maybe-weight auto: PubMedQA training records that are not maybe, divided by pqa_labeled training records
    that are not maybe; and the counts behind it."""
    pubmedqa = [it for it in items if it.get("qid") == PUBMEDQA]
    if any(it.get("source") not in ("pqa_labeled", "pqa_artificial") for it in pubmedqa):
        raise SystemExit("--maybe-weight auto needs each PubMedQA item's source; rebuild the items with src/train/build_items.py")
    other = sum(1 for it in pubmedqa if int(it["label"]) != MAYBE)
    labeled_not_maybe = sum(1 for it in pubmedqa if it["source"] == "pqa_labeled" and int(it["label"]) != MAYBE)
    counts = {"pubmedqa_records": len(pubmedqa), "maybe_records": len(pubmedqa) - other, "other_records": other,
              "pqa_labeled_not_maybe": labeled_not_maybe}
    return other / labeled_not_maybe, counts


def item_weight(item, w):
    return float(w) if w is not None and item.get("qid") == PUBMEDQA and int(item["label"]) == MAYBE else 1.0


def weighted_log_loss(logits, target, mask, weights):
    """Soft cross-entropy per row times the row's weight, summed and divided by the number of rows."""
    logp = torch.log_softmax(logits.masked_fill(~mask, -1e4), dim=-1)
    return (-(target * logp).sum(-1) * weights).sum() / logits.shape[0]


def fit_temperature(logit_rows, target_rows):
    """One temperature for a set of (logits, target) rows, by LBFGS on the log loss; clamped to [0.1, 10]."""
    if len(logit_rows) < 10:
        return 1.0
    kmax = max(len(z) for z in logit_rows)
    z_pad = torch.full((len(logit_rows), kmax), -1e4)
    t_pad = torch.zeros((len(logit_rows), kmax))
    for i, (z, t) in enumerate(zip(logit_rows, target_rows)):
        z_pad[i, : len(z)] = torch.tensor(z, dtype=torch.float32)
        t_pad[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = -(t_pad * torch.log_softmax(z_pad / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.clamp(log_t.exp(), 0.1, 10.0).item())


def run(model, items, pad_id, device, batch_size, autocast_dtype):
    """Raw logits for every item, as rows: key, id, label, logits, target, calibrate."""
    model.eval()
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            batch = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                  batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            logits = logits.float().cpu()
            for i, it in enumerate(chunk):
                k = len(it["markers"])
                rows.append({
                    "key": item_key(it), "id": str(it.get("id", start + i)), "label": int(it["label"]),
                    "logits": logits[i, :k].tolist(), "target": [float(x) for x in it["target"][:k]],
                    "calibrate": bool(it.get("calibrate", True)),
                })
    model.train()
    return rows


def probabilities(rows, temperature=None):
    """Softmax of each row's logits, divided first by its question's temperature when one is given."""
    out = []
    for r in rows:
        t = temperature.get(r["key"], 1.0) if temperature else 1.0
        out.append(torch.softmax(torch.tensor(r["logits"]) / t, -1).numpy())
    return out


def summarize(rows, probs, keep=lambda r: True):
    """Accuracy, log loss and ECE over the rows that keep() accepts."""
    idx = [i for i, r in enumerate(rows) if keep(r)]
    if not idx:
        return {"n": 0}
    conf = np.array([float(probs[i].max()) for i in idx])
    correct = np.array([int(probs[i].argmax()) == rows[i]["label"] for i in idx])
    losses = [float(-(np.array(rows[i]["target"]) * np.log(np.clip(probs[i], 1e-9, None))).sum()) for i in idx]
    return {"n": len(idx), "accuracy": round(float(correct.mean()), 4), "log_loss": round(float(np.mean(losses)), 4),
            "ece": round(ece_score(conf, correct), 4)}


def per_question(rows, probs, calibrate_only=False):
    keys = sorted({r["key"] for r in rows})
    return {k: summarize(rows, probs, lambda r, k=k: r["key"] == k and (r["calibrate"] or not calibrate_only)) for k in keys}


def selection_score(rows, probs):
    """Mean of the per-question log losses over calibrate rows (each question weighted equally)."""
    scores = [m["log_loss"] for m in per_question(rows, probs, calibrate_only=True).values() if m["n"]]
    return float(np.mean(scores)) if scores else float("inf")


def fit_temperatures(rows):
    temperature = {}
    for key in sorted({r["key"] for r in rows}):
        sel = [r for r in rows if r["key"] == key and r["calibrate"]]
        temperature[key] = fit_temperature([r["logits"] for r in sel], [r["target"] for r in sel]) if sel else 1.0
    return temperature


def write_predictions(directory, system, split, rows, probs):
    """One CSV per question: id, label, p0, p1, ... (the step 5 format of docs/prompt_session4.md), probabilities printed
    with 9 significant digits so that a small probability never prints as 0 (docs/prompt_session7.md, CALIBRATION;
    6 decimals before session 7)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in sorted({r["key"] for r in rows}):
        path = directory / f"{system}_{key}_{split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            idx = [i for i, r in enumerate(rows) if r["key"] == key]
            k = len(rows[idx[0]]["logits"])
            w.writerow(["id", "label"] + [f"p{j}" for j in range(k)])
            for i in idx:
                w.writerow([rows[i]["id"], rows[i]["label"]] + [f"{x:.9g}" for x in probs[i]])
        paths.append(path)
    return paths


def write_logits(directory, system, split, rows):
    """One CSV per question: id, label, z0, z1, ... with each row's raw logits (no temperature) at full precision."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for key in sorted({r["key"] for r in rows}):
        path = directory / f"{system}_{key}_{split}_logits.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            idx = [i for i, r in enumerate(rows) if r["key"] == key]
            k = len(rows[idx[0]]["logits"])
            w.writerow(["id", "label"] + [f"z{j}" for j in range(k)])
            for i in idx:
                w.writerow([rows[i]["id"], rows[i]["label"]] + [repr(float(x)) for x in rows[i]["logits"]])
        paths.append(path)
    return paths


def reinit_head(model, seed):
    """Fresh random initialization (PyTorch defaults, seeded) of every parameter outside the encoder."""
    torch.manual_seed(seed)
    n = 0
    for part in (model.head, model.type_emb, model.scorer, model.act_head):
        for module in part.modules():
            for name in ("reset_parameters", "_reset_parameters"):
                if hasattr(module, name) and callable(getattr(module, name)):
                    getattr(module, name)()
                    n += 1
                    break
    return n


def micro_batches(items, micro_batch, rng, by_question=False):
    """Lists of item indices, one per micro-batch, in training order for one epoch."""
    if not by_question:
        order = list(range(len(items)))
        rng.shuffle(order)
        return [order[i : i + micro_batch] for i in range(0, len(order), micro_batch)]
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(item_key(it), []).append(i)
    batches = []
    for key in sorted(groups):
        idx = groups[key]
        rng.shuffle(idx)
        batches.extend(idx[i : i + micro_batch] for i in range(0, len(idx), micro_batch))
    rng.shuffle(batches)
    return batches


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--train", type=Path, default=PRACTICE / "emotion_train.pt")
    parser.add_argument("--val", type=Path, default=PRACTICE / "emotion_validation.pt")
    parser.add_argument("--test", type=Path, default=PRACTICE / "emotion_test.pt")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models" / "laya-base")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "practice-laya-emotion-v0")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--micro-batch", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-len", type=int, default=None, help="max_len recorded in the saved config (items are built with it)")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--eval-every", type=int, default=0, help="optimizer steps between validations; 0 = once per epoch")
    parser.add_argument("--lr-encoder", type=float, default=2.5e-5)
    parser.add_argument("--lr-head", type=float, default=1.0e-4)
    parser.add_argument("--autocast", choices=("float16", "bfloat16"), default="float16",
                        help="mixed-precision type on CUDA; float16 uses a loss scaler (the T4 has no bfloat16); bfloat16 needs none")
    parser.add_argument("--reinit-head", action="store_true",
                        help="start the decision head (head layers, type embedding, scorer, action head) from a fresh random initialization, seeded, instead of the model folder's head")
    parser.add_argument("--freeze-encoder-steps", type=int, default=0,
                        help="optimizer steps at the start during which only the decision head trains (the encoder is run but not updated); 0 = none")
    parser.add_argument("--calibration-mode", choices=("separate", "config"), default="separate")
    parser.add_argument("--batching", choices=("mixed", "by-question"), default="mixed")
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--system-name", default=None, help="name used in prediction file names (default: the --out folder name)")
    parser.add_argument("--maybe-weight", choices=("none", "auto"), default="none",
                        help="auto: PubMedQA maybe records weighted by w (session 5); none: session 4's loss")
    parser.add_argument("--no-test", action="store_true", help="never read the test split; write no test prediction or metric")
    parser.add_argument("--keep", choices=("best", "last"), default="best",
                        help="best: the checkpoint with the lowest selection score (session 4); last: the final weights (session 7's earlier phases)")
    parser.add_argument("--logits-dir", type=Path, default=None,
                        help="also write each prediction row's raw logits to DIR/<system>_<qid>_<split>_logits.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "results" / "practice" / "emotion_metrics.json")
    parser.add_argument("--log", type=Path, default=ROOT / "results" / "practice" / "emotion_train_log.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    autocast_dtype = (torch.bfloat16 if args.autocast == "bfloat16" else torch.float16) if device.type == "cuda" else None
    system = args.system_name or args.out.name
    print(f"device: {device}, torch {torch.__version__}, calibration mode {args.calibration_mode}")

    config = json.loads((args.model_dir / "rl_agent_config.json").read_text(encoding="utf-8"))
    config["gradient_checkpointing"] = True
    if args.max_len:
        config["max_len"] = args.max_len
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir / "tokenizer"))
    model = build_model(config, encoder_dir=str(args.model_dir / "encoder"))
    model.load_state_dict(load_file(str(args.model_dir / "model.safetensors")), strict=True)
    if args.reinit_head:
        reinit_head(model, args.seed)
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head_checkpointing = True
    model.to(device)
    model.train()

    train_items = torch.load(args.train, weights_only=False)
    val_items = torch.load(args.val, weights_only=False)
    test_items = [] if args.no_test else torch.load(args.test, weights_only=False)
    if args.max_rows:
        train_items = train_items[: args.max_rows]
        val_items = val_items[: max(200, args.max_rows // 8)]
        test_items = test_items[: max(200, args.max_rows // 8)]
    pad_id = tokenizer.pad_token_id
    eval_batch = args.micro_batch * 2
    weight, weight_counts = maybe_weight(train_items) if args.maybe_weight == "auto" else (None, None)
    if weight is not None:
        print(f"maybe weight w = {weight:.4f} ({weight_counts})")

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([{"params": enc_params, "lr": args.lr_encoder}, {"params": head_params, "lr": args.lr_head}], weight_decay=0.01)
    n_micro = len(micro_batches(train_items, args.micro_batch, random.Random(0), args.batching == "by-question"))
    steps_per_epoch = math.ceil(n_micro / args.grad_accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=autocast_dtype == torch.float16)

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "loss", "lr_encoder", "elapsed_seconds"])
    best_dir = args.out / "checkpoint_best"
    history, best_val, last_val = [], float("inf"), float("inf")
    step, sequences, t0 = 0, 0, time.time()
    train_seconds = 0.0

    def validate(epoch):
        nonlocal best_val, last_val
        rows = run(model, val_items, pad_id, device, eval_batch, autocast_dtype)
        probs = probabilities(rows)
        score = selection_score(rows, probs)
        metrics = summarize(rows, probs)
        metrics.update({"selection_log_loss": round(score, 4), "per_question": per_question(rows, probs, calibrate_only=True),
                        "epoch": epoch, "step": step, "elapsed_seconds": round(time.time() - t0, 1)})
        history.append(metrics)
        print(f"validation after epoch {epoch}, step {step}: {metrics}")
        last_val = score
        if score < best_val:
            best_val = score
            if args.keep == "last":
                return
            best_dir.mkdir(parents=True, exist_ok=True)
            save_file({k: v.detach().to(torch.float16).contiguous().cpu() for k, v in model.state_dict().items()}, str(best_dir / "model.safetensors"))

    for epoch in range(1, args.epochs + 1):
        batches = micro_batches(train_items, args.micro_batch, random.Random(args.seed + epoch), args.batching == "by-question")
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for b, batch_idx in enumerate(batches):
            chunk = [train_items[i] for i in batch_idx]
            batch = collate(chunk, pad_id)
            t_batch = time.time()
            with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device),
                                    detach_encoder=step < args.freeze_encoder_steps)
            if weight is None:
                loss = log_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device)) + 0.0 * act.float().sum()
            else:
                weights = torch.tensor([item_weight(it, weight) for it in chunk], dtype=torch.float32, device=device)
                loss = weighted_log_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device),
                                         weights) + 0.0 * act.float().sum()
            scaler.scale(loss / args.grad_accum).backward()
            micro += 1
            sequences += len(chunk)
            if micro % args.grad_accum == 0 or b == len(batches) - 1:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                log_writer.writerow([epoch, step, round(float(loss.detach()), 5), scheduler.get_last_lr()[0], round(time.time() - t0, 1)])
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {float(loss.detach()):.4f} ({sequences / max(1e-6, train_seconds):.1f} sequences/s)")
                if args.eval_every and step % args.eval_every == 0:
                    validate(epoch)
            train_seconds += time.time() - t_batch
        validate(epoch)
    log_file.close()

    if args.keep == "best" and best_dir.exists():
        model.load_state_dict({k: v.to(torch.float32) for k, v in load_file(str(best_dir / "model.safetensors")).items()}, strict=True)
        model.to(device)
    val_rows = run(model, val_items, pad_id, device, eval_batch, autocast_dtype)
    temperature = fit_temperatures(val_rows)
    test_rows = [] if args.no_test else run(model, test_items, pad_id, device, eval_batch, autocast_dtype)
    val_raw, test_raw = probabilities(val_rows), probabilities(test_rows)
    val_cal, test_cal = probabilities(val_rows, temperature), probabilities(test_rows, temperature)

    args.out.mkdir(parents=True, exist_ok=True)
    config["fine_tuned"] = True
    config["model_name"] = args.out.name
    config.pop("temperature_by_options", None)
    if args.calibration_mode == "config":
        config["temperature"] = [temperature.get(name, 1.0) for name in TYPE_NAMES]
    else:
        config["temperature"] = [1.0, 1.0, 1.0]
        if hasattr(model, "temperature"):
            model.temperature.fill_(1.0)
        calibration = {
            "method": "temperature scaling, one temperature per question",
            "fitted_on": "validation items marked calibrate",
            "usage": "divide each question's logits by its temperature before the softmax",
            "temperature": {k: round(v, 4) for k, v in temperature.items()},
        }
        (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    save_file({k: v.detach().to(torch.float16).contiguous().cpu() for k, v in model.state_dict().items()}, str(args.out / "model.safetensors"))
    model.encoder.config.save_pretrained(str(args.out / "encoder"))
    tokenizer.save_pretrained(str(args.out / "tokenizer"))
    (args.out / "rl_agent_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if best_dir.exists():
        shutil.rmtree(best_dir)

    if args.predictions_dir:
        splits = (("validation", val_rows, val_raw, val_cal),) if args.no_test else (
            ("validation", val_rows, val_raw, val_cal), ("test", test_rows, test_raw, test_cal))
        for split, rows, raw, cal in splits:
            write_predictions(args.predictions_dir, system, split, rows, cal)
            write_predictions(args.predictions_dir, f"{system}-raw", split, rows, raw)
    if args.logits_dir:
        for split, rows in ((("validation", val_rows),) if args.no_test else (("validation", val_rows), ("test", test_rows))):
            write_logits(args.logits_dir, system, split, rows)

    metrics = {
        "system": system, "calibration_mode": args.calibration_mode, "batching": args.batching,
        "reinit_head": args.reinit_head, "freeze_encoder_steps": args.freeze_encoder_steps, "autocast": str(autocast_dtype), "eval_every": args.eval_every, "lr_encoder": args.lr_encoder, "lr_head": args.lr_head,
        "device": str(device), "torch": torch.__version__, "base_model": str(args.model_dir),
        "rows": {"train": len(train_items), "validation": len(val_items), "test": len(test_items)},
        "epochs": args.epochs, "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "optimizer_steps": step,
        "max_len": config.get("max_len"),
        "elapsed_seconds": round(time.time() - t0, 1), "train_seconds": round(train_seconds, 1),
        "sequences_per_second": round(sequences / max(1e-6, train_seconds), 2),
        "validation_history": history, "best_validation_selection_log_loss": round(best_val, 4),
        "best_validation_selection_log_loss_exact": best_val,
        "keep": args.keep,
        "kept_validation_selection_log_loss_exact": best_val if args.keep == "best" else last_val,
        "temperature": {k: round(v, 4) for k, v in temperature.items()},
        "maybe_weight": {"mode": args.maybe_weight, "w": weight, **(weight_counts or {})},
        "no_test": args.no_test,
        "validation_final": summarize(val_rows, val_raw),
    }
    if args.no_test:
        metrics["test"] = "not run (--no-test)"
    else:
        metrics.update({
            "test_before_temperature": summarize(test_rows, test_raw), "test_after_temperature": summarize(test_rows, test_cal),
            "test_per_question_before_temperature": per_question(test_rows, test_raw),
            "test_per_question_after_temperature": per_question(test_rows, test_cal),
        })
    if args.calibration_mode == "config" and not args.no_test:
        metrics["pass_mark"] = PASS_MARK
        metrics["passed"] = metrics["test_before_temperature"]["accuracy"] > PASS_MARK
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"saved model to {args.out}; metrics to {args.metrics}")


if __name__ == "__main__":
    sys.exit(main())

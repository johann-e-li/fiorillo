"""Arm B: Laya's decision head on Qwen3-4B-Base with LoRA adapters (docs/prompt_session7.md, ARM B).

Model. Qwen3's decoder without its language-model head (transformers' AutoModel) in bfloat16, frozen, with LoRA
adapters (src/train/lora.py: rank 16, alpha 32, dropout 0.05, on q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj
and down_proj of every layer). A reader wrapper (QwenReader) runs the adapted base, then a projection,
torch.nn.Linear(hidden size, 1024) in float32, on the base's last hidden state (after its final norm), and returns it
as last_hidden_state; its config gives hidden size 1,024, so laya.common.DecisionModel builds Laya's head on it
unchanged (two layers, 16 heads, feed-forward 4,096, dropout 0.1, norm first; the type embedding; the scorer; the
action head, built and not trained, as in v0.1). The adapters, the projection and the head start from one seeded
initialization (--seed). Head-only steps (--freeze-steps): the wrapper runs the base and adapters under
torch.no_grad(), and the projection and head learn; DecisionModel's detach_encoder is not used.

Training. train_laya.py's method and functions: soft cross-entropy; micro-batches of one question at a time
(--batching by-question); AdamW with weight decay 0.01, learning rate --lr-adapters for the adapters (parameters named
lora_A or lora_B) and --lr-head for everything else that trains; cosine schedule to 1e-6 over the phase; gradient
clipping at 1.0; bfloat16 autocast on CUDA (float32 on the CPU, for the tests); validation every --eval-every optimizer
steps and at the end of each epoch; --keep best keeps the checkpoint with the lowest selection score (session 4's),
--keep last the final weights. One temperature per question is fitted on the calibrate validation items and written to
calibration.json. --init <phase folder> starts from a previous phase's adapters, projection and head.

Outputs of a phase (--out): adapter.safetensors (every trainable parameter, float32), config.json (the base and its
revision, the adapter settings, max_len, the item format and the project's git commit, read from COMMIT.txt at the
repository root when it exists), the tokenizer files, calibration.json; the metrics JSON (--metrics) and the log CSV
(--log); with --predictions-dir and --logits-dir, validation predictions (raw and with the temperature) and logits as
train_laya.py writes them. The base's weights are never copied.

Subcommands:
  train    one phase (the options above).
  pilot    the pilot of docs/prompt_session7.md (THE PILOT AND THE COST RULE): --max-steps optimizer steps of the
           training method on --train with no head-only steps, the first micro-batch being the 8 longest items (the
           peak memory, in GiB); training items a second over steps 21 to the last; then one forward pass over --val,
           timed; writes pilot.json (--out-json) with the rule on gradient checkpointing. When the rule turns gradient
           checkpointing on, the pilot runs a second time with it on, so that the cost rule uses the rates of the
           setting the arm will run with (correction 2). No weights are written.
  predict  loads the base and a phase folder and writes raw predictions and logits for <records-dir>/<qid>_<split>.jsonl.
           Session 8 (docs/prompt_session8.md, ROOT): --run-summary names the run summary's path (default, as in session
           7, <out>/../<system>_run_summary.json), and the summary also records each split's forward-pass seconds (the
           model's passes over the built items, timed with the GPU synchronized; tokenizing and item building are not
           counted), seconds per item and items a second.
  pilot-s8 the pilot of docs/prompt_session8.md (THE PILOT AND THE COST RULE), beside session 7's pilot, which is
           unchanged: on --train (EI's training items), no head-only steps and no weights written, two settings of
           --steps (30) optimizer steps each, (a) micro-batch 1 with accumulation 32 and gradient checkpointing off and
           (b) micro-batch 4 with accumulation 8 and gradient checkpointing on, the first step of each on the micro-batch
           times accumulation longest items; for each, training items a second over steps 11 to 30, the peak GPU memory
           (GiB) and whether it ran out of memory. A setting passes if it did not run out of memory and its peak plus 16
           GiB (room for arm A beside it) is at most 76 GiB; the seeds use the passing setting with the higher training
           rate. If neither passes, (c) micro-batch 1 with accumulation 32 and gradient checkpointing on is run, and used if
           it passes; if it does not, pilot.json says to stop and ask. Then one forward pass over --val (the 300 monitoring
           items) at --val-batch-size (4, predict's batch size), timed: the validation items a second. Writes pilot.json.
           Session 9 (docs/prompt_session9.md, ROOT and THE PILOT AND THE COST RULE): --headroom-gib is the room held
           for another GPU job in that rule, default 16, which keeps session 8's rule exactly; session 9's pilot_long passes
           0, since nothing runs beside its training, so that a setting passes if its peak is at most 76 GiB. pilot.json
           records the headroom (headroom_gib).

Usage on the pod:
  python src/train/train_qwen_head.py train --base-dir /root/eddm_s7/models/qwen3_4b_base --train items/b_phase1.pt \\
      --val items/b_phase12_val.pt --out models/s7_b1 --system-name s7_b1 --freeze-steps 250 --keep last --no-test ...
  python src/train/train_qwen_head.py pilot --base-dir ... --train items/b_phase1.pt --val items/b_phase12_val.pt --out-json pilot.json
  python src/train/train_qwen_head.py predict --base-dir ... --phase-dir models/s7_b3 --system s7_b3 \\
      --records-dir data/processed/public --qids pubmedqa_answer readmit_window --splits test --out preds --logits-dir logits
  python src/train/train_qwen_head.py pilot-s8 --base-dir /root/eddm_s8/models/qwen3_4b_base --train items/ei_train.pt \\
      --val items/ei_monitor.pt --out-json results/fiorillo_v0_4/pilot.json
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from laya.common import DecisionModel
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
from build_items import QUESTIONS, load_questions, read_records  # noqa: E402
from lora import TARGETS, apply_lora, count_parameters, is_adapter  # noqa: E402
from qwen_items import build_qwen_items  # noqa: E402
from train_laya import (collate, fit_temperatures, log_loss, micro_batches, per_question, pick_device,  # noqa: E402
                        probabilities, run, selection_score, summarize, write_logits, write_predictions)

PROJECTION = 1024
HEAD_LAYERS = 2
N_ACT = 2  # Laya's act_costs has one entry (escalate), so len + 1, as laya.common.build_model sets it
ITEM_FORMAT = "state, then \\n\\n, then '<type> question: <instructions>', then \\n, then each option followed by one \\n token (its marker); src/train/qwen_items.py"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json")


class QwenReader(nn.Module):
    """The adapted base and a projection to Laya's head width, shaped like the encoder DecisionModel expects."""

    def __init__(self, base, width=PROJECTION):
        super().__init__()
        self.base = base
        self.proj = nn.Linear(base.config.hidden_size, width, dtype=torch.float32)
        self.config = SimpleNamespace(hidden_size=width)
        self.freeze_base = False

    def forward(self, input_ids=None, attention_mask=None):
        if self.freeze_base:
            with torch.no_grad():
                h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            h = self.base(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if not torch.is_autocast_enabled(h.device.type):
            h = h.to(self.proj.weight.dtype)
        return SimpleNamespace(last_hidden_state=self.proj(h))


def build_model(base, seed, r=16, alpha=32, dropout=0.05, gradient_checkpointing=False):
    """DecisionModel on the adapted base; the base is frozen first, then adapters, projection and head are made from
    one seeded initialization, in that order."""
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    wrapped = apply_lora(base, TARGETS, r, alpha, dropout)
    if gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reader = QwenReader(base)
    model = DecisionModel(reader, HEAD_LAYERS, N_ACT)
    model.head_checkpointing = False
    return model, wrapped


def load_base(base_dir, dtype=torch.bfloat16):
    from transformers import AutoModel

    return AutoModel.from_pretrained(str(base_dir), dtype=dtype, attn_implementation="sdpa")


def trainable_state(model):
    return {n: p.detach().to(torch.float32).contiguous().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, path):
    state = load_file(str(path))
    params = dict(model.named_parameters())
    want = {n for n, p in params.items() if p.requires_grad}
    if set(state) != want:
        missing, extra = sorted(want - set(state))[:5], sorted(set(state) - want)[:5]
        raise SystemExit(f"{path}: trainable parameters differ (missing {missing}, unexpected {extra})")
    with torch.no_grad():
        for n, t in state.items():
            if params[n].shape != t.shape:
                raise SystemExit(f"{path}: {n} has shape {tuple(t.shape)}, the model {tuple(params[n].shape)}")
            params[n].copy_(t.to(params[n].dtype))


def param_groups(model, lr_adapters, lr_head):
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and is_adapter(n)]
    rest = [p for n, p in model.named_parameters() if p.requires_grad and not is_adapter(n)]
    return [{"params": adapters, "lr": lr_adapters, "name": "adapters"}, {"params": rest, "lr": lr_head, "name": "projection_and_head"}]


def base_revision(base_dir):
    path = Path(base_dir) / "REVISION.txt"
    return path.read_text(encoding="utf-8").split()[1] if path.exists() and len(path.read_text(encoding="utf-8").split()) > 1 else None


def project_commit():
    path = ROOT / "COMMIT.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_phase(out, model, args, base_dir, extra):
    out.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state(model), str(out / "adapter.safetensors"))
    config = {
        "arm": "B", "plan": "docs/prompt_session7.md, ARM B",
        "base_dir": str(base_dir), "base_revision": base_revision(base_dir),
        "base_config_sha256": file_sha256(Path(base_dir) / "config.json") if (Path(base_dir) / "config.json").exists() else None,
        "adapters": {"targets": list(TARGETS), "r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "projection": PROJECTION, "head_layers": HEAD_LAYERS, "n_act": N_ACT, "max_len": args.max_len,
        "item_format": ITEM_FORMAT, "project_commit": project_commit(), "seed": args.seed,
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **extra,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in TOKENIZER_FILES:
        src = Path(base_dir) / name
        if src.exists():
            shutil.copyfile(src, out / name)


def autocast_type(device):
    return torch.bfloat16 if device.type == "cuda" else None


def step_batches(items, micro_batch, grad_accum, seed, epoch):
    batches = micro_batches(items, micro_batch, random.Random(seed + epoch), by_question=True)
    return batches, math.ceil(len(batches) / grad_accum)


def train_phase(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ac = autocast_type(device)
    base = load_base(args.base_dir, torch.bfloat16 if device.type == "cuda" else torch.float32)
    model, wrapped = build_model(base, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout, args.gradient_checkpointing)
    if args.init:
        load_trainable(model, Path(args.init) / "adapter.safetensors")
    model.to(device)
    model.train()
    reader = model.encoder
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_dir))
    pad_id = tokenizer.pad_token_id

    train_items = torch.load(args.train, weights_only=False)
    val_items = torch.load(args.val, weights_only=False)
    if args.max_rows:
        train_items, val_items = train_items[: args.max_rows], val_items[: max(50, args.max_rows // 4)]
    groups = param_groups(model, args.lr_adapters, args.lr_head)
    optimizer = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], weight_decay=0.01)
    n_micro = len(micro_batches(train_items, args.micro_batch, random.Random(0), True))
    total_steps = max(1, math.ceil(n_micro / args.grad_accum) * args.epochs)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    trainable = [p for p in model.parameters() if p.requires_grad]

    args.log.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(args.log, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["epoch", "step", "loss", "lr_adapters", "lr_head", "elapsed_seconds"])
    best_dir = args.out / "checkpoint_best"
    history, best_val, last_val = [], float("inf"), float("inf")
    step, sequences, t0, train_seconds = 0, 0, time.time(), 0.0
    eval_batch = args.micro_batch * 2

    def validate(epoch):
        nonlocal best_val, last_val
        reader.freeze_base = False
        rows = run(model, val_items, pad_id, device, eval_batch, ac)
        probs = probabilities(rows)
        score = selection_score(rows, probs)
        metrics = summarize(rows, probs)
        metrics.update({"selection_log_loss": round(score, 4), "per_question": per_question(rows, probs, calibrate_only=True),
                        "epoch": epoch, "step": step, "elapsed_seconds": round(time.time() - t0, 1)})
        history.append(metrics)
        print(f"validation after epoch {epoch}, step {step}: {metrics}", flush=True)
        last_val = score
        if score < best_val:
            best_val = score
            if args.keep == "best":
                best_dir.mkdir(parents=True, exist_ok=True)
                save_file(trainable_state(model), str(best_dir / "adapter.safetensors"))

    done = False
    for epoch in range(1, args.epochs + 1):
        batches, _ = step_batches(train_items, args.micro_batch, args.grad_accum, args.seed, epoch)
        optimizer.zero_grad(set_to_none=True)
        micro = 0
        for b, batch_idx in enumerate(batches):
            chunk = [train_items[i] for i in batch_idx]
            batch = collate(chunk, pad_id)
            t_batch = time.time()
            reader.freeze_base = step < args.freeze_steps
            with torch.autocast(device.type, dtype=ac, enabled=ac is not None):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            loss = log_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device)) + 0.0 * act.float().sum()
            (loss / args.grad_accum).backward()
            micro += 1
            sequences += len(chunk)
            if micro % args.grad_accum == 0 or b == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                lrs = scheduler.get_last_lr()
                log_writer.writerow([epoch, step, round(float(loss.detach()), 5), lrs[0], lrs[1], round(time.time() - t0, 1)])
                if step % 20 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {float(loss.detach()):.4f} "
                          f"({sequences / max(1e-6, train_seconds):.1f} sequences/s)", flush=True)
                if args.eval_every and step % args.eval_every == 0:
                    validate(epoch)
                if args.max_steps and step >= args.max_steps:
                    done = True
            train_seconds += time.time() - t_batch
            if done:
                break
        validate(epoch)
        if done:
            break
    log_file.close()
    reader.freeze_base = False

    if args.keep == "best" and best_dir.exists():
        load_trainable(model, best_dir / "adapter.safetensors")
    val_rows = run(model, val_items, pad_id, device, eval_batch, ac)
    temperature = fit_temperatures(val_rows)
    val_raw, val_cal = probabilities(val_rows), probabilities(val_rows, temperature)
    save_phase(args.out, model, args, args.base_dir, {"system": args.system_name, "keep": args.keep})
    calibration = {"method": "temperature scaling, one temperature per question", "fitted_on": "validation items marked calibrate",
                   "usage": "divide each question's logits by its temperature before the softmax",
                   "temperature": {k: round(v, 4) for k, v in temperature.items()}}
    (args.out / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n", encoding="utf-8")
    if best_dir.exists():
        shutil.rmtree(best_dir)
    if args.predictions_dir:
        write_predictions(args.predictions_dir, args.system_name, "validation", val_rows, val_cal)
        write_predictions(args.predictions_dir, f"{args.system_name}-raw", "validation", val_rows, val_raw)
    if args.logits_dir:
        write_logits(args.logits_dir, args.system_name, "validation", val_rows)
    adapters = [p for n, p in model.named_parameters() if p.requires_grad and is_adapter(n)]
    others = [p for n, p in model.named_parameters() if p.requires_grad and not is_adapter(n)]
    metrics = {
        "system": args.system_name, "arm": "B", "base_dir": str(args.base_dir), "init": str(args.init) if args.init else None,
        "keep": args.keep, "freeze_steps": args.freeze_steps, "eval_every": args.eval_every,
        "lr_adapters": args.lr_adapters, "lr_head": args.lr_head, "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                                                                          "wrapped_layers": wrapped},
        "gradient_checkpointing": args.gradient_checkpointing, "autocast": str(ac), "device": str(device), "torch": torch.__version__,
        "trainable_parameters": {"adapters": count_parameters(adapters), "projection_and_head": count_parameters(others)},
        "rows": {"train": len(train_items), "validation": len(val_items), "test": 0},
        "epochs": args.epochs, "micro_batch": args.micro_batch, "grad_accum": args.grad_accum, "optimizer_steps": step,
        "max_len": args.max_len, "elapsed_seconds": round(time.time() - t0, 1), "train_seconds": round(train_seconds, 1),
        "sequences_per_second": round(sequences / max(1e-6, train_seconds), 2),
        "validation_history": history, "best_validation_selection_log_loss": round(best_val, 4),
        "best_validation_selection_log_loss_exact": best_val,
        "kept_validation_selection_log_loss_exact": best_val if args.keep == "best" else last_val,
        "temperature": {k: round(v, 4) for k, v in temperature.items()},
        "no_test": True, "test": "not run (the test split is predicted only by the predict subcommand, after the freeze)",
        "validation_final": summarize(val_rows, val_raw),
        "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None,
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: metrics[k] for k in ("system", "optimizer_steps", "sequences_per_second", "kept_validation_selection_log_loss_exact",
                                              "temperature", "trainable_parameters")}, indent=2))
    return metrics


def pilot(args):
    """THE PILOT: training throughput with full steps, peak memory, and the forward rate on validation."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ac = autocast_type(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_dir))
    pad_id = tokenizer.pad_token_id
    train_items = torch.load(args.train, weights_only=False)
    val_items = torch.load(args.val, weights_only=False)
    result = {"plan": "docs/prompt_session7.md, THE PILOT AND THE COST RULE", "micro_batch": args.micro_batch,
              "grad_accum": args.grad_accum, "steps": args.max_steps, "attempts": []}

    def attempt(checkpointing):
        base = load_base(args.base_dir, torch.bfloat16 if device.type == "cuda" else torch.float32)
        model, _ = build_model(base, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout, checkpointing)
        model.to(device)
        model.train()
        groups = param_groups(model, args.lr_adapters, args.lr_head)
        optimizer = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], weight_decay=0.01)
        trainable = [p for p in model.parameters() if p.requires_grad]
        longest = sorted(range(len(train_items)), key=lambda i: -len(train_items[i]["ids"]))[: args.micro_batch]
        batches = [longest] + micro_batches(train_items, args.micro_batch, random.Random(args.seed + 1), True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        step, micro, seen, t_start, seen_start = 0, 0, 0, None, 0
        for batch_idx in batches:
            chunk = [train_items[i] for i in batch_idx]
            batch = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=ac, enabled=ac is not None):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            loss = log_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device)) + 0.0 * act.float().sum()
            (loss / args.grad_accum).backward()
            micro += 1
            seen += len(chunk)
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == 20:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start, seen_start = time.time(), seen
                if step >= args.max_steps:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_rate = (seen - seen_start) / max(1e-6, time.time() - t_start) if t_start else None
        peak = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else None  # GiB, as session 5 reported memory
        model.eval()
        t_val = time.time()
        run(model, val_items, pad_id, device, args.micro_batch * 2, ac)
        if device.type == "cuda":
            torch.cuda.synchronize()
        val_rate = len(val_items) / max(1e-6, time.time() - t_val)
        return {"gradient_checkpointing": checkpointing, "steps": step, "train_items_per_second": train_rate,
                "validation_items_per_second": val_rate, "peak_gpu_memory_gib": peak, "out_of_memory": False}

    oom = False
    try:
        first = attempt(False)
    except torch.cuda.OutOfMemoryError:
        oom = True
        first = {"gradient_checkpointing": False, "out_of_memory": True}
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    result["attempts"].append(first)
    needs = oom or (first.get("peak_gpu_memory_gib") is not None and first["peak_gpu_memory_gib"] + 16 > 76)
    if needs:
        # the rates the cost rule uses are those of the setting the arm will run with (correction 2)
        result["attempts"].append(attempt(True))
    chosen = result["attempts"][-1]
    result.update({
        "rule": ("gradient checkpointing on if the peak memory plus 16 GB is above 76 GB, or if the pilot ran out of memory at micro-batch 8; "
                 "GB here is GiB (2^30 bytes), the unit of the card's 80 GB and of session 5's memory figures (docs/prompt_session7.md, correction 2)"),
        "gradient_checkpointing_on": bool(needs),
        "train_items_per_second": chosen.get("train_items_per_second"),
        "validation_items_per_second": chosen.get("validation_items_per_second"),
        "peak_gpu_memory_gib": chosen.get("peak_gpu_memory_gib"),
        "peak_gpu_memory_gib_without_checkpointing": first.get("peak_gpu_memory_gib"),
        "items": {"train": len(train_items), "validation": len(val_items)},
        "device": str(device), "torch": torch.__version__,
        "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    })
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def predict(args):
    device = pick_device(args.device)
    ac = autocast_type(device)
    base = load_base(args.base_dir, torch.bfloat16 if device.type == "cuda" else torch.float32)
    cfg = json.loads((Path(args.phase_dir) / "config.json").read_text(encoding="utf-8"))
    a = cfg["adapters"]
    model, _ = build_model(base, cfg.get("seed", 20260927), a["r"], a["alpha"], a["dropout"], False)
    load_trainable(model, Path(args.phase_dir) / "adapter.safetensors")
    model.to(device)
    model.eval()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.phase_dir))
    questions = load_questions(args.questions)
    summary = {"system": args.system, "phase_dir": str(args.phase_dir), "device": str(device), "max_len": cfg["max_len"],
               "batch_size": args.batch_size, "questions": str(args.questions), "torch": torch.__version__}
    if device.type == "cuda":
        summary["gpu"] = torch.cuda.get_device_name(0)
    for split in args.splits:
        records = read_records([Path(args.records_dir) / f"{q}_{split}.jsonl" for q in args.qids])
        items, built = build_qwen_items(records, questions, tokenizer, cfg["max_len"])
        summary[f"items_{split}"] = built
        rows, seconds = timed_run(model, items, tokenizer.pad_token_id, device, args.batch_size, ac)
        summary[f"timing_{split}"] = {"items": len(items), "forward_seconds": seconds,
                                      "seconds_per_item": seconds / max(1, len(items)), "items_per_second": len(items) / max(1e-9, seconds),
                                      "note": "the model's passes over the built items, GPU synchronized; tokenizing and item building not counted"}
        write_predictions(args.out, f"{args.system}-raw", split, rows, probabilities(rows))
        if args.logits_dir:
            write_logits(args.logits_dir, args.system, split, rows)
        summary[f"{split}_per_question_raw"] = per_question(rows, probabilities(rows))
        print(f"{args.system} {split}: {len(rows)} rows, {seconds:.1f} forward seconds", flush=True)
    path = Path(args.run_summary) if args.run_summary else Path(args.out).parent / f"{args.system}_run_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def timed_run(model, items, pad_id, device, batch_size, ac):
    """run() timed alone: (rows, seconds), with the GPU synchronized before and after."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    rows = run(model, items, pad_id, device, batch_size, ac)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return rows, time.perf_counter() - t0


S8_SETTINGS = (("a", 1, 32, False), ("b", 4, 8, True))
S8_FALLBACK = ("c", 1, 32, True)
S8_LIMIT_GIB, S8_ROOM_GIB = 76.0, 16.0


def s8_passes(attempt, headroom=S8_ROOM_GIB):
    """A setting passes if it did not run out of memory and its peak plus the headroom (--headroom-gib: 16 GiB by default,
    session 8's room for arm A; 0 in session 9) is at most 76 GiB (no peak is measured on a CPU, where a setting that did
    not fail passes; tests only)."""
    if attempt.get("out_of_memory") or attempt.get("error"):
        return False
    peak = attempt.get("peak_gpu_memory_gib")
    return peak is None or peak + headroom <= S8_LIMIT_GIB


def choose_s8(attempts, headroom=S8_ROOM_GIB):
    """The passing setting with the higher training rate (the first listed among exact ties), or None."""
    passing = [a for a in attempts if s8_passes(a, headroom)]
    if not passing:
        return None
    best = max(passing, key=lambda a: a["train_items_per_second"])
    return next(a for a in passing if a["train_items_per_second"] == best["train_items_per_second"])


def s8_attempt(args, name, micro_batch, grad_accum, checkpointing, train_items, pad_id, device, ac):
    """One setting of the pilot: --steps optimizer steps, the first on the longest items; the rate over steps
    time-from + 1 to the last."""
    import gc

    out = {"setting": name, "micro_batch": micro_batch, "grad_accum": grad_accum, "gradient_checkpointing": checkpointing,
           "steps": 0, "train_items_per_second": None, "peak_gpu_memory_gib": None, "out_of_memory": False}
    model = optimizer = None
    try:
        base = load_base(args.base_dir, torch.bfloat16 if device.type == "cuda" else torch.float32)
        model, _ = build_model(base, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout, checkpointing)
        model.to(device)
        model.train()
        groups = param_groups(model, args.lr_adapters, args.lr_head)
        optimizer = torch.optim.AdamW([{k: v for k, v in g.items() if k != "name"} for g in groups], weight_decay=0.01)
        trainable = [p for p in model.parameters() if p.requires_grad]
        longest = sorted(range(len(train_items)), key=lambda i: (-len(train_items[i]["ids"]), i))[: micro_batch * grad_accum]
        first = [longest[k: k + micro_batch] for k in range(0, len(longest), micro_batch)]
        batches = first + micro_batches(train_items, micro_batch, random.Random(args.seed + 1), True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        step, micro, seen, t_start, seen_start = 0, 0, 0, None, 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx in batches:
            chunk = [train_items[i] for i in batch_idx]
            batch = collate(chunk, pad_id)
            with torch.autocast(device.type, dtype=ac, enabled=ac is not None):
                logits, act = model(batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                    batch["marker_pos"].to(device), batch["marker_mask"].to(device), batch["qtype"].to(device))
            loss = log_loss(logits.float(), batch["target"].to(device), batch["marker_mask"].to(device)) + 0.0 * act.float().sum()
            (loss / grad_accum).backward()
            micro += 1
            seen += len(chunk)
            if micro % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step == args.time_from:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t_start, seen_start = time.perf_counter(), seen
                if step >= args.steps:
                    break
        if device.type == "cuda":
            torch.cuda.synchronize()
        out["steps"] = step
        if t_start is not None and step > args.time_from:
            out["train_items_per_second"] = (seen - seen_start) / max(1e-9, time.perf_counter() - t_start)
            out["timed_steps"] = [args.time_from + 1, step]
        if device.type == "cuda":
            out["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    except torch.cuda.OutOfMemoryError:
        out["out_of_memory"] = True
        if device.type == "cuda":
            out["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated() / 2**30
    del model, optimizer
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    out["passes"] = s8_passes(out, getattr(args, "headroom_gib", S8_ROOM_GIB))
    print(json.dumps(out), flush=True)
    return out


def pilot_s8(args):
    """THE PILOT AND THE COST RULE of docs/prompt_session8.md."""
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    ac = autocast_type(device)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.base_dir))
    pad_id = tokenizer.pad_token_id
    train_items = torch.load(args.train, weights_only=False)
    val_items = torch.load(args.val, weights_only=False)
    headroom = getattr(args, "headroom_gib", S8_ROOM_GIB)
    if headroom == S8_ROOM_GIB:  # session 8's rule, as it was written
        plan = "docs/prompt_session8.md, THE PILOT AND THE COST RULE"
        room = "plus 16 GiB (room for arm A beside it)"
    else:
        plan = "docs/prompt_session8.md, THE PILOT AND THE COST RULE, with --headroom-gib (docs/prompt_session9.md, THE PILOT AND THE COST RULE)"
        room = f"plus {headroom:g} GiB (--headroom-gib, the room held for another GPU job)"
    result = {"plan": plan, "steps": args.steps, "timed_steps": f"{args.time_from + 1} to {args.steps}", "headroom_gib": headroom,
              "rule": (f"a setting passes if it did not run out of memory and its peak GPU memory {room} "
                       "is at most 76 GiB; the seeds use the passing setting with the higher training rate; if neither passes, "
                       "setting c is run and used if it passes; if it does not, stop and ask"),
              "settings": []}
    for name, mb, ga, ckpt in S8_SETTINGS:
        result["settings"].append(s8_attempt(args, name, mb, ga, ckpt, train_items, pad_id, device, ac))
    chosen = choose_s8(result["settings"], headroom)
    if chosen is None:
        name, mb, ga, ckpt = S8_FALLBACK
        result["settings"].append(s8_attempt(args, name, mb, ga, ckpt, train_items, pad_id, device, ac))
        chosen = choose_s8(result["settings"][-1:], headroom)
    result["chosen"] = ({k: chosen[k] for k in ("setting", "micro_batch", "grad_accum", "gradient_checkpointing", "train_items_per_second",
                                                 "peak_gpu_memory_gib")} if chosen else None)
    result["stop_and_ask"] = chosen is None
    # one forward pass over the monitoring items, timed, on a model that has not been trained (the rate does not depend on it)
    base = load_base(args.base_dir, torch.bfloat16 if device.type == "cuda" else torch.float32)
    model, _ = build_model(base, args.seed, args.lora_r, args.lora_alpha, args.lora_dropout, False)
    model.to(device)
    _, seconds = timed_run(model, val_items, pad_id, device, args.val_batch_size, ac)
    result.update({"validation_items": len(val_items), "validation_batch_size": args.val_batch_size, "validation_seconds": seconds,
                   "validation_items_per_second": len(val_items) / max(1e-9, seconds),
                   "items": {"train": len(train_items), "validation": len(val_items)}, "device": str(device), "torch": torch.__version__,
                   "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                   "written_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("chosen", "stop_and_ask", "validation_items_per_second")}, indent=2))
    return result


def add_model_options(p):
    p.add_argument("--base-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=20260927)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lr-adapters", type=float, default=5e-5, help="correction 1 of docs/prompt_session7.md (was 2e-4)")
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-len", type=int, default=768)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    add_model_options(t)
    t.add_argument("--train", type=Path, required=True)
    t.add_argument("--val", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--system-name", required=True)
    t.add_argument("--init", type=Path, default=None, help="a previous phase's folder (its adapter.safetensors)")
    t.add_argument("--epochs", type=int, default=1)
    t.add_argument("--keep", choices=("best", "last"), default="best")
    t.add_argument("--freeze-steps", type=int, default=0)
    t.add_argument("--eval-every", type=int, default=0)
    t.add_argument("--no-test", action="store_true", help="kept for the plan's wording; train never reads a test split")
    t.add_argument("--predictions-dir", type=Path, default=None)
    t.add_argument("--logits-dir", type=Path, default=None)
    t.add_argument("--max-steps", type=int, default=0)
    t.add_argument("--max-rows", type=int, default=None)
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.add_argument("--metrics", type=Path, required=True)
    t.add_argument("--log", type=Path, required=True)
    pl = sub.add_parser("pilot")
    add_model_options(pl)
    pl.add_argument("--train", type=Path, required=True)
    pl.add_argument("--val", type=Path, required=True)
    pl.add_argument("--max-steps", type=int, default=200)
    pl.add_argument("--out-json", type=Path, required=True)
    pr = sub.add_parser("predict")
    pr.add_argument("--base-dir", type=Path, required=True)
    pr.add_argument("--phase-dir", type=Path, required=True)
    pr.add_argument("--system", required=True)
    pr.add_argument("--records-dir", type=Path, required=True)
    pr.add_argument("--qids", nargs="+", required=True)
    pr.add_argument("--splits", nargs="+", choices=("validation", "test"), required=True)
    pr.add_argument("--questions", type=Path, default=QUESTIONS)
    pr.add_argument("--out", type=Path, required=True)
    pr.add_argument("--logits-dir", type=Path, default=None)
    pr.add_argument("--batch-size", type=int, default=16)
    pr.add_argument("--device", default="auto")
    pr.add_argument("--run-summary", type=Path, default=None, help="the run summary's path (session 8)")
    p8 = sub.add_parser("pilot-s8")
    add_model_options(p8)
    p8.add_argument("--train", type=Path, required=True)
    p8.add_argument("--val", type=Path, required=True)
    p8.add_argument("--steps", type=int, default=30)
    p8.add_argument("--time-from", type=int, default=10, help="the rate is timed from the end of this step to the last")
    p8.add_argument("--val-batch-size", type=int, default=4)
    p8.add_argument("--headroom-gib", type=float, default=S8_ROOM_GIB,
                    help="the room (GiB) held for another GPU job: a setting passes if its peak plus this is at most 76 GiB "
                         "(default 16, session 8's rule; session 9's pilot_long passes 0)")
    p8.add_argument("--out-json", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "train":
        return train_phase(args)
    if args.command == "pilot":
        return pilot(args)
    if args.command == "pilot-s8":
        return pilot_s8(args)
    return predict(args)


if __name__ == "__main__":
    main()

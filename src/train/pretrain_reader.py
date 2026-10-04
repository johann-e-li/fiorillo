"""Continued masked-language-model (MLM) training of Laya's reader for session 5 (docs/prompt_session5.md, READER
METHOD), on blocks made by src/corpus/tokenize_blocks.py.

Method, fixed by the prompt:
- Start: a ModernBertForMaskedLM folder (--init; src/train/build_mlm_init.py makes the first one), or a start-of-decay
  checkpoint with its resume.pt (--resume).
- Data: blocks of 1,024 tokens drawn without replacement in the mix's token shares by walking each source's
  permutation (tokenize_blocks.MixSampler); mixes and shares in src/configs/mixes_session5.yaml.
- Masking: each non-special token is selected with probability 0.30; of the selected, 80 percent become [MASK], 10
  percent a random token and 10 percent stay unchanged. The masks come from a generator seeded by the seed, the step and
  the micro-batch, so a resumed run masks exactly as an uninterrupted one. The loss is computed at selected positions
  only (sparse prediction).
- Optimizer: AdamW, betas 0.9 and 0.98, epsilon 1e-6, weight decay 0.01 on weights and none on biases and
  normalization weights, peak learning rate 1e-4, gradient clipping at 1.0, bfloat16 autocast on CUDA, attention sdpa.
- Batch: 256 blocks (262,144 tokens) per optimizer step, micro-batch 32 with accumulation 8 (16 and 16 if the timed
  run runs out of memory).
- Schedule (warmup, constant, decay): S steps for the run's tokens, rounded up; linear warmup over W = int(0.03 S)
  steps; the peak rate; linear decay to zero over the last ceil(0.10 S) steps, from step D. A 500-million-token run:
  S 1,908, W 57, D 1,717. --total-tokens sets S and D from the longer total (the long run: 2,000 million tokens, S
  7,630, D 6,867) while W stays the run's own; a resumed run keeps W from resume.pt. The task round (mix T) runs
  three passes: S is three times its training blocks divided by 256, rounded up.
- Checkpoints: the final one (<out>/final) and, in a comparison run (mix A, B or C, not resumed), the start-of-decay
  one (<out>/decay_start, after D steps) with resume.pt: the optimizer state, the schedule, every random-number state,
  the step and each source's position in its permutation.
- Logging: the training loss every 10 steps (--log CSV: step, tokens, learning rate, loss, gradient norm, tokens per
  second, seconds); held-out MLM loss per source every 250 steps and at the end, over every held-out block with a fixed
  mask (<out>/heldout_losses.json); throughput (<out>/throughput.json).
--timing-steps N is the timed run of step 8: N steps of the mix from --init, drawing blocks in a separate seeded order
so the permutations stay untouched, nothing kept; tokens per second over steps 21 to N and peak GPU memory go to
<out>/timing.json. Out of memory in the timed run exits with code 3 after writing timing.json.

Usage on the pod:
    python src/train/pretrain_reader.py --init /root/eddm_s5/models/mlm_init --mix A --tokens 500000000 \
        --out /root/eddm_s5/models/reader_mix_a --log /root/eddm_s5/logs/mlm_mix_a.csv
    python src/train/pretrain_reader.py --resume /root/eddm_s5/models/reader_mix_a/decay_start/resume.pt \
        --mix A --total-tokens 2000000000 --out /root/eddm_s5/models/reader_v1 --log /root/eddm_s5/logs/mlm_long.csv
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
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "corpus"))
from tokenize_blocks import BLOCK, SOURCES, MixSampler, open_source  # noqa: E402

SEED = 20260927
MIXES = ROOT / "src" / "configs" / "mixes_session5.yaml"
MASK_PROB, MASK_TOKEN, RANDOM_TOKEN = 0.30, 0.8, 0.1


def schedule(tokens_per_step, run_tokens=None, total_tokens=None, warmup=None, steps=None):
    """(S, W, D): total optimizer steps, warmup steps, first decay step. The run's own length (steps, or run_tokens
    rounded up to whole steps) sets W unless warmup is given; total_tokens, when given, sets S and D."""
    run_steps = steps if steps is not None else (math.ceil(run_tokens / tokens_per_step) if run_tokens else None)
    w = warmup if warmup is not None else int(0.03 * run_steps)
    s = math.ceil(total_tokens / tokens_per_step) if total_tokens else run_steps
    return s, w, s - math.ceil(0.10 * s)


def learning_rate(step, peak, s, w, d):
    """Learning rate of optimizer step `step` (0-based)."""
    if step < w:
        return peak * (step + 1) / w
    if step < d:
        return peak
    return peak * (s - step) / (s - d)


def mask_blocks(ids, step, micro, seed, special_ids, mask_id, vocab_size):
    """(inputs, labels) for a micro-batch of blocks; deterministic in (seed, step, micro)."""
    g = torch.Generator().manual_seed(int(seed) * 1_000_003 + int(step) * 1_009 + int(micro))
    special = torch.zeros_like(ids, dtype=torch.bool)
    for s in special_ids:
        special |= ids == s
    selected = (torch.rand(ids.shape, generator=g) < MASK_PROB) & ~special
    r = torch.rand(ids.shape, generator=g)
    random_ids = torch.randint(0, vocab_size, ids.shape, generator=g)
    inputs = ids.clone()
    inputs[selected & (r < MASK_TOKEN)] = mask_id
    to_random = selected & (r >= MASK_TOKEN) & (r < MASK_TOKEN + RANDOM_TOKEN)
    inputs[to_random] = random_ids[to_random]
    labels = torch.where(selected, ids, torch.full_like(ids, -100))
    return inputs, labels


def optimizer_groups(model):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in name or name.endswith(".bias") else decay).append(p)
    return [{"params": decay, "weight_decay": 0.01}, {"params": no_decay, "weight_decay": 0.0}]


def rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def copy_tokenizer(src_dir, dst_dir):
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if (Path(src_dir) / name).exists():
            shutil.copy(Path(src_dir) / name, Path(dst_dir) / name)


def save_checkpoint(model, folder, tokenizer_dir, info):
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(folder), safe_serialization=True)
    copy_tokenizer(tokenizer_dir, folder)
    (folder / "checkpoint.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


@torch.no_grad()
def heldout_losses(model, sources, special_ids, mask_id, vocab_size, device, autocast, seed, batch=32):
    """Mean MLM loss per source over every held-out block, with one fixed mask per source."""
    model.eval()
    out = {}
    for name, src in sources.items():
        blocks = np.asarray(src["heldout"])
        total, count = 0.0, 0
        for start in range(0, len(blocks), batch):
            ids = torch.from_numpy(blocks[start : start + batch].astype(np.int64))
            inputs, labels = mask_blocks(ids, 10_000_000 + SOURCES.index(name), start, seed, special_ids, mask_id, vocab_size)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                loss = model(input_ids=inputs.to(device), attention_mask=torch.ones_like(inputs).to(device),
                             labels=labels.to(device)).loss
            n = int((labels != -100).sum())
            total += float(loss) * n
            count += n
        out[name] = round(total / max(1, count), 5)
    model.train()
    return out


def run(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--init", type=Path, help="a ModernBertForMaskedLM folder to start from")
    parser.add_argument("--mix", required=True, help="a mix name in the mixes file (A, B, C or T)")
    parser.add_argument("--tokens", type=int, default=None, help="tokens for this run (default: the mix's budget)")
    parser.add_argument("--resume", type=Path, help="a resume.pt from a start-of-decay checkpoint")
    parser.add_argument("--total-tokens", type=int, default=None, help="the long run's total (sets S and the decay)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--blocks", type=Path, default=Path("/root/eddm_s5/blocks"))
    parser.add_argument("--mixes", type=Path, default=MIXES)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--peak-lr", type=float, default=1e-4, help="fixed at 1e-4 by the method; tests change it")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--timing-steps", type=int, default=0, help="the timed run: this many steps, nothing kept")
    args = parser.parse_args(argv)
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    autocast = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    mixes = yaml.safe_load(args.mixes.read_text(encoding="utf-8"))
    mix = mixes["mixes"][args.mix]
    shares, passes = {k: float(v) for k, v in mix["shares"].items()}, int(mix.get("passes", 1))
    blocks_per_step = args.micro_batch * args.grad_accum
    tokens_per_step = blocks_per_step * BLOCK
    held_names = [s for s in SOURCES if (args.blocks / s / "heldout.npy").exists()]
    sources = {s: open_source(args.blocks, s) for s in set(held_names) | set(shares)}

    resume = torch.load(args.resume, map_location="cpu", weights_only=False) if args.resume else None
    if resume:
        if resume["mix"] != args.mix or resume["blocks_per_step"] != blocks_per_step:
            raise SystemExit("resume.pt was written by another mix or batch; stop")
        for s in shares:
            if resume["perm_sha"][s] != hash_perm(sources[s]["perm"]):
                raise SystemExit(f"source {s}: its permutation differs from the one the run started with; stop")
        start_dir = args.resume.parent
        step = int(resume["step"])
        s_total, warmup, decay = schedule(tokens_per_step, total_tokens=args.total_tokens, warmup=resume["warmup"],
                                          steps=resume["steps"])
    else:
        if not args.init:
            raise SystemExit("--init or --resume is needed")
        start_dir, step = args.init, 0
        if args.mix == "T":
            n_task = len(sources["task"]["perm"])
            s_total, warmup, decay = schedule(tokens_per_step, steps=math.ceil(passes * n_task / blocks_per_step))
        else:
            s_total, warmup, decay = schedule(tokens_per_step, run_tokens=args.tokens or int(mix["tokens"]),
                                              total_tokens=args.total_tokens)
    if args.timing_steps:
        s_total_timing = args.timing_steps

    tokenizer = AutoTokenizer.from_pretrained(str(start_dir))
    special_ids = sorted({tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id})
    mask_id, vocab_size = tokenizer.mask_token_id, len(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(str(start_dir), attn_implementation="sdpa", dtype=torch.float32)
    model.config.sparse_prediction, model.config.sparse_pred_ignore_index = True, -100
    if hasattr(model, "sparse_prediction"):
        model.sparse_prediction, model.sparse_pred_ignore_index = True, -100
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(optimizer_groups(model), lr=args.peak_lr, betas=(0.9, 0.98), eps=1e-6)

    if args.timing_steps:
        perms = {s: np.random.default_rng([args.seed, SOURCES.index(s), 7]).permutation(len(sources[s]["perm"])) for s in shares}
        sampler = MixSampler(shares, perms, passes=passes, seed=args.seed)
    else:
        sampler = MixSampler(shares, {s: sources[s]["perm"] for s in shares}, passes=passes, seed=args.seed,
                             state=resume["sampler"] if resume else None)
        if passes > 1:  # the task round draws every block of every pass; its last step takes what remains (correction 5)
            needed = {s: len(sources[s]["perm"]) * passes for s in shares}
        else:
            needed = {s: math.ceil(shares[s] / sum(shares.values()) * s_total * blocks_per_step) + 1 for s in shares}
        short = {s: (needed[s], len(sources[s]["perm"]) * passes) for s in shares if needed[s] > len(sources[s]["perm"]) * passes}
        if short:
            raise SystemExit(f"not enough training blocks for {s_total} steps (needed, available): {short}")
    # the task round (passes above 1) stops after exactly its passes, so its last step can hold fewer blocks (correction 5)
    run_blocks = sum(len(sources[s]["perm"]) for s in shares) * passes if (passes > 1 and not args.timing_steps) else None

    def tokens_done(at_step):
        return sampler.total * BLOCK if run_blocks is not None else at_step * tokens_per_step

    if resume:
        optimizer.load_state_dict(resume["optimizer"])
        set_rng_state(resume["rng"])

    args.out.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    fresh_log = not resume or not args.log.exists() or args.log.stat().st_size == 0
    log_file = open(args.log, "w" if fresh_log else "a", newline="", encoding="utf-8")
    writer = csv.writer(log_file)
    if fresh_log:
        writer.writerow(["step", "tokens", "lr", "loss", "grad_norm", "tokens_per_second", "seconds"])
    held_path = args.out / "heldout_losses.json"
    held_history = json.loads(held_path.read_text()) if (resume and held_path.exists()) else []
    losses, window, window_t = {}, [], time.time()
    timing = {"steps": [], "tokens_per_second_21_on": None, "peak_memory_gb": None, "micro_batch": args.micro_batch,
              "grad_accum": args.grad_accum}
    end = s_total_timing if args.timing_steps else s_total
    train_start = time.time()
    run_info = {"mix": args.mix, "shares": shares, "passes": passes, "blocks_per_step": blocks_per_step,
                "tokens_per_step": tokens_per_step, "steps": s_total, "warmup": warmup, "decay_start": decay,
                "peak_lr": args.peak_lr, "seed": args.seed, "start": str(start_dir), "resumed_at": step if resume else None,
                "device": str(device), "torch": torch.__version__}
    (args.out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    t_step20 = None
    try:
        while step < end:
            if (not args.timing_steps and step == decay and args.mix in ("A", "B", "C") and not resume
                    and not (args.out / "decay_start").exists()):
                folder = args.out / "decay_start"
                save_checkpoint(model, folder, start_dir, dict(run_info, step=step, tokens=step * tokens_per_step))
                torch.save({"mix": args.mix, "step": step, "steps": s_total, "warmup": warmup, "tokens_run": args.tokens or mix.get("tokens"),
                            "blocks_per_step": blocks_per_step, "optimizer": optimizer.state_dict(), "rng": rng_state(),
                            "sampler": sampler.state(), "perm_sha": {s: hash_perm(sources[s]["perm"]) for s in shares},
                            "peak_lr": args.peak_lr}, folder / "resume.pt")
            lr = learning_rate(step, args.peak_lr, s_total, warmup, decay)
            for group in optimizer.param_groups:
                group["lr"] = lr
            draws = sampler.draw_many(blocks_per_step if run_blocks is None else min(blocks_per_step, run_blocks - sampler.total))
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for micro in range(args.grad_accum):
                chunk = draws[micro * args.micro_batch : (micro + 1) * args.micro_batch]
                if not chunk:
                    break
                ids = torch.from_numpy(np.stack([np.asarray(sources[s]["train"][b]) for s, b in chunk]).astype(np.int64))
                inputs, labels = mask_blocks(ids, step, micro, args.seed, special_ids, mask_id, vocab_size)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                    loss = model(input_ids=inputs.to(device), attention_mask=torch.ones_like(inputs).to(device),
                                 labels=labels.to(device)).loss
                # a full step divides by the accumulation as before; a short last step weights each micro-batch by its blocks
                scale = args.grad_accum if len(draws) == blocks_per_step else len(draws) / len(chunk)
                (loss / scale).backward()
                step_loss += float(loss.detach()) / scale
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            losses[step] = step_loss
            step += 1
            if device.type == "cuda" and args.timing_steps and step == 20:
                torch.cuda.synchronize()
                t_step20 = time.time()
            window.append(step_loss)
            if step % args.log_every == 0 or step == end:
                elapsed = time.time() - window_t
                tps = len(window) * tokens_per_step / max(1e-9, elapsed)
                writer.writerow([step, tokens_done(step), f"{lr:.3e}", round(float(np.mean(window)), 5),
                                 round(grad_norm, 4), round(tps, 1), round(time.time() - t0, 1)])
                log_file.flush()
                window, window_t = [], time.time()
            if not args.timing_steps and (step % args.eval_every == 0 or step == end) and held_names:
                record = {"step": step, "tokens": tokens_done(step),
                          "loss": heldout_losses(model, {s: sources[s] for s in held_names}, special_ids, mask_id,
                                                 vocab_size, device, autocast, args.seed)}
                held_history.append(record)
                held_path.write_text(json.dumps(held_history, indent=1) + "\n", encoding="utf-8")
    except torch.cuda.OutOfMemoryError:
        if args.timing_steps:
            timing.update({"oom": True, "at_step": step})
            (args.out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
            print("out of memory in the timed run")
            sys.exit(3)
        raise
    log_file.close()
    train_seconds = time.time() - train_start
    if args.timing_steps:
        if device.type == "cuda":
            torch.cuda.synchronize()
            steps_after = end - 20
            timing["tokens_per_second_21_on"] = round(steps_after * tokens_per_step / (time.time() - t_step20), 1) if t_step20 else None
            timing["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        timing.update({"steps": end, "oom": False, "seconds": round(train_seconds, 1)})
        (args.out / "timing.json").write_text(json.dumps(timing, indent=2) + "\n", encoding="utf-8")
        return losses
    save_checkpoint(model, args.out / "final", start_dir, dict(run_info, step=step, tokens=tokens_done(step)))
    done_steps = step - (run_info["resumed_at"] or 0)
    done_tokens = tokens_done(step) - (run_info["resumed_at"] or 0) * tokens_per_step
    throughput = {"steps_run": done_steps, "tokens_run": done_tokens, "seconds": round(train_seconds, 1),
                  "tokens_per_second": round(done_tokens / max(1e-9, train_seconds), 1),
                  "blocks_drawn": sampler.state()["drawn"], "peak_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if device.type == "cuda" else None}
    (args.out / "throughput.json").write_text(json.dumps(throughput, indent=2) + "\n", encoding="utf-8")
    return losses


def hash_perm(perm):
    import hashlib

    return hashlib.sha256(np.ascontiguousarray(perm).tobytes()).hexdigest()


def main():
    run()


if __name__ == "__main__":
    sys.exit(main())

"""The Mac-training check (docs/prompt_session7.md, step 5, with correction 6; idea I-24 of docs/ideas_register.md).
Run on Johann's Mac by scripts/check_mac_training_macos.sh, or for one arm with
`caffeinate -i uv run python src/train/mac_training_check.py --arms b`; never on the linked shell or the pod.

For each arm, 20 optimizer steps of that arm's training step on the Mac's GPU (MPS), with 64 phase-3 training records
(the first 32 of data/processed/public/pubmedqa_answer_train.jsonl and the first 32 of readmit_window_train.jsonl),
cycled in a fixed order, at 32 records per step: micro-batch 8 with accumulation 4 first, then 4 with 8, then 2 with 16
if memory runs out; if micro-batch 2 runs out of memory, once more with gradient checkpointing.
- Arm A: reader v1 in Laya's format (models/fiorillo_v0_3/reader_v1_laya_format/) with a fresh head, as
  src/train/train_laya.py runs it on MPS: float32, gradient checkpointing, AdamW 2.5e-5 (reader) and 1e-4 (head), items
  from src/train/build_items.py at max_len 768.
- Arm B: Qwen3-4B-Base (models/qwen3_4b_base/, downloaded at the fixed revision) with the adapters, projection and head
  of src/train/train_qwen_head.py: weights in bfloat16 (float16 if bfloat16 fails), adapters, projection and head in
  float32, AdamW 5e-5 (adapters, correction 1) and 1e-4 (projection and head), items from src/train/qwen_items.py at
  max_len 768.
Correction 6: each try runs in its own process (this file with --one), so a try that runs out of memory gives all of it
back before the next try starts; every try runs with PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0 (the GPU's memory capped at
Metal's recommended working set) and PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 (PyTorch refuses a low ratio above the high
one), so a try too big for memory fails as out of memory instead of running on in swap; and a try is stopped, and
counted as out of memory, when 600 seconds pass without an optimizer step finishing (for the first step, from the try's
start).
Recorded per arm: seconds per optimizer step over steps 6 to 20, peak memory (the largest of
torch.mps.driver_allocated_memory() after each step), the micro-batch that fit, gradient checkpointing, the precision,
and the hours that 100,000 records would take at 32 records per step; per try, how long it took, Metal's recommended
working set and the cap in GiB, and the swap in use before and after it (sysctl vm.swapusage); and the Mac's chip and
memory (sysctl), macOS's version (sw_vers) and the torch version. A run of some arms keeps the other arms' results
already in the file, with the time each was measured, if the file was written on this Mac. Nothing is saved but
results/fiorillo_v0_3/mac_training_check.json. It reads only this project's public records.

Usage: uv run python src/train/mac_training_check.py [--arms a b] [--steps 20]
"""
import argparse
import datetime as dt
import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
from build_items import build_items, load_questions, read_records  # noqa: E402
from train_laya import collate, log_loss  # noqa: E402

RECORDS = ROOT / "data" / "processed" / "public"
LAYA_DIR = ROOT / "models" / "fiorillo_v0_3" / "reader_v1_laya_format"
QWEN_DIR = ROOT / "models" / "qwen3_4b_base"
OUT = ROOT / "results" / "fiorillo_v0_3" / "mac_training_check.json"
RECORDS_PER_STEP = 32
SEED = 20260927
PLANS = [(8, False), (4, False), (2, False), (2, True)]  # (micro-batch, gradient checkpointing added)
MPS_ENV = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8"}  # correction 6
STEP_LIMIT = 600  # seconds without an optimizer step finishing before a try is stopped (correction 6)


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable ({type(e).__name__})"


def machine():
    mem = sh(["sysctl", "-n", "hw.memsize"])
    return {"chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "memory_gib": round(int(mem) / 2**30, 1) if mem.isdigit() else mem,
            "macos": sh(["sw_vers", "-productVersion"]), "python": platform.python_version(), "torch": torch.__version__,
            "mps_available": bool(torch.backends.mps.is_available())}


def records64(records_dir):
    out = []
    for qid in ("pubmedqa_answer", "readmit_window"):
        with open(Path(records_dir) / f"{qid}_train.jsonl", encoding="utf-8") as f:
            for k, line in enumerate(f):
                if k >= 32:
                    break
                out.append(json.loads(line))
    return out


def is_oom(e):
    text = str(e).lower()
    return "out of memory" in text or isinstance(e, getattr(torch, "OutOfMemoryError", ()))


def mps_memory():
    try:
        return torch.mps.driver_allocated_memory() / 2**30
    except Exception:  # noqa: BLE001
        return None


def run_steps(model, items, groups, forward, micro, steps, device):
    """steps optimizer steps of 32 records each; items cycled in order. Returns seconds per step over steps 6 to the last."""
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)
    trainable = [p for g in groups for p in g["params"]]
    accum = RECORDS_PER_STEP // micro
    pos, times, peak = 0, [], 0.0
    model.train()
    for step in range(1, steps + 1):
        t = time.time()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            chunk = [items[(pos + i) % len(items)] for i in range(micro)]
            pos = (pos + micro) % len(items)
            loss = forward(chunk) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        torch.mps.synchronize()
        times.append(time.time() - t)
        m = mps_memory()
        peak = max(peak, m or 0.0)
        print(f"step {step}: {times[-1]:.2f} s", flush=True)
    measured = times[5:]
    return sum(measured) / len(measured), peak


def arm_a(records, laya_dir, device, micro, steps, checkpointing=True):
    from laya.common import build_model
    from safetensors.torch import load_file
    from transformers import AutoTokenizer
    from train_laya import reinit_head

    config = json.loads((Path(laya_dir) / "rl_agent_config.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(str(Path(laya_dir) / "tokenizer"))
    model = build_model(config, encoder_dir=str(Path(laya_dir) / "encoder"))
    model.load_state_dict(load_file(str(Path(laya_dir) / "model.safetensors")), strict=True)
    reinit_head(model, SEED)
    if checkpointing:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    items, _ = build_items(records, load_questions(), tokenizer, 768, 192)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    groups = [{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}]

    def forward(chunk):
        b = collate(chunk, tokenizer.pad_token_id)
        logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                            b["marker_mask"].to(device), b["qtype"].to(device))
        return log_loss(logits.float(), b["target"].to(device), b["marker_mask"].to(device)) + 0.0 * act.float().sum()

    return run_steps(model, items, groups, forward, micro, steps, device), "float32"


def arm_b(records, qwen_dir, device, micro, steps, checkpointing=False, dtype=torch.bfloat16):
    from transformers import AutoTokenizer
    from qwen_items import build_qwen_items
    from train_qwen_head import build_model, load_base, param_groups

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_dir))
    base = load_base(qwen_dir, dtype)
    model, _ = build_model(base, SEED, 16, 32, 0.05, checkpointing)
    model.to(device)
    items, _ = build_qwen_items(records, load_questions(), tokenizer, 768)
    groups = [{k: v for k, v in g.items() if k != "name"} for g in param_groups(model, 5e-5, 1e-4)]

    def forward(chunk):
        b = collate(chunk, tokenizer.pad_token_id)
        logits, act = model(b["input_ids"].to(device), b["attention_mask"].to(device), b["marker_pos"].to(device),
                            b["marker_mask"].to(device), b["qtype"].to(device))
        return log_loss(logits.float(), b["target"].to(device), b["marker_mask"].to(device)) + 0.0 * act.float().sum()

    return run_steps(model, items, groups, forward, micro, steps, device), {torch.bfloat16: "bfloat16", torch.float16: "float16"}[dtype]


def recommended_gib():
    try:
        return torch.mps.recommended_max_memory() / 2**30
    except Exception:  # noqa: BLE001
        return None


def one_try(args):
    """One try, run in this process by check() (--one): its step lines, then one RESULT or ERROR line, on stdout."""
    rec = recommended_gib()
    ratio = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
    info = {"recommended_gib": rec, "cap_gib": rec * float(ratio) if rec is not None and ratio else None,
            "mps_env": {k: os.environ.get(k) for k in MPS_ENV}}
    torch.manual_seed(SEED)
    device = torch.device("mps")
    records = records64(args.records_dir)
    try:
        if args.arm == "a":
            (sec, peak), precision = arm_a(records, args.laya_dir, device, args.micro, args.steps)
        else:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
            (sec, peak), precision = arm_b(records, args.qwen_dir, device, args.micro, args.steps, args.checkpointing, dtype)
    except Exception as e:  # noqa: BLE001
        print("ERROR " + json.dumps({"error": f"{type(e).__name__}: {str(e)[:300]}", "oom": is_oom(e), **info}), flush=True)
        return 3
    print("RESULT " + json.dumps({"seconds_per_step": sec, "peak_memory_gib": peak, "precision": precision, **info}), flush=True)
    return 0


def swap_used_gib():
    """Swap in use in GiB, from sysctl vm.swapusage ("total = 2048.00M  used = 1024.00M  free = ..."), or None."""
    m = re.search(r"used = ([0-9.]+)([MG])", sh(["sysctl", "-n", "vm.swapusage"]))
    if not m:
        return None
    value = float(m.group(1))
    return round(value / 1024 if m.group(2) == "M" else value, 2)


def run_try(cmd, env, step_limit=STEP_LIMIT, echo=print, clock=time.monotonic):
    """Run one try as a child process, passing its step lines on. Returns (status, payload): "result" or "error" with
    the payload of the child's RESULT or ERROR line, or "stopped" when step_limit seconds pass without a step line
    (for the first step, from the start)."""
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, text=True, bufsize=1)
    lines = queue.Queue()

    def reader():
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=reader, daemon=True).start()
    last, status, payload = clock(), None, None
    while True:
        try:
            line = lines.get(timeout=0.5)
        except queue.Empty:
            if clock() - last > step_limit:
                proc.kill()
                proc.wait()
                if status is not None:
                    return status, payload
                return "stopped", {"error": f"stopped: {step_limit:g} s passed without an optimizer step finishing "
                                            "(counted as out of memory, correction 6)", "oom": True}
            continue
        if line is None:
            break
        line = line.rstrip("\n")
        if line.startswith("RESULT "):
            status, payload = "result", json.loads(line[len("RESULT "):])
        elif line.startswith("ERROR "):
            status, payload = "error", json.loads(line[len("ERROR "):])
        else:
            echo(line)
            if line.startswith("step "):
                last = clock()
    code = proc.wait()
    if status is None:
        status, payload = "error", {"error": f"the try's process ended with code {code} and printed no result", "oom": False}
    return status, payload


def try_command(arm, micro, checkpointing, dtype, args):
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", "--arm", arm, "--micro", str(micro), "--dtype", dtype,
           "--steps", str(args.steps), "--records-dir", str(args.records_dir), "--laya-dir", str(args.laya_dir),
           "--qwen-dir", str(args.qwen_dir)]
    return cmd + (["--checkpointing"] if checkpointing else [])


def check(arm, args, run=None):
    """The ladder for one arm, one process per try (correction 6). run(cmd, env) returns (status, payload); run_try by
    default."""
    if run is None:
        def run(cmd, env):
            return run_try(cmd, env, args.step_limit)
    attempts = []
    dtypes = ["bfloat16", "float16"] if arm == "b" else ["float32"]
    for dtype in dtypes:
        dtype_failed = False
        for micro, extra in PLANS:
            if arm == "a" and extra:
                continue  # arm A already runs with gradient checkpointing, as train_laya.py does
            ckpt = arm == "a" or extra
            print(f"try: micro-batch {micro}, gradient checkpointing {'on' if ckpt else 'off'}, {dtype}", flush=True)
            swap_before, start = swap_used_gib(), time.time()
            status, payload = run(try_command(arm, micro, extra, dtype, args), {**os.environ, **MPS_ENV})
            common = {"micro_batch": micro, "accumulation": RECORDS_PER_STEP // micro, "gradient_checkpointing": ckpt,
                      "precision": payload.get("precision", dtype), "try_seconds": round(time.time() - start, 1),
                      "swap_used_gib_before": swap_before, "swap_used_gib_after": swap_used_gib(),
                      "recommended_working_set_gib": payload.get("recommended_gib"), "cap_gib": payload.get("cap_gib")}
            if status == "result":
                sec = payload["seconds_per_step"]
                result = {**common, "seconds_per_step": sec, "peak_memory_gib": payload["peak_memory_gib"],
                          "hours_per_100000_records": (100000 / RECORDS_PER_STEP) * sec / 3600, "fit": True}
                attempts.append(result)
                return {"attempts": attempts, **result}
            attempts.append({**common, "error": payload.get("error"), "fit": False})
            if not payload.get("oom"):
                dtype_failed = True
                break
        if not dtype_failed:
            break
    return {"attempts": attempts, "fit": False}


def previous(path, info):
    """(arms, runs) already in the file, if it was written on this Mac (same chip and memory); else nothing."""
    try:
        old = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}, []
    if any(old.get("machine", {}).get(k) != info.get(k) for k in ("chip", "memory_gib")):
        return {}, []
    arms = old.get("arms", {})
    for res in arms.values():
        res.setdefault("time_utc", old.get("time_utc"))
        res.setdefault("tries_run", "in one process, before correction 6")
    runs = old.get("runs") or [{"time_utc": old.get("time_utc"), "arms_recorded": sorted(arms)}]
    return arms, runs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", nargs="+", choices=("a", "b"), default=["a", "b"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--laya-dir", type=Path, default=LAYA_DIR)
    parser.add_argument("--qwen-dir", type=Path, default=QWEN_DIR)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--step-limit", type=float, default=STEP_LIMIT)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)  # one try, in a child process
    parser.add_argument("--arm", choices=("a", "b"), help=argparse.SUPPRESS)
    parser.add_argument("--micro", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--dtype", default="float32", help=argparse.SUPPRESS)
    parser.add_argument("--checkpointing", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        return one_try(args)
    info = machine()
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    kept, runs = previous(args.out, info)
    out = {"plan": "docs/prompt_session7.md, step 5 (I-24), with correction 6", "machine": info,
           "records_per_step": RECORDS_PER_STEP, "steps": args.steps, "timed_steps": f"6 to {args.steps}",
           "tries": {"process": "one per try", "mps_env": MPS_ENV, "step_limit_seconds": args.step_limit},
           "arms": {k: v for k, v in kept.items() if k not in args.arms},
           "runs": runs + [{"time_utc": now, "arms": list(args.arms)}], "time_utc": now}

    def write():
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")

    if not info["mps_available"]:
        out["error"] = "MPS is not available on this Mac, so nothing was run"
    else:
        records64(args.records_dir)  # stop here, before any try, if the records are missing
        for arm in args.arms:
            print(f"arm {arm.upper()}", flush=True)
            out["arms"][arm] = {**check(arm, args), "time_utc": now, "tries_run": "one process per try (correction 6)"}
            write()
    write()
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""I-39, the weight average (docs/prompt_session9.md, THE RECIPE RULE).

s9_ei_wavg is one model built from the winning recipe's n runs (three; two if the cost rule left the winner with two,
correction 7): each adapted layer's n LoRA updates are averaged exactly, by joining the n runs' A matrices along the rank
and their B matrices divided by n (for three runs, rank 48 and alpha 96; for two, rank 32 and alpha 64; alpha divided by
the rank stays 2, and B_cat A_cat equals the mean of the n B A products); every other trainable weight (the projection
and Laya's head) is averaged elementwise. The folder has src/train/train_qwen_head.py's phase layout (adapter.safetensors,
config.json with the joined r and alpha, the tokenizer files), so train_qwen_head.py predict loads it as it loads a run.

The check: for every adapted layer, the joined adapter's update (scale times B_cat A_cat) equals the mean of the n
runs' updates within 1e-5 (the largest absolute difference of any entry); weight_average.json records the largest
difference per layer kind, the sources' SHA-256 and the time, and the build stops if any layer fails.

Usage (the pod): python src/eval/weight_average_session9.py build --runs MODELS/s9_x_1 MODELS/s9_x_2 [MODELS/s9_x_3] \\
                     --out MODELS/s9_ei_wavg [--results results/fiorillo_v0_5] [--device cuda]
"""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "fiorillo_v0_5"
TOLERANCE = 1e-5
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json")
PLAN = "docs/prompt_session9.md, THE RECIPE RULE (I-39, the weight average)"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_a(name):
    return ".lora_A." in name or name.startswith("lora_A.")


def is_b(name):
    return ".lora_B." in name or name.startswith("lora_B.")


def layer_of(name):
    return name.replace(".lora_A.", ".lora_X.").replace(".lora_B.", ".lora_X.")


def join(states, n=3):
    """The weight average's state from n runs' adapter states (each a {name: float32 tensor} dict with the same keys)."""
    keys = list(states[0])
    for s in states[1:]:
        if set(s) != set(keys):
            raise SystemExit("the runs' adapters hold different parameters")
    out = {}
    for k in keys:
        ts = [s[k].to(torch.float32) for s in states]
        if any(t.shape != ts[0].shape for t in ts):
            raise SystemExit(f"{k}: the runs' shapes differ")
        if is_a(k):
            out[k] = torch.cat(ts, dim=0).contiguous()  # (r, in) each, joined along the rank
        elif is_b(k):
            out[k] = (torch.cat(ts, dim=1) / n).contiguous()  # (out, r) each, joined along the rank, divided by n
        else:
            out[k] = (sum(ts) / n).contiguous()
    return out


def check(states, joined, scales, joined_scale, device="cpu"):
    """The largest absolute difference, per adapted layer, between the joined update and the mean of the runs' updates."""
    diffs = {}
    for k in joined:
        if not is_a(k):
            continue
        kb = k.replace(".lora_A.", ".lora_B.") if ".lora_A." in k else k.replace("lora_A.", "lora_B.", 1)
        with torch.no_grad():
            mean = sum(sc * (s[kb].to(device, torch.float64) @ s[k].to(device, torch.float64)) for s, sc in zip(states, scales)) / len(states)
            got = joined_scale * (joined[kb].to(device, torch.float64) @ joined[k].to(device, torch.float64))
            diffs[layer_of(k)] = float((got - mean).abs().max())
    return diffs


def build(runs, out, results=RESULTS, device="cpu"):
    runs, out = [Path(r) for r in runs], Path(out)
    if len(runs) < 2:
        raise SystemExit("the weight average takes the winning recipe's runs, at least two (correction 7)")
    configs = [json.loads((r / "config.json").read_text(encoding="utf-8")) for r in runs]
    for c in configs[1:]:
        for key in ("adapters", "projection", "head_layers", "n_act", "max_len", "base_revision", "item_format"):
            if c.get(key) != configs[0].get(key):
                raise SystemExit(f"the runs' config.json differ in {key}")
    a = configs[0]["adapters"]
    r, alpha = int(a["r"]), float(a["alpha"])
    states = [load_file(str(rr / "adapter.safetensors")) for rr in runs]
    joined = join(states, len(runs))
    joined_r, joined_alpha = r * len(runs), alpha * len(runs)
    diffs = check(states, joined, [alpha / r] * len(runs), joined_alpha / joined_r, device)
    worst = max(diffs.values()) if diffs else 0.0
    kinds = {}
    for layer, d in diffs.items():
        kind = layer.split(".lora_X.")[0].split(".")[-1]
        kinds[kind] = max(kinds.get(kind, 0.0), d)
    if worst > TOLERANCE:
        raise SystemExit(f"the joined adapter's update differs from the mean of the runs' updates by {worst:.3g} > {TOLERANCE}; stop")
    out.mkdir(parents=True, exist_ok=True)
    save_file(joined, str(out / "adapter.safetensors"))
    cfg = dict(configs[0])
    cfg["adapters"] = {**a, "r": joined_r, "alpha": joined_alpha}
    cfg.update({"system": "s9_ei_wavg", "written_utc": now(),
                "weight_average": {"plan": PLAN, "sources": [str(rr) for rr in runs],
                                   "source_systems": [c.get("system") for c in configs],
                                   "method": f"A matrices joined along the rank; B matrices joined along the rank and divided by {len(runs)}; "
                                             "every other trainable weight averaged elementwise"}})
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    for name in TOKENIZER_FILES:
        src = runs[0] / name
        if src.exists():
            shutil.copyfile(src, out / name)
    record = {"plan": PLAN, "written_utc": now(), "sources": {str(rr): {"system": c.get("system"),
                                                                        "adapter_sha256": sha256_file(rr / "adapter.safetensors")}
                                                              for rr, c in zip(runs, configs)},
              "rank": joined_r, "alpha": joined_alpha, "scale": joined_alpha / joined_r,
              "adapted_layers": len(diffs), "largest_difference": worst, "largest_difference_by_kind": kinds,
              "tolerance": TOLERANCE, "passed": True, "out": str(out),
              "adapter_sha256": sha256_file(out / "adapter.safetensors")}
    Path(results).mkdir(parents=True, exist_ok=True)
    (Path(results) / "weight_average.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("rank", "alpha", "adapted_layers", "largest_difference", "passed")}))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--runs", nargs="+", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--results", type=Path, default=RESULTS)
    b.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    build(args.runs, args.out, args.results, args.device)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Why did arm B ignore its input? A check of Laya-format models before training (docs/prompt_session4.md, correction 5).

Run 1's arm B (BioClinical ModernBERT-large's reader under Laya's decision head) gave nearly the same probabilities
for every example. This script compares model folders on the same items and reports, per question:
- marker spread: how much the reader's output at the answer slots (the [MASK] markers) varies across examples,
  as the mean over dimensions of the across-example standard deviation divided by the root mean square of the
  vectors. Near 0 means every example looks the same at the answer slots.
- head spread: the same after the decision head's two layers, which is what the scorer reads.
- probability spread: the standard deviation across examples of each option's predicted probability, averaged.
- outsized dimensions: the three largest per-dimension mean absolute values of the reader's output over all tokens,
  as multiples of the median dimension, and the mean token norm.
- frozen-feature probes: a logistic regression (scikit-learn; standardized features; C = 0.1) fitted on training
  items and scored on validation items, as log loss beside the training prior's log loss, for three feature sets:
  the reader's answer-slot vectors (averaged over options), the reader's mean over all tokens, and the head's
  answer-slot vectors (averaged). If the mean over tokens predicts the label and the answer slots do not, the
  information is in the reader but does not reach the answer slots.
- a numeric check: the largest difference between probabilities computed in float16 (as in training) and float32.
Validation items are drawn regardless of the calibrate flag (for PubMedQA most are pqa_artificial, yes or no only),
because the 50 calibrate items are too few for a probe.

Usage: python src/eval/diagnose_reader.py --model laya=/tmp/models/laya-base --model bioclinical=/tmp/models/arm_bioclinical_init \
    --train-items items/train.pt --val-items items/validation.pt --out outputs/diagnosis.json [--n-train 1500 --n-val 400]
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from laya.common import build_model
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "train"))
from train_laya import collate, item_key, pick_device  # noqa: E402

SEED = 20260927


def load(model_dir, device):
    config = json.loads((Path(model_dir) / "rl_agent_config.json").read_text(encoding="utf-8"))
    model = build_model(config, encoder_dir=str(Path(model_dir) / "encoder"), pretrained=False)
    model.load_state_dict({k: v.to(torch.float32) for k, v in load_file(str(Path(model_dir) / "model.safetensors")).items()}, strict=True)
    model.encoder.config.reference_compile = False
    return model.to(device).eval()


@torch.no_grad()
def features(model, items, device, batch_size, autocast_dtype):
    """Per item: reader answer-slot vectors, reader token mean, head answer-slot vectors, probabilities; plus dimension stats."""
    out = {"enc_slots": [], "enc_mean": [], "head_slots": [], "probs": [], "labels": []}
    abs_sum, n_tokens, norm_sum, nonfinite, forward_check = None, 0, 0.0, 0, None
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        b = collate(chunk, 0)
        ids, att = b["input_ids"].to(device), b["attention_mask"].to(device)
        mpos, mmask, qtype = b["marker_pos"].to(device), b["marker_mask"].to(device), b["qtype"].to(device)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
            enc = model.encoder(input_ids=ids, attention_mask=att).last_hidden_state
            h = enc + model.type_emb(qtype)[:, None, :]
            pad = ~att.bool()
            for layer in model.head.layers:
                h = layer(h, src_key_padding_mask=pad)
            idx = mpos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m_head = torch.gather(h, 1, idx)
            logits = model.scorer(m_head).squeeze(-1).float().masked_fill(~mmask, -1e4)
            if forward_check is None:  # the steps above must reproduce DecisionModel.forward
                reference, _ = model(ids, att, mpos, mmask, qtype)
                forward_check = float((reference.float() - logits).abs().max())
        enc, m_head = enc.float(), m_head.float()
        nonfinite += int((~torch.isfinite(enc)).sum())
        m_enc = torch.gather(enc, 1, idx)
        a = att.unsqueeze(-1).float()
        tok_abs = (enc.abs() * a).sum((0, 1))
        abs_sum = tok_abs if abs_sum is None else abs_sum + tok_abs
        n_tokens += int(att.sum())
        norm_sum += float((enc.norm(dim=-1) * att).sum())
        out["enc_mean"].append(((enc * a).sum(1) / a.sum(1)).cpu())
        out["enc_slots"].append(m_enc.cpu())
        out["head_slots"].append(m_head.cpu())
        out["probs"].append(torch.softmax(logits, -1).cpu())
        out["labels"].append(b["label"])
    result = {k: torch.cat(v).numpy() for k, v in out.items()}
    per_dim = (abs_sum / n_tokens).cpu().numpy()
    top = np.sort(per_dim)[::-1][:3]
    result["dims"] = {"top3_over_median": [round(float(x / np.median(per_dim)), 1) for x in top],
                      "mean_token_norm": round(norm_sum / n_tokens, 2), "nonfinite_values": nonfinite,
                      "forward_check_max_logit_difference": round(forward_check, 5)}
    return result


def spread(vectors):
    """Mean over dimensions of the across-example standard deviation, divided by the root mean square (per slot, averaged)."""
    values = []
    for j in range(vectors.shape[1]):
        v = vectors[:, j, :]
        values.append(float(v.std(0).mean() / np.sqrt((v ** 2).mean())))
    return round(float(np.mean(values)), 4)


def probe(train_x, train_y, val_x, val_y, k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_x)
    clf = LogisticRegression(C=0.1, max_iter=3000).fit(scaler.transform(train_x), train_y)
    p = np.full((len(val_y), k), 1e-6)
    p[:, clf.classes_] = clf.predict_proba(scaler.transform(val_x))
    p /= p.sum(1, keepdims=True)
    return round(float(log_loss(val_y, p, labels=list(range(k)))), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", required=True, help="name=folder, repeatable")
    parser.add_argument("--train-items", type=Path, required=True)
    parser.add_argument("--val-items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-val", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = pick_device(args.device)
    autocast_dtype = torch.float16 if device.type == "cuda" else None
    rng = random.Random(SEED)
    train_all = torch.load(args.train_items, weights_only=False)
    val_all = torch.load(args.val_items, weights_only=False)
    samples = {}
    for key in sorted({item_key(it) for it in val_all}):
        seen, train = set(), []
        for it in rng.sample([it for it in train_all if item_key(it) == key], k=sum(1 for it in train_all if item_key(it) == key)):
            if it["id"] not in seen:  # labeled PubMedQA items are repeated in the training file
                seen.add(it["id"])
                train.append(it)
            if len(train) == args.n_train:
                break
        val = rng.sample([it for it in val_all if item_key(it) == key], k=min(args.n_val, sum(1 for it in val_all if item_key(it) == key)))
        samples[key] = (train, val)

    report = {"settings": {"n_train": args.n_train, "n_val": args.n_val, "seed": SEED, "device": str(device),
                           "autocast": str(autocast_dtype), "probe": "logistic regression, standardized, C = 0.1"}, "models": {}}
    for spec in args.model:
        name, folder = spec.split("=", 1)
        model = load(folder, device)
        entry = {}
        for key, (train, val) in samples.items():
            k = len(val[0]["markers"])
            ft = features(model, train, device, args.batch_size, autocast_dtype)
            fv = features(model, val, device, args.batch_size, autocast_dtype)
            prior = (np.bincount(ft["labels"], minlength=k) + 1) / (len(ft["labels"]) + k)  # add-one, so no option has zero
            prior_ll = round(float(-np.mean(np.log(np.clip(prior[fv["labels"]], 1e-6, None)))), 4)
            check = val[: min(32, len(val))]
            f16 = features(model, check, device, args.batch_size, autocast_dtype)["probs"]
            f32 = features(model, check, device, args.batch_size, None)["probs"]
            entry[key] = {
                "n_train": len(train), "n_val": len(val),
                "marker_spread_reader": spread(fv["enc_slots"]),
                "marker_spread_head": spread(fv["head_slots"]),
                "probability_spread": round(float(fv["probs"].std(0).mean()), 4),
                "outsized_dimensions": fv["dims"],
                "probe_log_loss": {
                    "prior": prior_ll,
                    "reader_answer_slots": probe(ft["enc_slots"].mean(1), ft["labels"], fv["enc_slots"].mean(1), fv["labels"], k),
                    "reader_token_mean": probe(ft["enc_mean"], ft["labels"], fv["enc_mean"], fv["labels"], k),
                    "head_answer_slots": probe(ft["head_slots"].mean(1), ft["labels"], fv["head_slots"].mean(1), fv["labels"], k),
                },
                "float16_vs_float32_max_probability_difference": round(float(np.abs(f16 - f32).max()), 4),
            }
            print(f"{name} {key}: {json.dumps(entry[key])}", flush=True)
        report["models"][name] = entry
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())

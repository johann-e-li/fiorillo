"""Validation-only checks behind ideas I-10 and I-13 of docs/ideas_register.md.

Reads v0.1's (arm_laya's) PubMedQA validation predictions, keeps the 150 PQA-L items
(the 2,000 PQA-A monitoring items are left out), and reports:
  - accuracy, log loss, and each true answer's share of the log loss;
  - 10-fold cross-validation, repeated 20 times (seed 20260929), of two calibrations
    fitted on v0.1's raw probabilities: one temperature, and one temperature plus a
    bias for each answer (idea I-10);
  - accuracy when only the most confident share of items is answered (idea I-13).
The test split is not read. Usage: python3 src/eval/validation_idea_checks.py
Writes results/ideas_v0_3/validation_checks.json.
"""
import csv
import json
import math

import numpy as np

SEED = 20260929
PRED = "results/public_v0/predictions/arm_laya_pubmedqa_answer_validation.csv"
RAW = "results/public_v0/predictions/arm_laya-raw_pubmedqa_answer_validation.csv"
ITEMS = "data/processed/public/pubmedqa_answer_validation.jsonl"
OUT = "results/ideas_v0_3/validation_checks.json"
EPS = 1e-12


def load(path):
    ids, y, p = [], [], []
    for r in csv.DictReader(open(path)):
        ids.append(r["id"])
        y.append(int(r["label"]))
        p.append([float(r["p0"]), float(r["p1"]), float(r["p2"])])
    return ids, np.array(y), np.array(p)


def fit(z, y, use_bias, iters=4000, lr=0.05):
    a, b = 0.0, np.zeros(3)  # scale = exp(a)
    for _ in range(iters):
        s = math.exp(a)
        L = z * s + (b if use_bias else 0)
        L = L - L.max(1, keepdims=True)
        q = np.exp(L)
        q /= q.sum(1, keepdims=True)
        g = q.copy()
        g[np.arange(len(y)), y] -= 1
        g /= len(y)
        a -= lr * (g * z * s).sum()
        if use_bias:
            b -= lr * 10 * g.sum(0)
            b -= b.mean()
    return a, b


def apply(z, a, b, use_bias):
    L = z * math.exp(a) + (b if use_bias else 0)
    L = L - L.max(1, keepdims=True)
    q = np.exp(L)
    return q / q.sum(1, keepdims=True)


def main():
    rng = np.random.default_rng(SEED)
    labeled = set()
    for line in open(ITEMS):
        d = json.loads(line)
        if "pqa_labeled" in str(d.get("group")):
            labeled.add(d["id"])
    ids, y, p = load(PRED)
    ids_raw, y_raw, p_raw = load(RAW)
    assert ids == ids_raw and (y == y_raw).all()
    keep = np.array([i in labeled for i in ids])
    y, p, p_raw = y[keep], p[keep], p_raw[keep]
    n = len(y)
    nll = -np.log(np.clip(p[np.arange(n), y], EPS, 1))
    out = {
        "source": {"predictions": PRED, "raw_predictions": RAW, "items": ITEMS, "split": "validation (PQA-L items only); test not read"},
        "n": int(n),
        "label_counts_yes_no_maybe": np.bincount(y, minlength=3).tolist(),
        "accuracy": round(float((p.argmax(1) == y).mean()), 4),
        "log_loss_with_temperature": round(float(nll.mean()), 4),
        "by_true_answer": {},
        "maybe_top_answer_count": int((p.argmax(1) == 2).sum()),
        "mean_p_maybe_all_items": round(float(p[:, 2].mean()), 4),
    }
    for c, name in enumerate(["yes", "no", "maybe"]):
        s = y == c
        out["by_true_answer"][name] = {
            "n": int(s.sum()),
            "mean_log_loss": round(float(nll[s].mean()), 4),
            "share_of_total_log_loss": round(float(nll[s].sum() / nll.sum()), 4),
            "mean_p_of_this_answer": round(float(p[s, c].mean()), 4),
        }
    z = np.log(np.clip(p_raw, EPS, 1))
    res = {False: [], True: []}
    for _ in range(20):
        perm = rng.permutation(n)
        folds = np.array_split(perm, 10)
        for use_bias in (False, True):
            tot = 0.0
            for k in range(10):
                te = folds[k]
                tr = np.setdiff1d(perm, te)
                a, b = fit(z[tr], y[tr], use_bias)
                q = apply(z[te], a, b, use_bias)
                tot += -np.log(np.clip(q[np.arange(len(te)), y[te]], EPS, 1)).sum()
            res[use_bias].append(tot / n)
    d = np.array(res[False]) - np.array(res[True])
    a, b = fit(z, y, True)
    out["calibration_cv"] = {
        "design": "10-fold cross-validation on the 150 items, repeated 20 times, seed %d" % SEED,
        "temperature_only_log_loss": round(float(np.mean(res[False])), 4),
        "temperature_plus_bias_log_loss": round(float(np.mean(res[True])), 4),
        "difference_mean": round(float(d.mean()), 4),
        "difference_min_over_repeats": round(float(d.min()), 4),
        "difference_max_over_repeats": round(float(d.max()), 4),
        "fit_on_all_150": {"scale": round(math.exp(a), 3), "bias_yes_no_maybe": [round(float(v), 3) for v in b]},
    }
    order = np.argsort(-p.max(1))
    correct = (p.argmax(1) == y)[order]
    cov = []
    for share in (1.0, 0.8, 0.6, 0.5, 0.4, 0.2):
        k = int(round(share * n))
        cov.append({
            "answered_share": share,
            "answered_items": k,
            "answered_correct": int(correct[:k].sum()),
            "answered_accuracy": round(float(correct[:k].mean()), 4),
            "escalated_items": int(n - k),
            "escalated_correct_if_answered": int(correct[k:].sum()),
            "lowest_confidence_answered": round(float(p.max(1)[order][k - 1]), 4),
        })
    out["coverage"] = cov
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

"""Public v0 baselines (docs/prompt_session4.md, step 5): the prior for both questions, and LightGBM on UCI.

The prior gives every row the training-split label frequencies. LightGBM uses the same fields as the UCI state
(numeric fields as numbers; the rest, including each medication, as categories fitted on the training split),
at most 1,000 trees, learning rate 0.05, early stopping after 50 rounds without improvement in validation
multi-class log loss, seed 20260927. Predictions for validation and test go to
results/public_v0/predictions/<system>_<qid>_<split>.csv (id, label, p0, p1, ...); a summary goes to
results/public_v0/baselines.json.

Usage: python src/eval/baselines.py [--records data/processed/public] [--out results/public_v0]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "tasks"))
import records as R  # noqa: E402

QIDS = {"pubmedqa_answer": 3, "readmit_window": 3}
UCI_NUMERIC = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
               "number_emergency", "number_inpatient", "number_diagnoses"]
UCI_CATEGORICAL = ["age band", "sex", "admission type", "admission source", "discharge disposition", "diag_1",
                   "diag_2", "diag_3", "max_glu_serum", "A1Cresult", "change", "diabetesMed", "medical_specialty"]


def write_predictions(path, rows, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    k = probs.shape[1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(k)])
        for row, p in zip(rows, probs):
            w.writerow([row["id"], row["label"]] + [f"{x:.6f}" for x in p])


def prior(train_rows, k):
    counts = np.bincount([r["label"] for r in train_rows], minlength=k).astype(float)
    return counts / counts.sum()


def uci_frame(rows, categories=None):
    import pandas as pd
    from uci_readmission import MEDICATIONS

    data = {c: [r["state"].get(c, np.nan) for r in rows] for c in UCI_NUMERIC}
    for c in UCI_CATEGORICAL:
        data[c] = [r["state"].get(c) for r in rows]
    for m in MEDICATIONS:
        data[f"med_{m}"] = [r["state"].get("medications", {}).get(m, "No") for r in rows]
    frame = pd.DataFrame(data)
    cat_cols = UCI_CATEGORICAL + [f"med_{m}" for m in MEDICATIONS]
    if categories is None:
        categories = {c: sorted(x for x in frame[c].dropna().unique()) for c in cat_cols}
    for c in cat_cols:
        frame[c] = pd.Categorical(frame[c], categories=categories[c])
    for c in UCI_NUMERIC:
        frame[c] = frame[c].astype(float)
    return frame, categories


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records", type=Path, default=R.OUT)
    parser.add_argument("--out", type=Path, default=R.RESULTS)
    args = parser.parse_args()
    summary = {}
    data = {q: {s: R.read_jsonl(args.records / f"{q}_{s}.jsonl") for s in R.SPLITS} for q in QIDS}

    for qid, k in QIDS.items():
        p = prior(data[qid]["train"], k)
        summary[f"prior_{qid}"] = [round(float(x), 6) for x in p]
        for split in ("validation", "test"):
            rows = data[qid][split]
            write_predictions(args.out / "predictions" / f"prior_{qid}_{split}.csv", rows, np.tile(p, (len(rows), 1)))

    import lightgbm as lgb

    train, cats = uci_frame(data["readmit_window"]["train"])
    val, _ = uci_frame(data["readmit_window"]["validation"], cats)
    test, _ = uci_frame(data["readmit_window"]["test"], cats)
    y = {s: np.array([r["label"] for r in data["readmit_window"][s]]) for s in R.SPLITS}
    model = lgb.LGBMClassifier(objective="multiclass", num_class=3, n_estimators=1000, learning_rate=0.05,
                               random_state=R.SEED, deterministic=True, force_col_wise=True, n_jobs=4, verbose=-1)
    model.fit(train, y["train"], eval_set=[(val, y["validation"])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])
    for split, frame in (("validation", val), ("test", test)):
        probs = model.predict_proba(frame, num_iteration=model.best_iteration_)
        write_predictions(args.out / "predictions" / f"lightgbm_readmit_window_{split}.csv",
                          data["readmit_window"][split], probs)
    summary["lightgbm_readmit_window"] = {
        "best_iteration": int(model.best_iteration_),
        "validation_multi_logloss": round(float(model.best_score_["valid_0"]["multi_logloss"]), 6),
        "lightgbm_version": lgb.__version__, "features": len(train.columns),
    }
    (args.out / "baselines.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

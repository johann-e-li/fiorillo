"""How large the ECE of a perfectly calibrated model is, by chance alone, on a test split of a given size.

Takes a system's calibrated validation predictions (validation items marked calibrate), draws n rows with
replacement, draws each row's label from the row's own probabilities (so the model is calibrated by construction),
and computes the ECE with the project's definition (top answer, 15 bins, src/eval/metrics.py). Repeated many times,
this gives the chance distribution of the ECE that release criterion 1 compares with 0.05. It reads no test labels
and no test predictions; n is the test split's size. Exploratory: it has no bearing on the release bar.

Usage: python src/eval/ece_noise_floor.py --predictions DIR/arm_laya_pubmedqa_answer_validation.csv
       --records data/processed/public/pubmedqa_answer_validation.jsonl --n 500 [--draws 5000] [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M  # noqa: E402

SEED = 20260927
LIMIT = 0.05


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions", type=Path, required=True, help="a calibrated validation predictions CSV")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="rows per draw (the test split's size)")
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    keep = {json.loads(line)["id"] for line in args.records.open(encoding="utf-8") if json.loads(line).get("calibrate", True)}
    ids, _, probs = M.read_predictions(args.predictions)
    probs = np.asarray(probs)[[i for i, x in enumerate(ids) if x in keep]]
    rng = np.random.default_rng(SEED)
    cumulative = np.cumsum(probs, axis=1)
    values = np.empty(args.draws)
    for d in range(args.draws):
        rows = rng.integers(len(probs), size=args.n)
        u = rng.random(args.n)[:, None]
        labels = np.minimum((u > cumulative[rows]).sum(axis=1), probs.shape[1] - 1)
        values[d] = M.ece(labels, probs[rows])
    out = {"predictions": args.predictions.name, "validation_rows": int(len(probs)), "n": args.n, "draws": args.draws,
           "ece_if_calibrated": {"median": float(np.median(values)), "p05": float(np.quantile(values, 0.05)),
                                 "p95": float(np.quantile(values, 0.95))},
           "share_at_most_0.05": float(np.mean(values <= LIMIT))}
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

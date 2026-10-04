"""The hand-off analysis (docs/prompt_session7.md, HAND-OFF; the analysis part of idea I-13 of docs/ideas_register.md).
Secondary and descriptive: it is never part of the release bar.

For each system and question, items are ranked by the top calibrated probability, highest first (ties broken by record
ID). At coverage 100, 75 and 50 percent, the floor(coverage x n) most confident items are answered and the rest are
handed off (at 100 percent nothing is handed off). Reported: the accuracy on the answered items and on the items handed
off, and the area under the risk-coverage curve (the error rate of the k most confident items, averaged over
k = 1 to n), each with a 95 percent interval from 1,000 bootstrap resamples of items (numpy's default_rng with the seed
20260927, integers(0, n, n) per resample, as src/eval/metrics.py's bootstrap draws them), the ranking made again inside
each resample.

Items: on validation, the items marked calibrate in the question's validation records (PubMedQA's 150 expert-labeled
items; every readmission item; correction 2); on test, every test item, and only once release_candidate.json and
testpreds_log.json exist. Ties are broken by the record ID as text (so pubmedqa-15137012 comes before pubmedqa-7482275). A system's predictions are its calibrated file,
<predictions-dir>/<system>_<qid>_<split>.csv.

Usage: python3 src/eval/handoff_session7.py --split validation --systems arm_laya --predictions-dir results/public_v0/predictions \\
           --out results/fiorillo_v0_3/handoff_validation.json
       python3 src/eval/handoff_session7.py --split test --systems ... --out results/fiorillo_v0_3/handoff_test.json
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval import metrics as M  # noqa: E402

SEED = 20260927
RESAMPLES = 1000
COVERAGES = (1.0, 0.75, 0.5)
QIDS = ("pubmedqa_answer", "readmit_window")
RECORDS = ROOT / "data" / "processed" / "public"


def curve(tie, labels, probs):
    """Accuracy answered and handed off at each coverage, and the area under the risk-coverage curve. Rows are ranked by
    top probability, highest first; tie is a numeric key that breaks ties (the rank of the record ID)."""
    conf = probs.max(axis=1)
    order = np.lexsort((tie, -conf))
    correct = (probs.argmax(axis=1) == labels).astype(float)[order]
    n = len(order)
    out = {}
    for c in COVERAGES:
        k = int(np.floor(c * n))
        out[f"answered_accuracy_{int(c * 100)}"] = float(correct[:k].mean()) if k else None
        out[f"handed_off_accuracy_{int(c * 100)}"] = float(correct[k:].mean()) if k < n else None
        out[f"answered_{int(c * 100)}"] = k
    errors = 1.0 - correct
    out["aurc"] = float(np.mean(np.cumsum(errors) / np.arange(1, n + 1)))
    return out


def analyze(ids, labels, probs, resamples=RESAMPLES, seed=SEED):
    ids = list(ids)
    n = len(ids)
    id_rank = np.empty(n, dtype=np.int64)
    id_rank[np.argsort(np.array(ids, dtype=object), kind="stable")] = np.arange(n)
    point = curve(id_rank, labels, probs)
    rng = np.random.default_rng(seed)
    keys = [k for k in point if not k.startswith("answered_") or "accuracy" in k]
    draws = {k: [] for k in keys}
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        # a resample repeats items; copies of one item are ordered by their place in the resample
        tie = id_rank[idx] * n + np.arange(n)
        c = curve(tie, labels[idx], probs[idx])
        for k in keys:
            draws[k].append(c[k])
    out = {"n": n}
    for k in point:
        if k in draws:
            v = np.array([x for x in draws[k] if x is not None], dtype=float)
            if point[k] is None or not len(v):
                out[k] = {"estimate": point[k], "low": None, "high": None}
            else:
                lo, hi = np.percentile(v, [2.5, 97.5])
                out[k] = {"estimate": point[k], "low": float(lo), "high": float(hi)}
        else:
            out[k] = point[k]
    return out


def calibrate_ids(qid, records_dir=RECORDS):
    keep = set()
    with open(Path(records_dir) / f"{qid}_validation.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("calibrate", True):
                keep.add(r["id"])
    return keep


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--systems", nargs="+", required=True, help="system=predictions_dir, or a system name with --predictions-dir")
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "results" / "fiorillo_v0_3" / "predictions")
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--qids", nargs="+", default=list(QIDS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "fiorillo_v0_3",
                        help="the folder that must hold release_candidate.json and testpreds_log.json before a test split is read")
    args = parser.parse_args(argv)
    if args.split == "test":
        for name in ("release_candidate.json", "testpreds_log.json"):
            if not (args.results / name).exists():
                raise SystemExit(f"{name} does not exist in {args.results}; test predictions are read only after the freeze")
    out = {"plan": "docs/prompt_session7.md, HAND-OFF (secondary, descriptive)", "split": args.split, "coverages": list(COVERAGES),
           "resamples": args.resamples, "seed": SEED,
           "items": "validation: the calibrate items; test: every test item", "systems": {},
           "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    for spec in args.systems:
        system, _, directory = spec.partition("=")
        directory = Path(directory) if directory else args.predictions_dir
        out["systems"][system] = {}
        for qid in args.qids:
            path = directory / f"{system}_{qid}_{args.split}.csv"
            if not path.exists():
                out["systems"][system][qid] = {"missing": str(path)}
                continue
            ids, labels, probs = M.read_predictions(path)
            if args.split == "validation":
                keep = calibrate_ids(qid, args.records_dir)
                rows = [i for i, x in enumerate(ids) if x in keep]
                ids, labels, probs = [ids[i] for i in rows], labels[rows], probs[rows]
            out["systems"][system][qid] = {"file": str(path), **analyze(ids, labels, probs, args.resamples)}
            r = out["systems"][system][qid]
            print(f"{system} {qid} {args.split}: n {r['n']}, answered accuracy at 100, 75 and 50 percent "
                  f"{r['answered_accuracy_100']['estimate']:.3f} / {r['answered_accuracy_75']['estimate']:.3f} / "
                  f"{r['answered_accuracy_50']['estimate']:.3f}, AURC {r['aurc']['estimate']:.4f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

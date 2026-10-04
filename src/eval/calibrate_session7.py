"""Session 7's calibration rule (docs/prompt_session7.md, CALIBRATION; idea I-10 of docs/ideas_register.md).

For each system and question, on the validation items marked calibrate (from the question's validation records): the
temperature family and the temperature_intercepts family of src/eval/calibration_family_cv.py (unchanged; its fit(),
calibrated() and folds_for()), each scored by its out-of-fold log loss over 5 folds and 20 repeats (numpy's
default_rng with the seed 20260927, as calibration_family_cv.py draws its folds). The family with the lower mean
out-of-fold log loss is chosen, an exact tie going to temperature, and both families are fitted on all those items.

Inputs: a system's raw logits (<logits-dir>/<system>_<qid>_<split>_logits.csv, from train_laya.py, run_laya_model.py or
train_qwen_head.py with --logits-dir), whose log-softmax is the log-probability the families take; or, with
--from-raw, a raw predictions CSV (<system>-raw_<qid>_<split>.csv; v0.1 has no logits file), whose printed
probabilities are clipped at 1e-12 before the log, as calibration_family_cv.py does.

Outputs (written only by this script):
- fit: the entry for the system in --out-json (calibration_s7.json for s7_a3 and s7_b3; calibration_s7_arm_laya.json for
  v0.1), with both families' out-of-fold scores and parameters and the chosen family; and the calibrated validation
  predictions <name>_<qid>_validation.csv (the chosen family) and <name>_temperature_<qid>_validation.csv (the
  temperature family alone), probabilities printed with 9 significant digits.
- apply: the same two files for another split (test, after the freeze), with the parameters in --out-json; on the test
  split it refuses to run unless release_candidate.json and testpreds_log.json exist in --results.
<name> is the system's name, or --name (arm_laya_s7cal for v0.1).

Usage: python src/eval/calibrate_session7.py fit --system s7_a3 --logits-dir results/fiorillo_v0_3/logits \\
           --out-json results/fiorillo_v0_3/calibration_s7.json --predictions-dir results/fiorillo_v0_3/predictions
       python src/eval/calibrate_session7.py fit --system arm_laya --name arm_laya_s7cal --from-raw results/public_v0/predictions \\
           --out-json results/fiorillo_v0_3/calibration_s7_arm_laya.json --predictions-dir results/fiorillo_v0_3/predictions
       python src/eval/calibrate_session7.py apply --system s7_a3 --split test ...
"""
import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval import metrics as M  # noqa: E402
from eval.calibration_family_cv import T_MAX, T_MIN, calibrated, fit, folds_for  # noqa: E402

SEED = 20260927
FOLDS, REPEATS = 5, 20
FAMILIES = ("temperature", "temperature_intercepts")
QIDS = ("pubmedqa_answer", "readmit_window")
RECORDS = ROOT / "data" / "processed" / "public"


def read_logits(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    k = sum(1 for c in rows[0] if c.startswith("z") and c[1:].isdigit())
    ids = [r["id"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    z = np.array([[float(r[f"z{j}"]) for j in range(k)] for r in rows])
    return ids, labels, z


def log_softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def load_logp(system, qid, split, logits_dir=None, raw_dir=None):
    """(ids, labels, log-probabilities, source) for one system, question and split."""
    if raw_dir is not None:
        path = Path(raw_dir) / f"{system}-raw_{qid}_{split}.csv"
        ids, labels, probs = M.read_predictions(path)
        zeros = int((np.asarray(probs)[np.arange(len(labels)), labels] == 0).sum())
        return ids, labels, np.log(np.clip(np.asarray(probs), 1e-12, None)), {"file": str(path), "kind": "raw probabilities (printed)",
                                                                                "true_answer_printed_zero": zeros}
    path = Path(logits_dir) / f"{system}_{qid}_{split}_logits.csv"
    ids, labels, z = read_logits(path)
    return ids, labels, log_softmax(z), {"file": str(path), "kind": "logits"}


def calibrate_ids(qid, records_dir=RECORDS):
    keep = set()
    with open(Path(records_dir) / f"{qid}_validation.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("calibrate", True):
                keep.add(r["id"])
    return keep


def choose(logp, y, seed=SEED, folds=FOLDS, repeats=REPEATS):
    """Both families' out-of-fold log loss and their fit on all items; the chosen family."""
    k = logp.shape[1]
    rng = np.random.default_rng(seed)
    fold_sets = [folds_for(y, folds, rng) for _ in range(repeats)]
    out = {}
    for family in FAMILIES:
        losses = []
        for fold in fold_sets:
            oof = np.empty_like(logp)
            for f in range(folds):
                test = fold == f
                params = fit(logp[~test], y[~test], family)
                oof[test] = calibrated(logp[test], params, k)
            losses.append(float(np.mean(-np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, None)))))
        full = fit(logp, y, family)
        out[family] = {"out_of_fold_log_loss": {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses))},
                       "params": [float(v) for v in full],
                       "temperature": float(np.clip(np.exp(full[0]), T_MIN, T_MAX)),
                       "intercepts": [0.0] + [float(v) for v in full[1:]] if family == "temperature_intercepts" else [0.0] * k}
    a, b = out["temperature"]["out_of_fold_log_loss"]["mean"], out["temperature_intercepts"]["out_of_fold_log_loss"]["mean"]
    chosen = "temperature_intercepts" if b < a else "temperature"
    return chosen, out


def write_probs(path, ids, labels, probs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for i, y, p in zip(ids, labels, probs):
            w.writerow([i, int(y)] + [f"{x:.9g}" for x in p])


def write_calibrated(pred_dir, name, qid, split, ids, labels, logp, entry):
    k = logp.shape[1]
    chosen = calibrated(logp, entry["families"][entry["chosen"]]["params"], k)
    temp = calibrated(logp, entry["families"]["temperature"]["params"], k)
    write_probs(Path(pred_dir) / f"{name}_{qid}_{split}.csv", ids, labels, chosen)
    write_probs(Path(pred_dir) / f"{name}_temperature_{qid}_{split}.csv", ids, labels, temp)


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fit_system(system, name, qids, logits_dir, raw_dir, out_json, pred_dir, records_dir=RECORDS):
    data = load_json(out_json)
    data[name] = {}
    for qid in qids:
        ids, labels, logp, source = load_logp(system, qid, "validation", logits_dir, raw_dir)
        keep = calibrate_ids(qid, records_dir)
        rows = np.array([i for i, x in enumerate(ids) if x in keep])
        if len(rows) != len(keep):
            raise SystemExit(f"{system} {qid}: {len(keep)} calibrate records, {len(rows)} of them in {source['file']}")
        chosen, fams = choose(logp[rows], labels[rows])
        entry = {"source": source, "n_calibrate": int(len(rows)), "folds": FOLDS, "repeats": REPEATS, "seed": SEED,
                 "families": fams, "chosen": chosen,
                 "rule": "the family with the lower mean out-of-fold log loss; an exact tie goes to temperature",
                 "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
        data[name][qid] = entry
        write_calibrated(pred_dir, name, qid, "validation", ids, labels, logp, entry)
        print(f"{name} {qid}: temperature {fams['temperature']['out_of_fold_log_loss']['mean']:.4f}, "
              f"temperature_intercepts {fams['temperature_intercepts']['out_of_fold_log_loss']['mean']:.4f}: {chosen}", flush=True)
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(out_json).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data[name]


def test_guard(results):
    """Test predictions are read only after the freeze and the pod's test stage (as trees_session7.py's guard)."""
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; test predictions are read only after the freeze and the pod's test stage")


def apply_system(system, name, qids, split, logits_dir, raw_dir, out_json, pred_dir, results=None):
    if split == "test":
        test_guard(results if results is not None else Path(out_json).parent)
    data = load_json(out_json)
    if name not in data:
        raise SystemExit(f"{out_json} has no calibration for {name}")
    for qid in qids:
        ids, labels, logp, _ = load_logp(system, qid, split, logits_dir, raw_dir)
        write_calibrated(pred_dir, name, qid, split, ids, labels, logp, data[name][qid])
        print(f"{name} {qid} {split}: {len(ids)} rows written", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fit", "apply"))
    parser.add_argument("--system", required=True)
    parser.add_argument("--name", default=None, help="the calibrated system's name (default: the system's)")
    parser.add_argument("--qids", nargs="+", default=list(QIDS))
    parser.add_argument("--split", default="validation")
    parser.add_argument("--logits-dir", type=Path, default=ROOT / "results" / "fiorillo_v0_3" / "logits")
    parser.add_argument("--from-raw", type=Path, default=None, help="a folder of raw predictions, for a system with no logits file")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "results" / "fiorillo_v0_3" / "predictions")
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "fiorillo_v0_3",
                        help="the folder that must hold release_candidate.json and testpreds_log.json before a test split is read")
    args = parser.parse_args(argv)
    name = args.name or args.system
    if args.command == "fit":
        fit_system(args.system, name, args.qids, args.logits_dir, args.from_raw, args.out_json, args.predictions_dir, args.records_dir)
    else:
        apply_system(args.system, name, args.qids, args.split, args.logits_dir, args.from_raw, args.out_json, args.predictions_dir,
                     args.results)


if __name__ == "__main__":
    main()

"""s9_long_3's calibration and its test scores (docs/prompt_session10.md, S9_LONG_3; I-102). Descriptive.

It calibrates s9_long_3, the third run of session 9's long recipe, exactly as calibrate_session9.py's fit calibrated
s9_long_1 and s9_long_2, by calling that module's functions and, through them, calibrate_session7.py's, not copies of
them: session 8's rule (calibrate_session7.py's choose(): temperature or temperature_intercepts, the lower mean
out-of-fold log loss over its folds and repeats with its seed; an exact tie goes to temperature), fitted on the 654 clean
validation prompts (data/processed/session9/clean/evidence_effect_validation.jsonl), with the calibrated predictions
written for every validation row. The run's probabilities are the softmax of its logits (recipe_session9.py's
run_probs), and the calibration reads their logarithm floored at 1e-300, as calibrate_session9.py's fit does for every
run.

It writes these files, all new, and changes no file of session 9 (it refuses when one of them exists and was not written
by it):
  results/fiorillo_v0_5/calibration_s9_long_3.json   {"s9_long_3": {"evidence_effect": entry}}, entry as in
      calibration_s9.json, with the logits files' SHA-256
  results/fiorillo_v0_5/predictions/s9_long_3_evidence_effect_validation.csv and
      s9_long_3_temperature_evidence_effect_validation.csv, and the same two on test, named as the other runs' files are
  results/fiorillo_v0_5/s9_long_3_test.json   the test split's log loss, ECE (expected calibration error), Brier score,
      accuracy and macro-F1 of the chosen family's predictions, each with the interval session 9's comparisons use:
      compare_session8.py's cluster_bootstrap over articles (each test prompt's group.pmcid in
      data/processed/session8/evidence_effect_test.jsonl), with compare_session9.py's RESAMPLES and SEED, as
      compare_session9.py's resample calls it for Evidence Inference.

Usage (the linked shell, from the project folder): python3 src/eval/calibrate_s9_long_3_session10.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from eval import calibrate_session9 as C9  # noqa: E402
from eval import metrics as M  # noqa: E402
from eval import recipe_session9 as R  # noqa: E402
import compare_session9 as CMP9  # noqa: E402

CMP8 = CMP9.CMP  # compare_session8, as compare_session9 imports it
PLAN = "docs/prompt_session10.md, S9_LONG_3 (I-102)"
NAME = "s9_long_3"
EI = C9.EI
FLOOR = 1e-300
METRICS = ("log_loss", "ece", "brier", "accuracy", "macro_f1")
LABEL = ("descriptive: s9_long_3 was not calibrated in session 9 (I-102); no frozen system, comparison or file of session "
         "9 depends on it")


def outputs(results):
    pred = Path(results) / "predictions"
    return [Path(results) / "calibration_s9_long_3.json", Path(results) / f"{NAME}_test.json"] + [
        pred / f"{NAME}{fam}_{EI}_{split}.csv" for split in ("validation", "test") for fam in ("", "_temperature")]


def guard(results):
    """Refuse to write over a file this script did not write."""
    cal = Path(results) / "calibration_s9_long_3.json"
    if cal.exists():
        data = json.loads(cal.read_text(encoding="utf-8"))
        if data.get("plan") != PLAN:
            raise SystemExit(f"{C9.rel(cal)} exists and was not written by this script; nothing is overwritten")
        return
    clash = [p for p in outputs(results) if p.exists()]
    if clash:
        raise SystemExit(f"{C9.rel(clash[0])} exists and was not written by this script; nothing is overwritten")


def test_clusters(records8, ids, labels):
    """Each test prompt's article (group.pmcid), in the order of ids, with the labels checked against the records."""
    by_id = {}
    for line in open(Path(records8) / f"{EI}_test.jsonl", encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            by_id[r["id"]] = (r["group"]["pmcid"], int(r["label"]))
    missing = [i for i in ids if i not in by_id]
    if missing or len(by_id) != len(ids):
        raise SystemExit(f"the test records and s9_long_3's test logits hold different prompts ({len(missing)} missing)")
    if [by_id[i][1] for i in ids] != [int(y) for y in labels]:
        raise SystemExit("the test records' labels differ from s9_long_3's test logits")
    return [by_id[i][0] for i in ids]


def run(results=C9.RESULTS, records9=C9.RECORDS9, records8=C9.RECORDS8, resamples=CMP9.RESAMPLES):
    results = Path(results)
    guard(results)
    pred_dir = results / "predictions"
    v_logits, t_logits = R.logits_path(results, NAME, "validation"), R.logits_path(results, NAME, "test")

    # validation: session 8's rule on the 654 clean prompts, the calibrated predictions for every row
    ids, labels, p = R.run_probs(results, NAME, "validation")
    logp = C9.logp_of(p, FLOOR)
    keep = R.clean_ids(records9)
    rows = C9.rows_for(ids, keep, NAME)
    source = {"system": NAME, "kind": "the softmax of the run's logits (recipe_session9.py's run_probs)",
              "logits": C9.rel(v_logits), "logits_sha256": C9.sha256_file(v_logits),
              "calibrate_items": "the 654 clean validation prompts (data/processed/session9/clean/)"}
    entry = C9.session8_rule_entry(logp[rows], labels[rows], source)
    probs_v = C9.write_entry_predictions(pred_dir, NAME, EI, "validation", ids, labels, logp, entry)
    entry["validation_calibrate_items"] = {"raw_log_loss": M.log_loss(labels[rows], np.exp(logp[rows])),
                                           "log_loss": M.log_loss(labels[rows], probs_v[rows]),
                                           "ece": M.ece(labels[rows], probs_v[rows])}
    entry["validation_rows_written"] = int(len(ids))

    # test: the same fit applied
    ids_t, labels_t, p_t = R.run_probs(results, NAME, "test")
    logp_t = C9.logp_of(p_t, FLOOR)
    probs_t = C9.write_entry_predictions(pred_dir, NAME, EI, "test", ids_t, labels_t, logp_t, entry)
    data = {"plan": PLAN, "label": LABEL, NAME: {EI: entry},
            "test": {"logits": C9.rel(t_logits), "logits_sha256": C9.sha256_file(t_logits), "rows_written": int(len(ids_t))},
            "time_utc": C9.now()}
    C9.save_json(results / "calibration_s9_long_3.json", data)

    # the test scores with session 9's intervals (articles resampled)
    clusters = test_clusters(records8, ids_t, labels_t)
    rows_of = CMP8.article_rows(clusters)
    est, low, high = CMP8.cluster_bootstrap(lambda idx: CMP8.scores(METRICS, labels_t[idx], probs_t[idx]), rows_of,
                                            len(labels_t), resamples, CMP9.SEED)
    test = {"plan": PLAN, "label": LABEL, "system": NAME, "family": entry["chosen"],
            "predictions": C9.rel(pred_dir / f"{NAME}_{EI}_test.csv"), "n": int(len(labels_t)), "articles": len(rows_of),
            "interval": ("95 percent percentile interval over article resamples: compare_session8.py's cluster_bootstrap "
                         "with compare_session9.py's RESAMPLES and SEED"),
            "resamples": int(resamples), "seed": int(CMP9.SEED),
            "metrics": {m: {"estimate": CMP8.floats(est[j])[0], "low": CMP8.floats(low[j])[0], "high": CMP8.floats(high[j])[0]}
                        for j, m in enumerate(METRICS)},
            "time_utc": C9.now()}
    C9.save_json(results / f"{NAME}_test.json", test)
    print(f"{NAME}: chosen {entry['chosen']}; validation (clean) log loss {entry['validation_calibrate_items']['log_loss']:.4f}; "
          + "; ".join(f"test {m} {test['metrics'][m]['estimate']:.4f} ({test['metrics'][m]['low']:.4f} to {test['metrics'][m]['high']:.4f})"
                      for m in METRICS), flush=True)
    return data, test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=C9.RESULTS)
    parser.add_argument("--records9", type=Path, default=C9.RECORDS9)
    parser.add_argument("--records8", type=Path, default=C9.RECORDS8)
    parser.add_argument("--resamples", type=int, default=CMP9.RESAMPLES)
    args = parser.parse_args(argv)
    run(args.results, args.records9, args.records8, args.resamples)
    return 0


if __name__ == "__main__":
    sys.exit(main())

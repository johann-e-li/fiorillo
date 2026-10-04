"""Comparison of every public v0 system (docs/prompt_session4.md, step 9, the comparisons fixed before any run, and
correction 3).

Reads the predictions in results/public_v0/predictions/ (step 5 format) and the records in data/processed/public/
(row order, labels and the UCI subgroup fields), and writes results/public_v0/comparison.csv (one row per number)
and results/public_v0/comparison.json (the same numbers, nested, with the release bar and the contamination note).

Evaluations, all on the test split:
- pubmedqa_answer: the 500 official PubMedQA test items, three options (yes, no, maybe).
- pubmedqa_yes_no: the official test items labeled yes or no. Three-option probabilities are renormalized over yes
  and no. Duan's own noul question (options false, true) is read with true as yes and false as no.
- readmit_window: the UCI test split, three levels. Duan's own score question has the same three levels.
Metrics: log loss (primary), Brier score, ECE, accuracy and macro-F1 of the top answer; for readmit_window also
macro_f1_tuned (macro-F1 after decision weights fitted on the validation split, correction 3), quadratic weighted
kappa and, for readmission within 30 days (the probability of level 2), AUROC, calibration intercept and slope.
Every number has a 95 percent interval from 1,000 bootstrap resamples of test rows (seed 20260927, percentile
method); a difference between two systems uses the same resamples for both (paired). The decision weights are held
fixed in the resamples, so the intervals of macro_f1_tuned leave out the uncertainty of choosing the weights.
Resampling inflates ECE, so an ECE interval can lie above its estimate. A number that cannot be computed (a
calibration slope for a constant prediction, for example) is empty in the CSV and null in the JSON.

Comparisons, each a difference "system minus reference":
- C1: arm_laya minus Duan on pubmedqa_yes_no: macro-F1, log loss, ECE.
- C2: arm_laya minus Duan on readmit_window: macro-F1 (top answer and tuned), quadratic weighted kappa, log loss,
  ECE, AUROC within 30 days.
  Duan is run with its own wording and state format ("duan", "duan-raw"). Its result is the better of raw and
  recalibrated (a temperature fitted on our validation split), chosen for each metric by the point estimate on the
  compared rows; the variant used is recorded. The same comparisons against Duan run on our wording
  ("duan_ourwording", "duan_ourwording-raw") are the sensitivity analysis and are marked so.
- C3 (as changed by correction 5a): arm_bioclinical minus arm_laya_fresh on pubmedqa_answer and readmit_window, every
  metric: the two readers, each under a freshly initialized decision head trained the same way.
- C3b (correction 5a): arm_laya minus arm_laya_fresh, every metric: what Laya's own decision training adds. Correction 7:
  when run 2's arm L is present as arm_laya_run2, C3b uses it instead of arm_laya, because it was trained on the same
  records as arm_laya_fresh (run 3's arm_laya had 150 PubMedQA validation items instead of 50).
- C4: arm_laya and arm_bioclinical, each minus lightgbm, on readmit_window: AUROC within 30 days, log loss, ECE.
- C5: every system minus the prior (training-split label frequencies) on log loss, on every evaluation; on
  pubmedqa_yes_no the prior is renormalized over yes and no.
Subgroups: on readmit_window, for every system, calibration for readmission within 30 days (observed and expected
rates, observed/expected, calibration intercept and slope) and ECE, by sex, age band and race, for every group with
at least 100 test rows.
Release bar for arm_laya: (1) ECE at most 0.05 on pubmedqa_answer and on readmit_window; (2) log loss below the
prior's on both, with the 95 percent interval of the difference excluding zero (from C5); (3) macro-F1 not below
Duan's point estimate in C1 (top answer) and in C2 (tuned, correction 3), with Duan on its own wording. Each
criterion is PASS or FAIL; a criterion whose predictions are missing is FAIL, with the reason.

Session 5 (--session5; docs/prompt_session5.md, "COMPARISONS AND RELEASE BAR", step 14): the v0.2 candidate named in
results/fiorillo_v0_2/release_candidate.json, every session 5 decision-layer run (results/fiorillo_v0_2/predictions/) and
session 4's v0.1 (arm_laya), Duan, the prior and LightGBM, with the same metrics, resamples and seed. D1: the candidate
minus v0.1, every metric on every evaluation. D2: session 4's C1 and C2 with the candidate in place of arm L; criterion 3
needs both of Duan's variants. D3 (paper only): s5_mix_a, s5_mix_b, s5_mix_c and s5_bioclinical, each minus the
Laya-reader baseline (the run the maybe rule kept, from release_candidate.json), and s5_final_fresh_head minus
s5_bioclinical. D4: the candidate minus LightGBM (AUROC within 30 days, log loss, ECE). D5: every system minus the prior
on log loss. Subgroups as above, for the candidate and v0.1. The four release criteria, criterion 1 with the limits
fixed in release_candidate.json. Writes results/fiorillo_v0_2/comparison.csv and comparison.json.

Usage: python src/eval/compare.py [--predictions-dir DIR] [--records-dir DIR] [--out-dir DIR]
       python src/eval/compare.py --session5 [--s5-predictions-dir DIR] [--release-candidate FILE] [--out-dir DIR]
"""
import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SEED, RESAMPLES = 20260927, 1000
MIN_GROUP_ROWS = 100
ECE_LIMIT = 0.05
SYSTEMS = ("prior", "lightgbm", "laya_zeroshot", "laya_zeroshot-raw", "arm_laya", "arm_laya-raw",
           "arm_bioclinical", "arm_bioclinical-raw", "arm_laya_fresh", "arm_laya_fresh-raw",
           "arm_laya_run2", "arm_laya_run2-raw", "duan", "duan-raw", "duan_ourwording", "duan_ourwording-raw")
DUAN_OWN = ("duan", "duan-raw")
DUAN_OURS = ("duan_ourwording", "duan_ourwording-raw")
EVALUATIONS = ("pubmedqa_answer", "pubmedqa_yes_no", "readmit_window")
RECORDS_QID = {"pubmedqa_answer": "pubmedqa_answer", "pubmedqa_yes_no": "pubmedqa_answer", "readmit_window": "readmit_window"}
DUAN_QID = {"pubmedqa_yes_no": "duan_conclusion_supported", "readmit_window": "duan_readmit_window"}
BASE_METRICS = ("log_loss", "brier", "ece", "accuracy", "macro_f1")
READMIT_METRICS = BASE_METRICS + ("macro_f1_tuned", "quadratic_weighted_kappa", "auroc_30d", "calibration_intercept_30d",
                                  "calibration_slope_30d")
SUBGROUP_METRICS = ("observed_30d", "expected_30d", "observed_expected_30d", "calibration_intercept_30d",
                    "calibration_slope_30d", "ece")
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}
SUBGROUP_FIELDS = ("sex", "age_band", "race")
C1_METRICS = ("macro_f1", "log_loss", "ece")
C2_METRICS = ("macro_f1", "macro_f1_tuned", "quadratic_weighted_kappa", "log_loss", "ece", "auroc_30d")
C4_METRICS = ("auroc_30d", "log_loss", "ece")
CRITERION_3 = (("C1", "pubmedqa_yes_no", "macro_f1"), ("C2", "readmit_window", "macro_f1_tuned"))
CONTAMINATION = ("Known contamination: PubMedQA contexts are PubMed abstracts that PubMed-trained readers may have seen "
                 "with their conclusions; Duan trained on 445 PQA-L items that may overlap the official test list, and on "
                 "a random 20,000-encounter UCI sample that may overlap our test split. Both favor Duan.")
METRICS = {
    "log_loss": lambda y, p, w: M.log_loss(y, p),
    "brier": lambda y, p, w: M.brier(y, p),
    "ece": lambda y, p, w: M.ece(y, p),
    "accuracy": lambda y, p, w: M.accuracy(y, p),
    "macro_f1": lambda y, p, w: M.macro_f1(y, p),
    "macro_f1_tuned": lambda y, p, w: M.macro_f1(y, p * w) if w is not None else M.NAN,
    "quadratic_weighted_kappa": lambda y, p, w: M.quadratic_weighted_kappa(y, p),
    "auroc_30d": lambda y, p, w: M.auroc(y == 2, p[:, 2]),
    "calibration_intercept_30d": lambda y, p, w: M.calibration_intercept(y == 2, p[:, 2]),
    "calibration_slope_30d": lambda y, p, w: M.calibration_slope(y == 2, p[:, 2]),
    "observed_30d": lambda y, p, w: float(np.mean(y == 2)),
    "expected_30d": lambda y, p, w: float(np.mean(p[:, 2])),
    "observed_expected_30d": lambda y, p, w: M.observed_expected(y == 2, p[:, 2]),
}
CSV_COLUMNS = ("section", "evaluation", "system", "reference", "group", "metric", "estimate", "low", "high", "n",
               "system_estimate", "reference_estimate", "note")


def metric_names(evaluation):
    return READMIT_METRICS if evaluation == "readmit_window" else BASE_METRICS


def scores(names, labels, probs, weights=None):
    return np.array([METRICS[m](labels, probs, weights) for m in names])


def applies(system, evaluation):
    if system == "lightgbm":
        return evaluation == "readmit_window"
    if system in DUAN_OWN:
        return evaluation in DUAN_QID
    return True


def prediction_path(directory, system, evaluation, split="test"):
    qid = DUAN_QID[evaluation] if system in DUAN_OWN else RECORDS_QID[evaluation]
    return Path(directory) / f"{system}_{qid}_{split}.csv"


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reference_rows(records_dir, evaluation, split="test"):
    rows = read_jsonl(Path(records_dir) / f"{RECORDS_QID[evaluation]}_{split}.jsonl")
    return [r for r in rows if r["label"] in (0, 1)] if evaluation == "pubmedqa_yes_no" else rows


def load(path, system, evaluation, ref_ids, ref_labels):
    """Probabilities in the reference row order, in our option order; stops if IDs or labels disagree."""
    ids, labels, probs = M.read_predictions(path)
    if system in DUAN_OWN and evaluation == "pubmedqa_yes_no":
        probs, labels = probs[:, [1, 0]], 1 - labels  # Duan's [false, true] read as our [yes, no]
    elif evaluation == "pubmedqa_yes_no":
        keep = labels < 2
        ids = [i for i, k in zip(ids, keep) if k]
        labels, probs = labels[keep], probs[keep][:, :2]
        probs = probs / np.clip(probs.sum(1, keepdims=True), M.EPS, None)
    position = {i: n for n, i in enumerate(ids)}
    missing = [i for i in ref_ids if i not in position]
    if missing or len(position) != len(ref_ids):
        raise SystemExit(f"{path.name}: {len(position)} rows for {evaluation}, {len(ref_ids)} records, {len(missing)} IDs missing")
    order = np.array([position[i] for i in ref_ids])
    if not np.array_equal(labels[order], ref_labels):
        raise SystemExit(f"{path.name}: labels differ from the records")
    return probs[order]


def num(x):
    return round(float(x), 6) if x is not None and np.isfinite(x) else None


def interval(estimate, low, high):
    return {"estimate": num(estimate), "low": num(low), "high": num(high)}


def fmt(x):
    if isinstance(x, (float, np.floating)):
        return f"{x:.6f}" if np.isfinite(x) else ""
    return "" if x is None else str(x)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "results" / "public_v0" / "predictions",
                        help="session 4's predictions (with --session5: v0.1, Duan, the prior and LightGBM)")
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "processed" / "public")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default results/public_v0, or results/fiorillo_v0_2 with --session5")
    parser.add_argument("--session5", action="store_true", help="session 5: D1 to D5 and the v0.2 release bar")
    parser.add_argument("--s5-predictions-dir", type=Path, default=ROOT / "results" / "fiorillo_v0_2" / "predictions")
    parser.add_argument("--release-candidate", type=Path, default=ROOT / "results" / "fiorillo_v0_2" / "release_candidate.json")
    args = parser.parse_args(argv)
    if args.session5:
        return main_session5(args)
    if args.out_dir is None:
        args.out_dir = ROOT / "results" / "public_v0"

    data, missing = {}, []
    for ev in EVALUATIONS:
        rows = reference_rows(args.records_dir, ev)
        entry = {"ids": [r["id"] for r in rows], "labels": np.array([r["label"] for r in rows]),
                 "groups": [r.get("group", {}) for r in rows], "probs": {}, "weights": {}}
        for system in SYSTEMS:
            if not applies(system, ev):
                continue
            path = prediction_path(args.predictions_dir, system, ev)
            if path.exists():
                entry["probs"][system] = load(path, system, ev, entry["ids"], entry["labels"])
            else:
                missing.append(f"{system} on {ev} ({path.name})")
        data[ev] = entry

    # Correction 3: decision weights for the readmission window, fitted on the validation split.
    ev = "readmit_window"
    val_rows = reference_rows(args.records_dir, ev, "validation")
    val_ids, val_labels = [r["id"] for r in val_rows], np.array([r["label"] for r in val_rows])
    fitted = {}
    for system in data[ev]["probs"]:
        path = prediction_path(args.predictions_dir, system, ev, "validation")
        if not path.exists():
            missing.append(f"{system} validation on {ev} ({path.name}); macro_f1_tuned not computed")
            continue
        weights, val_f1 = M.decision_weights(val_labels, load(path, system, ev, val_ids, val_labels))
        data[ev]["weights"][system] = weights
        fitted[system] = {"weights": [num(w) for w in weights], "validation_macro_f1": num(val_f1), "validation_rows": len(val_ids)}

    out = {"created": date.today().isoformat(), "source": "src/eval/compare.py (docs/prompt_session4.md, step 9 and correction 3)",
           "settings": {"split": "test", "resamples": RESAMPLES, "seed": SEED, "interval": "95 percent, percentile, paired",
                        "ece_bins": 15, "subgroup_min_rows": MIN_GROUP_ROWS, "ece_limit": ECE_LIMIT,
                        "decision_weight_grid": "levels 1 and 2: 49 values evenly spaced on the log scale from 0.25 to 16; level 0 fixed at 1",
                        "notes": ["resampling inflates ECE, so an ECE interval can lie above its estimate",
                                  "decision weights are held fixed in the resamples"]},
           "contamination": CONTAMINATION,
           "systems": {e: list(data[e]["probs"]) for e in EVALUATIONS}, "missing_predictions": missing,
           "decision_weights_readmit_window": fitted, "metrics": {}, "comparisons": [], "subgroups": [], "release_bar": {}}
    table = []

    # Every system, every metric.
    for ev in EVALUATIONS:
        names, labels = metric_names(ev), data[ev]["labels"]
        out["metrics"][ev] = {}
        for system, probs in data[ev]["probs"].items():
            w = data[ev]["weights"].get(system)
            est, low, high = M.bootstrap(lambda idx: scores(names, labels[idx], probs[idx], w), len(labels), RESAMPLES, SEED)
            out["metrics"][ev][system] = {"n": len(labels), **{m: interval(est[j], low[j], high[j]) for j, m in enumerate(names)}}
            for j, m in enumerate(names):
                table.append({"section": "metrics", "evaluation": ev, "system": system, "metric": m, "estimate": est[j],
                              "low": low[j], "high": high[j], "n": len(labels)})
        print(f"metrics done: {ev} ({len(data[ev]['probs'])} systems)", flush=True)

    def point(ev, system, metric):
        return out["metrics"][ev][system][metric]["estimate"]

    def better(ev, variants, metric):
        found = [s for s in variants if s in data[ev]["probs"] and point(ev, s, metric) is not None]
        if not found:
            return None
        pick = min if metric in LOWER_IS_BETTER else max
        return pick(found, key=lambda s: point(ev, s, metric))

    def compare(section, ev, system, reference, names, note=""):
        entry = data[ev]
        if system not in entry["probs"] or reference not in entry["probs"]:
            return
        labels, a, b = entry["labels"], entry["probs"][system], entry["probs"][reference]
        wa, wb = entry["weights"].get(system), entry["weights"].get(reference)
        est, low, high = M.bootstrap(lambda idx: scores(names, labels[idx], a[idx], wa) - scores(names, labels[idx], b[idx], wb),
                                     len(labels), RESAMPLES, SEED)
        for j, m in enumerate(names):
            record = {"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                      "difference": interval(est[j], low[j], high[j]), "n": len(labels),
                      "system_estimate": point(ev, system, m), "reference_estimate": point(ev, reference, m), "note": note}
            out["comparisons"].append(record)
            table.append({"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                          "estimate": est[j], "low": low[j], "high": high[j], "n": len(labels),
                          "system_estimate": record["system_estimate"], "reference_estimate": record["reference_estimate"], "note": note})

    for section, ev, names in (("C1", "pubmedqa_yes_no", C1_METRICS), ("C2", "readmit_window", C2_METRICS)):
        for variants, label in ((DUAN_OWN, "Duan's own wording and state format (primary)"), (DUAN_OURS, "Duan on our wording (sensitivity)")):
            for m in names:
                chosen = better(ev, variants, m)
                if chosen:
                    compare(section, ev, "arm_laya", chosen, (m,), f"{label}; Duan variant with the better point estimate: {chosen}")
    for ev in ("pubmedqa_answer", "readmit_window"):
        compare("C3", ev, "arm_bioclinical", "arm_laya_fresh", metric_names(ev))
        run2 = "arm_laya_run2" in data[ev]["probs"]
        compare("C3b", ev, "arm_laya_run2" if run2 else "arm_laya", "arm_laya_fresh", metric_names(ev),
                "run 2's arm L (same records as arm L-fresh, correction 7)" if run2 else "")
    for arm in ("arm_laya", "arm_bioclinical"):
        compare("C4", "readmit_window", arm, "lightgbm", C4_METRICS)
    for ev in EVALUATIONS:
        for system in data[ev]["probs"]:
            if system != "prior":
                compare("C5", ev, system, "prior", ("log_loss",),
                        "prior renormalized over yes and no" if ev == "pubmedqa_yes_no" else "")
    print("comparisons done", flush=True)

    # Subgroup calibration on the UCI test split.
    ev = "readmit_window"
    labels, groups = data[ev]["labels"], data[ev]["groups"]
    for field in SUBGROUP_FIELDS:
        values = [g.get(field, "not recorded") for g in groups]
        for value in sorted(set(values)):
            rows = np.array([i for i, v in enumerate(values) if v == value])
            if len(rows) < MIN_GROUP_ROWS:
                continue
            for system, probs in data[ev]["probs"].items():
                y, p = labels[rows], probs[rows]
                est, low, high = M.bootstrap(lambda idx: scores(SUBGROUP_METRICS, y[idx], p[idx]), len(rows), RESAMPLES, SEED)
                out["subgroups"].append({"field": field, "group": value, "system": system, "n": len(rows),
                                         **{m: interval(est[j], low[j], high[j]) for j, m in enumerate(SUBGROUP_METRICS)}})
                for j, m in enumerate(SUBGROUP_METRICS):
                    table.append({"section": "subgroup", "evaluation": ev, "system": system, "group": f"{field}={value}",
                                  "metric": m, "estimate": est[j], "low": low[j], "high": high[j], "n": len(rows)})
    print("subgroups done", flush=True)

    out["release_bar"] = release_bar(out, data)
    for k, c in out["release_bar"]["criteria"].items():
        table.append({"section": "release_bar", "metric": f"criterion {k}", "note": f"{c['result']}: {c['rule']}"
                      + (f" ({c['reason']})" if c["reason"] else "") + f"; values {json.dumps(c['values'])}"})
    table.append({"section": "release_bar", "metric": "overall", "note": out["release_bar"]["overall"]})
    table.append({"section": "note", "metric": "contamination", "note": CONTAMINATION})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with open(args.out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({c: fmt(r.get(c)) for c in CSV_COLUMNS})
    print_summary(out)


def release_bar(out, data):
    def find(section, ev, system, metric, note_start=""):
        for c in out["comparisons"]:
            if (c["section"], c["evaluation"], c["system"], c["metric"]) == (section, ev, system, metric) and c["note"].startswith(note_start):
                return c
        return None

    bar = {}
    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        if "arm_laya" in data[ev]["probs"]:
            values[ev] = out["metrics"][ev]["arm_laya"]["ece"]["estimate"]
        else:
            reasons.append(f"arm_laya predictions missing on {ev}")
    ok = not reasons and all(v is not None and v <= ECE_LIMIT for v in values.values())
    bar["1"] = {"rule": f"ECE at most {ECE_LIMIT} on both questions after calibration", "values": values,
                "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        c = find("C5", ev, "arm_laya", "log_loss")
        if c:
            values[ev] = {"arm_laya": c["system_estimate"], "prior": c["reference_estimate"], "difference": c["difference"]}
        else:
            reasons.append(f"arm_laya or prior predictions missing on {ev}")
    ok = not reasons and all(v["difference"]["estimate"] < 0 and v["difference"]["high"] < 0 for v in values.values())
    bar["2"] = {"rule": "log loss below the prior's on both questions, 95 percent interval of the difference excluding zero",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for section, ev, metric in CRITERION_3:
        c = find(section, ev, "arm_laya", metric, "Duan's own wording")
        if c and c["system_estimate"] is not None and c["reference_estimate"] is not None:
            values[section] = {"metric": metric, "arm_laya": c["system_estimate"], c["reference"]: c["reference_estimate"],
                               "ok": c["system_estimate"] >= c["reference_estimate"]}
        else:
            reasons.append(f"arm_laya or Duan predictions missing for {section} ({metric})")
    ok = not reasons and all(v["ok"] for v in values.values())
    bar["3"] = {"rule": "macro-F1 not below Duan's point estimate in C1 (top answer) and C2 (decision weights fitted on the validation split, correction 3)",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    return {"system": "arm_laya", "criteria": bar, "overall": "PASS" if all(c["result"] == "PASS" for c in bar.values()) else "FAIL"}


def print_summary(out):
    print("\nPublic v0 comparison, test split (estimate [95% interval])")
    for ev, systems in out["metrics"].items():
        names = ("log_loss", "ece", "macro_f1", "accuracy") + (("macro_f1_tuned", "auroc_30d") if ev == "readmit_window" else ())
        print(f"\n{ev}: " + ", ".join(names))
        for system, m in systems.items():
            cells = [f"{m[k]['estimate']:.4f} [{m[k]['low']:.4f}, {m[k]['high']:.4f}]" if m[k]["low"] is not None else "n/a" for k in names]
            print(f"  {system:22s} " + "  ".join(cells))
    if out["missing_predictions"]:
        print("\nMissing predictions (skipped): " + "; ".join(out["missing_predictions"]))
    print("\nRelease bar for arm_laya:")
    for k, c in out["release_bar"]["criteria"].items():
        print(f"  criterion {k}: {c['result']}  {c['rule']}" + (f"  ({c['reason']})" if c["reason"] else ""))
        print(f"    {json.dumps(c['values'])}")
    print(f"  overall: {out['release_bar']['overall']}")
    print("\n" + out["contamination"])


# Session 5 (docs/prompt_session5.md, "COMPARISONS AND RELEASE BAR" and step 14): Fiorillo v0.2 against v0.1 and the rest.
S5_RUNS = ("s5_laya", "s5_laya_w", "s5_mix_a", "s5_mix_b", "s5_mix_c", "s5_bioclinical", "s5_task",
           "s5_final_fresh_head", "s5_final_laya_head")
S5_FROM_SESSION4 = ("prior", "lightgbm", "arm_laya", "arm_laya-raw", "duan", "duan-raw", "duan_ourwording", "duan_ourwording-raw")
V01 = "arm_laya"
D3_AGAINST_BASELINE = ("s5_mix_a", "s5_mix_b", "s5_mix_c", "s5_bioclinical")  # each minus the Laya-reader baseline
ECE_FLOOR = 0.05


def contamination_session5(task_round_kept):
    task = ("except that the kept task round read the contexts of session 4's PubMedQA training items (never a validation or "
            "test item)" if task_round_kept else "and the task round, which would have read the contexts of session 4's PubMedQA "
            "training items, was not kept")
    return ("Known contamination: Laya's and ModernBERT's pretraining text may include PubMed pages; reader v1's text excludes every "
            f"PubMedQA article by PMID and by the 13-word check, {task}; BioClinical's reader was trained on PubMed and PMC, "
            "PubMedQA's articles included. Session 4's notes on Duan stand: " + CONTAMINATION.replace("Known contamination: ", ""))


def main_session5(args):
    """D1 to D5 and the four release criteria for the v0.2 candidate named in release_candidate.json."""
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "results" / "fiorillo_v0_2"
    cand = json.loads(Path(args.release_candidate).read_text(encoding="utf-8"))
    candidate = cand["system"]
    limits = {q: float(cand["ece_limits"][q]["limit"]) for q in ("pubmedqa_answer", "readmit_window")}
    task_kept = bool(cand.get("selection", {}).get("task_rule", {}).get("task_round_kept"))
    baseline = cand.get("selection", {}).get("maybe_rule", {}).get("baseline", "s5_laya")  # the run the maybe rule kept
    where = {s: Path(args.predictions_dir) for s in S5_FROM_SESSION4}
    for run in S5_RUNS:
        where[run] = where[run + "-raw"] = Path(args.s5_predictions_dir)
    if candidate not in S5_RUNS:
        raise SystemExit(f"release_candidate.json names {candidate}, which is not a session 5 decision-layer run")

    data, missing = {}, []
    for ev in EVALUATIONS:
        rows = reference_rows(args.records_dir, ev)
        entry = {"ids": [r["id"] for r in rows], "labels": np.array([r["label"] for r in rows]),
                 "groups": [r.get("group", {}) for r in rows], "probs": {}, "weights": {}}
        for system, directory in where.items():
            if not applies(system, ev):
                continue
            path = prediction_path(directory, system, ev)
            if path.exists():
                entry["probs"][system] = load(path, system, ev, entry["ids"], entry["labels"])
            else:
                missing.append(f"{system} on {ev} ({path})")
        data[ev] = entry

    ev = "readmit_window"
    val_rows = reference_rows(args.records_dir, ev, "validation")
    val_ids, val_labels = [r["id"] for r in val_rows], np.array([r["label"] for r in val_rows])
    fitted = {}
    for system in data[ev]["probs"]:
        path = prediction_path(where[system], system, ev, "validation")
        if not path.exists():
            missing.append(f"{system} validation on {ev} ({path}); macro_f1_tuned not computed")
            continue
        weights, val_f1 = M.decision_weights(val_labels, load(path, system, ev, val_ids, val_labels))
        data[ev]["weights"][system] = weights
        fitted[system] = {"weights": [num(w) for w in weights], "validation_macro_f1": num(val_f1), "validation_rows": len(val_ids)}

    note = contamination_session5(task_kept)
    out = {"created": date.today().isoformat(),
           "source": "src/eval/compare.py --session5 (docs/prompt_session5.md, comparisons D1 to D5 and the release bar)",
           "candidate": {"system": candidate, "reader": cand.get("reader"), "recipe": cand.get("recipe"), "head": cand.get("head"),
                         "release_candidate_time_utc": cand.get("time_utc"), "ece_limits": limits},
           "v0_1": V01, "d3_baseline": baseline,
           "settings": {"split": "test", "resamples": RESAMPLES, "seed": SEED, "interval": "95 percent, percentile, paired",
                        "ece_bins": 15, "subgroup_min_rows": MIN_GROUP_ROWS,
                        "decision_weight_grid": "levels 1 and 2: 49 values evenly spaced on the log scale from 0.25 to 16; level 0 fixed at 1",
                        "notes": ["resampling inflates ECE, so an ECE interval can lie above its estimate",
                                  "decision weights are held fixed in the resamples",
                                  "D3 is for the paper only and has no bearing on the release"]},
           "contamination": note,
           "systems": {e: list(data[e]["probs"]) for e in EVALUATIONS}, "missing_predictions": missing,
           "decision_weights_readmit_window": fitted, "metrics": {}, "comparisons": [], "subgroups": [], "release_bar": {}}
    table = []

    for ev in EVALUATIONS:
        names, labels = metric_names(ev), data[ev]["labels"]
        out["metrics"][ev] = {}
        for system, probs in data[ev]["probs"].items():
            w = data[ev]["weights"].get(system)
            est, low, high = M.bootstrap(lambda idx: scores(names, labels[idx], probs[idx], w), len(labels), RESAMPLES, SEED)
            out["metrics"][ev][system] = {"n": len(labels), **{m: interval(est[j], low[j], high[j]) for j, m in enumerate(names)}}
            for j, m in enumerate(names):
                table.append({"section": "metrics", "evaluation": ev, "system": system, "metric": m, "estimate": est[j],
                              "low": low[j], "high": high[j], "n": len(labels)})
        print(f"metrics done: {ev} ({len(data[ev]['probs'])} systems)", flush=True)

    def point(ev, system, metric):
        return out["metrics"][ev][system][metric]["estimate"]

    def better(ev, variants, metric):
        found = [s for s in variants if s in data[ev]["probs"] and point(ev, s, metric) is not None]
        if not found:
            return None
        pick = min if metric in LOWER_IS_BETTER else max
        return pick(found, key=lambda s: point(ev, s, metric))

    def compare(section, ev, system, reference, names, note_text=""):
        entry = data[ev]
        if system not in entry["probs"] or reference not in entry["probs"]:
            missing.append(f"{section}: {system} or {reference} missing on {ev}")
            return
        labels, a, b = entry["labels"], entry["probs"][system], entry["probs"][reference]
        wa, wb = entry["weights"].get(system), entry["weights"].get(reference)
        est, low, high = M.bootstrap(lambda idx: scores(names, labels[idx], a[idx], wa) - scores(names, labels[idx], b[idx], wb),
                                     len(labels), RESAMPLES, SEED)
        for j, m in enumerate(names):
            record = {"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                      "difference": interval(est[j], low[j], high[j]), "n": len(labels),
                      "system_estimate": point(ev, system, m), "reference_estimate": point(ev, reference, m), "note": note_text}
            out["comparisons"].append(record)
            table.append({"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                          "estimate": est[j], "low": low[j], "high": high[j], "n": len(labels),
                          "system_estimate": record["system_estimate"], "reference_estimate": record["reference_estimate"],
                          "note": note_text})

    for ev in EVALUATIONS:
        compare("D1", ev, candidate, V01, metric_names(ev), "v0.2 candidate minus v0.1")
    for ev, names in (("pubmedqa_yes_no", C1_METRICS), ("readmit_window", C2_METRICS)):
        for variants, label in ((DUAN_OWN, "Duan's own wording and state format (primary)"), (DUAN_OURS, "Duan on our wording (sensitivity)")):
            for m in names:
                chosen = better(ev, variants, m)
                if chosen:
                    compare("D2", ev, candidate, chosen, (m,), f"{label}; Duan variant with the better point estimate: {chosen}")
    d3_pairs = [(s, baseline) for s in D3_AGAINST_BASELINE] + [("s5_final_fresh_head", "s5_bioclinical")]
    for ev in ("pubmedqa_answer", "readmit_window"):
        for system, reference in d3_pairs:
            if system in data[ev]["probs"] and reference in data[ev]["probs"]:
                compare("D3", ev, system, reference, metric_names(ev), "paper only")
    compare("D4", "readmit_window", candidate, "lightgbm", C4_METRICS)
    for ev in EVALUATIONS:
        for system in data[ev]["probs"]:
            if system != "prior":
                compare("D5", ev, system, "prior", ("log_loss",),
                        "prior renormalized over yes and no" if ev == "pubmedqa_yes_no" else "")
    print("comparisons done", flush=True)

    ev = "readmit_window"
    labels, groups = data[ev]["labels"], data[ev]["groups"]
    for field in SUBGROUP_FIELDS:
        values = [g.get(field, "not recorded") for g in groups]
        for value in sorted(set(values)):
            rows = np.array([i for i, v in enumerate(values) if v == value])
            if len(rows) < MIN_GROUP_ROWS:
                continue
            for system in (candidate, V01):
                if system not in data[ev]["probs"]:
                    continue
                y, p = labels[rows], data[ev]["probs"][system][rows]
                est, low, high = M.bootstrap(lambda idx: scores(SUBGROUP_METRICS, y[idx], p[idx]), len(rows), RESAMPLES, SEED)
                out["subgroups"].append({"field": field, "group": value, "system": system, "n": len(rows),
                                         **{m: interval(est[j], low[j], high[j]) for j, m in enumerate(SUBGROUP_METRICS)}})
                for j, m in enumerate(SUBGROUP_METRICS):
                    table.append({"section": "subgroup", "evaluation": ev, "system": system, "group": f"{field}={value}",
                                  "metric": m, "estimate": est[j], "low": low[j], "high": high[j], "n": len(rows)})
    print("subgroups done", flush=True)

    out["release_bar"] = release_bar_session5(out, data, candidate, limits)
    for k, c in out["release_bar"]["criteria"].items():
        table.append({"section": "release_bar", "metric": f"criterion {k}", "note": f"{c['result']}: {c['rule']}"
                      + (f" ({c['reason']})" if c["reason"] else "") + f"; values {json.dumps(c['values'])}"})
    table.append({"section": "release_bar", "metric": "overall", "note": out["release_bar"]["overall"]})
    table.append({"section": "note", "metric": "contamination", "note": note})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with open(out_dir / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({c: fmt(r.get(c)) for c in CSV_COLUMNS})
    print_summary_session5(out)
    return out


def release_bar_session5(out, data, candidate, limits):
    def find(section, ev, metric, note_start=""):
        for c in out["comparisons"]:
            if (c["section"], c["evaluation"], c["system"], c["metric"]) == (section, ev, candidate, metric) and c["note"].startswith(note_start):
                return c
        return None

    bar = {}
    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        if candidate in data[ev]["probs"]:
            values[ev] = {"ece": out["metrics"][ev][candidate]["ece"]["estimate"], "limit": limits[ev]}
        else:
            reasons.append(f"{candidate} predictions missing on {ev}")
    ok = not reasons and all(v["ece"] is not None and v["ece"] <= v["limit"] for v in values.values())
    bar["1"] = {"rule": f"ECE at most the limit on each question (the larger of {ECE_FLOOR} and the 95th percentile of a calibrated "
                        "model's ECE, fixed in release_candidate.json before any test prediction)",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        c = find("D5", ev, "log_loss")
        if c:
            values[ev] = {candidate: c["system_estimate"], "prior": c["reference_estimate"], "difference": c["difference"]}
        else:
            reasons.append(f"{candidate} or prior predictions missing on {ev}")
    ok = not reasons and all(v["difference"]["estimate"] < 0 and v["difference"]["high"] < 0 for v in values.values())
    bar["2"] = {"rule": "log loss below the prior's on both questions, 95 percent interval of the difference excluding zero",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for ev, metric in (("pubmedqa_yes_no", "macro_f1"), ("readmit_window", "macro_f1_tuned")):
        c = find("D2", ev, metric, "Duan's own wording")
        # the better of raw and recalibrated needs both of Duan's variants, or the criterion could pass on a partial comparison
        both = all(s in data[ev]["probs"] and out["metrics"][ev][s][metric]["estimate"] is not None for s in DUAN_OWN)
        if c and both and c["system_estimate"] is not None and c["reference_estimate"] is not None:
            values[ev] = {"metric": metric, candidate: c["system_estimate"], c["reference"]: c["reference_estimate"],
                          "ok": c["system_estimate"] >= c["reference_estimate"]}
        else:
            reasons.append(f"{candidate}'s predictions, or one of Duan's two variants ({', '.join(DUAN_OWN)}), missing for D2 on {ev} ({metric})")
    ok = not reasons and all(v["ok"] for v in values.values())
    bar["3"] = {"rule": "in D2, macro-F1 not below Duan's point estimate (top answer on the PubMedQA yes or no items; for "
                        "readmission, decision weights fitted on the validation split as in session 4's correction 3), Duan on its own wording",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    c_p, c_r = find("D1", "pubmedqa_answer", "log_loss"), find("D1", "readmit_window", "log_loss")
    if c_p:
        d = c_p["difference"]
        values["pubmedqa_answer"] = {candidate: c_p["system_estimate"], V01: c_p["reference_estimate"], "difference": d,
                                     "ok": d["estimate"] is not None and d["high"] is not None and d["estimate"] < 0 and d["high"] < 0}
    else:
        reasons.append(f"{candidate} or {V01} predictions missing on pubmedqa_answer")
    if c_r:
        d = c_r["difference"]
        values["readmit_window"] = {candidate: c_r["system_estimate"], V01: c_r["reference_estimate"], "difference": d,
                                    "ok": d["low"] is not None and d["low"] <= 0}
    else:
        reasons.append(f"{candidate} or {V01} predictions missing on readmit_window")
    ok = not reasons and all(v["ok"] for v in values.values())
    bar["4"] = {"rule": "PubMedQA log loss lower than v0.1's, the 95 percent interval of the difference (v0.2 minus v0.1) wholly "
                        "below zero, and readmission log loss not significantly worse than v0.1's (the interval's lower end at or below zero)",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    return {"system": candidate, "criteria": bar, "overall": "PASS" if all(c["result"] == "PASS" for c in bar.values()) else "FAIL"}


def print_summary_session5(out):
    cand = out["candidate"]["system"]
    print(f"\nFiorillo v0.2 comparison, test split (estimate [95% interval]); candidate {cand}, v0.1 {out['v0_1']}")
    shown = (cand, cand + "-raw", out["v0_1"], "prior", "lightgbm", "duan", "duan-raw")
    for ev, systems in out["metrics"].items():
        names = ("log_loss", "ece", "macro_f1", "accuracy") + (("macro_f1_tuned", "auroc_30d") if ev == "readmit_window" else ())
        print(f"\n{ev}: " + ", ".join(names))
        for system in shown:
            if system not in systems:
                continue
            m = systems[system]
            cells = [f"{m[k]['estimate']:.4f} [{m[k]['low']:.4f}, {m[k]['high']:.4f}]" if m[k]["low"] is not None else "n/a" for k in names]
            print(f"  {system:22s} " + "  ".join(cells))
    if out["missing_predictions"]:
        print("\nMissing predictions (skipped): " + "; ".join(out["missing_predictions"]))
    print(f"\nRelease bar for {cand}:")
    for k, c in out["release_bar"]["criteria"].items():
        print(f"  criterion {k}: {c['result']}  {c['rule']}" + (f"  ({c['reason']})" if c["reason"] else ""))
        print(f"    {json.dumps(c['values'])}")
    print(f"  overall: {out['release_bar']['overall']}")
    print("\n" + out["contamination"])


if __name__ == "__main__":
    main()

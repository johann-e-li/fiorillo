"""Session 7's comparisons and the v0.3 release bar (docs/prompt_session7.md, COMPARISONS and POWER AND THE BAR).

Reads the test predictions (session 4's format) in results/fiorillo_v0_3/predictions/ (session 7's systems),
results/public_v0/predictions/ (v0.1 as released, arm_laya; Duan; the prior; LightGBM) and
results/fiorillo_v0_2/predictions/ (the v0.2 candidate, s5_final_fresh_head), with the records in data/processed/public/
(row order and labels), and results/fiorillo_v0_3/release_candidate.json (the candidate, or none, and the ECE limits).
Metrics and reading functions are session 4's (src/eval/metrics.py; src/eval/compare.py, not changed), on the
evaluations pubmedqa_answer, pubmedqa_yes_no and readmit_window. Every number has a 95 percent interval from 1,000
bootstrap resamples of test rows (metrics.bootstrap: numpy's default_rng with the seed 20260927, integers(0, n, n) per
resample, percentile method); a difference "system minus reference" uses the same resamples for both (paired).
Criterion 4's two log-loss differences use 10,000 resamples of the same generator, for both their 95 and their 97.5
percent intervals. The decision weights of macro_f1_tuned (readmission) are fitted on each system's validation
predictions (session 4's correction 3) and held fixed in the resamples.

Systems: the candidate as named in release_candidate.json (its chosen calibration), <arm>_temperature and <arm>-raw for
s7_a3 and s7_b3; arm_laya and arm_laya-raw; arm_laya_s7cal; s5_final_fresh_head and its -raw; Duan's four files; the
prior; LightGBM and lightgbm_tuned; blend_v0_1 and blend_v0_3 (readmission only). A missing file is recorded and
skipped.

Comparisons: E1, the candidate minus arm_laya, every metric on every evaluation; E2, s7_a3_temperature minus
s5_final_fresh_head; E3, s7_b3_temperature minus s7_a3_temperature (E2 and E3 every metric on every evaluation);
E4, the candidate minus Duan as session 4's C1 and C2 (the Duan variant with the better point estimate, its own wording
primary and ours as sensitivity); E5, every system minus the prior on log loss; E6 (readmission: log loss, AUROC within
30 days, ECE), lightgbm_tuned minus lightgbm, blend_v0_3 minus lightgbm_tuned, blend_v0_3 minus the candidate,
blend_v0_1 minus arm_laya; E7, arm_laya_s7cal minus arm_laya, every metric on every evaluation. Sensitivity: E1 with the
candidate's -raw and _temperature predictions; E2 and E3 with each arm's chosen calibration and with -raw predictions.
Maybe (descriptive): for each system on pubmedqa_answer, the F1 of maybe and the number of items whose top answer is
maybe. Release bar (only with a candidate): criteria 1 to 4 as the plan fixes them; criterion 4 passes by route (a)
or route (b). With no candidate, E1, E4, E5's candidate row, E6's blend_v0_3 rows and the release bar are recorded as
NOT BUILT. zeros.json: for every system, split and question read, the rows whose probability of the true answer prints
as 0.

It refuses to run unless release_candidate.json and testpreds_log.json exist (test predictions are made only after
the freeze). Each block of results is saved in models/fiorillo_v0_3/compare_cache/ under a key that includes the SHA-256
of its input files and of the records files, so a run that the linked shell ends after about three minutes resumes;
--max-seconds stops it from starting a new block after that many seconds. Writes results/fiorillo_v0_3/comparison.json,
comparison.csv and zeros.json.

Usage: python3 src/eval/compare_session7.py [--max-seconds 150]
"""
import argparse
import csv
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
import compare as C  # noqa: E402
import metrics as M  # noqa: E402

SEED = 20260927
RESAMPLES = 1000
RESAMPLES_C4 = 10000
RESULTS = ROOT / "results" / "fiorillo_v0_3"
S4 = ROOT / "results" / "public_v0" / "predictions"
S5 = ROOT / "results" / "fiorillo_v0_2" / "predictions"
S7 = RESULTS / "predictions"
CACHE = ROOT / "models" / "fiorillo_v0_3" / "compare_cache"
EVALUATIONS = C.EVALUATIONS
V01 = "arm_laya"
ARMS = ("s7_a3", "s7_b3")
FROM_S4 = ("prior", "lightgbm", "arm_laya", "arm_laya-raw", "duan", "duan-raw", "duan_ourwording", "duan_ourwording-raw")
FROM_S5 = ("s5_final_fresh_head", "s5_final_fresh_head-raw")
FROM_S7 = tuple(f"{a}{s}" for a in ARMS for s in ("", "_temperature", "-raw")) + ("arm_laya_s7cal", "arm_laya_s7cal_temperature",
                                                                                  "lightgbm_tuned", "blend_v0_1", "blend_v0_3")
READMIT_ONLY = ("lightgbm", "lightgbm_tuned", "blend_v0_1", "blend_v0_3")
E6 = (("lightgbm_tuned", "lightgbm"), ("blend_v0_3", "lightgbm_tuned"), ("blend_v0_3", "@candidate"), ("blend_v0_1", V01))
ECE_FLOOR = 0.05
CONTAMINATION = (
    "Known contamination: Laya's and ModernBERT's pretraining text may include PubMed pages; reader v1's own training text "
    "excluded PubMedQA's articles. Qwen3's pretraining text is not published and may include PubMed pages, PubMedQA and the UCI "
    "data, which would favor arm B. Every PQA-A and PQA-U item used in session 7 passed the 13-word check against the 1,000 "
    "PQA-L items, except that phase 3 read session 4's training records unchanged, as v0.1 did. Session 4's notes on Duan stand: "
    + C.CONTAMINATION.replace("Known contamination: ", ""))
ONE_RUN = ("Each arm and each recipe ran once; retraining alone moved PubMedQA log loss by about 0.023 per run in this project "
           "(results/fiorillo_v0_2/retry_check.json), so a difference in E2 or E3, in either direction, can come partly from the run "
           "itself, and a worse result for arm B is not evidence that the base is worse in general.")


def directory_of(system):
    if system in FROM_S4:
        return S4
    if system in FROM_S5:
        return S5
    return S7


def applies(system, ev):
    if system in READMIT_ONLY:
        return ev == "readmit_window"
    if system in C.DUAN_OWN:
        return ev in C.DUAN_QID
    return True


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Cache:
    def __init__(self, folder, deadline):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.stopped = False
        self.computed_now = 0

    def get(self, key_parts, compute):
        key = hashlib.sha256(json.dumps(key_parts, sort_keys=True).encode("utf-8")).hexdigest()[:32]
        path = self.folder / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.computed_now and time.time() > self.deadline:  # every run computes at least one block, so each run makes progress
            self.stopped = True
            return None
        value = compute()
        self.computed_now += 1
        tmp = path.with_name(path.name + ".tmp")  # written whole, then renamed, so a stopped run never leaves half a block
        tmp.write_text(json.dumps(value) + "\n", encoding="utf-8")
        tmp.replace(path)
        return value


def floats(a):
    return [None if (x is None or not np.isfinite(x)) else float(x) for x in np.atleast_1d(a)]


def c4_interval(labels, a, b, resamples=RESAMPLES_C4, seed=SEED):
    """The paired log-loss difference a minus b, with 95 and 97.5 percent percentile intervals from `resamples` draws of
    metrics.bootstrap's generator."""
    la = -np.log(np.clip(a[np.arange(len(labels)), labels], M.EPS, None))
    lb = -np.log(np.clip(b[np.arange(len(labels)), labels], M.EPS, None))
    d = la - lb
    rng = np.random.default_rng(seed)
    n = len(d)
    vals = np.array([d[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return {"estimate": float(d.mean()), "low_95": float(np.percentile(vals, 2.5)), "high_95": float(np.percentile(vals, 97.5)),
            "low_97_5": float(np.percentile(vals, 1.25)), "high_97_5": float(np.percentile(vals, 98.75)), "resamples": resamples, "n": n}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--records-dir", type=Path, default=ROOT / "data" / "processed" / "public")
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--resamples-c4", type=int, default=RESAMPLES_C4)
    args = parser.parse_args(argv)
    t0 = time.time()
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (args.results / name).exists():
            raise SystemExit(f"{name} does not exist in {args.results}; the comparisons read test predictions, made only after the freeze")
    cache = Cache(args.cache, t0 + args.max_seconds)
    rc = json.loads((args.results / "release_candidate.json").read_text(encoding="utf-8"))
    candidate = rc.get("system")
    systems = FROM_S4 + FROM_S5 + FROM_S7

    data, missing, digests, zeros = {}, [], {}, {}
    records_sha = {q: sha256_file(Path(args.records_dir) / f"{q}_{split}.jsonl") for q in ("pubmedqa_answer", "readmit_window")
                   for split in ("test",)}
    records_sha["readmit_window_validation"] = sha256_file(Path(args.records_dir) / "readmit_window_validation.jsonl")
    for ev in EVALUATIONS:
        rows = C.reference_rows(args.records_dir, ev)
        entry = {"ids": [r["id"] for r in rows], "labels": np.array([r["label"] for r in rows]), "probs": {}, "weights": {}}
        for system in systems:
            if not applies(system, ev):
                continue
            path = C.prediction_path(directory_of(system), system, ev)
            if path.exists():
                entry["probs"][system] = C.load(path, system, ev, entry["ids"], entry["labels"])
                digests[(system, ev)] = sha256_file(path)
            else:
                missing.append(f"{system} on {ev} ({path})")
        data[ev] = entry

    for system in systems:
        own = [C.DUAN_QID[ev] for ev in ("pubmedqa_yes_no", "readmit_window")] if system in C.DUAN_OWN else None
        for qid in own or ("pubmedqa_answer", "readmit_window"):
            for split in ("validation", "test"):
                if own is None and not applies(system, qid):
                    continue
                path = directory_of(system) / f"{system}_{qid}_{split}.csv"
                if path.exists():
                    ids, labels, probs = M.read_predictions(path)
                    zeros.setdefault(system, {}).setdefault(split, {})[qid] = {
                        "rows": len(ids), "true_answer_printed_zero": int((probs[np.arange(len(labels)), labels] == 0).sum())}

    ev = "readmit_window"
    val_rows = C.reference_rows(args.records_dir, ev, "validation")
    val_ids, val_labels = [r["id"] for r in val_rows], np.array([r["label"] for r in val_rows])
    fitted = {}
    for system in data[ev]["probs"]:
        path = C.prediction_path(directory_of(system), system, ev, "validation")
        if not path.exists():
            missing.append(f"{system} validation on {ev} ({path}); macro_f1_tuned not computed")
            continue
        weights, val_f1 = M.decision_weights(val_labels, C.load(path, system, ev, val_ids, val_labels))
        data[ev]["weights"][system] = weights
        fitted[system] = {"weights": [C.num(w) for w in weights], "validation_macro_f1": C.num(val_f1)}

    out = {"created": dt.date.today().isoformat(), "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "src/eval/compare_session7.py (docs/prompt_session7.md, COMPARISONS and POWER AND THE BAR)",
           "candidate": candidate, "not_built": [] if candidate else [rc.get("not_built") or "NOT BUILT: v0.3 candidate"],
           "v0_1": V01, "ece_limits": rc.get("ece_limits"),
           "settings": {"split": "test", "resamples": args.resamples, "resamples_criterion_4": args.resamples_c4, "seed": SEED,
                        "interval": "95 percent, percentile, paired", "ece_bins": 15},
           "contamination": CONTAMINATION, "one_run": ONE_RUN,
           "systems": {e: list(data[e]["probs"]) for e in EVALUATIONS}, "missing_predictions": missing,
           "decision_weights_readmit_window": fitted, "metrics": {}, "comparisons": [], "criterion_4_intervals": {},
           "maybe": {}, "release_bar": {}}
    table = []

    for ev in EVALUATIONS:
        names, labels = C.metric_names(ev), data[ev]["labels"]
        out["metrics"][ev] = {}
        for system, probs in data[ev]["probs"].items():
            w = data[ev]["weights"].get(system)
            key = ["metrics", ev, system, digests[(system, ev)], list(names), args.resamples, SEED, floats(w) if w is not None else None,
                   records_sha]
            res = cache.get(key, lambda: [floats(v) for v in M.bootstrap(lambda idx: C.scores(names, labels[idx], probs[idx], w),
                                                                        len(labels), args.resamples, SEED)])
            if res is None:
                continue
            est, low, high = res
            out["metrics"][ev][system] = {"n": len(labels), **{m: {"estimate": est[j], "low": low[j], "high": high[j]} for j, m in enumerate(names)}}
            for j, m in enumerate(names):
                table.append({"section": "metrics", "evaluation": ev, "system": system, "metric": m, "estimate": est[j],
                              "low": low[j], "high": high[j], "n": len(labels)})

    def point(ev, system, metric):
        return out["metrics"].get(ev, {}).get(system, {}).get(metric, {}).get("estimate")

    def compare(section, ev, system, reference, names, note_text=""):
        entry = data[ev]
        if system not in entry["probs"] or reference not in entry["probs"]:
            missing.append(f"{section}: {system} or {reference} missing on {ev}")
            return
        labels, a, b = entry["labels"], entry["probs"][system], entry["probs"][reference]
        wa, wb = entry["weights"].get(system), entry["weights"].get(reference)
        key = ["diff", ev, system, reference, digests[(system, ev)], digests[(reference, ev)], list(names), args.resamples, SEED,
               floats(wa) if wa is not None else None, floats(wb) if wb is not None else None, records_sha]
        res = cache.get(key, lambda: [floats(v) for v in M.bootstrap(
            lambda idx: C.scores(names, labels[idx], a[idx], wa) - C.scores(names, labels[idx], b[idx], wb), len(labels), args.resamples, SEED)])
        if res is None:
            return
        est, low, high = res
        for j, m in enumerate(names):
            record = {"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                      "difference": {"estimate": est[j], "low": low[j], "high": high[j]}, "n": len(labels),
                      "system_estimate": point(ev, system, m), "reference_estimate": point(ev, reference, m), "note": note_text}
            out["comparisons"].append(record)
            table.append({"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                          "estimate": est[j], "low": low[j], "high": high[j], "n": len(labels),
                          "system_estimate": record["system_estimate"], "reference_estimate": record["reference_estimate"], "note": note_text})

    def better(ev, variants, metric):
        found = [s for s in variants if s in data[ev]["probs"] and point(ev, s, metric) is not None]
        if not found:
            return None
        pick = min if metric in C.LOWER_IS_BETTER else max
        return pick(found, key=lambda s: point(ev, s, metric))

    for ev in EVALUATIONS:
        if candidate:
            compare("E1", ev, candidate, V01, C.metric_names(ev), "the v0.3 candidate minus v0.1 as released")
            compare("E1_sensitivity", ev, candidate + "-raw", V01, C.metric_names(ev), "the candidate's raw predictions minus v0.1")
            compare("E1_sensitivity", ev, candidate + "_temperature", V01, C.metric_names(ev), "the candidate with one temperature per question minus v0.1")
        compare("E2", ev, "s7_a3_temperature", "s5_final_fresh_head", C.metric_names(ev), "phased recipe minus one mixed phase, reader v1, one temperature each; " + ONE_RUN)
        compare("E2_sensitivity", ev, "s7_a3", "s5_final_fresh_head", C.metric_names(ev), "s7_a3 with its chosen calibration")
        compare("E2_sensitivity", ev, "s7_a3-raw", "s5_final_fresh_head-raw", C.metric_names(ev), "raw predictions")
        compare("E3", ev, "s7_b3_temperature", "s7_a3_temperature", C.metric_names(ev), "Qwen3-4B-Base minus reader v1, same recipe and calibration; " + ONE_RUN)
        compare("E3_sensitivity", ev, "s7_b3", "s7_a3", C.metric_names(ev), "each arm's chosen calibration")
        compare("E3_sensitivity", ev, "s7_b3-raw", "s7_a3-raw", C.metric_names(ev), "raw predictions")
        compare("E7", ev, "arm_laya_s7cal", V01, C.metric_names(ev), "v0.1 with session 7's calibration rule minus v0.1 as released")
    if candidate:
        for ev, names in (("pubmedqa_yes_no", C.C1_METRICS), ("readmit_window", C.C2_METRICS)):
            for variants, label in ((C.DUAN_OWN, "Duan's own wording and state format (primary)"), (C.DUAN_OURS, "Duan on our wording (sensitivity)")):
                for m in names:
                    chosen = better(ev, variants, m)
                    if chosen:
                        compare("E4", ev, candidate, chosen, (m,), f"{label}; Duan variant with the better point estimate: {chosen}")
    for ev in EVALUATIONS:
        for system in data[ev]["probs"]:
            if system != "prior":
                compare("E5", ev, system, "prior", ("log_loss",), "prior renormalized over yes and no" if ev == "pubmedqa_yes_no" else "")
    for system, reference in E6:
        reference = candidate if reference == "@candidate" else reference
        if reference is None or (system == "blend_v0_3" and not candidate):
            continue
        compare("E6", "readmit_window", system, reference, C.C4_METRICS)

    labels = data["pubmedqa_answer"]["labels"]
    for system, probs in data["pubmedqa_answer"]["probs"].items():
        top = probs.argmax(axis=1)
        tp = int(((top == 2) & (labels == 2)).sum())
        fp, fn = int(((top == 2) & (labels != 2)).sum()), int(((top != 2) & (labels == 2)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        out["maybe"][system] = {"top_answer_maybe": int((top == 2).sum()), "maybe_items": int((labels == 2).sum()), "f1_maybe": f1}

    if candidate:
        for ev in ("pubmedqa_answer", "readmit_window"):
            entry = data[ev]
            if candidate in entry["probs"] and V01 in entry["probs"]:
                key = ["c4", ev, candidate, V01, digests[(candidate, ev)], digests[(V01, ev)], args.resamples_c4, SEED, records_sha]
                res = cache.get(key, lambda: c4_interval(entry["labels"], entry["probs"][candidate], entry["probs"][V01], args.resamples_c4, SEED))
                if res is not None:
                    out["criterion_4_intervals"][ev] = res

    if cache.stopped:
        print(json.dumps({"status": "stopped; run again to resume", "seconds": round(time.time() - t0, 1),
                          "cached_blocks": len(list(Path(args.cache).glob("*.json")))}))
        return None

    reason = out["not_built"][0] if out["not_built"] else "NOT BUILT: v0.3 candidate"
    out["release_bar"] = release_bar(out, data, candidate, rc) if candidate else {
        "overall": "NOT BUILT", "reason": reason, "outcome": f"v0.3 release bar not met: there is no v0.3 candidate ({reason})"}
    if not candidate:
        out["not_built"] += ["NOT BUILT: E1, E4, E5's candidate row, E6's blend_v0_3 rows and the release bar (no v0.3 candidate)"]
    for k, c in out["release_bar"].get("criteria", {}).items():
        table.append({"section": "release_bar", "metric": f"criterion {k}", "note": f"{c['result']}: {c['rule']}"
                      + (f" ({c['reason']})" if c["reason"] else "") + f"; values {json.dumps(c['values'])}"})
    table.append({"section": "release_bar", "metric": "overall", "note": f"{out['release_bar']['overall']}: {out['release_bar']['outcome']}"})
    table.append({"section": "note", "metric": "contamination", "note": CONTAMINATION})

    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / "comparison.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with open(args.results / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=C.CSV_COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({c: C.fmt(r.get(c)) for c in C.CSV_COLUMNS})
    (args.results / "zeros.json").write_text(json.dumps({"plan": "docs/prompt_session7.md, CALIBRATION", "rows_whose_true_answer_prints_as_0": zeros},
                                                        indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"status": "written", "candidate": candidate, "release_bar": out["release_bar"].get("overall"),
                      "missing": len(missing), "seconds": round(time.time() - t0, 1)}, indent=1))
    return out


def criterion4_routes(ci):
    """Criterion 4's two routes from the two log-loss differences (the candidate minus v0.1): route (a), PubMedQA's
    97.5 percent interval wholly below zero and readmission's 95 percent interval's lower end at or below zero; route (b),
    the same with the questions swapped."""
    p, r = ci["pubmedqa_answer"], ci["readmit_window"]
    routes = {
        "a": {"rule": "PubMedQA's 97.5 percent interval wholly below zero, and readmission's 95 percent interval's lower end at or below zero",
              "own": p["high_97_5"] < 0, "other_not_worse": r["low_95"] <= 0},
        "b": {"rule": "readmission's 97.5 percent interval wholly below zero, and PubMedQA's 95 percent interval's lower end at or below zero",
              "own": r["high_97_5"] < 0, "other_not_worse": p["low_95"] <= 0},
    }
    for v in routes.values():
        v["passes"] = bool(v["own"] and v["other_not_worse"])
    return routes


def release_bar(out, data, candidate, rc):
    def find(section, ev, metric, note_start=""):
        for c in out["comparisons"]:
            if (c["section"], c["evaluation"], c["system"], c["metric"]) == (section, ev, candidate, metric) and c["note"].startswith(note_start):
                return c
        return None

    limits = {q: float(v["limit"]) for q, v in (rc.get("ece_limits") or {}).items()}
    bar = {}
    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        m = out["metrics"].get(ev, {}).get(candidate)
        if m and ev in limits:
            values[ev] = {"ece": m["ece"]["estimate"], "limit": limits[ev]}
        else:
            reasons.append(f"{candidate}'s predictions or the ECE limit missing on {ev}")
    ok = not reasons and all(v["ece"] is not None and v["ece"] <= v["limit"] for v in values.values())
    bar["1"] = {"rule": f"ECE at most the limit on each question (the larger of {ECE_FLOOR} and the 95th percentile of a calibrated "
                        "model's ECE, fixed in release_candidate.json before any test prediction)",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for ev in ("pubmedqa_answer", "readmit_window"):
        c = find("E5", ev, "log_loss")
        if c:
            values[ev] = {candidate: c["system_estimate"], "prior": c["reference_estimate"], "difference": c["difference"]}
        else:
            reasons.append(f"{candidate} or prior predictions missing on {ev}")
    ok = not reasons and all(v["difference"]["estimate"] < 0 and v["difference"]["high"] < 0 for v in values.values())
    bar["2"] = {"rule": "log loss below the prior's on both questions, 95 percent interval of the difference excluding zero",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons, values = [], {}
    for ev, metric in (("pubmedqa_yes_no", "macro_f1"), ("readmit_window", "macro_f1_tuned")):
        c = find("E4", ev, metric, "Duan's own wording")
        both = all(s in data[ev]["probs"] and out["metrics"][ev].get(s, {}).get(metric, {}).get("estimate") is not None for s in C.DUAN_OWN)
        if c and both and c["system_estimate"] is not None and c["reference_estimate"] is not None:
            values[ev] = {"metric": metric, candidate: c["system_estimate"], c["reference"]: c["reference_estimate"],
                          "ok": c["system_estimate"] >= c["reference_estimate"]}
        else:
            reasons.append(f"{candidate}'s predictions, or one of Duan's two variants, missing for E4 on {ev} ({metric})")
    ok = not reasons and all(v["ok"] for v in values.values())
    bar["3"] = {"rule": "in E4, macro-F1 not below Duan's point estimate (top answer on the PubMedQA yes or no items; for readmission, "
                        "decision weights fitted on the validation split as in session 4's correction 3), Duan on its own wording",
                "values": values, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}

    reasons = []
    ci = out["criterion_4_intervals"]
    for ev in ("pubmedqa_answer", "readmit_window"):
        if ev not in ci:
            reasons.append(f"{candidate} or {V01} predictions missing on {ev}")
    routes = criterion4_routes(ci) if not reasons else {}
    ok = not reasons and any(v["passes"] for v in routes.values())
    bar["4"] = {"rule": "route (a) or route (b) (docs/prompt_session7.md, POWER AND THE BAR); differences are the candidate minus v0.1 "
                        "in log loss, with 10,000 paired resamples",
                "values": {"intervals": ci, "routes": routes}, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    overall = "PASS" if all(c["result"] == "PASS" for c in bar.values()) else "FAIL"
    return {"system": candidate, "criteria": bar, "overall": overall,
            "outcome": "every criterion passes; releasing v0.3 is Johann's decision after the closeout" if overall == "PASS"
            else "v0.3 release bar not met: " + ", ".join(f"criterion {k}" for k, c in bar.items() if c["result"] != "PASS")}


if __name__ == "__main__":
    main()

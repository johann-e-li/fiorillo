"""Session 7's selection rules and the freeze (docs/prompt_session7.md, SELECTION RULES and steps 11 and 12).

select: reads the metrics JSON of s7_a3 and s7_b3 (each run's selection score at its kept checkpoint:
kept_validation_selection_log_loss_exact, session 4's score before any calibration) and the v0.2 candidate's score
(head_rule.fresh_head_score in results/fiorillo_v0_2/selection.json, 0.7828), applies the candidate rule, and writes
every score and each rule's outcome to results/fiorillo_v0_3/selection.json, with the labeler's gate and subset copied
from labeler.json. Candidate rule: of s7_a3 and s7_b3 the lower score wins, an exact tie going to s7_a3; if the winner's
score is not below the v0.2 candidate's, there is no v0.3 candidate ("NOT BUILT: v0.3 candidate, no arm beat the v0.2
candidate on validation"). An arm whose metrics are missing is recorded and cannot win.

freeze: writes results/fiorillo_v0_3/release_candidate.json before any test prediction exists: the candidate or none,
every selection score, the calibration families (calibration_s7.json), both ECE limits (the larger of 0.05 and the
95th percentile in ece_limit_<qid>.json, from src/eval/ece_noise_floor.py on the candidate's calibrated validation
predictions) and the time in UTC. It refuses to run if any test prediction of s7_a3 or s7_b3 exists under
results/fiorillo_v0_3/predictions/ or the pod's staging folder, or if release_candidate.json already exists.

Usage: python src/eval/select_session7.py select [--results DIR]
       python src/eval/select_session7.py freeze [--results DIR] [--staging DIR]
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "fiorillo_v0_3"
V02_SELECTION = ROOT / "results" / "fiorillo_v0_2" / "selection.json"
ARMS = ("s7_a3", "s7_b3")
ECE_FLOOR = 0.05
ECE_DRAWS = 5000
TEST_ROWS = {"pubmedqa_answer": 500, "readmit_window": 6998}
QIDS = ("pubmedqa_answer", "readmit_window")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def kept_score(metrics):
    if "kept_validation_selection_log_loss_exact" in metrics:
        return float(metrics["kept_validation_selection_log_loss_exact"])
    return float(metrics["best_validation_selection_log_loss_exact"])


def candidate_rule(scores, v02_score):
    """scores: {arm: score or None}. Returns (candidate or None, steps, winner, not_built)."""
    present = {a: s for a, s in scores.items() if s is not None}
    steps = []
    if not present:
        return None, ["no arm has a selection score"], None, "NOT BUILT: v0.3 candidate, neither arm has a selection score"
    a, b = ARMS
    if a in present and b in present:
        winner = b if present[b] < present[a] else a
        steps.append(f"{a} {present[a]:.6f} against {b} {present[b]:.6f}: {winner} (an exact tie goes to {a})")
    else:
        winner = next(iter(present))
        steps.append(f"only {winner} has a selection score ({present[winner]:.6f})")
    if present[winner] < v02_score:
        steps.append(f"{winner} {present[winner]:.6f} is below the v0.2 candidate's {v02_score:.6f}: {winner} is the v0.3 candidate")
        return winner, steps, winner, None
    steps.append(f"{winner} {present[winner]:.6f} is not below the v0.2 candidate's {v02_score:.6f}: no v0.3 candidate")
    return None, steps, winner, "NOT BUILT: v0.3 candidate, no arm beat the v0.2 candidate on validation"


def select(results, v02_selection=V02_SELECTION):
    results = Path(results)
    scores, sources = {}, {}
    for arm in ARMS:
        path = results / "metrics" / f"{arm}_metrics.json"
        if path.exists():
            scores[arm] = kept_score(json.loads(path.read_text(encoding="utf-8")))
            sources[arm] = str(path.relative_to(results.parent.parent)) if results.parent.parent in path.parents else str(path)
        else:
            scores[arm] = None
            sources[arm] = f"missing: {path}"
    v02 = json.loads(Path(v02_selection).read_text(encoding="utf-8"))
    v02_score = float(v02["head_rule"]["fresh_head_score"])
    candidate, steps, winner, not_built = candidate_rule(scores, v02_score)
    labeler_path = results / "labeler.json"
    labeler = json.loads(labeler_path.read_text(encoding="utf-8")) if labeler_path.exists() else None
    out = {
        "plan": "docs/prompt_session7.md, SELECTION RULES",
        "time_utc": now(),
        "labeler": ({"phase2_built": labeler.get("phase2_built"), "not_built": labeler.get("not_built"), "gate": labeler.get("gate"),
                     "subset": labeler.get("subset")} if labeler else "labeler.json missing"),
        "candidate_rule": {"scores": scores, "sources": sources, "v0_2_candidate": {"system": v02["head_rule"].get("candidate"), "score": v02_score,
                                                                                    "source": "results/fiorillo_v0_2/selection.json, head_rule"},
                           "steps": steps, "winner": winner, "candidate": candidate, "not_built": not_built},
    }
    (results / "selection.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["candidate_rule"], indent=1))
    return out


def test_predictions_exist(dirs):
    found = []
    for d in dirs:
        d = Path(d)
        if d.exists():
            for arm in ARMS:
                found.extend(str(p) for p in d.rglob(f"{arm}*_test*.csv"))
    return sorted(found)


def freeze(results, staging=None):
    results = Path(results)
    out_path = results / "release_candidate.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; the freeze is written once")
    found = test_predictions_exist([results / "predictions", results / "logits"] + ([staging] if staging else []))
    if found:
        raise SystemExit(f"test predictions already exist, so the freeze cannot be written: {found[:3]}")
    sel = json.loads((results / "selection.json").read_text(encoding="utf-8"))
    rule = sel["candidate_rule"]
    candidate = rule["candidate"]
    cal_path = results / "calibration_s7.json"
    calibration = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else {}
    limits = None
    if candidate:
        limits = {}
        for qid in QIDS:
            path = results / f"ece_limit_{qid}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            want = {"predictions": f"{candidate}_{qid}_validation.csv", "n": TEST_ROWS[qid], "draws": ECE_DRAWS}
            got = {k: data.get(k) for k in want}
            if got != want:
                raise SystemExit(f"{path} holds {got}; the freeze needs {want} (the candidate's calibrated validation predictions at the test size)")
            p95 = float(data["ece_if_calibrated"]["p95"])
            limits[qid] = {"p95": p95, "floor": ECE_FLOOR, "limit": max(ECE_FLOOR, p95), "n": data["n"], "draws": data["draws"],
                           "file": path.name}
    out = {
        "plan": "docs/prompt_session7.md, SELECTION RULES and step 11",
        "system": candidate,
        "candidate_or_none": candidate if candidate else "none",
        "not_built": rule.get("not_built"),
        "selection_scores": rule["scores"],
        "v0_2_candidate_score": rule["v0_2_candidate"]["score"],
        "calibration": {s: {q: calibration[s][q]["chosen"] for q in calibration[s]} for s in calibration if s in ARMS},
        "ece_limits": limits,
        "labeler": sel.get("labeler"),
        "time_utc": now(),
    }
    out_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("select", "freeze"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--v02-selection", type=Path, default=V02_SELECTION)
    parser.add_argument("--staging", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "select":
        select(args.results, args.v02_selection)
    else:
        freeze(args.results, args.staging)


if __name__ == "__main__":
    main()

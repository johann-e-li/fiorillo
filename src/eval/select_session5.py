"""Session 5's selection rules (docs/prompt_session5.md, SELECTION RULES), applied in order and recorded in
results/fiorillo_v0_2/selection.json with every score and each rule's outcome. Each subcommand reads the metrics JSON
that src/train/train_laya.py wrote for each decision-layer run (<metrics-dir>/<system>_metrics.json; the selection
score is best_validation_selection_log_loss_exact, the mean of the per-question validation log losses on the calibrate
items at the kept checkpoint, before the temperature) and adds its rule to selection.json, keeping what is there.

maybe (step 9): s5_laya_w is kept over s5_laya only if its PubMedQA maybe-item out-of-fold log loss is lower and the
  mean of its two questions' out-of-fold log losses is not higher. The out-of-fold figures come from
  src/eval/calibration_family_cv.py (temperature family, 5 folds, 20 repeats, the seed), one JSON per run and question
  in --oof-dir, named <system>_<qid>_oof.json. The kept run is the Laya-reader baseline, and its recipe (weighted or
  not) is used by every later decision-layer run.
reader (step 10): start with mix A as the winner; B replaces it only if B's score is more than 0.01 below the winner's;
  then C replaces the winner only if C's score is more than 0.01 below it (a mix that was NOT BUILT is skipped). If the
  winner does not score below the baseline, there is no task round and no long run.
task (step 11): the task round is kept only if s5_task scores below the winner's run.
head (step 12): of the two head starts on reader v1 (s5_final_fresh_head, s5_final_laya_head) the lower score wins;
  with no long run, s5_final_laya_head (on Laya's reader) is compared with the baseline run as the fresh-head run. A tie
  goes to the fresh head. The winner is the v0.2 candidate.

Usage: python src/eval/select_session5.py maybe --metrics-dir results/fiorillo_v0_2/metrics --oof-dir results/fiorillo_v0_2/oof
       python src/eval/select_session5.py reader --metrics-dir results/fiorillo_v0_2/metrics
       python src/eval/select_session5.py task --metrics-dir ...
       python src/eval/select_session5.py head --metrics-dir ...
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SELECTION = ROOT / "results" / "fiorillo_v0_2" / "selection.json"
QIDS = ("pubmedqa_answer", "readmit_window")
MAYBE = 2
MARGIN = 0.01


def selection_score(metrics):
    if "best_validation_selection_log_loss_exact" in metrics:
        return float(metrics["best_validation_selection_log_loss_exact"])
    return float(metrics["best_validation_selection_log_loss"])


def oof_summary(oof_by_qid):
    """From calibration_family_cv outputs per question: PubMedQA overall and maybe-item OOF log loss, readmission
    overall, and the mean of the two questions (temperature family)."""
    fam = {q: oof_by_qid[q]["families"]["temperature"] for q in QIDS}
    pubmed = fam["pubmedqa_answer"]["out_of_fold_log_loss"]["mean"]
    maybe = fam["pubmedqa_answer"]["out_of_fold_log_loss_by_label"][MAYBE]
    readmit = fam["readmit_window"]["out_of_fold_log_loss"]["mean"]
    return {"pubmedqa": pubmed, "pubmedqa_maybe": maybe, "readmission": readmit, "mean_of_questions": (pubmed + readmit) / 2}


def maybe_rule(plain, weighted):
    """True if the weighted run is kept: lower maybe-item OOF log loss and a mean OOF log loss that is not higher."""
    return weighted["pubmedqa_maybe"] < plain["pubmedqa_maybe"] and weighted["mean_of_questions"] <= plain["mean_of_questions"]


def reader_rule(scores, baseline, margin=MARGIN):
    """scores: {"A": s, "B": s or None, "C": s or None} (None when NOT BUILT). Returns (winner, long_run, steps)."""
    winner, steps = "A", []
    for challenger in ("B", "C"):
        if scores.get(challenger) is None:
            steps.append(f"mix {challenger} NOT BUILT; skipped")
            continue
        replaced = scores[challenger] < scores[winner] - margin
        steps.append(f"mix {challenger} {scores[challenger]:.6f} vs winner {winner} {scores[winner]:.6f} - {margin}: "
                     + ("replaces it" if replaced else "does not replace it"))
        if replaced:
            winner = challenger
    return winner, scores[winner] < baseline, steps


def task_rule(task_score, winner_score):
    return task_score < winner_score


def head_rule(fresh_score, laya_score):
    """'fresh' or 'laya': the lower score wins; a tie goes to the fresh head."""
    return "laya" if laya_score < fresh_score else "fresh"


def load_metrics(metrics_dir, system):
    path = Path(metrics_dir) / f"{system}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_selection(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def save_selection(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("rule", choices=("maybe", "reader", "task", "head"))
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--oof-dir", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=SELECTION)
    args = parser.parse_args(argv)
    sel = load_selection(args.selection)

    if args.rule == "maybe":
        runs = {}
        for system in ("s5_laya", "s5_laya_w"):
            oof = {q: json.loads((args.oof_dir / f"{system}_{q}_oof.json").read_text(encoding="utf-8")) for q in QIDS}
            runs[system] = {"selection_score": selection_score(load_metrics(args.metrics_dir, system)), "out_of_fold": oof_summary(oof)}
        kept_weighted = maybe_rule(runs["s5_laya"]["out_of_fold"], runs["s5_laya_w"]["out_of_fold"])
        baseline = "s5_laya_w" if kept_weighted else "s5_laya"
        sel["maybe_rule"] = {"time": now(), "runs": runs, "weighted_kept": kept_weighted, "recipe": "--maybe-weight " + ("auto" if kept_weighted else "none"),
                             "baseline": baseline, "baseline_score": runs[baseline]["selection_score"]}
    elif args.rule == "reader":
        base = sel["maybe_rule"]
        scores = {}
        for mix in ("A", "B", "C"):
            m = load_metrics(args.metrics_dir, f"s5_mix_{mix.lower()}")
            scores[mix] = selection_score(m) if m else None
        if scores["A"] is None:
            raise SystemExit("s5_mix_a has no metrics; the reader rule cannot run")
        winner, long_run, steps = reader_rule(scores, base["baseline_score"])
        bioclinical = load_metrics(args.metrics_dir, "s5_bioclinical")
        sel["reader_rule"] = {"time": now(), "scores": scores, "margin": MARGIN, "steps": steps, "winner": winner,
                              "winner_score": scores[winner], "baseline": base["baseline"], "baseline_score": base["baseline_score"],
                              "long_run": long_run, "s5_bioclinical_score_paper_only": selection_score(bioclinical) if bioclinical else None}
        if not long_run:
            sel["reader_rule"]["note"] = "NOT BUILT: long run, no mix beat Laya's reader on validation"
    elif args.rule == "task":
        reader = sel["reader_rule"]
        task_score = selection_score(load_metrics(args.metrics_dir, "s5_task"))
        kept = task_rule(task_score, reader["winner_score"])
        sel["task_rule"] = {"time": now(), "s5_task_score": task_score, "winner_score": reader["winner_score"], "task_round_kept": kept}
    else:
        laya = selection_score(load_metrics(args.metrics_dir, "s5_final_laya_head"))
        if sel.get("reader_rule", {}).get("long_run"):
            fresh_name = "s5_final_fresh_head"
            fresh = selection_score(load_metrics(args.metrics_dir, fresh_name))
        else:
            fresh_name = sel["maybe_rule"]["baseline"]
            fresh = sel["maybe_rule"]["baseline_score"]
        pick = head_rule(fresh, laya)
        sel["head_rule"] = {"time": now(), "fresh_head_run": fresh_name, "fresh_head_score": fresh,
                            "laya_head_run": "s5_final_laya_head", "laya_head_score": laya, "winner": pick,
                            "candidate": fresh_name if pick == "fresh" else "s5_final_laya_head"}
    save_selection(args.selection, sel)
    print(json.dumps(sel, indent=2))
    return sel


if __name__ == "__main__":
    main()

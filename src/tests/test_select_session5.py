"""Tests for session 5's selection rules (src/eval/select_session5.py) on made-up scores."""
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC / "eval"))
import select_session5 as S  # noqa: E402


def oof(pubmed, maybe, readmit):
    return {"pubmedqa": pubmed, "pubmedqa_maybe": maybe, "readmission": readmit, "mean_of_questions": (pubmed + readmit) / 2}


def test_maybe_rule():
    plain = oof(0.80, 1.60, 0.90)
    assert S.maybe_rule(plain, oof(0.80, 1.40, 0.90))          # maybe better, mean equal: kept
    assert not S.maybe_rule(plain, oof(0.82, 1.40, 0.90))      # maybe better, mean higher: not kept
    assert not S.maybe_rule(plain, oof(0.70, 1.60, 0.90))      # maybe not lower: not kept


def test_reader_rule_margin_favors_the_simpler_mix():
    assert S.reader_rule({"A": 0.500, "B": 0.495, "C": 0.480}, baseline=0.52)[:2] == ("C", True)
    assert S.reader_rule({"A": 0.500, "B": 0.495, "C": 0.491}, baseline=0.52)[:2] == ("A", True)
    assert S.reader_rule({"A": 0.500, "B": 0.480, "C": 0.475}, baseline=0.52)[:2] == ("B", True)   # C within 0.01 of B
    assert S.reader_rule({"A": 0.500, "B": 0.480, "C": None}, baseline=0.52)[:2] == ("B", True)
    assert S.reader_rule({"A": 0.530, "B": 0.525, "C": 0.529}, baseline=0.52)[:2] == ("A", False)  # no long run
    assert S.reader_rule({"A": 0.52, "B": None, "C": None}, baseline=0.52)[1] is False           # equal is not below


def test_task_and_head_rules():
    assert S.task_rule(0.49, 0.50) and not S.task_rule(0.50, 0.50)
    assert S.head_rule(0.50, 0.49) == "laya" and S.head_rule(0.49, 0.50) == "fresh" and S.head_rule(0.5, 0.5) == "fresh"


def cv_output(overall, by_label):
    return {"families": {"temperature": {"out_of_fold_log_loss": {"mean": overall}, "out_of_fold_log_loss_by_label": by_label}}}


def test_command_line_writes_every_rule(tmp_path):
    metrics, oofd, sel = tmp_path / "metrics", tmp_path / "oof", tmp_path / "selection.json"
    metrics.mkdir()
    oofd.mkdir()
    scores = {"s5_laya": 0.60, "s5_laya_w": 0.61, "s5_mix_a": 0.58, "s5_mix_b": 0.565, "s5_mix_c": 0.56,
              "s5_bioclinical": 0.55, "s5_task": 0.555, "s5_final_fresh_head": 0.54, "s5_final_laya_head": 0.53}
    for system, score in scores.items():
        (metrics / f"{system}_metrics.json").write_text(json.dumps({"best_validation_selection_log_loss": round(score, 4),
                                                                   "best_validation_selection_log_loss_exact": score}))
    for system, maybe in (("s5_laya", 1.5), ("s5_laya_w", 1.3)):
        (oofd / f"{system}_pubmedqa_answer_oof.json").write_text(json.dumps(cv_output(0.7, [0.5, 0.6, maybe])))
        (oofd / f"{system}_readmit_window_oof.json").write_text(json.dumps(cv_output(0.9, [0.8, 1.0, 1.1])))
    S.main(["maybe", "--metrics-dir", str(metrics), "--oof-dir", str(oofd), "--selection", str(sel)])
    S.main(["reader", "--metrics-dir", str(metrics), "--selection", str(sel)])
    S.main(["task", "--metrics-dir", str(metrics), "--selection", str(sel)])
    out = S.main(["head", "--metrics-dir", str(metrics), "--selection", str(sel)])
    assert out["maybe_rule"]["weighted_kept"] and out["maybe_rule"]["baseline"] == "s5_laya_w"
    assert out["reader_rule"]["winner"] == "B" and out["reader_rule"]["long_run"]   # C is not 0.01 below B
    assert out["task_rule"]["task_round_kept"] is True                               # 0.555 is below B's 0.565
    assert out["head_rule"]["candidate"] == "s5_final_laya_head"
    assert json.loads(sel.read_text()) == out

"""Session 5 follow-up, at Johann's request (29 September 2026): would a second try at Fiorillo v0.2 that retrains only
the decision layer on reader v1 have a real chance of passing release criterion 4? Nothing is trained or selected here;
no model is changed; the release bar is not changed.

It reads the test predictions already written (results/fiorillo_v0_2/predictions/ and results/public_v0/predictions/)
and reports, for PubMedQA (500 test items) and readmission (6,998):
  1. the candidate minus v0.1 per item, with the paired bootstrap of compare.py (1,000 resamples, seed 20260927);
  2. the pass line of criterion 4's PubMedQA part (the 95% interval wholly below zero) and, for a second look at the
     same test items, a stricter line (a 97.5% interval, a Bonferroni split of 5% over two looks); and the true gain
     a model would need for an 80% chance of passing each line;
  3. how much the same reader moves between two training runs with nearly the same recipe (two pairs; a third pair,
     v0.1's own two runs, is shown but not counted because correction 7 changed the recipe between them), as a
     root-mean-square spread per run;
  4. the chance that a retry that is truly no better than the candidate passes by luck, from 1 to 3.
Writes results/fiorillo_v0_2/retry_check.json. Standard library and numpy only; runs with the linked shell's python3.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as M  # noqa: E402

S4 = ROOT / "results" / "public_v0" / "predictions"
S5 = ROOT / "results" / "fiorillo_v0_2" / "predictions"
QIDS = ("pubmedqa_answer", "readmit_window")
SEED, RESAMPLES = 20260927, 1000
Z95, Z975, Z80 = 1.959964, 2.241403, 0.841621
PAIRS = [  # (what, run in session 4, run in session 5 or the other session 4 run, counted in the spread)
    ("BioClinical's reader, fresh head", (S4, "arm_bioclinical"), (S5, "s5_bioclinical"), True),
    ("Laya's reader, fresh head", (S4, "arm_laya_fresh"), (S5, "s5_laya"), True),
    # not counted: correction 7 changed the recipe between these two runs (150 validation items instead of 50)
    ("Laya's reader and head: session 4 run 2, then run 3 (v0.1) after correction 7", (S4, "arm_laya_run2"), (S4, "arm_laya"), False),
]


def item_losses(folder, system, qid):
    path = folder / f"{system}_{qid}_test.csv"
    ids, labels, probs = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = sorted((c for c in reader.fieldnames if c.startswith("p") and c[1:].isdigit()), key=lambda c: int(c[1:]))
        for row in reader:
            ids.append(row["id"])
            labels.append(int(row["label"]))
            probs.append([float(row[c]) for c in cols])
    labels, probs = np.array(labels), np.array(probs)
    per_item = -np.log(np.clip(probs[np.arange(len(labels)), labels], M.EPS, None))
    assert abs(per_item.mean() - M.log_loss(labels, probs)) < 1e-12
    return dict(zip(ids, per_item))


def paired(a, b):
    ids = sorted(a)
    if set(ids) != set(b):
        raise SystemExit("the two prediction files do not hold the same test items")
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(SEED)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(RESAMPLES)])
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"n": len(d), "estimate": float(d.mean()), "low": float(low), "high": float(high),
            "se_bootstrap": float(boots.std(ddof=1)), "se_analytic": float(d.std(ddof=1) / math.sqrt(len(d)))}


def phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def main():
    out = {"source": "src/eval/retry_check_session5.py", "note": __doc__.split("\n\n")[0].replace("\n", " ")}
    cand = json.loads((ROOT / "results" / "fiorillo_v0_2" / "release_candidate.json").read_text(encoding="utf-8"))["system"]
    d1 = {q: paired(item_losses(S5, cand, q), item_losses(S4, "arm_laya", q)) for q in QIDS}
    out["candidate_minus_v0_1"] = d1
    se_p, se_r = d1["pubmedqa_answer"]["se_bootstrap"], d1["readmit_window"]["se_bootstrap"]
    out["pubmedqa_lines"] = {
        "pass_line_95": -Z95 * se_p, "pass_line_97_5_second_look": -Z975 * se_p,
        "true_gain_for_80_percent_95": (Z95 + Z80) * se_p, "true_gain_for_80_percent_97_5": (Z975 + Z80) * se_p}
    out["readmission_line"] = {"estimate_must_be_at_most": Z95 * se_r}
    spread = {q: [] for q in QIDS}
    rows = []
    for what, (fa, sa), (fb, sb), counted in PAIRS:
        row = {"what": what, "runs": [sa, sb], "counted_in_spread": counted}
        for q in QIDS:
            a, b = item_losses(fa, sa, q), item_losses(fb, sb, q)
            diff = float(np.mean([b[i] - a[i] for i in a]))
            row[q] = {sa: float(np.mean(list(a.values()))), sb: float(np.mean(list(b.values()))), "difference": diff}
            if counted:
                spread[q].append(diff)
        rows.append(row)
    out["run_to_run_pairs"] = rows
    sigma = {q: float(math.sqrt(np.mean(np.square(spread[q])) / 2)) for q in QIDS}
    out["run_to_run_sd_per_run"] = sigma
    est_p, est_r = d1["pubmedqa_answer"]["estimate"], d1["readmit_window"]["estimate"]
    lines = out["pubmedqa_lines"]
    p_read = phi((out["readmission_line"]["estimate_must_be_at_most"] - est_r) / sigma["readmit_window"])
    luck = {}
    for name, line in (("95", lines["pass_line_95"]), ("97_5_second_look", lines["pass_line_97_5_second_look"])):
        p_pub = phi((line - est_p) / sigma["pubmedqa_answer"])
        luck[name] = {"pubmedqa_part": p_pub, "readmission_part": p_read, "both": p_pub * p_read}
    out["chance_a_retry_no_better_than_the_candidate_passes"] = luck
    path = ROOT / "results" / "fiorillo_v0_2" / "retry_check.json"
    path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

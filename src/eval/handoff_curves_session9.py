"""Session 9's hand-off analyses (docs/prompt_session9.md, HAND-OFF ANALYSES: ideas I-34 and I-35 of
docs/ideas_register.md; K7 of COMPARISONS). Secondary and descriptive: v0.5 hands nothing off, these analyses are not part
of its bar, and I-35's rule is never released.

A pair is a Fiorillo system and a large model on one question and split (EI: s8_ei, or v0.5's EI specialist, with
gemma4_31b, or gemma4_31b_long under GEMMA ON THE LONGER INPUT; PubMedQA: s7_b3 with gemma4_31b). Inputs: calibrated
prediction files in session 4's format (id,label,p0,p1,p2); the question's records (JSONL with id and label, and for EI
group.pmcid, the article); for I-35, three raw prediction files of Fiorillo's runs (a weight-averaged system: its three
source runs). The items: on validation, the records marked calibrate (a record without the key counts as marked, as
session 8 read them: all 1,237 EI validation prompts, and PubMedQA's 150); on test, every record. Rows follow the large
model's file (handoff_session8.aligned); its labels must match Fiorillo's file and the records.

The ranking (session 8's, handoff_session8.combine): confidence is Fiorillo's calibrated top probability; the least
confident is handed off first; at hand-off rate r the floor(r n) least confident items go to the large model (session 8's
count, in integers here) and the rest are kept. Ties: among exact ties the lower record ID, as text, is handed off first
(in every order below), and inside a resample copies of one item are ordered by their place in the resample (the key is
the ID's rank times the resample's size plus the place).

I-34 (subcommand curves), for each pair:
- The risk-coverage curve: at coverages 5 to 100 percent in steps of 5, the error rate of the answers kept, which are
  those the confidence rule keeps at hand-off rate 100 minus the coverage: the n - floor((100 - c) n / 100) most confident.
- AURC, per item: the mean over k = 1 to n of the error rate of the k most confident answers, the most confident first in
  the reverse of the hand-off order (so among exact ties the higher ID counts as more confident).
- E-AURC: AURC minus the AURC, by the same rule, of a perfect ranking of the same answers (every right answer first).
- AUGRC (Traub et al. 2024, "Overcoming common flaws in the evaluation of selective classification systems",
  arXiv:2407.01032, equations 3, 6 and 7): the area under the generalized risk-coverage curve, where the generalized risk
  is the number of wrong answers among those kept divided by n, all items. Computed as the authors' code computes it
  (fd-shifts, generalized_risk_stats and the trapezoid): one point for each distinct confidence (tied answers enter
  together), the curve closed at coverage 0 with risk 0, the trapezoid rule between points; their code then multiplies
  the area by 1,000 for display, which is not done here. This equals their equation 7,
  (1 - AUROC_f) acc (1 - acc) + (1 - acc)^2 / 2 (AUROC_f: the area under the ROC curve of the confidence for telling
  right answers from wrong ones, ties counted half; acc: Fiorillo's accuracy), which is checked on every run. Without ties
  it is the per-item mean over k of (wrong answers among the k most confident) / n, less (1 - acc) / (2 n).
- Deferral curves, combined accuracy at hand-off rates 0 to 100 percent in steps of 5 (floor(r n) items answered by the
  large model, the rest by Fiorillo): confidence (session 8's order); random_expected (the exact expected value of
  handing off floor(r n) items drawn uniformly without replacement: acc_F + floor(r n) / n x (acc_L - acc_F)); oracle (a
  bound, not a rule: first the items Fiorillo gets wrong and the large model right, then those both get right or both
  wrong, last those Fiorillo gets right and the large model wrong, each group in session 8's order; no rule that hands off
  as many of the same items can do better); large_model_confidence (a reference, not a usable rule: the large model's
  calibrated top probability, the highest handed off first).
- Intervals: 95 percent, percentile method, from 1,000 resamples (handoff_session8.resample_indices: numpy's
  default_rng(20260927); EI resamples articles, integers(0, m, m) over the m articles sorted by PMCID as text, every prompt
  of a drawn article kept; PubMedQA resamples items, integers(0, n, n)), at the registered rates 10, 20, 30 and 50 percent
  for every curve, and for AURC, E-AURC and AUGRC; every statistic is computed again inside each resample, the items
  ranked again.

I-35 (subcommands fit-rule and apply-rule, and curves --rule; EI only):
- Target: Fiorillo's calibrated top answer wrong and the large model's right.
- Features: top_probability (Fiorillo's calibrated top probability); margin (its top probability minus its second);
  entropy (of its calibrated distribution, natural logarithm, 0 log 0 taken as 0); run_sd (the sample standard deviation,
  ddof 1, across Fiorillo's runs of the raw probability each run gives the averaged top answer, the answer with the
  highest mean of the runs' raw probabilities; three runs, or two if the cost rule left v0.5's winning recipe with two,
  correction 8). The features are standardized, so the logarithm's base and the ddof change nothing in the rule. A frozen
  rule is applied only with as many runs as it was fitted on.
- Model: scikit-learn's LogisticRegression with its L2 penalty, C 1.0, the lbfgs solver, the intercept not penalized and
  no class weights, on the features standardized with their validation mean and standard deviation (ddof 0), which are
  frozen with the coefficients. The fit is checked against the L2 optimum's gradient (C 1.0).
- The rule: at rate r, the floor(r n) items with the highest predicted probability are handed off.
- fit-rule writes the frozen rule and the room on validation (the items Fiorillo gets wrong and the large model right,
  and how many of them the confidence rule's 10 percent hand-off holds, and how many it hands off the other way).
- apply-rule (and curves --rule, which puts the same result in the pair's entry) applies a frozen rule to a split and
  compares it with the confidence rule at the registered rates: each one's combined accuracy, and the paired difference
  "rule minus confidence" from the same 1,000 resamples, with session 8's reading (compare_session8.reading).

Guards: on test, curves and apply-rule refuse to run unless release_candidate.json and testpreds_log.json exist in
--guard-dir (default results/fiorillo_v0_5); an input whose file name names the other split is refused; fit-rule reads
validation files only. Outputs are JSON (indent 1) with the keys plan and written_utc.

Subcommands:
  curves      I-34 for each --pair QUESTION FIORILLO FIORILLO_CSV LARGE_MODEL LARGE_CSV RECORDS (repeatable; default:
              step 6's two pairs, s8_ei and s7_b3 with gemma4_31b, on --split), and I-35's frozen rules (--rule FIORILLO
              RULE_JSON RUN1 RUN2 [RUN3], repeatable; each attaches to the EI pair of that Fiorillo system with the rule's
              large model, and its result is that pair's i35_rule). Default --out: on validation
              results/fiorillo_v0_5/validation_analyses/handoff_curves_validation_s8.json, on test
              results/fiorillo_v0_5/handoff_curves.json.
  fit-rule    I-35's rule on validation (default: s8_ei and gemma4_31b, s8_ei's three runs; --out
              results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json).
  apply-rule  a frozen rule on --split against the confidence rule (default files: s8_ei's on that split; --out required).

Usage: python3 src/eval/handoff_curves_session9.py curves --split validation
       python3 src/eval/handoff_curves_session9.py fit-rule
       python3 src/eval/handoff_curves_session9.py curves --split validation \\
           --pair evidence_effect s9_ei FIO_CSV gemma4_31b LARGE_CSV data/processed/session8/evidence_effect_validation.jsonl \\
           --out results/fiorillo_v0_5/handoff_validation.json
       python3 src/eval/handoff_curves_session9.py fit-rule --fiorillo-name s9_ei --fiorillo FIO_CSV --runs RAW1 RAW2 RAW3 \\
           --large-name gemma4_31b --large LARGE_CSV --records data/processed/session8/evidence_effect_validation.jsonl \\
           --out results/fiorillo_v0_5/handoff_rule_v05.json
       python3 src/eval/handoff_curves_session9.py curves --split test --pair ... [--pair ...] \\
           --rule s8_ei results/fiorillo_v0_5/validation_analyses/handoff_rule_s8.json RAW1 RAW2 RAW3 \\
           --out results/fiorillo_v0_5/handoff_curves.json
       python3 src/eval/handoff_curves_session9.py apply-rule --split test --rule RULE_JSON --fiorillo FIO_CSV \\
           --runs RAW1 RAW2 RAW3 --large LARGE_CSV --records RECORDS --out FILE
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
import compare_session8 as C8  # noqa: E402
import handoff_session8 as H8  # noqa: E402
import metrics as M  # noqa: E402

SEED = 20260927
RESAMPLES = 1000
RATES = tuple(range(0, 101, 5))  # hand-off rates, percent
COVERAGES = tuple(range(5, 101, 5))  # coverages, percent
REGISTERED = (10, 20, 30, 50)  # the registered hand-off rates, percent
EI = "evidence_effect"
PQA = "pubmedqa_answer"
QUESTIONS = (EI, PQA)
SPLITS = ("validation", "test")
CURVES = ("confidence", "random_expected", "oracle", "large_model_confidence")
AREAS = ("aurc", "e_aurc", "augrc")
FEATURES = ("top_probability", "margin", "entropy", "run_sd")
C_L2 = 1.0
RESULTS3 = ROOT / "results" / "fiorillo_v0_3"
RESULTS4 = ROOT / "results" / "fiorillo_v0_4"
RESULTS5 = ROOT / "results" / "fiorillo_v0_5"
ANALYSES = RESULTS5 / "validation_analyses"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
PLAN = "docs/prompt_session9.md, HAND-OFF ANALYSES"
FEATURE_TEXT = {
    "top_probability": "Fiorillo's calibrated top probability",
    "margin": "Fiorillo's calibrated top probability minus its second highest",
    "entropy": "the entropy of Fiorillo's calibrated distribution, natural logarithm, 0 log 0 taken as 0",
    "run_sd": ("the sample standard deviation (ddof 1) across Fiorillo's runs (three, or two under correction 8) of the raw "
               "probability each run gives the averaged top answer (the answer with the highest mean of the runs' raw probabilities)"),
}
DEFINITIONS = {
    "items": ("validation: the records marked calibrate (a record without the key counts as marked, as session 8 read them); "
              "test: every record; rows follow the large model's file (handoff_session8.aligned)"),
    "ranking": ("confidence is Fiorillo's calibrated top probability; the least confident is handed off first; at hand-off rate r "
                "the floor(r n) least confident items go to the large model (session 8's order and count)"),
    "ties": ("among exact ties the lower record ID, as text, is handed off first, in every order; inside a resample copies of one "
             "item are ordered by their place in the resample (key: the ID's rank times the resample's size plus the place)"),
    "risk_coverage": ("at coverage c percent, the error rate of the n - floor((100 - c) n / 100) most confident answers, those the "
                      "confidence rule keeps at hand-off rate 100 - c percent"),
    "aurc": ("per item: the mean over k = 1 to n of the error rate of the k most confident answers (the reverse of the hand-off "
             "order, so among exact ties the higher ID counts as more confident)"),
    "e_aurc": "AURC minus the AURC, by the same rule, of a perfect ranking of the same answers (every right answer first)",
    "augrc": ("Traub et al. 2024 (arXiv:2407.01032, equations 3, 6 and 7): the area under the generalized risk-coverage curve, "
              "the generalized risk being the wrong answers among those kept divided by n (all items); computed as the authors' "
              "code does: one point for each distinct confidence, the curve closed at coverage 0 with risk 0, the trapezoid rule, "
              "not multiplied by the 1,000 their code uses for display; "
              "equal to (1 - AUROC_f) acc (1 - acc) + (1 - acc)^2 / 2, checked; without ties, the per-item mean over k of "
              "(wrong among the k most confident) / n, less (1 - acc) / (2 n)"),
    "confidence": "combined accuracy, the floor(r n) least confident handed off (session 8's rule)",
    "random_expected": ("the exact expected combined accuracy when floor(r n) items drawn uniformly without replacement are "
                        "handed off: acc_F + floor(r n) / n x (acc_L - acc_F)"),
    "oracle": ("a bound, not a rule: first the items Fiorillo gets wrong and the large model right, then those both get right or "
               "both wrong, last those Fiorillo gets right and the large model wrong, each group in session 8's order; no rule "
               "that hands off as many of the same items can do better"),
    "large_model_confidence": ("a reference, not a usable rule (it needs the large model's answer on every item): the large "
                               "model's calibrated top probability, the highest handed off first"),
    "intervals": ("95 percent, percentile, from the resamples of handoff_session8.resample_indices (numpy default_rng(20260927); "
                  "EI: articles, integers(0, m, m) over the m articles sorted by PMCID as text, every prompt of a drawn article "
                  "kept; PubMedQA: items, integers(0, n, n)); at the registered rates for every curve, and for the three areas; "
                  "each statistic computed again inside each resample, the items ranked again"),
}


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if ROOT in path.parents else str(path)


def file_info(path, rows=None):
    out = {"path": rel(path), "sha256": C8.sha256_file(path)}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf-8")


def handed(rate, n):
    """Session 8's count at a hand-off rate in percent, floor(rate x n / 100), in integers."""
    return (int(rate) * int(n)) // 100


def check_split(split, paths):
    other = "_test" if split == "validation" else "_validation"
    named = [str(p) for p in paths if other in Path(p).name]
    if named:
        raise SystemExit(f"the split is {split}, but these files name the other split: {named}")


def guard_test(split, guard_dir):
    if split == "test":
        for name in ("release_candidate.json", "testpreds_log.json"):
            if not (Path(guard_dir) / name).exists():
                raise SystemExit(f"{name} does not exist in {guard_dir}; test predictions are read only after the freeze")


# ---------------------------------------------------------------- inputs

def read_items(records_path, split):
    """The items' IDs in the records' order, with each one's label and article (group.pmcid, or None)."""
    ids, label_of, article_of = [], {}, {}
    with open(records_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if split == "validation" and not r.get("calibrate", True):
                continue
            ids.append(r["id"])
            label_of[r["id"]] = int(r["label"])
            article_of[r["id"]] = (r.get("group") or {}).get("pmcid")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{Path(records_path).name}: a record ID is repeated")
    return ids, label_of, article_of


def load_pair(question, fio_path, big_path, records_path, split):
    """(ids, labels, Fiorillo's and the large model's calibrated probabilities, articles or None)."""
    if question not in QUESTIONS:
        raise SystemExit(f"unknown question {question}; expected one of {QUESTIONS}")
    check_split(split, [fio_path, big_path, records_path])
    keep, label_of, article_of = read_items(records_path, split)
    if not keep:
        raise SystemExit(f"{Path(records_path).name} holds no item for {split}")
    ids, labels, fio, big = H8.aligned(fio_path, big_path, set(keep))
    if not np.array_equal(labels, np.array([label_of[i] for i in ids])):
        raise SystemExit(f"labels differ between the prediction files and {Path(records_path).name}")
    clusters = None
    if question == EI:
        clusters = [article_of[i] for i in ids]
        if any(c is None for c in clusters):
            raise SystemExit(f"{Path(records_path).name}: an EI record has no group.pmcid")
    return ids, labels, fio, big, clusters


def load_runs(paths, ids, labels):
    """The runs' raw probabilities in the rows of ids; each file must hold every item with the same label."""
    if len(paths) not in (2, 3):
        raise SystemExit("I-35 reads two or three raw run files (correction 8)")
    runs = []
    for path in paths:
        r_ids, r_labels, r_probs = M.read_predictions(path)
        position = {i: k for k, i in enumerate(r_ids)}
        missing = [i for i in ids if i not in position]
        if missing:
            raise SystemExit(f"{Path(path).name} lacks {len(missing)} of the items, for example {missing[:3]}")
        order = np.array([position[i] for i in ids])
        if not np.array_equal(r_labels[order], labels):
            raise SystemExit(f"{Path(path).name}: labels differ from Fiorillo's file")
        runs.append(r_probs[order])
    return runs


# ---------------------------------------------------------------- orders and curves

def handoff_order(conf, tie):
    """Session 8's hand-off order (handoff_session8.combine): the least confident first, ties by the key (lower first)."""
    return np.lexsort((tie, conf))


def highest_first(score, tie):
    """The highest score handed off first, ties by the key (lower first)."""
    return np.lexsort((tie, -np.asarray(score, dtype=float)))


def oracle_order(right_f, right_l, conf, tie):
    """What handing an item off gains, the most first (+1: Fiorillo wrong and the large model right; 0: both right or both
    wrong; -1: Fiorillo right and the large model wrong); within a gain, session 8's order."""
    gain = right_l.astype(np.int64) - right_f.astype(np.int64)
    return np.lexsort((tie, conf, -gain))


def resample_tie(rank, idx):
    """The tie key inside a resample: the ID's rank as text, then the place in the resample."""
    m = len(idx)
    return rank[idx].astype(np.int64) * m + np.arange(m, dtype=np.int64)


def accuracy_after(order, right_f, right_l, counts):
    """Combined accuracy when the first h items of order are handed off, for each h in counts."""
    gain = right_l[order].astype(float) - right_f[order].astype(float)
    cum = np.concatenate([[0.0], np.cumsum(gain)])
    return (right_f.sum() + cum[np.asarray(counts, dtype=np.int64)]) / len(order)


def random_expected(right_f, right_l, counts):
    """The exact expected combined accuracy when h items drawn uniformly without replacement are handed off."""
    a_f, a_l = right_f.mean(), right_l.mean()
    return a_f + np.asarray(counts, dtype=float) / len(right_f) * (a_l - a_f)


def moved(order, right_f, right_l, counts):
    """Among the first h items of order: (Fiorillo wrong and the large model right, Fiorillo right and the large model wrong)."""
    up = np.concatenate([[0], np.cumsum((~right_f & right_l)[order])])
    down = np.concatenate([[0], np.cumsum((right_f & ~right_l)[order])])
    counts = np.asarray(counts, dtype=np.int64)
    return up[counts], down[counts]


def risk_coverage(conf, wrong, tie, coverages=COVERAGES):
    """(kept, wrong among them, error rate) at each coverage in percent; the kept are the confidence rule's at 100 - c."""
    n = len(conf)
    kept_order = handoff_order(conf, tie)[::-1]
    cum = np.concatenate([[0], np.cumsum(np.asarray(wrong, dtype=np.int64)[kept_order])])
    kept = np.array([n - handed(100 - c, n) for c in coverages], dtype=np.int64)
    return kept, cum[kept], cum[kept] / kept


def aurc_of(wrong_in_order):
    """Per item: the mean over k = 1 to n of the error rate of the first k answers."""
    w = np.asarray(wrong_in_order, dtype=float)
    return float(np.mean(np.cumsum(w) / np.arange(1, len(w) + 1)))


def augrc(conf, wrong):
    """Traub et al. 2024's AUGRC as their code computes it: a point for each distinct confidence, the curve closed at
    coverage 0 with generalized risk 0, the trapezoid rule; the generalized risk is the wrong answers kept divided by n."""
    conf = np.asarray(conf, dtype=float)
    n = len(conf)
    order = np.argsort(-conf, kind="stable")
    c, w = conf[order], np.asarray(wrong, dtype=float)[order]
    ends = np.concatenate([np.flatnonzero(c[1:] != c[:-1]), [n - 1]])
    err = np.cumsum(w)[ends]
    kept = (ends + 1).astype(float)
    err_before = np.concatenate([[0.0], err[:-1]])
    kept_before = np.concatenate([[0.0], kept[:-1]])
    return float(np.sum((kept - kept_before) * (err_before + err)) / (2.0 * n * n))


def augrc_right_sum(conf, wrong, tie):
    """The per-item right sum: the mean over k = 1 to n of (wrong among the k most confident) / n, ranked as for AURC."""
    w = np.asarray(wrong, dtype=float)[handoff_order(conf, tie)[::-1]]
    return float(np.mean(np.cumsum(w)) / len(w))


def augrc_closed_form(conf, wrong):
    """Traub et al. 2024, equation 7: (1 - AUROC_f) acc (1 - acc) + (1 - acc)^2 / 2; (value, AUROC_f or None)."""
    right = ~np.asarray(wrong, dtype=bool)
    acc = float(right.mean())
    if 0.0 < acc < 1.0:
        auroc_f = M.auroc(right.astype(int), np.asarray(conf, dtype=float))
        return (1.0 - auroc_f) * acc * (1.0 - acc) + 0.5 * (1.0 - acc) ** 2, auroc_f
    return 0.5 * (1.0 - acc) ** 2, None


def areas(conf, wrong, tie):
    """AURC (per item), the perfect ranking's AURC, E-AURC and AUGRC."""
    wrong = np.asarray(wrong, dtype=bool)
    aurc = aurc_of(wrong[handoff_order(conf, tie)[::-1]])
    perfect = aurc_of(np.sort(wrong.astype(float)))
    return {"aurc": aurc, "aurc_perfect_ranking": perfect, "e_aurc": aurc - perfect, "augrc": augrc(conf, wrong)}


def statistics(labels, fio, big, tie, rates):
    """Every curve at the rates (percent) and the areas, for these rows and this tie key."""
    n = len(labels)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    conf = fio.max(1)
    counts = np.array([handed(r, n) for r in rates], dtype=np.int64)
    curves = {"confidence": accuracy_after(handoff_order(conf, tie), right_f, right_l, counts),
              "random_expected": random_expected(right_f, right_l, counts),
              "oracle": accuracy_after(oracle_order(right_f, right_l, conf, tie), right_f, right_l, counts),
              "large_model_confidence": accuracy_after(highest_first(big.max(1), tie), right_f, right_l, counts)}
    return counts, curves, areas(conf, ~right_f, tie)


def interval(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None, None
    low, high = np.percentile(v, [2.5, 97.5])
    return float(low), float(high)


def ties(values):
    values = np.asarray(values, dtype=float)
    return int(len(values) - len(np.unique(values)))


def room(labels, fio, big, rank, rates=REGISTERED):
    """The items Fiorillo gets wrong and the large model right, and what the confidence rule's hand-off holds."""
    n = len(labels)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    counts = [handed(r, n) for r in rates]
    up, down = moved(handoff_order(fio.max(1), rank), right_f, right_l, counts)
    return {"items": n, "fiorillo_wrong_large_model_right": int((~right_f & right_l).sum()),
            "fiorillo_right_large_model_wrong": int((right_f & ~right_l).sum()), "both_right": int((right_f & right_l).sum()),
            "both_wrong": int((~right_f & ~right_l).sum()),
            "share_fiorillo_wrong_large_model_right": float((~right_f & right_l).mean()),
            "confidence_rule": {f"rate_{r}": {"handed_off": int(c), "fiorillo_wrong_large_model_right": int(u),
                                              "fiorillo_right_large_model_wrong": int(d)}
                                for r, c, u, d in zip(rates, counts, up, down)}}


def analyze_pair(ids, labels, fio, big, clusters, resamples=RESAMPLES, seed=SEED):
    """I-34 for one pair: the curves, the areas and their intervals."""
    n = len(ids)
    rank = H8.id_ranks(ids)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    counts, curves, area = statistics(labels, fio, big, rank, RATES)
    closed, auroc_f = augrc_closed_form(fio.max(1), ~right_f)
    if abs(closed - area["augrc"]) > 1e-9:
        raise SystemExit(f"AUGRC by the trapezoid ({area['augrc']}) and by equation 7 ({closed}) differ")
    kept, wrong_kept, risk = risk_coverage(fio.max(1), ~right_f, rank)
    draws = {k: [] for k in CURVES + AREAS}
    for idx in H8.resample_indices(n, clusters, resamples, seed):
        _, c, a = statistics(labels[idx], fio[idx], big[idx], resample_tie(rank, idx), REGISTERED)
        for k in CURVES:
            draws[k].append(c[k])
        for k in AREAS:
            draws[k].append(a[k])
    out = {"n": n, "articles": len(set(clusters)) if clusters is not None else None,
           "resampling": "articles, sorted by PMCID as text" if clusters is not None else "items",
           "fiorillo_alone": {"accuracy": float(right_f.mean()), "wrong": int((~right_f).sum())},
           "large_model_alone": {"accuracy": float(right_l.mean()), "wrong": int((~right_l).sum())},
           "ties": {"fiorillo_confidence": ties(fio.max(1)), "large_model_confidence": ties(big.max(1))},
           "room": room(labels, fio, big, rank),
           "risk_coverage": {"coverage_percent": list(COVERAGES), "kept": [int(k) for k in kept],
                             "wrong": [int(w) for w in wrong_kept], "risk": [float(r) for r in risk]},
           "areas": {}, "augrc_check": {"equation_7": float(closed), "auroc_f": auroc_f,
                                        "difference": float(area["augrc"] - closed),
                                        "per_item_right_sum": augrc_right_sum(fio.max(1), ~right_f, rank),
                                        "per_item_right_sum_note": ("the mean over k = 1 to n of (wrong among the k most "
                                                                    "confident) / n, ranked as for AURC; without ties it is "
                                                                    "AUGRC + (1 - acc) / (2 n); reported for reference, not as "
                                                                    "the AUGRC")},
           "deferral": {"rate_percent": list(RATES), "handed_off": [int(c) for c in counts],
                        **{k: [float(v) for v in curves[k]] for k in CURVES}},
           "registered_rates": {}}
    for k in AREAS:
        low, high = interval(draws[k])
        out["areas"][k] = {"estimate": float(area[k]), "low": low, "high": high}
    out["areas"]["aurc_perfect_ranking"] = {"estimate": float(area["aurc_perfect_ranking"])}
    for j, r in enumerate(REGISTERED):
        entry = {"handed_off": int(counts[RATES.index(r)])}
        for k in CURVES:
            low, high = interval(np.array(draws[k])[:, j])
            entry[k] = {"estimate": float(curves[k][RATES.index(r)]), "low": low, "high": high}
        out["registered_rates"][f"rate_{r}"] = entry
    return out


# ---------------------------------------------------------------- I-35

def rule_features(cal, runs):
    """The four features, one row per item, in the order of FEATURES; and the averaged top answer."""
    cal = np.asarray(cal, dtype=float)
    top2 = -np.sort(-cal, axis=1)[:, :2]
    entropy = -np.sum(np.where(cal > 0, cal * np.log(np.where(cal > 0, cal, 1.0)), 0.0), axis=1)
    stack = np.stack([np.asarray(r, dtype=float) for r in runs])  # runs x items x answers
    answer = stack.mean(axis=0).argmax(axis=1)
    given = stack[:, np.arange(cal.shape[0]), answer]  # runs x items
    run_sd = given.std(axis=0, ddof=1)
    return np.column_stack([top2[:, 0], top2[:, 0] - top2[:, 1], entropy, run_sd]), answer


def sigmoid(z):
    z = np.asarray(z, dtype=float)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def rule_arrays(rule):
    names = [f["name"] for f in rule["features"]]
    if tuple(names) != FEATURES:
        raise SystemExit(f"the rule's features are {names}, expected {list(FEATURES)}")
    mean = np.array([rule["scaling"]["mean"][k] for k in FEATURES], dtype=float)
    scale = np.array([rule["scaling"]["scale"][k] for k in FEATURES], dtype=float)
    coef = np.array([rule["coefficients"][k] for k in FEATURES], dtype=float)
    return mean, scale, coef, float(rule["intercept"])


def rule_probability(rule, x):
    """The frozen rule's predicted probability that Fiorillo is wrong and the large model right."""
    mean, scale, coef, intercept = rule_arrays(rule)
    return sigmoid(intercept + ((np.asarray(x, dtype=float) - mean) / scale) @ coef)


def fit_logistic(z, target, c=C_L2):
    """scikit-learn's logistic regression with its default L2 penalty (the penalty is not passed, so that the call is the
    same in scikit-learn 1.7 and in 1.8 and later, where the penalty argument is being replaced); (coefficients, intercept,
    details). The gradient of C x (the log loss summed) + |w|^2 / 2 at the fit is recorded: at most 1e-6 x C n at the L2
    optimum; above 1e-2 the fit is refused, since no fit of that objective gives it."""
    import sklearn
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=c, solver="lbfgs", max_iter=1000, tol=1e-8)
    clf.fit(z, target)
    coef, intercept = clf.coef_[0].astype(float), float(clf.intercept_[0])
    p = sigmoid(intercept + z @ coef)
    gradient = np.concatenate([coef + c * z.T @ (p - target), [c * np.sum(p - target)]])
    details = {"library": f"scikit-learn {sklearn.__version__}",
               "params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                          for k, v in sorted(clf.get_params().items())},
               "iterations": int(np.max(clf.n_iter_)),
               "gradient_max_abs": float(np.max(np.abs(gradient))),
               "gradient_tolerance": 1e-6 * max(1.0, c * len(target)),
               "numpy_minus_scikit_learn_max_abs": float(np.max(np.abs(p - clf.predict_proba(z)[:, 1])))}
    details["at_l2_optimum"] = bool(details["gradient_max_abs"] <= details["gradient_tolerance"])
    if details["gradient_max_abs"] > 1e-2:
        raise SystemExit(f"the fit is not the L2 logistic regression with C {c}: largest gradient {details['gradient_max_abs']}")
    return coef, intercept, details


def fit_rule(fio_name, fio_path, run_paths, big_name, big_path, records_path):
    """I-35's rule fitted on validation; the frozen rule as a dict."""
    check_split("validation", [fio_path, *run_paths, big_path, records_path])
    ids, labels, fio, big, clusters = load_pair(EI, fio_path, big_path, records_path, "validation")
    runs = load_runs(run_paths, ids, labels)
    x, answer = rule_features(fio, runs)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    target = (~right_f & right_l).astype(int)
    if target.sum() in (0, len(target)):
        raise SystemExit(f"the target has one value only ({int(target.sum())} of {len(target)}); the rule cannot be fitted")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    coef, intercept, details = fit_logistic((x - mean) / scale, target)
    rank = H8.id_ranks(ids)
    space = room(labels, fio, big, rank)
    up, ten = space["fiorillo_wrong_large_model_right"], space["confidence_rule"]["rate_10"]
    space["statement"] = (
        f"On validation {fio_name} was wrong where {big_name} was right on {up} of {len(ids):,} prompts "
        f"({100 * up / len(ids):.1f} percent), and the confidence rule's 10 percent hand-off ({ten['handed_off']} prompts) held "
        f"{ten['fiorillo_wrong_large_model_right']} of them and {ten['fiorillo_right_large_model_wrong']} prompts the other way "
        f"({fio_name} right, {big_name} wrong), so a better rule has room of at most about {100 * up / len(ids):.1f} points of "
        f"accuracy and few cases to learn from.")
    return {"plan": f"{PLAN} (I-35; fitted and frozen on validation; not part of v0.5 and never released)",
            "source": "src/eval/handoff_curves_session9.py fit-rule", "written_utc": now(),
            "question": EI, "split": "validation", "system": fio_name, "large_model": big_name,
            "target": "1 when Fiorillo's calibrated top answer is wrong and the large model's calibrated top answer is right, else 0",
            "features": [{"name": k, "definition": FEATURE_TEXT[k],
                          "source": "the raw run files" if k == "run_sd" else "Fiorillo's calibrated file"} for k in FEATURES],
            "scaling": {"method": "standardized: (x - mean) / scale, the mean and the standard deviation (ddof 0) on these validation items",
                        "mean": {k: float(v) for k, v in zip(FEATURES, mean)}, "scale": {k: float(v) for k, v in zip(FEATURES, scale)}},
            "model": {"family": "logistic regression", "penalty": "L2", "C": C_L2, "solver": "lbfgs", "intercept_penalized": False,
                      "class_weight": None, **details},
            "coefficients": {k: float(v) for k, v in zip(FEATURES, coef)}, "intercept": intercept,
            "apply": ("p = 1 / (1 + exp(-(intercept + sum over the features of coefficient x (x - mean) / scale))); at hand-off rate r "
                      "the floor(r n) items with the highest p are handed off, ties by record ID as text (lower first)"),
            "files": {"fiorillo": file_info(fio_path), "runs": [file_info(p) for p in run_paths], "large_model": file_info(big_path),
                      "records": file_info(records_path)},
            "n": len(ids), "articles": len(set(clusters)), "positives": int(target.sum()),
            "averaged_answer_differs_from_calibrated_answer": int((answer != fio.argmax(1)).sum()),
            "ties": {"rule_probability": ties(sigmoid(intercept + ((x - mean) / scale) @ coef))},
            "room": space}


def compare_rule(rule, ids, labels, fio, runs, big, clusters, split, resamples=RESAMPLES, seed=SEED):
    """The frozen rule against the confidence rule at the registered rates: combined accuracy and the paired difference."""
    if clusters is None:
        raise SystemExit("I-35 is for EI only (articles are the resampling unit)")
    x, answer = rule_features(fio, runs)
    prob = rule_probability(rule, x)
    n = len(ids)
    rank = H8.id_ranks(ids)
    right_f, right_l = fio.argmax(1) == labels, big.argmax(1) == labels
    conf = fio.max(1)

    def both(rows, tie):
        rf, rl = right_f[rows], right_l[rows]
        counts = [handed(r, len(rows)) for r in REGISTERED]
        a = accuracy_after(highest_first(prob[rows], tie), rf, rl, counts)
        b = accuracy_after(handoff_order(conf[rows], tie), rf, rl, counts)
        return np.concatenate([a, b, a - b])

    est = both(np.arange(n), rank)
    draws = np.array([both(idx, resample_tie(rank, idx)) for idx in H8.resample_indices(n, clusters, resamples, seed)])
    counts = [handed(r, n) for r in REGISTERED]
    rule_up, rule_down = moved(highest_first(prob, rank), right_f, right_l, counts)
    conf_up, conf_down = moved(handoff_order(conf, rank), right_f, right_l, counts)
    k = len(REGISTERED)
    rates = {}
    for j, r in enumerate(REGISTERED):
        cells = {}
        for name, col in (("rule", j), ("confidence", k + j), ("rule_minus_confidence", 2 * k + j)):
            low, high = interval(draws[:, col])
            cells[name] = {"estimate": float(est[col]), "low": low, "high": high}
        d = cells["rule_minus_confidence"]
        d["reading"] = C8.reading("accuracy", d["low"], d["high"])
        if d["reading"] == "no difference was detected":
            d["excludes"] = (f"a gain larger than {d['high']:.4f} and a loss larger than {-d['low']:.4f} in combined accuracy "
                             "(not equivalence)")
        rates[f"rate_{r}"] = {"handed_off": int(counts[j]), **cells,
                              "rule_hand_off": {"fiorillo_wrong_large_model_right": int(rule_up[j]),
                                                "fiorillo_right_large_model_wrong": int(rule_down[j])},
                              "confidence_hand_off": {"fiorillo_wrong_large_model_right": int(conf_up[j]),
                                                      "fiorillo_right_large_model_wrong": int(conf_down[j])}}
    return {"split": split, "system": rule["system"], "large_model": rule["large_model"],
            "in_sample": split == "validation",
            "note": ("in sample: the rule was fitted on these validation items; descriptive only" if split == "validation"
                     else "the rule as frozen on validation, applied to this split"),
            "n": n, "articles": len(set(clusters)), "resamples": resamples, "seed": seed,
            "averaged_answer_differs_from_calibrated_answer": int((answer != fio.argmax(1)).sum()),
            "ties": {"rule_probability": ties(prob)},
            "room": room(labels, fio, big, rank), "room_on_validation": (rule.get("room") or {}).get("statement"),
            "registered_rates": rates}


def check_rule_runs(rule, rule_path, run_paths):
    """Correction 8: two or three runs, and as many as the frozen rule was fitted on."""
    fitted = len((rule.get("files") or {}).get("runs") or [])
    if len(run_paths) not in (2, 3) or (fitted and len(run_paths) != fitted):
        raise SystemExit(f"{rule_path} was fitted on {fitted} runs; it is applied with {len(run_paths)} (two or three, the same number)")


def apply_rule(rule_path, split, fio_path, run_paths, big_path, records_path, resamples=RESAMPLES, seed=SEED,
               guard_dir=RESULTS5):
    guard_test(split, guard_dir)
    rule = json.loads(Path(rule_path).read_text(encoding="utf-8"))
    check_rule_runs(rule, rule_path, run_paths)
    check_split(split, [fio_path, *run_paths, big_path, records_path])
    ids, labels, fio, big, clusters = load_pair(EI, fio_path, big_path, records_path, split)
    runs = load_runs(run_paths, ids, labels)
    out = compare_rule(rule, ids, labels, fio, runs, big, clusters, split, resamples, seed)
    out["rule_file"] = {**file_info(rule_path), "written_utc": rule.get("written_utc")}
    out["files"] = {"fiorillo": file_info(fio_path, len(ids)), "runs": [file_info(p) for p in run_paths],
                    "large_model": file_info(big_path), "records": file_info(records_path)}
    return out


# ---------------------------------------------------------------- the commands

def default_pairs(split):
    """Step 6's pairs (session 8's files) on a split."""
    p3, p4 = RESULTS3 / "predictions", RESULTS4 / "predictions"
    return [(EI, "s8_ei", p4 / f"s8_ei_{EI}_{split}.csv", "gemma4_31b", p4 / f"gemma4_31b_{EI}_{split}.csv",
             RECORDS_EI / f"{EI}_{split}.jsonl"),
            (PQA, "s7_b3", p3 / f"s7_b3_{PQA}_{split}.csv", "gemma4_31b", p4 / f"gemma4_31b_{PQA}_{split}.csv",
             RECORDS_PUBLIC / f"{PQA}_{split}.jsonl")]


def default_runs(split):
    return [RESULTS4 / "predictions" / f"s8_ei_{k}-raw_{EI}_{split}.csv" for k in (1, 2, 3)]


def curves(split, pairs, rules=(), resamples=RESAMPLES, seed=SEED, guard_dir=RESULTS5):
    guard_test(split, guard_dir)
    out = {"plan": f"{PLAN} (I-34 and I-35; secondary and descriptive)", "source": "src/eval/handoff_curves_session9.py curves",
           "written_utc": None, "split": split, "seed": seed, "resamples": resamples, "hand_off_rates_percent": list(RATES),
           "coverages_percent": list(COVERAGES), "registered_rates_percent": list(REGISTERED), "definitions": DEFINITIONS,
           "pairs": []}
    loaded = []
    for question, fio_name, fio_path, big_name, big_path, records_path in pairs:
        ids, labels, fio, big, clusters = load_pair(question, Path(fio_path), Path(big_path), Path(records_path), split)
        entry = {"question": question, "fiorillo": fio_name, "large_model": big_name,
                 "files": {"fiorillo": file_info(fio_path), "large_model": file_info(big_path), "records": file_info(records_path)},
                 **analyze_pair(ids, labels, fio, big, clusters, resamples, seed)}
        out["pairs"].append(entry)
        loaded.append((entry, ids, labels, fio, big, clusters))
    for fio_name, rule_path, *run_paths in rules:
        rule = json.loads(Path(rule_path).read_text(encoding="utf-8"))
        if rule.get("system") != fio_name:
            raise SystemExit(f"{rule_path} is the rule of {rule.get('system')}, not of {fio_name}")
        check_rule_runs(rule, rule_path, run_paths)
        matches = [x for x in loaded if x[0]["question"] == EI and x[0]["fiorillo"] == fio_name
                   and x[0]["large_model"] == rule.get("large_model")]
        if len(matches) != 1:
            raise SystemExit(f"--rule {fio_name}: expected one EI pair of {fio_name} with {rule.get('large_model')}, "
                             f"found {len(matches)}")
        entry, ids, labels, fio, big, clusters = matches[0]
        check_split(split, run_paths)
        runs = load_runs([Path(p) for p in run_paths], ids, labels)
        entry["i35_rule"] = {**compare_rule(rule, ids, labels, fio, runs, big, clusters, split, resamples, seed),
                             "rule_file": {**file_info(rule_path), "written_utc": rule.get("written_utc")},
                             "runs": [file_info(p) for p in run_paths]}
    out["written_utc"] = now()
    return out


def summary(out):
    lines = []
    for p in out["pairs"]:
        reg = p["registered_rates"]
        lines.append({"question": p["question"], "fiorillo": p["fiorillo"], "large_model": p["large_model"], "n": p["n"],
                      "aurc": round(p["areas"]["aurc"]["estimate"], 5), "e_aurc": round(p["areas"]["e_aurc"]["estimate"], 5),
                      "augrc": round(p["areas"]["augrc"]["estimate"], 5),
                      "confidence": {r: round(v["confidence"]["estimate"], 4) for r, v in reg.items()},
                      "room": [p["room"]["fiorillo_wrong_large_model_right"],
                               p["room"]["confidence_rule"]["rate_10"]["fiorillo_wrong_large_model_right"],
                               p["room"]["confidence_rule"]["rate_10"]["fiorillo_right_large_model_wrong"]],
                      **({"i35_rule_minus_confidence": {r: round(v["rule_minus_confidence"]["estimate"], 4)
                                                        for r, v in p["i35_rule"]["registered_rates"].items()}}
                         if "i35_rule" in p else {})})
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("curves", help="I-34 for one or more pairs on one split, and I-35's frozen rules with --rule")
    c.add_argument("--split", choices=SPLITS, required=True)
    c.add_argument("--pair", nargs=6, action="append", default=None,
                   metavar=("QUESTION", "FIORILLO", "FIORILLO_CSV", "LARGE_MODEL", "LARGE_CSV", "RECORDS"))
    c.add_argument("--rule", nargs="+", action="append", default=None, metavar="FIORILLO RULE_JSON RUN1 RUN2 [RUN3]")
    c.add_argument("--resamples", type=int, default=RESAMPLES)
    c.add_argument("--guard-dir", type=Path, default=RESULTS5)
    c.add_argument("--out", type=Path, default=None)
    f = sub.add_parser("fit-rule", help="I-35's rule fitted and frozen on validation (EI)")
    f.add_argument("--fiorillo-name", default="s8_ei")
    f.add_argument("--fiorillo", type=Path, default=RESULTS4 / "predictions" / f"s8_ei_{EI}_validation.csv")
    f.add_argument("--runs", type=Path, nargs="+", default=default_runs("validation"), help="two or three raw run files")
    f.add_argument("--large-name", default="gemma4_31b")
    f.add_argument("--large", type=Path, default=RESULTS4 / "predictions" / f"gemma4_31b_{EI}_validation.csv")
    f.add_argument("--records", type=Path, default=RECORDS_EI / f"{EI}_validation.jsonl")
    f.add_argument("--out", type=Path, default=ANALYSES / "handoff_rule_s8.json")
    a = sub.add_parser("apply-rule", help="a frozen I-35 rule on a split against the confidence rule (EI)")
    a.add_argument("--split", choices=SPLITS, required=True)
    a.add_argument("--rule", type=Path, default=ANALYSES / "handoff_rule_s8.json")
    a.add_argument("--fiorillo", type=Path, default=None, help="default: s8_ei's calibrated file on the split")
    a.add_argument("--runs", type=Path, nargs="+", default=None, help="two or three; default: s8_ei's three raw run files on the split")
    a.add_argument("--large", type=Path, default=None, help="default: gemma4_31b's calibrated file on the split")
    a.add_argument("--records", type=Path, default=None, help="default: data/processed/session8/evidence_effect_<split>.jsonl")
    a.add_argument("--resamples", type=int, default=RESAMPLES)
    a.add_argument("--guard-dir", type=Path, default=RESULTS5)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.cmd == "curves" and any(len(r) not in (4, 5) for r in (args.rule or ())):
        parser.error("--rule takes FIORILLO RULE_JSON and two or three run files (correction 8)")
    if args.cmd in ("fit-rule", "apply-rule") and args.runs is not None and len(args.runs) not in (2, 3):
        parser.error("--runs takes two or three run files (correction 8)")
    if args.cmd == "curves":
        pairs = [(q, fn, Path(fp), bn, Path(bp), Path(rp)) for q, fn, fp, bn, bp, rp in args.pair] if args.pair else default_pairs(args.split)
        out = curves(args.split, pairs, args.rule or (), args.resamples, SEED, args.guard_dir)
        path = args.out or (ANALYSES / "handoff_curves_validation_s8.json" if args.split == "validation"
                            else RESULTS5 / "handoff_curves.json")
        write_json(path, out)
        print(json.dumps({"written": str(path), "pairs": summary(out)}, indent=1))
    elif args.cmd == "fit-rule":
        out = fit_rule(args.fiorillo_name, args.fiorillo, args.runs, args.large_name, args.large, args.records)
        write_json(args.out, out)
        print(json.dumps({"written": str(args.out), "positives": out["positives"], "coefficients": out["coefficients"],
                          "intercept": out["intercept"], "statement": out["room"]["statement"]}, indent=1))
    else:
        p4 = RESULTS4 / "predictions"
        out = apply_rule(args.rule, args.split, args.fiorillo or p4 / f"s8_ei_{EI}_{args.split}.csv",
                         args.runs or default_runs(args.split), args.large or p4 / f"gemma4_31b_{EI}_{args.split}.csv",
                         args.records or RECORDS_EI / f"{EI}_{args.split}.jsonl", args.resamples, SEED, args.guard_dir)
        out = {"plan": f"{PLAN} (I-35: the frozen rule against the confidence rule)",
               "source": "src/eval/handoff_curves_session9.py apply-rule", "written_utc": now(), **out}
        write_json(args.out, out)
        print(json.dumps({"written": str(args.out), **{r: v["rule_minus_confidence"] for r, v in out["registered_rates"].items()}},
                         indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

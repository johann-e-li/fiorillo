"""Session 9's comparisons and v0.5's bar (docs/prompt_session9.md: COMPARISONS, POWER AND THE BAR, step 15).

The plan's sections, with corrections 2 (K3's two runs) and 3 (K10, swap averaging). Run on the linked shell in step 15,
after calibrate_session9.py apply, trees_session9.py test, reading_checks_session9.py (score for v0.5, then combine) and
handoff_curves_session9.py on test. It refuses to run unless release_candidate.json and testpreds_log.json exist in
--results (session 8's guard): the comparisons read test predictions, which exist only after the freeze. Every comparison
is on the test split except K3's recipes, which are on the 654 clean validation prompts.

Imported from src/eval/compare_session8.py and not changed: the metrics (session 4's, src/eval/metrics.py through
src/eval/compare.py, plus each answer's F1 on EI), the resampling (numpy's default_rng(20260927), 95 percent percentile
intervals; EI resamples articles, integers(0, m, m) over the m articles sorted by PMCID as text, every prompt of a drawn
article kept; PubMedQA and readmission resample items), the paired differences (a difference "system minus reference" uses
the same resamples for both), the bar's intervals (criteria 2 and 3: 10,000 article resamples), the readings and the
cache. Every number has a 95 percent interval from 1,000 resamples, except criteria 2 and 3's (10,000 article resamples),
the costs (derived from timings) and calibration_s9.json's families table (reported as that file holds it); K4's and K7's
numbers are their files', each with its own interval. A difference whose interval includes zero reads "no difference was
detected", and its "excludes" gives the estimate, the interval and what the interval excludes; it is never read as
equivalence. The decision weights of macro_f1_tuned (readmission) are fitted on each system's validation predictions and
held fixed in the resamples, as in session 8.

Systems. v0.5's are read from release_candidate.json (the freeze): its EI specialist (specialists > evidence_effect >
system: s9_ei or s9_ei_wavg, with "_sa" when swap averaging is kept; base_system is the same system without it), the large
model of criterion 3 (reference > criterion_3_large_model: gemma4_31b, or gemma4_31b_long under GEMMA ON THE LONGER
INPUT) and the ECE limit (ece_limits); s7_b3 answers PubMedQA and readmission_pool_v2 (the pool readmission_pool_v2.json
chose) readmission. Calibrated test predictions (session 4's format, id,label,p0,p1,p2) are read from --results'
predictions/ (calibrate_session9.py apply; trees_session9.py test; biogpt_session9.py test, whose raw file is the system
biogpt_large_pqa-raw), --results8's (s8_ei, gemma4_31b, readmission_pool, xgboost_tuned, catboost_tuned), --results7's
(s7_b3, lightgbm_tuned) and --results-public's (the PubMedQA and readmission priors); lightgbm_tuned's refit, which
trees_session9.py writes as lightgbm_tuned in --results' predictions/, is the system lightgbm_tuned_refit here. The EI
prior is v0.4's: the answer shares of session 8's whole training split (--records-ei). Records: session 8's EI test
records (labels, PMCIDs and the README flags), the clean validation records (--records9, K3), and PubMedQA's and
readmission's public records. A system whose file does not exist (s9_ei_wavg not built, gemma4_31b_long not run, tabicl
NOT BUILT, and so on) is listed in not_built with its reason (with the marker's text where not_built.json holds one), every
comparison that needs it is listed in comparisons_not_made, and nothing fails for it.

Comparisons (the section names of comparison.json and comparison.csv):
- K1 (the bar): v0.5's EI specialist minus the prior, and minus the large model of criterion 3, every EI metric; criteria
  1 to 4 as compare_session8.release_bar computes them, with v0.5's specialist in s8_ei's place: (1) its ECE at most the
  limit; (2) and (3) its log loss below the prior's and the large model's, with the 95 percent interval of the difference
  (10,000 article resamples) wholly below zero; (4) its macro-F1 not below the large model's point estimate. If any
  fails: "v0.5 release bar not met: <the reason>".
- K2 (what the license cost, descriptive): s9_b minus s8_ei, and v0.5's specialist minus s8_ei.
- K3 (the recipes, validation, descriptive): each candidate's two-run average, and the combination's, minus the average of
  s9_b_1 and s9_b_2 (correction 2), on the 654 clean validation prompts. An average's probabilities are the mean of its
  runs' softmaxed validation logits (recipe_session9.py's average), calibrated by session 8's rule (temperature or
  temperature_intercepts, the lower mean out-of-fold log loss, a tie to temperature, as calibrate_session7.choose), each
  prompt's out-of-fold probabilities (calibration_family_cv.out_of_fold: 5 folds, 20 repeats, the seed) averaged over the
  20 repeats; 1,000 article resamples; beside recipe_selection.json's decisions. A candidate (or the combination) left
  with one run under the cost rule is compared run against the base run of the same seed. K3_test: on test, v0.5's
  specialist minus s9_b.
- K4 (reading): reading_checks.json (reading_checks_session9.py combine), with its readings, as it is.
- K5 (PubMedQA, descriptive): s7_b3 minus biogpt_large_pqa, its uncalibrated probabilities (biogpt_large_pqa-raw) primary
  and its calibrated ones beside them under the caveat that BioGPT-Large was fine-tuned on the 150 validation items; and
  biogpt_large_pqa minus gemma4_31b, the same two ways.
- K6 (readmission, descriptive): v0.5's pool (readmission_pool_v2) minus session 8's pool (readmission_pool);
  xgboost_codes minus xgboost_tuned; catboost_codes minus catboost_tuned; tabicl minus lightgbm_tuned (session 7's file);
  session 8's readmission metrics.
- K7 (the hand-off): handoff_curves.json (handoff_curves_session9.py on test), summarized as it is.
- K8 (the weight average, descriptive): s9_ei_wavg minus s9_ei, if s9_ei_wavg was built, both calibrated by v0.5's chosen
  family, each on its own clean validation predictions.
- K9 (calibration families, descriptive): calibration_s9.json's table, and v0.5's specialist under its chosen family minus
  under session 8's rule (<specialist>_s8rule).
- K10 (swap averaging, correction 3, descriptive): v0.5's specialist with swap averaging (<base_system>_sa) minus without
  it (<base_system>), each calibrated by v0.5's chosen family on its own clean validation predictions.
- K1_sensitivity: K1 without every test prompt the README flags (group.flag Incorrect, Questionable or Somewhat
  malformed; session 8's correction 13), descriptive, the unflagged prompts' articles resampled; the three systems'
  metrics on those prompts are K1_sensitivity_metrics.
- calibration_slope_intercept, with intervals, for v0.5's specialists: on EI and PubMedQA, each answer against the rest
  (slope: metrics.calibration_slope, the coefficient of a logistic regression of the answer's indicator on the logit of
  its calibrated probability, 1 ideal; intercept: metrics.calibration_intercept, the intercept with that logit as an
  offset, 0 ideal, as session 8 computed the readmission intercept; intercept_with_slope: the intercept of the same
  regression as the slope, 0 ideal); on readmission, session 8's 30-day intercept and slope from the metrics.
- cost_per_1000_questions, as session 8 computed it (handoff_session8.costs with nothing handed off): 1,000 x the forward
  seconds per question / 3,600 x the pod's hourly rate, read from pod_rate.txt's line "hourly_usd <rate>" (--pod-rate,
  default <results>/pod_rate.txt; package does not zip the pod's /root/eddm_s9/pod_rate.txt, so where no copy is here the
  rate is cost_rule.json's pod > hourly_usd, which the cost rule read from that line on the pod). v0.5's EI specialist:
  the forward seconds in testpreds' run summaries (<results>/run_summaries/<system>_test_run_summary.json for its runs or
  s9_ei_wavg; <results>/checks_v05/<system>_swap_run_summary.json for THE READING CHECKS' swap pass) of each of its
  single-pass systems and, with swap averaging, of each one's pass over the exchanged prompts, summed and divided by the
  number of test questions; s7_b3: session 8's testpreds (s7_b3_timing; s7_b3 is unchanged);
  the large model: its timing file's GPU seconds per test item; readmission: not computed (no test run summary times the
  pool's members on the GPU).
- zeros.json: for every system read (with the raw prediction files of the runs, of s9_ei_wavg and of the large model of
  criterion 3), split and question, the rows whose probability of the true answer prints as 0.
- Known contamination: session 8's note, and that EI's test split was read in session 8, so that this is the second time
  the bar is judged on it.

Writes results/fiorillo_v0_5/comparison.json, comparison.csv (compare.py's columns) and zeros.json, JSON with indent 1.
Each block of results is cached in models/fiorillo_v0_5/compare_cache/ under a key that includes the SHA-256 of its input
files and of the records, so that a run the linked shell ends resumes; --max-seconds stops it from starting a new block
after that many seconds (every run computes at least one), and it prints "stopped; run again to resume" and writes
nothing until every block is done. One command, with no subcommands, as compare_session8.py.

Usage: python3 src/eval/compare_session9.py [--max-seconds 150] [--results results/fiorillo_v0_5] [--pod-rate FILE]
           [--results8 ...] [--results7 ...] [--results-public ...] [--records-public ...] [--records-ei ...]
           [--records9 ...] [--cache ...] [--resamples 1000] [--resamples-bar 10000]
"""
import argparse
import collections
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
import calibration_family_cv as CFC  # noqa: E402
import compare as C  # noqa: E402
import compare_session8 as CMP  # noqa: E402
import metrics as M  # noqa: E402
import recipe_session9 as R  # noqa: E402

PLAN = "docs/prompt_session9.md, COMPARISONS"
SOURCE = ("src/eval/compare_session9.py (docs/prompt_session9.md, COMPARISONS and POWER AND THE BAR, with corrections 2 "
          "and 3)")
SEED = CMP.SEED
RESAMPLES = CMP.RESAMPLES
RESAMPLES_BAR = CMP.RESAMPLES_BAR
FOLDS, REPEATS = CFC.FOLDS, CFC.REPEATS
EI, PQA, RW = "evidence_effect", "pubmedqa_answer", "readmit_window"
EVALUATIONS = (EI, PQA, RW)
EI_METRICS = CMP.EI_METRICS
UNRANKED = {"calibration_intercept_30d", "calibration_slope_30d"}
S8_RULE = ("temperature", "temperature_intercepts")
CANDIDATES = ("swap", "long", "lr1", "lr2")
ANSWERS = {EI: tuple(CMP.ANSWERS), PQA: ("yes", "no", "maybe")}
SLOPE_STATS = ("intercept", "slope", "intercept_with_slope")
HANDOFF_CURVES = ("confidence", "random_expected", "oracle", "large_model_confidence")
HANDOFF_AREAS = ("aurc", "e_aurc", "augrc")
LOG_FLOOR = 1e-300  # recipe_session9.py's floor for the logarithm of an average
LONG_TOKENS = 6144
RESULTS = ROOT / "results" / "fiorillo_v0_5"
RESULTS8 = ROOT / "results" / "fiorillo_v0_4"
RESULTS7 = ROOT / "results" / "fiorillo_v0_3"
RESULTS_PUBLIC = ROOT / "results" / "public_v0"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
RECORDS9 = ROOT / "data" / "processed" / "session9"
CACHE = ROOT / "models" / "fiorillo_v0_5" / "compare_cache"
NOT_BUILT_STAGES = {"tabicl": ("tabicl",), "biogpt_large_pqa": ("biogpt",), "biogpt_large_pqa-raw": ("biogpt",),
                    "gemma4_31b_long": ("prepare_large", "smoke_large", "large_validation")}

CONTAMINATION = (CMP.CONTAMINATION + " EI's test split was read in session 8, for v0.4's bar, its comparisons and its checks; "
                 "v0.5's recipe, calibration and pool were chosen on validation only and frozen in release_candidate.json "
                 "before any test prediction of a v0.5 system existed, and this is the second time the bar is judged on this "
                 "split.")
BIOGPT_CAVEAT = ("BioGPT-Large-PubMedQA was fine-tuned on PQA-L's fold 0 training and development sets, which hold all 150 "
                 "of this project's PubMedQA validation items, so its calibration (session 8's rule on those items) was "
                 "fitted on items it was trained on: its uncalibrated probabilities, renormalized over the three answers, are "
                 "the primary result and its calibrated ones are secondary (PUBMEDQA BESIDE BIOGPT-LARGE)")
K3_RULE = ("each candidate's two-run average, and the combination's, minus the average of s9_b_1 and s9_b_2 (correction 2); "
           "an average is the mean of its runs' softmaxed validation logits (recipe_session9.py), calibrated by session 8's "
           "rule (temperature or temperature_intercepts, the lower mean out-of-fold log loss, a tie to temperature), each "
           "prompt's out-of-fold probabilities (calibration_family_cv.out_of_fold: 5 folds, 20 repeats, the seed) averaged "
           "over the repeats; a candidate or combination left with one run is compared run against the base run of the same "
           "seed; 1,000 resamples of the clean validation prompts' articles")
K3_SCORE_NOTE = ("recipe_selection.json's score of a candidate is the mean of its two runs' own scores (correction 2), while "
                 "K3 scores the two runs' probability average, so the two can differ; K3 is descriptive and decides nothing")
SLOPE_METHOD = ("each answer against the rest, on the calibrated test probabilities: slope, metrics.calibration_slope (the "
                "coefficient of a logistic regression of the answer's indicator on the logit of its probability, 1 ideal); "
                "intercept, metrics.calibration_intercept (the intercept with that logit as an offset, 0 ideal, as session 8 "
                "computed the readmission intercept); intercept_with_slope, the intercept of the same regression as the slope "
                "(0 ideal); probabilities clipped to 1e-6 and 1 - 1e-6 before the logit, as metrics.py does")
COST_FORMULA = ("1,000 x the forward seconds per question / 3,600 x the pod's hourly rate (handoff_session8.costs with nothing "
                "handed off); the seconds are the GPU forward passes timed in the test predictions, each job alone on the GPU")
SENSITIVITY_RULE = ("without every test prompt the README flags (group.flag Incorrect, Questionable or Somewhat malformed; "
                    "session 8's correction 13); descriptive")


class Paths:
    """Where the inputs are; every one is an argument with a default relative to the repository root."""

    def __init__(self, results=RESULTS, results8=RESULTS8, results7=RESULTS7, results_public=RESULTS_PUBLIC,
                 records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI, records9=RECORDS9, cache=CACHE, pod_rate=None):
        self.results, self.results8, self.results7 = Path(results), Path(results8), Path(results7)
        self.results_public, self.records_public = Path(results_public), Path(records_public)
        self.records_ei, self.records9, self.cache = Path(records_ei), Path(records9), Path(cache)
        self.pod_rate = Path(pod_rate) if pod_rate else self.results / "pod_rate.txt"


# ---------------------------------------------------------------- small helpers

def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fl(x):
    """A float for JSON; None when missing or not finite."""
    if x is None or isinstance(x, bool):
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def clean(obj):
    """obj with numpy scalars made plain and every float that is not finite replaced by None (comparison.json is written
    with allow_nan False)."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return fl(obj)
    return obj


def file_entry(path):
    return {"file": rel(path), "sha256": CMP.sha256_file(path)}


def wkey(w):
    return CMP.floats(w) if w is not None else None


def guard(results):
    """Session 8's guard: the comparisons read test predictions, which exist only after the freeze and testpreds."""
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; the comparisons read test predictions, made only after "
                             "the freeze")


def excludes_text(metric, est, low, high):
    """What the 95 percent interval of a difference (system minus reference) excludes, in words (COMPARISONS); a difference
    whose interval includes zero is "no difference was detected", never equivalence."""
    if est is None or low is None or high is None:
        return "not computed"
    if low > 0 or high < 0:
        return f"the interval, {low:.4f} to {high:.4f}, lies wholly {'above' if low > 0 else 'below'} zero"
    if low == 0 and high == 0:
        return ("no difference was detected: the difference is 0 in every resample (the two give the same value on these "
                "items)")
    if metric in UNRANKED:
        up, down = "higher", "lower"
    elif metric in CMP.LOWER_IS_BETTER:
        up, down = "higher, that is worse,", "lower, that is better,"
    else:
        up, down = "higher, that is better,", "lower, that is worse,"
    return (f"no difference was detected: the estimate is {est:.4f}, with a 95 percent interval of {low:.4f} to {high:.4f}; "
            f"the interval excludes the system being {up} than the reference by more than {high:.4f}, and {down} by more "
            f"than {abs(low):.4f}; it is not a finding of equivalence")


# ---------------------------------------------------------------- v0.5's systems and their files

def v05_names(rc):
    """v0.5's EI specialist, its two forms and its runs, and the large model of criterion 3, from release_candidate.json."""
    try:
        v = rc["specialists"][EI]
        v05 = str(v["system"])
        large = str(rc["reference"]["criterion_3_large_model"])
    except (KeyError, TypeError) as err:
        raise SystemExit("release_candidate.json does not name v0.5's EI specialist (specialists > evidence_effect > system) "
                         f"and the large model of criterion 3 (reference > criterion_3_large_model): {err!r}")
    x = str(v.get("base_system") or (v05[:-3] if v05.endswith("_sa") else v05))
    runs = [str(r) for r in v.get("runs") or []]
    singles = [str(s) for s in v.get("single_pass_systems") or (["s9_ei_wavg"] if x == "s9_ei_wavg" else runs)]
    return {"v05": v05, "x": x, "with_sa": f"{x}_sa", "swap_averaging": bool(v.get("swap_averaging", v05 != x)),
            "recipe": v.get("recipe"), "runs": runs, "single_pass_systems": singles, "max_len": v.get("max_len"),
            "family": v.get("family"), "large": large}


def system_specs(paths, names):
    """{evaluation: [(system, folder, file stem)]}: every system read on test, in the tables' order (the EI prior, computed
    here, has no folder). A test file is <folder>/<stem>_<evaluation>_test.csv, its validation twin the same with
    _validation."""
    p5, p4 = paths.results / "predictions", paths.results8 / "predictions"
    p3, pp = paths.results7 / "predictions", paths.results_public / "predictions"
    v05, x = names["v05"], names["x"]
    ei = [(v05, p5, v05), (f"{v05}_temperature", p5, f"{v05}_temperature"), (f"{v05}-raw", p5, f"{v05}-raw"),
          (names["with_sa"], p5, names["with_sa"]), (x, p5, x), (f"{v05}_s8rule", p5, f"{v05}_s8rule"),
          ("s9_ei", p5, "s9_ei"), ("s9_ei_wavg", p5, "s9_ei_wavg"), ("s9_b", p5, "s9_b")]
    ei += [(r, p5, r) for r in ["s9_b_1", "s9_b_2", "s9_b_3"] + names["runs"]]
    ei += [("s8_ei", p4, "s8_ei"), ("gemma4_31b", p4, "gemma4_31b"), ("gemma4_31b_long", p5, "gemma4_31b_long"),
           ("prior", None, None)]
    pqa = [("s7_b3", p3, "s7_b3"), ("biogpt_large_pqa-raw", p5, "biogpt_large_pqa-raw"),
           ("biogpt_large_pqa", p5, "biogpt_large_pqa"), ("gemma4_31b", p4, "gemma4_31b"), ("prior", pp, "prior")]
    rw = [("readmission_pool_v2", p5, "readmission_pool_v2"), ("readmission_pool", p4, "readmission_pool"),
          ("xgboost_codes", p5, "xgboost_codes"), ("xgboost_tuned", p4, "xgboost_tuned"),
          ("catboost_codes", p5, "catboost_codes"), ("catboost_tuned", p4, "catboost_tuned"), ("tabicl", p5, "tabicl"),
          ("lightgbm_tuned", p3, "lightgbm_tuned"), ("lightgbm_tuned_refit", p5, "lightgbm_tuned"), ("prior", pp, "prior")]
    out = {}
    for ev, entries in ((EI, ei), (PQA, pqa), (RW, rw)):
        seen, kept = set(), []
        for entry in entries:
            if entry[0] not in seen:
                seen.add(entry[0])
                kept.append(entry)
        out[ev] = kept
    return out


def marker_text(markers, stages):
    """The text not_built.json (package's list of NOT BUILT markers) gives for any of these stages, or None."""
    found = []

    def visit(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in stages:
                    found.append(v if isinstance(v, str) else json.dumps(v, sort_keys=True))
                elif isinstance(v, (dict, list)):
                    visit(v)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict) and item.get("stage") in stages:
                    found.append(str(item.get("reason") or json.dumps(item, sort_keys=True)))
                else:
                    visit(item)

    visit(markers)
    return "; ".join(found) or None


def absent_reason(system, path, names, sel, markers):
    """Why a system's test file is not here."""
    base = f"its prediction file {rel(path)} does not exist"
    if system == "gemma4_31b_long" and names.get("max_len") != LONG_TOKENS:
        return ("not run: v0.5's recipe reads 4,096 tokens, so GEMMA ON THE LONGER INPUT did not apply and gemma4_31b's "
                "session 8 files are used")
    if system == "s9_ei_wavg" and not ((sel or {}).get("weight_average") or {}).get("built"):
        return "not built: recipe_selection.json records no weight average; " + base
    text = marker_text(markers, NOT_BUILT_STAGES.get(system, (system,)))
    return f"not_built.json records: {text}; {base}" if text else base


def records_digests(paths):
    """The SHA-256 of every record file the comparisons read (part of every cache key)."""
    files = {"ei_test": paths.records_ei / f"{EI}_test.jsonl", "ei_train": paths.records_ei / f"{EI}_train.jsonl",
             f"{PQA}_test": paths.records_public / f"{PQA}_test.jsonl", f"{RW}_test": paths.records_public / f"{RW}_test.jsonl",
             f"{RW}_validation": paths.records_public / f"{RW}_validation.jsonl",
             "clean_validation": paths.records9 / "clean" / f"{EI}_validation.jsonl"}
    return {k: (CMP.sha256_file(v) if v.exists() else None) for k, v in files.items()}


def decision_weights(path, ids, labels):
    """Session 4's decision weights (metrics.decision_weights) on a system's readmission validation predictions."""
    weights, f1 = M.decision_weights(labels, CMP.load_aligned(path, ids, labels))
    return {"weights": [float(w) for w in weights], "validation_macro_f1": fl(f1)}


def subset_dataset(ds, rows, name):
    """The rows of a test dataset as a dataset of their own (session 8's subset: its articles resampled, every prompt of a
    drawn article kept)."""
    rows = np.asarray(rows, dtype=np.int64)
    clusters = [ds["clusters"][i] for i in rows]
    return {"name": f"{ds['name']}_{name}", "split": ds["split"], "group": name, "ids": [ds["ids"][i] for i in rows],
            "labels": ds["labels"][rows], "clusters": clusters, "rows_of": CMP.article_rows(clusters),
            "probs": {s: p[rows] for s, p in ds["probs"].items()}, "weights": {}, "digests": dict(ds["digests"]),
            "absent": dict(ds["absent"])}


# ---------------------------------------------------------------- K3: the recipes on validation

def average_runs(results, runs, ids, labels):
    """The mean of the runs' softmaxed validation logits in double precision (recipe_session9.py's average), rows in the
    order of ids; the logits must hold every one of ids with the same labels."""
    out = []
    for run in runs:
        r_ids, r_labels, p = R.run_probs(results, run, "validation")
        pos = {i: k for k, i in enumerate(r_ids)}
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit(f"{run}'s validation logits lack {len(missing)} of the clean prompts, for example {missing[:3]}")
        order = np.array([pos[i] for i in ids])
        if not np.array_equal(np.asarray(r_labels)[order], np.asarray(labels)):
            raise SystemExit(f"{run}'s validation labels differ from the clean records")
        out.append(np.asarray(p, dtype=np.float64)[order])
    return np.mean(out, axis=0)


def oof_session8_rule(logp, y, repeats=REPEATS, folds=FOLDS, seed=SEED):
    """Session 8's rule out of fold: both families' out-of-fold scores and probabilities (calibration_family_cv.out_of_fold,
    whose folds are those of calibrate_session7.choose), the family with the lower mean out-of-fold log loss (a tie to
    temperature, as calibrate_session7.choose decides), and each item's out-of-fold probabilities of that family averaged
    over the repeats. Returns (family, {family: summary}, probabilities)."""
    summaries, probs = {}, {}
    for family in S8_RULE:
        summaries[family], probs[family] = CFC.out_of_fold(family, logp, y, folds, repeats, seed)
    a = summaries["temperature"]["out_of_fold_log_loss"]["mean"]
    b = summaries["temperature_intercepts"]["out_of_fold_log_loss"]["mean"]
    chosen = "temperature_intercepts" if b < a else "temperature"
    return chosen, summaries, probs[chosen].mean(axis=0)


def k3_block(results, runs, ids, labels, repeats=REPEATS):
    """One K3 system: its runs' average on the clean prompts, then session 8's rule out of fold (a cache block)."""
    avg = average_runs(results, runs, ids, labels)
    chosen, summaries, oof = oof_session8_rule(np.log(np.maximum(avg, LOG_FLOOR)), np.asarray(labels), repeats)
    return {"chosen": chosen, "families": clean(summaries), "probs": [[float(v) for v in row] for row in oof]}


def k3_plan(sel):
    """K3's pairs from recipe_selection.json: [(system, runs, reference, reference runs, how)], and the candidates it cannot
    compare, with the reason."""
    items, skipped = [], []

    def item(label, runs):
        if len(runs) >= 2:
            return (f"s9_{label}_1_2", list(runs[:2]), "s9_b_1_2", ["s9_b_1", "s9_b_2"],
                    "the average of its two runs minus the average of s9_b_1 and s9_b_2 (correction 2)")
        k = runs[0].rsplit("_", 1)[-1]
        return (runs[0], [runs[0]], f"s9_b_{k}", [f"s9_b_{k}"],
                f"one run (its second was dropped under the cost rule) minus s9_b_{k}, the base run of the same seed")

    cand = sel.get("candidates") or {}
    for c in CANDIDATES:
        info = cand.get(c) or {}
        runs = [str(r) for r in info.get("runs") or []]
        if runs:
            items.append(item(c, runs))
        else:
            skipped.append({"candidate": c, "reason": str(info.get("status") or "recipe_selection.json lists no run of it")})
    combo = sel.get("combination") or {}
    runs = [str(r) for r in combo.get("runs") or []]
    if runs:
        items.append(item("combo", runs))
    else:
        skipped.append({"candidate": "combo", "reason": str(combo.get("status") or (
            "called, but recipe_selection.json lists no run of it" if combo.get("called")
            else "not called: fewer than two of the three changes passed"))})
    return items, skipped


def selection_extract(sel):
    """recipe_selection.json's decisions, beside K3 (the run settings left out)."""
    keep = ("margin", "s9_b", "candidates", "learning_rate", "passing_changes", "combination", "winner", "s9_ei",
            "weight_average", "swap_averaging", "v05_ei", "decisions", "dropped")
    out = {k: sel.get(k) for k in keep if k in sel}
    for k in ("combination", "winner"):
        if isinstance(out.get(k), dict):
            out[k] = {kk: v for kk, v in out[k].items() if kk != "settings"}
    return out


# ---------------------------------------------------------------- calibration slope and intercept

def slope_fit(y, p):
    """(slope, intercept_with_slope): the logistic regression of y on logit(p), fitted as metrics.calibration_slope fits it."""
    y = np.asarray(y, dtype=float)
    lp = M._logit(p)
    if y.min() == y.max() or np.ptp(lp) < 1e-9:
        return M.NAN, M.NAN
    beta = M._logistic_fit(np.column_stack([np.ones_like(lp), lp]), y, np.zeros_like(lp))
    return float(beta[1]), float(beta[0])


def one_against_rest(labels, probs):
    """For each answer j in order, of (labels == j) against probs[:, j]: intercept (metrics.calibration_intercept), slope
    (metrics.calibration_slope) and intercept_with_slope; 3 numbers an answer."""
    labels, probs = np.asarray(labels), np.asarray(probs, dtype=float)
    out = []
    with np.errstate(over="ignore", invalid="ignore"):  # a fit that diverges on a small resample gives nan, left out
        for j in range(probs.shape[1]):
            y = (labels == j).astype(float)
            slope, joint = slope_fit(y, probs[:, j])
            out += [M.calibration_intercept(y, probs[:, j]), slope, joint]
    return np.array(out, dtype=float)


# ---------------------------------------------------------------- the cost per 1,000 questions

def hourly_rate(paths):
    """(the pod's hourly rate in dollars, where it was read): pod_rate.txt's line "hourly_usd <rate>"; if that file is not
    here, cost_rule.json's pod > hourly_usd, which the cost rule read from that line on the pod."""
    path = paths.pod_rate
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "hourly_usd" and fl(parts[1]) is not None:
                return fl(parts[1]), f"{rel(path)}, its line 'hourly_usd <rate>'"
        return None, f"{rel(path)} has no line 'hourly_usd <rate>'"
    cost_rule = paths.results / "cost_rule.json"
    d = load_json(cost_rule, {})
    v = fl(((d if isinstance(d, dict) else {}).get("pod") or {}).get("hourly_usd"))
    if v is not None:
        return v, (f"{rel(cost_rule)}, pod > hourly_usd (the cost rule read it from the pod's pod_rate.txt line 'hourly_usd "
                   f"<rate>'; {rel(path)} is not here)")
    return None, f"neither {rel(path)} nor {rel(cost_rule)} gives the pod's hourly rate"


def run_summary_path(results, system, split="test"):
    """A test run summary as testpreds writes it (scripts/pod_session9.sh): a run's or s9_ei_wavg's
    <results>/run_summaries/<system>_<split>_run_summary.json; a reading-check pass's
    <results>/checks_v05/<system>_run_summary.json (<system> is <single-pass system>_<input>); or the first of those names
    under <results>/checks_v05/ or <results>/checks_v05/run_summaries/."""
    results = Path(results)
    for path in (results / "run_summaries" / f"{system}_{split}_run_summary.json",
                 results / "checks_v05" / f"{system}_run_summary.json",
                 results / "checks_v05" / f"{system}_{split}_run_summary.json",
                 results / "checks_v05" / "run_summaries" / f"{system}_{split}_run_summary.json"):
        if path.exists():
            return path
    return None


def usd_per_1000(seconds_per_question, rate):
    if seconds_per_question is None or rate is None:
        return None
    return 1000 * seconds_per_question / 3600 * rate


def timing_seconds(path, question):
    """A large model's GPU seconds per test item (its timing file, as handoff_session8.seconds_per_item reads it)."""
    d = load_json(path, {})
    return fl((((d if isinstance(d, dict) else {}).get(question) or {}).get("test") or {}).get("gpu_seconds_per_item"))


def cost_section(paths, names, n_questions):
    """The cost per 1,000 questions of v0.5's specialists and the large model (COMPARISONS, as session 8 computed it)."""
    rate, source = hourly_rate(paths)
    out = {"hourly_usd": rate, "rate_source": source, "formula": COST_FORMULA, "specialists": {}, "large_model": {}}
    passes = [(s, "the test prompts") for s in names["single_pass_systems"]]
    if names["swap_averaging"]:
        passes += [(f"{s}_swap", "the exchanged test prompts (THE READING CHECKS' swap input; correction 3)")
                   for s in names["single_pass_systems"]]
    rows, missing, total = [], [], 0.0
    for system, what in passes:
        path = run_summary_path(paths.results, system)
        d = load_json(path, {}) if path else {}
        timing = ((d if isinstance(d, dict) else {}).get("timing_test") or {})
        seconds = fl(timing.get("forward_seconds"))
        if seconds is None:
            missing.append(system)
            continue
        rows.append({"system": system, "input": what, "run_summary": rel(path), "items": timing.get("items"),
                     "forward_seconds": seconds, "seconds_per_question": seconds / n_questions})
        total += seconds / n_questions
    ei = {"system": names["v05"], "questions": int(n_questions), "passes": rows,
          "note": ("its forward seconds summed over its passes and divided by the number of test questions; for a pass over "
                   "every test prompt that is its seconds per item, as session 8 summed s8_ei's three runs'")}
    if not passes:
        ei.update({"seconds_per_question": None, "usd": None,
                   "not_computed": "release_candidate.json lists no single-pass system of v0.5's EI specialist"})
    elif missing:
        ei.update({"seconds_per_question": None, "usd": None,
                   "not_computed": f"no test run summary with forward seconds for {', '.join(missing)}"})
    else:
        ei.update({"seconds_per_question": total, "usd": usd_per_1000(total, rate)})
    out["specialists"][EI] = ei
    path = paths.results8 / "run_summaries" / "s7_b3_timing_test_run_summary.json"
    d = load_json(path, {})
    s = fl((((d if isinstance(d, dict) else {}).get("timing_test")) or {}).get("seconds_per_item"))
    pq = {"system": "s7_b3", "run_summary": rel(path), "seconds_per_question": s, "usd": usd_per_1000(s, rate),
          "note": ("s7_b3 is unchanged and session 9's test predictions do not run it, so its forward seconds are session 8's "
                   "testpreds' (s7_b3_timing, alone on the GPU, the same GPU type), at this session's rate")}
    if s is None:
        pq["not_computed"] = f"{rel(path)} gives no test seconds per item"
    out["specialists"][PQA] = pq
    out["specialists"][RW] = {"system": "readmission_pool_v2", "seconds_per_question": None, "usd": None,
                              "not_computed": ("its members are trees on the CPU and readers predicted in earlier sessions; no "
                                               "test run summary of session 9 times them on the GPU")}
    large = names["large"]
    files = {EI: (paths.results / "large_model" / "gemma4_31b_long_timing.json" if large == "gemma4_31b_long"
                  else paths.results8 / "large_model" / "gemma4_31b_timing.json"),
             PQA: paths.results8 / "large_model" / "gemma4_31b_timing.json"}
    for ev, system in ((EI, large), (PQA, "gemma4_31b")):
        s = timing_seconds(files[ev], ev)
        entry = {"system": system, "timing_file": rel(files[ev]), "seconds_per_question": s, "usd": usd_per_1000(s, rate)}
        if s is None:
            entry["not_computed"] = f"{rel(files[ev])} gives no test GPU seconds per item for {ev}"
        if system == "gemma4_31b":
            entry["note"] = "gemma4_31b's seconds are session 8's testpreds' (alone on the GPU), at this session's rate"
        out["large_model"][ev] = entry
    return out


# ---------------------------------------------------------------- zeros.json

def count_zeros(path):
    ids, labels, probs = M.read_predictions(path)
    return {"rows": len(ids), "true_answer_printed_zero": int((probs[np.arange(len(labels)), labels] == 0).sum())}


def zeros_table(specs, extra=()):
    """For every system read, split and question, the rows whose probability of the true answer prints as 0; extra is
    [(system, question, split, path)] for files read outside the specs (K3's runs' raw validation predictions)."""
    zeros = {}
    for ev, entries in specs.items():
        for system, folder, stem in entries:
            if folder is None:
                continue
            for split in ("validation", "test"):
                path = Path(folder) / f"{stem}_{ev}_{split}.csv"
                if path.exists():
                    zeros.setdefault(system, {}).setdefault(split, {})[ev] = count_zeros(path)
    for system, ev, split, path in extra:
        if Path(path).exists():
            zeros.setdefault(system, {}).setdefault(split, {})[ev] = count_zeros(path)
    return zeros


# ---------------------------------------------------------------- the bar

def release_bar(out, rc, v05, large):
    """Criteria 1 to 4 as compare_session8.release_bar computes them, with v0.5's EI specialist in s8_ei's place and the ECE
    limit and the large model of criterion 3 read from release_candidate.json."""
    crit = {}
    m = out["metrics"].get(EI, {})
    n_bar = f"{int((out.get('settings') or {}).get('resamples_bar') or RESAMPLES_BAR):,}"
    limit = fl(((rc.get("ece_limits") or {}).get(EI) or {}).get("limit"))
    reasons = []
    if v05 not in m:
        reasons.append(f"{v05}'s test predictions are missing")
    if limit is None:
        reasons.append("release_candidate.json holds no ECE limit for evidence_effect")
    ece = (m.get(v05, {}).get("ece") or {}).get("estimate")
    ok = not reasons and ece is not None and ece <= limit
    if not reasons and not ok:
        reasons.append(f"its ECE, {ece:.4f}, is above the limit, {limit:.4f}" if ece is not None else "its ECE was not computed")
    crit["1"] = {"rule": (f"{v05}'s ECE at most the limit (the larger of 0.05 and the 95th percentile of a calibrated model's "
                          "ECE at n 1,218, fixed in release_candidate.json before any test prediction)"),
                 "values": {"ece": ece, "limit": limit}, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    for k, key, ref, rule in (
            ("2", "v05_minus_prior", "the prior",
             f"{v05}'s log loss below the prior's (v0.4's prior, the answer shares of session 8's whole training split), with "
             f"the 95 percent interval of the difference ({v05} minus the prior; {n_bar} article resamples) wholly below zero"),
            ("3", "v05_minus_large_model", large,
             f"{v05}'s log loss below {large}'s (both calibrated on validation), with the 95 percent interval of the "
             f"difference ({n_bar} article resamples) wholly below zero")):
        iv = out["bar_intervals"].get(key)
        reasons = [] if iv else [f"{v05}'s or {ref}'s test predictions are missing"]
        ok = bool(iv) and iv["estimate"] < 0 and iv["high"] < 0
        if iv and not ok:
            reasons.append(f"{v05}'s log loss minus {ref}'s is {iv['estimate']:.4f}, with a 95 percent interval of "
                           f"{iv['low']:.4f} to {iv['high']:.4f}, which is not wholly below zero")
        crit[k] = {"rule": rule, "values": iv, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    a = (m.get(v05, {}).get("macro_f1") or {}).get("estimate")
    b = (m.get(large, {}).get("macro_f1") or {}).get("estimate")
    reasons = [] if (a is not None and b is not None) else [f"{v05}'s or {large}'s test predictions are missing"]
    ok = not reasons and a >= b
    if not reasons and not ok:
        reasons.append(f"its macro-F1, {a:.4f}, is below {large}'s, {b:.4f}")
    crit["4"] = {"rule": f"{v05}'s macro-F1 not below {large}'s point estimate", "values": {"specialist": a, "large_model": b},
                 "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    overall = "PASS" if all(c["result"] == "PASS" for c in crit.values()) else "FAIL"
    failed = [k for k, c in crit.items() if c["result"] != "PASS"]
    outcome = ("every criterion passes: v0.5's bar is met, and THE RELEASE follows (step 17)" if overall == "PASS" else
               "v0.5 release bar not met: " + "; ".join(f"criterion {k} ({crit[k]['reason']})" for k in failed))
    return {"system": f"{v05} on evidence_effect (v0.5's EI specialist)", "specialist": v05, "large_model": large,
            "criteria": crit, "overall": overall, "outcome": outcome,
            "note": "EI's test split was read in session 8; this is the second time the bar is judged on it"}


# ---------------------------------------------------------------- the run

class Study:
    """One run of the comparisons: the data, the cache, comparison.json's content and comparison.csv's rows."""

    def __init__(self, paths, resamples=RESAMPLES, resamples_bar=RESAMPLES_BAR, max_seconds=None, k3_repeats=REPEATS):
        self.paths, self.resamples, self.resamples_bar, self.k3_repeats = paths, resamples, resamples_bar, k3_repeats
        self.t0 = time.time()
        self.cache = CMP.Cache(paths.cache, None if max_seconds is None else self.t0 + max_seconds)
        self.table, self.not_built, self.not_made, self.comparisons = [], [], [], []
        self.waiting = set()  # (evaluation, system) whose decision weights wait for the next run (the cache stopped)
        self.records_sha = records_digests(paths)

    # ------------------------------------------------------------ resampling, metrics and differences

    def resample(self, ds, stat, rows=None):
        """compare_session8's resampling: articles for EI (cluster_bootstrap), items for the others (metrics.bootstrap)."""
        if ds["clusters"] is not None:
            if rows is None:
                return CMP.cluster_bootstrap(stat, ds["rows_of"], len(ds["labels"]), self.resamples, SEED)
            return CMP.cluster_bootstrap(stat, CMP.article_rows([ds["clusters"][i] for i in rows]), len(rows), self.resamples,
                                         SEED)
        return M.bootstrap(stat, len(ds["labels"]) if rows is None else len(rows), self.resamples, SEED)

    def metrics_block(self, ds, ev, system, names, section="metrics"):
        probs, labels, w = ds["probs"][system], ds["labels"], ds["weights"].get(system)
        key = ["metrics", ds["name"], ev, system, ds["digests"][system], list(names), self.resamples, SEED, wkey(w),
               self.records_sha]
        res = self.cache.get(key, lambda: [CMP.floats(v) for v in self.resample(
            ds, lambda idx: CMP.scores(names, labels[idx], probs[idx], w))])
        if res is None:
            return None
        est, low, high = res
        for j, m in enumerate(names):
            self.table.append({"section": section, "evaluation": ev, "system": system, "group": ds["group"], "metric": m,
                               "estimate": est[j], "low": low[j], "high": high[j], "n": int(len(labels))})
        return {"n": int(len(labels)), **{m: {"estimate": est[j], "low": low[j], "high": high[j]} for j, m in enumerate(names)}}

    def compare(self, section, ds, ev, system, reference, names, note="", subset=None, subset_name=None):
        """The paired difference system minus reference on every metric in names, with its reading and what its interval
        excludes; a comparison whose system or reference is not built is listed in comparisons_not_made."""
        absent = [s for s in (system, reference) if s not in ds["probs"]]
        if absent:
            if self.cache.stopped and not any(s in ds["absent"] for s in absent):
                return None  # still to be computed in the next run
            self.not_made.append({"section": section, "evaluation": ev, "split": ds["split"], "system": system,
                                  "reference": reference,
                                  "reason": "; ".join(f"{s}: {ds['absent'].get(s, 'not built')}" for s in absent)})
            return None
        if (ev, system) in self.waiting or (ev, reference) in self.waiting:
            return None
        rows = np.arange(len(ds["labels"])) if subset is None else np.asarray(subset)
        labels, a, b = ds["labels"][rows], ds["probs"][system][rows], ds["probs"][reference][rows]
        wa, wb = ds["weights"].get(system), ds["weights"].get(reference)
        key = ["diff", ds["name"], ev, system, reference, ds["digests"][system], ds["digests"][reference], list(names),
               self.resamples, SEED, wkey(wa), wkey(wb), self.records_sha, subset_name]
        res = self.cache.get(key, lambda: [CMP.floats(v) for v in self.resample(
            ds, lambda idx: CMP.scores(names, labels[idx], a[idx], wa) - CMP.scores(names, labels[idx], b[idx], wb),
            None if subset is None else rows)])
        if res is None:
            return None
        est, low, high = res
        sys_est, ref_est = CMP.floats(CMP.scores(names, labels, a, wa)), CMP.floats(CMP.scores(names, labels, b, wb))
        records = []
        for j, m in enumerate(names):
            reading = CMP.reading(m, low[j], high[j])
            record = {"section": section, "evaluation": ev, "split": ds["split"], "system": system, "reference": reference,
                      "metric": m, "difference": {"estimate": est[j], "low": low[j], "high": high[j]}, "reading": reading,
                      "excludes": excludes_text(m, est[j], low[j], high[j]), "ranked": m not in UNRANKED,
                      "n": int(len(labels)), "system_estimate": sys_est[j], "reference_estimate": ref_est[j], "note": note}
            if subset_name:
                record["subset"] = subset_name
            self.comparisons.append(record)
            text = record["excludes"] if reading == "no difference was detected" else reading
            if m in UNRANKED:
                text += "; not ranked (an intercept is best at 0 and a slope at 1)"
            self.table.append({"section": section, "evaluation": ev, "system": system, "reference": reference,
                               "group": subset_name or ds["group"], "metric": m, "estimate": est[j], "low": low[j],
                               "high": high[j], "n": int(len(labels)), "system_estimate": sys_est[j],
                               "reference_estimate": ref_est[j], "note": f"{text}; {note}" if note else text})
            records.append(record)
        return records

    # ------------------------------------------------------------ the data

    def load_test(self, specs, names, sel, markers):
        """Each question's test records and every system's calibrated test probabilities, in the records' order."""
        p = self.paths
        data, files = {}, {}
        ei_rows = read_jsonl(p.records_ei / f"{EI}_test.jsonl")
        shares, counts = CMP.ei_prior(p.records_ei)
        self.prior = {"counts": [int(c) for c in counts], "shares": CMP.floats(shares),
                      "source": (f"{rel(p.records_ei / (EI + '_train.jsonl'))}: v0.4's prior, the answer shares of session 8's "
                                 "whole training split")}
        for ev in EVALUATIONS:
            rows = ei_rows if ev == EI else C.reference_rows(p.records_public, ev)
            ds = {"name": f"test_{ev}", "split": "test", "group": "", "ids": [r["id"] for r in rows],
                  "labels": np.array([r["label"] for r in rows]), "clusters": None, "probs": {}, "weights": {},
                  "digests": {}, "absent": {}}
            if ev == EI:
                ds["clusters"] = [r["group"]["pmcid"] for r in rows]
                ds["flags"] = [r["group"].get("flag") or "none" for r in rows]
                ds["rows_of"] = CMP.article_rows(ds["clusters"])
            for system, folder, stem in specs[ev]:
                if system == "prior" and ev == EI:
                    ds["probs"]["prior"] = np.tile(shares, (len(rows), 1))
                    ds["digests"]["prior"] = CMP.sha256_text(json.dumps(counts))
                    continue
                path = Path(folder) / f"{stem}_{ev}_test.csv"
                if path.exists():
                    ds["probs"][system] = CMP.load_aligned(path, ds["ids"], ds["labels"])
                    ds["digests"][system] = CMP.sha256_file(path)
                    files.setdefault(ev, {})[system] = {"file": rel(path), "sha256": ds["digests"][system]}
                else:
                    reason = absent_reason(system, path, names, sel, markers)
                    ds["absent"][system] = reason
                    self.not_built.append({"evaluation": ev, "split": "test", "system": system, "file": rel(path),
                                           "reason": reason})
            data[ev] = ds
        return data, files

    def fit_weights(self, ds, specs):
        """macro_f1_tuned's decision weights for each readmission system, from its validation predictions (session 8)."""
        val_rows = C.reference_rows(self.paths.records_public, RW, "validation")
        val_ids, val_labels = [r["id"] for r in val_rows], np.array([r["label"] for r in val_rows])
        fitted = {}
        for system, folder, stem in specs[RW]:
            if system not in ds["probs"]:
                continue
            path = Path(folder) / f"{stem}_{RW}_validation.csv"
            if not path.exists():
                fitted[system] = {"not_computed": f"{rel(path)} does not exist, so macro_f1_tuned is not computed"}
                continue
            sha = CMP.sha256_file(path)
            res = self.cache.get(["weights", system, sha, self.records_sha[f"{RW}_validation"]],
                                 lambda: decision_weights(path, val_ids, val_labels))
            if res is None:
                self.waiting.add((RW, system))
                continue
            ds["weights"][system] = np.array(res["weights"], dtype=float)
            fitted[system] = {**res, "file": rel(path), "sha256": sha}
        return fitted

    # ------------------------------------------------------------ K3, K4 and K7

    def k3(self, sel):
        """K3 on the clean validation prompts, beside recipe_selection.json's decisions."""
        p = self.paths
        sec = {"split": "validation", "items": "the 654 clean validation prompts (data/processed/session9/clean/)",
               "rule": K3_RULE, "note": K3_SCORE_NOTE, "systems": {}, "metrics": {}, "not_made": []}
        if not isinstance(sel, dict):
            sec["not_built"] = f"{rel(p.results / 'recipe_selection.json')} does not exist or cannot be read"
            return sec, []
        sec["recipe_selection"] = {**file_entry(p.results / "recipe_selection.json"), **selection_extract(sel)}
        cand, combo = sel.get("candidates") or {}, sel.get("combination") or {}
        passing = [c for c in CANDIDATES if (cand.get(c) or {}).get("passes")]
        combo_state = ("not called" if not combo.get("called") else
                       "called and passed" if combo.get("passes") else "called, did not pass")
        self.table.append({"section": "K3", "evaluation": EI, "group": "validation", "metric": "recipe_selection",
                           "note": (f"recipe_selection.json's decisions: passing candidates {', '.join(passing) or 'none'}; "
                                    f"the combination {combo_state}; the winning recipe "
                                    f"{(sel.get('winner') or {}).get('recipe')}; the weight average used "
                                    f"{(sel.get('weight_average') or {}).get('uses_wavg')}; swap averaging kept "
                                    f"{(sel.get('swap_averaging') or {}).get('kept')}")})
        clean_path = p.records9 / "clean" / f"{EI}_validation.jsonl"
        if not clean_path.exists():
            sec["not_built"] = f"{rel(clean_path)} does not exist"
            return sec, []
        rows = read_jsonl(clean_path)
        ds = {"name": "clean_validation", "split": "validation", "group": "clean validation, out of fold",
              "ids": [r["id"] for r in rows], "labels": np.array([r["label"] for r in rows]),
              "clusters": [r["group"]["pmcid"] for r in rows], "probs": {}, "weights": {}, "digests": {}, "absent": {}}
        ds["rows_of"] = CMP.article_rows(ds["clusters"])
        sec["n"], sec["articles"] = len(rows), len(ds["rows_of"])
        plan, skipped = k3_plan(sel)
        sec["not_made"] = skipped
        systems = {}
        for name, runs, ref, ref_runs, _ in plan:
            systems.setdefault(ref, ref_runs)
            systems.setdefault(name, runs)
        used_runs = sorted({r for runs in systems.values() for r in runs})
        for name, runs in systems.items():
            files = [R.logits_path(p.results, r, "validation") for r in runs]
            absent = [r for r, f in zip(runs, files) if not f.exists()]
            if absent:
                ds["absent"][name] = f"validation logits missing for {', '.join(absent)}"
                sec["systems"][name] = {"runs": runs, "not_built": ds["absent"][name]}
                continue
            key = ["k3_oof", name, runs, [CMP.sha256_file(f) for f in files], self.records_sha.get("clean_validation"), FOLDS,
                   self.k3_repeats, SEED]
            res = self.cache.get(key, lambda: k3_block(p.results, runs, ds["ids"], ds["labels"], self.k3_repeats))
            if res is None:
                continue
            ds["probs"][name] = np.asarray(res["probs"], dtype=float)
            ds["digests"][name] = CMP.sha256_text(json.dumps(key, sort_keys=True))
            sec["systems"][name] = {"runs": runs, "probabilities": ("the mean of its runs' softmaxed logits" if len(runs) > 1
                                                                    else "its run's softmaxed logits"),
                                    "session8_rule_family": res["chosen"], "out_of_fold": res["families"]}
            entry = self.metrics_block(ds, EI, name, EI_METRICS, section="K3_metrics")
            if entry is not None:
                sec["metrics"][name] = entry
        for name, runs, ref, ref_runs, how in plan:
            self.compare("K3", ds, EI, name, ref, EI_METRICS, f"{name} minus {ref}: {how}; validation, out of fold, descriptive")
        return sec, used_runs

    def k4(self):
        """K4: reading_checks.json as reading_checks_session9.py combine wrote it."""
        path = self.paths.results / "reading_checks.json"
        if not path.exists():
            return {"not_built": f"{rel(path)} does not exist (reading_checks_session9.py combine writes it in step 15)"}
        d = load_json(path, {}) or {}
        for row in d.get("table") or []:
            system = row.get("name") or row.get("system")
            if row.get("statistic") == "reading":
                self.table.append({"section": "K4", "evaluation": EI, "system": system, "group": row.get("input"),
                                   "metric": "reading",
                                   "note": "; ".join([str(row.get("reading"))] + [str(c) for c in row.get("caveats") or []])})
            else:
                self.table.append({"section": "K4", "evaluation": EI, "system": system, "group": row.get("input"),
                                   "metric": row.get("statistic"), "estimate": fl(row.get("estimate")), "low": fl(row.get("low")),
                                   "high": fl(row.get("high")), "n": row.get("n")})
        return {**file_entry(path), "written_utc": d.get("written_utc"), "readings": d.get("readings"), "table": d.get("table"),
                "references": d.get("references"), "notes": d.get("notes"),
                "same_none_and_title_records": d.get("same_none_and_title_records"),
                "note": ("THE READING CHECKS as reading_checks_session9.py scored them (each number with a 95 percent interval "
                         "from 1,000 article resamples), with their readings, which gate no release")}

    def k7(self):
        """K7: handoff_curves.json as handoff_curves_session9.py wrote it on test, summarized."""
        path = self.paths.results / "handoff_curves.json"
        if not path.exists():
            return {"not_built": f"{rel(path)} does not exist (handoff_curves_session9.py curves --split test writes it)"}
        d = load_json(path, {}) or {}
        pairs = []
        for pair in d.get("pairs") or []:
            q, fio, big, n = pair.get("question"), pair.get("fiorillo"), pair.get("large_model"), pair.get("n")
            entry = {k: pair.get(k) for k in ("question", "fiorillo", "large_model", "n", "articles", "resampling",
                                               "fiorillo_alone", "large_model_alone", "areas", "registered_rates", "room")}
            for area in HANDOFF_AREAS:
                v = (pair.get("areas") or {}).get(area) or {}
                self.table.append({"section": "K7", "evaluation": q, "system": fio, "reference": big, "group": "areas",
                                   "metric": area, "estimate": fl(v.get("estimate")), "low": fl(v.get("low")),
                                   "high": fl(v.get("high")), "n": n})
            for rate, cells in (pair.get("registered_rates") or {}).items():
                for curve in HANDOFF_CURVES:
                    v = (cells or {}).get(curve) or {}
                    self.table.append({"section": "K7", "evaluation": q, "system": fio, "reference": big, "group": rate,
                                       "metric": f"{curve}_combined_accuracy", "estimate": fl(v.get("estimate")),
                                       "low": fl(v.get("low")), "high": fl(v.get("high")), "n": n})
            rule = pair.get("i35_rule")
            if rule:
                entry["i35_rule"] = {k: rule.get(k) for k in ("split", "system", "large_model", "n", "articles", "note",
                                                              "registered_rates", "room_on_validation")}
                for rate, cells in (rule.get("registered_rates") or {}).items():
                    v = (cells or {}).get("rule_minus_confidence") or {}
                    self.table.append({"section": "K7", "evaluation": q, "system": fio, "reference": big, "group": rate,
                                       "metric": "i35_rule_minus_confidence_accuracy", "estimate": fl(v.get("estimate")),
                                       "low": fl(v.get("low")), "high": fl(v.get("high")), "n": rule.get("n"),
                                       "note": "; ".join(str(v[k]) for k in ("reading", "excludes") if v.get(k))})
            pairs.append(entry)
        return {**file_entry(path), "split": d.get("split"), "written_utc": d.get("written_utc"), "pairs": pairs,
                "note": "HAND-OFF ANALYSES (I-34 and I-35; secondary and descriptive), as handoff_curves_session9.py wrote them"}

    # ------------------------------------------------------------ calibration slope and intercept

    def slopes(self, data, metrics, names):
        sec = {"method": SLOPE_METHOD, "resamples": self.resamples, "seed": SEED}
        for ev, system in ((EI, names["v05"]), (PQA, "s7_b3")):
            ds = data[ev]
            if system not in ds["probs"]:
                sec[ev] = {"system": system, "not_built": ds["absent"].get(system, "not built")}
                continue
            probs, labels = ds["probs"][system], ds["labels"]
            key = ["slopes", ds["name"], ev, system, ds["digests"][system], self.resamples, SEED, self.records_sha]
            res = self.cache.get(key, lambda: [CMP.floats(v) for v in self.resample(
                ds, lambda idx: one_against_rest(labels[idx], probs[idx]))])
            if res is None:
                continue
            est, low, high = res
            answers = {}
            for j, answer in enumerate(ANSWERS[ev]):
                cells = {}
                for s, stat in enumerate(SLOPE_STATS):
                    k = len(SLOPE_STATS) * j + s
                    cells[stat] = {"estimate": est[k], "low": low[k], "high": high[k]}
                    self.table.append({"section": "calibration_slope_intercept", "evaluation": ev, "system": system,
                                       "group": answer, "metric": stat, "estimate": est[k], "low": low[k], "high": high[k],
                                       "n": int(len(labels)), "note": f"{answer} against the rest"})
                answers[answer] = {**cells, "rows_with_this_answer": int((labels == j).sum())}
            sec[ev] = {"system": system, "n": int(len(labels)), "resampling": "articles" if ev == EI else "items",
                       "answers": answers}
        m = metrics.get(RW, {}).get("readmission_pool_v2")
        if m:
            sec[RW] = {"system": "readmission_pool_v2", "intercept_30d": m.get("calibration_intercept_30d"),
                       "slope_30d": m.get("calibration_slope_30d"),
                       "note": ("session 8's calibration intercept and slope for readmission within 30 days (the probability "
                                "of level 2), from the metrics")}
        else:
            sec[RW] = {"system": "readmission_pool_v2",
                       "not_built": data[RW]["absent"].get("readmission_pool_v2", "its metrics were not computed")}
        return sec

    # ------------------------------------------------------------ everything

    def run(self):
        p = self.paths
        guard(p.results)
        rc = load_json(p.results / "release_candidate.json")
        if not isinstance(rc, dict):
            raise SystemExit(f"{rel(p.results / 'release_candidate.json')} cannot be read")
        names = v05_names(rc)
        v05, x, large = names["v05"], names["x"], names["large"]
        sel = load_json(p.results / "recipe_selection.json")
        cal9 = load_json(p.results / "calibration_s9.json", {})
        cal9 = cal9 if isinstance(cal9, dict) else {}
        for name, named in (("calibration_s9.json", (cal9.get("v05_ei") or {}).get("system")),
                            ("recipe_selection.json", ((sel or {}).get("v05_ei") or {}).get("system") if isinstance(sel, dict)
                             else None)):
            if named is not None and named != v05:
                raise SystemExit(f"{name} names {named} as v0.5's EI specialist, but release_candidate.json names {v05}; stop")
        markers = load_json(p.results / "not_built.json")
        specs = system_specs(p, names)
        data, files = self.load_test(specs, names, sel, markers)
        fitted = self.fit_weights(data[RW], specs)
        ei, pq, rw = data[EI], data[PQA], data[RW]

        def family_of(system):
            return ((cal9.get(system) or {}).get(EI) or {}).get("chosen")

        family = family_of(v05) or names["family"]
        out = {"plan": PLAN, "written_utc": None, "source": SOURCE,
               "release_candidate": {**file_entry(p.results / "release_candidate.json"), "time_utc": rc.get("time_utc")},
               "testpreds_log": file_entry(p.results / "testpreds_log.json"), "ece_limits": rc.get("ece_limits"),
               "settings": {"split": "test, except K3 (the 654 clean validation prompts)", "resamples": self.resamples,
                            "resamples_bar": self.resamples_bar, "seed": SEED, "interval": "95 percent, percentile, paired",
                            "ei_resampling": "articles, sorted by PMCID as text", "other_resampling": "items", "ece_bins": 15,
                            "k3": {"folds": FOLDS, "repeats": self.k3_repeats, "calibration": "session 8's rule, out of fold"},
                            "metrics": {ev: list(CMP.metric_names(ev)) for ev in EVALUATIONS}},
               "contamination": CONTAMINATION,
               "v05": {EI: {"system": v05, "base_system": x, "swap_averaged_form": names["with_sa"],
                            "swap_averaging": names["swap_averaging"], "recipe": names["recipe"], "runs": names["runs"],
                            "single_pass_systems": names["single_pass_systems"], "max_len": names["max_len"],
                            "calibration_family": family},
                       "criterion_3_large_model": large, PQA: "s7_b3", RW: "readmission_pool_v2"},
               "ei_prior": self.prior, "systems": {ev: list(data[ev]["probs"]) for ev in EVALUATIONS}, "files": files,
               "not_built": self.not_built, "comparisons_not_made": self.not_made, "decision_weights_readmit_window": fitted,
               "metrics": {}, "comparisons": self.comparisons, "bar_intervals": {}}

        # every system, every metric
        for ev in EVALUATIONS:
            out["metrics"][ev] = {}
            for system in data[ev]["probs"]:
                if (ev, system) in self.waiting:
                    continue
                entry = self.metrics_block(data[ev], ev, system, CMP.metric_names(ev))
                if entry is not None:
                    out["metrics"][ev][system] = entry

        # K1: the bar's comparisons, every EI metric, and criteria 2 and 3's intervals
        self.compare("K1", ei, EI, v05, "prior", EI_METRICS,
                     f"{v05} (v0.5's EI specialist) minus the prior (v0.4's prior, the answer shares of session 8's whole "
                     "training split)")
        self.compare("K1", ei, EI, v05, large, EI_METRICS,
                     f"{v05} minus {large}, the large model of criterion 3, both calibrated on validation")
        for key, ref in (("v05_minus_prior", "prior"), ("v05_minus_large_model", large)):
            if v05 in ei["probs"] and ref in ei["probs"]:
                res = self.cache.get(["bar", v05, ref, ei["digests"][v05], ei["digests"][ref], self.resamples_bar, SEED,
                                      self.records_sha],
                                     lambda: CMP.bar_interval(ei["labels"], ei["probs"][v05], ei["probs"][ref], ei["rows_of"],
                                                              self.resamples_bar, SEED))
                if res is not None:
                    out["bar_intervals"][key] = {"system": v05, "reference": ref, **res}
        out["k1"] = {"specialist": v05, "large_model": large, "ece_limit": ((rc.get("ece_limits") or {}).get(EI) or {}),
                     "note": ("v0.4's four criteria with v0.5's EI specialist in s8_ei's place (POWER AND THE BAR); the ECE "
                              "limit and the large model are release_candidate.json's")}

        # K2: what the license cost
        self.compare("K2", ei, EI, "s9_b", "s8_ei", EI_METRICS,
                     "what the license cost: s9_b (trained on the clean articles) minus s8_ei (trained on all of them), both "
                     "three runs averaged and calibrated by session 8's rule (s9_b on the 654 clean validation prompts, s8_ei "
                     "on all 1,237); descriptive")
        self.compare("K2", ei, EI, v05, "s8_ei", EI_METRICS, f"{v05} (v0.5's EI specialist) minus s8_ei; descriptive")
        out["k2"] = {"note": "what the license cost (descriptive)", "systems": ["s9_b", v05], "reference": "s8_ei"}

        # K3: the recipes on validation, and v0.5's specialist minus s9_b on test
        out["k3"], k3_runs = self.k3(sel)
        self.compare("K3_test", ei, EI, v05, "s9_b", EI_METRICS,
                     f"on test: {v05} (v0.5's EI specialist) minus s9_b (base's three runs averaged, session 8's rule); "
                     "descriptive")

        # K4: the reading checks
        out["k4"] = self.k4()

        # K5: PubMedQA beside BioGPT-Large
        names_pq = CMP.metric_names(PQA)
        for system, reference, note in (
                ("s7_b3", "biogpt_large_pqa-raw", "s7_b3 (v0.5's PubMedQA specialist, session 7's calibration) minus "
                 "BioGPT-Large-PubMedQA's uncalibrated probabilities, renormalized over the three answers (primary); descriptive"),
                ("s7_b3", "biogpt_large_pqa", "s7_b3 minus BioGPT-Large-PubMedQA calibrated by session 8's rule on the 150 "
                 "validation items (secondary: it was fine-tuned on them); descriptive"),
                ("biogpt_large_pqa-raw", "gemma4_31b", "BioGPT-Large-PubMedQA's uncalibrated probabilities (primary) minus "
                 "gemma4_31b (calibrated on validation); descriptive"),
                ("biogpt_large_pqa", "gemma4_31b", "BioGPT-Large-PubMedQA calibrated (secondary: it was fine-tuned on the "
                 "calibration items) minus gemma4_31b; descriptive")):
            self.compare("K5", pq, PQA, system, reference, names_pq, note)
        biogpt = p.results / "biogpt.json"
        out["k5"] = {"primary": "biogpt_large_pqa-raw", "secondary": "biogpt_large_pqa", "caveat": BIOGPT_CAVEAT,
                     "biogpt_json": file_entry(biogpt) if biogpt.exists() else None}

        # K6: readmission
        pool_path = p.results / "readmission_pool_v2.json"
        pool = load_json(pool_path, {})
        pool = pool if isinstance(pool, dict) else {}
        chosen = pool.get("chosen")
        pool_note = {"new": "v0.5 uses the new pool (readmission_pool_v2.json chose it)",
                     "session8": ("v0.5 keeps session 8's pool (readmission_pool_v2.json), so readmission_pool_v2 is session "
                                  "8's weights and T applied again to its members' test files, and this difference is zero up "
                                  "to the printed digits")}.get(chosen, "readmission_pool_v2.json records no choice")
        names_rw = CMP.metric_names(RW)
        for system, reference, note in (
                ("readmission_pool_v2", "readmission_pool", f"v0.5's pool minus session 8's pool: {pool_note}; descriptive"),
                ("xgboost_codes", "xgboost_tuned", "session 8's xgboost_tuned setting refitted with the recoded diagnosis "
                 "fields, minus xgboost_tuned; descriptive"),
                ("catboost_codes", "catboost_tuned", "session 8's catboost_tuned setting refitted with the recoded diagnosis "
                 "fields, minus catboost_tuned; descriptive"),
                ("tabicl", "lightgbm_tuned", "TabICLv2 minus lightgbm_tuned (session 7's predictions); descriptive")):
            self.compare("K6", rw, RW, system, reference, names_rw, note)
        out["k6"] = {"v05_pool": "readmission_pool_v2", "chosen": chosen, "choice": pool.get("choice"), "note": pool_note,
                     "readmission_pool_v2_json": file_entry(pool_path) if pool_path.exists() else None,
                     "lightgbm_tuned": ("session 7's predictions (results/fiorillo_v0_3); its refit, which stands for it in "
                                        "the new pool, is lightgbm_tuned_refit in the metrics")}

        # K7: the hand-off
        out["k7"] = self.k7()

        # K8: the weight average
        self.compare("K8", ei, EI, "s9_ei_wavg", "s9_ei", EI_METRICS,
                     f"s9_ei_wavg (the weight average) minus s9_ei (the probability average of the same runs: three, or two under correction 7), both "
                     f"calibrated by v0.5's chosen family ({family}), each fitted on its own 654 clean validation predictions; "
                     "descriptive")
        out["k8"] = {"system": "s9_ei_wavg", "reference": "s9_ei", "v05_family": family,
                     "families": {s: family_of(s) for s in ("s9_ei_wavg", "s9_ei")},
                     "weight_average": (sel or {}).get("weight_average") if isinstance(sel, dict) else None}
        if "s9_ei_wavg" not in ei["probs"]:
            out["k8"]["not_built"] = ei["absent"].get("s9_ei_wavg", "not built")

        # K9: the calibration families
        s8rule = f"{v05}_s8rule"
        self.compare("K9", ei, EI, v05, s8rule, EI_METRICS,
                     f"{v05} under its chosen family ({family}) minus under session 8's rule ({family_of(s8rule)}), both "
                     "fitted on the 654 clean validation prompts; descriptive")
        table = cal9.get("families_table_v05")
        out["k9"] = {"families_table": table, "chosen": family, "session8_rule_system": s8rule,
                     "session8_rule_family": family_of(s8rule),
                     "calibration_s9_json": file_entry(p.results / "calibration_s9.json")
                     if (p.results / "calibration_s9.json").exists() else None,
                     "note": ("the table is calibration_s9.json's: each family's out-of-fold log loss on the 654 clean validation "
                              "prompts (5 folds, 20 repeats; mean, min and max over the repeats), reported whatever was chosen")}
        table_chosen = (table or {}).get("chosen") or family
        for fam, v in ((table or {}).get("table") or {}).items():
            v = v or {}
            spread = ", ".join(f"{k} {fl(v.get(k)):.5f}" for k in ("min", "max") if fl(v.get(k)) is not None)
            self.table.append({"section": "K9", "evaluation": EI, "system": (table or {}).get("system"), "group": fam,
                               "metric": "out_of_fold_log_loss", "estimate": fl(v.get("mean")),
                               "n": ((cal9.get(v05) or {}).get(EI) or {}).get("n_calibrate"),
                               "note": ("validation, calibration_s9.json; the mean over the repeats"
                                        + (f" ({spread})" if spread else "") + ("; chosen" if fam == table_chosen else ""))})

        # K10: swap averaging
        self.compare("K10", ei, EI, names["with_sa"], x, EI_METRICS,
                     f"{names['with_sa']} (swap averaging) minus {x} (one pass), each calibrated by v0.5's chosen family "
                     f"({family}) on its own 654 clean validation predictions; v0.5 "
                     f"{'uses' if names['swap_averaging'] else 'does not use'} swap averaging (correction 3); descriptive")
        out["k10"] = {"with_swap_averaging": names["with_sa"], "without": x, "v05": v05,
                      "kept": names["swap_averaging"], "families": {s: family_of(s) for s in (names["with_sa"], x)},
                      "decision": (sel or {}).get("swap_averaging") if isinstance(sel, dict) else None}

        # the sensitivity: K1 on the prompts the README does not flag, their articles resampled
        flags = ei["flags"]
        unflagged = [i for i, f in enumerate(flags) if f == "none"]
        out["sensitivity"] = {"rule": SENSITIVITY_RULE, "flagged_test_prompts": len(flags) - len(unflagged),
                              "unflagged_test_prompts": len(unflagged),
                              "flags": dict(sorted(collections.Counter(f for f in flags if f != "none").items())), "metrics": {}}
        if not unflagged:
            out["sensitivity"]["not_computed"] = "every test prompt is flagged, so no prompt is left"
        else:
            sub = subset_dataset(ei, unflagged, "unflagged")
            for system in (v05, "prior", large):
                if system in sub["probs"]:
                    entry = self.metrics_block(sub, EI, system, EI_METRICS, section="K1_sensitivity_metrics")
                    if entry is not None:
                        out["sensitivity"]["metrics"][system] = entry
            for ref in ("prior", large):
                self.compare("K1_sensitivity", sub, EI, v05, ref, EI_METRICS, f"K1 {SENSITIVITY_RULE}", subset_name="unflagged")

        # calibration slope and intercept; the cost per 1,000 questions
        out["calibration_slope_intercept"] = self.slopes(data, out["metrics"], names)
        out["cost_per_1000_questions"] = cost_section(p, names, len(ei["labels"]))

        if self.cache.stopped:
            print(json.dumps({"status": "stopped; run again to resume", "seconds": round(time.time() - self.t0, 1),
                              "cached_blocks": len(list(p.cache.glob("*.json")))}), flush=True)
            return None

        out["release_bar"] = release_bar(out, rc, v05, large)
        self.finish_table(out)
        out["written_utc"] = now()
        out = clean(out)
        p.results.mkdir(parents=True, exist_ok=True)
        (p.results / "comparison.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
        with open(p.results / "comparison.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=C.CSV_COLUMNS)
            w.writeheader()
            for r in self.table:
                w.writerow({c: C.fmt(r.get(c)) for c in C.CSV_COLUMNS})
        raw_runs = sorted(set(k3_runs) | set(names["runs"]) | {"s9_b_1", "s9_b_2", "s9_b_3", "s9_ei_wavg"})
        large_dir = (p.results if large == "gemma4_31b_long" else p.results8) / "large_model"
        extra = [(f"{r}-raw", EI, split, p.results / "predictions" / f"{r}-raw_{EI}_{split}.csv")
                 for r in raw_runs for split in ("validation", "test")]
        extra += [(f"{large}-raw", EI, split, large_dir / f"{large}-raw_{EI}_{split}.csv") for split in ("validation", "test")]
        zeros = {"plan": PLAN, "written_utc": now(), "rows_whose_true_answer_prints_as_0": zeros_table(specs, extra)}
        (p.results / "zeros.json").write_text(json.dumps(clean(zeros), indent=1) + "\n", encoding="utf-8")
        print(json.dumps({"status": "written", "release_bar": out["release_bar"]["overall"],
                          "outcome": out["release_bar"]["outcome"], "not_built": len(out["not_built"]),
                          "comparisons_not_made": len(out["comparisons_not_made"]),
                          "seconds": round(time.time() - self.t0, 1)}, indent=1), flush=True)
        return out

    def finish_table(self, out):
        """comparison.csv's last rows: the bar by criterion, the costs and the notes."""
        bar = out["release_bar"]
        for k, c in bar["criteria"].items():
            self.table.append({"section": "release_bar", "evaluation": EI, "system": bar["specialist"],
                               "reference": bar["large_model"] if k in ("3", "4") else ("prior" if k == "2" else ""),
                               "metric": f"criterion {k}",
                               "note": f"{c['result']}: {c['rule']}" + (f" ({c['reason']})" if c["reason"] else "")
                                       + f"; values {json.dumps(clean(c['values']), sort_keys=True)}"})
        self.table.append({"section": "release_bar", "evaluation": EI, "system": bar["specialist"], "metric": "overall",
                           "note": f"{bar['overall']}: {bar['outcome']}"})
        cost = out["cost_per_1000_questions"]
        for kind in ("specialists", "large_model"):
            for ev, e in cost[kind].items():
                self.table.append({"section": "cost_per_1000_questions", "evaluation": ev, "system": e.get("system"),
                                   "group": kind, "metric": "usd_per_1000_questions", "estimate": e.get("usd"),
                                   "note": (f"{e['seconds_per_question']:.4f} seconds a question at {cost['hourly_usd']} dollars "
                                            "an hour" if e.get("usd") is not None else e.get("not_computed", "not computed"))})
        for metric, note in (("contamination", CONTAMINATION), ("biogpt_caveat", BIOGPT_CAVEAT), ("k3", K3_SCORE_NOTE)):
            self.table.append({"section": "note", "metric": metric, "note": note})


def run(results=RESULTS, results8=RESULTS8, results7=RESULTS7, results_public=RESULTS_PUBLIC, records_public=RECORDS_PUBLIC,
        records_ei=RECORDS_EI, records9=RECORDS9, cache_dir=CACHE, pod_rate=None, max_seconds=None, resamples=RESAMPLES,
        resamples_bar=RESAMPLES_BAR, k3_repeats=REPEATS):
    """Every comparison. Writes comparison.json, comparison.csv and zeros.json in results and returns comparison.json's
    content, or None when max_seconds stopped the run before every block was done (run it again to resume). k3_repeats is
    the plan's 20; tests pass fewer."""
    paths = Paths(results, results8, results7, results_public, records_public, records_ei, records9, cache_dir, pod_rate)
    guard(paths.results)
    return Study(paths, resamples, resamples_bar, max_seconds, k3_repeats).run()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--results8", type=Path, default=RESULTS8)
    parser.add_argument("--results7", type=Path, default=RESULTS7)
    parser.add_argument("--results-public", type=Path, default=RESULTS_PUBLIC)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--records-ei", type=Path, default=RECORDS_EI)
    parser.add_argument("--records9", type=Path, default=RECORDS9)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--pod-rate", type=Path, default=None,
                        help="pod_rate.txt (default <results>/pod_rate.txt; if absent, cost_rule.json's pod > hourly_usd)")
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--resamples-bar", type=int, default=RESAMPLES_BAR)
    args = parser.parse_args(argv)
    run(args.results, args.results8, args.results7, args.results_public, args.records_public, args.records_ei, args.records9,
        args.cache, args.pod_rate, args.max_seconds, args.resamples, args.resamples_bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())

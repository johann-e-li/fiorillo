"""The reading checks, I-32 and I-37: scoring and readings (docs/prompt_session9.md, THE READING CHECKS; COMPARISONS, K4;
steps 10 and 15; corrections 3 and 6).

The checks ask how much of an EI specialist's answer comes from reading the article. src/tasks/ei_session9.py checks
builds their inputs from the test records, IDs and groups kept (data/processed/session9/checks/<input>/
evidence_effect_test.jsonl): none (the article set to the empty text; every one of the 1,218 test prompts); title (the
article set to the article's title by correction 6's rule; the 9 test prompts whose article gives no title left out);
and swap (the intervention and the comparator exchanged in the state, the article as in the system's own input:
checks/swap/ at 4,096 tokens, checks/swap_long/ at 6,144 for a v0.5 recipe that reads 6,144; the expected answer already
exchanged for "significantly increased" and "significantly decreased" and kept for "no significant difference"; prompts
whose intervention and comparator are the same text left out).

A system's probabilities on an input are its runs' raw probabilities (session 4's format, id,label,p0,p1,p2, as
train_qwen_head.py predict writes them; p0 increased, p1 decreased, p2 no difference) averaged, then its calibration.
s8_ei: its three runs averaged, then session 8's calibration (results/fiorillo_v0_4/calibration_s8.json, "s8_ei" >
"evidence_effect"); on its full input, results/fiorillo_v0_4/predictions/s8_ei-raw_evidence_effect_test.csv (the average
of its three runs, from calibrate_session8.py). v0.5's EI specialist: its runs (three for s9_ei, one for s9_ei_wavg)
averaged, then its own calibration (calibration_s9.json, the same structure). The chosen family is applied to the
logarithm of the averaged probabilities, floored at 1e-300 as calibrate_session8.py takes s8_ei's average (--log-floor
1e-12 matches a calibration fitted on printed probabilities clipped at 1e-12, as calibration_family_cv.py's command
does): temperature and temperature_intercepts here, by the formula of calibration_family_cv.calibrated that
calibrate_session8.py applies; vector and dirichlet_odir by calibration_family_cv.apply_family(name, params, logp),
imported only when one of them is chosen. An answer is the argmax of the calibrated probabilities.

v0.5 (correction 3). --swap-augmented: its recipe includes swap augmentation, and its swap reading carries the plan's
caveat. --swap-averaged: v0.5 keeps swap averaging, so its answer to a prompt averages its raw answer to the prompt with
the exchange (p0 and p1 exchanged) of its raw answer to the exchanged prompt, before calibration; its answer to an
exchanged prompt is then the exchange of its answer to the prompt, so its swap check is consistent by construction and
is reported as such, not read. Its none and title checks use its single pass, without swap averaging, and so does the
answer on the full input that they are compared with, so that the two answers differ only in the article; their
calibration is --single-pass-key's (the single-pass form's calibration in the same file, fitted for K10) if given,
otherwise its own.

Reported for each input, each with a 95 percent interval from 1,000 resamples of the input's articles (numpy's
default_rng(20260927); articles sorted by PMCID as text, every prompt of a drawn article kept; compare_session8.py's
article_rows and cluster_bootstrap, as session 8 resampled): accuracy and macro-F1 (metrics.py) against the original
answers (none and title) or the exchanged ones (swap); for none and title, the share of answers equal to the system's
answer on its full input; for swap, the share of the system's "increased" and "decreased" answers on its full input that
exchange on the exchanged input and the share of its "no significant difference" answers that stay, on every swap
prompt and on the prompts whose comparator names a time point or a baseline (TIME_POINT, matched against the comparator
of the prompt as written, which the exchanged record holds as its intervention; that subset's own articles resampled, as
session 8 resampled a subset); and the paired difference in macro-F1, title minus none, on the prompts in both inputs,
with the same resamples for both. A difference whose interval includes zero reads "no difference was detected", with the
estimate, the interval and what the interval excludes (COMPARISONS).

TIME_POINT: baseline, before, pre- (with an ASCII hyphen, or a Unicode hyphen or dash, U+2010 to U+2015 or U+2212),
pretreatment, prior to (any whitespace between the two words) or start, found anywhere in the text, ignoring case.

Readings (fixed in THE READING CHECKS; they gate no release; a value within 1e-12 of a line counts as on it): none, from
the macro-F1 with no article: at or below 0.60, "its answers depend on the article's text"; above 0.75, the plan's reading
that the prompt alone carries more than a classifier trained on prompts alone; otherwise "partly". title, from the
interval of the paired difference: wholly above 0.05, the title adds information about the result; wholly below 0.05,
"no added knowledge of 0.05 or more was detected"; otherwise "not settled". swap, from the accuracy against the exchanged
answers: at or above 0.85, "it follows the comparison as written"; below 0.70, "it does not follow the exchange (recall
of the trial or a bias toward one direction)"; otherwise "partly".

The stop: if s8_ei's macro-F1 with no article is above 0.85, score writes its JSON with "stop": true and the numbers,
then exits with code 3, and the driver stops the chain before the train stage.

References stated beside the results: the training split's answer shares (the prior) score a macro-F1 of 0.198 on the
test split (read from results/fiorillo_v0_4/comparison.json, metrics > evidence_effect > prior > macro_f1; recorded as not
read where that file is absent, and combine reads it again); and Windisch et al. 2026, JMIR AI, confirmed at its source
in step 0 (docs/prior_art.md).

Subcommands:
  score    one system's checks. --system s8_ei writes checks_s8.json (default runs:
           <results>/checks_s8/s8_ei_<k>_<input>-raw_evidence_effect_test.csv for k 1 to 3; full input, calibration and
           key as above). --system v05 --name <s9_ei or s9_ei_wavg> writes checks_v05.json (runs and full input given
           explicitly; calibration <results>/calibration_s9.json, key the name unless --calibration-key is given); it
           refuses to run unless release_candidate.json and testpreds_log.json exist in --results, since it reads v0.5's
           test predictions.
  combine  reading_checks.json for both systems, from checks_s8.json and checks_v05.json (K4), with the references.

Exit codes: 0 written; 3 the stop (its JSON written first); 1 any other failure, 2 an argument error.

Usage: python3 src/eval/reading_checks_session9.py score --system s8_ei [--runs-none F1 F2 F3] [--runs-title ...]
           [--runs-swap ...] [--out results/fiorillo_v0_5/checks_s8.json]
       python3 src/eval/reading_checks_session9.py score --system v05 --name s9_ei --runs-none F1 F2 F3 \\
           --runs-title F1 F2 F3 --runs-swap F1 F2 F3 --full F1 F2 F3 \\
           [--records-swap data/processed/session9/checks/swap_long/evidence_effect_test.jsonl] \\
           [--swap-augmented] [--swap-averaged [--single-pass-key KEY]]
       python3 src/eval/reading_checks_session9.py combine
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
import compare_session8 as CMP  # noqa: E402
import metrics as M  # noqa: E402
from ei_session9 import SWAP_LABEL  # noqa: E402

SEED = 20260927
RESAMPLES = 1000
EI = "evidence_effect"
ANSWERS = CMP.ANSWERS  # increased, decreased, no_difference: labels 0, 1 and 2
EXCHANGE = np.array([SWAP_LABEL[j] for j in range(3)])  # the answer each answer becomes when the two arms are exchanged
INPUTS = ("none", "title", "swap")
SYSTEMS = ("s8_ei", "v05")
RESULTS = ROOT / "results" / "fiorillo_v0_5"
RESULTS8 = ROOT / "results" / "fiorillo_v0_4"
CHECKS = ROOT / "data" / "processed" / "session9" / "checks"
OUT_S8 = RESULTS / "checks_s8.json"
OUT_V05 = RESULTS / "checks_v05.json"
OUT_COMBINED = RESULTS / "reading_checks.json"
CAL_S8 = RESULTS8 / "calibration_s8.json"
FULL_S8 = RESULTS8 / "predictions" / f"s8_ei-raw_{EI}_test.csv"
COMPARISON_S8 = RESULTS8 / "comparison.json"
PLAN = "docs/prompt_session9.md, THE READING CHECKS"
PLAN_COMBINED = "docs/prompt_session9.md, THE READING CHECKS and COMPARISONS (K4)"
T_MIN, T_MAX = 0.1, 10.0  # calibration_family_cv.py's bounds on the temperature
LOG_FLOOR = 1e-300  # calibrate_session8.py's floor for the log of an average
TIE = 1e-12  # a value within this of a line counts as on it (floating-point rounding)
STOP_EXIT = 3
TEMPERATURE_FAMILIES = ("temperature", "temperature_intercepts")
LATER_FAMILIES = ("vector", "dirichlet_odir")
HYPHENS = "-" + "".join(chr(c) for c in range(0x2010, 0x2016)) + chr(0x2212)  # the ASCII hyphen, U+2010 to U+2015 and U+2212
TIME_POINT = re.compile("baseline|before|pre[" + HYPHENS + r"]|pretreatment|prior\s+to|start", re.IGNORECASE)

NONE_LOW, NONE_HIGH = 0.60, 0.75
TITLE_LINE = 0.05
SWAP_HIGH, SWAP_LOW = 0.85, 0.70
STOP_ABOVE = 0.85
PARTLY = "partly"
NONE_DEPENDS = "its answers depend on the article's text"
NONE_PROMPT = ("the prompt alone carries more than a classifier trained on prompts alone (0.511 macro-F1, DeYoung et al. 2020, "
               "Table 1); it may come from EI's training prompts (a model of 4 billion parameters may learn stronger priors from "
               "them than that classifier did), from pretraining, or from EI's public labels, and these checks cannot tell them "
               "apart")
TITLE_ADDS = ("the title adds information about the result: a title can state the result, or the model may recall the trial, "
              "and these checks cannot tell them apart")
TITLE_NONE = "no added knowledge of 0.05 or more was detected"
NOT_SETTLED = "not settled"
SWAP_FOLLOWS = "it follows the comparison as written"
SWAP_NOT = "it does not follow the exchange (recall of the trial or a bias toward one direction)"
NONE_RULE = (f"macro-F1 with no article (its estimate) at or below 0.60: \"{NONE_DEPENDS}\"; above 0.75: \"{NONE_PROMPT}\"; "
             "otherwise \"partly\"")
TITLE_RULE = (f"the 95 percent interval of the paired difference in macro-F1, title minus none: wholly above 0.05: \"{TITLE_ADDS}\"; "
              f"wholly below 0.05: \"{TITLE_NONE}\"; otherwise \"not settled\"")
SWAP_RULE = (f"accuracy against the exchanged answers (its estimate) at or above 0.85: \"{SWAP_FOLLOWS}\"; below 0.70: "
             f"\"{SWAP_NOT}\"; otherwise \"partly\"")
STOP_RULE = ("if s8_ei's macro-F1 with no article is above 0.85, the chain stops before the train stage (THE READING CHECKS); "
             "score then exits with code 3 after writing this file")
CAVEAT_AUGMENTED = ("v0.5's recipe includes swap augmentation, so its swap reading measures what the augmentation taught as much "
                    "as reading; s8_ei's swap check is the one that bears on reading")
CONSISTENT = ("consistent by construction (swap averaging kept): v0.5's answer to an exchanged prompt is the exchange of its "
              "answer to the prompt, so its swap check is reported as such and not read as a reading check; s8_ei's swap check "
              "is the one that bears on reading (correction 3)")
CAVEAT_CALIBRATION = ("the exchange holds exactly before calibration; a calibration that treats \"increased\" and \"decreased\" "
                      "differently (unequal intercepts, or a matrix not symmetric in the two) can move an answer that lies near a "
                      "boundary, so the exchange and stay shares can fall short of 1")
CAVEAT_SINGLE_PASS = ("v0.5's none and title checks use its single pass, without swap averaging (correction 3); the answers on "
                      "its full input that they are compared with are its single pass too")
PRIOR_TEXT = "the training split's answer shares score a macro-F1 of {e:.3f} on the test split ({l:.3f} to {h:.3f})"
WINDISCH = {
    "text": ("frontier models scored an accuracy of 0.79 to 0.88 from trial titles alone on 250 oncology trials, on a two-answer "
             "task (positive or negative), against a majority class of 0.584"),
    "accuracy_low": 0.79, "accuracy_high": 0.88, "trials": 250, "answers": 2, "majority_class_accuracy": 0.584,
    "source": ("Windisch et al. 2026, JMIR AI 2026;5:e95565, doi 10.2196/95565 (PubMed 42525868; full text PMC13419288), "
               "Table 1"),
    "verified": "confirmed at its source in step 0 (docs/prior_art.md, the session 9 check of 1 October 2026)"}
CONTAMINATION = CMP.CONTAMINATION + " EI's test split was read in session 8."


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rel(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def num(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def file_entry(path, rows=None):
    out = {"file": rel(path), "sha256": CMP.sha256_file(path)}
    if rows is not None:
        out["rows"] = int(rows)
    return out


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------- inputs

def load_records(path):
    """One check input's records: IDs, answers (labels), articles (group.pmcid as text) and the state's two arms."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{rel(path)} does not exist")
    ids, labels, clusters, interventions, comparators = [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                ids.append(r["id"])
                labels.append(int(r["label"]))
                clusters.append(str(r["group"]["pmcid"]))
                state = r.get("state") or {}
                interventions.append(str(state.get("intervention", "")))
                comparators.append(str(state.get("comparator", "")))
    if not ids:
        raise SystemExit(f"{rel(path)} holds no records")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"{rel(path)}: an ID appears twice")
    if any(y not in (0, 1, 2) for y in labels):
        raise SystemExit(f"{rel(path)}: a label is not 0, 1 or 2")
    return {"path": path, "ids": ids, "labels": np.array(labels, dtype=np.int64), "clusters": clusters,
            "interventions": interventions, "comparators": comparators}


def average_runs(paths, rec):
    """The runs' raw probabilities averaged, rows in the records' order; each file must hold exactly the records' prompts
    with the records' answers (compare_session8.load_aligned)."""
    probs = []
    for p in paths:
        if not Path(p).exists():
            raise SystemExit(f"{rel(p)} does not exist")
        got = CMP.load_aligned(p, rec["ids"], rec["labels"])
        if got.shape[1] != 3:
            raise SystemExit(f"{Path(p).name}: {got.shape[1]} probability columns, not 3")
        probs.append(got)
    return np.mean(probs, axis=0)


def full_input(paths, ids, labels):
    """The system's raw probabilities on its full input, averaged over the files given, for the prompts `ids` (rows in that
    order); each file may hold more prompts, and must hold these with the answers `labels`."""
    probs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise SystemExit(f"{rel(p)} does not exist")
        f_ids, f_labels, f_probs = M.read_predictions(p)
        pos = {i: k for k, i in enumerate(f_ids)}
        if len(pos) != len(f_ids):
            raise SystemExit(f"{p.name}: an ID appears twice")
        missing = [i for i in ids if i not in pos]
        if missing:
            raise SystemExit(f"{p.name}: {len(missing)} of the check input's {len(ids)} prompts are missing (the first: {missing[0]})")
        order = np.array([pos[i] for i in ids], dtype=np.int64)
        if not np.array_equal(f_labels[order], labels):
            raise SystemExit(f"{p.name}: its answers differ from the check input's answers on the full input")
        if f_probs.shape[1] != 3:
            raise SystemExit(f"{p.name}: {f_probs.shape[1]} probability columns, not 3")
        probs.append(f_probs[order])
    return np.mean(probs, axis=0)


def exchange(probs):
    """The probabilities of "increased" and "decreased" exchanged (correction 3)."""
    return np.asarray(probs)[:, EXCHANGE]


# ---------------------------------------------------------------- calibration

def calibration_entry(path, key, qid=EI):
    """The chosen family and its parameters for one system, from a file shaped as calibration_s8.json
    ({system: {qid: {"chosen": family, "families": {family: {"params": [...]}}}}})."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{rel(path)} does not exist")
    data = json.loads(path.read_text(encoding="utf-8"))
    entry = data.get(key)
    if isinstance(entry, dict) and qid in entry:
        entry = entry[qid]
    if not isinstance(entry, dict) or "chosen" not in entry or entry["chosen"] not in entry.get("families", {}):
        raise SystemExit(f"{rel(path)} holds no calibration for {key} on {qid}")
    family = entry["chosen"]
    params = entry["families"][family]["params"]
    if family not in TEMPERATURE_FAMILIES + LATER_FAMILIES:
        raise SystemExit(f"{rel(path)}: {key}'s calibration family {family!r} is not one of {TEMPERATURE_FAMILIES + LATER_FAMILIES}")
    return {"file": rel(path), "sha256": CMP.sha256_file(path), "key": key, "qid": qid, "family": family, "params": params}


def temperature_family(logp, params, family):
    """softmax(log p / T + b) with T = exp(params[0]) clipped to [0.1, 10], b[0] = 0 and b[1:] = params[1:] (the
    temperature family has no b): calibration_family_cv.calibrated, as calibrate_session8.py applies it."""
    params = [float(v) for v in np.atleast_1d(params)]
    k = logp.shape[1]
    if len(params) != (1 if family == "temperature" else k):
        raise SystemExit(f"the {family} family has {len(params)} parameters, not {1 if family == 'temperature' else k}")
    t = float(np.clip(np.exp(params[0]), T_MIN, T_MAX))
    b = np.zeros(k)
    if len(params) > 1:
        b[1:] = params[1:]
    z = logp / t + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibrated(probs, cal, log_floor=LOG_FLOOR):
    """The calibration's chosen family applied to the logarithm of the (averaged) raw probabilities, floored at log_floor."""
    logp = np.log(np.maximum(np.asarray(probs, dtype=np.float64), log_floor))
    if cal["family"] in TEMPERATURE_FAMILIES:
        return temperature_family(logp, cal["params"], cal["family"])
    if cal["family"] not in LATER_FAMILIES:
        raise SystemExit(f"unknown calibration family {cal['family']!r}")
    import calibration_family_cv as CFC  # only here: vector and dirichlet_odir are session 9's additions to that module

    out = np.asarray(CFC.apply_family(cal["family"], cal["params"], logp), dtype=np.float64)
    if out.shape != logp.shape or not np.all(np.isfinite(out)) or np.any(np.abs(out.sum(axis=1) - 1.0) > 1e-6):
        raise SystemExit(f"calibration_family_cv.apply_family({cal['family']!r}) did not return probabilities of shape {logp.shape}")
    return out


# ---------------------------------------------------------------- statistics

def share(hit, among):
    """The share of the rows in `among` where `hit` holds; nan when `among` is empty."""
    d = int(np.sum(among))
    return float(np.sum(hit & among) / d) if d else float("nan")


def resampled(stat, clusters, resamples=RESAMPLES):
    """compare_session8.cluster_bootstrap over the articles of these rows: (estimate, low, high, the number of resamples in
    which each statistic is defined, the number of articles)."""
    rows_of = CMP.article_rows(clusters)
    seen = []

    def wrapped(idx):
        v = np.atleast_1d(np.asarray(stat(idx), dtype=float))
        seen.append(v)
        return v

    est, low, high = CMP.cluster_bootstrap(wrapped, rows_of, len(clusters), resamples, SEED)
    used = np.isfinite(np.array(seen[1:])).sum(axis=0) if resamples else np.zeros(len(est), dtype=int)
    return est, low, high, used, len(rows_of)


def interval(est, low, high, used=None, resamples=None):
    out = {"estimate": num(est), "low": num(low), "high": num(high)}
    if used is not None:
        out["resamples_defined"] = int(used)
        out["resamples"] = int(resamples)
    return out


def answer_counts(ans):
    return {a: int(c) for a, c in zip(ANSWERS, np.bincount(np.asarray(ans, dtype=np.int64), minlength=3))}


def exchange_shares(full_ans, ans):
    """(share of "increased" and "decreased" answers on the full input that exchange, share of "no significant difference"
    answers that stay), as a vector for the resampling."""
    moved = full_ans != 2
    return [share(ans == EXCHANGE[full_ans], moved), share(ans == 2, ~moved)]


def share_entry(est, low, high, used, resamples, hits, among):
    return {**interval(est, low, high, used, resamples), "count": int(hits), "of": int(among)}


# ---------------------------------------------------------------- readings

def none_reading(macro_f1):
    if macro_f1 is None:
        return "not computed"
    if macro_f1 <= NONE_LOW + TIE:
        return NONE_DEPENDS
    if macro_f1 > NONE_HIGH + TIE:
        return NONE_PROMPT
    return PARTLY


def title_reading(low, high):
    if low is None or high is None:
        return "not computed"
    if low > TITLE_LINE + TIE:
        return TITLE_ADDS
    if high < TITLE_LINE - TIE:
        return TITLE_NONE
    return NOT_SETTLED


def swap_reading(accuracy):
    if accuracy is None:
        return "not computed"
    if accuracy >= SWAP_HIGH - TIE:
        return SWAP_FOLLOWS
    if accuracy < SWAP_LOW - TIE:
        return SWAP_NOT
    return PARTLY


def stops(none_macro_f1):
    """The stop of THE READING CHECKS: s8_ei's macro-F1 with no article above 0.85."""
    return none_macro_f1 is not None and none_macro_f1 > STOP_ABOVE + TIE


def difference_reading(low, high):
    """COMPARISONS: a difference whose interval includes zero reads "no difference was detected"."""
    if low is None or high is None:
        return "not computed"
    if low <= 0 <= high:
        return "no difference was detected"
    return "higher with the title than with no article" if low > 0 else "lower with the title than with no article"


def excludes_text(est, low, high):
    """What the interval of title minus none excludes, in words (COMPARISONS)."""
    if est is None or low is None or high is None:
        return "not computed"
    zero = "zero is inside it" if low <= 0 <= high else "it excludes zero"
    line = {"inside the interval": "0.05 is inside it", "excluded: above the interval": "it excludes 0.05 (above the interval)",
            "excluded: below the interval": "it excludes 0.05 (below the interval)"}[CMP.excludes(low, high, TITLE_LINE)]
    return (f"title minus none in macro-F1 is {est:.3f}, with a 95 percent interval of {low:.3f} to {high:.3f}: the interval "
            f"excludes a difference below {low:.3f} and one above {high:.3f}; {zero}; {line}")


def prior_reference(comparison=COMPARISON_S8):
    """The prior's macro-F1 on the test split, as session 8's comparisons stored it."""
    path = Path(comparison)
    where = "metrics > evidence_effect > prior > macro_f1"
    if not path.exists():
        return {"not_read": f"{rel(path)} does not exist here; combine reads it again", "path": where}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        m = data["metrics"][EI]["prior"]
        f1 = m["macro_f1"]
        e, lo, hi = float(f1["estimate"]), float(f1["low"]), float(f1["high"])
    except (KeyError, TypeError, ValueError) as err:
        return {"not_read": f"{rel(path)} has no prior macro-F1 at {where} ({err!r})", "path": where}
    return {"text": PRIOR_TEXT.format(e=e, l=lo, h=hi), "macro_f1": {"estimate": e, "low": lo, "high": hi}, "n": m.get("n"),
            "source": f"{rel(path)}, {where} (session 8's comparisons; no prediction file is read here)",
            "sha256": CMP.sha256_file(path)}


# ---------------------------------------------------------------- score

def guard(results=RESULTS):
    """v0.5's checks are test predictions made after the freeze (as calibrate_session8.py apply and compare_session8.py)."""
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {rel(results)}; v0.5's checks are read only after the freeze and testpreds")


def score(system, name, runs, full, calibration, calibration_key, records, out_path, comparison=COMPARISON_S8,
          swap_augmented=False, swap_averaged=False, single_pass_key=None, results=RESULTS, resamples=RESAMPLES,
          log_floor=LOG_FLOOR):
    """One system's checks. runs: {input: [raw prediction files]}; full: [raw prediction files on its full input];
    records: {input: records file}. Writes out_path and returns (the output, the exit code: 0, or 3 for the stop)."""
    if system not in SYSTEMS:
        raise SystemExit(f"--system is s8_ei or v05, not {system!r}")
    if system == "s8_ei" and (swap_augmented or swap_averaged or single_pass_key):
        raise SystemExit("--swap-augmented, --swap-averaged and --single-pass-key are for v0.5's EI specialist only")
    if single_pass_key and not swap_averaged:
        raise SystemExit("--single-pass-key applies only with --swap-averaged")
    if resamples < 1:
        raise SystemExit("--resamples must be at least 1")
    if not 0 < log_floor < 1:
        raise SystemExit("--log-floor must lie between 0 and 1")
    if system == "v05":
        guard(results)
    if system == "s8_ei" and any(len(runs[i]) != 3 for i in INPUTS):
        raise SystemExit("s8_ei has three runs on each input: " + ", ".join(f"{i} {len(runs[i])}" for i in INPUTS))
    if not full or any(not runs[i] for i in INPUTS):
        raise SystemExit("every input needs at least one run, and the full input at least one file")
    rec = {i: load_records(records[i]) for i in INPUTS}
    own = calibration_entry(calibration, calibration_key)
    single = calibration_entry(calibration, single_pass_key) if single_pass_key else own
    none_ids = set(rec["none"]["ids"])
    for i in ("title", "swap"):
        extra = [x for x in rec[i]["ids"] if x not in none_ids]
        if extra:
            raise SystemExit(f"{rel(rec[i]['path'])} holds {len(extra)} prompts that the none input lacks (the first: {extra[0]})")

    raw = {i: average_runs(runs[i], rec[i]) for i in INPUTS}
    full_labels = {"none": rec["none"]["labels"], "title": rec["title"]["labels"], "swap": EXCHANGE[rec["swap"]["labels"]]}
    full_raw = {i: full_input(full, rec[i]["ids"], full_labels[i]) for i in INPUTS}

    # calibrated probabilities on each input, and on the full input for the same prompts
    probs, full_probs = {}, {}
    check_cal = single if swap_averaged else own
    for i in ("none", "title"):
        probs[i], full_probs[i] = calibrated(raw[i], check_cal, log_floor), calibrated(full_raw[i], check_cal, log_floor)
    if swap_averaged:
        averaged = (full_raw["swap"] + exchange(raw["swap"])) / 2.0  # its answer to each prompt as written, with swap averaging
        full_probs["swap"], probs["swap"] = calibrated(averaged, own, log_floor), calibrated(exchange(averaged), own, log_floor)
    else:
        full_probs["swap"], probs["swap"] = calibrated(full_raw["swap"], own, log_floor), calibrated(raw["swap"], own, log_floor)
    ans = {i: probs[i].argmax(1) for i in INPUTS}
    full_ans = {i: full_probs[i].argmax(1) for i in INPUTS}

    def runs_text(k):
        return f"{k} run{'s' if k != 1 else ''}"

    if len(full) > 1:
        full_text = f"the raw probabilities of the {len(full)} files given, averaged"
    elif system == "s8_ei":
        full_text = "the raw probabilities of s8_ei-raw (the average of its three runs)"
    else:
        full_text = "the raw probabilities of the one file given"
    formed = {i: (f"the runs' raw probabilities averaged ({runs_text(len(runs[i]))}), then {check_cal['key']}'s calibration "
                  f"({check_cal['family']}); on the full input, {full_text}, then the same calibration"
                  + ("; single pass, without swap averaging" if swap_averaged else "")) for i in ("none", "title")}
    if swap_averaged:
        formed["swap"] = ("swap averaging (correction 3): on the full input, each prompt's raw answer averaged with the exchange of "
                          f"its raw answer to the exchanged prompt, then {own['key']}'s calibration ({own['family']}); on the "
                          "exchanged input, the exchange of that average, then the same calibration")
    else:
        formed["swap"] = (f"the runs' raw probabilities averaged ({runs_text(len(runs['swap']))}), then {own['key']}'s calibration "
                          f"({own['family']}); on the full input, {full_text}, then the same calibration")

    inputs = {}
    for i in ("none", "title"):
        y, p, a, fa = rec[i]["labels"], probs[i], ans[i], full_ans[i]
        est, low, high, used, arts = resampled(
            lambda idx: [M.accuracy(y[idx], p[idx]), M.macro_f1(y[idx], p[idx]), share(a[idx] == fa[idx], np.ones(len(idx), bool))],
            rec[i]["clusters"], resamples)
        inputs[i] = {"records": file_entry(rec[i]["path"], len(y)), "n": int(len(y)), "articles": int(arts),
                     "probabilities": formed[i],
                     "accuracy": interval(est[0], low[0], high[0]), "macro_f1": interval(est[1], low[1], high[1]),
                     "agreement_with_full_input": share_entry(est[2], low[2], high[2], used[2], resamples, np.sum(a == fa), len(a)),
                     "answers": answer_counts(a), "full_input_answers": answer_counts(fa)}
    inputs["title"]["left_out_no_title"] = len(none_ids - set(rec["title"]["ids"]))

    y, p, a, fa = rec["swap"]["labels"], probs["swap"], ans["swap"], full_ans["swap"]
    est, low, high, used, arts = resampled(
        lambda idx: [M.accuracy(y[idx], p[idx]), M.macro_f1(y[idx], p[idx])] + exchange_shares(fa[idx], a[idx]),
        rec["swap"]["clusters"], resamples)
    moved = fa != 2
    inputs["swap"] = {"records": file_entry(rec["swap"]["path"], len(y)), "n": int(len(y)), "articles": int(arts),
                      "left_out_same_text": len(none_ids - set(rec["swap"]["ids"])), "probabilities": formed["swap"],
                      "accuracy": interval(est[0], low[0], high[0]), "macro_f1": interval(est[1], low[1], high[1]),
                      "exchange_share": share_entry(est[2], low[2], high[2], used[2], resamples,
                                                    np.sum(moved & (a == EXCHANGE[fa])), np.sum(moved)),
                      "stay_share": share_entry(est[3], low[3], high[3], used[3], resamples, np.sum(~moved & (a == 2)), np.sum(~moved)),
                      "answers": answer_counts(a), "full_input_answers": answer_counts(fa)}
    # the comparator of the prompt as written is the exchanged record's intervention
    sub = np.array([bool(TIME_POINT.search(c)) for c in rec["swap"]["interventions"]])
    tp = {"rule": ("the comparator of the prompt as written (the exchanged record's intervention) matches TIME_POINT, ignoring case; "
                   "the subset's own articles resampled"),
          "pattern": TIME_POINT.pattern, "ignore_case": True, "n": int(sub.sum())}
    if sub.any():
        rows = np.flatnonzero(sub)
        fs, as_ = fa[rows], a[rows]
        est, low, high, used, arts = resampled(lambda idx: exchange_shares(fs[idx], as_[idx]), [rec["swap"]["clusters"][r] for r in rows],
                                               resamples)
        m2 = fs != 2
        tp.update({"articles": int(arts),
                   "exchange_share": share_entry(est[0], low[0], high[0], used[0], resamples, np.sum(m2 & (as_ == EXCHANGE[fs])), np.sum(m2)),
                   "stay_share": share_entry(est[1], low[1], high[1], used[1], resamples, np.sum(~m2 & (as_ == 2)), np.sum(~m2))})
    else:
        tp.update({"articles": 0, "exchange_share": None, "stay_share": None,
                   "note": "no swap prompt's comparator matches, so the subset's shares are not computed"})
    inputs["swap"]["time_point_or_baseline"] = tp

    # title minus none, on the prompts in both inputs, the same resamples for both
    pos_none = {x: k for k, x in enumerate(rec["none"]["ids"])}
    t_rows = np.arange(len(rec["title"]["ids"]))
    n_rows = np.array([pos_none[x] for x in rec["title"]["ids"]], dtype=np.int64)
    yc = rec["title"]["labels"]
    if not np.array_equal(rec["none"]["labels"][n_rows], yc):
        raise SystemExit("the title and none inputs give different answers for the same prompt")
    pt, pn = probs["title"][t_rows], probs["none"][n_rows]
    est, low, high, _, arts = resampled(
        lambda idx: [M.macro_f1(yc[idx], pt[idx]) - M.macro_f1(yc[idx], pn[idx]), M.macro_f1(yc[idx], pt[idx]), M.macro_f1(yc[idx], pn[idx])],
        rec["title"]["clusters"], resamples)
    d_est, d_low, d_high = num(est[0]), num(low[0]), num(high[0])
    title_minus_none = {"prompts": "the prompts in both inputs (every title prompt)", "n": int(len(yc)), "articles": int(arts),
                        "macro_f1_difference": interval(est[0], low[0], high[0]),
                        "title_macro_f1": interval(est[1], low[1], high[1]), "none_macro_f1": interval(est[2], low[2], high[2]),
                        "difference_reading": difference_reading(d_low, d_high), "excludes": excludes_text(d_est, d_low, d_high)}

    # readings
    f1_none, acc_swap = inputs["none"]["macro_f1"]["estimate"], inputs["swap"]["accuracy"]["estimate"]
    single_caveats = [CAVEAT_SINGLE_PASS] if swap_averaged else []
    swap_caveats = ([CAVEAT_AUGMENTED] if swap_augmented else []) + ([CAVEAT_CALIBRATION] if swap_averaged else [])
    readings = {
        "none": {"reading": none_reading(f1_none), "value": f1_none, "from": "macro-F1 with no article, its estimate",
                 "rule": NONE_RULE, "caveats": single_caveats},
        "title": {"reading": title_reading(d_low, d_high), "value": title_minus_none["macro_f1_difference"],
                  "from": "the 95 percent interval of the paired difference in macro-F1, title minus none", "rule": TITLE_RULE,
                  "caveats": single_caveats},
        "swap": {"reading": CONSISTENT if swap_averaged else swap_reading(acc_swap), "value": acc_swap,
                 "from": "accuracy against the exchanged answers, its estimate", "rule": SWAP_RULE, "caveats": swap_caveats}}
    if swap_averaged:
        readings["swap"]["read_as_a_reading_check"] = False
    stop = system == "s8_ei" and stops(f1_none)

    n_runs = {i: len(runs[i]) for i in INPUTS}
    if system == "s8_ei":
        description = "s8_ei (its three runs' raw probabilities averaged, then session 8's calibration)"
    else:
        k = max(n_runs.values())
        description = (f"v0.5's EI specialist, {name} ({'its ' + str(k) + ' runs' if k > 1 else 'its one run'}: raw probabilities "
                       f"averaged, then its own calibration, {own['family']})")
        if swap_averaged:
            description += ("; with swap averaging (correction 3) on its full input and the exchanged input, and its single pass on "
                            "none and title")
    cal_note = {"own": own, "applied_to": {"none": "single pass" if swap_averaged else "own", "title": "single pass" if swap_averaged else "own",
                                           "swap": "own"}}
    if swap_averaged:
        cal_note["single_pass"] = single
        cal_note["single_pass_note"] = ("--single-pass-key's calibration (the single-pass form's)" if single_pass_key else
                                        "no --single-pass-key was given, so the single pass is calibrated by its own calibration")
    out = {
        "plan": PLAN + (" and correction 3" if system == "v05" else ""), "written_utc": now(),
        "source": "src/eval/reading_checks_session9.py score", "system": system, "name": name, "description": description,
        "flags": {"swap_augmented": bool(swap_augmented), "swap_averaged": bool(swap_averaged)},
        "settings": {"split": "test", "resamples": resamples, "seed": SEED, "interval": "95 percent, percentile",
                     "resampling": ("articles (group.pmcid), sorted by PMCID as text, every prompt of a drawn article kept "
                                    "(compare_session8.py); each input over its own articles, the time-point subset over its "
                                    "own, title minus none over the articles of the prompts in both inputs, the same draws for both"),
                     "answers": list(ANSWERS), "answer": "the argmax of the calibrated probabilities",
                     "log_floor": log_floor, "line_tolerance": TIE},
        "calibration": cal_note,
        "files": {"runs": {i: [file_entry(p) for p in runs[i]] for i in INPUTS}, "full_input": [file_entry(p) for p in full]},
        "inputs": inputs, "title_minus_none": title_minus_none, "readings": readings,
        "stop": bool(stop), "stop_rule": STOP_RULE if system == "s8_ei" else "the stop applies to s8_ei only",
        "references": {"prior": prior_reference(comparison), "windisch_2026": WINDISCH},
        "contamination": CONTAMINATION,
        "notes": ["Every number here is computed from the answers, so a calibration matters only where it moves an argmax "
                  "(a temperature alone moves none).",
                  f"title leaves out {inputs['title']['left_out_no_title']} prompts whose article gives no title (correction 6); swap "
                  f"leaves out {inputs['swap']['left_out_same_text']} whose intervention and comparator are the same text."]}
    if stop:
        out["stop_numbers"] = {"none_macro_f1": inputs["none"]["macro_f1"], "line": STOP_ABOVE,
                               "text": (f"s8_ei's macro-F1 with no article is {f1_none:.4f}, above 0.85: the chain stops before the "
                                        "train stage")}
    write_json(out_path, out)
    summary = {"written": rel(out_path), "system": system, "name": name, "stop": out["stop"],
               "none_macro_f1": f1_none, "title_minus_none": title_minus_none["macro_f1_difference"],
               "swap_accuracy": acc_swap, "readings": {i: readings[i]["reading"] for i in INPUTS}}
    print(json.dumps(summary, indent=1), flush=True)
    if stop:
        print(f"STOP: {out['stop_numbers']['text']} (exit code {STOP_EXIT})", flush=True)
        return out, STOP_EXIT
    return out, 0


# ---------------------------------------------------------------- combine

def table_rows(system, d):
    rows = []
    for i in INPUTS:
        e = d["inputs"][i]
        stats = ["accuracy", "macro_f1"] + (["agreement_with_full_input"] if i != "swap" else ["exchange_share", "stay_share"])
        for s in stats:
            rows.append({"system": system, "name": d.get("name"), "input": i, "statistic": s, **{k: e[s].get(k) for k in ("estimate", "low", "high")},
                         "n": e[s].get("of", e["n"]), "articles": e["articles"]})
        if i == "swap" and e["time_point_or_baseline"].get("exchange_share") is not None:
            tp = e["time_point_or_baseline"]
            for s in ("exchange_share", "stay_share"):
                rows.append({"system": system, "name": d.get("name"), "input": "swap, comparator a time point or baseline", "statistic": s,
                             **{k: tp[s].get(k) for k in ("estimate", "low", "high")}, "n": tp[s].get("of"), "articles": tp["articles"]})
    t = d["title_minus_none"]
    for s in ("macro_f1_difference", "title_macro_f1", "none_macro_f1"):
        rows.append({"system": system, "name": d.get("name"), "input": "title minus none (prompts in both)", "statistic": s,
                     **{k: t[s].get(k) for k in ("estimate", "low", "high")}, "n": t["n"], "articles": t["articles"]})
    for i in INPUTS:
        rows.append({"system": system, "name": d.get("name"), "input": i, "statistic": "reading", "reading": d["readings"][i]["reading"],
                     "caveats": d["readings"][i].get("caveats", [])})
    return rows


def combine(checks_s8=OUT_S8, checks_v05=OUT_V05, comparison=COMPARISON_S8, out_path=OUT_COMBINED):
    """reading_checks.json for both systems (K4)."""
    data, files = {}, {}
    for key, path in (("s8_ei", checks_s8), ("v05", checks_v05)):
        path = Path(path)
        if not path.exists():
            raise SystemExit(f"{rel(path)} does not exist; combine needs both systems' checks")
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("system") != key:
            raise SystemExit(f"{rel(path)} holds the checks of {d.get('system')!r}, not {key}")
        data[key], files[key] = d, file_entry(path)
    prior = prior_reference(comparison)
    if "not_read" in prior and "not_read" not in data["s8_ei"]["references"]["prior"]:
        prior = data["s8_ei"]["references"]["prior"]
    same = {i: data["s8_ei"]["inputs"][i]["records"]["sha256"] == data["v05"]["inputs"][i]["records"]["sha256"] for i in ("none", "title")}
    notes = ["The readings are fixed in THE READING CHECKS and gate no release."]
    v = data["v05"]
    if v["flags"].get("swap_averaged"):
        notes.append("v0.5's swap check is " + CONSISTENT + "; " + CAVEAT_SINGLE_PASS + ".")
    if v["flags"].get("swap_augmented"):
        notes.append(CAVEAT_AUGMENTED + ".")
    if not all(same.values()):
        notes.append("The two systems' none or title records differ (SHA-256); see each system's files.")
    if data["s8_ei"].get("stop"):
        notes.append("The stop applied to s8_ei: " + data["s8_ei"].get("stop_numbers", {}).get("text", "") + ".")
    out = {"plan": PLAN_COMBINED, "written_utc": now(), "source": "src/eval/reading_checks_session9.py combine", "files": files,
           "same_none_and_title_records": same,
           "readings": {k: {i: {"reading": d["readings"][i]["reading"], "caveats": d["readings"][i].get("caveats", [])} for i in INPUTS}
                        for k, d in data.items()},
           "table": table_rows("s8_ei", data["s8_ei"]) + table_rows("v05", data["v05"]),
           "systems": data, "references": {"prior": prior, "windisch_2026": WINDISCH}, "contamination": CONTAMINATION, "notes": notes}
    write_json(out_path, out)
    print(json.dumps({"written": rel(out_path), "readings": out["readings"], "stop_s8_ei": bool(data["s8_ei"].get("stop"))}, indent=1),
          flush=True)
    return out


# ---------------------------------------------------------------- command line

def default_runs(input_name, results=RESULTS):
    """s8_ei's raw predictions on a check input, as the checks_s8 stage names them (system s8_ei_<k>_<input>)."""
    return [Path(results) / "checks_s8" / f"s8_ei_{k}_{input_name}-raw_{EI}_test.csv" for k in (1, 2, 3)]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="one system's checks (checks_s8.json or checks_v05.json)")
    s.add_argument("--system", choices=SYSTEMS, required=True)
    s.add_argument("--name", default=None, help="v0.5's EI specialist's name (s9_ei or s9_ei_wavg); s8_ei for s8_ei")
    for i in INPUTS:
        s.add_argument(f"--runs-{i}", nargs="+", type=Path, default=None, help=f"raw predictions on the {i} input, one file per run")
    s.add_argument("--full", nargs="+", type=Path, default=None, help="raw predictions on the system's full input (averaged if several)")
    s.add_argument("--calibration", type=Path, default=None, help="calibration_s8.json for s8_ei, calibration_s9.json for v05")
    s.add_argument("--calibration-key", default=None, help="the system's key in the calibration file (default s8_ei, or --name)")
    s.add_argument("--single-pass-key", default=None, help="with --swap-averaged: the single-pass form's key in the calibration file")
    for i in INPUTS:
        s.add_argument(f"--records-{i}", type=Path, default=CHECKS / i / f"{EI}_test.jsonl")
    s.add_argument("--comparison-s8", type=Path, default=COMPARISON_S8, help="session 8's comparison.json (the prior's macro-F1)")
    s.add_argument("--swap-augmented", action="store_true", help="v0.5's recipe includes swap augmentation")
    s.add_argument("--swap-averaged", action="store_true", help="v0.5 keeps swap averaging (correction 3)")
    s.add_argument("--results", type=Path, default=RESULTS, help="results/fiorillo_v0_5: default runs and outputs, and v0.5's guard")
    s.add_argument("--resamples", type=int, default=RESAMPLES)
    s.add_argument("--log-floor", type=float, default=LOG_FLOOR,
                   help="the floor of the averaged probabilities before the log: 1e-300 as calibrate_session8.py takes s8_ei's "
                        "average; 1e-12 for a calibration fitted on clipped printed probabilities")
    s.add_argument("--out", type=Path, default=None, help="default <results>/checks_s8.json or <results>/checks_v05.json")
    c = sub.add_parser("combine", help="reading_checks.json for both systems")
    c.add_argument("--checks-s8", type=Path, default=OUT_S8)
    c.add_argument("--checks-v05", type=Path, default=OUT_V05)
    c.add_argument("--comparison-s8", type=Path, default=COMPARISON_S8)
    c.add_argument("--out", type=Path, default=OUT_COMBINED)
    args = parser.parse_args(argv)
    if args.cmd == "combine":
        combine(args.checks_s8, args.checks_v05, args.comparison_s8, args.out)
        return 0
    records = {i: getattr(args, f"records_{i}") for i in INPUTS}
    if args.system == "s8_ei":
        if args.swap_augmented or args.swap_averaged or args.single_pass_key:
            parser.error("--swap-augmented, --swap-averaged and --single-pass-key are for --system v05")
        name = args.name or "s8_ei"
        runs = {i: getattr(args, f"runs_{i}") or default_runs(i, args.results) for i in INPUTS}
        full = args.full or [FULL_S8]
        calibration, key = args.calibration or CAL_S8, args.calibration_key or "s8_ei"
        out = args.out or Path(args.results) / "checks_s8.json"
    else:
        missing = [f"--runs-{i}" for i in INPUTS if not getattr(args, f"runs_{i}")] + ([] if args.full else ["--full"])
        missing += [] if args.name else ["--name"]
        if missing:
            parser.error("--system v05 needs " + ", ".join(missing))
        if args.single_pass_key and not args.swap_averaged:
            parser.error("--single-pass-key applies only with --swap-averaged")
        name = args.name
        runs = {i: getattr(args, f"runs_{i}") for i in INPUTS}
        full = args.full
        calibration, key = args.calibration or Path(args.results) / "calibration_s9.json", args.calibration_key or args.name
        out = args.out or Path(args.results) / "checks_v05.json"
    _, code = score(args.system, name, runs, full, calibration, key, records, out, args.comparison_s8, args.swap_augmented,
                    args.swap_averaged, args.single_pass_key, args.results, args.resamples, args.log_floor)
    return code


if __name__ == "__main__":
    sys.exit(main())

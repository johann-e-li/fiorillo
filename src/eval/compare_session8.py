"""Session 8's comparisons and v0.4's bar (docs/prompt_session8.md, COMPARISONS and POWER AND THE BAR).

Reads the test predictions (session 4's format) of every system named below, the records (row order and labels:
data/processed/session8/evidence_effect_test.jsonl for EI, with each prompt's PMCID, flag and accepted labels;
data/processed/public/ for PubMedQA and readmission) and results/fiorillo_v0_4/release_candidate.json (the ECE limit).
Metrics and reading functions are session 4's (src/eval/metrics.py; src/eval/compare.py, not changed), plus each answer's
F1 on EI. Every number has a 95 percent interval from 1,000 resamples (numpy's default_rng with the seed 20260927;
percentile method): EI resamples articles (integers(0, m, m) over the m articles sorted by PMCID as text, every prompt of
each drawn article kept, as power_session8.py draws them), PubMedQA and readmission resample items (metrics.bootstrap).
A difference "system minus reference" uses the same resamples for both (paired). Criteria 2 and 3 use 10,000 article
resamples of the same generator. The decision weights of macro_f1_tuned (readmission) are fitted on each system's
validation predictions (session 4's correction 3) and held fixed in the resamples. A difference whose interval includes
zero reads "no difference was detected".

Systems: EI: s8_ei, s8_ei_temperature and s8_ei-raw; each seed (s8_ei_<k>, s8_ei_<k>_temperature, s8_ei_<k>-raw);
gemma4_31b, gemma4_31b_temperature and gemma4_31b-raw; the prior (the training split's answer shares, computed here).
PubMedQA: s7_b3 (v0.4's specialist, session 7's calibration), its _temperature and -raw; arm_laya (v0.1); gemma4_31b and
its variants; s8_a3 and its variants; s7_a3 and s7_a3_temperature; the prior. Readmission: readmission_pool,
xgboost_tuned, catboost_tuned, lightgbm_tuned, lightgbm, blend_v0_1, arm_laya, s7_b3, s7_a3, s8_a3 and its variants, the
prior. A missing file is recorded and skipped.

Comparisons: G1 (the bar): s8_ei minus the prior and s8_ei minus gemma4_31b, every EI metric, and criteria 1 to 4.
G2 (seeds): s8_ei minus each seed alone (each calibrated by its own rule), and the spread of the three seeds' log loss.
G3 (published results): s8_ei's macro-F1 beside DeYoung et al. 2020's 0.780 and 0.851, and its lenient macro-F1 (a
prompt's gold label replaced by the prediction when the prediction is one of its accepted labels, group.accepted_labels,
as Buchmann et al. 2024 scored) beside their LED (0.7430 to 0.7649) and LongT5 (0.8154), each with what s8_ei's interval
excludes. G4 (PubMedQA): s7_b3 minus arm_laya; s7_b3 minus gemma4_31b; s8_a3 minus s7_a3; s8_a3 minus arm_laya; and the
check that s7_b3 was loaded as it was (the largest difference between s7_b3_timing's raw test probabilities and session
7's s7_b3-raw test file, from s7_b3_timing's run summary; correction 4). G5 (readmission): the pool minus lightgbm_tuned,
minus blend_v0_1 and minus arm_laya; xgboost_tuned and catboost_tuned minus lightgbm_tuned; s8_a3 minus s7_a3; s8_a3
minus arm_laya. G6: the hand-off (handoff_test.json, reported as it is). Sensitivity: G1 without every test prompt the
README flags (Incorrect, Questionable or Somewhat malformed), and G1 with s8_ei's raw and temperature-only predictions.
zeros.json: for every system, split and question read, the rows whose probability of the true answer prints as 0.

It refuses to run unless release_candidate.json and testpreds_log.json exist. Each block of results is saved in
models/fiorillo_v0_4/compare_cache/ under a key that includes the SHA-256 of its input files and of the records, so a run
that the linked shell ends resumes; --max-seconds stops it from starting a new block after that many seconds. Writes
results/fiorillo_v0_4/comparison.json, comparison.csv and zeros.json.

Usage: python3 src/eval/compare_session8.py [--max-seconds 150]
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
RESAMPLES_BAR = 10000
RESULTS = ROOT / "results" / "fiorillo_v0_4"
R7 = ROOT / "results" / "fiorillo_v0_3"
S4 = ROOT / "results" / "public_v0" / "predictions"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
CACHE = ROOT / "models" / "fiorillo_v0_4" / "compare_cache"
EI = "evidence_effect"
EVALUATIONS = (EI, "pubmedqa_answer", "readmit_window")
ANSWERS = ("increased", "decreased", "no_difference")
EI_METRICS = ("log_loss", "brier", "accuracy", "macro_f1", "f1_increased", "f1_decreased", "f1_no_difference", "ece")
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}
ECE_FLOOR = 0.05
SEEDS = ("s8_ei_1", "s8_ei_2", "s8_ei_3")
PUBLISHED = {"deyoung_2020_pipeline": (0.780, "DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa)"),
             "deyoung_2020_oracle": (0.851, "DeYoung et al. 2020, with the right evidence given")}
PUBLISHED_LENIENT = {"buchmann_2024_led_low": (0.7430, "Buchmann et al. 2024, LED, the lower end of its range"),
                     "buchmann_2024_led_high": (0.7649, "Buchmann et al. 2024, LED, the upper end of its range"),
                     "buchmann_2024_longt5": (0.8154, "Buchmann et al. 2024, LongT5")}
CONTAMINATION = (
    "Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PMC articles (EI's articles are "
    "PMC's), EI's repository (public since 2019), PubMedQA and the UCI data, which would favor both; EI's test labels are public "
    "(the project's site serves them to its leaderboard), so no evaluation of EI can be blind; Laya's, ModernBERT's and reader v1's "
    "notes and session 4's notes on Duan stand.")
RUNS = ("Each EI seed ran once and the three are reported apart as well as averaged; s8_a3 and s7_a3 differ in the keep rule and in "
        "the run itself, so s8_a3 minus s7_a3 cannot separate the two; retraining alone moved PubMedQA log loss by about 0.023 per run "
        "in this project (results/fiorillo_v0_2/retry_check.json).")
BIOD = ("BioDecision-4B's 70.6 was measured, by its model card, on the bigbio copy of EI, whose context is one evidence span and in "
        "which 73.4 percent of rows carry another prompt's span (the check of 30 September), so it is named and not compared. The "
        "published results counted the 6 test prompts of the empty article, which are left out here.")


def ei_systems():
    names = ["s8_ei", "s8_ei_temperature", "s8_ei-raw"]
    for s in SEEDS:
        names += [s, f"{s}_temperature", f"{s}-raw"]
    return tuple(names + ["gemma4_31b", "gemma4_31b_temperature", "gemma4_31b-raw", "prior"])


SYSTEMS = {
    EI: ei_systems(),
    "pubmedqa_answer": ("s7_b3", "s7_b3_temperature", "s7_b3-raw", "arm_laya", "gemma4_31b", "gemma4_31b_temperature", "gemma4_31b-raw",
                        "s8_a3", "s8_a3_temperature", "s8_a3-raw", "s7_a3", "s7_a3_temperature", "prior"),
    "readmit_window": ("readmission_pool", "xgboost_tuned", "catboost_tuned", "lightgbm_tuned", "lightgbm", "blend_v0_1", "arm_laya",
                       "s7_b3", "s7_a3", "s8_a3", "s8_a3_temperature", "s8_a3-raw", "prior"),
}


def directory_of(system, results=RESULTS):
    results = Path(results)
    base = results.parent
    s4 = base / "public_v0" / "predictions"
    r7 = base / "fiorillo_v0_3" / "predictions"
    if system in ("arm_laya", "lightgbm", "prior"):
        return s4
    if system.startswith(("s7_b3", "s7_a3")) or system in ("lightgbm_tuned", "blend_v0_1"):
        return r7
    if system == "gemma4_31b-raw":
        return results / "large_model"
    return results / "predictions"


def metric_names(ev):
    return EI_METRICS if ev == EI else C.metric_names(ev)


def f1_of(labels, probs, c):
    preds = probs.argmax(1)
    tp = np.sum((preds == c) & (labels == c))
    fp = np.sum((preds == c) & (labels != c))
    fn = np.sum((preds != c) & (labels == c))
    return 0.0 if tp == 0 else float(2 * tp / (2 * tp + fp + fn))


def scores(names, labels, probs, weights=None):
    out = []
    for m in names:
        if m.startswith("f1_"):
            out.append(f1_of(labels, probs, ANSWERS.index(m[3:])))
        else:
            out.append(C.METRICS[m](labels, probs, weights))
    return np.array(out, dtype=float)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        if self.computed_now and self.deadline is not None and time.time() > self.deadline:
            self.stopped = True  # every run computes at least one block, so each run makes progress
            return None
        value = compute()
        self.computed_now += 1
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(value) + "\n", encoding="utf-8")
        tmp.replace(path)
        return value


def floats(a):
    return [None if (x is None or not np.isfinite(x)) else float(x) for x in np.atleast_1d(np.asarray(a, dtype=float))]


# ---------------------------------------------------------------- resampling

def article_rows(clusters):
    """The rows of each article, articles sorted by PMCID as text."""
    articles = sorted(set(clusters))
    rows = {a: [] for a in articles}
    for i, c in enumerate(clusters):
        rows[c].append(i)
    return [np.array(rows[a], dtype=np.int64) for a in articles]


def cluster_bootstrap(stat, rows_of, n, resamples=RESAMPLES, seed=SEED):
    """Estimate and 95 percent percentile interval of stat(row indices) over article resamples; non-finite values left out."""
    rng = np.random.default_rng(seed)
    first = np.atleast_1d(np.asarray(stat(np.arange(n)), dtype=float))
    m = len(rows_of)
    values = np.array([np.atleast_1d(np.asarray(stat(np.concatenate([rows_of[j] for j in rng.integers(0, m, m)])), dtype=float))
                       for _ in range(resamples)])
    low, high = np.full(len(first), np.nan), np.full(len(first), np.nan)
    for j in range(len(first)):
        v = values[:, j][np.isfinite(values[:, j])]
        if len(v):
            low[j], high[j] = np.percentile(v, [2.5, 97.5])
    return first, low, high


def bar_interval(labels, a, b, rows_of, resamples=RESAMPLES_BAR, seed=SEED):
    """The paired log-loss difference a minus b over article resamples (criteria 2 and 3)."""
    la = -np.log(np.clip(a[np.arange(len(labels)), labels], M.EPS, None))
    lb = -np.log(np.clip(b[np.arange(len(labels)), labels], M.EPS, None))
    d = la - lb
    sums = np.array([d[r].sum() for r in rows_of])
    counts = np.array([len(r) for r in rows_of])
    rng = np.random.default_rng(seed)
    m = len(rows_of)
    vals = np.empty(resamples)
    for k in range(resamples):
        draw = rng.integers(0, m, m)
        vals[k] = sums[draw].sum() / counts[draw].sum()
    return {"estimate": float(d.mean()), "low": float(np.percentile(vals, 2.5)), "high": float(np.percentile(vals, 97.5)),
            "resamples": resamples, "n": int(len(d)), "articles": m}


def reading(metric, low, high):
    if low is None or high is None:
        return "not computed"
    if low <= 0 <= high:
        return "no difference was detected"
    lower = high < 0
    if metric in LOWER_IS_BETTER:
        return "lower (better)" if lower else "higher (worse)"
    return "lower (worse)" if lower else "higher (better)"


def excludes(low, high, value):
    if low is None or high is None:
        return "not computed"
    if value < low:
        return "excluded: below the interval"
    if value > high:
        return "excluded: above the interval"
    return "inside the interval"


# ---------------------------------------------------------------- data

def load_aligned(path, ref_ids, ref_labels):
    ids, labels, probs = M.read_predictions(path)
    position = {i: n for n, i in enumerate(ids)}
    missing = [i for i in ref_ids if i not in position]
    if missing or len(position) != len(ref_ids):
        raise SystemExit(f"{Path(path).name}: {len(position)} rows, {len(ref_ids)} records, {len(missing)} IDs missing")
    order = np.array([position[i] for i in ref_ids])
    if not np.array_equal(labels[order], ref_labels):
        raise SystemExit(f"{Path(path).name}: labels differ from the records")
    return probs[order]


def ei_prior(records_ei):
    labels = [json.loads(line)["label"] for line in open(Path(records_ei) / f"{EI}_train.jsonl", encoding="utf-8") if line.strip()]
    counts = np.bincount(labels, minlength=3).astype(float)
    return counts / counts.sum(), counts.astype(int).tolist()


def load_data(results, records_public, records_ei, missing, digests):
    data = {}
    ei_rows = [json.loads(line) for line in open(Path(records_ei) / f"{EI}_test.jsonl", encoding="utf-8") if line.strip()]
    prior_shares, prior_counts = ei_prior(records_ei)
    for ev in EVALUATIONS:
        rows = ei_rows if ev == EI else C.reference_rows(records_public, ev)
        entry = {"ids": [r["id"] for r in rows], "labels": np.array([r["label"] for r in rows]), "probs": {}, "weights": {}}
        if ev == EI:
            entry["clusters"] = [r["group"]["pmcid"] for r in rows]
            entry["flags"] = [r["group"].get("flag") or "none" for r in rows]
            entry["accepted"] = [set(r["group"]["accepted_labels"]) for r in rows]
            entry["rows_of"] = article_rows(entry["clusters"])
        for system in SYSTEMS[ev]:
            if system == "prior" and ev == EI:
                entry["probs"]["prior"] = np.tile(prior_shares, (len(rows), 1))
                digests[("prior", ev)] = sha256_text(json.dumps(prior_counts))
                entry["prior_counts"] = prior_counts
                continue
            path = directory_of(system, results) / f"{system}_{ev}_test.csv"
            if path.exists():
                entry["probs"][system] = load_aligned(path, entry["ids"], entry["labels"])
                digests[(system, ev)] = sha256_file(path)
            else:
                missing.append(f"{system} on {ev} ({path})")
        data[ev] = entry
    return data


def zeros_table(results):
    zeros = {}
    for ev in EVALUATIONS:
        for system in SYSTEMS[ev]:
            if system == "prior" and ev == EI:
                continue
            for split in ("validation", "test"):
                path = directory_of(system, results) / f"{system}_{ev}_{split}.csv"
                if path.exists():
                    ids, labels, probs = M.read_predictions(path)
                    zeros.setdefault(system, {}).setdefault(split, {})[ev] = {
                        "rows": len(ids), "true_answer_printed_zero": int((probs[np.arange(len(labels)), labels] == 0).sum())}
    return zeros


def fit_weights(data, results, records_public, missing):
    ev = "readmit_window"
    val_rows = C.reference_rows(records_public, ev, "validation")
    val_ids, val_labels = [r["id"] for r in val_rows], np.array([r["label"] for r in val_rows])
    fitted = {}
    for system in data[ev]["probs"]:
        path = directory_of(system, results) / f"{system}_{ev}_validation.csv"
        if not path.exists():
            missing.append(f"{system} validation on {ev} ({path}); macro_f1_tuned not computed")
            continue
        weights, val_f1 = M.decision_weights(val_labels, load_aligned(path, val_ids, val_labels))
        data[ev]["weights"][system] = weights
        fitted[system] = {"weights": [C.num(w) for w in weights], "validation_macro_f1": C.num(val_f1)}
    return fitted


# ---------------------------------------------------------------- the run

def run(results=RESULTS, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI, cache_dir=CACHE, max_seconds=None,
        resamples=RESAMPLES, resamples_bar=RESAMPLES_BAR):
    t0 = time.time()
    results = Path(results)
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (results / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; the comparisons read test predictions, made only after the freeze")
    rc = json.loads((results / "release_candidate.json").read_text(encoding="utf-8"))
    cache = Cache(cache_dir, None if max_seconds is None else t0 + max_seconds)
    missing, digests = [], {}
    data = load_data(results, records_public, records_ei, missing, digests)
    fitted = fit_weights(data, results, records_public, missing)
    records_sha = {"ei_test": sha256_file(Path(records_ei) / f"{EI}_test.jsonl"), "ei_train": sha256_file(Path(records_ei) / f"{EI}_train.jsonl"),
                   **{f"{q}_{s}": sha256_file(Path(records_public) / f"{q}_{s}.jsonl") for q in ("pubmedqa_answer", "readmit_window")
                      for s in ("test",)},
                   "readmit_window_validation": sha256_file(Path(records_public) / "readmit_window_validation.jsonl")}
    out = {"created": dt.date.today().isoformat(), "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "source": "src/eval/compare_session8.py (docs/prompt_session8.md, COMPARISONS and POWER AND THE BAR)",
           "release_candidate_time_utc": rc.get("time_utc"), "ece_limits": rc.get("ece_limits"),
           "settings": {"split": "test", "resamples": resamples, "resamples_bar": resamples_bar, "seed": SEED,
                        "interval": "95 percent, percentile, paired", "ei_resampling": "articles, sorted by PMCID as text",
                        "other_resampling": "items", "ece_bins": 15},
           "contamination": CONTAMINATION, "runs": RUNS, "biodecision": BIOD,
           "ei_prior": {"counts": data[EI].get("prior_counts"), "shares": floats(data[EI]["probs"]["prior"][0])},
           "systems": {e: list(data[e]["probs"]) for e in EVALUATIONS}, "missing_predictions": missing,
           "decision_weights_readmit_window": fitted, "metrics": {}, "comparisons": [], "bar_intervals": {}, "g2_spread": {},
           "g3": {}, "g4_load_check": {}, "g6": {}, "release_bar": {}, "sensitivity": {}}
    table = []

    def boot(ev, stat, rows=None):
        entry = data[ev]
        if ev == EI:
            rows_of = entry["rows_of"] if rows is None else article_rows([entry["clusters"][i] for i in rows])
            n = len(entry["labels"]) if rows is None else len(rows)
            return cluster_bootstrap(stat, rows_of, n, resamples, SEED)
        return M.bootstrap(stat, len(entry["labels"]), resamples, SEED)

    # every system, every metric
    for ev in EVALUATIONS:
        names, labels = metric_names(ev), data[ev]["labels"]
        out["metrics"][ev] = {}
        for system, probs in data[ev]["probs"].items():
            w = data[ev]["weights"].get(system)
            key = ["metrics", ev, system, digests[(system, ev)], list(names), resamples, SEED, floats(w) if w is not None else None, records_sha]
            res = cache.get(key, lambda: [floats(v) for v in boot(ev, lambda idx: scores(names, labels[idx], probs[idx], w))])
            if res is None:
                continue
            est, low, high = res
            out["metrics"][ev][system] = {"n": len(labels), **{m: {"estimate": est[j], "low": low[j], "high": high[j]} for j, m in enumerate(names)}}
            for j, m in enumerate(names):
                table.append({"section": "metrics", "evaluation": ev, "system": system, "metric": m, "estimate": est[j], "low": low[j],
                              "high": high[j], "n": len(labels)})

    def point(ev, system, metric):
        return out["metrics"].get(ev, {}).get(system, {}).get(metric, {}).get("estimate")

    def compare(section, ev, system, reference, names, note_text="", subset=None, subset_name=None):
        entry = data[ev]
        if system not in entry["probs"] or reference not in entry["probs"]:
            missing.append(f"{section}: {system} or {reference} missing on {ev}")
            return
        rows = np.arange(len(entry["labels"])) if subset is None else np.asarray(subset)
        labels, a, b = entry["labels"][rows], entry["probs"][system][rows], entry["probs"][reference][rows]
        wa, wb = entry["weights"].get(system), entry["weights"].get(reference)
        key = ["diff", ev, system, reference, digests[(system, ev)], digests[(reference, ev)], list(names), resamples, SEED,
               floats(wa) if wa is not None else None, floats(wb) if wb is not None else None, records_sha, subset_name]
        res = cache.get(key, lambda: [floats(v) for v in boot(
            ev, lambda idx: scores(names, labels[idx], a[idx], wa) - scores(names, labels[idx], b[idx], wb), None if subset is None else rows)])
        if res is None:
            return
        est, low, high = res
        for j, m in enumerate(names):
            sys_est = scores((m,), labels, a, wa)[0] if subset is not None else point(ev, system, m)
            ref_est = scores((m,), labels, b, wb)[0] if subset is not None else point(ev, reference, m)
            record = {"section": section, "evaluation": ev, "system": system, "reference": reference, "metric": m,
                      "difference": {"estimate": est[j], "low": low[j], "high": high[j]}, "reading": reading(m, low[j], high[j]),
                      "n": int(len(labels)), "system_estimate": floats(sys_est)[0], "reference_estimate": floats(ref_est)[0], "note": note_text}
            if subset_name:
                record["subset"] = subset_name
            out["comparisons"].append(record)
            table.append({"section": section, "evaluation": ev, "system": system, "reference": reference, "group": subset_name or "",
                          "metric": m, "estimate": est[j], "low": low[j], "high": high[j], "n": int(len(labels)),
                          "system_estimate": record["system_estimate"], "reference_estimate": record["reference_estimate"],
                          "note": f"{record['reading']}; {note_text}" if note_text else record["reading"]})

    # G1: the bar's comparisons, every EI metric
    compare("G1", EI, "s8_ei", "prior", EI_METRICS, "s8_ei minus the prior (the training split's answer shares)")
    compare("G1", EI, "s8_ei", "gemma4_31b", EI_METRICS, "s8_ei minus gemma4_31b, both calibrated on validation")
    for ref in ("prior", "gemma4_31b"):
        e = data[EI]
        if "s8_ei" in e["probs"] and ref in e["probs"]:
            key = ["bar", ref, digests[("s8_ei", EI)], digests[(ref, EI)], resamples_bar, SEED, records_sha]
            res = cache.get(key, lambda: bar_interval(e["labels"], e["probs"]["s8_ei"], e["probs"][ref], e["rows_of"], resamples_bar, SEED))
            if res is not None:
                out["bar_intervals"][f"s8_ei_minus_{ref}"] = res

    # G2: the seeds
    for s in SEEDS:
        compare("G2", EI, "s8_ei", s, EI_METRICS, f"the average minus {s} alone (each calibrated by its own rule)")
    losses = {s: point(EI, s, "log_loss") for s in SEEDS if point(EI, s, "log_loss") is not None}
    if len(losses) == len(SEEDS):
        v = np.array(list(losses.values()))
        out["g2_spread"] = {"log_loss": losses, "max_minus_min": float(v.max() - v.min()), "standard_deviation": float(np.std(v, ddof=1))}

    # G3: published results
    e = data[EI]
    if "s8_ei" in e["probs"]:
        probs, labels = e["probs"]["s8_ei"], e["labels"]
        preds = probs.argmax(1)
        lenient = np.array([p if p in acc else y for p, acc, y in zip(preds, e["accepted"], labels)])

        def lenient_f1(idx):
            return M.macro_f1(lenient[idx], probs[idx])

        key = ["g3", digests[("s8_ei", EI)], resamples, SEED, records_sha]
        res = cache.get(key, lambda: [floats(v) for v in boot(EI, lambda idx: np.array([M.macro_f1(labels[idx], probs[idx]), lenient_f1(idx)]))])
        if res is not None:
            est, low, high = res
            out["g3"] = {"macro_f1": {"estimate": est[0], "low": low[0], "high": high[0]},
                         "lenient_macro_f1": {"estimate": est[1], "low": low[1], "high": high[1],
                                              "rule": "gold replaced by the prediction when the prediction is one of the prompt's accepted labels"},
                         "published": {k: {"value": v, "source": s, "s8_ei": excludes(low[0], high[0], v)} for k, (v, s) in PUBLISHED.items()},
                         "published_lenient": {k: {"value": v, "source": s, "s8_ei": excludes(low[1], high[1], v)}
                                               for k, (v, s) in PUBLISHED_LENIENT.items()},
                         "lenient_changed_prompts": int((lenient != labels).sum()), "note": BIOD}

    # G4: PubMedQA
    names = C.metric_names("pubmedqa_answer")
    for system, reference, note_text in (("s7_b3", "arm_laya", "v0.4 (s7_b3) minus v0.1"), ("s7_b3", "gemma4_31b", "v0.4 (s7_b3) minus gemma4_31b"),
                                         ("s8_a3", "s7_a3", "the last weights minus the kept checkpoint, one run each; " + RUNS),
                                         ("s8_a3", "arm_laya", "s8_a3 minus v0.1")):
        compare("G4", "pubmedqa_answer", system, reference, names, note_text)
    timing = results / "run_summaries" / "s7_b3_timing_test_run_summary.json"
    if timing.exists():
        t = json.loads(timing.read_text(encoding="utf-8"))
        out["g4_load_check"] = {"file": str(timing.name), "max_abs_difference_vs_session7": t.get("max_abs_difference_vs_session7"),
                                "note": ("s7_b3_timing's raw test probabilities against session 7's s7_b3-raw test file (correction 4). Session 7 "
                                         "predicted both questions together at batch size 16 and s7_b3_timing PubMedQA alone, so the padding "
                                         "differs; under bfloat16 a difference of about 1e-3 is padding, not a loading fault (derived)")}
    else:
        missing.append(f"G4's load check: {timing} missing")

    # G5: readmission
    names = C.metric_names("readmit_window")
    for system, reference in (("readmission_pool", "lightgbm_tuned"), ("readmission_pool", "blend_v0_1"), ("readmission_pool", "arm_laya"),
                              ("xgboost_tuned", "lightgbm_tuned"), ("catboost_tuned", "lightgbm_tuned"), ("s8_a3", "s7_a3"), ("s8_a3", "arm_laya")):
        compare("G5", "readmit_window", system, reference, names, RUNS if system == "s8_a3" else "")

    # G6: the hand-off
    handoff = results / "handoff_test.json"
    if handoff.exists():
        out["g6"] = {"file": handoff.name, "sha256": sha256_file(handoff),
                     "note": "the hand-off's numbers are in handoff_test.json (handoff_session8.py), reported as they are"}
    else:
        missing.append(f"G6: {handoff} missing")

    # Sensitivity
    flagged = [i for i, f in enumerate(data[EI]["flags"]) if f != "none"]
    unflagged = [i for i, f in enumerate(data[EI]["flags"]) if f == "none"]
    out["sensitivity"]["flagged_test_prompts"] = len(flagged)
    for ref in ("prior", "gemma4_31b"):
        compare("G1_sensitivity", EI, "s8_ei", ref, EI_METRICS, "without every test prompt the README flags", unflagged, "unflagged")
    for variant in ("s8_ei-raw", "s8_ei_temperature"):
        for ref in ("prior", "gemma4_31b"):
            compare("G1_sensitivity", EI, variant, ref, EI_METRICS, f"{variant} in place of s8_ei")

    if cache.stopped:
        print(json.dumps({"status": "stopped; run again to resume", "seconds": round(time.time() - t0, 1),
                          "cached_blocks": len(list(Path(cache_dir).glob("*.json")))}))
        return None

    out["release_bar"] = release_bar(out, rc)
    for k, c in out["release_bar"]["criteria"].items():
        table.append({"section": "release_bar", "metric": f"criterion {k}", "note": f"{c['result']}: {c['rule']}"
                      + (f" ({c['reason']})" if c["reason"] else "") + f"; values {json.dumps(c['values'])}"})
    table.append({"section": "release_bar", "metric": "overall", "note": f"{out['release_bar']['overall']}: {out['release_bar']['outcome']}"})
    table.append({"section": "note", "metric": "contamination", "note": CONTAMINATION})
    results.mkdir(parents=True, exist_ok=True)
    (results / "comparison.json").write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    with open(results / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=C.CSV_COLUMNS)
        w.writeheader()
        for r in table:
            w.writerow({c: C.fmt(r.get(c)) for c in C.CSV_COLUMNS})
    (results / "zeros.json").write_text(json.dumps({"plan": "docs/prompt_session8.md, COMPARISONS",
                                                    "rows_whose_true_answer_prints_as_0": zeros_table(results)}, indent=1) + "\n",
                                        encoding="utf-8")
    print(json.dumps({"status": "written", "release_bar": out["release_bar"]["overall"], "outcome": out["release_bar"]["outcome"],
                      "missing": len(missing), "seconds": round(time.time() - t0, 1)}, indent=1))
    return out


def release_bar(out, rc):
    crit = {}
    m = out["metrics"].get(EI, {})
    limit = ((rc.get("ece_limits") or {}).get(EI) or {}).get("limit")
    reasons = []
    if "s8_ei" not in m:
        reasons.append("s8_ei's test predictions are missing")
    if limit is None:
        reasons.append("release_candidate.json holds no ECE limit for evidence_effect")
    ece = m.get("s8_ei", {}).get("ece", {}).get("estimate")
    ok = not reasons and ece is not None and ece <= float(limit)
    crit["1"] = {"rule": "s8_ei's ECE at most the limit (the larger of 0.05 and the 95th percentile of a calibrated model's ECE at n 1,218, "
                         "fixed in release_candidate.json before any test prediction)",
                 "values": {"ece": ece, "limit": limit}, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    for k, ref, rule in (("2", "prior", "s8_ei's log loss below the prior's, with the 95 percent interval of the difference (s8_ei minus "
                                         "the prior; 10,000 article resamples) wholly below zero"),
                         ("3", "gemma4_31b", "s8_ei's log loss below gemma4_31b's (both calibrated on validation), with the 95 percent "
                                             "interval of the difference (10,000 article resamples) wholly below zero")):
        iv = out["bar_intervals"].get(f"s8_ei_minus_{ref}")
        reasons = [] if iv else [f"s8_ei's or {ref}'s test predictions are missing"]
        ok = bool(iv) and iv["estimate"] < 0 and iv["high"] < 0
        crit[k] = {"rule": rule, "values": iv, "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    a, b = m.get("s8_ei", {}).get("macro_f1", {}).get("estimate"), m.get("gemma4_31b", {}).get("macro_f1", {}).get("estimate")
    reasons = [] if (a is not None and b is not None) else ["s8_ei's or gemma4_31b's test predictions are missing"]
    ok = not reasons and a >= b
    crit["4"] = {"rule": "s8_ei's macro-F1 not below gemma4_31b's point estimate", "values": {"s8_ei": a, "gemma4_31b": b},
                 "result": "PASS" if ok else "FAIL", "reason": "; ".join(reasons)}
    overall = "PASS" if all(c["result"] == "PASS" for c in crit.values()) else "FAIL"
    failed = [k for k, c in crit.items() if c["result"] != "PASS"]
    return {"system": "s8_ei on evidence_effect (v0.4)", "criteria": crit, "overall": overall,
            "outcome": ("every criterion passes; releasing v0.4 is my decision after the closeout" if overall == "PASS"
                        else "v0.4 release bar not met: " + ", ".join(f"criterion {k}" for k in failed))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--records-ei", type=Path, default=RECORDS_EI)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--resamples-bar", type=int, default=RESAMPLES_BAR)
    args = parser.parse_args(argv)
    run(args.results, args.records_public, args.records_ei, args.cache, args.max_seconds, args.resamples, args.resamples_bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())

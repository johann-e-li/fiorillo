"""The hand-off (docs/prompt_session8.md, HAND-OFF; idea I-13 of docs/ideas_register.md, in full, without Laya's escalation
head: I-31). Secondary and descriptive: it is never part of the bar.

For pubmedqa_answer (Fiorillo: s7_b3 as calibrated in session 7, results/fiorillo_v0_3/predictions/s7_b3_pubmedqa_answer_<split>.csv)
and evidence_effect (Fiorillo: s8_ei, results/fiorillo_v0_4/predictions/s8_ei_evidence_effect_<split>.csv), with the large
model's calibrated answers (results/fiorillo_v0_4/predictions/gemma4_31b_<qid>_<split>.csv): items are ranked by
Fiorillo's top calibrated probability, least confident first, ties broken by record ID as text. At hand-off rates 0, 10,
20, 30 and 50 percent, the floor(rate x n) least confident items are answered by gemma4_31b and the rest by Fiorillo.

Reported, each with a 95 percent interval (1,000 resamples; numpy's default_rng with the seed 20260927; EI resamples
articles, integers(0, m, m) over the m articles sorted by PMCID as text, keeping every prompt of each drawn article;
PubMedQA resamples items, integers(0, n, n)), the items ranked again inside each resample (copies of one item ordered by
their place in the resample): the combined answers' accuracy, macro-F1 and log loss; on the handed-off items, Fiorillo's
and the large model's accuracy; the large model alone (accuracy, macro-F1, log loss); and the cost per 1,000 questions
at the pod's hourly rate (--hourly-rate), from the forward-pass GPU seconds per item: Fiorillo answers every item and the
large model the handed-off share, so the cost is 1,000 x (Fiorillo's seconds per item + rate x the large model's) / 3,600
x the hourly rate. The seconds: s8_ei, the sum of its three seeds' (results/fiorillo_v0_4/run_summaries/
s8_ei_<k>_<split>_run_summary.json); PubMedQA, s7_b3_timing's (test only; there is no session 8 run of s7_b3 on
validation, so the validation cost for PubMedQA is left empty; correction 4); the large model, gemma4_31b_timing.json's for
the split. The EI seeds' validation passes can share the GPU with arm A, which handoff_validation.json says beside its costs.
Validation (step 12): the items marked calibrate (every EI validation prompt; PubMedQA's 150). Test (step 15): every
test item, only once release_candidate.json and testpreds_log.json exist.

Usage: python src/eval/handoff_session8.py --split validation --hourly-rate 3.52
       python3 src/eval/handoff_session8.py --split test --hourly-rate 3.52
"""
import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval import metrics as M  # noqa: E402

SEED = 20260927
RESAMPLES = 1000
RATES = (0.0, 0.1, 0.2, 0.3, 0.5)
RESULTS = ROOT / "results" / "fiorillo_v0_4"
RESULTS7 = ROOT / "results" / "fiorillo_v0_3"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
LARGE = "gemma4_31b"
QUESTIONS = {"pubmedqa_answer": {"fiorillo": "s7_b3", "fiorillo_dir": "fiorillo_v0_3", "resample": "items"},
             "evidence_effect": {"fiorillo": "s8_ei", "fiorillo_dir": "fiorillo_v0_4", "resample": "articles"}}
SEEDS = ("s8_ei_1", "s8_ei_2", "s8_ei_3")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def id_ranks(ids):
    rank = np.empty(len(ids), dtype=np.int64)
    rank[np.argsort(np.array(ids, dtype=object), kind="stable")] = np.arange(len(ids))
    return rank


def combine(tie, labels, fio, big, rate):
    """The combined answers at one rate: (combined probabilities, handed-off mask). Least confident first; tie is a
    numeric key that breaks ties (lower first)."""
    n = len(labels)
    conf = fio.max(axis=1)
    order = np.lexsort((tie, conf))  # least confident first
    k = int(np.floor(rate * n + 1e-9))
    handed = np.zeros(n, dtype=bool)
    handed[order[:k]] = True
    return np.where(handed[:, None], big, fio), handed


def point(tie, labels, fio, big):
    out = {}
    for rate in RATES:
        probs, handed = combine(tie, labels, fio, big, rate)
        key = f"rate_{int(round(rate * 100))}"
        out[key] = {"handed_off": int(handed.sum()), "accuracy": M.accuracy(labels, probs), "macro_f1": M.macro_f1(labels, probs),
                    "log_loss": M.log_loss(labels, probs),
                    "handed_off_fiorillo_accuracy": M.accuracy(labels[handed], fio[handed]) if handed.any() else None,
                    "handed_off_large_model_accuracy": M.accuracy(labels[handed], big[handed]) if handed.any() else None}
    out["large_model_alone"] = {"accuracy": M.accuracy(labels, big), "macro_f1": M.macro_f1(labels, big), "log_loss": M.log_loss(labels, big)}
    out["fiorillo_alone"] = {"accuracy": M.accuracy(labels, fio), "macro_f1": M.macro_f1(labels, fio), "log_loss": M.log_loss(labels, fio)}
    return out


def resample_indices(n, clusters, resamples, seed):
    """Row indices of each resample: articles (clusters sorted as text) or items."""
    rng = np.random.default_rng(seed)
    if clusters is None:
        for _ in range(resamples):
            yield rng.integers(0, n, n)
        return
    articles = sorted(set(clusters))
    rows_of = {a: [] for a in articles}
    for i, c in enumerate(clusters):
        rows_of[c].append(i)
    rows_of = [np.array(rows_of[a], dtype=np.int64) for a in articles]
    m = len(articles)
    for _ in range(resamples):
        yield np.concatenate([rows_of[j] for j in rng.integers(0, m, m)])


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flatten(v, f"{prefix}{k}/"))
        else:
            out[f"{prefix}{k}"] = v
    return out


def analyze(ids, labels, fio, big, clusters=None, resamples=RESAMPLES, seed=SEED):
    n = len(ids)
    rank = id_ranks(ids)
    est = point(rank, labels, fio, big)
    flat_est = flatten(est)
    draws = {k: [] for k in flat_est}
    for idx in resample_indices(n, clusters, resamples, seed):
        tie = rank[idx] * n + np.arange(len(idx))  # copies of one item ordered by their place in the resample
        for k, v in flatten(point(tie, labels[idx], fio[idx], big[idx])).items():
            if v is not None and not k.endswith("handed_off"):
                draws[k].append(v)
    out = {}
    for k, v in flat_est.items():
        if k.endswith("handed_off"):
            out[k] = v
            continue
        vals = np.array(draws[k], dtype=float)
        lo, hi = (np.percentile(vals, [2.5, 97.5]) if len(vals) else (None, None))
        out[k] = {"estimate": v, "low": None if lo is None else float(lo), "high": None if hi is None else float(hi)}
    nested = {}
    for k, v in out.items():
        a, b = k.split("/", 1)
        nested.setdefault(a, {})[b] = v
    return nested


def aligned(fio_path, big_path, keep_ids=None):
    """(ids, labels, Fiorillo's and the large model's probabilities), rows in the large model's file order restricted to
    keep_ids; both files must hold those rows with the same labels."""
    f_ids, f_labels, f_probs = M.read_predictions(fio_path)
    b_ids, b_labels, b_probs = M.read_predictions(big_path)
    ids = [i for i in b_ids if keep_ids is None or i in keep_ids]
    if keep_ids is not None and set(ids) != set(keep_ids):
        raise SystemExit(f"{Path(big_path).name} lacks {len(set(keep_ids) - set(ids))} of the items")
    fpos, bpos = {i: k for k, i in enumerate(f_ids)}, {i: k for k, i in enumerate(b_ids)}
    missing = [i for i in ids if i not in fpos]
    if missing:
        raise SystemExit(f"{Path(fio_path).name} lacks {len(missing)} items, for example {missing[:3]}")
    fo, bo = np.array([fpos[i] for i in ids]), np.array([bpos[i] for i in ids])
    if not np.array_equal(f_labels[fo], b_labels[bo]):
        raise SystemExit("labels differ between Fiorillo's and the large model's files")
    return ids, b_labels[bo], f_probs[fo], b_probs[bo]


def seconds_per_item(results, qid, split):
    """(Fiorillo's seconds per item, the large model's, notes)."""
    notes = []
    fio = None
    if qid == "evidence_effect":
        total = 0.0
        for s in SEEDS:
            path = Path(results) / "run_summaries" / f"{s}_{split}_run_summary.json"
            if not path.exists():
                notes.append(f"{path.name} missing")
                total = None
                break
            total += json.loads(path.read_text(encoding="utf-8"))[f"timing_{split}"]["seconds_per_item"]
        fio = total
    else:
        path = Path(results) / "run_summaries" / f"s7_b3_timing_{split}_run_summary.json"
        if path.exists():
            fio = json.loads(path.read_text(encoding="utf-8"))[f"timing_{split}"]["seconds_per_item"]
        else:
            notes.append(f"no session 8 timing of s7_b3 on {split} ({path.name})")
    big = None
    path = Path(results) / "large_model" / f"{LARGE}_timing.json"
    if path.exists():
        t = json.loads(path.read_text(encoding="utf-8"))
        big = t.get(qid, {}).get(split, {}).get("gpu_seconds_per_item")
    if big is None:
        notes.append(f"no large-model timing for {qid} {split}")
    return fio, big, notes


def costs(fio_s, big_s, hourly_rate):
    out = {}
    for rate in RATES:
        key = f"rate_{int(round(rate * 100))}"
        if None in (fio_s, big_s, hourly_rate):
            out[key] = None
        else:
            out[key] = 1000 * (fio_s + rate * big_s) / 3600 * hourly_rate
    return out


def calibrate_ids(qid, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    folder = Path(records_ei) if qid == "evidence_effect" else Path(records_public)
    return {r["id"] for r in read_jsonl(folder / f"{qid}_validation.jsonl") if r.get("calibrate", True)}


def pmcids(split, records_ei=RECORDS_EI):
    return {r["id"]: r["group"]["pmcid"] for r in read_jsonl(Path(records_ei) / f"evidence_effect_{split}.jsonl")}


def run(split, results=RESULTS, results7=RESULTS7, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI, hourly_rate=None,
        resamples=RESAMPLES, qids=tuple(QUESTIONS)):
    if split == "test":
        for name in ("release_candidate.json", "testpreds_log.json"):
            if not (Path(results) / name).exists():
                raise SystemExit(f"{name} does not exist in {results}; test predictions are read only after the freeze")
    out = {"plan": "docs/prompt_session8.md, HAND-OFF (secondary, descriptive)", "split": split, "rates": list(RATES),
           "resamples": resamples, "seed": SEED, "large_model": LARGE, "hourly_rate": hourly_rate,
           "items": "validation: the items marked calibrate; test: every test item", "questions": {},
           "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    if split == "validation":
        out["costs_note"] = ("The EI seeds' validation passes can share the GPU with arm A (the plan runs them side by side; the ei stage "
                             "log lists the GPU's processes before each pass), so these costs can be higher than a run alone would give; "
                             "PubMedQA's is left empty, since no session 8 run of s7_b3 reads the validation split. The test costs come "
                             "from runs alone on the GPU (correction 4).")
    for qid in qids:
        spec = QUESTIONS[qid]
        fdir = Path(results7) if spec["fiorillo_dir"] == "fiorillo_v0_3" else Path(results)
        fio_path = fdir / "predictions" / f"{spec['fiorillo']}_{qid}_{split}.csv"
        big_path = Path(results) / "predictions" / f"{LARGE}_{qid}_{split}.csv"
        missing = [str(p) for p in (fio_path, big_path) if not p.exists()]
        if missing:
            out["questions"][qid] = {"missing": missing}
            continue
        keep = calibrate_ids(qid, records_public, records_ei) if split == "validation" else None
        ids, labels, fio, big = aligned(fio_path, big_path, keep)
        clusters = None
        if spec["resample"] == "articles":
            of = pmcids(split, records_ei)
            clusters = [of[i] for i in ids]
        fio_s, big_s, notes = seconds_per_item(results, qid, split)
        out["questions"][qid] = {"fiorillo": spec["fiorillo"], "fiorillo_file": str(fio_path), "large_model_file": str(big_path),
                                 "n": len(ids), "articles": len(set(clusters)) if clusters else None,
                                 "resampling": spec["resample"], **analyze(ids, labels, fio, big, clusters, resamples),
                                 "seconds_per_item": {"fiorillo": fio_s, "large_model": big_s, "notes": notes},
                                 "cost_per_1000_questions": costs(fio_s, big_s, hourly_rate)}
        r = out["questions"][qid]
        print(f"{qid} {split}: n {r['n']}; accuracy at 0, 10, 20, 30 and 50 percent "
              + ", ".join(f"{r[f'rate_{p}']['accuracy']['estimate']:.3f}" for p in (0, 10, 20, 30, 50)), flush=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--results7", type=Path, default=RESULTS7)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--records-ei", type=Path, default=RECORDS_EI)
    parser.add_argument("--hourly-rate", type=float, default=None, help="the pod's hourly rate in dollars")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out = run(args.split, args.results, args.results7, args.records_public, args.records_ei, args.hourly_rate, args.resamples)
    path = args.out or Path(args.results) / f"handoff_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

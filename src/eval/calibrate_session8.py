"""Session 8's calibration and the freeze (docs/prompt_session8.md, CALIBRATION and steps 12 and 15).

Systems, questions and inputs:
- s8_ei_1, s8_ei_2 and s8_ei_3 on evidence_effect: each seed's logits (<results>/logits/<system>_evidence_effect_<split>_logits.csv,
  from train_qwen_head.py predict), whose log-softmax is the log-probability the families take.
- s8_ei, the average of the three seeds, on evidence_effect: each prompt's probabilities are the mean of the three seeds'
  raw probabilities (the softmax of each seed's logits, in double precision; the printed raw files round them), written
  here as <results>/predictions/s8_ei-raw_evidence_effect_<split>.csv; its log-probabilities, the logarithm of that mean,
  are what the rule reads.
- s8_a3 on pubmedqa_answer and readmit_window: its logits (<results>/logits/s8_a3_<qid>_<split>_logits.csv).
- gemma4_31b on evidence_effect and pubmedqa_answer: its raw predictions (<results>/large_model/gemma4_31b-raw_<qid>_<split>.csv),
  clipped at 1e-12 before the log, as calibrate_session7.py does with raw files. It gets the same rule as Fiorillo.
A system whose validation inputs are missing (s8_a3 when arm A was not built) is recorded in calibration_s8.json as NOT
BUILT and the others go on (correction 5).
The rule is session 7's (src/eval/calibrate_session7.py's choose(), unchanged): on the validation items marked calibrate
(every EI validation prompt; PubMedQA's 150; every readmission validation record), the temperature and
temperature_intercepts families of src/eval/calibration_family_cv.py, each scored by its out-of-fold log loss (5 folds, 20
repeats, the seed); the lower is chosen (an exact tie goes to temperature) and it is fitted on all those items.

Subcommands:
  fit     on validation: calibration_s8.json ({system: {qid: entry}}, written only here), and in <results>/predictions/ the
          calibrated predictions <system>_<qid>_validation.csv (the chosen family) and <system>_temperature_<qid>_validation.csv
          (temperature alone), for every row of the system's validation file as session 7 wrote them, 9 significant digits;
          for s8_ei also s8_ei-raw_evidence_effect_validation.csv.
  freeze  release_candidate.json, written once and before any test prediction exists (it refuses if one does, in results or
          in --staging, and it refuses without readmission_pool.json unless --pool-not-built gives the reason; correction
          5): v0.4's three specialists with their calibrations (s7_b3's from calibration_s7.json; s8_ei's from
          calibration_s8.json; the readmission pool's weights and T from readmission_pool.json), gemma4_31b's calibration,
          the ECE limit for EI (the larger of 0.05 and the 95th percentile in ece_limit_evidence_effect.json), the SHA-256
          of calibration_s7.json, calibration_s8.json, readmission_pool.json, ece_limit_evidence_effect.json and each pool
          member's validation file, and the time in UTC.
  apply   on test (step 15): refuses unless release_candidate.json and testpreds_log.json exist, and unless calibration_s8.json
          and calibration_s7.json have the SHA-256 the freeze recorded; writes s8_ei-raw's test average and the calibrated
          test predictions with the parameters fitted on validation.

Usage: python src/eval/calibrate_session8.py fit [--systems s8_ei s8_ei_1 ...]
       python src/eval/calibrate_session8.py freeze --staging /root/eddm_s8/predictions_staging
       python3 src/eval/calibrate_session8.py apply
"""
import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval import metrics as M  # noqa: E402
from eval.calibrate_session7 import FOLDS, REPEATS, SEED, calibrated, choose, log_softmax, read_logits, write_probs  # noqa: E402

RESULTS = ROOT / "results" / "fiorillo_v0_4"
RESULTS7 = ROOT / "results" / "fiorillo_v0_3"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
RECORDS_EI = ROOT / "data" / "processed" / "session8"
SEEDS = ("s8_ei_1", "s8_ei_2", "s8_ei_3")
AVERAGE = "s8_ei"
EI = "evidence_effect"
SYSTEMS = {  # system: (questions, source kind)
    "s8_ei": ((EI,), "average"),
    "s8_ei_1": ((EI,), "logits"), "s8_ei_2": ((EI,), "logits"), "s8_ei_3": ((EI,), "logits"),
    "s8_a3": (("pubmedqa_answer", "readmit_window"), "logits"),
    "gemma4_31b": ((EI, "pubmedqa_answer"), "large_model"),
}
ECE_FLOOR = 0.05
PLAN = "docs/prompt_session8.md, CALIBRATION"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def records_path(qid, split, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    return (Path(records_ei) if qid == EI else Path(records_public)) / f"{qid}_{split}.jsonl"


def calibrate_ids(qid, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    keep = set()
    with open(records_path(qid, "validation", records_public, records_ei), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("calibrate", True):
                    keep.add(r["id"])
    return keep


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def seed_average(split, results=RESULTS, seeds=SEEDS):
    """(ids, labels, mean probabilities) of the seeds' softmaxed logits, rows in the first seed's order; the seeds must hold
    the same prompts with the same labels."""
    ids = labels = None
    probs = []
    for s in seeds:
        path = Path(results) / "logits" / f"{s}_{EI}_{split}_logits.csv"
        if not path.exists():
            raise SystemExit(f"{rel(path)} does not exist; the average needs all three seeds")
        s_ids, s_labels, z = read_logits(path)
        if ids is None:
            ids, labels = s_ids, s_labels
            order = np.arange(len(ids))
        else:
            pos = {i: k for k, i in enumerate(s_ids)}
            if len(pos) != len(ids) or set(pos) != set(ids):
                raise SystemExit(f"{path.name} holds different prompts from {seeds[0]}'s file")
            order = np.array([pos[i] for i in ids])
            if not np.array_equal(s_labels[order], labels):
                raise SystemExit(f"{path.name}: labels differ from {seeds[0]}'s file")
        probs.append(softmax(np.asarray(z, dtype=np.float64)[order]))
    return ids, labels, np.mean(probs, axis=0)


def load_logp(system, qid, split, results=RESULTS):
    """(ids, labels, log-probabilities, source) for one system, question and split."""
    kind = SYSTEMS[system][1]
    if kind == "average":
        ids, labels, mean = seed_average(split, results)
        raw_path = Path(results) / "predictions" / f"{AVERAGE}-raw_{EI}_{split}.csv"
        write_probs(raw_path, ids, labels, mean)
        return ids, labels, np.log(np.maximum(mean, 1e-300)), {"file": rel(raw_path), "kind": "mean of the three seeds' softmaxed logits",
                                                               "seeds": list(SEEDS)}
    if kind == "logits":
        path = Path(results) / "logits" / f"{system}_{qid}_{split}_logits.csv"
        ids, labels, z = read_logits(path)
        return ids, labels, log_softmax(np.asarray(z, dtype=np.float64)), {"file": rel(path), "kind": "logits"}
    path = Path(results) / "large_model" / f"{system}-raw_{qid}_{split}.csv"
    ids, labels, probs = M.read_predictions(path)
    zeros = int((np.asarray(probs)[np.arange(len(labels)), labels] == 0).sum())
    return ids, labels, np.log(np.clip(np.asarray(probs), 1e-12, None)), {"file": rel(path), "kind": "raw probabilities (printed)",
                                                                           "true_answer_printed_zero": zeros}


def write_calibrated(pred_dir, name, qid, split, ids, labels, logp, entry):
    k = logp.shape[1]
    chosen = calibrated(logp, entry["families"][entry["chosen"]]["params"], k)
    temp = calibrated(logp, entry["families"]["temperature"]["params"], k)
    write_probs(Path(pred_dir) / f"{name}_{qid}_{split}.csv", ids, labels, chosen)
    write_probs(Path(pred_dir) / f"{name}_temperature_{qid}_{split}.csv", ids, labels, temp)
    return chosen


def load_json(path):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def inputs_present(system, split, results=RESULTS):
    """Whether the system's input files for the split exist (a missing optional system is recorded as NOT BUILT)."""
    qids, kind = SYSTEMS[system]
    if kind == "average":
        return all((Path(results) / "logits" / f"{s}_{EI}_{split}_logits.csv").exists() for s in SEEDS)
    if kind == "logits":
        return all((Path(results) / "logits" / f"{system}_{q}_{split}_logits.csv").exists() for q in qids)
    return all((Path(results) / "large_model" / f"{system}-raw_{q}_{split}.csv").exists() for q in qids)


def fit(systems=tuple(SYSTEMS), results=RESULTS, records_public=RECORDS_PUBLIC, records_ei=RECORDS_EI):
    out_json = Path(results) / "calibration_s8.json"
    data = load_json(out_json)
    data.setdefault("plan", PLAN)
    for system in systems:
        qids, _ = SYSTEMS[system]
        if not inputs_present(system, "validation", results):
            data[system] = {"not_built": f"NOT BUILT: {system}'s calibration, its validation logits or raw predictions are missing", "time_utc": now()}
            save_json(out_json, data)
            print(data[system]["not_built"], flush=True)
            continue
        data[system] = {}
        for qid in qids:
            ids, labels, logp, source = load_logp(system, qid, "validation", results)
            keep = calibrate_ids(qid, records_public, records_ei)
            rows = np.array([i for i, x in enumerate(ids) if x in keep])
            if len(rows) != len(keep):
                raise SystemExit(f"{system} {qid}: {len(keep)} calibrate records, {len(rows)} of them in {source['file']}")
            chosen, fams = choose(logp[rows], labels[rows])
            entry = {"source": source, "n_calibrate": int(len(rows)), "folds": FOLDS, "repeats": REPEATS, "seed": SEED,
                     "families": fams, "chosen": chosen,
                     "rule": "the family with the lower mean out-of-fold log loss; an exact tie goes to temperature (session 7's rule)",
                     "time_utc": now()}
            data[system][qid] = entry
            probs = write_calibrated(Path(results) / "predictions", system, qid, "validation", ids, labels, logp, entry)
            entry["validation_calibrate_items"] = {"raw_log_loss": M.log_loss(labels[rows], np.exp(logp[rows])),
                                                   "log_loss": M.log_loss(labels[rows], probs[rows]), "ece": M.ece(labels[rows], probs[rows])}
            print(f"{system} {qid}: temperature {fams['temperature']['out_of_fold_log_loss']['mean']:.4f}, temperature_intercepts "
                  f"{fams['temperature_intercepts']['out_of_fold_log_loss']['mean']:.4f}: {chosen}", flush=True)
        save_json(out_json, data)  # after each system, so a later system's failure keeps the earlier ones
    return data


# ---------------------------------------------------------------- the freeze and the test split

def test_files(results=RESULTS, staging=None):
    found = []
    folders = [Path(results) / "predictions", Path(results) / "logits", Path(results) / "large_model"]
    if staging:
        folders.append(Path(staging))
    for folder in folders:
        if folder.exists():
            found += [str(p) for p in folder.rglob("*") if p.is_file() and "_test" in p.name]
    return sorted(found)


def family_entry(entry):
    fam = entry["families"][entry["chosen"]]
    return {"family": entry["chosen"], "params": fam["params"], "temperature": fam["temperature"], "intercepts": fam["intercepts"],
            "out_of_fold_log_loss": fam["out_of_fold_log_loss"]}


def freeze(results=RESULTS, results7=RESULTS7, staging=None, pool_not_built=None):
    results, results7 = Path(results), Path(results7)
    out_path = results / "release_candidate.json"
    if out_path.exists():
        raise SystemExit("release_candidate.json exists; it is written once")
    found = test_files(results, staging)
    if found:
        raise SystemExit(f"a test prediction already exists ({found[0]}); the freeze comes before any test prediction")
    cal7_path, cal8_path = results7 / "calibration_s7.json", results / "calibration_s8.json"
    pool_path, ece_path = results / "readmission_pool.json", results / "ece_limit_evidence_effect.json"
    for path in (cal7_path, cal8_path, ece_path):
        if not path.exists():
            raise SystemExit(f"{rel(path)} does not exist; the freeze needs it")
    if not pool_path.exists() and not pool_not_built:
        raise SystemExit(f"{rel(pool_path)} does not exist; the freeze waits for the trees stage (or takes --pool-not-built REASON)")
    cal7, cal8, ece = load_json(cal7_path), load_json(cal8_path), load_json(ece_path)
    if AVERAGE not in cal8 or EI not in cal8[AVERAGE]:
        raise SystemExit("calibration_s8.json has no calibration for s8_ei; run fit first")
    sha = {rel(cal7_path): sha256_file(cal7_path), rel(cal8_path): sha256_file(cal8_path), rel(ece_path): sha256_file(ece_path)}
    specialists = {
        "pubmedqa_answer": {"system": "s7_b3", "calibration_source": rel(cal7_path), **family_entry(cal7["s7_b3"]["pubmedqa_answer"])},
        EI: {"system": AVERAGE, "seeds": list(SEEDS), "calibration_source": rel(cal8_path), **family_entry(cal8[AVERAGE][EI])},
    }
    if pool_path.exists():
        pool = load_json(pool_path)
        sha[rel(pool_path)] = sha256_file(pool_path)
        for m in pool["members"]:
            f = pool["member_files"][m]["file"]
            path = Path(f) if Path(f).is_absolute() else ROOT / f
            sha[rel(path)] = sha256_file(path)
            if sha[rel(path)] != pool["member_files"][m]["sha256"]:
                raise SystemExit(f"{f} changed after the pool was fitted; stop")
        specialists["readmit_window"] = {"system": "readmission_pool", "source": rel(pool_path), "members": pool["members"],
                                         "weights": pool["weights"], "T": pool["T"], "validation_log_loss": pool["validation_log_loss"]}
    else:
        specialists["readmit_window"] = {"not_built": f"NOT BUILT: readmit_window's specialist, {pool_not_built}"}
    reference = {"gemma4_31b": {q: family_entry(e) for q, e in cal8.get("gemma4_31b", {}).items() if q != "not_built" and isinstance(e, dict) and "chosen" in e}}
    if not reference["gemma4_31b"]:
        raise SystemExit("calibration_s8.json has no calibration for gemma4_31b; the bar's reference needs it")
    p95 = float(ece["ece_if_calibrated"]["p95"])
    out = {"plan": "docs/prompt_session8.md, step 12 (the freeze)", "system": "fiorillo_v0_4",
           "specialists": specialists, "reference": reference,
           "ece_limits": {EI: {"limit": max(ECE_FLOOR, p95), "floor": ECE_FLOOR, "p95": p95, "n": ece.get("n"), "draws": ece.get("draws"),
                               "source": rel(ece_path),
                               "note": "the noise floor draws prompts one by one, not by article, which makes the limit a little strict"}},
           "sha256": sha, "time_utc": now()}
    save_json(out_path, out)
    print(json.dumps({"written": rel(out_path), "ece_limit": out["ece_limits"][EI]["limit"], "time_utc": out["time_utc"]}, indent=1))
    return out


def guard(results=RESULTS):
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; test predictions are read only after the freeze and the pod's test stage")
    return load_json(Path(results) / "release_candidate.json")


def check_frozen(rc, path):
    key = rel(path)
    want = (rc.get("sha256") or {}).get(key)
    if want is None:
        raise SystemExit(f"release_candidate.json records no SHA-256 for {key}")
    got = sha256_file(path)
    if got != want:
        raise SystemExit(f"{key} changed after the freeze (SHA-256 {got}, frozen {want}); stop")


def apply(systems=tuple(SYSTEMS), results=RESULTS, results7=RESULTS7, split="test"):
    rc = guard(results)
    check_frozen(rc, Path(results) / "calibration_s8.json")
    check_frozen(rc, Path(results7) / "calibration_s7.json")
    data = load_json(Path(results) / "calibration_s8.json")
    written = {}
    for system in systems:
        if system not in data:
            raise SystemExit(f"calibration_s8.json has no calibration for {system}")
        if "not_built" in data[system]:
            print(data[system]["not_built"], flush=True)
            continue
        if not inputs_present(system, split, results):
            print(f"NOT BUILT: {system}'s {split} predictions, its {split} logits or raw predictions are missing", flush=True)
            continue
        for qid in SYSTEMS[system][0]:
            ids, labels, logp, _ = load_logp(system, qid, split, results)
            write_calibrated(Path(results) / "predictions", system, qid, split, ids, labels, logp, data[system][qid])
            written[f"{system} {qid}"] = len(ids)
            print(f"{system} {qid} {split}: {len(ids)} rows written", flush=True)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("fit", "freeze", "apply"))
    parser.add_argument("--systems", nargs="+", default=list(SYSTEMS))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--results7", type=Path, default=RESULTS7)
    parser.add_argument("--records-public", type=Path, default=RECORDS_PUBLIC)
    parser.add_argument("--records-ei", type=Path, default=RECORDS_EI)
    parser.add_argument("--staging", type=Path, default=None, help="freeze: the pod's staging folder, checked for test files too")
    parser.add_argument("--pool-not-built", default=None, help="freeze: the reason the readmission pool is not built (a stated NOT BUILT)")
    args = parser.parse_args(argv)
    unknown = [s for s in args.systems if s not in SYSTEMS]
    if unknown:
        parser.error(f"unknown systems {unknown}")
    if args.command == "fit":
        fit(tuple(args.systems), args.results, args.records_public, args.records_ei)
    elif args.command == "freeze":
        freeze(args.results, args.results7, args.staging, args.pool_not_built)
    else:
        apply(tuple(args.systems), args.results, args.results7)
    return 0


if __name__ == "__main__":
    sys.exit(main())

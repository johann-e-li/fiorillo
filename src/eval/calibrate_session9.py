"""Session 9's calibration, the ECE limit, the freeze and the apply step (docs/prompt_session9.md: CALIBRATION, POWER AND
THE BAR, the pod's calibrate and freeze stages, steps 6 and 15, and corrections 2 and 3).

Systems (their log-probabilities on a split; recipe_session9.py's names):
- a run (s9_b_1 to s9_b_3, s9_<candidate>_<k>, s9_combo_<k>) or s9_ei_wavg: the log-softmax of its logits
  (<results>/logits/<system>_evidence_effect_<split>_logits.csv);
- s9_b (the base runs' average) and s9_ei (the winning recipe's runs, recipe_selection.json): the logarithm of the mean of
  the runs' softmaxed logits, as session 8 averaged s8_ei;
- <X>_sa, X swap-averaged (correction 3): the mean of X's probabilities and the exchange of its answer to the exchanged
  prompt; on validation the exchanged answers are <X>_exch's (recipe_session9.py), on test the reading checks' swap inputs
  (<results>/checks_v05/logits/<single-pass system>_swap_evidence_effect_test_logits.csv), averaged over X's runs for s9_ei;
- gemma4_31b_long (under GEMMA ON THE LONGER INPUT) and biogpt_large_pqa: their raw probabilities, clipped at 1e-12 before
  the log, as session 8 read raw files.
The rules:
- v0.5's EI specialist (recipe_selection.json's v05_ei.system): on the 654 clean validation prompts, the family with the
  lowest out-of-fold log loss among temperature, temperature_intercepts, vector and dirichlet_odir (5 folds, 20 repeats,
  the seed; src/eval/calibration_family_cv.py's choose(); an exact tie, within 1e-12, goes to the family listed first),
  fitted on all 654. The other form of v0.5's system (with or without swap averaging, for K10) and the one of s9_ei and
  s9_ei_wavg that v0.5 does not use (for K8) are calibrated by the same family, each fitted on its own 654 clean
  predictions. v0.5's specialist is also calibrated by session 8's rule, as <system>_s8rule (for K9).
- every other system keeps session 8's rule (src/eval/calibrate_session7.py's choose(): temperature or
  temperature_intercepts): s9_b and each run on the 654 clean prompts; gemma4_31b_long on all 1,237 EI validation prompts;
  biogpt_large_pqa on PubMedQA's 150 calibrate items. s8_ei, s7_b3 and gemma4_31b keep session 8's and session 7's fits
  (calibration_s8.json and calibration_s7.json, read, not refitted).
Every v0.5 system's calibrated predictions are written for every row of its file (all 1,237 validation prompts), with the
calibration fitted on the clean ones (the hand-off analyses and power_session9.py validation read all 1,237).

Subcommands:
  fit          calibration_s9.json ({system: {qid: entry}}, as calibration_s8.json), with the four families' out-of-fold
               table for v0.5's specialist (I-36); the calibrated validation predictions
               <results>/predictions/<system>_<qid>_validation.csv (the chosen family) and <system>_temperature_<qid>_validation.csv
  ece          the ECE limit for criterion 1: ece_noise_floor.py (unchanged; 5,000 draws, the seed, n 1,218) on v0.5's
               specialist's calibrated clean validation predictions; ece_limit_evidence_effect.json
  families-s8  step 6: the four families' out-of-fold table for s8_ei on all 1,237 validation prompts
               (validation_analyses/calibration_families_s8.json; descriptive; s8_ei keeps session 8's calibration)
  freeze       release_candidate.json, before any test prediction of a v0.5 system exists (checks_s8/ holds s8_ei's and is
               not counted); it refuses before trees, tabicl, biogpt and calibrate have finished (or written a NOT BUILT
               marker) and before osf_filed.txt exists
  apply        step 15, on test: refuses unless release_candidate.json and testpreds_log.json exist and the frozen files
               have the SHA-256 the freeze recorded; writes every system's calibrated test predictions

Usage (the pod, from /root/eddm_s9/code): python src/eval/calibrate_session9.py fit | ece | freeze --base /root/eddm_s9
          (the linked shell): python3 src/eval/calibrate_session9.py families-s8 | apply
"""
import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from eval import metrics as M  # noqa: E402
from eval import calibration_family_cv as CFC  # noqa: E402
from eval.calibrate_session7 import FOLDS, REPEATS, SEED, calibrated, choose, log_softmax, read_logits, write_probs  # noqa: E402
from eval import recipe_session9 as R  # noqa: E402

RESULTS = ROOT / "results" / "fiorillo_v0_5"
RESULTS8 = ROOT / "results" / "fiorillo_v0_4"
RESULTS7 = ROOT / "results" / "fiorillo_v0_3"
RECORDS8 = ROOT / "data" / "processed" / "session8"
RECORDS9 = ROOT / "data" / "processed" / "session9"
RECORDS_PUBLIC = ROOT / "data" / "processed" / "public"
EI = "evidence_effect"
PQA = "pubmedqa_answer"
FAMILIES = ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")
ECE_FLOOR = 0.05
N_TEST = 1218
PLAN = "docs/prompt_session9.md, CALIBRATION"


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def load_json(path, default=None):
    return R.load_json(path, default)


def save_json(path, data):
    R.save_json(path, data)


# ---------------------------------------------------------------- systems' probabilities

def selection(results):
    sel = load_json(Path(results) / "recipe_selection.json", None)
    if not sel or "v05_ei" not in sel:
        raise SystemExit("recipe_selection.json has no v05_ei: the train stage has not finished")
    return sel


def probs_of(name, split, results, sel):
    """(ids, labels, probabilities) of a v0.5 EI system on a split (validation or test), in double precision."""
    results = Path(results)
    base_runs = [R.run_name("b", k) for k in (1, 2, 3)]
    if name == "s9_b":
        return R.mean_probs(results, base_runs, split)
    if name == "s9_ei":
        return R.mean_probs(results, list(sel["s9_ei"]["runs"]), split)
    if name.endswith("_sa"):
        x = name[:-3]
        ids, labels, p = probs_of(x, split, results, sel)
        e_ids, _, e_p = exch_probs(x, split, results, sel)
        return ids, labels, R.sa_probs(ids, labels, p, e_ids, e_p)
    return R.run_probs(results, name, split)


def exch_probs(x, split, results, sel):
    """X's answers to the exchanged prompts: on validation <X>_exch's logits; on test the reading checks' swap inputs."""
    singles = list(sel["s9_ei"]["runs"]) if x == "s9_ei" else [x]
    files = []
    for s in singles:
        if split == "validation":
            files.append(R.logits_path(results, f"{s}_exch", split))
        else:
            files.append(Path(results) / "checks_v05" / "logits" / f"{s}_swap_{EI}_{split}_logits.csv")
    out = []
    ids = labels = None
    for f in files:
        if not f.exists():
            raise SystemExit(f"{rel(f)} does not exist; the swap-averaged system needs it")
        i2, l2, z = read_logits(f)
        p = R.softmax(np.asarray(z, dtype=np.float64))
        if ids is None:
            ids, labels = list(i2), np.asarray(l2)
            out.append(p)
        else:
            out.append(R.align(ids, list(i2), l2, p, labels, rel(f)))
    return ids, labels, np.mean(out, axis=0)


def raw_probs(path):
    ids, labels, probs = M.read_predictions(path)
    return list(ids), np.asarray(labels), np.asarray(probs, dtype=np.float64)


def logp_of(probs, floor):
    return np.log(np.maximum(np.asarray(probs), floor))


def ids_of(path, calibrate_only=True):
    out = []
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if not calibrate_only or r.get("calibrate", True):
                out.append(r["id"])
    return out


def rows_for(ids, keep, what):
    pos = {i: k for k, i in enumerate(ids)}
    missing = [i for i in keep if i not in pos]
    if missing:
        raise SystemExit(f"{what}: {len(missing)} of the calibrate items are missing, for example {missing[:3]}")
    return np.array([pos[i] for i in keep])


# ---------------------------------------------------------------- fit

def session8_rule_entry(logp, labels, source):
    chosen, fams = choose(logp, labels)
    return {"source": source, "n_calibrate": int(len(labels)), "folds": FOLDS, "repeats": REPEATS, "seed": SEED, "families": fams,
            "chosen": chosen, "rule": "session 8's rule: the lower mean out-of-fold log loss of temperature and temperature_intercepts; "
                                      "an exact tie goes to temperature", "time_utc": now()}


def four_family_entry(logp, labels, source):
    result, _ = CFC.choose(logp, labels, families=FAMILIES, folds=FOLDS, repeats=REPEATS, seed=SEED)
    fams = {f: dict(v) for f, v in result["families"].items()}
    return {"source": source, "n_calibrate": int(len(labels)), "folds": FOLDS, "repeats": REPEATS, "seed": SEED, "families": fams,
            "chosen": result["chosen"], "rule": ("the lowest mean out-of-fold log loss among temperature, temperature_intercepts, "
                                                 "vector and dirichlet_odir; an exact tie (within 1e-12) goes to the family listed first"),
            "time_utc": now()}


def same_family_entry(logp, labels, family, source, seed=SEED):
    params = CFC.fit_family(family, logp, labels, seed=seed)
    params = params.tolist() if hasattr(params, "tolist") else params
    return {"source": source, "n_calibrate": int(len(labels)), "families": {family: {"params": params}}, "chosen": family,
            "rule": "the family chosen for v0.5's EI specialist, fitted on this system's own clean validation predictions",
            "time_utc": now()}


def apply_entry(entry, logp):
    fam = entry["chosen"]
    return np.asarray(CFC.apply_family(fam, entry["families"][fam]["params"], logp))


def write_entry_predictions(pred_dir, name, qid, split, ids, labels, logp, entry):
    probs = apply_entry(entry, logp)
    write_probs(Path(pred_dir) / f"{name}_{qid}_{split}.csv", ids, labels, probs)
    if "temperature" in entry["families"] and "params" in entry["families"]["temperature"]:
        temp = calibrated(logp, entry["families"]["temperature"]["params"], logp.shape[1])
        write_probs(Path(pred_dir) / f"{name}_temperature_{qid}_{split}.csv", ids, labels, temp)
    return probs


def fit(results=RESULTS, records9=RECORDS9, records8=RECORDS8, records_public=RECORDS_PUBLIC, results8=RESULTS8, results7=RESULTS7):
    results = Path(results)
    sel = selection(results)
    v = sel["v05_ei"]
    keep = R.clean_ids(records9)
    out_json = results / "calibration_s9.json"
    data = load_json(out_json, {}) or {}
    data["plan"] = PLAN
    data["v05_ei"] = {"system": v["system"], "base_system": v["base_system"], "swap_averaging": v["swap_averaging"],
                      "other_form": v["other_form"], "recipe": v["recipe"]}
    pred_dir = results / "predictions"

    def clean_fit(name, how, family=None):
        ids, labels, p = probs_of(name, "validation", results, sel)
        logp = logp_of(p, 1e-300)
        rows = rows_for(ids, keep, name)
        source = {"system": name, "kind": "logits (runs) or the mean of the runs' softmaxed logits; swap-averaged where named _sa",
                  "calibrate_items": "the 654 clean validation prompts (data/processed/session9/clean/)"}
        if how == "four":
            entry = four_family_entry(logp[rows], labels[rows], source)
        elif how == "same":
            entry = same_family_entry(logp[rows], labels[rows], family, source)
        else:
            entry = session8_rule_entry(logp[rows], labels[rows], source)
        probs = write_entry_predictions(pred_dir, name if how != "s8rule_of_v05" else f"{name}_s8rule", EI, "validation", ids, labels, logp, entry)
        entry["validation_calibrate_items"] = {"raw_log_loss": M.log_loss(labels[rows], np.exp(logp[rows])),
                                               "log_loss": M.log_loss(labels[rows], probs[rows]), "ece": M.ece(labels[rows], probs[rows])}
        return entry

    v_name = v["system"]
    data[v_name] = {EI: clean_fit(v_name, "four")}
    family = data[v_name][EI]["chosen"]
    print(f"v0.5's EI specialist {v_name}: chosen {family}; out-of-fold "
          + ", ".join(f"{f} {data[v_name][EI]['families'][f]['out_of_fold_log_loss']['mean']:.5f}" for f in FAMILIES), flush=True)
    save_json(out_json, data)
    data[f"{v_name}_s8rule"] = {EI: clean_fit(v_name, "s8rule_of_v05")}
    others = [v["other_form"]] + [s for s in ("s9_ei", "s9_ei_wavg") if s not in (v_name, v["other_form"]) and
                                  (s != "s9_ei_wavg" or R.has_run(results, "s9_ei_wavg"))]
    for name in others:
        data[name] = {EI: clean_fit(name, "same", family)}
        save_json(out_json, data)
    runs = sorted({r for r in (sel.get("scores") or {}) if R.has_run(results, r)} | set(R.run_name("b", k) for k in (1, 2, 3)))
    for name in ["s9_b"] + runs:
        if name in data:
            continue
        data[name] = {EI: clean_fit(name, "s8rule")}
        save_json(out_json, data)
    long_raw = results / "large_model" / f"gemma4_31b_long-raw_{EI}_validation.csv"
    if long_raw.exists():
        ids, labels, p = raw_probs(long_raw)
        keep_all = ids_of(Path(records8) / f"{EI}_validation.jsonl")
        rows = rows_for(ids, keep_all, "gemma4_31b_long")
        logp = logp_of(p, 1e-12)
        entry = session8_rule_entry(logp[rows], labels[rows], {"file": rel(long_raw), "kind": "raw probabilities (printed)"})
        probs = write_entry_predictions(pred_dir, "gemma4_31b_long", EI, "validation", ids, labels, logp, entry)
        entry["validation_calibrate_items"] = {"log_loss": M.log_loss(labels[rows], probs[rows]), "ece": M.ece(labels[rows], probs[rows])}
        data["gemma4_31b_long"] = {EI: entry}
        save_json(out_json, data)
    bio_raw = pred_dir / f"biogpt_large_pqa-raw_{PQA}_validation.csv"
    if bio_raw.exists():
        ids, labels, p = raw_probs(bio_raw)
        keep_pqa = ids_of(Path(records_public) / f"{PQA}_validation.jsonl")
        rows = rows_for(ids, keep_pqa, "biogpt_large_pqa")
        logp = logp_of(p, 1e-12)
        entry = session8_rule_entry(logp[rows], labels[rows], {"file": rel(bio_raw), "kind": "raw probabilities (printed)",
                                                                 "caveat": "BioGPT-Large was fine-tuned on these 150 items, so its "
                                                                           "calibrated results are secondary (PUBMEDQA BESIDE BIOGPT-LARGE)"})
        probs = write_entry_predictions(pred_dir, "biogpt_large_pqa", PQA, "validation", ids, labels, logp, entry)
        entry["validation_calibrate_items"] = {"raw_log_loss": M.log_loss(labels[rows], np.exp(logp[rows])),
                                               "log_loss": M.log_loss(labels[rows], probs[rows]), "ece": M.ece(labels[rows], probs[rows])}
        data["biogpt_large_pqa"] = {PQA: entry}
    else:
        data["biogpt_large_pqa"] = {"not_built": "NOT BUILT: biogpt_large_pqa's calibration, its raw validation predictions are missing"}
    data["kept_from_earlier_sessions"] = {
        "s8_ei": {"source": "results/fiorillo_v0_4/calibration_s8.json", "sha256": sha256_file(Path(results8) / "calibration_s8.json")},
        "gemma4_31b": {"source": "results/fiorillo_v0_4/calibration_s8.json"},
        "s7_b3": {"source": "results/fiorillo_v0_3/calibration_s7.json", "sha256": sha256_file(Path(results7) / "calibration_s7.json")}}
    data["families_table_v05"] = {"system": v_name, "table": {f: data[v_name][EI]["families"][f]["out_of_fold_log_loss"] for f in FAMILIES},
                                  "chosen": family, "note": "I-36, reported whatever is chosen"}
    data["time_utc"] = now()
    save_json(out_json, data)
    return data


# ---------------------------------------------------------------- the ECE limit

def ece(results=RESULTS, records9=RECORDS9):
    results = Path(results)
    cal = load_json(results / "calibration_s9.json", None)
    if not cal or "v05_ei" not in cal:
        raise SystemExit("calibration_s9.json has no v0.5 specialist; run fit first")
    v_name = cal["v05_ei"]["system"]
    pred = results / "predictions" / f"{v_name}_{EI}_validation.csv"
    out = results / "ece_limit_evidence_effect.json"
    cmd = [sys.executable, str(ROOT / "src" / "eval" / "ece_noise_floor.py"), "--predictions", str(pred), "--records",
           str(Path(records9) / "clean" / f"{EI}_validation.jsonl"), "--n", str(N_TEST), "--draws", "5000", "--out", str(out)]
    subprocess.run(cmd, check=True)
    d = load_json(out)
    d["system"] = v_name
    d["plan"] = "docs/prompt_session9.md, CALIBRATION (the ECE limit for criterion 1)"
    d["limit"] = max(ECE_FLOOR, float(d["ece_if_calibrated"]["p95"]))
    d["written_utc"] = now()
    save_json(out, d)
    print(json.dumps({"system": v_name, "limit": d["limit"], "p95": d["ece_if_calibrated"]["p95"]}))
    return d


# ---------------------------------------------------------------- step 6: the families for s8_ei

def families_s8(results8=RESULTS8, records8=RECORDS8, out=RESULTS / "validation_analyses" / "calibration_families_s8.json"):
    seeds = [f"s8_ei_{k}" for k in (1, 2, 3)]
    ids = labels = None
    probs = []
    for s in seeds:
        i2, l2, z = read_logits(Path(results8) / "logits" / f"{s}_{EI}_validation_logits.csv")
        p = R.softmax(np.asarray(z, dtype=np.float64))
        if ids is None:
            ids, labels = list(i2), np.asarray(l2)
            probs.append(p)
        else:
            probs.append(R.align(ids, list(i2), l2, p, labels, s))
    logp = np.log(np.maximum(np.mean(probs, axis=0), 1e-300))
    keep = ids_of(Path(records8) / f"{EI}_validation.jsonl")
    rows = rows_for(ids, keep, "s8_ei")
    result, _ = CFC.choose(logp[rows], labels[rows], families=FAMILIES, folds=FOLDS, repeats=REPEATS, seed=SEED)
    cal8 = load_json(Path(results8) / "calibration_s8.json")
    data = {"plan": "docs/prompt_session9.md, STEP 6 (I-36 for s8_ei; descriptive)", "written_utc": now(), "system": "s8_ei",
            "n": int(len(rows)), "folds": FOLDS, "repeats": REPEATS, "seed": SEED,
            "table": {f: result["families"][f]["out_of_fold_log_loss"] for f in FAMILIES},
            "lowest": result["chosen"], "params": {f: result["families"][f]["params"] for f in FAMILIES},
            "s8_ei_keeps": {"family": cal8["s8_ei"][EI]["chosen"], "source": "results/fiorillo_v0_4/calibration_s8.json"},
            "note": "descriptive: s8_ei keeps session 8's calibration; this table is not used to change it"}
    save_json(out, data)
    print(json.dumps({"table": {f: round(v["mean"], 5) for f, v in data["table"].items()}, "lowest": data["lowest"]}))
    return data


# ---------------------------------------------------------------- the freeze

def test_files(results=RESULTS, staging=None):
    """Every test prediction of a v0.5 system: results' predictions, logits, large_model and checks_v05 folders, and the
    staging folder. checks_s8/ holds s8_ei's reading checks and is not counted."""
    found = []
    folders = [Path(results) / d for d in ("predictions", "logits", "large_model", "checks_v05")]
    if staging:
        folders.append(Path(staging))
    for folder in folders:
        if folder.exists():
            found += [str(p) for p in folder.rglob("*") if p.is_file() and "_test" in p.name]
    return sorted(found)


def family_entry(entry):
    fam = entry["chosen"]
    f = entry["families"][fam]
    return {"family": fam, "params": f["params"], "out_of_fold_log_loss": f.get("out_of_fold_log_loss")}


def stage_finished(base, stage, results, marker_names=None):
    """A stage counts as finished when the driver wrote its done marker (<base>/stage_done/<stage>.txt, written only when
    every command of the stage succeeded) and its outputs exist, or when it wrote a NOT BUILT marker
    (<base>/not_built/<stage>.txt)."""
    nb = Path(base) / "not_built" / f"{stage}.txt"
    if nb.exists():
        return "not built: " + nb.read_text(encoding="utf-8").strip()
    if not (Path(base) / "stage_done" / f"{stage}.txt").exists():
        return None
    for name in marker_names or []:
        if not (Path(results) / name).exists():
            return None
    return "finished"


def read_osf_filed(base):
    path = Path(base) / "osf_filed.txt"
    if not path.exists():
        raise SystemExit(f"{path} does not exist: the OSF update's filing is not recorded; the freeze waits")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if " " in line.strip():
            k, val = line.strip().split(" ", 1)
            out[k] = val.strip()
    if not out.get("link") or not out.get("filed_utc"):
        raise SystemExit(f"{path} needs the lines 'link <the OSF update's address>' and 'filed_utc <time>'")
    return out


def freeze(results=RESULTS, results8=RESULTS8, results7=RESULTS7, base=Path("/root/eddm_s9"), staging=None):
    results, results8, results7 = Path(results), Path(results8), Path(results7)
    sys.path.insert(0, str(ROOT / "src" / "eval"))
    out_path = results / "release_candidate.json"
    if out_path.exists():
        raise SystemExit("release_candidate.json exists; it is written once")
    osf = read_osf_filed(base)
    found = test_files(results, staging)
    if found:
        raise SystemExit(f"a test prediction of a v0.5 system already exists ({found[0]}); the freeze comes before any")
    status = {"trees": stage_finished(base, "trees", results, ["trees_v2.json"]),
              "tabicl": stage_finished(base, "tabicl", results, ["tabicl.json"]),
              "biogpt": stage_finished(base, "biogpt", results, ["biogpt.json", f"predictions/biogpt_large_pqa-raw_{PQA}_validation.csv"]),
              "calibrate": stage_finished(base, "calibrate", results, ["calibration_s9.json", "ece_limit_evidence_effect.json",
                                                                       "readmission_pool_v2.json", "handoff_rule_v05.json",
                                                                       "handoff_validation.json", "power_validation.json"])}
    waiting = [k for k, v in status.items() if v is None]
    if waiting:
        raise SystemExit(f"the freeze waits for {waiting} to finish or write a NOT BUILT marker")
    sel, cal9 = selection(results), load_json(results / "calibration_s9.json")
    cal7, cal8 = load_json(results7 / "calibration_s7.json"), load_json(results8 / "calibration_s8.json")
    ece_d = load_json(results / "ece_limit_evidence_effect.json")
    v = sel["v05_ei"]
    v_name = v["system"]
    if cal9.get("v05_ei", {}).get("system") != v_name:
        raise SystemExit("calibration_s9.json was fitted for another v0.5 system than recipe_selection.json names; stop")
    files = [results / "recipe_selection.json", results / "calibration_s9.json", results7 / "calibration_s7.json",
             results / "ece_limit_evidence_effect.json", results / "validation_analyses" / "handoff_rule_s8.json",
             results / "handoff_rule_v05.json", results / "readmission_pool_v2.json"]
    if (results / "weight_average.json").exists():
        files.append(results / "weight_average.json")
    sha = {}
    for f in files:
        if not f.exists():
            raise SystemExit(f"{rel(f)} does not exist; the freeze needs it")
        sha[rel(f)] = sha256_file(f)
    import trees_session9 as T9
    for k, digest in T9.frozen_inputs(results).items():
        sha[k] = digest
    pool = load_json(results / "readmission_pool_v2.json")
    large = "gemma4_31b_long" if v["max_len"] == 6144 and "gemma4_31b_long" in cal9 else "gemma4_31b"
    if v["max_len"] == 6144 and large != "gemma4_31b_long":
        raise SystemExit("v0.5's recipe reads 6,144 tokens, but gemma4_31b_long has no calibration; stop (GEMMA ON THE LONGER INPUT)")
    reference = ({EI: family_entry(cal9["gemma4_31b_long"][EI]), "calibration_source": rel(results / "calibration_s9.json")}
                 if large == "gemma4_31b_long" else
                 {EI: family_entry(cal8["gemma4_31b"][EI]), PQA: family_entry(cal8["gemma4_31b"][PQA]),
                  "calibration_source": rel(results8 / "calibration_s8.json")})
    out = {"plan": "docs/prompt_session9.md, the freeze (STEP 3's freeze stage)", "system": "fiorillo_v0_5",
           "specialists": {
               PQA: {"system": "s7_b3", "calibration_source": rel(results7 / "calibration_s7.json"), **family_entry(cal7["s7_b3"][PQA])},
               EI: {"system": v_name, "base_system": v["base_system"], "swap_averaging": v["swap_averaging"], "recipe": v["recipe"],
                    "runs": v["runs"], "single_pass_systems": v["single_pass_systems"], "max_len": v["max_len"],
                    "calibration_source": rel(results / "calibration_s9.json"), **family_entry(cal9[v_name][EI])},
               "readmit_window": {"system": "readmission_pool_v2", "chosen": pool.get("chosen"), "members": pool.get("members"),
                                  "weights": pool.get("weights"), "T": pool.get("T"),
                                  "validation_log_loss": pool.get("validation_log_loss"), "source": rel(results / "readmission_pool_v2.json")}},
           "reference": {"criterion_3_large_model": large, large: reference},
           "biogpt_large_pqa": (family_entry(cal9["biogpt_large_pqa"][PQA]) if PQA in cal9.get("biogpt_large_pqa", {}) else
                                cal9.get("biogpt_large_pqa")),
           "handoff_rules": {"s8_ei": rel(results / "validation_analyses" / "handoff_rule_s8.json"),
                             "v05": rel(results / "handoff_rule_v05.json")},
           "ece_limits": {EI: {"limit": max(ECE_FLOOR, float(ece_d["ece_if_calibrated"]["p95"])), "floor": ECE_FLOOR,
                               "p95": float(ece_d["ece_if_calibrated"]["p95"]), "n": ece_d.get("n"), "draws": ece_d.get("draws"),
                               "source": rel(results / "ece_limit_evidence_effect.json")}},
           "stages": status, "osf_update": osf, "sha256": sha, "time_utc": now()}
    save_json(out_path, out)
    print(json.dumps({"written": rel(out_path), "v05_ei": v_name, "large_model": large, "ece_limit": out["ece_limits"][EI]["limit"],
                      "time_utc": out["time_utc"]}, indent=1))
    return out


# ---------------------------------------------------------------- apply (test)

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


def apply(results=RESULTS, results7=RESULTS7, split="test"):
    results = Path(results)
    rc = guard(results)
    for f in (results / "calibration_s9.json", results / "recipe_selection.json", results7 / "calibration_s7.json"):
        check_frozen(rc, f)
    sel, cal9 = selection(results), load_json(results / "calibration_s9.json")
    pred_dir = results / "predictions"
    written = {}
    for name, entry in cal9.items():
        if not isinstance(entry, dict) or name in ("v05_ei", "kept_from_earlier_sessions", "families_table_v05"):
            continue
        if EI in entry:
            if name == "gemma4_31b_long":
                path = results / "large_model" / f"gemma4_31b_long-raw_{EI}_{split}.csv"
                if not path.exists():
                    print(f"NOT BUILT: gemma4_31b_long's {split} predictions are missing", flush=True)
                    continue
                ids, labels, p = raw_probs(path)
                logp = logp_of(p, 1e-12)
            else:
                source = name[: -len("_s8rule")] if name.endswith("_s8rule") else name
                try:
                    ids, labels, p = probs_of(source, split, results, sel)
                except (SystemExit, OSError) as e:  # a candidate's runs have no test predictions (testpreds predicts the
                    # base runs, the winning recipe's runs and the weight average only)
                    print(f"no {split} predictions for {name}: {e}", flush=True)
                    continue
                logp = logp_of(p, 1e-300)
                if source == "s9_ei" or source in ("s9_b",) or source.endswith("_sa"):
                    write_probs(pred_dir / f"{source}-raw_{EI}_{split}.csv", ids, labels, p)
            write_entry_predictions(pred_dir, name, EI, split, ids, labels, logp, entry[EI])
            written[name] = len(ids)
        elif PQA in entry:
            path = pred_dir / f"{name}-raw_{PQA}_{split}.csv"
            if not path.exists():
                print(f"NOT BUILT: {name}'s {split} predictions are missing", flush=True)
                continue
            ids, labels, p = raw_probs(path)
            write_entry_predictions(pred_dir, name, PQA, split, ids, labels, logp_of(p, 1e-12), entry[PQA])
            written[name] = len(ids)
    print(json.dumps(written, indent=1))
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--results", type=Path, default=RESULTS)
    e = sub.add_parser("ece")
    e.add_argument("--results", type=Path, default=RESULTS)
    sub.add_parser("families-s8")
    fr = sub.add_parser("freeze")
    fr.add_argument("--results", type=Path, default=RESULTS)
    fr.add_argument("--base", type=Path, default=Path("/root/eddm_s9"))
    fr.add_argument("--staging", type=Path, default=None)
    ap = sub.add_parser("apply")
    ap.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args(argv)
    if args.cmd == "fit":
        fit(args.results)
    elif args.cmd == "ece":
        ece(args.results)
    elif args.cmd == "families-s8":
        families_s8()
    elif args.cmd == "freeze":
        freeze(args.results, base=args.base, staging=args.staging)
    else:
        apply(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

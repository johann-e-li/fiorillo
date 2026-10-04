"""The readmission pool (docs/prompt_session8.md, THE READMISSION POOL, with corrections 2 and 3; idea I-28 of
docs/ideas_register.md). Run on the pod's CPU in the trees stage (correction 3), and pool-test on the linked shell.

Members: lightgbm_tuned (session 7's chosen setting, its predictions as they are), xgboost_tuned and catboost_tuned
(tuned here), arm_laya (v0.1 as released) and s7_b3 (its session 7 calibrated readmission predictions). Each member's
prediction files are <member>_readmit_window_<split>.csv in session 4's format: lightgbm_tuned and s7_b3 in
results/fiorillo_v0_3/predictions/, arm_laya in results/public_v0/predictions/, xgboost_tuned and catboost_tuned in
results/fiorillo_v0_4/predictions/.

The fields are the frames that src/eval/trees_session7.py's frames() builds (session 4's fields through
src/eval/baselines.py's uci_frame(), categories fitted on the training split). XGBoost reads the categorical columns
natively (enable_categorical true, tree_method hist); CatBoost reads them as strings, a missing value (or a category not
seen in training, which the frames make missing) as "missing", given as categorical features.

xgboost_tuned: 50 distinct settings drawn with the seed (random.Random(20260927).sample, without replacement) from the
1,620 combinations of max_depth (3, 4, 6, 8, 10), min_child_weight (1, 5, 10, 20), eta (0.02, 0.05, 0.1), subsample
(0.6, 0.8, 1.0), colsample_bytree (0.6, 0.8, 1.0) and lambda (0, 1, 10), taken in that order by itertools.product.
catboost_tuned: 30 distinct settings drawn the same way (a new random.Random(20260927)) from the 144 combinations of
depth (4, 6, 8, 10), learning_rate (0.03, 0.05, 0.1), l2_leaf_reg (1, 3, 10), bagging_temperature (0, 1) and
border_count (128, 254). CatBoost's bootstrap is set to Bayesian, its default for the multi-class loss (checked with
CatBoost 1.2.10's get_all_params), because bagging_temperature is a setting of that bootstrap.
Each setting is scored as in session 7: 5-fold stratified cross-validation on the training records (scikit-learn's
StratifiedKFold, shuffled with the seed), at most 3,000 trees, stopping after 100 rounds without improvement in the
held-out fold's multi-class log loss; the fold's score is the log loss of src/eval/metrics.py on the held-out fold at the
best round (the libraries' own best scores are recorded beside it). best_iteration below is the number of trees at the
best round (XGBoost's best_iteration + 1; CatBoost's get_best_iteration() + 1), as LightGBM's best_iteration_ counts
them in session 7. Progress is saved after each setting and fold in trees_progress.json, so the search resumes when run
again; --max-seconds stops it from starting a new fit after that many seconds. Correction 3: a fit no longer resumes
part way (the pod has no three-minute limit), so no fit saves its own state. The setting with the lowest mean
out-of-fold log loss (ties to the lower setting index) is refitted on all training records with the mean of its folds'
best iterations (rounded).

The pool: softmax((sum over members of w_m log p_m) / T), log p clipped at 1e-12 as in session 7's blends; the weights on
the grid of steps of 0.1 that sum to 1 (1,001 points for five members; the points of itertools.product over 0 to 10
tenths for each member, in that order, that sum to 10), T fitted by maximum likelihood at each point (T in [0.1, 10],
by the damped Newton method of session 7's fit_blend on 1 / T), on the readmission validation records; the point with the
lowest validation log loss is kept (the first in grid order among exact ties). Its out-of-fold validation log loss (5
folds, 20 repeats; the folds of src/eval/calibration_family_cv.py's folds_for with numpy's default_rng and the seed, as
session 7's blends drew them; weights and T refitted in each fold; each fold's fit saved in trees_progress.json as it
finishes) is reported beside it and labeled optimistic, because two members' calibrations (arm_laya's and s7_b3's) were
fitted on the same records. A tuned member whose validation file does not exist is left out and recorded as NOT BUILT; a
missing uploaded member (lightgbm_tuned, arm_laya, s7_b3) stops the run (correction 5).

Subcommands:
  search      the search of both members (--members to choose), resumable (trees_progress.json).
  validation  the refits on all training records; the fitted models saved in --models-dir (xgboost_tuned.json,
              catboost_tuned.cbm); the validation predictions; trees.json (every setting's score, the chosen setting,
              the number of trees, the model files' SHA-256).
  pool        the pool on validation: readmission_pool.json and readmission_pool_readmit_window_validation.csv.
  testpreds   on the pod, in the testpreds stage: refuses to run unless release_candidate.json and testpreds_log.json
              exist; checks each tree member's validation file against the SHA-256 in release_candidate.json and that its
              saved model reproduces it; writes xgboost_tuned's and catboost_tuned's test predictions; adds its check to
              testpreds_log.json under "trees".
  pool-test   step 15, on the linked shell: the same guard; checks the SHA-256 of readmission_pool.json and of every
              member's validation file against release_candidate.json; the kept weights and T applied to the members'
              test files; trees_test.json (the members' test files and their row counts), readmission_pool_test.json and
              readmission_pool_readmit_window_test.csv.
New prediction files print 9 significant digits.

Usage (pod): python src/eval/trees_session8.py search --threads 12
             python src/eval/trees_session8.py validation --threads 12 --models-dir /root/eddm_s8/models/trees
             python src/eval/trees_session8.py pool --processes 12
             python src/eval/trees_session8.py testpreds --threads 12 --models-dir /root/eddm_s8/models/trees
Usage (linked shell, step 15): python3 src/eval/trees_session8.py pool-test
"""
import argparse
import datetime as dt
import hashlib
import itertools
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tasks"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
from eval import metrics as M  # noqa: E402
from eval.calibration_family_cv import folds_for  # noqa: E402
from eval.trees_session7 import frames, load_json, save_json, write_probs  # noqa: E402

SEED = 20260927
RESULTS = ROOT / "results" / "fiorillo_v0_4"
RESULTS7 = ROOT / "results" / "fiorillo_v0_3"
PUBLIC = ROOT / "results" / "public_v0" / "predictions"
RECORDS = ROOT / "data" / "processed" / "public"
MODELS = ROOT / "models" / "fiorillo_v0_4" / "trees"
QID = "readmit_window"
FOLDS = 5
MAX_TREES = 3000
EARLY_STOP = 100
POOL_FOLDS, POOL_REPEATS = 5, 20
T_MIN, T_MAX = 0.1, 10.0
GRID_STEPS = 10  # tenths
CHUNK = 128  # grid points fitted at once
PLAN = "docs/prompt_session8.md, THE READMISSION POOL (corrections 2 and 3)"
SEARCHED = {
    "xgboost_tuned": {"n_settings": 50, "file": "xgboost_tuned.json",
                      "grid": (("max_depth", (3, 4, 6, 8, 10)), ("min_child_weight", (1, 5, 10, 20)), ("eta", (0.02, 0.05, 0.1)),
                               ("subsample", (0.6, 0.8, 1.0)), ("colsample_bytree", (0.6, 0.8, 1.0)), ("lambda", (0, 1, 10)))},
    "catboost_tuned": {"n_settings": 30, "file": "catboost_tuned.cbm",
                       "grid": (("depth", (4, 6, 8, 10)), ("learning_rate", (0.03, 0.05, 0.1)), ("l2_leaf_reg", (1, 3, 10)),
                                ("bagging_temperature", (0, 1)), ("border_count", (128, 254)))},
}
MEMBERS = ("lightgbm_tuned", "xgboost_tuned", "catboost_tuned", "arm_laya", "s7_b3")


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    """A path as the repository sees it (results/...), so the pod's and the Mac's records name the same file."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def member_dir(member, results=RESULTS):
    if member in ("lightgbm_tuned", "s7_b3"):
        return RESULTS7 / "predictions" if Path(results) == RESULTS else Path(results).parent / "fiorillo_v0_3" / "predictions"
    if member == "arm_laya":
        return PUBLIC if Path(results) == RESULTS else Path(results).parent / "public_v0" / "predictions"
    return Path(results) / "predictions"


def member_file(member, split, results=RESULTS):
    return member_dir(member, results) / f"{member}_{QID}_{split}.csv"


# ---------------------------------------------------------------- the search

def settings(member):
    spec = SEARCHED[member]
    names = [n for n, _ in spec["grid"]]
    combos = list(itertools.product(*[v for _, v in spec["grid"]]))
    chosen = random.Random(SEED).sample(range(len(combos)), spec["n_settings"])
    return [{"index": k, "combination": c, **dict(zip(names, combos[c]))} for k, c in enumerate(chosen)], len(combos)


def xgb_params(s, threads, early_stopping=True):
    p = {"objective": "multi:softprob", "n_estimators": MAX_TREES, "max_depth": s["max_depth"], "min_child_weight": s["min_child_weight"],
         "learning_rate": s["eta"], "subsample": s["subsample"], "colsample_bytree": s["colsample_bytree"], "reg_lambda": s["lambda"],
         "tree_method": "hist", "enable_categorical": True, "eval_metric": "mlogloss", "random_state": SEED, "n_jobs": threads}
    if early_stopping:
        p["early_stopping_rounds"] = EARLY_STOP
    return p


def cat_params(s, threads, early_stopping=True):
    p = {"loss_function": "MultiClass", "eval_metric": "MultiClass", "iterations": MAX_TREES, "depth": s["depth"],
         "learning_rate": s["learning_rate"], "l2_leaf_reg": s["l2_leaf_reg"], "bootstrap_type": "Bayesian",
         "bagging_temperature": s["bagging_temperature"], "border_count": s["border_count"], "random_seed": SEED,
         "thread_count": threads, "allow_writing_files": False, "verbose": False}
    if early_stopping:
        p.update({"od_type": "Iter", "od_wait": EARLY_STOP, "use_best_model": True})
    return p


def cat_frame(x):
    """CatBoost's copy of a frame: the categorical columns as strings, a missing value as "missing"."""
    import pandas as pd
    x = x.copy()
    cols = [c for c in x.columns if isinstance(x[c].dtype, pd.CategoricalDtype)]
    for c in cols:
        x[c] = x[c].astype(object).where(x[c].notna(), "missing").astype(str)
    return x, cols


def fit_member(member, s, x_tr, y_tr, x_te=None, y_te=None, threads=4, n_trees=None):
    """One fit. With a held-out fold: early stopping on it, and (model, trees at the best round, library's best score).
    Without: n_trees trees, and (model, n_trees, None)."""
    early = x_te is not None
    if member == "xgboost_tuned":
        import xgboost as xgb
        params = xgb_params(s, threads, early)
        if not early:
            params["n_estimators"] = int(n_trees)
        model = xgb.XGBClassifier(**params)
        if early:
            model.fit(x_tr, y_tr, eval_set=[(x_te, y_te)], verbose=False)
            return model, int(model.best_iteration) + 1, float(model.best_score)
        model.fit(x_tr, y_tr, verbose=False)
        return model, int(n_trees), None
    import catboost as cb
    params = cat_params(s, threads, early)
    if not early:
        params["iterations"] = int(n_trees)
    xc_tr, cols = cat_frame(x_tr)
    model = cb.CatBoostClassifier(**params)
    if early:
        xc_te, _ = cat_frame(x_te)
        model.fit(xc_tr, y_tr, cat_features=cols, eval_set=(xc_te, y_te))
        return model, int(model.get_best_iteration()) + 1, float(model.get_best_score()["validation"]["MultiClass"])
    model.fit(xc_tr, y_tr, cat_features=cols)
    return model, int(n_trees), None


def predict_member(member, model, x, threads=None):
    if member == "catboost_tuned":
        x, _ = cat_frame(x)
        return np.asarray(model.predict_proba(x, thread_count=threads or -1), dtype=float)
    if threads:
        model.set_params(n_jobs=threads)
    return np.asarray(model.predict_proba(x), dtype=float)


def versions():
    out = {}
    for name in ("xgboost", "catboost", "sklearn", "pandas", "numpy"):
        try:
            out[name] = __import__(name).__version__
        except ImportError:
            out[name] = None
    return out


def search(results=RESULTS, records_dir=RECORDS, members=("xgboost_tuned", "catboost_tuned"), threads=4, max_seconds=None,
           n_settings=None):
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    progress_path = Path(results) / "trees_progress.json"
    progress = load_json(progress_path, {"plan": PLAN, "members": {}})
    progress["versions"] = versions()
    progress.setdefault("threads", {})
    data = frames(records_dir, ("train",))
    x, y, _ = data["train"]
    folds = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    done, fitted_now, total = 0, 0, 0
    for member in members:
        grid, n_combos = settings(member)
        grid = grid[: n_settings or SEARCHED[member]["n_settings"]]
        entry = progress["members"].setdefault(member, {"fits": {}})
        entry["settings_drawn"] = {"from": n_combos, "drawn": len(grid), "without_replacement": True}
        total += len(grid) * FOLDS
        for s in grid:
            for f, (tr, te) in enumerate(folds):
                key = f"{s['index']}/{f}"
                if key in entry["fits"]:
                    done += 1
                    continue
                if fitted_now and max_seconds is not None and time.time() - t0 > max_seconds:
                    save_json(progress_path, progress)  # every run fits at least once, so each run makes progress
                    print(json.dumps({"status": "stopped; run again to resume", "fits_done": done, "seconds": round(time.time() - t0, 1)}))
                    return progress
                t_fit = time.time()
                model, trees, library = fit_member(member, s, x.iloc[tr], y[tr], x.iloc[te], y[te], threads)
                probs = predict_member(member, model, x.iloc[te])
                entry["fits"][key] = {"best_iteration": trees, "log_loss": M.log_loss(y[te], probs), "library_best_score": library,
                                      "seconds": round(time.time() - t_fit, 2)}
                progress["threads"][member] = threads
                save_json(progress_path, progress)
                done += 1
                fitted_now += 1
                print(f"{member} setting {s['index']} fold {f}: {trees} trees, log loss {entry['fits'][key]['log_loss']:.5f}, "
                      f"{entry['fits'][key]['seconds']} s ({done} fits done)", flush=True)
        entry["complete"] = True
    save_json(progress_path, progress)
    print(json.dumps({"status": "complete", "fits": done, "of": total, "seconds": round(time.time() - t0, 1)}))
    return progress


def summarize_search(progress, member, n_settings=None):
    grid, n_combos = settings(member)
    grid = grid[: n_settings or SEARCHED[member]["n_settings"]]
    fits_all = progress.get("members", {}).get(member, {}).get("fits", {})
    table = []
    for s in grid:
        fits = [fits_all.get(f"{s['index']}/{f}") for f in range(FOLDS)]
        if any(v is None for v in fits):
            raise SystemExit(f"{member}: setting {s['index']} is not finished; run search again")
        table.append({**s, "mean_out_of_fold_log_loss": float(np.mean([v["log_loss"] for v in fits])),
                      "best_iterations": [v["best_iteration"] for v in fits],
                      "n_estimators": int(round(np.mean([v["best_iteration"] for v in fits])))})
    best = min(table, key=lambda r: (r["mean_out_of_fold_log_loss"], r["index"]))
    return table, best, n_combos


def save_model(member, model, models_dir):
    path = Path(models_dir) / SEARCHED[member]["file"]
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def load_model(member, models_dir):
    path = Path(models_dir) / SEARCHED[member]["file"]
    if not path.exists():
        raise SystemExit(f"{path} does not exist; run validation first")
    if member == "xgboost_tuned":
        import xgboost as xgb
        model = xgb.XGBClassifier()
    else:
        import catboost as cb
        model = cb.CatBoostClassifier()
    model.load_model(str(path))
    return model, path


def validation(results=RESULTS, records_dir=RECORDS, models_dir=MODELS, members=("xgboost_tuned", "catboost_tuned"), threads=4,
               n_settings=None):
    progress = load_json(Path(results) / "trees_progress.json", None)
    if not progress:
        raise SystemExit("trees_progress.json is missing; run search first")
    data = frames(records_dir, ("train", "validation"))
    x, y, _ = data["train"]
    xv, yv, rows = data["validation"]
    trees_path = Path(results) / "trees.json"
    out = load_json(trees_path, {"plan": PLAN, "members": {}})
    out.update({"seed": SEED, "folds": FOLDS, "max_trees": MAX_TREES, "early_stopping_rounds": EARLY_STOP, "versions": versions()})
    for member in members:
        table, best, n_combos = summarize_search(progress, member, n_settings)
        t_fit = time.time()
        model, _, _ = fit_member(member, best, x, y, threads=threads, n_trees=best["n_estimators"])
        model_path = save_model(member, model, models_dir)
        probs = predict_member(member, model, xv)
        pred_path = member_file(member, "validation", results)
        write_probs(pred_path, rows, probs)
        out["members"][member] = {
            "settings_drawn": {"from": n_combos, "drawn": len(table), "without_replacement": True}, "settings": table, "chosen": best,
            "refit": {"rows": int(len(y)), "trees": best["n_estimators"], "seconds": round(time.time() - t_fit, 2), "threads": threads},
            "model_file": model_path.name, "model_sha256": sha256_file(model_path),
            "validation": {"rows": int(len(yv)), "log_loss": M.log_loss(yv, probs), "auroc_30d": M.auroc(yv == 2, probs[:, 2]),
                           "ece": M.ece(yv, probs)},
            "predictions": rel(pred_path), "predictions_sha256": sha256_file(pred_path), "time_utc": now()}
        print(json.dumps({"member": member, "chosen": best, "validation": out["members"][member]["validation"]}, indent=1), flush=True)
    out["time_utc"] = now()
    save_json(trees_path, out)
    return out


# ---------------------------------------------------------------- the pool

def weight_grid(m, steps=GRID_STEPS):
    """Every weight vector of m members in steps of 1 / steps summing to 1, in itertools.product order."""
    return np.array([k for k in itertools.product(range(steps + 1), repeat=m) if sum(k) == steps], dtype=float) / steps


def pool_probs(logs, w, t):
    """softmax((sum_m w_m log p_m) / T); logs has shape (members, rows, answers)."""
    z = np.tensordot(np.asarray(w, dtype=float), logs, axes=(0, 0)) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit_chunk(logs, y, grid, iterations=40, tolerance=1e-8):
    """For each weight vector in grid: 1 / T by damped Newton in [0.1, 10] (session 7's fit_blend), and the log loss."""
    z = np.einsum("gm,mra->agr", grid, logs)  # (answers, grid points, rows)
    zy = np.take_along_axis(z, np.broadcast_to(y[None, None, :], (1,) + z.shape[1:]), axis=0)[0]

    def stats(beta):
        s = z * beta[None, :, None]
        smax = s.max(axis=0)
        e = np.exp(s - smax[None])
        total = e.sum(axis=0)
        p = e / total[None]
        loss = (np.log(total) + smax - beta[:, None] * zy).mean(axis=1)
        ez = (p * z).sum(axis=0)
        grad = (ez - zy).mean(axis=1)
        hess = ((p * z * z).sum(axis=0) - ez ** 2).mean(axis=1)
        return loss, grad, hess

    lo, hi = 1.0 / T_MAX, 1.0 / T_MIN
    beta = np.ones(len(grid))
    loss, grad, hess = stats(beta)
    for _ in range(iterations):
        proposal = np.clip(beta - grad / np.maximum(hess, 1e-12), lo, hi)
        new_loss, new_grad, new_hess = stats(proposal)
        worse = new_loss > loss + 1e-13  # a rise within rounding is not a rise
        if worse.any():
            proposal = np.where(worse, (beta + proposal) / 2, proposal)
            new_loss, new_grad, new_hess = stats(proposal)
        moved = np.abs(proposal - beta).max()
        beta, loss, grad, hess = proposal, new_loss, new_grad, new_hess
        if moved < tolerance:
            break
    return loss, beta


def fit_pool(logs, y, grid=None, chunk=CHUNK):
    """The grid point with the lowest log loss (the first among exact ties) and its T."""
    grid = weight_grid(logs.shape[0]) if grid is None else grid
    losses, betas = [], []
    for start in range(0, len(grid), chunk):
        loss, beta = _fit_chunk(logs, y, grid[start: start + chunk])
        losses.append(loss)
        betas.append(beta)
    losses, betas = np.concatenate(losses), np.concatenate(betas)
    k = int(np.argmin(losses))
    return {"grid_index": k, "weights": [float(v) for v in grid[k]], "T": float(1.0 / betas[k]), "validation_log_loss": float(losses[k])}


def _fold_job(args):
    logs, y, train_rows = args
    return fit_pool(logs[:, train_rows], y[train_rows])


def read_members(split, results=RESULTS, members=MEMBERS, allow_missing=True):
    """(present members, ids, labels, log-probabilities (members, rows, answers), files, not built). Rows follow the first
    present member's file; every other file must hold the same IDs with the same labels. Only a tuned member
    (xgboost_tuned, catboost_tuned) may be missing, and is then NOT BUILT; a missing uploaded member stops the run."""
    present, logs, files, not_built = [], [], {}, []
    ids = labels = None
    for member in members:
        path = member_file(member, split, results)
        if not path.exists():
            if not allow_missing or member not in SEARCHED:
                raise SystemExit(f"{rel(path)} does not exist; {member}'s file must be there")
            not_built.append(f"NOT BUILT: {member}, {rel(path)} does not exist")
            continue
        m_ids, m_labels, m_probs = M.read_predictions(path)
        if ids is None:
            ids, labels = m_ids, m_labels
            order = np.arange(len(ids))
        else:
            pos = {i: k for k, i in enumerate(m_ids)}
            if set(m_ids) != set(ids) or len(m_ids) != len(ids):
                raise SystemExit(f"{path.name} holds different rows from {Path(files[present[0]]).name}")
            order = np.array([pos[i] for i in ids])
            if not np.array_equal(m_labels[order], labels):
                raise SystemExit(f"{path.name}: labels differ from {Path(files[present[0]]).name}")
        present.append(member)
        logs.append(np.log(np.clip(np.asarray(m_probs)[order], 1e-12, None)))
        files[member] = rel(path)
    if not present:
        raise SystemExit("no member has a prediction file")
    return present, ids, labels, np.stack(logs), files, not_built


class _workers:
    """A process pool (fork, on Linux) for the out-of-fold fits, or none with processes <= 1."""

    def __init__(self, processes):
        self.processes = processes

    def __enter__(self):
        if not self.processes or self.processes <= 1:
            self.p = None
            return None
        import multiprocessing as mp
        method = "fork" if sys.platform.startswith("linux") else "spawn"
        self.p = mp.get_context(method).Pool(self.processes)
        return self.p

    def __exit__(self, *exc):
        if self.p is not None:
            self.p.close()
            self.p.join()


def pool(results=RESULTS, members=MEMBERS, processes=4, repeats=POOL_REPEATS):
    t0 = time.time()
    present, ids, y, logs, files, not_built = read_members("validation", results, members)
    grid = weight_grid(len(present))
    kept = fit_pool(logs, y, grid)
    digests = {m: sha256_file(ROOT / files[m]) if not Path(files[m]).is_absolute() else sha256_file(files[m]) for m in present}
    progress_path = Path(results) / "trees_progress.json"
    progress = load_json(progress_path, {"plan": PLAN, "members": {}})
    key = hashlib.sha256(json.dumps({"members": present, "sha256": digests, "folds": POOL_FOLDS, "seed": SEED}, sort_keys=True)
                         .encode("utf-8")).hexdigest()[:16]
    oof_store = progress.setdefault("pool_out_of_fold", {})
    if oof_store.get("key") != key:
        oof_store.clear()
        oof_store.update({"key": key, "members": present, "fits": {}})
    rng = np.random.default_rng(SEED)
    fold_sets = [folds_for(y, POOL_FOLDS, rng) for _ in range(repeats)]
    jobs = [(r, f) for r in range(repeats) for f in range(POOL_FOLDS) if f"{r}/{f}" not in oof_store["fits"]]
    with _workers(processes) as workers:
        args = [(logs, y, fold_sets[r] != f) for r, f in jobs]
        results_iter = workers.imap(_fold_job, args) if workers else (_fold_job(a) for a in args)
        for (r, f), fitted in zip(jobs, results_iter):
            oof_store["fits"][f"{r}/{f}"] = {"grid_index": fitted["grid_index"], "weights": fitted["weights"], "T": fitted["T"]}
            save_json(progress_path, progress)
    losses = []
    for r, fold in enumerate(fold_sets):
        oof = np.empty((len(y), logs.shape[2]))
        for f in range(POOL_FOLDS):
            te = fold == f
            fit_f = oof_store["fits"][f"{r}/{f}"]
            oof[te] = pool_probs(logs[:, te], fit_f["weights"], fit_f["T"])
        losses.append(M.log_loss(y, oof))
    probs = pool_probs(logs, kept["weights"], kept["T"])
    out_path = Path(results) / "predictions" / f"readmission_pool_{QID}_validation.csv"
    write_probs(out_path, [{"id": i, "label": int(v)} for i, v in zip(ids, y)], probs)
    trees = load_json(Path(results) / "trees.json", {})
    trees7 = load_json(RESULTS7 / "trees.json" if Path(results) == RESULTS else Path(results).parent / "fiorillo_v0_3" / "trees.json", {})
    search_record = {}
    for m in present:
        if m in (trees.get("members") or {}):
            t = trees["members"][m]
            search_record[m] = {"settings": t["settings"], "chosen": t["chosen"], "settings_drawn": t["settings_drawn"],
                                "source": "results/fiorillo_v0_4/trees.json"}
        elif m == "lightgbm_tuned" and trees7:
            search_record[m] = {"settings": trees7.get("settings"), "chosen": trees7.get("chosen"),
                                "settings_drawn": trees7.get("settings_drawn"), "source": "results/fiorillo_v0_3/trees.json (session 7)"}
        else:
            search_record[m] = {"source": {"arm_laya": "v0.1 as released (session 4); no search",
                                           "s7_b3": "session 7's calibrated readmission predictions; no search"}.get(m, "no search")}
    out = {"plan": PLAN, "members": present, "not_built": not_built, "rows": int(len(y)),
           "weight_grid": f"steps of 0.1 summing to 1, {len(grid)} points in itertools.product order", "T_bounds": [T_MIN, T_MAX],
           "weights": dict(zip(present, kept["weights"])), "T": kept["T"], "grid_index": kept["grid_index"],
           "validation_log_loss": M.log_loss(y, probs), "validation_log_loss_fit": kept["validation_log_loss"],
           "validation": {"auroc_30d": M.auroc(y == 2, probs[:, 2]), "ece": M.ece(y, probs)},
           "out_of_fold_log_loss": {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses)),
                                    "folds": POOL_FOLDS, "repeats": repeats, "seed": SEED,
                                    "note": ("optimistic: arm_laya's and s7_b3's calibrations were fitted on the same validation "
                                             "records, and the members' settings were chosen by cross-validation on the training records")},
           "members_alone_validation_log_loss": {m: M.log_loss(y, np.exp(logs[k])) for k, m in enumerate(present)},
           "member_files": {m: {"file": files[m], "sha256": digests[m]} for m in present},
           "member_search": search_record,
           "predictions": rel(out_path), "predictions_sha256": sha256_file(out_path), "seconds": round(time.time() - t0, 1),
           "time_utc": now()}
    save_json(Path(results) / "readmission_pool.json", out)
    print(json.dumps({k: out[k] for k in ("members", "weights", "T", "validation_log_loss", "out_of_fold_log_loss",
                                          "members_alone_validation_log_loss", "not_built")}, indent=1), flush=True)
    return out


# ---------------------------------------------------------------- after the freeze

def guard(results):
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; test predictions are made only after the freeze and the pod's test stage")
    return json.loads((Path(results) / "release_candidate.json").read_text(encoding="utf-8"))


def frozen_sha(rc, path):
    """The SHA-256 the freeze recorded for a file (by its repository path)."""
    table = rc.get("sha256") or {}
    key = rel(path)
    if key not in table:
        raise SystemExit(f"release_candidate.json records no SHA-256 for {key}")
    return table[key]


def check_frozen(rc, path):
    want, got = frozen_sha(rc, path), sha256_file(path)
    if want != got:
        raise SystemExit(f"{rel(path)} changed after the freeze (SHA-256 {got}, frozen {want}); stop")
    return got


def testpreds(results=RESULTS, records_dir=RECORDS, models_dir=MODELS, members=("xgboost_tuned", "catboost_tuned"), threads=4):
    rc = guard(results)
    data = frames(records_dir, ("train", "validation", "test"))
    xv, _, _ = data["validation"]
    xt, yt, rows = data["test"]
    log = {}
    for member in members:
        val_path = member_file(member, "validation", results)
        check_frozen(rc, val_path)
        model, model_path = load_model(member, models_dir)
        ids, _, saved = M.read_predictions(val_path)
        again = predict_member(member, model, xv, threads)
        diff = float(np.max(np.abs(again - saved)))
        if diff > 1e-8:
            raise SystemExit(f"{member}: the saved model does not reproduce its validation predictions (largest difference {diff})")
        test_path = member_file(member, "test", results)
        write_probs(test_path, rows, predict_member(member, model, xt, threads))
        log[member] = {"model": model_path.name, "model_sha256": sha256_file(model_path), "validation_reproduced_max_abs_difference": diff,
                       "test_rows": int(len(yt)), "test_predictions": rel(test_path), "time_utc": now()}
        print(f"{member}: validation reproduced (largest difference {diff:.2e}); {len(yt)} test rows written", flush=True)
    path = Path(results) / "testpreds_log.json"
    data_log = json.loads(path.read_text(encoding="utf-8"))
    data_log["trees"] = log
    save_json(path, data_log)
    return log


def pool_test(results=RESULTS, members=MEMBERS):
    rc = guard(results)
    kept = load_json(Path(results) / "readmission_pool.json", None)
    if not kept:
        raise SystemExit("readmission_pool.json is missing; run pool first")
    check_frozen(rc, Path(results) / "readmission_pool.json")
    for m in kept["members"]:
        check_frozen(rc, ROOT / kept["member_files"][m]["file"] if not Path(kept["member_files"][m]["file"]).is_absolute()
                     else Path(kept["member_files"][m]["file"]))
    present, ids, y, logs, files, _ = read_members("test", results, tuple(kept["members"]), allow_missing=False)
    if present != kept["members"]:
        raise SystemExit(f"test members {present} differ from the pool's {kept['members']}")
    probs = pool_probs(logs, [kept["weights"][m] for m in present], kept["T"])
    out_path = Path(results) / "predictions" / f"readmission_pool_{QID}_test.csv"
    write_probs(out_path, [{"id": i, "label": int(v)} for i, v in zip(ids, y)], probs)
    trees_test = {"plan": "docs/prompt_session8.md, step 15 (correction 3)", "rows": int(len(y)),
                  "members": {m: {"file": files[m], "rows": int(len(y)), "sha256": sha256_file(ROOT / files[m])
                                  if not Path(files[m]).is_absolute() else sha256_file(files[m])} for m in present},
                  "time_utc": now()}
    save_json(Path(results) / "trees_test.json", trees_test)
    out = {"plan": "docs/prompt_session8.md, THE READMISSION POOL and step 15", "members": present,
           "weights": kept["weights"], "T": kept["T"], "rows": int(len(y)), "predictions": rel(out_path),
           "predictions_sha256": sha256_file(out_path), "readmission_pool_json_sha256": frozen_sha(rc, Path(results) / "readmission_pool.json"),
           "time_utc": now()}
    save_json(Path(results) / "readmission_pool_test.json", out)
    print(json.dumps({"rows": out["rows"], "members": present, "predictions": out["predictions"]}, indent=1))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("search", "validation", "pool", "testpreds", "pool-test"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--models-dir", type=Path, default=MODELS)
    parser.add_argument("--members", nargs="+", default=None, help="search, validation and testpreds: the tuned members (default both)")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--processes", type=int, default=4, help="pool: processes for the out-of-fold fits")
    parser.add_argument("--max-seconds", type=float, default=None, help="search: start no new fit after this many seconds")
    parser.add_argument("--n-settings", type=int, default=None, help="tests only: fewer settings")
    parser.add_argument("--repeats", type=int, default=POOL_REPEATS, help="tests only: fewer out-of-fold repeats")
    args = parser.parse_args(argv)
    tuned = tuple(args.members) if args.members else ("xgboost_tuned", "catboost_tuned")
    if any(m not in SEARCHED for m in tuned):
        parser.error(f"--members takes {sorted(SEARCHED)}")
    if args.command == "search":
        search(args.results, args.records_dir, tuned, args.threads, args.max_seconds, args.n_settings)
    elif args.command == "validation":
        validation(args.results, args.records_dir, args.models_dir, tuned, args.threads, args.n_settings)
    elif args.command == "pool":
        pool(args.results, MEMBERS, args.processes, args.repeats)
    elif args.command == "testpreds":
        testpreds(args.results, args.records_dir, args.models_dir, tuned, args.threads)
    else:
        pool_test(args.results)
    return 0


if __name__ == "__main__":
    sys.exit(main())

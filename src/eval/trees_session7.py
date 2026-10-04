"""A tuned LightGBM for readmission and the readmission blends (docs/prompt_session7.md, READMISSION TREES AND BLEND;
ideas I-02 and I-11 of docs/ideas_register.md). Run on the linked shell's CPU.

lightgbm_tuned: LightGBM on the fields src/eval/baselines.py uses (its uci_frame(), categories fitted on the training
split). 50 distinct settings are drawn with the seed (random.Random(20260927).sample, without replacement) from the
1,620 combinations of num_leaves (15, 31, 63, 127), min_child_samples (10, 20, 50, 100, 200), learning_rate (0.02,
0.05, 0.1), feature_fraction (0.6, 0.8, 1.0), bagging_fraction (0.6, 0.8, 1.0; bagging_freq 1, or 0 with 1.0) and
lambda_l2 (0, 1, 10), taken in that order by itertools.product. Each setting is scored by 5-fold stratified
cross-validation on the training records (scikit-learn's StratifiedKFold, shuffled with the seed): at most 3,000 trees,
stopping after 100 rounds without improvement in the held-out fold's multi-class log loss. Progress is saved after each
setting and fold in results/fiorillo_v0_3/trees_progress.json, so the search resumes when run again; --max-seconds
stops it from starting a new fit after that many seconds. The setting with the lowest mean out-of-fold log loss is
refitted on all training records with the mean of its folds' best iterations (rounded).

Subcommands:
  search      the search (resumable).
  validation  the refit, and lightgbm_tuned's validation predictions; trees.json (every setting's score, the chosen
              setting and the number of trees).
  test        refuses to run unless results/fiorillo_v0_3/release_candidate.json and testpreds_log.json exist; refits,
              checks that the refit reproduces the validation predictions written by validation, and writes the test
              predictions; trees_test.json.
  blend       a log-linear pool of a Fiorillo system's calibrated readmission probabilities p_F and lightgbm_tuned's
              p_T, softmax((w log p_F + (1 - w) log p_T) / T): w from 0.00 to 1.00 in steps of 0.01, T fitted by maximum
              likelihood at each w on the readmission validation records, the pair with the lowest validation log loss
              kept; its out-of-fold validation log loss (5 folds, 20 repeats, the seed; w and T refitted in each fold),
              which is optimistic because p_F's own calibration was fitted on the same records; the blend's validation
              predictions; blend_<name>.json.
  blend-test  the kept w and T applied to the test predictions (the same guard as test).
Prediction files are session 4's format with probabilities printed to 9 significant digits.

Usage: python3 src/eval/trees_session7.py search [--max-seconds 150]
       python3 src/eval/trees_session7.py validation
       python3 src/eval/trees_session7.py blend --name blend_v0_1 --fiorillo results/public_v0/predictions/arm_laya_readmit_window_validation.csv
       python3 src/eval/trees_session7.py test
       python3 src/eval/trees_session7.py blend-test --name blend_v0_1 --fiorillo results/public_v0/predictions/arm_laya_readmit_window_test.csv
"""
import argparse
import csv
import datetime as dt
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

SEED = 20260927
RESULTS = ROOT / "results" / "fiorillo_v0_3"
RECORDS = ROOT / "data" / "processed" / "public"
QID = "readmit_window"
N_SETTINGS = 50
FOLDS = 5
MAX_TREES = 3000
EARLY_STOP = 100
GRID = (("num_leaves", (15, 31, 63, 127)), ("min_child_samples", (10, 20, 50, 100, 200)), ("learning_rate", (0.02, 0.05, 0.1)),
        ("feature_fraction", (0.6, 0.8, 1.0)), ("bagging_fraction", (0.6, 0.8, 1.0)), ("lambda_l2", (0, 1, 10)))
W_GRID = [round(i / 100, 2) for i in range(101)]
BLEND_FOLDS, BLEND_REPEATS = 5, 20


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def settings():
    names = [n for n, _ in GRID]
    combos = list(itertools.product(*[v for _, v in GRID]))
    chosen = random.Random(SEED).sample(range(len(combos)), N_SETTINGS)
    return [{"index": k, "combination": c, **dict(zip(names, combos[c]))} for k, c in enumerate(chosen)], len(combos)


def lgb_params(s):
    bagging_freq = 0 if s["bagging_fraction"] == 1.0 else 1
    return {"objective": "multiclass", "num_class": 3, "n_estimators": MAX_TREES, "learning_rate": s["learning_rate"],
            "num_leaves": s["num_leaves"], "min_child_samples": s["min_child_samples"], "colsample_bytree": s["feature_fraction"],
            "subsample": s["bagging_fraction"], "subsample_freq": bagging_freq, "reg_lambda": s["lambda_l2"],
            "random_state": SEED, "deterministic": True, "force_col_wise": True, "n_jobs": 4, "verbose": -1}


def frames(records_dir, splits):
    from baselines import uci_frame
    rows = {s: M_read(Path(records_dir) / f"{QID}_{s}.jsonl") for s in ("train",) + tuple(x for x in splits if x != "train")}
    train, cats = uci_frame(rows["train"])
    out = {"train": (train, np.array([r["label"] for r in rows["train"]]), rows["train"])}
    for s in splits:
        if s == "train":
            continue
        frame, _ = uci_frame(rows[s], cats)
        out[s] = (frame, np.array([r["label"] for r in rows[s]]), rows[s])
    return out


def M_read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(path, default):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def search(results=RESULTS, records_dir=RECORDS, max_seconds=150.0, n_settings=None):
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    t0 = time.time()
    progress_path = Path(results) / "trees_progress.json"
    progress = load_json(progress_path, {"plan": "docs/prompt_session7.md, READMISSION TREES AND BLEND", "lightgbm": lgb.__version__, "fits": {}})
    grid, n_combos = settings()
    grid = grid[: n_settings or N_SETTINGS]
    data = frames(records_dir, ("train",))
    x, y, _ = data["train"]
    folds = list(StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
    done, fitted_now = 0, 0
    for s in grid:
        for f, (tr, te) in enumerate(folds):
            key = f"{s['index']}/{f}"
            if key in progress["fits"]:
                done += 1
                continue
            if fitted_now and time.time() - t0 > max_seconds:  # every run fits at least once, so each run makes progress
                save_json(progress_path, progress)
                print(json.dumps({"status": "stopped; run again to resume", "fits_done": done, "fits": len(grid) * FOLDS,
                                  "seconds": round(time.time() - t0, 1)}))
                return progress
            model = lgb.LGBMClassifier(**lgb_params(s))
            t_fit = time.time()
            model.fit(x.iloc[tr], y[tr], eval_set=[(x.iloc[te], y[te])], eval_metric="multi_logloss",
                      callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            progress["fits"][key] = {"best_iteration": int(model.best_iteration_),
                                     "log_loss": float(model.best_score_["valid_0"]["multi_logloss"]),
                                     "seconds": round(time.time() - t_fit, 2)}
            save_json(progress_path, progress)
            done += 1
            fitted_now += 1
    progress["complete"] = True
    progress["settings_drawn"] = {"from": n_combos, "drawn": len(grid)}
    save_json(progress_path, progress)
    print(json.dumps({"status": "complete", "fits": done, "seconds": round(time.time() - t0, 1)}))
    return progress


def summarize_search(progress, n_settings=None):
    grid, n_combos = settings()
    grid = grid[: n_settings or N_SETTINGS]
    table = []
    for s in grid:
        fits = [progress["fits"].get(f"{s['index']}/{f}") for f in range(FOLDS)]
        if any(v is None for v in fits):
            raise SystemExit(f"setting {s['index']} is not finished; run search again")
        table.append({**s, "mean_out_of_fold_log_loss": float(np.mean([v["log_loss"] for v in fits])),
                      "best_iterations": [v["best_iteration"] for v in fits],
                      "n_estimators": int(round(np.mean([v["best_iteration"] for v in fits])))})
    best = min(table, key=lambda r: (r["mean_out_of_fold_log_loss"], r["index"]))
    return table, best, n_combos


def refit(best, data):
    import lightgbm as lgb
    params = lgb_params(best)
    params["n_estimators"] = best["n_estimators"]
    x, y, _ = data["train"]
    model = lgb.LGBMClassifier(**params)
    model.fit(x, y)
    return model


def write_probs(path, rows, probs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"] + [f"p{j}" for j in range(probs.shape[1])])
        for r, p in zip(rows, probs):
            w.writerow([r["id"], r["label"]] + [f"{x:.9g}" for x in p])


def validation(results=RESULTS, records_dir=RECORDS, n_settings=None):
    import lightgbm as lgb
    progress = load_json(Path(results) / "trees_progress.json", None)
    if not progress:
        raise SystemExit("trees_progress.json is missing; run search first")
    table, best, n_combos = summarize_search(progress, n_settings)
    data = frames(records_dir, ("train", "validation"))
    model = refit(best, data)
    xv, yv, rows = data["validation"]
    probs = model.predict_proba(xv)
    path = Path(results) / "predictions" / f"lightgbm_tuned_{QID}_validation.csv"
    write_probs(path, rows, probs)
    out = {"plan": "docs/prompt_session7.md, READMISSION TREES AND BLEND", "lightgbm": lgb.__version__, "seed": SEED,
           "settings_drawn": {"from": n_combos, "drawn": len(table), "without_replacement": True},
           "folds": FOLDS, "max_trees": MAX_TREES, "early_stopping_rounds": EARLY_STOP,
           "settings": table, "chosen": best,
           "validation": {"rows": len(yv), "log_loss": M.log_loss(yv, probs), "auroc_30d": M.auroc(yv == 2, probs[:, 2]), "ece": M.ece(yv, probs)},
           "predictions": str(path.relative_to(ROOT)) if ROOT in path.parents else str(path), "time_utc": now()}
    save_json(Path(results) / "trees.json", out)
    print(json.dumps({"chosen": best, "validation": out["validation"]}, indent=1))
    return out


def guard(results):
    for name in ("release_candidate.json", "testpreds_log.json"):
        if not (Path(results) / name).exists():
            raise SystemExit(f"{name} does not exist in {results}; test predictions are made only after the freeze and the pod's test stage")


def test(results=RESULTS, records_dir=RECORDS):
    guard(results)
    trees = load_json(Path(results) / "trees.json", None)
    data = frames(records_dir, ("train", "validation", "test"))
    model = refit(trees["chosen"], data)
    xv, _, _ = data["validation"]
    _, _, saved = M.read_predictions(Path(results) / "predictions" / f"lightgbm_tuned_{QID}_validation.csv")
    diff = float(np.max(np.abs(model.predict_proba(xv) - saved)))
    if diff > 1e-8:
        raise SystemExit(f"the refit does not reproduce the saved validation predictions (largest difference {diff})")
    xt, yt, rows = data["test"]
    path = Path(results) / "predictions" / f"lightgbm_tuned_{QID}_test.csv"
    write_probs(path, rows, model.predict_proba(xt))
    out = {"plan": "docs/prompt_session7.md, step 14", "rows": len(yt), "validation_reproduced_max_abs_difference": diff,
           "predictions": path.name, "time_utc": now()}
    save_json(Path(results) / "trees_test.json", out)
    print(json.dumps(out, indent=1))
    return out


def pool(lf, lt, w, t):
    z = (w * lf + (1 - w) * lt) / t
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_blend(lf, lt, y, iterations=40, tolerance=1e-8):
    """For every w on the grid, T by maximum likelihood, T in [0.1, 10]; the pair with the lowest validation log loss
    (the lowest w among exact ties). The negative log likelihood of softmax(z / T) is convex in 1 / T, so 1 / T is
    found by a damped Newton method, run for all w at once and kept inside [0.1, 10]."""
    w = np.array(W_GRID)[None, :, None]
    z = w * lf.T[:, None, :] + (1 - w) * lt.T[:, None, :]  # (answers, w, rows)
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

    beta = np.ones(len(W_GRID))
    loss, grad, hess = stats(beta)
    for _ in range(iterations):
        proposal = np.clip(beta - grad / np.maximum(hess, 1e-12), 0.1, 10.0)
        new_loss, new_grad, new_hess = stats(proposal)
        worse = new_loss > loss + 1e-13  # a rise within rounding is not a rise
        if worse.any():
            proposal = np.where(worse, (beta + proposal) / 2, proposal)
            new_loss, new_grad, new_hess = stats(proposal)
        moved = np.abs(proposal - beta).max()
        beta, loss, grad, hess = proposal, new_loss, new_grad, new_hess
        if moved < tolerance:
            break
    k = int(np.argmin(loss))
    return {"validation_log_loss": float(loss[k]), "w": W_GRID[k], "T": float(1.0 / beta[k])}


def read_aligned(path_f, path_t):
    ids_f, y_f, p_f = M.read_predictions(path_f)
    ids_t, y_t, p_t = M.read_predictions(path_t)
    pos = {i: k for k, i in enumerate(ids_t)}
    if set(ids_f) != set(ids_t):
        raise SystemExit(f"{Path(path_f).name} and {Path(path_t).name} hold different rows")
    order = np.array([pos[i] for i in ids_f])
    if not np.array_equal(y_f, y_t[order]):
        raise SystemExit("labels differ between the two prediction files")
    return ids_f, y_f, np.log(np.clip(p_f, 1e-12, None)), np.log(np.clip(p_t[order], 1e-12, None))


class _pool:
    """A process pool (fork, on Linux) for the out-of-fold blend fits, or none with processes <= 1."""

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


def blend(name, fiorillo_path, results=RESULTS, processes=4):
    tree_path = Path(results) / "predictions" / f"lightgbm_tuned_{QID}_validation.csv"
    ids, y, lf, lt = read_aligned(fiorillo_path, tree_path)
    kept = fit_blend(lf, lt, y)
    rng = np.random.default_rng(SEED)
    fold_sets = [folds_for(y, BLEND_FOLDS, rng) for _ in range(BLEND_REPEATS)]
    jobs = [(lf[fold != f], lt[fold != f], y[fold != f]) for fold in fold_sets for f in range(BLEND_FOLDS)]
    with _pool(processes) as workers:
        fitted = workers.starmap(fit_blend, jobs) if workers else [fit_blend(*j) for j in jobs]
    losses = []
    for r, fold in enumerate(fold_sets):
        oof = np.empty_like(lf)
        for f in range(BLEND_FOLDS):
            te = fold == f
            fit_f = fitted[r * BLEND_FOLDS + f]
            oof[te] = pool(lf[te], lt[te], fit_f["w"], fit_f["T"])
        losses.append(float(np.mean(-np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, None)))))
    probs = pool(lf, lt, kept["w"], kept["T"])
    rows = [{"id": i, "label": int(v)} for i, v in zip(ids, y)]
    out_path = Path(results) / "predictions" / f"{name}_{QID}_validation.csv"
    write_probs(out_path, rows, probs)
    out = {"plan": "docs/prompt_session7.md, READMISSION TREES AND BLEND", "name": name,
           "fiorillo_validation": str(fiorillo_path), "trees_validation": tree_path.name, "rows": len(y),
           "w_grid": "0.00 to 1.00 in steps of 0.01", "T_bounds": [0.1, 10.0], **kept,
           "out_of_fold_log_loss": {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses)),
                                    "folds": BLEND_FOLDS, "repeats": BLEND_REPEATS,
                                    "note": "optimistic: the Fiorillo system's own calibration was fitted on the same validation records"},
           "fiorillo_alone_log_loss": M.log_loss(y, np.exp(lf)), "trees_alone_log_loss": M.log_loss(y, np.exp(lt)),
           "time_utc": now()}
    save_json(Path(results) / f"{name}.json", out)
    print(json.dumps({k: out[k] for k in ("name", "w", "T", "validation_log_loss", "out_of_fold_log_loss", "fiorillo_alone_log_loss",
                                          "trees_alone_log_loss")}, indent=1))
    return out


def blend_test(name, fiorillo_path, results=RESULTS):
    guard(results)
    kept = load_json(Path(results) / f"{name}.json", None)
    if not kept:
        raise SystemExit(f"{name}.json is missing; run blend first")
    fitted_on = Path(kept["fiorillo_validation"]).name
    if Path(fiorillo_path).name != fitted_on.replace("_validation.csv", "_test.csv"):
        raise SystemExit(f"{name} was fitted on {fitted_on}; {Path(fiorillo_path).name} is not the same system's test file")
    tree_path = Path(results) / "predictions" / f"lightgbm_tuned_{QID}_test.csv"
    ids, y, lf, lt = read_aligned(fiorillo_path, tree_path)
    probs = pool(lf, lt, kept["w"], kept["T"])
    rows = [{"id": i, "label": int(v)} for i, v in zip(ids, y)]
    write_probs(Path(results) / "predictions" / f"{name}_{QID}_test.csv", rows, probs)
    print(f"{name}: {len(ids)} test rows written")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("search", "validation", "test", "blend", "blend-test"))
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--records-dir", type=Path, default=RECORDS)
    parser.add_argument("--max-seconds", type=float, default=150.0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--fiorillo", type=Path, default=None)
    parser.add_argument("--n-settings", type=int, default=None, help="tests only: fewer settings")
    parser.add_argument("--processes", type=int, default=4, help="processes for the blend's out-of-fold fits")
    args = parser.parse_args(argv)
    if args.command == "search":
        search(args.results, args.records_dir, args.max_seconds, args.n_settings)
    elif args.command == "validation":
        validation(args.results, args.records_dir, args.n_settings)
    elif args.command == "test":
        test(args.results, args.records_dir)
    elif args.command == "blend":
        blend(args.name, args.fiorillo, args.results, args.processes)
    else:
        blend_test(args.name, args.fiorillo, args.results)


if __name__ == "__main__":
    main()

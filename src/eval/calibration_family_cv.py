"""Validation-only comparison of two calibration families for one question (correction 7, point 5; session 5 input).

Families, each fitted by maximum likelihood on raw (uncalibrated) probabilities p:
- temperature: softmax(log p / T), T in [0.1, 10] (what train_laya.py fits and the laya runtime applies);
- temperature_intercepts: softmax(log p / T + b), b[0] = 0 (bias-corrected temperature scaling; it can move one
  answer's share, such as maybe, without flattening the others; the laya runtime cannot apply b today).
Only validation items marked calibrate are used, and every score is out of fold: repeated stratified K-fold, the
family fitted on K-1 folds and scored on the held-out fold; pooled out-of-fold log loss, ECE (top answer, the
project's bins) and per-label log loss; also the 95 percent bootstrap range of the temperature fitted on all the items.
No test split is read. Exploratory: it has no bearing on the release bar.
Session 9 (docs/prompt_session9.md, CALIBRATION and THE RECIPE RULE) adds four families as functions, which
calibrate_session9.py and recipe_session9.py call: fit_family(), apply_family(), out_of_fold() and choose(). The
command (main()) is unchanged and still scores only temperature and temperature_intercepts; the functions fit and apply
those two through fit() and calibrated(), with their parameters as calibration_s8.json stores them. The new families:
- vector: softmax(s * log p + b), a scale s_j and an intercept b_j for each answer, with b_0 = 0 because adding the same
  constant to every intercept changes nothing (2k - 1 free parameters, 5 for three answers); each s_j in [0.1, 10] (the
  temperature family's range of 1/T) and each free b_j in [-10, 10]; started at s = 1 and b = 0 (the identity map).
- dirichlet_odir: softmax(W log p + b), W a k by k matrix and b an intercept for each answer, with the off-diagonal and
  intercept regularization (ODIR) of Kull et al. 2019 ("Beyond temperature scaling", section 3): it minimizes the mean
  negative log-likelihood, plus lambda_od times the mean of W's k(k - 1) squared off-diagonal entries, plus lambda_b
  times the mean of the k squared intercepts, with every entry of W and b in [-10, 10]. Each strength is one of 0.0001,
  0.001, 0.01, 0.1 and 1 (25 pairs), chosen by 5-fold cross-validation inside each fitting set (odir_select()); the
  family is then refitted on the whole fitting set with the chosen pair, from W = I and b = 0 (Kull et al. averaged the
  inner folds' models instead; the plan refits).
Both are fitted by maximum likelihood (the exact log-softmax, nothing clipped) with L-BFGS-B, analytic gradients and
LBFGS_OPTIONS (with scipy's default tolerances, some ODIR fits stopped early enough to move a pair's inner log loss by up
to 5e-4 on made-up data, more than the gaps between neighboring pairs); every out-of-fold log loss is scored as main()
scores it, with the probabilities clipped at 1e-12.

Usage: python src/eval/calibration_family_cv.py --predictions DIR/arm_laya-raw_pubmedqa_answer_validation.csv
       --records data/processed/public/pubmedqa_answer_validation.jsonl [--folds 5] [--repeats 20] [--bootstrap 1000] [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval import metrics as M  # noqa: E402

SEED = 20260927
T_MIN, T_MAX = 0.1, 10.0


def calibrated(logp, params, k):
    t = float(np.clip(np.exp(params[0]), T_MIN, T_MAX))
    b = np.zeros(k)
    if len(params) > 1:
        b[1:] = params[1:]
    z = logp / t + b
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(logp, y, family):
    k = logp.shape[1]
    start = np.zeros(1 if family == "temperature" else k)

    def nll(params):
        p = calibrated(logp, params, k)
        return -np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)))

    bounds = [(np.log(T_MIN), np.log(T_MAX))] + [(-10.0, 10.0)] * (len(start) - 1)
    return minimize(nll, start, method="L-BFGS-B", bounds=bounds).x


def folds_for(y, k_folds, rng):
    """Stratified fold labels: each answer's items dealt round-robin after a shuffle."""
    fold = np.empty(len(y), dtype=int)
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + rng.integers(k_folds)) % k_folds
    return fold


# ---------------------------------------------------------------- session 9: the four families as functions

FOLDS, REPEATS = 5, 20
FAMILIES = ("temperature", "temperature_intercepts", "vector", "dirichlet_odir")
TIE = 1e-12  # scores within this of the lowest are an exact tie, which goes to the first of them in their order
S_MIN, S_MAX = 1.0 / T_MAX, 1.0 / T_MIN  # vector: each answer's scale in [0.1, 10], the temperature family's range of 1/T
B_MAX = 10.0  # vector: each free intercept in [-10, 10], the bound fit() gives temperature_intercepts' intercepts
ODIR_GRID = (0.0001, 0.001, 0.01, 0.1, 1.0)  # dirichlet_odir: the strengths tried for each penalty (25 pairs)
ODIR_INNER_FOLDS = 5
ODIR_BOUND = 10.0  # dirichlet_odir: every entry of W and b in [-10, 10]
LBFGS_OPTIONS = {"ftol": 0.0, "gtol": 1e-8, "maxcor": 20}  # vector and dirichlet_odir: stop once no projected gradient entry exceeds 1e-8


def lowest(scores, tie=TIE):
    """Index of the lowest score; the scores within tie of it are an exact tie, which goes to the first of them."""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError(f"scores must be finite and not empty: {scores.tolist()}")
    return int(np.flatnonzero(scores <= scores.min() + tie)[0])


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _finite(logp):
    logp = np.asarray(logp, dtype=float)
    if logp.ndim != 2 or not np.all(np.isfinite(logp)):
        raise ValueError("logp must be a finite array of items by answers (clip probabilities at 1e-12 before the log, as main() does)")
    return logp


def _onehot(y, k):
    out = np.zeros((len(y), k))
    out[np.arange(len(y)), y] = 1.0
    return out


def _vector_objective(theta, logp, rows, onehot):
    """The mean negative log-likelihood of softmax(s * log p + b), b_0 = 0, and its gradient; theta = [s_0, ..., b_1, ...]."""
    n, k = logp.shape
    z = logp * theta[:k]
    z[:, 1:] += theta[k:]
    z -= z.max(axis=1, keepdims=True)
    logq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    r = (np.exp(logq) - onehot) / n
    return -logq[rows].mean(), np.concatenate(((r * logp).sum(axis=0), r[:, 1:].sum(axis=0)))


def _vector_fit(logp, y):
    n, k = logp.shape
    start = np.concatenate((np.ones(k), np.zeros(k - 1)))  # the identity map
    bounds = [(S_MIN, S_MAX)] * k + [(-B_MAX, B_MAX)] * (k - 1)
    res = minimize(_vector_objective, start, args=(logp, (np.arange(n), y), _onehot(y, k)), jac=True, method="L-BFGS-B", bounds=bounds,
                   options=dict(LBFGS_OPTIONS))
    return [float(v) for v in res.x]


def odir_objective(theta, logp, y, lambda_od, lambda_b):
    """dirichlet_odir's objective and its gradient, theta = W's rows, then b (Kull et al. 2019, section 3, ODIR, with
    lambda = lambda_od and mu = lambda_b): the mean negative log-likelihood of softmax(W log p + b), plus lambda_od times the
    mean of W's k(k - 1) squared off-diagonal entries, plus lambda_b times the mean of the k squared intercepts."""
    logp, y = np.asarray(logp, dtype=float), np.asarray(y)
    return _odir_objective(np.asarray(theta, dtype=float), logp, (np.arange(len(y)), y), _onehot(y, logp.shape[1]), lambda_od, lambda_b)


def _odir_objective(theta, logp, rows, onehot, lambda_od, lambda_b):
    n, k = logp.shape
    w, b = theta[:k * k].reshape(k, k), theta[k * k:]
    off = 1.0 - np.eye(k)
    z = logp @ w.T + b
    z -= z.max(axis=1, keepdims=True)
    logq = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    r = (np.exp(logq) - onehot) / n
    value = -logq[rows].mean() + lambda_od * ((w * off) ** 2).sum() / (k * (k - 1)) + lambda_b * (b ** 2).mean()
    grad_w = r.T @ logp + (2.0 * lambda_od / (k * (k - 1))) * w * off
    grad_b = r.sum(axis=0) + (2.0 * lambda_b / k) * b
    return value, np.concatenate((grad_w.ravel(), grad_b))


def _odir_fit(logp, y, lambda_od, lambda_b, start=None):
    """theta (W's rows, then b) that minimizes the ODIR objective, from start (W = I and b = 0 when start is None)."""
    n, k = logp.shape
    if start is None:
        start = np.concatenate((np.eye(k).ravel(), np.zeros(k)))
    res = minimize(_odir_objective, start, args=(logp, (np.arange(n), y), _onehot(y, k), lambda_od, lambda_b), jac=True,
                   method="L-BFGS-B", bounds=[(-ODIR_BOUND, ODIR_BOUND)] * (k * k + k), options=dict(LBFGS_OPTIONS))
    return res.x


def _odir_apply(theta, logp):
    k = logp.shape[1]
    return _softmax(logp @ theta[:k * k].reshape(k, k).T + theta[k * k:])


def odir_select(logp, y, seed=SEED):
    """dirichlet_odir's penalty pair, chosen by 5-fold cross-validation inside (logp, y), its folds drawn by
    folds_for(y, 5, np.random.default_rng(seed)). Each pair's score is its held-out log loss pooled over every item
    (probabilities clipped at 1e-12, as the out-of-fold scores are); the lowest wins, and a tie (within 1e-12) goes to the
    first pair in the order (lambda_od, lambda_b) ascending. Within one inner fold the 25 fits follow a path (lambda_od
    ascending; lambda_b ascending, then descending for the next lambda_od, and so on), the first from W = I and b = 0 and
    each later one from the fit before it (a warm start: the objective is convex, so the start changes the time, not the
    solution beyond the optimizer's tolerance).
    Returns ((lambda_od, lambda_b), {"pairs": [[lambda_od, lambda_b], ...], "inner_log_loss": [...]}), pairs in that order."""
    logp, y = _finite(logp), np.asarray(y)
    g = len(ODIR_GRID)
    pairs = [(a, b) for a in ODIR_GRID for b in ODIR_GRID]
    path = [i * g + (j if i % 2 == 0 else g - 1 - j) for i in range(g) for j in range(g)]
    fold = folds_for(y, ODIR_INNER_FOLDS, np.random.default_rng(seed))
    total = np.zeros(len(pairs))
    for f in range(ODIR_INNER_FOLDS):
        test = fold == f
        if not test.any():
            continue
        train_logp, train_y, test_logp = logp[~test], y[~test], logp[test]
        test_rows = (np.arange(len(test_logp)), y[test])
        theta = None
        for p in path:
            theta = _odir_fit(train_logp, train_y, *pairs[p], start=theta)
            total[p] += -np.log(np.clip(_odir_apply(theta, test_logp)[test_rows], 1e-12, None)).sum()
    scores = total / len(y)
    return pairs[lowest(scores)], {"pairs": [list(p) for p in pairs], "inner_log_loss": [float(s) for s in scores]}


def fit_family(name, logp, y, seed=SEED):
    """name's parameters, fitted on (logp, y), in the form apply_family() takes and calibration_s9.json stores:
    temperature [log T] and temperature_intercepts [log T, b_1, ..., b_(k-1)] (fit(), as calibration_s8.json stores them);
    vector [s_0, ..., s_(k-1), b_1, ..., b_(k-1)] (b_0 = 0); dirichlet_odir {"W": k rows of k numbers, "b": k numbers,
    "lambda_od": the chosen strength, "lambda_b": the chosen strength}, where W's row j holds the weights of log p_0, ...,
    log p_(k-1) in answer j's score. seed draws dirichlet_odir's inner folds; the other families do not use it."""
    if name in ("temperature", "temperature_intercepts"):
        return [float(v) for v in fit(np.asarray(logp, dtype=float), np.asarray(y), name)]
    if name == "vector":
        return _vector_fit(_finite(logp), np.asarray(y))
    if name == "dirichlet_odir":
        logp, y = _finite(logp), np.asarray(y)
        k = logp.shape[1]
        (lambda_od, lambda_b), _ = odir_select(logp, y, seed)
        theta = _odir_fit(logp, y, lambda_od, lambda_b)
        return {"W": theta[:k * k].reshape(k, k).tolist(), "b": theta[k * k:].tolist(), "lambda_od": lambda_od, "lambda_b": lambda_b}
    raise ValueError(f"unknown calibration family {name!r}; the families are {FAMILIES}")


def apply_family(name, params, logp):
    """Calibrated probabilities (items by answers): name with params, as fit_family() returns them, applied to logp."""
    if name in ("temperature", "temperature_intercepts"):
        logp = np.asarray(logp, dtype=float)
        k = logp.shape[1]
        if len(params) != (1 if name == "temperature" else k):
            raise ValueError(f"{name} takes {1 if name == 'temperature' else k} parameters, not {len(params)}")
        return calibrated(logp, params, k)
    if name not in FAMILIES:
        raise ValueError(f"unknown calibration family {name!r}; the families are {FAMILIES}")
    logp = _finite(logp)
    k = logp.shape[1]
    if name == "vector":
        if len(params) != 2 * k - 1:
            raise ValueError(f"vector takes {2 * k - 1} parameters, not {len(params)}")
        z = logp * np.asarray(params[:k], dtype=float)
        z[:, 1:] += np.asarray(params[k:], dtype=float)
        return _softmax(z)
    w, b = np.asarray(params["W"], dtype=float), np.asarray(params["b"], dtype=float)
    if w.shape != (k, k) or b.shape != (k,):
        raise ValueError(f"dirichlet_odir takes W of {k} by {k} and b of {k}, not {w.shape} and {b.shape}")
    return _softmax(logp @ w.T + b)


def out_of_fold(name, logp, y, folds=FOLDS, repeats=REPEATS, seed=SEED):
    """name's out-of-fold scores on (logp, y), computed as main() computes them: the fold sets of repeated stratified
    K-fold drawn as main() draws them (rng = np.random.default_rng(seed); folds_for(y, folds, rng) for each repeat), the
    family fitted with fit_family() on each training part (dirichlet_odir with its own inner folds, from the same seed)
    and applied to the held-out part. Returns (summary, probs): summary holds out_of_fold_log_loss and out_of_fold_ece
    (mean, min and max over the repeats) and out_of_fold_log_loss_by_label, the keys and numbers main() writes; probs is
    the out-of-fold probabilities of every repeat, an array of repeats by items by answers."""
    if name not in FAMILIES:
        raise ValueError(f"unknown calibration family {name!r}; the families are {FAMILIES}")
    logp, y = np.asarray(logp, dtype=float), np.asarray(y)
    n, k = logp.shape
    rng = np.random.default_rng(seed)
    fold_sets = [folds_for(y, folds, rng) for _ in range(repeats)]
    probs = np.empty((repeats, n, k))
    losses, eces, per_label = [], [], []
    for r, fold in enumerate(fold_sets):
        oof = probs[r]
        for f in range(folds):
            test = fold == f
            params = fit_family(name, logp[~test], y[~test], seed)
            oof[test] = apply_family(name, params, logp[test])
        item_loss = -np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, None))
        losses.append(item_loss.mean())
        eces.append(M.ece(y, oof))
        per_label.append([item_loss[y == c].mean() if (y == c).any() else np.nan for c in range(k)])
    summary = {"out_of_fold_log_loss": {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses))},
               "out_of_fold_ece": {"mean": float(np.mean(eces)), "min": float(np.min(eces)), "max": float(np.max(eces))},
               "out_of_fold_log_loss_by_label": [float(v) for v in np.nanmean(np.array(per_label), axis=0)]}
    return summary, probs


def choose(logp, y, families=FAMILIES, folds=FOLDS, repeats=REPEATS, seed=SEED):
    """The family with the lowest mean out-of-fold log loss (an exact tie, within 1e-12, goes to the family listed first).
    Returns (result, probs): result = {"families": {family: out_of_fold()'s summary with "params", the family fitted on
    all items}, "chosen", "params" (the chosen family's), "folds", "repeats", "seed", "rule"}, ready for JSON; probs =
    {family: out_of_fold()'s probabilities, repeats by items by answers}."""
    families = tuple(families)
    if not families or len(set(families)) != len(families) or any(f not in FAMILIES for f in families):
        raise ValueError(f"families must be distinct names from {FAMILIES}, not {families}")
    logp, y = np.asarray(logp, dtype=float), np.asarray(y)
    table, probs = {}, {}
    for family in families:
        table[family], probs[family] = out_of_fold(family, logp, y, folds, repeats, seed)
        table[family]["params"] = fit_family(family, logp, y, seed)
    chosen = families[lowest([table[f]["out_of_fold_log_loss"]["mean"] for f in families])]
    result = {"families": table, "chosen": chosen, "params": table[chosen]["params"], "folds": folds, "repeats": repeats, "seed": seed,
              "rule": "the lowest mean out-of-fold log loss; an exact tie (within 1e-12) goes to the family listed first"}
    return result, probs


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--predictions", type=Path, required=True, help="a -raw validation predictions CSV")
    parser.add_argument("--records", type=Path, required=True, help="the validation records the model was checked on")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000, help="resamples for the temperature's 95 percent range")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    keep = set()
    for line in args.records.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("calibrate", True):
            keep.add(r["id"])
    ids, labels, probs = M.read_predictions(args.predictions)
    rows = np.array([i for i, x in enumerate(ids) if x in keep])
    if len(rows) != len(keep):
        raise SystemExit(f"{len(keep)} calibrate records but {len(rows)} of them in {args.predictions.name}")
    y, logp = np.asarray(labels)[rows], np.log(np.clip(np.asarray(probs)[rows], 1e-12, None))
    k = logp.shape[1]
    rng = np.random.default_rng(SEED)
    out = {"predictions": args.predictions.name, "n": int(len(y)), "label_counts": np.bincount(y, minlength=k).tolist(),
           "folds": args.folds, "repeats": args.repeats, "families": {}}
    raw = np.exp(logp)
    out["raw"] = {"log_loss": M.log_loss(y, raw), "ece": M.ece(y, raw)}
    boot_rng = np.random.default_rng(SEED + 1)  # separate from the folds' generator
    boot = [fit(logp[idx], y[idx], "temperature")[0] for idx in (boot_rng.integers(len(y), size=len(y)) for _ in range(args.bootstrap))]
    out["temperature_bootstrap"] = {"resamples": args.bootstrap, "range_95": [float(np.clip(np.exp(q), T_MIN, T_MAX)) for q in np.quantile(boot, [0.025, 0.975])]}
    fold_sets = [folds_for(y, args.folds, rng) for _ in range(args.repeats)]
    for family in ("temperature", "temperature_intercepts"):
        full = fit(logp, y, family)
        losses, eces, per_label = [], [], []
        for fold in fold_sets:
            oof = np.empty_like(logp)
            for f in range(args.folds):
                test = fold == f
                params = fit(logp[~test], y[~test], family)
                oof[test] = calibrated(logp[test], params, k)
            item_loss = -np.log(np.clip(oof[np.arange(len(y)), y], 1e-12, None))
            losses.append(item_loss.mean())
            eces.append(M.ece(y, oof))
            per_label.append([item_loss[y == c].mean() if (y == c).any() else np.nan for c in range(k)])
        out["families"][family] = {
            "fitted_on_all": {"temperature": float(np.clip(np.exp(full[0]), T_MIN, T_MAX)),
                              "intercepts": [0.0] + [float(v) for v in full[1:]]},
            "out_of_fold_log_loss": {"mean": float(np.mean(losses)), "min": float(np.min(losses)), "max": float(np.max(losses))},
            "out_of_fold_ece": {"mean": float(np.mean(eces)), "min": float(np.min(eces)), "max": float(np.max(eces))},
            "out_of_fold_log_loss_by_label": [float(v) for v in np.nanmean(np.array(per_label), axis=0)],
        }
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

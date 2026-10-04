"""Metrics for public v0 (docs/prompt_session4.md, step 9): one implementation used for every system.

Inputs are aligned arrays: labels (option index per row) and probabilities (rows by options). Log loss is the
mean of -log p(label) (targets are one-hot). The Brier score sums squared errors over options. ECE uses 15
equal-width bins on top-label confidence, with the bin edges of laya.common.ece_score (the first bin closed, the
others open below and closed above). Macro-F1 averages F1 over the classes present in the labels or the
predictions (the scikit-learn default). Quadratic weighted kappa uses the argmax level. AUROC is the Mann-Whitney
statistic with ties counted half. For a binary outcome y and predicted probability p: the calibration intercept is
the intercept of a logistic regression of y with logit(p) as an offset (0 is ideal); the calibration slope is the
coefficient of logit(p) in a logistic regression of y on logit(p) (1 is ideal); observed/expected is
sum(y) / sum(p). Decision weights (docs/prompt_session4.md, correction 3) are the level weights that maximize
macro-F1 of argmax(probability times weight) on the rows given; they are fitted on validation rows and then used on
test rows. Confidence intervals come from paired bootstrap resamples of rows (percentile method).
"""
import csv

import numpy as np

EPS = 1e-12
NAN = float("nan")


def read_predictions(path):
    """id list, label array, probability matrix from a predictions CSV (id, label, p0, p1, ...)."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    k = sum(1 for c in rows[0] if c.startswith("p") and c[1:].isdigit())
    ids = [r["id"] for r in rows]
    labels = np.array([int(r["label"]) for r in rows])
    probs = np.array([[float(r[f"p{j}"]) for j in range(k)] for r in rows])
    return ids, labels, probs


def log_loss(labels, probs):
    return float(np.mean(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, None))))


def brier(labels, probs):
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(((probs - onehot) ** 2).sum(1)))


def ece(labels, probs, bins=15):
    conf = probs.max(1)
    correct = (probs.argmax(1) == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.searchsorted(edges[1:-1], conf, side="left")  # bin i holds edges[i] < conf <= edges[i+1]; bin 0 includes 0
    gap = np.bincount(which, conf, bins) - np.bincount(which, correct, bins)
    return float(np.abs(gap).sum() / len(conf))


def accuracy(labels, probs):
    return float(np.mean(probs.argmax(1) == labels))


def macro_f1(labels, probs):
    preds = probs.argmax(1)
    scores = []
    for c in sorted(set(labels.tolist()) | set(preds.tolist())):
        tp = np.sum((preds == c) & (labels == c))
        fp = np.sum((preds == c) & (labels != c))
        fn = np.sum((preds != c) & (labels == c))
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(scores))


def decision_weights(labels, probs, low=0.25, high=16.0, steps=49):
    """Weights for a three-level question (level 0 fixed at 1; levels 1 and 2 each from `steps` values evenly spaced
    on the log scale from `low` to `high`) that maximize macro-F1 of argmax(probs * weights) on these rows. Ties go to
    the pair nearest to no change (smallest |log w1| + |log w2|). Returns the weights and their macro-F1."""
    if probs.shape[1] != 3:
        raise ValueError("decision_weights is defined for three levels")
    grid = np.exp(np.linspace(np.log(low), np.log(high), steps))
    best_f1, best_cost, best = -1.0, np.inf, np.ones(3)
    for w1 in grid:
        for w2 in grid:
            w = np.array([1.0, w1, w2])
            f1 = macro_f1(labels, probs * w)
            cost = abs(np.log(w1)) + abs(np.log(w2))
            if f1 > best_f1 + 1e-12 or (abs(f1 - best_f1) <= 1e-12 and cost < best_cost):
                best_f1, best_cost, best = f1, cost, w
    return best, float(best_f1)


def quadratic_weighted_kappa(labels, probs):
    k = probs.shape[1]
    preds = probs.argmax(1)
    observed = np.bincount(labels * k + preds, minlength=k * k).reshape(k, k).astype(float)
    weights = (np.subtract.outer(np.arange(k), np.arange(k)) ** 2) / (k - 1) ** 2
    expected = np.outer(observed.sum(1), observed.sum(0)) / observed.sum()
    denominator = (weights * expected).sum()
    return float(1.0 - (weights * observed).sum() / denominator) if denominator > 0 else NAN


def auroc(y, score):
    from scipy.stats import rankdata

    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return NAN
    ranks = rankdata(score, method="average")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _logistic_fit(design, y, offset, iterations=50):
    """Newton-Raphson for logit P(y=1) = design @ beta + offset; nan when it does not converge to finite values."""
    beta = np.zeros(design.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(design @ beta + offset)))
        gradient = design.T @ (y - p)
        hessian = design.T @ (design * (p * (1 - p))[:, None])
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return np.full(design.shape[1], NAN)
        beta += step
        if not np.all(np.isfinite(beta)):
            return np.full(design.shape[1], NAN)
        if np.max(np.abs(step)) < 1e-10:
            break
    return beta


def calibration_intercept(y, p):
    y = np.asarray(y, dtype=float)
    if y.min() == y.max():
        return NAN
    lp = _logit(p)
    return float(_logistic_fit(np.ones((len(y), 1)), y, lp)[0])


def calibration_slope(y, p):
    y = np.asarray(y, dtype=float)
    lp = _logit(p)
    if y.min() == y.max() or np.ptp(lp) < 1e-9:
        return NAN
    return float(_logistic_fit(np.column_stack([np.ones_like(lp), lp]), y, np.zeros_like(lp))[1])


def observed_expected(y, p):
    return float(np.sum(y) / max(EPS, np.sum(p)))


def bootstrap(stat, n, resamples=1000, seed=20260927):
    """Estimate and 95 percent percentile interval of stat(row_indices) over paired resamples of n rows.

    stat may return one number or a vector; the result has the same shape (floats, or arrays for a vector).
    The same seed and n give the same resamples, so two systems scored on the same rows are paired.
    Resamples where a statistic is not finite (for example AUROC with no positive rows) are left out of its
    interval.
    """
    rng = np.random.default_rng(seed)
    first = stat(np.arange(n))
    full = np.atleast_1d(np.asarray(first, dtype=float))
    values = np.array([np.atleast_1d(np.asarray(stat(rng.integers(0, n, n)), dtype=float)) for _ in range(resamples)])
    low, high = np.full(len(full), NAN), np.full(len(full), NAN)
    for j in range(len(full)):
        v = values[:, j][np.isfinite(values[:, j])]
        if len(v):
            low[j], high[j] = np.percentile(v, [2.5, 97.5])
    if np.ndim(first) == 0:
        return float(full[0]), float(low[0]), float(high[0])
    return full, low, high

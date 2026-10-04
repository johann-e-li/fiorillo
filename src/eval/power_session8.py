"""Power of v0.4's bar, criterion 3 (docs/prompt_session8.md, POWER AND THE BAR and steps 1 and 12).

Criterion 3 compares s8_ei with gemma4_31b (both calibrated on validation) on Evidence Inference 2.0's (EI's) test log
loss, as the paired difference "s8_ei minus gemma4_31b" (negative is a gain), and passes when the 95 percent interval of
the difference, from resampling articles, is wholly below zero.

Method (derived; normal approximation). With a per-prompt standard deviation s of the log-loss difference and a design
effect D for the articles (the variance of the mean difference when articles are resampled, divided by its variance
when prompts are resampled), the standard error on the n = 1,218 test prompts is se = s sqrt(D / n). The criterion passes
when the estimate falls below -z se, with z the 97.5th percentile of the standard normal (1.9600); at a real gain g its
chance is Phi(g / se - z); the real gain that gives an 80 percent chance is (z + z80) se, with z80 the 80th percentile.

Subcommands:
  plan        the grid of the plan (s 0.6, 0.8 and 1.0; D 1.0, 1.5 and 2.0; real gains 0 to 0.3 in steps of 0.01);
              writes results/fiorillo_v0_4/power.json. No prediction of any kind is read.
  validation  step 12, on the pod: s and D measured on EI's validation split from the calibrated validation predictions
              of s8_ei and gemma4_31b (1,000 article resamples and 1,000 prompt resamples, numpy's default_rng with the
              seed; articles sorted by PMCID as text); the pass line and the chance at the measured values, for the record
              only (the bar does not change); writes power_validation.json. No test prediction is read.

Usage: python3 src/eval/power_session8.py plan [--out FILE]
       python3 src/eval/power_session8.py validation --records data/processed/session8/evidence_effect_validation.jsonl \\
           --system results/fiorillo_v0_4/predictions/s8_ei_evidence_effect_validation.csv \\
           --reference results/fiorillo_v0_4/predictions/gemma4_31b_evidence_effect_validation.csv [--out FILE]
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "fiorillo_v0_4" / "power.json"
OUT_VALIDATION = ROOT / "results" / "fiorillo_v0_4" / "power_validation.json"
N_TEST = 1218
S_VALUES = (0.6, 0.8, 1.0)
D_VALUES = (1.0, 1.5, 2.0)
GAINS = [round(0.01 * i, 2) for i in range(31)]
SEED = 20260927
RESAMPLES = 1000


def phi(x):
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inverse(p):
    """Standard normal quantile, by bisection on phi (exact to about 1e-12)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


Z = phi_inverse(0.975)
Z80 = phi_inverse(0.80)


def standard_error(s, d, n=N_TEST):
    return s * math.sqrt(d / n)


def chance(gain, se, z=Z):
    """Chance that the estimate falls below the pass line -z se when the real gain is `gain`."""
    return phi(gain / se - z)


def plan():
    out = {"source": "src/eval/power_session8.py plan", "plan": "docs/prompt_session8.md, POWER AND THE BAR and step 1",
           "note": ("Derived: normal approximation. Differences are 's8_ei minus gemma4_31b' in EI test log loss; a gain is the "
                    "negative of a difference. se = s sqrt(D / n), with n the test prompts, s the per-prompt standard deviation "
                    "of the difference and D the design effect of resampling articles. No prediction is read."),
           "n_test": N_TEST, "z_95": Z, "z80": Z80, "cases": []}
    for s in S_VALUES:
        for d in D_VALUES:
            se = standard_error(s, d)
            out["cases"].append({"s": s, "D": d, "se": se, "pass_line": -Z * se, "real_gain_for_80_percent": (Z + Z80) * se,
                                 "chance_by_real_gain": [{"real_gain": g, "chance": chance(g, se)} for g in GAINS]})
    return out


def read_probs(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    k = sum(1 for c in rows[0] if c.startswith("p") and c[1:].isdigit())
    return {r["id"]: (int(r["label"]), [float(r[f"p{j}"]) for j in range(k)]) for r in rows}


def per_item_log_loss(pred, ids):
    return np.array([-math.log(max(pred[i][1][pred[i][0]], 1e-15)) for i in ids])


def design(diff, clusters, resamples=RESAMPLES, seed=SEED):
    """(variance of the mean over article resamples, variance over prompt resamples, D)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    item_means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    articles = sorted(set(clusters))
    index = {a: [i for i, c in enumerate(clusters) if c == a] for a in articles}
    m = len(articles)
    rng = np.random.default_rng(seed)
    cluster_means = []
    for _ in range(resamples):
        draw = rng.integers(0, m, m)
        rows = [i for j in draw for i in index[articles[j]]]
        cluster_means.append(diff[rows].mean())
    v_item, v_cluster = float(np.var(item_means, ddof=1)), float(np.var(cluster_means, ddof=1))
    return v_cluster, v_item, v_cluster / v_item


def validation(records_path, system_path, reference_path):
    records = [json.loads(line) for line in open(records_path, encoding="utf-8") if line.strip()]
    ids = [r["id"] for r in records]
    clusters = [r["group"]["pmcid"] for r in records]
    sys_pred, ref_pred = read_probs(system_path), read_probs(reference_path)
    missing = [i for i in ids if i not in sys_pred or i not in ref_pred]
    if missing:
        raise SystemExit(f"{len(missing)} validation prompts have no prediction, for example {missing[:3]}")
    diff = per_item_log_loss(sys_pred, ids) - per_item_log_loss(ref_pred, ids)
    s = float(np.std(diff, ddof=1))
    v_cluster, v_item, d = design(diff, clusters)
    se = standard_error(s, d)
    return {"source": "src/eval/power_session8.py validation", "plan": "docs/prompt_session8.md, POWER AND THE BAR (step 12)",
            "note": "For the record only; the bar does not change. Measured on EI's validation split; no test prediction is read.",
            "records": str(records_path), "system": str(system_path), "reference": str(reference_path),
            "n_validation": len(ids), "articles_validation": len(set(clusters)),
            "validation_mean_difference": float(diff.mean()), "s": s,
            "variance_article_resamples": v_cluster, "variance_prompt_resamples": v_item, "D": d,
            "n_test": N_TEST, "se_test": se, "pass_line": -Z * se, "real_gain_for_80_percent": (Z + Z80) * se,
            "chance_at_validation_difference": chance(-float(diff.mean()), se)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, default=OUT)
    v = sub.add_parser("validation")
    v.add_argument("--records", type=Path, required=True)
    v.add_argument("--system", type=Path, required=True)
    v.add_argument("--reference", type=Path, required=True)
    v.add_argument("--out", type=Path, default=OUT_VALIDATION)
    args = parser.parse_args(argv)
    out = plan() if args.cmd == "plan" else validation(args.records, args.system, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    if args.cmd == "plan":
        c = next(c for c in out["cases"] if c["s"] == 0.8 and c["D"] == 1.5)
        print(json.dumps({"se_at_s0.8_D1.5": round(c["se"], 4), "pass_line": round(c["pass_line"], 4),
                          "real_gain_for_80_percent": round(c["real_gain_for_80_percent"], 4), "out": str(args.out)}))
    else:
        print(json.dumps({k: out[k] for k in ("s", "D", "se_test", "pass_line", "validation_mean_difference")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

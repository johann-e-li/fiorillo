"""Power of release criterion 4 for Fiorillo v0.3 (docs/prompt_session7.md, POWER AND THE BAR and step 1).

Criterion 4 compares the v0.3 candidate with v0.1 (arm_laya) on each question's test log loss, as the paired
difference "candidate minus v0.1" (negative is a gain). It has two routes, and passes if either passes:
- route (a), PubMedQA: the 97.5 percent interval of the PubMedQA difference wholly below zero, and readmission not
  significantly worse (the lower end of its 95 percent interval at or below zero);
- route (b), readmission: the 97.5 percent interval of the readmission difference wholly below zero, and PubMedQA not
  significantly worse (the lower end of its 95 percent interval at or below zero).
Each route runs at half the error rate (a Bonferroni split of the 2.5 percent that v0.2's single 95 percent interval
allowed on its side).

Method (derived; normal approximation): each question's estimated difference D is taken as normal with mean the real
difference d and standard deviation se, where se is session 5's paired bootstrap standard error of the v0.2 candidate
minus v0.1 (se_bootstrap in results/fiorillo_v0_2/retry_check.json). The two questions' test items are different, so
their estimates are taken as independent. With z1 the 98.75th percentile of the standard normal (2.2414) and z2 the
97.5th (1.9600), and a gain g = -d:
- a route's own condition passes when D < -z1 se, which has chance Phi(g / se - z1);
- "not significantly worse" on the other question holds when D <= z2 se, which has chance Phi(z2 - d / se);
- route (a) and route (b) both pass only when both estimates are below their own -z1 se lines, so
  P(criterion 4) = P(a) + P(b) - Phi(g_p / se_p - z1) Phi(g_r / se_r - z1).
The pass line of a route is -z1 se; the real gain that gives an 80 percent chance on one question alone is
(z1 + z80) se, with z80 the 80th percentile (0.8416). v0.2's criterion 4 (PubMedQA's 95 percent interval wholly below
zero, readmission not significantly worse) is given beside it, and its pass line and 80 percent gain must equal
retry_check.json's own pubmedqa_lines (a check that stops the script if they do not).

No test prediction is read. Output: results/fiorillo_v0_3/power.json.

Usage: python3 src/eval/power_session7.py [--retry-check results/fiorillo_v0_2/retry_check.json] [--out FILE]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RETRY = ROOT / "results" / "fiorillo_v0_2" / "retry_check.json"
OUT = ROOT / "results" / "fiorillo_v0_3" / "power.json"
QUESTIONS = ("pubmedqa_answer", "readmit_window")
GRID = {"pubmedqa_answer": (0.0, 0.08, 0.002), "readmit_window": (0.0, 0.006, 0.0002)}
SE_SCALES = (1.0, 1.5)


def phi(x):
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi_inverse(p):
    """Standard normal quantile, by bisection on phi (exact to about 1e-12, enough for these tables)."""
    lo, hi = -40.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


Z1 = phi_inverse(1 - 0.025 / 2)  # 97.5 percent two-sided interval: the route's own condition
Z2 = phi_inverse(1 - 0.05 / 2)  # 95 percent two-sided interval: "not significantly worse"
Z80 = phi_inverse(0.80)


def own(gain, se, z=Z1):
    """Chance that a question's estimate falls below its pass line -z se when the real gain is `gain`."""
    return phi(gain / se - z)


def not_worse(difference, se, z=Z2):
    """Chance that the lower end of the 95 percent interval is at or below zero when the real difference is `difference`."""
    return phi(z - difference / se)


def criterion4(d_p, d_r, se_p, se_r):
    """Chance of passing route (a), route (b) and criterion 4, for real differences d (candidate minus v0.1)."""
    a = own(-d_p, se_p) * not_worse(d_r, se_r)
    b = own(-d_r, se_r) * not_worse(d_p, se_p)
    both = own(-d_p, se_p) * own(-d_r, se_r)
    return {"route_a": a, "route_b": b, "criterion_4": a + b - both}


def grid(start, stop, step):
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def compute(retry):
    diff = retry["candidate_minus_v0_1"]
    se = {q: float(diff[q]["se_bootstrap"]) for q in QUESTIONS}
    observed = {q: float(diff[q]["estimate"]) for q in QUESTIONS}
    other = {"pubmedqa_answer": "readmit_window", "readmit_window": "pubmedqa_answer"}
    route = {"pubmedqa_answer": "route_a", "readmit_window": "route_b"}

    out = {
        "source": "src/eval/power_session7.py",
        "plan": "docs/prompt_session7.md, POWER AND THE BAR and step 1",
        "note": ("Derived: normal approximation with session 5's paired bootstrap standard errors (se_bootstrap in "
                 "results/fiorillo_v0_2/retry_check.json, the v0.2 candidate minus v0.1 on the same test items). "
                 "Differences are 'candidate minus v0.1' in log loss; a gain is the negative of a difference. "
                 "The two questions' estimates are taken as independent (different test items). "
                 "No test prediction is read."),
        "z": {"z1_route_97_5": Z1, "z2_not_worse_95": Z2, "z80": Z80},
        "se_bootstrap": se,
        "v0_2_observed_difference": observed,
        "false_pass_rate_no_real_gain": criterion4(0.0, 0.0, se["pubmedqa_answer"], se["readmit_window"])["criterion_4"],
        "false_pass_rate_note": ("Chance of passing criterion 4 when neither question has a real difference; at most "
                                 "2.5 percent by the Bonferroni split, and lower because each route also asks the other "
                                 "question to be not significantly worse. It is per attempt: the same test items "
                                 "carried v0.2's criterion 4."),
        "routes": {},
        "v0_2_criterion_4": {},
    }

    for q in QUESTIONS:
        r = route[q]
        entry = {"question": q, "own_condition": "97.5 percent interval of the difference wholly below zero",
                 "other_question": other[q], "other_condition": "lower end of its 95 percent interval at or below zero",
                 "by_se_scale": {}}
        for scale in SE_SCALES:
            s_q = se[q] * scale
            s_o = se[other[q]] * scale
            start, stop, step = GRID[q]
            rows = []
            for g in grid(start, stop, step):
                d_q = -g
                row = {"real_gain": g, "alone": own(g, s_q)}
                for label, d_o in (("other_at_0", 0.0), ("other_at_v0_2_observed", observed[other[q]])):
                    if q == "pubmedqa_answer":
                        c = criterion4(d_q, d_o, s_q, s_o)
                    else:
                        c = criterion4(d_o, d_q, s_o, s_q)
                    row[f"joint_{label}"] = c[r]
                    row[f"criterion_4_{label}"] = c["criterion_4"]
                rows.append(row)
            entry["by_se_scale"][str(scale)] = {
                "se": s_q,
                "pass_line": -Z1 * s_q,
                "real_gain_for_80_percent_alone": (Z1 + Z80) * s_q,
                "joint_at_real_gain_for_80_percent_alone": {
                    "other_at_0": own((Z1 + Z80) * s_q, s_q) * not_worse(0.0, s_o),
                    "other_at_v0_2_observed": own((Z1 + Z80) * s_q, s_q) * not_worse(observed[other[q]], s_o),
                },
                "grid": rows,
            }
        out["routes"][r] = entry

    s_p = se["pubmedqa_answer"]
    v02 = {"rule": "PubMedQA 95 percent interval wholly below zero, and readmission not significantly worse",
           "pass_line": -Z2 * s_p, "real_gain_for_80_percent_alone": (Z2 + Z80) * s_p}
    lines = retry.get("pubmedqa_lines", {})
    for key, mine in (("pass_line_95", v02["pass_line"]), ("true_gain_for_80_percent_95", v02["real_gain_for_80_percent_alone"])):
        if key in lines and abs(float(lines[key]) - mine) > 1e-6:
            raise SystemExit(f"power_session7.py disagrees with retry_check.json's {key}: {mine} against {lines[key]}")
    v02["matches_retry_check"] = all(k in lines for k in ("pass_line_95", "true_gain_for_80_percent_95"))
    v02["joint_at_real_gain_for_80_percent_alone"] = {
        "readmission_at_0": own((Z2 + Z80) * s_p, s_p, Z2) * not_worse(0.0, se["readmit_window"]),
        "readmission_at_v0_2_observed": own((Z2 + Z80) * s_p, s_p, Z2) * not_worse(observed["readmit_window"], se["readmit_window"]),
    }
    out["v0_2_criterion_4"] = v02
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--retry-check", type=Path, default=RETRY)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    retry = json.loads(args.retry_check.read_text(encoding="utf-8"))
    out = compute(retry)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    a = out["routes"]["route_a"]["by_se_scale"]["1.0"]
    b = out["routes"]["route_b"]["by_se_scale"]["1.0"]
    print(json.dumps({
        "route_a_pass_line": round(a["pass_line"], 5), "route_a_gain_80_alone": round(a["real_gain_for_80_percent_alone"], 5),
        "route_a_joint_at_that_gain": {k: round(v, 4) for k, v in a["joint_at_real_gain_for_80_percent_alone"].items()},
        "route_b_pass_line": round(b["pass_line"], 6), "route_b_gain_80_alone": round(b["real_gain_for_80_percent_alone"], 6),
        "route_b_joint_at_that_gain": {k: round(v, 4) for k, v in b["joint_at_real_gain_for_80_percent_alone"].items()},
        "false_pass_rate_no_real_gain": round(out["false_pass_rate_no_real_gain"], 5),
        "v0_2_pass_line": round(out["v0_2_criterion_4"]["pass_line"], 5),
        "v0_2_gain_80_alone": round(out["v0_2_criterion_4"]["real_gain_for_80_percent_alone"], 5),
        "out": str(args.out),
    }, indent=1))


if __name__ == "__main__":
    sys.exit(main())

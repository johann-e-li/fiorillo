"""Power of v0.5's bar, criterion 3 (docs/prompt_session9.md, POWER AND THE BAR and steps 1 and 12).

Criterion 3 compares v0.5's EI specialist with the large model of criterion 3 (gemma4_31b, or gemma4_31b_long under
GEMMA ON THE LONGER INPUT; both calibrated on validation) on EI's test log loss, as the paired difference "specialist
minus large model" (negative is a gain), and passes when the 95 percent interval of the difference, from resampling
articles, is wholly below zero.

Method (derived): session 8's normal approximation (src/eval/power_session8.py), at the per-prompt standard deviation s
and the design effect D that session 8 measured on EI's validation split (results/fiorillo_v0_4/power_validation.json):
se = s sqrt(D / n) on the n = 1,218 test prompts; the criterion passes when the estimate falls below -z se; at a real
gain g its chance is Phi(g / se - z).

Subcommands:
  plan        criterion 3's pass line and chance at real gains from 0 to 0.3 in steps of 0.01, at session 8's measured
              s and D; and its chance if v0.5's EI specialist is worse than s8_ei by 0.025, 0.05, 0.075 and 0.1 in log
              loss, taking s8_ei's test gain over gemma4_31b (results/fiorillo_v0_4/comparison.json, release_bar,
              criterion 3) as the real gain from which those amounts are taken. Writes results/fiorillo_v0_5/power.json.
              It reads session 8's two result files only; no prediction file is read.
  validation  the calibrate stage, on the pod: session 8's validation function (power_session8.validation) on v0.5's EI
              specialist's and the criterion-3 large model's calibrated predictions on all 1,237 EI validation prompts;
              for the record only (the bar does not change); writes results/fiorillo_v0_5/power_validation.json. No test
              prediction is read.

Usage (the --system file is calibrate_session9.py fit's, with all 1,237 validation rows; correction 7):
       python3 src/eval/power_session9.py plan [--out FILE]
       python3 src/eval/power_session9.py validation --records data/processed/session8/evidence_effect_validation.jsonl \\
           --system results/fiorillo_v0_5/predictions/<system>_evidence_effect_validation.csv \\
           --reference results/fiorillo_v0_4/predictions/gemma4_31b_evidence_effect_validation.csv [--out FILE]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.eval import power_session8 as p8  # noqa: E402

OUT = ROOT / "results" / "fiorillo_v0_5" / "power.json"
OUT_VALIDATION = ROOT / "results" / "fiorillo_v0_5" / "power_validation.json"
S8_POWER_VALIDATION = ROOT / "results" / "fiorillo_v0_4" / "power_validation.json"
S8_COMPARISON = ROOT / "results" / "fiorillo_v0_4" / "comparison.json"
WORSE_BY = (0.025, 0.05, 0.075, 0.1)
GAINS = [round(0.01 * i, 2) for i in range(31)]
ENOUGH = 0.8


def s8_inputs(power_validation=S8_POWER_VALIDATION, comparison=S8_COMPARISON):
    """Session 8's measured s and D, and s8_ei's test gain over gemma4_31b (criterion 3's estimate, negated)."""
    pv = json.loads(Path(power_validation).read_text(encoding="utf-8"))
    cmp = json.loads(Path(comparison).read_text(encoding="utf-8"))
    c3 = cmp["release_bar"]["criteria"]["3"]["values"]
    return {"s": float(pv["s"]), "D": float(pv["D"]), "n_test": int(pv["n_test"]),
            "s8_ei_minus_gemma4_31b": {k: float(c3[k]) for k in ("estimate", "low", "high")},
            "s8_gain": -float(c3["estimate"])}


def plan(power_validation=S8_POWER_VALIDATION, comparison=S8_COMPARISON):
    inp = s8_inputs(power_validation, comparison)
    se = p8.standard_error(inp["s"], inp["D"], inp["n_test"])
    worse = []
    for w in WORSE_BY:
        gain = inp["s8_gain"] - w
        worse.append({"worse_than_s8_ei_by": w, "real_gain": gain, "chance": p8.chance(gain, se)})
    chance_at_01 = next(r["chance"] for r in worse if r["worse_than_s8_ei_by"] == 0.1)
    return {"source": "src/eval/power_session9.py plan", "plan": "docs/prompt_session9.md, POWER AND THE BAR and step 1",
            "note": ("Derived: session 8's normal approximation. Differences are 'v0.5's EI specialist minus the large model of "
                     "criterion 3' in EI test log loss; a gain is the negative of a difference. se = s sqrt(D / n), with s and D "
                     "as session 8 measured them on validation for s8_ei and gemma4_31b (they stand in for v0.5's specialist and "
                     "the criterion-3 large model). The real gains of the 'worse than s8_ei' rows are s8_ei's test gain over "
                     "gemma4_31b (session 8's criterion 3 estimate) less each amount. No prediction file is read."),
            "inputs": {"power_validation": str(Path(power_validation).relative_to(ROOT)) if Path(power_validation).is_absolute() and ROOT in Path(power_validation).parents else str(power_validation),
                       "comparison": str(Path(comparison).relative_to(ROOT)) if Path(comparison).is_absolute() and ROOT in Path(comparison).parents else str(comparison),
                       **inp},
            "z_95": p8.Z, "se": se, "pass_line": -p8.Z * se, "real_gain_for_80_percent": (p8.Z + p8.Z80) * se,
            "chance_by_real_gain": [{"real_gain": g, "chance": p8.chance(g, se)} for g in GAINS],
            "if_worse_than_s8_ei": worse,
            "step_1_rule": ("tell Johann before step 8 if criterion 3's chance is below 0.8 when v0.5's EI specialist is worse "
                            "than s8_ei by 0.1 in log loss"),
            "chance_if_worse_by_0.1": chance_at_01, "tell_johann_before_step_8": bool(chance_at_01 < ENOUGH)}


def validation(records_path, system_path, reference_path):
    out = p8.validation(records_path, system_path, reference_path)
    out["source"] = "src/eval/power_session9.py validation (power_session8.validation)"
    out["plan"] = "docs/prompt_session9.md, POWER AND THE BAR (the calibrate stage)"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--power-validation-s8", type=Path, default=S8_POWER_VALIDATION)
    p.add_argument("--comparison-s8", type=Path, default=S8_COMPARISON)
    v = sub.add_parser("validation")
    v.add_argument("--records", type=Path, required=True)
    v.add_argument("--system", type=Path, required=True)
    v.add_argument("--reference", type=Path, required=True)
    v.add_argument("--out", type=Path, default=OUT_VALIDATION)
    args = parser.parse_args(argv)
    if args.cmd == "plan":
        out = plan(args.power_validation_s8, args.comparison_s8)
    else:
        out = validation(args.records, args.system, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    if args.cmd == "plan":
        print(json.dumps({"se": round(out["se"], 4), "pass_line": round(out["pass_line"], 4),
                          "real_gain_for_80_percent": round(out["real_gain_for_80_percent"], 4),
                          "if_worse_than_s8_ei": [(r["worse_than_s8_ei_by"], round(r["chance"], 4)) for r in out["if_worse_than_s8_ei"]],
                          "tell_johann_before_step_8": out["tell_johann_before_step_8"], "out": str(args.out)}))
    else:
        print(json.dumps({k: out[k] for k in ("s", "D", "se_test", "pass_line", "validation_mean_difference")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

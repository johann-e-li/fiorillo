"""Write docs/model_card_v0_2.md, the model card for v0.2 (docs/prompt_session5.md, step 14, with correction 1 (b)).

Every number is read from results/fiorillo_v0_2/ (comparison.json from src/eval/compare.py --session5, release_candidate.json
with the selection record, and the candidate's metrics JSON), results/reader_v1/ (the reader's runs and text manifests),
the v0.2 and v0.1 prediction files and src/configs/mixes_session5.yaml, so the card cannot drift from the results. The
prose is below. {{NAME}} and {{REPO}} are left in the card for scripts/release_fiorillo_v0_2.sh. The card is written only
for a candidate that passed all four release criteria. FAMILY_CHECK_DATE and FAMILY_MODELS are v0.1's paragraph on the
other medical models in the Laya family; the release-day check (step 15) brings them up to date before the card is built
for release.

Usage: python scripts/build_model_card_v0_2.py --usage-check-max-diff X [--results DIR] [--reader-results DIR]
       [--v01-predictions DIR] [--v01-metrics FILE] [--out FILE]
(X is the largest difference found by step 14's usage check on six test items; the card is not written unless X <= 0.001)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_model_card_v0 import LABELS, ci, f, verdict  # noqa: E402

FAMILY_CHECK_DATE = "27 September 2026"
FAMILY_MODELS = ("Duan (wipen/Duan), licensed CC BY-NC 4.0 for non-commercial use, and impacte/laya-medical-escalation, an "
                 "emergency severity index (ESI) triage model licensed Apache 2.0")
ACK = {True: ("This model was trained in part on data from the U.S. National Library of Medicine (NLM): PubMed citations and "
              "abstracts, and full-text articles from PubMed Central (PMC). NLM is not responsible for this model and has not "
              "endorsed it, and the data it was trained on may not reflect the most current data available from NLM."),
       False: ("This model was trained in part on data from the U.S. National Library of Medicine (NLM): PubMed citations and "
               "abstracts. NLM is not responsible for this model and has not endorsed it, and the data it was trained on may not "
               "reflect the most current data available from NLM.")}
SOURCES = {
    "abstracts": ("PubMed titles and abstracts (corpus v1: the PubMed 2026 annual baseline, every PubMedQA article and every "
                  "13-word overlap with PubMedQA's labeled items removed)", "NLM's terms for PubMed; abstracts keep their own copyright"),
    "pmc": ("Full-text PMC articles other than case reports (PMC Article Datasets on AWS; body text only, references and tables "
            "removed)", "CC0 or CC BY, read from each article's own metadata; NLM's terms for PMC"),
    "case_reports": ("Full-text PMC case reports", "CC0 or CC BY, read from each article's own metadata; NLM's terms for PMC"),
    "web": ("Medical web text (TheBlueScrubs-v1 documents rated 5 of 5 on all three of its ratings)",
            "Apache 2.0 for the collection; the texts keep their sources' terms (Common Crawl's terms of use; CC BY-SA for "
            "Wikipedia and StackExchange)"),
    "task": ("The states of the training records of both questions (no validation or test record)",
             "as the records: PubMedQA MIT; UCI data CC BY 4.0"),
}
SOURCE_ORDER = ("abstracts", "pmc", "case_reports", "web", "task")
QNAMES = {"pubmedqa_answer": "PubMedQA, three answers", "pubmedqa_yes_no": "PubMedQA, yes or no items", "readmit_window": "Readmission"}
FINALS = ("s5_final_fresh_head", "s5_final_laya_head")
SHORT = {"abstracts": "abstracts", "pmc": "PMC articles", "case_reports": "case reports", "web": "web text", "task": "task text"}


def for_this_model(metric, d):
    return {"Significant, better": "significantly better for this model", "Significant, worse": "significantly worse for this model",
            "No difference detected": "no difference detected"}.get(verdict(metric, d), "not defined")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def top_counts(path):
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    top = [max(range(3), key=lambda k: float(r[f"p{k}"])) for r in rows]
    labels = [int(r["label"]) for r in rows]
    return sum(t == 2 for t in top), sum(t == 2 and y == 2 for t, y in zip(top, labels)), sum(y == 2 for y in labels)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "fiorillo_v0_2")
    parser.add_argument("--reader-results", type=Path, default=ROOT / "results" / "reader_v1")
    parser.add_argument("--v01-predictions", type=Path, default=ROOT / "results" / "public_v0" / "predictions")
    parser.add_argument("--mixes", type=Path, default=ROOT / "src" / "configs" / "mixes_session5.yaml")
    parser.add_argument("--v01-metrics", type=Path, default=ROOT / "results" / "public_v0" / "arm_laya_metrics.json")
    parser.add_argument("--usage-check-max-diff", type=float, required=True,
                        help="the largest difference between the usage code's probabilities and the evaluation's on six test "
                             "items (step 14), as measured and recorded in docs/log.md; at most 0.001")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "model_card_v0_2.md")
    args = parser.parse_args(argv)
    assert 0 <= args.usage_check_max_diff <= 0.001, "the usage check must reproduce the evaluation within 0.001"
    res, rr = args.results, args.reader_results

    comp, rc = read(res / "comparison.json"), read(res / "release_candidate.json")
    bar = comp["release_bar"]
    assert bar["overall"] == "PASS", "the card is written only for a candidate that passed all four release criteria"
    cand, v01 = comp["candidate"]["system"], comp["v0_1"]
    assert cand == rc["system"] == bar["system"], "comparison.json and release_candidate.json name different candidates"
    sel = rc["selection"]
    run = read(res / "metrics" / f"{cand}_metrics.json")
    M, C = comp["metrics"], comp["comparisons"]
    best = round(run["best_validation_selection_log_loss_exact"], 4)
    kept_steps = [e["step"] for e in run["validation_history"] if e["selection_log_loss"] == best]
    kept_words = (f"step {kept_steps[0]:,}" if len(kept_steps) == 1 else
                  "one of steps " + " and ".join(f"{s:,}" for s in kept_steps) + " (tied at four decimals)")
    v01_run = read(args.v01_metrics)
    temps = run["temperature"]

    reader_rule, task_rule, head_rule, maybe_rule = sel["reader_rule"], sel.get("task_rule", {}), sel["head_rule"], sel["maybe_rule"]
    long_run = bool(reader_rule.get("long_run"))
    task_kept = bool(task_rule.get("task_round_kept"))
    winner = reader_rule["winner"]
    mixes = yaml.safe_load(args.mixes.read_text(encoding="utf-8"))["mixes"]
    shares = mixes[winner]["shares"] if long_run else {}
    uses_pmc = any(s in shares for s in ("pmc", "case_reports"))

    baseline = maybe_rule.get("baseline", "s5_laya")
    names = {cand: "**{{NAME}} v0.2 (this model)**", v01: "{{NAME}} v0.1",
             "s5_laya_w": "Laya's reader, new decision head, maybe weight",
             "s5_final_fresh_head": "Reader v1, new decision head", "s5_final_laya_head": "Reader v1, Laya's decision head",
             "s5_laya": "Laya's reader, new decision head", "s5_bioclinical": "BioClinical ModernBERT reader, new decision head",
             "s5_mix_a": "Mix A reader, new decision head", "s5_mix_b": "Mix B reader, new decision head",
             "s5_mix_c": "Mix C reader, new decision head", "s5_task": "Mix winner plus task round, new decision head",
             "lightgbm": "LightGBM on the same fields", "duan": "Duan, its own wording and input format",
             "duan-raw": "Duan, its own wording and input format", "duan_ourwording": "Duan, this model's wording",
             "prior": "Prior (training label frequencies)"}
    names[cand] = "**{{NAME}} v0.2 (this model)**"
    order = [cand, v01] + [s for s in FINALS if s != cand] + [baseline, "s5_bioclinical", "lightgbm", "duan", "duan_ourwording", "prior"]

    def metric_table(evaluation, metrics):
        rows = M[evaluation]
        lines = ["| System | n | " + " | ".join(LABELS[m] for m in metrics) + " |", "|" + "---|" * (len(metrics) + 2)]
        for system in order:
            if system in rows:
                r = rows[system]
                lines.append(f"| {names[system]} | {r['n']:,} | " + " | ".join(ci(r[m]) for m in metrics) + " |")
        return "\n".join(lines)

    def rows_for(section, evaluation, system, references, metrics):
        found = [c for c in C if c["section"] == section and c["evaluation"] == evaluation and c["system"] == system
                 and c["reference"] in references and c["metric"] in metrics]
        return sorted(found, key=lambda c: metrics.index(c["metric"]))

    def diff_lines(spec, first="This model"):
        lines = [f"| Comparison | Question | Measure | {first} | Reference | Difference (95% interval) | Verdict |",
                 "|---|---|---|---|---|---|---|"]
        for label, section, evaluation, system, references, metrics in spec:
            for c in rows_for(section, evaluation, system, references, metrics):
                d = c["difference"]
                lines.append(f"| {label} | {QNAMES[evaluation]} | {LABELS[c['metric']]} | {f(c['system_estimate'])} | "
                             f"{f(c['reference_estimate'])} | {f(d['estimate'])} ({f(d['low'])} to {f(d['high'])}) | "
                             f"{verdict(c['metric'], d)} |")
        return "\n".join(lines)

    duan = ("duan", "duan-raw")
    main_diffs = diff_lines([
        ("v0.1", "D1", "pubmedqa_answer", cand, (v01,), ["log_loss", "ece", "accuracy", "macro_f1"]),
        ("v0.1", "D1", "pubmedqa_yes_no", cand, (v01,), ["log_loss", "ece", "macro_f1"]),
        ("v0.1", "D1", "readmit_window", cand, (v01,), ["log_loss", "ece", "auroc_30d", "macro_f1_tuned"]),
        ("Prior", "D5", "pubmedqa_answer", cand, ("prior",), ["log_loss"]),
        ("Prior", "D5", "readmit_window", cand, ("prior",), ["log_loss"]),
        ("Duan, its own format", "D2", "pubmedqa_yes_no", cand, duan, ["macro_f1", "log_loss", "ece"]),
        ("Duan, its own format", "D2", "readmit_window", cand, duan,
         ["macro_f1_tuned", "macro_f1", "quadratic_weighted_kappa", "log_loss", "ece", "auroc_30d"]),
        ("LightGBM", "D4", "readmit_window", cand, ("lightgbm",), ["auroc_30d", "log_loss", "ece"]),
    ])
    sens = diff_lines([("Duan, this model's wording", "D2", ev, cand, ("duan_ourwording", "duan_ourwording-raw"), metrics)
                       for ev, metrics in (("pubmedqa_yes_no", ["macro_f1", "log_loss", "ece"]),
                                           ("readmit_window", ["macro_f1_tuned", "macro_f1", "quadratic_weighted_kappa",
                                                               "log_loss", "ece", "auroc_30d"]))])
    d3_spec = []
    for system, reference, label in (("s5_mix_a", baseline, "Mix A reader minus Laya's reader"),
                                     ("s5_mix_b", baseline, "Mix B reader minus Laya's reader"),
                                     ("s5_mix_c", baseline, "Mix C reader minus Laya's reader"),
                                     ("s5_bioclinical", baseline, "BioClinical reader minus Laya's reader"),
                                     ("s5_final_fresh_head", "s5_bioclinical", "Reader v1 minus BioClinical's reader")):
        for ev, metrics in (("pubmedqa_answer", ["accuracy", "log_loss", "ece"]), ("readmit_window", ["auroc_30d", "log_loss", "ece"])):
            d3_spec.append((label, "D3", ev, system, (reference,), metrics))
    d3 = diff_lines(d3_spec, first="First")

    crit = bar["criteria"]
    c1, c2, c3, c4 = (crit[k]["values"] for k in ("1", "2", "3", "4"))
    duan_ref = lambda v: next(v[k] for k in v if k not in ("metric", cand, "ok"))
    bar_table = "\n".join([
        "| Criterion (fixed before training) | Result | Values |",
        "|---|---|---|",
        f"| 1. Expected calibration error at most the limit on each question (the larger of 0.05 and the 95th percentile of a "
        f"calibrated model's error, fixed before any test prediction) | {crit['1']['result']} | PubMedQA {f(c1['pubmedqa_answer']['ece'])} "
        f"against a limit of {f(c1['pubmedqa_answer']['limit'])}; readmission {f(c1['readmit_window']['ece'])} against "
        f"{f(c1['readmit_window']['limit'])} |",
        f"| 2. Log loss below the prior's on both questions, 95% interval of the difference excluding zero | {crit['2']['result']} | "
        f"PubMedQA {f(c2['pubmedqa_answer'][cand])} against {f(c2['pubmedqa_answer']['prior'])}, difference "
        f"{ci(c2['pubmedqa_answer']['difference'])}; readmission {f(c2['readmit_window'][cand])} against "
        f"{f(c2['readmit_window']['prior'])}, difference {ci(c2['readmit_window']['difference'])} |",
        f"| 3. Macro-F1 not below Duan's point estimate (PubMedQA yes or no items, top answer; readmission, decision weights) | "
        f"{crit['3']['result']} | PubMedQA {f(c3['pubmedqa_yes_no'][cand])} against {f(duan_ref(c3['pubmedqa_yes_no']))}; "
        f"readmission {f(c3['readmit_window'][cand])} against {f(duan_ref(c3['readmit_window']))} |",
        f"| 4. Better than v0.1: PubMedQA log loss lower, the 95% interval of the difference wholly below zero; readmission log "
        f"loss not significantly worse | {crit['4']['result']} | PubMedQA {f(c4['pubmedqa_answer'][cand])} against "
        f"{f(c4['pubmedqa_answer'][v01])}, difference {ci(c4['pubmedqa_answer']['difference'])}; readmission "
        f"{f(c4['readmit_window'][cand])} against {f(c4['readmit_window'][v01])}, difference "
        f"{ci(c4['readmit_window']['difference'])} |",
    ])

    sub = {}
    for s in comp["subgroups"]:
        sub.setdefault((s["field"], s["group"]), {})[s["system"]] = s
    sub_lines = ["| Group | n | Observed within 30 days | v0.2 expected | v0.1 expected | v0.2 observed / expected | "
                 "v0.1 observed / expected | v0.2 ECE | v0.1 ECE |", "|---|---|---|---|---|---|---|---|---|"]
    for (field, group), by in sub.items():
        if cand in by and v01 in by:
            a, b = by[cand], by[v01]
            sub_lines.append(f"| {({'sex': 'Sex', 'age_band': 'Age', 'race': 'Race'})[field]}: "
                             f"{({'AfricanAmerican': 'African American'}).get(group, group)} | {a['n']:,} | {ci(a['observed_30d'])} | "
                             f"{ci(a['expected_30d'])} | {ci(b['expected_30d'])} | {ci(a['observed_expected_30d'], 3)} | "
                             f"{ci(b['observed_expected_30d'], 3)} | {ci(a['ece'])} | {ci(b['ece'])} |")

    heldout = []
    runs = [(f"Mix {m}", rr / f"mlm_mix_{m.lower()}" / "heldout_losses.json") for m in ("A", "B", "C")]
    runs.append(("Reader v1 (the long run)", rr / "mlm_long" / "heldout_losses.json"))
    if task_kept:
        runs.append(("Reader v1 (after the task round)", rr / "mlm_long_task" / "heldout_losses.json"))
    for label, path in runs:
        if path.exists():
            last = read(path)[-1]
            heldout.append(f"| {label} | {last['step']:,} | {last['tokens']:,} | "
                           + " | ".join(f(last["loss"].get(s)) for s in SOURCE_ORDER) + " |")
    heldout_table = "\n".join(["| Run | Steps | Tokens | Abstracts | PMC articles | Case reports | Web text | Task text |",
                               "|---|---|---|---|---|---|---|---|"] + heldout)

    mix_lines = ["| Mix | Text (token shares) | Decision-layer selection score on validation |", "|---|---|---|"]
    for m in ("A", "B", "C"):
        share = ", ".join(f"{SHORT[k]} {v:g}%" for k, v in mixes[m]["shares"].items())
        score = reader_rule["scores"].get(m)
        mix_lines.append(f"| {m} | {share} | {f(score) if score is not None else 'not built'} |")
    mix_lines.append(f"| None (Laya's own reader, the baseline) | none | {f(reader_rule['baseline_score'])} |")

    if long_run:
        lrun = read(rr / "mlm_long" / "run.json")
        lthr = read(rr / "mlm_long" / "throughput.json")
        tokens = lrun["steps"] * lrun["tokens_per_step"]
        drawn = lthr["blocks_drawn"]
        text_rows = []
        for s in SOURCE_ORDER:
            if s in shares:
                man = rr / "text_manifests" / f"{s}_manifest.json"
                pool = read(man).get("tokens") if man.exists() else None
                text_rows.append(f"| {SOURCES[s][0]} | {SOURCES[s][1]} | {shares[s]:g}% | "
                                 f"{f'{pool:,}' if pool else 'not recorded'} | {drawn.get(s, 0) * 1024:,} |")
        if task_kept:
            tman = rr / "text_manifests" / "task_manifest.json"
            tpool = read(tman).get("tokens") if tman.exists() else None
            ttok = read(rr / "mlm_long_task" / "throughput.json")["tokens_run"]
            text_rows.append(f"| {SOURCES['task'][0]} | {SOURCES['task'][1]} | three passes after the long run | "
                             f"{f'{tpool:,}' if tpool else 'not recorded'} | {ttok:,} |")
        text_table = "\n".join(["| Text | Licence and terms | Share | Tokens available | Tokens read by reader v1 |",
                                "|---|---|---|---|---|"] + text_rows)
        schedule = (f"linear warmup over the first {lrun['warmup']:,} steps, the peak rate until step {lrun['decay_start']:,}, "
                    f"then linear decay to zero over the last {lrun['steps'] - lrun['decay_start']:,} of its {lrun['steps']:,} steps")
        reader_summary = (f"Reader v1 is Laya's reader trained further by masked-language modeling on {tokens:,} tokens of mix "
                          f"{winner} ({lrun['steps']:,} steps of {lrun['tokens_per_step']:,} tokens)"
                          + (", then on the task text" if task_kept else "") + ".")
        mlm_tps = f"{lthr['tokens_per_second']:,.0f}"
    else:
        text_table, reader_summary, mlm_tps = "No mix beat Laya's own reader on validation, so v0.2 keeps Laya's reader.", \
            "No long run was made: v0.2 keeps Laya's reader.", "not applicable"
        schedule = "not applicable"

    v02_maybe, v02_hits, n_maybe = top_counts(res / "predictions" / f"{cand}_pubmedqa_answer_test.csv")
    v01_maybe, v01_hits, _ = top_counts(args.v01_predictions / f"{v01}_pubmedqa_answer_test.csv")
    d4 = {c["metric"]: c for c in C if (c["section"], c["system"], c["reference"]) == ("D4", cand, "lightgbm")}
    rw = M["readmit_window"][cand]
    head_name = "the new head" if head_rule["winner"] == "fresh" else "Laya's head"
    v01_freeze, v01_reinit = int(v01_run.get("freeze_encoder_steps", 0)), bool(v01_run.get("reinit_head", False))
    v01_head = ("where v0.1 trained Laya's head together with the reader from the first step" if not v01_reinit and v01_freeze == 0
                else f"where v0.1 used {'a new head' if v01_reinit else 'Laya' + chr(39) + 's head'} with {v01_freeze} head-only steps")
    w = run.get("maybe_weight", {}).get("w")
    maybe_note = (f"; PubMedQA maybe items carried a loss weight of {w:.2f}" if maybe_rule["weighted_kept"] and w is not None else "")
    built = [m for m in ("A", "B", "C") if reader_rule["scores"].get(m) is not None]
    mixes_run = {3: "Three mixes were", 2: "Two mixes were", 1: "One mix was"}[len(built)]
    head_words = {"fresh": "a new decision head, trained alone for its first 250 steps and then with the reader",
                  "laya": "Laya's own decision head, trained alone for its first 250 steps and then with the reader"}[head_rule["winner"]]
    values = {
        "READER_SUMMARY": reader_summary, "HEAD_WORDS": head_words, "WINNER": winner, "V01_HEAD": v01_head,
        "MAYBE_NOTE": maybe_note, "MIXES_RUN": mixes_run, "SCHEDULE": schedule, "KEPT_WORDS": kept_words,
        "USAGE_DIFF": f"{args.usage_check_max_diff:.4f}",
        "FAMILY": f"Other medical models in the Laya family (Hugging Face check, {FAMILY_CHECK_DATE}): {FAMILY_MODELS}.",
        "ACK": ACK[uses_pmc], "TEXT_TABLE": text_table, "MIX_TABLE": "\n".join(mix_lines), "HELDOUT_TABLE": heldout_table,
        "MLM_TPS": mlm_tps,
        "MAYBE_RULE": (f"the extra loss weight on PubMedQA maybe items was {'kept' if maybe_rule['weighted_kept'] else 'not kept'} "
                       f"(out-of-fold log loss on the maybe items {f(maybe_rule['runs']['s5_laya_w']['out_of_fold']['pubmedqa_maybe'])} "
                       f"with it, {f(maybe_rule['runs']['s5_laya']['out_of_fold']['pubmedqa_maybe'])} without)"),
        "READER_RULE": "; ".join(reader_rule["steps"]) + f"; the winner, mix {winner}, scored {f(reader_rule['winner_score'])} "
                       f"against Laya's reader's {f(reader_rule['baseline_score'])}",
        "TASK_RULE": (f"the task round was {'kept' if task_kept else 'not kept'} (selection score {f(task_rule['s5_task_score'])} "
                      f"against the winner's {f(task_rule['winner_score'])})") if task_rule else "there was no task round",
        "HEAD_RULE": (f"{head_name} won "
                      f"(new head {f(head_rule['fresh_head_score'])}, Laya's head {f(head_rule['laya_head_score'])})"),
        "PUBMED_TABLE": metric_table("pubmedqa_answer", ["log_loss", "ece", "macro_f1", "accuracy"]),
        "YESNO_TABLE": metric_table("pubmedqa_yes_no", ["log_loss", "ece", "macro_f1", "accuracy"]),
        "READMIT_TABLE": metric_table("readmit_window", ["log_loss", "ece", "accuracy", "macro_f1", "macro_f1_tuned", "auroc_30d"]),
        "READMIT_CAL_TABLE": metric_table("readmit_window", ["calibration_intercept_30d", "calibration_slope_30d",
                                                             "quadratic_weighted_kappa"]),
        "MAIN_DIFFS": main_diffs, "SENS_TABLE": sens, "D3_TABLE": d3, "BAR_TABLE": bar_table,
        "SUBGROUP_TABLE": "\n".join(sub_lines),
        "T_PUB": f"{temps['pubmedqa_answer']:.4f}", "T_READ": f"{temps['readmit_window']:.4f}",
        "STEPS": f"{run['optimizer_steps']:,}", "TRAIN_S": f"{run['train_seconds']:,.0f}",
        "RAW_PUB_LL": f(M["pubmedqa_answer"][cand + "-raw"]["log_loss"]["estimate"]),
        "CAL_PUB_LL": f(M["pubmedqa_answer"][cand]["log_loss"]["estimate"]),
        "RAW_PUB_ECE": f(M["pubmedqa_answer"][cand + "-raw"]["ece"]["estimate"]),
        "CAL_PUB_ECE": f(M["pubmedqa_answer"][cand]["ece"]["estimate"]),
        "RAW_RW_LL": f(M["readmit_window"][cand + "-raw"]["log_loss"]["estimate"]), "CAL_RW_LL": f(rw["log_loss"]["estimate"]),
        "RAW_RW_ECE": f(M["readmit_window"][cand + "-raw"]["ece"]["estimate"]), "CAL_RW_ECE": f(rw["ece"]["estimate"]),
        "RW_INTERCEPT": ci(rw["calibration_intercept_30d"]), "RW_SLOPE": ci(rw["calibration_slope_30d"]),
        "LIM_PUB": f(rc["ece_limits"]["pubmedqa_answer"]["limit"]), "P95_PUB": f(rc["ece_limits"]["pubmedqa_answer"]["p95_if_calibrated"]),
        "LIM_RW": f(rc["ece_limits"]["readmit_window"]["limit"]), "P95_RW": f(rc["ece_limits"]["readmit_window"]["p95_if_calibrated"]),
        "RC_TIME": rc["time_utc"],
        "V02_MAYBE_TOP": str(v02_maybe), "V01_MAYBE_TOP": str(v01_maybe), "V02_MAYBE_HITS": str(v02_hits),
        "V01_MAYBE_HITS": str(v01_hits), "N_MAYBE": str(n_maybe),
        "LGBM_LL": f(d4["log_loss"]["reference_estimate"]), "OUR_RW_LL": f(d4["log_loss"]["system_estimate"]),
        "D4_LL": ci(d4["log_loss"]["difference"]), "D4_LL_VERDICT": for_this_model("log_loss", d4["log_loss"]["difference"]),
        "D4_AUC": ci(d4["auroc_30d"]["difference"]), "D4_AUC_VERDICT": for_this_model("auroc_30d", d4["auroc_30d"]["difference"]),
        "CONTAMINATION": comp["contamination"].replace("Known contamination: ", "")
                         .replace("445 PQA-L items", "445 expert-labeled PubMedQA items")
                         .replace("may overlap our test split", "may overlap the test split used here"),
    }
    card = TEMPLATE
    for k, v in values.items():
        card = card.replace("<<" + k + ">>", v)
    assert "<<" not in card, "a placeholder was not filled"
    assert chr(0x2014) not in card and chr(0x2013) not in card, "dashes"
    lower = card.lower()
    assert not any(p in lower for p in ("first open", "the first medical", "first typed", "first commercially")), \
        "no priority claim in the card"
    args.out.write_text(card, encoding="utf-8")
    print(f"wrote {args.out} ({len(card.split()):,} words)")


TEMPLATE = """# Model card for v0.2
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 28 September 2026
Sources: docs/prompt_session5.md (step 14, with corrections 1 to 5); results/fiorillo_v0_2/comparison.json; results/fiorillo_v0_2/release_candidate.json; the candidate's metrics JSON; results/reader_v1/; docs/data_sources_session5.md; docs/model_card_v0.md; written by scripts/build_model_card_v0_2.py
Status: complete (the name and repository are filled in by scripts/release_fiorillo_v0_2.sh)

Everything below the next line becomes the Hugging Face README.md.

<!-- model card begins -->
---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: laya
language:
- en
tags:
- laya
- medical
- typed-decisions
- calibrated-probabilities
- pubmedqa
- hospital-readmission
datasets:
- qiaojin/PubMedQA
---

# {{NAME}} v0.2

{{NAME}} v0.2 is a medical typed-decision model. It answers two questions with calibrated probabilities in one forward pass: the answer to a biomedical research question from its abstract (PubMedQA: yes, no or maybe), and how soon a patient with diabetes is readmitted after a hospital discharge (UCI Diabetes 130-US hospitals, 1999 to 2008: none, after 30 days, or within 30 days). It is for research. It is not a medical device and must not be used to make clinical decisions.

What is new in v0.2: the reader. <<READER_SUMMARY>> I then trained the decision layer on reader v1 with v0.1's method and settings except for the head: <<HEAD_WORDS>>, <<V01_HEAD>><<MAYBE_NOTE>>. v0.2 is released only because it passed four criteria fixed before training, one of which is beating v0.1 on PubMedQA.

<<FAMILY>>

<<ACK>>

Named after my violin, which carries the name of Federigo Fiorillo, whose 36 Études, or studies, are where violinists learn to play in tune.

Abbreviations: ECE is expected calibration error (top answer, 15 equal-width bins); AUROC is the area under the receiver operating characteristic curve; UCI is the University of California, Irvine, Machine Learning Repository; NLM is the U.S. National Library of Medicine; PMC is PubMed Central; MLM is masked-language modeling; MIMIC is the Medical Information Mart for Intensive Care; NRD is the Nationwide Readmissions Database; NIS is the National Inpatient Sample; NHAMCS is the National Hospital Ambulatory Medical Care Survey.

## The two questions

The model was trained and evaluated on these definitions, word for word. Other questions are outside what was tested.

| Question id | Type | Instructions | Options |
|---|---|---|---|
| pubmedqa_answer | choice | Based only on this abstract, what is the answer to the research question? | yes: the reported results support yes; no: the reported results support no; maybe: the reported results are inconclusive |
| readmit_window | score | How soon after this hospital discharge is the patient readmitted? | 0: no readmission after this discharge; 1: readmitted more than 30 days after this discharge; 2: readmitted within 30 days of this discharge |

## How to use

Install laya 0.3.20 (`pip install laya==0.3.20`). The weights ship with every temperature set to 1, so laya returns uncalibrated probabilities until the temperatures in calibration.json are applied. The code below applies them. In laya 0.3.20 the temperature is looked up by question type and number of options, so these two settings calibrate any choice question with 3 to 5 options and any score question with 3 to 5 levels; only the two questions above were calibrated and tested.

```python
import json
from huggingface_hub import hf_hub_download
from laya import Agent

repo = "{{REPO}}"
agent = Agent(repo)
temperature = json.load(open(hf_hub_download(repo, "calibration.json")))["temperature"]
agent.temperature_by_options = {"choice:3-5": temperature["pubmedqa_answer"],
                                "score:3-5": temperature["readmit_window"]}

questions = {
    "pubmedqa_answer": {
        "type": "choice",
        "instructions": "Based only on this abstract, what is the answer to the research question?",
        "criteria": {"yes": "the reported results support yes",
                     "no": "the reported results support no",
                     "maybe": "the reported results are inconclusive"}},
    "readmit_window": {
        "type": "score",
        "instructions": "How soon after this hospital discharge is the patient readmitted?",
        "criteria": ["no readmission after this discharge",
                     "readmitted more than 30 days after this discharge",
                     "readmitted within 30 days of this discharge"]},
}

# A PubMedQA-style state: the research question and the abstract's sections, without its conclusion.
abstract = {"question": "Does early mobilization shorten hospital stay after hip fracture surgery?",
            "contexts": [{"section": "BACKGROUND", "text": "..."},
                         {"section": "METHODS", "text": "..."},
                         {"section": "RESULTS", "text": "..."}]}
print(agent.system_one(abstract, {"pubmedqa_answer": questions["pubmedqa_answer"]})["answers"])

# A UCI-style encounter (an invented patient), in the field names and values the model was trained on.
encounter = {"age band": "[70-80)", "sex": "Female", "admission type": "Emergency",
             "admission source": "Emergency Room", "discharge disposition": "Discharged to home",
             "time_in_hospital": 4, "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18,
             "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9,
             "diag_1": "428 (Diseases of the circulatory system)",
             "diag_2": "250 (Endocrine, nutritional and metabolic diseases, and immunity disorders)",
             "diag_3": "401 (Diseases of the circulatory system)",
             "max_glu_serum": "None", "A1Cresult": ">8", "medications": {"insulin": "Up"},
             "change": "Ch", "diabetesMed": "Yes"}
print(agent.system_one(encounter, {"readmit_window": questions["readmit_window"]})["answers"])
```

Each answer carries the probability of every option. Run through this code on a central processing unit (CPU), the released model reproduced its evaluation probabilities to within 0.001 on the six test items I checked (largest difference <<USAGE_DIFF>>).

## Training data

### Text for the reader (new in v0.2)

<<TEXT_TABLE>>

The attribution for every PMC article fetched while the text was built (a superset of any used) is training_manifest.csv in this repository, with each article's PMCID, version and licence.

### Records for the decision layer (as in v0.1)

| Data | License | Used | Splits |
|---|---|---|---|
| PubMedQA (qiaojin/PubMedQA, revision 9001f2853fb87cab8d220904e0de81ac6973b318; official test list at commit 3e7616d288b52edaf38f5fd5c59821c7806bde86) | MIT | The question and the abstract's sections in source order. Left out: the conclusion (long_answer, which states the answer), the label, the original baselines' predictions and MeSH terms | Train 21,050 records: 350 expert-labeled items written 3 times and 20,000 automatically labeled items. Validation: 150 expert-labeled items (checkpoint choice and temperature) and 2,000 automatically labeled items (monitoring only). Test: the 500 official test items |
| Diabetes 130-US Hospitals for Years 1999 to 2008 (UCI dataset 296) | CC BY 4.0 | The first encounter of each patient; encounters ending in death or hospice dropped; identifiers, race (kept only to report calibration by group), payer code and weight left out | Split by patient, 80, 10 and 10 percent: 55,978 train, 6,997 validation, 6,998 test |

Citations. PubMedQA: Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019: 2567 to 2577. UCI data: Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J. Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research International 2014: 781670.

No MIMIC, NRD, NIS or NHAMCS data were used to train this model.

## Training

### Reader v1

I built the starting point from Laya's encoder (convaiinnovations/laya, revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851) and ModernBERT-large's masked-language-model head (answerdotai/ModernBERT-large, revision 45bb4654a4d5aaff24dd11d4781fa46d39bf8c13). Training: 30 percent of tokens masked; AdamW (peak learning rate 1e-4, betas 0.9 and 0.98, weight decay 0.01); 256 blocks of 1,024 tokens a step; <<SCHEDULE>>; bfloat16 mixed precision on one NVIDIA H100 SXM, about <<MLM_TPS>> tokens a second.

The text was chosen by rules fixed before any training run. <<MIXES_RUN>> each run for 500 million tokens, and each reader then had a new decision head trained on it the same way; the score is the mean validation log loss of the two questions (lower is better). A mix replaced the current winner only if its score was more than 0.01 lower, which favors the simpler mix because smaller differences are not reliable.

<<MIX_TABLE>>

The rules as applied, in order: <<MAYBE_RULE>>; <<READER_RULE>>; <<TASK_RULE>>; for the decision layer on reader v1, <<HEAD_RULE>>.

Held-out masked-language-model loss at the end of each run, on text never trained on (lower is better):

<<HELDOUT_TABLE>>

### Decision layer

Trained on reader v1 with v0.1's settings except for the head (<<HEAD_WORDS>>)<<MAYBE_NOTE>>: log loss over the options; AdamW (encoder learning rate 2.5e-5, head 1.0e-4, weight decay 0.01), cosine schedule, gradient clipping at 1.0, float16 mixed precision, gradient checkpointing; one epoch, <<STEPS>> optimizer steps of 32 sequences, each micro-batch from one question; states up to 768 tokens, decision head 192. Validation every 250 steps; I kept the step with the lowest mean of the two questions' validation log losses (<<KEPT_WORDS>>). One temperature per question, fitted on the validation items and shipped in calibration.json: PubMedQA <<T_PUB>>, readmission <<T_READ>>. <<TRAIN_S>> seconds of training on one NVIDIA H100 SXM.

Every choice above (the text, the task round, the head, the temperatures and the calibration limits of criterion 1) was fixed on validation data before any v0.2 test prediction existed; the candidate was fixed at <<RC_TIME>>.

## Evaluation

All results are on the test splits: 500 PubMedQA items (445 of them yes or no) and 6,998 readmission encounters. Intervals are 95% paired percentile bootstrap intervals (1,000 resamples, seed 20260927). Each neural model has one temperature per question fitted on the validation split; the prior and LightGBM are used as they are. In the comparisons with Duan, Duan is scored with and without its temperature and the better of the two is used for each measure. Systems: this model and v0.1; the other decision head on reader v1 and research readers with new heads (not released); LightGBM (gradient-boosted trees) on the same UCI fields; Duan (wipen/Duan, revision 50e0615b576e4829cbad5c193d42edd0cadf2784, licensed for non-commercial use, evaluated only); and the prior, which gives every item the training split's label frequencies.

### PubMedQA, three answers (500 items)

<<PUBMED_TABLE>>

This model gave maybe as its top answer on <<V02_MAYBE_TOP>> of 500 test items (v0.1: <<V01_MAYBE_TOP>>), and named <<V02_MAYBE_HITS>> of the <<N_MAYBE>> maybe items correctly (v0.1: <<V01_MAYBE_HITS>>).

### PubMedQA, yes or no items (445 items; maybe items removed and the other two probabilities rescaled to sum to 1)

<<YESNO_TABLE>>

### Readmission window (6,998 encounters)

Macro-F1 with decision weights: before scoring, each system's decision is tuned on the validation split by weighting the probability of each level (level 0 fixed at 1; levels 1 and 2 on a grid from 1/4 to 16), because readmission within 30 days is only 8.7% of test encounters and a calibrated model seldom makes it the top answer. AUROC is for readmission within 30 days against the rest.

<<READMIT_TABLE>>

<<READMIT_CAL_TABLE>>

### Paired comparisons with this model

Differences are this model minus the reference; a negative difference in log loss or ECE favors this model.

<<MAIN_DIFFS>>

Duan in this model's wording (sensitivity analysis):

<<SENS_TABLE>>

What the reader text contributes (research readers, each with a new decision head, not released):

<<D3_TABLE>>

### Release criteria

These four criteria were fixed before training. The model is released because all four passed on the test split.

<<BAR_TABLE>>

## Calibration

The temperature changed PubMedQA log loss from <<RAW_PUB_LL>> to <<CAL_PUB_LL>> and ECE from <<RAW_PUB_ECE>> to <<CAL_PUB_ECE>>; for readmission, log loss from <<RAW_RW_LL>> to <<CAL_RW_LL>> and ECE from <<RAW_RW_ECE>> to <<CAL_RW_ECE>>. For readmission within 30 days, the calibration intercept is <<RW_INTERCEPT>> and the slope is <<RW_SLOPE>> (1 is ideal).

The ECE limits of criterion 1 were fixed on validation data before any test prediction: a model calibrated by construction, with this model's confidence profile, shows an ECE at or below <<P95_PUB>> on 500 PubMedQA items 95% of the time, so the PubMedQA limit is <<LIM_PUB>>; for readmission the 95th percentile is <<P95_RW>> and the limit <<LIM_RW>>. Resampling inflates ECE, so an ECE interval can lie above its own estimate.

Calibration for readmission within 30 days by group, for groups with at least 100 test encounters, with v0.1 beside this model:

<<SUBGROUP_TABLE>>

## Known contamination

<<CONTAMINATION>>

## Limitations

- Research use only. It is not a medical device, has not been validated prospectively or at any other site, and must not guide care.
- Two questions only. Other questions, wordings or input formats are untested and uncalibrated.
- The readmission data are from 130 US hospitals between 1999 and 2008, first encounters only, and practice has changed since. On these data, this model's log loss was <<OUR_RW_LL>> against LightGBM's <<LGBM_LL>> (this model minus LightGBM, <<D4_LL>>: <<D4_LL_VERDICT>>); for AUROC within 30 days the difference was <<D4_AUC>> (<<D4_AUC_VERDICT>>).
- The PubMedQA temperature was fitted on 150 items.
- The PubMedQA and readmission test splits were also used to evaluate v0.1; v0.2's own choices were all made on validation data.
- English only.

## Lineage

ModernBERT-large (answerdotai/ModernBERT-large, Apache 2.0) was fine-tuned into Laya (convaiinnovations/laya, Apache 2.0); Laya's reader was trained further into reader v1 (mix <<WINNER>>); reader v1 with a decision layer is this model. The previous version is {{NAME}} v0.1 (johanneli/fiorillo-v0.1).

## License and citation

Apache 2.0. The training data keep their own licenses and terms (listed above); the NOTICE file carries the attributions and NLM's acknowledgement.

Li JE. {{NAME}} v0.2: a calibrated medical typed-decision model built on Laya. Hugging Face, 2026. https://huggingface.co/{{REPO}}
"""

if __name__ == "__main__":
    main()

"""Write docs/model_card_v0.md, the model card for public v0 (docs/prompt_session4.md, step 10).

Every number in the evaluation, calibration and comparison tables is read from results/public_v0/comparison.json
(src/eval/compare.py) and results/public_v0/arm_laya_metrics.json (run 3's training script), so the card cannot drift
from the results. The prose is below. {{NAME}} and {{REPO}} are left in the card: scripts/release_public_v0.sh fills
them from docs/naming.md and the Hugging Face account when it assembles README.md, and stops if either is left.

Usage: python scripts/build_model_card_v0.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "public_v0"
OUT = ROOT / "docs" / "model_card_v0.md"

NAMES = {
    "prior": "Prior (training label frequencies)",
    "lightgbm": "LightGBM on the same fields",
    "laya_zeroshot": "Laya, not fine-tuned",
    "arm_laya": "**{{NAME}} v0.1 (this model)**",
    "arm_laya_run2": "Earlier run of this model (run 2)",
    "arm_laya_fresh": "Laya reader, new decision head",
    "arm_bioclinical": "BioClinical ModernBERT reader, new decision head",
    "duan": "Duan, its own wording and input format",
    "duan_ourwording": "Duan, this model's wording",
}
ORDER = ["arm_laya", "arm_laya_run2", "arm_laya_fresh", "arm_bioclinical", "laya_zeroshot", "lightgbm",
         "duan", "duan_ourwording", "prior"]
LOWER_IS_BETTER = {"log_loss", "brier", "ece"}
LABELS = {"log_loss": "Log loss", "ece": "ECE", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
          "macro_f1_tuned": "Macro-F1, decision weights", "auroc_30d": "AUROC, within 30 days",
          "quadratic_weighted_kappa": "Quadratic weighted kappa", "brier": "Brier score",
          "calibration_intercept_30d": "Calibration intercept, within 30 days",
          "calibration_slope_30d": "Calibration slope, within 30 days"}


def f(x, digits=4):
    return "not defined" if x is None else f"{x:.{digits}f}"


def ci(m, digits=4):
    if m is None or m.get("estimate") is None:
        return "not defined"
    return f"{f(m['estimate'], digits)} ({f(m['low'], digits)} to {f(m['high'], digits)})"


def metric_table(metrics, evaluation, names):
    rows = metrics[evaluation]
    head = "| System | n | " + " | ".join(LABELS[m] for m in names) + " |"
    lines = [head, "|" + "---|" * (len(names) + 2)]
    for system in ORDER:
        if system in rows:
            r = rows[system]
            lines.append(f"| {NAMES[system]} | {r['n']:,} | " + " | ".join(ci(r[m]) for m in names) + " |")
    return "\n".join(lines)


def verdict(metric, d):
    if d["low"] is None or d["high"] is None:
        return "not defined"
    if d["low"] <= 0 <= d["high"]:
        return "No difference detected"
    better = (d["estimate"] < 0) == (metric in LOWER_IS_BETTER)
    return "Significant, better" if better else "Significant, worse"


def comparison_rows(comparisons, section, evaluation, system, reference, metrics):
    out = []
    for c in comparisons:
        if (c["section"], c["evaluation"], c["system"], c["reference"]) == (section, evaluation, system, reference) \
                and c["metric"] in metrics:
            d = c["difference"]
            out.append((c["metric"], d, c["system_estimate"], c["reference_estimate"]))
    out.sort(key=lambda t: metrics.index(t[0]))
    return out


def main():
    comp = json.loads((RESULTS / "comparison.json").read_text(encoding="utf-8"))
    run = json.loads((RESULTS / "arm_laya_metrics.json").read_text(encoding="utf-8"))
    M, C = comp["metrics"], comp["comparisons"]
    temps = run["temperature"]
    kept = min(run["validation_history"], key=lambda e: e["selection_log_loss"])
    raw_q, cal_q = run["test_per_question_before_temperature"], run["test_per_question_after_temperature"]
    bar = comp["release_bar"]
    assert bar["overall"] == "PASS", "the card is written only for a model that passed the release bar"

    def diff_table(spec):
        lines = ["| Comparison | Question | Measure | This model | Reference | Difference (95% interval) | Verdict |",
                 "|---|---|---|---|---|---|---|"]
        for label, section, evaluation, system, reference, metrics in spec:
            for metric, d, a, b in comparison_rows(C, section, evaluation, system, reference, metrics):
                lines.append(f"| {label} | {QNAMES[evaluation]} | {LABELS[metric]} | {f(a)} | {f(b)} | "
                             f"{f(d['estimate'])} ({f(d['low'])} to {f(d['high'])}) | {verdict(metric, d)} |")
        return "\n".join(lines)

    QNAMES = {"pubmedqa_answer": "PubMedQA, three answers", "pubmedqa_yes_no": "PubMedQA, yes or no items",
              "readmit_window": "Readmission"}
    main_diffs = diff_table([
        ("Prior", "C5", "pubmedqa_answer", "arm_laya", "prior", ["log_loss"]),
        ("Prior", "C5", "readmit_window", "arm_laya", "prior", ["log_loss"]),
        ("Duan, its own format", "C1", "pubmedqa_yes_no", "arm_laya", "duan", ["macro_f1", "log_loss", "ece"]),
        ("Duan, its own format", "C2", "readmit_window", "arm_laya", "duan",
         ["macro_f1_tuned", "macro_f1", "quadratic_weighted_kappa", "log_loss", "ece", "auroc_30d"]),
        ("LightGBM", "C4", "readmit_window", "arm_laya", "lightgbm", ["auroc_30d", "log_loss", "ece"]),
    ])
    sens = [c for c in C if c["section"] in ("C1", "C2") and c["reference"].startswith("duan_ourwording")]
    sens_lines = ["| Question | Measure | This model | Duan, this model's wording | Difference (95% interval) | Verdict |",
                  "|---|---|---|---|---|---|"]
    for c in sens:
        d = c["difference"]
        sens_lines.append(f"| {QNAMES[c['evaluation']]} | {LABELS[c['metric']]} | {f(c['system_estimate'])} | "
                          f"{f(c['reference_estimate'])} | {f(d['estimate'])} ({f(d['low'])} to {f(d['high'])}) | "
                          f"{verdict(c['metric'], d)} |")
    reader = []
    for label, section, system, reference in (("BioClinical reader minus Laya reader, both with a new head", "C3",
                                               "arm_bioclinical", "arm_laya_fresh"),
                                              ("Laya's own head minus a new head, both on Laya's reader (run 2)", "C3b",
                                               "arm_laya_run2", "arm_laya_fresh")):
        for evaluation, metrics in (("pubmedqa_answer", ["accuracy", "log_loss", "ece"]),
                                    ("readmit_window", ["auroc_30d", "log_loss", "ece"])):
            for metric, d, a, b in comparison_rows(C, section, evaluation, system, reference, metrics):
                reader.append(f"| {label} | {QNAMES[evaluation]} | {LABELS[metric]} | {f(a)} | {f(b)} | "
                              f"{f(d['estimate'])} ({f(d['low'])} to {f(d['high'])}) | {verdict(metric, d)} |")
    reader_table = "\n".join(["| Comparison | Question | Measure | First | Second | Difference (95% interval) | Verdict |",
                              "|---|---|---|---|---|---|---|"] + reader)

    crit = bar["criteria"]
    c2 = crit["2"]["values"]
    c3 = crit["3"]["values"]
    bar_table = "\n".join([
        "| Criterion (fixed before training) | Result | Values |",
        "|---|---|---|",
        f"| 1. Expected calibration error at most 0.05 on both questions | {crit['1']['result']} | PubMedQA "
        f"{f(crit['1']['values']['pubmedqa_answer'])}; readmission {f(crit['1']['values']['readmit_window'])} |",
        f"| 2. Log loss below the prior's on both questions, 95% interval of the difference excluding zero | "
        f"{crit['2']['result']} | PubMedQA {f(c2['pubmedqa_answer']['arm_laya'])} against {f(c2['pubmedqa_answer']['prior'])}, "
        f"difference {ci(c2['pubmedqa_answer']['difference'])}; readmission {f(c2['readmit_window']['arm_laya'])} against "
        f"{f(c2['readmit_window']['prior'])}, difference {ci(c2['readmit_window']['difference'])} |",
        f"| 3. Macro-F1 not below Duan's point estimate (PubMedQA yes or no items, top answer; readmission, decision "
        f"weights) | {crit['3']['result']} | PubMedQA {f(c3['C1']['arm_laya'])} against {f(c3['C1']['duan'])}; readmission "
        f"{f(c3['C2']['arm_laya'])} against {f(c3['C2']['duan'])} |",
    ])

    sub_lines = ["| Group | n | Observed within 30 days | Expected within 30 days | Observed / expected | ECE |",
                 "|---|---|---|---|---|---|"]
    for s in comp["subgroups"]:
        if s["system"] == "arm_laya":
            field = {"sex": "Sex", "age_band": "Age", "race": "Race"}[s["field"]]
            group = {"AfricanAmerican": "African American"}.get(s["group"], s["group"])
            sub_lines.append(f"| {field}: {group} | {s['n']:,} | {ci(s['observed_30d'])} | {ci(s['expected_30d'])} | "
                             f"{ci(s['observed_expected_30d'], 3)} | {ci(s['ece'])} |")

    maybe_top, maybe_hits, n_maybe = {}, {}, 0
    for system in ORDER:
        path = RESULTS / "predictions" / f"{system}_pubmedqa_answer_test.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        top = [max(range(3), key=lambda k: float(r[f"p{k}"])) for r in rows]
        labels = [int(r["label"]) for r in rows]
        maybe_top[system] = sum(t == 2 for t in top)
        maybe_hits[system] = sum(t == 2 and y == 2 for t, y in zip(top, labels))
        n_maybe = sum(y == 2 for y in labels)
    t2 = json.loads((RESULTS / "run2" / "calibration_family_cv_arm_laya_pubmedqa_answer.json").read_text())["temperature_bootstrap"]["range_95"]
    t3 = json.loads((RESULTS / "run3" / "calibration_family_cv_arm_laya_pubmedqa_answer.json").read_text())["temperature_bootstrap"]["range_95"]
    floor = json.loads((RESULTS / "run3" / "ece_noise_floor_arm_laya_pubmedqa_answer.json").read_text())
    rw = M["readmit_window"]["arm_laya"]
    c2d = next(c["difference"] for c in C if (c["section"], c["reference"], c["metric"]) == ("C2", "duan", "macro_f1_tuned"))
    card = TEMPLATE
    values = {
        "PUBMED_TABLE": metric_table(M, "pubmedqa_answer", ["log_loss", "ece", "macro_f1", "accuracy"]),
        "YESNO_TABLE": metric_table(M, "pubmedqa_yes_no", ["log_loss", "ece", "macro_f1", "accuracy"]),
        "READMIT_TABLE": metric_table(M, "readmit_window", ["log_loss", "ece", "accuracy", "macro_f1", "macro_f1_tuned",
                                                            "auroc_30d"]),
        "READMIT_CAL_TABLE": metric_table(M, "readmit_window", ["calibration_intercept_30d", "calibration_slope_30d",
                                                                "quadratic_weighted_kappa"]),
        "MAIN_DIFFS": main_diffs,
        "SENS_TABLE": "\n".join(sens_lines),
        "READER_TABLE": reader_table,
        "BAR_TABLE": bar_table,
        "SUBGROUP_TABLE": "\n".join(sub_lines),
        "T_PUB": f"{temps['pubmedqa_answer']:.3f}",
        "T_READ": f"{temps['readmit_window']:.4f}",
        "KEPT_STEP": f"{kept['step']:,}",
        "STEPS": f"{run['optimizer_steps']:,}",
        "TRAIN_S": f"{run['train_seconds']:,.0f}",
        "RAW_PUB_ECE": f(raw_q["pubmedqa_answer"]["ece"]), "CAL_PUB_ECE": f(cal_q["pubmedqa_answer"]["ece"]),
        "RAW_PUB_LL": f(raw_q["pubmedqa_answer"]["log_loss"]), "CAL_PUB_LL": f(cal_q["pubmedqa_answer"]["log_loss"]),
        "RAW_RW_ECE": f(raw_q["readmit_window"]["ece"]), "CAL_RW_ECE": f(cal_q["readmit_window"]["ece"]),
        "RAW_RW_LL": f(raw_q["readmit_window"]["log_loss"]), "CAL_RW_LL": f(cal_q["readmit_window"]["log_loss"]),
        "RW_INTERCEPT": ci(rw["calibration_intercept_30d"]), "RW_SLOPE": ci(rw["calibration_slope_30d"]),
        "LGBM_LL": f(M["readmit_window"]["lightgbm"]["log_loss"]["estimate"]),
        "OUR_RW_LL": f(rw["log_loss"]["estimate"]),
        "CONTAMINATION": comp["contamination"].replace("Known contamination: ", "")
                         .replace("445 PQA-L items", "445 expert-labeled PubMedQA items")
                         .replace("may overlap our test split", "may overlap the test split used here"),
        "C2_DIFF": ci(c2d),
        "RUN2_T_LOW": f"{t2[0]:.2f}", "RUN2_T_HIGH": f"{t2[1]:.2f}",
        "RUN3_T_LOW": f"{t3[0]:.2f}", "RUN3_T_HIGH": f"{t3[1]:.2f}",
        "ECE_CHANCE": f"{100 * (1 - floor['share_at_most_0.05']):.0f}",
        "OURS_MAYBE_TOP": str(maybe_top["arm_laya"]),
        "N_MAYBE": str(n_maybe),
        "BEST_MAYBE_HITS": str(max(maybe_hits.values())),
        "C2_LOW_ABS": f(abs(c2d["low"])),
    }
    for k, v in values.items():
        card = card.replace("<<" + k + ">>", v)
    assert "<<" not in card, "a placeholder was not filled"
    assert "—" not in card and "–" not in card, "dashes"
    OUT.write_text(card, encoding="utf-8")
    print(f"wrote {OUT} ({len(card.split()):,} words)")


TEMPLATE = """# Model card for public v0
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 27 September 2026
Sources: docs/prompt_session4.md (step 10, with corrections 1 to 7); results/public_v0/comparison.json; results/public_v0/arm_laya_metrics.json; docs/data_sources_public.md; docs/prior_art.md (release-day check, 27 September 2026); written by scripts/build_model_card_v0.py
Status: complete (the name and repository are filled in by scripts/release_public_v0.sh from docs/naming.md); corrected 27 September 2026 under correction 1 in docs/prompt_session5.md (the priority sentence replaced)

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

# {{NAME}} v0.1

{{NAME}} v0.1 is a medical typed-decision model. I fine-tuned Laya (convaiinnovations/laya), a 421-million-parameter decision model built on ModernBERT-large, to answer two questions with calibrated probabilities in one forward pass: the answer to a biomedical research question from its abstract (PubMedQA: yes, no or maybe), and how soon a patient with diabetes is readmitted after a hospital discharge (UCI Diabetes 130-US hospitals, 1999 to 2008: none, after 30 days, or within 30 days). It is for research. It is not a medical device and must not be used to make clinical decisions.

Other medical models in the Laya family (Hugging Face check, 27 September 2026): Duan (wipen/Duan), licensed CC BY-NC 4.0 for non-commercial use, and impacte/laya-medical-escalation, an emergency severity index (ESI) triage model licensed Apache 2.0.

Correction: the version of this card at revision 0a2e55e (DOI 10.57967/hf/10639) said this was the first openly licensed medical typed-decision model built on Laya that permits commercial use. That sentence rested on a check at 21:23 UTC on 27 September 2026. impacte/laya-medical-escalation was published at 22:15 UTC, before this model was made public at 23:53 UTC, so I removed the sentence.

Named after my violin, which carries the name of Federigo Fiorillo, whose 36 Études, or studies, are where violinists learn to play in tune.

Abbreviations: ECE is expected calibration error (top answer, 15 equal-width bins); AUROC is the area under the receiver operating characteristic curve; UCI is the University of California, Irvine, Machine Learning Repository; MIMIC is the Medical Information Mart for Intensive Care; NRD is the Nationwide Readmissions Database; NIS is the National Inpatient Sample; NHAMCS is the National Hospital Ambulatory Medical Care Survey.

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

Each answer carries the probability of every option. Run through this code on a central processing unit (CPU), the released model reproduced its evaluation probabilities to within 0.001 on the six test items I checked.

## Training data

| Data | License | Used | Splits |
|---|---|---|---|
| PubMedQA (qiaojin/PubMedQA, revision 9001f2853fb87cab8d220904e0de81ac6973b318; official test list at commit 3e7616d288b52edaf38f5fd5c59821c7806bde86) | MIT | The question and the abstract's sections in source order. Left out: the conclusion (long_answer, which states the answer), the label, the original baselines' predictions and MeSH terms. Labels: the expert labels (yes, no, maybe) and, for the automatically labeled items, yes or no | Train 21,050 records: 350 expert-labeled items written 3 times and 20,000 automatically labeled items (yes and no in the expert items' ratio, 0.620 yes). Validation: 150 expert-labeled items (checkpoint choice and temperature) and 2,000 automatically labeled items (monitoring only). Test: the 500 official test items. No PubMed ID appears in two splits |
| Diabetes 130-US Hospitals for Years 1999 to 2008 (UCI dataset 296) | CC BY 4.0 | The first encounter of each patient; encounters ending in death or hospice dropped. Fields: age band, sex, admission type and source, discharge disposition, length of stay, counts of laboratory tests, procedures, medications, prior visits and diagnoses, three diagnoses coded in the International Classification of Diseases, Ninth Revision (ICD-9) with their chapter names, serum glucose and hemoglobin A1c results, diabetes medications and changes to them. Left out: identifiers, race (kept only to report calibration by group), payer code and weight | Split by patient, 80, 10 and 10 percent: 55,978 train, 6,997 validation, 6,998 test |

Citations. PubMedQA: Jin Q, Dhingra B, Liu Z, Cohen W, Lu X. PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP-IJCNLP 2019: 2567 to 2577. UCI data: Clore J, Cios K, DeShazo J, Strack B. Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository, 2014. https://doi.org/10.24432/C5230J. Strack B, DeShazo JP, Gennings C, Olmo JL, Ventura S, Cios KJ, Clore JN. Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. BioMed Research International 2014: 781670.

No MIMIC, NRD, NIS or NHAMCS data were used to train this model. Laya's model card does not list PubMedQA or the UCI data among its training data (read 27 September 2026).

## Training

I started from Laya (convaiinnovations/laya, revision 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851), its encoder and its decision head, and trained both on the two questions together.

- Loss: log loss over the options. AdamW (encoder learning rate 2.5e-5, head 1.0e-4, weight decay 0.01), cosine schedule, gradient clipping at 1.0, float16 mixed precision, gradient checkpointing.
- One epoch, <<STEPS>> optimizer steps, 32 sequences a step (8 per micro-batch, 4 accumulated), each micro-batch from one question; states up to 768 tokens, decision head 192.
- Checkpoint: validation every 250 steps; I kept the step with the lowest mean of the two questions' validation log losses (step <<KEPT_STEP>>).
- Calibration: one temperature per question, fitted on the validation items and shipped in calibration.json: PubMedQA <<T_PUB>>, readmission <<T_READ>>.
- Hardware: one NVIDIA H100 PCIe; <<TRAIN_S>> seconds of training.

Development history. This is the third training run, all on 27 September 2026. Run 1 wrote each expert-labeled PubMedQA training item 10 times and checked validation every 1,000 steps; the model memorized those items. Run 2 wrote them 3 times and checked every 250 steps, but its PubMedQA checkpoint choice and temperature came from 50 validation items and were poorly determined (the temperature's bootstrap 95% interval ran from <<RUN2_T_LOW>> to <<RUN2_T_HIGH>>). For run 3 I moved 100 expert-labeled items from training to validation (150 in validation, 350 in training) and changed nothing else. Each change to this model's training was written down, with reasons drawn from training logs and validation data, before the run it applied to. The test split was first scored in run 1, so these results are not from a test split used once; every run's test results are kept, and run 2's appear below as "Earlier run of this model".

## Evaluation

All results are on the test splits: 500 PubMedQA items (445 of them yes or no) and 6,998 readmission encounters. Intervals are 95% paired percentile bootstrap intervals (1,000 resamples, seed 20260927). Each neural model has one temperature per question fitted on the validation split; the prior and LightGBM are used as they are. In the comparisons with Duan, Duan is scored with and without its temperature and the better of the two is used for each measure, a rule set in the evaluation code before Duan was run. Systems:

- This model, and the earlier run of it (run 2).
- Two research arms from run 2, to compare readers: each has a new, randomly initialized decision head, trained alone for the first 250 steps and then with the reader, on run 2's data (50 PubMedQA validation items, 450 expert-labeled training items). One uses Laya's own reader; the other uses the reader of BioClinical ModernBERT-large (thomas-sounack/BioClinical-ModernBERT-large, whose pretraining included MIMIC clinical notes). Neither is released.
- Laya as published, not fine-tuned.
- LightGBM (gradient-boosted trees) on the same UCI fields, readmission only.
- Duan (wipen/Duan, revision 50e0615b576e4829cbad5c193d42edd0cadf2784), a Laya-family medical model licensed for non-commercial use, evaluated only: in its own question wording and input format, and in this model's wording. Duan's PubMedQA question is yes or no, so it is scored on the 445 yes or no items.
- The prior: every item gets the training split's label frequencies.

### PubMedQA, three answers (500 items)

<<PUBMED_TABLE>>

Resampling inflates ECE, so some ECE intervals lie above their own estimate (see Calibration).

This model never gave maybe as its top answer (<<OURS_MAYBE_TOP>> of 500 test items; <<N_MAYBE>> items are maybe), which holds its three-answer macro-F1 down. No system named more than <<BEST_MAYBE_HITS>> of the <<N_MAYBE>> maybe items correctly.

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

What the reader and the decision head contribute (research arms, not released):

<<READER_TABLE>>

### Release criteria

These three criteria were fixed before training. The model is released because all three passed on the test split.

<<BAR_TABLE>>

The readmission part of criterion 3 passed on the point estimate; the difference from Duan, <<C2_DIFF>>, is not significant, and its interval excludes Duan being better by more than <<C2_LOW_ABS>>.

## Calibration

The temperature lowered PubMedQA log loss from <<RAW_PUB_LL>> to <<CAL_PUB_LL>> and ECE from <<RAW_PUB_ECE>> to <<CAL_PUB_ECE>>; for readmission, log loss from <<RAW_RW_LL>> to <<CAL_RW_LL>> and ECE from <<RAW_RW_ECE>> to <<CAL_RW_ECE>>. For readmission within 30 days, the calibration intercept is <<RW_INTERCEPT>> and the slope is <<RW_SLOPE>> (1 is ideal): the model predicts slightly more 30-day readmissions than occur, and its predictions are slightly too close together.

Resampling inflates ECE, so an ECE interval can lie above its own estimate. The ECE on 500 items is also noisy: on validation data, a model calibrated by construction with this model's confidence profile would show an ECE above 0.05 on 500 items about <<ECE_CHANCE>>% of the time.

Calibration for readmission within 30 days by group, for groups with at least 100 test encounters:

<<SUBGROUP_TABLE>>

## Known contamination

<<CONTAMINATION>>

## Limitations

- Research use only. It is not a medical device, has not been validated prospectively or at any other site, and must not guide care.
- Two questions only. Other questions, wordings or input formats are untested and uncalibrated.
- The readmission data are from 130 US hospitals between 1999 and 2008, first encounters only, and practice has changed since. On these data, LightGBM on the same fields had lower log loss (<<LGBM_LL>> against <<OUR_RW_LL>>), and no difference was detected in AUROC.
- The model never gave maybe as its top answer on the test split, and cannot yet tell maybe abstracts apart from the others.
- The PubMedQA temperature was fitted on 150 items; its bootstrap 95% interval runs from <<RUN3_T_LOW>> to <<RUN3_T_HIGH>>.
- The test split was used by all three training runs (see Development history).
- English only.

## Lineage

ModernBERT-large (answerdotai/ModernBERT-large, Apache 2.0) was fine-tuned into Laya (convaiinnovations/laya, Apache 2.0), which was fine-tuned into this model.

## License and citation

Apache 2.0. The training data keep their own licenses (PubMedQA MIT; UCI data CC BY 4.0); the NOTICE file carries the attributions.

Li JE. {{NAME}} v0.1: a calibrated medical typed-decision model built on Laya. Hugging Face, 2026. https://huggingface.co/{{REPO}}
"""

if __name__ == "__main__":
    main()

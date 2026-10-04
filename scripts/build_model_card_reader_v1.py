"""Model card for reader v1 (docs/prompt_session5.md, correction 6).

  python3 scripts/build_model_card_reader_v1.py usage-check   runs the card's usage code on models/reader_v1 on the CPU and
                                                               writes results/reader_v1/usage_check.json
  python3 scripts/build_model_card_reader_v1.py build         writes docs/model_card_reader_v1.md

Every number in the card is read from results/reader_v1/card_numbers.json (src/eval/reader_v1_numbers.py) or from
usage_check.json. {{NAME}} and {{REPO}} are left for scripts/release_reader_v1.sh to fill from docs/naming.md.
Standard library only, except the usage check, which needs transformers and torch.
"""
import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMBERS = ROOT / "results" / "reader_v1" / "card_numbers.json"
USAGE = ROOT / "results" / "reader_v1" / "usage_check.json"
CARD = ROOT / "docs" / "model_card_reader_v1.md"
MODEL_DIR = ROOT / "models" / "reader_v1"
MODEL_SHA256 = "28e9a7b62be95615ae6ec80fbb3a92a4694644e27422c3c3048e01ec18c2d201"  # correction 6 (a)

USAGE_CODE = '''from transformers import pipeline

fill = pipeline("fill-mask", model="{{REPO}}")
for text in ["Metformin is a first-line treatment for type 2 [MASK].",
             "The patient was treated with intravenous [MASK] for bacterial pneumonia.",
             "Aspirin irreversibly inhibits platelet [MASK]."]:
    print(text, [(p["token_str"].strip(), round(p["score"], 3)) for p in fill(text, top_k=3)])
'''

CHECK_DATES = "27 and 28 September 2026"  # the Hugging Face checks behind the paragraph on other readers; update after a new check
LABELS = {"s5_laya": "Laya's own reader", "s5_bioclinical": "BioClinical ModernBERT-large's reader",
          "s5_mix_a": "Mix A (abstracts only), 500 million tokens",
          "s5_mix_b": "Mix B (abstracts, PMC full text, case reports), 500 million tokens",
          "s5_mix_c": "Mix C (mix B's sources and medical web text), 500 million tokens",
          "s5_final_fresh_head": "Reader v1, 2.0 billion tokens",
          "arm_laya": "{{NAME}} v0.1 (Laya's reader and Laya's own decision head)"}
HELDOUT_ROWS = [("abstracts", "PubMed abstracts (the training text)"), ("pmc", "PMC full-text articles"),
                ("case_reports", "PMC case reports"), ("web", "Medical web text"),
                ("task", "Training records of the two evaluation questions")]
DIFF_ROWS = [("s5_mix_a minus s5_laya", "Mix A minus Laya's reader"), ("s5_mix_b minus s5_laya", "Mix B minus Laya's reader"),
             ("s5_mix_c minus s5_laya", "Mix C minus Laya's reader"),
             ("s5_bioclinical minus s5_laya", "BioClinical ModernBERT-large's reader minus Laya's reader"),
             ("s5_final_fresh_head minus s5_bioclinical", "Reader v1 minus BioClinical ModernBERT-large's reader")]
HELDOUT_NOUNS = {"abstracts": "abstracts", "pmc": "articles", "case_reports": "case reports", "web": "web documents",
                 "task": "records"}
SOURCE_WORDS = {"abstracts": "abstracts", "pmc": "PMC full text", "case_reports": "case reports", "web": "medical web text"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def usage_check():
    import torch
    import transformers
    torch.set_num_threads(4)
    got = sha256(MODEL_DIR / "model.safetensors")
    if got != MODEL_SHA256:
        sys.exit(f"models/reader_v1/model.safetensors has SHA-256 {got}, not the one correction 6 names")
    buf, t0 = io.StringIO(), time.time()
    with contextlib.redirect_stdout(buf):
        exec(compile(USAGE_CODE.replace("{{REPO}}", str(MODEL_DIR)), "model-card-usage", "exec"), {})
    out = {"code": USAGE_CODE, "model_dir": "models/reader_v1", "model_sha256": got,
           "output": buf.getvalue().rstrip("\n").split("\n"), "seconds": round(time.time() - t0, 1),
           "transformers": transformers.__version__, "torch": torch.__version__, "device": "CPU (the linked shell, 4 threads)",
           "time_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    USAGE.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("\n".join(out["output"]))


def n(x):
    return f"{x:,}"


def diff(d, places=4):
    return f"{d['estimate']:+.{places}f} ({d['low']:+.{places}f} to {d['high']:+.{places}f})"


def verdict(d):
    return "significant" if d["low"] > 0 or d["high"] < 0 else "no difference detected"


def value(m, places=4):
    return f"{m['estimate']:.{places}f} ({m['low']:.{places}f} to {m['high']:.{places}f})"


def shares(mix):
    return ", ".join(f"{SOURCE_WORDS[k]} {v} percent" for k, v in mix.items())


def build():
    num = json.loads(NUMBERS.read_text(encoding="utf-8"))
    use = json.loads(USAGE.read_text(encoding="utf-8"))
    if use["code"] != USAGE_CODE or use["model_sha256"] != MODEL_SHA256:
        sys.exit("usage_check.json is not for this card's code and model; run usage-check again")
    t, tr, met, d3 = num["text"], num["training"], num["test_metrics"], num["d3_planned"]
    after = num["after_the_fact"]["results"]
    c4 = num["v0_2_release_bar"]["criterion_4"]["values"]
    rr, rules, mixes = num["run_to_run"], num["rules"], num["mixes"]
    hours = tr["gpu_seconds_for_reader_v1_derived"] / 3600

    rows_abs = []
    for sysname in ("s5_laya", "s5_bioclinical", "s5_mix_a", "s5_mix_b", "s5_mix_c", "s5_final_fresh_head", "arm_laya"):
        pq, rd = met["pubmedqa_answer"][sysname], met["readmit_window"][sysname]
        rows_abs.append(f"| {LABELS[sysname]} | {value(pq['log_loss'])} | {value(pq['accuracy'], 3)} | {value(rd['log_loss'])} | "
                        f"{value(rd['auroc_30d'])} |")

    def comp_row(label, block, fixed):
        pq, rd = block["pubmedqa_answer"], block["readmit_window"]
        cells = [f"{diff(pq['log_loss']['difference'])}, {verdict(pq['log_loss']['difference'])}",
                 f"{diff(pq['accuracy']['difference'], 3)}, {verdict(pq['accuracy']['difference'])}",
                 f"{diff(rd['log_loss']['difference'])}, {verdict(rd['log_loss']['difference'])}",
                 f"{diff(rd['auroc_30d']['difference'])}, {verdict(rd['auroc_30d']['difference'])}"]
        return f"| {label} | " + " | ".join(cells) + f" | {fixed} |"

    rows_diff = [comp_row(label, d3[key], "yes") for key, label in DIFF_ROWS]
    rows_diff.append(comp_row("Reader v1 minus Laya's reader", after, "no, computed after the results"))
    held, counts = num["heldout"], num["heldout"]["counts"]
    rows_held = [f"| {label} | {n(counts[k]['heldout_blocks'])} (from {n(counts[k]['heldout_documents'])} {HELDOUT_NOUNS[k]}) | "
                 f"{held['mix_a_500m']['loss'][k]:.5f} | {held['reader_v1']['loss'][k]:.5f} |" for k, label in HELDOUT_ROWS]
    ar = after["readmit_window"]["log_loss"]["difference"]

    p4, r4 = c4["pubmedqa_answer"], c4["readmit_window"]
    sc = rules["reader_rule"]["scores"]
    card = TEMPLATE
    fill = {
        "<<CHECK_DATES>>": CHECK_DATES, "<<PARAMS>>": "395,881,664", "<<TOKENS>>": n(tr["tokens"]),
        "<<DOCS>>": n(t["documents"]), "<<TEXT_TOKENS>>": n(t["tokens"]), "<<SHARDS>>": n(t["shards_used"]),
        "<<PUBIDS>>": n(t["pubmedqa_pubids_excluded"]), "<<DROP_PMID>>": n(t["drops"]["pubmedqa_pmid"]),
        "<<DROP_13>>": n(t["drops"]["thirteen_word"]), "<<HELD_DOCS>>": n(t["heldout_documents"]),
        "<<BLOCK>>": n(t["block_tokens"]), "<<TRAIN_BLOCKS>>": n(t["train_blocks"]), "<<DRAWN>>": n(tr["blocks_drawn"]),
        "<<STEPS>>": n(tr["steps"]), "<<PER_STEP_BLOCKS>>": n(tr["blocks_per_step"]), "<<PER_STEP_TOKENS>>": n(tr["tokens_per_step"]),
        "<<WARMUP>>": n(tr["warmup"]), "<<DECAY>>": n(tr["decay_start"]), "<<RESUMED>>": n(tr["resumed_at"]),
        "<<LR>>": f"{tr['peak_lr']:g}", "<<SEED>>": str(tr["seed"]), "<<HOURS>>": f"{hours:.1f}",
        "<<TPS>>": n(round(tr["tokens_per_second_long"])), "<<PEAK_MEM>>": f"{tr['peak_memory_gb']}",
        "<<MIX_B>>": shares(mixes["B"]), "<<MIX_C>>": shares(mixes["C"]),
        "<<MARGIN>>": f"{rules['reader_rule']['margin']:.2f}",
        "<<SCORES>>": f"A {sc['A']:.5f}, B {sc['B']:.5f}, C {sc['C']:.5f}",
        "<<TASK_SCORE>>": f"{rules['task_rule']['s5_task_score']:.5f}",
        "<<RR_P>>": f"{rr['pubmedqa_answer']['largest_move']:.3f}", "<<RR_R>>": f"{rr['readmit_window']['largest_move']:.3f}",
        "<<AFTER_R>>": f"{ar['estimate']:+.4f}",
        "<<USAGE_CODE>>": USAGE_CODE.rstrip("\n"), "<<USAGE_OUTPUT>>": "\n".join(use["output"]),
        "<<USAGE_VERSIONS>>": f"transformers {use['transformers']}, torch {use['torch']}",
        "<<USAGE_DATE>>": dt.datetime.strptime(use["time_utc"], "%Y-%m-%dT%H:%M:%SZ").strftime("%-d %B %Y, %H:%M UTC"),
        "<<HELD_ROWS>>": "\n".join(rows_held), "<<ABS_ROWS>>": "\n".join(rows_abs), "<<DIFF_ROWS>>": "\n".join(rows_diff),
        "<<N_PQA>>": n(after["pubmedqa_answer"]["n"]), "<<N_READ>>": n(after["readmit_window"]["n"]),
        "<<C4_P_V2>>": f"{p4['s5_final_fresh_head']:.4f}", "<<C4_P_V1>>": f"{p4['arm_laya']:.4f}",
        "<<C4_P_DIFF>>": diff(p4["difference"]), "<<C4_R_V2>>": f"{r4['s5_final_fresh_head']:.4f}",
        "<<C4_R_V1>>": f"{r4['arm_laya']:.4f}", "<<C4_R_DIFF>>": diff(r4["difference"]),
        "<<PREPARED>>": dt.date.today().strftime("%-d %B %Y"),
    }
    for k, v in fill.items():
        card = card.replace(k, v)
    assert "<<" not in card, "a placeholder was not filled"
    assert chr(0x2014) not in card and chr(0x2013) not in card, "dashes"
    lower = card.lower()
    for phrase in ("first open", "first medical", "first biomedical", "first model", "first public", "first commercial",
                   "first mimic", "first to ", "first reader trained", "novel", "unique", "state-of-the-art", "state of the art",
                   "notably", "importantly", "it is worth noting", "robust", "leverage", "delve", "crucially", "equivalent",
                   "comparable", "only mimic-free", "the only"):
        assert phrase not in lower, f"banned phrase: {phrase}"
    CARD.write_text(card, encoding="utf-8")
    print(f"wrote {CARD.relative_to(ROOT)} ({len(card):,} characters)")


TEMPLATE = """# Model card for reader v1
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: <<PREPARED>>
Sources: docs/prompt_session5.md (correction 6); results/reader_v1/card_numbers.json (src/eval/reader_v1_numbers.py, from the session 5 results); results/reader_v1/usage_check.json; docs/prior_art.md (release-day check, 28 September 2026, reader v1); docs/data_sources_session5.md; written by scripts/build_model_card_reader_v1.py
Status: complete (the name and repository are filled in by scripts/release_reader_v1.sh from docs/naming.md); revised after an independent review on 29 September 2026

Everything below the next line becomes the Hugging Face README.md.

<!-- model card begins -->
---
license: apache-2.0
base_model: convaiinnovations/laya
base_model_relation: finetune
library_name: transformers
pipeline_tag: fill-mask
language:
- en
tags:
- modernbert
- fill-mask
- biomedical
- medical
---

# {{NAME}} reader v1

Abbreviations: MIMIC is the Medical Information Mart for Intensive Care, a credentialed database of clinical records; NLM is the U.S. National Library of Medicine; PMC is PubMed Central; UCI is the University of California, Irvine, whose Machine Learning Repository holds the readmission data; AUROC is the area under the receiver operating characteristic curve; CPU is central processing unit; NRD is the Nationwide Readmissions Database; NIS is the National Inpatient Sample; NHAMCS is the National Hospital Ambulatory Medical Care Survey.

{{NAME}} reader v1 is a biomedical masked-language model with <<PARAMS>> parameters and the ModernBERT-large architecture. I made it from Laya (convaiinnovations/laya): I took Laya's encoder, gave it ModernBERT-large's masked-language-model head, and trained it further on <<TOKENS>> tokens of PubMed titles and abstracts. I added no MIMIC or other credentialed clinical data: its further training read PubMed abstracts only, and neither Laya's card nor ModernBERT-large's lists clinical notes among their training data (their full training text is not published). It is the reader I trained for {{NAME}} v0.2, a medical typed-decision model. I release it on its own because the decision layer built on it did not clear the bar I set before training (see What it did not achieve). In a comparison I computed after the results, the decision layer on reader v1 had lower log loss than the same decision layer on Laya's own reader on both questions and higher PubMedQA accuracy, with no difference detected in readmission AUROC; the comparison fixed before training, at 500 million tokens, showed the same pattern. It is for research and for fine-tuning. It is not a medical device and must not be used to make clinical decisions.

The openly licensed biomedical ModernBERT-large readers I found in Hugging Face checks of <<CHECK_DATES>> were each trained on tens of billions of tokens that include clinical notes from MIMIC: BioClinical ModernBERT-large (thomas-sounack/BioClinical-ModernBERT-large, MIT license; 53.5 billion tokens including MIMIC-III and MIMIC-IV notes), Bio ModernBERT-large (thomas-sounack/Bio-ModernBERT-large, MIT license; its first phase included the same clinical data) and ModernBERT-bio-large (almanach/ModernBERT-bio-large, Apache 2.0; 55 billion tokens, a fifth of them MIMIC clinical notes). The other biomedical models in the ModernBERT family that those checks found are base size, gated, without a license, in another language or without public weights; one of them, Clinical ModernBERT (Simonlee711/Clinical_ModernBERT, MIT license, base size), also includes MIMIC-IV notes.

It carries the name of {{NAME}} v0.1, this project's decision model.

## How to use

```python
<<USAGE_CODE>>
```

On a CPU, with this repository's files, it printed (<<USAGE_DATE>>; <<USAGE_VERSIONS>>):

```
<<USAGE_OUTPUT>>
```

For classification, load it with `AutoModelForSequenceClassification` and fine-tune it; for sentence vectors, load it with `AutoModel` and pool the last hidden states. Its encoder has the same architecture and tokenizer as Laya's, so it can also take the place of Laya's encoder in a Laya decision model, which is how I used it. It was trained on blocks of <<BLOCK>> tokens. The architecture accepts up to 8,192 tokens, as ModernBERT-large does, but longer inputs were not tested.

## Training data

- <<DOCS>> PubMed titles and abstracts (<<TEXT_TOKENS>> tokens) from NLM's 2026 annual baseline, read from <<SHARDS>> of the baseline's 1,334 files in a random order fixed by a seed (<<SEED>>).
- I removed every article in PubMedQA (<<PUBIDS>> distinct PubMed identifiers across its three sets), which took out <<DROP_PMID>> abstracts, and <<DROP_13>> more abstracts that shared a run of 13 words with the contexts or long answer of one of PubMedQA's 1,000 expert-labeled items.
- I held out <<HELD_DOCS>> abstracts (1,000 blocks of <<BLOCK>> tokens), used only to measure loss.
- Not used: full-text articles, clinical notes, and any MIMIC, NRD, NIS or NHAMCS data.

## Training

- Start: Laya's encoder (convaiinnovations/laya, revision 55cf4c4e) with the masked-language-model head of ModernBERT-large (answerdotai/ModernBERT-large, revision 45bb4654); the output layer shares its weights with the input embeddings. The two models have the same vocabulary and merge rules, so Laya's tokenizer is used unchanged.
- Objective: masked-language modeling; 30 percent of tokens are selected, and of those 80 percent become [MASK], 10 percent a random token and 10 percent stay unchanged.
- Batches: <<PER_STEP_BLOCKS>> blocks of <<BLOCK>> tokens a step (<<PER_STEP_TOKENS>> tokens); <<STEPS>> steps and <<TOKENS>> tokens in all; <<DRAWN>> of the <<TRAIN_BLOCKS>> training blocks were drawn, each at most once.
- Optimizer: AdamW (betas 0.9 and 0.98, epsilon 1e-6, weight decay 0.01 except on biases and normalization weights), peak learning rate <<LR>>, gradient clipping at 1.0, bfloat16 autocast; linear warmup over <<WARMUP>> steps, then constant, then linear decay to zero from step <<DECAY>>.
- How I chose the text: I compared three mixes at 500 million tokens each, by token share: A, abstracts only; B, <<MIX_B>>; C, <<MIX_C>>. A rule I fixed before training kept A unless another mix scored more than <<MARGIN>> better on the validation data of the two questions below. Neither did (selection scores <<SCORES>>; lower is better), so I continued A's run from the start of its decay (step <<RESUMED>>) to <<TOKENS>> tokens. I also tried a round of training on the two questions' own training records and, by a rule I fixed before training, did not keep it (score <<TASK_SCORE>>).
- Hardware: one NVIDIA H100 (80 GB); about <<HOURS>> hours for this model's <<STEPS>> steps (derived from the logs of the two runs), <<TPS>> tokens a second, peak memory <<PEAK_MEM>> GiB.

## Evaluation

### Held-out masked-language loss

Lower is better. Only abstracts were in the further training, so the other rows are text this training did not see. The first number column is mix A's run at 500 million tokens, at the end of its own schedule.

| Held-out text | Held-out blocks of <<BLOCK>> tokens | Mix A, 500 million tokens | Reader v1, 2.0 billion tokens |
|---|---|---|---|
<<HELD_ROWS>>

### With a decision layer

To see whether the reader helps a model that answers questions, I trained the same decision layer on each reader: a new decision head, trained with {{NAME}} v0.1's method on two questions (the reader is fine-tuned with it after the first 250 steps) and calibrated with a temperature fitted on validation data. The questions are the answer to a biomedical research question from its abstract (PubMedQA: yes, no or maybe; <<N_PQA>> test items) and how soon a patient with diabetes is readmitted after a hospital discharge (UCI Diabetes 130-US hospitals, 1999 to 2008: none, after 30 days, or within 30 days; <<N_READ>> test encounters). Every number is on the test split, with a 95 percent interval from 1,000 bootstrap resamples of the test rows (paired for differences). {{NAME}} v0.1, which keeps Laya's reader and Laya's own decision head, is in the last row for reference.

Each reader got one decision-layer training run, and the intervals cover the sampling of test rows only, not the randomness of training. In this project, training the same reader again with nearly the same recipe moved PubMedQA log loss by up to <<RR_P>> and readmission log loss by up to <<RR_R>>, which is more than every readmission difference below. Read the differences with that in mind.

| Reader, same decision layer | PubMedQA log loss | PubMedQA accuracy | Readmission log loss | Readmission AUROC within 30 days |
|---|---|---|---|---|
<<ABS_ROWS>>

Differences (the first reader minus the second; a negative log loss difference favors the first reader). The first five rows are every reader comparison I fixed before training:

| Comparison | PubMedQA log loss | PubMedQA accuracy | Readmission log loss | Readmission AUROC within 30 days | Fixed before training |
|---|---|---|---|---|---|
<<DIFF_ROWS>>

### What it did not achieve

The decision layer on reader v1 was the candidate for {{NAME}} v0.2. Against {{NAME}} v0.1, its PubMedQA log loss was <<C4_P_V2>> against <<C4_P_V1>>, a difference of <<C4_P_DIFF>>, and its readmission log loss <<C4_R_V2>> against <<C4_R_V1>>, a difference of <<C4_R_DIFF>>. Before training I had required the PubMedQA interval to lie wholly below zero. It did not, so v0.2 was not released, and v0.1 remains the current decision model.

## Known contamination

The pretraining text of Laya and of ModernBERT-large is not published and may include PubMed pages. Reader v1's further training excluded every PubMedQA article, by PubMed identifier and by the 13-word check above. BioClinical ModernBERT-large was trained on PubMed and PMC text that includes PubMedQA's articles, which favors its reader on PubMedQA in the tables above. I did not exclude other benchmarks built from PubMed abstracts (for example BioASQ, PubMed 200k RCT, BC5CDR and NCBI-disease), so results on them may overstate what the reader learned.

## Limitations

- It was evaluated only by held-out masked-language loss and by two questions under one decision-layer recipe, with one training run per reader; the differences between readers are within what retraining alone can move on readmission.
- Its further training read English titles and abstracts only: no full-text articles and no clinical notes. How it does on clinical notes was not tested.
- It was trained on blocks of <<BLOCK>> tokens; inputs longer than that were not tested.
- It is a reader, not a decision model: no classification head is included, so it needs fine-tuning before it classifies text or answers questions.
- The literature it read carries its own biases, such as the questions researchers chose to study and what got published.
- It is not a medical device and must not be used to make clinical decisions.

## Lineage

ModernBERT-large (answerdotai/ModernBERT-large, Apache 2.0) was fine-tuned into Laya (convaiinnovations/laya, Apache 2.0). This model is Laya's encoder with ModernBERT-large's masked-language-model head, trained further on PubMed titles and abstracts. {{NAME}} v0.1 (johanneli/fiorillo-v0.1) is built on Laya's own reader, not on this one.

## License and citation

Apache 2.0, like Laya and ModernBERT-large; the LICENSE file holds the license text and the NOTICE file carries the attributions.

This model was trained in part on data from the U.S. National Library of Medicine (NLM): PubMed citations and abstracts. NLM is not responsible for this model and has not endorsed it, and the data it was trained on may not reflect the most current data available from NLM. Some abstracts may be protected by copyright; only the model's weights are released, and no abstract text is distributed with them.

Li JE. {{NAME}} reader v1: a biomedical ModernBERT-large reader further trained on PubMed abstracts. Hugging Face, 2026. https://huggingface.co/{{REPO}}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["usage-check", "build"])
    args = parser.parse_args()
    usage_check() if args.command == "usage-check" else build()


if __name__ == "__main__":
    main()

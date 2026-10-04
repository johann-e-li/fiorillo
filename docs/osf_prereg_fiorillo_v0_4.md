# OSF pre-registration draft: Fiorillo v0.4 on Evidence Inference 2.0

Draft of 30 September 2026, written from docs/prompt_session8.md at commit 77087bd (corrections 1 to 13) and checked against that plan and the code by a separate agent. Each section below is one field of OSF's "OSF Preregistration" template, in the template's order. The filing notes at the end are not part of the registration.

## Study information

### Title

Fiorillo v0.4 against Gemma 4 31B-it on Evidence Inference 2.0: a test fixed before any test prediction

### Description

Fiorillo is an open medical model I am building. It answers typed questions (one answer from a fixed set) with calibrated probabilities, and it can hand its least confident answers to a larger model. Version 0.4 routes each question to a specialist. This registration concerns its new question, from Evidence Inference 2.0 (EI; DeYoung et al., 2020): given a randomized trial's article and a prompt that names an intervention, a comparator and an outcome, did the outcome significantly increase, significantly decrease, or show no significant difference?

I fixed the test below in my plan (docs/prompt_session8.md) before training began, and I am registering it before any model has made a prediction on EI's test split. After the exclusions under Data exclusion, the test split has 1,218 prompts in 333 articles. Before any test prediction, it was read only to build its records and to compute the figures listed under Explanation of existing data.

The EI specialist, s8_ei, averages three fine-tuning runs of Qwen3-4B-Base with LoRA adapters and Laya's decision head. Each run starts from my session 7 model s7_b3, v0.4's PubMedQA specialist, and differs only in its seed. The reference is Gemma 4 31B-it, used zero-shot, with each answer's probability read from its option letters. Both are calibrated on EI's validation split by the same rule.

PubMedQA and the readmission window (no readmission, readmission after 30 days, readmission within 30 days), v0.4's other two questions, are reported descriptively only, because I read their test splits in earlier sessions.

The attached plan (commit 77087bd, corrections 1 to 13) gives every detail. Where this form and the attached plan differ, the attached plan governs. Any later correction to the plan is a deviation from this registration and will be reported as one.

### Hypotheses

v0.4 passes its bar only if all four criteria hold on EI's test split. The system tested is s8_ei with its calibration; the comparisons are the prior and Gemma 4 31B-it with its calibration. Every criterion is computed over all 1,218 test prompts.

- H1 (calibration). s8_ei's expected calibration error (ECE) is at most the ECE limit. The limit is the larger of 0.05 and the 95th percentile of the ECE that a perfectly calibrated model shows by chance on 1,218 prompts: 5,000 simulated test sets of 1,218 prompts, drawn with replacement one prompt at a time (not by article) from s8_ei's 1,237 calibrated validation predictions, each label drawn from the row's own probabilities (numpy's default_rng(20260927)). Drawing prompts rather than articles makes the limit a little strict. The limit is computed and frozen before any test prediction exists.
- H2 (better than the prior). s8_ei's log loss is below the prior's, and the 95 percent interval of the difference (s8_ei minus the prior) lies wholly below zero. The prior gives every prompt the training split's answer shares.
- H3 (the specialist's claim). s8_ei's log loss is below Gemma 4 31B-it's, and the 95 percent interval of the difference (s8_ei minus Gemma 4 31B-it) lies wholly below zero.
- H4 (top answers not worse by macro-F1). s8_ei's macro-F1 is not below Gemma 4 31B-it's point estimate.

If any criterion fails, I record "v0.4 release bar not met" with the reason.

## Design plan

### Study type

Other: a pre-specified evaluation of machine-learning models on the held-out test split of an existing public dataset. No data are collected from people.

### Blinding

No blinding is involved in this study.

### Is there any additional blinding in this study?

EI's test labels are public, so the evaluation cannot be blind. What I fixed instead is the order. Before any test prediction exists, a freeze step writes release_candidate.json, which records each system's calibration, the ECE limit and the SHA-256 of the files they come from. The test-prediction stage refuses to start without it, and first records its SHA-256 and the stage's start time in testpreds_log.json. Before the freeze, the test labels are read only by the record builder, which writes them into the test records and counts them, and, before the plan was written, by the check described under Explanation of existing data. After the freeze, the test-prediction stage copies them into the prediction files and computes raw scores in its run summaries, and the analysis reads them. Before anything from the rented GPU is unpacked on my Mac, the unpacking step stops unless testpreds_log.json names release_candidate.json's SHA-256 and a start time later than the freeze, and the test calibration stops unless the calibration files match the SHA-256 the freeze recorded.

### Study design

Inputs. Both models read the same selected text of each article. It is chosen so that Qwen's whole input (the intervention, comparator, outcome and selected text, then the question and its options) fits 4,096 tokens of Qwen's tokenizer with 16 to spare. The title and abstract come first, cut from their end if they alone do not fit. Passages are then taken while they fit: results-section passages before the others, each group in order of BM25 score against the prompt's intervention, comparator and outcome, ties by position. BM25's statistics come from the passages of the training split's articles only and are used for every split. The kept passages follow the title and abstract in their article order. On the training split, among the 9,850 prompts with usable evidence offsets, at least one accepted evidence span lies wholly inside the kept text for 9,487 (96.3 percent).

s8_ei. Three runs (seeds 20260927, 20260928 and 20260929) of Qwen3-4B-Base (revision 906bfd4b4dc7f14ee4320094d8b41684abff8539) with LoRA adapters (rank 16, alpha 32, dropout 0.05) and Laya's decision head (two transformer layers and a scorer over the options). Each run starts from s7_b3's adapters, projection and head and trains for 2 epochs on EI's 10,046 training prompts: AdamW at 5e-5 for the adapters and 1e-4 for the projection and head, a cosine schedule to 1e-6, gradient clipping at 1.0, weight decay 0.01, bfloat16, and 32 prompts per optimizer step (micro-batch 1 with accumulation 32, chosen by a timed pilot between two memory settings of the same training). The last weights are kept; nothing selects a checkpoint. s8_ei's probabilities for a prompt are the mean of the three runs' probabilities.

Gemma 4 31B-it. Revision 842da3794eaa0b77d5f08bae87a17459d91ff475, bfloat16, through vLLM 0.30.0, text only, thinking off. The prompt is its chat template with one user message: the intervention, comparator and outcome, the same selected article text, the question, the three options lettered A, B and C, and "Answer with the letter of one option only." Each prompt gets one greedy request for a single new token (temperature 0, the top 20 log-probabilities), and more only if a letter is missing. An option's probability is the summed probability of its letter's one-token variants (with and without a leading space) for that token, and the three are renormalized. A prompt missing a letter in the top 20 is asked again with the top 1,000; a prompt still missing a letter is scored in a second process from the log-probability of the prompt followed by each letter variant. Each prompt's method is recorded.

The prior. The training split's answer shares: significantly increased 3,054, significantly decreased 2,454 and no significant difference 4,538, of 10,046.

Calibration. Each system's rule is fitted on all 1,237 EI validation prompts. There are two families: temperature, softmax(log p / T) with T from 0.1 to 10; and temperature with per-answer intercepts, softmax(log p / T + b), with b fixed at 0 for the first answer. Each is fitted by maximum likelihood and scored by out-of-fold log loss (repeated stratified 5-fold cross-validation, 20 repeats, seed 20260927); the lower one is chosen (an exact tie goes to temperature) and fitted on all validation prompts. Gemma 4 31B-it gets the same rule, so a bias toward one letter can be corrected as s8_ei's bias can.

Order. Training and validation predictions; Gemma 4 31B-it on validation; calibration and the ECE limit; the freeze; then test predictions for every system; then the analysis.

### Randomization

No assignment is randomized. Every random step uses a fixed seed: 20260927 everywhere, the three EI runs 20260927, 20260928 and 20260929, and the bootstrap numpy's default_rng(20260927).

## Sampling plan

### Existing data

Registration following analysis of the data.

### Explanation of existing data

EI 2.0 is public (github.com/jayded/evidence-inference at commit a661e8c14f973398380c8865cf2f27a535aaaf6d, MIT license), and I downloaded it at that commit. Before the plan was written, a check of the annotations counted 1,224 test prompts with accepted rows (of 1,228 with any row) and 3 whose accepted labels disagree, and found the test article PMC2206488 empty. Building the records then read the test split's prompts, annotations and article texts to write its records, choosing each prompt's text by BM25 with statistics from the training split only. It computed these figures on the test split: 1,218 prompts kept in 333 articles; 28 prompts left out for having no accepted annotation and 6 for an empty article text (PMC2206488); answers significantly increased 411, significantly decreased 294 and no significant difference 513; 3 prompts whose accepted labels disagree; tokens per record (mean 3,870, 90th percentile 4,076, largest 4,080); and each record file's SHA-256. For each test prompt, the builder also marked whether an accepted evidence span lies inside the kept text; this is saved in its working files and was never summarized. The rented GPU rebuilt the same records from the saved selection and checked them against those SHA-256s. Europe PMC was asked for every article's license, test articles included. No model has made a prediction on any test prompt. The validation split is used as planned: 300 of its prompts for monitoring during training; 4 of its EI prompts and 2 PubMedQA prompts for a smoke test of the large model's engine with a small Gemma 4 model, whose outputs are used for nothing; and the whole split for validation predictions, calibration, the ECE limit, the check of Gemma 4 31B-it's letter probabilities, a power check and a validation run of the hand-off. The EI paper and its leaderboard publish results on this test split, and its labels are public.

### Data collection procedures

The records are built from the fixed commit by src/tasks/evidence_inference.py. A prompt's answer is the most common label among its accepted annotation rows (Valid Label and Valid Reasoning both true, and a non-empty annotation text). A tie goes to the smallest of EI's Label Codes: -1 (significantly decreased), then 0 (no significant difference), then 1 (significantly increased). The records store the answer as an index in the question's order (significantly increased 0, significantly decreased 1, no significant difference 2).

### Sample size

1,218 test prompts in 333 articles: every test prompt with an accepted answer and a readable article text.

### Sample size rationale

The size is fixed. Power for H3 (normal approximation, derived before any prediction): with a per-prompt standard deviation s of the log-loss difference and an article design effect D, the standard error is s times the square root of D / 1,218. At s 0.8 and D 1.5 it is about 0.028, the pass line is a difference of about -0.055, and the chance of passing is about 80 percent at a real gain of about 0.079 in log loss. The attached power.json gives the grid (s 0.6, 0.8 and 1.0; D 1.0, 1.5 and 2.0; gains from 0 to 0.3). An estimate of s and D on validation is recorded for information only; it does not change the bar.

### Stopping rule

None. Each system's test predictions are made once. If the test-prediction stage stops on an error, it is run again only for the jobs whose test files are not all there, and testpreds_log.json keeps its first entry and adds the retry's time; nothing is run again to change a completed test result.

## Variables

### Manipulated variables

None. The systems compared are s8_ei with its calibration, Gemma 4 31B-it with its calibration, and the prior.

### Measured variables

For each test prompt and system: the three answer probabilities (calibrated for s8_ei and Gemma 4 31B-it; the training shares for the prior; also raw and temperature-only for s8_ei). For each test prompt: its answer, its article (PMCID), its flag in EI's annotations README (Incorrect, Questionable, Somewhat malformed, or none) and its accepted labels (used only for the lenient scoring in the published comparison).

### Indices

- Log loss: the mean over prompts of the negative natural log of the probability given to the answer, with probabilities floored at 1e-12.
- ECE: 15 equal-width bins on the top answer's probability; the sum over bins of the absolute gap between mean confidence and accuracy, weighted by the bin's share of prompts.
- Macro-F1: the mean F1 over the answers that appear among the labels or the top answers.
- Also reported: the Brier score (summed over the three answers), accuracy and each answer's F1.
- The top answer is the one with the highest probability; an exact tie goes to the first option in the order significantly increased, significantly decreased, no significant difference.

## Analysis plan

### Statistical models

A paired bootstrap by article. Prompts from one article are not independent, so each resample draws the 333 articles with replacement (numpy's default_rng(20260927), integers(0, m, m) over the articles sorted by PMCID as text) and keeps every prompt of each drawn article. Both systems of a difference are scored on the same resamples. Intervals are percentile intervals (the 2.5th and 97.5th percentiles). H2 and H3 use 10,000 resamples; every other EI interval uses 1,000.

### Transformations

s8_ei's raw probabilities are the mean of its three runs' probabilities, and its calibration reads their logarithm. Gemma 4 31B-it's letter probabilities are renormalized over the three options before its calibration. Calibration is as described under Study design.

### Inference criteria

H1: the ECE point estimate is at most the limit. H2 and H3: the log-loss difference (s8_ei minus the comparison, over all 1,218 prompts) is below zero, and the upper end of its 95 percent percentile interval from 10,000 article resamples is below zero; comparing the upper end of a two-sided 95 percent interval with zero is a one-sided test at the 2.5 percent level. H4: s8_ei's macro-F1 point estimate is at least Gemma 4 31B-it's. The 1,000-resample intervals of the same differences in the descriptive tables do not decide H2 or H3. A criterion whose inputs are missing (a system's test predictions, or the ECE limit in release_candidate.json) fails. The bar is the conjunction of the four, so no correction for multiple comparisons is applied to it. A difference whose interval includes zero is reported as "no difference was detected," with the estimate, the interval and what the interval excludes, never as equivalence.

### Data exclusion

Training leaves out the 70 training prompts that EI's annotations README lists as "Incorrect". Of the 1,252 test prompts, the 28 with no accepted annotation and the 6 of the empty article PMC2206488 are left out (published results counted those 6); no other test prompt is left out, and the primary analysis uses all 1,218.

### Missing data

Each system must produce all 1,218 test rows; the stages check the row counts and stop otherwise. A prompt whose letters Gemma 4 31B-it leaves out of its top 20 is answered by the further methods under Study design. Before any test prediction: if the median of the three letters' summed probability in Gemma 4 31B-it's EI validation run is below 0.5, I append a correction to the plan saying how its answers are read, and I report it as a deviation.

### Exploratory analysis

Secondary and descriptive, with 95 percent intervals (1,000 resamples; EI by article, the others by item) and without tests:
- Seeds: s8_ei against each of its three runs alone (each run calibrated by its own rule), and the spread of the three runs' log loss.
- Published results: s8_ei's macro-F1 beside DeYoung et al. 2020 (0.780 from the full text, 0.851 with the evidence given), and, scored leniently as Buchmann et al. 2024 scored, beside their LED (0.7430 to 0.7649) and LongT5 (0.8154). The lenient score replaces a prompt's answer with the prediction when the prediction is one of its accepted labels. BioDecision-4B's 70.6 was measured on a different copy of EI and is named, not compared. Each is stated with what s8_ei's interval excludes. These systems read the articles in other ways, so these are not tests.
- PubMedQA: v0.4 (s7_b3) minus v0.1; v0.4 minus Gemma 4 31B-it; s8_a3 (arm A retrained keeping the last weights) minus s7_a3; s8_a3 minus v0.1; and a check that s7_b3 loaded as it was.
- Readmission: the readmission pool minus lightgbm_tuned, minus blend_v0_1 and minus v0.1; xgboost_tuned and catboost_tuned minus lightgbm_tuned; s8_a3 minus s7_a3; s8_a3 minus v0.1. I read both of these test splits in earlier sessions.
- The hand-off, for EI (s8_ei) and PubMedQA (s7_b3): items ranked by the specialist's top calibrated probability, ties by record ID; at 0, 10, 20, 30 and 50 percent, the floor of rate times n least confident items go to Gemma 4 31B-it (calibrated), ranked again within each resample. Reported: the combined answers' accuracy, macro-F1 and log loss; on the handed-off items, each model's accuracy; Gemma 4 31B-it alone; and the cost per 1,000 questions from forward-pass GPU seconds measured in the test run, at $3.51 an hour.
- Sensitivity, descriptive and without a pass or fail: the differences s8_ei minus the prior and s8_ei minus Gemma 4 31B-it, on every metric, without every flagged test prompt (Incorrect, Questionable or Somewhat malformed), and the same differences with s8_ei's raw and temperature-only predictions in place of s8_ei.
- For every system, the rows whose probability of the true answer prints as 0.

Anything else is labeled exploratory where it appears.

## Other

Deviations made before this registration. Each was written into the plan as a numbered correction before any test prediction existed. None changes the four criteria, the calibration rule or the list of comparisons.
- Correction 5 changes how Gemma 4 31B-it's letter probabilities are obtained for a prompt missing a letter from its top 20 (its top 1,000 first, then the second process), which is an input to H3 and H4; the quantity read is the same, its next-token probabilities. It also fixes a text-only engine without CUDA graphs.
- Corrections 1, 7, 8, 10 and 11 change how the engine is installed, checked and run, not the model, its prompt or how its probabilities are read. Correction 11 uses vLLM's own sampler, which cannot change an answer, because every request is greedy with one new token and the log-probabilities are read from the raw logits.
- Correction 6: the rented GPU as deployed (a 150 GB disk instead of 200 GB, a CUDA 12.8 driver, $3.51 an hour), and a stop before any EI seed if the zips could not be copied before the server's maintenance from 1 October at 19:12 UTC.
- Corrections 2 and 3: where the readmission tree models are fitted (descriptive part only).
- Correction 4: session 7's s7_b3-raw PubMedQA test file is uploaded to the rented GPU, where only the check that s7_b3 loaded as it was reads it; the test of a tree fit resumed from its saved state is not built, because correction 3 removed that state; the validation hand-off's PubMedQA cost is left empty, and its EI costs are marked as possibly measured while arm A shared the GPU.
- Correction 9: code fixes from two independent reviews, and a smoke test that runs a small Gemma 4 model on six validation prompts, whose outputs are used for nothing.
- Correction 12: each vLLM process shuts its engine down on every way out, also after an error, and stops any child process left running; the smoke test's checks are changed (agreement within one engine instead of across processes) and extended to the script's own passes, with timeouts.
- Correction 13: the test-prediction stage waits for this registration, and the sensitivity analyses are stated as the code computes them (the differences, descriptive, without a pass or fail).

Known contamination. The pretraining text of Qwen3 and Gemma 4 is not published and may include PubMed Central articles (EI's articles come from it), EI's repository (public since 2019), PubMedQA and the UCI readmission data; this would favor both models. EI's test labels are public, so no evaluation of EI can be blind.

Files attached, with their SHA-256 at commit 77087bd:
- docs/prompt_session8.md: 64d50830d90960e01f89a270ff67f5355373b7214dfe57ecaaa86422db125ac3
- src/eval/metrics.py: 9a75d2a14d3dda024dbf3f78a6fcc54450f88c0eb41878bb14d80f31e667b507
- src/eval/compare_session8.py: 688ced2f527f528f91905903cb336ac43ac8c63945e95bb110db4143ca5d9125
- src/eval/calibrate_session8.py: 6d5fb5c9fe3f6903cf37d74ca9865a030539befb836e578adc652eb62a76282e
- src/eval/calibrate_session7.py: 6efee6afcbdbe0d18d4ee35af24615213958fc382b766fab5c1c3d550044e117
- src/eval/calibration_family_cv.py: 295e5f60a2f3ccdf10bec332ee2a7ca2c24bae3be9d2939a98932160398c122f
- src/eval/ece_noise_floor.py: a99aae2c17b5a959bc9a04ea08be0dc7b9399b480025ce6c54df67e49841e37e
- src/eval/large_model_session8.py: 371ea5b2072a4c186585d13470c55aef0c5f7ea6569c169da7f30587d38f2327
- src/tasks/evidence_inference.py: 5abc1c80185e491f9da823521f37753ce48f87671cd4bfbcf0ac0d2f956ccd91
- src/eval/power_session8.py: 95d2852e6d64b21846e875905e5408633af498d995ab8334b0ceb04eafce44b8
- src/eval/handoff_session8.py: c880e07bc481ab6ab79b0892a068900549733001cf46e0f298b222191bb07d85
- results/fiorillo_v0_4/power.json: c4786c96885124d9452ebe5df7196d1d26216afd9aea7baac6fef84d26b1c0d6
- results/fiorillo_v0_4/fixed_versions.json: c13c78221e4bb4d73b3f85b08c0e8a4131b264ff0e2e21e584a770a66338fba3
- results/fiorillo_v0_4/records_manifest.json: c7207d3678b51a47a769dc3b36fc6b3aa57277141da8c0c3fa3fc06f4f4506f9

---

## Filing notes (not part of the registration)

1. Template: OSF Preregistration, as for Study D. The checker found a July 2026 text copy of OSF's form with different sections ("Foreknowledge of data or evidence" and others). Check the live form first. If it differs from these fields, map each section by meaning, and tell me if one has no home.
2. Timing: the plan now holds the test predictions until you tell me it is filed (correction 13). The freeze should come around 7 to 7:30 pm Eastern. The latest safe time is set by correction 6's schedule, and I will ask you before it matters.
3. Public or embargoed: an embargo (up to four years) keeps it private but timestamped. Filing it public now counts as disclosure for any later patent.
4. Attach the 14 files listed above, from the Mac folder at commit 77087bd; their SHA-256 values are in the form so anyone can check them. The plan names file paths, the rented GPU and costs, and no secret; the pod's password appears only as a file name.
5. Existing data: I chose "following analysis of the data". Building the records computed summary statistics of the test split, its answer counts among them, and OSF's definition of "prior to analysis" excludes those. The explanation field says exactly what was computed, and that no model has predicted on the test split.
6. Before you submit, read each field against this file once; in Study D two fields went in from an older draft.
7. If you want a conflicts line in "Other": "I may later build a company on this model."

---

## As filed

I registered this at 8:36:37 pm Eastern on 30 September 2026 (00:36:37 UTC on 1 October): https://osf.io/kaxmn/, OSF Registries, OSF Preregistration template, CC-BY 4.0, embargoed until 1 October 2027. Claude filled the form in my Chrome and submitted it at my request.

OSF's OSF Preregistration form had changed since Study D, so each field above went into the field of the same meaning:

| Field in this draft | Field on OSF's form |
|---|---|
| Hypotheses | Research questions or hypotheses |
| Existing data ("Registration following analysis of the data") | Foreknowledge of data or evidence: "Authors have observed the data, but have not performed the proposed analyses." |
| Explanation of existing data | Explanation of foreknowledge and managing unintended influences |
| Study type ("Other: ...") | Study type: Other; its sentence opens Study design as "Study type (Other): ..." |
| (none) | Intention for causal interpretation: "No causal relationship inferred" |
| Blinding | Blinding of experimental treatments: "No blinding is involved." |
| Is there any additional blinding in this study? | Additional blinding during research or analysis |
| Stopping rule | Starting and stopping rules |
| Data exclusion | Data inclusion and exclusion |
| Exploratory analysis | Other planned analysis |
| Other | Context and additional information, with "Conflicts of interest. I may later build a company on this model." before the file list |

Every other field kept its name. In the Description and in the additional blinding text, "under Data exclusion" became "under Data inclusion and exclusion" and "under Explanation of existing data" became "under Explanation of foreknowledge and managing unintended influences"; no other word changed.

The 14 files went into the form's upload fields, at most five a question: Study design (the plan, fixed_versions.json, large_model_session8.py, calibrate_session8.py, calibrate_session7.py), Data collection procedures (evidence_inference.py, records_manifest.json), Indices (metrics.py, ece_noise_floor.py, calibration_family_cv.py) and Statistical models (compare_session8.py, power_session8.py, power.json, handoff_session8.py). OSF's SHA-256 for each matches the list above.

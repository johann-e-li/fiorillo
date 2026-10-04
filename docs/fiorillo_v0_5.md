# Fiorillo v0.5

Session 9's report (docs/prompt_session9.md). Step 0 wrote the Licenses section on 1 October 2026; step 16 wrote the rest on 2 October 2026.

## Licenses (read 1 October 2026)

- EI's annotations: the MIT License, as read in session 8 (docs/fiorillo_v0_4.md, Licenses).
- EI's article texts. Each article in PubMed Central's open-access subset keeps its own license, and PubMed Central says users are responsible for following it (docs/fiorillo_v0_4.md, Licenses). Europe PMC's records for EI's 3,348 articles (read 30 September 2026; results/fiorillo_v0_4/article_licenses.csv) gave CC BY for 1,807, CC0 for 28 and no license for 197. The license recheck of 1 October 2026, which Johann asked for (src/tools/ei_license_recheck.py, audited by a second agent; results/fiorillo_v0_5/license_recheck.csv and license_recheck.json), read each article's own statement as PubMed Central serves it through NCBI's E-utilities, from 06:30:15 to 06:36:09 UTC: its license links, its license type, its license text and its copyright statements. By that statement the 3,348 articles are: CC BY 1,747; CC BY-NC 585; CC BY-NC-SA 359; CC BY-NC-ND 301; a copyright notice only 92; other non-commercial or no-modification terms in words 79; an open link or statement with restricting words in its text 41; CC BY-NC, BY-ND or BY-SA in words 31; permission for verbatim copying only 28; CC0 19; the publisher's own reuse terms 16; CC BY in words 15; no statement 12; the public domain 9; CC BY-ND 6; conflicting license links 5; CC BY-SA 2; a general statement only 1.
- The clean rule (docs/prompt_session9.md, DECISIONS 3): an article is clean when its own statement is CC BY (any version, by link or in words), CC0 or the public domain for the article itself, and no part of the statement restricts commercial use, derivatives or the license of what is made from it; where the parts disagree, the more restrictive part decides; a publisher's own terms (iOpenAccess) are not clean even beside Creative Commons words; and CC BY-SA is not clean, because Creative Commons' conservative guidance, as a research agent read it, is that a model trained on ShareAlike works should carry the same license. Clean: 1,790 of the 3,348 articles (6,546 of 12,501 prompts); in the training split 1,431 of 2,657 articles (5,279 of 10,046 prompts), in validation 179 of 340 (654 of 1,237) and in test 170 of 333 (613 of 1,218).
- Where Europe PMC's records and the articles' own statements disagree: in the training split Europe PMC lists as CC BY 47 articles that are not clean by their own statements (33 with restricting words beside an open link or statement, 2 with conflicting license links, 4 with a copyright notice only, 4 with no statement, 1 with CC BY-NC, BY-ND or BY-SA in words, 1 with other non-commercial or no-modification terms, 1 with the publisher's own terms and 1 allowing verbatim copying only), and lists with no license 12 that are CC BY by their own statements (7 by link and 5 in words); in validation the counts are 7 and 1, and in test 5 and 1.
- Why v0.5 trains only on clean articles. Whether weights trained on non-commercial or no-derivatives texts may be published is UNVERIFIABLE: no source read addresses it (docs/fiorillo_v0_4.md, Licenses). v0.5 is the version meant to be made public if it meets v0.4's bar (DECISIONS 1), so its EI specialist learns from, is chosen on and is calibrated on clean articles only (DECISIONS 2 and 3); s8_ei, trained on all of them, stays private. The one exception is the passage selection's word statistics (data/processed/session8/bm25_stats.json), counted over the passages of all 2,657 training articles; they hold single words and counts but no sentence of any article, and they are kept so that every system's input is cut by the same rule (DECISIONS 3).
- Qwen3-4B-Base at revision 906bfd4b4dc7f14ee4320094d8b41684abff8539, the revision s8_ei used and the repository's current one when read (2026-10-01T07:56:06Z): the card's license is apache-2.0, and the repository holds the Apache License 2.0's text (LICENSE); not gated. It allows use, modification and redistribution, commercial use included, with the license and notices kept.
- tabicl 2.2.0, the newest release on the Python Package Index when read (uploaded 2 September 2026): the LICENSE in its wheel is the BSD 3-Clause License ("Copyright (c) 2025, Soda team @ Inria"), which allows use, modification and redistribution, commercial use included, with the notice kept and without using the holders' names to endorse; the code under src/tabicl/forecast, derived from TabPFN-TS, is under the Apache License 2.0 and is not used here. Its checkpoint, tabicl-classifier-v2-20260212.ckpt (the one it loads by default), comes from Hugging Face's jingang/TabICL at revision 4dcd344ece2c00be9e831fdd35bed57b5ad83e19, whose card's license is bsd-3-clause with no other terms; not gated. Both allow commercial use, so tabicl is built (step 0 (c)).
- BioGPT-Large-PubMedQA at revision 39c04fb31dfc8094b0f04c7bb7973cf612606c0c: its card's license is MIT (the repository holds no license file), and microsoft/BioGPT's code is under the MIT License (GitHub's record of the repository). Here the model is only run on PubMedQA, for research, and its measured results are published beside Fiorillo's; it is not fine-tuned or redistributed, and its outputs train nothing.
- sacremoses 0.2.0, which BioGPT's tokenizer needs: the MIT License (its classifier on the Python Package Index).
- TCGA-Reports, for THE NEXT QUESTION'S RECORDS (Mendeley Data, dataset hyg5xkznpx, version 1, doi 10.17632/hyg5xkznpx.1, published 3 January 2024; read through Mendeley Data's public interface at 07:57 UTC): the Creative Commons Attribution 4.0 International license. As Mendeley states it, the data may be shared, copied and changed with credit, a link to the license and a note of any changes, without implying endorsement, and content that belongs to a third party may need further permission. The reports were first published by TCGA as PDF files (the dataset's description). The register's UNVERIFIED for this license (I-38) is resolved.
- The GDC's labels: the GDC's page on data access says that open-access data include most clinical data elements and that no login is needed. Its page on data access policies, under the NIH Genomic Data Sharing Policy, asks users to acknowledge in their publications the datasets and the NIH-designated repository through which they reached the data, and not to attempt to identify individual participants. Session 9 fetches open-access fields only, attempts no identification, and its documents and the preprint name TCGA and the GDC as the sources.
- Gemma 4 31B-it and vLLM 0.30.0: as read in session 8 (docs/fiorillo_v0_4.md, Licenses), at the same revision and version.
- PubMedQA: as read in session 7 (docs/fiorillo_v0_3.md, Licenses).

## What session 9 built

Fiorillo v0.5 is the license-clean version of v0.4's routed system of specialists (I-25). Each typed question goes to the specialist chosen for it, and the system is judged as one, question by question. PubMedQA goes to s7_b3 with its session 7 calibration, unchanged. Readmission goes to the pool that the readmission rule chooses on validation (I-28 continued). EI's question goes to a new specialist, trained only on clean EI articles (I-27), with a recipe chosen on validation among four candidates: swap augmentation (I-37), a 6,144-token input (I-30) and two learning rates for the adapters. A weight average of its runs (I-39) and the average of each answer with the answer to the exchanged prompt (correction 3) were each kept only if validation favored them. v0.5 hands nothing off to a larger model. Before any training, the session asked s8_ei, v0.4's EI specialist, EI's test prompts with no article, with the title alone and with the intervention and the comparator exchanged (I-32 and I-37, the reading checks). It also ran the hand-off analyses (I-34 and I-35), wider calibration families (I-36), BioGPT-Large-PubMedQA beside s7_b3, and prepared, without training, the records of the next question, TCGA's pathology reports (I-38; docs/tcga_prep_session9.md). The bar is v0.4's four criteria on EI's test split, with v0.5's EI specialist in s8_ei's place, and I filed the analysis as an update to the OSF registration before any test prediction of a v0.5 system existed.

The outcome, in brief (the Results give every figure with its interval). v0.5's bar is met: every criterion passes (Table 9), so THE RELEASE follows (step 17). v0.5's EI specialist is s9_ei_wavg, the weight average of the long recipe's three runs (6,144-token inputs, no swap averaging, the temperature family). On EI's test split it is less accurate than v0.4's s8_ei, which learned from all 2,657 training articles where it learned from the 1,431 clean ones: 1.40 points of accuracy lower (0.25 to 2.45), while no difference in log loss was detected (Table 10). The long recipe led base by 0.0286 in log loss on the clean validation prompts, but on test s9_ei_wavg and s9_b do not differ (Table 12). THE READING CHECKS read the same for v0.5 as for v0.4 (Table 14). TabICL was NOT BUILT, by Claude's mistake in its version check, and the new readmission pool fell short of its margin, so v0.5 keeps session 8's pool (Tables 17 to 19). The winning recipe's third run, s9_long_3, was left without a calibration of its own, also Claude's mistake (Table 4's note); it changes no frozen system.

## Method

The method was fixed in docs/prompt_session9.md before any training. Corrections 1 to 8 were written there before the work each one changed. The OSF update (docs/osf_update_session9.md; filed as an update to https://osf.io/kaxmn/, OSF revision 6abe7a9d0c9675529d443184; the clock read 15:31:21 UTC on 1 October 2026 just after) states the method as corrections 1 to 8 left it, from the plan at commit af59af5 (SHA-256 02b9142c...). The corrections are listed at the end of this section, and how each stage ran is in the next section.

### The license-clean records

- Clean: an article whose own license statement, as PubMed Central serves it, is CC BY (any version, by link or in words), CC0 or the public domain, with no part of the statement restricting commercial use, derivatives or the license of what is made from it; where the parts disagree, the more restrictive part decides; a publisher's own terms are not clean even beside Creative Commons words; CC BY-SA is not clean (Licenses, above; license_recheck.csv's clean column).
- The clean training records: session 8's training records whose article is clean, in the same order, byte for byte: 5,279 of the 10,046 training prompts, in 1,431 of the 2,657 training articles.
- The clean validation records: 654 of the 1,237 validation prompts, in 179 of the 340 validation articles. They are the only validation prompts that any choice for v0.5 reads.
- The monitoring subset: 200 clean validation prompts in 113 articles, drawn with random.Random(20260927).sample over the clean validation records sorted by ID as text. Only the training runs' logged validation reads it, never anything that chooses.
- The test records: session 8's, all 1,218 prompts in 333 articles, unchanged; v0.5's bar is judged on all of them (the plan's decision 2).
- The exception: the passage selection's word statistics (data/processed/session8/bm25_stats.json), counted over the passages of all 2,657 training articles. They hold single words and counts but no sentence of any article, and they are kept so that every input is cut by the same rule.

### The recipes

Every run is src/train/train_qwen_head.py train from s7_b3's folder (--init, --keep last, --freeze-steps 0): it starts from s7_b3's adapters, projection and head, and the last weights are kept. Shared settings, as s8_ei's in session 8: Qwen3-4B-Base at revision 906bfd4b4dc7f14ee4320094d8b41684abff8539; LoRA rank 16, alpha 32, dropout 0.05; AdamW, 1e-4 for the projection and head; a cosine schedule to 1e-6; gradient clipping at 1.0; weight decay 0.01; bfloat16; 32 prompts an optimizer step; the monitoring subset's validation every 80 steps, logged and never used to choose. After training, each run predicts the whole validation split (1,237 prompts); only the 654 clean prompts are scored.

- base (b): 2 epochs on the clean training records, 4,096 tokens, the adapters' learning rate 5e-5. Three runs, s9_b_1, s9_b_2 and s9_b_3 (seeds 20260927, 20260928 and 20260929); their probability average is s9_b.
- swap: the clean training records plus a copy of each with the intervention and the comparator exchanged and "significantly increased" and "significantly decreased" exchanged in the label ("no significant difference" kept); 10,557 records (one prompt names the same text twice and gets no copy); one epoch, so a run takes about as many optimizer steps as base.
- long: inputs selected by session 8's rule at 6,144 tokens (16 to spare; BM25's statistics unchanged), for training, monitoring, validation and test; the marked evidence is kept for 98.68 percent of the clean training prompts. A pilot chooses the memory setting.
- lr1 and lr2: the adapters' learning rate 1e-4 and 2e-4.
- Two runs each (correction 2): swap, long, lr1 and lr2 with seeds 20260927 and 20260928, every first run before any second run.

### The recipe rule

Applied by the train stage on the pod by itself; nothing in it reads a test file.
- A run's score: from its raw validation predictions on the 654 clean prompts, the out-of-fold log loss of session 8's calibration rule (the lower of the temperature and temperature_intercepts families'; 5 folds, 20 repeats, seed 20260927). An average's score is the same on the averaged probabilities.
- The margin m: the largest minus the smallest of s9_b_1's, s9_b_2's and s9_b_3's scores, and at least 0.005.
- A candidate passes if the mean of its two runs' scores is below the mean of s9_b_1's and s9_b_2's by more than m, and each of its runs scores below the base run of the same seed. lr1 and lr2 count as one change; if both pass, the lower mean is kept.
- The combination: if two or more of swap, long and the learning rate pass, one more recipe uses every passing change, in two runs (s9_combo_1 and s9_combo_2). It passes if its mean score is below the best passing candidate's by more than m and each run scores below that candidate's run of the same seed.
- The winning recipe: the combination if it passed; else the passing candidate with the lowest score; else base. A winning recipe other than base is trained once more, with seed 20260929; s9_ei is the probability average of its three runs.
- The weight average, s9_ei_wavg (I-39): each layer's three LoRA updates averaged exactly, by joining the A matrices along the rank and dividing the B matrices by 3 (rank 48, alpha 96), with the projection's and head's weights averaged elementwise. v0.5 uses it in place of s9_ei if its score is not above s9_ei's by more than m. With two runs (correction 7 (b)), rank 32, alpha 64 and B divided by 2.
- Swap averaging (correction 3): each prompt is also answered with its intervention and comparator exchanged; in that answer the probabilities of the two directions are exchanged, and the two answers' raw probabilities are averaged before calibration. v0.5 uses it if the swap-averaged score is below the plain one by more than m.
- One run left (correction 8 (b)): a candidate passes if that run's score is below s9_b_1's by more than m; a combination left with one run, or whose best candidate has one, passes if its first run's score is below the best candidate's first run's by more than m.
- recipe_selection.json records every score, m, every decision with its numbers and the time of each.

### The pilot and the cost rule

- pilot_long: train_qwen_head.py pilot-s8 on the long items, alone on the GPU, settings a (micro-batch 1, accumulation 32, no gradient checkpointing) and b (micro-batch 4, accumulation 8, gradient checkpointing), 30 optimizer steps each, timed over steps 11 to 30; a setting passes if it does not run out of memory and its peak is at most 76 GiB (no room held for another job); the long runs use the passing setting with the higher rate.
- The cost rule (corrections 1 and 8 (c)): before its first run, the train stage projects the session's total as the spend so far plus the longest path still possible (every run still ahead at its pilot's rate scaled by session 8's ratio of run rate to pilot rate, 2.70 against 2.86; Gemma 4 31B-it on the longer input whenever long's pilot passed; and fixed hours for the remaining stages). Above $115 it stops and asks me, with a fixed order of what can be dropped; the cap is $125.

### The readmission pool, version 2

- Recoded diagnosis fields added to session 8's frames for diag_1, diag_2 and diag_3: the three-character ICD-9-CM code, its chapter (20 values), 20 chapter indicators and an out-of-fold target encoding of each field's code (9 columns), 35 columns in all.
- xgboost_codes and catboost_codes: session 8's chosen settings, refitted on all training records with the recoded fields, with session 8's numbers of trees.
- lightgbm_tuned's refit from session 7's chosen setting, kept only if its validation predictions match session 7's within 0.001.
- tabicl: TabICLv2 (tabicl 2.2.0, checkpoint jingang/TabICL at 4dcd344e; BSD-3-Clause) with its defaults and random_state 20260927, on the GPU in its own environment.
- The new pool: session 8's method (the softmax of the members' weighted log-probabilities divided by T; weights on a grid of steps of 0.1 summing to 1; T by maximum likelihood) over session 8's five members and the new ones. v0.5 uses it if its out-of-fold log loss (5 folds, 20 repeats) is lower than session 8's pool's by at least 0.002; otherwise it keeps session 8's pool (readmission_pool_v2.json).

### PubMedQA beside BioGPT-Large (descriptive)

BioGPT-Large-PubMedQA (microsoft/BioGPT-Large-PubMedQA at revision 39c04fb3, MIT) reads each item as it was fine-tuned (the question and the contexts without the long answer, cut at 384 tokens; its nine learned prompt tokens; then "the answer to the question given the context is"), and each answer's probability is the next token's probability of yes, no or maybe, renormalized. Its fine-tuning covered all 150 of the project's PubMedQA validation items, so its primary result is its uncalibrated probabilities; a calibration on those items is secondary (K5). Before scoring, the stage reads the checkpoint's own arguments (correction 5).

### The large model

Gemma 4 31B-it as session 8 ran and calibrated it (gemma4_31b) is criterion 3's reference. If the winning recipe reads 6,144 tokens, Gemma 4 31B-it answers the 6,144-token inputs as gemma4_31b_long, with session 8's code and corrections unchanged, calibrated by session 8's rule on all 1,237 validation prompts, and criterion 3, K1 and K7 use it, so that both models read the same text. Its outputs never train, tune or select a Fiorillo model.

### Calibration

- v0.5's EI specialist (as the rules above leave it) is calibrated on the 654 clean validation prompts by the family with the lowest out-of-fold log loss among four (I-36): temperature; temperature_intercepts; vector (a scale and an intercept for each answer); and dirichlet_odir (a 3 by 3 matrix on the log-probabilities with off-diagonal and intercept penalties chosen by inner cross-validation). An exact tie goes to the family listed first.
- Every other system keeps session 8's rule (temperature or temperature_intercepts): s9_b and each run on the 654 clean prompts; s8_ei, gemma4_31b and s7_b3 as before; gemma4_31b_long on all 1,237 EI validation prompts; BioGPT-Large-PubMedQA on PubMedQA's 150 calibrate items (secondary).
- The ECE limit for criterion 1: the larger of 0.05 and the 95th percentile of the ECE that a perfectly calibrated model shows by chance on 1,218 prompts (src/eval/ece_noise_floor.py, unchanged; 5,000 simulated test sets from v0.5's specialist's calibrated clean validation predictions).

### The freeze and the registration

The freeze writes release_candidate.json before any test prediction of a v0.5 system exists: v0.5's three specialists with their calibrations, the reference's, BioGPT-Large's, the two frozen hand-off rules, the ECE limit, the SHA-256 of every file they come from, the OSF update's link and filing time, and its own time. It refuses to run until the OSF update is filed (osf_filed.txt on the pod). The test-prediction stage writes testpreds_log.json (that file's SHA-256 and its own start time) before any test prediction, and the unpacking on my Mac stops unless testpreds_log.json names the frozen file and a later start time.

### The reading checks (I-32 and I-37)

- Inputs, from the test records: none (the article set to empty text; 1,218 prompts); title (the article set to its title, by correction 7 (e)'s rule; 1,209 prompts, 9 with no title left out); swap (the intervention and the comparator exchanged, the expected answer exchanged for the two directions and kept for no difference; the article as in the system's own input; 1,218 prompts).
- Systems: s8_ei (its three runs' raw probabilities averaged, then session 8's calibration), on the pod before any training; v0.5's EI specialist with its own calibration, after the freeze (single pass for none and title if swap averaging is kept).
- Reported, each with a 95 percent interval from 1,000 article resamples: accuracy and macro-F1 against the original answers (none, title) or the exchanged ones (swap); for none and title, agreement with the system's answer on its full input; for swap, the share of direction answers that exchange and of no-difference answers that stay, overall and for comparators that name a time point or a baseline; and title minus none on the 1,209 prompts in both inputs (correction 8 (g)).
- Readings, fixed before the run: none, macro-F1 at or below 0.60 "its answers depend on the article's text", above 0.75 "the prompt alone carries more than a classifier trained on prompts alone (0.511 macro-F1, DeYoung et al. 2020, Table 1); ...", otherwise "partly"; title, the difference's interval wholly above 0.05 "the title adds information about the result: a title can state the result, or the model may recall the trial, and these checks cannot tell them apart", wholly below "no added knowledge of 0.05 or more was detected", otherwise "not settled"; swap, accuracy at or above 0.85 "it follows the comparison as written", below 0.70 "it does not follow the exchange (recall of the trial or a bias toward one direction)", otherwise "partly". If swap averaging is kept, v0.5's swap check is consistent by construction and is not read as a reading check.
- The stop: if s8_ei's macro-F1 with no article is above 0.85, the chain stops before training and asks me.

### Hand-off (secondary, descriptive; I-34 and I-35)

Pairs: EI with s8_ei and Gemma 4 31B-it; EI with v0.5's specialist and Gemma 4 31B-it (gemma4_31b_long under the longer input); PubMedQA with s7_b3 and Gemma 4 31B-it; on validation before the freeze and on test after it. Reported: the risk-coverage curve with AURC, E-AURC and AUGRC; deferral curves for the confidence rule, random hand-off, the oracle and the large model's own confidence; intervals at 10, 20, 30 and 50 percent from 1,000 resamples (EI by article, PubMedQA by item); and, for EI, a logistic regression (L2, C 1.0) fitted and frozen on validation to pick prompts where the specialist is wrong and the large model right, compared on test with the confidence rule at the same rates (corrections 7 (b), 8 (a) and 8 (f) for two runs and swap averaging). None of this is part of v0.5, which hands nothing off.

### The release bar

v0.4's four criteria on EI's test split, over all 1,218 prompts, with v0.5's EI specialist (with its calibration) in s8_ei's place: (1) its ECE is at most the ECE limit; (2) its log loss is below the prior's (v0.4's prior, the answer shares of the whole training split), with the 95 percent interval of the difference wholly below zero; (3) its log loss is below the large model's (gemma4_31b, or gemma4_31b_long under the longer input), with the interval wholly below zero; (4) its macro-F1 is not below the large model's point estimate. Criteria 2 and 3 use 10,000 article resamples. If every criterion passes, v0.5 is released publicly, and I end the OSF embargo the same day; if any fails, "v0.5 release bar not met: <the reason>" is recorded and nothing is released. Power (power.json): the chance of passing criterion 3 is 0.80 at a real gain of 0.0718 in log loss, and above 0.98 if v0.5's specialist is worse than s8_ei by up to 0.1.

### Comparisons

Session 8's metrics, resampling and paired differences; every number has a 95 percent interval from 1,000 resamples (EI by article, the other questions by item), except criteria 2 and 3 (10,000), the costs and K9's out-of-fold table (none), and K4's and K7's numbers (their own files' intervals) (correction 8 (e)). A difference whose interval includes zero is reported as "no difference was detected", with the estimate, the interval and what it excludes. PubMedQA's and readmission's test splits were read in earlier sessions, so K5 and K6 are descriptive.
- K1 (the bar): v0.5's EI specialist minus the prior and minus the large model, on every EI metric; criteria 1 to 4.
- K2 (what the license cost; descriptive): s9_b minus s8_ei, and v0.5's specialist minus s8_ei.
- K3 (the recipes): each candidate's two-run average, and the combination's, minus the average of s9_b_1 and s9_b_2 on the clean validation prompts (out-of-fold calibrated probabilities, 1,000 article resamples), beside recipe_selection.json; on test, v0.5's specialist minus s9_b.
- K4: the reading checks. K5 (descriptive): s7_b3 minus BioGPT-Large-PubMedQA, and BioGPT-Large-PubMedQA minus Gemma 4 31B-it. K6 (descriptive): v0.5's pool minus session 8's pool, and each new member minus its session 8 counterpart. K7: the hand-off analyses. K8 (descriptive): s9_ei_wavg minus s9_ei. K9 (descriptive): the four calibration families' out-of-fold table, and v0.5's specialist under its family minus under session 8's rule. K10 (descriptive): with swap averaging minus without.
- Also: a sensitivity analysis without the test prompts EI's README flags; the rows whose true-answer probability prints as 0; calibration slope and intercept; the cost per 1,000 questions (correction 8 (d)); and the published EI results of v0.4's G3 section beside v0.5's specialist and s8_ei.

### Corrections

1. (07:42 UTC, before step 0) My advance approval of the pod's deployment, the copy and the shutdown; the budget raised to a $125 cap with the cost rule's stop at $115, given a balance of at least $140.
2. (07:42 UTC) Two runs for each candidate, and the rule for two runs.
3. (07:42 UTC) Swap averaging, decided on validation.
4. (07:42 UTC) No larger base model in this session.
5. (08:12 UTC, during step 0) BioGPT-Large's training arguments read from its checkpoint on the pod, with fixed fallbacks.
6. (08:21 UTC, during step 2) The title input's rule, before any prediction.
7. (11:37 UTC, during steps 3 and 4) Three rules of the license recheck fixed (no article's class changed); the weight average with two runs; the pod's rebuild keeps a copy of the uploaded selection; a usage text; the title rule as built; and what the model card says about PubMedQA (kept, by my choice).
8. (14:12 UTC, during step 5, amended at 14:53 UTC) The two-run path end to end; one-run candidates and combinations; the cost projection as computed; the cost per 1,000 questions; the intervals' exceptions; the run spread under swap averaging; and the title difference on the 1,209 prompts in both inputs.

## How each stage was run

Everything on the GPU ran on one Runpod pod (gn7mjxvzwz4xne: one H100 SXM GPU with 80 GB, Secure Cloud US-NE-1, On-Demand; 26 virtual processors and 251 GB of memory; the image runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404; a 200 GB container disk and no volume; $3.52 an hour, $3.49 for compute and $0.028 for the disk; pod_console.json), driven by scripts/pod_session9.sh from the linked shell through the pod's Jupyter server. By the console's audit log (pod_console.json), I created it at 15:55:18 UTC on 1 October and stopped it at 15:55:27 UTC, Claude started it at 16:00:24 UTC, and its Jupyter server started at 16:30:55 UTC. It had been made with a 150 GB disk, and Claude set the disk to the plan's 200 GB before starting it (docs/log.md). The pod then sat idle until about 18:55 UTC, because Claude stopped checking it after 16:15 UTC (2 hours 24 minutes, about $8.45, derived), and again from 19:09:56 to 19:19:31 UTC (about $0.56, derived), because Claude did not start the next stage at once, and from the chain's end at 15:18:29 UTC on 2 October until the copy began at 15:23:53 UTC (about $0.32, derived); all three were Claude's mistakes (docs/log.md). Claude stopped the pod at 15:29:23 UTC on 2 October and deleted it at 15:29:57 UTC (pod_console.json). It ran 23.49 hours from its start, about $82.67 at $3.52 an hour (derived). The Runpod balance was $140 before the pod and $59.07 at 15:32 UTC on 2 October (pod_console.json), and the console notes that its billing runs about an hour behind. Times below are UTC, and every stage's log is in results/fiorillo_v0_5/logs/pod/.

### Table 1. The stages on the pod (UTC; 1 October, then 2 October from s9_swap_1's finish on)

| Stage | Started | Finished | What it did |
|---|---|---|---|
| upload | about 18:55 | 19:05:30 | 174 files placed and checked by SHA-256; pod_rate.txt and osf_filed.txt written (docs/log.md) |
| prepare | 19:05:39 | 19:09:56 | installs; the fixed revisions; every record file rebuilt and checked; the items for every recipe and for the reading checks (5,279 clean, 10,557 swap and 5,279 long training items; 200 monitoring items), none cut or skipped |
| checks_s8 | 19:19:32 | 19:33:25 | the reading checks on s8_ei, alone on the GPU, before any training (checks_s8.json; checks_s8.log) |
| pilot_long | 19:33:25 | 20:05:41 | alone on the GPU: settings a and b on the long items (pilot_long.json; pilot_long.log) |
| trees, on the pod's processors | 20:05:41 | 20:09:26 | the recoded fields; xgboost_codes' and catboost_codes' refits; lightgbm_tuned's refit and its check |
| train: the cost rule | 20:05:41 | 20:05:41 | cost_rule.json |
| train, s9_b_1 | 20:05:42 | 21:16:54 | trained alone on the GPU; validation predictions |
| train, s9_b_2 | 21:16:55 | 22:27:35 | trained alone on the GPU; validation predictions |
| train, s9_b_3 | 22:27:35 | 23:38:47 | trained alone on the GPU; validation predictions |
| train, s9_swap_1 | 23:38:50 | 00:49:20 (2 October) | trained alone on the GPU; validation predictions |
| train, s9_long_1 | 00:49:22 | 02:35:45 | trained alone on the GPU at 6,144 tokens; validation predictions (train.log) |
| train, s9_lr1_1 | 02:35:48 | 03:46:29 | trained alone on the GPU; validation predictions |
| train, s9_lr2_1 | 03:46:32 | 04:57:17 | trained alone on the GPU; validation predictions |
| train, s9_swap_2 | 04:57:20 | 06:07:41 | trained alone on the GPU; validation predictions |
| train, s9_long_2 | 06:07:43 | 07:54:39 | trained alone on the GPU at 6,144 tokens; validation predictions (train.log) |
| train, s9_lr1_2 | 07:54:41 | 09:05:45 | trained alone on the GPU; validation predictions |
| train, s9_lr2_2 | 09:05:48 | 10:16:42 | trained alone on the GPU; validation predictions |
| train: THE RECIPE RULE | 10:16:42 | 10:16:48 | the candidates' scores and passes; long the winner (recipe_selection.json; train.log) |
| train, s9_long_3 | 10:16:48 | 12:03:46 | trained alone on the GPU at 6,144 tokens; validation predictions (train.log) |
| train: the weight average | 12:03:53 | 12:10:28 | s9_ei_wavg built from s9_long_1 to s9_long_3 (weight_average.json; train.log); its validation predictions from 12:04:01 |
| train: the exchanged records' validation predictions | 12:10:35 | 12:17:02 | s9_ei_wavg_exch: the 1,237 exchanged validation prompts at 6,144 tokens (correction 3; train.log) |
| train: THE RECIPE RULE done | 12:17:02 | 12:17:10 | swap averaging's decision; v0.5's EI specialist, s9_ei_wavg (recipe_selection.json; train.log) |
| tabicl | 12:17:10 | 12:17:20 | NOT BUILT: its version check read tabicl 2.2.0's version as None (Claude's mistake; not_built/tabicl.txt) |
| prepare_large | 12:17:20 | 12:25:13 | vLLM 0.30.0's CUDA 13.0 build (the driver's CUDA version is 13.2) and its checks; Gemma 4 31B-it downloaded and its weights checked (62,546,338,248 bytes) |
| smoke_large | 12:25:13 | 12:30:42 | session 8's smoke test with the E2B model and the 31B's chat template: four processes, each passing and leaving the GPU empty |
| large_validation | 12:30:42 | 13:28:21 | Gemma 4 31B-it on the 1,237 EI validation prompts at 6,144 tokens, alone on the GPU; its files copied to the name gemma4_31b_long |
| biogpt | 13:28:21 | 13:39:25 | BioGPT-Large-PubMedQA downloaded at the fixed revision; the checkpoint's arguments not read (its archive's host does not resolve), so fixed_versions.json's settings (correction 5); PubMedQA's 150 calibrate validation items (biogpt.log) |
| calibrate | 13:39:25 | 13:40:31 | calibration_s9.json; the ECE limit; readmission_pool_v2.json; the validation hand-off analyses and handoff_rule_v05.json; power_validation.json (calibrate.log) |
| freeze | 13:40:31 | 13:40:31 | release_candidate.json |
| testpreds | 13:40:31 | 15:18:19 | testpreds_log.json first; then every test prediction, one job at a time on the GPU (Table 1b; testpreds.log) |
| package | 15:18:19 | 15:18:29 | the four zips and SHA256SUMS.txt; the package check passed (207 files in the results zip; docs/log.md, since the zipped stage logs were written before the stage's last lines) |
| the copy to the Mac (step 13) | 15:23:53 | 15:27:24 | 3,742,122,608 bytes (3,568.8 MiB) in two rounds of byte ranges; every size and SHA-256 matched; unzip -t found no errors (docs/log.md) |
| the pod stopped, then deleted | 15:29:23 | 15:29:57 | in the Runpod console (its audit log's times; pod_console.json) |
| unpack (step 14, on the linked shell) | 15:33:56 | 15:37:05 | the guard passed (the test predictions' log names the frozen file's SHA-256 and a later start); 328 files extracted and checked by CRC-32; 230 copied into results/fiorillo_v0_5/ (4 already there with the same content); 94 model files; no results file over 1,000,000 bytes (docs/log.md) |

The reading checks on s8_ei ran each of its three runs on the 1,218 prompts with no article (about 13 forward seconds a run), the 1,209 with the title alone (13 seconds) and the 1,218 swapped prompts (221 seconds) (checks_s8.log). At each load of a run's tokenizer, transformers warned of "an incorrect regex pattern", the check written for Mistral's tokenizers that sessions 7 and 8 met; s8_ei's tokenizer files are byte-identical to Qwen3-4B-Base's (docs/log.md, session 8).

The pilot: setting a (micro-batch 1, `grad_accum` 32, no gradient checkpointing) trained 1.90 items a second on the long items with a peak of 50.3 GiB, and setting b (micro-batch 4, `grad_accum` 8, gradient checkpointing) 0.70 items a second with a peak of 19.0 GiB; both passed, and the long runs use setting a (pilot_long.json; its validation pass ran at 3.05 items a second).

The cost rule (cost_rule.json, written 20:05:41 UTC): spent so far $14.69 (4.17 hours from the pod's creation, which counts the minutes before Claude's start); the projection's rates, the pilots' times 2.70/2.86, 2.70 items a second for the base-length recipes and 1.79 for long; 1.15 hours for each base-length run and 1.74 for each long, combination or third run; Gemma 4 31B-it on the longer input among the fixed hours (2.5 hours, counted because long's pilot passed); remaining 24.74 hours; projected total $101.76, under the $115 stop, so nothing stopped and nothing was dropped; the copy and its checks projected for 20:49:51 UTC on 2 October, with no maintenance announced before then. The fixed hours were 5.67 in all (derived: the sum of cost_rule.json's fixed_hours).

The trees stage: the recoded fields added 35 columns to session 8's 44. xgboost_codes (333 trees, 9.9 seconds) had a validation log loss of 0.84230 and catboost_codes (1,222 trees, 181.1 seconds) 0.83800. lightgbm_tuned's refit matched session 7's validation predictions (6,997 rows; largest difference 5.0e-10), so it stands for lightgbm_tuned in the new pool. pandas warned that building a categorical column from codes outside its categories is deprecated: a validation or test code never seen in training becomes a missing value, as written.

Gemma on the longer input (gemma4_31b_long_timing.json; large_validation.log): large_validation answered the 1,237 EI validation prompts at 5,142.9 prompt tokens an item in 3,321.1 wall seconds: a first pass of 668.2 seconds, 1,081 items asked again with the top 1,000 log-probabilities (578.9 seconds), and 576 items answered from the prompt's log-probabilities, one sequence at a time (2,074.0 seconds); model loading is not counted. In session 8, at 4,096 tokens, the same took 2,358.5 wall seconds at 3,754.2 prompt tokens an item (results/fiorillo_v0_4/large_model/gemma4_31b_timing.json). The letters' summed probability before renormalization had a median of 0.99999993, a 10th percentile of 0.99999822 and a share of 0.0024 below one half (gemma4_31b_long_letters.json). Session 8's were 0.99999993, 0.99999833 and 0.0016 (results/fiorillo_v0_4/large_model/gemma4_31b_letters.json). So the model's next token stayed on the answer letters with the longer input, and no correction was needed. Its files were copied to the name gemma4_31b_long and the originals kept (large_validation.log). PubMedQA's prompts were written, because session 8's smoke test reads both questions' files, but not answered: session 8's PubMedQA predictions are used as they are.

BioGPT-Large (biogpt.json): correction 5's reader could not read the original checkpoint's arguments, since in 40 attempts over 587 seconds the archive's host name did not resolve from the pod (Hugging Face's did), so the input used fixed_versions.json's settings (a learned prompt of 9 tokens, prefix false, sources cut at 384 positions), as correction 5 provides. On PubMedQA's 150 calibrate validation items, 18 sources were cut, the three answer tokens' summed probability before renormalization had a median of 0.9982 (smallest 0.8999), and the most likely next token was an answer on all 150 (4.6 forward seconds; biogpt.json). BioGPT-Large-PubMedQA was fine-tuned on sets that hold all 150 validation items, so its validation figures are not a fair measure.

Calibration (calibration_s9.json): v0.5's EI specialist s9_ei_wavg chose temperature among the four families (mean out-of-fold log loss 0.17808; temperature_intercepts 0.18084, vector 0.18364, dirichlet_odir 0.18290), as session 8's two-family rule would. gemma4_31b_long, calibrated on all 1,237 validation prompts, chose temperature_intercepts (0.33840), and biogpt_large_pqa temperature_intercepts on its 150 items (0.50558) (calibration_s9.json). Session 8's gemma4_31b, at 4,096 tokens, had 0.34244 (results/fiorillo_v0_4/calibration_s8.json). The ECE limit for criterion 1 is the floor, 0.05: the 95th percentile of s9_ei_wavg's ECE if calibrated, over 5,000 draws of simulated test sets the size of EI's test split, drawn from its 654 validation rows, was 0.0184 (ece_limit_evidence_effect.json). The new readmission pool (session 8's members with xgboost_codes and catboost_codes; tabicl NOT BUILT) scored 0.836806 out of fold against session 8's pool's 0.837153, short of the 0.002 the rule asks, so v0.5 keeps session 8's pool (readmission_pool_v2.json).

On validation (descriptive; handoff_validation.json), s9_ei_wavg answered 0.9378 of the 1,237 prompts right and gemma4_31b_long 0.8909; s9_ei_wavg was wrong where Gemma was right on 28 prompts and right where Gemma was wrong on 86, and handing the least confident 10 percent to Gemma gave 0.9353 (s9_ei_wavg alone: 0.9378; a random hand-off of as many: 0.9331 expected). Session 8's pair on the same prompts, s8_ei with gemma4_31b, answered 0.9369 and 0.8860 right (validation_analyses/handoff_curves_validation_s8.json). The power calculation (power_validation.json, for the record only) put the chance that criterion 3 passes at 0.9993 if the validation difference in log loss, -0.1390, held on test. The freeze wrote release_candidate.json at 13:40:31 UTC (its SHA-256 bd45820bef38b523092813b406949c44af7f40babed6eb20658559ff3e83ea55), and testpreds began at once.

### Table 1b. The test predictions (logs/pod/testpreds.log; UTC, 2 October)

| Job | Started | Finished | Rows | Forward seconds | Input |
|---|---|---|---|---|---|
| s9_b_1 | 13:40:32 | 13:44:23 | 1,218 | 221.2 | session 8's test records |
| s9_b_2 | 13:44:23 | 13:48:15 | 1,218 | 221.3 | the same |
| s9_b_3 | 13:48:15 | 13:52:06 | 1,218 | 221.1 | the same |
| s9_long_1 | 13:52:06 | 13:58:31 | 1,218 | 369.3 | the long test records |
| s9_long_2 | 13:58:31 | 14:04:51 | 1,218 | 369.4 | the same |
| s9_long_3 | 14:04:51 | 14:11:12 | 1,218 | 369.0 | the same |
| s9_ei_wavg | 14:11:12 | 14:17:34 | 1,218 | 369.5 | the same |
| s9_ei_wavg, no article | 14:17:34 | 14:17:58 | 1,218 | 13.1 | checks/none |
| s9_ei_wavg, title only | 14:17:58 | 14:18:23 | 1,209 | 13.4 | checks/title |
| s9_ei_wavg, exchanged | 14:18:23 | 14:24:45 | 1,218 | 369.2 | checks/swap_long (its end derived: the next stamped line) |
| gemma4_31b_long | 14:24:45 | 15:17:42 | 1,218 | 3,042.8 (wall) | the long test prompts (gemma4_31b_long_timing.json), started when the GPU was free |
| biogpt_large_pqa | 15:17:42 | 15:18:11 | 500 | 15.1 | PubMedQA's test items |
| xgboost_codes, catboost_codes, lightgbm_tuned | 15:18:11 | 15:18:19 | 6,998 each | (processors) | readmission's test records |

Gemma 4 31B-it on the long test prompts (gemma4_31b_long_timing.json; testpreds.log) read 5,126.7 prompt tokens an item: a first pass of 657.0 seconds, 1,051 items asked again with the top 1,000 log-probabilities (558.6 seconds), and 512 answered from the prompt's log-probabilities (1,827.1 seconds). The letters' summed probability before renormalization had a median of 0.99999992 and a share of 0.0008 below one half (gemma4_31b_long_letters.json), so no correction was needed. BioGPT-Large read the 500 PubMedQA test items with validation's settings: 84 sources cut, the answer tokens' summed probability a median of 0.9982 with a share of 0.002 below one half, and the most likely next token an answer on all 500 (biogpt.json). Each tree member reproduced its validation predictions (largest difference 5.0e-10) before writing its 6,998 test rows (testpreds.log), and TabICL wrote none, being NOT BUILT. package wrote the four zips and passed its check at 15:18:29 UTC (docs/log.md). The copy to the Mac took 15:23:53 to 15:27:24 UTC (docs/log.md). The pod, stopped at 15:29:23 and deleted at 15:29:57 UTC (pod_console.json), ran from its start at 16:00:24 UTC on 1 October. That is 23.49 hours, about $82.67 at $3.52 an hour (derived).

### Table 2. The training runs (validation figures before calibration, whole validation split)

| Run | Recipe and seed | Optimizer steps | Training seconds (sequences a second) | Monitoring at the last step: accuracy, log loss | Whole validation split: accuracy, log loss, ECE |
|---|---|---|---|---|---|
| s9_b_1 | base, 20260927 | 330 | 3,895.2 (2.71) | 0.940, 0.1833 | 0.9305, 0.2427, 0.0366 |
| s9_b_2 | base, 20260928 | 330 | 3,867.6 (2.73) | 0.940, 0.1876 | 0.9297, 0.2288, 0.0350 |
| s9_b_3 | base, 20260929 | 330 | 3,894.8 (2.71) | 0.940, 0.1894 | 0.9345, 0.2490, 0.0351 |
| s9_swap_1 | swap, 20260927 | 330 | 3,890.5 (2.71) | 0.950, 0.1667 | 0.9272, 0.2289, 0.0302 |
| s9_long_1 | long, 20260927 | 330 | 5,777.2 (1.83) | 0.960, 0.1409 | 0.9402, 0.2192, 0.0351 |
| s9_lr1_1 | lr1, 20260927 | 330 | 3,870.4 (2.73) | 0.930, 0.1957 | 0.9329, 0.2313, 0.0386 |
| s9_lr2_1 | lr2, 20260927 | 330 | 3,874.4 (2.73) | 0.935, 0.1989 | 0.9361, 0.2471, 0.0420 |
| s9_swap_2 | swap, 20260928 | 330 | 3,881.5 (2.72) | 0.940, 0.1797 | 0.9289, 0.2286, 0.0231 |
| s9_long_2 | long, 20260928 | 330 | 5,803.6 (1.82) | 0.950, 0.1437 | 0.9313, 0.2245, 0.0375 |
| s9_lr1_2 | lr1, 20260928 | 330 | 3,888.4 (2.72) | 0.945, 0.1876 | 0.9369, 0.2326, 0.0380 |
| s9_lr2_2 | lr2, 20260928 | 330 | 3,880.2 (2.72) | 0.935, 0.1901 | 0.9345, 0.2169, 0.0339 |
| s9_long_3 | long, 20260929 | 330 | 5,808.7 (1.82) | 0.960, 0.1284 | 0.9378, 0.2171, 0.0374 |

The whole-split figures are read as a check that each run predicted; the recipe rule scores the 654 clean prompts, and nothing was chosen from these. For comparison, session 8's s8_ei_1, s8_ei_2 and s8_ei_3 scored 0.9386, 0.2301 and 0.0348; 0.9345, 0.2470 and 0.0424; and 0.9337, 0.2343 and 0.0373, on the same prompts (docs/log.md, session 8).

### Table 3. THE RECIPE RULE (recipe_selection.json; out-of-fold calibrated log loss on the 654 clean validation prompts)

| Recipe | Run, seed 20260927 | Run, seed 20260928 | Mean of the two (the candidate's score) | Below the base mean by more than m | Each run below base's run of the same seed | Passes |
|---|---|---|---|---|---|---|
| base | s9_b_1 0.2171 | s9_b_2 0.2061 | 0.2116 | (the reference) | (the reference) | (the reference) |
| swap | s9_swap_1 0.2100 | s9_swap_2 0.2174 | 0.2137 | no | no (s9_swap_1 yes, s9_swap_2 no) | no |
| long | s9_long_1 0.1816 | s9_long_2 0.1852 | 0.1834 | yes | yes | yes |
| lr1 | s9_lr1_1 0.2153 | s9_lr1_2 0.2015 | 0.2084 | no | yes | no |
| lr2 | s9_lr2_1 0.2053 | s9_lr2_2 0.1983 | 0.2018 | no | yes | no |

A score is session 8's calibration rule's out-of-fold log loss (the lower of the temperature and temperature_intercepts families; 5 folds, 20 repeats), from each run's logits; temperature was the lower family for every run but s9_lr2_1. s9_b_3 scored 0.2094, so the margin m, the largest minus the smallest of base's three scores, was 0.0110 (above the floor of 0.005; decided at 23:38:49 UTC on 1 October, when s9_b_3 had finished). A candidate's mean had to be below 0.2006 (derived: the mean of s9_b_1's and s9_b_2's scores less m). Only long passed, so the combination was not called (it needs two or more passing changes), and long won at 10:16:48 UTC on 2 October; its third run, s9_long_3 (seed 20260929), started at once. s9_b, the average of base's three runs, scored 0.2067. After s9_long_3, s9_ei, the probability average of the three long runs, scored 0.1790, and the weight average s9_ei_wavg 0.1781, not above s9_ei's by more than m, so v0.5 uses it (recipe_selection.json, 12:10:35 UTC). The weight average has rank 48 and alpha 96, and the largest difference between an adapted layer's joined update and the mean of the runs' updates is 7.6e-11 (weight_average.json). Swap averaging on s9_ei_wavg, with its answers to the 1,237 exchanged validation prompts, scored 0.2515 against 0.1781, so it was not kept (12:17:10 UTC). v0.5's EI specialist is s9_ei_wavg, one model, at 6,144 tokens and without swap averaging.

While the pod trained, step 18 ran on the linked shell (docs/tcga_prep_session9.md), and the linked shell and Chrome could not be reached from after 21:35 to 22:07 UTC, from after 22:30 to 23:02 UTC, from after 02:37 to 04:42 UTC on 2 October, and from after 09:47 to 10:18 UTC (the Mac asleep or the app closed; docs/log.md); the pod trained on without a break. The linked shell's virtual machine restarted at about 00:18 UTC and again before 04:42 UTC on 2 October, each time removing its temporary folders (/tmp/eddm_tmp, /tmp/eddm_pylibs and /tmp/eddm_s9) but not the project, and the helpers were recreated (docs/log.md); since the first restart the project folder does not allow deletion, and git's leftover lock files are moved into .git/_to_delete/ after each commit.

## Results (test split)

EI's test split was read for v0.5 once, after the freeze: release_candidate.json was written at 13:40:31 UTC on 2 October. testpreds started at 13:40:32 UTC and wrote that file's SHA-256 first (testpreds_log.json). The split was read once before, in session 8, for v0.4's bar, so this is the second time a bar is judged on it (comparison.json's release_bar). EI's test split has 1,218 prompts in 333 articles, and PubMedQA's has 500 items (comparison.json). Readmission's has 6,998 encounters (readmission_pool_v2_test.json).

PubMedQA's and readmission's test splits were read in sessions 4 to 8, and s7_b3 and the readmission pool were chosen after those results, so K5 and K6 are descriptive.

Every interval is a 95 percent percentile interval from 1,000 paired bootstrap resamples (seed 20260927): EI's resample articles, PubMedQA's and readmission's resample items. Criteria 2 and 3 are decided by 10,000 article resamples (Table 9); the intervals of the same differences in Table 8 are descriptive. A system's name alone is its chosen calibration; _temperature marks the temperature family alone, -raw the predictions before calibration, _sa the swap-averaged form and _s8rule session 8's rule.

### Calibration chosen on validation

v0.5's EI specialist, s9_ei_wavg, chose temperature among the four families on the 654 clean validation prompts (calibration_s9.json; Table 4), and session 8's two-family rule chose the same (temperature), so K9's difference is zero. gemma4_31b_long, calibrated on all 1,237 validation prompts, chose temperature_intercepts, and biogpt_large_pqa, calibrated on PubMedQA's 150 calibrate items, chose temperature_intercepts (calibration_s9.json). The temperature column is calibration_s9.json's temperature of the chosen family, or, for the two forms fitted only, the exponential of the fitted parameter, as calibration_family_cv.py applies it (derived).

### Table 4. Calibration chosen on validation (calibration_s9.json)

| System | Question | Prompts | Out-of-fold log loss, temperature | Temperature and intercepts | Vector | Dirichlet (ODIR) | Chosen | Temperature |
|---|---|---|---|---|---|---|---|---|
| `s9_ei_wavg` | Evidence Inference | 654 | 0.1781 | 0.1808 | 0.1836 | 0.1829 | temperature | 1.340 (derived) |
| `s9_ei_wavg_s8rule` | Evidence Inference | 654 | 0.1781 | 0.1808 |  |  | temperature | 1.340 |
| `s9_ei_wavg_sa` | Evidence Inference | 654 | fitted only |  |  |  | temperature | 0.843 (derived) |
| `s9_ei` | Evidence Inference | 654 | fitted only |  |  |  | temperature | 1.274 (derived) |
| `s9_b` | Evidence Inference | 654 | 0.2067 | 0.2087 |  |  | temperature | 1.386 |
| `s9_b_1` | Evidence Inference | 654 | 0.2171 | 0.2192 |  |  | temperature | 1.441 |
| `s9_b_2` | Evidence Inference | 654 | 0.2061 | 0.2085 |  |  | temperature | 1.396 |
| `s9_b_3` | Evidence Inference | 654 | 0.2094 | 0.2106 |  |  | temperature | 1.470 |
| `s9_long_1` | Evidence Inference | 654 | 0.1816 | 0.1843 |  |  | temperature | 1.356 |
| `s9_long_2` | Evidence Inference | 654 | 0.1852 | 0.1881 |  |  | temperature | 1.317 |
| `s9_lr1_1` | Evidence Inference | 654 | 0.2153 | 0.2183 |  |  | temperature | 1.444 |
| `s9_lr1_2` | Evidence Inference | 654 | 0.2015 | 0.2039 |  |  | temperature | 1.484 |
| `s9_lr2_1` | Evidence Inference | 654 | 0.2060 | 0.2053 |  |  | temperature_intercepts | 1.516 |
| `s9_lr2_2` | Evidence Inference | 654 | 0.1983 | 0.1999 |  |  | temperature | 1.404 |
| `s9_swap_1` | Evidence Inference | 654 | 0.2100 | 0.2123 |  |  | temperature | 1.204 |
| `s9_swap_2` | Evidence Inference | 654 | 0.2174 | 0.2198 |  |  | temperature | 1.156 |
| `gemma4_31b_long` | Evidence Inference | 1,237 | 0.3896 | 0.3384 |  |  | temperature_intercepts | 6.229 |
| `biogpt_large_pqa` | PubMedQA | 150 | 0.5077 | 0.5056 |  |  | temperature_intercepts | 1.463 |

s9_ei_wavg_sa and s9_ei take v0.5's specialist's family, fitted on their own clean validation predictions, so no out-of-fold table is computed for them ("fitted only"). The winning recipe's third run, s9_long_3, has no row: calibrate_session9.py calibrated the runs recipe_selection.json scored and base's three, and so left it out, against the plan's "each run" (Claude's omission, listed in the closeout); its raw predictions enter s9_ei and s9_ei_wavg as built.

### Gemma 4 31B-it on the longer input

gemma4_31b_long's three letters held nearly all of its next-token probability on both splits (Table 5): the median summed probability was 1.000000 on validation and 1.000000 on test (gemma4_31b_long_letters.json), so its answers were read as the plan fixed.

### Table 5. gemma4_31b_long's letters and timing (gemma4_31b_long_letters.json and gemma4_31b_long_timing.json)

| Split | Items | Letters' summed probability, median | 10th percentile | Share below one half | Answered from the top 20 | From the top 1,000 | From the prompt's log-probabilities | Prompt tokens an item | GPU seconds an item |
|---|---|---|---|---|---|---|---|---|---|
| validation | 1,237 | 1.000000 | 0.999998 | 0.0024 | 156 | 505 | 576 | 5,142.9 | 2.685 |
| test | 1,218 | 1.000000 | 0.999997 | 0.0008 | 167 | 539 | 512 | 5,126.7 | 2.498 |

### Evidence Inference: every system

v0.5's specialist, s9_ei_wavg, has a macro-F1 of 0.9248 (0.9059 to 0.9409) and a log loss of 0.2164 (0.1714 to 0.2645); v0.4's s8_ei has 0.9399 (0.9225 to 0.9544) and 0.1994 (0.1573 to 0.2486); gemma4_31b_long has 0.8668 (0.8390 to 0.8935) and 0.3993 (0.3301 to 0.4793) (Table 6).

### Table 6. Evidence Inference, every system (test split)

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 1.0767 (1.0580 to 1.0940) | 0.6525 (0.6395 to 0.6643) | 0.0305 (0.0018 to 0.0663) | 0.4212 (0.3855 to 0.4584) | 0.1976 (0.1855 to 0.2095) |
| `s9_ei_wavg` | 0.2164 (0.1714 to 0.2645) | 0.1101 (0.0871 to 0.1363) | 0.0168 (0.0124 to 0.0348) | 0.9269 (0.9089 to 0.9426) | 0.9248 (0.9059 to 0.9409) |
| `s9_ei_wavg_temperature` | 0.2164 (0.1714 to 0.2645) | 0.1101 (0.0871 to 0.1363) | 0.0168 (0.0124 to 0.0348) | 0.9269 (0.9089 to 0.9426) | 0.9248 (0.9059 to 0.9409) |
| `s9_ei_wavg-raw` | 0.2405 (0.1826 to 0.3037) | 0.1136 (0.0885 to 0.1417) | 0.0373 (0.0281 to 0.0546) | 0.9269 (0.9089 to 0.9426) | 0.9248 (0.9059 to 0.9409) |
| `s9_ei_wavg_s8rule` | 0.2164 (0.1714 to 0.2645) | 0.1101 (0.0871 to 0.1363) | 0.0168 (0.0124 to 0.0348) | 0.9269 (0.9089 to 0.9426) | 0.9248 (0.9059 to 0.9409) |
| `s9_ei_wavg_sa` | 0.3086 (0.2545 to 0.3692) | 0.1520 (0.1279 to 0.1776) | 0.0648 (0.0510 to 0.0818) | 0.9212 (0.9037 to 0.9380) | 0.9198 (0.9022 to 0.9374) |
| `s9_ei` | 0.2116 (0.1692 to 0.2594) | 0.1075 (0.0854 to 0.1329) | 0.0115 (0.0103 to 0.0316) | 0.9302 (0.9125 to 0.9457) | 0.9279 (0.9095 to 0.9442) |
| `s9_b` | 0.2164 (0.1714 to 0.2663) | 0.1107 (0.0869 to 0.1364) | 0.0126 (0.0105 to 0.0328) | 0.9286 (0.9116 to 0.9446) | 0.9266 (0.9093 to 0.9435) |
| `s9_b_1` | 0.2239 (0.1789 to 0.2746) | 0.1148 (0.0902 to 0.1417) | 0.0153 (0.0107 to 0.0321) | 0.9269 (0.9093 to 0.9438) | 0.9257 (0.9078 to 0.9426) |
| `s9_b_2` | 0.2248 (0.1793 to 0.2731) | 0.1152 (0.0911 to 0.1414) | 0.0118 (0.0101 to 0.0306) | 0.9245 (0.9058 to 0.9412) | 0.9227 (0.9039 to 0.9397) |
| `s9_b_3` | 0.2189 (0.1735 to 0.2704) | 0.1111 (0.0865 to 0.1383) | 0.0060 (0.0093 to 0.0282) | 0.9278 (0.9099 to 0.9436) | 0.9255 (0.9067 to 0.9421) |
| `s9_long_1` | 0.2177 (0.1728 to 0.2671) | 0.1111 (0.0875 to 0.1367) | 0.0215 (0.0137 to 0.0374) | 0.9245 (0.9072 to 0.9408) | 0.9226 (0.9046 to 0.9398) |
| `s9_long_2` | 0.2235 (0.1794 to 0.2738) | 0.1134 (0.0899 to 0.1390) | 0.0226 (0.0139 to 0.0392) | 0.9294 (0.9118 to 0.9447) | 0.9271 (0.9086 to 0.9436) |
| `s8_ei` | 0.1994 (0.1573 to 0.2486) | 0.1001 (0.0778 to 0.1266) | 0.0119 (0.0119 to 0.0300) | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| `gemma4_31b_long` | 0.3993 (0.3301 to 0.4793) | 0.2005 (0.1649 to 0.2395) | 0.0283 (0.0238 to 0.0567) | 0.8752 (0.8489 to 0.8991) | 0.8668 (0.8390 to 0.8935) |
| `gemma4_31b` | 0.4078 (0.3373 to 0.4804) | 0.2040 (0.1671 to 0.2418) | 0.0283 (0.0221 to 0.0560) | 0.8711 (0.8450 to 0.8949) | 0.8626 (0.8333 to 0.8879) |

### Table 7. Evidence Inference, each answer's F1 (test split)

| System | F1, significantly increased | F1, significantly decreased | F1, no significant difference |
|---|---|---|---|
| `prior` | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 0.5927 (0.5564 to 0.6286) |
| `s9_ei_wavg` | 0.9185 (0.8940 to 0.9390) | 0.9158 (0.8876 to 0.9437) | 0.9399 (0.9218 to 0.9555) |
| `s9_ei_wavg_temperature` | 0.9185 (0.8940 to 0.9390) | 0.9158 (0.8876 to 0.9437) | 0.9399 (0.9218 to 0.9555) |
| `s9_ei_wavg-raw` | 0.9185 (0.8940 to 0.9390) | 0.9158 (0.8876 to 0.9437) | 0.9399 (0.9218 to 0.9555) |
| `s9_ei_wavg_s8rule` | 0.9185 (0.8940 to 0.9390) | 0.9158 (0.8876 to 0.9437) | 0.9399 (0.9218 to 0.9555) |
| `s9_ei_wavg_sa` | 0.9112 (0.8868 to 0.9303) | 0.9176 (0.8901 to 0.9426) | 0.9307 (0.9103 to 0.9478) |
| `s9_ei` | 0.9248 (0.9023 to 0.9442) | 0.9161 (0.8881 to 0.9430) | 0.9427 (0.9245 to 0.9576) |
| `s9_b` | 0.9204 (0.8982 to 0.9401) | 0.9186 (0.8912 to 0.9451) | 0.9407 (0.9212 to 0.9563) |
| `s9_b_1` | 0.9178 (0.8936 to 0.9384) | 0.9226 (0.8946 to 0.9475) | 0.9367 (0.9163 to 0.9543) |
| `s9_b_2` | 0.9146 (0.8903 to 0.9360) | 0.9168 (0.8889 to 0.9424) | 0.9367 (0.9161 to 0.9546) |
| `s9_b_3` | 0.9253 (0.9027 to 0.9436) | 0.9134 (0.8853 to 0.9393) | 0.9379 (0.9187 to 0.9555) |
| `s9_long_1` | 0.9167 (0.8913 to 0.9373) | 0.9153 (0.8885 to 0.9414) | 0.9359 (0.9168 to 0.9521) |
| `s9_long_2` | 0.9231 (0.8995 to 0.9429) | 0.9164 (0.8858 to 0.9433) | 0.9419 (0.9236 to 0.9569) |
| `s8_ei` | 0.9358 (0.9152 to 0.9529) | 0.9365 (0.9113 to 0.9577) | 0.9475 (0.9278 to 0.9630) |
| `gemma4_31b_long` | 0.8473 (0.8103 to 0.8811) | 0.8301 (0.7833 to 0.8721) | 0.9229 (0.8993 to 0.9420) |
| `gemma4_31b` | 0.8455 (0.8088 to 0.8793) | 0.8241 (0.7787 to 0.8627) | 0.9182 (0.8947 to 0.9395) |

### K1 and the release bar

Every criterion passes (Table 9): s9_ei_wavg's ECE is 0.0168 against the limit 0.05; its log loss minus the prior's is -0.8603 (-0.9078 to -0.8104) and minus gemma4_31b_long's -0.1829 (-0.2598 to -0.1164), each over 333 articles; and its macro-F1, 0.9248, is not below gemma4_31b_long's, 0.8668.

compare_session9.py's outcome for the bar: every criterion passes, v0.5's bar is met, and THE RELEASE follows (step 17). EI's test split was read in session 8; this is the second time the bar is judged on it.

### Table 8. K1: s9_ei_wavg minus the prior and minus gemma4_31b_long

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_ei_wavg` minus `prior` | Log loss | 0.2164 | 1.0767 | -0.8603 (-0.9102 to -0.8093) | lower (better) |
| `s9_ei_wavg` minus `prior` | Brier score | 0.1101 | 0.6525 | -0.5424 (-0.5692 to -0.5138) | lower (better) |
| `s9_ei_wavg` minus `prior` | Accuracy | 0.9269 | 0.4212 | 0.5057 (0.4656 to 0.5419) | higher (better) |
| `s9_ei_wavg` minus `prior` | Macro-F1 | 0.9248 | 0.1976 | 0.7272 (0.7050 to 0.7479) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, significantly increased | 0.9185 | 0.0000 | 0.9185 (0.8940 to 0.9390) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, significantly decreased | 0.9158 | 0.0000 | 0.9158 (0.8876 to 0.9437) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, no significant difference | 0.9399 | 0.5927 | 0.3472 (0.3123 to 0.3804) | higher (better) |
| `s9_ei_wavg` minus `prior` | ECE | 0.0168 | 0.0305 | -0.0138 (-0.0434 to 0.0233) | no difference was detected; excludes below -0.0434 and above 0.0233 |
| `s9_ei_wavg` minus `gemma4_31b_long` | Log loss | 0.2164 | 0.3993 | -0.1829 (-0.2573 to -0.1164) | lower (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Brier score | 0.1101 | 0.2005 | -0.0904 (-0.1273 to -0.0570) | lower (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Accuracy | 0.9269 | 0.8752 | 0.0517 (0.0303 to 0.0760) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Macro-F1 | 0.9248 | 0.8668 | 0.0580 (0.0340 to 0.0849) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, significantly increased | 0.9185 | 0.8473 | 0.0712 (0.0366 to 0.1078) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, significantly decreased | 0.9158 | 0.8301 | 0.0858 (0.0488 to 0.1262) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, no significant difference | 0.9399 | 0.9229 | 0.0170 (-0.0021 to 0.0366) | no difference was detected; excludes below -0.0021 and above 0.0366 |
| `s9_ei_wavg` minus `gemma4_31b_long` | ECE | 0.0168 | 0.0283 | -0.0116 (-0.0345 to 0.0027) | no difference was detected; excludes below -0.0345 and above 0.0027 |

### Table 9. The release bar (criteria 2 and 3 with the bar's 10,000 article resamples)

| Criterion | Measure | s9_ei_wavg, or the difference | Limit, 95 percent interval, or gemma4_31b_long | Result |
|---|---|---|---|---|
| 1 | ECE | 0.0168 | 0.0500 | PASS |
| 2 | Log loss, s9_ei_wavg minus the prior | -0.8603 | -0.9078 to -0.8104 | PASS |
| 3 | Log loss, s9_ei_wavg minus gemma4_31b_long | -0.1829 | -0.2598 to -0.1164 | PASS |
| 4 | Macro-F1, s9_ei_wavg against gemma4_31b_long | 0.9248 | 0.8668 | PASS |
| The bar | all four |  |  | PASS |

### K2: what the license cost (descriptive)

v0.5's specialist learned from the 1,431 clean training articles, and v0.4's s8_ei from all 2,657 (license_recheck.json). s9_ei_wavg minus s8_ei is 0.0169 (-0.0095 to 0.0397) in log loss (no difference was detected; excludes below -0.0095 and above 0.0397) and -0.0140 (-0.0245 to -0.0025) in accuracy (lower (worse)); s9_b, the base recipe's three runs averaged, minus s8_ei is 0.0170 (0.0058 to 0.0284) in log loss (higher (worse)) (Table 10).

### Table 10. K2: what the license cost (descriptive)

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_b` minus `s8_ei` | Log loss | 0.2164 | 0.1994 | 0.0170 (0.0058 to 0.0284) | higher (worse) |
| `s9_b` minus `s8_ei` | Brier score | 0.1107 | 0.1001 | 0.0106 (0.0037 to 0.0174) | higher (worse) |
| `s9_b` minus `s8_ei` | Accuracy | 0.9286 | 0.9409 | -0.0123 (-0.0207 to -0.0034) | lower (worse) |
| `s9_b` minus `s8_ei` | Macro-F1 | 0.9266 | 0.9399 | -0.0133 (-0.0226 to -0.0041) | lower (worse) |
| `s9_b` minus `s8_ei` | F1, significantly increased | 0.9204 | 0.9358 | -0.0154 (-0.0284 to -0.0039) | lower (worse) |
| `s9_b` minus `s8_ei` | F1, significantly decreased | 0.9186 | 0.9365 | -0.0178 (-0.0329 to -0.0040) | lower (worse) |
| `s9_b` minus `s8_ei` | F1, no significant difference | 0.9407 | 0.9475 | -0.0068 (-0.0150 to 0.0016) | no difference was detected; excludes below -0.0150 and above 0.0016 |
| `s9_b` minus `s8_ei` | ECE | 0.0126 | 0.0119 | 0.0007 (-0.0105 to 0.0102) | no difference was detected; excludes below -0.0105 and above 0.0102 |
| `s9_ei_wavg` minus `s8_ei` | Log loss | 0.2164 | 0.1994 | 0.0169 (-0.0095 to 0.0397) | no difference was detected; excludes below -0.0095 and above 0.0397 |
| `s9_ei_wavg` minus `s8_ei` | Brier score | 0.1101 | 0.1001 | 0.0100 (-0.0055 to 0.0231) | no difference was detected; excludes below -0.0055 and above 0.0231 |
| `s9_ei_wavg` minus `s8_ei` | Accuracy | 0.9269 | 0.9409 | -0.0140 (-0.0245 to -0.0025) | lower (worse) |
| `s9_ei_wavg` minus `s8_ei` | Macro-F1 | 0.9248 | 0.9399 | -0.0152 (-0.0261 to -0.0036) | lower (worse) |
| `s9_ei_wavg` minus `s8_ei` | F1, significantly increased | 0.9185 | 0.9358 | -0.0173 (-0.0312 to -0.0041) | lower (worse) |
| `s9_ei_wavg` minus `s8_ei` | F1, significantly decreased | 0.9158 | 0.9365 | -0.0206 (-0.0358 to -0.0057) | lower (worse) |
| `s9_ei_wavg` minus `s8_ei` | F1, no significant difference | 0.9399 | 0.9475 | -0.0075 (-0.0189 to 0.0050) | no difference was detected; excludes below -0.0189 and above 0.0050 |
| `s9_ei_wavg` minus `s8_ei` | ECE | 0.0168 | 0.0119 | 0.0048 (-0.0093 to 0.0144) | no difference was detected; excludes below -0.0093 and above 0.0144 |

### K3: the recipes

On the 654 clean validation prompts in 179 articles, each candidate's two runs averaged, calibrated out of fold by session 8's rule, minus base's first two runs averaged (Tables 11 and 12; descriptive). THE RECIPE RULE itself decided on recipe_selection.json's scores (Table 3). As comparison.json notes, recipe_selection.json's score of a candidate is the mean of its two runs' own scores (correction 2), while K3 scores the two runs' probability average, so the two can differ; K3 is descriptive and decides nothing. The combination was not called: not called: fewer than two of the three changes passed.

On the test split, s9_ei_wavg minus s9_b is 0.0000 (-0.0202 to 0.0174) in log loss (no difference was detected; excludes below -0.0202 and above 0.0174) and -0.0018 (-0.0112 to 0.0089) in macro-F1 (no difference was detected; excludes below -0.0112 and above 0.0089): the long recipe's lead on validation is not seen on test (Table 12).

### Table 11. K3: the two-run averages on the clean validation prompts (out-of-fold calibrated)

| Two-run average | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `s9_b_1_2` | 0.2089 (0.1525 to 0.2752) | 0.1073 (0.0767 to 0.1416) | 0.0122 (0.0117 to 0.0366) | 0.9281 (0.9042 to 0.9489) | 0.9229 (0.8977 to 0.9457) |
| `s9_swap_1_2` | 0.2102 (0.1556 to 0.2718) | 0.1082 (0.0772 to 0.1415) | 0.0219 (0.0183 to 0.0454) | 0.9358 (0.9112 to 0.9567) | 0.9314 (0.9059 to 0.9535) |
| `s9_long_1_2` | 0.1802 (0.1326 to 0.2397) | 0.0892 (0.0630 to 0.1203) | 0.0074 (0.0098 to 0.0303) | 0.9450 (0.9242 to 0.9626) | 0.9407 (0.9178 to 0.9594) |
| `s9_lr1_1_2` | 0.2028 (0.1467 to 0.2715) | 0.1026 (0.0715 to 0.1382) | 0.0225 (0.0163 to 0.0452) | 0.9343 (0.9105 to 0.9554) | 0.9290 (0.9033 to 0.9523) |
| `s9_lr2_1_2` | 0.1933 (0.1387 to 0.2547) | 0.0985 (0.0688 to 0.1330) | 0.0161 (0.0127 to 0.0369) | 0.9327 (0.9082 to 0.9544) | 0.9287 (0.9027 to 0.9515) |

### Table 12. K3: the recipes minus base on validation, and v0.5 minus s9_b on test

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_swap_1_2` minus `s9_b_1_2` | Log loss | 0.2102 | 0.2089 | 0.0014 (-0.0156 to 0.0187) | no difference was detected; excludes below -0.0156 and above 0.0187 |
| `s9_swap_1_2` minus `s9_b_1_2` | Brier score | 0.1082 | 0.1073 | 0.0010 (-0.0088 to 0.0104) | no difference was detected; excludes below -0.0088 and above 0.0104 |
| `s9_swap_1_2` minus `s9_b_1_2` | Accuracy | 0.9358 | 0.9281 | 0.0076 (-0.0017 to 0.0181) | no difference was detected; excludes below -0.0017 and above 0.0181 |
| `s9_swap_1_2` minus `s9_b_1_2` | Macro-F1 | 0.9314 | 0.9229 | 0.0085 (-0.0022 to 0.0198) | no difference was detected; excludes below -0.0022 and above 0.0198 |
| `s9_swap_1_2` minus `s9_b_1_2` | F1, significantly increased | 0.9492 | 0.9372 | 0.0120 (-0.0028 to 0.0293) | no difference was detected; excludes below -0.0028 and above 0.0293 |
| `s9_swap_1_2` minus `s9_b_1_2` | F1, significantly decreased | 0.8997 | 0.8882 | 0.0115 (-0.0055 to 0.0280) | no difference was detected; excludes below -0.0055 and above 0.0280 |
| `s9_swap_1_2` minus `s9_b_1_2` | F1, no significant difference | 0.9454 | 0.9432 | 0.0022 (-0.0065 to 0.0117) | no difference was detected; excludes below -0.0065 and above 0.0117 |
| `s9_swap_1_2` minus `s9_b_1_2` | ECE | 0.0219 | 0.0122 | 0.0098 (-0.0064 to 0.0221) | no difference was detected; excludes below -0.0064 and above 0.0221 |
| `s9_long_1_2` minus `s9_b_1_2` | Log loss | 0.1802 | 0.2089 | -0.0286 (-0.0496 to -0.0097) | lower (better) |
| `s9_long_1_2` minus `s9_b_1_2` | Brier score | 0.0892 | 0.1073 | -0.0181 (-0.0304 to -0.0067) | lower (better) |
| `s9_long_1_2` minus `s9_b_1_2` | Accuracy | 0.9450 | 0.9281 | 0.0168 (0.0056 to 0.0297) | higher (better) |
| `s9_long_1_2` minus `s9_b_1_2` | Macro-F1 | 0.9407 | 0.9229 | 0.0178 (0.0063 to 0.0314) | higher (better) |
| `s9_long_1_2` minus `s9_b_1_2` | F1, significantly increased | 0.9489 | 0.9372 | 0.0117 (0.0000 to 0.0258) | no difference was detected; excludes below 0.0000 and above 0.0258 |
| `s9_long_1_2` minus `s9_b_1_2` | F1, significantly decreased | 0.9143 | 0.8882 | 0.0261 (0.0067 to 0.0486) | higher (better) |
| `s9_long_1_2` minus `s9_b_1_2` | F1, no significant difference | 0.9588 | 0.9432 | 0.0156 (0.0035 to 0.0295) | higher (better) |
| `s9_long_1_2` minus `s9_b_1_2` | ECE | 0.0074 | 0.0122 | -0.0047 (-0.0179 to 0.0093) | no difference was detected; excludes below -0.0179 and above 0.0093 |
| `s9_lr1_1_2` minus `s9_b_1_2` | Log loss | 0.2028 | 0.2089 | -0.0060 (-0.0181 to 0.0057) | no difference was detected; excludes below -0.0181 and above 0.0057 |
| `s9_lr1_1_2` minus `s9_b_1_2` | Brier score | 0.1026 | 0.1073 | -0.0047 (-0.0117 to 0.0019) | no difference was detected; excludes below -0.0117 and above 0.0019 |
| `s9_lr1_1_2` minus `s9_b_1_2` | Accuracy | 0.9343 | 0.9281 | 0.0061 (-0.0048 to 0.0167) | no difference was detected; excludes below -0.0048 and above 0.0167 |
| `s9_lr1_1_2` minus `s9_b_1_2` | Macro-F1 | 0.9290 | 0.9229 | 0.0061 (-0.0056 to 0.0183) | no difference was detected; excludes below -0.0056 and above 0.0183 |
| `s9_lr1_1_2` minus `s9_b_1_2` | F1, significantly increased | 0.9415 | 0.9372 | 0.0043 (-0.0081 to 0.0196) | no difference was detected; excludes below -0.0081 and above 0.0196 |
| `s9_lr1_1_2` minus `s9_b_1_2` | F1, significantly decreased | 0.8952 | 0.8882 | 0.0071 (-0.0128 to 0.0265) | no difference was detected; excludes below -0.0128 and above 0.0265 |
| `s9_lr1_1_2` minus `s9_b_1_2` | F1, no significant difference | 0.9503 | 0.9432 | 0.0071 (-0.0033 to 0.0176) | no difference was detected; excludes below -0.0033 and above 0.0176 |
| `s9_lr1_1_2` minus `s9_b_1_2` | ECE | 0.0225 | 0.0122 | 0.0103 (-0.0072 to 0.0205) | no difference was detected; excludes below -0.0072 and above 0.0205 |
| `s9_lr2_1_2` minus `s9_b_1_2` | Log loss | 0.1933 | 0.2089 | -0.0156 (-0.0316 to 0.0021) | no difference was detected; excludes below -0.0316 and above 0.0021 |
| `s9_lr2_1_2` minus `s9_b_1_2` | Brier score | 0.0985 | 0.1073 | -0.0087 (-0.0181 to 0.0003) | no difference was detected; excludes below -0.0181 and above 0.0003 |
| `s9_lr2_1_2` minus `s9_b_1_2` | Accuracy | 0.9327 | 0.9281 | 0.0046 (-0.0057 to 0.0141) | no difference was detected; excludes below -0.0057 and above 0.0141 |
| `s9_lr2_1_2` minus `s9_b_1_2` | Macro-F1 | 0.9287 | 0.9229 | 0.0058 (-0.0050 to 0.0161) | no difference was detected; excludes below -0.0050 and above 0.0161 |
| `s9_lr2_1_2` minus `s9_b_1_2` | F1, significantly increased | 0.9353 | 0.9372 | -0.0019 (-0.0181 to 0.0140) | no difference was detected; excludes below -0.0181 and above 0.0140 |
| `s9_lr2_1_2` minus `s9_b_1_2` | F1, significantly decreased | 0.9055 | 0.8882 | 0.0174 (-0.0005 to 0.0337) | no difference was detected; excludes below -0.0005 and above 0.0337 |
| `s9_lr2_1_2` minus `s9_b_1_2` | F1, no significant difference | 0.9452 | 0.9432 | 0.0020 (-0.0080 to 0.0118) | no difference was detected; excludes below -0.0080 and above 0.0118 |
| `s9_lr2_1_2` minus `s9_b_1_2` | ECE | 0.0161 | 0.0122 | 0.0039 (-0.0119 to 0.0145) | no difference was detected; excludes below -0.0119 and above 0.0145 |
| `s9_ei_wavg` minus `s9_b` | Log loss | 0.2164 | 0.2164 | 0.0000 (-0.0202 to 0.0174) | no difference was detected; excludes below -0.0202 and above 0.0174 |
| `s9_ei_wavg` minus `s9_b` | Brier score | 0.1101 | 0.1107 | -0.0006 (-0.0130 to 0.0089) | no difference was detected; excludes below -0.0130 and above 0.0089 |
| `s9_ei_wavg` minus `s9_b` | Accuracy | 0.9269 | 0.9286 | -0.0016 (-0.0106 to 0.0087) | no difference was detected; excludes below -0.0106 and above 0.0087 |
| `s9_ei_wavg` minus `s9_b` | Macro-F1 | 0.9248 | 0.9266 | -0.0018 (-0.0112 to 0.0089) | no difference was detected; excludes below -0.0112 and above 0.0089 |
| `s9_ei_wavg` minus `s9_b` | F1, significantly increased | 0.9185 | 0.9204 | -0.0019 (-0.0136 to 0.0105) | no difference was detected; excludes below -0.0136 and above 0.0105 |
| `s9_ei_wavg` minus `s9_b` | F1, significantly decreased | 0.9158 | 0.9186 | -0.0028 (-0.0174 to 0.0126) | no difference was detected; excludes below -0.0174 and above 0.0126 |
| `s9_ei_wavg` minus `s9_b` | F1, no significant difference | 0.9399 | 0.9407 | -0.0008 (-0.0095 to 0.0100) | no difference was detected; excludes below -0.0095 and above 0.0100 |
| `s9_ei_wavg` minus `s9_b` | ECE | 0.0168 | 0.0126 | 0.0042 (-0.0094 to 0.0153) | no difference was detected; excludes below -0.0094 and above 0.0153 |

### K8, K9 and K10

K8: the weight average minus the probability average of the same three runs, s9_ei_wavg minus s9_ei, is 0.0048 (0.0010 to 0.0092) in log loss (higher (worse)). K10: with swap averaging minus without, s9_ei_wavg_sa minus s9_ei_wavg, is 0.0922 (0.0672 to 0.1206) in log loss (higher (worse)) and 0.0481 (0.0259 to 0.0602) in ECE (higher (worse)); on validation it scored 0.2515 against 0.1781, so it was not kept. K9: s9_ei_wavg under its chosen family minus under session 8's rule is zero, both being temperature (Table 13); the four families' out-of-fold table is Table 4's first row.

### Table 13. K8, K9 and K10 (descriptive)

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_ei_wavg` minus `s9_ei` | Log loss | 0.2164 | 0.2116 | 0.0048 (0.0010 to 0.0092) | higher (worse) |
| `s9_ei_wavg` minus `s9_ei` | Brier score | 0.1101 | 0.1075 | 0.0026 (0.0006 to 0.0047) | higher (worse) |
| `s9_ei_wavg` minus `s9_ei` | Accuracy | 0.9269 | 0.9302 | -0.0033 (-0.0072 to 0.0000) | no difference was detected; excludes below -0.0072 and above 0.0000 |
| `s9_ei_wavg` minus `s9_ei` | Macro-F1 | 0.9248 | 0.9279 | -0.0031 (-0.0068 to 0.0003) | no difference was detected; excludes below -0.0068 and above 0.0003 |
| `s9_ei_wavg` minus `s9_ei` | F1, significantly increased | 0.9185 | 0.9248 | -0.0063 (-0.0116 to -0.0014) | lower (worse) |
| `s9_ei_wavg` minus `s9_ei` | F1, significantly decreased | 0.9158 | 0.9161 | -0.0003 (-0.0054 to 0.0046) | no difference was detected; excludes below -0.0054 and above 0.0046 |
| `s9_ei_wavg` minus `s9_ei` | F1, no significant difference | 0.9399 | 0.9427 | -0.0027 (-0.0070 to 0.0012) | no difference was detected; excludes below -0.0070 and above 0.0012 |
| `s9_ei_wavg` minus `s9_ei` | ECE | 0.0168 | 0.0115 | 0.0052 (-0.0049 to 0.0101) | no difference was detected; excludes below -0.0049 and above 0.0101 |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | Log loss | 0.2164 | 0.2164 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | Brier score | 0.1101 | 0.1101 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | Accuracy | 0.9269 | 0.9269 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | Macro-F1 | 0.9248 | 0.9248 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | F1, significantly increased | 0.9185 | 0.9185 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | F1, significantly decreased | 0.9158 | 0.9158 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | F1, no significant difference | 0.9399 | 0.9399 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg` minus `s9_ei_wavg_s8rule` | ECE | 0.0168 | 0.0168 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | Log loss | 0.3086 | 0.2164 | 0.0922 (0.0672 to 0.1206) | higher (worse) |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | Brier score | 0.1520 | 0.1101 | 0.0419 (0.0279 to 0.0562) | higher (worse) |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | Accuracy | 0.9212 | 0.9269 | -0.0057 (-0.0187 to 0.0066) | no difference was detected; excludes below -0.0187 and above 0.0066 |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | Macro-F1 | 0.9198 | 0.9248 | -0.0049 (-0.0184 to 0.0075) | no difference was detected; excludes below -0.0184 and above 0.0075 |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | F1, significantly increased | 0.9112 | 0.9185 | -0.0074 (-0.0266 to 0.0117) | no difference was detected; excludes below -0.0266 and above 0.0117 |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | F1, significantly decreased | 0.9176 | 0.9158 | 0.0018 (-0.0162 to 0.0188) | no difference was detected; excludes below -0.0162 and above 0.0188 |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | F1, no significant difference | 0.9307 | 0.9399 | -0.0092 (-0.0225 to 0.0034) | no difference was detected; excludes below -0.0225 and above 0.0034 |
| `s9_ei_wavg_sa` minus `s9_ei_wavg` | ECE | 0.0648 | 0.0168 | 0.0481 (0.0259 to 0.0602) | higher (worse) |

### K4: THE READING CHECKS

The readings were fixed before any check ran, and they gate no release. Both systems read the same on each input (reading_checks.json; the Reading column of Table 14).

### Table 14. K4: THE READING CHECKS on the test prompts (comparison.json's k4 table, from checks_s8.json and checks_v05.json)

| System | Input | Prompts | Accuracy | Macro-F1 | Agreement with its answer on the full input | Reading |
|---|---|---|---|---|---|---|
| v0.4 (`s8_ei`) | none | 1,218 | 0.5090 (0.4779 to 0.5439) | 0.4289 (0.3941 to 0.4634) | 0.5140 (0.4812 to 0.5473) | its answers depend on the article's text |
| v0.4 (`s8_ei`) | title | 1,209 | 0.5715 (0.5346 to 0.6065) | 0.5338 (0.4964 to 0.5733) | 0.5682 (0.5343 to 0.6027) | the title adds information about the result: a title can state the result, or the model may recall the trial, and these checks cannot tell them apart |
| v0.4 (`s8_ei`) | swap | 1,218 | 0.8103 (0.7802 to 0.8400) | 0.7958 (0.7633 to 0.8283) |  | partly |
| v0.5 (`s9_ei_wavg`) | none | 1,218 | 0.5156 (0.4874 to 0.5494) | 0.4384 (0.4085 to 0.4726) | 0.5238 (0.4955 to 0.5571) | its answers depend on the article's text |
| v0.5 (`s9_ei_wavg`) | title | 1,209 | 0.5715 (0.5389 to 0.6041) | 0.5327 (0.4995 to 0.5692) | 0.5782 (0.5471 to 0.6108) | the title adds information about the result: a title can state the result, or the model may recall the trial, and these checks cannot tell them apart |
| v0.5 (`s9_ei_wavg`) | swap | 1,218 | 0.7808 (0.7493 to 0.8117) | 0.7600 (0.7268 to 0.7946) |  | partly |

### Table 14b. K4: the exchanged prompts (comparison.json's k4 table)

| System | Input | Direction answers | Share of them that exchange | No-difference answers | Share of them that stay |
|---|---|---|---|---|---|
| v0.4 (`s8_ei`) | swap | 703 | 0.7226 (0.6765 to 0.7714) | 515 | 0.9806 (0.9657 to 0.9931) |
| v0.4 (`s8_ei`) | swap, comparator a time point or baseline | 38 | 0.1316 (0.0250 to 0.2632) | 10 | 1.0000 (1.0000 to 1.0000) |
| v0.5 (`s9_ei_wavg`) | swap | 699 | 0.6652 (0.6208 to 0.7157) | 519 | 0.9672 (0.9477 to 0.9841) |
| v0.5 (`s9_ei_wavg`) | swap, comparator a time point or baseline | 37 | 0.2162 (0.0857 to 0.3439) | 11 | 1.0000 (1.0000 to 1.0000) |

### Table 14c. K4: the title against no article, on the prompts in both inputs (comparison.json's k4 table)

| System | Prompts in both inputs | Macro-F1, title | Macro-F1, none | Macro-F1, title minus none |
|---|---|---|---|---|
| v0.4 (`s8_ei`) | 1,209 | 0.5338 (0.4964 to 0.5733) | 0.4314 (0.3967 to 0.4650) | 0.1024 (0.0736 to 0.1331) |
| v0.5 (`s9_ei_wavg`) | 1,209 | 0.5327 (0.4995 to 0.5692) | 0.4388 (0.4076 to 0.4723) | 0.0939 (0.0655 to 0.1234) |

### PubMedQA and K5

v0.5 answers PubMedQA with s7_b3 and its session 7 calibration: log loss 0.5797 (0.5070 to 0.6509) and accuracy 0.7960 (0.7600 to 0.8340) (Table 15). BioGPT-Large-PubMedQA was fine-tuned on PQA-L's fold 0 training and development sets, which hold all 150 of this project's PubMedQA validation items, so its calibration (session 8's rule on those items) was fitted on items it was trained on: its uncalibrated probabilities, renormalized over the three answers, are the primary result and its calibrated ones are secondary (PUBMEDQA BESIDE BIOGPT-LARGE).

### Table 15. PubMedQA, three answers (test split)

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 1.1593 (1.0401 to 1.2920) | 0.5855 (0.5503 to 0.6208) | 0.0646 (0.0226 to 0.1086) | 0.5520 (0.5080 to 0.5940) | 0.2371 (0.2246 to 0.2484) |
| `s7_b3` | 0.5797 (0.5070 to 0.6509) | 0.3147 (0.2683 to 0.3595) | 0.0401 (0.0315 to 0.0800) | 0.7960 (0.7600 to 0.8340) | 0.5698 (0.5406 to 0.6039) |
| `biogpt_large_pqa-raw` | 0.8899 (0.7428 to 1.0347) | 0.4058 (0.3462 to 0.4623) | 0.1371 (0.1114 to 0.1803) | 0.7520 (0.7120 to 0.7900) | 0.5481 (0.5119 to 0.5922) |
| `biogpt_large_pqa` | 0.7340 (0.6341 to 0.8367) | 0.3886 (0.3352 to 0.4389) | 0.0876 (0.0667 to 0.1299) | 0.7380 (0.6980 to 0.7761) | 0.5269 (0.4963 to 0.5638) |
| `gemma4_31b` | 0.6131 (0.5309 to 0.6957) | 0.3209 (0.2728 to 0.3696) | 0.0575 (0.0431 to 0.0970) | 0.7840 (0.7460 to 0.8200) | 0.5862 (0.5527 to 0.6252) |

### Table 16. K5: s7_b3 beside BioGPT-Large-PubMedQA, and BioGPT-Large-PubMedQA beside Gemma 4 31B-it (descriptive)

| Comparison | Metric | Difference (95 percent interval) | Reading |
|---|---|---|---|
| `s7_b3` minus `biogpt_large_pqa-raw` | Log loss | -0.3102 (-0.4219 to -0.1996) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa-raw` | Brier score | -0.0910 (-0.1352 to -0.0505) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa-raw` | ECE | -0.0970 (-0.1298 to -0.0501) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa-raw` | Accuracy | 0.0440 (0.0139 to 0.0780) | higher (better) |
| `s7_b3` minus `biogpt_large_pqa-raw` | Macro-F1 | 0.0216 (-0.0248 to 0.0696) | no difference was detected; excludes below -0.0248 and above 0.0696 |
| `s7_b3` minus `biogpt_large_pqa` | Log loss | -0.1543 (-0.2286 to -0.0844) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa` | Brier score | -0.0738 (-0.1116 to -0.0388) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa` | ECE | -0.0475 (-0.0779 to -0.0064) | lower (better) |
| `s7_b3` minus `biogpt_large_pqa` | Accuracy | 0.0580 (0.0260 to 0.0920) | higher (better) |
| `s7_b3` minus `biogpt_large_pqa` | Macro-F1 | 0.0428 (0.0024 to 0.0852) | higher (better) |
| `biogpt_large_pqa-raw` minus `gemma4_31b` | Log loss | 0.2768 (0.1695 to 0.3925) | higher (worse) |
| `biogpt_large_pqa-raw` minus `gemma4_31b` | Brier score | 0.0848 (0.0413 to 0.1314) | higher (worse) |
| `biogpt_large_pqa-raw` minus `gemma4_31b` | ECE | 0.0796 (0.0373 to 0.1163) | higher (worse) |
| `biogpt_large_pqa-raw` minus `gemma4_31b` | Accuracy | -0.0320 (-0.0680 to 0.0040) | no difference was detected; excludes below -0.0680 and above 0.0040 |
| `biogpt_large_pqa-raw` minus `gemma4_31b` | Macro-F1 | -0.0381 (-0.0867 to 0.0139) | no difference was detected; excludes below -0.0867 and above 0.0139 |
| `biogpt_large_pqa` minus `gemma4_31b` | Log loss | 0.1209 (0.0514 to 0.1987) | higher (worse) |
| `biogpt_large_pqa` minus `gemma4_31b` | Brier score | 0.0676 (0.0287 to 0.1083) | higher (worse) |
| `biogpt_large_pqa` minus `gemma4_31b` | ECE | 0.0301 (-0.0083 to 0.0677) | no difference was detected; excludes below -0.0083 and above 0.0677 |
| `biogpt_large_pqa` minus `gemma4_31b` | Accuracy | -0.0460 (-0.0840 to -0.0080) | lower (worse) |
| `biogpt_large_pqa` minus `gemma4_31b` | Macro-F1 | -0.0593 (-0.1058 to -0.0135) | lower (worse) |

### Readmission and K6

The new pool scored 0.836806 out of fold against session 8's pool's 0.837153, 0.000347 lower, short of the 0.002 the rule asks, so v0.5 keeps session 8's pool (readmission_pool_v2.json, 13:40:26 UTC), and readmission_pool_v2 is session 8's pool on the same test files (Tables 17 to 19).

### Table 17. Readmission (test split), part one

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 0.8861 (0.8734 to 0.8984) | 0.5377 (0.5298 to 0.5453) | 0.0017 (0.0002 to 0.0127) | 0.5942 (0.5829 to 0.6047) | 0.2485 (0.2455 to 0.2512) |
| `readmission_pool_v2` | 0.8275 (0.8132 to 0.8411) | 0.4972 (0.4882 to 0.5062) | 0.0066 (0.0085 to 0.0225) | 0.6157 (0.6042 to 0.6269) | 0.3695 (0.3579 to 0.3813) |
| `readmission_pool` | 0.8275 (0.8132 to 0.8411) | 0.4972 (0.4882 to 0.5062) | 0.0066 (0.0085 to 0.0225) | 0.6157 (0.6042 to 0.6269) | 0.3695 (0.3579 to 0.3813) |
| `xgboost_codes` | 0.8326 (0.8184 to 0.8462) | 0.5011 (0.4921 to 0.5100) | 0.0128 (0.0097 to 0.0258) | 0.6099 (0.5986 to 0.6210) | 0.3682 (0.3564 to 0.3799) |
| `xgboost_tuned` | 0.8310 (0.8166 to 0.8448) | 0.4997 (0.4907 to 0.5085) | 0.0103 (0.0081 to 0.0241) | 0.6139 (0.6026 to 0.6250) | 0.3748 (0.3625 to 0.3871) |
| `catboost_codes` | 0.8289 (0.8146 to 0.8428) | 0.4980 (0.4889 to 0.5072) | 0.0091 (0.0085 to 0.0232) | 0.6146 (0.6029 to 0.6270) | 0.3666 (0.3552 to 0.3783) |
| `catboost_tuned` | 0.8291 (0.8149 to 0.8429) | 0.4981 (0.4889 to 0.5072) | 0.0078 (0.0085 to 0.0230) | 0.6147 (0.6035 to 0.6262) | 0.3671 (0.3566 to 0.3781) |
| `lightgbm_tuned` | 0.8347 (0.8203 to 0.8487) | 0.5026 (0.4937 to 0.5114) | 0.0071 (0.0074 to 0.0225) | 0.6157 (0.6043 to 0.6275) | 0.3569 (0.3477 to 0.3660) |
| `lightgbm_tuned_refit` | 0.8347 (0.8203 to 0.8487) | 0.5026 (0.4937 to 0.5114) | 0.0071 (0.0074 to 0.0225) | 0.6157 (0.6043 to 0.6275) | 0.3569 (0.3477 to 0.3660) |

### Table 18. Readmission (test split), part two

| System | Macro-F1, decision weights | Quadratic weighted kappa | AUROC, 30 days | Calibration intercept, 30 days | Calibration slope, 30 days |
|---|---|---|---|---|---|
| `prior` | 0.2485 (0.2455 to 0.2512) | 0.0000 (0.0000 to 0.0000) | 0.5000 (0.5000 to 0.5000) | -0.0299 (-0.1120 to 0.0536) | not computed (not computed to not computed) |
| `readmission_pool_v2` | 0.4632 (0.4488 to 0.4780) | 0.1301 (0.1101 to 0.1491) | 0.6622 (0.6394 to 0.6855) | -0.0192 (-0.1031 to 0.0646) | 1.0493 (0.9108 to 1.1940) |
| `readmission_pool` | 0.4632 (0.4488 to 0.4780) | 0.1301 (0.1101 to 0.1491) | 0.6622 (0.6394 to 0.6855) | -0.0192 (-0.1031 to 0.0646) | 1.0493 (0.9108 to 1.1940) |
| `xgboost_codes` | 0.4578 (0.4439 to 0.4718) | 0.1242 (0.1057 to 0.1435) | 0.6584 (0.6344 to 0.6822) | -0.0208 (-0.1048 to 0.0634) | 1.0146 (0.8761 to 1.1579) |
| `xgboost_tuned` | 0.4551 (0.4410 to 0.4690) | 0.1371 (0.1168 to 0.1583) | 0.6584 (0.6355 to 0.6817) | -0.0338 (-0.1181 to 0.0509) | 0.9969 (0.8635 to 1.1430) |
| `catboost_codes` | 0.4595 (0.4453 to 0.4743) | 0.1325 (0.1139 to 0.1520) | 0.6565 (0.6333 to 0.6793) | -0.0363 (-0.1202 to 0.0474) | 1.0151 (0.8781 to 1.1532) |
| `catboost_tuned` | 0.4604 (0.4469 to 0.4749) | 0.1274 (0.1084 to 0.1459) | 0.6581 (0.6347 to 0.6823) | -0.0287 (-0.1120 to 0.0561) | 1.0160 (0.8806 to 1.1574) |
| `lightgbm_tuned` | 0.4484 (0.4335 to 0.4643) | 0.1200 (0.1029 to 0.1377) | 0.6577 (0.6341 to 0.6807) | -0.0177 (-0.1011 to 0.0680) | 1.0935 (0.9379 to 1.2504) |
| `lightgbm_tuned_refit` | 0.4484 (0.4335 to 0.4643) | 0.1200 (0.1029 to 0.1377) | 0.6577 (0.6341 to 0.6807) | -0.0177 (-0.1011 to 0.0680) | 1.0935 (0.9379 to 1.2504) |

### Table 19. K6: readmission (descriptive)

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `readmission_pool_v2` minus `readmission_pool` | Log loss | 0.8275 | 0.8275 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Brier score | 0.4972 | 0.4972 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | ECE | 0.0066 | 0.0066 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Accuracy | 0.6157 | 0.6157 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Macro-F1 | 0.3695 | 0.3695 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Macro-F1, decision weights | 0.4632 | 0.4632 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Quadratic weighted kappa | 0.1301 | 0.1301 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | AUROC, 30 days | 0.6622 | 0.6622 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Calibration intercept, 30 days | -0.0192 | -0.0192 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `readmission_pool_v2` minus `readmission_pool` | Calibration slope, 30 days | 1.0493 | 1.0493 | 0.0000 (0.0000 to 0.0000) | no difference was detected; the two are the same on these items |
| `xgboost_codes` minus `xgboost_tuned` | Log loss | 0.8326 | 0.8310 | 0.0016 (-0.0004 to 0.0037) | no difference was detected; excludes below -0.0004 and above 0.0037 |
| `xgboost_codes` minus `xgboost_tuned` | Brier score | 0.5011 | 0.4997 | 0.0014 (0.0000 to 0.0028) | higher (worse) |
| `xgboost_codes` minus `xgboost_tuned` | ECE | 0.0128 | 0.0103 | 0.0026 (-0.0076 to 0.0115) | no difference was detected; excludes below -0.0076 and above 0.0115 |
| `xgboost_codes` minus `xgboost_tuned` | Accuracy | 0.6099 | 0.6139 | -0.0040 (-0.0087 to 0.0010) | no difference was detected; excludes below -0.0087 and above 0.0010 |
| `xgboost_codes` minus `xgboost_tuned` | Macro-F1 | 0.3682 | 0.3748 | -0.0066 (-0.0140 to 0.0006) | no difference was detected; excludes below -0.0140 and above 0.0006 |
| `xgboost_codes` minus `xgboost_tuned` | Macro-F1, decision weights | 0.4578 | 0.4551 | 0.0027 (-0.0058 to 0.0115) | no difference was detected; excludes below -0.0058 and above 0.0115 |
| `xgboost_codes` minus `xgboost_tuned` | Quadratic weighted kappa | 0.1242 | 0.1371 | -0.0129 (-0.0248 to -0.0006) | lower (worse) |
| `xgboost_codes` minus `xgboost_tuned` | AUROC, 30 days | 0.6584 | 0.6584 | 0.0000 (-0.0065 to 0.0067) | no difference was detected; excludes below -0.0065 and above 0.0067 |
| `xgboost_codes` minus `xgboost_tuned` | Calibration intercept, 30 days | -0.0208 | -0.0338 | 0.0130 (0.0080 to 0.0184) | higher (better) |
| `xgboost_codes` minus `xgboost_tuned` | Calibration slope, 30 days | 1.0146 | 0.9969 | 0.0177 (-0.0222 to 0.0565) | no difference was detected; excludes below -0.0222 and above 0.0565 |
| `catboost_codes` minus `catboost_tuned` | Log loss | 0.8289 | 0.8291 | -0.0002 (-0.0023 to 0.0017) | no difference was detected; excludes below -0.0023 and above 0.0017 |
| `catboost_codes` minus `catboost_tuned` | Brier score | 0.4980 | 0.4981 | -0.0001 (-0.0015 to 0.0013) | no difference was detected; excludes below -0.0015 and above 0.0013 |
| `catboost_codes` minus `catboost_tuned` | ECE | 0.0091 | 0.0078 | 0.0013 (-0.0083 to 0.0085) | no difference was detected; excludes below -0.0083 and above 0.0085 |
| `catboost_codes` minus `catboost_tuned` | Accuracy | 0.6146 | 0.6147 | -0.0001 (-0.0047 to 0.0047) | no difference was detected; excludes below -0.0047 and above 0.0047 |
| `catboost_codes` minus `catboost_tuned` | Macro-F1 | 0.3666 | 0.3671 | -0.0005 (-0.0072 to 0.0067) | no difference was detected; excludes below -0.0072 and above 0.0067 |
| `catboost_codes` minus `catboost_tuned` | Macro-F1, decision weights | 0.4595 | 0.4604 | -0.0009 (-0.0095 to 0.0077) | no difference was detected; excludes below -0.0095 and above 0.0077 |
| `catboost_codes` minus `catboost_tuned` | Quadratic weighted kappa | 0.1325 | 0.1274 | 0.0051 (-0.0056 to 0.0167) | no difference was detected; excludes below -0.0056 and above 0.0167 |
| `catboost_codes` minus `catboost_tuned` | AUROC, 30 days | 0.6565 | 0.6581 | -0.0016 (-0.0086 to 0.0051) | no difference was detected; excludes below -0.0086 and above 0.0051 |
| `catboost_codes` minus `catboost_tuned` | Calibration intercept, 30 days | -0.0363 | -0.0287 | -0.0076 (-0.0126 to -0.0024) | lower (worse) |
| `catboost_codes` minus `catboost_tuned` | Calibration slope, 30 days | 1.0151 | 1.0160 | -0.0010 (-0.0419 to 0.0388) | no difference was detected; excludes below -0.0419 and above 0.0388 |

TabICL is NOT BUILT, so K6's TabICL minus lightgbm_tuned was not made (comparison.json's comparisons_not_made).

### K7: the hand-off (secondary, descriptive)

v0.5 hands nothing off; these analyses ask what a hand-off to Gemma 4 31B-it would do (handoff_curves.json on test, handoff_validation.json on validation). At each rate the least confident answers go to the large model; the random column is the exact expectation of handing off as many at random, the oracle a bound, and the I-35 rule the logistic rule frozen on validation.

### Table 20. K7: the hand-off on the test split (combined accuracy)

| Pair | Percent handed off | Items handed off | Confidence rule | Random hand-off (expected) | Oracle (a bound) | I-35 rule | I-35 rule minus confidence rule | Reading |
|---|---|---|---|---|---|---|---|---|
| `s8_ei` with `gemma4_31b` (Evidence Inference) | 0 | 0 | 0.9409 |  |  |  |  | |
| `s8_ei` with `gemma4_31b` (Evidence Inference) | 10 | 121 | 0.9294 (0.9102 to 0.9447) | 0.9340 (0.9183 to 0.9485) | 0.9581 (0.9433 to 0.9705) | 0.9286 (0.9102 to 0.9447) | -0.0008 (-0.0024 to 0.0016) | no difference was detected; excludes below -0.0024 and above 0.0016 |
| `s8_ei` with `gemma4_31b` (Evidence Inference) | 20 | 243 | 0.9163 (0.8985 to 0.9322) | 0.9270 (0.9110 to 0.9416) | 0.9581 (0.9433 to 0.9705) | 0.9163 (0.8986 to 0.9324) | 0.0000 (-0.0016 to 0.0016) | no difference was detected; excludes below -0.0016 and above 0.0016 |
| `s8_ei` with `gemma4_31b` (Evidence Inference) | 30 | 365 | 0.8998 (0.8782 to 0.9192) | 0.9200 (0.9036 to 0.9352) | 0.9581 (0.9433 to 0.9705) | 0.8998 (0.8782 to 0.9193) | 0.0000 (-0.0008 to 0.0016) | no difference was detected; excludes below -0.0008 and above 0.0016 |
| `s8_ei` with `gemma4_31b` (Evidence Inference) | 50 | 609 | 0.8785 (0.8552 to 0.9014) | 0.9060 (0.8876 to 0.9229) | 0.9581 (0.9433 to 0.9705) | 0.8785 (0.8552 to 0.9014) | 0.0000 (0.0000 to 0.0008) | no difference was detected; excludes below 0.0000 and above 0.0008 |
| `s8_ei` with `gemma4_31b` (Evidence Inference) | the large model alone |  | 0.8711 |  |  |  |  | |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 0 | 0 | 0.9269 |  |  |  |  | |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 10 | 121 | 0.9335 (0.9160 to 0.9492) | 0.9218 (0.9040 to 0.9369) | 0.9573 (0.9442 to 0.9693) | 0.9335 (0.9150 to 0.9493) | 0.0000 (-0.0048 to 0.0041) | no difference was detected; excludes below -0.0048 and above 0.0041 |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 20 | 243 | 0.9171 (0.8986 to 0.9328) | 0.9166 (0.8988 to 0.9322) | 0.9573 (0.9442 to 0.9693) | 0.9163 (0.8993 to 0.9336) | -0.0008 (-0.0024 to 0.0032) | no difference was detected; excludes below -0.0024 and above 0.0032 |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 30 | 365 | 0.8966 (0.8751 to 0.9173) | 0.9114 (0.8935 to 0.9276) | 0.9573 (0.9442 to 0.9693) | 0.8966 (0.8750 to 0.9170) | 0.0000 (-0.0024 to 0.0017) | no difference was detected; excludes below -0.0024 and above 0.0017 |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 50 | 609 | 0.8768 (0.8535 to 0.9013) | 0.9011 (0.8821 to 0.9182) | 0.9573 (0.9442 to 0.9693) | 0.8777 (0.8532 to 0.9014) | 0.0008 (-0.0009 to 0.0009) | no difference was detected; excludes below -0.0009 and above 0.0009 |
| `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | the large model alone |  | 0.8752 |  |  |  |  | |
| `s7_b3` with `gemma4_31b` (PubMedQA) | 0 | 0 | 0.7960 |  |  |  |  | |
| `s7_b3` with `gemma4_31b` (PubMedQA) | 10 | 50 | 0.8000 (0.7620 to 0.8360) | 0.7948 (0.7604 to 0.8308) | 0.8380 (0.8040 to 0.8720) |  |  |  |
| `s7_b3` with `gemma4_31b` (PubMedQA) | 20 | 100 | 0.7960 (0.7600 to 0.8340) | 0.7936 (0.7596 to 0.8284) | 0.8380 (0.8040 to 0.8720) |  |  |  |
| `s7_b3` with `gemma4_31b` (PubMedQA) | 30 | 150 | 0.7940 (0.7580 to 0.8320) | 0.7924 (0.7580 to 0.8262) | 0.8380 (0.8040 to 0.8720) |  |  |  |
| `s7_b3` with `gemma4_31b` (PubMedQA) | 50 | 250 | 0.7960 (0.7600 to 0.8340) | 0.7900 (0.7550 to 0.8230) | 0.8380 (0.8040 to 0.8720) |  |  |  |
| `s7_b3` with `gemma4_31b` (PubMedQA) | the large model alone |  | 0.7840 |  |  |  |  | |

### Table 21. K7: the risk-coverage areas (test, then validation)

| Split | Pair | AURC (95 percent interval) | E-AURC | AUGRC | Specialist wrong, large model right | Specialist right, large model wrong |
|---|---|---|---|---|---|---|
| test | `s8_ei` with `gemma4_31b` (Evidence Inference) | 0.01158 (0.00713 to 0.01743) | 0.00977 (0.00607 to 0.01468) | 0.00938 (0.00600 to 0.01365) | 21 | 106 |
| test | `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 0.01280 (0.00819 to 0.01824) | 0.01004 (0.00639 to 0.01454) | 0.01053 (0.00693 to 0.01468) | 37 | 100 |
| test | `s7_b3` with `gemma4_31b` (PubMedQA) | 0.09176 (0.06638 to 0.11833) | 0.06917 (0.04915 to 0.09005) | 0.06153 (0.04657 to 0.07666) | 21 | 27 |
| validation | `s9_ei_wavg` with `gemma4_31b_long` (Evidence Inference) | 0.01114 (0.00664 to 0.01682) | 0.00914 (0.00521 to 0.01439) | 0.00878 (0.00560 to 0.01246) | 28 | 86 |

### Sensitivity, calibration slope and intercept, power and printed zeros

Without the 6 test prompts that EI's README flags (Incorrect 6), K1's differences are in Table 22 (descriptive).

### Table 22. Sensitivity: K1 without the flagged test prompts (descriptive)

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s9_ei_wavg` minus `prior` | Log loss | 0.2163 | 1.0763 | -0.8600 (-0.9087 to -0.8109) | lower (better) |
| `s9_ei_wavg` minus `prior` | Brier score | 0.1098 | 0.6522 | -0.5424 (-0.5681 to -0.5164) | lower (better) |
| `s9_ei_wavg` minus `prior` | Accuracy | 0.9274 | 0.4224 | 0.5050 (0.4630 to 0.5413) | higher (better) |
| `s9_ei_wavg` minus `prior` | Macro-F1 | 0.9253 | 0.1980 | 0.7274 (0.7051 to 0.7471) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, significantly increased | 0.9189 | 0.0000 | 0.9189 (0.8964 to 0.9378) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, significantly decreased | 0.9174 | 0.0000 | 0.9174 (0.8864 to 0.9415) | higher (better) |
| `s9_ei_wavg` minus `prior` | F1, no significant difference | 0.9398 | 0.5940 | 0.3458 (0.3089 to 0.3787) | higher (better) |
| `s9_ei_wavg` minus `prior` | ECE | 0.0174 | 0.0293 | -0.0118 (-0.0413 to 0.0224) | no difference was detected; excludes below -0.0413 and above 0.0224 |
| `s9_ei_wavg` minus `gemma4_31b_long` | Log loss | 0.2163 | 0.4010 | -0.1848 (-0.2606 to -0.1164) | lower (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Brier score | 0.1098 | 0.2015 | -0.0917 (-0.1275 to -0.0574) | lower (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Accuracy | 0.9274 | 0.8746 | 0.0528 (0.0309 to 0.0764) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | Macro-F1 | 0.9253 | 0.8660 | 0.0593 (0.0354 to 0.0856) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, significantly increased | 0.9189 | 0.8458 | 0.0731 (0.0396 to 0.1097) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, significantly decreased | 0.9174 | 0.8295 | 0.0879 (0.0503 to 0.1287) | higher (better) |
| `s9_ei_wavg` minus `gemma4_31b_long` | F1, no significant difference | 0.9398 | 0.9228 | 0.0170 (-0.0016 to 0.0359) | no difference was detected; excludes below -0.0016 and above 0.0359 |
| `s9_ei_wavg` minus `gemma4_31b_long` | ECE | 0.0174 | 0.0287 | -0.0113 (-0.0341 to 0.0013) | no difference was detected; excludes below -0.0341 and above 0.0013 |

### Table 23. Calibration slope and intercept on the test split (each answer against the rest)

| Question | System | Answer | Slope (95 percent interval) | Intercept | Intercept with the slope | Rows with this answer |
|---|---|---|---|---|---|---|
| Evidence Inference | `s9_ei_wavg` | increased | 0.9112 (0.8188 to 1.0405) | 0.2244 (-0.1219 to 0.5996) | 0.1712 (-0.1248 to 0.5211) | 411 |
| Evidence Inference | `s9_ei_wavg` | decreased | 0.9919 (0.8773 to 1.1833) | -0.1294 (-0.4867 to 0.1917) | -0.1324 (-0.4750 to 0.2415) | 294 |
| Evidence Inference | `s9_ei_wavg` | no difference | 0.8850 (0.7943 to 1.0011) | -0.1139 (-0.5151 to 0.2811) | -0.1230 (-0.4569 to 0.2077) | 513 |
| PubMedQA | `s7_b3` | yes | 0.9647 (0.8394 to 1.1381) | -0.0217 (-0.2800 to 0.2453) | -0.0145 (-0.2728 to 0.2469) | 276 |
| PubMedQA | `s7_b3` | no | 1.1869 (1.0296 to 1.4260) | -0.1513 (-0.3927 to 0.0973) | -0.1468 (-0.4504 to 0.1496) | 169 |
| PubMedQA | `s7_b3` | maybe | 0.8990 (0.5392 to 1.3122) | 0.2371 (-0.0890 to 0.5250) | 0.0231 (-0.8637 to 0.8928) | 55 |

Power, derived before any prediction (power.json) and measured on validation for the record only (power_validation.json): the validation mean difference in log loss, s9_ei_wavg minus gemma4_31b_long, was -0.1390; s 0.7520, design effect 1.5716; on 1,218 test prompts a standard error of 0.0270, a pass line of -0.0529, an 80 percent chance of passing at a real gain of 0.0757, and a chance of 0.9993 at the validation difference.

0 of 194,637 prediction rows of any system, split or question have a true answer whose probability prints as 0 (zeros.json; derived as the sum over its entries).

### The cost per 1,000 questions

The cost per 1,000 questions is the GPU forward time a question at the pod's rate of $3.52 an hour, from the forward passes timed in the test predictions, each job alone on the GPU (comparison.json's cost_per_1000_questions, whose formula field gives the computation; Table 24).

### Table 24. The cost per 1,000 questions (GPU forward time at the pod's rate)

| Question | System | Forward seconds a question | Cost per 1,000 questions |
|---|---|---|---|
| Evidence Inference | `s9_ei_wavg` | 0.3034 | $0.297 |
| PubMedQA | `s7_b3` | 0.0200 | $0.020 |
| Readmission | `readmission_pool_v2` | not computed | not computed |
| Evidence Inference | `gemma4_31b_long` | 2.4982 | $2.443 |
| PubMedQA | `gemma4_31b` | 0.8040 | $0.786 |

Readmission's cost is not computed: its members are trees on the CPU and readers predicted in earlier sessions; no test run summary of session 9 times them on the GPU.

### Known contamination

Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PMC articles (EI's articles are PMC's), EI's repository (public since 2019), PubMedQA and the UCI data, which would favor both; EI's test labels are public (the project's site serves them to its leaderboard), so no evaluation of EI can be blind; Laya's, ModernBERT's and reader v1's notes and session 4's notes on Duan stand. EI's test split was read in session 8, for v0.4's bar, its comparisons and its checks; v0.5's recipe, calibration and pool were chosen on validation only and frozen in release_candidate.json before any test prediction of a v0.5 system existed, and this is the second time the bar is judged on this split.

## Published results beside G3, checked at their sources

G3 is v0.4's comparison of its EI specialist with published EI results (docs/fiorillo_v0_4.md, G3, descriptive, since the published systems were scored and counted differently); the plan asks that they be shown beside v0.5's specialist and s8_ei (Table 25). Session 9's comparisons did not compute the lenient scoring that Buchmann et al. used, so v0.5's figure in those rows is its strict macro-F1 (lenient scoring can only raise it, and in session 8 it changed the gold label of 1 prompt), and s8_ei's lenient figure is session 8's (results/fiorillo_v0_4/comparison.json). Each published value lies below v0.5's interval, as below s8_ei's.

### Table 25. G3's published results beside s8_ei and v0.5's specialist (descriptive)

| Published system | Scoring | Its macro-F1 | s8_ei's macro-F1 (95 percent interval) | s9_ei_wavg's macro-F1, strict (95 percent interval) | The published value against s9_ei_wavg's interval |
|---|---|---|---|---|---|
| DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa) | strict | 0.780 | 0.9399 (0.9225 to 0.9544) | 0.9248 (0.9059 to 0.9409) | excluded: below the interval |
| DeYoung et al. 2020, with the right evidence given | strict | 0.851 | 0.9399 (0.9225 to 0.9544) | 0.9248 (0.9059 to 0.9409) | excluded: below the interval |
| Buchmann et al. 2024, LED, the lower end of its range | lenient | 0.7430 | 0.9407 (0.9240 to 0.9551), lenient | 0.9248 (0.9059 to 0.9409) | excluded: below the interval |
| Buchmann et al. 2024, LED, the upper end of its range | lenient | 0.7649 | 0.9407 (0.9240 to 0.9551), lenient | 0.9248 (0.9059 to 0.9409) | excluded: below the interval |
| Buchmann et al. 2024, LongT5 | lenient | 0.8154 | 0.9407 (0.9240 to 0.9551), lenient | 0.9248 (0.9059 to 0.9409) | excluded: below the interval |

The facts below come from the publications, read on 30 September and 1 October 2026 and recorded in docs/prior_art.md and docs/fiorillo_v0_4.md; they are not this study's results, and the figure check does not cover them. Step 17's prior-art check of 2 October 2026 is in docs/prior_art.md.

- DeYoung et al. 2020, "Evidence Inference 2.0: More Data, Better Models" (BioNLP 2020), Table 1, macro-averaged over the three answers, on what the paper calls the augmented test set: the Biomed RoBERTa pipeline conditioned on the prompt has an F of 0.780, the same classifier given the human-selected evidence span ("Oracle Spans") 0.851, and a classifier given the prompt alone 0.511. Its Table 13 counts 10,150 training, 1,238 validation and 1,228 test prompts, in 2,672, 340 and 334 articles. At the fixed commit the plan's check of 30 September found 1,228 test prompts with any annotation row and 1,224 with an accepted one, the 6 prompts of the empty article among them; that count is the basis of G3's statement that the published results counted those 6 prompts, and the paper does not mention the empty article. Buchmann et al.'s EACL paper does not state its count of test prompts; the LAB benchmark's 1,218 test instances equal this study's count. The paper does not say how disagreeing annotations become one label.
- Buchmann, Eichler, Bodensohn, Kuznetsov and Gurevych 2024, "Document Structure in Long-Document Transformers" (EACL 2024), Table 3: test-set F1 averaged over 3 seeds, for base-size models given 15,000 steps of further pretraining on EI's documents and then fine-tuned on EI. G3's LED range, "74.30 to 76.49", runs from LED without structure infusion (74.30) to its best setting (76.49); across its ten settings LED's lowest is 73.78. LongT5's 81.54 is its run without structure infusion, which is also its best. The paper scores a prompt with several annotations by the annotation that gives the highest score, and it describes the original evaluation as taking the most often annotated class as the gold label. Its code (UKPLab/eacl2024-doc-structure, infusion/evaluation/tasks/evidence_inference.py) gives every annotation the same score when the prediction matches none of them and then takes the first one stored, where compare_session8.py keeps the prompt's own answer, the most common accepted label. The authors note that their numbers can only be compared among their own models (section 5.1).
- Buchmann, Liu and Gurevych 2024, "Attribute or Abstain: Large Language Models as Long Document Assistants" (the LAB benchmark; EMNLP 2024), Table 3: on EI 2.0's test set (1,218 instances by its Table 1), classification macro-F1 of 86 for a fine-tuned Flan-T5-XL answering from the whole article and 87 from ten retrieved segments, and 87 for GPT-4, on 100 test instances only. These are single runs, and how prompts with several labels were scored is not stated. They are the best full-article results found, and they are not part of the registered G3.
- BioDecision-4B (lighteternal/biodecision-tev1-4b, a model card titled BioDecision-4B, September 2026): a macro-F1 of 70.6 on 1,216 test prompts of EI, measured on the BigBio copy, whose context is one evidence span (the check of 30 September); PubMedQA's expert-labeled accuracy 76.6 on 500; a research-only license on the card.
- A research agent's search found no published EI result that reports calibration (ECE, Brier score or log loss); the search was short and is not exhaustive (UNVERIFIED). s8_ei's log loss and ECE therefore have no published comparison here.

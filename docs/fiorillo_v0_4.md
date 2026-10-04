# Fiorillo v0.4
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 1 October 2026
Sources: docs/prompt_session8.md (with corrections 1 to 15; corrections 1 to 13 as registered on OSF, https://osf.io/kaxmn/); docs/osf_prereg_fiorillo_v0_4.md; docs/log.md (session 8's entries); docs/prior_art.md (the check of 1 October 2026); results/fiorillo_v0_4/ (comparison.json, handoff_validation.json, handoff_test.json, power.json, power_validation.json, article_selection.json, records_manifest.json, trees.json, readmission_pool.json, readmission_pool_test.json, trees_test.json, calibration_s8.json, ece_limit_evidence_effect.json, pilot.json, release_candidate.json, testpreds_log.json, zeros.json, contamination_checks.json, large_model/gemma4_31b_letters.json, large_model/gemma4_31b_timing.json, metrics/, run_summaries/ and logs/pod/); models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt; Evidence Inference 2.0's LICENSE, README.md and annotations/README.md at commit a661e8c14f973398380c8865cf2f27a535aaaf6d (https://github.com/jayded/evidence-inference); Europe PMC's REST service (queried 30 September 2026); Gemma 4 31B-it's model card, chat template and API record at revision 842da3794eaa0b77d5f08bae87a17459d91ff475 (https://huggingface.co/google/gemma-4-31B-it) and the Gemma 4 license page (https://ai.google.dev/gemma/docs/gemma_4_license); vLLM's LICENSE at tag v0.30.0; the Python Package Index's records for xgboost and catboost; the licenses read 30 September 2026
Status: complete (session 8; the v0.4 release bar was met; releasing v0.4 is my decision after the closeout)

Abbreviations: EI is Evidence Inference 2.0; OSF is the Open Science Framework; PMC is PubMed Central, and PMCID a PubMed Central identifier; MIT is the MIT License; CC BY, CC BY-NC, CC BY-ND and CC BY-SA are the Creative Commons Attribution licenses (NC non-commercial, ND no derivatives, SA share alike), and CC0 is the Creative Commons public-domain dedication; REST is the web-service style of Europe PMC's query service; PQA-L, PQA-A and PQA-U are PubMedQA's expert-labeled, automatically labeled and unlabeled parts; LoRA is low-rank adaptation (small trainable matrices added beside a model's fixed weights); BM25 is Okapi BM25, a word-matching score; GPU is graphics processing unit; GiB is 2^30 bytes; UTC is Coordinated Universal Time; ECE is expected calibration error; AUROC is the area under the receiver operating characteristic curve; SHA-256 is Secure Hash Algorithm 256; CUDA is NVIDIA's platform for computing on its GPUs; vLLM is an open-source engine that serves language models; UCI is the University of California, Irvine, Machine Learning Repository; LED is the Longformer Encoder-Decoder; LAB is Buchmann et al.'s long-document benchmark; ID is identifier; G1 to G6 are the plan's comparisons (docs/prompt_session8.md, COMPARISONS) and H1 to H4 the registration's hypotheses; API is application programming interface; CPU is central processing unit; GB and MB are 10^9 and 10^6 bytes; SXM is NVIDIA's socketed module form of the H100; AdamW is the Adam optimizer with decoupled weight decay; F1 is the harmonic mean of precision and recall, and macro-F1 its mean over the answers; BioNLP is the biomedical natural language processing workshop of the Association for Computational Linguistics (ACL), EACL the Conference of the European Chapter of the ACL, and EMNLP the Conference on Empirical Methods in Natural Language Processing; the linked shell is the Linux virtual machine that the chat linked to my Mac runs commands in.

## Licenses (read 30 September 2026)

- EI: the repository's LICENSE at the fixed commit is the MIT License, "Copyright (c) 2019 Eric Lehman, Jay DeYoung, Regina Barzilay, Byron C. Wallace". It allows use, copying, modification and publication, with the notice kept in copies. annotations/README.md says nothing about the article texts' licenses; the repository says the articles were taken from PMC's open-access subset.
- EI's article texts: each article in PMC's open-access subset keeps its own license, and PMC says users are responsible for following them. Europe PMC's licenses for EI's 3,348 articles (read 30 September 2026; results/fiorillo_v0_4/article_licenses.csv gives each article's): CC BY 1,807, CC BY-NC 650, CC BY-NC-SA 360, CC BY-NC-ND 298, none recorded 197, CC0 28, CC BY-ND 6, CC BY-SA 2; non-commercial 1,308 (39.1 percent), no derivatives 304 (9.1 percent). The morning's check, in decision 3, counted 3,346 articles and 648 CC BY-NC; every other count is the same. No source read addresses training models on these texts (UNVERIFIABLE; this sentence revised on 1 October 2026 after an independent review), and 197 of the articles have no license recorded. Whether weights trained on non-commercial or no-derivatives texts may be published is UNVERIFIABLE: no source found addresses it. This session releases nothing, and a release is my decision later (docs/prompt_session8.md, decision 3; idea I-27).
- Gemma 4 31B-it at revision 842da3794eaa0b77d5f08bae87a17459d91ff475: the model card's license is apache-2.0, linking the Gemma 4 license page, which gives the Apache License 2.0; the repository is not gated. The site's navigation beside that page also links the Gemma Terms of Use and the Prohibited Use Policy, written for earlier Gemma models; the license page does not say that they bind Gemma 4 (UNVERIFIABLE). Here the model is only run, for research, and its measured results are published; it is not fine-tuned or redistributed, and its outputs train nothing.
- vLLM 0.30.0: the Apache License 2.0 (its LICENSE at tag v0.30.0).
- XGBoost: Apache-2.0; CatBoost: the Apache License, Version 2.0 (their records on the Python Package Index).
- Qwen3-4B-Base and PubMedQA: as read in session 7 (docs/fiorillo_v0_3.md, Licenses).

## What session 8 built

Fiorillo v0.4 is a routed system of specialists (idea I-25): each typed question goes to the specialist chosen for it, and the system is judged as one, question by question. PubMedQA goes to s7_b3 with its session 7 calibration; readmission goes to a pool of tree models (I-28); and a new question, from Evidence Inference 2.0 (EI; DeYoung et al., 2020; I-12), goes to s8_ei, the average of three arm B runs trained for it from s7_b3, each keeping its last weights (I-04, I-26). EI asks, given a randomized trial's article and a prompt that names an intervention, a comparator and an outcome, whether the outcome significantly increased, significantly decreased, or showed no significant difference. A large open model, Gemma 4 31B-it, is the reference in the release bar and, in a secondary, descriptive analysis, answers the least confident questions in a hand-off (I-13); v0.4 itself does not hand off. Arm A was trained again keeping its last weights (I-29). The bar was fixed before training, on EI's test split, on which no model of this project had predicted (its records were built and counted, as the registration states), and I registered the analysis on OSF before any prediction on that split existed (https://osf.io/kaxmn/, submitted under an embargo to 1 October 2027).

Outcome, in words (the figures are in the Results section): the v0.4 release bar was met. On EI's test split s8_ei's ECE was within its limit, its log loss was below the prior's and below Gemma 4 31B-it's, each difference's interval wholly below zero, and its macro-F1 was not below the large model's. The reference was not trained on EI: it answered with thinking off, its answer was read from its next token's letter probabilities, and it was calibrated on EI's validation split. s8_ei's macro-F1 is above each published value G3 names, and its interval excludes each; it is also above the best full-article results found, though the published systems were scored and counted differently and GPT-4's result came from 100 prompts (descriptive). Of the checks added after the results (corrections 14 and 15), the first found both models less accurate where no annotated evidence span lay inside the input, which is consistent with their reading the evidence and also with those prompts being harder; the second detected no difference on the test articles whose abstracts s8_ei's starting model had read in PubMedQA; the third found no article shared across EI's splits. None of them can rule out that the base model's pretraining saw EI's public labels. The hand-off, PubMedQA and readmission are descriptive, and PubMedQA's and readmission's test splits had been read before their specialists were chosen. Releasing v0.4 is my decision after the closeout.

## Method

The method was fixed in docs/prompt_session8.md before any training. Corrections 1 to 13 were written there before the work each one changed, except correction 6, which records a pod already deployed; the OSF registration (docs/osf_prereg_fiorillo_v0_4.md, with its As filed section) states the method as corrections 1 to 13 left it. Corrections 14 and 15 came after the results: they add descriptive checks, they are deviations from the registration, and they change nothing it fixed. The corrections are listed at the end of this section, and how each stage ran is in the next section.

### EI's records and question

- Source: EI 2.0's repository at commit a661e8c14f973398380c8865cf2f27a535aaaf6d (MIT): its prompts, annotations, split files and article texts. Texts and evidence offsets come from this one source.
- A prompt belongs to the split whose article list holds its PMCID; a prompt whose PMCID is in no split file is dropped. An annotation row is accepted when Valid Label and Valid Reasoning are both true and its text is not empty, and a prompt is kept when it has one. Its answer is the most common accepted label, a tie going to the smallest of EI's codes (significantly decreased, then no significant difference, then significantly increased), as EI's own code does.
- Also dropped: the test prompts of the one article whose text file is empty (PMC2206488), and, in the training split only, the prompts that EI's annotations/README.md lists as "Incorrect". Validation and test keep every prompt with an accepted answer and an article text.
- The monitoring subset: 300 validation prompts drawn with the seed 20260927. The training runs' logged validation and the pilot read it, and nothing that chooses reads it; its prompts stay in the validation split that the calibration, the ECE limit and the validation hand-off use.
- The question (src/questions/session8.yaml): "Based only on this article, how did the outcome differ with the intervention compared with the comparator?", with the options significantly increased, significantly decreased and no significant difference, in that order.
- The article's text: its title and abstract first, cut from their end if they alone do not fit; then, in the order below, every passage that still fits. A passage that does not fit is skipped and the next one tried, and if the exact count of tokens then runs over, the passages taken last are dropped until it fits. Passages in results sections come first, a passage's section being the last "BODY" heading before it and a results section one whose heading holds RESULT, FINDING, OUTCOME, EFFICAC, SAFETY, ADVERSE, TABLE, RESPONSE or END POINT; within each group the highest BM25 score against the prompt's intervention, comparator and outcome comes first (statistics from the training split's articles only), ties by position. The kept passages follow the title and abstract in their article order. The whole input fits 4,096 tokens of Qwen's tokenizer with 16 to spare, so no record is cut.

### s8_ei: arm B on EI

- s8_ei_1, s8_ei_2 and s8_ei_3: Qwen3-4B-Base at revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 with LoRA adapters (rank 16, alpha 32, dropout 0.05), the projection that carries the base model's hidden state to the head, and Laya's decision head, each starting from s7_b3's adapters, projection and head, with seeds 20260927, 20260928 and 20260929. Two epochs over EI's training prompts, 32 prompts an optimizer step (the last step of each epoch holds the remainder): one prompt at a time, with the gradients of 32 added up before each step (the pilot's choice), and without gradient checkpointing (recomputing activations to save memory). AdamW at 5e-5 for the adapters and 1e-4 for the projection and head, a cosine schedule to 1e-6, gradient clipping at 1.0, weight decay 0.01, in bfloat16, a 16-bit number format. The last weights are kept; the monitoring subset's validation every 80 steps was logged and never used to choose.
- s8_ei: each prompt's probabilities are the mean of the three runs' raw probabilities, and its calibration reads the logarithm of that mean. It costs three forward passes a question.

### Arm A, keeping the last weights

s8_a1, s8_a2 and s8_a3 are session 7's s7_a1, s7_a2 and s7_a3 (reader v1 in Laya's format; PQA-A, then PQA-U labeled by session 7's labeler, then session 7's corrected phase 3), run again with the last weights kept in every phase. Phase 2's records were rebuilt on the pod from session 7's labels and had to match labeler.json's SHA-256 before arm A could run; they matched (docs/log.md, steps 8 and 9).

### The pilot and the cost rule

The pilot ran alone on the GPU on EI's training items, 30 optimizer steps a setting, timed over steps 11 to 30: setting a, one prompt at a time with 32 accumulated and no gradient checkpointing, and setting b, four at a time with 8 accumulated and gradient checkpointing. A setting passed if it did not run out of memory and its peak plus 16 GiB (room for arm A beside it) was at most 76 GiB; the seeds use the passing setting with the higher rate. The cost rule then projected the total as the spend so far plus the work still to do as if nothing shared the GPU, and would have asked me above $75. The results are in the next section.

### The readmission pool

Run on the pod's processors beside the GPU work (correction 3), on session 4's readmission records. xgboost_tuned and catboost_tuned: settings drawn without replacement from fixed grids (50 of XGBoost's 1,620 combinations, 30 of CatBoost's 144), each scored by 5-fold stratified cross-validation on the training records, at most 3,000 trees and stopping after 100 rounds without improvement in the held-out fold's multi-class log loss; the best setting refitted on all training records with the mean of its folds' best iteration counts. The pool: softmax((sum over the members m of w_m log p_m) / T), where the softmax turns scores into probabilities, over lightgbm_tuned (session 7), xgboost_tuned, catboost_tuned, v0.1 and s7_b3, with the weights on a grid of steps of 0.1 that sum to 1 (1,001 points) and T fitted by maximum likelihood at each point, on the validation records; the point with the lowest validation log loss is kept. Its out-of-fold log loss (5 folds, 20 repeats) is reported beside it and labeled optimistic, because two members' calibrations were fitted on the same records.

### The large model

Gemma 4 31B-it at revision 842da3794eaa0b77d5f08bae87a17459d91ff475, in bfloat16, served by vLLM 0.30.0 (its CUDA 12.9 build with torch 2.13.0, on the pod's CUDA 12.8 driver: corrections 1, 7, 8 and 10) in its own environment, text only, thinking off, at most 8,192 tokens, without CUDA graphs (recorded sequences of GPU work). Each prompt is its chat template with one user message. For EI it holds the intervention, the comparator and the outcome, then "Article:" and the same selected text that s8_ei reads; for PubMedQA, "Question:" and the question, then "Abstract:" and each context as "<section>: <text>". Then come the question's instructions, the options lettered A, B and C, and "Answer with the letter of one option only." One greedy request (it takes the most likely token) for one new token reads the top 20 log-probabilities; an option's probability is the summed probability of its letter's one-token variants (with and without a leading space), and the three are renormalized. A prompt missing a letter from the top 20 is asked again at once with the top 1,000, and one still missing a letter is scored in a second process from the log-probability of the prompt followed by each letter variant (correction 5). vLLM's own sampler is used (correction 11), and each process shuts its engine down on every way out (correction 12). The large model's processes ran alone on the GPU, and its outputs never trained, tuned or selected a Fiorillo model. Before the validation run, a smoke test ran the four longest EI validation prompts (by characters) and the first two PubMedQA validation prompts through the same code with the small Gemma 4 E2B-it, whose outputs were used for nothing (corrections 9 and 12).

### Calibration

Session 7's rule (src/eval/calibrate_session8.py): for each system and question, two families are fitted on the validation items, a single temperature, and a temperature with an intercept for each answer after the first (the first's fixed at 0); the temperature is bounded between 0.1 and 10 and each intercept between -10 and 10. Each family is scored by its out-of-fold log loss over 5 folds and 20 repeats; the lower is chosen (an exact tie goes to the temperature) and fitted on all those items. The items: every EI validation prompt, PubMedQA's 150 calibrate items, and every readmission validation record. The systems: s8_ei and its three runs on EI; s8_a3 on PubMedQA and readmission; Gemma 4 31B-it on EI and PubMedQA. s7_b3 keeps its session 7 calibration. A system's name alone is its chosen calibration; _temperature marks the temperature family alone, and -raw the predictions before calibration.

### The freeze and the registration

The freeze wrote results/fiorillo_v0_4/release_candidate.json before any prediction on EI's test split existed: every specialist's calibration, the reference's, the ECE limit for EI, and the SHA-256 of the files they come from. The test stage then waited for the OSF registration (correction 13), filed at 00:36:37 UTC on 1 October 2026 (https://osf.io/kaxmn/; OSF Preregistration; submitted under an embargo to 1 October 2027, pending OSF's approval of the embargo when checked at about 01:13 UTC). testpreds first wrote testpreds_log.json with the frozen file's SHA-256 and its own start time. The unpacking on my Mac was set to stop unless testpreds_log.json named that file's SHA-256 and a start time later than the freeze; that check, the extraction and the copies ran from a helper in the linked shell's /tmp, outside the repository (docs/log.md, step 14). The test calibration and the pool's test step were set to stop unless their inputs had the SHA-256 the freeze recorded.

### Hand-off (secondary, descriptive)

For EI (s8_ei) and PubMedQA (s7_b3), items are ranked by the specialist's top calibrated probability, ties by record ID; at 0, 10, 20, 30 and 50 percent, the floor of rate times n least confident items are answered by Gemma 4 31B-it (calibrated) and the rest by the specialist, and the items are ranked again within each resample. Reported, each with a 95 percent interval from 1,000 resamples (EI's by article, PubMedQA's by item): the combined answers' accuracy, macro-F1 and log loss; each model's accuracy on the handed-off items; Gemma 4 31B-it alone; and the cost of GPU time for 1,000 questions at the pod's rate, $3.51 an hour, that is 1,000 × (s + r × g) / 3,600 × the rate, where s is the specialist's GPU seconds an item, g the large model's and r the share handed off: the specialist is counted on every question and the large model on the handed-off share. The seconds come from the test run, where each job ran alone: s8_ei's are the sum of its three runs', PubMedQA's are s7_b3_timing's (s7_b3 run again on the pod for its timing), and the large model's are its test run's. The large model's second process answered EI and PubMedQA items together, and the code shares its time between them by item count (large_model_session8.py; the plan does not fix this, and it enters only the cost). The time follows prompt length, and an EI prompt is about nine times as long as a PubMedQA one, so this share understates EI's seconds and overstates PubMedQA's. Shared by prompt tokens instead, taking the items in the second process to have their question's mean prompt length, the test run's figures would be about 1.879 GPU seconds an EI item and 0.174 a PubMedQA item, against the 1.620 and 0.804 the code reports (derived from gemma4_31b_timing.json); a thousand EI questions with 50 percent handed off would then cost about $1.45 rather than $1.32, and a thousand PubMedQA questions about $0.10 rather than $0.41 (derived). The same analysis on validation (handoff_validation.json) leaves PubMedQA's cost empty, since no session 8 run of s7_b3 read the validation split, and its EI costs may include time when the seeds' validation passes shared the GPU with arm A (correction 4).

### The release bar

Fixed before training, on EI's test split only, over all 1,218 test prompts: (1) s8_ei's ECE (15 bins) is at most the limit, the larger of 0.05 and the 95th percentile of the ECE that a perfectly calibrated model shows by chance on 1,218 prompts, from 5,000 simulated test sets drawn from s8_ei's calibrated validation predictions with each label drawn from its row's own probabilities (src/eval/ece_noise_floor.py; prompts are drawn one at a time, not by article, which makes the limit a little strict); (2) s8_ei's log loss is below the prior's (the training split's answer shares), with the 95 percent interval of the difference wholly below zero; (3) s8_ei's log loss is below Gemma 4 31B-it's, both calibrated on validation, with the 95 percent interval of the difference wholly below zero; (4) s8_ei's macro-F1 is not below Gemma 4 31B-it's point estimate. Criteria 2 and 3 are decided by intervals from 10,000 resamples of articles (prompts from one article are not independent); the 1,000-resample intervals of the same differences in G1 are descriptive. These are the registration's H1 to H4.

### Comparisons

On the test split, with session 4's metrics: log loss, Brier score, ECE, accuracy and macro-F1; for EI also each answer's F1; for readmission also macro-F1 with decision weights, quadratic weighted kappa, the AUROC for readmission within 30 days, and the calibration intercept and slope at 30 days. Every interval is a 95 percent percentile interval from 1,000 paired bootstrap resamples with the seed 20260927 (EI's by article, PubMedQA's and readmission's by item), except the bar's criteria 2 and 3.

- G1 (the bar): s8_ei minus the prior, and minus Gemma 4 31B-it, on every metric.
- G2 (descriptive): s8_ei minus each run alone, each calibrated by its own rule, and the spread of the three runs' log loss.
- G3 (descriptive): s8_ei's macro-F1 beside DeYoung et al. 2020's 0.780 (the full-text pipeline) and 0.851 (with the right evidence given), and, scored leniently as Buchmann et al. 2024 scored, beside their LED (74.30 to 76.49) and LongT5 (81.54); BioDecision-4B named and not compared.
- G4 (descriptive): on PubMedQA, v0.4 (s7_b3) minus v0.1, v0.4 minus Gemma 4 31B-it, s8_a3 minus s7_a3, and s8_a3 minus v0.1; and the check that s7_b3 loaded as it was.
- G5 (descriptive): on readmission, the pool minus lightgbm_tuned, minus blend_v0_1 and minus v0.1; xgboost_tuned and catboost_tuned minus lightgbm_tuned; s8_a3 minus s7_a3; and s8_a3 minus v0.1.
- G6 (secondary, descriptive): the hand-off.
- Sensitivity (descriptive; correction 13): G1 without every test prompt that EI's README flags, and with s8_ei's raw and temperature-only predictions.

A difference whose interval includes zero is reported as "no difference was detected"; otherwise the side of zero the interval lies on gives the reading. The readmission calibration intercept and slope are not ranked, since their ideals are 0 and 1.

### Corrections

1. (30 September, step 0) vLLM 0.30.0 has no CUDA 12.8 build, so the pod installs its CUDA 12.9 build when the driver is older than CUDA 13.
2. (step 3) XGBoost and CatBoost were installed on the linked shell in /tmp/eddm_pylibs, since its home volume had about 310 MB free.
3. (step 3) The readmission pool's search and fits moved to the pod's processors, and the tree models are zipped and copied.
4. (step 3) Session 7's s7_b3-raw PubMedQA test file went to the pod for the check that s7_b3 loaded as it was; the test of a tree fit resumed part way was not built (NOT BUILT); the validation hand-off's PubMedQA cost was left empty.
5. (step 3, after an independent review) The large model's further ways: the top 1,000 first, then a second process; a text-only engine without CUDA graphs.
6. (step 9, after the pod was deployed) The pod as deployed: a 150 GB disk instead of 200 GB, a CUDA 12.8 driver, and the server's maintenance from 1 October at 19:12 UTC, which bounded the schedule.
7. and 8. (step 11) prepare_large's checks of vLLM's kernels and of Triton were wrong, not the install, and were fixed.
9. (step 11, after two independent reviews) A smoke test with Gemma 4 E2B-it before large_validation, four code fixes, all five zips copied and checked, and the test hand-off's rate fixed at $3.51 an hour.
10. (step 11) torchcodec's CPU build in the vLLM environment, after the smoke test found a CUDA 13 build.
11. (step 11) vLLM's own sampler instead of FlashInfer's, after the smoke test found that the latter could not compile.
12. (step 11) Every vLLM process shuts its engine down on every way out, after the smoke test found a process that did not exit; the smoke test's checks changed.
13. (step 11) The test stage waited for the OSF registration; the sensitivity analyses are stated as the code computes them.
14. (1 October, after step 15) Three descriptive checks, not pre-registered: s8_ei's and Gemma 4 31B-it's accuracy by whether the evidence is inside the input, the overlap of EI's articles with PubMedQA, and duplicates across EI's splits.
15. (1 October, after correction 14's checks ran) Intervals for the differences between correction 14's groups.

## How each stage was run

Everything on the GPU ran on one Runpod pod (`0k22u0lbzm8r9t`: one H100 SXM GPU, Secure Cloud `CA-MTL-1`, a 150 GB container disk, $3.51 an hour), driven by scripts/pod_session8.sh from the linked shell. Its container started at 13:37:21 UTC on 30 September; it was stopped at about 02:44 UTC on 1 October and terminated at 02:44:54 UTC, after 13 hours 7 minutes, about $46.02 at that rate (derived). The Runpod balance was $91.37 before the pod (derived: $21.37 left by session 7 and $70 added) and $45.26 after it, so the session's Runpod spend was about $46.11 (derived), under the $80 cap. Times below are UTC, and every stage's log is in results/fiorillo_v0_4/logs/pod/.

### Table 1. The stages on the pod (UTC; 30 September, then 1 October from testpreds on)

| Stage | Started | Finished | What it did |
|---|---|---|---|
| prepare | 13:46:35 | 13:49:48 | installs; the fixed revisions; EI's records rebuilt on the pod from the selection file and checked against the Mac's |
| prepare_a | 13:50:53 | 13:55:41 | reader v1 in Laya's format; phase 2's records rebuilt and checked against labeler.json |
| pilot | 13:56:42 | 14:15:54 | alone on the GPU: the two settings and the cost rule |
| trees, on the pod's processors | 14:16:12 | 18:56:14 | xgboost_tuned's and catboost_tuned's searches and refits; the readmission pool on validation |
| ei, `s8_ei_1` | 14:16:54 | 17:49:19 | trained beside arm A, then beside the first three smoke tests; validation predictions |
| ei, `s8_ei_2` | 17:49:19 | 20:00:53 | trained beside the fourth smoke test, and otherwise alone; validation predictions |
| ei, `s8_ei_3` | 20:00:53 | 22:12:11 | trained alone on the GPU; validation predictions |
| arm_a, `s8_a1` | 14:16:55 | 15:58:44 | beside `s8_ei_1` |
| arm_a, `s8_a2` | 15:58:44 | 16:14:23 | beside `s8_ei_1` |
| arm_a, `s8_a3` | 16:14:23 | 16:57:23 | beside `s8_ei_1`; validation predictions |
| prepare_large | 14:16:57 | 17:01:04 | the large model's environment and weights, in four runs within these times (corrections 7, 8 and 10) |
| smoke_large | 16:57:30 | 18:19:32 | the smoke test with Gemma 4 E2B-it, in four runs within these times (corrections 9 to 12) |
| large_validation | 22:12:42 | 22:56:40 | Gemma 4 31B-it alone on the GPU, on the validation items |
| calibrate | 22:59:01 | 22:59:17 | calibration_s8.json; the ECE limit |
| freeze | 23:00:14 | 23:00:15 | release_candidate.json |
| testpreds (1 October) | 01:13:53 | 02:09:13 | after the OSF registration (correction 13): testpreds_log.json first, then every test prediction |
| package | 02:36:29 | 02:36:36 | the five zips and SHA256SUMS.txt |
| copy | 02:37:32 | 02:42:19 | the zips copied to the Mac and checked |

EI's records (step 2, built on the Mac and rebuilt on the pod from the selection file): 29 prompts whose PMCID is in no split file were dropped, with 210 training, 21 validation and 28 test prompts without an accepted row, the 6 test prompts of the empty article, and the 70 training prompts listed as Incorrect. Kept (records_manifest.json): training 10,046 prompts in 2,657 articles (significantly increased 3,054, significantly decreased 2,454, no significant difference 4,538); validation 1,237 in 340; test 1,218 in 333. The accepted labels of 16 training, 1 validation and 3 test prompts disagree.

Tokens per record: mean 3,839 (training), 3,831 (validation) and 3,870 (test); the largest is 4,080, 4,080 and 4,080, under the limit of 4,096, so no record was cut.

An accepted evidence span lies wholly inside the kept text for 9,487 of the 9,850 training prompts with usable offsets (96.31 percent; the plan's check found 95.76) and for 1,151 of 1,206 validation prompts (95.44 percent) (article_selection.json).

Europe PMC reported the licenses of 3,348 articles: CC BY 1,807, CC BY-NC 650, CC BY-NC-SA 360, CC BY-NC-ND 298, none recorded 197, CC0 28, CC BY-ND 6 and CC BY-SA 2 (records_manifest.json).

The pilot ran alone on the GPU: setting a (micro-batch 1, accumulation 32, no gradient checkpointing) trained 2.8558 items a second at a peak of 36.40 GiB, and setting b (micro-batch 4, accumulation 8, gradient checkpointing) 1.2923 items a second at 15.39 GiB; both passed, and the seeds used setting a. The forward pass over the 300 monitoring prompts ran at 5.3708 items a second (pilot.json). The cost rule's projection was $43.38 (derived), under the $75 line.

`s8_ei_1` trained beside arm A until 16:57:23 and then beside the first three smoke tests, which started at 16:57:30, until 17:45:23; `s8_ei_2` trained beside the fourth smoke test from 18:15:08 to 18:19:32 and otherwise alone, until 19:56:53; `s8_ei_3` trained alone, until 22:08:13 (Table 2). Arm A's runs kept their last weights: `s8_a1` 6,539 optimizer steps, `s8_a2` 957 and `s8_a3` 2,739 (35.75 items a second).

### Table 2. The three EI runs (validation figures before calibration)

| Run | Optimizer steps | Training seconds | Training items a second | Peak GPU memory, GiB | Monitoring log loss at the last step | Validation accuracy | Validation log loss | Validation ECE |
|---|---|---|---|---|---|---|---|---|
| `s8_ei_1` | 628 | 12,219.5 | 1.64 | 36.4 | 0.1815 | 0.9386 | 0.2301 | 0.0348 |
| `s8_ei_2` | 628 | 7,447.3 | 2.70 | 36.4 | 0.1901 | 0.9345 | 0.2470 | 0.0424 |
| `s8_ei_3` | 628 | 7,430.4 | 2.70 | 36.4 | 0.1949 | 0.9337 | 0.2343 | 0.0373 |

The trees, on the pod's processors (trees.json): `xgboost_tuned` is the best of 50 settings drawn from 1,620 by 5-fold cross-validated log loss on the 55,978 training records (maximum depth 10, minimum child weight 20, learning rate 0.02, row and column samples 0.8 and 0.8, L2 penalty 10), refitted with 333 trees; `catboost_tuned` is the best of 30 drawn from 144 (depth 8, learning rate 0.03, L2 penalty 3, bagging temperature 1, border count 254), refitted with 1,222 trees.

The readmission pool's weights, fitted on the 6,997 validation records on a grid of tenths: `xgboost_tuned` 0.4, `catboost_tuned` 0.6, `lightgbm_tuned` 0.0, v0.1 0.0 and `s7_b3` 0.0, with temperature 0.9957; its validation log loss is 0.83679, and out of fold 0.83715, which is optimistic because two members' calibrations were fitted on the same records (readmission_pool.json).

Gemma 4 31B-it read EI's 1,237 validation prompts (mean 3,754 prompt tokens) at 1.907 GPU seconds an item, and its 1,218 test prompts at 1.620; PubMedQA's 150 calibrate items at 0.945 and 500 test items at 0.804 (gemma4_31b_timing.json; Table 4).

The ECE limit for EI is 0.05: the 95th percentile of a calibrated model's ECE at 1,218 prompts, from 5,000 draws, was 0.0202, below the floor of 0.05 (ece_limit_evidence_effect.json).

So the floor set the limit, not the percentile. The noise floor draws prompts one by one, not by article, which would make a limit set by the percentile a little strict.

testpreds waited for the OSF registration from the freeze to its start (correction 13), about $7.80 of the pod's time (derived; docs/log.md).

In testpreds the three EI runs took 221.8, 221.8 and 222.1 forward seconds for their 1,218 prompts each, and `s7_b3_timing` 10.0 seconds for PubMedQA's 500 items; the two tree models reproduced their validation predictions to within one billionth before predicting the test split (testpreds_log.json).

package wrote the 5 zips, 3,411,585,854 bytes in all (SHA256SUMS.txt). The pod sat idle from the end of testpreds to the start of package, 27 minutes (about $1.60, derived), while Claude ran background checks; Claude should have kept those checks shorter or started them after the copy.

Every zip's size and SHA-256 matched SHA256SUMS.txt, unzip -t found no errors in any of them, and models/fiorillo_v0_4/check_package.py found nothing missing; the pod was then stopped and terminated.

When testpreds loaded each EI run's tokenizer, transformers warned of "an incorrect regex pattern", a check written for Mistral's tokenizers. The three runs' tokenizer files are byte-identical to s7_b3's, which session 7 found identical to Qwen3-4B-Base's (docs/log.md, steps 12 and 14).

The logs zip was made while package ran, so results/fiorillo_v0_4/logs/pod/package.log holds only that stage's first lines; its output, the zips' sizes and SHA-256, is in docs/log.md (step 13).

Step 14 (the unpack) and step 15 (the analyses, from 02:52:31 to 02:54:23 UTC) ran on the linked shell.

## Results (test split)

EI's test split was read once, after the freeze: release_candidate.json was written at 23:00:15 UTC on 30 September, and testpreds started at 01:13:53 UTC on 1 October and wrote that file's SHA-256 first. EI's test split has 1,218 prompts in 333 articles (significantly increased 411, significantly decreased 294, no significant difference 513); PubMedQA's has 500 items and readmission's 6,998 encounters.

PubMedQA's and readmission's test splits were read in sessions 4 to 7, and s7_b3 and the readmission pool's members were chosen for v0.4 after those results, so Tables 11 to 15 are descriptive and favor the chosen specialists.

Every interval is a 95 percent percentile interval from 1,000 paired bootstrap resamples (seed 20260927): EI's resample articles, PubMedQA's and readmission's resample items. Criteria 2 and 3 are decided by 10,000 article resamples (Table 8); the intervals of the same differences in Table 7 are descriptive. A system's name alone is its chosen calibration; _temperature marks the temperature family alone, and -raw the predictions before calibration.

### Calibration and the large model on validation

s8_ei and each of its three runs chose the temperature family on validation (s8_ei's temperature is 1.460); Gemma 4 31B-it chose the temperature with intercepts on both questions, and its PubMedQA temperature, 10.000, is the family's upper bound (Table 3). On EI's validation prompts Gemma 4 31B-it's log loss was 1.9906 before calibration and 0.3403 after, in sample for the calibration (out of fold 0.3424).

### Table 3. Calibration chosen on validation

| System | Question | Out-of-fold log loss, temperature | Out-of-fold log loss, temperature and intercepts | Chosen | Temperature | Intercept, second answer | Intercept, third answer |
|---|---|---|---|---|---|---|---|
| `s8_ei` | Evidence Inference | 0.1994 | 0.2004 | temperature | 1.460 | 0.000 | 0.000 |
| `s8_ei_1` | Evidence Inference | 0.1990 | 0.2005 | temperature | 1.512 | 0.000 | 0.000 |
| `s8_ei_2` | Evidence Inference | 0.2083 | 0.2090 | temperature | 1.567 | 0.000 | 0.000 |
| `s8_ei_3` | Evidence Inference | 0.2070 | 0.2073 | temperature | 1.459 | 0.000 | 0.000 |
| `gemma4_31b` | Evidence Inference | 0.3956 | 0.3424 | temperature_intercepts | 6.255 | -1.850 | -1.304 |
| `gemma4_31b` | PubMedQA | 0.6579 | 0.6004 | temperature_intercepts | 10.000 | -0.318 | -1.348 |
| `s8_a3` | PubMedQA | 0.6609 | 0.6089 | temperature_intercepts | 2.318 | -0.309 | -1.271 |
| `s8_a3` | Readmission | 0.8537 | 0.8536 | temperature_intercepts | 0.962 | -0.009 | 0.088 |

Gemma 4 31B-it's three letters held nearly all of its next-token probability on every split (Table 4), so its answers were read as the plan fixed; the last columns count how each prompt's letters were found.

### Table 4. Gemma 4 31B-it's letters

| Question | Split | Items | Letters' summed probability, median | 10th percentile | Share below one half | Answered from the top 20 | From the top 1,000 | From the prompt's log-probabilities | GPU seconds an item |
|---|---|---|---|---|---|---|---|---|---|
| Evidence Inference | validation | 1,237 | 1.000000 | 0.999998 | 0.0016 | 153 | 497 | 587 | 1.907 |
| Evidence Inference | test | 1,218 | 1.000000 | 0.999997 | 0.0000 | 166 | 529 | 523 | 1.620 |
| PubMedQA | validation | 150 | 0.999999 | 0.999999 | 0.0000 | 41 | 56 | 53 | 0.945 |
| PubMedQA | test | 500 | 0.999999 | 0.999998 | 0.0000 | 138 | 186 | 176 | 0.804 |

### Evidence Inference: every system

s8_ei's macro-F1 is 0.9399 (0.9225 to 0.9544) and its log loss 0.1994 (0.1573 to 0.2486); Gemma 4 31B-it's are 0.8626 (0.8333 to 0.8879) and 0.4078 (0.3373 to 0.4804) (Table 5).

The lower end of s8_ei's ECE interval lies a little above its estimate (both are 0.0119 at four decimals): ECE computed on a resample is biased upward, so its percentile interval can sit above the estimate. Criterion 1 reads the estimate, as fixed.

### Table 5. Evidence Inference, every system (test split)

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 1.0767 (1.0580 to 1.0940) | 0.6525 (0.6395 to 0.6643) | 0.0305 (0.0018 to 0.0663) | 0.4212 (0.3855 to 0.4584) | 0.1976 (0.1855 to 0.2095) |
| `s8_ei` | 0.1994 (0.1573 to 0.2486) | 0.1001 (0.0778 to 0.1266) | 0.0119 (0.0119 to 0.0300) | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| `s8_ei_temperature` | 0.1994 (0.1573 to 0.2486) | 0.1001 (0.0778 to 0.1266) | 0.0119 (0.0119 to 0.0300) | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| `s8_ei-raw` | 0.2238 (0.1651 to 0.2915) | 0.1024 (0.0781 to 0.1313) | 0.0325 (0.0244 to 0.0504) | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| `s8_ei_1` | 0.2085 (0.1646 to 0.2575) | 0.1050 (0.0819 to 0.1315) | 0.0138 (0.0114 to 0.0321) | 0.9384 (0.9212 to 0.9540) | 0.9372 (0.9198 to 0.9532) |
| `s8_ei_1_temperature` | 0.2085 (0.1646 to 0.2575) | 0.1050 (0.0819 to 0.1315) | 0.0138 (0.0114 to 0.0321) | 0.9384 (0.9212 to 0.9540) | 0.9372 (0.9198 to 0.9532) |
| `s8_ei_1-raw` | 0.2438 (0.1812 to 0.3190) | 0.1082 (0.0834 to 0.1372) | 0.0369 (0.0276 to 0.0552) | 0.9384 (0.9212 to 0.9540) | 0.9372 (0.9198 to 0.9532) |
| `s8_ei_2` | 0.2098 (0.1670 to 0.2605) | 0.1054 (0.0820 to 0.1327) | 0.0188 (0.0140 to 0.0331) | 0.9368 (0.9193 to 0.9524) | 0.9358 (0.9179 to 0.9516) |
| `s8_ei_2_temperature` | 0.2098 (0.1670 to 0.2605) | 0.1054 (0.0820 to 0.1327) | 0.0188 (0.0140 to 0.0331) | 0.9368 (0.9193 to 0.9524) | 0.9358 (0.9179 to 0.9516) |
| `s8_ei_2-raw` | 0.2468 (0.1831 to 0.3247) | 0.1098 (0.0834 to 0.1391) | 0.0406 (0.0295 to 0.0578) | 0.9368 (0.9193 to 0.9524) | 0.9358 (0.9179 to 0.9516) |
| `s8_ei_3` | 0.1998 (0.1577 to 0.2489) | 0.0991 (0.0768 to 0.1253) | 0.0137 (0.0112 to 0.0295) | 0.9384 (0.9210 to 0.9532) | 0.9379 (0.9203 to 0.9529) |
| `s8_ei_3_temperature` | 0.1998 (0.1577 to 0.2489) | 0.0991 (0.0768 to 0.1253) | 0.0137 (0.0112 to 0.0295) | 0.9384 (0.9210 to 0.9532) | 0.9379 (0.9203 to 0.9529) |
| `s8_ei_3-raw` | 0.2232 (0.1647 to 0.2894) | 0.1021 (0.0778 to 0.1304) | 0.0365 (0.0256 to 0.0523) | 0.9384 (0.9210 to 0.9532) | 0.9379 (0.9203 to 0.9529) |
| `gemma4_31b` | 0.4078 (0.3373 to 0.4804) | 0.2040 (0.1671 to 0.2418) | 0.0283 (0.0221 to 0.0560) | 0.8711 (0.8450 to 0.8949) | 0.8626 (0.8333 to 0.8879) |
| `gemma4_31b_temperature` | 0.4424 (0.3761 to 0.5095) | 0.2205 (0.1823 to 0.2622) | 0.0321 (0.0248 to 0.0561) | 0.8695 (0.8430 to 0.8940) | 0.8608 (0.8315 to 0.8874) |
| `gemma4_31b-raw` | 2.2911 (1.8275 to 2.7759) | 0.2615 (0.2124 to 0.3148) | 0.1317 (0.1069 to 0.1587) | 0.8695 (0.8430 to 0.8940) | 0.8608 (0.8315 to 0.8874) |

### Table 6. Evidence Inference, each answer's F1 (test split)

| System | F1, significantly increased | F1, significantly decreased | F1, no significant difference |
|---|---|---|---|
| `prior` | 0.0000 (0.0000 to 0.0000) | 0.0000 (0.0000 to 0.0000) | 0.5927 (0.5564 to 0.6286) |
| `s8_ei` | 0.9358 (0.9152 to 0.9529) | 0.9365 (0.9113 to 0.9577) | 0.9475 (0.9278 to 0.9630) |
| `s8_ei_temperature` | 0.9358 (0.9152 to 0.9529) | 0.9365 (0.9113 to 0.9577) | 0.9475 (0.9278 to 0.9630) |
| `s8_ei-raw` | 0.9358 (0.9152 to 0.9529) | 0.9365 (0.9113 to 0.9577) | 0.9475 (0.9278 to 0.9630) |
| `s8_ei_1` | 0.9312 (0.9103 to 0.9490) | 0.9338 (0.9097 to 0.9561) | 0.9468 (0.9268 to 0.9630) |
| `s8_ei_1_temperature` | 0.9312 (0.9103 to 0.9490) | 0.9338 (0.9097 to 0.9561) | 0.9468 (0.9268 to 0.9630) |
| `s8_ei_1-raw` | 0.9312 (0.9103 to 0.9490) | 0.9338 (0.9097 to 0.9561) | 0.9468 (0.9268 to 0.9630) |
| `s8_ei_2` | 0.9337 (0.9135 to 0.9516) | 0.9316 (0.9059 to 0.9534) | 0.9423 (0.9213 to 0.9584) |
| `s8_ei_2_temperature` | 0.9337 (0.9135 to 0.9516) | 0.9316 (0.9059 to 0.9534) | 0.9423 (0.9213 to 0.9584) |
| `s8_ei_2-raw` | 0.9337 (0.9135 to 0.9516) | 0.9316 (0.9059 to 0.9534) | 0.9423 (0.9213 to 0.9584) |
| `s8_ei_3` | 0.9325 (0.9121 to 0.9505) | 0.9378 (0.9118 to 0.9596) | 0.9435 (0.9237 to 0.9595) |
| `s8_ei_3_temperature` | 0.9325 (0.9121 to 0.9505) | 0.9378 (0.9118 to 0.9596) | 0.9435 (0.9237 to 0.9595) |
| `s8_ei_3-raw` | 0.9325 (0.9121 to 0.9505) | 0.9378 (0.9118 to 0.9596) | 0.9435 (0.9237 to 0.9595) |
| `gemma4_31b` | 0.8455 (0.8088 to 0.8793) | 0.8241 (0.7787 to 0.8627) | 0.9182 (0.8947 to 0.9395) |
| `gemma4_31b_temperature` | 0.8399 (0.8026 to 0.8743) | 0.8248 (0.7785 to 0.8640) | 0.9178 (0.8950 to 0.9388) |
| `gemma4_31b-raw` | 0.8399 (0.8026 to 0.8743) | 0.8248 (0.7785 to 0.8640) | 0.9178 (0.8950 to 0.9388) |

### G1 and the release bar

Every criterion passes (Table 8): s8_ei's ECE is 0.0119 against the limit 0.05; its log loss minus the prior's is -0.8773 (-0.9225 to -0.8281) and minus Gemma 4 31B-it's -0.2084 (-0.2720 to -0.1505), each over 333 articles; and its macro-F1, 0.9399, is not below Gemma 4 31B-it's, 0.8626.

The bar's outcome, as compare_session8.py wrote it: "every criterion passes; releasing v0.4 is my decision after the closeout".

### Table 7. G1: s8_ei minus the prior and minus Gemma 4 31B-it

| Comparison | Metric | System | Reference | Difference (95 percent interval) | Reading |
|---|---|---|---|---|---|
| `s8_ei` minus `prior` | Log loss | 0.1994 | 1.0767 | -0.8773 (-0.9201 to -0.8274) | s8_ei better |
| `s8_ei` minus `prior` | Brier score | 0.1001 | 0.6525 | -0.5524 (-0.5767 to -0.5247) | s8_ei better |
| `s8_ei` minus `prior` | ECE | 0.0119 | 0.0305 | -0.0186 (-0.0461 to 0.0187) | no difference was detected; excludes below -0.0461 and above 0.0187 |
| `s8_ei` minus `prior` | Accuracy | 0.9409 | 0.4212 | 0.5197 (0.4790 to 0.5541) | s8_ei better |
| `s8_ei` minus `prior` | Macro-F1 | 0.9399 | 0.1976 | 0.7423 (0.7217 to 0.7602) | s8_ei better |
| `s8_ei` minus `prior` | F1, significantly increased | 0.9358 | 0.0000 | 0.9358 (0.9152 to 0.9529) | s8_ei better |
| `s8_ei` minus `prior` | F1, significantly decreased | 0.9365 | 0.0000 | 0.9365 (0.9113 to 0.9577) | s8_ei better |
| `s8_ei` minus `prior` | F1, no significant difference | 0.9475 | 0.5927 | 0.3547 (0.3208 to 0.3874) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | Log loss | 0.1994 | 0.4078 | -0.2084 (-0.2690 to -0.1498) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | Brier score | 0.1001 | 0.2040 | -0.1039 (-0.1357 to -0.0741) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | ECE | 0.0119 | 0.0283 | -0.0164 (-0.0369 to -0.0007) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | Accuracy | 0.9409 | 0.8711 | 0.0698 (0.0487 to 0.0923) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | Macro-F1 | 0.9399 | 0.8626 | 0.0773 (0.0543 to 0.1022) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | F1, significantly increased | 0.9358 | 0.8455 | 0.0903 (0.0588 to 0.1243) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | F1, significantly decreased | 0.9365 | 0.8241 | 0.1124 (0.0771 to 0.1521) | s8_ei better |
| `s8_ei` minus `gemma4_31b` | F1, no significant difference | 0.9475 | 0.9182 | 0.0293 (0.0120 to 0.0476) | s8_ei better |

### Table 8. The release bar (criteria 2 and 3 with the bar's article resamples)

| Criterion | Measure | s8_ei, or the difference | Limit, 95 percent interval, or Gemma 4 31B-it | Result |
|---|---|---|---|---|
| 1 | ECE | 0.0119 | 0.0500 | PASS |
| 2 | Log loss, s8_ei minus the prior | -0.8773 | -0.9225 to -0.8281 | PASS |
| 3 | Log loss, s8_ei minus Gemma 4 31B-it | -0.2084 | -0.2720 to -0.1505 | PASS |
| 4 | Macro-F1, s8_ei against Gemma 4 31B-it | 0.9399 | 0.8626 | PASS |
| The bar | all four |  |  | PASS |

### G2: the three runs

Each run calibrated by its own rule had a test log loss of 0.2085, 0.2098 and 0.1998: a spread, largest minus smallest, of 0.0100 and a standard deviation of 0.0054, the run-to-run noise of this recipe from data order and dropout (descriptive). s8_ei minus each run: Table 9.

Each EI seed ran once and the three are reported apart as well as averaged; s8_a3 and s7_a3 differ in the keep rule and in the run itself, so s8_a3 minus s7_a3 cannot separate the two; retraining alone moved PubMedQA log loss by about 0.023 per run in this project (results/fiorillo_v0_2/retry_check.json).

### Table 9. G2: s8_ei minus each run alone (descriptive)

| Comparison | Metric | Difference (95 percent interval) | Reading |
|---|---|---|---|
| `s8_ei` minus `s8_ei_1` | Log loss | -0.0091 (-0.0151 to -0.0030) | s8_ei better |
| `s8_ei` minus `s8_ei_1` | Brier score | -0.0048 (-0.0088 to -0.0013) | s8_ei better |
| `s8_ei` minus `s8_ei_1` | ECE | -0.0019 (-0.0100 to 0.0095) | no difference was detected; excludes below -0.0100 and above 0.0095 |
| `s8_ei` minus `s8_ei_1` | Accuracy | 0.0025 (-0.0034 to 0.0093) | no difference was detected; excludes below -0.0034 and above 0.0093 |
| `s8_ei` minus `s8_ei_1` | Macro-F1 | 0.0027 (-0.0038 to 0.0104) | no difference was detected; excludes below -0.0038 and above 0.0104 |
| `s8_ei` minus `s8_ei_1` | F1, significantly increased | 0.0046 (-0.0030 to 0.0126) | no difference was detected; excludes below -0.0030 and above 0.0126 |
| `s8_ei` minus `s8_ei_1` | F1, significantly decreased | 0.0027 (-0.0090 to 0.0156) | no difference was detected; excludes below -0.0090 and above 0.0156 |
| `s8_ei` minus `s8_ei_1` | F1, no significant difference | 0.0007 (-0.0051 to 0.0067) | no difference was detected; excludes below -0.0051 and above 0.0067 |
| `s8_ei` minus `s8_ei_2` | Log loss | -0.0104 (-0.0178 to -0.0030) | s8_ei better |
| `s8_ei` minus `s8_ei_2` | Brier score | -0.0053 (-0.0097 to -0.0013) | s8_ei better |
| `s8_ei` minus `s8_ei_2` | ECE | -0.0068 (-0.0137 to 0.0085) | no difference was detected; excludes below -0.0137 and above 0.0085 |
| `s8_ei` minus `s8_ei_2` | Accuracy | 0.0041 (-0.0015 to 0.0099) | no difference was detected; excludes below -0.0015 and above 0.0099 |
| `s8_ei` minus `s8_ei_2` | Macro-F1 | 0.0041 (-0.0017 to 0.0099) | no difference was detected; excludes below -0.0017 and above 0.0099 |
| `s8_ei` minus `s8_ei_2` | F1, significantly increased | 0.0021 (-0.0045 to 0.0088) | no difference was detected; excludes below -0.0045 and above 0.0088 |
| `s8_ei` minus `s8_ei_2` | F1, significantly decreased | 0.0049 (-0.0051 to 0.0145) | no difference was detected; excludes below -0.0051 and above 0.0145 |
| `s8_ei` minus `s8_ei_2` | F1, no significant difference | 0.0051 (-0.0009 to 0.0112) | no difference was detected; excludes below -0.0009 and above 0.0112 |
| `s8_ei` minus `s8_ei_3` | Log loss | -0.0004 (-0.0073 to 0.0062) | no difference was detected; excludes below -0.0073 and above 0.0062 |
| `s8_ei` minus `s8_ei_3` | Brier score | 0.0010 (-0.0028 to 0.0046) | no difference was detected; excludes below -0.0028 and above 0.0046 |
| `s8_ei` minus `s8_ei_3` | ECE | -0.0018 (-0.0115 to 0.0119) | no difference was detected; excludes below -0.0115 and above 0.0119 |
| `s8_ei` minus `s8_ei_3` | Accuracy | 0.0025 (-0.0025 to 0.0079) | no difference was detected; excludes below -0.0025 and above 0.0079 |
| `s8_ei` minus `s8_ei_3` | Macro-F1 | 0.0020 (-0.0033 to 0.0076) | no difference was detected; excludes below -0.0033 and above 0.0076 |
| `s8_ei` minus `s8_ei_3` | F1, significantly increased | 0.0033 (-0.0042 to 0.0115) | no difference was detected; excludes below -0.0042 and above 0.0115 |
| `s8_ei` minus `s8_ei_3` | F1, significantly decreased | -0.0014 (-0.0085 to 0.0057) | no difference was detected; excludes below -0.0085 and above 0.0057 |
| `s8_ei` minus `s8_ei_3` | F1, no significant difference | 0.0040 (-0.0011 to 0.0097) | no difference was detected; excludes below -0.0011 and above 0.0097 |

### G3: published results

s8_ei's macro-F1, 0.9399 (0.9225 to 0.9544), is above each published value G3 names, and its interval excludes each (Table 10; descriptive, since the published systems were scored and counted differently). Scored leniently, as Buchmann et al. scored (a prompt's gold label is replaced by the prediction when the prediction is one of its accepted labels), it is 0.9407: the rule changed the gold label of 1 prompt, since only 3 test prompts have accepted labels that disagree.

BioDecision-4B's 70.6 was measured, by its model card, on the bigbio copy of EI, whose context is one evidence span and in which 73.4 percent of rows carry another prompt's span (the check of 30 September), so it is named and not compared. The published results counted the 6 test prompts of the empty article, which are left out here.

What the published papers say about these figures, checked at their sources, is in the section after this one.

### Table 10. G3: published results (descriptive)

| Published system | Scoring | Its macro-F1 | s8_ei's macro-F1 (95 percent interval) | The published value against s8_ei's interval |
|---|---|---|---|---|
| DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa) | strict | 0.780 | 0.9399 (0.9225 to 0.9544) | excluded: below the interval |
| DeYoung et al. 2020, with the right evidence given | strict | 0.851 | 0.9399 (0.9225 to 0.9544) | excluded: below the interval |
| Buchmann et al. 2024, LED, the lower end of its range | lenient | 0.7430 | 0.9407 (0.9240 to 0.9551) | excluded: below the interval |
| Buchmann et al. 2024, LED, the upper end of its range | lenient | 0.7649 | 0.9407 (0.9240 to 0.9551) | excluded: below the interval |
| Buchmann et al. 2024, LongT5 | lenient | 0.8154 | 0.9407 (0.9240 to 0.9551) | excluded: below the interval |

### PubMedQA

v0.4 answers PubMedQA with s7_b3 and its session 7 calibration: log loss 0.5797 (0.5070 to 0.6509) and accuracy 0.7960 (0.7600 to 0.8340) (Table 11).

The load check: s7_b3, loaded again on the pod as `s7_b3_timing`, gave raw test probabilities whose largest difference from session 7's file is 0.0101 in one probability, more than the code's note expects from padding alone. Its raw test accuracy, log loss and ECE, 0.7860, 0.6710 and 0.0767, are session 7's `s7_b3-raw` figures in Table 11, which is consistent with the model having loaded as it was.

### Table 11. PubMedQA, three answers (test split)

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 1.1593 (1.0401 to 1.2920) | 0.5855 (0.5503 to 0.6208) | 0.0646 (0.0226 to 0.1086) | 0.5520 (0.5080 to 0.5940) | 0.2371 (0.2246 to 0.2484) |
| `arm_laya` | 0.7428 (0.6698 to 0.8164) | 0.4138 (0.3686 to 0.4582) | 0.0276 (0.0301 to 0.0818) | 0.7140 (0.6720 to 0.7560) | 0.5005 (0.4744 to 0.5258) |
| `s7_a3` | 0.6506 (0.5731 to 0.7302) | 0.3584 (0.3119 to 0.4061) | 0.0462 (0.0371 to 0.0905) | 0.7540 (0.7160 to 0.7920) | 0.5293 (0.5049 to 0.5514) |
| `s7_a3_temperature` | 0.6628 (0.5880 to 0.7442) | 0.3654 (0.3184 to 0.4145) | 0.0557 (0.0423 to 0.0983) | 0.7520 (0.7140 to 0.7900) | 0.5746 (0.5274 to 0.6210) |
| `s7_b3` | 0.5797 (0.5070 to 0.6509) | 0.3147 (0.2683 to 0.3595) | 0.0401 (0.0315 to 0.0800) | 0.7960 (0.7600 to 0.8340) | 0.5698 (0.5406 to 0.6039) |
| `s7_b3_temperature` | 0.6044 (0.5336 to 0.6767) | 0.3277 (0.2837 to 0.3711) | 0.0401 (0.0345 to 0.0820) | 0.7860 (0.7500 to 0.8220) | 0.6223 (0.5780 to 0.6688) |
| `s7_b3-raw` | 0.6710 (0.5587 to 0.7837) | 0.3361 (0.2817 to 0.3887) | 0.0767 (0.0592 to 0.1162) | 0.7860 (0.7500 to 0.8220) | 0.6223 (0.5780 to 0.6688) |
| `s8_a3` | 0.6073 (0.5350 to 0.6805) | 0.3250 (0.2799 to 0.3684) | 0.0571 (0.0448 to 0.0993) | 0.7900 (0.7540 to 0.8260) | 0.5567 (0.5354 to 0.5779) |
| `s8_a3_temperature` | 0.6368 (0.5731 to 0.7028) | 0.3502 (0.3076 to 0.3923) | 0.0537 (0.0417 to 0.0940) | 0.7640 (0.7260 to 0.8040) | 0.6209 (0.5796 to 0.6651) |
| `s8_a3-raw` | 0.7526 (0.6285 to 0.8751) | 0.3696 (0.3134 to 0.4267) | 0.1077 (0.0852 to 0.1479) | 0.7640 (0.7260 to 0.8040) | 0.6209 (0.5796 to 0.6651) |
| `gemma4_31b` | 0.6131 (0.5309 to 0.6957) | 0.3209 (0.2728 to 0.3696) | 0.0575 (0.0431 to 0.0970) | 0.7840 (0.7460 to 0.8200) | 0.5862 (0.5527 to 0.6252) |
| `gemma4_31b_temperature` | 0.6374 (0.5707 to 0.7090) | 0.3450 (0.2999 to 0.3912) | 0.0945 (0.0709 to 0.1328) | 0.7720 (0.7340 to 0.8080) | 0.6273 (0.5865 to 0.6750) |
| `gemma4_31b-raw` | 3.6433 (2.9804 to 4.2976) | 0.4488 (0.3762 to 0.5225) | 0.2226 (0.1880 to 0.2616) | 0.7720 (0.7340 to 0.8080) | 0.6273 (0.5865 to 0.6750) |

### Table 12. G4: PubMedQA (descriptive)

| Comparison | Metric | Difference (95 percent interval) | Reading |
|---|---|---|---|
| `s7_b3` minus `arm_laya` | Log loss | -0.1631 (-0.2082 to -0.1196) | v0.4 better |
| `s7_b3` minus `arm_laya` | Brier score | -0.0991 (-0.1303 to -0.0707) | v0.4 better |
| `s7_b3` minus `arm_laya` | ECE | 0.0126 (-0.0340 to 0.0320) | no difference was detected; excludes below -0.0340 and above 0.0320 |
| `s7_b3` minus `arm_laya` | Accuracy | 0.0820 (0.0500 to 0.1160) | v0.4 better |
| `s7_b3` minus `arm_laya` | Macro-F1 | 0.0692 (0.0398 to 0.1031) | v0.4 better |
| `s7_b3` minus `gemma4_31b` | Log loss | -0.0334 (-0.0785 to 0.0137) | no difference was detected; excludes below -0.0785 and above 0.0137 |
| `s7_b3` minus `gemma4_31b` | Brier score | -0.0062 (-0.0351 to 0.0209) | no difference was detected; excludes below -0.0351 and above 0.0209 |
| `s7_b3` minus `gemma4_31b` | ECE | -0.0174 (-0.0469 to 0.0208) | no difference was detected; excludes below -0.0469 and above 0.0208 |
| `s7_b3` minus `gemma4_31b` | Accuracy | 0.0120 (-0.0140 to 0.0400) | no difference was detected; excludes below -0.0140 and above 0.0400 |
| `s7_b3` minus `gemma4_31b` | Macro-F1 | -0.0164 (-0.0561 to 0.0237) | no difference was detected; excludes below -0.0561 and above 0.0237 |
| `s8_a3` minus `s7_a3` | Log loss | -0.0433 (-0.0752 to -0.0147) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Brier score | -0.0334 (-0.0541 to -0.0140) | s8_a3 better |
| `s8_a3` minus `s7_a3` | ECE | 0.0109 (-0.0292 to 0.0450) | no difference was detected; excludes below -0.0292 and above 0.0450 |
| `s8_a3` minus `s7_a3` | Accuracy | 0.0360 (0.0120 to 0.0600) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Macro-F1 | 0.0274 (0.0096 to 0.0454) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Log loss | -0.1355 (-0.1753 to -0.0905) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Brier score | -0.0889 (-0.1165 to -0.0597) | s8_a3 better |
| `s8_a3` minus `arm_laya` | ECE | 0.0295 (-0.0235 to 0.0513) | no difference was detected; excludes below -0.0235 and above 0.0513 |
| `s8_a3` minus `arm_laya` | Accuracy | 0.0760 (0.0420 to 0.1061) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Macro-F1 | 0.0562 (0.0317 to 0.0790) | s8_a3 better |

### Readmission

The pool's log loss is 0.8275 (0.8132 to 0.8411), against `lightgbm_tuned`'s 0.8347 and `blend_v0_1`'s 0.8340 (Tables 13 to 15; descriptive). Macro-F1 with decision weights multiplies each answer's probability by a weight fitted on validation to maximize macro-F1 (session 4's rule).

### Table 13. Readmission (test split), part one

| System | Log loss | Brier score | ECE | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| `prior` | 0.8861 (0.8734 to 0.8984) | 0.5377 (0.5298 to 0.5453) | 0.0017 (0.0002 to 0.0127) | 0.5942 (0.5829 to 0.6047) | 0.2485 (0.2455 to 0.2512) |
| `arm_laya` | 0.8524 (0.8384 to 0.8662) | 0.5149 (0.5063 to 0.5237) | 0.0169 (0.0101 to 0.0295) | 0.6045 (0.5935 to 0.6160) | 0.2934 (0.2863 to 0.3002) |
| `lightgbm` | 0.8400 (0.8254 to 0.8538) | 0.5059 (0.4969 to 0.5148) | 0.0066 (0.0071 to 0.0214) | 0.6096 (0.5975 to 0.6216) | 0.3494 (0.3396 to 0.3593) |
| `lightgbm_tuned` | 0.8347 (0.8203 to 0.8487) | 0.5026 (0.4937 to 0.5114) | 0.0071 (0.0074 to 0.0225) | 0.6157 (0.6043 to 0.6275) | 0.3569 (0.3477 to 0.3660) |
| `blend_v0_1` | 0.8340 (0.8197 to 0.8479) | 0.5021 (0.4933 to 0.5108) | 0.0074 (0.0078 to 0.0226) | 0.6116 (0.5997 to 0.6227) | 0.3402 (0.3316 to 0.3492) |
| `xgboost_tuned` | 0.8310 (0.8166 to 0.8448) | 0.4997 (0.4907 to 0.5085) | 0.0103 (0.0081 to 0.0241) | 0.6139 (0.6026 to 0.6250) | 0.3748 (0.3625 to 0.3871) |
| `catboost_tuned` | 0.8291 (0.8149 to 0.8429) | 0.4981 (0.4889 to 0.5072) | 0.0078 (0.0085 to 0.0230) | 0.6147 (0.6035 to 0.6262) | 0.3671 (0.3566 to 0.3781) |
| `readmission_pool` | 0.8275 (0.8132 to 0.8411) | 0.4972 (0.4882 to 0.5062) | 0.0066 (0.0085 to 0.0225) | 0.6157 (0.6042 to 0.6269) | 0.3695 (0.3579 to 0.3813) |
| `s7_a3` | 0.8580 (0.8443 to 0.8713) | 0.5184 (0.5098 to 0.5273) | 0.0254 (0.0188 to 0.0377) | 0.6012 (0.5896 to 0.6130) | 0.3347 (0.3265 to 0.3430) |
| `s7_b3` | 0.8605 (0.8466 to 0.8738) | 0.5196 (0.5110 to 0.5280) | 0.0149 (0.0121 to 0.0299) | 0.6025 (0.5907 to 0.6140) | 0.3223 (0.3143 to 0.3307) |
| `s8_a3` | 0.8445 (0.8304 to 0.8585) | 0.5086 (0.4998 to 0.5173) | 0.0150 (0.0104 to 0.0281) | 0.6085 (0.5969 to 0.6199) | 0.3499 (0.3406 to 0.3600) |
| `s8_a3_temperature` | 0.8441 (0.8300 to 0.8580) | 0.5085 (0.4997 to 0.5171) | 0.0183 (0.0110 to 0.0305) | 0.6087 (0.5972 to 0.6202) | 0.3465 (0.3379 to 0.3551) |
| `s8_a3-raw` | 0.8442 (0.8302 to 0.8580) | 0.5086 (0.4999 to 0.5170) | 0.0172 (0.0110 to 0.0298) | 0.6087 (0.5972 to 0.6202) | 0.3465 (0.3379 to 0.3551) |

### Table 14. Readmission (test split), part two

| System | Macro-F1, decision weights | Quadratic weighted kappa | AUROC, 30 days | Calibration intercept, 30 days | Calibration slope, 30 days |
|---|---|---|---|---|---|
| `prior` | 0.2485 (0.2455 to 0.2512) | 0.0000 (0.0000 to 0.0000) | 0.5000 (0.5000 to 0.5000) | -0.0299 (-0.1120 to 0.0536) | not computed |
| `arm_laya` | 0.4261 (0.4125 to 0.4409) | 0.0537 (0.0422 to 0.0657) | 0.6423 (0.6185 to 0.6661) | -0.0957 (-0.1781 to -0.0103) | 1.1883 (1.0064 to 1.3756) |
| `lightgbm` | 0.4417 (0.4285 to 0.4564) | 0.1081 (0.0912 to 0.1262) | 0.6469 (0.6226 to 0.6715) | -0.0306 (-0.1171 to 0.0549) | 0.9781 (0.8240 to 1.1292) |
| `lightgbm_tuned` | 0.4484 (0.4335 to 0.4643) | 0.1200 (0.1029 to 0.1377) | 0.6577 (0.6341 to 0.6807) | -0.0177 (-0.1011 to 0.0680) | 1.0935 (0.9379 to 1.2504) |
| `blend_v0_1` | 0.4520 (0.4370 to 0.4678) | 0.0986 (0.0822 to 0.1153) | 0.6612 (0.6377 to 0.6855) | -0.0100 (-0.0930 to 0.0751) | 1.1752 (1.0133 to 1.3358) |
| `xgboost_tuned` | 0.4551 (0.4410 to 0.4690) | 0.1371 (0.1168 to 0.1583) | 0.6584 (0.6355 to 0.6817) | -0.0338 (-0.1181 to 0.0509) | 0.9969 (0.8635 to 1.1430) |
| `catboost_tuned` | 0.4604 (0.4469 to 0.4749) | 0.1274 (0.1084 to 0.1459) | 0.6581 (0.6347 to 0.6823) | -0.0287 (-0.1120 to 0.0561) | 1.0160 (0.8806 to 1.1574) |
| `readmission_pool` | 0.4632 (0.4488 to 0.4780) | 0.1301 (0.1101 to 0.1491) | 0.6622 (0.6394 to 0.6855) | -0.0192 (-0.1031 to 0.0646) | 1.0493 (0.9108 to 1.1940) |
| `s7_a3` | 0.4213 (0.4091 to 0.4344) | 0.0883 (0.0726 to 0.1064) | 0.6255 (0.6014 to 0.6494) | -0.1138 (-0.1989 to -0.0296) | 1.1156 (0.9211 to 1.3149) |
| `s7_b3` | 0.4183 (0.4054 to 0.4309) | 0.0768 (0.0623 to 0.0923) | 0.6293 (0.6047 to 0.6549) | -0.1246 (-0.2100 to -0.0378) | 0.7651 (0.6200 to 0.9123) |
| `s8_a3` | 0.4380 (0.4249 to 0.4518) | 0.1120 (0.0950 to 0.1307) | 0.6433 (0.6198 to 0.6679) | -0.1174 (-0.2015 to -0.0303) | 0.9539 (0.8097 to 1.0969) |
| `s8_a3_temperature` | 0.4367 (0.4234 to 0.4511) | 0.1106 (0.0933 to 0.1280) | 0.6434 (0.6198 to 0.6677) | -0.0732 (-0.1573 to 0.0134) | 0.9866 (0.8376 to 1.1348) |
| `s8_a3-raw` | 0.4367 (0.4238 to 0.4505) | 0.1106 (0.0933 to 0.1280) | 0.6434 (0.6198 to 0.6678) | -0.0852 (-0.1695 to 0.0013) | 0.9949 (0.8447 to 1.1445) |

### Table 15. G5: readmission (descriptive)

| Comparison | Metric | Difference (95 percent interval) | Reading |
|---|---|---|---|
| `readmission_pool` minus `lightgbm_tuned` | Log loss | -0.0072 (-0.0104 to -0.0042) | the pool better |
| `readmission_pool` minus `lightgbm_tuned` | Brier score | -0.0054 (-0.0075 to -0.0034) | the pool better |
| `readmission_pool` minus `lightgbm_tuned` | ECE | -0.0005 (-0.0092 to 0.0103) | no difference was detected; excludes below -0.0092 and above 0.0103 |
| `readmission_pool` minus `lightgbm_tuned` | Accuracy | 0.0000 (-0.0061 to 0.0059) | no difference was detected; excludes below -0.0061 and above 0.0059 |
| `readmission_pool` minus `lightgbm_tuned` | Macro-F1 | 0.0126 (0.0031 to 0.0218) | the pool better |
| `readmission_pool` minus `lightgbm_tuned` | Macro-F1, decision weights | 0.0148 (0.0051 to 0.0265) | the pool better |
| `readmission_pool` minus `lightgbm_tuned` | Quadratic weighted kappa | 0.0102 (-0.0048 to 0.0240) | no difference was detected; excludes below -0.0048 and above 0.0240 |
| `readmission_pool` minus `lightgbm_tuned` | AUROC, 30 days | 0.0045 (-0.0063 to 0.0160) | no difference was detected; excludes below -0.0063 and above 0.0160 |
| `readmission_pool` minus `lightgbm_tuned` | Calibration intercept, 30 days | -0.0015 (-0.0096 to 0.0058) | not ranked |
| `readmission_pool` minus `lightgbm_tuned` | Calibration slope, 30 days | -0.0443 (-0.1130 to 0.0250) | not ranked |
| `readmission_pool` minus `blend_v0_1` | Log loss | -0.0065 (-0.0094 to -0.0038) | the pool better |
| `readmission_pool` minus `blend_v0_1` | Brier score | -0.0049 (-0.0069 to -0.0030) | the pool better |
| `readmission_pool` minus `blend_v0_1` | ECE | -0.0009 (-0.0086 to 0.0093) | no difference was detected; excludes below -0.0086 and above 0.0093 |
| `readmission_pool` minus `blend_v0_1` | Accuracy | 0.0041 (-0.0014 to 0.0094) | no difference was detected; excludes below -0.0014 and above 0.0094 |
| `readmission_pool` minus `blend_v0_1` | Macro-F1 | 0.0293 (0.0199 to 0.0393) | the pool better |
| `readmission_pool` minus `blend_v0_1` | Macro-F1, decision weights | 0.0112 (0.0021 to 0.0214) | the pool better |
| `readmission_pool` minus `blend_v0_1` | Quadratic weighted kappa | 0.0315 (0.0173 to 0.0456) | the pool better |
| `readmission_pool` minus `blend_v0_1` | AUROC, 30 days | 0.0010 (-0.0087 to 0.0110) | no difference was detected; excludes below -0.0087 and above 0.0110 |
| `readmission_pool` minus `blend_v0_1` | Calibration intercept, 30 days | -0.0092 (-0.0168 to -0.0028) | not ranked |
| `readmission_pool` minus `blend_v0_1` | Calibration slope, 30 days | -0.1259 (-0.1896 to -0.0606) | not ranked |
| `readmission_pool` minus `arm_laya` | Log loss | -0.0248 (-0.0297 to -0.0202) | the pool better |
| `readmission_pool` minus `arm_laya` | Brier score | -0.0177 (-0.0212 to -0.0143) | the pool better |
| `readmission_pool` minus `arm_laya` | ECE | -0.0103 (-0.0158 to 0.0077) | no difference was detected; excludes below -0.0158 and above 0.0077 |
| `readmission_pool` minus `arm_laya` | Accuracy | 0.0113 (0.0037 to 0.0186) | the pool better |
| `readmission_pool` minus `arm_laya` | Macro-F1 | 0.0761 (0.0646 to 0.0875) | the pool better |
| `readmission_pool` minus `arm_laya` | Macro-F1, decision weights | 0.0372 (0.0248 to 0.0494) | the pool better |
| `readmission_pool` minus `arm_laya` | Quadratic weighted kappa | 0.0764 (0.0583 to 0.0927) | the pool better |
| `readmission_pool` minus `arm_laya` | AUROC, 30 days | 0.0198 (0.0043 to 0.0357) | the pool better |
| `readmission_pool` minus `arm_laya` | Calibration intercept, 30 days | 0.0765 (0.0662 to 0.0852) | not ranked |
| `readmission_pool` minus `arm_laya` | Calibration slope, 30 days | -0.1390 (-0.2524 to -0.0325) | not ranked |
| `xgboost_tuned` minus `lightgbm_tuned` | Log loss | -0.0037 (-0.0069 to -0.0008) | xgboost_tuned better |
| `xgboost_tuned` minus `lightgbm_tuned` | Brier score | -0.0029 (-0.0050 to -0.0010) | xgboost_tuned better |
| `xgboost_tuned` minus `lightgbm_tuned` | ECE | 0.0032 (-0.0084 to 0.0108) | no difference was detected; excludes below -0.0084 and above 0.0108 |
| `xgboost_tuned` minus `lightgbm_tuned` | Accuracy | -0.0019 (-0.0079 to 0.0034) | no difference was detected; excludes below -0.0079 and above 0.0034 |
| `xgboost_tuned` minus `lightgbm_tuned` | Macro-F1 | 0.0179 (0.0083 to 0.0280) | xgboost_tuned better |
| `xgboost_tuned` minus `lightgbm_tuned` | Macro-F1, decision weights | 0.0067 (-0.0032 to 0.0168) | no difference was detected; excludes below -0.0032 and above 0.0168 |
| `xgboost_tuned` minus `lightgbm_tuned` | Quadratic weighted kappa | 0.0171 (0.0026 to 0.0307) | xgboost_tuned better |
| `xgboost_tuned` minus `lightgbm_tuned` | AUROC, 30 days | 0.0007 (-0.0102 to 0.0107) | no difference was detected; excludes below -0.0102 and above 0.0107 |
| `xgboost_tuned` minus `lightgbm_tuned` | Calibration intercept, 30 days | -0.0161 (-0.0237 to -0.0088) | not ranked |
| `xgboost_tuned` minus `lightgbm_tuned` | Calibration slope, 30 days | -0.0966 (-0.1635 to -0.0325) | not ranked |
| `catboost_tuned` minus `lightgbm_tuned` | Log loss | -0.0056 (-0.0095 to -0.0020) | catboost_tuned better |
| `catboost_tuned` minus `lightgbm_tuned` | Brier score | -0.0045 (-0.0071 to -0.0021) | catboost_tuned better |
| `catboost_tuned` minus `lightgbm_tuned` | ECE | 0.0007 (-0.0096 to 0.0109) | no difference was detected; excludes below -0.0096 and above 0.0109 |
| `catboost_tuned` minus `lightgbm_tuned` | Accuracy | -0.0010 (-0.0077 to 0.0054) | no difference was detected; excludes below -0.0077 and above 0.0054 |
| `catboost_tuned` minus `lightgbm_tuned` | Macro-F1 | 0.0102 (0.0005 to 0.0196) | catboost_tuned better |
| `catboost_tuned` minus `lightgbm_tuned` | Macro-F1, decision weights | 0.0120 (0.0013 to 0.0235) | catboost_tuned better |
| `catboost_tuned` minus `lightgbm_tuned` | Quadratic weighted kappa | 0.0075 (-0.0077 to 0.0223) | no difference was detected; excludes below -0.0077 and above 0.0223 |
| `catboost_tuned` minus `lightgbm_tuned` | AUROC, 30 days | 0.0004 (-0.0124 to 0.0140) | no difference was detected; excludes below -0.0124 and above 0.0140 |
| `catboost_tuned` minus `lightgbm_tuned` | Calibration intercept, 30 days | -0.0110 (-0.0202 to -0.0019) | not ranked |
| `catboost_tuned` minus `lightgbm_tuned` | Calibration slope, 30 days | -0.0775 (-0.1536 to 0.0039) | not ranked |
| `s8_a3` minus `s7_a3` | Log loss | -0.0136 (-0.0177 to -0.0095) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Brier score | -0.0098 (-0.0128 to -0.0070) | s8_a3 better |
| `s8_a3` minus `s7_a3` | ECE | -0.0105 (-0.0226 to 0.0045) | no difference was detected; excludes below -0.0226 and above 0.0045 |
| `s8_a3` minus `s7_a3` | Accuracy | 0.0073 (0.0003 to 0.0140) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Macro-F1 | 0.0153 (0.0065 to 0.0236) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Macro-F1, decision weights | 0.0167 (0.0065 to 0.0271) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Quadratic weighted kappa | 0.0237 (0.0095 to 0.0385) | s8_a3 better |
| `s8_a3` minus `s7_a3` | AUROC, 30 days | 0.0178 (0.0032 to 0.0318) | s8_a3 better |
| `s8_a3` minus `s7_a3` | Calibration intercept, 30 days | -0.0036 (-0.0106 to 0.0037) | not ranked |
| `s8_a3` minus `s7_a3` | Calibration slope, 30 days | -0.1617 (-0.2544 to -0.0662) | not ranked |
| `s8_a3` minus `arm_laya` | Log loss | -0.0079 (-0.0121 to -0.0043) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Brier score | -0.0062 (-0.0094 to -0.0035) | s8_a3 better |
| `s8_a3` minus `arm_laya` | ECE | -0.0019 (-0.0151 to 0.0148) | no difference was detected; excludes below -0.0151 and above 0.0148 |
| `s8_a3` minus `arm_laya` | Accuracy | 0.0040 (-0.0031 to 0.0110) | no difference was detected; excludes below -0.0031 and above 0.0110 |
| `s8_a3` minus `arm_laya` | Macro-F1 | 0.0566 (0.0470 to 0.0657) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Macro-F1, decision weights | 0.0119 (0.0017 to 0.0225) | s8_a3 better |
| `s8_a3` minus `arm_laya` | Quadratic weighted kappa | 0.0583 (0.0432 to 0.0735) | s8_a3 better |
| `s8_a3` minus `arm_laya` | AUROC, 30 days | 0.0009 (-0.0131 to 0.0154) | no difference was detected; excludes below -0.0131 and above 0.0154 |
| `s8_a3` minus `arm_laya` | Calibration intercept, 30 days | -0.0217 (-0.0297 to -0.0142) | not ranked |
| `s8_a3` minus `arm_laya` | Calibration slope, 30 days | -0.2344 (-0.3331 to -0.1413) | not ranked |

### G6: the hand-off

On EI the combined answers' accuracy was 0.9409 with no hand-off and 0.9294, 0.9163, 0.8998 and 0.8785 with 10, 20, 30 and 50 percent handed off; on the handed-off items s8_ei's accuracy was 0.6942 and Gemma 4 31B-it's 0.5785 at 10 percent (Table 16; descriptive, and the rates were not compared with each other).

On PubMedQA it was 0.7960 with no hand-off and 0.7960 with 50 percent handed off.

At $3.51 an hour, a thousand EI questions cost $0.53 of GPU time with no hand-off and $1.32 with 50 percent handed off (derived), from the GPU seconds an item measured in testpreds, each job alone: s8_ei 0.5466 (its three runs' sum) and Gemma 4 31B-it 1.6198. The specialist is counted on every question and the large model on the handed-off share. A thousand PubMedQA questions cost $0.02 and $0.41 (`s7_b3_timing` 0.0200 and Gemma 4 31B-it 0.8040 seconds an item).

The large model's seconds an item come from its second process's time shared between the two questions by item count, as the code does; the Method gives what a share by prompt tokens would make them (derived).

handoff_test.json is the file comparison.json names; the figure check compares their SHA-256.

### Table 16. G6: the hand-off on the test split (secondary, descriptive)

| Question | Percent handed off | Items handed off | Accuracy | Macro-F1 | Log loss | Specialist's accuracy on the handed-off items | Gemma 4 31B-it's accuracy on them | Cost per 1,000 questions |
|---|---|---|---|---|---|---|---|---|
| Evidence Inference | 0 | 0 | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) | 0.1994 (0.1573 to 0.2486) |  |  | $0.53 |
| Evidence Inference | 10 | 121 | 0.9294 (0.9102 to 0.9447) | 0.9264 (0.9059 to 0.9429) | 0.2374 (0.1910 to 0.2910) | 0.6942 (0.6186 to 0.7900) | 0.5785 (0.4746 to 0.6829) | $0.69 |
| Evidence Inference | 20 | 243 | 0.9163 (0.8985 to 0.9322) | 0.9134 (0.8941 to 0.9302) | 0.2704 (0.2223 to 0.3198) | 0.7901 (0.7266 to 0.8452) | 0.6667 (0.5958 to 0.7317) | $0.85 |
| Evidence Inference | 30 | 365 | 0.8998 (0.8782 to 0.9192) | 0.8954 (0.8722 to 0.9152) | 0.3042 (0.2478 to 0.3631) | 0.8356 (0.7896 to 0.8781) | 0.6986 (0.6342 to 0.7586) | $1.01 |
| Evidence Inference | 50 | 609 | 0.8785 (0.8552 to 0.9014) | 0.8710 (0.8444 to 0.8947) | 0.3593 (0.2976 to 0.4208) | 0.8883 (0.8583 to 0.9144) | 0.7635 (0.7187 to 0.8077) | $1.32 |
| Evidence Inference | Gemma 4 31B-it alone |  | 0.8711 (0.8450 to 0.8949) | 0.8626 (0.8333 to 0.8879) | 0.4078 (0.3373 to 0.4804) |  |  |  |
| PubMedQA | 0 | 0 | 0.7960 (0.7600 to 0.8340) | 0.5698 (0.5406 to 0.6039) | 0.5797 (0.5070 to 0.6509) |  |  | $0.02 |
| PubMedQA | 10 | 50 | 0.8000 (0.7620 to 0.8360) | 0.5625 (0.5411 to 0.5828) | 0.5774 (0.5070 to 0.6525) | 0.5600 (0.4200 to 0.7000) | 0.6000 (0.4600 to 0.7200) | $0.10 |
| PubMedQA | 20 | 100 | 0.7960 (0.7600 to 0.8340) | 0.5729 (0.5465 to 0.6056) | 0.5908 (0.5051 to 0.6638) | 0.5800 (0.4800 to 0.6800) | 0.5800 (0.4900 to 0.6800) | $0.18 |
| PubMedQA | 30 | 150 | 0.7940 (0.7580 to 0.8320) | 0.5739 (0.5470 to 0.6039) | 0.5833 (0.5036 to 0.6632) | 0.6000 (0.5200 to 0.6800) | 0.5933 (0.5133 to 0.6668) | $0.25 |
| PubMedQA | 50 | 250 | 0.7960 (0.7600 to 0.8340) | 0.5932 (0.5591 to 0.6325) | 0.5921 (0.5081 to 0.6750) | 0.6680 (0.6040 to 0.7360) | 0.6680 (0.6040 to 0.7320) | $0.41 |
| PubMedQA | Gemma 4 31B-it alone |  | 0.7840 (0.7460 to 0.8200) | 0.5862 (0.5527 to 0.6252) | 0.6131 (0.5309 to 0.6957) |  |  |  |

On validation the hand-off ran on every EI validation prompt and on PubMedQA's calibrate items (Table 17). Its predictions are calibrated on the items it scores, so its figures are in sample. PubMedQA's validation cost is not computed, since no session 8 run of s7_b3 read the validation split (correction 4).

handoff_validation.json's note allows that the seeds' validation passes shared the GPU with arm A; by the stage logs none did, since arm A had finished at 16:57:23. Its EI costs differ from the test's mainly through the large model's seconds an item.

### Table 17. The hand-off on validation (secondary, descriptive)

| Question | Percent handed off | Items handed off | Accuracy | Macro-F1 | Log loss | Specialist's accuracy on the handed-off items | Gemma 4 31B-it's accuracy on them | Cost per 1,000 questions |
|---|---|---|---|---|---|---|---|---|
| Evidence Inference | 0 | 0 | 0.9369 (0.9206 to 0.9515) | 0.9332 (0.9160 to 0.9491) | 0.1987 (0.1572 to 0.2460) |  |  | $0.53 |
| Evidence Inference | 10 | 123 | 0.9369 (0.9218 to 0.9525) | 0.9314 (0.9139 to 0.9477) | 0.2108 (0.1657 to 0.2611) | 0.6748 (0.5785 to 0.7619) | 0.6748 (0.5691 to 0.7787) | $0.72 |
| Evidence Inference | 20 | 247 | 0.9192 (0.9028 to 0.9345) | 0.9114 (0.8934 to 0.9281) | 0.2557 (0.2068 to 0.3084) | 0.7652 (0.7029 to 0.8193) | 0.6761 (0.6064 to 0.7399) | $0.90 |
| Evidence Inference | 30 | 371 | 0.9062 (0.8862 to 0.9222) | 0.8970 (0.8764 to 0.9142) | 0.2702 (0.2232 to 0.3281) | 0.8113 (0.7628 to 0.8568) | 0.7089 (0.6489 to 0.7609) | $1.09 |
| Evidence Inference | 50 | 618 | 0.8925 (0.8716 to 0.9139) | 0.8816 (0.8578 to 0.9055) | 0.2994 (0.2443 to 0.3578) | 0.8770 (0.8467 to 0.9062) | 0.7880 (0.7460 to 0.8320) | $1.46 |
| Evidence Inference | Gemma 4 31B-it alone |  | 0.8860 (0.8607 to 0.9095) | 0.8747 (0.8461 to 0.8996) | 0.3403 (0.2826 to 0.4036) |  |  |  |
| PubMedQA | 0 | 0 | 0.7867 (0.7200 to 0.8467) | 0.5455 (0.4994 to 0.5842) | 0.5603 (0.4324 to 0.6997) |  |  | not computed |
| PubMedQA | 10 | 15 | 0.8200 (0.7533 to 0.8800) | 0.5709 (0.5283 to 0.6044) | 0.5486 (0.4087 to 0.6968) | 0.3333 (0.1333 to 0.5333) | 0.6667 (0.4000 to 0.9333) | not computed |
| PubMedQA | 20 | 30 | 0.8133 (0.7533 to 0.8733) | 0.5737 (0.5365 to 0.6097) | 0.5631 (0.4141 to 0.7028) | 0.4333 (0.2658 to 0.6000) | 0.5667 (0.4000 to 0.7333) | not computed |
| PubMedQA | 30 | 45 | 0.8000 (0.7467 to 0.8667) | 0.5683 (0.5341 to 0.6057) | 0.5663 (0.4165 to 0.7119) | 0.5111 (0.3556 to 0.6667) | 0.5556 (0.4439 to 0.7111) | not computed |
| PubMedQA | 50 | 75 | 0.7933 (0.7332 to 0.8535) | 0.5672 (0.5286 to 0.6010) | 0.5769 (0.4251 to 0.7298) | 0.6533 (0.5467 to 0.7733) | 0.6667 (0.5600 to 0.7733) | not computed |
| PubMedQA | Gemma 4 31B-it alone |  | 0.7800 (0.7133 to 0.8467) | 0.5635 (0.5247 to 0.5980) | 0.5934 (0.4470 to 0.7488) |  |  |  |

### Sensitivity, power and printed zeros

Without the 6 test prompts that EI's README flags, and with s8_ei's raw or temperature-only predictions, the G1 differences are in Table 18 (descriptive).

### Table 18. Sensitivity: G1 without the flagged test prompts, and with s8_ei's raw and temperature-only predictions (descriptive)

| Prompts | Comparison | Metric | Difference (95 percent interval) | Reading |
|---|---|---|---|---|
| without the flagged prompts | `s8_ei` minus `prior` | Log loss | -0.8764 (-0.9209 to -0.8293) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | Brier score | -0.5518 (-0.5772 to -0.5249) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | ECE | -0.0175 (-0.0453 to 0.0180) | no difference was detected; excludes below -0.0453 and above 0.0180 |
| without the flagged prompts | `s8_ei` minus `prior` | Accuracy | 0.5182 (0.4753 to 0.5558) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | Macro-F1 | 0.7416 (0.7208 to 0.7606) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | F1, significantly increased | 0.9352 (0.9150 to 0.9526) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | F1, significantly decreased | 0.9362 (0.9118 to 0.9573) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `prior` | F1, no significant difference | 0.3534 (0.3159 to 0.3878) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | Log loss | -0.2097 (-0.2758 to -0.1512) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | Brier score | -0.1046 (-0.1377 to -0.0748) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | ECE | -0.0169 (-0.0357 to -0.0005) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | Accuracy | 0.0701 (0.0489 to 0.0941) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | Macro-F1 | 0.0778 (0.0554 to 0.1027) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | F1, significantly increased | 0.0913 (0.0584 to 0.1265) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | F1, significantly decreased | 0.1127 (0.0778 to 0.1518) | s8_ei better |
| without the flagged prompts | `s8_ei` minus `gemma4_31b` | F1, no significant difference | 0.0293 (0.0126 to 0.0480) | s8_ei better |
| all prompts | `s8_ei-raw` minus `prior` | Log loss | -0.8530 (-0.9128 to -0.7831) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | Brier score | -0.5501 (-0.5761 to -0.5212) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | ECE | 0.0019 (-0.0308 to 0.0369) | no difference was detected; excludes below -0.0308 and above 0.0369 |
| all prompts | `s8_ei-raw` minus `prior` | Accuracy | 0.5197 (0.4790 to 0.5541) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | Macro-F1 | 0.7423 (0.7217 to 0.7602) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | F1, significantly increased | 0.9358 (0.9152 to 0.9529) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | F1, significantly decreased | 0.9365 (0.9113 to 0.9577) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `prior` | F1, no significant difference | 0.3547 (0.3208 to 0.3874) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | Log loss | -0.1841 (-0.2512 to -0.1164) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | Brier score | -0.1016 (-0.1333 to -0.0716) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | ECE | 0.0042 (-0.0208 to 0.0170) | no difference was detected; excludes below -0.0208 and above 0.0170 |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | Accuracy | 0.0698 (0.0487 to 0.0923) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | Macro-F1 | 0.0773 (0.0543 to 0.1022) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | F1, significantly increased | 0.0903 (0.0588 to 0.1243) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | F1, significantly decreased | 0.1124 (0.0771 to 0.1521) | s8_ei-raw better |
| all prompts | `s8_ei-raw` minus `gemma4_31b` | F1, no significant difference | 0.0293 (0.0120 to 0.0476) | s8_ei-raw better |
| all prompts | `s8_ei_temperature` minus `prior` | Log loss | -0.8773 (-0.9201 to -0.8274) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | Brier score | -0.5524 (-0.5767 to -0.5247) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | ECE | -0.0186 (-0.0461 to 0.0187) | no difference was detected; excludes below -0.0461 and above 0.0187 |
| all prompts | `s8_ei_temperature` minus `prior` | Accuracy | 0.5197 (0.4790 to 0.5541) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | Macro-F1 | 0.7423 (0.7217 to 0.7602) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | F1, significantly increased | 0.9358 (0.9152 to 0.9529) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | F1, significantly decreased | 0.9365 (0.9113 to 0.9577) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `prior` | F1, no significant difference | 0.3547 (0.3208 to 0.3874) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | Log loss | -0.2084 (-0.2690 to -0.1498) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | Brier score | -0.1039 (-0.1357 to -0.0741) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | ECE | -0.0164 (-0.0369 to -0.0007) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | Accuracy | 0.0698 (0.0487 to 0.0923) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | Macro-F1 | 0.0773 (0.0543 to 0.1022) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | F1, significantly increased | 0.0903 (0.0588 to 0.1243) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | F1, significantly decreased | 0.1124 (0.0771 to 0.1521) | s8_ei_temperature better |
| all prompts | `s8_ei_temperature` minus `gemma4_31b` | F1, no significant difference | 0.0293 (0.0120 to 0.0476) | s8_ei_temperature better |

Power, derived before any prediction (power.json): at a per-prompt standard deviation of 0.8 and a design effect of 1.5, criterion 3's difference had a standard error of 0.0281, a pass line of -0.0550, and an 80 percent chance of passing at a real gain of 0.0787. Measured on validation, for the record only (power_validation.json): s 0.6975, design effect 1.6465, and s8_ei minus Gemma 4 31B-it -0.1417 in log loss.

No prediction row of any system, split or question has a true answer whose probability prints as 0: 0 of 247,210 rows (zeros.json), and 0 prediction files were missing.

### Known contamination

Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PMC articles (EI's articles are PMC's), EI's repository (public since 2019), PubMedQA and the UCI data, which would favor both; EI's test labels are public (the project's site serves them to its leaderboard), so no evaluation of EI can be blind; Laya's, ModernBERT's and reader v1's notes and session 4's notes on Duan stand.

The checks after the Results (correction 14) measure part of this; they cannot reach the pretraining text.

## Published results beside G3, checked at their sources

The figures in this section come from the publications, read on 30 September and 1 October 2026 and recorded in docs/prior_art.md (the check of 1 October 2026), where Claude re-opened each paper, model card and code file named here except where a line says otherwise; they are not this study's results, and the figure check does not cover them. G3 stands as registered, and these facts stand beside it.

- DeYoung et al. 2020, "Evidence Inference 2.0: More Data, Better Models" (BioNLP 2020), Table 1, macro-averaged over the three answers, on what the paper calls the augmented test set: the Biomed RoBERTa pipeline conditioned on the prompt has an F of 0.780, the same classifier given the human-selected evidence span ("Oracle Spans") 0.851, and a classifier given the prompt alone 0.511. Its Table 13 counts 10,150 training, 1,238 validation and 1,228 test prompts, in 2,672, 340 and 334 articles. At the fixed commit the plan's check of 30 September found 1,228 test prompts with any annotation row and 1,224 with an accepted one, the 6 prompts of the empty article among them; that count is the basis of G3's statement that the published results counted those 6 prompts, and the paper does not mention the empty article. Buchmann et al.'s EACL paper does not state its count of test prompts; the LAB benchmark's 1,218 test instances equal this study's count. The paper does not say how disagreeing annotations become one label.
- Buchmann, Eichler, Bodensohn, Kuznetsov and Gurevych 2024, "Document Structure in Long-Document Transformers" (EACL 2024), Table 3: test-set F1 averaged over 3 seeds, for base-size models given 15,000 steps of further pretraining on EI's documents and then fine-tuned on EI. G3's LED range, "74.30 to 76.49", runs from LED without structure infusion (74.30) to its best setting (76.49); across its ten settings LED's lowest is 73.78. LongT5's 81.54 is its run without structure infusion, which is also its best. The paper scores a prompt with several annotations by the annotation that gives the highest score, and it describes the original evaluation as taking the most often annotated class as the gold label. Its code (UKPLab/eacl2024-doc-structure, infusion/evaluation/tasks/evidence_inference.py) gives every annotation the same score when the prediction matches none of them and then takes the first one stored, where compare_session8.py keeps the prompt's own answer, the most common accepted label. The authors note that their numbers can only be compared among their own models (section 5.1).
- Buchmann, Liu and Gurevych 2024, "Attribute or Abstain: Large Language Models as Long Document Assistants" (the LAB benchmark; EMNLP 2024), Table 3: on EI 2.0's test set (1,218 instances by its Table 1), classification macro-F1 of 86 for a fine-tuned Flan-T5-XL answering from the whole article and 87 from ten retrieved segments, and 87 for GPT-4, on 100 test instances only. These are single runs, and how prompts with several labels were scored is not stated. They are the best full-article results found, and they are not part of the registered G3.
- BioDecision-4B (lighteternal/biodecision-tev1-4b, a model card titled BioDecision-4B, September 2026): a macro-F1 of 70.6 on 1,216 test prompts of EI, measured on the BigBio copy, whose context is one evidence span (the check of 30 September); PubMedQA's expert-labeled accuracy 76.6 on 500; a research-only license on the card.
- A research agent's search found no published EI result that reports calibration (ECE, Brier score or log loss); the search was short and is not exhaustive (UNVERIFIED). s8_ei's log loss and ECE therefore have no published comparison here.

## Checks not pre-registered (corrections 14 and 15)

Corrections 14 and 15 added these checks after the results, because s8_ei's macro-F1 is above each published EI result found (descriptive). They were not registered, they have no pass or fail, and they change no figure above.

src/eval/contamination_checks_session8.py wrote them to results/fiorillo_v0_4/contamination_checks.json, first recounting the builder's validation coverage, which matched article_selection.json (1,151 of 1,206).

Reading the evidence. On the test split the selection kept an accepted evidence span inside the input for 1,153 of the 1,212 prompts with usable offsets (95.13 percent). s8_ei's accuracy is 0.9488 (0.9335 to 0.9622) where an annotated evidence span lies inside the input and 0.7966 (0.6571 to 0.9260) on the 59 prompts in 25 articles where none does, though the result may still be stated elsewhere in the input; Gemma 4 31B-it's is 0.8829 and 0.6610. On validation s8_ei's is 0.9461 and 0.7273 (55 prompts outside).

Outside minus inside (correction 15), s8_ei's accuracy differs by -0.1522 (-0.2967 to -0.0327) on the test split and -0.2189 (-0.4055 to -0.0666) on validation, and Gemma 4 31B-it's by -0.2219 (-0.4175 to -0.0783) and -0.1650 (-0.3469 to -0.0103) (Table 22).

Each interval lies below zero: both models are less accurate where no annotated evidence span lies inside the input. This is consistent with their reading the evidence; it is also consistent with those prompts being harder for any reader, and it does not rule out that a model recalls some answers. The groups are small, and their intervals are wide.

### Table 20. Accuracy where the evidence is inside the input and where it is not (correction 14; descriptive, not pre-registered)

| Split | System | Prompts | Number | Articles | Accuracy (95 percent interval) | Macro-F1 (95 percent interval) |
|---|---|---|---|---|---|---|
| test | s8_ei | all | 1,218 | 333 | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| test | s8_ei | evidence inside the input | 1,153 | 328 | 0.9488 (0.9335 to 0.9622) | 0.9477 (0.9325 to 0.9609) |
| test | s8_ei | evidence outside the input | 59 | 25 | 0.7966 (0.6571 to 0.9260) | 0.8002 (0.6402 to 0.9220) |
| test | s8_ei | no usable evidence offsets | 6 | 5 | 0.8333 (0.4000 to 1.0000) | 0.6000 (0.3333 to 1.0000) |
| test | s8_ei | evidence in the abstract | 614 | 247 | 0.9577 (0.9404 to 0.9743) | 0.9575 (0.9392 to 0.9744) |
| test | s8_ei | evidence not in the abstract | 604 | 251 | 0.9238 (0.8986 to 0.9450) | 0.9183 (0.8908 to 0.9408) |
| test | Gemma 4 31B-it | all | 1,218 | 333 | 0.8711 (0.8450 to 0.8949) | 0.8626 (0.8333 to 0.8879) |
| test | Gemma 4 31B-it | evidence inside the input | 1,153 | 328 | 0.8829 (0.8578 to 0.9046) | 0.8745 (0.8461 to 0.8978) |
| test | Gemma 4 31B-it | evidence outside the input | 59 | 25 | 0.6610 (0.4918 to 0.8136) | 0.6507 (0.4553 to 0.7962) |
| test | Gemma 4 31B-it | no usable evidence offsets | 6 | 5 | 0.6667 (0.2000 to 1.0000) | 0.5333 (0.1667 to 1.0000) |
| test | Gemma 4 31B-it | evidence in the abstract | 614 | 247 | 0.8811 (0.8444 to 0.9143) | 0.8796 (0.8422 to 0.9139) |
| test | Gemma 4 31B-it | evidence not in the abstract | 604 | 251 | 0.8609 (0.8227 to 0.8943) | 0.8424 (0.8004 to 0.8799) |
| validation | s8_ei | all | 1,237 | 340 | 0.9369 (0.9206 to 0.9515) | 0.9332 (0.9160 to 0.9491) |
| validation | s8_ei | evidence inside the input | 1,151 | 332 | 0.9461 (0.9312 to 0.9602) | 0.9426 (0.9264 to 0.9578) |
| validation | s8_ei | evidence outside the input | 55 | 30 | 0.7273 (0.5510 to 0.8689) | 0.7093 (0.5355 to 0.8519) |
| validation | s8_ei | no usable evidence offsets | 31 | 16 | 0.9677 (0.8749 to 1.0000) | 0.9666 (0.8249 to 1.0000) |
| validation | s8_ei | evidence in the abstract | 623 | 253 | 0.9551 (0.9366 to 0.9715) | 0.9527 (0.9327 to 0.9697) |
| validation | s8_ei | evidence not in the abstract | 614 | 244 | 0.9186 (0.8882 to 0.9443) | 0.9123 (0.8781 to 0.9405) |
| validation | Gemma 4 31B-it | all | 1,237 | 340 | 0.8860 (0.8607 to 0.9095) | 0.8747 (0.8461 to 0.8996) |
| validation | Gemma 4 31B-it | evidence inside the input | 1,151 | 332 | 0.8923 (0.8660 to 0.9165) | 0.8815 (0.8538 to 0.9076) |
| validation | Gemma 4 31B-it | evidence outside the input | 55 | 30 | 0.7273 (0.5490 to 0.8772) | 0.6800 (0.5042 to 0.8641) |
| validation | Gemma 4 31B-it | no usable evidence offsets | 31 | 16 | 0.9355 (0.8077 to 1.0000) | 0.9069 (0.6310 to 1.0000) |
| validation | Gemma 4 31B-it | evidence in the abstract | 623 | 253 | 0.9021 (0.8647 to 0.9316) | 0.8951 (0.8561 to 0.9263) |
| validation | Gemma 4 31B-it | evidence not in the abstract | 614 | 244 | 0.8697 (0.8339 to 0.9063) | 0.8533 (0.8129 to 0.8944) |

PubMedQA. 10 test articles and 11 validation articles share at least two normalized sentences of 60 characters or more with one PubMedQA abstract (the matched abstracts: 16 in PQA-A, 5 in PQA-U and 0 in PQA-L), and arm B read every one of those abstracts in training (10 and 11 articles). s8_ei's test accuracy is 0.8378 (0.6667 to 0.9445) on their 37 prompts and 0.9441 on the other 1,181 (Table 21); the difference, -0.1063 (-0.2817 to 0.0051), includes zero, so no difference was detected, and the interval excludes an accuracy on the shared articles more than 0.0051 above the rest's (Table 22).

EI's own splits. 0 test and 0 validation articles share a title, or two or more such sentences, with one of EI's 2,657 training articles.

### Table 21. s8_ei's accuracy on articles that overlap a training text, and on the rest (correction 14; descriptive, not pre-registered)

| Split | Which articles | Prompts | Articles | Accuracy (95 percent interval) |
|---|---|---|---|---|
| test | articles that share sentences with a PubMedQA abstract | 37 | 10 | 0.8378 (0.6667 to 0.9445) |
| test | the rest | 1,181 | 323 | 0.9441 (0.9265 to 0.9585) |
| validation | articles that share sentences with a PubMedQA abstract | 36 | 11 | 0.9722 (0.9130 to 1.0000) |
| validation | the rest | 1,201 | 329 | 0.9359 (0.9183 to 0.9513) |
| test | articles that share a title or sentences with an EI training article | 0 | 0 | none |
| test | the rest | 1,218 | 333 | 0.9409 (0.9246 to 0.9554) |

### Table 22. Differences in accuracy between the groups (correction 15; descriptive, not pre-registered)

| Split | System | Difference in accuracy | Estimate (95 percent interval) | Reading |
|---|---|---|---|---|
| test | s8_ei | where the evidence is outside the input, minus inside | -0.1522 (-0.2967 to -0.0327) | lower outside the input |
| test | Gemma 4 31B-it | where the evidence is outside the input, minus inside | -0.2219 (-0.4175 to -0.0783) | lower outside the input |
| validation | s8_ei | where the evidence is outside the input, minus inside | -0.2189 (-0.4055 to -0.0666) | lower outside the input |
| validation | Gemma 4 31B-it | where the evidence is outside the input, minus inside | -0.1650 (-0.3469 to -0.0103) | lower outside the input |
| test | s8_ei | articles that share sentences with a PubMedQA abstract, minus the rest | -0.1063 (-0.2817 to 0.0051) | no difference was detected; excludes below -0.2817 and above 0.0051 |
| validation | s8_ei | articles that share sentences with a PubMedQA abstract, minus the rest | 0.0363 (-0.0234 to 0.0760) | no difference was detected; excludes below -0.0234 and above 0.0760 |

These checks cannot show whether Qwen3-4B-Base's or Gemma 4's pretraining text held EI's annotations. EI's labels have been public since 2019, so that stays possible, and it would favor both models.

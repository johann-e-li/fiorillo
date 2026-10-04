---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Base
language:
- en
tags:
- medical
- typed-decisions
- calibrated-probabilities
- evidence-inference
- pubmedqa
- hospital-readmission
datasets:
- qiaojin/PubMedQA
---

# Fiorillo v0.5
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 2 October 2026
Sources: results/fiorillo_v0_5/comparison.json; results/fiorillo_v0_5/release_candidate.json; results/fiorillo_v0_5/calibration_s9.json; results/fiorillo_v0_5/recipe_selection.json; results/fiorillo_v0_5/readmission_pool_v2.json; results/fiorillo_v0_5/license_recheck.json; results/fiorillo_v0_5/fixed_versions.json; results/fiorillo_v0_5/coverage_long.json; results/fiorillo_v0_4/article_selection.json; results/fiorillo_v0_3/calibration_s7.json; results/fiorillo_v0_4/comparison.json; results/fiorillo_v0_3/comparison.json; results/fiorillo_v0_2/comparison.json; results/public_v0/comparison.json; docs/prior_art.md; written by scripts/build_model_card_v0_5.py
Status: complete

Fiorillo v0.5 answers three medical questions with calibrated probabilities, each routed to its own specialist: whether a trial's article reports an outcome significantly higher, significantly lower or not significantly different with an intervention than with a comparator (Evidence Inference 2.0); the answer to a biomedical research question from its abstract (PubMedQA: yes, no or maybe); and how soon a patient with diabetes is readmitted after a hospital discharge (UCI Diabetes 130-US hospitals: no readmission, after more than 30 days, or within 30 days). It is for research. It is not a medical device and must not be used to make clinical decisions.

Abbreviations: EI is Evidence Inference 2.0 (DeYoung et al. 2020); ECE is expected calibration error; PMC is PubMed Central; PMCID is a PubMed Central identifier; LoRA is low-rank adaptation; BM25 is Okapi BM25; UCI is the University of California, Irvine, Machine Learning Repository; ICD-9-CM is the International Classification of Diseases, Ninth Revision, Clinical Modification; CC BY is the Creative Commons Attribution license and CC0 the Creative Commons public-domain dedication; OSF is the Open Science Framework; (derived) marks a number this card computes from other numbers.

## What v0.5 is

Fiorillo v0.5 routes each question to its own specialist and hands nothing off: as released, its hand-off rate is zero, and every answer is the specialist's own. The rate was fixed at zero before v0.5 was trained: in v0.4's evaluation, handing the least confident answers to Gemma 4 31B-it lowered Evidence Inference's accuracy at every rate tested and barely changed PubMedQA's. The hand-off analyses in the question cards are descriptive. A reader pass is one forward pass of a language-model reader; the tree members of the readmission pool need none.

| Question id | Specialist | Reader passes a question | Calibration |
|---|---|---|---|
| evidence_effect | s9_ei_wavg: one model whose adapters, projection and head average its recipe's three runs exactly (s9_ei_wavg) | one | temperature, fitted on the clean validation prompts |
| pubmedqa_answer | s7_b3, unchanged from v0.3 | one | temperature_intercepts, v0.3's fit |
| readmit_window | the readmission pool: xgboost_tuned, weight 0.4; catboost_tuned, weight 0.6 | zero | the pool's temperature T, 0.9957 |

Every choice for v0.5 (the recipe, the weight average, swap averaging, the calibrations, the readmission pool and the limit of the release bar's first criterion) was made on validation data and frozen at 2 October 2026, 13:40 UTC, before any test prediction of v0.5's new Evidence Inference specialist existed. The PubMedQA specialist and the readmission pool that the pool rule kept are unchanged from v0.3 and v0.4, so their test results were known before the freeze; both were kept by decisions fixed before any v0.5 training. The bar, the recipe rule, the calibration rules and the reading checks' readings had been filed on 1 October 2026, 15:31 UTC as an update to the project's registration on the Open Science Framework (OSF): https://osf.io/kaxmn/.

## Why v0.4's EI specialist is not released

v0.4's EI specialist, s8_ei, learned from all 2,657 articles (10,046 prompts) of Evidence Inference 2.0's training split. By their own license statements, 1,226 of them (derived) are under non-commercial, no-derivatives, share-alike, verbatim-only or publishers' own terms, or state no license, and whether weights trained on them may be published is UNVERIFIABLE: no source says. s8_ei therefore stays a research model: its weights are not released, and its test results appear in the Evidence Inference card only for reference. v0.5's EI specialist learned only from the 1,431 training articles (5,279 prompts) whose own statement is CC BY, CC0 or the public domain.

## Licenses, and the articles credited

- Evidence Inference 2.0: its annotations are under the MIT License (DeYoung et al. 2020; Lehman et al. 2019). Every article that v0.5's EI specialist learned from, or was calibrated and chosen on, states CC BY (any version), CC0 or the public domain in its own license statement as PubMed Central serves it, and no part of the statement restricts commercial use, derivatives or the license of what is made from it: 1,431 training and 179 validation articles, 1,610 in all (derived). The statements were read from PubMed Central between 1 October 2026, 06:30 UTC and 1 October 2026, 06:36 UTC and classified at 1 October 2026, 06:36 UTC. NOTICE credits each of these articles with its PMCID, its license and its citation as Europe PMC's REST service gives it.
- PubMedQA: MIT License (Jin et al. 2019); see the next section.
- UCI Diabetes 130-US hospitals for years 1999 to 2008: CC BY 4.0 (Strack et al. 2014).
- Qwen3-4B-Base: Apache-2.0 (Qwen/Qwen3-4B-Base at revision 906bfd4b4dc7f14ee4320094d8b41684abff8539).
- The laya package, whose decision head the readers use: Apache-2.0.
- This release: Apache-2.0 (LICENSE). NOTICE carries the attributions, the clean articles and the line "Courtesy of the U.S. National Library of Medicine".

## PubMedQA's abstracts

Every Evidence Inference article that v0.5's EI specialist learned from, or was calibrated and chosen on, is CC BY, CC0 or the public domain. The specialist's training started from s7_b3's weights, and s7_b3 learned from PubMedQA's abstracts (its automatically labeled items, PQA-A; its unlabeled items, PQA-U; and the training part of its expert-labeled items, PQA-L) under PubMedQA's MIT License. Unlike Evidence Inference's articles, those abstracts' own licenses were not checked one by one, and most PubMed abstracts are the publishers'; they rest on PubMedQA's license. s7_b3's labels for PQA-U came from this project's own labeler, not from an outside language model. The labeler is a model built on Fiorillo reader v1 (johanneli/fiorillo-reader-v1) that reads each abstract's conclusion. s7_b3 is v0.5's PubMedQA specialist and is in the readmission pool with no weight, so it does not answer readmission.

## Prior art

The prior-art check of the release day, 2 October 2026, is in the project's docs/prior_art.md. This card makes no claim that v0.5 is the first of its kind; a claim of a first would be worded "to our knowledge". The check found nothing published since the previous check that makes v0.5 redundant.

## Question card: evidence_effect (Evidence Inference 2.0)

### Intended use and user

For researchers who extract or check the results of randomized trials. Given a trial's article and one comparison in it (an intervention, a comparator and an outcome), it gives the probability that the article reports the outcome significantly higher with the intervention than with the comparator, significantly lower, or not significantly different. It reads the article as given; its only other source about a trial is what its base model may have learned in pretraining, which the reading checks below look for but cannot rule out. It is for research only, not for clinical decisions, and it has been tested only on Evidence Inference 2.0's prompts, in English.

### Inputs

A record whose state is {"intervention", "comparator", "outcome", "article"}: the comparison's three parts as short texts, and the article's full text in Evidence Inference's text format (the title and abstract, then the body, whose sections begin with lines that begin with "BODY."). The passage selection below cuts the article so that the whole input, article and question together, fits 6,144 tokens. The question, word for word: "Based only on this article, how did the outcome differ with the intervention compared with the comparator?", with the options significantly increased (the outcome was significantly higher with the intervention than with the comparator), significantly decreased (significantly lower) and no significant difference (the article reports no significant difference in the outcome between the intervention and the comparator).

### The specialist

v0.5's EI specialist is s9_ei_wavg: one model whose adapters, projection and head average its recipe's three runs exactly (s9_ei_wavg). Its reader is Qwen3-4B-Base (Qwen/Qwen3-4B-Base, revision 906bfd4b4dc7f14ee4320094d8b41684abff8539) with low-rank adaptation (LoRA) adapters on every linear layer of the decoder, a projection and Laya's decision head, trained from s7_b3's weights; the base model's own weights stay fixed. Its recipe, chosen by the recipe rule on the 654 clean validation prompts, is longer inputs. It trains for two epochs with inputs of 6,144 tokens and the adapters' learning rate 5e-5. Its runs are s9_long_1, s9_long_2, s9_long_3. On the test split, its log loss minus that of the base recipe's runs (s9_b) was -0.0000 (-0.0202 to 0.0174): no difference was detected. The recipe's lead on the clean validation prompts, 0.0282 (derived), was not seen on test: the interval excludes a lead that large. The weight average of the runs scored 0.1781 on the clean validation prompts against 0.1790 for their probability average, with a margin of 0.0110: not above it by more than the margin, so v0.5 uses the weight average, one reader in place of the runs. Swap averaging is not kept: with it the score was 0.2515, against 0.1781 without it, not below by more than the margin, 0.0110, so each question takes one pass of its reader.

The recipe rule's scores, each the out-of-fold log loss of v0.4's calibration rule (the lower of the two families temperature and temperature_intercepts) on the clean validation prompts (lower is better). A candidate's score is the mean of its runs' scores; it passes if that is below the base recipe's, 0.2116 (derived: the mean of s9_b_1's 0.2171 and s9_b_2's 0.2061), by more than the margin m, 0.0110 (the spread of the three base runs' scores, 0.0110, with a floor of 0.0050), and each of its runs scores below the base run with the same seed.

| Candidate | Runs | Score | Passes |
|---|---|---|---|
| swap | s9_swap_1, s9_swap_2 | 0.2137 | no |
| long | s9_long_1, s9_long_2 | 0.1834 | yes |
| lr1 | s9_lr1_1, s9_lr1_2 | 0.2084 | no |
| lr2 | s9_lr2_1, s9_lr2_2 | 0.2018 | no |

The winning recipe: long.

Calibration: the family with the lowest out-of-fold log loss on the clean validation prompts, among four, fitted on all of them; the chosen family is temperature.

| Family | Out-of-fold log loss: mean over the repeats (lowest and highest repeat) |
|---|---|
| temperature (chosen) | 0.1781 (0.1765 to 0.1808) |
| temperature_intercepts | 0.1808 (0.1775 to 0.1860) |
| vector | 0.1836 (0.1791 to 0.1910) |
| dirichlet_odir | 0.1829 (0.1781 to 0.1924) |

The families: temperature, one temperature for all the answers; temperature_intercepts, a temperature and an intercept for each answer but the first; vector, a scale and an intercept for each answer; dirichlet_odir, a full matrix on the log probabilities and an intercept for each answer, with the matrix's off-diagonal entries regularized (Kull et al. 2019).

Cost: one reader pass a question. In the test predictions, run alone on a rented GPU (an NVIDIA H100), it took 0.303 GPU seconds a question, which at its rate of 3.52 dollars an hour is 0.297 dollars per thousand questions.

### Training data and their licenses

Evidence Inference 2.0 (DeYoung et al. 2020): its training split's prompts whose article is clean, 5,279 of 10,046 prompts in 1,431 of 2,657 articles. The specialist was calibrated and chosen on the clean validation prompts, 654 of 1,237 in 179 of 340 articles. An article is clean when its own license statement, as PubMed Central serves it, is CC BY (any version), CC0 or the public domain and no part of it restricts commercial use, derivatives or the license of what is made from it (Licenses, above). The annotations are under the MIT License. Before Evidence Inference, the readers started from s7_b3, whose training data are in the PubMedQA card.

### Validation and test splits, and when they were read

- Validation: Evidence Inference's validation split, 1,237 prompts in 340 articles. Every choice for v0.5 read only its 654 clean prompts: the recipe, the weight average, swap averaging and the calibration, all frozen at 2 October 2026, 13:40 UTC. The large model's calibration and the hand-off analyses read all of them.
- Test: Evidence Inference's test split, 1,218 prompts. It was first read for v0.4, whose comparisons are dated 1 October 2026, 02:53 UTC; then, before v0.5's training, by the reading checks on v0.4's specialist, after the registration's update was filed (1 October 2026, 15:31 UTC); and for v0.5 after the freeze, its comparisons written 2 October 2026, 15:39 UTC. v0.5's bar is the second release bar judged on this split.

### Test results

Each test result has an interval (95 percent, percentile, paired) from 1,000 resamples: Evidence Inference resamples its articles, the other two questions their items. ECE is the expected calibration error of the top answer in 15 equal-width bins. A percentile interval of ECE can lie above its estimate: ECE adds up the gaps between confidence and accuracy over the bins, and the noise of a resample adds to those gaps, so resampled values tend to exceed the value on the whole split.

| System | n | Log loss | ECE | Brier score | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| **v0.5's EI specialist (s9_ei_wavg)** | 1,218 | 0.2164 (0.1714 to 0.2645) | 0.0168 (0.0124 to 0.0348) | 0.1101 (0.0871 to 0.1363) | 0.9269 (0.9089 to 0.9426) | 0.9248 (0.9059 to 0.9409) |
| Gemma 4 31B-it reading the same longer inputs, calibrated on validation | 1,218 | 0.3993 (0.3301 to 0.4793) | 0.0283 (0.0238 to 0.0567) | 0.2005 (0.1649 to 0.2395) | 0.8752 (0.8489 to 0.8991) | 0.8668 (0.8390 to 0.8935) |
| v0.4's EI specialist, s8_ei (not released) | 1,218 | 0.1994 (0.1573 to 0.2486) | 0.0119 (0.0119 to 0.0300) | 0.1001 (0.0778 to 0.1266) | 0.9409 (0.9246 to 0.9554) | 0.9399 (0.9225 to 0.9544) |
| Prior: the training split's answer shares | 1,218 | 1.0767 (1.0580 to 1.0940) | 0.0305 (0.0018 to 0.0663) | 0.6525 (0.6395 to 0.6643) | 0.4212 (0.3855 to 0.4584) | 0.1976 (0.1855 to 0.2095) |

Calibration slope and intercept on the test split, each answer against the rest: the slope is the coefficient of a logistic regression of the answer's indicator on the logit of its calibrated probability (ideal one); the intercept is that regression's intercept with the logit as an offset (ideal zero); the third column is the intercept fitted together with the slope (ideal zero). 1,218 test items.

| Answer | Items with this answer | Intercept | Slope | Intercept fitted with the slope |
|---|---|---|---|---|
| significantly increased | 411 | 0.2244 (-0.1219 to 0.5996) | 0.9112 (0.8188 to 1.0405) | 0.1712 (-0.1248 to 0.5211) |
| significantly decreased | 294 | -0.1294 (-0.4867 to 0.1917) | 0.9919 (0.8773 to 1.1833) | -0.1324 (-0.4750 to 0.2415) |
| no significant difference | 513 | -0.1139 (-0.5151 to 0.2811) | 0.8850 (0.7943 to 1.0011) | -0.1230 (-0.4569 to 0.2077) |

v0.5 was released because it met v0.4's four criteria on this test split, with v0.5's EI specialist in v0.4's place (overall: PASS).

| Criterion | Result | Values |
|---|---|---|
| ECE at most the limit: the larger of 0.05 and the 95th percentile of the ECE of a model calibrated by construction at the test split's size, fixed in the freeze | PASS | ECE 0.0168 against a limit of 0.0500 (the percentile: 0.0184) |
| Log loss below the prior's, the interval of the difference (10,000 article resamples) wholly below zero | PASS | 0.2164 against 1.0767; difference -0.8603 (-0.9078 to -0.8104) |
| Log loss below that of Gemma 4 31B-it on the same longer inputs (calibrated on validation), the interval of the difference (10,000 article resamples) wholly below zero | PASS | 0.2164 against 0.3993; difference -0.1829 (-0.2598 to -0.1164) |
| Macro-F1 not below that large model's point estimate | PASS | 0.9248 against 0.8668 |

### Hand-off

As released, v0.5 hands nothing off: its accuracy on the 1,218 test items is 0.9269, and Gemma 4 31B-it on the same longer inputs alone scored 0.8752. Descriptive, and not part of the release: had the specialist handed some answers to that model, the combined accuracy on the test split would have been as follows. The confidence rule hands off the least confident answers first; random hand-off is that rate's expected value; the oracle first hands off the items the specialist gets wrong and the large model right, which bounds what any rule could gain; the large model's own confidence is a reference, not a usable rule.

| Hand-off rate, percent | Items handed off | Confidence rule | Random hand-off, expected | Oracle, a bound | Large model's own confidence |
|---|---|---|---|---|---|
| 10 | 121 | 0.9335 (0.9160 to 0.9492) | 0.9218 (0.9040 to 0.9369) | 0.9573 (0.9442 to 0.9693) | 0.9269 (0.9097 to 0.9428) |
| 20 | 243 | 0.9171 (0.8986 to 0.9328) | 0.9166 (0.8988 to 0.9322) | 0.9573 (0.9442 to 0.9693) | 0.9261 (0.9076 to 0.9430) |
| 30 | 365 | 0.8966 (0.8751 to 0.9173) | 0.9114 (0.8935 to 0.9276) | 0.9573 (0.9442 to 0.9693) | 0.9245 (0.9055 to 0.9423) |
| 50 | 609 | 0.8768 (0.8535 to 0.9013) | 0.9011 (0.8821 to 0.9182) | 0.9573 (0.9442 to 0.9693) | 0.9261 (0.9073 to 0.9441) |

Selective answering on its own (no hand-off): the area under the risk-coverage curve (AURC) is 0.0128 (0.0082 to 0.0182), its excess over a perfect ranking's (E-AURC) 0.0100 (0.0064 to 0.0145), and the area under the generalized risk-coverage curve (AUGRC; Traub et al. 2024) 0.0105 (0.0069 to 0.0147).

### The reading checks

The reading checks ask how much of an EI specialist's answer comes from reading the article: with no article (none), with the article's title alone (title), and with the intervention and the comparator exchanged and the expected answer exchanged (swap). Accuracy and macro-F1 are against the original answers (none, title) or the exchanged ones (swap); agreement is the share of answers equal to the specialist's answer on its full input; for swap, the exchange share is the share of its increased and decreased answers that exchange, and the stay share the share of its no-significant-difference answers that stay. The readings were fixed before the checks ran and gate no release.

| System | Input | Accuracy | Macro-F1 | Agreement, or exchange and stay shares | Reading |
|---|---|---|---|---|---|
| v0.5's EI specialist | none: no article | 0.5156 (0.4874 to 0.5494) | 0.4384 (0.4085 to 0.4726) | 0.5238 (0.4955 to 0.5571) | its answers depend on the article's text |
| v0.5's EI specialist | title: the title alone | 0.5715 (0.5389 to 0.6041) | 0.5327 (0.4995 to 0.5692) | 0.5782 (0.5471 to 0.6108) | the title adds information about the result: a title can state the result, or the model may recall the trial, and these checks cannot tell them apart |
| v0.5's EI specialist | swap: intervention and comparator exchanged | 0.7808 (0.7493 to 0.8117) | 0.7600 (0.7268 to 0.7946) | exchange 0.6652 (0.6208 to 0.7157); stay 0.9672 (0.9477 to 0.9841) | partly |
| v0.4's EI specialist, s8_ei | none: no article | 0.5090 (0.4779 to 0.5439) | 0.4289 (0.3941 to 0.4634) | 0.5140 (0.4812 to 0.5473) | its answers depend on the article's text |
| v0.4's EI specialist, s8_ei | title: the title alone | 0.5715 (0.5346 to 0.6065) | 0.5338 (0.4964 to 0.5733) | 0.5682 (0.5343 to 0.6027) | the title adds information about the result: a title can state the result, or the model may recall the trial, and these checks cannot tell them apart |
| v0.4's EI specialist, s8_ei | swap: intervention and comparator exchanged | 0.8103 (0.7802 to 0.8400) | 0.7958 (0.7633 to 0.8283) | exchange 0.7226 (0.6765 to 0.7714); stay 0.9806 (0.9657 to 0.9931) | partly |

On the exchanged prompts whose comparator is a time point or baseline, where exchanging the two need not reverse the answer: v0.5's specialist exchanged 0.2162 (0.0857 to 0.3439) of its 37 direction answers and kept 1.0000 (1.0000 to 1.0000) of its 11 no-significant-difference answers; v0.4's exchanged 0.1316 (0.0250 to 0.2632) of its 38 direction answers and kept 1.0000 (1.0000 to 1.0000) of its 10 no-significant-difference answers.

The title minus none, macro-F1 on the prompts in both inputs: v0.5's specialist 0.0939 (0.0655 to 0.1234); v0.4's 0.1024 (0.0736 to 0.1331).

References: the training split's answer shares score a macro-F1 of 0.198 on the test split (0.185 to 0.210); frontier models scored an accuracy of 0.79 to 0.88 from trial titles alone on 250 oncology trials, on a two-answer task (positive or negative), against a majority class of 0.584 (Windisch et al. 2026, JMIR AI 2026;5:e95565, doi 10.2196/95565 (PubMed 42525868; full text PMC13419288), Table 1).

### The passage selection, and the exception for its word statistics

The article is cut by v0.4's selection rule at 6,144 tokens, the specialist's input length: the title and abstract first, cut from their end if they alone do not fit; then the passages, results sections first and then the others, each ranked by Okapi BM25 (a word-matching score; k1 1.2, b 0.75) against the distinct words of the intervention, the comparator and the outcome, each passage kept if it still fits; the kept passages are joined again in article order. At this length the selection kept an annotator-marked evidence span for 98.68 percent of the clean training prompts that have one, the prompts the specialist learned from (5,100 of 5,168), and for 99.03 percent of all the training prompts that have one (9,754 of 9,850).

The selection's functions ship in evidence_effect/passage_selection/evidence_inference.py with its word statistics, bm25_stats.json: for each of 115,286 words, the number of the 44,280 passages of the 2,657 training articles that contain it, with the passages' mean length, 149.09 words. These statistics are the one exception to the rule that nothing the EI specialist uses comes from an Evidence Inference article that is not clean: they were counted over every training article, clean or not. They hold single words and counts but no sentence of any article. They are kept so that every input, the specialist's and the large model's, is cut by the same rule, and they are shipped so that a new article can be cut the same way.

### Known failure modes and contamination

- Evidence outside the input: for 1.32 percent of the clean training prompts that have marked evidence (derived), the selection cut it away, and the specialist answers without it.
- Answers without the article: with no article at all its macro-F1 was 0.4384 (0.4085 to 0.4726) (the reading checks, above, say how far its answers depend on the article's text).
- Comparators that are a time point or baseline: when the intervention and such a comparator are exchanged, the specialist seldom reverses its answer (the reading checks, above); whether its answers to such prompts follow the order of the comparison is not known.
- Only Evidence Inference's kind of prompt was tested: one comparison in a randomized trial's article, in English. Other questions, wordings and kinds of document are untested and uncalibrated.
- Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PMC articles (EI's articles are PMC's), EI's repository (public since 2019), PubMedQA and the UCI data, which would favor both; EI's test labels are public (Evidence Inference's site serves them to its leaderboard), so no evaluation of EI can be blind. EI's test split was read before, for v0.4's bar, its comparisons and its checks; v0.5's recipe and calibration were chosen on validation only and frozen before any test prediction of v0.5's Evidence Inference specialist existed, and this is the second time the bar is judged on this split.

### Update plan

This version is fixed as released. The project's next step for this question is a test on trials published after the base model was released, so that a good answer cannot come from recalling the trial; a later Evidence Inference specialist will be released only if it meets a bar fixed before it is trained, as this one did.

## Question card: pubmedqa_answer (PubMedQA)

### Intended use and user

For researchers who want a calibrated yes, no or maybe for a biomedical research question from its abstract, as PubMedQA frames it: whether the abstract's reported results support yes, support no, or are inconclusive. It is for research only, not for clinical decisions, and it has been tested only on PubMedQA's questions, in English.

### Inputs

A record whose state is {"question", "contexts": [{"section", "text"}, ...]}: the research question and the abstract's sections in their order, without the conclusion (which states the answer). The input is cut from the end of the abstract if it does not fit the reader's input length (max_len in s7_b3's config.json). The question, word for word: "Based only on this abstract, what is the answer to the research question?", with the options yes (the reported results support yes), no (the reported results support no) and maybe (the reported results are inconclusive).

### The specialist

s7_b3, unchanged from v0.3: Qwen3-4B-Base with LoRA adapters, a projection and Laya's decision head, trained on PubMedQA (and, as one model, on the readmission question). Its calibration is temperature_intercepts, chosen by v0.3's rule (the lower out-of-fold log loss of temperature and temperature with intercepts) and fitted on PubMedQA's 150 expert-labeled validation items at 30 September 2026, 06:20 UTC. One reader pass a question.

### Training data and their licenses

PubMedQA (Jin et al. 2019; MIT License): its automatically labeled items (PQA-A), its unlabeled items (PQA-U), labeled by this project's own labeler and not by an outside language model, and the training part of its expert-labeled items (PQA-L). Those abstracts' own licenses were not checked one by one: they rest on PubMedQA's license (PubMedQA's abstracts, above).

### Validation and test splits, and when they were read

- Validation: PubMedQA's 150 expert-labeled validation items, read for s7_b3's calibration at 30 September 2026, 06:20 UTC.
- Test: PubMedQA's official test list, 500 items, read in the evaluation of each version: v0.1 (27 September 2026), v0.2 (28 September 2026), v0.3 (30 September 2026, 06:45 UTC), v0.4 (1 October 2026, 02:53 UTC), v0.5 (2 October 2026, 15:39 UTC). s7_b3's test answers are v0.3's, reused. The split was also read on 29 September 2026, for a comparison that this card does not report.

### Test results

| System | n | Log loss | ECE | Brier score | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| **v0.5's PubMedQA specialist (s7_b3)** | 500 | 0.5797 (0.5070 to 0.6509) | 0.0401 (0.0315 to 0.0800) | 0.3147 (0.2683 to 0.3595) | 0.7960 (0.7600 to 0.8340) | 0.5698 (0.5406 to 0.6039) |
| BioGPT-Large-PubMedQA, uncalibrated (fine-tuned on PubMedQA) | 500 | 0.8899 (0.7428 to 1.0347) | 0.1371 (0.1114 to 0.1803) | 0.4058 (0.3462 to 0.4623) | 0.7520 (0.7120 to 0.7900) | 0.5481 (0.5119 to 0.5922) |
| Gemma 4 31B-it, calibrated on validation | 500 | 0.6131 (0.5309 to 0.6957) | 0.0575 (0.0431 to 0.0970) | 0.3209 (0.2728 to 0.3696) | 0.7840 (0.7460 to 0.8200) | 0.5862 (0.5527 to 0.6252) |
| Prior: the training labels' shares | 500 | 1.1593 (1.0401 to 1.2920) | 0.0646 (0.0226 to 0.1086) | 0.5855 (0.5503 to 0.6208) | 0.5520 (0.5080 to 0.5940) | 0.2371 (0.2246 to 0.2484) |

Calibration slope and intercept on the test split, each answer against the rest: the slope is the coefficient of a logistic regression of the answer's indicator on the logit of its calibrated probability (ideal one); the intercept is that regression's intercept with the logit as an offset (ideal zero); the third column is the intercept fitted together with the slope (ideal zero). 500 test items.

| Answer | Items with this answer | Intercept | Slope | Intercept fitted with the slope |
|---|---|---|---|---|
| yes | 276 | -0.0217 (-0.2800 to 0.2453) | 0.9647 (0.8394 to 1.1381) | -0.0145 (-0.2728 to 0.2469) |
| no | 169 | -0.1513 (-0.3927 to 0.0973) | 1.1869 (1.0296 to 1.4260) | -0.1468 (-0.4504 to 0.1496) |
| maybe | 55 | 0.2371 (-0.0890 to 0.5250) | 0.8990 (0.5392 to 1.3122) | 0.0231 (-0.8637 to 0.8928) |

### Hand-off

As released, v0.5 hands nothing off: its accuracy on the 500 test items is 0.7960, and Gemma 4 31B-it alone scored 0.7840. Descriptive, and not part of the release: had the specialist handed some answers to that model, the combined accuracy on the test split would have been as follows. The confidence rule hands off the least confident answers first; random hand-off is that rate's expected value; the oracle first hands off the items the specialist gets wrong and the large model right, which bounds what any rule could gain; the large model's own confidence is a reference, not a usable rule.

| Hand-off rate, percent | Items handed off | Confidence rule | Random hand-off, expected | Oracle, a bound | Large model's own confidence |
|---|---|---|---|---|---|
| 10 | 50 | 0.8000 (0.7620 to 0.8360) | 0.7948 (0.7604 to 0.8308) | 0.8380 (0.8040 to 0.8720) | 0.8000 (0.7640 to 0.8380) |
| 20 | 100 | 0.7960 (0.7600 to 0.8340) | 0.7936 (0.7596 to 0.8284) | 0.8380 (0.8040 to 0.8720) | 0.8000 (0.7660 to 0.8380) |
| 30 | 150 | 0.7940 (0.7580 to 0.8320) | 0.7924 (0.7580 to 0.8262) | 0.8380 (0.8040 to 0.8720) | 0.8040 (0.7700 to 0.8400) |
| 50 | 250 | 0.7960 (0.7600 to 0.8340) | 0.7900 (0.7550 to 0.8230) | 0.8380 (0.8040 to 0.8720) | 0.8020 (0.7680 to 0.8380) |

Selective answering on its own (no hand-off): the area under the risk-coverage curve (AURC) is 0.0918 (0.0664 to 0.1183), its excess over a perfect ranking's (E-AURC) 0.0692 (0.0492 to 0.0901), and the area under the generalized risk-coverage curve (AUGRC; Traub et al. 2024) 0.0615 (0.0466 to 0.0767).

### Known failure modes and contamination

- Only PubMedQA's kind of question was tested, in English; other questions and wordings are untested.
- About the BioGPT-Large-PubMedQA row: BioGPT-Large-PubMedQA was fine-tuned on PQA-L's fold 0 training and development sets, which hold all 150 of this project's PubMedQA validation items, so its calibration (v0.4's calibration rule on those items) was fitted on items it was trained on: its uncalibrated probabilities, renormalized over the three answers, are the primary result and its calibrated ones are secondary (docs/fiorillo_v0_5.md, the project's report).
- Known contamination: Qwen3's and Gemma 4's pretraining text is not published and may include PubMedQA and the PubMed abstracts it is built from, which would favor both. PubMedQA's test split was read in the evaluation of every version, and s7_b3 was chosen in v0.3 after earlier versions' test results were known, so its comparisons here are descriptive.

### Update plan

s7_b3 is unchanged since v0.3 and no retraining of it is planned. The comparison with BioGPT-Large-PubMedQA is descriptive and changes nothing in this release.

## Question card: readmit_window (UCI Diabetes 130-US hospitals)

### Intended use and user

For researchers who study readmission risk on the UCI Diabetes 130-US hospitals data: the probability that a patient with diabetes is not readmitted after a hospital discharge, readmitted after more than 30 days, or readmitted within 30 days. The data are from 130 US hospitals between 1999 and 2008, first encounters only, and care has changed since. It is for research only, not for clinical decisions.

### Inputs

A record whose state holds one encounter's fields: age band, sex, admission type, admission source and discharge disposition (this project's names for the UCI data's age, gender and three _id columns, with the codes written out), then, as the UCI data name them, time_in_hospital, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, number_diagnoses, diag_1, diag_2 and diag_3 (an ICD-9-CM code followed by its chapter's name in parentheses), max_glu_serum, A1Cresult, medications (each diabetes medication and its change), change, diabetesMed and medical_specialty; a field that is not known is left out. The question, word for word: "How soon after this hospital discharge is the patient readmitted?", with the levels no readmission after this discharge, readmitted more than 30 days after this discharge, and readmitted within 30 days of this discharge.

### The specialist

The readmission pool combines its members' probabilities: softmax((sum over the members of w log p) / T), each member's probabilities clipped at a floor before the log, with the weights on a grid of tenths and T fitted on the validation split. T is 0.9957. The members with a weight above zero:

| Member | What it is | Weight |
|---|---|---|
| xgboost_tuned | XGBoost, tuned for v0.4 | 0.4 |
| catboost_tuned | CatBoost, tuned for v0.4 | 0.6 |

v0.5 uses v0.4's pool: the new pool's out-of-fold log loss on validation was 0.83681 against 0.83715 for v0.4's pool, a difference of 0.00035, and the rule needed at least 0.002 (the choice made at 2 October 2026, 13:40 UTC). The new pool added to v0.4's members: xgboost_codes, XGBoost with v0.4's tuned setting, refitted with the recoded diagnosis fields; catboost_codes, CatBoost with v0.4's tuned setting, refitted with the recoded diagnosis fields. TabICL, planned as a further member, was not built, because of an error in this project's check of its version.

Reader passes a question: zero; the tree members need none.

### Training data and their licenses

The UCI Diabetes 130-US hospitals data (CC BY 4.0; Strack et al. 2014): the first encounter of each patient, encounters that ended in death or hospice left out, split by patient into training, validation and test parts. The members were fitted on the training part; the pool's weights and T on the 6,997 validation encounters.

### Validation and test splits, and when they were read

- Validation: 6,997 encounters, read in each version: v0.4's pool's weights and T were fitted on them before v0.4's evaluation (1 October 2026, 02:53 UTC), the new pool's in v0.5, and the choice between the two pools was made at 2 October 2026, 13:40 UTC.
- Test: 6,998 encounters, read in the evaluation of each version: v0.1 (27 September 2026), v0.2 (28 September 2026), v0.3 (30 September 2026, 06:45 UTC), v0.4 (1 October 2026, 02:53 UTC), v0.5 (2 October 2026, 15:39 UTC). The split was also read on 29 September 2026, for a comparison that this card does not report.

### Test results

| System | n | Log loss | ECE | Brier score | Accuracy | Macro-F1 |
|---|---|---|---|---|---|---|
| **v0.5's readmission pool** | 6,998 | 0.8275 (0.8132 to 0.8411) | 0.0066 (0.0085 to 0.0225) | 0.4972 (0.4882 to 0.5062) | 0.6157 (0.6042 to 0.6269) | 0.3695 (0.3579 to 0.3813) |
| v0.4's readmission pool | 6,998 | 0.8275 (0.8132 to 0.8411) | 0.0066 (0.0085 to 0.0225) | 0.4972 (0.4882 to 0.5062) | 0.6157 (0.6042 to 0.6269) | 0.3695 (0.3579 to 0.3813) |
| LightGBM, tuned for v0.3 | 6,998 | 0.8347 (0.8203 to 0.8487) | 0.0071 (0.0074 to 0.0225) | 0.5026 (0.4937 to 0.5114) | 0.6157 (0.6043 to 0.6275) | 0.3569 (0.3477 to 0.3660) |
| Prior: the training labels' shares | 6,998 | 0.8861 (0.8734 to 0.8984) | 0.0017 (0.0002 to 0.0127) | 0.5377 (0.5298 to 0.5453) | 0.5942 (0.5829 to 0.6047) | 0.2485 (0.2455 to 0.2512) |

Calibration for readmission within 30 days (the probability of the last level against the rest) on the test split: the intercept is -0.0192 (-0.1031 to 0.0646) (ideal zero) and the slope 1.0493 (0.9108 to 1.1940) (ideal one).

### Hand-off

The readmission pool hands nothing off, and no hand-off to a large model was analyzed for this question.

### Known failure modes and contamination

- Old data from one source: 130 US hospitals between 1999 and 2008, first encounters only; care has changed since, and the pool has not been tested on any other data.
- Readmission within 30 days is rare in these data, so a calibrated model seldom makes it the top answer: use the probabilities, not the top answer.
- Known contamination: the pool's members are tree models fitted only on the training part, with no pretraining, and the UCI data are public. The readmission test split was read in the evaluation of every version, and v0.4's pool was chosen after earlier versions' test results were known, so its comparisons here are descriptive.

### Update plan

No refit of the pool is planned; a change would be a new version, with its own bar.

## How to use

The release holds the readers' adapters and heads, not Qwen3-4B-Base itself, which predict_v0_5.py loads from Qwen/Qwen3-4B-Base at the revision in each reader's config.json (or from a local copy given with --base). Its files:

- evidence_effect/: calibration.json (the specialist's form and calibration), s9_ei_wavg/ (the reader's folder: adapter.safetensors, config.json and the tokenizer files), and passage_selection/ (evidence_inference.py and bm25_stats.json)
- pubmedqa_answer/: calibration.json and s7_b3/ (adapter.safetensors, config.json and the tokenizer files)
- readmit_window/: pool.json (the members, weights and T), recode_tables.json, xgboost_tuned.json, catboost_tuned.cbm
- config.json (a short description of the release's parts, which predict_v0_5.py does not read; the Hugging Face Hub counts a download each time it is requested), predict_v0_5.py, LICENSE, NOTICE and SHA256SUMS.txt

Install torch, transformers, safetensors, numpy, pandas and laya==0.3.20, and for readmission xgboost, catboost. Then write one record a line in a JSON Lines file and run predict_v0_5.py on it from the release folder:

```bash
python predict_v0_5.py --records records.jsonl --out answers.jsonl
```

Example records (the state's fields are the inputs of each question card):

```json
{"id": "my-trial-1", "qid": "evidence_effect", "state": {"intervention": "early mobilization", "comparator": "usual care", "outcome": "length of hospital stay", "article": "TITLE: ...\nABSTRACT ...\nBODY.METHODS:\n..."}}
{"id": "my-question-1", "qid": "pubmedqa_answer", "state": {"question": "Does early mobilization shorten hospital stay after hip fracture surgery?", "contexts": [{"section": "BACKGROUND", "text": "..."}, {"section": "RESULTS", "text": "..."}]}}
{"id": "my-encounter-1", "qid": "readmit_window", "state": {"age band": "[70-80)", "sex": "Female", "admission type": "Emergency", "admission source": "Emergency Room", "discharge disposition": "Discharged to home", "time_in_hospital": 4, "num_lab_procedures": 45, "num_procedures": 1, "num_medications": 18, "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "number_diagnoses": 9, "diag_1": "428 (Diseases of the circulatory system)", "max_glu_serum": "None", "A1Cresult": ">8", "medications": {"insulin": "Up"}, "change": "Ch", "diabetesMed": "Yes"}}
```

Each answer gives every option's calibrated probability, the most probable option and the reader passes it took. For Evidence Inference, predict_v0_5.py cuts the article with the shipped passage selection before the specialist reads it. For readmit_window the options are the levels 0 (no readmission), 1 (after more than 30 days) and 2 (within 30 days).

Change since the first public revision (56e5944): on 3 October 2026, predict_v0_5.py was changed in the order in which it loads its libraries. On macOS, the readmission pool's tree libraries (xgboost, catboost) and torch, on which the readers run, cannot work in one process: whichever runs second ends it with a segmentation fault, and at the first revision the readmission question failed this way. The script now loads xgboost and catboost before torch when every record in the file asks readmit_window, and does not load them first otherwise. On macOS, put each question's records in a file of their own (a file that mixes them, like the example above, is refused there with a message) and run the script once for each file. Run this way on a Mac, on the usage check's records (all 6,998 readmission test rows, all 500 PubMedQA test items and 60 Evidence Inference test prompts), the answers agreed with the evaluated test predictions within the check's limits: readmission within 0.0001 on every row; Evidence Inference and PubMedQA with every top answer whose evaluated probability is at least 0.6 unchanged and a mean absolute difference of at most 0.02. No weight, calibration or other file changed.

## License and citation

Apache-2.0 (LICENSE). The training data keep their own licenses (Licenses, above); NOTICE carries the attributions. The previous public version is Fiorillo v0.1 (johanneli/fiorillo-v0.1).

Li JE. Fiorillo v0.5: calibrated answers to medical typed questions from routed specialists. Hugging Face, 2026. https://huggingface.co/johanneli/fiorillo-v0.5

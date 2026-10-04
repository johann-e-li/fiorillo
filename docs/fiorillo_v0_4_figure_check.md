# Fiorillo v0.4: figure check
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 1 October 2026
Sources: docs/fiorillo_v0_4.md (its sections How each stage was run, Results (test split) and Checks not pre-registered); results/fiorillo_v0_4/ (comparison.json, release_candidate.json, testpreds_log.json, calibration_s8.json, ece_limit_evidence_effect.json, handoff_validation.json, handoff_test.json, power.json, power_validation.json, pilot.json, trees.json, readmission_pool.json, readmission_pool_test.json, trees_test.json, zeros.json, article_selection.json, records_manifest.json, contamination_checks.json, large_model/gemma4_31b_letters.json, large_model/gemma4_31b_timing.json, metrics/, run_summaries/ and logs/pod/); docs/log.md; models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt; data/processed/session8/ and data/processed/public/ (the test records' counts)
Status: complete

Abbreviations: EI is Evidence Inference 2.0; ECE is expected calibration error; AUROC is the area under the receiver operating characteristic curve; UTC is Coordinated Universal Time; SHA-256 is Secure Hash Algorithm 256; GPU is graphics processing unit; GiB is 2^30 bytes; PMCID is a PubMed Central identifier; PQA-A, PQA-U and PQA-L are PubMedQA's automatically labeled, unlabeled and expert-labeled parts; OSF is the Open Science Framework; G1 to G6 are the plan's comparisons (docs/prompt_session8.md, COMPARISONS).

## How the check was made

src/eval/check_fiorillo_v0_4_figures.py read three sections of docs/fiorillo_v0_4.md as written (SHA-256 of the file checked: 915229d4fa200f4ea16f6455b4eead4e0d963195c1c489d5118df48d54151e49): How each stage was run, Results (test split), and Checks not pre-registered (corrections 14 and 15). It found every number in them and compared each with its source at the printed precision: a number printed with four decimals passes when it is the source value rounded to four decimals, a count when it equals the source, a time when it equals the source's time of day. Table cells were matched to their sources by the table's heading and its rows and columns, which the script fixes for each table. Numbers in the prose were matched, in order, to the source keys of the line's template in the script, and each key was resolved by the script's own code from the results files, the pod's stage logs, docs/log.md (the pod's times and cost) and SHA256SUMS.txt. Lines that state a source's text were checked whole against it. Every reading was checked against its interval and the direction of its metric, read as compare_session8.py reads it, and every PASS or FAIL against the rule's own result. The script also checked that handoff_test.json is the file comparison.json names, by its SHA-256. Before the document was checked, 14 errors were planted in copies of it, one at a time, and each was reported. The script checks numbers and readings, not the wording of claims. Every figure's source is also listed in results/fiorillo_v0_4/figure_sources.json.

Not checked here: the document's other sections, which hold the plan's fixed settings (docs/prompt_session8.md) and the published figures, each cited to its paper and to docs/prior_art.md (the check of 1 October 2026).

Not counted as numbers, wherever they appear: version names (v0.1 to v0.4, vLLM 0.30.0, CUDA versions), model names (Qwen3-4B-Base, Gemma 4 31B-it, Gemma 4 E2B-it, Evidence Inference 2.0), idea numbers (I-01 and so on), G1 to G6 and H1 to H4; table, correction, criterion, session, step and seed numbers; dates and the years 2010 to 2029; the phrases "95 percent", "95th", "30 days", "80 percent chance" and "prints as 0"; readmission's levels 0 to 2; and anything in code type or holding an underscore or a slash. Line numbers count each section's heading as line 1.

## Result

2,576 numbers and whole lines: 2,576 PASS, 0 MISMATCH. 202 readings: 202 PASS, 0 MISMATCH.

## Every number

| # | Where | As written | Source | Source value | Check |
|---|---|---|---|---|---|
| 1 | stages, line 3 (pod), number 1 | 150 | docs/log.md, the entry "session 8, steps 8 and 9: the credit", 'a (\\d+) GB container disk' | 150 | PASS |
| 2 | stages, line 3 (pod), number 2 | $3.51 | handoff_test.json hourly_rate | 3.510000 | PASS |
| 3 | stages, line 3 (pod), number 3 | 13:37:21 | docs/log.md, the entry "session 8, steps 8 and 9: the credit", 'its container started at (\\S+) UTC' | 13:37:21 | PASS |
| 4 | stages, line 3 (pod), number 4 | 02:44 | docs/log.md, the entry "session 8, step 13: test predictions", 'stopped at about (\\S+) UTC' | 02:44 | PASS |
| 5 | stages, line 3 (pod), number 5 | 02:44:54 | docs/log.md, the entry "session 8, step 13: test predictions", 'terminated at (\\S+) UTC' | 02:44:54 | PASS |
| 6 | stages, line 3 (pod), number 6 | 13 | docs/log.md, the entry "session 8, step 13: test predictions", '(\\d+) hours \\d+ minutes, about' | 13 | PASS |
| 7 | stages, line 3 (pod), number 7 | 7 | docs/log.md, the entry "session 8, step 13: test predictions", '\\d+ hours (\\d+) minutes, about' | 7 | PASS |
| 8 | stages, line 3 (pod), number 8 | $46.02 | docs/log.md, the entry "session 8, step 13: test predictions", 'about \\$(\\S+) at that rate' | 46.020000 | PASS |
| 9 | stages, line 3 (pod), number 9 | $91.37 | docs/log.md, the entry "session 8, step 13: test predictions", 'the balance was \\$(\\S+) before the pod' | 91.370000 | PASS |
| 10 | stages, line 3 (pod), number 10 | $21.37 | docs/log.md, the entry "session 8, steps 8 and 9: the credit", 'added to \\$(\\S+), less' | 21.370000 | PASS |
| 11 | stages, line 3 (pod), number 11 | $70 | docs/log.md, the entry "session 8, steps 8 and 9: the credit", '\\(\\$(\\d+) added to \\$' | 70.000000 | PASS |
| 12 | stages, line 3 (pod), number 12 | $45.26 | docs/log.md, the entry "session 8, step 13: test predictions", 'and \\$(\\S+) after the shutdown' | 45.260000 | PASS |
| 13 | stages, line 3 (pod), number 13 | $46.11 | docs/log.md, the entry "session 8, step 13: test predictions", 'spend is about \\$(\\S+), under' | 46.110000 | PASS |
| 14 | stages, line 3 (pod), number 14 | $80 | docs/log.md, the entry "session 8, step 13: test predictions", 'under the \\$(\\d+) cap' | 80.000000 | PASS |
| 15 | Table 1, prepare, installs; the fixed revisions; EI's records rebuilt on the pod from the selection file and checked against the Mac's, Started | 13:46:35 | logs/pod/prepare.log, the first line holding 'stage prepare started' | 13:46:35 | PASS |
| 16 | Table 1, prepare, installs; the fixed revisions; EI's records rebuilt on the pod from the selection file and checked against the Mac's, Finished | 13:49:48 | logs/pod/prepare.log, the last line holding 'stage prepare finished' | 13:49:48 | PASS |
| 17 | Table 1, prepare_a, reader v1 in Laya's format; phase 2's records rebuilt and checked against labeler.json, Started | 13:50:53 | logs/pod/prepare_a.log, the first line holding 'stage prepare_a started' | 13:50:53 | PASS |
| 18 | Table 1, prepare_a, reader v1 in Laya's format; phase 2's records rebuilt and checked against labeler.json, Finished | 13:55:41 | logs/pod/prepare_a.log, the last line holding 'stage prepare_a finished' | 13:55:41 | PASS |
| 19 | Table 1, pilot, alone on the GPU: the two settings and the cost rule, Started | 13:56:42 | logs/pod/pilot.log, the first line holding 'stage pilot started' | 13:56:42 | PASS |
| 20 | Table 1, pilot, alone on the GPU: the two settings and the cost rule, Finished | 14:15:54 | logs/pod/pilot.log, the last line holding 'stage pilot finished' | 14:15:54 | PASS |
| 21 | Table 1, trees, on the pod's processors, xgboost_tuned's and catboost_tuned's searches and refits; the readmission pool on validation, Started | 14:16:12 | logs/pod/trees.log, the first line holding 'stage trees started' | 14:16:12 | PASS |
| 22 | Table 1, trees, on the pod's processors, xgboost_tuned's and catboost_tuned's searches and refits; the readmission pool on validation, Finished | 18:56:14 | logs/pod/trees.log, the last line holding 'stage trees finished' | 18:56:14 | PASS |
| 23 | Table 1, ei, `s8_ei_1`, trained beside arm A, then beside the first three smoke tests; validation predictions, Started | 14:16:54 | logs/pod/ei.log, the first line holding 'EI: s8_ei_1, seed' | 14:16:54 | PASS |
| 24 | Table 1, ei, `s8_ei_1`, trained beside arm A, then beside the first three smoke tests; validation predictions, Finished | 17:49:19 | logs/pod/ei.log, the first line holding 'EI: s8_ei_1 finished' | 17:49:19 | PASS |
| 25 | Table 1, ei, `s8_ei_2`, trained beside the fourth smoke test, and otherwise alone; validation predictions, Started | 17:49:19 | logs/pod/ei.log, the first line holding 'EI: s8_ei_2, seed' | 17:49:19 | PASS |
| 26 | Table 1, ei, `s8_ei_2`, trained beside the fourth smoke test, and otherwise alone; validation predictions, Finished | 20:00:53 | logs/pod/ei.log, the first line holding 'EI: s8_ei_2 finished' | 20:00:53 | PASS |
| 27 | Table 1, ei, `s8_ei_3`, trained alone on the GPU; validation predictions, Started | 20:00:53 | logs/pod/ei.log, the first line holding 'EI: s8_ei_3, seed' | 20:00:53 | PASS |
| 28 | Table 1, ei, `s8_ei_3`, trained alone on the GPU; validation predictions, Finished | 22:12:11 | logs/pod/ei.log, the first line holding 'EI: s8_ei_3 finished' | 22:12:11 | PASS |
| 29 | Table 1, arm_a, `s8_a1`, beside `s8_ei_1`, Started | 14:16:55 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a1 from' | 14:16:55 | PASS |
| 30 | Table 1, arm_a, `s8_a1`, beside `s8_ei_1`, Finished | 15:58:44 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a1 done' | 15:58:44 | PASS |
| 31 | Table 1, arm_a, `s8_a2`, beside `s8_ei_1`, Started | 15:58:44 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a2 from' | 15:58:44 | PASS |
| 32 | Table 1, arm_a, `s8_a2`, beside `s8_ei_1`, Finished | 16:14:23 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a2 done' | 16:14:23 | PASS |
| 33 | Table 1, arm_a, `s8_a3`, beside `s8_ei_1`; validation predictions, Started | 16:14:23 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a3 from' | 16:14:23 | PASS |
| 34 | Table 1, arm_a, `s8_a3`, beside `s8_ei_1`; validation predictions, Finished | 16:57:23 | logs/pod/arm_a.log, the first line holding 'arm A: s8_a3 done' | 16:57:23 | PASS |
| 35 | Table 1, prepare_large, the large model's environment and weights, in four runs within these times (corrections 7, 8 and 10), Started | 14:16:57 | logs/pod/prepare_large.log, the first line holding 'stage prepare_large started' | 14:16:57 | PASS |
| 36 | Table 1, prepare_large, the large model's environment and weights, in four runs within these times (corrections 7, 8 and 10), Finished | 17:01:04 | logs/pod/prepare_large.log, the last line holding 'stage prepare_large finished' | 17:01:04 | PASS |
| 37 | Table 1, smoke_large, the smoke test with Gemma 4 E2B-it, in four runs within these times (corrections 9 to 12), Started | 16:57:30 | logs/pod/smoke_large.log, the first line holding 'stage smoke_large started' | 16:57:30 | PASS |
| 38 | Table 1, smoke_large, the smoke test with Gemma 4 E2B-it, in four runs within these times (corrections 9 to 12), Finished | 18:19:32 | logs/pod/smoke_large.log, the last line holding 'stage smoke_large finished' | 18:19:32 | PASS |
| 39 | Table 1, large_validation, Gemma 4 31B-it alone on the GPU, on the validation items, Started | 22:12:42 | logs/pod/large_validation.log, the first line holding 'stage large_validation started' | 22:12:42 | PASS |
| 40 | Table 1, large_validation, Gemma 4 31B-it alone on the GPU, on the validation items, Finished | 22:56:40 | logs/pod/large_validation.log, the last line holding 'stage large_validation finished' | 22:56:40 | PASS |
| 41 | Table 1, calibrate, calibration_s8.json; the ECE limit, Started | 22:59:01 | logs/pod/calibrate.log, the first line holding 'stage calibrate started' | 22:59:01 | PASS |
| 42 | Table 1, calibrate, calibration_s8.json; the ECE limit, Finished | 22:59:17 | logs/pod/calibrate.log, the last line holding 'stage calibrate finished' | 22:59:17 | PASS |
| 43 | Table 1, freeze, release_candidate.json, Started | 23:00:14 | logs/pod/freeze.log, the first line holding 'stage freeze started' | 23:00:14 | PASS |
| 44 | Table 1, freeze, release_candidate.json, Finished | 23:00:15 | logs/pod/freeze.log, the last line holding 'stage freeze finished' | 23:00:15 | PASS |
| 45 | Table 1, testpreds (1 October), after the OSF registration (correction 13): testpreds_log.json first, then every test prediction, Started | 01:13:53 | logs/pod/testpreds.log, the first line holding 'stage testpreds started' | 01:13:53 | PASS |
| 46 | Table 1, testpreds (1 October), after the OSF registration (correction 13): testpreds_log.json first, then every test prediction, Finished | 02:09:13 | logs/pod/testpreds.log, the last line holding 'stage testpreds finished' | 02:09:13 | PASS |
| 47 | Table 1, package, the five zips and SHA256SUMS.txt, Started | 02:36:29 | logs/pod/package.log, the first line holding 'stage package started' | 02:36:29 | PASS |
| 48 | Table 1, package, the five zips and SHA256SUMS.txt, Finished | 02:36:36 | docs/log.md, the entry "session 8, step 13: test predictions", 'package ran from \\S+ to (\\S+) UTC' | 02:36:36 | PASS |
| 49 | Table 1, copy, the zips copied to the Mac and checked, Started | 02:37:32 | docs/log.md, the entry "session 8, step 13: test predictions", 'ran from (\\S+) to \\S+ UTC with a helper' | 02:37:32 | PASS |
| 50 | Table 1, copy, the zips copied to the Mac and checked, Finished | 02:42:19 | docs/log.md, the entry "session 8, step 13: test predictions", 'ran from \\S+ to (\\S+) UTC with a helper' | 02:42:19 | PASS |
| 51 | stages, line 28 (records), number 1 | 29 | records_manifest.json drops.PMCID in no split file | 29 | PASS |
| 52 | stages, line 28 (records), number 2 | 210 | records_manifest.json drops.train: no accepted row | 210 | PASS |
| 53 | stages, line 28 (records), number 3 | 21 | records_manifest.json drops.validation: no accepted row | 21 | PASS |
| 54 | stages, line 28 (records), number 4 | 28 | records_manifest.json drops.test: no accepted row | 28 | PASS |
| 55 | stages, line 28 (records), number 5 | 6 | records_manifest.json drops.test: empty article text | 6 | PASS |
| 56 | stages, line 28 (records), number 6 | 70 | records_manifest.json drops.train: listed as Incorrect in annotations/README.md | 70 | PASS |
| 57 | stages, line 28 (records), number 7 | 10,046 | records_manifest.json counts.train.prompts | 10,046 | PASS |
| 58 | stages, line 28 (records), number 8 | 2,657 | records_manifest.json counts.train.articles | 2,657 | PASS |
| 59 | stages, line 28 (records), number 9 | 3,054 | records_manifest.json counts.train.answers.0 | 3,054 | PASS |
| 60 | stages, line 28 (records), number 10 | 2,454 | records_manifest.json counts.train.answers.1 | 2,454 | PASS |
| 61 | stages, line 28 (records), number 11 | 4,538 | records_manifest.json counts.train.answers.2 | 4,538 | PASS |
| 62 | stages, line 28 (records), number 12 | 1,237 | records_manifest.json counts.validation.prompts | 1,237 | PASS |
| 63 | stages, line 28 (records), number 13 | 340 | records_manifest.json counts.validation.articles | 340 | PASS |
| 64 | stages, line 28 (records), number 14 | 1,218 | records_manifest.json counts.test.prompts | 1,218 | PASS |
| 65 | stages, line 28 (records), number 15 | 333 | records_manifest.json counts.test.articles | 333 | PASS |
| 66 | stages, line 28 (records), number 16 | 16 | records_manifest.json accepted_labels_disagree.train | 16 | PASS |
| 67 | stages, line 28 (records), number 17 | 1 | records_manifest.json accepted_labels_disagree.validation | 1 | PASS |
| 68 | stages, line 28 (records), number 18 | 3 | records_manifest.json accepted_labels_disagree.test | 3 | PASS |
| 69 | stages, line 30 (tokens), number 1 | 3,839 | records_manifest.json tokens_per_record.train.mean | 3839.037826 | PASS |
| 70 | stages, line 30 (tokens), number 2 | 3,831 | records_manifest.json tokens_per_record.validation.mean | 3831.042037 | PASS |
| 71 | stages, line 30 (tokens), number 3 | 3,870 | records_manifest.json tokens_per_record.test.mean | 3869.834154 | PASS |
| 72 | stages, line 30 (tokens), number 4 | 4,080 | records_manifest.json tokens_per_record.train.largest | 4,080 | PASS |
| 73 | stages, line 30 (tokens), number 5 | 4,080 | records_manifest.json tokens_per_record.validation.largest | 4,080 | PASS |
| 74 | stages, line 30 (tokens), number 6 | 4,080 | records_manifest.json tokens_per_record.test.largest | 4,080 | PASS |
| 75 | stages, line 30 (tokens), number 7 | 4,096 | article_selection.json max_len | 4,096 | PASS |
| 76 | stages, line 32 (coverage), number 1 | 9,487 | article_selection.json coverage.train.covered | 9,487 | PASS |
| 77 | stages, line 32 (coverage), number 2 | 9,850 | article_selection.json coverage.train.prompts_with_usable_offsets | 9,850 | PASS |
| 78 | stages, line 32 (coverage), number 3 | 96.31 | article_selection.json coverage.train.coverage | 96.314721 | PASS |
| 79 | stages, line 32 (coverage), number 4 | 95.76 | article_selection.json check_of_30_september_train | 95.760000 | PASS |
| 80 | stages, line 32 (coverage), number 5 | 1,151 | article_selection.json coverage.validation.covered | 1,151 | PASS |
| 81 | stages, line 32 (coverage), number 6 | 1,206 | article_selection.json coverage.validation.prompts_with_usable_offsets | 1,206 | PASS |
| 82 | stages, line 32 (coverage), number 7 | 95.44 | article_selection.json coverage.validation.coverage | 95.439469 | PASS |
| 83 | stages, line 34 (licenses), number 1 | 3,348 | records_manifest.json article_licenses.articles | 3,348 | PASS |
| 84 | stages, line 34 (licenses), number 2 | 1,807 | records_manifest.json article_licenses.tally.cc by | 1,807 | PASS |
| 85 | stages, line 34 (licenses), number 3 | 650 | records_manifest.json article_licenses.tally.cc by-nc | 650 | PASS |
| 86 | stages, line 34 (licenses), number 4 | 360 | records_manifest.json article_licenses.tally.cc by-nc-sa | 360 | PASS |
| 87 | stages, line 34 (licenses), number 5 | 298 | records_manifest.json article_licenses.tally.cc by-nc-nd | 298 | PASS |
| 88 | stages, line 34 (licenses), number 6 | 197 | records_manifest.json article_licenses.tally.none recorded | 197 | PASS |
| 89 | stages, line 34 (licenses), number 7 | 28 | records_manifest.json article_licenses.tally.cc0 | 28 | PASS |
| 90 | stages, line 34 (licenses), number 8 | 6 | records_manifest.json article_licenses.tally.cc by-nd | 6 | PASS |
| 91 | stages, line 34 (licenses), number 9 | 2 | records_manifest.json article_licenses.tally.cc by-sa | 2 | PASS |
| 92 | stages, line 36 (pilot), number 1 | 1 | pilot.json settings.0.micro_batch | 1 | PASS |
| 93 | stages, line 36 (pilot), number 2 | 32 | pilot.json settings.0.grad_accum | 32 | PASS |
| 94 | stages, line 36 (pilot), number 3 | 2.8558 | pilot.json settings.0.train_items_per_second | 2.855767 | PASS |
| 95 | stages, line 36 (pilot), number 4 | 36.40 | pilot.json settings.0.peak_gpu_memory_gib | 36.397950 | PASS |
| 96 | stages, line 36 (pilot), number 5 | 4 | pilot.json settings.1.micro_batch | 4 | PASS |
| 97 | stages, line 36 (pilot), number 6 | 8 | pilot.json settings.1.grad_accum | 8 | PASS |
| 98 | stages, line 36 (pilot), number 7 | 1.2923 | pilot.json settings.1.train_items_per_second | 1.292277 | PASS |
| 99 | stages, line 36 (pilot), number 8 | 15.39 | pilot.json settings.1.peak_gpu_memory_gib | 15.386986 | PASS |
| 100 | stages, line 36 (pilot), number 9 | 300 | pilot.json validation_items | 300 | PASS |
| 101 | stages, line 36 (pilot), number 10 | 5.3708 | pilot.json validation_items_per_second | 5.370822 | PASS |
| 102 | stages, line 36 (pilot), number 11 | $43.38 | docs/log.md, the entry "session 8, step 10: the pilot and the cost rule", 'projected total \\$(\\S+), under' | 43.380000 | PASS |
| 103 | stages, line 36 (pilot), number 12 | $75 | docs/log.md, the entry "session 8, step 10: the pilot and the cost rule", 'under the \\$(\\d+) line' | 75.000000 | PASS |
| 104 | stages, line 38 (seeds), number 1 | 16:57:23 | logs/pod/arm_a.log, the last line holding 'stage arm_a finished' | 16:57:23 | PASS |
| 105 | stages, line 38 (seeds), number 2 | 16:57:30 | logs/pod/smoke_large.log, the first line holding 'stage smoke_large started' | 16:57:30 | PASS |
| 106 | stages, line 38 (seeds), number 3 | 17:45:23 | logs/pod/ei.log, the first line holding 'EI: s8_ei_1 trained' | 17:45:23 | PASS |
| 107 | stages, line 38 (seeds), number 4 | 18:15:08 | logs/pod/smoke_large.log, the last line holding 'stage smoke_large started' | 18:15:08 | PASS |
| 108 | stages, line 38 (seeds), number 5 | 18:19:32 | logs/pod/smoke_large.log, the last line holding 'stage smoke_large finished' | 18:19:32 | PASS |
| 109 | stages, line 38 (seeds), number 6 | 19:56:53 | logs/pod/ei.log, the first line holding 'EI: s8_ei_2 trained' | 19:56:53 | PASS |
| 110 | stages, line 38 (seeds), number 7 | 22:08:13 | logs/pod/ei.log, the first line holding 'EI: s8_ei_3 trained' | 22:08:13 | PASS |
| 111 | stages, line 38 (seeds), number 8 | 6,539 | metrics/s8_a1_metrics.json optimizer_steps | 6,539 | PASS |
| 112 | stages, line 38 (seeds), number 9 | 957 | metrics/s8_a2_metrics.json optimizer_steps | 957 | PASS |
| 113 | stages, line 38 (seeds), number 10 | 2,739 | metrics/s8_a3_metrics.json optimizer_steps | 2,739 | PASS |
| 114 | stages, line 38 (seeds), number 11 | 35.75 | metrics/s8_a3_metrics.json sequences_per_second | 35.750000 | PASS |
| 115 | Table 2, `s8_ei_1`, Optimizer steps | 628 | metrics/s8_ei_1_metrics.json optimizer_steps | 628 | PASS |
| 116 | Table 2, `s8_ei_1`, Training seconds | 12,219.5 | metrics/s8_ei_1_metrics.json train_seconds | 12219.500000 | PASS |
| 117 | Table 2, `s8_ei_1`, Training items a second | 1.64 | metrics/s8_ei_1_metrics.json sequences_per_second | 1.640000 | PASS |
| 118 | Table 2, `s8_ei_1`, Peak GPU memory, GiB | 36.4 | metrics/s8_ei_1_metrics.json peak_gpu_memory_gib | 36.400000 | PASS |
| 119 | Table 2, `s8_ei_1`, Monitoring log loss at the last step | 0.1815 | metrics/s8_ei_1_metrics.json kept_validation_selection_log_loss_exact | 0.181500 | PASS |
| 120 | Table 2, `s8_ei_1`, Validation accuracy | 0.9386 | run_summaries/s8_ei_1_validation_run_summary.json validation_per_question_raw.evidence_effect.accuracy | 0.938600 | PASS |
| 121 | Table 2, `s8_ei_1`, Validation log loss | 0.2301 | run_summaries/s8_ei_1_validation_run_summary.json validation_per_question_raw.evidence_effect.log_loss | 0.230100 | PASS |
| 122 | Table 2, `s8_ei_1`, Validation ECE | 0.0348 | run_summaries/s8_ei_1_validation_run_summary.json validation_per_question_raw.evidence_effect.ece | 0.034800 | PASS |
| 123 | Table 2, `s8_ei_2`, Optimizer steps | 628 | metrics/s8_ei_2_metrics.json optimizer_steps | 628 | PASS |
| 124 | Table 2, `s8_ei_2`, Training seconds | 7,447.3 | metrics/s8_ei_2_metrics.json train_seconds | 7447.300000 | PASS |
| 125 | Table 2, `s8_ei_2`, Training items a second | 2.70 | metrics/s8_ei_2_metrics.json sequences_per_second | 2.700000 | PASS |
| 126 | Table 2, `s8_ei_2`, Peak GPU memory, GiB | 36.4 | metrics/s8_ei_2_metrics.json peak_gpu_memory_gib | 36.400000 | PASS |
| 127 | Table 2, `s8_ei_2`, Monitoring log loss at the last step | 0.1901 | metrics/s8_ei_2_metrics.json kept_validation_selection_log_loss_exact | 0.190100 | PASS |
| 128 | Table 2, `s8_ei_2`, Validation accuracy | 0.9345 | run_summaries/s8_ei_2_validation_run_summary.json validation_per_question_raw.evidence_effect.accuracy | 0.934500 | PASS |
| 129 | Table 2, `s8_ei_2`, Validation log loss | 0.2470 | run_summaries/s8_ei_2_validation_run_summary.json validation_per_question_raw.evidence_effect.log_loss | 0.247000 | PASS |
| 130 | Table 2, `s8_ei_2`, Validation ECE | 0.0424 | run_summaries/s8_ei_2_validation_run_summary.json validation_per_question_raw.evidence_effect.ece | 0.042400 | PASS |
| 131 | Table 2, `s8_ei_3`, Optimizer steps | 628 | metrics/s8_ei_3_metrics.json optimizer_steps | 628 | PASS |
| 132 | Table 2, `s8_ei_3`, Training seconds | 7,430.4 | metrics/s8_ei_3_metrics.json train_seconds | 7430.400000 | PASS |
| 133 | Table 2, `s8_ei_3`, Training items a second | 2.70 | metrics/s8_ei_3_metrics.json sequences_per_second | 2.700000 | PASS |
| 134 | Table 2, `s8_ei_3`, Peak GPU memory, GiB | 36.4 | metrics/s8_ei_3_metrics.json peak_gpu_memory_gib | 36.400000 | PASS |
| 135 | Table 2, `s8_ei_3`, Monitoring log loss at the last step | 0.1949 | metrics/s8_ei_3_metrics.json kept_validation_selection_log_loss_exact | 0.194900 | PASS |
| 136 | Table 2, `s8_ei_3`, Validation accuracy | 0.9337 | run_summaries/s8_ei_3_validation_run_summary.json validation_per_question_raw.evidence_effect.accuracy | 0.933700 | PASS |
| 137 | Table 2, `s8_ei_3`, Validation log loss | 0.2343 | run_summaries/s8_ei_3_validation_run_summary.json validation_per_question_raw.evidence_effect.log_loss | 0.234300 | PASS |
| 138 | Table 2, `s8_ei_3`, Validation ECE | 0.0373 | run_summaries/s8_ei_3_validation_run_summary.json validation_per_question_raw.evidence_effect.ece | 0.037300 | PASS |
| 139 | stages, line 48 (trees), number 1 | 50 | trees.json members.xgboost_tuned.settings_drawn.drawn | 50 | PASS |
| 140 | stages, line 48 (trees), number 2 | 1,620 | trees.json members.xgboost_tuned.settings_drawn.from | 1,620 | PASS |
| 141 | stages, line 48 (trees), number 3 | 5 | trees.json folds | 5 | PASS |
| 142 | stages, line 48 (trees), number 4 | 55,978 | trees.json members.xgboost_tuned.refit.rows | 55,978 | PASS |
| 143 | stages, line 48 (trees), number 5 | 10 | trees.json members.xgboost_tuned.chosen.max_depth | 10 | PASS |
| 144 | stages, line 48 (trees), number 6 | 20 | trees.json members.xgboost_tuned.chosen.min_child_weight | 20 | PASS |
| 145 | stages, line 48 (trees), number 7 | 0.02 | trees.json members.xgboost_tuned.chosen.eta | 0.020000 | PASS |
| 146 | stages, line 48 (trees), number 8 | 0.8 | trees.json members.xgboost_tuned.chosen.subsample | 0.800000 | PASS |
| 147 | stages, line 48 (trees), number 9 | 0.8 | trees.json members.xgboost_tuned.chosen.colsample_bytree | 0.800000 | PASS |
| 148 | stages, line 48 (trees), number 10 | 10 | trees.json members.xgboost_tuned.chosen.lambda | 10 | PASS |
| 149 | stages, line 48 (trees), number 11 | 333 | trees.json members.xgboost_tuned.refit.trees | 333 | PASS |
| 150 | stages, line 48 (trees), number 12 | 30 | trees.json members.catboost_tuned.settings_drawn.drawn | 30 | PASS |
| 151 | stages, line 48 (trees), number 13 | 144 | trees.json members.catboost_tuned.settings_drawn.from | 144 | PASS |
| 152 | stages, line 48 (trees), number 14 | 8 | trees.json members.catboost_tuned.chosen.depth | 8 | PASS |
| 153 | stages, line 48 (trees), number 15 | 0.03 | trees.json members.catboost_tuned.chosen.learning_rate | 0.030000 | PASS |
| 154 | stages, line 48 (trees), number 16 | 3 | trees.json members.catboost_tuned.chosen.l2_leaf_reg | 3 | PASS |
| 155 | stages, line 48 (trees), number 17 | 1 | trees.json members.catboost_tuned.chosen.bagging_temperature | 1 | PASS |
| 156 | stages, line 48 (trees), number 18 | 254 | trees.json members.catboost_tuned.chosen.border_count | 254 | PASS |
| 157 | stages, line 48 (trees), number 19 | 1,222 | trees.json members.catboost_tuned.refit.trees | 1,222 | PASS |
| 158 | stages, line 50 (pool), number 1 | 6,997 | readmission_pool.json rows | 6,997 | PASS |
| 159 | stages, line 50 (pool), number 2 | 0.4 | readmission_pool.json weights.xgboost_tuned | 0.400000 | PASS |
| 160 | stages, line 50 (pool), number 3 | 0.6 | readmission_pool.json weights.catboost_tuned | 0.600000 | PASS |
| 161 | stages, line 50 (pool), number 4 | 0.0 | readmission_pool.json weights.lightgbm_tuned | 0.000000 | PASS |
| 162 | stages, line 50 (pool), number 5 | 0.0 | readmission_pool.json weights.arm_laya | 0.000000 | PASS |
| 163 | stages, line 50 (pool), number 6 | 0.0 | readmission_pool.json weights.s7_b3 | 0.000000 | PASS |
| 164 | stages, line 50 (pool), number 7 | 0.9957 | readmission_pool.json T | 0.995725 | PASS |
| 165 | stages, line 50 (pool), number 8 | 0.83679 | readmission_pool.json validation_log_loss | 0.836789 | PASS |
| 166 | stages, line 50 (pool), number 9 | 0.83715 | readmission_pool.json out_of_fold_log_loss.mean | 0.837153 | PASS |
| 167 | stages, line 52 (large), number 1 | 1,237 | large_model/gemma4_31b_timing.json evidence_effect.validation.items | 1,237 | PASS |
| 168 | stages, line 52 (large), number 2 | 3,754 | large_model/gemma4_31b_timing.json evidence_effect.validation.prompt_tokens_per_item | 3754.189976 | PASS |
| 169 | stages, line 52 (large), number 3 | 1.907 | large_model/gemma4_31b_timing.json evidence_effect.validation.gpu_seconds_per_item | 1.906613 | PASS |
| 170 | stages, line 52 (large), number 4 | 1,218 | large_model/gemma4_31b_timing.json evidence_effect.test.items | 1,218 | PASS |
| 171 | stages, line 52 (large), number 5 | 1.620 | large_model/gemma4_31b_timing.json evidence_effect.test.gpu_seconds_per_item | 1.619837 | PASS |
| 172 | stages, line 52 (large), number 6 | 150 | large_model/gemma4_31b_timing.json pubmedqa_answer.validation.items | 150 | PASS |
| 173 | stages, line 52 (large), number 7 | 0.945 | large_model/gemma4_31b_timing.json pubmedqa_answer.validation.gpu_seconds_per_item | 0.945057 | PASS |
| 174 | stages, line 52 (large), number 8 | 500 | large_model/gemma4_31b_timing.json pubmedqa_answer.test.items | 500 | PASS |
| 175 | stages, line 52 (large), number 9 | 0.804 | large_model/gemma4_31b_timing.json pubmedqa_answer.test.gpu_seconds_per_item | 0.804037 | PASS |
| 176 | stages, line 54 (freeze), number 1 | 0.05 | comparison.json ece_limits evidence_effect.limit | 0.050000 | PASS |
| 177 | stages, line 54 (freeze), number 2 | 1,218 | ece_limit_evidence_effect.json n | 1,218 | PASS |
| 178 | stages, line 54 (freeze), number 3 | 5,000 | ece_limit_evidence_effect.json draws | 5,000 | PASS |
| 179 | stages, line 54 (freeze), number 4 | 0.0202 | ece_limit_evidence_effect.json ece_if_calibrated.p95 | 0.020244 | PASS |
| 180 | stages, line 54 (freeze), number 5 | 0.05 | comparison.json ece_limits evidence_effect.floor | 0.050000 | PASS |
| 181 | stages, line 58 (wait), number 1 | $7.80 | docs/log.md, the entry "session 8: the OSF registration filed", 'was about \\$(\\S+)' | 7.800000 | PASS |
| 182 | stages, line 60 (testpreds), number 1 | 221.8 | run_summaries/s8_ei_1_test_run_summary.json timing_test.forward_seconds | 221.814813 | PASS |
| 183 | stages, line 60 (testpreds), number 2 | 221.8 | run_summaries/s8_ei_2_test_run_summary.json timing_test.forward_seconds | 221.836435 | PASS |
| 184 | stages, line 60 (testpreds), number 3 | 222.1 | run_summaries/s8_ei_3_test_run_summary.json timing_test.forward_seconds | 222.111533 | PASS |
| 185 | stages, line 60 (testpreds), number 4 | 1,218 | run_summaries/s8_ei_1_test_run_summary.json timing_test.items | 1,218 | PASS |
| 186 | stages, line 60 (testpreds), number 5 | 10.0 | run_summaries/s7_b3_timing_test_run_summary.json timing_test.forward_seconds | 9.992877 | PASS |
| 187 | stages, line 60 (testpreds), number 6 | 500 | run_summaries/s7_b3_timing_test_run_summary.json timing_test.items | 500 | PASS |
| 188 | stages, line 62 (package), number 1 | 5 | models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt, its lines | 5 | PASS |
| 189 | stages, line 62 (package), number 2 | 3,411,585,854 | models/fiorillo_v0_4/runpod_zips/SHA256SUMS.txt, the sum of the sizes | 3,411,585,854 | PASS |
| 190 | stages, line 62 (package), number 3 | 27 | docs/log.md, the entry "session 8, step 13: test predictions", 'idle from \\S+ to \\S+ UTC, (\\d+) minutes' | 27 | PASS |
| 191 | stages, line 62 (package), number 4 | $1.60 | docs/log.md, the entry "session 8, step 13: test predictions", 'minutes \\(about \\$(\\S+)\\)' | 1.600000 | PASS |
| 192 | stages, line 70 (analyses), number 1 | 02:52:31 | docs/log.md, the entry "session 8, step 15: results", 'apply on the test split at (\\S+) UTC' | 02:52:31 | PASS |
| 193 | stages, line 70 (analyses), number 2 | 02:54:23 | docs/log.md, the entry "session 8, step 15: results", 'compare_session8.py from \\S+ to (\\S+) UTC' | 02:54:23 | PASS |
| 194 | results, line 3 (intro), number 1 | 23:00:15 | logs/pod/freeze.log, the last line holding 'stage freeze finished' | 23:00:15 | PASS |
| 195 | results, line 3 (intro), number 2 | 01:13:53 | testpreds_log.json testpreds_start_utc | 01:13:53 | PASS |
| 196 | results, line 3 (intro), number 3 | 1,218 | derived: data/processed/session8/evidence_effect_test.jsonl: records | 1,218 | PASS |
| 197 | results, line 3 (intro), number 4 | 333 | records_manifest.json counts.test.articles | 333 | PASS |
| 198 | results, line 3 (intro), number 5 | 411 | derived: data/processed/session8/evidence_effect_test.jsonl: records with label 0 | 411 | PASS |
| 199 | results, line 3 (intro), number 6 | 294 | derived: data/processed/session8/evidence_effect_test.jsonl: records with label 1 | 294 | PASS |
| 200 | results, line 3 (intro), number 7 | 513 | derived: data/processed/session8/evidence_effect_test.jsonl: records with label 2 | 513 | PASS |
| 201 | results, line 3 (intro), number 8 | 500 | derived: data/processed/public/pubmedqa_answer_test.jsonl: records | 500 | PASS |
| 202 | results, line 3 (intro), number 9 | 6,998 | derived: data/processed/public/readmit_window_test.jsonl: records | 6,998 | PASS |
| 203 | results, line 7 (intervals), number 1 | 1,000 | comparison.json settings resamples | 1,000 | PASS |
| 204 | results, line 7 (intervals), number 2 | 10,000 | comparison.json settings resamples_bar | 10,000 | PASS |
| 205 | results, line 11 (cal), number 1 | 1.460 | calibration_s8.json s8_ei.evidence_effect.families.temperature.temperature | 1.460042 | PASS |
| 206 | results, line 11 (cal), number 2 | 10.000 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature_intercepts.temperature | 10.000000 | PASS |
| 207 | results, line 11 (cal), number 3 | 1.9906 | calibration_s8.json gemma4_31b.evidence_effect.validation_calibrate_items.raw_log_loss | 1.990639 | PASS |
| 208 | results, line 11 (cal), number 4 | 0.3403 | calibration_s8.json gemma4_31b.evidence_effect.validation_calibrate_items.log_loss | 0.340311 | PASS |
| 209 | results, line 11 (cal), number 5 | 0.3424 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.342442 | PASS |
| 210 | Table 3, `s8_ei`, Evidence Inference, Out-of-fold log loss, temperature | 0.1994 | calibration_s8.json s8_ei.evidence_effect.families.temperature.out_of_fold_log_loss.mean | 0.199444 | PASS |
| 211 | Table 3, `s8_ei`, Evidence Inference, Out-of-fold log loss, temperature and intercepts | 0.2004 | calibration_s8.json s8_ei.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.200380 | PASS |
| 212 | Table 3, `s8_ei`, Evidence Inference, Temperature | 1.460 | calibration_s8.json s8_ei.evidence_effect.families.temperature.temperature | 1.460042 | PASS |
| 213 | Table 3, `s8_ei`, Evidence Inference, Intercept, second answer | 0.000 | calibration_s8.json s8_ei.evidence_effect.families.temperature.intercepts.1 | 0.000000 | PASS |
| 214 | Table 3, `s8_ei`, Evidence Inference, Intercept, third answer | 0.000 | calibration_s8.json s8_ei.evidence_effect.families.temperature.intercepts.2 | 0.000000 | PASS |
| 215 | Table 3, `s8_ei_1`, Evidence Inference, Out-of-fold log loss, temperature | 0.1990 | calibration_s8.json s8_ei_1.evidence_effect.families.temperature.out_of_fold_log_loss.mean | 0.199044 | PASS |
| 216 | Table 3, `s8_ei_1`, Evidence Inference, Out-of-fold log loss, temperature and intercepts | 0.2005 | calibration_s8.json s8_ei_1.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.200534 | PASS |
| 217 | Table 3, `s8_ei_1`, Evidence Inference, Temperature | 1.512 | calibration_s8.json s8_ei_1.evidence_effect.families.temperature.temperature | 1.511885 | PASS |
| 218 | Table 3, `s8_ei_1`, Evidence Inference, Intercept, second answer | 0.000 | calibration_s8.json s8_ei_1.evidence_effect.families.temperature.intercepts.1 | 0.000000 | PASS |
| 219 | Table 3, `s8_ei_1`, Evidence Inference, Intercept, third answer | 0.000 | calibration_s8.json s8_ei_1.evidence_effect.families.temperature.intercepts.2 | 0.000000 | PASS |
| 220 | Table 3, `s8_ei_2`, Evidence Inference, Out-of-fold log loss, temperature | 0.2083 | calibration_s8.json s8_ei_2.evidence_effect.families.temperature.out_of_fold_log_loss.mean | 0.208299 | PASS |
| 221 | Table 3, `s8_ei_2`, Evidence Inference, Out-of-fold log loss, temperature and intercepts | 0.2090 | calibration_s8.json s8_ei_2.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.209031 | PASS |
| 222 | Table 3, `s8_ei_2`, Evidence Inference, Temperature | 1.567 | calibration_s8.json s8_ei_2.evidence_effect.families.temperature.temperature | 1.567308 | PASS |
| 223 | Table 3, `s8_ei_2`, Evidence Inference, Intercept, second answer | 0.000 | calibration_s8.json s8_ei_2.evidence_effect.families.temperature.intercepts.1 | 0.000000 | PASS |
| 224 | Table 3, `s8_ei_2`, Evidence Inference, Intercept, third answer | 0.000 | calibration_s8.json s8_ei_2.evidence_effect.families.temperature.intercepts.2 | 0.000000 | PASS |
| 225 | Table 3, `s8_ei_3`, Evidence Inference, Out-of-fold log loss, temperature | 0.2070 | calibration_s8.json s8_ei_3.evidence_effect.families.temperature.out_of_fold_log_loss.mean | 0.206982 | PASS |
| 226 | Table 3, `s8_ei_3`, Evidence Inference, Out-of-fold log loss, temperature and intercepts | 0.2073 | calibration_s8.json s8_ei_3.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.207312 | PASS |
| 227 | Table 3, `s8_ei_3`, Evidence Inference, Temperature | 1.459 | calibration_s8.json s8_ei_3.evidence_effect.families.temperature.temperature | 1.459329 | PASS |
| 228 | Table 3, `s8_ei_3`, Evidence Inference, Intercept, second answer | 0.000 | calibration_s8.json s8_ei_3.evidence_effect.families.temperature.intercepts.1 | 0.000000 | PASS |
| 229 | Table 3, `s8_ei_3`, Evidence Inference, Intercept, third answer | 0.000 | calibration_s8.json s8_ei_3.evidence_effect.families.temperature.intercepts.2 | 0.000000 | PASS |
| 230 | Table 3, `gemma4_31b`, Evidence Inference, Out-of-fold log loss, temperature | 0.3956 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature.out_of_fold_log_loss.mean | 0.395649 | PASS |
| 231 | Table 3, `gemma4_31b`, Evidence Inference, Out-of-fold log loss, temperature and intercepts | 0.3424 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.342442 | PASS |
| 232 | Table 3, `gemma4_31b`, Evidence Inference, Temperature | 6.255 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature_intercepts.temperature | 6.254625 | PASS |
| 233 | Table 3, `gemma4_31b`, Evidence Inference, Intercept, second answer | -1.850 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature_intercepts.intercepts.1 | -1.850335 | PASS |
| 234 | Table 3, `gemma4_31b`, Evidence Inference, Intercept, third answer | -1.304 | calibration_s8.json gemma4_31b.evidence_effect.families.temperature_intercepts.intercepts.2 | -1.304220 | PASS |
| 235 | Table 3, `gemma4_31b`, PubMedQA, Out-of-fold log loss, temperature | 0.6579 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature.out_of_fold_log_loss.mean | 0.657889 | PASS |
| 236 | Table 3, `gemma4_31b`, PubMedQA, Out-of-fold log loss, temperature and intercepts | 0.6004 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.600398 | PASS |
| 237 | Table 3, `gemma4_31b`, PubMedQA, Temperature | 10.000 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature_intercepts.temperature | 10.000000 | PASS |
| 238 | Table 3, `gemma4_31b`, PubMedQA, Intercept, second answer | -0.318 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature_intercepts.intercepts.1 | -0.318247 | PASS |
| 239 | Table 3, `gemma4_31b`, PubMedQA, Intercept, third answer | -1.348 | calibration_s8.json gemma4_31b.pubmedqa_answer.families.temperature_intercepts.intercepts.2 | -1.348346 | PASS |
| 240 | Table 3, `s8_a3`, PubMedQA, Out-of-fold log loss, temperature | 0.6609 | calibration_s8.json s8_a3.pubmedqa_answer.families.temperature.out_of_fold_log_loss.mean | 0.660917 | PASS |
| 241 | Table 3, `s8_a3`, PubMedQA, Out-of-fold log loss, temperature and intercepts | 0.6089 | calibration_s8.json s8_a3.pubmedqa_answer.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.608893 | PASS |
| 242 | Table 3, `s8_a3`, PubMedQA, Temperature | 2.318 | calibration_s8.json s8_a3.pubmedqa_answer.families.temperature_intercepts.temperature | 2.317514 | PASS |
| 243 | Table 3, `s8_a3`, PubMedQA, Intercept, second answer | -0.309 | calibration_s8.json s8_a3.pubmedqa_answer.families.temperature_intercepts.intercepts.1 | -0.309133 | PASS |
| 244 | Table 3, `s8_a3`, PubMedQA, Intercept, third answer | -1.271 | calibration_s8.json s8_a3.pubmedqa_answer.families.temperature_intercepts.intercepts.2 | -1.270532 | PASS |
| 245 | Table 3, `s8_a3`, Readmission, Out-of-fold log loss, temperature | 0.8537 | calibration_s8.json s8_a3.readmit_window.families.temperature.out_of_fold_log_loss.mean | 0.853658 | PASS |
| 246 | Table 3, `s8_a3`, Readmission, Out-of-fold log loss, temperature and intercepts | 0.8536 | calibration_s8.json s8_a3.readmit_window.families.temperature_intercepts.out_of_fold_log_loss.mean | 0.853603 | PASS |
| 247 | Table 3, `s8_a3`, Readmission, Temperature | 0.962 | calibration_s8.json s8_a3.readmit_window.families.temperature_intercepts.temperature | 0.962112 | PASS |
| 248 | Table 3, `s8_a3`, Readmission, Intercept, second answer | -0.009 | calibration_s8.json s8_a3.readmit_window.families.temperature_intercepts.intercepts.1 | -0.009317 | PASS |
| 249 | Table 3, `s8_a3`, Readmission, Intercept, third answer | 0.088 | calibration_s8.json s8_a3.readmit_window.families.temperature_intercepts.intercepts.2 | 0.088365 | PASS |
| 250 | Table 4, Evidence Inference, validation, Items | 1,237 | large_model/gemma4_31b_letters.json evidence_effect.validation.items | 1,237 | PASS |
| 251 | Table 4, Evidence Inference, validation, Letters' summed probability, median | 1.000000 | large_model/gemma4_31b_letters.json evidence_effect.validation.letters_summed_probability.median | 1.000000 | PASS |
| 252 | Table 4, Evidence Inference, validation, 10th percentile | 0.999998 | large_model/gemma4_31b_letters.json evidence_effect.validation.letters_summed_probability.p10 | 0.999998 | PASS |
| 253 | Table 4, Evidence Inference, validation, Share below one half | 0.0016 | large_model/gemma4_31b_letters.json evidence_effect.validation.letters_summed_probability.share_below_0_5 | 0.001617 | PASS |
| 254 | Table 4, Evidence Inference, validation, Answered from the top 20 | 153 | large_model/gemma4_31b_letters.json evidence_effect.validation.by_method.top20 | 153 | PASS |
| 255 | Table 4, Evidence Inference, validation, From the top 1,000 | 497 | large_model/gemma4_31b_letters.json evidence_effect.validation.by_method.top1000 | 497 | PASS |
| 256 | Table 4, Evidence Inference, validation, From the prompt's log-probabilities | 587 | large_model/gemma4_31b_letters.json evidence_effect.validation.by_method.prompt | 587 | PASS |
| 257 | Table 4, Evidence Inference, validation, GPU seconds an item | 1.907 | large_model/gemma4_31b_timing.json evidence_effect.validation.gpu_seconds_per_item | 1.906613 | PASS |
| 258 | Table 4, Evidence Inference, test, Items | 1,218 | large_model/gemma4_31b_letters.json evidence_effect.test.items | 1,218 | PASS |
| 259 | Table 4, Evidence Inference, test, Letters' summed probability, median | 1.000000 | large_model/gemma4_31b_letters.json evidence_effect.test.letters_summed_probability.median | 1.000000 | PASS |
| 260 | Table 4, Evidence Inference, test, 10th percentile | 0.999997 | large_model/gemma4_31b_letters.json evidence_effect.test.letters_summed_probability.p10 | 0.999997 | PASS |
| 261 | Table 4, Evidence Inference, test, Share below one half | 0.0000 | large_model/gemma4_31b_letters.json evidence_effect.test.letters_summed_probability.share_below_0_5 | 0.000000 | PASS |
| 262 | Table 4, Evidence Inference, test, Answered from the top 20 | 166 | large_model/gemma4_31b_letters.json evidence_effect.test.by_method.top20 | 166 | PASS |
| 263 | Table 4, Evidence Inference, test, From the top 1,000 | 529 | large_model/gemma4_31b_letters.json evidence_effect.test.by_method.top1000 | 529 | PASS |
| 264 | Table 4, Evidence Inference, test, From the prompt's log-probabilities | 523 | large_model/gemma4_31b_letters.json evidence_effect.test.by_method.prompt | 523 | PASS |
| 265 | Table 4, Evidence Inference, test, GPU seconds an item | 1.620 | large_model/gemma4_31b_timing.json evidence_effect.test.gpu_seconds_per_item | 1.619837 | PASS |
| 266 | Table 4, PubMedQA, validation, Items | 150 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.items | 150 | PASS |
| 267 | Table 4, PubMedQA, validation, Letters' summed probability, median | 0.999999 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.letters_summed_probability.median | 0.999999 | PASS |
| 268 | Table 4, PubMedQA, validation, 10th percentile | 0.999999 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.letters_summed_probability.p10 | 0.999999 | PASS |
| 269 | Table 4, PubMedQA, validation, Share below one half | 0.0000 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.letters_summed_probability.share_below_0_5 | 0.000000 | PASS |
| 270 | Table 4, PubMedQA, validation, Answered from the top 20 | 41 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.by_method.top20 | 41 | PASS |
| 271 | Table 4, PubMedQA, validation, From the top 1,000 | 56 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.by_method.top1000 | 56 | PASS |
| 272 | Table 4, PubMedQA, validation, From the prompt's log-probabilities | 53 | large_model/gemma4_31b_letters.json pubmedqa_answer.validation.by_method.prompt | 53 | PASS |
| 273 | Table 4, PubMedQA, validation, GPU seconds an item | 0.945 | large_model/gemma4_31b_timing.json pubmedqa_answer.validation.gpu_seconds_per_item | 0.945057 | PASS |
| 274 | Table 4, PubMedQA, test, Items | 500 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.items | 500 | PASS |
| 275 | Table 4, PubMedQA, test, Letters' summed probability, median | 0.999999 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.letters_summed_probability.median | 0.999999 | PASS |
| 276 | Table 4, PubMedQA, test, 10th percentile | 0.999998 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.letters_summed_probability.p10 | 0.999998 | PASS |
| 277 | Table 4, PubMedQA, test, Share below one half | 0.0000 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.letters_summed_probability.share_below_0_5 | 0.000000 | PASS |
| 278 | Table 4, PubMedQA, test, Answered from the top 20 | 138 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.by_method.top20 | 138 | PASS |
| 279 | Table 4, PubMedQA, test, From the top 1,000 | 186 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.by_method.top1000 | 186 | PASS |
| 280 | Table 4, PubMedQA, test, From the prompt's log-probabilities | 176 | large_model/gemma4_31b_letters.json pubmedqa_answer.test.by_method.prompt | 176 | PASS |
| 281 | Table 4, PubMedQA, test, GPU seconds an item | 0.804 | large_model/gemma4_31b_timing.json pubmedqa_answer.test.gpu_seconds_per_item | 0.804037 | PASS |
| 282 | results, line 39 (ei), number 1 | 0.9399 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.estimate | 0.939909 | PASS |
| 283 | results, line 39 (ei), number 2 | 0.9225 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.low | 0.922543 | PASS |
| 284 | results, line 39 (ei), number 3 | 0.9544 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.high | 0.954353 | PASS |
| 285 | results, line 39 (ei), number 4 | 0.1994 | comparison.json metrics.evidence_effect.s8_ei.log_loss.estimate | 0.199449 | PASS |
| 286 | results, line 39 (ei), number 5 | 0.1573 | comparison.json metrics.evidence_effect.s8_ei.log_loss.low | 0.157254 | PASS |
| 287 | results, line 39 (ei), number 6 | 0.2486 | comparison.json metrics.evidence_effect.s8_ei.log_loss.high | 0.248619 | PASS |
| 288 | results, line 39 (ei), number 7 | 0.8626 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.estimate | 0.862587 | PASS |
| 289 | results, line 39 (ei), number 8 | 0.8333 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.low | 0.833263 | PASS |
| 290 | results, line 39 (ei), number 9 | 0.8879 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.high | 0.887909 | PASS |
| 291 | results, line 39 (ei), number 10 | 0.4078 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.estimate | 0.407805 | PASS |
| 292 | results, line 39 (ei), number 11 | 0.3373 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.low | 0.337272 | PASS |
| 293 | results, line 39 (ei), number 12 | 0.4804 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.high | 0.480403 | PASS |
| 294 | results, line 41 (ece_note), number 1 | 0.0119 | comparison.json metrics.evidence_effect.s8_ei.ece.estimate | 0.011914 | PASS |
| 295 | Table 5, `prior`, Log loss, estimate | 1.0767 | comparison.json metrics.evidence_effect.prior.log_loss.estimate | 1.076717 | PASS |
| 296 | Table 5, `prior`, Log loss, low | 1.0580 | comparison.json metrics.evidence_effect.prior.log_loss.low | 1.058008 | PASS |
| 297 | Table 5, `prior`, Log loss, high | 1.0940 | comparison.json metrics.evidence_effect.prior.log_loss.high | 1.094031 | PASS |
| 298 | Table 5, `prior`, Brier score, estimate | 0.6525 | comparison.json metrics.evidence_effect.prior.brier.estimate | 0.652536 | PASS |
| 299 | Table 5, `prior`, Brier score, low | 0.6395 | comparison.json metrics.evidence_effect.prior.brier.low | 0.639467 | PASS |
| 300 | Table 5, `prior`, Brier score, high | 0.6643 | comparison.json metrics.evidence_effect.prior.brier.high | 0.664342 | PASS |
| 301 | Table 5, `prior`, ECE, estimate | 0.0305 | comparison.json metrics.evidence_effect.prior.ece.estimate | 0.030540 | PASS |
| 302 | Table 5, `prior`, ECE, low | 0.0018 | comparison.json metrics.evidence_effect.prior.ece.low | 0.001807 | PASS |
| 303 | Table 5, `prior`, ECE, high | 0.0663 | comparison.json metrics.evidence_effect.prior.ece.high | 0.066258 | PASS |
| 304 | Table 5, `prior`, Accuracy, estimate | 0.4212 | comparison.json metrics.evidence_effect.prior.accuracy.estimate | 0.421182 | PASS |
| 305 | Table 5, `prior`, Accuracy, low | 0.3855 | comparison.json metrics.evidence_effect.prior.accuracy.low | 0.385464 | PASS |
| 306 | Table 5, `prior`, Accuracy, high | 0.4584 | comparison.json metrics.evidence_effect.prior.accuracy.high | 0.458370 | PASS |
| 307 | Table 5, `prior`, Macro-F1, estimate | 0.1976 | comparison.json metrics.evidence_effect.prior.macro_f1.estimate | 0.197574 | PASS |
| 308 | Table 5, `prior`, Macro-F1, low | 0.1855 | comparison.json metrics.evidence_effect.prior.macro_f1.low | 0.185480 | PASS |
| 309 | Table 5, `prior`, Macro-F1, high | 0.2095 | comparison.json metrics.evidence_effect.prior.macro_f1.high | 0.209535 | PASS |
| 310 | Table 5, `s8_ei`, Log loss, estimate | 0.1994 | comparison.json metrics.evidence_effect.s8_ei.log_loss.estimate | 0.199449 | PASS |
| 311 | Table 5, `s8_ei`, Log loss, low | 0.1573 | comparison.json metrics.evidence_effect.s8_ei.log_loss.low | 0.157254 | PASS |
| 312 | Table 5, `s8_ei`, Log loss, high | 0.2486 | comparison.json metrics.evidence_effect.s8_ei.log_loss.high | 0.248619 | PASS |
| 313 | Table 5, `s8_ei`, Brier score, estimate | 0.1001 | comparison.json metrics.evidence_effect.s8_ei.brier.estimate | 0.100116 | PASS |
| 314 | Table 5, `s8_ei`, Brier score, low | 0.0778 | comparison.json metrics.evidence_effect.s8_ei.brier.low | 0.077771 | PASS |
| 315 | Table 5, `s8_ei`, Brier score, high | 0.1266 | comparison.json metrics.evidence_effect.s8_ei.brier.high | 0.126561 | PASS |
| 316 | Table 5, `s8_ei`, ECE, estimate | 0.0119 | comparison.json metrics.evidence_effect.s8_ei.ece.estimate | 0.011914 | PASS |
| 317 | Table 5, `s8_ei`, ECE, low | 0.0119 | comparison.json metrics.evidence_effect.s8_ei.ece.low | 0.011950 | PASS |
| 318 | Table 5, `s8_ei`, ECE, high | 0.0300 | comparison.json metrics.evidence_effect.s8_ei.ece.high | 0.029970 | PASS |
| 319 | Table 5, `s8_ei`, Accuracy, estimate | 0.9409 | comparison.json metrics.evidence_effect.s8_ei.accuracy.estimate | 0.940887 | PASS |
| 320 | Table 5, `s8_ei`, Accuracy, low | 0.9246 | comparison.json metrics.evidence_effect.s8_ei.accuracy.low | 0.924604 | PASS |
| 321 | Table 5, `s8_ei`, Accuracy, high | 0.9554 | comparison.json metrics.evidence_effect.s8_ei.accuracy.high | 0.955372 | PASS |
| 322 | Table 5, `s8_ei`, Macro-F1, estimate | 0.9399 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.estimate | 0.939909 | PASS |
| 323 | Table 5, `s8_ei`, Macro-F1, low | 0.9225 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.low | 0.922543 | PASS |
| 324 | Table 5, `s8_ei`, Macro-F1, high | 0.9544 | comparison.json metrics.evidence_effect.s8_ei.macro_f1.high | 0.954353 | PASS |
| 325 | Table 5, `s8_ei_temperature`, Log loss, estimate | 0.1994 | comparison.json metrics.evidence_effect.s8_ei_temperature.log_loss.estimate | 0.199449 | PASS |
| 326 | Table 5, `s8_ei_temperature`, Log loss, low | 0.1573 | comparison.json metrics.evidence_effect.s8_ei_temperature.log_loss.low | 0.157254 | PASS |
| 327 | Table 5, `s8_ei_temperature`, Log loss, high | 0.2486 | comparison.json metrics.evidence_effect.s8_ei_temperature.log_loss.high | 0.248619 | PASS |
| 328 | Table 5, `s8_ei_temperature`, Brier score, estimate | 0.1001 | comparison.json metrics.evidence_effect.s8_ei_temperature.brier.estimate | 0.100116 | PASS |
| 329 | Table 5, `s8_ei_temperature`, Brier score, low | 0.0778 | comparison.json metrics.evidence_effect.s8_ei_temperature.brier.low | 0.077771 | PASS |
| 330 | Table 5, `s8_ei_temperature`, Brier score, high | 0.1266 | comparison.json metrics.evidence_effect.s8_ei_temperature.brier.high | 0.126561 | PASS |
| 331 | Table 5, `s8_ei_temperature`, ECE, estimate | 0.0119 | comparison.json metrics.evidence_effect.s8_ei_temperature.ece.estimate | 0.011914 | PASS |
| 332 | Table 5, `s8_ei_temperature`, ECE, low | 0.0119 | comparison.json metrics.evidence_effect.s8_ei_temperature.ece.low | 0.011950 | PASS |
| 333 | Table 5, `s8_ei_temperature`, ECE, high | 0.0300 | comparison.json metrics.evidence_effect.s8_ei_temperature.ece.high | 0.029970 | PASS |
| 334 | Table 5, `s8_ei_temperature`, Accuracy, estimate | 0.9409 | comparison.json metrics.evidence_effect.s8_ei_temperature.accuracy.estimate | 0.940887 | PASS |
| 335 | Table 5, `s8_ei_temperature`, Accuracy, low | 0.9246 | comparison.json metrics.evidence_effect.s8_ei_temperature.accuracy.low | 0.924604 | PASS |
| 336 | Table 5, `s8_ei_temperature`, Accuracy, high | 0.9554 | comparison.json metrics.evidence_effect.s8_ei_temperature.accuracy.high | 0.955372 | PASS |
| 337 | Table 5, `s8_ei_temperature`, Macro-F1, estimate | 0.9399 | comparison.json metrics.evidence_effect.s8_ei_temperature.macro_f1.estimate | 0.939909 | PASS |
| 338 | Table 5, `s8_ei_temperature`, Macro-F1, low | 0.9225 | comparison.json metrics.evidence_effect.s8_ei_temperature.macro_f1.low | 0.922543 | PASS |
| 339 | Table 5, `s8_ei_temperature`, Macro-F1, high | 0.9544 | comparison.json metrics.evidence_effect.s8_ei_temperature.macro_f1.high | 0.954353 | PASS |
| 340 | Table 5, `s8_ei-raw`, Log loss, estimate | 0.2238 | comparison.json metrics.evidence_effect.s8_ei-raw.log_loss.estimate | 0.223751 | PASS |
| 341 | Table 5, `s8_ei-raw`, Log loss, low | 0.1651 | comparison.json metrics.evidence_effect.s8_ei-raw.log_loss.low | 0.165132 | PASS |
| 342 | Table 5, `s8_ei-raw`, Log loss, high | 0.2915 | comparison.json metrics.evidence_effect.s8_ei-raw.log_loss.high | 0.291510 | PASS |
| 343 | Table 5, `s8_ei-raw`, Brier score, estimate | 0.1024 | comparison.json metrics.evidence_effect.s8_ei-raw.brier.estimate | 0.102420 | PASS |
| 344 | Table 5, `s8_ei-raw`, Brier score, low | 0.0781 | comparison.json metrics.evidence_effect.s8_ei-raw.brier.low | 0.078061 | PASS |
| 345 | Table 5, `s8_ei-raw`, Brier score, high | 0.1313 | comparison.json metrics.evidence_effect.s8_ei-raw.brier.high | 0.131320 | PASS |
| 346 | Table 5, `s8_ei-raw`, ECE, estimate | 0.0325 | comparison.json metrics.evidence_effect.s8_ei-raw.ece.estimate | 0.032459 | PASS |
| 347 | Table 5, `s8_ei-raw`, ECE, low | 0.0244 | comparison.json metrics.evidence_effect.s8_ei-raw.ece.low | 0.024438 | PASS |
| 348 | Table 5, `s8_ei-raw`, ECE, high | 0.0504 | comparison.json metrics.evidence_effect.s8_ei-raw.ece.high | 0.050419 | PASS |
| 349 | Table 5, `s8_ei-raw`, Accuracy, estimate | 0.9409 | comparison.json metrics.evidence_effect.s8_ei-raw.accuracy.estimate | 0.940887 | PASS |
| 350 | Table 5, `s8_ei-raw`, Accuracy, low | 0.9246 | comparison.json metrics.evidence_effect.s8_ei-raw.accuracy.low | 0.924604 | PASS |
| 351 | Table 5, `s8_ei-raw`, Accuracy, high | 0.9554 | comparison.json metrics.evidence_effect.s8_ei-raw.accuracy.high | 0.955372 | PASS |
| 352 | Table 5, `s8_ei-raw`, Macro-F1, estimate | 0.9399 | comparison.json metrics.evidence_effect.s8_ei-raw.macro_f1.estimate | 0.939909 | PASS |
| 353 | Table 5, `s8_ei-raw`, Macro-F1, low | 0.9225 | comparison.json metrics.evidence_effect.s8_ei-raw.macro_f1.low | 0.922543 | PASS |
| 354 | Table 5, `s8_ei-raw`, Macro-F1, high | 0.9544 | comparison.json metrics.evidence_effect.s8_ei-raw.macro_f1.high | 0.954353 | PASS |
| 355 | Table 5, `s8_ei_1`, Log loss, estimate | 0.2085 | comparison.json metrics.evidence_effect.s8_ei_1.log_loss.estimate | 0.208516 | PASS |
| 356 | Table 5, `s8_ei_1`, Log loss, low | 0.1646 | comparison.json metrics.evidence_effect.s8_ei_1.log_loss.low | 0.164566 | PASS |
| 357 | Table 5, `s8_ei_1`, Log loss, high | 0.2575 | comparison.json metrics.evidence_effect.s8_ei_1.log_loss.high | 0.257498 | PASS |
| 358 | Table 5, `s8_ei_1`, Brier score, estimate | 0.1050 | comparison.json metrics.evidence_effect.s8_ei_1.brier.estimate | 0.104955 | PASS |
| 359 | Table 5, `s8_ei_1`, Brier score, low | 0.0819 | comparison.json metrics.evidence_effect.s8_ei_1.brier.low | 0.081868 | PASS |
| 360 | Table 5, `s8_ei_1`, Brier score, high | 0.1315 | comparison.json metrics.evidence_effect.s8_ei_1.brier.high | 0.131482 | PASS |
| 361 | Table 5, `s8_ei_1`, ECE, estimate | 0.0138 | comparison.json metrics.evidence_effect.s8_ei_1.ece.estimate | 0.013812 | PASS |
| 362 | Table 5, `s8_ei_1`, ECE, low | 0.0114 | comparison.json metrics.evidence_effect.s8_ei_1.ece.low | 0.011442 | PASS |
| 363 | Table 5, `s8_ei_1`, ECE, high | 0.0321 | comparison.json metrics.evidence_effect.s8_ei_1.ece.high | 0.032149 | PASS |
| 364 | Table 5, `s8_ei_1`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_1.accuracy.estimate | 0.938424 | PASS |
| 365 | Table 5, `s8_ei_1`, Accuracy, low | 0.9212 | comparison.json metrics.evidence_effect.s8_ei_1.accuracy.low | 0.921164 | PASS |
| 366 | Table 5, `s8_ei_1`, Accuracy, high | 0.9540 | comparison.json metrics.evidence_effect.s8_ei_1.accuracy.high | 0.954025 | PASS |
| 367 | Table 5, `s8_ei_1`, Macro-F1, estimate | 0.9372 | comparison.json metrics.evidence_effect.s8_ei_1.macro_f1.estimate | 0.937249 | PASS |
| 368 | Table 5, `s8_ei_1`, Macro-F1, low | 0.9198 | comparison.json metrics.evidence_effect.s8_ei_1.macro_f1.low | 0.919836 | PASS |
| 369 | Table 5, `s8_ei_1`, Macro-F1, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_1.macro_f1.high | 0.953187 | PASS |
| 370 | Table 5, `s8_ei_1_temperature`, Log loss, estimate | 0.2085 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.log_loss.estimate | 0.208516 | PASS |
| 371 | Table 5, `s8_ei_1_temperature`, Log loss, low | 0.1646 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.log_loss.low | 0.164566 | PASS |
| 372 | Table 5, `s8_ei_1_temperature`, Log loss, high | 0.2575 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.log_loss.high | 0.257498 | PASS |
| 373 | Table 5, `s8_ei_1_temperature`, Brier score, estimate | 0.1050 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.brier.estimate | 0.104955 | PASS |
| 374 | Table 5, `s8_ei_1_temperature`, Brier score, low | 0.0819 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.brier.low | 0.081868 | PASS |
| 375 | Table 5, `s8_ei_1_temperature`, Brier score, high | 0.1315 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.brier.high | 0.131482 | PASS |
| 376 | Table 5, `s8_ei_1_temperature`, ECE, estimate | 0.0138 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.ece.estimate | 0.013812 | PASS |
| 377 | Table 5, `s8_ei_1_temperature`, ECE, low | 0.0114 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.ece.low | 0.011442 | PASS |
| 378 | Table 5, `s8_ei_1_temperature`, ECE, high | 0.0321 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.ece.high | 0.032149 | PASS |
| 379 | Table 5, `s8_ei_1_temperature`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.accuracy.estimate | 0.938424 | PASS |
| 380 | Table 5, `s8_ei_1_temperature`, Accuracy, low | 0.9212 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.accuracy.low | 0.921164 | PASS |
| 381 | Table 5, `s8_ei_1_temperature`, Accuracy, high | 0.9540 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.accuracy.high | 0.954025 | PASS |
| 382 | Table 5, `s8_ei_1_temperature`, Macro-F1, estimate | 0.9372 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.macro_f1.estimate | 0.937249 | PASS |
| 383 | Table 5, `s8_ei_1_temperature`, Macro-F1, low | 0.9198 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.macro_f1.low | 0.919836 | PASS |
| 384 | Table 5, `s8_ei_1_temperature`, Macro-F1, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.macro_f1.high | 0.953187 | PASS |
| 385 | Table 5, `s8_ei_1-raw`, Log loss, estimate | 0.2438 | comparison.json metrics.evidence_effect.s8_ei_1-raw.log_loss.estimate | 0.243769 | PASS |
| 386 | Table 5, `s8_ei_1-raw`, Log loss, low | 0.1812 | comparison.json metrics.evidence_effect.s8_ei_1-raw.log_loss.low | 0.181201 | PASS |
| 387 | Table 5, `s8_ei_1-raw`, Log loss, high | 0.3190 | comparison.json metrics.evidence_effect.s8_ei_1-raw.log_loss.high | 0.319048 | PASS |
| 388 | Table 5, `s8_ei_1-raw`, Brier score, estimate | 0.1082 | comparison.json metrics.evidence_effect.s8_ei_1-raw.brier.estimate | 0.108196 | PASS |
| 389 | Table 5, `s8_ei_1-raw`, Brier score, low | 0.0834 | comparison.json metrics.evidence_effect.s8_ei_1-raw.brier.low | 0.083396 | PASS |
| 390 | Table 5, `s8_ei_1-raw`, Brier score, high | 0.1372 | comparison.json metrics.evidence_effect.s8_ei_1-raw.brier.high | 0.137172 | PASS |
| 391 | Table 5, `s8_ei_1-raw`, ECE, estimate | 0.0369 | comparison.json metrics.evidence_effect.s8_ei_1-raw.ece.estimate | 0.036944 | PASS |
| 392 | Table 5, `s8_ei_1-raw`, ECE, low | 0.0276 | comparison.json metrics.evidence_effect.s8_ei_1-raw.ece.low | 0.027578 | PASS |
| 393 | Table 5, `s8_ei_1-raw`, ECE, high | 0.0552 | comparison.json metrics.evidence_effect.s8_ei_1-raw.ece.high | 0.055221 | PASS |
| 394 | Table 5, `s8_ei_1-raw`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_1-raw.accuracy.estimate | 0.938424 | PASS |
| 395 | Table 5, `s8_ei_1-raw`, Accuracy, low | 0.9212 | comparison.json metrics.evidence_effect.s8_ei_1-raw.accuracy.low | 0.921164 | PASS |
| 396 | Table 5, `s8_ei_1-raw`, Accuracy, high | 0.9540 | comparison.json metrics.evidence_effect.s8_ei_1-raw.accuracy.high | 0.954025 | PASS |
| 397 | Table 5, `s8_ei_1-raw`, Macro-F1, estimate | 0.9372 | comparison.json metrics.evidence_effect.s8_ei_1-raw.macro_f1.estimate | 0.937249 | PASS |
| 398 | Table 5, `s8_ei_1-raw`, Macro-F1, low | 0.9198 | comparison.json metrics.evidence_effect.s8_ei_1-raw.macro_f1.low | 0.919836 | PASS |
| 399 | Table 5, `s8_ei_1-raw`, Macro-F1, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_1-raw.macro_f1.high | 0.953187 | PASS |
| 400 | Table 5, `s8_ei_2`, Log loss, estimate | 0.2098 | comparison.json metrics.evidence_effect.s8_ei_2.log_loss.estimate | 0.209829 | PASS |
| 401 | Table 5, `s8_ei_2`, Log loss, low | 0.1670 | comparison.json metrics.evidence_effect.s8_ei_2.log_loss.low | 0.166996 | PASS |
| 402 | Table 5, `s8_ei_2`, Log loss, high | 0.2605 | comparison.json metrics.evidence_effect.s8_ei_2.log_loss.high | 0.260539 | PASS |
| 403 | Table 5, `s8_ei_2`, Brier score, estimate | 0.1054 | comparison.json metrics.evidence_effect.s8_ei_2.brier.estimate | 0.105418 | PASS |
| 404 | Table 5, `s8_ei_2`, Brier score, low | 0.0820 | comparison.json metrics.evidence_effect.s8_ei_2.brier.low | 0.081952 | PASS |
| 405 | Table 5, `s8_ei_2`, Brier score, high | 0.1327 | comparison.json metrics.evidence_effect.s8_ei_2.brier.high | 0.132702 | PASS |
| 406 | Table 5, `s8_ei_2`, ECE, estimate | 0.0188 | comparison.json metrics.evidence_effect.s8_ei_2.ece.estimate | 0.018759 | PASS |
| 407 | Table 5, `s8_ei_2`, ECE, low | 0.0140 | comparison.json metrics.evidence_effect.s8_ei_2.ece.low | 0.014037 | PASS |
| 408 | Table 5, `s8_ei_2`, ECE, high | 0.0331 | comparison.json metrics.evidence_effect.s8_ei_2.ece.high | 0.033098 | PASS |
| 409 | Table 5, `s8_ei_2`, Accuracy, estimate | 0.9368 | comparison.json metrics.evidence_effect.s8_ei_2.accuracy.estimate | 0.936782 | PASS |
| 410 | Table 5, `s8_ei_2`, Accuracy, low | 0.9193 | comparison.json metrics.evidence_effect.s8_ei_2.accuracy.low | 0.919279 | PASS |
| 411 | Table 5, `s8_ei_2`, Accuracy, high | 0.9524 | comparison.json metrics.evidence_effect.s8_ei_2.accuracy.high | 0.952385 | PASS |
| 412 | Table 5, `s8_ei_2`, Macro-F1, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei_2.macro_f1.estimate | 0.935847 | PASS |
| 413 | Table 5, `s8_ei_2`, Macro-F1, low | 0.9179 | comparison.json metrics.evidence_effect.s8_ei_2.macro_f1.low | 0.917900 | PASS |
| 414 | Table 5, `s8_ei_2`, Macro-F1, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2.macro_f1.high | 0.951555 | PASS |
| 415 | Table 5, `s8_ei_2_temperature`, Log loss, estimate | 0.2098 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.log_loss.estimate | 0.209829 | PASS |
| 416 | Table 5, `s8_ei_2_temperature`, Log loss, low | 0.1670 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.log_loss.low | 0.166996 | PASS |
| 417 | Table 5, `s8_ei_2_temperature`, Log loss, high | 0.2605 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.log_loss.high | 0.260539 | PASS |
| 418 | Table 5, `s8_ei_2_temperature`, Brier score, estimate | 0.1054 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.brier.estimate | 0.105418 | PASS |
| 419 | Table 5, `s8_ei_2_temperature`, Brier score, low | 0.0820 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.brier.low | 0.081952 | PASS |
| 420 | Table 5, `s8_ei_2_temperature`, Brier score, high | 0.1327 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.brier.high | 0.132702 | PASS |
| 421 | Table 5, `s8_ei_2_temperature`, ECE, estimate | 0.0188 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.ece.estimate | 0.018759 | PASS |
| 422 | Table 5, `s8_ei_2_temperature`, ECE, low | 0.0140 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.ece.low | 0.014037 | PASS |
| 423 | Table 5, `s8_ei_2_temperature`, ECE, high | 0.0331 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.ece.high | 0.033098 | PASS |
| 424 | Table 5, `s8_ei_2_temperature`, Accuracy, estimate | 0.9368 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.accuracy.estimate | 0.936782 | PASS |
| 425 | Table 5, `s8_ei_2_temperature`, Accuracy, low | 0.9193 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.accuracy.low | 0.919279 | PASS |
| 426 | Table 5, `s8_ei_2_temperature`, Accuracy, high | 0.9524 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.accuracy.high | 0.952385 | PASS |
| 427 | Table 5, `s8_ei_2_temperature`, Macro-F1, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.macro_f1.estimate | 0.935847 | PASS |
| 428 | Table 5, `s8_ei_2_temperature`, Macro-F1, low | 0.9179 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.macro_f1.low | 0.917900 | PASS |
| 429 | Table 5, `s8_ei_2_temperature`, Macro-F1, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.macro_f1.high | 0.951555 | PASS |
| 430 | Table 5, `s8_ei_2-raw`, Log loss, estimate | 0.2468 | comparison.json metrics.evidence_effect.s8_ei_2-raw.log_loss.estimate | 0.246801 | PASS |
| 431 | Table 5, `s8_ei_2-raw`, Log loss, low | 0.1831 | comparison.json metrics.evidence_effect.s8_ei_2-raw.log_loss.low | 0.183073 | PASS |
| 432 | Table 5, `s8_ei_2-raw`, Log loss, high | 0.3247 | comparison.json metrics.evidence_effect.s8_ei_2-raw.log_loss.high | 0.324720 | PASS |
| 433 | Table 5, `s8_ei_2-raw`, Brier score, estimate | 0.1098 | comparison.json metrics.evidence_effect.s8_ei_2-raw.brier.estimate | 0.109752 | PASS |
| 434 | Table 5, `s8_ei_2-raw`, Brier score, low | 0.0834 | comparison.json metrics.evidence_effect.s8_ei_2-raw.brier.low | 0.083436 | PASS |
| 435 | Table 5, `s8_ei_2-raw`, Brier score, high | 0.1391 | comparison.json metrics.evidence_effect.s8_ei_2-raw.brier.high | 0.139058 | PASS |
| 436 | Table 5, `s8_ei_2-raw`, ECE, estimate | 0.0406 | comparison.json metrics.evidence_effect.s8_ei_2-raw.ece.estimate | 0.040644 | PASS |
| 437 | Table 5, `s8_ei_2-raw`, ECE, low | 0.0295 | comparison.json metrics.evidence_effect.s8_ei_2-raw.ece.low | 0.029538 | PASS |
| 438 | Table 5, `s8_ei_2-raw`, ECE, high | 0.0578 | comparison.json metrics.evidence_effect.s8_ei_2-raw.ece.high | 0.057779 | PASS |
| 439 | Table 5, `s8_ei_2-raw`, Accuracy, estimate | 0.9368 | comparison.json metrics.evidence_effect.s8_ei_2-raw.accuracy.estimate | 0.936782 | PASS |
| 440 | Table 5, `s8_ei_2-raw`, Accuracy, low | 0.9193 | comparison.json metrics.evidence_effect.s8_ei_2-raw.accuracy.low | 0.919279 | PASS |
| 441 | Table 5, `s8_ei_2-raw`, Accuracy, high | 0.9524 | comparison.json metrics.evidence_effect.s8_ei_2-raw.accuracy.high | 0.952385 | PASS |
| 442 | Table 5, `s8_ei_2-raw`, Macro-F1, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei_2-raw.macro_f1.estimate | 0.935847 | PASS |
| 443 | Table 5, `s8_ei_2-raw`, Macro-F1, low | 0.9179 | comparison.json metrics.evidence_effect.s8_ei_2-raw.macro_f1.low | 0.917900 | PASS |
| 444 | Table 5, `s8_ei_2-raw`, Macro-F1, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2-raw.macro_f1.high | 0.951555 | PASS |
| 445 | Table 5, `s8_ei_3`, Log loss, estimate | 0.1998 | comparison.json metrics.evidence_effect.s8_ei_3.log_loss.estimate | 0.199843 | PASS |
| 446 | Table 5, `s8_ei_3`, Log loss, low | 0.1577 | comparison.json metrics.evidence_effect.s8_ei_3.log_loss.low | 0.157687 | PASS |
| 447 | Table 5, `s8_ei_3`, Log loss, high | 0.2489 | comparison.json metrics.evidence_effect.s8_ei_3.log_loss.high | 0.248904 | PASS |
| 448 | Table 5, `s8_ei_3`, Brier score, estimate | 0.0991 | comparison.json metrics.evidence_effect.s8_ei_3.brier.estimate | 0.099113 | PASS |
| 449 | Table 5, `s8_ei_3`, Brier score, low | 0.0768 | comparison.json metrics.evidence_effect.s8_ei_3.brier.low | 0.076843 | PASS |
| 450 | Table 5, `s8_ei_3`, Brier score, high | 0.1253 | comparison.json metrics.evidence_effect.s8_ei_3.brier.high | 0.125321 | PASS |
| 451 | Table 5, `s8_ei_3`, ECE, estimate | 0.0137 | comparison.json metrics.evidence_effect.s8_ei_3.ece.estimate | 0.013697 | PASS |
| 452 | Table 5, `s8_ei_3`, ECE, low | 0.0112 | comparison.json metrics.evidence_effect.s8_ei_3.ece.low | 0.011247 | PASS |
| 453 | Table 5, `s8_ei_3`, ECE, high | 0.0295 | comparison.json metrics.evidence_effect.s8_ei_3.ece.high | 0.029494 | PASS |
| 454 | Table 5, `s8_ei_3`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_3.accuracy.estimate | 0.938424 | PASS |
| 455 | Table 5, `s8_ei_3`, Accuracy, low | 0.9210 | comparison.json metrics.evidence_effect.s8_ei_3.accuracy.low | 0.920990 | PASS |
| 456 | Table 5, `s8_ei_3`, Accuracy, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_3.accuracy.high | 0.953165 | PASS |
| 457 | Table 5, `s8_ei_3`, Macro-F1, estimate | 0.9379 | comparison.json metrics.evidence_effect.s8_ei_3.macro_f1.estimate | 0.937933 | PASS |
| 458 | Table 5, `s8_ei_3`, Macro-F1, low | 0.9203 | comparison.json metrics.evidence_effect.s8_ei_3.macro_f1.low | 0.920293 | PASS |
| 459 | Table 5, `s8_ei_3`, Macro-F1, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei_3.macro_f1.high | 0.952877 | PASS |
| 460 | Table 5, `s8_ei_3_temperature`, Log loss, estimate | 0.1998 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.log_loss.estimate | 0.199843 | PASS |
| 461 | Table 5, `s8_ei_3_temperature`, Log loss, low | 0.1577 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.log_loss.low | 0.157687 | PASS |
| 462 | Table 5, `s8_ei_3_temperature`, Log loss, high | 0.2489 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.log_loss.high | 0.248904 | PASS |
| 463 | Table 5, `s8_ei_3_temperature`, Brier score, estimate | 0.0991 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.brier.estimate | 0.099113 | PASS |
| 464 | Table 5, `s8_ei_3_temperature`, Brier score, low | 0.0768 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.brier.low | 0.076843 | PASS |
| 465 | Table 5, `s8_ei_3_temperature`, Brier score, high | 0.1253 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.brier.high | 0.125321 | PASS |
| 466 | Table 5, `s8_ei_3_temperature`, ECE, estimate | 0.0137 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.ece.estimate | 0.013697 | PASS |
| 467 | Table 5, `s8_ei_3_temperature`, ECE, low | 0.0112 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.ece.low | 0.011247 | PASS |
| 468 | Table 5, `s8_ei_3_temperature`, ECE, high | 0.0295 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.ece.high | 0.029494 | PASS |
| 469 | Table 5, `s8_ei_3_temperature`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.accuracy.estimate | 0.938424 | PASS |
| 470 | Table 5, `s8_ei_3_temperature`, Accuracy, low | 0.9210 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.accuracy.low | 0.920990 | PASS |
| 471 | Table 5, `s8_ei_3_temperature`, Accuracy, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.accuracy.high | 0.953165 | PASS |
| 472 | Table 5, `s8_ei_3_temperature`, Macro-F1, estimate | 0.9379 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.macro_f1.estimate | 0.937933 | PASS |
| 473 | Table 5, `s8_ei_3_temperature`, Macro-F1, low | 0.9203 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.macro_f1.low | 0.920293 | PASS |
| 474 | Table 5, `s8_ei_3_temperature`, Macro-F1, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.macro_f1.high | 0.952877 | PASS |
| 475 | Table 5, `s8_ei_3-raw`, Log loss, estimate | 0.2232 | comparison.json metrics.evidence_effect.s8_ei_3-raw.log_loss.estimate | 0.223215 | PASS |
| 476 | Table 5, `s8_ei_3-raw`, Log loss, low | 0.1647 | comparison.json metrics.evidence_effect.s8_ei_3-raw.log_loss.low | 0.164659 | PASS |
| 477 | Table 5, `s8_ei_3-raw`, Log loss, high | 0.2894 | comparison.json metrics.evidence_effect.s8_ei_3-raw.log_loss.high | 0.289407 | PASS |
| 478 | Table 5, `s8_ei_3-raw`, Brier score, estimate | 0.1021 | comparison.json metrics.evidence_effect.s8_ei_3-raw.brier.estimate | 0.102054 | PASS |
| 479 | Table 5, `s8_ei_3-raw`, Brier score, low | 0.0778 | comparison.json metrics.evidence_effect.s8_ei_3-raw.brier.low | 0.077846 | PASS |
| 480 | Table 5, `s8_ei_3-raw`, Brier score, high | 0.1304 | comparison.json metrics.evidence_effect.s8_ei_3-raw.brier.high | 0.130427 | PASS |
| 481 | Table 5, `s8_ei_3-raw`, ECE, estimate | 0.0365 | comparison.json metrics.evidence_effect.s8_ei_3-raw.ece.estimate | 0.036507 | PASS |
| 482 | Table 5, `s8_ei_3-raw`, ECE, low | 0.0256 | comparison.json metrics.evidence_effect.s8_ei_3-raw.ece.low | 0.025634 | PASS |
| 483 | Table 5, `s8_ei_3-raw`, ECE, high | 0.0523 | comparison.json metrics.evidence_effect.s8_ei_3-raw.ece.high | 0.052312 | PASS |
| 484 | Table 5, `s8_ei_3-raw`, Accuracy, estimate | 0.9384 | comparison.json metrics.evidence_effect.s8_ei_3-raw.accuracy.estimate | 0.938424 | PASS |
| 485 | Table 5, `s8_ei_3-raw`, Accuracy, low | 0.9210 | comparison.json metrics.evidence_effect.s8_ei_3-raw.accuracy.low | 0.920990 | PASS |
| 486 | Table 5, `s8_ei_3-raw`, Accuracy, high | 0.9532 | comparison.json metrics.evidence_effect.s8_ei_3-raw.accuracy.high | 0.953165 | PASS |
| 487 | Table 5, `s8_ei_3-raw`, Macro-F1, estimate | 0.9379 | comparison.json metrics.evidence_effect.s8_ei_3-raw.macro_f1.estimate | 0.937933 | PASS |
| 488 | Table 5, `s8_ei_3-raw`, Macro-F1, low | 0.9203 | comparison.json metrics.evidence_effect.s8_ei_3-raw.macro_f1.low | 0.920293 | PASS |
| 489 | Table 5, `s8_ei_3-raw`, Macro-F1, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei_3-raw.macro_f1.high | 0.952877 | PASS |
| 490 | Table 5, `gemma4_31b`, Log loss, estimate | 0.4078 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.estimate | 0.407805 | PASS |
| 491 | Table 5, `gemma4_31b`, Log loss, low | 0.3373 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.low | 0.337272 | PASS |
| 492 | Table 5, `gemma4_31b`, Log loss, high | 0.4804 | comparison.json metrics.evidence_effect.gemma4_31b.log_loss.high | 0.480403 | PASS |
| 493 | Table 5, `gemma4_31b`, Brier score, estimate | 0.2040 | comparison.json metrics.evidence_effect.gemma4_31b.brier.estimate | 0.204016 | PASS |
| 494 | Table 5, `gemma4_31b`, Brier score, low | 0.1671 | comparison.json metrics.evidence_effect.gemma4_31b.brier.low | 0.167069 | PASS |
| 495 | Table 5, `gemma4_31b`, Brier score, high | 0.2418 | comparison.json metrics.evidence_effect.gemma4_31b.brier.high | 0.241809 | PASS |
| 496 | Table 5, `gemma4_31b`, ECE, estimate | 0.0283 | comparison.json metrics.evidence_effect.gemma4_31b.ece.estimate | 0.028293 | PASS |
| 497 | Table 5, `gemma4_31b`, ECE, low | 0.0221 | comparison.json metrics.evidence_effect.gemma4_31b.ece.low | 0.022099 | PASS |
| 498 | Table 5, `gemma4_31b`, ECE, high | 0.0560 | comparison.json metrics.evidence_effect.gemma4_31b.ece.high | 0.056017 | PASS |
| 499 | Table 5, `gemma4_31b`, Accuracy, estimate | 0.8711 | comparison.json metrics.evidence_effect.gemma4_31b.accuracy.estimate | 0.871100 | PASS |
| 500 | Table 5, `gemma4_31b`, Accuracy, low | 0.8450 | comparison.json metrics.evidence_effect.gemma4_31b.accuracy.low | 0.845020 | PASS |
| 501 | Table 5, `gemma4_31b`, Accuracy, high | 0.8949 | comparison.json metrics.evidence_effect.gemma4_31b.accuracy.high | 0.894911 | PASS |
| 502 | Table 5, `gemma4_31b`, Macro-F1, estimate | 0.8626 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.estimate | 0.862587 | PASS |
| 503 | Table 5, `gemma4_31b`, Macro-F1, low | 0.8333 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.low | 0.833263 | PASS |
| 504 | Table 5, `gemma4_31b`, Macro-F1, high | 0.8879 | comparison.json metrics.evidence_effect.gemma4_31b.macro_f1.high | 0.887909 | PASS |
| 505 | Table 5, `gemma4_31b_temperature`, Log loss, estimate | 0.4424 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.log_loss.estimate | 0.442366 | PASS |
| 506 | Table 5, `gemma4_31b_temperature`, Log loss, low | 0.3761 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.log_loss.low | 0.376106 | PASS |
| 507 | Table 5, `gemma4_31b_temperature`, Log loss, high | 0.5095 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.log_loss.high | 0.509456 | PASS |
| 508 | Table 5, `gemma4_31b_temperature`, Brier score, estimate | 0.2205 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.brier.estimate | 0.220458 | PASS |
| 509 | Table 5, `gemma4_31b_temperature`, Brier score, low | 0.1823 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.brier.low | 0.182277 | PASS |
| 510 | Table 5, `gemma4_31b_temperature`, Brier score, high | 0.2622 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.brier.high | 0.262168 | PASS |
| 511 | Table 5, `gemma4_31b_temperature`, ECE, estimate | 0.0321 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.ece.estimate | 0.032144 | PASS |
| 512 | Table 5, `gemma4_31b_temperature`, ECE, low | 0.0248 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.ece.low | 0.024849 | PASS |
| 513 | Table 5, `gemma4_31b_temperature`, ECE, high | 0.0561 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.ece.high | 0.056136 | PASS |
| 514 | Table 5, `gemma4_31b_temperature`, Accuracy, estimate | 0.8695 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.accuracy.estimate | 0.869458 | PASS |
| 515 | Table 5, `gemma4_31b_temperature`, Accuracy, low | 0.8430 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.accuracy.low | 0.843034 | PASS |
| 516 | Table 5, `gemma4_31b_temperature`, Accuracy, high | 0.8940 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.accuracy.high | 0.893973 | PASS |
| 517 | Table 5, `gemma4_31b_temperature`, Macro-F1, estimate | 0.8608 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.macro_f1.estimate | 0.860839 | PASS |
| 518 | Table 5, `gemma4_31b_temperature`, Macro-F1, low | 0.8315 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.macro_f1.low | 0.831526 | PASS |
| 519 | Table 5, `gemma4_31b_temperature`, Macro-F1, high | 0.8874 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.macro_f1.high | 0.887385 | PASS |
| 520 | Table 5, `gemma4_31b-raw`, Log loss, estimate | 2.2911 | comparison.json metrics.evidence_effect.gemma4_31b-raw.log_loss.estimate | 2.291077 | PASS |
| 521 | Table 5, `gemma4_31b-raw`, Log loss, low | 1.8275 | comparison.json metrics.evidence_effect.gemma4_31b-raw.log_loss.low | 1.827532 | PASS |
| 522 | Table 5, `gemma4_31b-raw`, Log loss, high | 2.7759 | comparison.json metrics.evidence_effect.gemma4_31b-raw.log_loss.high | 2.775923 | PASS |
| 523 | Table 5, `gemma4_31b-raw`, Brier score, estimate | 0.2615 | comparison.json metrics.evidence_effect.gemma4_31b-raw.brier.estimate | 0.261526 | PASS |
| 524 | Table 5, `gemma4_31b-raw`, Brier score, low | 0.2124 | comparison.json metrics.evidence_effect.gemma4_31b-raw.brier.low | 0.212387 | PASS |
| 525 | Table 5, `gemma4_31b-raw`, Brier score, high | 0.3148 | comparison.json metrics.evidence_effect.gemma4_31b-raw.brier.high | 0.314808 | PASS |
| 526 | Table 5, `gemma4_31b-raw`, ECE, estimate | 0.1317 | comparison.json metrics.evidence_effect.gemma4_31b-raw.ece.estimate | 0.131669 | PASS |
| 527 | Table 5, `gemma4_31b-raw`, ECE, low | 0.1069 | comparison.json metrics.evidence_effect.gemma4_31b-raw.ece.low | 0.106897 | PASS |
| 528 | Table 5, `gemma4_31b-raw`, ECE, high | 0.1587 | comparison.json metrics.evidence_effect.gemma4_31b-raw.ece.high | 0.158728 | PASS |
| 529 | Table 5, `gemma4_31b-raw`, Accuracy, estimate | 0.8695 | comparison.json metrics.evidence_effect.gemma4_31b-raw.accuracy.estimate | 0.869458 | PASS |
| 530 | Table 5, `gemma4_31b-raw`, Accuracy, low | 0.8430 | comparison.json metrics.evidence_effect.gemma4_31b-raw.accuracy.low | 0.843034 | PASS |
| 531 | Table 5, `gemma4_31b-raw`, Accuracy, high | 0.8940 | comparison.json metrics.evidence_effect.gemma4_31b-raw.accuracy.high | 0.893973 | PASS |
| 532 | Table 5, `gemma4_31b-raw`, Macro-F1, estimate | 0.8608 | comparison.json metrics.evidence_effect.gemma4_31b-raw.macro_f1.estimate | 0.860839 | PASS |
| 533 | Table 5, `gemma4_31b-raw`, Macro-F1, low | 0.8315 | comparison.json metrics.evidence_effect.gemma4_31b-raw.macro_f1.low | 0.831526 | PASS |
| 534 | Table 5, `gemma4_31b-raw`, Macro-F1, high | 0.8874 | comparison.json metrics.evidence_effect.gemma4_31b-raw.macro_f1.high | 0.887385 | PASS |
| 535 | Table 6, `prior`, F1, significantly increased, estimate | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_increased.estimate | 0.000000 | PASS |
| 536 | Table 6, `prior`, F1, significantly increased, low | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_increased.low | 0.000000 | PASS |
| 537 | Table 6, `prior`, F1, significantly increased, high | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_increased.high | 0.000000 | PASS |
| 538 | Table 6, `prior`, F1, significantly decreased, estimate | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_decreased.estimate | 0.000000 | PASS |
| 539 | Table 6, `prior`, F1, significantly decreased, low | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_decreased.low | 0.000000 | PASS |
| 540 | Table 6, `prior`, F1, significantly decreased, high | 0.0000 | comparison.json metrics.evidence_effect.prior.f1_decreased.high | 0.000000 | PASS |
| 541 | Table 6, `prior`, F1, no significant difference, estimate | 0.5927 | comparison.json metrics.evidence_effect.prior.f1_no_difference.estimate | 0.592721 | PASS |
| 542 | Table 6, `prior`, F1, no significant difference, low | 0.5564 | comparison.json metrics.evidence_effect.prior.f1_no_difference.low | 0.556441 | PASS |
| 543 | Table 6, `prior`, F1, no significant difference, high | 0.6286 | comparison.json metrics.evidence_effect.prior.f1_no_difference.high | 0.628606 | PASS |
| 544 | Table 6, `s8_ei`, F1, significantly increased, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei.f1_increased.estimate | 0.935802 | PASS |
| 545 | Table 6, `s8_ei`, F1, significantly increased, low | 0.9152 | comparison.json metrics.evidence_effect.s8_ei.f1_increased.low | 0.915164 | PASS |
| 546 | Table 6, `s8_ei`, F1, significantly increased, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei.f1_increased.high | 0.952917 | PASS |
| 547 | Table 6, `s8_ei`, F1, significantly decreased, estimate | 0.9365 | comparison.json metrics.evidence_effect.s8_ei.f1_decreased.estimate | 0.936455 | PASS |
| 548 | Table 6, `s8_ei`, F1, significantly decreased, low | 0.9113 | comparison.json metrics.evidence_effect.s8_ei.f1_decreased.low | 0.911256 | PASS |
| 549 | Table 6, `s8_ei`, F1, significantly decreased, high | 0.9577 | comparison.json metrics.evidence_effect.s8_ei.f1_decreased.high | 0.957656 | PASS |
| 550 | Table 6, `s8_ei`, F1, no significant difference, estimate | 0.9475 | comparison.json metrics.evidence_effect.s8_ei.f1_no_difference.estimate | 0.947471 | PASS |
| 551 | Table 6, `s8_ei`, F1, no significant difference, low | 0.9278 | comparison.json metrics.evidence_effect.s8_ei.f1_no_difference.low | 0.927764 | PASS |
| 552 | Table 6, `s8_ei`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei.f1_no_difference.high | 0.963037 | PASS |
| 553 | Table 6, `s8_ei_temperature`, F1, significantly increased, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_increased.estimate | 0.935802 | PASS |
| 554 | Table 6, `s8_ei_temperature`, F1, significantly increased, low | 0.9152 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_increased.low | 0.915164 | PASS |
| 555 | Table 6, `s8_ei_temperature`, F1, significantly increased, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_increased.high | 0.952917 | PASS |
| 556 | Table 6, `s8_ei_temperature`, F1, significantly decreased, estimate | 0.9365 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_decreased.estimate | 0.936455 | PASS |
| 557 | Table 6, `s8_ei_temperature`, F1, significantly decreased, low | 0.9113 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_decreased.low | 0.911256 | PASS |
| 558 | Table 6, `s8_ei_temperature`, F1, significantly decreased, high | 0.9577 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_decreased.high | 0.957656 | PASS |
| 559 | Table 6, `s8_ei_temperature`, F1, no significant difference, estimate | 0.9475 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_no_difference.estimate | 0.947471 | PASS |
| 560 | Table 6, `s8_ei_temperature`, F1, no significant difference, low | 0.9278 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_no_difference.low | 0.927764 | PASS |
| 561 | Table 6, `s8_ei_temperature`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei_temperature.f1_no_difference.high | 0.963037 | PASS |
| 562 | Table 6, `s8_ei-raw`, F1, significantly increased, estimate | 0.9358 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_increased.estimate | 0.935802 | PASS |
| 563 | Table 6, `s8_ei-raw`, F1, significantly increased, low | 0.9152 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_increased.low | 0.915164 | PASS |
| 564 | Table 6, `s8_ei-raw`, F1, significantly increased, high | 0.9529 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_increased.high | 0.952917 | PASS |
| 565 | Table 6, `s8_ei-raw`, F1, significantly decreased, estimate | 0.9365 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_decreased.estimate | 0.936455 | PASS |
| 566 | Table 6, `s8_ei-raw`, F1, significantly decreased, low | 0.9113 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_decreased.low | 0.911256 | PASS |
| 567 | Table 6, `s8_ei-raw`, F1, significantly decreased, high | 0.9577 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_decreased.high | 0.957656 | PASS |
| 568 | Table 6, `s8_ei-raw`, F1, no significant difference, estimate | 0.9475 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_no_difference.estimate | 0.947471 | PASS |
| 569 | Table 6, `s8_ei-raw`, F1, no significant difference, low | 0.9278 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_no_difference.low | 0.927764 | PASS |
| 570 | Table 6, `s8_ei-raw`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei-raw.f1_no_difference.high | 0.963037 | PASS |
| 571 | Table 6, `s8_ei_1`, F1, significantly increased, estimate | 0.9312 | comparison.json metrics.evidence_effect.s8_ei_1.f1_increased.estimate | 0.931204 | PASS |
| 572 | Table 6, `s8_ei_1`, F1, significantly increased, low | 0.9103 | comparison.json metrics.evidence_effect.s8_ei_1.f1_increased.low | 0.910344 | PASS |
| 573 | Table 6, `s8_ei_1`, F1, significantly increased, high | 0.9490 | comparison.json metrics.evidence_effect.s8_ei_1.f1_increased.high | 0.948959 | PASS |
| 574 | Table 6, `s8_ei_1`, F1, significantly decreased, estimate | 0.9338 | comparison.json metrics.evidence_effect.s8_ei_1.f1_decreased.estimate | 0.933786 | PASS |
| 575 | Table 6, `s8_ei_1`, F1, significantly decreased, low | 0.9097 | comparison.json metrics.evidence_effect.s8_ei_1.f1_decreased.low | 0.909672 | PASS |
| 576 | Table 6, `s8_ei_1`, F1, significantly decreased, high | 0.9561 | comparison.json metrics.evidence_effect.s8_ei_1.f1_decreased.high | 0.956081 | PASS |
| 577 | Table 6, `s8_ei_1`, F1, no significant difference, estimate | 0.9468 | comparison.json metrics.evidence_effect.s8_ei_1.f1_no_difference.estimate | 0.946757 | PASS |
| 578 | Table 6, `s8_ei_1`, F1, no significant difference, low | 0.9268 | comparison.json metrics.evidence_effect.s8_ei_1.f1_no_difference.low | 0.926827 | PASS |
| 579 | Table 6, `s8_ei_1`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei_1.f1_no_difference.high | 0.963037 | PASS |
| 580 | Table 6, `s8_ei_1_temperature`, F1, significantly increased, estimate | 0.9312 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_increased.estimate | 0.931204 | PASS |
| 581 | Table 6, `s8_ei_1_temperature`, F1, significantly increased, low | 0.9103 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_increased.low | 0.910344 | PASS |
| 582 | Table 6, `s8_ei_1_temperature`, F1, significantly increased, high | 0.9490 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_increased.high | 0.948959 | PASS |
| 583 | Table 6, `s8_ei_1_temperature`, F1, significantly decreased, estimate | 0.9338 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_decreased.estimate | 0.933786 | PASS |
| 584 | Table 6, `s8_ei_1_temperature`, F1, significantly decreased, low | 0.9097 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_decreased.low | 0.909672 | PASS |
| 585 | Table 6, `s8_ei_1_temperature`, F1, significantly decreased, high | 0.9561 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_decreased.high | 0.956081 | PASS |
| 586 | Table 6, `s8_ei_1_temperature`, F1, no significant difference, estimate | 0.9468 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_no_difference.estimate | 0.946757 | PASS |
| 587 | Table 6, `s8_ei_1_temperature`, F1, no significant difference, low | 0.9268 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_no_difference.low | 0.926827 | PASS |
| 588 | Table 6, `s8_ei_1_temperature`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei_1_temperature.f1_no_difference.high | 0.963037 | PASS |
| 589 | Table 6, `s8_ei_1-raw`, F1, significantly increased, estimate | 0.9312 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_increased.estimate | 0.931204 | PASS |
| 590 | Table 6, `s8_ei_1-raw`, F1, significantly increased, low | 0.9103 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_increased.low | 0.910344 | PASS |
| 591 | Table 6, `s8_ei_1-raw`, F1, significantly increased, high | 0.9490 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_increased.high | 0.948959 | PASS |
| 592 | Table 6, `s8_ei_1-raw`, F1, significantly decreased, estimate | 0.9338 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_decreased.estimate | 0.933786 | PASS |
| 593 | Table 6, `s8_ei_1-raw`, F1, significantly decreased, low | 0.9097 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_decreased.low | 0.909672 | PASS |
| 594 | Table 6, `s8_ei_1-raw`, F1, significantly decreased, high | 0.9561 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_decreased.high | 0.956081 | PASS |
| 595 | Table 6, `s8_ei_1-raw`, F1, no significant difference, estimate | 0.9468 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_no_difference.estimate | 0.946757 | PASS |
| 596 | Table 6, `s8_ei_1-raw`, F1, no significant difference, low | 0.9268 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_no_difference.low | 0.926827 | PASS |
| 597 | Table 6, `s8_ei_1-raw`, F1, no significant difference, high | 0.9630 | comparison.json metrics.evidence_effect.s8_ei_1-raw.f1_no_difference.high | 0.963037 | PASS |
| 598 | Table 6, `s8_ei_2`, F1, significantly increased, estimate | 0.9337 | comparison.json metrics.evidence_effect.s8_ei_2.f1_increased.estimate | 0.933661 | PASS |
| 599 | Table 6, `s8_ei_2`, F1, significantly increased, low | 0.9135 | comparison.json metrics.evidence_effect.s8_ei_2.f1_increased.low | 0.913519 | PASS |
| 600 | Table 6, `s8_ei_2`, F1, significantly increased, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2.f1_increased.high | 0.951632 | PASS |
| 601 | Table 6, `s8_ei_2`, F1, significantly decreased, estimate | 0.9316 | comparison.json metrics.evidence_effect.s8_ei_2.f1_decreased.estimate | 0.931553 | PASS |
| 602 | Table 6, `s8_ei_2`, F1, significantly decreased, low | 0.9059 | comparison.json metrics.evidence_effect.s8_ei_2.f1_decreased.low | 0.905851 | PASS |
| 603 | Table 6, `s8_ei_2`, F1, significantly decreased, high | 0.9534 | comparison.json metrics.evidence_effect.s8_ei_2.f1_decreased.high | 0.953384 | PASS |
| 604 | Table 6, `s8_ei_2`, F1, no significant difference, estimate | 0.9423 | comparison.json metrics.evidence_effect.s8_ei_2.f1_no_difference.estimate | 0.942326 | PASS |
| 605 | Table 6, `s8_ei_2`, F1, no significant difference, low | 0.9213 | comparison.json metrics.evidence_effect.s8_ei_2.f1_no_difference.low | 0.921345 | PASS |
| 606 | Table 6, `s8_ei_2`, F1, no significant difference, high | 0.9584 | comparison.json metrics.evidence_effect.s8_ei_2.f1_no_difference.high | 0.958370 | PASS |
| 607 | Table 6, `s8_ei_2_temperature`, F1, significantly increased, estimate | 0.9337 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_increased.estimate | 0.933661 | PASS |
| 608 | Table 6, `s8_ei_2_temperature`, F1, significantly increased, low | 0.9135 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_increased.low | 0.913519 | PASS |
| 609 | Table 6, `s8_ei_2_temperature`, F1, significantly increased, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_increased.high | 0.951632 | PASS |
| 610 | Table 6, `s8_ei_2_temperature`, F1, significantly decreased, estimate | 0.9316 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_decreased.estimate | 0.931553 | PASS |
| 611 | Table 6, `s8_ei_2_temperature`, F1, significantly decreased, low | 0.9059 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_decreased.low | 0.905851 | PASS |
| 612 | Table 6, `s8_ei_2_temperature`, F1, significantly decreased, high | 0.9534 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_decreased.high | 0.953384 | PASS |
| 613 | Table 6, `s8_ei_2_temperature`, F1, no significant difference, estimate | 0.9423 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_no_difference.estimate | 0.942326 | PASS |
| 614 | Table 6, `s8_ei_2_temperature`, F1, no significant difference, low | 0.9213 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_no_difference.low | 0.921345 | PASS |
| 615 | Table 6, `s8_ei_2_temperature`, F1, no significant difference, high | 0.9584 | comparison.json metrics.evidence_effect.s8_ei_2_temperature.f1_no_difference.high | 0.958370 | PASS |
| 616 | Table 6, `s8_ei_2-raw`, F1, significantly increased, estimate | 0.9337 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_increased.estimate | 0.933661 | PASS |
| 617 | Table 6, `s8_ei_2-raw`, F1, significantly increased, low | 0.9135 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_increased.low | 0.913519 | PASS |
| 618 | Table 6, `s8_ei_2-raw`, F1, significantly increased, high | 0.9516 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_increased.high | 0.951632 | PASS |
| 619 | Table 6, `s8_ei_2-raw`, F1, significantly decreased, estimate | 0.9316 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_decreased.estimate | 0.931553 | PASS |
| 620 | Table 6, `s8_ei_2-raw`, F1, significantly decreased, low | 0.9059 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_decreased.low | 0.905851 | PASS |
| 621 | Table 6, `s8_ei_2-raw`, F1, significantly decreased, high | 0.9534 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_decreased.high | 0.953384 | PASS |
| 622 | Table 6, `s8_ei_2-raw`, F1, no significant difference, estimate | 0.9423 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_no_difference.estimate | 0.942326 | PASS |
| 623 | Table 6, `s8_ei_2-raw`, F1, no significant difference, low | 0.9213 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_no_difference.low | 0.921345 | PASS |
| 624 | Table 6, `s8_ei_2-raw`, F1, no significant difference, high | 0.9584 | comparison.json metrics.evidence_effect.s8_ei_2-raw.f1_no_difference.high | 0.958370 | PASS |
| 625 | Table 6, `s8_ei_3`, F1, significantly increased, estimate | 0.9325 | comparison.json metrics.evidence_effect.s8_ei_3.f1_increased.estimate | 0.932515 | PASS |
| 626 | Table 6, `s8_ei_3`, F1, significantly increased, low | 0.9121 | comparison.json metrics.evidence_effect.s8_ei_3.f1_increased.low | 0.912098 | PASS |
| 627 | Table 6, `s8_ei_3`, F1, significantly increased, high | 0.9505 | comparison.json metrics.evidence_effect.s8_ei_3.f1_increased.high | 0.950522 | PASS |
| 628 | Table 6, `s8_ei_3`, F1, significantly decreased, estimate | 0.9378 | comparison.json metrics.evidence_effect.s8_ei_3.f1_decreased.estimate | 0.937815 | PASS |
| 629 | Table 6, `s8_ei_3`, F1, significantly decreased, low | 0.9118 | comparison.json metrics.evidence_effect.s8_ei_3.f1_decreased.low | 0.911849 | PASS |
| 630 | Table 6, `s8_ei_3`, F1, significantly decreased, high | 0.9596 | comparison.json metrics.evidence_effect.s8_ei_3.f1_decreased.high | 0.959582 | PASS |
| 631 | Table 6, `s8_ei_3`, F1, no significant difference, estimate | 0.9435 | comparison.json metrics.evidence_effect.s8_ei_3.f1_no_difference.estimate | 0.943470 | PASS |
| 632 | Table 6, `s8_ei_3`, F1, no significant difference, low | 0.9237 | comparison.json metrics.evidence_effect.s8_ei_3.f1_no_difference.low | 0.923703 | PASS |
| 633 | Table 6, `s8_ei_3`, F1, no significant difference, high | 0.9595 | comparison.json metrics.evidence_effect.s8_ei_3.f1_no_difference.high | 0.959502 | PASS |
| 634 | Table 6, `s8_ei_3_temperature`, F1, significantly increased, estimate | 0.9325 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_increased.estimate | 0.932515 | PASS |
| 635 | Table 6, `s8_ei_3_temperature`, F1, significantly increased, low | 0.9121 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_increased.low | 0.912098 | PASS |
| 636 | Table 6, `s8_ei_3_temperature`, F1, significantly increased, high | 0.9505 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_increased.high | 0.950522 | PASS |
| 637 | Table 6, `s8_ei_3_temperature`, F1, significantly decreased, estimate | 0.9378 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_decreased.estimate | 0.937815 | PASS |
| 638 | Table 6, `s8_ei_3_temperature`, F1, significantly decreased, low | 0.9118 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_decreased.low | 0.911849 | PASS |
| 639 | Table 6, `s8_ei_3_temperature`, F1, significantly decreased, high | 0.9596 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_decreased.high | 0.959582 | PASS |
| 640 | Table 6, `s8_ei_3_temperature`, F1, no significant difference, estimate | 0.9435 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_no_difference.estimate | 0.943470 | PASS |
| 641 | Table 6, `s8_ei_3_temperature`, F1, no significant difference, low | 0.9237 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_no_difference.low | 0.923703 | PASS |
| 642 | Table 6, `s8_ei_3_temperature`, F1, no significant difference, high | 0.9595 | comparison.json metrics.evidence_effect.s8_ei_3_temperature.f1_no_difference.high | 0.959502 | PASS |
| 643 | Table 6, `s8_ei_3-raw`, F1, significantly increased, estimate | 0.9325 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_increased.estimate | 0.932515 | PASS |
| 644 | Table 6, `s8_ei_3-raw`, F1, significantly increased, low | 0.9121 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_increased.low | 0.912098 | PASS |
| 645 | Table 6, `s8_ei_3-raw`, F1, significantly increased, high | 0.9505 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_increased.high | 0.950522 | PASS |
| 646 | Table 6, `s8_ei_3-raw`, F1, significantly decreased, estimate | 0.9378 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_decreased.estimate | 0.937815 | PASS |
| 647 | Table 6, `s8_ei_3-raw`, F1, significantly decreased, low | 0.9118 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_decreased.low | 0.911849 | PASS |
| 648 | Table 6, `s8_ei_3-raw`, F1, significantly decreased, high | 0.9596 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_decreased.high | 0.959582 | PASS |
| 649 | Table 6, `s8_ei_3-raw`, F1, no significant difference, estimate | 0.9435 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_no_difference.estimate | 0.943470 | PASS |
| 650 | Table 6, `s8_ei_3-raw`, F1, no significant difference, low | 0.9237 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_no_difference.low | 0.923703 | PASS |
| 651 | Table 6, `s8_ei_3-raw`, F1, no significant difference, high | 0.9595 | comparison.json metrics.evidence_effect.s8_ei_3-raw.f1_no_difference.high | 0.959502 | PASS |
| 652 | Table 6, `gemma4_31b`, F1, significantly increased, estimate | 0.8455 | comparison.json metrics.evidence_effect.gemma4_31b.f1_increased.estimate | 0.845466 | PASS |
| 653 | Table 6, `gemma4_31b`, F1, significantly increased, low | 0.8088 | comparison.json metrics.evidence_effect.gemma4_31b.f1_increased.low | 0.808773 | PASS |
| 654 | Table 6, `gemma4_31b`, F1, significantly increased, high | 0.8793 | comparison.json metrics.evidence_effect.gemma4_31b.f1_increased.high | 0.879348 | PASS |
| 655 | Table 6, `gemma4_31b`, F1, significantly decreased, estimate | 0.8241 | comparison.json metrics.evidence_effect.gemma4_31b.f1_decreased.estimate | 0.824104 | PASS |
| 656 | Table 6, `gemma4_31b`, F1, significantly decreased, low | 0.7787 | comparison.json metrics.evidence_effect.gemma4_31b.f1_decreased.low | 0.778712 | PASS |
| 657 | Table 6, `gemma4_31b`, F1, significantly decreased, high | 0.8627 | comparison.json metrics.evidence_effect.gemma4_31b.f1_decreased.high | 0.862701 | PASS |
| 658 | Table 6, `gemma4_31b`, F1, no significant difference, estimate | 0.9182 | comparison.json metrics.evidence_effect.gemma4_31b.f1_no_difference.estimate | 0.918191 | PASS |
| 659 | Table 6, `gemma4_31b`, F1, no significant difference, low | 0.8947 | comparison.json metrics.evidence_effect.gemma4_31b.f1_no_difference.low | 0.894729 | PASS |
| 660 | Table 6, `gemma4_31b`, F1, no significant difference, high | 0.9395 | comparison.json metrics.evidence_effect.gemma4_31b.f1_no_difference.high | 0.939537 | PASS |
| 661 | Table 6, `gemma4_31b_temperature`, F1, significantly increased, estimate | 0.8399 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_increased.estimate | 0.839895 | PASS |
| 662 | Table 6, `gemma4_31b_temperature`, F1, significantly increased, low | 0.8026 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_increased.low | 0.802636 | PASS |
| 663 | Table 6, `gemma4_31b_temperature`, F1, significantly increased, high | 0.8743 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_increased.high | 0.874304 | PASS |
| 664 | Table 6, `gemma4_31b_temperature`, F1, significantly decreased, estimate | 0.8248 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_decreased.estimate | 0.824841 | PASS |
| 665 | Table 6, `gemma4_31b_temperature`, F1, significantly decreased, low | 0.7785 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_decreased.low | 0.778545 | PASS |
| 666 | Table 6, `gemma4_31b_temperature`, F1, significantly decreased, high | 0.8640 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_decreased.high | 0.864026 | PASS |
| 667 | Table 6, `gemma4_31b_temperature`, F1, no significant difference, estimate | 0.9178 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_no_difference.estimate | 0.917782 | PASS |
| 668 | Table 6, `gemma4_31b_temperature`, F1, no significant difference, low | 0.8950 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_no_difference.low | 0.895038 | PASS |
| 669 | Table 6, `gemma4_31b_temperature`, F1, no significant difference, high | 0.9388 | comparison.json metrics.evidence_effect.gemma4_31b_temperature.f1_no_difference.high | 0.938811 | PASS |
| 670 | Table 6, `gemma4_31b-raw`, F1, significantly increased, estimate | 0.8399 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_increased.estimate | 0.839895 | PASS |
| 671 | Table 6, `gemma4_31b-raw`, F1, significantly increased, low | 0.8026 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_increased.low | 0.802636 | PASS |
| 672 | Table 6, `gemma4_31b-raw`, F1, significantly increased, high | 0.8743 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_increased.high | 0.874304 | PASS |
| 673 | Table 6, `gemma4_31b-raw`, F1, significantly decreased, estimate | 0.8248 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_decreased.estimate | 0.824841 | PASS |
| 674 | Table 6, `gemma4_31b-raw`, F1, significantly decreased, low | 0.7785 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_decreased.low | 0.778545 | PASS |
| 675 | Table 6, `gemma4_31b-raw`, F1, significantly decreased, high | 0.8640 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_decreased.high | 0.864026 | PASS |
| 676 | Table 6, `gemma4_31b-raw`, F1, no significant difference, estimate | 0.9178 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_no_difference.estimate | 0.917782 | PASS |
| 677 | Table 6, `gemma4_31b-raw`, F1, no significant difference, low | 0.8950 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_no_difference.low | 0.895038 | PASS |
| 678 | Table 6, `gemma4_31b-raw`, F1, no significant difference, high | 0.9388 | comparison.json metrics.evidence_effect.gemma4_31b-raw.f1_no_difference.high | 0.938811 | PASS |
| 679 | results, line 87 (bar), number 1 | 0.0119 | comparison.json release_bar criteria.1.values.ece | 0.011914 | PASS |
| 680 | results, line 87 (bar), number 2 | 0.05 | comparison.json release_bar criteria.1.values.limit | 0.050000 | PASS |
| 681 | results, line 87 (bar), number 3 | -0.8773 | comparison.json bar_intervals.s8_ei_minus_prior estimate | -0.877268 | PASS |
| 682 | results, line 87 (bar), number 4 | -0.9225 | comparison.json bar_intervals.s8_ei_minus_prior low | -0.922517 | PASS |
| 683 | results, line 87 (bar), number 5 | -0.8281 | comparison.json bar_intervals.s8_ei_minus_prior high | -0.828053 | PASS |
| 684 | results, line 87 (bar), number 6 | -0.2084 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b estimate | -0.208355 | PASS |
| 685 | results, line 87 (bar), number 7 | -0.2720 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b low | -0.272016 | PASS |
| 686 | results, line 87 (bar), number 8 | -0.1505 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b high | -0.150517 | PASS |
| 687 | results, line 87 (bar), number 9 | 333 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b articles | 333 | PASS |
| 688 | results, line 87 (bar), number 10 | 0.9399 | comparison.json release_bar criteria.4.values.s8_ei | 0.939909 | PASS |
| 689 | results, line 87 (bar), number 11 | 0.8626 | comparison.json release_bar criteria.4.values.gemma4_31b | 0.862587 | PASS |
| 690 | results, line 89 (bar_outcome, the whole line) | verbatim | comparison.json release_bar.outcome | the same text | PASS |
| 691 | Table 7, `s8_ei` minus `prior`, Log loss, System | 0.1994 | comparison.json comparisons[0].system_estimate | 0.199449 | PASS |
| 692 | Table 7, `s8_ei` minus `prior`, Log loss, Reference | 1.0767 | comparison.json comparisons[0].reference_estimate | 1.076717 | PASS |
| 693 | Table 7, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), estimate | -0.8773 | comparison.json comparisons[0] (G1, evidence_effect, s8_ei minus prior, log_loss).difference estimate | -0.877268 | PASS |
| 694 | Table 7, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), low | -0.9201 | comparison.json comparisons[0] (G1, evidence_effect, s8_ei minus prior, log_loss).difference low | -0.920105 | PASS |
| 695 | Table 7, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), high | -0.8274 | comparison.json comparisons[0] (G1, evidence_effect, s8_ei minus prior, log_loss).difference high | -0.827397 | PASS |
| 696 | Table 7, `s8_ei` minus `prior`, Brier score, System | 0.1001 | comparison.json comparisons[1].system_estimate | 0.100116 | PASS |
| 697 | Table 7, `s8_ei` minus `prior`, Brier score, Reference | 0.6525 | comparison.json comparisons[1].reference_estimate | 0.652536 | PASS |
| 698 | Table 7, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), estimate | -0.5524 | comparison.json comparisons[1] (G1, evidence_effect, s8_ei minus prior, brier).difference estimate | -0.552420 | PASS |
| 699 | Table 7, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), low | -0.5767 | comparison.json comparisons[1] (G1, evidence_effect, s8_ei minus prior, brier).difference low | -0.576715 | PASS |
| 700 | Table 7, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), high | -0.5247 | comparison.json comparisons[1] (G1, evidence_effect, s8_ei minus prior, brier).difference high | -0.524697 | PASS |
| 701 | Table 7, `s8_ei` minus `prior`, ECE, System | 0.0119 | comparison.json comparisons[7].system_estimate | 0.011914 | PASS |
| 702 | Table 7, `s8_ei` minus `prior`, ECE, Reference | 0.0305 | comparison.json comparisons[7].reference_estimate | 0.030540 | PASS |
| 703 | Table 7, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), estimate | -0.0186 | comparison.json comparisons[7] (G1, evidence_effect, s8_ei minus prior, ece).difference estimate | -0.018626 | PASS |
| 704 | Table 7, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), low | -0.0461 | comparison.json comparisons[7] (G1, evidence_effect, s8_ei minus prior, ece).difference low | -0.046084 | PASS |
| 705 | Table 7, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), high | 0.0187 | comparison.json comparisons[7] (G1, evidence_effect, s8_ei minus prior, ece).difference high | 0.018718 | PASS |
| 706 | Table 7, `s8_ei` minus `prior`, Accuracy, System | 0.9409 | comparison.json comparisons[2].system_estimate | 0.940887 | PASS |
| 707 | Table 7, `s8_ei` minus `prior`, Accuracy, Reference | 0.4212 | comparison.json comparisons[2].reference_estimate | 0.421182 | PASS |
| 708 | Table 7, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), estimate | 0.5197 | comparison.json comparisons[2] (G1, evidence_effect, s8_ei minus prior, accuracy).difference estimate | 0.519704 | PASS |
| 709 | Table 7, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), low | 0.4790 | comparison.json comparisons[2] (G1, evidence_effect, s8_ei minus prior, accuracy).difference low | 0.479002 | PASS |
| 710 | Table 7, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), high | 0.5541 | comparison.json comparisons[2] (G1, evidence_effect, s8_ei minus prior, accuracy).difference high | 0.554125 | PASS |
| 711 | Table 7, `s8_ei` minus `prior`, Macro-F1, System | 0.9399 | comparison.json comparisons[3].system_estimate | 0.939909 | PASS |
| 712 | Table 7, `s8_ei` minus `prior`, Macro-F1, Reference | 0.1976 | comparison.json comparisons[3].reference_estimate | 0.197574 | PASS |
| 713 | Table 7, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), estimate | 0.7423 | comparison.json comparisons[3] (G1, evidence_effect, s8_ei minus prior, macro_f1).difference estimate | 0.742336 | PASS |
| 714 | Table 7, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), low | 0.7217 | comparison.json comparisons[3] (G1, evidence_effect, s8_ei minus prior, macro_f1).difference low | 0.721684 | PASS |
| 715 | Table 7, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), high | 0.7602 | comparison.json comparisons[3] (G1, evidence_effect, s8_ei minus prior, macro_f1).difference high | 0.760240 | PASS |
| 716 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, System | 0.9358 | comparison.json comparisons[4].system_estimate | 0.935802 | PASS |
| 717 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, Reference | 0.0000 | comparison.json comparisons[4].reference_estimate | 0.000000 | PASS |
| 718 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), estimate | 0.9358 | comparison.json comparisons[4] (G1, evidence_effect, s8_ei minus prior, f1_increased).difference estimate | 0.935802 | PASS |
| 719 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), low | 0.9152 | comparison.json comparisons[4] (G1, evidence_effect, s8_ei minus prior, f1_increased).difference low | 0.915164 | PASS |
| 720 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), high | 0.9529 | comparison.json comparisons[4] (G1, evidence_effect, s8_ei minus prior, f1_increased).difference high | 0.952917 | PASS |
| 721 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, System | 0.9365 | comparison.json comparisons[5].system_estimate | 0.936455 | PASS |
| 722 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, Reference | 0.0000 | comparison.json comparisons[5].reference_estimate | 0.000000 | PASS |
| 723 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.9365 | comparison.json comparisons[5] (G1, evidence_effect, s8_ei minus prior, f1_decreased).difference estimate | 0.936455 | PASS |
| 724 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), low | 0.9113 | comparison.json comparisons[5] (G1, evidence_effect, s8_ei minus prior, f1_decreased).difference low | 0.911256 | PASS |
| 725 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), high | 0.9577 | comparison.json comparisons[5] (G1, evidence_effect, s8_ei minus prior, f1_decreased).difference high | 0.957656 | PASS |
| 726 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, System | 0.9475 | comparison.json comparisons[6].system_estimate | 0.947471 | PASS |
| 727 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, Reference | 0.5927 | comparison.json comparisons[6].reference_estimate | 0.592721 | PASS |
| 728 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), estimate | 0.3547 | comparison.json comparisons[6] (G1, evidence_effect, s8_ei minus prior, f1_no_difference).difference estimate | 0.354750 | PASS |
| 729 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), low | 0.3208 | comparison.json comparisons[6] (G1, evidence_effect, s8_ei minus prior, f1_no_difference).difference low | 0.320787 | PASS |
| 730 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), high | 0.3874 | comparison.json comparisons[6] (G1, evidence_effect, s8_ei minus prior, f1_no_difference).difference high | 0.387380 | PASS |
| 731 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, System | 0.1994 | comparison.json comparisons[8].system_estimate | 0.199449 | PASS |
| 732 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, Reference | 0.4078 | comparison.json comparisons[8].reference_estimate | 0.407805 | PASS |
| 733 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), estimate | -0.2084 | comparison.json comparisons[8] (G1, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference estimate | -0.208355 | PASS |
| 734 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), low | -0.2690 | comparison.json comparisons[8] (G1, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference low | -0.268980 | PASS |
| 735 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), high | -0.1498 | comparison.json comparisons[8] (G1, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference high | -0.149845 | PASS |
| 736 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, System | 0.1001 | comparison.json comparisons[9].system_estimate | 0.100116 | PASS |
| 737 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, Reference | 0.2040 | comparison.json comparisons[9].reference_estimate | 0.204016 | PASS |
| 738 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), estimate | -0.1039 | comparison.json comparisons[9] (G1, evidence_effect, s8_ei minus gemma4_31b, brier).difference estimate | -0.103900 | PASS |
| 739 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), low | -0.1357 | comparison.json comparisons[9] (G1, evidence_effect, s8_ei minus gemma4_31b, brier).difference low | -0.135694 | PASS |
| 740 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), high | -0.0741 | comparison.json comparisons[9] (G1, evidence_effect, s8_ei minus gemma4_31b, brier).difference high | -0.074131 | PASS |
| 741 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, System | 0.0119 | comparison.json comparisons[15].system_estimate | 0.011914 | PASS |
| 742 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, Reference | 0.0283 | comparison.json comparisons[15].reference_estimate | 0.028293 | PASS |
| 743 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), estimate | -0.0164 | comparison.json comparisons[15] (G1, evidence_effect, s8_ei minus gemma4_31b, ece).difference estimate | -0.016378 | PASS |
| 744 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), low | -0.0369 | comparison.json comparisons[15] (G1, evidence_effect, s8_ei minus gemma4_31b, ece).difference low | -0.036914 | PASS |
| 745 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), high | -0.0007 | comparison.json comparisons[15] (G1, evidence_effect, s8_ei minus gemma4_31b, ece).difference high | -0.000722 | PASS |
| 746 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, System | 0.9409 | comparison.json comparisons[10].system_estimate | 0.940887 | PASS |
| 747 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, Reference | 0.8711 | comparison.json comparisons[10].reference_estimate | 0.871100 | PASS |
| 748 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), estimate | 0.0698 | comparison.json comparisons[10] (G1, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference estimate | 0.069787 | PASS |
| 749 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), low | 0.0487 | comparison.json comparisons[10] (G1, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference low | 0.048738 | PASS |
| 750 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), high | 0.0923 | comparison.json comparisons[10] (G1, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference high | 0.092342 | PASS |
| 751 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, System | 0.9399 | comparison.json comparisons[11].system_estimate | 0.939909 | PASS |
| 752 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, Reference | 0.8626 | comparison.json comparisons[11].reference_estimate | 0.862587 | PASS |
| 753 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), estimate | 0.0773 | comparison.json comparisons[11] (G1, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference estimate | 0.077322 | PASS |
| 754 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), low | 0.0543 | comparison.json comparisons[11] (G1, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference low | 0.054326 | PASS |
| 755 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), high | 0.1022 | comparison.json comparisons[11] (G1, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference high | 0.102247 | PASS |
| 756 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, System | 0.9358 | comparison.json comparisons[12].system_estimate | 0.935802 | PASS |
| 757 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Reference | 0.8455 | comparison.json comparisons[12].reference_estimate | 0.845466 | PASS |
| 758 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0903 | comparison.json comparisons[12] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference estimate | 0.090336 | PASS |
| 759 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), low | 0.0588 | comparison.json comparisons[12] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference low | 0.058806 | PASS |
| 760 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), high | 0.1243 | comparison.json comparisons[12] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference high | 0.124283 | PASS |
| 761 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, System | 0.9365 | comparison.json comparisons[13].system_estimate | 0.936455 | PASS |
| 762 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Reference | 0.8241 | comparison.json comparisons[13].reference_estimate | 0.824104 | PASS |
| 763 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.1124 | comparison.json comparisons[13] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference estimate | 0.112351 | PASS |
| 764 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), low | 0.0771 | comparison.json comparisons[13] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference low | 0.077092 | PASS |
| 765 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), high | 0.1521 | comparison.json comparisons[13] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference high | 0.152118 | PASS |
| 766 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, System | 0.9475 | comparison.json comparisons[14].system_estimate | 0.947471 | PASS |
| 767 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Reference | 0.9182 | comparison.json comparisons[14].reference_estimate | 0.918191 | PASS |
| 768 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0293 | comparison.json comparisons[14] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference estimate | 0.029280 | PASS |
| 769 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), low | 0.0120 | comparison.json comparisons[14] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference low | 0.012040 | PASS |
| 770 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), high | 0.0476 | comparison.json comparisons[14] (G1, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference high | 0.047610 | PASS |
| 771 | Table 8, 1, ECE, s8_ei, or the difference | 0.0119 | comparison.json release_bar criteria.1.values.ece | 0.011914 | PASS |
| 772 | Table 8, 1, ECE, Limit, 95 percent interval, or Gemma 4 31B-it | 0.0500 | comparison.json release_bar criteria.1.values.limit | 0.050000 | PASS |
| 773 | Table 8, 2, Log loss, s8_ei minus the prior, s8_ei, or the difference | -0.8773 | comparison.json bar_intervals.s8_ei_minus_prior estimate | -0.877268 | PASS |
| 774 | Table 8, 2, Log loss, s8_ei minus the prior, Limit, 95 percent interval, or Gemma 4 31B-it, low | -0.9225 | comparison.json bar_intervals.s8_ei_minus_prior low | -0.922517 | PASS |
| 775 | Table 8, 2, Log loss, s8_ei minus the prior, Limit, 95 percent interval, or Gemma 4 31B-it, high | -0.8281 | comparison.json bar_intervals.s8_ei_minus_prior high | -0.828053 | PASS |
| 776 | Table 8, 3, Log loss, s8_ei minus Gemma 4 31B-it, s8_ei, or the difference | -0.2084 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b estimate | -0.208355 | PASS |
| 777 | Table 8, 3, Log loss, s8_ei minus Gemma 4 31B-it, Limit, 95 percent interval, or Gemma 4 31B-it, low | -0.2720 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b low | -0.272016 | PASS |
| 778 | Table 8, 3, Log loss, s8_ei minus Gemma 4 31B-it, Limit, 95 percent interval, or Gemma 4 31B-it, high | -0.1505 | comparison.json bar_intervals.s8_ei_minus_gemma4_31b high | -0.150517 | PASS |
| 779 | Table 8, 4, Macro-F1, s8_ei against Gemma 4 31B-it, s8_ei, or the difference | 0.9399 | comparison.json release_bar criteria.4.values.s8_ei | 0.939909 | PASS |
| 780 | Table 8, 4, Macro-F1, s8_ei against Gemma 4 31B-it, Limit, 95 percent interval, or Gemma 4 31B-it | 0.8626 | comparison.json release_bar criteria.4.values.gemma4_31b | 0.862587 | PASS |
| 781 | results, line 124 (g2), number 1 | 0.2085 | comparison.json g2_spread log_loss.s8_ei_1 | 0.208516 | PASS |
| 782 | results, line 124 (g2), number 2 | 0.2098 | comparison.json g2_spread log_loss.s8_ei_2 | 0.209829 | PASS |
| 783 | results, line 124 (g2), number 3 | 0.1998 | comparison.json g2_spread log_loss.s8_ei_3 | 0.199843 | PASS |
| 784 | results, line 124 (g2), number 4 | 0.0100 | comparison.json g2_spread max_minus_min | 0.009986 | PASS |
| 785 | results, line 124 (g2), number 5 | 0.0054 | comparison.json g2_spread standard_deviation | 0.005426 | PASS |
| 786 | results, line 126 (runs, the whole line) | verbatim | comparison.json runs | the same text | PASS |
| 787 | Table 9, `s8_ei` minus `s8_ei_1`, Log loss, Difference (95 percent interval), estimate | -0.0091 | comparison.json comparisons[16] (G2, evidence_effect, s8_ei minus s8_ei_1, log_loss).difference estimate | -0.009066 | PASS |
| 788 | Table 9, `s8_ei` minus `s8_ei_1`, Log loss, Difference (95 percent interval), low | -0.0151 | comparison.json comparisons[16] (G2, evidence_effect, s8_ei minus s8_ei_1, log_loss).difference low | -0.015062 | PASS |
| 789 | Table 9, `s8_ei` minus `s8_ei_1`, Log loss, Difference (95 percent interval), high | -0.0030 | comparison.json comparisons[16] (G2, evidence_effect, s8_ei minus s8_ei_1, log_loss).difference high | -0.002993 | PASS |
| 790 | Table 9, `s8_ei` minus `s8_ei_1`, Brier score, Difference (95 percent interval), estimate | -0.0048 | comparison.json comparisons[17] (G2, evidence_effect, s8_ei minus s8_ei_1, brier).difference estimate | -0.004840 | PASS |
| 791 | Table 9, `s8_ei` minus `s8_ei_1`, Brier score, Difference (95 percent interval), low | -0.0088 | comparison.json comparisons[17] (G2, evidence_effect, s8_ei minus s8_ei_1, brier).difference low | -0.008797 | PASS |
| 792 | Table 9, `s8_ei` minus `s8_ei_1`, Brier score, Difference (95 percent interval), high | -0.0013 | comparison.json comparisons[17] (G2, evidence_effect, s8_ei minus s8_ei_1, brier).difference high | -0.001279 | PASS |
| 793 | Table 9, `s8_ei` minus `s8_ei_1`, ECE, Difference (95 percent interval), estimate | -0.0019 | comparison.json comparisons[23] (G2, evidence_effect, s8_ei minus s8_ei_1, ece).difference estimate | -0.001898 | PASS |
| 794 | Table 9, `s8_ei` minus `s8_ei_1`, ECE, Difference (95 percent interval), low | -0.0100 | comparison.json comparisons[23] (G2, evidence_effect, s8_ei minus s8_ei_1, ece).difference low | -0.010027 | PASS |
| 795 | Table 9, `s8_ei` minus `s8_ei_1`, ECE, Difference (95 percent interval), high | 0.0095 | comparison.json comparisons[23] (G2, evidence_effect, s8_ei minus s8_ei_1, ece).difference high | 0.009451 | PASS |
| 796 | Table 9, `s8_ei` minus `s8_ei_1`, Accuracy, Difference (95 percent interval), estimate | 0.0025 | comparison.json comparisons[18] (G2, evidence_effect, s8_ei minus s8_ei_1, accuracy).difference estimate | 0.002463 | PASS |
| 797 | Table 9, `s8_ei` minus `s8_ei_1`, Accuracy, Difference (95 percent interval), low | -0.0034 | comparison.json comparisons[18] (G2, evidence_effect, s8_ei minus s8_ei_1, accuracy).difference low | -0.003387 | PASS |
| 798 | Table 9, `s8_ei` minus `s8_ei_1`, Accuracy, Difference (95 percent interval), high | 0.0093 | comparison.json comparisons[18] (G2, evidence_effect, s8_ei minus s8_ei_1, accuracy).difference high | 0.009269 | PASS |
| 799 | Table 9, `s8_ei` minus `s8_ei_1`, Macro-F1, Difference (95 percent interval), estimate | 0.0027 | comparison.json comparisons[19] (G2, evidence_effect, s8_ei minus s8_ei_1, macro_f1).difference estimate | 0.002660 | PASS |
| 800 | Table 9, `s8_ei` minus `s8_ei_1`, Macro-F1, Difference (95 percent interval), low | -0.0038 | comparison.json comparisons[19] (G2, evidence_effect, s8_ei minus s8_ei_1, macro_f1).difference low | -0.003827 | PASS |
| 801 | Table 9, `s8_ei` minus `s8_ei_1`, Macro-F1, Difference (95 percent interval), high | 0.0104 | comparison.json comparisons[19] (G2, evidence_effect, s8_ei minus s8_ei_1, macro_f1).difference high | 0.010400 | PASS |
| 802 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0046 | comparison.json comparisons[20] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_increased).difference estimate | 0.004599 | PASS |
| 803 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly increased, Difference (95 percent interval), low | -0.0030 | comparison.json comparisons[20] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_increased).difference low | -0.002964 | PASS |
| 804 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly increased, Difference (95 percent interval), high | 0.0126 | comparison.json comparisons[20] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_increased).difference high | 0.012643 | PASS |
| 805 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.0027 | comparison.json comparisons[21] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_decreased).difference estimate | 0.002669 | PASS |
| 806 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly decreased, Difference (95 percent interval), low | -0.0090 | comparison.json comparisons[21] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_decreased).difference low | -0.009027 | PASS |
| 807 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly decreased, Difference (95 percent interval), high | 0.0156 | comparison.json comparisons[21] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_decreased).difference high | 0.015627 | PASS |
| 808 | Table 9, `s8_ei` minus `s8_ei_1`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0007 | comparison.json comparisons[22] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_no_difference).difference estimate | 0.000714 | PASS |
| 809 | Table 9, `s8_ei` minus `s8_ei_1`, F1, no significant difference, Difference (95 percent interval), low | -0.0051 | comparison.json comparisons[22] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_no_difference).difference low | -0.005141 | PASS |
| 810 | Table 9, `s8_ei` minus `s8_ei_1`, F1, no significant difference, Difference (95 percent interval), high | 0.0067 | comparison.json comparisons[22] (G2, evidence_effect, s8_ei minus s8_ei_1, f1_no_difference).difference high | 0.006683 | PASS |
| 811 | Table 9, `s8_ei` minus `s8_ei_2`, Log loss, Difference (95 percent interval), estimate | -0.0104 | comparison.json comparisons[24] (G2, evidence_effect, s8_ei minus s8_ei_2, log_loss).difference estimate | -0.010380 | PASS |
| 812 | Table 9, `s8_ei` minus `s8_ei_2`, Log loss, Difference (95 percent interval), low | -0.0178 | comparison.json comparisons[24] (G2, evidence_effect, s8_ei minus s8_ei_2, log_loss).difference low | -0.017834 | PASS |
| 813 | Table 9, `s8_ei` minus `s8_ei_2`, Log loss, Difference (95 percent interval), high | -0.0030 | comparison.json comparisons[24] (G2, evidence_effect, s8_ei minus s8_ei_2, log_loss).difference high | -0.003007 | PASS |
| 814 | Table 9, `s8_ei` minus `s8_ei_2`, Brier score, Difference (95 percent interval), estimate | -0.0053 | comparison.json comparisons[25] (G2, evidence_effect, s8_ei minus s8_ei_2, brier).difference estimate | -0.005302 | PASS |
| 815 | Table 9, `s8_ei` minus `s8_ei_2`, Brier score, Difference (95 percent interval), low | -0.0097 | comparison.json comparisons[25] (G2, evidence_effect, s8_ei minus s8_ei_2, brier).difference low | -0.009737 | PASS |
| 816 | Table 9, `s8_ei` minus `s8_ei_2`, Brier score, Difference (95 percent interval), high | -0.0013 | comparison.json comparisons[25] (G2, evidence_effect, s8_ei minus s8_ei_2, brier).difference high | -0.001264 | PASS |
| 817 | Table 9, `s8_ei` minus `s8_ei_2`, ECE, Difference (95 percent interval), estimate | -0.0068 | comparison.json comparisons[31] (G2, evidence_effect, s8_ei minus s8_ei_2, ece).difference estimate | -0.006844 | PASS |
| 818 | Table 9, `s8_ei` minus `s8_ei_2`, ECE, Difference (95 percent interval), low | -0.0137 | comparison.json comparisons[31] (G2, evidence_effect, s8_ei minus s8_ei_2, ece).difference low | -0.013703 | PASS |
| 819 | Table 9, `s8_ei` minus `s8_ei_2`, ECE, Difference (95 percent interval), high | 0.0085 | comparison.json comparisons[31] (G2, evidence_effect, s8_ei minus s8_ei_2, ece).difference high | 0.008531 | PASS |
| 820 | Table 9, `s8_ei` minus `s8_ei_2`, Accuracy, Difference (95 percent interval), estimate | 0.0041 | comparison.json comparisons[26] (G2, evidence_effect, s8_ei minus s8_ei_2, accuracy).difference estimate | 0.004105 | PASS |
| 821 | Table 9, `s8_ei` minus `s8_ei_2`, Accuracy, Difference (95 percent interval), low | -0.0015 | comparison.json comparisons[26] (G2, evidence_effect, s8_ei minus s8_ei_2, accuracy).difference low | -0.001535 | PASS |
| 822 | Table 9, `s8_ei` minus `s8_ei_2`, Accuracy, Difference (95 percent interval), high | 0.0099 | comparison.json comparisons[26] (G2, evidence_effect, s8_ei minus s8_ei_2, accuracy).difference high | 0.009927 | PASS |
| 823 | Table 9, `s8_ei` minus `s8_ei_2`, Macro-F1, Difference (95 percent interval), estimate | 0.0041 | comparison.json comparisons[27] (G2, evidence_effect, s8_ei minus s8_ei_2, macro_f1).difference estimate | 0.004063 | PASS |
| 824 | Table 9, `s8_ei` minus `s8_ei_2`, Macro-F1, Difference (95 percent interval), low | -0.0017 | comparison.json comparisons[27] (G2, evidence_effect, s8_ei minus s8_ei_2, macro_f1).difference low | -0.001662 | PASS |
| 825 | Table 9, `s8_ei` minus `s8_ei_2`, Macro-F1, Difference (95 percent interval), high | 0.0099 | comparison.json comparisons[27] (G2, evidence_effect, s8_ei minus s8_ei_2, macro_f1).difference high | 0.009943 | PASS |
| 826 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0021 | comparison.json comparisons[28] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_increased).difference estimate | 0.002142 | PASS |
| 827 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly increased, Difference (95 percent interval), low | -0.0045 | comparison.json comparisons[28] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_increased).difference low | -0.004543 | PASS |
| 828 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly increased, Difference (95 percent interval), high | 0.0088 | comparison.json comparisons[28] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_increased).difference high | 0.008837 | PASS |
| 829 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.0049 | comparison.json comparisons[29] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_decreased).difference estimate | 0.004902 | PASS |
| 830 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly decreased, Difference (95 percent interval), low | -0.0051 | comparison.json comparisons[29] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_decreased).difference low | -0.005109 | PASS |
| 831 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly decreased, Difference (95 percent interval), high | 0.0145 | comparison.json comparisons[29] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_decreased).difference high | 0.014461 | PASS |
| 832 | Table 9, `s8_ei` minus `s8_ei_2`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0051 | comparison.json comparisons[30] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_no_difference).difference estimate | 0.005144 | PASS |
| 833 | Table 9, `s8_ei` minus `s8_ei_2`, F1, no significant difference, Difference (95 percent interval), low | -0.0009 | comparison.json comparisons[30] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_no_difference).difference low | -0.000883 | PASS |
| 834 | Table 9, `s8_ei` minus `s8_ei_2`, F1, no significant difference, Difference (95 percent interval), high | 0.0112 | comparison.json comparisons[30] (G2, evidence_effect, s8_ei minus s8_ei_2, f1_no_difference).difference high | 0.011228 | PASS |
| 835 | Table 9, `s8_ei` minus `s8_ei_3`, Log loss, Difference (95 percent interval), estimate | -0.0004 | comparison.json comparisons[32] (G2, evidence_effect, s8_ei minus s8_ei_3, log_loss).difference estimate | -0.000394 | PASS |
| 836 | Table 9, `s8_ei` minus `s8_ei_3`, Log loss, Difference (95 percent interval), low | -0.0073 | comparison.json comparisons[32] (G2, evidence_effect, s8_ei minus s8_ei_3, log_loss).difference low | -0.007324 | PASS |
| 837 | Table 9, `s8_ei` minus `s8_ei_3`, Log loss, Difference (95 percent interval), high | 0.0062 | comparison.json comparisons[32] (G2, evidence_effect, s8_ei minus s8_ei_3, log_loss).difference high | 0.006209 | PASS |
| 838 | Table 9, `s8_ei` minus `s8_ei_3`, Brier score, Difference (95 percent interval), estimate | 0.0010 | comparison.json comparisons[33] (G2, evidence_effect, s8_ei minus s8_ei_3, brier).difference estimate | 0.001002 | PASS |
| 839 | Table 9, `s8_ei` minus `s8_ei_3`, Brier score, Difference (95 percent interval), low | -0.0028 | comparison.json comparisons[33] (G2, evidence_effect, s8_ei minus s8_ei_3, brier).difference low | -0.002815 | PASS |
| 840 | Table 9, `s8_ei` minus `s8_ei_3`, Brier score, Difference (95 percent interval), high | 0.0046 | comparison.json comparisons[33] (G2, evidence_effect, s8_ei minus s8_ei_3, brier).difference high | 0.004614 | PASS |
| 841 | Table 9, `s8_ei` minus `s8_ei_3`, ECE, Difference (95 percent interval), estimate | -0.0018 | comparison.json comparisons[39] (G2, evidence_effect, s8_ei minus s8_ei_3, ece).difference estimate | -0.001783 | PASS |
| 842 | Table 9, `s8_ei` minus `s8_ei_3`, ECE, Difference (95 percent interval), low | -0.0115 | comparison.json comparisons[39] (G2, evidence_effect, s8_ei minus s8_ei_3, ece).difference low | -0.011528 | PASS |
| 843 | Table 9, `s8_ei` minus `s8_ei_3`, ECE, Difference (95 percent interval), high | 0.0119 | comparison.json comparisons[39] (G2, evidence_effect, s8_ei minus s8_ei_3, ece).difference high | 0.011920 | PASS |
| 844 | Table 9, `s8_ei` minus `s8_ei_3`, Accuracy, Difference (95 percent interval), estimate | 0.0025 | comparison.json comparisons[34] (G2, evidence_effect, s8_ei minus s8_ei_3, accuracy).difference estimate | 0.002463 | PASS |
| 845 | Table 9, `s8_ei` minus `s8_ei_3`, Accuracy, Difference (95 percent interval), low | -0.0025 | comparison.json comparisons[34] (G2, evidence_effect, s8_ei minus s8_ei_3, accuracy).difference low | -0.002496 | PASS |
| 846 | Table 9, `s8_ei` minus `s8_ei_3`, Accuracy, Difference (95 percent interval), high | 0.0079 | comparison.json comparisons[34] (G2, evidence_effect, s8_ei minus s8_ei_3, accuracy).difference high | 0.007930 | PASS |
| 847 | Table 9, `s8_ei` minus `s8_ei_3`, Macro-F1, Difference (95 percent interval), estimate | 0.0020 | comparison.json comparisons[35] (G2, evidence_effect, s8_ei minus s8_ei_3, macro_f1).difference estimate | 0.001976 | PASS |
| 848 | Table 9, `s8_ei` minus `s8_ei_3`, Macro-F1, Difference (95 percent interval), low | -0.0033 | comparison.json comparisons[35] (G2, evidence_effect, s8_ei minus s8_ei_3, macro_f1).difference low | -0.003254 | PASS |
| 849 | Table 9, `s8_ei` minus `s8_ei_3`, Macro-F1, Difference (95 percent interval), high | 0.0076 | comparison.json comparisons[35] (G2, evidence_effect, s8_ei minus s8_ei_3, macro_f1).difference high | 0.007557 | PASS |
| 850 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0033 | comparison.json comparisons[36] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_increased).difference estimate | 0.003287 | PASS |
| 851 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly increased, Difference (95 percent interval), low | -0.0042 | comparison.json comparisons[36] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_increased).difference low | -0.004187 | PASS |
| 852 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly increased, Difference (95 percent interval), high | 0.0115 | comparison.json comparisons[36] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_increased).difference high | 0.011522 | PASS |
| 853 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly decreased, Difference (95 percent interval), estimate | -0.0014 | comparison.json comparisons[37] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_decreased).difference estimate | -0.001360 | PASS |
| 854 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly decreased, Difference (95 percent interval), low | -0.0085 | comparison.json comparisons[37] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_decreased).difference low | -0.008483 | PASS |
| 855 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly decreased, Difference (95 percent interval), high | 0.0057 | comparison.json comparisons[37] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_decreased).difference high | 0.005709 | PASS |
| 856 | Table 9, `s8_ei` minus `s8_ei_3`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0040 | comparison.json comparisons[38] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_no_difference).difference estimate | 0.004001 | PASS |
| 857 | Table 9, `s8_ei` minus `s8_ei_3`, F1, no significant difference, Difference (95 percent interval), low | -0.0011 | comparison.json comparisons[38] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_no_difference).difference low | -0.001055 | PASS |
| 858 | Table 9, `s8_ei` minus `s8_ei_3`, F1, no significant difference, Difference (95 percent interval), high | 0.0097 | comparison.json comparisons[38] (G2, evidence_effect, s8_ei minus s8_ei_3, f1_no_difference).difference high | 0.009701 | PASS |
| 859 | results, line 159 (g3), number 1 | 0.9399 | comparison.json g3 macro_f1.estimate | 0.939909 | PASS |
| 860 | results, line 159 (g3), number 2 | 0.9225 | comparison.json g3 macro_f1.low | 0.922543 | PASS |
| 861 | results, line 159 (g3), number 3 | 0.9544 | comparison.json g3 macro_f1.high | 0.954353 | PASS |
| 862 | results, line 159 (g3), number 4 | 0.9407 | comparison.json g3 lenient_macro_f1.estimate | 0.940654 | PASS |
| 863 | results, line 159 (g3), number 5 | 1 | comparison.json g3 lenient_changed_prompts | 1 | PASS |
| 864 | results, line 159 (g3), number 6 | 3 | records_manifest.json accepted_labels_disagree.test | 3 | PASS |
| 865 | results, line 161 (g3_note, the whole line) | verbatim | comparison.json g3.note | the same text | PASS |
| 866 | Table 10, DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa), strict, Its macro-F1 | 0.780 | comparison.json g3 published.deyoung_2020_pipeline.value | 0.780000 | PASS |
| 867 | Table 10, DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa), strict, s8_ei's macro-F1 (95 percent interval), estimate | 0.9399 | comparison.json g3 macro_f1.estimate | 0.939909 | PASS |
| 868 | Table 10, DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa), strict, s8_ei's macro-F1 (95 percent interval), low | 0.9225 | comparison.json g3 macro_f1.low | 0.922543 | PASS |
| 869 | Table 10, DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa), strict, s8_ei's macro-F1 (95 percent interval), high | 0.9544 | comparison.json g3 macro_f1.high | 0.954353 | PASS |
| 870 | Table 10, DeYoung et al. 2020, with the right evidence given, strict, Its macro-F1 | 0.851 | comparison.json g3 published.deyoung_2020_oracle.value | 0.851000 | PASS |
| 871 | Table 10, DeYoung et al. 2020, with the right evidence given, strict, s8_ei's macro-F1 (95 percent interval), estimate | 0.9399 | comparison.json g3 macro_f1.estimate | 0.939909 | PASS |
| 872 | Table 10, DeYoung et al. 2020, with the right evidence given, strict, s8_ei's macro-F1 (95 percent interval), low | 0.9225 | comparison.json g3 macro_f1.low | 0.922543 | PASS |
| 873 | Table 10, DeYoung et al. 2020, with the right evidence given, strict, s8_ei's macro-F1 (95 percent interval), high | 0.9544 | comparison.json g3 macro_f1.high | 0.954353 | PASS |
| 874 | Table 10, Buchmann et al. 2024, LED, the lower end of its range, lenient, Its macro-F1 | 0.7430 | comparison.json g3 published_lenient.buchmann_2024_led_low.value | 0.743000 | PASS |
| 875 | Table 10, Buchmann et al. 2024, LED, the lower end of its range, lenient, s8_ei's macro-F1 (95 percent interval), estimate | 0.9407 | comparison.json g3 lenient_macro_f1.estimate | 0.940654 | PASS |
| 876 | Table 10, Buchmann et al. 2024, LED, the lower end of its range, lenient, s8_ei's macro-F1 (95 percent interval), low | 0.9240 | comparison.json g3 lenient_macro_f1.low | 0.923951 | PASS |
| 877 | Table 10, Buchmann et al. 2024, LED, the lower end of its range, lenient, s8_ei's macro-F1 (95 percent interval), high | 0.9551 | comparison.json g3 lenient_macro_f1.high | 0.955079 | PASS |
| 878 | Table 10, Buchmann et al. 2024, LED, the upper end of its range, lenient, Its macro-F1 | 0.7649 | comparison.json g3 published_lenient.buchmann_2024_led_high.value | 0.764900 | PASS |
| 879 | Table 10, Buchmann et al. 2024, LED, the upper end of its range, lenient, s8_ei's macro-F1 (95 percent interval), estimate | 0.9407 | comparison.json g3 lenient_macro_f1.estimate | 0.940654 | PASS |
| 880 | Table 10, Buchmann et al. 2024, LED, the upper end of its range, lenient, s8_ei's macro-F1 (95 percent interval), low | 0.9240 | comparison.json g3 lenient_macro_f1.low | 0.923951 | PASS |
| 881 | Table 10, Buchmann et al. 2024, LED, the upper end of its range, lenient, s8_ei's macro-F1 (95 percent interval), high | 0.9551 | comparison.json g3 lenient_macro_f1.high | 0.955079 | PASS |
| 882 | Table 10, Buchmann et al. 2024, LongT5, lenient, Its macro-F1 | 0.8154 | comparison.json g3 published_lenient.buchmann_2024_longt5.value | 0.815400 | PASS |
| 883 | Table 10, Buchmann et al. 2024, LongT5, lenient, s8_ei's macro-F1 (95 percent interval), estimate | 0.9407 | comparison.json g3 lenient_macro_f1.estimate | 0.940654 | PASS |
| 884 | Table 10, Buchmann et al. 2024, LongT5, lenient, s8_ei's macro-F1 (95 percent interval), low | 0.9240 | comparison.json g3 lenient_macro_f1.low | 0.923951 | PASS |
| 885 | Table 10, Buchmann et al. 2024, LongT5, lenient, s8_ei's macro-F1 (95 percent interval), high | 0.9551 | comparison.json g3 lenient_macro_f1.high | 0.955079 | PASS |
| 886 | results, line 177 (pqa), number 1 | 0.5797 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.estimate | 0.579666 | PASS |
| 887 | results, line 177 (pqa), number 2 | 0.5070 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.low | 0.506978 | PASS |
| 888 | results, line 177 (pqa), number 3 | 0.6509 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.high | 0.650873 | PASS |
| 889 | results, line 177 (pqa), number 4 | 0.7960 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.estimate | 0.796000 | PASS |
| 890 | results, line 177 (pqa), number 5 | 0.7600 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.low | 0.760000 | PASS |
| 891 | results, line 177 (pqa), number 6 | 0.8340 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.high | 0.834000 | PASS |
| 892 | results, line 179 (load), number 1 | 0.0101 | comparison.json g4_load_check max_abs_difference_vs_session7 | 0.010125 | PASS |
| 893 | results, line 179 (load), number 2 | 0.7860 | run_summaries/s7_b3_timing_test_run_summary.json test_per_question_raw.pubmedqa_answer.accuracy | 0.786000 | PASS |
| 894 | results, line 179 (load), number 3 | 0.6710 | run_summaries/s7_b3_timing_test_run_summary.json test_per_question_raw.pubmedqa_answer.log_loss | 0.671000 | PASS |
| 895 | results, line 179 (load), number 4 | 0.0767 | run_summaries/s7_b3_timing_test_run_summary.json test_per_question_raw.pubmedqa_answer.ece | 0.076700 | PASS |
| 896 | Table 11, `prior`, Log loss, estimate | 1.1593 | comparison.json metrics.pubmedqa_answer.prior.log_loss.estimate | 1.159279 | PASS |
| 897 | Table 11, `prior`, Log loss, low | 1.0401 | comparison.json metrics.pubmedqa_answer.prior.log_loss.low | 1.040141 | PASS |
| 898 | Table 11, `prior`, Log loss, high | 1.2920 | comparison.json metrics.pubmedqa_answer.prior.log_loss.high | 1.292050 | PASS |
| 899 | Table 11, `prior`, Brier score, estimate | 0.5855 | comparison.json metrics.pubmedqa_answer.prior.brier.estimate | 0.585499 | PASS |
| 900 | Table 11, `prior`, Brier score, low | 0.5503 | comparison.json metrics.pubmedqa_answer.prior.brier.low | 0.550341 | PASS |
| 901 | Table 11, `prior`, Brier score, high | 0.6208 | comparison.json metrics.pubmedqa_answer.prior.brier.high | 0.620792 | PASS |
| 902 | Table 11, `prior`, ECE, estimate | 0.0646 | comparison.json metrics.pubmedqa_answer.prior.ece.estimate | 0.064627 | PASS |
| 903 | Table 11, `prior`, ECE, low | 0.0226 | comparison.json metrics.pubmedqa_answer.prior.ece.low | 0.022627 | PASS |
| 904 | Table 11, `prior`, ECE, high | 0.1086 | comparison.json metrics.pubmedqa_answer.prior.ece.high | 0.108627 | PASS |
| 905 | Table 11, `prior`, Accuracy, estimate | 0.5520 | comparison.json metrics.pubmedqa_answer.prior.accuracy.estimate | 0.552000 | PASS |
| 906 | Table 11, `prior`, Accuracy, low | 0.5080 | comparison.json metrics.pubmedqa_answer.prior.accuracy.low | 0.508000 | PASS |
| 907 | Table 11, `prior`, Accuracy, high | 0.5940 | comparison.json metrics.pubmedqa_answer.prior.accuracy.high | 0.594000 | PASS |
| 908 | Table 11, `prior`, Macro-F1, estimate | 0.2371 | comparison.json metrics.pubmedqa_answer.prior.macro_f1.estimate | 0.237113 | PASS |
| 909 | Table 11, `prior`, Macro-F1, low | 0.2246 | comparison.json metrics.pubmedqa_answer.prior.macro_f1.low | 0.224580 | PASS |
| 910 | Table 11, `prior`, Macro-F1, high | 0.2484 | comparison.json metrics.pubmedqa_answer.prior.macro_f1.high | 0.248432 | PASS |
| 911 | Table 11, `arm_laya`, Log loss, estimate | 0.7428 | comparison.json metrics.pubmedqa_answer.arm_laya.log_loss.estimate | 0.742808 | PASS |
| 912 | Table 11, `arm_laya`, Log loss, low | 0.6698 | comparison.json metrics.pubmedqa_answer.arm_laya.log_loss.low | 0.669808 | PASS |
| 913 | Table 11, `arm_laya`, Log loss, high | 0.8164 | comparison.json metrics.pubmedqa_answer.arm_laya.log_loss.high | 0.816414 | PASS |
| 914 | Table 11, `arm_laya`, Brier score, estimate | 0.4138 | comparison.json metrics.pubmedqa_answer.arm_laya.brier.estimate | 0.413810 | PASS |
| 915 | Table 11, `arm_laya`, Brier score, low | 0.3686 | comparison.json metrics.pubmedqa_answer.arm_laya.brier.low | 0.368633 | PASS |
| 916 | Table 11, `arm_laya`, Brier score, high | 0.4582 | comparison.json metrics.pubmedqa_answer.arm_laya.brier.high | 0.458223 | PASS |
| 917 | Table 11, `arm_laya`, ECE, estimate | 0.0276 | comparison.json metrics.pubmedqa_answer.arm_laya.ece.estimate | 0.027569 | PASS |
| 918 | Table 11, `arm_laya`, ECE, low | 0.0301 | comparison.json metrics.pubmedqa_answer.arm_laya.ece.low | 0.030068 | PASS |
| 919 | Table 11, `arm_laya`, ECE, high | 0.0818 | comparison.json metrics.pubmedqa_answer.arm_laya.ece.high | 0.081787 | PASS |
| 920 | Table 11, `arm_laya`, Accuracy, estimate | 0.7140 | comparison.json metrics.pubmedqa_answer.arm_laya.accuracy.estimate | 0.714000 | PASS |
| 921 | Table 11, `arm_laya`, Accuracy, low | 0.6720 | comparison.json metrics.pubmedqa_answer.arm_laya.accuracy.low | 0.672000 | PASS |
| 922 | Table 11, `arm_laya`, Accuracy, high | 0.7560 | comparison.json metrics.pubmedqa_answer.arm_laya.accuracy.high | 0.756000 | PASS |
| 923 | Table 11, `arm_laya`, Macro-F1, estimate | 0.5005 | comparison.json metrics.pubmedqa_answer.arm_laya.macro_f1.estimate | 0.500519 | PASS |
| 924 | Table 11, `arm_laya`, Macro-F1, low | 0.4744 | comparison.json metrics.pubmedqa_answer.arm_laya.macro_f1.low | 0.474432 | PASS |
| 925 | Table 11, `arm_laya`, Macro-F1, high | 0.5258 | comparison.json metrics.pubmedqa_answer.arm_laya.macro_f1.high | 0.525804 | PASS |
| 926 | Table 11, `s7_a3`, Log loss, estimate | 0.6506 | comparison.json metrics.pubmedqa_answer.s7_a3.log_loss.estimate | 0.650607 | PASS |
| 927 | Table 11, `s7_a3`, Log loss, low | 0.5731 | comparison.json metrics.pubmedqa_answer.s7_a3.log_loss.low | 0.573118 | PASS |
| 928 | Table 11, `s7_a3`, Log loss, high | 0.7302 | comparison.json metrics.pubmedqa_answer.s7_a3.log_loss.high | 0.730229 | PASS |
| 929 | Table 11, `s7_a3`, Brier score, estimate | 0.3584 | comparison.json metrics.pubmedqa_answer.s7_a3.brier.estimate | 0.358394 | PASS |
| 930 | Table 11, `s7_a3`, Brier score, low | 0.3119 | comparison.json metrics.pubmedqa_answer.s7_a3.brier.low | 0.311913 | PASS |
| 931 | Table 11, `s7_a3`, Brier score, high | 0.4061 | comparison.json metrics.pubmedqa_answer.s7_a3.brier.high | 0.406114 | PASS |
| 932 | Table 11, `s7_a3`, ECE, estimate | 0.0462 | comparison.json metrics.pubmedqa_answer.s7_a3.ece.estimate | 0.046217 | PASS |
| 933 | Table 11, `s7_a3`, ECE, low | 0.0371 | comparison.json metrics.pubmedqa_answer.s7_a3.ece.low | 0.037142 | PASS |
| 934 | Table 11, `s7_a3`, ECE, high | 0.0905 | comparison.json metrics.pubmedqa_answer.s7_a3.ece.high | 0.090492 | PASS |
| 935 | Table 11, `s7_a3`, Accuracy, estimate | 0.7540 | comparison.json metrics.pubmedqa_answer.s7_a3.accuracy.estimate | 0.754000 | PASS |
| 936 | Table 11, `s7_a3`, Accuracy, low | 0.7160 | comparison.json metrics.pubmedqa_answer.s7_a3.accuracy.low | 0.716000 | PASS |
| 937 | Table 11, `s7_a3`, Accuracy, high | 0.7920 | comparison.json metrics.pubmedqa_answer.s7_a3.accuracy.high | 0.792000 | PASS |
| 938 | Table 11, `s7_a3`, Macro-F1, estimate | 0.5293 | comparison.json metrics.pubmedqa_answer.s7_a3.macro_f1.estimate | 0.529344 | PASS |
| 939 | Table 11, `s7_a3`, Macro-F1, low | 0.5049 | comparison.json metrics.pubmedqa_answer.s7_a3.macro_f1.low | 0.504887 | PASS |
| 940 | Table 11, `s7_a3`, Macro-F1, high | 0.5514 | comparison.json metrics.pubmedqa_answer.s7_a3.macro_f1.high | 0.551386 | PASS |
| 941 | Table 11, `s7_a3_temperature`, Log loss, estimate | 0.6628 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.log_loss.estimate | 0.662802 | PASS |
| 942 | Table 11, `s7_a3_temperature`, Log loss, low | 0.5880 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.log_loss.low | 0.587997 | PASS |
| 943 | Table 11, `s7_a3_temperature`, Log loss, high | 0.7442 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.log_loss.high | 0.744235 | PASS |
| 944 | Table 11, `s7_a3_temperature`, Brier score, estimate | 0.3654 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.brier.estimate | 0.365436 | PASS |
| 945 | Table 11, `s7_a3_temperature`, Brier score, low | 0.3184 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.brier.low | 0.318359 | PASS |
| 946 | Table 11, `s7_a3_temperature`, Brier score, high | 0.4145 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.brier.high | 0.414505 | PASS |
| 947 | Table 11, `s7_a3_temperature`, ECE, estimate | 0.0557 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.ece.estimate | 0.055699 | PASS |
| 948 | Table 11, `s7_a3_temperature`, ECE, low | 0.0423 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.ece.low | 0.042272 | PASS |
| 949 | Table 11, `s7_a3_temperature`, ECE, high | 0.0983 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.ece.high | 0.098272 | PASS |
| 950 | Table 11, `s7_a3_temperature`, Accuracy, estimate | 0.7520 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.accuracy.estimate | 0.752000 | PASS |
| 951 | Table 11, `s7_a3_temperature`, Accuracy, low | 0.7140 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.accuracy.low | 0.714000 | PASS |
| 952 | Table 11, `s7_a3_temperature`, Accuracy, high | 0.7900 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.accuracy.high | 0.790000 | PASS |
| 953 | Table 11, `s7_a3_temperature`, Macro-F1, estimate | 0.5746 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.macro_f1.estimate | 0.574638 | PASS |
| 954 | Table 11, `s7_a3_temperature`, Macro-F1, low | 0.5274 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.macro_f1.low | 0.527397 | PASS |
| 955 | Table 11, `s7_a3_temperature`, Macro-F1, high | 0.6210 | comparison.json metrics.pubmedqa_answer.s7_a3_temperature.macro_f1.high | 0.620969 | PASS |
| 956 | Table 11, `s7_b3`, Log loss, estimate | 0.5797 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.estimate | 0.579666 | PASS |
| 957 | Table 11, `s7_b3`, Log loss, low | 0.5070 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.low | 0.506978 | PASS |
| 958 | Table 11, `s7_b3`, Log loss, high | 0.6509 | comparison.json metrics.pubmedqa_answer.s7_b3.log_loss.high | 0.650873 | PASS |
| 959 | Table 11, `s7_b3`, Brier score, estimate | 0.3147 | comparison.json metrics.pubmedqa_answer.s7_b3.brier.estimate | 0.314738 | PASS |
| 960 | Table 11, `s7_b3`, Brier score, low | 0.2683 | comparison.json metrics.pubmedqa_answer.s7_b3.brier.low | 0.268273 | PASS |
| 961 | Table 11, `s7_b3`, Brier score, high | 0.3595 | comparison.json metrics.pubmedqa_answer.s7_b3.brier.high | 0.359516 | PASS |
| 962 | Table 11, `s7_b3`, ECE, estimate | 0.0401 | comparison.json metrics.pubmedqa_answer.s7_b3.ece.estimate | 0.040133 | PASS |
| 963 | Table 11, `s7_b3`, ECE, low | 0.0315 | comparison.json metrics.pubmedqa_answer.s7_b3.ece.low | 0.031459 | PASS |
| 964 | Table 11, `s7_b3`, ECE, high | 0.0800 | comparison.json metrics.pubmedqa_answer.s7_b3.ece.high | 0.079987 | PASS |
| 965 | Table 11, `s7_b3`, Accuracy, estimate | 0.7960 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.estimate | 0.796000 | PASS |
| 966 | Table 11, `s7_b3`, Accuracy, low | 0.7600 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.low | 0.760000 | PASS |
| 967 | Table 11, `s7_b3`, Accuracy, high | 0.8340 | comparison.json metrics.pubmedqa_answer.s7_b3.accuracy.high | 0.834000 | PASS |
| 968 | Table 11, `s7_b3`, Macro-F1, estimate | 0.5698 | comparison.json metrics.pubmedqa_answer.s7_b3.macro_f1.estimate | 0.569754 | PASS |
| 969 | Table 11, `s7_b3`, Macro-F1, low | 0.5406 | comparison.json metrics.pubmedqa_answer.s7_b3.macro_f1.low | 0.540643 | PASS |
| 970 | Table 11, `s7_b3`, Macro-F1, high | 0.6039 | comparison.json metrics.pubmedqa_answer.s7_b3.macro_f1.high | 0.603864 | PASS |
| 971 | Table 11, `s7_b3_temperature`, Log loss, estimate | 0.6044 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.log_loss.estimate | 0.604423 | PASS |
| 972 | Table 11, `s7_b3_temperature`, Log loss, low | 0.5336 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.log_loss.low | 0.533633 | PASS |
| 973 | Table 11, `s7_b3_temperature`, Log loss, high | 0.6767 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.log_loss.high | 0.676698 | PASS |
| 974 | Table 11, `s7_b3_temperature`, Brier score, estimate | 0.3277 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.brier.estimate | 0.327721 | PASS |
| 975 | Table 11, `s7_b3_temperature`, Brier score, low | 0.2837 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.brier.low | 0.283709 | PASS |
| 976 | Table 11, `s7_b3_temperature`, Brier score, high | 0.3711 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.brier.high | 0.371100 | PASS |
| 977 | Table 11, `s7_b3_temperature`, ECE, estimate | 0.0401 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.ece.estimate | 0.040142 | PASS |
| 978 | Table 11, `s7_b3_temperature`, ECE, low | 0.0345 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.ece.low | 0.034470 | PASS |
| 979 | Table 11, `s7_b3_temperature`, ECE, high | 0.0820 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.ece.high | 0.081976 | PASS |
| 980 | Table 11, `s7_b3_temperature`, Accuracy, estimate | 0.7860 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.accuracy.estimate | 0.786000 | PASS |
| 981 | Table 11, `s7_b3_temperature`, Accuracy, low | 0.7500 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.accuracy.low | 0.750000 | PASS |
| 982 | Table 11, `s7_b3_temperature`, Accuracy, high | 0.8220 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.accuracy.high | 0.822000 | PASS |
| 983 | Table 11, `s7_b3_temperature`, Macro-F1, estimate | 0.6223 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.macro_f1.estimate | 0.622284 | PASS |
| 984 | Table 11, `s7_b3_temperature`, Macro-F1, low | 0.5780 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.macro_f1.low | 0.578015 | PASS |
| 985 | Table 11, `s7_b3_temperature`, Macro-F1, high | 0.6688 | comparison.json metrics.pubmedqa_answer.s7_b3_temperature.macro_f1.high | 0.668815 | PASS |
| 986 | Table 11, `s7_b3-raw`, Log loss, estimate | 0.6710 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.log_loss.estimate | 0.671036 | PASS |
| 987 | Table 11, `s7_b3-raw`, Log loss, low | 0.5587 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.log_loss.low | 0.558663 | PASS |
| 988 | Table 11, `s7_b3-raw`, Log loss, high | 0.7837 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.log_loss.high | 0.783724 | PASS |
| 989 | Table 11, `s7_b3-raw`, Brier score, estimate | 0.3361 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.brier.estimate | 0.336053 | PASS |
| 990 | Table 11, `s7_b3-raw`, Brier score, low | 0.2817 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.brier.low | 0.281712 | PASS |
| 991 | Table 11, `s7_b3-raw`, Brier score, high | 0.3887 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.brier.high | 0.388725 | PASS |
| 992 | Table 11, `s7_b3-raw`, ECE, estimate | 0.0767 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.ece.estimate | 0.076706 | PASS |
| 993 | Table 11, `s7_b3-raw`, ECE, low | 0.0592 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.ece.low | 0.059175 | PASS |
| 994 | Table 11, `s7_b3-raw`, ECE, high | 0.1162 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.ece.high | 0.116203 | PASS |
| 995 | Table 11, `s7_b3-raw`, Accuracy, estimate | 0.7860 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.accuracy.estimate | 0.786000 | PASS |
| 996 | Table 11, `s7_b3-raw`, Accuracy, low | 0.7500 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.accuracy.low | 0.750000 | PASS |
| 997 | Table 11, `s7_b3-raw`, Accuracy, high | 0.8220 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.accuracy.high | 0.822000 | PASS |
| 998 | Table 11, `s7_b3-raw`, Macro-F1, estimate | 0.6223 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.macro_f1.estimate | 0.622284 | PASS |
| 999 | Table 11, `s7_b3-raw`, Macro-F1, low | 0.5780 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.macro_f1.low | 0.578015 | PASS |
| 1000 | Table 11, `s7_b3-raw`, Macro-F1, high | 0.6688 | comparison.json metrics.pubmedqa_answer.s7_b3-raw.macro_f1.high | 0.668815 | PASS |
| 1001 | Table 11, `s8_a3`, Log loss, estimate | 0.6073 | comparison.json metrics.pubmedqa_answer.s8_a3.log_loss.estimate | 0.607278 | PASS |
| 1002 | Table 11, `s8_a3`, Log loss, low | 0.5350 | comparison.json metrics.pubmedqa_answer.s8_a3.log_loss.low | 0.535048 | PASS |
| 1003 | Table 11, `s8_a3`, Log loss, high | 0.6805 | comparison.json metrics.pubmedqa_answer.s8_a3.log_loss.high | 0.680485 | PASS |
| 1004 | Table 11, `s8_a3`, Brier score, estimate | 0.3250 | comparison.json metrics.pubmedqa_answer.s8_a3.brier.estimate | 0.324958 | PASS |
| 1005 | Table 11, `s8_a3`, Brier score, low | 0.2799 | comparison.json metrics.pubmedqa_answer.s8_a3.brier.low | 0.279913 | PASS |
| 1006 | Table 11, `s8_a3`, Brier score, high | 0.3684 | comparison.json metrics.pubmedqa_answer.s8_a3.brier.high | 0.368390 | PASS |
| 1007 | Table 11, `s8_a3`, ECE, estimate | 0.0571 | comparison.json metrics.pubmedqa_answer.s8_a3.ece.estimate | 0.057087 | PASS |
| 1008 | Table 11, `s8_a3`, ECE, low | 0.0448 | comparison.json metrics.pubmedqa_answer.s8_a3.ece.low | 0.044831 | PASS |
| 1009 | Table 11, `s8_a3`, ECE, high | 0.0993 | comparison.json metrics.pubmedqa_answer.s8_a3.ece.high | 0.099261 | PASS |
| 1010 | Table 11, `s8_a3`, Accuracy, estimate | 0.7900 | comparison.json metrics.pubmedqa_answer.s8_a3.accuracy.estimate | 0.790000 | PASS |
| 1011 | Table 11, `s8_a3`, Accuracy, low | 0.7540 | comparison.json metrics.pubmedqa_answer.s8_a3.accuracy.low | 0.754000 | PASS |
| 1012 | Table 11, `s8_a3`, Accuracy, high | 0.8260 | comparison.json metrics.pubmedqa_answer.s8_a3.accuracy.high | 0.826000 | PASS |
| 1013 | Table 11, `s8_a3`, Macro-F1, estimate | 0.5567 | comparison.json metrics.pubmedqa_answer.s8_a3.macro_f1.estimate | 0.556696 | PASS |
| 1014 | Table 11, `s8_a3`, Macro-F1, low | 0.5354 | comparison.json metrics.pubmedqa_answer.s8_a3.macro_f1.low | 0.535438 | PASS |
| 1015 | Table 11, `s8_a3`, Macro-F1, high | 0.5779 | comparison.json metrics.pubmedqa_answer.s8_a3.macro_f1.high | 0.577882 | PASS |
| 1016 | Table 11, `s8_a3_temperature`, Log loss, estimate | 0.6368 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.log_loss.estimate | 0.636850 | PASS |
| 1017 | Table 11, `s8_a3_temperature`, Log loss, low | 0.5731 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.log_loss.low | 0.573108 | PASS |
| 1018 | Table 11, `s8_a3_temperature`, Log loss, high | 0.7028 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.log_loss.high | 0.702816 | PASS |
| 1019 | Table 11, `s8_a3_temperature`, Brier score, estimate | 0.3502 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.brier.estimate | 0.350231 | PASS |
| 1020 | Table 11, `s8_a3_temperature`, Brier score, low | 0.3076 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.brier.low | 0.307584 | PASS |
| 1021 | Table 11, `s8_a3_temperature`, Brier score, high | 0.3923 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.brier.high | 0.392325 | PASS |
| 1022 | Table 11, `s8_a3_temperature`, ECE, estimate | 0.0537 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.ece.estimate | 0.053699 | PASS |
| 1023 | Table 11, `s8_a3_temperature`, ECE, low | 0.0417 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.ece.low | 0.041654 | PASS |
| 1024 | Table 11, `s8_a3_temperature`, ECE, high | 0.0940 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.ece.high | 0.094037 | PASS |
| 1025 | Table 11, `s8_a3_temperature`, Accuracy, estimate | 0.7640 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.accuracy.estimate | 0.764000 | PASS |
| 1026 | Table 11, `s8_a3_temperature`, Accuracy, low | 0.7260 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.accuracy.low | 0.726000 | PASS |
| 1027 | Table 11, `s8_a3_temperature`, Accuracy, high | 0.8040 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.accuracy.high | 0.804000 | PASS |
| 1028 | Table 11, `s8_a3_temperature`, Macro-F1, estimate | 0.6209 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.macro_f1.estimate | 0.620862 | PASS |
| 1029 | Table 11, `s8_a3_temperature`, Macro-F1, low | 0.5796 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.macro_f1.low | 0.579620 | PASS |
| 1030 | Table 11, `s8_a3_temperature`, Macro-F1, high | 0.6651 | comparison.json metrics.pubmedqa_answer.s8_a3_temperature.macro_f1.high | 0.665088 | PASS |
| 1031 | Table 11, `s8_a3-raw`, Log loss, estimate | 0.7526 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.log_loss.estimate | 0.752566 | PASS |
| 1032 | Table 11, `s8_a3-raw`, Log loss, low | 0.6285 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.log_loss.low | 0.628502 | PASS |
| 1033 | Table 11, `s8_a3-raw`, Log loss, high | 0.8751 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.log_loss.high | 0.875053 | PASS |
| 1034 | Table 11, `s8_a3-raw`, Brier score, estimate | 0.3696 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.brier.estimate | 0.369558 | PASS |
| 1035 | Table 11, `s8_a3-raw`, Brier score, low | 0.3134 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.brier.low | 0.313378 | PASS |
| 1036 | Table 11, `s8_a3-raw`, Brier score, high | 0.4267 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.brier.high | 0.426747 | PASS |
| 1037 | Table 11, `s8_a3-raw`, ECE, estimate | 0.1077 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.ece.estimate | 0.107719 | PASS |
| 1038 | Table 11, `s8_a3-raw`, ECE, low | 0.0852 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.ece.low | 0.085221 | PASS |
| 1039 | Table 11, `s8_a3-raw`, ECE, high | 0.1479 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.ece.high | 0.147927 | PASS |
| 1040 | Table 11, `s8_a3-raw`, Accuracy, estimate | 0.7640 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.accuracy.estimate | 0.764000 | PASS |
| 1041 | Table 11, `s8_a3-raw`, Accuracy, low | 0.7260 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.accuracy.low | 0.726000 | PASS |
| 1042 | Table 11, `s8_a3-raw`, Accuracy, high | 0.8040 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.accuracy.high | 0.804000 | PASS |
| 1043 | Table 11, `s8_a3-raw`, Macro-F1, estimate | 0.6209 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.macro_f1.estimate | 0.620862 | PASS |
| 1044 | Table 11, `s8_a3-raw`, Macro-F1, low | 0.5796 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.macro_f1.low | 0.579620 | PASS |
| 1045 | Table 11, `s8_a3-raw`, Macro-F1, high | 0.6651 | comparison.json metrics.pubmedqa_answer.s8_a3-raw.macro_f1.high | 0.665088 | PASS |
| 1046 | Table 11, `gemma4_31b`, Log loss, estimate | 0.6131 | comparison.json metrics.pubmedqa_answer.gemma4_31b.log_loss.estimate | 0.613082 | PASS |
| 1047 | Table 11, `gemma4_31b`, Log loss, low | 0.5309 | comparison.json metrics.pubmedqa_answer.gemma4_31b.log_loss.low | 0.530868 | PASS |
| 1048 | Table 11, `gemma4_31b`, Log loss, high | 0.6957 | comparison.json metrics.pubmedqa_answer.gemma4_31b.log_loss.high | 0.695734 | PASS |
| 1049 | Table 11, `gemma4_31b`, Brier score, estimate | 0.3209 | comparison.json metrics.pubmedqa_answer.gemma4_31b.brier.estimate | 0.320942 | PASS |
| 1050 | Table 11, `gemma4_31b`, Brier score, low | 0.2728 | comparison.json metrics.pubmedqa_answer.gemma4_31b.brier.low | 0.272804 | PASS |
| 1051 | Table 11, `gemma4_31b`, Brier score, high | 0.3696 | comparison.json metrics.pubmedqa_answer.gemma4_31b.brier.high | 0.369646 | PASS |
| 1052 | Table 11, `gemma4_31b`, ECE, estimate | 0.0575 | comparison.json metrics.pubmedqa_answer.gemma4_31b.ece.estimate | 0.057493 | PASS |
| 1053 | Table 11, `gemma4_31b`, ECE, low | 0.0431 | comparison.json metrics.pubmedqa_answer.gemma4_31b.ece.low | 0.043073 | PASS |
| 1054 | Table 11, `gemma4_31b`, ECE, high | 0.0970 | comparison.json metrics.pubmedqa_answer.gemma4_31b.ece.high | 0.096993 | PASS |
| 1055 | Table 11, `gemma4_31b`, Accuracy, estimate | 0.7840 | comparison.json metrics.pubmedqa_answer.gemma4_31b.accuracy.estimate | 0.784000 | PASS |
| 1056 | Table 11, `gemma4_31b`, Accuracy, low | 0.7460 | comparison.json metrics.pubmedqa_answer.gemma4_31b.accuracy.low | 0.745950 | PASS |
| 1057 | Table 11, `gemma4_31b`, Accuracy, high | 0.8200 | comparison.json metrics.pubmedqa_answer.gemma4_31b.accuracy.high | 0.820050 | PASS |
| 1058 | Table 11, `gemma4_31b`, Macro-F1, estimate | 0.5862 | comparison.json metrics.pubmedqa_answer.gemma4_31b.macro_f1.estimate | 0.586196 | PASS |
| 1059 | Table 11, `gemma4_31b`, Macro-F1, low | 0.5527 | comparison.json metrics.pubmedqa_answer.gemma4_31b.macro_f1.low | 0.552652 | PASS |
| 1060 | Table 11, `gemma4_31b`, Macro-F1, high | 0.6252 | comparison.json metrics.pubmedqa_answer.gemma4_31b.macro_f1.high | 0.625214 | PASS |
| 1061 | Table 11, `gemma4_31b_temperature`, Log loss, estimate | 0.6374 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.log_loss.estimate | 0.637449 | PASS |
| 1062 | Table 11, `gemma4_31b_temperature`, Log loss, low | 0.5707 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.log_loss.low | 0.570704 | PASS |
| 1063 | Table 11, `gemma4_31b_temperature`, Log loss, high | 0.7090 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.log_loss.high | 0.708971 | PASS |
| 1064 | Table 11, `gemma4_31b_temperature`, Brier score, estimate | 0.3450 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.brier.estimate | 0.344978 | PASS |
| 1065 | Table 11, `gemma4_31b_temperature`, Brier score, low | 0.2999 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.brier.low | 0.299884 | PASS |
| 1066 | Table 11, `gemma4_31b_temperature`, Brier score, high | 0.3912 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.brier.high | 0.391242 | PASS |
| 1067 | Table 11, `gemma4_31b_temperature`, ECE, estimate | 0.0945 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.ece.estimate | 0.094493 | PASS |
| 1068 | Table 11, `gemma4_31b_temperature`, ECE, low | 0.0709 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.ece.low | 0.070896 | PASS |
| 1069 | Table 11, `gemma4_31b_temperature`, ECE, high | 0.1328 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.ece.high | 0.132795 | PASS |
| 1070 | Table 11, `gemma4_31b_temperature`, Accuracy, estimate | 0.7720 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.accuracy.estimate | 0.772000 | PASS |
| 1071 | Table 11, `gemma4_31b_temperature`, Accuracy, low | 0.7340 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.accuracy.low | 0.734000 | PASS |
| 1072 | Table 11, `gemma4_31b_temperature`, Accuracy, high | 0.8080 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.accuracy.high | 0.808000 | PASS |
| 1073 | Table 11, `gemma4_31b_temperature`, Macro-F1, estimate | 0.6273 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.macro_f1.estimate | 0.627265 | PASS |
| 1074 | Table 11, `gemma4_31b_temperature`, Macro-F1, low | 0.5865 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.macro_f1.low | 0.586496 | PASS |
| 1075 | Table 11, `gemma4_31b_temperature`, Macro-F1, high | 0.6750 | comparison.json metrics.pubmedqa_answer.gemma4_31b_temperature.macro_f1.high | 0.674984 | PASS |
| 1076 | Table 11, `gemma4_31b-raw`, Log loss, estimate | 3.6433 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.log_loss.estimate | 3.643300 | PASS |
| 1077 | Table 11, `gemma4_31b-raw`, Log loss, low | 2.9804 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.log_loss.low | 2.980415 | PASS |
| 1078 | Table 11, `gemma4_31b-raw`, Log loss, high | 4.2976 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.log_loss.high | 4.297614 | PASS |
| 1079 | Table 11, `gemma4_31b-raw`, Brier score, estimate | 0.4488 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.brier.estimate | 0.448821 | PASS |
| 1080 | Table 11, `gemma4_31b-raw`, Brier score, low | 0.3762 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.brier.low | 0.376150 | PASS |
| 1081 | Table 11, `gemma4_31b-raw`, Brier score, high | 0.5225 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.brier.high | 0.522534 | PASS |
| 1082 | Table 11, `gemma4_31b-raw`, ECE, estimate | 0.2226 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.ece.estimate | 0.222589 | PASS |
| 1083 | Table 11, `gemma4_31b-raw`, ECE, low | 0.1880 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.ece.low | 0.188033 | PASS |
| 1084 | Table 11, `gemma4_31b-raw`, ECE, high | 0.2616 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.ece.high | 0.261588 | PASS |
| 1085 | Table 11, `gemma4_31b-raw`, Accuracy, estimate | 0.7720 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.accuracy.estimate | 0.772000 | PASS |
| 1086 | Table 11, `gemma4_31b-raw`, Accuracy, low | 0.7340 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.accuracy.low | 0.734000 | PASS |
| 1087 | Table 11, `gemma4_31b-raw`, Accuracy, high | 0.8080 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.accuracy.high | 0.808000 | PASS |
| 1088 | Table 11, `gemma4_31b-raw`, Macro-F1, estimate | 0.6273 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.macro_f1.estimate | 0.627265 | PASS |
| 1089 | Table 11, `gemma4_31b-raw`, Macro-F1, low | 0.5865 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.macro_f1.low | 0.586496 | PASS |
| 1090 | Table 11, `gemma4_31b-raw`, Macro-F1, high | 0.6750 | comparison.json metrics.pubmedqa_answer.gemma4_31b-raw.macro_f1.high | 0.674984 | PASS |
| 1091 | Table 12, `s7_b3` minus `arm_laya`, Log loss, Difference (95 percent interval), estimate | -0.1631 | comparison.json comparisons[40] (G4, pubmedqa_answer, s7_b3 minus arm_laya, log_loss).difference estimate | -0.163143 | PASS |
| 1092 | Table 12, `s7_b3` minus `arm_laya`, Log loss, Difference (95 percent interval), low | -0.2082 | comparison.json comparisons[40] (G4, pubmedqa_answer, s7_b3 minus arm_laya, log_loss).difference low | -0.208216 | PASS |
| 1093 | Table 12, `s7_b3` minus `arm_laya`, Log loss, Difference (95 percent interval), high | -0.1196 | comparison.json comparisons[40] (G4, pubmedqa_answer, s7_b3 minus arm_laya, log_loss).difference high | -0.119589 | PASS |
| 1094 | Table 12, `s7_b3` minus `arm_laya`, Brier score, Difference (95 percent interval), estimate | -0.0991 | comparison.json comparisons[41] (G4, pubmedqa_answer, s7_b3 minus arm_laya, brier).difference estimate | -0.099072 | PASS |
| 1095 | Table 12, `s7_b3` minus `arm_laya`, Brier score, Difference (95 percent interval), low | -0.1303 | comparison.json comparisons[41] (G4, pubmedqa_answer, s7_b3 minus arm_laya, brier).difference low | -0.130318 | PASS |
| 1096 | Table 12, `s7_b3` minus `arm_laya`, Brier score, Difference (95 percent interval), high | -0.0707 | comparison.json comparisons[41] (G4, pubmedqa_answer, s7_b3 minus arm_laya, brier).difference high | -0.070657 | PASS |
| 1097 | Table 12, `s7_b3` minus `arm_laya`, ECE, Difference (95 percent interval), estimate | 0.0126 | comparison.json comparisons[42] (G4, pubmedqa_answer, s7_b3 minus arm_laya, ece).difference estimate | 0.012565 | PASS |
| 1098 | Table 12, `s7_b3` minus `arm_laya`, ECE, Difference (95 percent interval), low | -0.0340 | comparison.json comparisons[42] (G4, pubmedqa_answer, s7_b3 minus arm_laya, ece).difference low | -0.034033 | PASS |
| 1099 | Table 12, `s7_b3` minus `arm_laya`, ECE, Difference (95 percent interval), high | 0.0320 | comparison.json comparisons[42] (G4, pubmedqa_answer, s7_b3 minus arm_laya, ece).difference high | 0.032030 | PASS |
| 1100 | Table 12, `s7_b3` minus `arm_laya`, Accuracy, Difference (95 percent interval), estimate | 0.0820 | comparison.json comparisons[43] (G4, pubmedqa_answer, s7_b3 minus arm_laya, accuracy).difference estimate | 0.082000 | PASS |
| 1101 | Table 12, `s7_b3` minus `arm_laya`, Accuracy, Difference (95 percent interval), low | 0.0500 | comparison.json comparisons[43] (G4, pubmedqa_answer, s7_b3 minus arm_laya, accuracy).difference low | 0.050000 | PASS |
| 1102 | Table 12, `s7_b3` minus `arm_laya`, Accuracy, Difference (95 percent interval), high | 0.1160 | comparison.json comparisons[43] (G4, pubmedqa_answer, s7_b3 minus arm_laya, accuracy).difference high | 0.116000 | PASS |
| 1103 | Table 12, `s7_b3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), estimate | 0.0692 | comparison.json comparisons[44] (G4, pubmedqa_answer, s7_b3 minus arm_laya, macro_f1).difference estimate | 0.069235 | PASS |
| 1104 | Table 12, `s7_b3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), low | 0.0398 | comparison.json comparisons[44] (G4, pubmedqa_answer, s7_b3 minus arm_laya, macro_f1).difference low | 0.039782 | PASS |
| 1105 | Table 12, `s7_b3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), high | 0.1031 | comparison.json comparisons[44] (G4, pubmedqa_answer, s7_b3 minus arm_laya, macro_f1).difference high | 0.103085 | PASS |
| 1106 | Table 12, `s7_b3` minus `gemma4_31b`, Log loss, Difference (95 percent interval), estimate | -0.0334 | comparison.json comparisons[45] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, log_loss).difference estimate | -0.033417 | PASS |
| 1107 | Table 12, `s7_b3` minus `gemma4_31b`, Log loss, Difference (95 percent interval), low | -0.0785 | comparison.json comparisons[45] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, log_loss).difference low | -0.078544 | PASS |
| 1108 | Table 12, `s7_b3` minus `gemma4_31b`, Log loss, Difference (95 percent interval), high | 0.0137 | comparison.json comparisons[45] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, log_loss).difference high | 0.013697 | PASS |
| 1109 | Table 12, `s7_b3` minus `gemma4_31b`, Brier score, Difference (95 percent interval), estimate | -0.0062 | comparison.json comparisons[46] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, brier).difference estimate | -0.006204 | PASS |
| 1110 | Table 12, `s7_b3` minus `gemma4_31b`, Brier score, Difference (95 percent interval), low | -0.0351 | comparison.json comparisons[46] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, brier).difference low | -0.035092 | PASS |
| 1111 | Table 12, `s7_b3` minus `gemma4_31b`, Brier score, Difference (95 percent interval), high | 0.0209 | comparison.json comparisons[46] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, brier).difference high | 0.020871 | PASS |
| 1112 | Table 12, `s7_b3` minus `gemma4_31b`, ECE, Difference (95 percent interval), estimate | -0.0174 | comparison.json comparisons[47] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, ece).difference estimate | -0.017360 | PASS |
| 1113 | Table 12, `s7_b3` minus `gemma4_31b`, ECE, Difference (95 percent interval), low | -0.0469 | comparison.json comparisons[47] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, ece).difference low | -0.046911 | PASS |
| 1114 | Table 12, `s7_b3` minus `gemma4_31b`, ECE, Difference (95 percent interval), high | 0.0208 | comparison.json comparisons[47] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, ece).difference high | 0.020827 | PASS |
| 1115 | Table 12, `s7_b3` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), estimate | 0.0120 | comparison.json comparisons[48] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, accuracy).difference estimate | 0.012000 | PASS |
| 1116 | Table 12, `s7_b3` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), low | -0.0140 | comparison.json comparisons[48] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, accuracy).difference low | -0.014000 | PASS |
| 1117 | Table 12, `s7_b3` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), high | 0.0400 | comparison.json comparisons[48] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, accuracy).difference high | 0.040000 | PASS |
| 1118 | Table 12, `s7_b3` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), estimate | -0.0164 | comparison.json comparisons[49] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, macro_f1).difference estimate | -0.016442 | PASS |
| 1119 | Table 12, `s7_b3` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), low | -0.0561 | comparison.json comparisons[49] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, macro_f1).difference low | -0.056117 | PASS |
| 1120 | Table 12, `s7_b3` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), high | 0.0237 | comparison.json comparisons[49] (G4, pubmedqa_answer, s7_b3 minus gemma4_31b, macro_f1).difference high | 0.023700 | PASS |
| 1121 | Table 12, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), estimate | -0.0433 | comparison.json comparisons[50] (G4, pubmedqa_answer, s8_a3 minus s7_a3, log_loss).difference estimate | -0.043329 | PASS |
| 1122 | Table 12, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), low | -0.0752 | comparison.json comparisons[50] (G4, pubmedqa_answer, s8_a3 minus s7_a3, log_loss).difference low | -0.075243 | PASS |
| 1123 | Table 12, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), high | -0.0147 | comparison.json comparisons[50] (G4, pubmedqa_answer, s8_a3 minus s7_a3, log_loss).difference high | -0.014748 | PASS |
| 1124 | Table 12, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), estimate | -0.0334 | comparison.json comparisons[51] (G4, pubmedqa_answer, s8_a3 minus s7_a3, brier).difference estimate | -0.033435 | PASS |
| 1125 | Table 12, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), low | -0.0541 | comparison.json comparisons[51] (G4, pubmedqa_answer, s8_a3 minus s7_a3, brier).difference low | -0.054101 | PASS |
| 1126 | Table 12, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), high | -0.0140 | comparison.json comparisons[51] (G4, pubmedqa_answer, s8_a3 minus s7_a3, brier).difference high | -0.013981 | PASS |
| 1127 | Table 12, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), estimate | 0.0109 | comparison.json comparisons[52] (G4, pubmedqa_answer, s8_a3 minus s7_a3, ece).difference estimate | 0.010870 | PASS |
| 1128 | Table 12, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), low | -0.0292 | comparison.json comparisons[52] (G4, pubmedqa_answer, s8_a3 minus s7_a3, ece).difference low | -0.029225 | PASS |
| 1129 | Table 12, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), high | 0.0450 | comparison.json comparisons[52] (G4, pubmedqa_answer, s8_a3 minus s7_a3, ece).difference high | 0.045009 | PASS |
| 1130 | Table 12, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), estimate | 0.0360 | comparison.json comparisons[53] (G4, pubmedqa_answer, s8_a3 minus s7_a3, accuracy).difference estimate | 0.036000 | PASS |
| 1131 | Table 12, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), low | 0.0120 | comparison.json comparisons[53] (G4, pubmedqa_answer, s8_a3 minus s7_a3, accuracy).difference low | 0.012000 | PASS |
| 1132 | Table 12, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), high | 0.0600 | comparison.json comparisons[53] (G4, pubmedqa_answer, s8_a3 minus s7_a3, accuracy).difference high | 0.060000 | PASS |
| 1133 | Table 12, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), estimate | 0.0274 | comparison.json comparisons[54] (G4, pubmedqa_answer, s8_a3 minus s7_a3, macro_f1).difference estimate | 0.027352 | PASS |
| 1134 | Table 12, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), low | 0.0096 | comparison.json comparisons[54] (G4, pubmedqa_answer, s8_a3 minus s7_a3, macro_f1).difference low | 0.009594 | PASS |
| 1135 | Table 12, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), high | 0.0454 | comparison.json comparisons[54] (G4, pubmedqa_answer, s8_a3 minus s7_a3, macro_f1).difference high | 0.045389 | PASS |
| 1136 | Table 12, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), estimate | -0.1355 | comparison.json comparisons[55] (G4, pubmedqa_answer, s8_a3 minus arm_laya, log_loss).difference estimate | -0.135530 | PASS |
| 1137 | Table 12, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), low | -0.1753 | comparison.json comparisons[55] (G4, pubmedqa_answer, s8_a3 minus arm_laya, log_loss).difference low | -0.175294 | PASS |
| 1138 | Table 12, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), high | -0.0905 | comparison.json comparisons[55] (G4, pubmedqa_answer, s8_a3 minus arm_laya, log_loss).difference high | -0.090549 | PASS |
| 1139 | Table 12, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), estimate | -0.0889 | comparison.json comparisons[56] (G4, pubmedqa_answer, s8_a3 minus arm_laya, brier).difference estimate | -0.088852 | PASS |
| 1140 | Table 12, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), low | -0.1165 | comparison.json comparisons[56] (G4, pubmedqa_answer, s8_a3 minus arm_laya, brier).difference low | -0.116534 | PASS |
| 1141 | Table 12, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), high | -0.0597 | comparison.json comparisons[56] (G4, pubmedqa_answer, s8_a3 minus arm_laya, brier).difference high | -0.059688 | PASS |
| 1142 | Table 12, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), estimate | 0.0295 | comparison.json comparisons[57] (G4, pubmedqa_answer, s8_a3 minus arm_laya, ece).difference estimate | 0.029519 | PASS |
| 1143 | Table 12, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), low | -0.0235 | comparison.json comparisons[57] (G4, pubmedqa_answer, s8_a3 minus arm_laya, ece).difference low | -0.023539 | PASS |
| 1144 | Table 12, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), high | 0.0513 | comparison.json comparisons[57] (G4, pubmedqa_answer, s8_a3 minus arm_laya, ece).difference high | 0.051254 | PASS |
| 1145 | Table 12, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), estimate | 0.0760 | comparison.json comparisons[58] (G4, pubmedqa_answer, s8_a3 minus arm_laya, accuracy).difference estimate | 0.076000 | PASS |
| 1146 | Table 12, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), low | 0.0420 | comparison.json comparisons[58] (G4, pubmedqa_answer, s8_a3 minus arm_laya, accuracy).difference low | 0.042000 | PASS |
| 1147 | Table 12, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), high | 0.1061 | comparison.json comparisons[58] (G4, pubmedqa_answer, s8_a3 minus arm_laya, accuracy).difference high | 0.106050 | PASS |
| 1148 | Table 12, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), estimate | 0.0562 | comparison.json comparisons[59] (G4, pubmedqa_answer, s8_a3 minus arm_laya, macro_f1).difference estimate | 0.056178 | PASS |
| 1149 | Table 12, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), low | 0.0317 | comparison.json comparisons[59] (G4, pubmedqa_answer, s8_a3 minus arm_laya, macro_f1).difference low | 0.031748 | PASS |
| 1150 | Table 12, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), high | 0.0790 | comparison.json comparisons[59] (G4, pubmedqa_answer, s8_a3 minus arm_laya, macro_f1).difference high | 0.078999 | PASS |
| 1151 | results, line 226 (readmit), number 1 | 0.8275 | comparison.json metrics.readmit_window.readmission_pool.log_loss.estimate | 0.827515 | PASS |
| 1152 | results, line 226 (readmit), number 2 | 0.8132 | comparison.json metrics.readmit_window.readmission_pool.log_loss.low | 0.813195 | PASS |
| 1153 | results, line 226 (readmit), number 3 | 0.8411 | comparison.json metrics.readmit_window.readmission_pool.log_loss.high | 0.841139 | PASS |
| 1154 | results, line 226 (readmit), number 4 | 0.8347 | comparison.json metrics.readmit_window.lightgbm_tuned.log_loss.estimate | 0.834734 | PASS |
| 1155 | results, line 226 (readmit), number 5 | 0.8340 | comparison.json metrics.readmit_window.blend_v0_1.log_loss.estimate | 0.833995 | PASS |
| 1156 | Table 13, `prior`, Log loss, estimate | 0.8861 | comparison.json metrics.readmit_window.prior.log_loss.estimate | 0.886096 | PASS |
| 1157 | Table 13, `prior`, Log loss, low | 0.8734 | comparison.json metrics.readmit_window.prior.log_loss.low | 0.873382 | PASS |
| 1158 | Table 13, `prior`, Log loss, high | 0.8984 | comparison.json metrics.readmit_window.prior.log_loss.high | 0.898439 | PASS |
| 1159 | Table 13, `prior`, Brier score, estimate | 0.5377 | comparison.json metrics.readmit_window.prior.brier.estimate | 0.537695 | PASS |
| 1160 | Table 13, `prior`, Brier score, low | 0.5298 | comparison.json metrics.readmit_window.prior.brier.low | 0.529761 | PASS |
| 1161 | Table 13, `prior`, Brier score, high | 0.5453 | comparison.json metrics.readmit_window.prior.brier.high | 0.545345 | PASS |
| 1162 | Table 13, `prior`, ECE, estimate | 0.0017 | comparison.json metrics.readmit_window.prior.ece.estimate | 0.001669 | PASS |
| 1163 | Table 13, `prior`, ECE, low | 0.0002 | comparison.json metrics.readmit_window.prior.ece.low | 0.000189 | PASS |
| 1164 | Table 13, `prior`, ECE, high | 0.0127 | comparison.json metrics.readmit_window.prior.ece.high | 0.012675 | PASS |
| 1165 | Table 13, `prior`, Accuracy, estimate | 0.5942 | comparison.json metrics.readmit_window.prior.accuracy.estimate | 0.594170 | PASS |
| 1166 | Table 13, `prior`, Accuracy, low | 0.5829 | comparison.json metrics.readmit_window.prior.accuracy.low | 0.582877 | PASS |
| 1167 | Table 13, `prior`, Accuracy, high | 0.6047 | comparison.json metrics.readmit_window.prior.accuracy.high | 0.604744 | PASS |
| 1168 | Table 13, `prior`, Macro-F1, estimate | 0.2485 | comparison.json metrics.readmit_window.prior.macro_f1.estimate | 0.248476 | PASS |
| 1169 | Table 13, `prior`, Macro-F1, low | 0.2455 | comparison.json metrics.readmit_window.prior.macro_f1.low | 0.245493 | PASS |
| 1170 | Table 13, `prior`, Macro-F1, high | 0.2512 | comparison.json metrics.readmit_window.prior.macro_f1.high | 0.251232 | PASS |
| 1171 | Table 13, `arm_laya`, Log loss, estimate | 0.8524 | comparison.json metrics.readmit_window.arm_laya.log_loss.estimate | 0.852353 | PASS |
| 1172 | Table 13, `arm_laya`, Log loss, low | 0.8384 | comparison.json metrics.readmit_window.arm_laya.log_loss.low | 0.838387 | PASS |
| 1173 | Table 13, `arm_laya`, Log loss, high | 0.8662 | comparison.json metrics.readmit_window.arm_laya.log_loss.high | 0.866234 | PASS |
| 1174 | Table 13, `arm_laya`, Brier score, estimate | 0.5149 | comparison.json metrics.readmit_window.arm_laya.brier.estimate | 0.514852 | PASS |
| 1175 | Table 13, `arm_laya`, Brier score, low | 0.5063 | comparison.json metrics.readmit_window.arm_laya.brier.low | 0.506270 | PASS |
| 1176 | Table 13, `arm_laya`, Brier score, high | 0.5237 | comparison.json metrics.readmit_window.arm_laya.brier.high | 0.523738 | PASS |
| 1177 | Table 13, `arm_laya`, ECE, estimate | 0.0169 | comparison.json metrics.readmit_window.arm_laya.ece.estimate | 0.016885 | PASS |
| 1178 | Table 13, `arm_laya`, ECE, low | 0.0101 | comparison.json metrics.readmit_window.arm_laya.ece.low | 0.010068 | PASS |
| 1179 | Table 13, `arm_laya`, ECE, high | 0.0295 | comparison.json metrics.readmit_window.arm_laya.ece.high | 0.029545 | PASS |
| 1180 | Table 13, `arm_laya`, Accuracy, estimate | 0.6045 | comparison.json metrics.readmit_window.arm_laya.accuracy.estimate | 0.604458 | PASS |
| 1181 | Table 13, `arm_laya`, Accuracy, low | 0.5935 | comparison.json metrics.readmit_window.arm_laya.accuracy.low | 0.593452 | PASS |
| 1182 | Table 13, `arm_laya`, Accuracy, high | 0.6160 | comparison.json metrics.readmit_window.arm_laya.accuracy.high | 0.616033 | PASS |
| 1183 | Table 13, `arm_laya`, Macro-F1, estimate | 0.2934 | comparison.json metrics.readmit_window.arm_laya.macro_f1.estimate | 0.293383 | PASS |
| 1184 | Table 13, `arm_laya`, Macro-F1, low | 0.2863 | comparison.json metrics.readmit_window.arm_laya.macro_f1.low | 0.286323 | PASS |
| 1185 | Table 13, `arm_laya`, Macro-F1, high | 0.3002 | comparison.json metrics.readmit_window.arm_laya.macro_f1.high | 0.300236 | PASS |
| 1186 | Table 13, `lightgbm`, Log loss, estimate | 0.8400 | comparison.json metrics.readmit_window.lightgbm.log_loss.estimate | 0.839976 | PASS |
| 1187 | Table 13, `lightgbm`, Log loss, low | 0.8254 | comparison.json metrics.readmit_window.lightgbm.log_loss.low | 0.825446 | PASS |
| 1188 | Table 13, `lightgbm`, Log loss, high | 0.8538 | comparison.json metrics.readmit_window.lightgbm.log_loss.high | 0.853777 | PASS |
| 1189 | Table 13, `lightgbm`, Brier score, estimate | 0.5059 | comparison.json metrics.readmit_window.lightgbm.brier.estimate | 0.505876 | PASS |
| 1190 | Table 13, `lightgbm`, Brier score, low | 0.4969 | comparison.json metrics.readmit_window.lightgbm.brier.low | 0.496938 | PASS |
| 1191 | Table 13, `lightgbm`, Brier score, high | 0.5148 | comparison.json metrics.readmit_window.lightgbm.brier.high | 0.514770 | PASS |
| 1192 | Table 13, `lightgbm`, ECE, estimate | 0.0066 | comparison.json metrics.readmit_window.lightgbm.ece.estimate | 0.006558 | PASS |
| 1193 | Table 13, `lightgbm`, ECE, low | 0.0071 | comparison.json metrics.readmit_window.lightgbm.ece.low | 0.007077 | PASS |
| 1194 | Table 13, `lightgbm`, ECE, high | 0.0214 | comparison.json metrics.readmit_window.lightgbm.ece.high | 0.021409 | PASS |
| 1195 | Table 13, `lightgbm`, Accuracy, estimate | 0.6096 | comparison.json metrics.readmit_window.lightgbm.accuracy.estimate | 0.609603 | PASS |
| 1196 | Table 13, `lightgbm`, Accuracy, low | 0.5975 | comparison.json metrics.readmit_window.lightgbm.accuracy.low | 0.597456 | PASS |
| 1197 | Table 13, `lightgbm`, Accuracy, high | 0.6216 | comparison.json metrics.readmit_window.lightgbm.accuracy.high | 0.621606 | PASS |
| 1198 | Table 13, `lightgbm`, Macro-F1, estimate | 0.3494 | comparison.json metrics.readmit_window.lightgbm.macro_f1.estimate | 0.349382 | PASS |
| 1199 | Table 13, `lightgbm`, Macro-F1, low | 0.3396 | comparison.json metrics.readmit_window.lightgbm.macro_f1.low | 0.339578 | PASS |
| 1200 | Table 13, `lightgbm`, Macro-F1, high | 0.3593 | comparison.json metrics.readmit_window.lightgbm.macro_f1.high | 0.359309 | PASS |
| 1201 | Table 13, `lightgbm_tuned`, Log loss, estimate | 0.8347 | comparison.json metrics.readmit_window.lightgbm_tuned.log_loss.estimate | 0.834734 | PASS |
| 1202 | Table 13, `lightgbm_tuned`, Log loss, low | 0.8203 | comparison.json metrics.readmit_window.lightgbm_tuned.log_loss.low | 0.820257 | PASS |
| 1203 | Table 13, `lightgbm_tuned`, Log loss, high | 0.8487 | comparison.json metrics.readmit_window.lightgbm_tuned.log_loss.high | 0.848693 | PASS |
| 1204 | Table 13, `lightgbm_tuned`, Brier score, estimate | 0.5026 | comparison.json metrics.readmit_window.lightgbm_tuned.brier.estimate | 0.502580 | PASS |
| 1205 | Table 13, `lightgbm_tuned`, Brier score, low | 0.4937 | comparison.json metrics.readmit_window.lightgbm_tuned.brier.low | 0.493736 | PASS |
| 1206 | Table 13, `lightgbm_tuned`, Brier score, high | 0.5114 | comparison.json metrics.readmit_window.lightgbm_tuned.brier.high | 0.511372 | PASS |
| 1207 | Table 13, `lightgbm_tuned`, ECE, estimate | 0.0071 | comparison.json metrics.readmit_window.lightgbm_tuned.ece.estimate | 0.007108 | PASS |
| 1208 | Table 13, `lightgbm_tuned`, ECE, low | 0.0074 | comparison.json metrics.readmit_window.lightgbm_tuned.ece.low | 0.007421 | PASS |
| 1209 | Table 13, `lightgbm_tuned`, ECE, high | 0.0225 | comparison.json metrics.readmit_window.lightgbm_tuned.ece.high | 0.022507 | PASS |
| 1210 | Table 13, `lightgbm_tuned`, Accuracy, estimate | 0.6157 | comparison.json metrics.readmit_window.lightgbm_tuned.accuracy.estimate | 0.615747 | PASS |
| 1211 | Table 13, `lightgbm_tuned`, Accuracy, low | 0.6043 | comparison.json metrics.readmit_window.lightgbm_tuned.accuracy.low | 0.604316 | PASS |
| 1212 | Table 13, `lightgbm_tuned`, Accuracy, high | 0.6275 | comparison.json metrics.readmit_window.lightgbm_tuned.accuracy.high | 0.627472 | PASS |
| 1213 | Table 13, `lightgbm_tuned`, Macro-F1, estimate | 0.3569 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1.estimate | 0.356902 | PASS |
| 1214 | Table 13, `lightgbm_tuned`, Macro-F1, low | 0.3477 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1.low | 0.347657 | PASS |
| 1215 | Table 13, `lightgbm_tuned`, Macro-F1, high | 0.3660 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1.high | 0.365999 | PASS |
| 1216 | Table 13, `blend_v0_1`, Log loss, estimate | 0.8340 | comparison.json metrics.readmit_window.blend_v0_1.log_loss.estimate | 0.833995 | PASS |
| 1217 | Table 13, `blend_v0_1`, Log loss, low | 0.8197 | comparison.json metrics.readmit_window.blend_v0_1.log_loss.low | 0.819688 | PASS |
| 1218 | Table 13, `blend_v0_1`, Log loss, high | 0.8479 | comparison.json metrics.readmit_window.blend_v0_1.log_loss.high | 0.847930 | PASS |
| 1219 | Table 13, `blend_v0_1`, Brier score, estimate | 0.5021 | comparison.json metrics.readmit_window.blend_v0_1.brier.estimate | 0.502140 | PASS |
| 1220 | Table 13, `blend_v0_1`, Brier score, low | 0.4933 | comparison.json metrics.readmit_window.blend_v0_1.brier.low | 0.493266 | PASS |
| 1221 | Table 13, `blend_v0_1`, Brier score, high | 0.5108 | comparison.json metrics.readmit_window.blend_v0_1.brier.high | 0.510806 | PASS |
| 1222 | Table 13, `blend_v0_1`, ECE, estimate | 0.0074 | comparison.json metrics.readmit_window.blend_v0_1.ece.estimate | 0.007449 | PASS |
| 1223 | Table 13, `blend_v0_1`, ECE, low | 0.0078 | comparison.json metrics.readmit_window.blend_v0_1.ece.low | 0.007772 | PASS |
| 1224 | Table 13, `blend_v0_1`, ECE, high | 0.0226 | comparison.json metrics.readmit_window.blend_v0_1.ece.high | 0.022609 | PASS |
| 1225 | Table 13, `blend_v0_1`, Accuracy, estimate | 0.6116 | comparison.json metrics.readmit_window.blend_v0_1.accuracy.estimate | 0.611603 | PASS |
| 1226 | Table 13, `blend_v0_1`, Accuracy, low | 0.5997 | comparison.json metrics.readmit_window.blend_v0_1.accuracy.low | 0.599739 | PASS |
| 1227 | Table 13, `blend_v0_1`, Accuracy, high | 0.6227 | comparison.json metrics.readmit_window.blend_v0_1.accuracy.high | 0.622749 | PASS |
| 1228 | Table 13, `blend_v0_1`, Macro-F1, estimate | 0.3402 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1.estimate | 0.340193 | PASS |
| 1229 | Table 13, `blend_v0_1`, Macro-F1, low | 0.3316 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1.low | 0.331645 | PASS |
| 1230 | Table 13, `blend_v0_1`, Macro-F1, high | 0.3492 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1.high | 0.349206 | PASS |
| 1231 | Table 13, `xgboost_tuned`, Log loss, estimate | 0.8310 | comparison.json metrics.readmit_window.xgboost_tuned.log_loss.estimate | 0.830997 | PASS |
| 1232 | Table 13, `xgboost_tuned`, Log loss, low | 0.8166 | comparison.json metrics.readmit_window.xgboost_tuned.log_loss.low | 0.816581 | PASS |
| 1233 | Table 13, `xgboost_tuned`, Log loss, high | 0.8448 | comparison.json metrics.readmit_window.xgboost_tuned.log_loss.high | 0.844752 | PASS |
| 1234 | Table 13, `xgboost_tuned`, Brier score, estimate | 0.4997 | comparison.json metrics.readmit_window.xgboost_tuned.brier.estimate | 0.499664 | PASS |
| 1235 | Table 13, `xgboost_tuned`, Brier score, low | 0.4907 | comparison.json metrics.readmit_window.xgboost_tuned.brier.low | 0.490681 | PASS |
| 1236 | Table 13, `xgboost_tuned`, Brier score, high | 0.5085 | comparison.json metrics.readmit_window.xgboost_tuned.brier.high | 0.508485 | PASS |
| 1237 | Table 13, `xgboost_tuned`, ECE, estimate | 0.0103 | comparison.json metrics.readmit_window.xgboost_tuned.ece.estimate | 0.010262 | PASS |
| 1238 | Table 13, `xgboost_tuned`, ECE, low | 0.0081 | comparison.json metrics.readmit_window.xgboost_tuned.ece.low | 0.008106 | PASS |
| 1239 | Table 13, `xgboost_tuned`, ECE, high | 0.0241 | comparison.json metrics.readmit_window.xgboost_tuned.ece.high | 0.024107 | PASS |
| 1240 | Table 13, `xgboost_tuned`, Accuracy, estimate | 0.6139 | comparison.json metrics.readmit_window.xgboost_tuned.accuracy.estimate | 0.613890 | PASS |
| 1241 | Table 13, `xgboost_tuned`, Accuracy, low | 0.6026 | comparison.json metrics.readmit_window.xgboost_tuned.accuracy.low | 0.602601 | PASS |
| 1242 | Table 13, `xgboost_tuned`, Accuracy, high | 0.6250 | comparison.json metrics.readmit_window.xgboost_tuned.accuracy.high | 0.625039 | PASS |
| 1243 | Table 13, `xgboost_tuned`, Macro-F1, estimate | 0.3748 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1.estimate | 0.374791 | PASS |
| 1244 | Table 13, `xgboost_tuned`, Macro-F1, low | 0.3625 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1.low | 0.362505 | PASS |
| 1245 | Table 13, `xgboost_tuned`, Macro-F1, high | 0.3871 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1.high | 0.387050 | PASS |
| 1246 | Table 13, `catboost_tuned`, Log loss, estimate | 0.8291 | comparison.json metrics.readmit_window.catboost_tuned.log_loss.estimate | 0.829101 | PASS |
| 1247 | Table 13, `catboost_tuned`, Log loss, low | 0.8149 | comparison.json metrics.readmit_window.catboost_tuned.log_loss.low | 0.814933 | PASS |
| 1248 | Table 13, `catboost_tuned`, Log loss, high | 0.8429 | comparison.json metrics.readmit_window.catboost_tuned.log_loss.high | 0.842893 | PASS |
| 1249 | Table 13, `catboost_tuned`, Brier score, estimate | 0.4981 | comparison.json metrics.readmit_window.catboost_tuned.brier.estimate | 0.498072 | PASS |
| 1250 | Table 13, `catboost_tuned`, Brier score, low | 0.4889 | comparison.json metrics.readmit_window.catboost_tuned.brier.low | 0.488885 | PASS |
| 1251 | Table 13, `catboost_tuned`, Brier score, high | 0.5072 | comparison.json metrics.readmit_window.catboost_tuned.brier.high | 0.507156 | PASS |
| 1252 | Table 13, `catboost_tuned`, ECE, estimate | 0.0078 | comparison.json metrics.readmit_window.catboost_tuned.ece.estimate | 0.007804 | PASS |
| 1253 | Table 13, `catboost_tuned`, ECE, low | 0.0085 | comparison.json metrics.readmit_window.catboost_tuned.ece.low | 0.008461 | PASS |
| 1254 | Table 13, `catboost_tuned`, ECE, high | 0.0230 | comparison.json metrics.readmit_window.catboost_tuned.ece.high | 0.023026 | PASS |
| 1255 | Table 13, `catboost_tuned`, Accuracy, estimate | 0.6147 | comparison.json metrics.readmit_window.catboost_tuned.accuracy.estimate | 0.614747 | PASS |
| 1256 | Table 13, `catboost_tuned`, Accuracy, low | 0.6035 | comparison.json metrics.readmit_window.catboost_tuned.accuracy.low | 0.603458 | PASS |
| 1257 | Table 13, `catboost_tuned`, Accuracy, high | 0.6262 | comparison.json metrics.readmit_window.catboost_tuned.accuracy.high | 0.626182 | PASS |
| 1258 | Table 13, `catboost_tuned`, Macro-F1, estimate | 0.3671 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1.estimate | 0.367136 | PASS |
| 1259 | Table 13, `catboost_tuned`, Macro-F1, low | 0.3566 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1.low | 0.356577 | PASS |
| 1260 | Table 13, `catboost_tuned`, Macro-F1, high | 0.3781 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1.high | 0.378065 | PASS |
| 1261 | Table 13, `readmission_pool`, Log loss, estimate | 0.8275 | comparison.json metrics.readmit_window.readmission_pool.log_loss.estimate | 0.827515 | PASS |
| 1262 | Table 13, `readmission_pool`, Log loss, low | 0.8132 | comparison.json metrics.readmit_window.readmission_pool.log_loss.low | 0.813195 | PASS |
| 1263 | Table 13, `readmission_pool`, Log loss, high | 0.8411 | comparison.json metrics.readmit_window.readmission_pool.log_loss.high | 0.841139 | PASS |
| 1264 | Table 13, `readmission_pool`, Brier score, estimate | 0.4972 | comparison.json metrics.readmit_window.readmission_pool.brier.estimate | 0.497196 | PASS |
| 1265 | Table 13, `readmission_pool`, Brier score, low | 0.4882 | comparison.json metrics.readmit_window.readmission_pool.brier.low | 0.488222 | PASS |
| 1266 | Table 13, `readmission_pool`, Brier score, high | 0.5062 | comparison.json metrics.readmit_window.readmission_pool.brier.high | 0.506167 | PASS |
| 1267 | Table 13, `readmission_pool`, ECE, estimate | 0.0066 | comparison.json metrics.readmit_window.readmission_pool.ece.estimate | 0.006583 | PASS |
| 1268 | Table 13, `readmission_pool`, ECE, low | 0.0085 | comparison.json metrics.readmit_window.readmission_pool.ece.low | 0.008520 | PASS |
| 1269 | Table 13, `readmission_pool`, ECE, high | 0.0225 | comparison.json metrics.readmit_window.readmission_pool.ece.high | 0.022508 | PASS |
| 1270 | Table 13, `readmission_pool`, Accuracy, estimate | 0.6157 | comparison.json metrics.readmit_window.readmission_pool.accuracy.estimate | 0.615747 | PASS |
| 1271 | Table 13, `readmission_pool`, Accuracy, low | 0.6042 | comparison.json metrics.readmit_window.readmission_pool.accuracy.low | 0.604173 | PASS |
| 1272 | Table 13, `readmission_pool`, Accuracy, high | 0.6269 | comparison.json metrics.readmit_window.readmission_pool.accuracy.high | 0.626897 | PASS |
| 1273 | Table 13, `readmission_pool`, Macro-F1, estimate | 0.3695 | comparison.json metrics.readmit_window.readmission_pool.macro_f1.estimate | 0.369494 | PASS |
| 1274 | Table 13, `readmission_pool`, Macro-F1, low | 0.3579 | comparison.json metrics.readmit_window.readmission_pool.macro_f1.low | 0.357856 | PASS |
| 1275 | Table 13, `readmission_pool`, Macro-F1, high | 0.3813 | comparison.json metrics.readmit_window.readmission_pool.macro_f1.high | 0.381300 | PASS |
| 1276 | Table 13, `s7_a3`, Log loss, estimate | 0.8580 | comparison.json metrics.readmit_window.s7_a3.log_loss.estimate | 0.858028 | PASS |
| 1277 | Table 13, `s7_a3`, Log loss, low | 0.8443 | comparison.json metrics.readmit_window.s7_a3.log_loss.low | 0.844302 | PASS |
| 1278 | Table 13, `s7_a3`, Log loss, high | 0.8713 | comparison.json metrics.readmit_window.s7_a3.log_loss.high | 0.871327 | PASS |
| 1279 | Table 13, `s7_a3`, Brier score, estimate | 0.5184 | comparison.json metrics.readmit_window.s7_a3.brier.estimate | 0.518420 | PASS |
| 1280 | Table 13, `s7_a3`, Brier score, low | 0.5098 | comparison.json metrics.readmit_window.s7_a3.brier.low | 0.509836 | PASS |
| 1281 | Table 13, `s7_a3`, Brier score, high | 0.5273 | comparison.json metrics.readmit_window.s7_a3.brier.high | 0.527323 | PASS |
| 1282 | Table 13, `s7_a3`, ECE, estimate | 0.0254 | comparison.json metrics.readmit_window.s7_a3.ece.estimate | 0.025447 | PASS |
| 1283 | Table 13, `s7_a3`, ECE, low | 0.0188 | comparison.json metrics.readmit_window.s7_a3.ece.low | 0.018753 | PASS |
| 1284 | Table 13, `s7_a3`, ECE, high | 0.0377 | comparison.json metrics.readmit_window.s7_a3.ece.high | 0.037696 | PASS |
| 1285 | Table 13, `s7_a3`, Accuracy, estimate | 0.6012 | comparison.json metrics.readmit_window.s7_a3.accuracy.estimate | 0.601172 | PASS |
| 1286 | Table 13, `s7_a3`, Accuracy, low | 0.5896 | comparison.json metrics.readmit_window.s7_a3.accuracy.low | 0.589597 | PASS |
| 1287 | Table 13, `s7_a3`, Accuracy, high | 0.6130 | comparison.json metrics.readmit_window.s7_a3.accuracy.high | 0.613032 | PASS |
| 1288 | Table 13, `s7_a3`, Macro-F1, estimate | 0.3347 | comparison.json metrics.readmit_window.s7_a3.macro_f1.estimate | 0.334694 | PASS |
| 1289 | Table 13, `s7_a3`, Macro-F1, low | 0.3265 | comparison.json metrics.readmit_window.s7_a3.macro_f1.low | 0.326492 | PASS |
| 1290 | Table 13, `s7_a3`, Macro-F1, high | 0.3430 | comparison.json metrics.readmit_window.s7_a3.macro_f1.high | 0.343002 | PASS |
| 1291 | Table 13, `s7_b3`, Log loss, estimate | 0.8605 | comparison.json metrics.readmit_window.s7_b3.log_loss.estimate | 0.860512 | PASS |
| 1292 | Table 13, `s7_b3`, Log loss, low | 0.8466 | comparison.json metrics.readmit_window.s7_b3.log_loss.low | 0.846621 | PASS |
| 1293 | Table 13, `s7_b3`, Log loss, high | 0.8738 | comparison.json metrics.readmit_window.s7_b3.log_loss.high | 0.873849 | PASS |
| 1294 | Table 13, `s7_b3`, Brier score, estimate | 0.5196 | comparison.json metrics.readmit_window.s7_b3.brier.estimate | 0.519591 | PASS |
| 1295 | Table 13, `s7_b3`, Brier score, low | 0.5110 | comparison.json metrics.readmit_window.s7_b3.brier.low | 0.511013 | PASS |
| 1296 | Table 13, `s7_b3`, Brier score, high | 0.5280 | comparison.json metrics.readmit_window.s7_b3.brier.high | 0.527951 | PASS |
| 1297 | Table 13, `s7_b3`, ECE, estimate | 0.0149 | comparison.json metrics.readmit_window.s7_b3.ece.estimate | 0.014889 | PASS |
| 1298 | Table 13, `s7_b3`, ECE, low | 0.0121 | comparison.json metrics.readmit_window.s7_b3.ece.low | 0.012068 | PASS |
| 1299 | Table 13, `s7_b3`, ECE, high | 0.0299 | comparison.json metrics.readmit_window.s7_b3.ece.high | 0.029902 | PASS |
| 1300 | Table 13, `s7_b3`, Accuracy, estimate | 0.6025 | comparison.json metrics.readmit_window.s7_b3.accuracy.estimate | 0.602458 | PASS |
| 1301 | Table 13, `s7_b3`, Accuracy, low | 0.5907 | comparison.json metrics.readmit_window.s7_b3.accuracy.low | 0.590740 | PASS |
| 1302 | Table 13, `s7_b3`, Accuracy, high | 0.6140 | comparison.json metrics.readmit_window.s7_b3.accuracy.high | 0.614036 | PASS |
| 1303 | Table 13, `s7_b3`, Macro-F1, estimate | 0.3223 | comparison.json metrics.readmit_window.s7_b3.macro_f1.estimate | 0.322324 | PASS |
| 1304 | Table 13, `s7_b3`, Macro-F1, low | 0.3143 | comparison.json metrics.readmit_window.s7_b3.macro_f1.low | 0.314291 | PASS |
| 1305 | Table 13, `s7_b3`, Macro-F1, high | 0.3307 | comparison.json metrics.readmit_window.s7_b3.macro_f1.high | 0.330677 | PASS |
| 1306 | Table 13, `s8_a3`, Log loss, estimate | 0.8445 | comparison.json metrics.readmit_window.s8_a3.log_loss.estimate | 0.844460 | PASS |
| 1307 | Table 13, `s8_a3`, Log loss, low | 0.8304 | comparison.json metrics.readmit_window.s8_a3.log_loss.low | 0.830443 | PASS |
| 1308 | Table 13, `s8_a3`, Log loss, high | 0.8585 | comparison.json metrics.readmit_window.s8_a3.log_loss.high | 0.858490 | PASS |
| 1309 | Table 13, `s8_a3`, Brier score, estimate | 0.5086 | comparison.json metrics.readmit_window.s8_a3.brier.estimate | 0.508631 | PASS |
| 1310 | Table 13, `s8_a3`, Brier score, low | 0.4998 | comparison.json metrics.readmit_window.s8_a3.brier.low | 0.499756 | PASS |
| 1311 | Table 13, `s8_a3`, Brier score, high | 0.5173 | comparison.json metrics.readmit_window.s8_a3.brier.high | 0.517265 | PASS |
| 1312 | Table 13, `s8_a3`, ECE, estimate | 0.0150 | comparison.json metrics.readmit_window.s8_a3.ece.estimate | 0.014986 | PASS |
| 1313 | Table 13, `s8_a3`, ECE, low | 0.0104 | comparison.json metrics.readmit_window.s8_a3.ece.low | 0.010389 | PASS |
| 1314 | Table 13, `s8_a3`, ECE, high | 0.0281 | comparison.json metrics.readmit_window.s8_a3.ece.high | 0.028083 | PASS |
| 1315 | Table 13, `s8_a3`, Accuracy, estimate | 0.6085 | comparison.json metrics.readmit_window.s8_a3.accuracy.estimate | 0.608460 | PASS |
| 1316 | Table 13, `s8_a3`, Accuracy, low | 0.5969 | comparison.json metrics.readmit_window.s8_a3.accuracy.low | 0.596881 | PASS |
| 1317 | Table 13, `s8_a3`, Accuracy, high | 0.6199 | comparison.json metrics.readmit_window.s8_a3.accuracy.high | 0.619895 | PASS |
| 1318 | Table 13, `s8_a3`, Macro-F1, estimate | 0.3499 | comparison.json metrics.readmit_window.s8_a3.macro_f1.estimate | 0.349948 | PASS |
| 1319 | Table 13, `s8_a3`, Macro-F1, low | 0.3406 | comparison.json metrics.readmit_window.s8_a3.macro_f1.low | 0.340551 | PASS |
| 1320 | Table 13, `s8_a3`, Macro-F1, high | 0.3600 | comparison.json metrics.readmit_window.s8_a3.macro_f1.high | 0.360035 | PASS |
| 1321 | Table 13, `s8_a3_temperature`, Log loss, estimate | 0.8441 | comparison.json metrics.readmit_window.s8_a3_temperature.log_loss.estimate | 0.844090 | PASS |
| 1322 | Table 13, `s8_a3_temperature`, Log loss, low | 0.8300 | comparison.json metrics.readmit_window.s8_a3_temperature.log_loss.low | 0.829979 | PASS |
| 1323 | Table 13, `s8_a3_temperature`, Log loss, high | 0.8580 | comparison.json metrics.readmit_window.s8_a3_temperature.log_loss.high | 0.858045 | PASS |
| 1324 | Table 13, `s8_a3_temperature`, Brier score, estimate | 0.5085 | comparison.json metrics.readmit_window.s8_a3_temperature.brier.estimate | 0.508516 | PASS |
| 1325 | Table 13, `s8_a3_temperature`, Brier score, low | 0.4997 | comparison.json metrics.readmit_window.s8_a3_temperature.brier.low | 0.499737 | PASS |
| 1326 | Table 13, `s8_a3_temperature`, Brier score, high | 0.5171 | comparison.json metrics.readmit_window.s8_a3_temperature.brier.high | 0.517055 | PASS |
| 1327 | Table 13, `s8_a3_temperature`, ECE, estimate | 0.0183 | comparison.json metrics.readmit_window.s8_a3_temperature.ece.estimate | 0.018283 | PASS |
| 1328 | Table 13, `s8_a3_temperature`, ECE, low | 0.0110 | comparison.json metrics.readmit_window.s8_a3_temperature.ece.low | 0.010978 | PASS |
| 1329 | Table 13, `s8_a3_temperature`, ECE, high | 0.0305 | comparison.json metrics.readmit_window.s8_a3_temperature.ece.high | 0.030480 | PASS |
| 1330 | Table 13, `s8_a3_temperature`, Accuracy, estimate | 0.6087 | comparison.json metrics.readmit_window.s8_a3_temperature.accuracy.estimate | 0.608745 | PASS |
| 1331 | Table 13, `s8_a3_temperature`, Accuracy, low | 0.5972 | comparison.json metrics.readmit_window.s8_a3_temperature.accuracy.low | 0.597171 | PASS |
| 1332 | Table 13, `s8_a3_temperature`, Accuracy, high | 0.6202 | comparison.json metrics.readmit_window.s8_a3_temperature.accuracy.high | 0.620181 | PASS |
| 1333 | Table 13, `s8_a3_temperature`, Macro-F1, estimate | 0.3465 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1.estimate | 0.346484 | PASS |
| 1334 | Table 13, `s8_a3_temperature`, Macro-F1, low | 0.3379 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1.low | 0.337874 | PASS |
| 1335 | Table 13, `s8_a3_temperature`, Macro-F1, high | 0.3551 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1.high | 0.355125 | PASS |
| 1336 | Table 13, `s8_a3-raw`, Log loss, estimate | 0.8442 | comparison.json metrics.readmit_window.s8_a3-raw.log_loss.estimate | 0.844184 | PASS |
| 1337 | Table 13, `s8_a3-raw`, Log loss, low | 0.8302 | comparison.json metrics.readmit_window.s8_a3-raw.log_loss.low | 0.830164 | PASS |
| 1338 | Table 13, `s8_a3-raw`, Log loss, high | 0.8580 | comparison.json metrics.readmit_window.s8_a3-raw.log_loss.high | 0.858026 | PASS |
| 1339 | Table 13, `s8_a3-raw`, Brier score, estimate | 0.5086 | comparison.json metrics.readmit_window.s8_a3-raw.brier.estimate | 0.508555 | PASS |
| 1340 | Table 13, `s8_a3-raw`, Brier score, low | 0.4999 | comparison.json metrics.readmit_window.s8_a3-raw.brier.low | 0.499851 | PASS |
| 1341 | Table 13, `s8_a3-raw`, Brier score, high | 0.5170 | comparison.json metrics.readmit_window.s8_a3-raw.brier.high | 0.517036 | PASS |
| 1342 | Table 13, `s8_a3-raw`, ECE, estimate | 0.0172 | comparison.json metrics.readmit_window.s8_a3-raw.ece.estimate | 0.017224 | PASS |
| 1343 | Table 13, `s8_a3-raw`, ECE, low | 0.0110 | comparison.json metrics.readmit_window.s8_a3-raw.ece.low | 0.011003 | PASS |
| 1344 | Table 13, `s8_a3-raw`, ECE, high | 0.0298 | comparison.json metrics.readmit_window.s8_a3-raw.ece.high | 0.029754 | PASS |
| 1345 | Table 13, `s8_a3-raw`, Accuracy, estimate | 0.6087 | comparison.json metrics.readmit_window.s8_a3-raw.accuracy.estimate | 0.608745 | PASS |
| 1346 | Table 13, `s8_a3-raw`, Accuracy, low | 0.5972 | comparison.json metrics.readmit_window.s8_a3-raw.accuracy.low | 0.597171 | PASS |
| 1347 | Table 13, `s8_a3-raw`, Accuracy, high | 0.6202 | comparison.json metrics.readmit_window.s8_a3-raw.accuracy.high | 0.620181 | PASS |
| 1348 | Table 13, `s8_a3-raw`, Macro-F1, estimate | 0.3465 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1.estimate | 0.346484 | PASS |
| 1349 | Table 13, `s8_a3-raw`, Macro-F1, low | 0.3379 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1.low | 0.337874 | PASS |
| 1350 | Table 13, `s8_a3-raw`, Macro-F1, high | 0.3551 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1.high | 0.355125 | PASS |
| 1351 | Table 14, `prior`, Macro-F1, decision weights, estimate | 0.2485 | comparison.json metrics.readmit_window.prior.macro_f1_tuned.estimate | 0.248476 | PASS |
| 1352 | Table 14, `prior`, Macro-F1, decision weights, low | 0.2455 | comparison.json metrics.readmit_window.prior.macro_f1_tuned.low | 0.245493 | PASS |
| 1353 | Table 14, `prior`, Macro-F1, decision weights, high | 0.2512 | comparison.json metrics.readmit_window.prior.macro_f1_tuned.high | 0.251232 | PASS |
| 1354 | Table 14, `prior`, Quadratic weighted kappa, estimate | 0.0000 | comparison.json metrics.readmit_window.prior.quadratic_weighted_kappa.estimate | 0.000000 | PASS |
| 1355 | Table 14, `prior`, Quadratic weighted kappa, low | 0.0000 | comparison.json metrics.readmit_window.prior.quadratic_weighted_kappa.low | 0.000000 | PASS |
| 1356 | Table 14, `prior`, Quadratic weighted kappa, high | 0.0000 | comparison.json metrics.readmit_window.prior.quadratic_weighted_kappa.high | 0.000000 | PASS |
| 1357 | Table 14, `prior`, AUROC, 30 days, estimate | 0.5000 | comparison.json metrics.readmit_window.prior.auroc_30d.estimate | 0.500000 | PASS |
| 1358 | Table 14, `prior`, AUROC, 30 days, low | 0.5000 | comparison.json metrics.readmit_window.prior.auroc_30d.low | 0.500000 | PASS |
| 1359 | Table 14, `prior`, AUROC, 30 days, high | 0.5000 | comparison.json metrics.readmit_window.prior.auroc_30d.high | 0.500000 | PASS |
| 1360 | Table 14, `prior`, Calibration intercept, 30 days, estimate | -0.0299 | comparison.json metrics.readmit_window.prior.calibration_intercept_30d.estimate | -0.029918 | PASS |
| 1361 | Table 14, `prior`, Calibration intercept, 30 days, low | -0.1120 | comparison.json metrics.readmit_window.prior.calibration_intercept_30d.low | -0.111998 | PASS |
| 1362 | Table 14, `prior`, Calibration intercept, 30 days, high | 0.0536 | comparison.json metrics.readmit_window.prior.calibration_intercept_30d.high | 0.053650 | PASS |
| 1363 | Table 14, `prior`, Calibration slope, 30 days | not computed | comparison.json metrics.readmit_window.prior.calibration_slope_30d.estimate | null | PASS |
| 1364 | Table 14, `arm_laya`, Macro-F1, decision weights, estimate | 0.4261 | comparison.json metrics.readmit_window.arm_laya.macro_f1_tuned.estimate | 0.426052 | PASS |
| 1365 | Table 14, `arm_laya`, Macro-F1, decision weights, low | 0.4125 | comparison.json metrics.readmit_window.arm_laya.macro_f1_tuned.low | 0.412501 | PASS |
| 1366 | Table 14, `arm_laya`, Macro-F1, decision weights, high | 0.4409 | comparison.json metrics.readmit_window.arm_laya.macro_f1_tuned.high | 0.440872 | PASS |
| 1367 | Table 14, `arm_laya`, Quadratic weighted kappa, estimate | 0.0537 | comparison.json metrics.readmit_window.arm_laya.quadratic_weighted_kappa.estimate | 0.053685 | PASS |
| 1368 | Table 14, `arm_laya`, Quadratic weighted kappa, low | 0.0422 | comparison.json metrics.readmit_window.arm_laya.quadratic_weighted_kappa.low | 0.042249 | PASS |
| 1369 | Table 14, `arm_laya`, Quadratic weighted kappa, high | 0.0657 | comparison.json metrics.readmit_window.arm_laya.quadratic_weighted_kappa.high | 0.065671 | PASS |
| 1370 | Table 14, `arm_laya`, AUROC, 30 days, estimate | 0.6423 | comparison.json metrics.readmit_window.arm_laya.auroc_30d.estimate | 0.642349 | PASS |
| 1371 | Table 14, `arm_laya`, AUROC, 30 days, low | 0.6185 | comparison.json metrics.readmit_window.arm_laya.auroc_30d.low | 0.618451 | PASS |
| 1372 | Table 14, `arm_laya`, AUROC, 30 days, high | 0.6661 | comparison.json metrics.readmit_window.arm_laya.auroc_30d.high | 0.666095 | PASS |
| 1373 | Table 14, `arm_laya`, Calibration intercept, 30 days, estimate | -0.0957 | comparison.json metrics.readmit_window.arm_laya.calibration_intercept_30d.estimate | -0.095655 | PASS |
| 1374 | Table 14, `arm_laya`, Calibration intercept, 30 days, low | -0.1781 | comparison.json metrics.readmit_window.arm_laya.calibration_intercept_30d.low | -0.178132 | PASS |
| 1375 | Table 14, `arm_laya`, Calibration intercept, 30 days, high | -0.0103 | comparison.json metrics.readmit_window.arm_laya.calibration_intercept_30d.high | -0.010297 | PASS |
| 1376 | Table 14, `arm_laya`, Calibration slope, 30 days, estimate | 1.1883 | comparison.json metrics.readmit_window.arm_laya.calibration_slope_30d.estimate | 1.188288 | PASS |
| 1377 | Table 14, `arm_laya`, Calibration slope, 30 days, low | 1.0064 | comparison.json metrics.readmit_window.arm_laya.calibration_slope_30d.low | 1.006405 | PASS |
| 1378 | Table 14, `arm_laya`, Calibration slope, 30 days, high | 1.3756 | comparison.json metrics.readmit_window.arm_laya.calibration_slope_30d.high | 1.375559 | PASS |
| 1379 | Table 14, `lightgbm`, Macro-F1, decision weights, estimate | 0.4417 | comparison.json metrics.readmit_window.lightgbm.macro_f1_tuned.estimate | 0.441682 | PASS |
| 1380 | Table 14, `lightgbm`, Macro-F1, decision weights, low | 0.4285 | comparison.json metrics.readmit_window.lightgbm.macro_f1_tuned.low | 0.428486 | PASS |
| 1381 | Table 14, `lightgbm`, Macro-F1, decision weights, high | 0.4564 | comparison.json metrics.readmit_window.lightgbm.macro_f1_tuned.high | 0.456410 | PASS |
| 1382 | Table 14, `lightgbm`, Quadratic weighted kappa, estimate | 0.1081 | comparison.json metrics.readmit_window.lightgbm.quadratic_weighted_kappa.estimate | 0.108113 | PASS |
| 1383 | Table 14, `lightgbm`, Quadratic weighted kappa, low | 0.0912 | comparison.json metrics.readmit_window.lightgbm.quadratic_weighted_kappa.low | 0.091197 | PASS |
| 1384 | Table 14, `lightgbm`, Quadratic weighted kappa, high | 0.1262 | comparison.json metrics.readmit_window.lightgbm.quadratic_weighted_kappa.high | 0.126239 | PASS |
| 1385 | Table 14, `lightgbm`, AUROC, 30 days, estimate | 0.6469 | comparison.json metrics.readmit_window.lightgbm.auroc_30d.estimate | 0.646855 | PASS |
| 1386 | Table 14, `lightgbm`, AUROC, 30 days, low | 0.6226 | comparison.json metrics.readmit_window.lightgbm.auroc_30d.low | 0.622621 | PASS |
| 1387 | Table 14, `lightgbm`, AUROC, 30 days, high | 0.6715 | comparison.json metrics.readmit_window.lightgbm.auroc_30d.high | 0.671493 | PASS |
| 1388 | Table 14, `lightgbm`, Calibration intercept, 30 days, estimate | -0.0306 | comparison.json metrics.readmit_window.lightgbm.calibration_intercept_30d.estimate | -0.030582 | PASS |
| 1389 | Table 14, `lightgbm`, Calibration intercept, 30 days, low | -0.1171 | comparison.json metrics.readmit_window.lightgbm.calibration_intercept_30d.low | -0.117061 | PASS |
| 1390 | Table 14, `lightgbm`, Calibration intercept, 30 days, high | 0.0549 | comparison.json metrics.readmit_window.lightgbm.calibration_intercept_30d.high | 0.054874 | PASS |
| 1391 | Table 14, `lightgbm`, Calibration slope, 30 days, estimate | 0.9781 | comparison.json metrics.readmit_window.lightgbm.calibration_slope_30d.estimate | 0.978150 | PASS |
| 1392 | Table 14, `lightgbm`, Calibration slope, 30 days, low | 0.8240 | comparison.json metrics.readmit_window.lightgbm.calibration_slope_30d.low | 0.824001 | PASS |
| 1393 | Table 14, `lightgbm`, Calibration slope, 30 days, high | 1.1292 | comparison.json metrics.readmit_window.lightgbm.calibration_slope_30d.high | 1.129168 | PASS |
| 1394 | Table 14, `lightgbm_tuned`, Macro-F1, decision weights, estimate | 0.4484 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1_tuned.estimate | 0.448428 | PASS |
| 1395 | Table 14, `lightgbm_tuned`, Macro-F1, decision weights, low | 0.4335 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1_tuned.low | 0.433529 | PASS |
| 1396 | Table 14, `lightgbm_tuned`, Macro-F1, decision weights, high | 0.4643 | comparison.json metrics.readmit_window.lightgbm_tuned.macro_f1_tuned.high | 0.464257 | PASS |
| 1397 | Table 14, `lightgbm_tuned`, Quadratic weighted kappa, estimate | 0.1200 | comparison.json metrics.readmit_window.lightgbm_tuned.quadratic_weighted_kappa.estimate | 0.119972 | PASS |
| 1398 | Table 14, `lightgbm_tuned`, Quadratic weighted kappa, low | 0.1029 | comparison.json metrics.readmit_window.lightgbm_tuned.quadratic_weighted_kappa.low | 0.102909 | PASS |
| 1399 | Table 14, `lightgbm_tuned`, Quadratic weighted kappa, high | 0.1377 | comparison.json metrics.readmit_window.lightgbm_tuned.quadratic_weighted_kappa.high | 0.137652 | PASS |
| 1400 | Table 14, `lightgbm_tuned`, AUROC, 30 days, estimate | 0.6577 | comparison.json metrics.readmit_window.lightgbm_tuned.auroc_30d.estimate | 0.657741 | PASS |
| 1401 | Table 14, `lightgbm_tuned`, AUROC, 30 days, low | 0.6341 | comparison.json metrics.readmit_window.lightgbm_tuned.auroc_30d.low | 0.634105 | PASS |
| 1402 | Table 14, `lightgbm_tuned`, AUROC, 30 days, high | 0.6807 | comparison.json metrics.readmit_window.lightgbm_tuned.auroc_30d.high | 0.680674 | PASS |
| 1403 | Table 14, `lightgbm_tuned`, Calibration intercept, 30 days, estimate | -0.0177 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_intercept_30d.estimate | -0.017698 | PASS |
| 1404 | Table 14, `lightgbm_tuned`, Calibration intercept, 30 days, low | -0.1011 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_intercept_30d.low | -0.101102 | PASS |
| 1405 | Table 14, `lightgbm_tuned`, Calibration intercept, 30 days, high | 0.0680 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_intercept_30d.high | 0.067960 | PASS |
| 1406 | Table 14, `lightgbm_tuned`, Calibration slope, 30 days, estimate | 1.0935 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_slope_30d.estimate | 1.093528 | PASS |
| 1407 | Table 14, `lightgbm_tuned`, Calibration slope, 30 days, low | 0.9379 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_slope_30d.low | 0.937853 | PASS |
| 1408 | Table 14, `lightgbm_tuned`, Calibration slope, 30 days, high | 1.2504 | comparison.json metrics.readmit_window.lightgbm_tuned.calibration_slope_30d.high | 1.250393 | PASS |
| 1409 | Table 14, `blend_v0_1`, Macro-F1, decision weights, estimate | 0.4520 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1_tuned.estimate | 0.452023 | PASS |
| 1410 | Table 14, `blend_v0_1`, Macro-F1, decision weights, low | 0.4370 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1_tuned.low | 0.437022 | PASS |
| 1411 | Table 14, `blend_v0_1`, Macro-F1, decision weights, high | 0.4678 | comparison.json metrics.readmit_window.blend_v0_1.macro_f1_tuned.high | 0.467818 | PASS |
| 1412 | Table 14, `blend_v0_1`, Quadratic weighted kappa, estimate | 0.0986 | comparison.json metrics.readmit_window.blend_v0_1.quadratic_weighted_kappa.estimate | 0.098636 | PASS |
| 1413 | Table 14, `blend_v0_1`, Quadratic weighted kappa, low | 0.0822 | comparison.json metrics.readmit_window.blend_v0_1.quadratic_weighted_kappa.low | 0.082210 | PASS |
| 1414 | Table 14, `blend_v0_1`, Quadratic weighted kappa, high | 0.1153 | comparison.json metrics.readmit_window.blend_v0_1.quadratic_weighted_kappa.high | 0.115280 | PASS |
| 1415 | Table 14, `blend_v0_1`, AUROC, 30 days, estimate | 0.6612 | comparison.json metrics.readmit_window.blend_v0_1.auroc_30d.estimate | 0.661231 | PASS |
| 1416 | Table 14, `blend_v0_1`, AUROC, 30 days, low | 0.6377 | comparison.json metrics.readmit_window.blend_v0_1.auroc_30d.low | 0.637721 | PASS |
| 1417 | Table 14, `blend_v0_1`, AUROC, 30 days, high | 0.6855 | comparison.json metrics.readmit_window.blend_v0_1.auroc_30d.high | 0.685474 | PASS |
| 1418 | Table 14, `blend_v0_1`, Calibration intercept, 30 days, estimate | -0.0100 | comparison.json metrics.readmit_window.blend_v0_1.calibration_intercept_30d.estimate | -0.009975 | PASS |
| 1419 | Table 14, `blend_v0_1`, Calibration intercept, 30 days, low | -0.0930 | comparison.json metrics.readmit_window.blend_v0_1.calibration_intercept_30d.low | -0.092951 | PASS |
| 1420 | Table 14, `blend_v0_1`, Calibration intercept, 30 days, high | 0.0751 | comparison.json metrics.readmit_window.blend_v0_1.calibration_intercept_30d.high | 0.075126 | PASS |
| 1421 | Table 14, `blend_v0_1`, Calibration slope, 30 days, estimate | 1.1752 | comparison.json metrics.readmit_window.blend_v0_1.calibration_slope_30d.estimate | 1.175202 | PASS |
| 1422 | Table 14, `blend_v0_1`, Calibration slope, 30 days, low | 1.0133 | comparison.json metrics.readmit_window.blend_v0_1.calibration_slope_30d.low | 1.013255 | PASS |
| 1423 | Table 14, `blend_v0_1`, Calibration slope, 30 days, high | 1.3358 | comparison.json metrics.readmit_window.blend_v0_1.calibration_slope_30d.high | 1.335825 | PASS |
| 1424 | Table 14, `xgboost_tuned`, Macro-F1, decision weights, estimate | 0.4551 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1_tuned.estimate | 0.455123 | PASS |
| 1425 | Table 14, `xgboost_tuned`, Macro-F1, decision weights, low | 0.4410 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1_tuned.low | 0.440973 | PASS |
| 1426 | Table 14, `xgboost_tuned`, Macro-F1, decision weights, high | 0.4690 | comparison.json metrics.readmit_window.xgboost_tuned.macro_f1_tuned.high | 0.469032 | PASS |
| 1427 | Table 14, `xgboost_tuned`, Quadratic weighted kappa, estimate | 0.1371 | comparison.json metrics.readmit_window.xgboost_tuned.quadratic_weighted_kappa.estimate | 0.137122 | PASS |
| 1428 | Table 14, `xgboost_tuned`, Quadratic weighted kappa, low | 0.1168 | comparison.json metrics.readmit_window.xgboost_tuned.quadratic_weighted_kappa.low | 0.116834 | PASS |
| 1429 | Table 14, `xgboost_tuned`, Quadratic weighted kappa, high | 0.1583 | comparison.json metrics.readmit_window.xgboost_tuned.quadratic_weighted_kappa.high | 0.158276 | PASS |
| 1430 | Table 14, `xgboost_tuned`, AUROC, 30 days, estimate | 0.6584 | comparison.json metrics.readmit_window.xgboost_tuned.auroc_30d.estimate | 0.658421 | PASS |
| 1431 | Table 14, `xgboost_tuned`, AUROC, 30 days, low | 0.6355 | comparison.json metrics.readmit_window.xgboost_tuned.auroc_30d.low | 0.635458 | PASS |
| 1432 | Table 14, `xgboost_tuned`, AUROC, 30 days, high | 0.6817 | comparison.json metrics.readmit_window.xgboost_tuned.auroc_30d.high | 0.681687 | PASS |
| 1433 | Table 14, `xgboost_tuned`, Calibration intercept, 30 days, estimate | -0.0338 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_intercept_30d.estimate | -0.033816 | PASS |
| 1434 | Table 14, `xgboost_tuned`, Calibration intercept, 30 days, low | -0.1181 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_intercept_30d.low | -0.118058 | PASS |
| 1435 | Table 14, `xgboost_tuned`, Calibration intercept, 30 days, high | 0.0509 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_intercept_30d.high | 0.050939 | PASS |
| 1436 | Table 14, `xgboost_tuned`, Calibration slope, 30 days, estimate | 0.9969 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_slope_30d.estimate | 0.996908 | PASS |
| 1437 | Table 14, `xgboost_tuned`, Calibration slope, 30 days, low | 0.8635 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_slope_30d.low | 0.863492 | PASS |
| 1438 | Table 14, `xgboost_tuned`, Calibration slope, 30 days, high | 1.1430 | comparison.json metrics.readmit_window.xgboost_tuned.calibration_slope_30d.high | 1.142972 | PASS |
| 1439 | Table 14, `catboost_tuned`, Macro-F1, decision weights, estimate | 0.4604 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1_tuned.estimate | 0.460437 | PASS |
| 1440 | Table 14, `catboost_tuned`, Macro-F1, decision weights, low | 0.4469 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1_tuned.low | 0.446874 | PASS |
| 1441 | Table 14, `catboost_tuned`, Macro-F1, decision weights, high | 0.4749 | comparison.json metrics.readmit_window.catboost_tuned.macro_f1_tuned.high | 0.474925 | PASS |
| 1442 | Table 14, `catboost_tuned`, Quadratic weighted kappa, estimate | 0.1274 | comparison.json metrics.readmit_window.catboost_tuned.quadratic_weighted_kappa.estimate | 0.127445 | PASS |
| 1443 | Table 14, `catboost_tuned`, Quadratic weighted kappa, low | 0.1084 | comparison.json metrics.readmit_window.catboost_tuned.quadratic_weighted_kappa.low | 0.108413 | PASS |
| 1444 | Table 14, `catboost_tuned`, Quadratic weighted kappa, high | 0.1459 | comparison.json metrics.readmit_window.catboost_tuned.quadratic_weighted_kappa.high | 0.145904 | PASS |
| 1445 | Table 14, `catboost_tuned`, AUROC, 30 days, estimate | 0.6581 | comparison.json metrics.readmit_window.catboost_tuned.auroc_30d.estimate | 0.658112 | PASS |
| 1446 | Table 14, `catboost_tuned`, AUROC, 30 days, low | 0.6347 | comparison.json metrics.readmit_window.catboost_tuned.auroc_30d.low | 0.634750 | PASS |
| 1447 | Table 14, `catboost_tuned`, AUROC, 30 days, high | 0.6823 | comparison.json metrics.readmit_window.catboost_tuned.auroc_30d.high | 0.682305 | PASS |
| 1448 | Table 14, `catboost_tuned`, Calibration intercept, 30 days, estimate | -0.0287 | comparison.json metrics.readmit_window.catboost_tuned.calibration_intercept_30d.estimate | -0.028683 | PASS |
| 1449 | Table 14, `catboost_tuned`, Calibration intercept, 30 days, low | -0.1120 | comparison.json metrics.readmit_window.catboost_tuned.calibration_intercept_30d.low | -0.111994 | PASS |
| 1450 | Table 14, `catboost_tuned`, Calibration intercept, 30 days, high | 0.0561 | comparison.json metrics.readmit_window.catboost_tuned.calibration_intercept_30d.high | 0.056069 | PASS |
| 1451 | Table 14, `catboost_tuned`, Calibration slope, 30 days, estimate | 1.0160 | comparison.json metrics.readmit_window.catboost_tuned.calibration_slope_30d.estimate | 1.016025 | PASS |
| 1452 | Table 14, `catboost_tuned`, Calibration slope, 30 days, low | 0.8806 | comparison.json metrics.readmit_window.catboost_tuned.calibration_slope_30d.low | 0.880575 | PASS |
| 1453 | Table 14, `catboost_tuned`, Calibration slope, 30 days, high | 1.1574 | comparison.json metrics.readmit_window.catboost_tuned.calibration_slope_30d.high | 1.157419 | PASS |
| 1454 | Table 14, `readmission_pool`, Macro-F1, decision weights, estimate | 0.4632 | comparison.json metrics.readmit_window.readmission_pool.macro_f1_tuned.estimate | 0.463226 | PASS |
| 1455 | Table 14, `readmission_pool`, Macro-F1, decision weights, low | 0.4488 | comparison.json metrics.readmit_window.readmission_pool.macro_f1_tuned.low | 0.448849 | PASS |
| 1456 | Table 14, `readmission_pool`, Macro-F1, decision weights, high | 0.4780 | comparison.json metrics.readmit_window.readmission_pool.macro_f1_tuned.high | 0.478001 | PASS |
| 1457 | Table 14, `readmission_pool`, Quadratic weighted kappa, estimate | 0.1301 | comparison.json metrics.readmit_window.readmission_pool.quadratic_weighted_kappa.estimate | 0.130131 | PASS |
| 1458 | Table 14, `readmission_pool`, Quadratic weighted kappa, low | 0.1101 | comparison.json metrics.readmit_window.readmission_pool.quadratic_weighted_kappa.low | 0.110128 | PASS |
| 1459 | Table 14, `readmission_pool`, Quadratic weighted kappa, high | 0.1491 | comparison.json metrics.readmit_window.readmission_pool.quadratic_weighted_kappa.high | 0.149150 | PASS |
| 1460 | Table 14, `readmission_pool`, AUROC, 30 days, estimate | 0.6622 | comparison.json metrics.readmit_window.readmission_pool.auroc_30d.estimate | 0.662194 | PASS |
| 1461 | Table 14, `readmission_pool`, AUROC, 30 days, low | 0.6394 | comparison.json metrics.readmit_window.readmission_pool.auroc_30d.low | 0.639414 | PASS |
| 1462 | Table 14, `readmission_pool`, AUROC, 30 days, high | 0.6855 | comparison.json metrics.readmit_window.readmission_pool.auroc_30d.high | 0.685497 | PASS |
| 1463 | Table 14, `readmission_pool`, Calibration intercept, 30 days, estimate | -0.0192 | comparison.json metrics.readmit_window.readmission_pool.calibration_intercept_30d.estimate | -0.019197 | PASS |
| 1464 | Table 14, `readmission_pool`, Calibration intercept, 30 days, low | -0.1031 | comparison.json metrics.readmit_window.readmission_pool.calibration_intercept_30d.low | -0.103092 | PASS |
| 1465 | Table 14, `readmission_pool`, Calibration intercept, 30 days, high | 0.0646 | comparison.json metrics.readmit_window.readmission_pool.calibration_intercept_30d.high | 0.064556 | PASS |
| 1466 | Table 14, `readmission_pool`, Calibration slope, 30 days, estimate | 1.0493 | comparison.json metrics.readmit_window.readmission_pool.calibration_slope_30d.estimate | 1.049268 | PASS |
| 1467 | Table 14, `readmission_pool`, Calibration slope, 30 days, low | 0.9108 | comparison.json metrics.readmit_window.readmission_pool.calibration_slope_30d.low | 0.910825 | PASS |
| 1468 | Table 14, `readmission_pool`, Calibration slope, 30 days, high | 1.1940 | comparison.json metrics.readmit_window.readmission_pool.calibration_slope_30d.high | 1.194045 | PASS |
| 1469 | Table 14, `s7_a3`, Macro-F1, decision weights, estimate | 0.4213 | comparison.json metrics.readmit_window.s7_a3.macro_f1_tuned.estimate | 0.421325 | PASS |
| 1470 | Table 14, `s7_a3`, Macro-F1, decision weights, low | 0.4091 | comparison.json metrics.readmit_window.s7_a3.macro_f1_tuned.low | 0.409070 | PASS |
| 1471 | Table 14, `s7_a3`, Macro-F1, decision weights, high | 0.4344 | comparison.json metrics.readmit_window.s7_a3.macro_f1_tuned.high | 0.434405 | PASS |
| 1472 | Table 14, `s7_a3`, Quadratic weighted kappa, estimate | 0.0883 | comparison.json metrics.readmit_window.s7_a3.quadratic_weighted_kappa.estimate | 0.088316 | PASS |
| 1473 | Table 14, `s7_a3`, Quadratic weighted kappa, low | 0.0726 | comparison.json metrics.readmit_window.s7_a3.quadratic_weighted_kappa.low | 0.072619 | PASS |
| 1474 | Table 14, `s7_a3`, Quadratic weighted kappa, high | 0.1064 | comparison.json metrics.readmit_window.s7_a3.quadratic_weighted_kappa.high | 0.106415 | PASS |
| 1475 | Table 14, `s7_a3`, AUROC, 30 days, estimate | 0.6255 | comparison.json metrics.readmit_window.s7_a3.auroc_30d.estimate | 0.625494 | PASS |
| 1476 | Table 14, `s7_a3`, AUROC, 30 days, low | 0.6014 | comparison.json metrics.readmit_window.s7_a3.auroc_30d.low | 0.601438 | PASS |
| 1477 | Table 14, `s7_a3`, AUROC, 30 days, high | 0.6494 | comparison.json metrics.readmit_window.s7_a3.auroc_30d.high | 0.649390 | PASS |
| 1478 | Table 14, `s7_a3`, Calibration intercept, 30 days, estimate | -0.1138 | comparison.json metrics.readmit_window.s7_a3.calibration_intercept_30d.estimate | -0.113798 | PASS |
| 1479 | Table 14, `s7_a3`, Calibration intercept, 30 days, low | -0.1989 | comparison.json metrics.readmit_window.s7_a3.calibration_intercept_30d.low | -0.198860 | PASS |
| 1480 | Table 14, `s7_a3`, Calibration intercept, 30 days, high | -0.0296 | comparison.json metrics.readmit_window.s7_a3.calibration_intercept_30d.high | -0.029595 | PASS |
| 1481 | Table 14, `s7_a3`, Calibration slope, 30 days, estimate | 1.1156 | comparison.json metrics.readmit_window.s7_a3.calibration_slope_30d.estimate | 1.115552 | PASS |
| 1482 | Table 14, `s7_a3`, Calibration slope, 30 days, low | 0.9211 | comparison.json metrics.readmit_window.s7_a3.calibration_slope_30d.low | 0.921055 | PASS |
| 1483 | Table 14, `s7_a3`, Calibration slope, 30 days, high | 1.3149 | comparison.json metrics.readmit_window.s7_a3.calibration_slope_30d.high | 1.314933 | PASS |
| 1484 | Table 14, `s7_b3`, Macro-F1, decision weights, estimate | 0.4183 | comparison.json metrics.readmit_window.s7_b3.macro_f1_tuned.estimate | 0.418275 | PASS |
| 1485 | Table 14, `s7_b3`, Macro-F1, decision weights, low | 0.4054 | comparison.json metrics.readmit_window.s7_b3.macro_f1_tuned.low | 0.405384 | PASS |
| 1486 | Table 14, `s7_b3`, Macro-F1, decision weights, high | 0.4309 | comparison.json metrics.readmit_window.s7_b3.macro_f1_tuned.high | 0.430897 | PASS |
| 1487 | Table 14, `s7_b3`, Quadratic weighted kappa, estimate | 0.0768 | comparison.json metrics.readmit_window.s7_b3.quadratic_weighted_kappa.estimate | 0.076828 | PASS |
| 1488 | Table 14, `s7_b3`, Quadratic weighted kappa, low | 0.0623 | comparison.json metrics.readmit_window.s7_b3.quadratic_weighted_kappa.low | 0.062276 | PASS |
| 1489 | Table 14, `s7_b3`, Quadratic weighted kappa, high | 0.0923 | comparison.json metrics.readmit_window.s7_b3.quadratic_weighted_kappa.high | 0.092287 | PASS |
| 1490 | Table 14, `s7_b3`, AUROC, 30 days, estimate | 0.6293 | comparison.json metrics.readmit_window.s7_b3.auroc_30d.estimate | 0.629336 | PASS |
| 1491 | Table 14, `s7_b3`, AUROC, 30 days, low | 0.6047 | comparison.json metrics.readmit_window.s7_b3.auroc_30d.low | 0.604701 | PASS |
| 1492 | Table 14, `s7_b3`, AUROC, 30 days, high | 0.6549 | comparison.json metrics.readmit_window.s7_b3.auroc_30d.high | 0.654904 | PASS |
| 1493 | Table 14, `s7_b3`, Calibration intercept, 30 days, estimate | -0.1246 | comparison.json metrics.readmit_window.s7_b3.calibration_intercept_30d.estimate | -0.124616 | PASS |
| 1494 | Table 14, `s7_b3`, Calibration intercept, 30 days, low | -0.2100 | comparison.json metrics.readmit_window.s7_b3.calibration_intercept_30d.low | -0.209962 | PASS |
| 1495 | Table 14, `s7_b3`, Calibration intercept, 30 days, high | -0.0378 | comparison.json metrics.readmit_window.s7_b3.calibration_intercept_30d.high | -0.037766 | PASS |
| 1496 | Table 14, `s7_b3`, Calibration slope, 30 days, estimate | 0.7651 | comparison.json metrics.readmit_window.s7_b3.calibration_slope_30d.estimate | 0.765133 | PASS |
| 1497 | Table 14, `s7_b3`, Calibration slope, 30 days, low | 0.6200 | comparison.json metrics.readmit_window.s7_b3.calibration_slope_30d.low | 0.620019 | PASS |
| 1498 | Table 14, `s7_b3`, Calibration slope, 30 days, high | 0.9123 | comparison.json metrics.readmit_window.s7_b3.calibration_slope_30d.high | 0.912280 | PASS |
| 1499 | Table 14, `s8_a3`, Macro-F1, decision weights, estimate | 0.4380 | comparison.json metrics.readmit_window.s8_a3.macro_f1_tuned.estimate | 0.437998 | PASS |
| 1500 | Table 14, `s8_a3`, Macro-F1, decision weights, low | 0.4249 | comparison.json metrics.readmit_window.s8_a3.macro_f1_tuned.low | 0.424922 | PASS |
| 1501 | Table 14, `s8_a3`, Macro-F1, decision weights, high | 0.4518 | comparison.json metrics.readmit_window.s8_a3.macro_f1_tuned.high | 0.451807 | PASS |
| 1502 | Table 14, `s8_a3`, Quadratic weighted kappa, estimate | 0.1120 | comparison.json metrics.readmit_window.s8_a3.quadratic_weighted_kappa.estimate | 0.112016 | PASS |
| 1503 | Table 14, `s8_a3`, Quadratic weighted kappa, low | 0.0950 | comparison.json metrics.readmit_window.s8_a3.quadratic_weighted_kappa.low | 0.095001 | PASS |
| 1504 | Table 14, `s8_a3`, Quadratic weighted kappa, high | 0.1307 | comparison.json metrics.readmit_window.s8_a3.quadratic_weighted_kappa.high | 0.130673 | PASS |
| 1505 | Table 14, `s8_a3`, AUROC, 30 days, estimate | 0.6433 | comparison.json metrics.readmit_window.s8_a3.auroc_30d.estimate | 0.643296 | PASS |
| 1506 | Table 14, `s8_a3`, AUROC, 30 days, low | 0.6198 | comparison.json metrics.readmit_window.s8_a3.auroc_30d.low | 0.619751 | PASS |
| 1507 | Table 14, `s8_a3`, AUROC, 30 days, high | 0.6679 | comparison.json metrics.readmit_window.s8_a3.auroc_30d.high | 0.667855 | PASS |
| 1508 | Table 14, `s8_a3`, Calibration intercept, 30 days, estimate | -0.1174 | comparison.json metrics.readmit_window.s8_a3.calibration_intercept_30d.estimate | -0.117360 | PASS |
| 1509 | Table 14, `s8_a3`, Calibration intercept, 30 days, low | -0.2015 | comparison.json metrics.readmit_window.s8_a3.calibration_intercept_30d.low | -0.201517 | PASS |
| 1510 | Table 14, `s8_a3`, Calibration intercept, 30 days, high | -0.0303 | comparison.json metrics.readmit_window.s8_a3.calibration_intercept_30d.high | -0.030325 | PASS |
| 1511 | Table 14, `s8_a3`, Calibration slope, 30 days, estimate | 0.9539 | comparison.json metrics.readmit_window.s8_a3.calibration_slope_30d.estimate | 0.953888 | PASS |
| 1512 | Table 14, `s8_a3`, Calibration slope, 30 days, low | 0.8097 | comparison.json metrics.readmit_window.s8_a3.calibration_slope_30d.low | 0.809696 | PASS |
| 1513 | Table 14, `s8_a3`, Calibration slope, 30 days, high | 1.0969 | comparison.json metrics.readmit_window.s8_a3.calibration_slope_30d.high | 1.096904 | PASS |
| 1514 | Table 14, `s8_a3_temperature`, Macro-F1, decision weights, estimate | 0.4367 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1_tuned.estimate | 0.436712 | PASS |
| 1515 | Table 14, `s8_a3_temperature`, Macro-F1, decision weights, low | 0.4234 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1_tuned.low | 0.423426 | PASS |
| 1516 | Table 14, `s8_a3_temperature`, Macro-F1, decision weights, high | 0.4511 | comparison.json metrics.readmit_window.s8_a3_temperature.macro_f1_tuned.high | 0.451061 | PASS |
| 1517 | Table 14, `s8_a3_temperature`, Quadratic weighted kappa, estimate | 0.1106 | comparison.json metrics.readmit_window.s8_a3_temperature.quadratic_weighted_kappa.estimate | 0.110601 | PASS |
| 1518 | Table 14, `s8_a3_temperature`, Quadratic weighted kappa, low | 0.0933 | comparison.json metrics.readmit_window.s8_a3_temperature.quadratic_weighted_kappa.low | 0.093298 | PASS |
| 1519 | Table 14, `s8_a3_temperature`, Quadratic weighted kappa, high | 0.1280 | comparison.json metrics.readmit_window.s8_a3_temperature.quadratic_weighted_kappa.high | 0.128043 | PASS |
| 1520 | Table 14, `s8_a3_temperature`, AUROC, 30 days, estimate | 0.6434 | comparison.json metrics.readmit_window.s8_a3_temperature.auroc_30d.estimate | 0.643360 | PASS |
| 1521 | Table 14, `s8_a3_temperature`, AUROC, 30 days, low | 0.6198 | comparison.json metrics.readmit_window.s8_a3_temperature.auroc_30d.low | 0.619800 | PASS |
| 1522 | Table 14, `s8_a3_temperature`, AUROC, 30 days, high | 0.6677 | comparison.json metrics.readmit_window.s8_a3_temperature.auroc_30d.high | 0.667732 | PASS |
| 1523 | Table 14, `s8_a3_temperature`, Calibration intercept, 30 days, estimate | -0.0732 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_intercept_30d.estimate | -0.073191 | PASS |
| 1524 | Table 14, `s8_a3_temperature`, Calibration intercept, 30 days, low | -0.1573 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_intercept_30d.low | -0.157265 | PASS |
| 1525 | Table 14, `s8_a3_temperature`, Calibration intercept, 30 days, high | 0.0134 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_intercept_30d.high | 0.013421 | PASS |
| 1526 | Table 14, `s8_a3_temperature`, Calibration slope, 30 days, estimate | 0.9866 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_slope_30d.estimate | 0.986552 | PASS |
| 1527 | Table 14, `s8_a3_temperature`, Calibration slope, 30 days, low | 0.8376 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_slope_30d.low | 0.837636 | PASS |
| 1528 | Table 14, `s8_a3_temperature`, Calibration slope, 30 days, high | 1.1348 | comparison.json metrics.readmit_window.s8_a3_temperature.calibration_slope_30d.high | 1.134848 | PASS |
| 1529 | Table 14, `s8_a3-raw`, Macro-F1, decision weights, estimate | 0.4367 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1_tuned.estimate | 0.436724 | PASS |
| 1530 | Table 14, `s8_a3-raw`, Macro-F1, decision weights, low | 0.4238 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1_tuned.low | 0.423750 | PASS |
| 1531 | Table 14, `s8_a3-raw`, Macro-F1, decision weights, high | 0.4505 | comparison.json metrics.readmit_window.s8_a3-raw.macro_f1_tuned.high | 0.450507 | PASS |
| 1532 | Table 14, `s8_a3-raw`, Quadratic weighted kappa, estimate | 0.1106 | comparison.json metrics.readmit_window.s8_a3-raw.quadratic_weighted_kappa.estimate | 0.110601 | PASS |
| 1533 | Table 14, `s8_a3-raw`, Quadratic weighted kappa, low | 0.0933 | comparison.json metrics.readmit_window.s8_a3-raw.quadratic_weighted_kappa.low | 0.093298 | PASS |
| 1534 | Table 14, `s8_a3-raw`, Quadratic weighted kappa, high | 0.1280 | comparison.json metrics.readmit_window.s8_a3-raw.quadratic_weighted_kappa.high | 0.128043 | PASS |
| 1535 | Table 14, `s8_a3-raw`, AUROC, 30 days, estimate | 0.6434 | comparison.json metrics.readmit_window.s8_a3-raw.auroc_30d.estimate | 0.643390 | PASS |
| 1536 | Table 14, `s8_a3-raw`, AUROC, 30 days, low | 0.6198 | comparison.json metrics.readmit_window.s8_a3-raw.auroc_30d.low | 0.619827 | PASS |
| 1537 | Table 14, `s8_a3-raw`, AUROC, 30 days, high | 0.6678 | comparison.json metrics.readmit_window.s8_a3-raw.auroc_30d.high | 0.667756 | PASS |
| 1538 | Table 14, `s8_a3-raw`, Calibration intercept, 30 days, estimate | -0.0852 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_intercept_30d.estimate | -0.085242 | PASS |
| 1539 | Table 14, `s8_a3-raw`, Calibration intercept, 30 days, low | -0.1695 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_intercept_30d.low | -0.169458 | PASS |
| 1540 | Table 14, `s8_a3-raw`, Calibration intercept, 30 days, high | 0.0013 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_intercept_30d.high | 0.001263 | PASS |
| 1541 | Table 14, `s8_a3-raw`, Calibration slope, 30 days, estimate | 0.9949 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_slope_30d.estimate | 0.994881 | PASS |
| 1542 | Table 14, `s8_a3-raw`, Calibration slope, 30 days, low | 0.8447 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_slope_30d.low | 0.844725 | PASS |
| 1543 | Table 14, `s8_a3-raw`, Calibration slope, 30 days, high | 1.1445 | comparison.json metrics.readmit_window.s8_a3-raw.calibration_slope_30d.high | 1.144514 | PASS |
| 1544 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), estimate | -0.0072 | comparison.json comparisons[60] (G5, readmit_window, readmission_pool minus lightgbm_tuned, log_loss).difference estimate | -0.007219 | PASS |
| 1545 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), low | -0.0104 | comparison.json comparisons[60] (G5, readmit_window, readmission_pool minus lightgbm_tuned, log_loss).difference low | -0.010408 | PASS |
| 1546 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), high | -0.0042 | comparison.json comparisons[60] (G5, readmit_window, readmission_pool minus lightgbm_tuned, log_loss).difference high | -0.004230 | PASS |
| 1547 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), estimate | -0.0054 | comparison.json comparisons[61] (G5, readmit_window, readmission_pool minus lightgbm_tuned, brier).difference estimate | -0.005384 | PASS |
| 1548 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), low | -0.0075 | comparison.json comparisons[61] (G5, readmit_window, readmission_pool minus lightgbm_tuned, brier).difference low | -0.007468 | PASS |
| 1549 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), high | -0.0034 | comparison.json comparisons[61] (G5, readmit_window, readmission_pool minus lightgbm_tuned, brier).difference high | -0.003442 | PASS |
| 1550 | Table 15, `readmission_pool` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), estimate | -0.0005 | comparison.json comparisons[62] (G5, readmit_window, readmission_pool minus lightgbm_tuned, ece).difference estimate | -0.000525 | PASS |
| 1551 | Table 15, `readmission_pool` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), low | -0.0092 | comparison.json comparisons[62] (G5, readmit_window, readmission_pool minus lightgbm_tuned, ece).difference low | -0.009187 | PASS |
| 1552 | Table 15, `readmission_pool` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), high | 0.0103 | comparison.json comparisons[62] (G5, readmit_window, readmission_pool minus lightgbm_tuned, ece).difference high | 0.010337 | PASS |
| 1553 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), estimate | 0.0000 | comparison.json comparisons[63] (G5, readmit_window, readmission_pool minus lightgbm_tuned, accuracy).difference estimate | 0.000000 | PASS |
| 1554 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), low | -0.0061 | comparison.json comparisons[63] (G5, readmit_window, readmission_pool minus lightgbm_tuned, accuracy).difference low | -0.006148 | PASS |
| 1555 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), high | 0.0059 | comparison.json comparisons[63] (G5, readmit_window, readmission_pool minus lightgbm_tuned, accuracy).difference high | 0.005859 | PASS |
| 1556 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), estimate | 0.0126 | comparison.json comparisons[64] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1).difference estimate | 0.012592 | PASS |
| 1557 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), low | 0.0031 | comparison.json comparisons[64] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1).difference low | 0.003065 | PASS |
| 1558 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), high | 0.0218 | comparison.json comparisons[64] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1).difference high | 0.021767 | PASS |
| 1559 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0148 | comparison.json comparisons[65] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1_tuned).difference estimate | 0.014798 | PASS |
| 1560 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0051 | comparison.json comparisons[65] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1_tuned).difference low | 0.005146 | PASS |
| 1561 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0265 | comparison.json comparisons[65] (G5, readmit_window, readmission_pool minus lightgbm_tuned, macro_f1_tuned).difference high | 0.026524 | PASS |
| 1562 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0102 | comparison.json comparisons[66] (G5, readmit_window, readmission_pool minus lightgbm_tuned, quadratic_weighted_kappa).difference estimate | 0.010159 | PASS |
| 1563 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), low | -0.0048 | comparison.json comparisons[66] (G5, readmit_window, readmission_pool minus lightgbm_tuned, quadratic_weighted_kappa).difference low | -0.004829 | PASS |
| 1564 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0240 | comparison.json comparisons[66] (G5, readmit_window, readmission_pool minus lightgbm_tuned, quadratic_weighted_kappa).difference high | 0.024028 | PASS |
| 1565 | Table 15, `readmission_pool` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0045 | comparison.json comparisons[67] (G5, readmit_window, readmission_pool minus lightgbm_tuned, auroc_30d).difference estimate | 0.004452 | PASS |
| 1566 | Table 15, `readmission_pool` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), low | -0.0063 | comparison.json comparisons[67] (G5, readmit_window, readmission_pool minus lightgbm_tuned, auroc_30d).difference low | -0.006327 | PASS |
| 1567 | Table 15, `readmission_pool` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), high | 0.0160 | comparison.json comparisons[67] (G5, readmit_window, readmission_pool minus lightgbm_tuned, auroc_30d).difference high | 0.015970 | PASS |
| 1568 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0015 | comparison.json comparisons[68] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_intercept_30d).difference estimate | -0.001499 | PASS |
| 1569 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0096 | comparison.json comparisons[68] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_intercept_30d).difference low | -0.009562 | PASS |
| 1570 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), high | 0.0058 | comparison.json comparisons[68] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_intercept_30d).difference high | 0.005844 | PASS |
| 1571 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.0443 | comparison.json comparisons[69] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_slope_30d).difference estimate | -0.044260 | PASS |
| 1572 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.1130 | comparison.json comparisons[69] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_slope_30d).difference low | -0.112996 | PASS |
| 1573 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), high | 0.0250 | comparison.json comparisons[69] (G5, readmit_window, readmission_pool minus lightgbm_tuned, calibration_slope_30d).difference high | 0.025035 | PASS |
| 1574 | Table 15, `readmission_pool` minus `blend_v0_1`, Log loss, Difference (95 percent interval), estimate | -0.0065 | comparison.json comparisons[70] (G5, readmit_window, readmission_pool minus blend_v0_1, log_loss).difference estimate | -0.006481 | PASS |
| 1575 | Table 15, `readmission_pool` minus `blend_v0_1`, Log loss, Difference (95 percent interval), low | -0.0094 | comparison.json comparisons[70] (G5, readmit_window, readmission_pool minus blend_v0_1, log_loss).difference low | -0.009372 | PASS |
| 1576 | Table 15, `readmission_pool` minus `blend_v0_1`, Log loss, Difference (95 percent interval), high | -0.0038 | comparison.json comparisons[70] (G5, readmit_window, readmission_pool minus blend_v0_1, log_loss).difference high | -0.003806 | PASS |
| 1577 | Table 15, `readmission_pool` minus `blend_v0_1`, Brier score, Difference (95 percent interval), estimate | -0.0049 | comparison.json comparisons[71] (G5, readmit_window, readmission_pool minus blend_v0_1, brier).difference estimate | -0.004945 | PASS |
| 1578 | Table 15, `readmission_pool` minus `blend_v0_1`, Brier score, Difference (95 percent interval), low | -0.0069 | comparison.json comparisons[71] (G5, readmit_window, readmission_pool minus blend_v0_1, brier).difference low | -0.006949 | PASS |
| 1579 | Table 15, `readmission_pool` minus `blend_v0_1`, Brier score, Difference (95 percent interval), high | -0.0030 | comparison.json comparisons[71] (G5, readmit_window, readmission_pool minus blend_v0_1, brier).difference high | -0.003013 | PASS |
| 1580 | Table 15, `readmission_pool` minus `blend_v0_1`, ECE, Difference (95 percent interval), estimate | -0.0009 | comparison.json comparisons[72] (G5, readmit_window, readmission_pool minus blend_v0_1, ece).difference estimate | -0.000866 | PASS |
| 1581 | Table 15, `readmission_pool` minus `blend_v0_1`, ECE, Difference (95 percent interval), low | -0.0086 | comparison.json comparisons[72] (G5, readmit_window, readmission_pool minus blend_v0_1, ece).difference low | -0.008594 | PASS |
| 1582 | Table 15, `readmission_pool` minus `blend_v0_1`, ECE, Difference (95 percent interval), high | 0.0093 | comparison.json comparisons[72] (G5, readmit_window, readmission_pool minus blend_v0_1, ece).difference high | 0.009270 | PASS |
| 1583 | Table 15, `readmission_pool` minus `blend_v0_1`, Accuracy, Difference (95 percent interval), estimate | 0.0041 | comparison.json comparisons[73] (G5, readmit_window, readmission_pool minus blend_v0_1, accuracy).difference estimate | 0.004144 | PASS |
| 1584 | Table 15, `readmission_pool` minus `blend_v0_1`, Accuracy, Difference (95 percent interval), low | -0.0014 | comparison.json comparisons[73] (G5, readmit_window, readmission_pool minus blend_v0_1, accuracy).difference low | -0.001433 | PASS |
| 1585 | Table 15, `readmission_pool` minus `blend_v0_1`, Accuracy, Difference (95 percent interval), high | 0.0094 | comparison.json comparisons[73] (G5, readmit_window, readmission_pool minus blend_v0_1, accuracy).difference high | 0.009431 | PASS |
| 1586 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, Difference (95 percent interval), estimate | 0.0293 | comparison.json comparisons[74] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1).difference estimate | 0.029300 | PASS |
| 1587 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, Difference (95 percent interval), low | 0.0199 | comparison.json comparisons[74] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1).difference low | 0.019856 | PASS |
| 1588 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, Difference (95 percent interval), high | 0.0393 | comparison.json comparisons[74] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1).difference high | 0.039255 | PASS |
| 1589 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0112 | comparison.json comparisons[75] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1_tuned).difference estimate | 0.011203 | PASS |
| 1590 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0021 | comparison.json comparisons[75] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1_tuned).difference low | 0.002134 | PASS |
| 1591 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0214 | comparison.json comparisons[75] (G5, readmit_window, readmission_pool minus blend_v0_1, macro_f1_tuned).difference high | 0.021431 | PASS |
| 1592 | Table 15, `readmission_pool` minus `blend_v0_1`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0315 | comparison.json comparisons[76] (G5, readmit_window, readmission_pool minus blend_v0_1, quadratic_weighted_kappa).difference estimate | 0.031496 | PASS |
| 1593 | Table 15, `readmission_pool` minus `blend_v0_1`, Quadratic weighted kappa, Difference (95 percent interval), low | 0.0173 | comparison.json comparisons[76] (G5, readmit_window, readmission_pool minus blend_v0_1, quadratic_weighted_kappa).difference low | 0.017330 | PASS |
| 1594 | Table 15, `readmission_pool` minus `blend_v0_1`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0456 | comparison.json comparisons[76] (G5, readmit_window, readmission_pool minus blend_v0_1, quadratic_weighted_kappa).difference high | 0.045617 | PASS |
| 1595 | Table 15, `readmission_pool` minus `blend_v0_1`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0010 | comparison.json comparisons[77] (G5, readmit_window, readmission_pool minus blend_v0_1, auroc_30d).difference estimate | 0.000963 | PASS |
| 1596 | Table 15, `readmission_pool` minus `blend_v0_1`, AUROC, 30 days, Difference (95 percent interval), low | -0.0087 | comparison.json comparisons[77] (G5, readmit_window, readmission_pool minus blend_v0_1, auroc_30d).difference low | -0.008696 | PASS |
| 1597 | Table 15, `readmission_pool` minus `blend_v0_1`, AUROC, 30 days, Difference (95 percent interval), high | 0.0110 | comparison.json comparisons[77] (G5, readmit_window, readmission_pool minus blend_v0_1, auroc_30d).difference high | 0.011003 | PASS |
| 1598 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0092 | comparison.json comparisons[78] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_intercept_30d).difference estimate | -0.009222 | PASS |
| 1599 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0168 | comparison.json comparisons[78] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_intercept_30d).difference low | -0.016823 | PASS |
| 1600 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration intercept, 30 days, Difference (95 percent interval), high | -0.0028 | comparison.json comparisons[78] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_intercept_30d).difference high | -0.002764 | PASS |
| 1601 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.1259 | comparison.json comparisons[79] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_slope_30d).difference estimate | -0.125934 | PASS |
| 1602 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.1896 | comparison.json comparisons[79] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_slope_30d).difference low | -0.189645 | PASS |
| 1603 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration slope, 30 days, Difference (95 percent interval), high | -0.0606 | comparison.json comparisons[79] (G5, readmit_window, readmission_pool minus blend_v0_1, calibration_slope_30d).difference high | -0.060639 | PASS |
| 1604 | Table 15, `readmission_pool` minus `arm_laya`, Log loss, Difference (95 percent interval), estimate | -0.0248 | comparison.json comparisons[80] (G5, readmit_window, readmission_pool minus arm_laya, log_loss).difference estimate | -0.024838 | PASS |
| 1605 | Table 15, `readmission_pool` minus `arm_laya`, Log loss, Difference (95 percent interval), low | -0.0297 | comparison.json comparisons[80] (G5, readmit_window, readmission_pool minus arm_laya, log_loss).difference low | -0.029679 | PASS |
| 1606 | Table 15, `readmission_pool` minus `arm_laya`, Log loss, Difference (95 percent interval), high | -0.0202 | comparison.json comparisons[80] (G5, readmit_window, readmission_pool minus arm_laya, log_loss).difference high | -0.020228 | PASS |
| 1607 | Table 15, `readmission_pool` minus `arm_laya`, Brier score, Difference (95 percent interval), estimate | -0.0177 | comparison.json comparisons[81] (G5, readmit_window, readmission_pool minus arm_laya, brier).difference estimate | -0.017657 | PASS |
| 1608 | Table 15, `readmission_pool` minus `arm_laya`, Brier score, Difference (95 percent interval), low | -0.0212 | comparison.json comparisons[81] (G5, readmit_window, readmission_pool minus arm_laya, brier).difference low | -0.021211 | PASS |
| 1609 | Table 15, `readmission_pool` minus `arm_laya`, Brier score, Difference (95 percent interval), high | -0.0143 | comparison.json comparisons[81] (G5, readmit_window, readmission_pool minus arm_laya, brier).difference high | -0.014263 | PASS |
| 1610 | Table 15, `readmission_pool` minus `arm_laya`, ECE, Difference (95 percent interval), estimate | -0.0103 | comparison.json comparisons[82] (G5, readmit_window, readmission_pool minus arm_laya, ece).difference estimate | -0.010302 | PASS |
| 1611 | Table 15, `readmission_pool` minus `arm_laya`, ECE, Difference (95 percent interval), low | -0.0158 | comparison.json comparisons[82] (G5, readmit_window, readmission_pool minus arm_laya, ece).difference low | -0.015813 | PASS |
| 1612 | Table 15, `readmission_pool` minus `arm_laya`, ECE, Difference (95 percent interval), high | 0.0077 | comparison.json comparisons[82] (G5, readmit_window, readmission_pool minus arm_laya, ece).difference high | 0.007671 | PASS |
| 1613 | Table 15, `readmission_pool` minus `arm_laya`, Accuracy, Difference (95 percent interval), estimate | 0.0113 | comparison.json comparisons[83] (G5, readmit_window, readmission_pool minus arm_laya, accuracy).difference estimate | 0.011289 | PASS |
| 1614 | Table 15, `readmission_pool` minus `arm_laya`, Accuracy, Difference (95 percent interval), low | 0.0037 | comparison.json comparisons[83] (G5, readmit_window, readmission_pool minus arm_laya, accuracy).difference low | 0.003712 | PASS |
| 1615 | Table 15, `readmission_pool` minus `arm_laya`, Accuracy, Difference (95 percent interval), high | 0.0186 | comparison.json comparisons[83] (G5, readmit_window, readmission_pool minus arm_laya, accuracy).difference high | 0.018577 | PASS |
| 1616 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, Difference (95 percent interval), estimate | 0.0761 | comparison.json comparisons[84] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1).difference estimate | 0.076110 | PASS |
| 1617 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, Difference (95 percent interval), low | 0.0646 | comparison.json comparisons[84] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1).difference low | 0.064631 | PASS |
| 1618 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, Difference (95 percent interval), high | 0.0875 | comparison.json comparisons[84] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1).difference high | 0.087500 | PASS |
| 1619 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0372 | comparison.json comparisons[85] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1_tuned).difference estimate | 0.037174 | PASS |
| 1620 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0248 | comparison.json comparisons[85] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1_tuned).difference low | 0.024814 | PASS |
| 1621 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0494 | comparison.json comparisons[85] (G5, readmit_window, readmission_pool minus arm_laya, macro_f1_tuned).difference high | 0.049362 | PASS |
| 1622 | Table 15, `readmission_pool` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0764 | comparison.json comparisons[86] (G5, readmit_window, readmission_pool minus arm_laya, quadratic_weighted_kappa).difference estimate | 0.076447 | PASS |
| 1623 | Table 15, `readmission_pool` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), low | 0.0583 | comparison.json comparisons[86] (G5, readmit_window, readmission_pool minus arm_laya, quadratic_weighted_kappa).difference low | 0.058299 | PASS |
| 1624 | Table 15, `readmission_pool` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0927 | comparison.json comparisons[86] (G5, readmit_window, readmission_pool minus arm_laya, quadratic_weighted_kappa).difference high | 0.092715 | PASS |
| 1625 | Table 15, `readmission_pool` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0198 | comparison.json comparisons[87] (G5, readmit_window, readmission_pool minus arm_laya, auroc_30d).difference estimate | 0.019845 | PASS |
| 1626 | Table 15, `readmission_pool` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), low | 0.0043 | comparison.json comparisons[87] (G5, readmit_window, readmission_pool minus arm_laya, auroc_30d).difference low | 0.004327 | PASS |
| 1627 | Table 15, `readmission_pool` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), high | 0.0357 | comparison.json comparisons[87] (G5, readmit_window, readmission_pool minus arm_laya, auroc_30d).difference high | 0.035727 | PASS |
| 1628 | Table 15, `readmission_pool` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | 0.0765 | comparison.json comparisons[88] (G5, readmit_window, readmission_pool minus arm_laya, calibration_intercept_30d).difference estimate | 0.076458 | PASS |
| 1629 | Table 15, `readmission_pool` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), low | 0.0662 | comparison.json comparisons[88] (G5, readmit_window, readmission_pool minus arm_laya, calibration_intercept_30d).difference low | 0.066236 | PASS |
| 1630 | Table 15, `readmission_pool` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), high | 0.0852 | comparison.json comparisons[88] (G5, readmit_window, readmission_pool minus arm_laya, calibration_intercept_30d).difference high | 0.085227 | PASS |
| 1631 | Table 15, `readmission_pool` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.1390 | comparison.json comparisons[89] (G5, readmit_window, readmission_pool minus arm_laya, calibration_slope_30d).difference estimate | -0.139020 | PASS |
| 1632 | Table 15, `readmission_pool` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.2524 | comparison.json comparisons[89] (G5, readmit_window, readmission_pool minus arm_laya, calibration_slope_30d).difference low | -0.252428 | PASS |
| 1633 | Table 15, `readmission_pool` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), high | -0.0325 | comparison.json comparisons[89] (G5, readmit_window, readmission_pool minus arm_laya, calibration_slope_30d).difference high | -0.032478 | PASS |
| 1634 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), estimate | -0.0037 | comparison.json comparisons[90] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, log_loss).difference estimate | -0.003737 | PASS |
| 1635 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), low | -0.0069 | comparison.json comparisons[90] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, log_loss).difference low | -0.006851 | PASS |
| 1636 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), high | -0.0008 | comparison.json comparisons[90] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, log_loss).difference high | -0.000779 | PASS |
| 1637 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), estimate | -0.0029 | comparison.json comparisons[91] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, brier).difference estimate | -0.002916 | PASS |
| 1638 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), low | -0.0050 | comparison.json comparisons[91] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, brier).difference low | -0.004998 | PASS |
| 1639 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), high | -0.0010 | comparison.json comparisons[91] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, brier).difference high | -0.000966 | PASS |
| 1640 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), estimate | 0.0032 | comparison.json comparisons[92] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, ece).difference estimate | 0.003154 | PASS |
| 1641 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), low | -0.0084 | comparison.json comparisons[92] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, ece).difference low | -0.008402 | PASS |
| 1642 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), high | 0.0108 | comparison.json comparisons[92] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, ece).difference high | 0.010795 | PASS |
| 1643 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), estimate | -0.0019 | comparison.json comparisons[93] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, accuracy).difference estimate | -0.001858 | PASS |
| 1644 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), low | -0.0079 | comparison.json comparisons[93] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, accuracy).difference low | -0.007859 | PASS |
| 1645 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), high | 0.0034 | comparison.json comparisons[93] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, accuracy).difference high | 0.003430 | PASS |
| 1646 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), estimate | 0.0179 | comparison.json comparisons[94] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1).difference estimate | 0.017889 | PASS |
| 1647 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), low | 0.0083 | comparison.json comparisons[94] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1).difference low | 0.008292 | PASS |
| 1648 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), high | 0.0280 | comparison.json comparisons[94] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1).difference high | 0.027989 | PASS |
| 1649 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0067 | comparison.json comparisons[95] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference estimate | 0.006695 | PASS |
| 1650 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), low | -0.0032 | comparison.json comparisons[95] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference low | -0.003239 | PASS |
| 1651 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0168 | comparison.json comparisons[95] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference high | 0.016841 | PASS |
| 1652 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0171 | comparison.json comparisons[96] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference estimate | 0.017150 | PASS |
| 1653 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), low | 0.0026 | comparison.json comparisons[96] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference low | 0.002650 | PASS |
| 1654 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0307 | comparison.json comparisons[96] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference high | 0.030749 | PASS |
| 1655 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0007 | comparison.json comparisons[97] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, auroc_30d).difference estimate | 0.000680 | PASS |
| 1656 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), low | -0.0102 | comparison.json comparisons[97] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, auroc_30d).difference low | -0.010196 | PASS |
| 1657 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), high | 0.0107 | comparison.json comparisons[97] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, auroc_30d).difference high | 0.010731 | PASS |
| 1658 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0161 | comparison.json comparisons[98] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference estimate | -0.016118 | PASS |
| 1659 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0237 | comparison.json comparisons[98] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference low | -0.023693 | PASS |
| 1660 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), high | -0.0088 | comparison.json comparisons[98] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference high | -0.008839 | PASS |
| 1661 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.0966 | comparison.json comparisons[99] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference estimate | -0.096620 | PASS |
| 1662 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.1635 | comparison.json comparisons[99] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference low | -0.163527 | PASS |
| 1663 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), high | -0.0325 | comparison.json comparisons[99] (G5, readmit_window, xgboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference high | -0.032533 | PASS |
| 1664 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), estimate | -0.0056 | comparison.json comparisons[100] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, log_loss).difference estimate | -0.005633 | PASS |
| 1665 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), low | -0.0095 | comparison.json comparisons[100] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, log_loss).difference low | -0.009453 | PASS |
| 1666 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Log loss, Difference (95 percent interval), high | -0.0020 | comparison.json comparisons[100] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, log_loss).difference high | -0.002036 | PASS |
| 1667 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), estimate | -0.0045 | comparison.json comparisons[101] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, brier).difference estimate | -0.004508 | PASS |
| 1668 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), low | -0.0071 | comparison.json comparisons[101] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, brier).difference low | -0.007135 | PASS |
| 1669 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Brier score, Difference (95 percent interval), high | -0.0021 | comparison.json comparisons[101] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, brier).difference high | -0.002062 | PASS |
| 1670 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), estimate | 0.0007 | comparison.json comparisons[102] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, ece).difference estimate | 0.000695 | PASS |
| 1671 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), low | -0.0096 | comparison.json comparisons[102] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, ece).difference low | -0.009579 | PASS |
| 1672 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, ECE, Difference (95 percent interval), high | 0.0109 | comparison.json comparisons[102] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, ece).difference high | 0.010935 | PASS |
| 1673 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), estimate | -0.0010 | comparison.json comparisons[103] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, accuracy).difference estimate | -0.001000 | PASS |
| 1674 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), low | -0.0077 | comparison.json comparisons[103] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, accuracy).difference low | -0.007716 | PASS |
| 1675 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Accuracy, Difference (95 percent interval), high | 0.0054 | comparison.json comparisons[103] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, accuracy).difference high | 0.005430 | PASS |
| 1676 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), estimate | 0.0102 | comparison.json comparisons[104] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1).difference estimate | 0.010235 | PASS |
| 1677 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), low | 0.0005 | comparison.json comparisons[104] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1).difference low | 0.000520 | PASS |
| 1678 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, Difference (95 percent interval), high | 0.0196 | comparison.json comparisons[104] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1).difference high | 0.019643 | PASS |
| 1679 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0120 | comparison.json comparisons[105] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference estimate | 0.012009 | PASS |
| 1680 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0013 | comparison.json comparisons[105] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference low | 0.001278 | PASS |
| 1681 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0235 | comparison.json comparisons[105] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, macro_f1_tuned).difference high | 0.023490 | PASS |
| 1682 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0075 | comparison.json comparisons[106] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference estimate | 0.007472 | PASS |
| 1683 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), low | -0.0077 | comparison.json comparisons[106] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference low | -0.007720 | PASS |
| 1684 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0223 | comparison.json comparisons[106] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, quadratic_weighted_kappa).difference high | 0.022268 | PASS |
| 1685 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0004 | comparison.json comparisons[107] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, auroc_30d).difference estimate | 0.000371 | PASS |
| 1686 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), low | -0.0124 | comparison.json comparisons[107] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, auroc_30d).difference low | -0.012396 | PASS |
| 1687 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Difference (95 percent interval), high | 0.0140 | comparison.json comparisons[107] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, auroc_30d).difference high | 0.014000 | PASS |
| 1688 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0110 | comparison.json comparisons[108] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference estimate | -0.010986 | PASS |
| 1689 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0202 | comparison.json comparisons[108] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference low | -0.020156 | PASS |
| 1690 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Difference (95 percent interval), high | -0.0019 | comparison.json comparisons[108] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_intercept_30d).difference high | -0.001877 | PASS |
| 1691 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.0775 | comparison.json comparisons[109] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference estimate | -0.077504 | PASS |
| 1692 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.1536 | comparison.json comparisons[109] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference low | -0.153597 | PASS |
| 1693 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Difference (95 percent interval), high | 0.0039 | comparison.json comparisons[109] (G5, readmit_window, catboost_tuned minus lightgbm_tuned, calibration_slope_30d).difference high | 0.003909 | PASS |
| 1694 | Table 15, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), estimate | -0.0136 | comparison.json comparisons[110] (G5, readmit_window, s8_a3 minus s7_a3, log_loss).difference estimate | -0.013568 | PASS |
| 1695 | Table 15, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), low | -0.0177 | comparison.json comparisons[110] (G5, readmit_window, s8_a3 minus s7_a3, log_loss).difference low | -0.017735 | PASS |
| 1696 | Table 15, `s8_a3` minus `s7_a3`, Log loss, Difference (95 percent interval), high | -0.0095 | comparison.json comparisons[110] (G5, readmit_window, s8_a3 minus s7_a3, log_loss).difference high | -0.009538 | PASS |
| 1697 | Table 15, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), estimate | -0.0098 | comparison.json comparisons[111] (G5, readmit_window, s8_a3 minus s7_a3, brier).difference estimate | -0.009789 | PASS |
| 1698 | Table 15, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), low | -0.0128 | comparison.json comparisons[111] (G5, readmit_window, s8_a3 minus s7_a3, brier).difference low | -0.012837 | PASS |
| 1699 | Table 15, `s8_a3` minus `s7_a3`, Brier score, Difference (95 percent interval), high | -0.0070 | comparison.json comparisons[111] (G5, readmit_window, s8_a3 minus s7_a3, brier).difference high | -0.007000 | PASS |
| 1700 | Table 15, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), estimate | -0.0105 | comparison.json comparisons[112] (G5, readmit_window, s8_a3 minus s7_a3, ece).difference estimate | -0.010461 | PASS |
| 1701 | Table 15, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), low | -0.0226 | comparison.json comparisons[112] (G5, readmit_window, s8_a3 minus s7_a3, ece).difference low | -0.022595 | PASS |
| 1702 | Table 15, `s8_a3` minus `s7_a3`, ECE, Difference (95 percent interval), high | 0.0045 | comparison.json comparisons[112] (G5, readmit_window, s8_a3 minus s7_a3, ece).difference high | 0.004477 | PASS |
| 1703 | Table 15, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), estimate | 0.0073 | comparison.json comparisons[113] (G5, readmit_window, s8_a3 minus s7_a3, accuracy).difference estimate | 0.007288 | PASS |
| 1704 | Table 15, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), low | 0.0003 | comparison.json comparisons[113] (G5, readmit_window, s8_a3 minus s7_a3, accuracy).difference low | 0.000286 | PASS |
| 1705 | Table 15, `s8_a3` minus `s7_a3`, Accuracy, Difference (95 percent interval), high | 0.0140 | comparison.json comparisons[113] (G5, readmit_window, s8_a3 minus s7_a3, accuracy).difference high | 0.014004 | PASS |
| 1706 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), estimate | 0.0153 | comparison.json comparisons[114] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1).difference estimate | 0.015254 | PASS |
| 1707 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), low | 0.0065 | comparison.json comparisons[114] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1).difference low | 0.006499 | PASS |
| 1708 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, Difference (95 percent interval), high | 0.0236 | comparison.json comparisons[114] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1).difference high | 0.023564 | PASS |
| 1709 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0167 | comparison.json comparisons[115] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1_tuned).difference estimate | 0.016673 | PASS |
| 1710 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0065 | comparison.json comparisons[115] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1_tuned).difference low | 0.006483 | PASS |
| 1711 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0271 | comparison.json comparisons[115] (G5, readmit_window, s8_a3 minus s7_a3, macro_f1_tuned).difference high | 0.027062 | PASS |
| 1712 | Table 15, `s8_a3` minus `s7_a3`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0237 | comparison.json comparisons[116] (G5, readmit_window, s8_a3 minus s7_a3, quadratic_weighted_kappa).difference estimate | 0.023699 | PASS |
| 1713 | Table 15, `s8_a3` minus `s7_a3`, Quadratic weighted kappa, Difference (95 percent interval), low | 0.0095 | comparison.json comparisons[116] (G5, readmit_window, s8_a3 minus s7_a3, quadratic_weighted_kappa).difference low | 0.009527 | PASS |
| 1714 | Table 15, `s8_a3` minus `s7_a3`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0385 | comparison.json comparisons[116] (G5, readmit_window, s8_a3 minus s7_a3, quadratic_weighted_kappa).difference high | 0.038498 | PASS |
| 1715 | Table 15, `s8_a3` minus `s7_a3`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0178 | comparison.json comparisons[117] (G5, readmit_window, s8_a3 minus s7_a3, auroc_30d).difference estimate | 0.017802 | PASS |
| 1716 | Table 15, `s8_a3` minus `s7_a3`, AUROC, 30 days, Difference (95 percent interval), low | 0.0032 | comparison.json comparisons[117] (G5, readmit_window, s8_a3 minus s7_a3, auroc_30d).difference low | 0.003220 | PASS |
| 1717 | Table 15, `s8_a3` minus `s7_a3`, AUROC, 30 days, Difference (95 percent interval), high | 0.0318 | comparison.json comparisons[117] (G5, readmit_window, s8_a3 minus s7_a3, auroc_30d).difference high | 0.031850 | PASS |
| 1718 | Table 15, `s8_a3` minus `s7_a3`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0036 | comparison.json comparisons[118] (G5, readmit_window, s8_a3 minus s7_a3, calibration_intercept_30d).difference estimate | -0.003562 | PASS |
| 1719 | Table 15, `s8_a3` minus `s7_a3`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0106 | comparison.json comparisons[118] (G5, readmit_window, s8_a3 minus s7_a3, calibration_intercept_30d).difference low | -0.010553 | PASS |
| 1720 | Table 15, `s8_a3` minus `s7_a3`, Calibration intercept, 30 days, Difference (95 percent interval), high | 0.0037 | comparison.json comparisons[118] (G5, readmit_window, s8_a3 minus s7_a3, calibration_intercept_30d).difference high | 0.003707 | PASS |
| 1721 | Table 15, `s8_a3` minus `s7_a3`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.1617 | comparison.json comparisons[119] (G5, readmit_window, s8_a3 minus s7_a3, calibration_slope_30d).difference estimate | -0.161664 | PASS |
| 1722 | Table 15, `s8_a3` minus `s7_a3`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.2544 | comparison.json comparisons[119] (G5, readmit_window, s8_a3 minus s7_a3, calibration_slope_30d).difference low | -0.254420 | PASS |
| 1723 | Table 15, `s8_a3` minus `s7_a3`, Calibration slope, 30 days, Difference (95 percent interval), high | -0.0662 | comparison.json comparisons[119] (G5, readmit_window, s8_a3 minus s7_a3, calibration_slope_30d).difference high | -0.066244 | PASS |
| 1724 | Table 15, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), estimate | -0.0079 | comparison.json comparisons[120] (G5, readmit_window, s8_a3 minus arm_laya, log_loss).difference estimate | -0.007893 | PASS |
| 1725 | Table 15, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), low | -0.0121 | comparison.json comparisons[120] (G5, readmit_window, s8_a3 minus arm_laya, log_loss).difference low | -0.012076 | PASS |
| 1726 | Table 15, `s8_a3` minus `arm_laya`, Log loss, Difference (95 percent interval), high | -0.0043 | comparison.json comparisons[120] (G5, readmit_window, s8_a3 minus arm_laya, log_loss).difference high | -0.004333 | PASS |
| 1727 | Table 15, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), estimate | -0.0062 | comparison.json comparisons[121] (G5, readmit_window, s8_a3 minus arm_laya, brier).difference estimate | -0.006221 | PASS |
| 1728 | Table 15, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), low | -0.0094 | comparison.json comparisons[121] (G5, readmit_window, s8_a3 minus arm_laya, brier).difference low | -0.009412 | PASS |
| 1729 | Table 15, `s8_a3` minus `arm_laya`, Brier score, Difference (95 percent interval), high | -0.0035 | comparison.json comparisons[121] (G5, readmit_window, s8_a3 minus arm_laya, brier).difference high | -0.003464 | PASS |
| 1730 | Table 15, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), estimate | -0.0019 | comparison.json comparisons[122] (G5, readmit_window, s8_a3 minus arm_laya, ece).difference estimate | -0.001899 | PASS |
| 1731 | Table 15, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), low | -0.0151 | comparison.json comparisons[122] (G5, readmit_window, s8_a3 minus arm_laya, ece).difference low | -0.015070 | PASS |
| 1732 | Table 15, `s8_a3` minus `arm_laya`, ECE, Difference (95 percent interval), high | 0.0148 | comparison.json comparisons[122] (G5, readmit_window, s8_a3 minus arm_laya, ece).difference high | 0.014841 | PASS |
| 1733 | Table 15, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), estimate | 0.0040 | comparison.json comparisons[123] (G5, readmit_window, s8_a3 minus arm_laya, accuracy).difference estimate | 0.004001 | PASS |
| 1734 | Table 15, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), low | -0.0031 | comparison.json comparisons[123] (G5, readmit_window, s8_a3 minus arm_laya, accuracy).difference low | -0.003144 | PASS |
| 1735 | Table 15, `s8_a3` minus `arm_laya`, Accuracy, Difference (95 percent interval), high | 0.0110 | comparison.json comparisons[123] (G5, readmit_window, s8_a3 minus arm_laya, accuracy).difference high | 0.011007 | PASS |
| 1736 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), estimate | 0.0566 | comparison.json comparisons[124] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1).difference estimate | 0.056565 | PASS |
| 1737 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), low | 0.0470 | comparison.json comparisons[124] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1).difference low | 0.046969 | PASS |
| 1738 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, Difference (95 percent interval), high | 0.0657 | comparison.json comparisons[124] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1).difference high | 0.065734 | PASS |
| 1739 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), estimate | 0.0119 | comparison.json comparisons[125] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1_tuned).difference estimate | 0.011946 | PASS |
| 1740 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), low | 0.0017 | comparison.json comparisons[125] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1_tuned).difference low | 0.001744 | PASS |
| 1741 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, decision weights, Difference (95 percent interval), high | 0.0225 | comparison.json comparisons[125] (G5, readmit_window, s8_a3 minus arm_laya, macro_f1_tuned).difference high | 0.022477 | PASS |
| 1742 | Table 15, `s8_a3` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), estimate | 0.0583 | comparison.json comparisons[126] (G5, readmit_window, s8_a3 minus arm_laya, quadratic_weighted_kappa).difference estimate | 0.058331 | PASS |
| 1743 | Table 15, `s8_a3` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), low | 0.0432 | comparison.json comparisons[126] (G5, readmit_window, s8_a3 minus arm_laya, quadratic_weighted_kappa).difference low | 0.043196 | PASS |
| 1744 | Table 15, `s8_a3` minus `arm_laya`, Quadratic weighted kappa, Difference (95 percent interval), high | 0.0735 | comparison.json comparisons[126] (G5, readmit_window, s8_a3 minus arm_laya, quadratic_weighted_kappa).difference high | 0.073520 | PASS |
| 1745 | Table 15, `s8_a3` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), estimate | 0.0009 | comparison.json comparisons[127] (G5, readmit_window, s8_a3 minus arm_laya, auroc_30d).difference estimate | 0.000947 | PASS |
| 1746 | Table 15, `s8_a3` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), low | -0.0131 | comparison.json comparisons[127] (G5, readmit_window, s8_a3 minus arm_laya, auroc_30d).difference low | -0.013058 | PASS |
| 1747 | Table 15, `s8_a3` minus `arm_laya`, AUROC, 30 days, Difference (95 percent interval), high | 0.0154 | comparison.json comparisons[127] (G5, readmit_window, s8_a3 minus arm_laya, auroc_30d).difference high | 0.015377 | PASS |
| 1748 | Table 15, `s8_a3` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), estimate | -0.0217 | comparison.json comparisons[128] (G5, readmit_window, s8_a3 minus arm_laya, calibration_intercept_30d).difference estimate | -0.021705 | PASS |
| 1749 | Table 15, `s8_a3` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), low | -0.0297 | comparison.json comparisons[128] (G5, readmit_window, s8_a3 minus arm_laya, calibration_intercept_30d).difference low | -0.029665 | PASS |
| 1750 | Table 15, `s8_a3` minus `arm_laya`, Calibration intercept, 30 days, Difference (95 percent interval), high | -0.0142 | comparison.json comparisons[128] (G5, readmit_window, s8_a3 minus arm_laya, calibration_intercept_30d).difference high | -0.014240 | PASS |
| 1751 | Table 15, `s8_a3` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), estimate | -0.2344 | comparison.json comparisons[129] (G5, readmit_window, s8_a3 minus arm_laya, calibration_slope_30d).difference estimate | -0.234401 | PASS |
| 1752 | Table 15, `s8_a3` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), low | -0.3331 | comparison.json comparisons[129] (G5, readmit_window, s8_a3 minus arm_laya, calibration_slope_30d).difference low | -0.333116 | PASS |
| 1753 | Table 15, `s8_a3` minus `arm_laya`, Calibration slope, 30 days, Difference (95 percent interval), high | -0.1413 | comparison.json comparisons[129] (G5, readmit_window, s8_a3 minus arm_laya, calibration_slope_30d).difference high | -0.141333 | PASS |
| 1754 | results, line 341 (handoff_ei), number 1 | 0.9409 | handoff_test.json questions.evidence_effect.rate_0.accuracy.estimate | 0.940887 | PASS |
| 1755 | results, line 341 (handoff_ei), number 2 | 0.9294 | handoff_test.json questions.evidence_effect.rate_10.accuracy.estimate | 0.929392 | PASS |
| 1756 | results, line 341 (handoff_ei), number 3 | 0.9163 | handoff_test.json questions.evidence_effect.rate_20.accuracy.estimate | 0.916256 | PASS |
| 1757 | results, line 341 (handoff_ei), number 4 | 0.8998 | handoff_test.json questions.evidence_effect.rate_30.accuracy.estimate | 0.899836 | PASS |
| 1758 | results, line 341 (handoff_ei), number 5 | 0.8785 | handoff_test.json questions.evidence_effect.rate_50.accuracy.estimate | 0.878489 | PASS |
| 1759 | results, line 341 (handoff_ei), number 6 | 10 | handoff_test.json rates.1 | 10.000000 | PASS |
| 1760 | results, line 341 (handoff_ei), number 7 | 20 | handoff_test.json rates.2 | 20.000000 | PASS |
| 1761 | results, line 341 (handoff_ei), number 8 | 30 | handoff_test.json rates.3 | 30.000000 | PASS |
| 1762 | results, line 341 (handoff_ei), number 9 | 50 | handoff_test.json rates.4 | 50.000000 | PASS |
| 1763 | results, line 341 (handoff_ei), number 10 | 0.6942 | handoff_test.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.estimate | 0.694215 | PASS |
| 1764 | results, line 341 (handoff_ei), number 11 | 0.5785 | handoff_test.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.estimate | 0.578512 | PASS |
| 1765 | results, line 341 (handoff_ei), number 12 | 10 | handoff_test.json rates.1 | 10.000000 | PASS |
| 1766 | results, line 343 (handoff_pqa), number 1 | 0.7960 | handoff_test.json questions.pubmedqa_answer.rate_0.accuracy.estimate | 0.796000 | PASS |
| 1767 | results, line 343 (handoff_pqa), number 2 | 0.7960 | handoff_test.json questions.pubmedqa_answer.rate_50.accuracy.estimate | 0.796000 | PASS |
| 1768 | results, line 343 (handoff_pqa), number 3 | 50 | handoff_test.json rates.4 | 50.000000 | PASS |
| 1769 | results, line 345 (cost), number 1 | $3.51 | handoff_test.json hourly_rate | 3.510000 | PASS |
| 1770 | results, line 345 (cost), number 2 | $0.53 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_0 | 0.532938 | PASS |
| 1771 | results, line 345 (cost), number 3 | $1.32 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_50 | 1.322609 | PASS |
| 1772 | results, line 345 (cost), number 4 | 50 | handoff_test.json rates.4 | 50.000000 | PASS |
| 1773 | results, line 345 (cost), number 5 | 0.5466 | handoff_test.json questions.evidence_effect.seconds_per_item.fiorillo | 0.546603 | PASS |
| 1774 | results, line 345 (cost), number 6 | 1.6198 | handoff_test.json questions.evidence_effect.seconds_per_item.large_model | 1.619837 | PASS |
| 1775 | results, line 345 (cost), number 7 | $0.02 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_0 | 0.019486 | PASS |
| 1776 | results, line 345 (cost), number 8 | $0.41 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_50 | 0.411454 | PASS |
| 1777 | results, line 345 (cost), number 9 | 0.0200 | handoff_test.json questions.pubmedqa_answer.seconds_per_item.fiorillo | 0.019986 | PASS |
| 1778 | results, line 345 (cost), number 10 | 0.8040 | handoff_test.json questions.pubmedqa_answer.seconds_per_item.large_model | 0.804037 | PASS |
| 1779 | Table 16, Evidence Inference, 0, Items handed off | 0 | handoff_test.json questions.evidence_effect.rate_0.handed_off | 0 | PASS |
| 1780 | Table 16, Evidence Inference, 0, Accuracy, estimate | 0.9409 | handoff_test.json questions.evidence_effect.rate_0.accuracy.estimate | 0.940887 | PASS |
| 1781 | Table 16, Evidence Inference, 0, Accuracy, low | 0.9246 | handoff_test.json questions.evidence_effect.rate_0.accuracy.low | 0.924604 | PASS |
| 1782 | Table 16, Evidence Inference, 0, Accuracy, high | 0.9554 | handoff_test.json questions.evidence_effect.rate_0.accuracy.high | 0.955372 | PASS |
| 1783 | Table 16, Evidence Inference, 0, Macro-F1, estimate | 0.9399 | handoff_test.json questions.evidence_effect.rate_0.macro_f1.estimate | 0.939909 | PASS |
| 1784 | Table 16, Evidence Inference, 0, Macro-F1, low | 0.9225 | handoff_test.json questions.evidence_effect.rate_0.macro_f1.low | 0.922543 | PASS |
| 1785 | Table 16, Evidence Inference, 0, Macro-F1, high | 0.9544 | handoff_test.json questions.evidence_effect.rate_0.macro_f1.high | 0.954353 | PASS |
| 1786 | Table 16, Evidence Inference, 0, Log loss, estimate | 0.1994 | handoff_test.json questions.evidence_effect.rate_0.log_loss.estimate | 0.199449 | PASS |
| 1787 | Table 16, Evidence Inference, 0, Log loss, low | 0.1573 | handoff_test.json questions.evidence_effect.rate_0.log_loss.low | 0.157254 | PASS |
| 1788 | Table 16, Evidence Inference, 0, Log loss, high | 0.2486 | handoff_test.json questions.evidence_effect.rate_0.log_loss.high | 0.248619 | PASS |
| 1789 | Table 16, Evidence Inference, 0, Cost per 1,000 questions | $0.53 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_0 | 0.532938 | PASS |
| 1790 | Table 16, Evidence Inference, 10, Items handed off | 121 | handoff_test.json questions.evidence_effect.rate_10.handed_off | 121 | PASS |
| 1791 | Table 16, Evidence Inference, 10, Accuracy, estimate | 0.9294 | handoff_test.json questions.evidence_effect.rate_10.accuracy.estimate | 0.929392 | PASS |
| 1792 | Table 16, Evidence Inference, 10, Accuracy, low | 0.9102 | handoff_test.json questions.evidence_effect.rate_10.accuracy.low | 0.910194 | PASS |
| 1793 | Table 16, Evidence Inference, 10, Accuracy, high | 0.9447 | handoff_test.json questions.evidence_effect.rate_10.accuracy.high | 0.944721 | PASS |
| 1794 | Table 16, Evidence Inference, 10, Macro-F1, estimate | 0.9264 | handoff_test.json questions.evidence_effect.rate_10.macro_f1.estimate | 0.926446 | PASS |
| 1795 | Table 16, Evidence Inference, 10, Macro-F1, low | 0.9059 | handoff_test.json questions.evidence_effect.rate_10.macro_f1.low | 0.905924 | PASS |
| 1796 | Table 16, Evidence Inference, 10, Macro-F1, high | 0.9429 | handoff_test.json questions.evidence_effect.rate_10.macro_f1.high | 0.942873 | PASS |
| 1797 | Table 16, Evidence Inference, 10, Log loss, estimate | 0.2374 | handoff_test.json questions.evidence_effect.rate_10.log_loss.estimate | 0.237404 | PASS |
| 1798 | Table 16, Evidence Inference, 10, Log loss, low | 0.1910 | handoff_test.json questions.evidence_effect.rate_10.log_loss.low | 0.190991 | PASS |
| 1799 | Table 16, Evidence Inference, 10, Log loss, high | 0.2910 | handoff_test.json questions.evidence_effect.rate_10.log_loss.high | 0.291013 | PASS |
| 1800 | Table 16, Evidence Inference, 10, Specialist's accuracy on the handed-off items, estimate | 0.6942 | handoff_test.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.estimate | 0.694215 | PASS |
| 1801 | Table 16, Evidence Inference, 10, Specialist's accuracy on the handed-off items, low | 0.6186 | handoff_test.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.low | 0.618563 | PASS |
| 1802 | Table 16, Evidence Inference, 10, Specialist's accuracy on the handed-off items, high | 0.7900 | handoff_test.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.high | 0.790003 | PASS |
| 1803 | Table 16, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, estimate | 0.5785 | handoff_test.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.estimate | 0.578512 | PASS |
| 1804 | Table 16, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, low | 0.4746 | handoff_test.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.low | 0.474565 | PASS |
| 1805 | Table 16, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, high | 0.6829 | handoff_test.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.high | 0.682937 | PASS |
| 1806 | Table 16, Evidence Inference, 10, Cost per 1,000 questions | $0.69 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_10 | 0.690872 | PASS |
| 1807 | Table 16, Evidence Inference, 20, Items handed off | 243 | handoff_test.json questions.evidence_effect.rate_20.handed_off | 243 | PASS |
| 1808 | Table 16, Evidence Inference, 20, Accuracy, estimate | 0.9163 | handoff_test.json questions.evidence_effect.rate_20.accuracy.estimate | 0.916256 | PASS |
| 1809 | Table 16, Evidence Inference, 20, Accuracy, low | 0.8985 | handoff_test.json questions.evidence_effect.rate_20.accuracy.low | 0.898482 | PASS |
| 1810 | Table 16, Evidence Inference, 20, Accuracy, high | 0.9322 | handoff_test.json questions.evidence_effect.rate_20.accuracy.high | 0.932196 | PASS |
| 1811 | Table 16, Evidence Inference, 20, Macro-F1, estimate | 0.9134 | handoff_test.json questions.evidence_effect.rate_20.macro_f1.estimate | 0.913420 | PASS |
| 1812 | Table 16, Evidence Inference, 20, Macro-F1, low | 0.8941 | handoff_test.json questions.evidence_effect.rate_20.macro_f1.low | 0.894114 | PASS |
| 1813 | Table 16, Evidence Inference, 20, Macro-F1, high | 0.9302 | handoff_test.json questions.evidence_effect.rate_20.macro_f1.high | 0.930179 | PASS |
| 1814 | Table 16, Evidence Inference, 20, Log loss, estimate | 0.2704 | handoff_test.json questions.evidence_effect.rate_20.log_loss.estimate | 0.270436 | PASS |
| 1815 | Table 16, Evidence Inference, 20, Log loss, low | 0.2223 | handoff_test.json questions.evidence_effect.rate_20.log_loss.low | 0.222285 | PASS |
| 1816 | Table 16, Evidence Inference, 20, Log loss, high | 0.3198 | handoff_test.json questions.evidence_effect.rate_20.log_loss.high | 0.319765 | PASS |
| 1817 | Table 16, Evidence Inference, 20, Specialist's accuracy on the handed-off items, estimate | 0.7901 | handoff_test.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.estimate | 0.790123 | PASS |
| 1818 | Table 16, Evidence Inference, 20, Specialist's accuracy on the handed-off items, low | 0.7266 | handoff_test.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.low | 0.726561 | PASS |
| 1819 | Table 16, Evidence Inference, 20, Specialist's accuracy on the handed-off items, high | 0.8452 | handoff_test.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.high | 0.845196 | PASS |
| 1820 | Table 16, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, estimate | 0.6667 | handoff_test.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.estimate | 0.666667 | PASS |
| 1821 | Table 16, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, low | 0.5958 | handoff_test.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.low | 0.595797 | PASS |
| 1822 | Table 16, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, high | 0.7317 | handoff_test.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.high | 0.731713 | PASS |
| 1823 | Table 16, Evidence Inference, 20, Cost per 1,000 questions | $0.85 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_20 | 0.848806 | PASS |
| 1824 | Table 16, Evidence Inference, 30, Items handed off | 365 | handoff_test.json questions.evidence_effect.rate_30.handed_off | 365 | PASS |
| 1825 | Table 16, Evidence Inference, 30, Accuracy, estimate | 0.8998 | handoff_test.json questions.evidence_effect.rate_30.accuracy.estimate | 0.899836 | PASS |
| 1826 | Table 16, Evidence Inference, 30, Accuracy, low | 0.8782 | handoff_test.json questions.evidence_effect.rate_30.accuracy.low | 0.878226 | PASS |
| 1827 | Table 16, Evidence Inference, 30, Accuracy, high | 0.9192 | handoff_test.json questions.evidence_effect.rate_30.accuracy.high | 0.919169 | PASS |
| 1828 | Table 16, Evidence Inference, 30, Macro-F1, estimate | 0.8954 | handoff_test.json questions.evidence_effect.rate_30.macro_f1.estimate | 0.895435 | PASS |
| 1829 | Table 16, Evidence Inference, 30, Macro-F1, low | 0.8722 | handoff_test.json questions.evidence_effect.rate_30.macro_f1.low | 0.872156 | PASS |
| 1830 | Table 16, Evidence Inference, 30, Macro-F1, high | 0.9152 | handoff_test.json questions.evidence_effect.rate_30.macro_f1.high | 0.915159 | PASS |
| 1831 | Table 16, Evidence Inference, 30, Log loss, estimate | 0.3042 | handoff_test.json questions.evidence_effect.rate_30.log_loss.estimate | 0.304177 | PASS |
| 1832 | Table 16, Evidence Inference, 30, Log loss, low | 0.2478 | handoff_test.json questions.evidence_effect.rate_30.log_loss.low | 0.247795 | PASS |
| 1833 | Table 16, Evidence Inference, 30, Log loss, high | 0.3631 | handoff_test.json questions.evidence_effect.rate_30.log_loss.high | 0.363095 | PASS |
| 1834 | Table 16, Evidence Inference, 30, Specialist's accuracy on the handed-off items, estimate | 0.8356 | handoff_test.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.estimate | 0.835616 | PASS |
| 1835 | Table 16, Evidence Inference, 30, Specialist's accuracy on the handed-off items, low | 0.7896 | handoff_test.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.low | 0.789600 | PASS |
| 1836 | Table 16, Evidence Inference, 30, Specialist's accuracy on the handed-off items, high | 0.8781 | handoff_test.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.high | 0.878050 | PASS |
| 1837 | Table 16, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, estimate | 0.6986 | handoff_test.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.estimate | 0.698630 | PASS |
| 1838 | Table 16, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, low | 0.6342 | handoff_test.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.low | 0.634216 | PASS |
| 1839 | Table 16, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, high | 0.7586 | handoff_test.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.high | 0.758644 | PASS |
| 1840 | Table 16, Evidence Inference, 30, Cost per 1,000 questions | $1.01 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_30 | 1.006741 | PASS |
| 1841 | Table 16, Evidence Inference, 50, Items handed off | 609 | handoff_test.json questions.evidence_effect.rate_50.handed_off | 609 | PASS |
| 1842 | Table 16, Evidence Inference, 50, Accuracy, estimate | 0.8785 | handoff_test.json questions.evidence_effect.rate_50.accuracy.estimate | 0.878489 | PASS |
| 1843 | Table 16, Evidence Inference, 50, Accuracy, low | 0.8552 | handoff_test.json questions.evidence_effect.rate_50.accuracy.low | 0.855178 | PASS |
| 1844 | Table 16, Evidence Inference, 50, Accuracy, high | 0.9014 | handoff_test.json questions.evidence_effect.rate_50.accuracy.high | 0.901432 | PASS |
| 1845 | Table 16, Evidence Inference, 50, Macro-F1, estimate | 0.8710 | handoff_test.json questions.evidence_effect.rate_50.macro_f1.estimate | 0.870958 | PASS |
| 1846 | Table 16, Evidence Inference, 50, Macro-F1, low | 0.8444 | handoff_test.json questions.evidence_effect.rate_50.macro_f1.low | 0.844437 | PASS |
| 1847 | Table 16, Evidence Inference, 50, Macro-F1, high | 0.8947 | handoff_test.json questions.evidence_effect.rate_50.macro_f1.high | 0.894689 | PASS |
| 1848 | Table 16, Evidence Inference, 50, Log loss, estimate | 0.3593 | handoff_test.json questions.evidence_effect.rate_50.log_loss.estimate | 0.359260 | PASS |
| 1849 | Table 16, Evidence Inference, 50, Log loss, low | 0.2976 | handoff_test.json questions.evidence_effect.rate_50.log_loss.low | 0.297621 | PASS |
| 1850 | Table 16, Evidence Inference, 50, Log loss, high | 0.4208 | handoff_test.json questions.evidence_effect.rate_50.log_loss.high | 0.420795 | PASS |
| 1851 | Table 16, Evidence Inference, 50, Specialist's accuracy on the handed-off items, estimate | 0.8883 | handoff_test.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.estimate | 0.888342 | PASS |
| 1852 | Table 16, Evidence Inference, 50, Specialist's accuracy on the handed-off items, low | 0.8583 | handoff_test.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.low | 0.858320 | PASS |
| 1853 | Table 16, Evidence Inference, 50, Specialist's accuracy on the handed-off items, high | 0.9144 | handoff_test.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.high | 0.914378 | PASS |
| 1854 | Table 16, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, estimate | 0.7635 | handoff_test.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.estimate | 0.763547 | PASS |
| 1855 | Table 16, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, low | 0.7187 | handoff_test.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.low | 0.718726 | PASS |
| 1856 | Table 16, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, high | 0.8077 | handoff_test.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.high | 0.807709 | PASS |
| 1857 | Table 16, Evidence Inference, 50, Cost per 1,000 questions | $1.32 | handoff_test.json questions.evidence_effect.cost_per_1000_questions.rate_50 | 1.322609 | PASS |
| 1858 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Accuracy, estimate | 0.8711 | handoff_test.json questions.evidence_effect.large_model_alone.accuracy.estimate | 0.871100 | PASS |
| 1859 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Accuracy, low | 0.8450 | handoff_test.json questions.evidence_effect.large_model_alone.accuracy.low | 0.845020 | PASS |
| 1860 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Accuracy, high | 0.8949 | handoff_test.json questions.evidence_effect.large_model_alone.accuracy.high | 0.894911 | PASS |
| 1861 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, estimate | 0.8626 | handoff_test.json questions.evidence_effect.large_model_alone.macro_f1.estimate | 0.862587 | PASS |
| 1862 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, low | 0.8333 | handoff_test.json questions.evidence_effect.large_model_alone.macro_f1.low | 0.833263 | PASS |
| 1863 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, high | 0.8879 | handoff_test.json questions.evidence_effect.large_model_alone.macro_f1.high | 0.887909 | PASS |
| 1864 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Log loss, estimate | 0.4078 | handoff_test.json questions.evidence_effect.large_model_alone.log_loss.estimate | 0.407805 | PASS |
| 1865 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Log loss, low | 0.3373 | handoff_test.json questions.evidence_effect.large_model_alone.log_loss.low | 0.337272 | PASS |
| 1866 | Table 16, Evidence Inference, Gemma 4 31B-it alone, Log loss, high | 0.4804 | handoff_test.json questions.evidence_effect.large_model_alone.log_loss.high | 0.480403 | PASS |
| 1867 | Table 16, PubMedQA, 0, Items handed off | 0 | handoff_test.json questions.pubmedqa_answer.rate_0.handed_off | 0 | PASS |
| 1868 | Table 16, PubMedQA, 0, Accuracy, estimate | 0.7960 | handoff_test.json questions.pubmedqa_answer.rate_0.accuracy.estimate | 0.796000 | PASS |
| 1869 | Table 16, PubMedQA, 0, Accuracy, low | 0.7600 | handoff_test.json questions.pubmedqa_answer.rate_0.accuracy.low | 0.760000 | PASS |
| 1870 | Table 16, PubMedQA, 0, Accuracy, high | 0.8340 | handoff_test.json questions.pubmedqa_answer.rate_0.accuracy.high | 0.834000 | PASS |
| 1871 | Table 16, PubMedQA, 0, Macro-F1, estimate | 0.5698 | handoff_test.json questions.pubmedqa_answer.rate_0.macro_f1.estimate | 0.569754 | PASS |
| 1872 | Table 16, PubMedQA, 0, Macro-F1, low | 0.5406 | handoff_test.json questions.pubmedqa_answer.rate_0.macro_f1.low | 0.540643 | PASS |
| 1873 | Table 16, PubMedQA, 0, Macro-F1, high | 0.6039 | handoff_test.json questions.pubmedqa_answer.rate_0.macro_f1.high | 0.603864 | PASS |
| 1874 | Table 16, PubMedQA, 0, Log loss, estimate | 0.5797 | handoff_test.json questions.pubmedqa_answer.rate_0.log_loss.estimate | 0.579666 | PASS |
| 1875 | Table 16, PubMedQA, 0, Log loss, low | 0.5070 | handoff_test.json questions.pubmedqa_answer.rate_0.log_loss.low | 0.506978 | PASS |
| 1876 | Table 16, PubMedQA, 0, Log loss, high | 0.6509 | handoff_test.json questions.pubmedqa_answer.rate_0.log_loss.high | 0.650873 | PASS |
| 1877 | Table 16, PubMedQA, 0, Cost per 1,000 questions | $0.02 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_0 | 0.019486 | PASS |
| 1878 | Table 16, PubMedQA, 10, Items handed off | 50 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off | 50 | PASS |
| 1879 | Table 16, PubMedQA, 10, Accuracy, estimate | 0.8000 | handoff_test.json questions.pubmedqa_answer.rate_10.accuracy.estimate | 0.800000 | PASS |
| 1880 | Table 16, PubMedQA, 10, Accuracy, low | 0.7620 | handoff_test.json questions.pubmedqa_answer.rate_10.accuracy.low | 0.762000 | PASS |
| 1881 | Table 16, PubMedQA, 10, Accuracy, high | 0.8360 | handoff_test.json questions.pubmedqa_answer.rate_10.accuracy.high | 0.836000 | PASS |
| 1882 | Table 16, PubMedQA, 10, Macro-F1, estimate | 0.5625 | handoff_test.json questions.pubmedqa_answer.rate_10.macro_f1.estimate | 0.562471 | PASS |
| 1883 | Table 16, PubMedQA, 10, Macro-F1, low | 0.5411 | handoff_test.json questions.pubmedqa_answer.rate_10.macro_f1.low | 0.541088 | PASS |
| 1884 | Table 16, PubMedQA, 10, Macro-F1, high | 0.5828 | handoff_test.json questions.pubmedqa_answer.rate_10.macro_f1.high | 0.582767 | PASS |
| 1885 | Table 16, PubMedQA, 10, Log loss, estimate | 0.5774 | handoff_test.json questions.pubmedqa_answer.rate_10.log_loss.estimate | 0.577430 | PASS |
| 1886 | Table 16, PubMedQA, 10, Log loss, low | 0.5070 | handoff_test.json questions.pubmedqa_answer.rate_10.log_loss.low | 0.506970 | PASS |
| 1887 | Table 16, PubMedQA, 10, Log loss, high | 0.6525 | handoff_test.json questions.pubmedqa_answer.rate_10.log_loss.high | 0.652515 | PASS |
| 1888 | Table 16, PubMedQA, 10, Specialist's accuracy on the handed-off items, estimate | 0.5600 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.estimate | 0.560000 | PASS |
| 1889 | Table 16, PubMedQA, 10, Specialist's accuracy on the handed-off items, low | 0.4200 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.low | 0.420000 | PASS |
| 1890 | Table 16, PubMedQA, 10, Specialist's accuracy on the handed-off items, high | 0.7000 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.high | 0.700000 | PASS |
| 1891 | Table 16, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, estimate | 0.6000 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.estimate | 0.600000 | PASS |
| 1892 | Table 16, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, low | 0.4600 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.low | 0.460000 | PASS |
| 1893 | Table 16, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, high | 0.7200 | handoff_test.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.high | 0.720000 | PASS |
| 1894 | Table 16, PubMedQA, 10, Cost per 1,000 questions | $0.10 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_10 | 0.097880 | PASS |
| 1895 | Table 16, PubMedQA, 20, Items handed off | 100 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off | 100 | PASS |
| 1896 | Table 16, PubMedQA, 20, Accuracy, estimate | 0.7960 | handoff_test.json questions.pubmedqa_answer.rate_20.accuracy.estimate | 0.796000 | PASS |
| 1897 | Table 16, PubMedQA, 20, Accuracy, low | 0.7600 | handoff_test.json questions.pubmedqa_answer.rate_20.accuracy.low | 0.760000 | PASS |
| 1898 | Table 16, PubMedQA, 20, Accuracy, high | 0.8340 | handoff_test.json questions.pubmedqa_answer.rate_20.accuracy.high | 0.834000 | PASS |
| 1899 | Table 16, PubMedQA, 20, Macro-F1, estimate | 0.5729 | handoff_test.json questions.pubmedqa_answer.rate_20.macro_f1.estimate | 0.572855 | PASS |
| 1900 | Table 16, PubMedQA, 20, Macro-F1, low | 0.5465 | handoff_test.json questions.pubmedqa_answer.rate_20.macro_f1.low | 0.546514 | PASS |
| 1901 | Table 16, PubMedQA, 20, Macro-F1, high | 0.6056 | handoff_test.json questions.pubmedqa_answer.rate_20.macro_f1.high | 0.605632 | PASS |
| 1902 | Table 16, PubMedQA, 20, Log loss, estimate | 0.5908 | handoff_test.json questions.pubmedqa_answer.rate_20.log_loss.estimate | 0.590847 | PASS |
| 1903 | Table 16, PubMedQA, 20, Log loss, low | 0.5051 | handoff_test.json questions.pubmedqa_answer.rate_20.log_loss.low | 0.505135 | PASS |
| 1904 | Table 16, PubMedQA, 20, Log loss, high | 0.6638 | handoff_test.json questions.pubmedqa_answer.rate_20.log_loss.high | 0.663767 | PASS |
| 1905 | Table 16, PubMedQA, 20, Specialist's accuracy on the handed-off items, estimate | 0.5800 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.estimate | 0.580000 | PASS |
| 1906 | Table 16, PubMedQA, 20, Specialist's accuracy on the handed-off items, low | 0.4800 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.low | 0.480000 | PASS |
| 1907 | Table 16, PubMedQA, 20, Specialist's accuracy on the handed-off items, high | 0.6800 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.high | 0.680000 | PASS |
| 1908 | Table 16, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, estimate | 0.5800 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.estimate | 0.580000 | PASS |
| 1909 | Table 16, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, low | 0.4900 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.low | 0.490000 | PASS |
| 1910 | Table 16, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, high | 0.6800 | handoff_test.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.high | 0.680000 | PASS |
| 1911 | Table 16, PubMedQA, 20, Cost per 1,000 questions | $0.18 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_20 | 0.176273 | PASS |
| 1912 | Table 16, PubMedQA, 30, Items handed off | 150 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off | 150 | PASS |
| 1913 | Table 16, PubMedQA, 30, Accuracy, estimate | 0.7940 | handoff_test.json questions.pubmedqa_answer.rate_30.accuracy.estimate | 0.794000 | PASS |
| 1914 | Table 16, PubMedQA, 30, Accuracy, low | 0.7580 | handoff_test.json questions.pubmedqa_answer.rate_30.accuracy.low | 0.758000 | PASS |
| 1915 | Table 16, PubMedQA, 30, Accuracy, high | 0.8320 | handoff_test.json questions.pubmedqa_answer.rate_30.accuracy.high | 0.832000 | PASS |
| 1916 | Table 16, PubMedQA, 30, Macro-F1, estimate | 0.5739 | handoff_test.json questions.pubmedqa_answer.rate_30.macro_f1.estimate | 0.573875 | PASS |
| 1917 | Table 16, PubMedQA, 30, Macro-F1, low | 0.5470 | handoff_test.json questions.pubmedqa_answer.rate_30.macro_f1.low | 0.547012 | PASS |
| 1918 | Table 16, PubMedQA, 30, Macro-F1, high | 0.6039 | handoff_test.json questions.pubmedqa_answer.rate_30.macro_f1.high | 0.603934 | PASS |
| 1919 | Table 16, PubMedQA, 30, Log loss, estimate | 0.5833 | handoff_test.json questions.pubmedqa_answer.rate_30.log_loss.estimate | 0.583346 | PASS |
| 1920 | Table 16, PubMedQA, 30, Log loss, low | 0.5036 | handoff_test.json questions.pubmedqa_answer.rate_30.log_loss.low | 0.503633 | PASS |
| 1921 | Table 16, PubMedQA, 30, Log loss, high | 0.6632 | handoff_test.json questions.pubmedqa_answer.rate_30.log_loss.high | 0.663180 | PASS |
| 1922 | Table 16, PubMedQA, 30, Specialist's accuracy on the handed-off items, estimate | 0.6000 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.estimate | 0.600000 | PASS |
| 1923 | Table 16, PubMedQA, 30, Specialist's accuracy on the handed-off items, low | 0.5200 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.low | 0.520000 | PASS |
| 1924 | Table 16, PubMedQA, 30, Specialist's accuracy on the handed-off items, high | 0.6800 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.high | 0.680000 | PASS |
| 1925 | Table 16, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, estimate | 0.5933 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.estimate | 0.593333 | PASS |
| 1926 | Table 16, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, low | 0.5133 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.low | 0.513333 | PASS |
| 1927 | Table 16, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, high | 0.6668 | handoff_test.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.high | 0.666833 | PASS |
| 1928 | Table 16, PubMedQA, 30, Cost per 1,000 questions | $0.25 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_30 | 0.254667 | PASS |
| 1929 | Table 16, PubMedQA, 50, Items handed off | 250 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off | 250 | PASS |
| 1930 | Table 16, PubMedQA, 50, Accuracy, estimate | 0.7960 | handoff_test.json questions.pubmedqa_answer.rate_50.accuracy.estimate | 0.796000 | PASS |
| 1931 | Table 16, PubMedQA, 50, Accuracy, low | 0.7600 | handoff_test.json questions.pubmedqa_answer.rate_50.accuracy.low | 0.759950 | PASS |
| 1932 | Table 16, PubMedQA, 50, Accuracy, high | 0.8340 | handoff_test.json questions.pubmedqa_answer.rate_50.accuracy.high | 0.834000 | PASS |
| 1933 | Table 16, PubMedQA, 50, Macro-F1, estimate | 0.5932 | handoff_test.json questions.pubmedqa_answer.rate_50.macro_f1.estimate | 0.593238 | PASS |
| 1934 | Table 16, PubMedQA, 50, Macro-F1, low | 0.5591 | handoff_test.json questions.pubmedqa_answer.rate_50.macro_f1.low | 0.559096 | PASS |
| 1935 | Table 16, PubMedQA, 50, Macro-F1, high | 0.6325 | handoff_test.json questions.pubmedqa_answer.rate_50.macro_f1.high | 0.632529 | PASS |
| 1936 | Table 16, PubMedQA, 50, Log loss, estimate | 0.5921 | handoff_test.json questions.pubmedqa_answer.rate_50.log_loss.estimate | 0.592063 | PASS |
| 1937 | Table 16, PubMedQA, 50, Log loss, low | 0.5081 | handoff_test.json questions.pubmedqa_answer.rate_50.log_loss.low | 0.508079 | PASS |
| 1938 | Table 16, PubMedQA, 50, Log loss, high | 0.6750 | handoff_test.json questions.pubmedqa_answer.rate_50.log_loss.high | 0.675009 | PASS |
| 1939 | Table 16, PubMedQA, 50, Specialist's accuracy on the handed-off items, estimate | 0.6680 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.estimate | 0.668000 | PASS |
| 1940 | Table 16, PubMedQA, 50, Specialist's accuracy on the handed-off items, low | 0.6040 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.low | 0.604000 | PASS |
| 1941 | Table 16, PubMedQA, 50, Specialist's accuracy on the handed-off items, high | 0.7360 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.high | 0.736000 | PASS |
| 1942 | Table 16, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, estimate | 0.6680 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.estimate | 0.668000 | PASS |
| 1943 | Table 16, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, low | 0.6040 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.low | 0.604000 | PASS |
| 1944 | Table 16, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, high | 0.7320 | handoff_test.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.high | 0.732000 | PASS |
| 1945 | Table 16, PubMedQA, 50, Cost per 1,000 questions | $0.41 | handoff_test.json questions.pubmedqa_answer.cost_per_1000_questions.rate_50 | 0.411454 | PASS |
| 1946 | Table 16, PubMedQA, Gemma 4 31B-it alone, Accuracy, estimate | 0.7840 | handoff_test.json questions.pubmedqa_answer.large_model_alone.accuracy.estimate | 0.784000 | PASS |
| 1947 | Table 16, PubMedQA, Gemma 4 31B-it alone, Accuracy, low | 0.7460 | handoff_test.json questions.pubmedqa_answer.large_model_alone.accuracy.low | 0.745950 | PASS |
| 1948 | Table 16, PubMedQA, Gemma 4 31B-it alone, Accuracy, high | 0.8200 | handoff_test.json questions.pubmedqa_answer.large_model_alone.accuracy.high | 0.820050 | PASS |
| 1949 | Table 16, PubMedQA, Gemma 4 31B-it alone, Macro-F1, estimate | 0.5862 | handoff_test.json questions.pubmedqa_answer.large_model_alone.macro_f1.estimate | 0.586196 | PASS |
| 1950 | Table 16, PubMedQA, Gemma 4 31B-it alone, Macro-F1, low | 0.5527 | handoff_test.json questions.pubmedqa_answer.large_model_alone.macro_f1.low | 0.552652 | PASS |
| 1951 | Table 16, PubMedQA, Gemma 4 31B-it alone, Macro-F1, high | 0.6252 | handoff_test.json questions.pubmedqa_answer.large_model_alone.macro_f1.high | 0.625214 | PASS |
| 1952 | Table 16, PubMedQA, Gemma 4 31B-it alone, Log loss, estimate | 0.6131 | handoff_test.json questions.pubmedqa_answer.large_model_alone.log_loss.estimate | 0.613082 | PASS |
| 1953 | Table 16, PubMedQA, Gemma 4 31B-it alone, Log loss, low | 0.5309 | handoff_test.json questions.pubmedqa_answer.large_model_alone.log_loss.low | 0.530868 | PASS |
| 1954 | Table 16, PubMedQA, Gemma 4 31B-it alone, Log loss, high | 0.6957 | handoff_test.json questions.pubmedqa_answer.large_model_alone.log_loss.high | 0.695734 | PASS |
| 1955 | results, line 370 (hv_note), number 1 | 16:57:23 | logs/pod/arm_a.log, the last line holding 'stage arm_a finished' | 16:57:23 | PASS |
| 1956 | Table 17, Evidence Inference, 0, Items handed off | 0 | handoff_validation.json questions.evidence_effect.rate_0.handed_off | 0 | PASS |
| 1957 | Table 17, Evidence Inference, 0, Accuracy, estimate | 0.9369 | handoff_validation.json questions.evidence_effect.rate_0.accuracy.estimate | 0.936944 | PASS |
| 1958 | Table 17, Evidence Inference, 0, Accuracy, low | 0.9206 | handoff_validation.json questions.evidence_effect.rate_0.accuracy.low | 0.920618 | PASS |
| 1959 | Table 17, Evidence Inference, 0, Accuracy, high | 0.9515 | handoff_validation.json questions.evidence_effect.rate_0.accuracy.high | 0.951496 | PASS |
| 1960 | Table 17, Evidence Inference, 0, Macro-F1, estimate | 0.9332 | handoff_validation.json questions.evidence_effect.rate_0.macro_f1.estimate | 0.933174 | PASS |
| 1961 | Table 17, Evidence Inference, 0, Macro-F1, low | 0.9160 | handoff_validation.json questions.evidence_effect.rate_0.macro_f1.low | 0.916046 | PASS |
| 1962 | Table 17, Evidence Inference, 0, Macro-F1, high | 0.9491 | handoff_validation.json questions.evidence_effect.rate_0.macro_f1.high | 0.949058 | PASS |
| 1963 | Table 17, Evidence Inference, 0, Log loss, estimate | 0.1987 | handoff_validation.json questions.evidence_effect.rate_0.log_loss.estimate | 0.198651 | PASS |
| 1964 | Table 17, Evidence Inference, 0, Log loss, low | 0.1572 | handoff_validation.json questions.evidence_effect.rate_0.log_loss.low | 0.157237 | PASS |
| 1965 | Table 17, Evidence Inference, 0, Log loss, high | 0.2460 | handoff_validation.json questions.evidence_effect.rate_0.log_loss.high | 0.246046 | PASS |
| 1966 | Table 17, Evidence Inference, 0, Cost per 1,000 questions | $0.53 | handoff_validation.json questions.evidence_effect.cost_per_1000_questions.rate_0 | 0.531326 | PASS |
| 1967 | Table 17, Evidence Inference, 10, Items handed off | 123 | handoff_validation.json questions.evidence_effect.rate_10.handed_off | 123 | PASS |
| 1968 | Table 17, Evidence Inference, 10, Accuracy, estimate | 0.9369 | handoff_validation.json questions.evidence_effect.rate_10.accuracy.estimate | 0.936944 | PASS |
| 1969 | Table 17, Evidence Inference, 10, Accuracy, low | 0.9218 | handoff_validation.json questions.evidence_effect.rate_10.accuracy.low | 0.921800 | PASS |
| 1970 | Table 17, Evidence Inference, 10, Accuracy, high | 0.9525 | handoff_validation.json questions.evidence_effect.rate_10.accuracy.high | 0.952459 | PASS |
| 1971 | Table 17, Evidence Inference, 10, Macro-F1, estimate | 0.9314 | handoff_validation.json questions.evidence_effect.rate_10.macro_f1.estimate | 0.931426 | PASS |
| 1972 | Table 17, Evidence Inference, 10, Macro-F1, low | 0.9139 | handoff_validation.json questions.evidence_effect.rate_10.macro_f1.low | 0.913945 | PASS |
| 1973 | Table 17, Evidence Inference, 10, Macro-F1, high | 0.9477 | handoff_validation.json questions.evidence_effect.rate_10.macro_f1.high | 0.947703 | PASS |
| 1974 | Table 17, Evidence Inference, 10, Log loss, estimate | 0.2108 | handoff_validation.json questions.evidence_effect.rate_10.log_loss.estimate | 0.210781 | PASS |
| 1975 | Table 17, Evidence Inference, 10, Log loss, low | 0.1657 | handoff_validation.json questions.evidence_effect.rate_10.log_loss.low | 0.165718 | PASS |
| 1976 | Table 17, Evidence Inference, 10, Log loss, high | 0.2611 | handoff_validation.json questions.evidence_effect.rate_10.log_loss.high | 0.261118 | PASS |
| 1977 | Table 17, Evidence Inference, 10, Specialist's accuracy on the handed-off items, estimate | 0.6748 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.estimate | 0.674797 | PASS |
| 1978 | Table 17, Evidence Inference, 10, Specialist's accuracy on the handed-off items, low | 0.5785 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.low | 0.578503 | PASS |
| 1979 | Table 17, Evidence Inference, 10, Specialist's accuracy on the handed-off items, high | 0.7619 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_fiorillo_accuracy.high | 0.761905 | PASS |
| 1980 | Table 17, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, estimate | 0.6748 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.estimate | 0.674797 | PASS |
| 1981 | Table 17, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, low | 0.5691 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.low | 0.569073 | PASS |
| 1982 | Table 17, Evidence Inference, 10, Gemma 4 31B-it's accuracy on them, high | 0.7787 | handoff_validation.json questions.evidence_effect.rate_10.handed_off_large_model_accuracy.high | 0.778689 | PASS |
| 1983 | Table 17, Evidence Inference, 10, Cost per 1,000 questions | $0.72 | handoff_validation.json questions.evidence_effect.cost_per_1000_questions.rate_10 | 0.717220 | PASS |
| 1984 | Table 17, Evidence Inference, 20, Items handed off | 247 | handoff_validation.json questions.evidence_effect.rate_20.handed_off | 247 | PASS |
| 1985 | Table 17, Evidence Inference, 20, Accuracy, estimate | 0.9192 | handoff_validation.json questions.evidence_effect.rate_20.accuracy.estimate | 0.919159 | PASS |
| 1986 | Table 17, Evidence Inference, 20, Accuracy, low | 0.9028 | handoff_validation.json questions.evidence_effect.rate_20.accuracy.low | 0.902811 | PASS |
| 1987 | Table 17, Evidence Inference, 20, Accuracy, high | 0.9345 | handoff_validation.json questions.evidence_effect.rate_20.accuracy.high | 0.934544 | PASS |
| 1988 | Table 17, Evidence Inference, 20, Macro-F1, estimate | 0.9114 | handoff_validation.json questions.evidence_effect.rate_20.macro_f1.estimate | 0.911392 | PASS |
| 1989 | Table 17, Evidence Inference, 20, Macro-F1, low | 0.8934 | handoff_validation.json questions.evidence_effect.rate_20.macro_f1.low | 0.893403 | PASS |
| 1990 | Table 17, Evidence Inference, 20, Macro-F1, high | 0.9281 | handoff_validation.json questions.evidence_effect.rate_20.macro_f1.high | 0.928066 | PASS |
| 1991 | Table 17, Evidence Inference, 20, Log loss, estimate | 0.2557 | handoff_validation.json questions.evidence_effect.rate_20.log_loss.estimate | 0.255695 | PASS |
| 1992 | Table 17, Evidence Inference, 20, Log loss, low | 0.2068 | handoff_validation.json questions.evidence_effect.rate_20.log_loss.low | 0.206839 | PASS |
| 1993 | Table 17, Evidence Inference, 20, Log loss, high | 0.3084 | handoff_validation.json questions.evidence_effect.rate_20.log_loss.high | 0.308422 | PASS |
| 1994 | Table 17, Evidence Inference, 20, Specialist's accuracy on the handed-off items, estimate | 0.7652 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.estimate | 0.765182 | PASS |
| 1995 | Table 17, Evidence Inference, 20, Specialist's accuracy on the handed-off items, low | 0.7029 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.low | 0.702920 | PASS |
| 1996 | Table 17, Evidence Inference, 20, Specialist's accuracy on the handed-off items, high | 0.8193 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_fiorillo_accuracy.high | 0.819278 | PASS |
| 1997 | Table 17, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, estimate | 0.6761 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.estimate | 0.676113 | PASS |
| 1998 | Table 17, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, low | 0.6064 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.low | 0.606426 | PASS |
| 1999 | Table 17, Evidence Inference, 20, Gemma 4 31B-it's accuracy on them, high | 0.7399 | handoff_validation.json questions.evidence_effect.rate_20.handed_off_large_model_accuracy.high | 0.739864 | PASS |
| 2000 | Table 17, Evidence Inference, 20, Cost per 1,000 questions | $0.90 | handoff_validation.json questions.evidence_effect.cost_per_1000_questions.rate_20 | 0.903115 | PASS |
| 2001 | Table 17, Evidence Inference, 30, Items handed off | 371 | handoff_validation.json questions.evidence_effect.rate_30.handed_off | 371 | PASS |
| 2002 | Table 17, Evidence Inference, 30, Accuracy, estimate | 0.9062 | handoff_validation.json questions.evidence_effect.rate_30.accuracy.estimate | 0.906225 | PASS |
| 2003 | Table 17, Evidence Inference, 30, Accuracy, low | 0.8862 | handoff_validation.json questions.evidence_effect.rate_30.accuracy.low | 0.886174 | PASS |
| 2004 | Table 17, Evidence Inference, 30, Accuracy, high | 0.9222 | handoff_validation.json questions.evidence_effect.rate_30.accuracy.high | 0.922222 | PASS |
| 2005 | Table 17, Evidence Inference, 30, Macro-F1, estimate | 0.8970 | handoff_validation.json questions.evidence_effect.rate_30.macro_f1.estimate | 0.896976 | PASS |
| 2006 | Table 17, Evidence Inference, 30, Macro-F1, low | 0.8764 | handoff_validation.json questions.evidence_effect.rate_30.macro_f1.low | 0.876363 | PASS |
| 2007 | Table 17, Evidence Inference, 30, Macro-F1, high | 0.9142 | handoff_validation.json questions.evidence_effect.rate_30.macro_f1.high | 0.914200 | PASS |
| 2008 | Table 17, Evidence Inference, 30, Log loss, estimate | 0.2702 | handoff_validation.json questions.evidence_effect.rate_30.log_loss.estimate | 0.270182 | PASS |
| 2009 | Table 17, Evidence Inference, 30, Log loss, low | 0.2232 | handoff_validation.json questions.evidence_effect.rate_30.log_loss.low | 0.223171 | PASS |
| 2010 | Table 17, Evidence Inference, 30, Log loss, high | 0.3281 | handoff_validation.json questions.evidence_effect.rate_30.log_loss.high | 0.328130 | PASS |
| 2011 | Table 17, Evidence Inference, 30, Specialist's accuracy on the handed-off items, estimate | 0.8113 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.estimate | 0.811321 | PASS |
| 2012 | Table 17, Evidence Inference, 30, Specialist's accuracy on the handed-off items, low | 0.7628 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.low | 0.762797 | PASS |
| 2013 | Table 17, Evidence Inference, 30, Specialist's accuracy on the handed-off items, high | 0.8568 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_fiorillo_accuracy.high | 0.856771 | PASS |
| 2014 | Table 17, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, estimate | 0.7089 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.estimate | 0.708895 | PASS |
| 2015 | Table 17, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, low | 0.6489 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.low | 0.648920 | PASS |
| 2016 | Table 17, Evidence Inference, 30, Gemma 4 31B-it's accuracy on them, high | 0.7609 | handoff_validation.json questions.evidence_effect.rate_30.handed_off_large_model_accuracy.high | 0.760919 | PASS |
| 2017 | Table 17, Evidence Inference, 30, Cost per 1,000 questions | $1.09 | handoff_validation.json questions.evidence_effect.cost_per_1000_questions.rate_30 | 1.089010 | PASS |
| 2018 | Table 17, Evidence Inference, 50, Items handed off | 618 | handoff_validation.json questions.evidence_effect.rate_50.handed_off | 618 | PASS |
| 2019 | Table 17, Evidence Inference, 50, Accuracy, estimate | 0.8925 | handoff_validation.json questions.evidence_effect.rate_50.accuracy.estimate | 0.892482 | PASS |
| 2020 | Table 17, Evidence Inference, 50, Accuracy, low | 0.8716 | handoff_validation.json questions.evidence_effect.rate_50.accuracy.low | 0.871593 | PASS |
| 2021 | Table 17, Evidence Inference, 50, Accuracy, high | 0.9139 | handoff_validation.json questions.evidence_effect.rate_50.accuracy.high | 0.913941 | PASS |
| 2022 | Table 17, Evidence Inference, 50, Macro-F1, estimate | 0.8816 | handoff_validation.json questions.evidence_effect.rate_50.macro_f1.estimate | 0.881634 | PASS |
| 2023 | Table 17, Evidence Inference, 50, Macro-F1, low | 0.8578 | handoff_validation.json questions.evidence_effect.rate_50.macro_f1.low | 0.857837 | PASS |
| 2024 | Table 17, Evidence Inference, 50, Macro-F1, high | 0.9055 | handoff_validation.json questions.evidence_effect.rate_50.macro_f1.high | 0.905502 | PASS |
| 2025 | Table 17, Evidence Inference, 50, Log loss, estimate | 0.2994 | handoff_validation.json questions.evidence_effect.rate_50.log_loss.estimate | 0.299368 | PASS |
| 2026 | Table 17, Evidence Inference, 50, Log loss, low | 0.2443 | handoff_validation.json questions.evidence_effect.rate_50.log_loss.low | 0.244337 | PASS |
| 2027 | Table 17, Evidence Inference, 50, Log loss, high | 0.3578 | handoff_validation.json questions.evidence_effect.rate_50.log_loss.high | 0.357760 | PASS |
| 2028 | Table 17, Evidence Inference, 50, Specialist's accuracy on the handed-off items, estimate | 0.8770 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.estimate | 0.877023 | PASS |
| 2029 | Table 17, Evidence Inference, 50, Specialist's accuracy on the handed-off items, low | 0.8467 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.low | 0.846666 | PASS |
| 2030 | Table 17, Evidence Inference, 50, Specialist's accuracy on the handed-off items, high | 0.9062 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_fiorillo_accuracy.high | 0.906159 | PASS |
| 2031 | Table 17, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, estimate | 0.7880 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.estimate | 0.788026 | PASS |
| 2032 | Table 17, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, low | 0.7460 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.low | 0.746029 | PASS |
| 2033 | Table 17, Evidence Inference, 50, Gemma 4 31B-it's accuracy on them, high | 0.8320 | handoff_validation.json questions.evidence_effect.rate_50.handed_off_large_model_accuracy.high | 0.831981 | PASS |
| 2034 | Table 17, Evidence Inference, 50, Cost per 1,000 questions | $1.46 | handoff_validation.json questions.evidence_effect.cost_per_1000_questions.rate_50 | 1.460800 | PASS |
| 2035 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Accuracy, estimate | 0.8860 | handoff_validation.json questions.evidence_effect.large_model_alone.accuracy.estimate | 0.886015 | PASS |
| 2036 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Accuracy, low | 0.8607 | handoff_validation.json questions.evidence_effect.large_model_alone.accuracy.low | 0.860661 | PASS |
| 2037 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Accuracy, high | 0.9095 | handoff_validation.json questions.evidence_effect.large_model_alone.accuracy.high | 0.909464 | PASS |
| 2038 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, estimate | 0.8747 | handoff_validation.json questions.evidence_effect.large_model_alone.macro_f1.estimate | 0.874682 | PASS |
| 2039 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, low | 0.8461 | handoff_validation.json questions.evidence_effect.large_model_alone.macro_f1.low | 0.846120 | PASS |
| 2040 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Macro-F1, high | 0.8996 | handoff_validation.json questions.evidence_effect.large_model_alone.macro_f1.high | 0.899619 | PASS |
| 2041 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Log loss, estimate | 0.3403 | handoff_validation.json questions.evidence_effect.large_model_alone.log_loss.estimate | 0.340311 | PASS |
| 2042 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Log loss, low | 0.2826 | handoff_validation.json questions.evidence_effect.large_model_alone.log_loss.low | 0.282557 | PASS |
| 2043 | Table 17, Evidence Inference, Gemma 4 31B-it alone, Log loss, high | 0.4036 | handoff_validation.json questions.evidence_effect.large_model_alone.log_loss.high | 0.403646 | PASS |
| 2044 | Table 17, PubMedQA, 0, Items handed off | 0 | handoff_validation.json questions.pubmedqa_answer.rate_0.handed_off | 0 | PASS |
| 2045 | Table 17, PubMedQA, 0, Accuracy, estimate | 0.7867 | handoff_validation.json questions.pubmedqa_answer.rate_0.accuracy.estimate | 0.786667 | PASS |
| 2046 | Table 17, PubMedQA, 0, Accuracy, low | 0.7200 | handoff_validation.json questions.pubmedqa_answer.rate_0.accuracy.low | 0.720000 | PASS |
| 2047 | Table 17, PubMedQA, 0, Accuracy, high | 0.8467 | handoff_validation.json questions.pubmedqa_answer.rate_0.accuracy.high | 0.846667 | PASS |
| 2048 | Table 17, PubMedQA, 0, Macro-F1, estimate | 0.5455 | handoff_validation.json questions.pubmedqa_answer.rate_0.macro_f1.estimate | 0.545483 | PASS |
| 2049 | Table 17, PubMedQA, 0, Macro-F1, low | 0.4994 | handoff_validation.json questions.pubmedqa_answer.rate_0.macro_f1.low | 0.499361 | PASS |
| 2050 | Table 17, PubMedQA, 0, Macro-F1, high | 0.5842 | handoff_validation.json questions.pubmedqa_answer.rate_0.macro_f1.high | 0.584239 | PASS |
| 2051 | Table 17, PubMedQA, 0, Log loss, estimate | 0.5603 | handoff_validation.json questions.pubmedqa_answer.rate_0.log_loss.estimate | 0.560272 | PASS |
| 2052 | Table 17, PubMedQA, 0, Log loss, low | 0.4324 | handoff_validation.json questions.pubmedqa_answer.rate_0.log_loss.low | 0.432431 | PASS |
| 2053 | Table 17, PubMedQA, 0, Log loss, high | 0.6997 | handoff_validation.json questions.pubmedqa_answer.rate_0.log_loss.high | 0.699653 | PASS |
| 2054 | Table 17, PubMedQA, 0, Cost per 1,000 questions | not computed | handoff_validation.json questions.pubmedqa_answer.cost_per_1000_questions.rate_0 | null | PASS |
| 2055 | Table 17, PubMedQA, 10, Items handed off | 15 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off | 15 | PASS |
| 2056 | Table 17, PubMedQA, 10, Accuracy, estimate | 0.8200 | handoff_validation.json questions.pubmedqa_answer.rate_10.accuracy.estimate | 0.820000 | PASS |
| 2057 | Table 17, PubMedQA, 10, Accuracy, low | 0.7533 | handoff_validation.json questions.pubmedqa_answer.rate_10.accuracy.low | 0.753333 | PASS |
| 2058 | Table 17, PubMedQA, 10, Accuracy, high | 0.8800 | handoff_validation.json questions.pubmedqa_answer.rate_10.accuracy.high | 0.880000 | PASS |
| 2059 | Table 17, PubMedQA, 10, Macro-F1, estimate | 0.5709 | handoff_validation.json questions.pubmedqa_answer.rate_10.macro_f1.estimate | 0.570877 | PASS |
| 2060 | Table 17, PubMedQA, 10, Macro-F1, low | 0.5283 | handoff_validation.json questions.pubmedqa_answer.rate_10.macro_f1.low | 0.528333 | PASS |
| 2061 | Table 17, PubMedQA, 10, Macro-F1, high | 0.6044 | handoff_validation.json questions.pubmedqa_answer.rate_10.macro_f1.high | 0.604426 | PASS |
| 2062 | Table 17, PubMedQA, 10, Log loss, estimate | 0.5486 | handoff_validation.json questions.pubmedqa_answer.rate_10.log_loss.estimate | 0.548574 | PASS |
| 2063 | Table 17, PubMedQA, 10, Log loss, low | 0.4087 | handoff_validation.json questions.pubmedqa_answer.rate_10.log_loss.low | 0.408658 | PASS |
| 2064 | Table 17, PubMedQA, 10, Log loss, high | 0.6968 | handoff_validation.json questions.pubmedqa_answer.rate_10.log_loss.high | 0.696823 | PASS |
| 2065 | Table 17, PubMedQA, 10, Specialist's accuracy on the handed-off items, estimate | 0.3333 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.estimate | 0.333333 | PASS |
| 2066 | Table 17, PubMedQA, 10, Specialist's accuracy on the handed-off items, low | 0.1333 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.low | 0.133333 | PASS |
| 2067 | Table 17, PubMedQA, 10, Specialist's accuracy on the handed-off items, high | 0.5333 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_fiorillo_accuracy.high | 0.533333 | PASS |
| 2068 | Table 17, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, estimate | 0.6667 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.estimate | 0.666667 | PASS |
| 2069 | Table 17, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, low | 0.4000 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.low | 0.400000 | PASS |
| 2070 | Table 17, PubMedQA, 10, Gemma 4 31B-it's accuracy on them, high | 0.9333 | handoff_validation.json questions.pubmedqa_answer.rate_10.handed_off_large_model_accuracy.high | 0.933333 | PASS |
| 2071 | Table 17, PubMedQA, 10, Cost per 1,000 questions | not computed | handoff_validation.json questions.pubmedqa_answer.cost_per_1000_questions.rate_10 | null | PASS |
| 2072 | Table 17, PubMedQA, 20, Items handed off | 30 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off | 30 | PASS |
| 2073 | Table 17, PubMedQA, 20, Accuracy, estimate | 0.8133 | handoff_validation.json questions.pubmedqa_answer.rate_20.accuracy.estimate | 0.813333 | PASS |
| 2074 | Table 17, PubMedQA, 20, Accuracy, low | 0.7533 | handoff_validation.json questions.pubmedqa_answer.rate_20.accuracy.low | 0.753333 | PASS |
| 2075 | Table 17, PubMedQA, 20, Accuracy, high | 0.8733 | handoff_validation.json questions.pubmedqa_answer.rate_20.accuracy.high | 0.873333 | PASS |
| 2076 | Table 17, PubMedQA, 20, Macro-F1, estimate | 0.5737 | handoff_validation.json questions.pubmedqa_answer.rate_20.macro_f1.estimate | 0.573707 | PASS |
| 2077 | Table 17, PubMedQA, 20, Macro-F1, low | 0.5365 | handoff_validation.json questions.pubmedqa_answer.rate_20.macro_f1.low | 0.536500 | PASS |
| 2078 | Table 17, PubMedQA, 20, Macro-F1, high | 0.6097 | handoff_validation.json questions.pubmedqa_answer.rate_20.macro_f1.high | 0.609715 | PASS |
| 2079 | Table 17, PubMedQA, 20, Log loss, estimate | 0.5631 | handoff_validation.json questions.pubmedqa_answer.rate_20.log_loss.estimate | 0.563136 | PASS |
| 2080 | Table 17, PubMedQA, 20, Log loss, low | 0.4141 | handoff_validation.json questions.pubmedqa_answer.rate_20.log_loss.low | 0.414080 | PASS |
| 2081 | Table 17, PubMedQA, 20, Log loss, high | 0.7028 | handoff_validation.json questions.pubmedqa_answer.rate_20.log_loss.high | 0.702758 | PASS |
| 2082 | Table 17, PubMedQA, 20, Specialist's accuracy on the handed-off items, estimate | 0.4333 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.estimate | 0.433333 | PASS |
| 2083 | Table 17, PubMedQA, 20, Specialist's accuracy on the handed-off items, low | 0.2658 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.low | 0.265833 | PASS |
| 2084 | Table 17, PubMedQA, 20, Specialist's accuracy on the handed-off items, high | 0.6000 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_fiorillo_accuracy.high | 0.600000 | PASS |
| 2085 | Table 17, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, estimate | 0.5667 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.estimate | 0.566667 | PASS |
| 2086 | Table 17, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, low | 0.4000 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.low | 0.400000 | PASS |
| 2087 | Table 17, PubMedQA, 20, Gemma 4 31B-it's accuracy on them, high | 0.7333 | handoff_validation.json questions.pubmedqa_answer.rate_20.handed_off_large_model_accuracy.high | 0.733333 | PASS |
| 2088 | Table 17, PubMedQA, 20, Cost per 1,000 questions | not computed | handoff_validation.json questions.pubmedqa_answer.cost_per_1000_questions.rate_20 | null | PASS |
| 2089 | Table 17, PubMedQA, 30, Items handed off | 45 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off | 45 | PASS |
| 2090 | Table 17, PubMedQA, 30, Accuracy, estimate | 0.8000 | handoff_validation.json questions.pubmedqa_answer.rate_30.accuracy.estimate | 0.800000 | PASS |
| 2091 | Table 17, PubMedQA, 30, Accuracy, low | 0.7467 | handoff_validation.json questions.pubmedqa_answer.rate_30.accuracy.low | 0.746667 | PASS |
| 2092 | Table 17, PubMedQA, 30, Accuracy, high | 0.8667 | handoff_validation.json questions.pubmedqa_answer.rate_30.accuracy.high | 0.866667 | PASS |
| 2093 | Table 17, PubMedQA, 30, Macro-F1, estimate | 0.5683 | handoff_validation.json questions.pubmedqa_answer.rate_30.macro_f1.estimate | 0.568336 | PASS |
| 2094 | Table 17, PubMedQA, 30, Macro-F1, low | 0.5341 | handoff_validation.json questions.pubmedqa_answer.rate_30.macro_f1.low | 0.534083 | PASS |
| 2095 | Table 17, PubMedQA, 30, Macro-F1, high | 0.6057 | handoff_validation.json questions.pubmedqa_answer.rate_30.macro_f1.high | 0.605715 | PASS |
| 2096 | Table 17, PubMedQA, 30, Log loss, estimate | 0.5663 | handoff_validation.json questions.pubmedqa_answer.rate_30.log_loss.estimate | 0.566281 | PASS |
| 2097 | Table 17, PubMedQA, 30, Log loss, low | 0.4165 | handoff_validation.json questions.pubmedqa_answer.rate_30.log_loss.low | 0.416503 | PASS |
| 2098 | Table 17, PubMedQA, 30, Log loss, high | 0.7119 | handoff_validation.json questions.pubmedqa_answer.rate_30.log_loss.high | 0.711947 | PASS |
| 2099 | Table 17, PubMedQA, 30, Specialist's accuracy on the handed-off items, estimate | 0.5111 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.estimate | 0.511111 | PASS |
| 2100 | Table 17, PubMedQA, 30, Specialist's accuracy on the handed-off items, low | 0.3556 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.low | 0.355556 | PASS |
| 2101 | Table 17, PubMedQA, 30, Specialist's accuracy on the handed-off items, high | 0.6667 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_fiorillo_accuracy.high | 0.666667 | PASS |
| 2102 | Table 17, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, estimate | 0.5556 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.estimate | 0.555556 | PASS |
| 2103 | Table 17, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, low | 0.4439 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.low | 0.443889 | PASS |
| 2104 | Table 17, PubMedQA, 30, Gemma 4 31B-it's accuracy on them, high | 0.7111 | handoff_validation.json questions.pubmedqa_answer.rate_30.handed_off_large_model_accuracy.high | 0.711111 | PASS |
| 2105 | Table 17, PubMedQA, 30, Cost per 1,000 questions | not computed | handoff_validation.json questions.pubmedqa_answer.cost_per_1000_questions.rate_30 | null | PASS |
| 2106 | Table 17, PubMedQA, 50, Items handed off | 75 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off | 75 | PASS |
| 2107 | Table 17, PubMedQA, 50, Accuracy, estimate | 0.7933 | handoff_validation.json questions.pubmedqa_answer.rate_50.accuracy.estimate | 0.793333 | PASS |
| 2108 | Table 17, PubMedQA, 50, Accuracy, low | 0.7332 | handoff_validation.json questions.pubmedqa_answer.rate_50.accuracy.low | 0.733167 | PASS |
| 2109 | Table 17, PubMedQA, 50, Accuracy, high | 0.8535 | handoff_validation.json questions.pubmedqa_answer.rate_50.accuracy.high | 0.853500 | PASS |
| 2110 | Table 17, PubMedQA, 50, Macro-F1, estimate | 0.5672 | handoff_validation.json questions.pubmedqa_answer.rate_50.macro_f1.estimate | 0.567247 | PASS |
| 2111 | Table 17, PubMedQA, 50, Macro-F1, low | 0.5286 | handoff_validation.json questions.pubmedqa_answer.rate_50.macro_f1.low | 0.528574 | PASS |
| 2112 | Table 17, PubMedQA, 50, Macro-F1, high | 0.6010 | handoff_validation.json questions.pubmedqa_answer.rate_50.macro_f1.high | 0.601013 | PASS |
| 2113 | Table 17, PubMedQA, 50, Log loss, estimate | 0.5769 | handoff_validation.json questions.pubmedqa_answer.rate_50.log_loss.estimate | 0.576924 | PASS |
| 2114 | Table 17, PubMedQA, 50, Log loss, low | 0.4251 | handoff_validation.json questions.pubmedqa_answer.rate_50.log_loss.low | 0.425114 | PASS |
| 2115 | Table 17, PubMedQA, 50, Log loss, high | 0.7298 | handoff_validation.json questions.pubmedqa_answer.rate_50.log_loss.high | 0.729789 | PASS |
| 2116 | Table 17, PubMedQA, 50, Specialist's accuracy on the handed-off items, estimate | 0.6533 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.estimate | 0.653333 | PASS |
| 2117 | Table 17, PubMedQA, 50, Specialist's accuracy on the handed-off items, low | 0.5467 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.low | 0.546667 | PASS |
| 2118 | Table 17, PubMedQA, 50, Specialist's accuracy on the handed-off items, high | 0.7733 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_fiorillo_accuracy.high | 0.773333 | PASS |
| 2119 | Table 17, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, estimate | 0.6667 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.estimate | 0.666667 | PASS |
| 2120 | Table 17, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, low | 0.5600 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.low | 0.560000 | PASS |
| 2121 | Table 17, PubMedQA, 50, Gemma 4 31B-it's accuracy on them, high | 0.7733 | handoff_validation.json questions.pubmedqa_answer.rate_50.handed_off_large_model_accuracy.high | 0.773333 | PASS |
| 2122 | Table 17, PubMedQA, 50, Cost per 1,000 questions | not computed | handoff_validation.json questions.pubmedqa_answer.cost_per_1000_questions.rate_50 | null | PASS |
| 2123 | Table 17, PubMedQA, Gemma 4 31B-it alone, Accuracy, estimate | 0.7800 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.accuracy.estimate | 0.780000 | PASS |
| 2124 | Table 17, PubMedQA, Gemma 4 31B-it alone, Accuracy, low | 0.7133 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.accuracy.low | 0.713333 | PASS |
| 2125 | Table 17, PubMedQA, Gemma 4 31B-it alone, Accuracy, high | 0.8467 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.accuracy.high | 0.846667 | PASS |
| 2126 | Table 17, PubMedQA, Gemma 4 31B-it alone, Macro-F1, estimate | 0.5635 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.macro_f1.estimate | 0.563527 | PASS |
| 2127 | Table 17, PubMedQA, Gemma 4 31B-it alone, Macro-F1, low | 0.5247 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.macro_f1.low | 0.524730 | PASS |
| 2128 | Table 17, PubMedQA, Gemma 4 31B-it alone, Macro-F1, high | 0.5980 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.macro_f1.high | 0.597984 | PASS |
| 2129 | Table 17, PubMedQA, Gemma 4 31B-it alone, Log loss, estimate | 0.5934 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.log_loss.estimate | 0.593360 | PASS |
| 2130 | Table 17, PubMedQA, Gemma 4 31B-it alone, Log loss, low | 0.4470 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.log_loss.low | 0.446989 | PASS |
| 2131 | Table 17, PubMedQA, Gemma 4 31B-it alone, Log loss, high | 0.7488 | handoff_validation.json questions.pubmedqa_answer.large_model_alone.log_loss.high | 0.748809 | PASS |
| 2132 | results, line 391 (sens), number 1 | 6 | comparison.json sensitivity flagged_test_prompts | 6 | PASS |
| 2133 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), estimate | -0.8764 | comparison.json comparisons[130] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, log_loss).difference estimate | -0.876398 | PASS |
| 2134 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), low | -0.9209 | comparison.json comparisons[130] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, log_loss).difference low | -0.920861 | PASS |
| 2135 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Log loss, Difference (95 percent interval), high | -0.8293 | comparison.json comparisons[130] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, log_loss).difference high | -0.829314 | PASS |
| 2136 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), estimate | -0.5518 | comparison.json comparisons[131] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, brier).difference estimate | -0.551812 | PASS |
| 2137 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), low | -0.5772 | comparison.json comparisons[131] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, brier).difference low | -0.577166 | PASS |
| 2138 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Brier score, Difference (95 percent interval), high | -0.5249 | comparison.json comparisons[131] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, brier).difference high | -0.524863 | PASS |
| 2139 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), estimate | -0.0175 | comparison.json comparisons[137] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, ece).difference estimate | -0.017519 | PASS |
| 2140 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), low | -0.0453 | comparison.json comparisons[137] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, ece).difference low | -0.045261 | PASS |
| 2141 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, ECE, Difference (95 percent interval), high | 0.0180 | comparison.json comparisons[137] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, ece).difference high | 0.017964 | PASS |
| 2142 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), estimate | 0.5182 | comparison.json comparisons[132] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, accuracy).difference estimate | 0.518152 | PASS |
| 2143 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), low | 0.4753 | comparison.json comparisons[132] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, accuracy).difference low | 0.475324 | PASS |
| 2144 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Accuracy, Difference (95 percent interval), high | 0.5558 | comparison.json comparisons[132] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, accuracy).difference high | 0.555751 | PASS |
| 2145 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), estimate | 0.7416 | comparison.json comparisons[133] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, macro_f1).difference estimate | 0.741602 | PASS |
| 2146 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), low | 0.7208 | comparison.json comparisons[133] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, macro_f1).difference low | 0.720833 | PASS |
| 2147 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Macro-F1, Difference (95 percent interval), high | 0.7606 | comparison.json comparisons[133] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, macro_f1).difference high | 0.760598 | PASS |
| 2148 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), estimate | 0.9352 | comparison.json comparisons[134] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_increased).difference estimate | 0.935162 | PASS |
| 2149 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), low | 0.9150 | comparison.json comparisons[134] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_increased).difference low | 0.914979 | PASS |
| 2150 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly increased, Difference (95 percent interval), high | 0.9526 | comparison.json comparisons[134] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_increased).difference high | 0.952621 | PASS |
| 2151 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.9362 | comparison.json comparisons[135] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_decreased).difference estimate | 0.936242 | PASS |
| 2152 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), low | 0.9118 | comparison.json comparisons[135] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_decreased).difference low | 0.911817 | PASS |
| 2153 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly decreased, Difference (95 percent interval), high | 0.9573 | comparison.json comparisons[135] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_decreased).difference high | 0.957293 | PASS |
| 2154 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), estimate | 0.3534 | comparison.json comparisons[136] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_no_difference).difference estimate | 0.353401 | PASS |
| 2155 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), low | 0.3159 | comparison.json comparisons[136] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_no_difference).difference low | 0.315943 | PASS |
| 2156 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, no significant difference, Difference (95 percent interval), high | 0.3878 | comparison.json comparisons[136] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus prior, f1_no_difference).difference high | 0.387816 | PASS |
| 2157 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), estimate | -0.2097 | comparison.json comparisons[138] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference estimate | -0.209700 | PASS |
| 2158 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), low | -0.2758 | comparison.json comparisons[138] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference low | -0.275792 | PASS |
| 2159 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Log loss, Difference (95 percent interval), high | -0.1512 | comparison.json comparisons[138] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, log_loss).difference high | -0.151199 | PASS |
| 2160 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), estimate | -0.1046 | comparison.json comparisons[139] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, brier).difference estimate | -0.104605 | PASS |
| 2161 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), low | -0.1377 | comparison.json comparisons[139] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, brier).difference low | -0.137652 | PASS |
| 2162 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Brier score, Difference (95 percent interval), high | -0.0748 | comparison.json comparisons[139] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, brier).difference high | -0.074757 | PASS |
| 2163 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), estimate | -0.0169 | comparison.json comparisons[145] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, ece).difference estimate | -0.016886 | PASS |
| 2164 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), low | -0.0357 | comparison.json comparisons[145] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, ece).difference low | -0.035669 | PASS |
| 2165 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, ECE, Difference (95 percent interval), high | -0.0005 | comparison.json comparisons[145] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, ece).difference high | -0.000504 | PASS |
| 2166 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), estimate | 0.0701 | comparison.json comparisons[140] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference estimate | 0.070132 | PASS |
| 2167 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), low | 0.0489 | comparison.json comparisons[140] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference low | 0.048876 | PASS |
| 2168 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), high | 0.0941 | comparison.json comparisons[140] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, accuracy).difference high | 0.094100 | PASS |
| 2169 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), estimate | 0.0778 | comparison.json comparisons[141] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference estimate | 0.077780 | PASS |
| 2170 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), low | 0.0554 | comparison.json comparisons[141] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference low | 0.055396 | PASS |
| 2171 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), high | 0.1027 | comparison.json comparisons[141] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, macro_f1).difference high | 0.102743 | PASS |
| 2172 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0913 | comparison.json comparisons[142] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference estimate | 0.091291 | PASS |
| 2173 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), low | 0.0584 | comparison.json comparisons[142] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference low | 0.058395 | PASS |
| 2174 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), high | 0.1265 | comparison.json comparisons[142] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_increased).difference high | 0.126474 | PASS |
| 2175 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.1127 | comparison.json comparisons[143] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference estimate | 0.112712 | PASS |
| 2176 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), low | 0.0778 | comparison.json comparisons[143] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference low | 0.077782 | PASS |
| 2177 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), high | 0.1518 | comparison.json comparisons[143] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_decreased).difference high | 0.151759 | PASS |
| 2178 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0293 | comparison.json comparisons[144] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference estimate | 0.029336 | PASS |
| 2179 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), low | 0.0126 | comparison.json comparisons[144] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference low | 0.012573 | PASS |
| 2180 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), high | 0.0480 | comparison.json comparisons[144] (G1_sensitivity@unflagged, evidence_effect, s8_ei minus gemma4_31b, f1_no_difference).difference high | 0.047986 | PASS |
| 2181 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Log loss, Difference (95 percent interval), estimate | -0.8530 | comparison.json comparisons[146] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, log_loss).difference estimate | -0.852966 | PASS |
| 2182 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Log loss, Difference (95 percent interval), low | -0.9128 | comparison.json comparisons[146] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, log_loss).difference low | -0.912766 | PASS |
| 2183 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Log loss, Difference (95 percent interval), high | -0.7831 | comparison.json comparisons[146] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, log_loss).difference high | -0.783128 | PASS |
| 2184 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Brier score, Difference (95 percent interval), estimate | -0.5501 | comparison.json comparisons[147] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, brier).difference estimate | -0.550116 | PASS |
| 2185 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Brier score, Difference (95 percent interval), low | -0.5761 | comparison.json comparisons[147] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, brier).difference low | -0.576146 | PASS |
| 2186 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Brier score, Difference (95 percent interval), high | -0.5212 | comparison.json comparisons[147] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, brier).difference high | -0.521223 | PASS |
| 2187 | Table 18, all prompts, `s8_ei-raw` minus `prior`, ECE, Difference (95 percent interval), estimate | 0.0019 | comparison.json comparisons[153] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, ece).difference estimate | 0.001919 | PASS |
| 2188 | Table 18, all prompts, `s8_ei-raw` minus `prior`, ECE, Difference (95 percent interval), low | -0.0308 | comparison.json comparisons[153] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, ece).difference low | -0.030760 | PASS |
| 2189 | Table 18, all prompts, `s8_ei-raw` minus `prior`, ECE, Difference (95 percent interval), high | 0.0369 | comparison.json comparisons[153] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, ece).difference high | 0.036902 | PASS |
| 2190 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Accuracy, Difference (95 percent interval), estimate | 0.5197 | comparison.json comparisons[148] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, accuracy).difference estimate | 0.519704 | PASS |
| 2191 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Accuracy, Difference (95 percent interval), low | 0.4790 | comparison.json comparisons[148] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, accuracy).difference low | 0.479002 | PASS |
| 2192 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Accuracy, Difference (95 percent interval), high | 0.5541 | comparison.json comparisons[148] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, accuracy).difference high | 0.554125 | PASS |
| 2193 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Macro-F1, Difference (95 percent interval), estimate | 0.7423 | comparison.json comparisons[149] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, macro_f1).difference estimate | 0.742336 | PASS |
| 2194 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Macro-F1, Difference (95 percent interval), low | 0.7217 | comparison.json comparisons[149] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, macro_f1).difference low | 0.721684 | PASS |
| 2195 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Macro-F1, Difference (95 percent interval), high | 0.7602 | comparison.json comparisons[149] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, macro_f1).difference high | 0.760240 | PASS |
| 2196 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly increased, Difference (95 percent interval), estimate | 0.9358 | comparison.json comparisons[150] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_increased).difference estimate | 0.935802 | PASS |
| 2197 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly increased, Difference (95 percent interval), low | 0.9152 | comparison.json comparisons[150] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_increased).difference low | 0.915164 | PASS |
| 2198 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly increased, Difference (95 percent interval), high | 0.9529 | comparison.json comparisons[150] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_increased).difference high | 0.952917 | PASS |
| 2199 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.9365 | comparison.json comparisons[151] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_decreased).difference estimate | 0.936455 | PASS |
| 2200 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly decreased, Difference (95 percent interval), low | 0.9113 | comparison.json comparisons[151] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_decreased).difference low | 0.911256 | PASS |
| 2201 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly decreased, Difference (95 percent interval), high | 0.9577 | comparison.json comparisons[151] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_decreased).difference high | 0.957656 | PASS |
| 2202 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, no significant difference, Difference (95 percent interval), estimate | 0.3547 | comparison.json comparisons[152] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_no_difference).difference estimate | 0.354750 | PASS |
| 2203 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, no significant difference, Difference (95 percent interval), low | 0.3208 | comparison.json comparisons[152] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_no_difference).difference low | 0.320787 | PASS |
| 2204 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, no significant difference, Difference (95 percent interval), high | 0.3874 | comparison.json comparisons[152] (G1_sensitivity, evidence_effect, s8_ei-raw minus prior, f1_no_difference).difference high | 0.387380 | PASS |
| 2205 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Log loss, Difference (95 percent interval), estimate | -0.1841 | comparison.json comparisons[154] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, log_loss).difference estimate | -0.184054 | PASS |
| 2206 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Log loss, Difference (95 percent interval), low | -0.2512 | comparison.json comparisons[154] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, log_loss).difference low | -0.251214 | PASS |
| 2207 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Log loss, Difference (95 percent interval), high | -0.1164 | comparison.json comparisons[154] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, log_loss).difference high | -0.116408 | PASS |
| 2208 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Brier score, Difference (95 percent interval), estimate | -0.1016 | comparison.json comparisons[155] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, brier).difference estimate | -0.101596 | PASS |
| 2209 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Brier score, Difference (95 percent interval), low | -0.1333 | comparison.json comparisons[155] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, brier).difference low | -0.133342 | PASS |
| 2210 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Brier score, Difference (95 percent interval), high | -0.0716 | comparison.json comparisons[155] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, brier).difference high | -0.071583 | PASS |
| 2211 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, ECE, Difference (95 percent interval), estimate | 0.0042 | comparison.json comparisons[161] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, ece).difference estimate | 0.004166 | PASS |
| 2212 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, ECE, Difference (95 percent interval), low | -0.0208 | comparison.json comparisons[161] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, ece).difference low | -0.020786 | PASS |
| 2213 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, ECE, Difference (95 percent interval), high | 0.0170 | comparison.json comparisons[161] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, ece).difference high | 0.016957 | PASS |
| 2214 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), estimate | 0.0698 | comparison.json comparisons[156] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, accuracy).difference estimate | 0.069787 | PASS |
| 2215 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), low | 0.0487 | comparison.json comparisons[156] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, accuracy).difference low | 0.048738 | PASS |
| 2216 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), high | 0.0923 | comparison.json comparisons[156] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, accuracy).difference high | 0.092342 | PASS |
| 2217 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), estimate | 0.0773 | comparison.json comparisons[157] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, macro_f1).difference estimate | 0.077322 | PASS |
| 2218 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), low | 0.0543 | comparison.json comparisons[157] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, macro_f1).difference low | 0.054326 | PASS |
| 2219 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), high | 0.1022 | comparison.json comparisons[157] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, macro_f1).difference high | 0.102247 | PASS |
| 2220 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0903 | comparison.json comparisons[158] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_increased).difference estimate | 0.090336 | PASS |
| 2221 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), low | 0.0588 | comparison.json comparisons[158] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_increased).difference low | 0.058806 | PASS |
| 2222 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), high | 0.1243 | comparison.json comparisons[158] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_increased).difference high | 0.124283 | PASS |
| 2223 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.1124 | comparison.json comparisons[159] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_decreased).difference estimate | 0.112351 | PASS |
| 2224 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), low | 0.0771 | comparison.json comparisons[159] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_decreased).difference low | 0.077092 | PASS |
| 2225 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), high | 0.1521 | comparison.json comparisons[159] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_decreased).difference high | 0.152118 | PASS |
| 2226 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0293 | comparison.json comparisons[160] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_no_difference).difference estimate | 0.029280 | PASS |
| 2227 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), low | 0.0120 | comparison.json comparisons[160] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_no_difference).difference low | 0.012040 | PASS |
| 2228 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), high | 0.0476 | comparison.json comparisons[160] (G1_sensitivity, evidence_effect, s8_ei-raw minus gemma4_31b, f1_no_difference).difference high | 0.047610 | PASS |
| 2229 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Log loss, Difference (95 percent interval), estimate | -0.8773 | comparison.json comparisons[162] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, log_loss).difference estimate | -0.877268 | PASS |
| 2230 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Log loss, Difference (95 percent interval), low | -0.9201 | comparison.json comparisons[162] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, log_loss).difference low | -0.920105 | PASS |
| 2231 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Log loss, Difference (95 percent interval), high | -0.8274 | comparison.json comparisons[162] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, log_loss).difference high | -0.827397 | PASS |
| 2232 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Brier score, Difference (95 percent interval), estimate | -0.5524 | comparison.json comparisons[163] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, brier).difference estimate | -0.552420 | PASS |
| 2233 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Brier score, Difference (95 percent interval), low | -0.5767 | comparison.json comparisons[163] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, brier).difference low | -0.576715 | PASS |
| 2234 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Brier score, Difference (95 percent interval), high | -0.5247 | comparison.json comparisons[163] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, brier).difference high | -0.524697 | PASS |
| 2235 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, ECE, Difference (95 percent interval), estimate | -0.0186 | comparison.json comparisons[169] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, ece).difference estimate | -0.018626 | PASS |
| 2236 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, ECE, Difference (95 percent interval), low | -0.0461 | comparison.json comparisons[169] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, ece).difference low | -0.046084 | PASS |
| 2237 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, ECE, Difference (95 percent interval), high | 0.0187 | comparison.json comparisons[169] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, ece).difference high | 0.018718 | PASS |
| 2238 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Accuracy, Difference (95 percent interval), estimate | 0.5197 | comparison.json comparisons[164] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, accuracy).difference estimate | 0.519704 | PASS |
| 2239 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Accuracy, Difference (95 percent interval), low | 0.4790 | comparison.json comparisons[164] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, accuracy).difference low | 0.479002 | PASS |
| 2240 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Accuracy, Difference (95 percent interval), high | 0.5541 | comparison.json comparisons[164] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, accuracy).difference high | 0.554125 | PASS |
| 2241 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Macro-F1, Difference (95 percent interval), estimate | 0.7423 | comparison.json comparisons[165] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, macro_f1).difference estimate | 0.742336 | PASS |
| 2242 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Macro-F1, Difference (95 percent interval), low | 0.7217 | comparison.json comparisons[165] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, macro_f1).difference low | 0.721684 | PASS |
| 2243 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Macro-F1, Difference (95 percent interval), high | 0.7602 | comparison.json comparisons[165] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, macro_f1).difference high | 0.760240 | PASS |
| 2244 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly increased, Difference (95 percent interval), estimate | 0.9358 | comparison.json comparisons[166] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_increased).difference estimate | 0.935802 | PASS |
| 2245 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly increased, Difference (95 percent interval), low | 0.9152 | comparison.json comparisons[166] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_increased).difference low | 0.915164 | PASS |
| 2246 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly increased, Difference (95 percent interval), high | 0.9529 | comparison.json comparisons[166] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_increased).difference high | 0.952917 | PASS |
| 2247 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.9365 | comparison.json comparisons[167] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_decreased).difference estimate | 0.936455 | PASS |
| 2248 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly decreased, Difference (95 percent interval), low | 0.9113 | comparison.json comparisons[167] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_decreased).difference low | 0.911256 | PASS |
| 2249 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly decreased, Difference (95 percent interval), high | 0.9577 | comparison.json comparisons[167] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_decreased).difference high | 0.957656 | PASS |
| 2250 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, no significant difference, Difference (95 percent interval), estimate | 0.3547 | comparison.json comparisons[168] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_no_difference).difference estimate | 0.354750 | PASS |
| 2251 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, no significant difference, Difference (95 percent interval), low | 0.3208 | comparison.json comparisons[168] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_no_difference).difference low | 0.320787 | PASS |
| 2252 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, no significant difference, Difference (95 percent interval), high | 0.3874 | comparison.json comparisons[168] (G1_sensitivity, evidence_effect, s8_ei_temperature minus prior, f1_no_difference).difference high | 0.387380 | PASS |
| 2253 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Log loss, Difference (95 percent interval), estimate | -0.2084 | comparison.json comparisons[170] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, log_loss).difference estimate | -0.208355 | PASS |
| 2254 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Log loss, Difference (95 percent interval), low | -0.2690 | comparison.json comparisons[170] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, log_loss).difference low | -0.268980 | PASS |
| 2255 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Log loss, Difference (95 percent interval), high | -0.1498 | comparison.json comparisons[170] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, log_loss).difference high | -0.149845 | PASS |
| 2256 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Brier score, Difference (95 percent interval), estimate | -0.1039 | comparison.json comparisons[171] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, brier).difference estimate | -0.103900 | PASS |
| 2257 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Brier score, Difference (95 percent interval), low | -0.1357 | comparison.json comparisons[171] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, brier).difference low | -0.135694 | PASS |
| 2258 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Brier score, Difference (95 percent interval), high | -0.0741 | comparison.json comparisons[171] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, brier).difference high | -0.074131 | PASS |
| 2259 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, ECE, Difference (95 percent interval), estimate | -0.0164 | comparison.json comparisons[177] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, ece).difference estimate | -0.016378 | PASS |
| 2260 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, ECE, Difference (95 percent interval), low | -0.0369 | comparison.json comparisons[177] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, ece).difference low | -0.036914 | PASS |
| 2261 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, ECE, Difference (95 percent interval), high | -0.0007 | comparison.json comparisons[177] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, ece).difference high | -0.000722 | PASS |
| 2262 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), estimate | 0.0698 | comparison.json comparisons[172] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, accuracy).difference estimate | 0.069787 | PASS |
| 2263 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), low | 0.0487 | comparison.json comparisons[172] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, accuracy).difference low | 0.048738 | PASS |
| 2264 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Accuracy, Difference (95 percent interval), high | 0.0923 | comparison.json comparisons[172] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, accuracy).difference high | 0.092342 | PASS |
| 2265 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), estimate | 0.0773 | comparison.json comparisons[173] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, macro_f1).difference estimate | 0.077322 | PASS |
| 2266 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), low | 0.0543 | comparison.json comparisons[173] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, macro_f1).difference low | 0.054326 | PASS |
| 2267 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Macro-F1, Difference (95 percent interval), high | 0.1022 | comparison.json comparisons[173] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, macro_f1).difference high | 0.102247 | PASS |
| 2268 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), estimate | 0.0903 | comparison.json comparisons[174] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_increased).difference estimate | 0.090336 | PASS |
| 2269 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), low | 0.0588 | comparison.json comparisons[174] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_increased).difference low | 0.058806 | PASS |
| 2270 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly increased, Difference (95 percent interval), high | 0.1243 | comparison.json comparisons[174] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_increased).difference high | 0.124283 | PASS |
| 2271 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), estimate | 0.1124 | comparison.json comparisons[175] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_decreased).difference estimate | 0.112351 | PASS |
| 2272 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), low | 0.0771 | comparison.json comparisons[175] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_decreased).difference low | 0.077092 | PASS |
| 2273 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly decreased, Difference (95 percent interval), high | 0.1521 | comparison.json comparisons[175] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_decreased).difference high | 0.152118 | PASS |
| 2274 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), estimate | 0.0293 | comparison.json comparisons[176] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_no_difference).difference estimate | 0.029280 | PASS |
| 2275 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), low | 0.0120 | comparison.json comparisons[176] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_no_difference).difference low | 0.012040 | PASS |
| 2276 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, no significant difference, Difference (95 percent interval), high | 0.0476 | comparison.json comparisons[176] (G1_sensitivity, evidence_effect, s8_ei_temperature minus gemma4_31b, f1_no_difference).difference high | 0.047610 | PASS |
| 2277 | results, line 446 (power), number 1 | 0.8 | power.json cases.4.s | 0.800000 | PASS |
| 2278 | results, line 446 (power), number 2 | 1.5 | power.json cases.4.D | 1.500000 | PASS |
| 2279 | results, line 446 (power), number 3 | 0.0281 | power.json cases.4.se | 0.028074 | PASS |
| 2280 | results, line 446 (power), number 4 | -0.0550 | power.json cases.4.pass_line | -0.055025 | PASS |
| 2281 | results, line 446 (power), number 5 | 0.0787 | power.json cases.4.real_gain_for_80_percent | 0.078653 | PASS |
| 2282 | results, line 446 (power), number 6 | 0.6975 | power_validation.json s | 0.697470 | PASS |
| 2283 | results, line 446 (power), number 7 | 1.6465 | power_validation.json D | 1.646484 | PASS |
| 2284 | results, line 446 (power), number 8 | -0.1417 | power_validation.json validation_mean_difference | -0.141660 | PASS |
| 2285 | results, line 448 (zeros), number 1 | 0 | derived: zeros.json: the sum of true_answer_printed_zero over every system, split and question | 0 | PASS |
| 2286 | results, line 448 (zeros), number 2 | 247,210 | derived: zeros.json: the sum of rows over every system, split and question | 247,210 | PASS |
| 2287 | results, line 448 (zeros), number 3 | 0 | derived: comparison.json: missing_predictions, its length | 0 | PASS |
| 2288 | results, line 452 (contamination, the whole line) | verbatim | comparison.json contamination | the same text | PASS |
| 2289 | corrections 14 and 15, line 5 (cc_recount), number 1 | 1,151 | contamination_checks.json reading_the_evidence.coverage_recount_validation.covered | 1,151 | PASS |
| 2290 | corrections 14 and 15, line 5 (cc_recount), number 2 | 1,206 | contamination_checks.json reading_the_evidence.coverage_recount_validation.prompts_with_usable_offsets | 1,206 | PASS |
| 2291 | corrections 14 and 15, line 7 (cc_reading), number 1 | 1,153 | contamination_checks.json reading_the_evidence.coverage_test.covered | 1,153 | PASS |
| 2292 | corrections 14 and 15, line 7 (cc_reading), number 2 | 1,212 | contamination_checks.json reading_the_evidence.coverage_test.prompts_with_usable_offsets | 1,212 | PASS |
| 2293 | corrections 14 and 15, line 7 (cc_reading), number 3 | 95.13 | contamination_checks.json reading_the_evidence.coverage_test.coverage | 95.132013 | PASS |
| 2294 | corrections 14 and 15, line 7 (cc_reading), number 4 | 0.9488 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.estimate | 0.948829 | PASS |
| 2295 | corrections 14 and 15, line 7 (cc_reading), number 5 | 0.9335 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.low | 0.933501 | PASS |
| 2296 | corrections 14 and 15, line 7 (cc_reading), number 6 | 0.9622 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.high | 0.962236 | PASS |
| 2297 | corrections 14 and 15, line 7 (cc_reading), number 7 | 0.7966 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.estimate | 0.796610 | PASS |
| 2298 | corrections 14 and 15, line 7 (cc_reading), number 8 | 0.6571 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.low | 0.657143 | PASS |
| 2299 | corrections 14 and 15, line 7 (cc_reading), number 9 | 0.9260 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.high | 0.925960 | PASS |
| 2300 | corrections 14 and 15, line 7 (cc_reading), number 10 | 59 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.prompts | 59 | PASS |
| 2301 | corrections 14 and 15, line 7 (cc_reading), number 11 | 25 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.articles | 25 | PASS |
| 2302 | corrections 14 and 15, line 7 (cc_reading), number 12 | 0.8829 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.accuracy.estimate | 0.882914 | PASS |
| 2303 | corrections 14 and 15, line 7 (cc_reading), number 13 | 0.6610 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.accuracy.estimate | 0.661017 | PASS |
| 2304 | corrections 14 and 15, line 7 (cc_reading), number 14 | 0.9461 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.accuracy.estimate | 0.946134 | PASS |
| 2305 | corrections 14 and 15, line 7 (cc_reading), number 15 | 0.7273 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.accuracy.estimate | 0.727273 | PASS |
| 2306 | corrections 14 and 15, line 7 (cc_reading), number 16 | 55 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.prompts | 55 | PASS |
| 2307 | corrections 14 and 15, line 9 (cc_diff), number 1 | -0.1522 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.estimate | -0.152219 | PASS |
| 2308 | corrections 14 and 15, line 9 (cc_diff), number 2 | -0.2967 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.low | -0.296723 | PASS |
| 2309 | corrections 14 and 15, line 9 (cc_diff), number 3 | -0.0327 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.high | -0.032740 | PASS |
| 2310 | corrections 14 and 15, line 9 (cc_diff), number 4 | -0.2189 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.estimate | -0.218861 | PASS |
| 2311 | corrections 14 and 15, line 9 (cc_diff), number 5 | -0.4055 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.low | -0.405548 | PASS |
| 2312 | corrections 14 and 15, line 9 (cc_diff), number 6 | -0.0666 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.high | -0.066572 | PASS |
| 2313 | corrections 14 and 15, line 9 (cc_diff), number 7 | -0.2219 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.estimate | -0.221897 | PASS |
| 2314 | corrections 14 and 15, line 9 (cc_diff), number 8 | -0.4175 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.low | -0.417494 | PASS |
| 2315 | corrections 14 and 15, line 9 (cc_diff), number 9 | -0.0783 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.high | -0.078274 | PASS |
| 2316 | corrections 14 and 15, line 9 (cc_diff), number 10 | -0.1650 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.estimate | -0.164995 | PASS |
| 2317 | corrections 14 and 15, line 9 (cc_diff), number 11 | -0.3469 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.low | -0.346916 | PASS |
| 2318 | corrections 14 and 15, line 9 (cc_diff), number 12 | -0.0103 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.high | -0.010298 | PASS |
| 2319 | Table 20, test, s8_ei, all, Number | 1,218 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.prompts | 1,218 | PASS |
| 2320 | Table 20, test, s8_ei, all, Articles | 333 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.articles | 333 | PASS |
| 2321 | Table 20, test, s8_ei, all, Accuracy (95 percent interval), estimate | 0.9409 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.accuracy.estimate | 0.940887 | PASS |
| 2322 | Table 20, test, s8_ei, all, Accuracy (95 percent interval), low | 0.9246 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.accuracy.low | 0.924604 | PASS |
| 2323 | Table 20, test, s8_ei, all, Accuracy (95 percent interval), high | 0.9554 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.accuracy.high | 0.955372 | PASS |
| 2324 | Table 20, test, s8_ei, all, Macro-F1 (95 percent interval), estimate | 0.9399 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.macro_f1.estimate | 0.939909 | PASS |
| 2325 | Table 20, test, s8_ei, all, Macro-F1 (95 percent interval), low | 0.9225 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.macro_f1.low | 0.922543 | PASS |
| 2326 | Table 20, test, s8_ei, all, Macro-F1 (95 percent interval), high | 0.9544 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.all.macro_f1.high | 0.954353 | PASS |
| 2327 | Table 20, test, s8_ei, evidence inside the input, Number | 1,153 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.prompts | 1,153 | PASS |
| 2328 | Table 20, test, s8_ei, evidence inside the input, Articles | 328 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.articles | 328 | PASS |
| 2329 | Table 20, test, s8_ei, evidence inside the input, Accuracy (95 percent interval), estimate | 0.9488 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.estimate | 0.948829 | PASS |
| 2330 | Table 20, test, s8_ei, evidence inside the input, Accuracy (95 percent interval), low | 0.9335 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.low | 0.933501 | PASS |
| 2331 | Table 20, test, s8_ei, evidence inside the input, Accuracy (95 percent interval), high | 0.9622 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.accuracy.high | 0.962236 | PASS |
| 2332 | Table 20, test, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), estimate | 0.9477 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.macro_f1.estimate | 0.947746 | PASS |
| 2333 | Table 20, test, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), low | 0.9325 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.macro_f1.low | 0.932483 | PASS |
| 2334 | Table 20, test, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), high | 0.9609 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence inside the input.macro_f1.high | 0.960891 | PASS |
| 2335 | Table 20, test, s8_ei, evidence outside the input, Number | 59 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.prompts | 59 | PASS |
| 2336 | Table 20, test, s8_ei, evidence outside the input, Articles | 25 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.articles | 25 | PASS |
| 2337 | Table 20, test, s8_ei, evidence outside the input, Accuracy (95 percent interval), estimate | 0.7966 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.estimate | 0.796610 | PASS |
| 2338 | Table 20, test, s8_ei, evidence outside the input, Accuracy (95 percent interval), low | 0.6571 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.low | 0.657143 | PASS |
| 2339 | Table 20, test, s8_ei, evidence outside the input, Accuracy (95 percent interval), high | 0.9260 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.accuracy.high | 0.925960 | PASS |
| 2340 | Table 20, test, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), estimate | 0.8002 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.macro_f1.estimate | 0.800176 | PASS |
| 2341 | Table 20, test, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), low | 0.6402 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.macro_f1.low | 0.640206 | PASS |
| 2342 | Table 20, test, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), high | 0.9220 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence outside the input.macro_f1.high | 0.921964 | PASS |
| 2343 | Table 20, test, s8_ei, no usable evidence offsets, Number | 6 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.prompts | 6 | PASS |
| 2344 | Table 20, test, s8_ei, no usable evidence offsets, Articles | 5 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.articles | 5 | PASS |
| 2345 | Table 20, test, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), estimate | 0.8333 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.accuracy.estimate | 0.833333 | PASS |
| 2346 | Table 20, test, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), low | 0.4000 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.accuracy.low | 0.400000 | PASS |
| 2347 | Table 20, test, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.accuracy.high | 1.000000 | PASS |
| 2348 | Table 20, test, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), estimate | 0.6000 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.macro_f1.estimate | 0.600000 | PASS |
| 2349 | Table 20, test, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), low | 0.3333 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.macro_f1.low | 0.333333 | PASS |
| 2350 | Table 20, test, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.no usable evidence offsets.macro_f1.high | 1.000000 | PASS |
| 2351 | Table 20, test, s8_ei, evidence in the abstract, Number | 614 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.prompts | 614 | PASS |
| 2352 | Table 20, test, s8_ei, evidence in the abstract, Articles | 247 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.articles | 247 | PASS |
| 2353 | Table 20, test, s8_ei, evidence in the abstract, Accuracy (95 percent interval), estimate | 0.9577 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.accuracy.estimate | 0.957655 | PASS |
| 2354 | Table 20, test, s8_ei, evidence in the abstract, Accuracy (95 percent interval), low | 0.9404 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.accuracy.low | 0.940406 | PASS |
| 2355 | Table 20, test, s8_ei, evidence in the abstract, Accuracy (95 percent interval), high | 0.9743 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.accuracy.high | 0.974289 | PASS |
| 2356 | Table 20, test, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), estimate | 0.9575 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.macro_f1.estimate | 0.957514 | PASS |
| 2357 | Table 20, test, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), low | 0.9392 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.macro_f1.low | 0.939209 | PASS |
| 2358 | Table 20, test, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), high | 0.9744 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence in the abstract.macro_f1.high | 0.974385 | PASS |
| 2359 | Table 20, test, s8_ei, evidence not in the abstract, Number | 604 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.prompts | 604 | PASS |
| 2360 | Table 20, test, s8_ei, evidence not in the abstract, Articles | 251 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.articles | 251 | PASS |
| 2361 | Table 20, test, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), estimate | 0.9238 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.accuracy.estimate | 0.923841 | PASS |
| 2362 | Table 20, test, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), low | 0.8986 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.accuracy.low | 0.898639 | PASS |
| 2363 | Table 20, test, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), high | 0.9450 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.accuracy.high | 0.944972 | PASS |
| 2364 | Table 20, test, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), estimate | 0.9183 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.macro_f1.estimate | 0.918250 | PASS |
| 2365 | Table 20, test, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), low | 0.8908 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.macro_f1.low | 0.890828 | PASS |
| 2366 | Table 20, test, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), high | 0.9408 | contamination_checks.json reading_the_evidence.by_subset.test.s8_ei.evidence not in the abstract.macro_f1.high | 0.940824 | PASS |
| 2367 | Table 20, test, Gemma 4 31B-it, all, Number | 1,218 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.prompts | 1,218 | PASS |
| 2368 | Table 20, test, Gemma 4 31B-it, all, Articles | 333 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.articles | 333 | PASS |
| 2369 | Table 20, test, Gemma 4 31B-it, all, Accuracy (95 percent interval), estimate | 0.8711 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.accuracy.estimate | 0.871100 | PASS |
| 2370 | Table 20, test, Gemma 4 31B-it, all, Accuracy (95 percent interval), low | 0.8450 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.accuracy.low | 0.845020 | PASS |
| 2371 | Table 20, test, Gemma 4 31B-it, all, Accuracy (95 percent interval), high | 0.8949 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.accuracy.high | 0.894911 | PASS |
| 2372 | Table 20, test, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), estimate | 0.8626 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.macro_f1.estimate | 0.862587 | PASS |
| 2373 | Table 20, test, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), low | 0.8333 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.macro_f1.low | 0.833263 | PASS |
| 2374 | Table 20, test, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), high | 0.8879 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.all.macro_f1.high | 0.887909 | PASS |
| 2375 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Number | 1,153 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.prompts | 1,153 | PASS |
| 2376 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Articles | 328 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.articles | 328 | PASS |
| 2377 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), estimate | 0.8829 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.accuracy.estimate | 0.882914 | PASS |
| 2378 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), low | 0.8578 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.accuracy.low | 0.857752 | PASS |
| 2379 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), high | 0.9046 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.accuracy.high | 0.904556 | PASS |
| 2380 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), estimate | 0.8745 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.macro_f1.estimate | 0.874464 | PASS |
| 2381 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), low | 0.8461 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.macro_f1.low | 0.846108 | PASS |
| 2382 | Table 20, test, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), high | 0.8978 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence inside the input.macro_f1.high | 0.897753 | PASS |
| 2383 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Number | 59 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.prompts | 59 | PASS |
| 2384 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Articles | 25 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.articles | 25 | PASS |
| 2385 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), estimate | 0.6610 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.accuracy.estimate | 0.661017 | PASS |
| 2386 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), low | 0.4918 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.accuracy.low | 0.491796 | PASS |
| 2387 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), high | 0.8136 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.accuracy.high | 0.813637 | PASS |
| 2388 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), estimate | 0.6507 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.macro_f1.estimate | 0.650700 | PASS |
| 2389 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), low | 0.4553 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.macro_f1.low | 0.455338 | PASS |
| 2390 | Table 20, test, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), high | 0.7962 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence outside the input.macro_f1.high | 0.796188 | PASS |
| 2391 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Number | 6 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.prompts | 6 | PASS |
| 2392 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Articles | 5 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.articles | 5 | PASS |
| 2393 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), estimate | 0.6667 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.accuracy.estimate | 0.666667 | PASS |
| 2394 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), low | 0.2000 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.accuracy.low | 0.200000 | PASS |
| 2395 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.accuracy.high | 1.000000 | PASS |
| 2396 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), estimate | 0.5333 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.macro_f1.estimate | 0.533333 | PASS |
| 2397 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), low | 0.1667 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.macro_f1.low | 0.166667 | PASS |
| 2398 | Table 20, test, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.no usable evidence offsets.macro_f1.high | 1.000000 | PASS |
| 2399 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Number | 614 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.prompts | 614 | PASS |
| 2400 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Articles | 247 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.articles | 247 | PASS |
| 2401 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), estimate | 0.8811 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.accuracy.estimate | 0.881107 | PASS |
| 2402 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), low | 0.8444 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.accuracy.low | 0.844357 | PASS |
| 2403 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), high | 0.9143 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.accuracy.high | 0.914303 | PASS |
| 2404 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), estimate | 0.8796 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.macro_f1.estimate | 0.879629 | PASS |
| 2405 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), low | 0.8422 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.macro_f1.low | 0.842207 | PASS |
| 2406 | Table 20, test, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), high | 0.9139 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence in the abstract.macro_f1.high | 0.913922 | PASS |
| 2407 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Number | 604 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.prompts | 604 | PASS |
| 2408 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Articles | 251 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.articles | 251 | PASS |
| 2409 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), estimate | 0.8609 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.accuracy.estimate | 0.860927 | PASS |
| 2410 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), low | 0.8227 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.accuracy.low | 0.822717 | PASS |
| 2411 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), high | 0.8943 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.accuracy.high | 0.894296 | PASS |
| 2412 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), estimate | 0.8424 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.macro_f1.estimate | 0.842412 | PASS |
| 2413 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), low | 0.8004 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.macro_f1.low | 0.800436 | PASS |
| 2414 | Table 20, test, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), high | 0.8799 | contamination_checks.json reading_the_evidence.by_subset.test.gemma4_31b.evidence not in the abstract.macro_f1.high | 0.879911 | PASS |
| 2415 | Table 20, validation, s8_ei, all, Number | 1,237 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.prompts | 1,237 | PASS |
| 2416 | Table 20, validation, s8_ei, all, Articles | 340 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.articles | 340 | PASS |
| 2417 | Table 20, validation, s8_ei, all, Accuracy (95 percent interval), estimate | 0.9369 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.accuracy.estimate | 0.936944 | PASS |
| 2418 | Table 20, validation, s8_ei, all, Accuracy (95 percent interval), low | 0.9206 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.accuracy.low | 0.920618 | PASS |
| 2419 | Table 20, validation, s8_ei, all, Accuracy (95 percent interval), high | 0.9515 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.accuracy.high | 0.951496 | PASS |
| 2420 | Table 20, validation, s8_ei, all, Macro-F1 (95 percent interval), estimate | 0.9332 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.macro_f1.estimate | 0.933174 | PASS |
| 2421 | Table 20, validation, s8_ei, all, Macro-F1 (95 percent interval), low | 0.9160 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.macro_f1.low | 0.916046 | PASS |
| 2422 | Table 20, validation, s8_ei, all, Macro-F1 (95 percent interval), high | 0.9491 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.all.macro_f1.high | 0.949058 | PASS |
| 2423 | Table 20, validation, s8_ei, evidence inside the input, Number | 1,151 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.prompts | 1,151 | PASS |
| 2424 | Table 20, validation, s8_ei, evidence inside the input, Articles | 332 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.articles | 332 | PASS |
| 2425 | Table 20, validation, s8_ei, evidence inside the input, Accuracy (95 percent interval), estimate | 0.9461 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.accuracy.estimate | 0.946134 | PASS |
| 2426 | Table 20, validation, s8_ei, evidence inside the input, Accuracy (95 percent interval), low | 0.9312 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.accuracy.low | 0.931245 | PASS |
| 2427 | Table 20, validation, s8_ei, evidence inside the input, Accuracy (95 percent interval), high | 0.9602 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.accuracy.high | 0.960247 | PASS |
| 2428 | Table 20, validation, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), estimate | 0.9426 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.macro_f1.estimate | 0.942550 | PASS |
| 2429 | Table 20, validation, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), low | 0.9264 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.macro_f1.low | 0.926422 | PASS |
| 2430 | Table 20, validation, s8_ei, evidence inside the input, Macro-F1 (95 percent interval), high | 0.9578 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence inside the input.macro_f1.high | 0.957754 | PASS |
| 2431 | Table 20, validation, s8_ei, evidence outside the input, Number | 55 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.prompts | 55 | PASS |
| 2432 | Table 20, validation, s8_ei, evidence outside the input, Articles | 30 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.articles | 30 | PASS |
| 2433 | Table 20, validation, s8_ei, evidence outside the input, Accuracy (95 percent interval), estimate | 0.7273 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.accuracy.estimate | 0.727273 | PASS |
| 2434 | Table 20, validation, s8_ei, evidence outside the input, Accuracy (95 percent interval), low | 0.5510 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.accuracy.low | 0.550995 | PASS |
| 2435 | Table 20, validation, s8_ei, evidence outside the input, Accuracy (95 percent interval), high | 0.8689 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.accuracy.high | 0.868890 | PASS |
| 2436 | Table 20, validation, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), estimate | 0.7093 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.macro_f1.estimate | 0.709350 | PASS |
| 2437 | Table 20, validation, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), low | 0.5355 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.macro_f1.low | 0.535474 | PASS |
| 2438 | Table 20, validation, s8_ei, evidence outside the input, Macro-F1 (95 percent interval), high | 0.8519 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence outside the input.macro_f1.high | 0.851854 | PASS |
| 2439 | Table 20, validation, s8_ei, no usable evidence offsets, Number | 31 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.prompts | 31 | PASS |
| 2440 | Table 20, validation, s8_ei, no usable evidence offsets, Articles | 16 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.articles | 16 | PASS |
| 2441 | Table 20, validation, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), estimate | 0.9677 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.accuracy.estimate | 0.967742 | PASS |
| 2442 | Table 20, validation, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), low | 0.8749 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.accuracy.low | 0.874864 | PASS |
| 2443 | Table 20, validation, s8_ei, no usable evidence offsets, Accuracy (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.accuracy.high | 1.000000 | PASS |
| 2444 | Table 20, validation, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), estimate | 0.9666 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.macro_f1.estimate | 0.966607 | PASS |
| 2445 | Table 20, validation, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), low | 0.8249 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.macro_f1.low | 0.824865 | PASS |
| 2446 | Table 20, validation, s8_ei, no usable evidence offsets, Macro-F1 (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.no usable evidence offsets.macro_f1.high | 1.000000 | PASS |
| 2447 | Table 20, validation, s8_ei, evidence in the abstract, Number | 623 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.prompts | 623 | PASS |
| 2448 | Table 20, validation, s8_ei, evidence in the abstract, Articles | 253 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.articles | 253 | PASS |
| 2449 | Table 20, validation, s8_ei, evidence in the abstract, Accuracy (95 percent interval), estimate | 0.9551 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.accuracy.estimate | 0.955056 | PASS |
| 2450 | Table 20, validation, s8_ei, evidence in the abstract, Accuracy (95 percent interval), low | 0.9366 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.accuracy.low | 0.936607 | PASS |
| 2451 | Table 20, validation, s8_ei, evidence in the abstract, Accuracy (95 percent interval), high | 0.9715 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.accuracy.high | 0.971528 | PASS |
| 2452 | Table 20, validation, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), estimate | 0.9527 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.macro_f1.estimate | 0.952749 | PASS |
| 2453 | Table 20, validation, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), low | 0.9327 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.macro_f1.low | 0.932726 | PASS |
| 2454 | Table 20, validation, s8_ei, evidence in the abstract, Macro-F1 (95 percent interval), high | 0.9697 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence in the abstract.macro_f1.high | 0.969708 | PASS |
| 2455 | Table 20, validation, s8_ei, evidence not in the abstract, Number | 614 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.prompts | 614 | PASS |
| 2456 | Table 20, validation, s8_ei, evidence not in the abstract, Articles | 244 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.articles | 244 | PASS |
| 2457 | Table 20, validation, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), estimate | 0.9186 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.accuracy.estimate | 0.918567 | PASS |
| 2458 | Table 20, validation, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), low | 0.8882 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.accuracy.low | 0.888225 | PASS |
| 2459 | Table 20, validation, s8_ei, evidence not in the abstract, Accuracy (95 percent interval), high | 0.9443 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.accuracy.high | 0.944267 | PASS |
| 2460 | Table 20, validation, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), estimate | 0.9123 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.macro_f1.estimate | 0.912260 | PASS |
| 2461 | Table 20, validation, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), low | 0.8781 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.macro_f1.low | 0.878098 | PASS |
| 2462 | Table 20, validation, s8_ei, evidence not in the abstract, Macro-F1 (95 percent interval), high | 0.9405 | contamination_checks.json reading_the_evidence.by_subset.validation.s8_ei.evidence not in the abstract.macro_f1.high | 0.940478 | PASS |
| 2463 | Table 20, validation, Gemma 4 31B-it, all, Number | 1,237 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.prompts | 1,237 | PASS |
| 2464 | Table 20, validation, Gemma 4 31B-it, all, Articles | 340 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.articles | 340 | PASS |
| 2465 | Table 20, validation, Gemma 4 31B-it, all, Accuracy (95 percent interval), estimate | 0.8860 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.accuracy.estimate | 0.886015 | PASS |
| 2466 | Table 20, validation, Gemma 4 31B-it, all, Accuracy (95 percent interval), low | 0.8607 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.accuracy.low | 0.860661 | PASS |
| 2467 | Table 20, validation, Gemma 4 31B-it, all, Accuracy (95 percent interval), high | 0.9095 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.accuracy.high | 0.909464 | PASS |
| 2468 | Table 20, validation, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), estimate | 0.8747 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.macro_f1.estimate | 0.874682 | PASS |
| 2469 | Table 20, validation, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), low | 0.8461 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.macro_f1.low | 0.846120 | PASS |
| 2470 | Table 20, validation, Gemma 4 31B-it, all, Macro-F1 (95 percent interval), high | 0.8996 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.all.macro_f1.high | 0.899619 | PASS |
| 2471 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Number | 1,151 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.prompts | 1,151 | PASS |
| 2472 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Articles | 332 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.articles | 332 | PASS |
| 2473 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), estimate | 0.8923 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.accuracy.estimate | 0.892268 | PASS |
| 2474 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), low | 0.8660 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.accuracy.low | 0.865953 | PASS |
| 2475 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Accuracy (95 percent interval), high | 0.9165 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.accuracy.high | 0.916466 | PASS |
| 2476 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), estimate | 0.8815 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.macro_f1.estimate | 0.881490 | PASS |
| 2477 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), low | 0.8538 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.macro_f1.low | 0.853762 | PASS |
| 2478 | Table 20, validation, Gemma 4 31B-it, evidence inside the input, Macro-F1 (95 percent interval), high | 0.9076 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence inside the input.macro_f1.high | 0.907612 | PASS |
| 2479 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Number | 55 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.prompts | 55 | PASS |
| 2480 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Articles | 30 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.articles | 30 | PASS |
| 2481 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), estimate | 0.7273 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.accuracy.estimate | 0.727273 | PASS |
| 2482 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), low | 0.5490 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.accuracy.low | 0.549004 | PASS |
| 2483 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Accuracy (95 percent interval), high | 0.8772 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.accuracy.high | 0.877202 | PASS |
| 2484 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), estimate | 0.6800 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.macro_f1.estimate | 0.679989 | PASS |
| 2485 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), low | 0.5042 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.macro_f1.low | 0.504180 | PASS |
| 2486 | Table 20, validation, Gemma 4 31B-it, evidence outside the input, Macro-F1 (95 percent interval), high | 0.8641 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence outside the input.macro_f1.high | 0.864079 | PASS |
| 2487 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Number | 31 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.prompts | 31 | PASS |
| 2488 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Articles | 16 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.articles | 16 | PASS |
| 2489 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), estimate | 0.9355 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.accuracy.estimate | 0.935484 | PASS |
| 2490 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), low | 0.8077 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.accuracy.low | 0.807692 | PASS |
| 2491 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Accuracy (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.accuracy.high | 1.000000 | PASS |
| 2492 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), estimate | 0.9069 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.macro_f1.estimate | 0.906926 | PASS |
| 2493 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), low | 0.6310 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.macro_f1.low | 0.631022 | PASS |
| 2494 | Table 20, validation, Gemma 4 31B-it, no usable evidence offsets, Macro-F1 (95 percent interval), high | 1.0000 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.no usable evidence offsets.macro_f1.high | 1.000000 | PASS |
| 2495 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Number | 623 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.prompts | 623 | PASS |
| 2496 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Articles | 253 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.articles | 253 | PASS |
| 2497 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), estimate | 0.9021 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.accuracy.estimate | 0.902087 | PASS |
| 2498 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), low | 0.8647 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.accuracy.low | 0.864681 | PASS |
| 2499 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Accuracy (95 percent interval), high | 0.9316 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.accuracy.high | 0.931603 | PASS |
| 2500 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), estimate | 0.8951 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.macro_f1.estimate | 0.895134 | PASS |
| 2501 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), low | 0.8561 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.macro_f1.low | 0.856108 | PASS |
| 2502 | Table 20, validation, Gemma 4 31B-it, evidence in the abstract, Macro-F1 (95 percent interval), high | 0.9263 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence in the abstract.macro_f1.high | 0.926333 | PASS |
| 2503 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Number | 614 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.prompts | 614 | PASS |
| 2504 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Articles | 244 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.articles | 244 | PASS |
| 2505 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), estimate | 0.8697 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.accuracy.estimate | 0.869707 | PASS |
| 2506 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), low | 0.8339 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.accuracy.low | 0.833865 | PASS |
| 2507 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Accuracy (95 percent interval), high | 0.9063 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.accuracy.high | 0.906253 | PASS |
| 2508 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), estimate | 0.8533 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.macro_f1.estimate | 0.853292 | PASS |
| 2509 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), low | 0.8129 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.macro_f1.low | 0.812866 | PASS |
| 2510 | Table 20, validation, Gemma 4 31B-it, evidence not in the abstract, Macro-F1 (95 percent interval), high | 0.8944 | contamination_checks.json reading_the_evidence.by_subset.validation.gemma4_31b.evidence not in the abstract.macro_f1.high | 0.894407 | PASS |
| 2511 | corrections 14 and 15, line 42 (cc_pubmedqa), number 1 | 10 | derived: contamination_checks.json: pubmedqa.matched_articles.test, its length | 10 | PASS |
| 2512 | corrections 14 and 15, line 42 (cc_pubmedqa), number 2 | 11 | derived: contamination_checks.json: pubmedqa.matched_articles.validation, its length | 11 | PASS |
| 2513 | corrections 14 and 15, line 42 (cc_pubmedqa), number 3 | 60 | contamination_checks.json settings.sentence_min_chars | 60 | PASS |
| 2514 | corrections 14 and 15, line 42 (cc_pubmedqa), number 4 | 16 | derived: contamination_checks.json: pubmedqa.matches with source pqa_artificial | 16 | PASS |
| 2515 | corrections 14 and 15, line 42 (cc_pubmedqa), number 5 | 5 | derived: contamination_checks.json: pubmedqa.matches with source pqa_unlabeled | 5 | PASS |
| 2516 | corrections 14 and 15, line 42 (cc_pubmedqa), number 6 | 0 | derived: contamination_checks.json: pubmedqa.matches with source pqa_labeled | 0 | PASS |
| 2517 | corrections 14 and 15, line 42 (cc_pubmedqa), number 7 | 10 | derived: contamination_checks.json: pubmedqa.matched_articles_whose_abstract_arm_b_trained_on.test, its length | 10 | PASS |
| 2518 | corrections 14 and 15, line 42 (cc_pubmedqa), number 8 | 11 | derived: contamination_checks.json: pubmedqa.matched_articles_whose_abstract_arm_b_trained_on.validation, its length | 11 | PASS |
| 2519 | corrections 14 and 15, line 42 (cc_pubmedqa), number 9 | 0.8378 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.estimate | 0.837838 | PASS |
| 2520 | corrections 14 and 15, line 42 (cc_pubmedqa), number 10 | 0.6667 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.low | 0.666667 | PASS |
| 2521 | corrections 14 and 15, line 42 (cc_pubmedqa), number 11 | 0.9445 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.high | 0.944482 | PASS |
| 2522 | corrections 14 and 15, line 42 (cc_pubmedqa), number 12 | 37 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.prompts | 37 | PASS |
| 2523 | corrections 14 and 15, line 42 (cc_pubmedqa), number 13 | 0.9441 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.accuracy.estimate | 0.944115 | PASS |
| 2524 | corrections 14 and 15, line 42 (cc_pubmedqa), number 14 | 1,181 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.prompts | 1,181 | PASS |
| 2525 | corrections 14 and 15, line 42 (cc_pubmedqa), number 15 | -0.1063 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.estimate | -0.106277 | PASS |
| 2526 | corrections 14 and 15, line 42 (cc_pubmedqa), number 16 | -0.2817 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.low | -0.281745 | PASS |
| 2527 | corrections 14 and 15, line 42 (cc_pubmedqa), number 17 | 0.0051 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.high | 0.005136 | PASS |
| 2528 | corrections 14 and 15, line 42 (cc_pubmedqa), number 18 | 0.0051 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.high | 0.005136 | PASS |
| 2529 | corrections 14 and 15, line 44 (cc_splits), number 1 | 0 | derived: contamination_checks.json: ei_splits.flagged_articles.test, its length | 0 | PASS |
| 2530 | corrections 14 and 15, line 44 (cc_splits), number 2 | 0 | derived: contamination_checks.json: ei_splits.flagged_articles.validation, its length | 0 | PASS |
| 2531 | corrections 14 and 15, line 44 (cc_splits), number 3 | 2,657 | contamination_checks.json ei_splits.train_articles | 2,657 | PASS |
| 2532 | Table 21, test, articles that share sentences with a PubMedQA abstract, Prompts | 37 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.prompts | 37 | PASS |
| 2533 | Table 21, test, articles that share sentences with a PubMedQA abstract, Articles | 10 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.articles | 10 | PASS |
| 2534 | Table 21, test, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), estimate | 0.8378 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.estimate | 0.837838 | PASS |
| 2535 | Table 21, test, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), low | 0.6667 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.low | 0.666667 | PASS |
| 2536 | Table 21, test, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), high | 0.9445 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged.accuracy.high | 0.944482 | PASS |
| 2537 | Table 21, test, the rest, Prompts | 1,181 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.prompts | 1,181 | PASS |
| 2538 | Table 21, test, the rest, Articles | 323 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.articles | 323 | PASS |
| 2539 | Table 21, test, the rest, Accuracy (95 percent interval), estimate | 0.9441 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.accuracy.estimate | 0.944115 | PASS |
| 2540 | Table 21, test, the rest, Accuracy (95 percent interval), low | 0.9265 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.accuracy.low | 0.926469 | PASS |
| 2541 | Table 21, test, the rest, Accuracy (95 percent interval), high | 0.9585 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.the rest.accuracy.high | 0.958514 | PASS |
| 2542 | Table 21, validation, articles that share sentences with a PubMedQA abstract, Prompts | 36 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged.prompts | 36 | PASS |
| 2543 | Table 21, validation, articles that share sentences with a PubMedQA abstract, Articles | 11 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged.articles | 11 | PASS |
| 2544 | Table 21, validation, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), estimate | 0.9722 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged.accuracy.estimate | 0.972222 | PASS |
| 2545 | Table 21, validation, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), low | 0.9130 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged.accuracy.low | 0.913012 | PASS |
| 2546 | Table 21, validation, articles that share sentences with a PubMedQA abstract, Accuracy (95 percent interval), high | 1.0000 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged.accuracy.high | 1.000000 | PASS |
| 2547 | Table 21, validation, the rest, Prompts | 1,201 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.the rest.prompts | 1,201 | PASS |
| 2548 | Table 21, validation, the rest, Articles | 329 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.the rest.articles | 329 | PASS |
| 2549 | Table 21, validation, the rest, Accuracy (95 percent interval), estimate | 0.9359 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.the rest.accuracy.estimate | 0.935887 | PASS |
| 2550 | Table 21, validation, the rest, Accuracy (95 percent interval), low | 0.9183 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.the rest.accuracy.low | 0.918332 | PASS |
| 2551 | Table 21, validation, the rest, Accuracy (95 percent interval), high | 0.9513 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.the rest.accuracy.high | 0.951327 | PASS |
| 2552 | Table 21, test, articles that share a title or sentences with an EI training article, none, Prompts | 0 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.flagged.prompts | 0 | PASS |
| 2553 | Table 21, test, articles that share a title or sentences with an EI training article, none, Articles | 0 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.flagged.articles | 0 | PASS |
| 2554 | Table 21, test, the rest, Prompts | 1,218 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.the rest.prompts | 1,218 | PASS |
| 2555 | Table 21, test, the rest, Articles | 333 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.the rest.articles | 333 | PASS |
| 2556 | Table 21, test, the rest, Accuracy (95 percent interval), estimate | 0.9409 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.the rest.accuracy.estimate | 0.940887 | PASS |
| 2557 | Table 21, test, the rest, Accuracy (95 percent interval), low | 0.9246 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.the rest.accuracy.low | 0.924604 | PASS |
| 2558 | Table 21, test, the rest, Accuracy (95 percent interval), high | 0.9554 | contamination_checks.json ei_splits.s8_ei_test_flagged_or_not.the rest.accuracy.high | 0.955372 | PASS |
| 2559 | Table 22, test, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), estimate | -0.1522 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.estimate | -0.152219 | PASS |
| 2560 | Table 22, test, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), low | -0.2967 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.low | -0.296723 | PASS |
| 2561 | Table 22, test, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), high | -0.0327 | contamination_checks.json reading_the_evidence.differences.test.s8_ei.outside minus inside.high | -0.032740 | PASS |
| 2562 | Table 22, test, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), estimate | -0.2219 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.estimate | -0.221897 | PASS |
| 2563 | Table 22, test, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), low | -0.4175 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.low | -0.417494 | PASS |
| 2564 | Table 22, test, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), high | -0.0783 | contamination_checks.json reading_the_evidence.differences.test.gemma4_31b.outside minus inside.high | -0.078274 | PASS |
| 2565 | Table 22, validation, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), estimate | -0.2189 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.estimate | -0.218861 | PASS |
| 2566 | Table 22, validation, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), low | -0.4055 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.low | -0.405548 | PASS |
| 2567 | Table 22, validation, s8_ei, where the evidence is outside the input, minus inside, Estimate (95 percent interval), high | -0.0666 | contamination_checks.json reading_the_evidence.differences.validation.s8_ei.outside minus inside.high | -0.066572 | PASS |
| 2568 | Table 22, validation, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), estimate | -0.1650 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.estimate | -0.164995 | PASS |
| 2569 | Table 22, validation, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), low | -0.3469 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.low | -0.346916 | PASS |
| 2570 | Table 22, validation, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Estimate (95 percent interval), high | -0.0103 | contamination_checks.json reading_the_evidence.differences.validation.gemma4_31b.outside minus inside.high | -0.010298 | PASS |
| 2571 | Table 22, test, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), estimate | -0.1063 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.estimate | -0.106277 | PASS |
| 2572 | Table 22, test, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), low | -0.2817 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.low | -0.281745 | PASS |
| 2573 | Table 22, test, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), high | 0.0051 | contamination_checks.json pubmedqa.s8_ei_test_matched_or_not.flagged minus the rest.high | 0.005136 | PASS |
| 2574 | Table 22, validation, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), estimate | 0.0363 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged minus the rest.estimate | 0.036335 | PASS |
| 2575 | Table 22, validation, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), low | -0.0234 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged minus the rest.low | -0.023369 | PASS |
| 2576 | Table 22, validation, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Estimate (95 percent interval), high | 0.0760 | contamination_checks.json pubmedqa.s8_ei_validation_matched_or_not.flagged minus the rest.high | 0.075971 | PASS |

## Every reading

| # | Where | As written | From the source | Check |
|---|---|---|---|---|
| 1 | Table 3, `s8_ei`, Evidence Inference, Chosen | temperature | temperature (calibration_s8.json s8_ei.evidence_effect.chosen) | PASS |
| 2 | Table 3, `s8_ei_1`, Evidence Inference, Chosen | temperature | temperature (calibration_s8.json s8_ei_1.evidence_effect.chosen) | PASS |
| 3 | Table 3, `s8_ei_2`, Evidence Inference, Chosen | temperature | temperature (calibration_s8.json s8_ei_2.evidence_effect.chosen) | PASS |
| 4 | Table 3, `s8_ei_3`, Evidence Inference, Chosen | temperature | temperature (calibration_s8.json s8_ei_3.evidence_effect.chosen) | PASS |
| 5 | Table 3, `gemma4_31b`, Evidence Inference, Chosen | temperature_intercepts | temperature_intercepts (calibration_s8.json gemma4_31b.evidence_effect.chosen) | PASS |
| 6 | Table 3, `gemma4_31b`, PubMedQA, Chosen | temperature_intercepts | temperature_intercepts (calibration_s8.json gemma4_31b.pubmedqa_answer.chosen) | PASS |
| 7 | Table 3, `s8_a3`, PubMedQA, Chosen | temperature_intercepts | temperature_intercepts (calibration_s8.json s8_a3.pubmedqa_answer.chosen) | PASS |
| 8 | Table 3, `s8_a3`, Readmission, Chosen | temperature_intercepts | temperature_intercepts (calibration_s8.json s8_a3.readmit_window.chosen) | PASS |
| 9 | Table 7, `s8_ei` minus `prior`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/log_loss) | PASS |
| 10 | Table 7, `s8_ei` minus `prior`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/brier) | PASS |
| 11 | Table 7, `s8_ei` minus `prior`, ECE, Reading | no difference was detected; excludes below -0.0461 and above 0.0187 | no difference was detected; excludes below -0.0461 and above 0.0187 (from the interval of C/G1/evidence_effect/s8_ei/prior/ece) | PASS |
| 12 | Table 7, `s8_ei` minus `prior`, Accuracy, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/accuracy) | PASS |
| 13 | Table 7, `s8_ei` minus `prior`, Macro-F1, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/macro_f1) | PASS |
| 14 | Table 7, `s8_ei` minus `prior`, F1, significantly increased, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/f1_increased) | PASS |
| 15 | Table 7, `s8_ei` minus `prior`, F1, significantly decreased, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/f1_decreased) | PASS |
| 16 | Table 7, `s8_ei` minus `prior`, F1, no significant difference, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/prior/f1_no_difference) | PASS |
| 17 | Table 7, `s8_ei` minus `gemma4_31b`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/log_loss) | PASS |
| 18 | Table 7, `s8_ei` minus `gemma4_31b`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/brier) | PASS |
| 19 | Table 7, `s8_ei` minus `gemma4_31b`, ECE, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/ece) | PASS |
| 20 | Table 7, `s8_ei` minus `gemma4_31b`, Accuracy, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/accuracy) | PASS |
| 21 | Table 7, `s8_ei` minus `gemma4_31b`, Macro-F1, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/macro_f1) | PASS |
| 22 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/f1_increased) | PASS |
| 23 | Table 7, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/f1_decreased) | PASS |
| 24 | Table 7, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Reading | s8_ei better | s8_ei better (from the interval of C/G1/evidence_effect/s8_ei/gemma4_31b/f1_no_difference) | PASS |
| 25 | Table 8, 1, ECE, Result | PASS | PASS (comparison.json release_bar criteria.1.result) | PASS |
| 26 | Table 8, 2, Log loss, s8_ei minus the prior, Result | PASS | PASS (comparison.json release_bar criteria.2.result) | PASS |
| 27 | Table 8, 3, Log loss, s8_ei minus Gemma 4 31B-it, Result | PASS | PASS (comparison.json release_bar criteria.3.result) | PASS |
| 28 | Table 8, 4, Macro-F1, s8_ei against Gemma 4 31B-it, Result | PASS | PASS (comparison.json release_bar criteria.4.result) | PASS |
| 29 | Table 8, The bar, all four, Result | PASS | PASS (comparison.json release_bar overall) | PASS |
| 30 | Table 9, `s8_ei` minus `s8_ei_1`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/log_loss) | PASS |
| 31 | Table 9, `s8_ei` minus `s8_ei_1`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/brier) | PASS |
| 32 | Table 9, `s8_ei` minus `s8_ei_1`, ECE, Reading | no difference was detected; excludes below -0.0100 and above 0.0095 | no difference was detected; excludes below -0.0100 and above 0.0095 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/ece) | PASS |
| 33 | Table 9, `s8_ei` minus `s8_ei_1`, Accuracy, Reading | no difference was detected; excludes below -0.0034 and above 0.0093 | no difference was detected; excludes below -0.0034 and above 0.0093 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/accuracy) | PASS |
| 34 | Table 9, `s8_ei` minus `s8_ei_1`, Macro-F1, Reading | no difference was detected; excludes below -0.0038 and above 0.0104 | no difference was detected; excludes below -0.0038 and above 0.0104 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/macro_f1) | PASS |
| 35 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly increased, Reading | no difference was detected; excludes below -0.0030 and above 0.0126 | no difference was detected; excludes below -0.0030 and above 0.0126 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/f1_increased) | PASS |
| 36 | Table 9, `s8_ei` minus `s8_ei_1`, F1, significantly decreased, Reading | no difference was detected; excludes below -0.0090 and above 0.0156 | no difference was detected; excludes below -0.0090 and above 0.0156 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/f1_decreased) | PASS |
| 37 | Table 9, `s8_ei` minus `s8_ei_1`, F1, no significant difference, Reading | no difference was detected; excludes below -0.0051 and above 0.0067 | no difference was detected; excludes below -0.0051 and above 0.0067 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_1/f1_no_difference) | PASS |
| 38 | Table 9, `s8_ei` minus `s8_ei_2`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/log_loss) | PASS |
| 39 | Table 9, `s8_ei` minus `s8_ei_2`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/brier) | PASS |
| 40 | Table 9, `s8_ei` minus `s8_ei_2`, ECE, Reading | no difference was detected; excludes below -0.0137 and above 0.0085 | no difference was detected; excludes below -0.0137 and above 0.0085 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/ece) | PASS |
| 41 | Table 9, `s8_ei` minus `s8_ei_2`, Accuracy, Reading | no difference was detected; excludes below -0.0015 and above 0.0099 | no difference was detected; excludes below -0.0015 and above 0.0099 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/accuracy) | PASS |
| 42 | Table 9, `s8_ei` minus `s8_ei_2`, Macro-F1, Reading | no difference was detected; excludes below -0.0017 and above 0.0099 | no difference was detected; excludes below -0.0017 and above 0.0099 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/macro_f1) | PASS |
| 43 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly increased, Reading | no difference was detected; excludes below -0.0045 and above 0.0088 | no difference was detected; excludes below -0.0045 and above 0.0088 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/f1_increased) | PASS |
| 44 | Table 9, `s8_ei` minus `s8_ei_2`, F1, significantly decreased, Reading | no difference was detected; excludes below -0.0051 and above 0.0145 | no difference was detected; excludes below -0.0051 and above 0.0145 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/f1_decreased) | PASS |
| 45 | Table 9, `s8_ei` minus `s8_ei_2`, F1, no significant difference, Reading | no difference was detected; excludes below -0.0009 and above 0.0112 | no difference was detected; excludes below -0.0009 and above 0.0112 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_2/f1_no_difference) | PASS |
| 46 | Table 9, `s8_ei` minus `s8_ei_3`, Log loss, Reading | no difference was detected; excludes below -0.0073 and above 0.0062 | no difference was detected; excludes below -0.0073 and above 0.0062 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/log_loss) | PASS |
| 47 | Table 9, `s8_ei` minus `s8_ei_3`, Brier score, Reading | no difference was detected; excludes below -0.0028 and above 0.0046 | no difference was detected; excludes below -0.0028 and above 0.0046 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/brier) | PASS |
| 48 | Table 9, `s8_ei` minus `s8_ei_3`, ECE, Reading | no difference was detected; excludes below -0.0115 and above 0.0119 | no difference was detected; excludes below -0.0115 and above 0.0119 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/ece) | PASS |
| 49 | Table 9, `s8_ei` minus `s8_ei_3`, Accuracy, Reading | no difference was detected; excludes below -0.0025 and above 0.0079 | no difference was detected; excludes below -0.0025 and above 0.0079 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/accuracy) | PASS |
| 50 | Table 9, `s8_ei` minus `s8_ei_3`, Macro-F1, Reading | no difference was detected; excludes below -0.0033 and above 0.0076 | no difference was detected; excludes below -0.0033 and above 0.0076 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/macro_f1) | PASS |
| 51 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly increased, Reading | no difference was detected; excludes below -0.0042 and above 0.0115 | no difference was detected; excludes below -0.0042 and above 0.0115 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/f1_increased) | PASS |
| 52 | Table 9, `s8_ei` minus `s8_ei_3`, F1, significantly decreased, Reading | no difference was detected; excludes below -0.0085 and above 0.0057 | no difference was detected; excludes below -0.0085 and above 0.0057 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/f1_decreased) | PASS |
| 53 | Table 9, `s8_ei` minus `s8_ei_3`, F1, no significant difference, Reading | no difference was detected; excludes below -0.0011 and above 0.0097 | no difference was detected; excludes below -0.0011 and above 0.0097 (from the interval of C/G2/evidence_effect/s8_ei/s8_ei_3/f1_no_difference) | PASS |
| 54 | Table 10, DeYoung et al. 2020, the full-text pipeline (BioMed RoBERTa), strict, The published value against s8_ei's interval | excluded: below the interval | excluded: below the interval (comparison.json g3 published.deyoung_2020_pipeline.s8_ei) | PASS |
| 55 | Table 10, DeYoung et al. 2020, with the right evidence given, strict, The published value against s8_ei's interval | excluded: below the interval | excluded: below the interval (comparison.json g3 published.deyoung_2020_oracle.s8_ei) | PASS |
| 56 | Table 10, Buchmann et al. 2024, LED, the lower end of its range, lenient, The published value against s8_ei's interval | excluded: below the interval | excluded: below the interval (comparison.json g3 published_lenient.buchmann_2024_led_low.s8_ei) | PASS |
| 57 | Table 10, Buchmann et al. 2024, LED, the upper end of its range, lenient, The published value against s8_ei's interval | excluded: below the interval | excluded: below the interval (comparison.json g3 published_lenient.buchmann_2024_led_high.s8_ei) | PASS |
| 58 | Table 10, Buchmann et al. 2024, LongT5, lenient, The published value against s8_ei's interval | excluded: below the interval | excluded: below the interval (comparison.json g3 published_lenient.buchmann_2024_longt5.s8_ei) | PASS |
| 59 | Table 12, `s7_b3` minus `arm_laya`, Log loss, Reading | v0.4 better | v0.4 better (from the interval of C/G4/pubmedqa_answer/s7_b3/arm_laya/log_loss) | PASS |
| 60 | Table 12, `s7_b3` minus `arm_laya`, Brier score, Reading | v0.4 better | v0.4 better (from the interval of C/G4/pubmedqa_answer/s7_b3/arm_laya/brier) | PASS |
| 61 | Table 12, `s7_b3` minus `arm_laya`, ECE, Reading | no difference was detected; excludes below -0.0340 and above 0.0320 | no difference was detected; excludes below -0.0340 and above 0.0320 (from the interval of C/G4/pubmedqa_answer/s7_b3/arm_laya/ece) | PASS |
| 62 | Table 12, `s7_b3` minus `arm_laya`, Accuracy, Reading | v0.4 better | v0.4 better (from the interval of C/G4/pubmedqa_answer/s7_b3/arm_laya/accuracy) | PASS |
| 63 | Table 12, `s7_b3` minus `arm_laya`, Macro-F1, Reading | v0.4 better | v0.4 better (from the interval of C/G4/pubmedqa_answer/s7_b3/arm_laya/macro_f1) | PASS |
| 64 | Table 12, `s7_b3` minus `gemma4_31b`, Log loss, Reading | no difference was detected; excludes below -0.0785 and above 0.0137 | no difference was detected; excludes below -0.0785 and above 0.0137 (from the interval of C/G4/pubmedqa_answer/s7_b3/gemma4_31b/log_loss) | PASS |
| 65 | Table 12, `s7_b3` minus `gemma4_31b`, Brier score, Reading | no difference was detected; excludes below -0.0351 and above 0.0209 | no difference was detected; excludes below -0.0351 and above 0.0209 (from the interval of C/G4/pubmedqa_answer/s7_b3/gemma4_31b/brier) | PASS |
| 66 | Table 12, `s7_b3` minus `gemma4_31b`, ECE, Reading | no difference was detected; excludes below -0.0469 and above 0.0208 | no difference was detected; excludes below -0.0469 and above 0.0208 (from the interval of C/G4/pubmedqa_answer/s7_b3/gemma4_31b/ece) | PASS |
| 67 | Table 12, `s7_b3` minus `gemma4_31b`, Accuracy, Reading | no difference was detected; excludes below -0.0140 and above 0.0400 | no difference was detected; excludes below -0.0140 and above 0.0400 (from the interval of C/G4/pubmedqa_answer/s7_b3/gemma4_31b/accuracy) | PASS |
| 68 | Table 12, `s7_b3` minus `gemma4_31b`, Macro-F1, Reading | no difference was detected; excludes below -0.0561 and above 0.0237 | no difference was detected; excludes below -0.0561 and above 0.0237 (from the interval of C/G4/pubmedqa_answer/s7_b3/gemma4_31b/macro_f1) | PASS |
| 69 | Table 12, `s8_a3` minus `s7_a3`, Log loss, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/s7_a3/log_loss) | PASS |
| 70 | Table 12, `s8_a3` minus `s7_a3`, Brier score, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/s7_a3/brier) | PASS |
| 71 | Table 12, `s8_a3` minus `s7_a3`, ECE, Reading | no difference was detected; excludes below -0.0292 and above 0.0450 | no difference was detected; excludes below -0.0292 and above 0.0450 (from the interval of C/G4/pubmedqa_answer/s8_a3/s7_a3/ece) | PASS |
| 72 | Table 12, `s8_a3` minus `s7_a3`, Accuracy, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/s7_a3/accuracy) | PASS |
| 73 | Table 12, `s8_a3` minus `s7_a3`, Macro-F1, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/s7_a3/macro_f1) | PASS |
| 74 | Table 12, `s8_a3` minus `arm_laya`, Log loss, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/arm_laya/log_loss) | PASS |
| 75 | Table 12, `s8_a3` minus `arm_laya`, Brier score, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/arm_laya/brier) | PASS |
| 76 | Table 12, `s8_a3` minus `arm_laya`, ECE, Reading | no difference was detected; excludes below -0.0235 and above 0.0513 | no difference was detected; excludes below -0.0235 and above 0.0513 (from the interval of C/G4/pubmedqa_answer/s8_a3/arm_laya/ece) | PASS |
| 77 | Table 12, `s8_a3` minus `arm_laya`, Accuracy, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/arm_laya/accuracy) | PASS |
| 78 | Table 12, `s8_a3` minus `arm_laya`, Macro-F1, Reading | s8_a3 better | s8_a3 better (from the interval of C/G4/pubmedqa_answer/s8_a3/arm_laya/macro_f1) | PASS |
| 79 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Log loss, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/log_loss) | PASS |
| 80 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Brier score, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/brier) | PASS |
| 81 | Table 15, `readmission_pool` minus `lightgbm_tuned`, ECE, Reading | no difference was detected; excludes below -0.0092 and above 0.0103 | no difference was detected; excludes below -0.0092 and above 0.0103 (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/ece) | PASS |
| 82 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Accuracy, Reading | no difference was detected; excludes below -0.0061 and above 0.0059 | no difference was detected; excludes below -0.0061 and above 0.0059 (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/accuracy) | PASS |
| 83 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/macro_f1) | PASS |
| 84 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Macro-F1, decision weights, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/macro_f1_tuned) | PASS |
| 85 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Quadratic weighted kappa, Reading | no difference was detected; excludes below -0.0048 and above 0.0240 | no difference was detected; excludes below -0.0048 and above 0.0240 (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/quadratic_weighted_kappa) | PASS |
| 86 | Table 15, `readmission_pool` minus `lightgbm_tuned`, AUROC, 30 days, Reading | no difference was detected; excludes below -0.0063 and above 0.0160 | no difference was detected; excludes below -0.0063 and above 0.0160 (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/auroc_30d) | PASS |
| 87 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/calibration_intercept_30d) | PASS |
| 88 | Table 15, `readmission_pool` minus `lightgbm_tuned`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/lightgbm_tuned/calibration_slope_30d) | PASS |
| 89 | Table 15, `readmission_pool` minus `blend_v0_1`, Log loss, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/log_loss) | PASS |
| 90 | Table 15, `readmission_pool` minus `blend_v0_1`, Brier score, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/brier) | PASS |
| 91 | Table 15, `readmission_pool` minus `blend_v0_1`, ECE, Reading | no difference was detected; excludes below -0.0086 and above 0.0093 | no difference was detected; excludes below -0.0086 and above 0.0093 (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/ece) | PASS |
| 92 | Table 15, `readmission_pool` minus `blend_v0_1`, Accuracy, Reading | no difference was detected; excludes below -0.0014 and above 0.0094 | no difference was detected; excludes below -0.0014 and above 0.0094 (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/accuracy) | PASS |
| 93 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/macro_f1) | PASS |
| 94 | Table 15, `readmission_pool` minus `blend_v0_1`, Macro-F1, decision weights, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/macro_f1_tuned) | PASS |
| 95 | Table 15, `readmission_pool` minus `blend_v0_1`, Quadratic weighted kappa, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/quadratic_weighted_kappa) | PASS |
| 96 | Table 15, `readmission_pool` minus `blend_v0_1`, AUROC, 30 days, Reading | no difference was detected; excludes below -0.0087 and above 0.0110 | no difference was detected; excludes below -0.0087 and above 0.0110 (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/auroc_30d) | PASS |
| 97 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/calibration_intercept_30d) | PASS |
| 98 | Table 15, `readmission_pool` minus `blend_v0_1`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/blend_v0_1/calibration_slope_30d) | PASS |
| 99 | Table 15, `readmission_pool` minus `arm_laya`, Log loss, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/log_loss) | PASS |
| 100 | Table 15, `readmission_pool` minus `arm_laya`, Brier score, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/brier) | PASS |
| 101 | Table 15, `readmission_pool` minus `arm_laya`, ECE, Reading | no difference was detected; excludes below -0.0158 and above 0.0077 | no difference was detected; excludes below -0.0158 and above 0.0077 (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/ece) | PASS |
| 102 | Table 15, `readmission_pool` minus `arm_laya`, Accuracy, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/accuracy) | PASS |
| 103 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/macro_f1) | PASS |
| 104 | Table 15, `readmission_pool` minus `arm_laya`, Macro-F1, decision weights, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/macro_f1_tuned) | PASS |
| 105 | Table 15, `readmission_pool` minus `arm_laya`, Quadratic weighted kappa, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/quadratic_weighted_kappa) | PASS |
| 106 | Table 15, `readmission_pool` minus `arm_laya`, AUROC, 30 days, Reading | the pool better | the pool better (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/auroc_30d) | PASS |
| 107 | Table 15, `readmission_pool` minus `arm_laya`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/calibration_intercept_30d) | PASS |
| 108 | Table 15, `readmission_pool` minus `arm_laya`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/readmission_pool/arm_laya/calibration_slope_30d) | PASS |
| 109 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Log loss, Reading | xgboost_tuned better | xgboost_tuned better (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/log_loss) | PASS |
| 110 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Brier score, Reading | xgboost_tuned better | xgboost_tuned better (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/brier) | PASS |
| 111 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, ECE, Reading | no difference was detected; excludes below -0.0084 and above 0.0108 | no difference was detected; excludes below -0.0084 and above 0.0108 (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/ece) | PASS |
| 112 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Accuracy, Reading | no difference was detected; excludes below -0.0079 and above 0.0034 | no difference was detected; excludes below -0.0079 and above 0.0034 (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/accuracy) | PASS |
| 113 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, Reading | xgboost_tuned better | xgboost_tuned better (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/macro_f1) | PASS |
| 114 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Reading | no difference was detected; excludes below -0.0032 and above 0.0168 | no difference was detected; excludes below -0.0032 and above 0.0168 (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/macro_f1_tuned) | PASS |
| 115 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Reading | xgboost_tuned better | xgboost_tuned better (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/quadratic_weighted_kappa) | PASS |
| 116 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Reading | no difference was detected; excludes below -0.0102 and above 0.0107 | no difference was detected; excludes below -0.0102 and above 0.0107 (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/auroc_30d) | PASS |
| 117 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/calibration_intercept_30d) | PASS |
| 118 | Table 15, `xgboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/xgboost_tuned/lightgbm_tuned/calibration_slope_30d) | PASS |
| 119 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Log loss, Reading | catboost_tuned better | catboost_tuned better (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/log_loss) | PASS |
| 120 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Brier score, Reading | catboost_tuned better | catboost_tuned better (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/brier) | PASS |
| 121 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, ECE, Reading | no difference was detected; excludes below -0.0096 and above 0.0109 | no difference was detected; excludes below -0.0096 and above 0.0109 (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/ece) | PASS |
| 122 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Accuracy, Reading | no difference was detected; excludes below -0.0077 and above 0.0054 | no difference was detected; excludes below -0.0077 and above 0.0054 (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/accuracy) | PASS |
| 123 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, Reading | catboost_tuned better | catboost_tuned better (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/macro_f1) | PASS |
| 124 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Macro-F1, decision weights, Reading | catboost_tuned better | catboost_tuned better (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/macro_f1_tuned) | PASS |
| 125 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Quadratic weighted kappa, Reading | no difference was detected; excludes below -0.0077 and above 0.0223 | no difference was detected; excludes below -0.0077 and above 0.0223 (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/quadratic_weighted_kappa) | PASS |
| 126 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, AUROC, 30 days, Reading | no difference was detected; excludes below -0.0124 and above 0.0140 | no difference was detected; excludes below -0.0124 and above 0.0140 (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/auroc_30d) | PASS |
| 127 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/calibration_intercept_30d) | PASS |
| 128 | Table 15, `catboost_tuned` minus `lightgbm_tuned`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/catboost_tuned/lightgbm_tuned/calibration_slope_30d) | PASS |
| 129 | Table 15, `s8_a3` minus `s7_a3`, Log loss, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/log_loss) | PASS |
| 130 | Table 15, `s8_a3` minus `s7_a3`, Brier score, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/brier) | PASS |
| 131 | Table 15, `s8_a3` minus `s7_a3`, ECE, Reading | no difference was detected; excludes below -0.0226 and above 0.0045 | no difference was detected; excludes below -0.0226 and above 0.0045 (from the interval of C/G5/readmit_window/s8_a3/s7_a3/ece) | PASS |
| 132 | Table 15, `s8_a3` minus `s7_a3`, Accuracy, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/accuracy) | PASS |
| 133 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/macro_f1) | PASS |
| 134 | Table 15, `s8_a3` minus `s7_a3`, Macro-F1, decision weights, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/macro_f1_tuned) | PASS |
| 135 | Table 15, `s8_a3` minus `s7_a3`, Quadratic weighted kappa, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/quadratic_weighted_kappa) | PASS |
| 136 | Table 15, `s8_a3` minus `s7_a3`, AUROC, 30 days, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/s7_a3/auroc_30d) | PASS |
| 137 | Table 15, `s8_a3` minus `s7_a3`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/s8_a3/s7_a3/calibration_intercept_30d) | PASS |
| 138 | Table 15, `s8_a3` minus `s7_a3`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/s8_a3/s7_a3/calibration_slope_30d) | PASS |
| 139 | Table 15, `s8_a3` minus `arm_laya`, Log loss, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/arm_laya/log_loss) | PASS |
| 140 | Table 15, `s8_a3` minus `arm_laya`, Brier score, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/arm_laya/brier) | PASS |
| 141 | Table 15, `s8_a3` minus `arm_laya`, ECE, Reading | no difference was detected; excludes below -0.0151 and above 0.0148 | no difference was detected; excludes below -0.0151 and above 0.0148 (from the interval of C/G5/readmit_window/s8_a3/arm_laya/ece) | PASS |
| 142 | Table 15, `s8_a3` minus `arm_laya`, Accuracy, Reading | no difference was detected; excludes below -0.0031 and above 0.0110 | no difference was detected; excludes below -0.0031 and above 0.0110 (from the interval of C/G5/readmit_window/s8_a3/arm_laya/accuracy) | PASS |
| 143 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/arm_laya/macro_f1) | PASS |
| 144 | Table 15, `s8_a3` minus `arm_laya`, Macro-F1, decision weights, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/arm_laya/macro_f1_tuned) | PASS |
| 145 | Table 15, `s8_a3` minus `arm_laya`, Quadratic weighted kappa, Reading | s8_a3 better | s8_a3 better (from the interval of C/G5/readmit_window/s8_a3/arm_laya/quadratic_weighted_kappa) | PASS |
| 146 | Table 15, `s8_a3` minus `arm_laya`, AUROC, 30 days, Reading | no difference was detected; excludes below -0.0131 and above 0.0154 | no difference was detected; excludes below -0.0131 and above 0.0154 (from the interval of C/G5/readmit_window/s8_a3/arm_laya/auroc_30d) | PASS |
| 147 | Table 15, `s8_a3` minus `arm_laya`, Calibration intercept, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/s8_a3/arm_laya/calibration_intercept_30d) | PASS |
| 148 | Table 15, `s8_a3` minus `arm_laya`, Calibration slope, 30 days, Reading | not ranked | not ranked (from the interval of C/G5/readmit_window/s8_a3/arm_laya/calibration_slope_30d) | PASS |
| 149 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/log_loss) | PASS |
| 150 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/brier) | PASS |
| 151 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, ECE, Reading | no difference was detected; excludes below -0.0453 and above 0.0180 | no difference was detected; excludes below -0.0453 and above 0.0180 (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/ece) | PASS |
| 152 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Accuracy, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/accuracy) | PASS |
| 153 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, Macro-F1, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/macro_f1) | PASS |
| 154 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly increased, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/f1_increased) | PASS |
| 155 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, significantly decreased, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/f1_decreased) | PASS |
| 156 | Table 18, without the flagged prompts, `s8_ei` minus `prior`, F1, no significant difference, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/prior/f1_no_difference) | PASS |
| 157 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Log loss, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/log_loss) | PASS |
| 158 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Brier score, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/brier) | PASS |
| 159 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, ECE, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/ece) | PASS |
| 160 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Accuracy, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/accuracy) | PASS |
| 161 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, Macro-F1, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/macro_f1) | PASS |
| 162 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly increased, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/f1_increased) | PASS |
| 163 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, significantly decreased, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/f1_decreased) | PASS |
| 164 | Table 18, without the flagged prompts, `s8_ei` minus `gemma4_31b`, F1, no significant difference, Reading | s8_ei better | s8_ei better (from the interval of C/G1_sensitivity@unflagged/evidence_effect/s8_ei/gemma4_31b/f1_no_difference) | PASS |
| 165 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Log loss, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/log_loss) | PASS |
| 166 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Brier score, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/brier) | PASS |
| 167 | Table 18, all prompts, `s8_ei-raw` minus `prior`, ECE, Reading | no difference was detected; excludes below -0.0308 and above 0.0369 | no difference was detected; excludes below -0.0308 and above 0.0369 (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/ece) | PASS |
| 168 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Accuracy, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/accuracy) | PASS |
| 169 | Table 18, all prompts, `s8_ei-raw` minus `prior`, Macro-F1, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/macro_f1) | PASS |
| 170 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly increased, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/f1_increased) | PASS |
| 171 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, significantly decreased, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/f1_decreased) | PASS |
| 172 | Table 18, all prompts, `s8_ei-raw` minus `prior`, F1, no significant difference, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/prior/f1_no_difference) | PASS |
| 173 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Log loss, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/log_loss) | PASS |
| 174 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Brier score, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/brier) | PASS |
| 175 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, ECE, Reading | no difference was detected; excludes below -0.0208 and above 0.0170 | no difference was detected; excludes below -0.0208 and above 0.0170 (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/ece) | PASS |
| 176 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Accuracy, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/accuracy) | PASS |
| 177 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, Macro-F1, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/macro_f1) | PASS |
| 178 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly increased, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/f1_increased) | PASS |
| 179 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, significantly decreased, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/f1_decreased) | PASS |
| 180 | Table 18, all prompts, `s8_ei-raw` minus `gemma4_31b`, F1, no significant difference, Reading | s8_ei-raw better | s8_ei-raw better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei-raw/gemma4_31b/f1_no_difference) | PASS |
| 181 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Log loss, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/log_loss) | PASS |
| 182 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Brier score, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/brier) | PASS |
| 183 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, ECE, Reading | no difference was detected; excludes below -0.0461 and above 0.0187 | no difference was detected; excludes below -0.0461 and above 0.0187 (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/ece) | PASS |
| 184 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Accuracy, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/accuracy) | PASS |
| 185 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, Macro-F1, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/macro_f1) | PASS |
| 186 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly increased, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/f1_increased) | PASS |
| 187 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, significantly decreased, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/f1_decreased) | PASS |
| 188 | Table 18, all prompts, `s8_ei_temperature` minus `prior`, F1, no significant difference, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/prior/f1_no_difference) | PASS |
| 189 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Log loss, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/log_loss) | PASS |
| 190 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Brier score, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/brier) | PASS |
| 191 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, ECE, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/ece) | PASS |
| 192 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Accuracy, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/accuracy) | PASS |
| 193 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, Macro-F1, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/macro_f1) | PASS |
| 194 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly increased, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/f1_increased) | PASS |
| 195 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, significantly decreased, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/f1_decreased) | PASS |
| 196 | Table 18, all prompts, `s8_ei_temperature` minus `gemma4_31b`, F1, no significant difference, Reading | s8_ei_temperature better | s8_ei_temperature better (from the interval of C/G1_sensitivity/evidence_effect/s8_ei_temperature/gemma4_31b/f1_no_difference) | PASS |
| 197 | Table 22, test, s8_ei, where the evidence is outside the input, minus inside, Reading | lower outside the input | lower outside the input (from the interval of CC/reading_the_evidence/differences/test/s8_ei/outside minus inside) | PASS |
| 198 | Table 22, test, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Reading | lower outside the input | lower outside the input (from the interval of CC/reading_the_evidence/differences/test/gemma4_31b/outside minus inside) | PASS |
| 199 | Table 22, validation, s8_ei, where the evidence is outside the input, minus inside, Reading | lower outside the input | lower outside the input (from the interval of CC/reading_the_evidence/differences/validation/s8_ei/outside minus inside) | PASS |
| 200 | Table 22, validation, Gemma 4 31B-it, where the evidence is outside the input, minus inside, Reading | lower outside the input | lower outside the input (from the interval of CC/reading_the_evidence/differences/validation/gemma4_31b/outside minus inside) | PASS |
| 201 | Table 22, test, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Reading | no difference was detected; excludes below -0.2817 and above 0.0051 | no difference was detected; excludes below -0.2817 and above 0.0051 (from the interval of CC/pubmedqa/s8_ei_test_matched_or_not/flagged minus the rest) | PASS |
| 202 | Table 22, validation, s8_ei, articles that share sentences with a PubMedQA abstract, minus the rest, Reading | no difference was detected; excludes below -0.0234 and above 0.0760 | no difference was detected; excludes below -0.0234 and above 0.0760 (from the interval of CC/pubmedqa/s8_ei_validation_matched_or_not/flagged minus the rest) | PASS |

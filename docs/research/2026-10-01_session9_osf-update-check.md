# Check of the OSF update draft for session 9 (step 5)
Study: ed-decision-model (open medical typed-decision encoder built on Laya)   Prepared: 1 October 2026
Sources: docs/osf_update_session9.md (the draft, SHA-256 8edebf1be1e2523a6f201a3f095e899a845e72b89525120aac509181630503a1); docs/prompt_session9.md (the plan with corrections 1 to 7, SHA-256 82dca6b37f83f8bf6d4937d99acc4b45a6979187ab2c649113069a517892a6f9); docs/osf_prereg_fiorillo_v0_4.md; docs/fiorillo_v0_4.md; docs/log.md (session 9's entries, lines 177 to 181, SHA-256 c685a208cb16ff7c830d7c88954e5d316ccc4ba3217c7ed47f7615bca21b72be); results/fiorillo_v0_5/power.json and coverage_long.json; the code named in the task (working tree at commit a660e83 plus the uncommitted session 9 files)
Status: complete

## How the check was done

- Written by the agent "osfreview", which did not write the draft. The draft was read and not edited.
- Every section of the draft was compared with the plan as corrected (corrections 1 to 7), and every rule it states with the code that carries it out. The code was read, not run. No training and no test was run.
- Every number in the draft was traced to the plan, docs/fiorillo_v0_4.md, docs/log.md or results/fiorillo_v0_5/power.json.
- Not opened: results/fiorillo_v0_5/records_manifest.json, and any record, label or prediction file with "test" in its name under data/processed/, results/ or models/.
- Line numbers below are the files' as they stand now. The draft's lines are its own.

## Findings

### 1. The EI specialist's training data are stated more broadly than correction 7 (f) allows
- Location: "Why this update", paragraph "v0.4's licenses" (draft line 16): "Its EI specialist learns only from articles whose own statement".
- What is wrong: v0.5's EI runs start from s7_b3's folder, and s7_b3 learned from PubMedQA's abstracts, whose own licenses were not checked. Read alone, the sentence says the specialist learned only from CC BY, CC0 or public-domain articles. Correction 7 (f) was written to keep that claim out; the draft's own last section says otherwise. The same broad wording is in "What session 9 does" ("trained only on clean articles") and at the start of "The license-clean records" ("learns only from clean articles").
- Where it shows: the plan, correction 7 (f) (line 289): "every EI article v0.5's EI specialist learned from, or was calibrated and chosen on, is CC BY, CC0 or the public domain; PubMedQA's abstracts rest on PubMedQA's license". The older, broader wording in the plan's purpose (line 10) predates correction 7. Code: scripts/pod_session9.sh, train_run, `--init "$MODELS/s7_b3"`.
- Fix: "Every EI article its EI specialist learns from is CC BY, CC0 or the public domain by its own statement. The specialist starts from s7_b3, which learned from PubMedQA's abstracts (see the last section)." Add "EI" before "articles" in the other two places.
- Severity: should fix.

### 2. The disclosure says more than the log records, and leaves out what it records
- Location: "What in session 9 has touched EI's test split so far", third bullet (line 198): "It opened no record, label or prediction file."
- What is wrong: The log says the agent "opened no test record, label or prediction file". The draft claims no record file of any split, which the log does not support. The bullet also leaves out four things the log records: it happened during step 3; it was the agent that wrote scripts/pod_session9.sh; it printed the section to learn its structure; and every agent prompt since forbids reading that section.
- Where it shows: docs/log.md line 181 (the disclosure, written at 13:25 UTC).
- Fix: "During step 3, the agent that wrote scripts/pod_session9.sh printed the counts section of results/fiorillo_v0_5/records_manifest.json to learn its structure. That section holds the builder's tallies, by answer, of the test record files. It opened no test record, label or prediction file. Every agent prompt since forbids reading that section."
- Severity: should fix.

### 3. Two summary sentences leave out the printed test tallies
- Location: same section, closing paragraph (line 200): "Apart from the builder, nothing in session 9 has read a test label." Also "Why this update", last paragraph (line 22): "I file it before anything in session 9 reads EI's test labels".
- What is wrong: The bullet above them records that an agent printed the per-answer tallies of the test records, which are counts of the test labels, before this update is filed. The two sentences name only the builder, so they say less than the record. Also, the builder's manifest command prints the same counts, test files included, whenever it runs (src/tasks/ei_session9.py, build_manifest, line 529), so they were printed when the manifest was built in step 2. The plan's exception covers the builder ("which writes and counts them"), but the log does not say the counts were printed.
- Where it shows: the plan, THE OSF UPDATE (line 149); docs/log.md line 181.
- Fix: "Apart from the builder, which wrote and counted them, and the printing of its tallies described above, nothing in session 9 has read a test label or a count of them." Make the sentence in "Why this update" match. Before filing, Claude checks whether the step 2 print was read and records the answer in docs/log.md.
- Severity: should fix.

### 4. The builder's account leaves out per-prompt evidence marks on the test split
- Location: same section, first bullet (line 196): "It chose the 6,144-token input of all 12,501 prompts".
- What is wrong: Choosing the 6,144-token input also read each test prompt's accepted evidence spans and marked whether one lies inside the kept text. The marks are written to the working file data/processed/session9/parts/selection_long_test.jsonl. coverage_long.json summarizes training and validation only. The registration disclosed the same mark for session 8's builder ("marked whether an accepted evidence span lies inside the kept text; ... never summarized"). The bullet also leaves out the in-memory replay of rebuild, which read the test records again and reproduced all 15 files (docs/log.md line 179), and the tokens per record of the 6,144-token test file that the manifest holds.
- Where it shows: src/tasks/ei_session9.py, select_long, line 264 (`"covered": ei8.coverage_of(...)`), coverage_table (train, validation and train_clean only), build_manifest, lines 494 to 496 (tokens of long/evidence_effect_test.jsonl); docs/osf_prereg_fiorillo_v0_4.md line 74.
- Fix: Add: "For each test prompt, the selection also marked whether an accepted evidence span lies inside the 6,144-token input. This is saved in its working files and was never summarized. The builder also recorded the 6,144-token test records' tokens per record, and a replay of its rebuild, in memory, reproduced every file byte for byte." Claude first confirms that nothing summarized the test marks.
- Severity: should fix.

### 5. The title input's history does not match the log
- Location: same bullet (line 196): "Once the label was read on its own line" and "would have given the 38 test prompts".
- What is wrong: The log says the first version looked for a line that is exactly "TITLE:", and the fix was "reading the label on its line", that is, taking the title from the label's line. "On its own line" reads as the opposite. The log also says the rule as written "then ran on into the abstract's text" and that "the checks and the manifest were rebuilt after it". So title inputs holding abstract text for those prompts were built, then replaced before any prediction. "Would have given" says they never were.
- Where it shows: docs/log.md line 179.
- Fix: "The first version looked for a line holding only 'TITLE:' and found no title for 949 test prompts. Reading the title on the label's line fixed that, but the rule as written then ran on into the abstract's text, which often states the result, for 38 test prompts of 12 test articles. Correction 6 changed the rule, and the title input was rebuilt, before any prediction. Correction 7 (e) states the rule as built."
- Severity: should fix.

### 6. Session 8's reads of the test split are understated
- Location: "EI's test split was read in session 8" (line 184): "EI's test split was read once in session 8, after that session's freeze".
- What is wrong: The registration itself says that before session 8's freeze the test labels were read by the record builder, and, before session 8's plan, by a check of the annotations. "Once" and "after that session's freeze" contradict it. The plan says only that the split "was read in session 8, for v0.4's bar, its comparisons and its checks".
- Where it shows: the plan, POWER AND THE BAR (line 128); docs/osf_prereg_fiorillo_v0_4.md line 46 ("Before the freeze, the test labels are read only by the record builder ...") and its Explanation of existing data.
- Fix: "In session 8, EI's test split was read by the records builder before that session's freeze, and by a check of its annotations before that session's plan, as this registration states. After the freeze (release_candidate.json written at 23:00:15 UTC on 30 September 2026; test predictions from 01:13:53 UTC on 1 October), it was read for v0.4's bar, its comparisons (G1 to G6 in session 8's plan) and its checks (session 8's corrections 14 and 15, descriptive and not registered)."
- Severity: should fix.

### 7. The summary of corrections 14 and 15 keeps one reading and drops the other
- Location: "Why this update", paragraph "The checks v0.4 needs" (line 18): "That is consistent with reading the evidence, but those checks cannot show".
- What is wrong: The v0.4 report gives two readings of the result: it "is consistent with their reading the evidence; it is also consistent with those prompts being harder for any reader, and it does not rule out that a model recalls some answers." The draft keeps only the first, a stronger claim than its source.
- Where it shows: docs/fiorillo_v0_4.md line 650 ("Checks not pre-registered (corrections 14 and 15)").
- Fix: "That is consistent with reading the evidence. It is also consistent with those prompts being harder for any reader, and it does not rule out that a model recalls some answers. Those checks cannot show whether the pretraining text held EI's annotations."
- Severity: should fix.

### 8. The code cannot finish the two-run path that the update promises
- Location: "The recipes and the recipe rule", the weight average bullet (line 118): "s9_ei_wavg is built from those two (rank 32, alpha 64". Also the order, item 5 (line 32): "of the winning recipe's three runs (if it is not base)".
- What is wrong: The update promises correction 7 (b)'s path, which is reached whenever the cost rule's options drop the winner's second run (correction 2). weight_average_session9.py build and the driver's wavg action accept two runs, but two later steps do not. The calibrate stage passes the winner's runs to handoff_curves_session9.py fit-rule, which takes exactly three: argparse refuses two (--runs nargs=3) and load_runs stops on any other count. The testpreds stage stops unless recipe_selection.json names exactly three runs. Step 15's `curves --rule` also takes three. On that path the chain stops before the freeze, and a fix then would change K7's code after this update is filed.
- Where it shows: the plan, correction 7 (b) (line 285) and correction 2 (lines 256 and 259). Code: src/eval/handoff_curves_session9.py lines 249, 697 (curves --rule, three run files), 704 and 713; scripts/pod_session9.sh line 1172 (fit-rule in calibrate) and line 1216 (testpreds: `if [ "${#RUNS[@]}" != 3 ]`).
- Fix: Before filing, make I-35's fit and apply and the testpreds stage accept two runs, with tests, and say in the update that with two runs I-35's run spread is over those two. Otherwise say in the update that the two-run path stops the chain and is decided by a correction.
- Severity: should fix.

### 9. How a combination left with one run is judged is not stated
- Location: "The recipes and the recipe rule" (line 120): "A candidate that the cost rule leaves with one run is judged".
- What is wrong: The combination's second run is one of correction 2's options, but the update does not say how a one-run combination is judged, or how K3 compares a one-run candidate. The code decides both. The combination's one run must score below the best candidate's first run by more than m, and the same applies when the combination has two runs but the best candidate has one. K3 compares a one-run candidate with the base run of the same seed. The plan does not fix either, and the first can choose the recipe.
- Where it shows: src/eval/recipe_session9.py lines 358 to 370; src/eval/compare_session9.py docstring, K3 (line 48); the plan, correction 2 (line 259).
- Fix: Append a correction to the plan that states both as the code does, then state them in the update.
- Severity: minor.

### 10. The cost rule's projection is stated loosely
- Location: "The cost rule", "The projection" (line 178): "the winning recipe's further runs at the longest candidate's rate".
- What is wrong: Since correction 2 there is one further run, the third. The code takes the pilots' training rates times 2.70 / 2.86 (session 8's ratio of run to pilot rate), not the measured rates alone. Its fixed items add "the weight average's build" (10 minutes), which the plan does not list, and leave out the reading checks on v0.5, which the plan counts.
- Where it shows: src/eval/recipe_session9.py lines 455 to 464 (FIXED_HOURS, S8_RUN_TO_PILOT); the plan, THE PILOT AND THE COST RULE (line 122, "the reading checks about 30 minutes in all") and correction 2 (line 256).
- Fix: "the winning recipe's third run at the longest candidate's rate", and "at the pilots' training rates, scaled by session 8's ratio of run to pilot rate"; then either list the fixed items as the code counts them or make the code's list match the plan's.
- Severity: minor.

### 11. The cost per 1,000 questions does not come only from this session's run summaries
- Location: "The comparisons", "Also reported" (line 70): "from the forward-pass seconds in the test-prediction stage's run summaries".
- What is wrong: The code reads session 9's run summaries only for v0.5's EI specialist. s7_b3's seconds, and Gemma 4 31B-it's at 4,096 tokens, are session 8's testpreds'. The readmission pool's cost is not computed. The plan has the update's wording (COMPARISONS, line 146), so plan and code differ.
- Where it shows: src/eval/compare_session9.py, cost_section, lines 541 to 599 (lines 573, 584 to 588).
- Fix: "for v0.5's EI specialist, from this session's test-prediction stage; for s7_b3 and for Gemma 4 31B-it at 4,096 tokens, from session 8's; each job alone on the GPU, at this session's hourly rate; not computed for readmission." Or change the code to match the plan.
- Severity: minor.

### 12. "Every number has a 95 percent interval" has more exceptions in the code
- Location: "The comparisons" (line 53): "Every number has a 95 percent interval from 1,000 resamples".
- What is wrong: The code also leaves the costs (derived from timings) and calibration_s9.json's out-of-fold families table (K9) without intervals, and K4 and K7 carry their own files' intervals. The plan has the same sentence (line 133).
- Where it shows: src/eval/compare_session9.py docstring, line 14.
- Fix: Add "the costs and K9's out-of-fold table" to the exceptions.
- Severity: minor.

### 13. The title check's paired difference does not say which prompts it uses
- Location: "The readings of the reading checks, and the stop", title (line 140): "The paired difference in macro-F1, title minus none, with its interval."
- What is wrong: The code computes the difference on the 1,209 prompts in both inputs, with the same resamples, not none's 1,218 against title's 1,209. The update does not say which.
- Where it shows: src/eval/reading_checks_session9.py lines 572 to 587.
- Fix: "The paired difference in macro-F1, title minus none, on the 1,209 prompts in both inputs, with its interval."
- Severity: minor.

### 14. Item 6 of the order lists the analyses in another order than step 15 runs them
- Location: the order, item 6 (line 33): "6. The analyses, after the freeze: v0.5's bar (K1".
- What is wrong: Step 15 runs calibrate_session9.py apply, trees_session9.py test, the reading checks on v0.5, the hand-off on test, and then compare_session9.py, which computes the bar and K1 to K10. Item 6 puts the bar first. The plan's own list (line 149) is in the draft's order and the bar depends on none of the others, so this is wording only.
- Where it shows: the plan, STEP 15 (line 229).
- Fix: Give step 15's order in item 6, or say that item 6 is a list, not an order.
- Severity: minor.

### 15. A report-level test comparison is not named
- Location: "Why this update", last paragraph (line 22): "This update names every analysis that session 9 runs on EI's test split".
- What is wrong: STEP 16 puts the published EI results of v0.4's G3 section beside v0.5's specialist and s8_ei. The update does not name this. compare_session9.py computes no G3 comparison or lenient score.
- Where it shows: the plan, STEP 16 (line 232).
- Fix: Add to "Also reported": "the published EI results of my v0.4 report's G3 section, beside v0.5's EI specialist and s8_ei (descriptive; no new score)".
- Severity: minor.

### 16. The list of what touched the test split needs its time, step 6, and two more items checked against the log
- Location: "What in session 9 has touched EI's test split so far" (lines 195 and 200): "Up to this draft:" and "The tests of this session's code use made-up inputs."
- What is wrong: (a) The plan changed after this draft (13:23 UTC), and step 6 ran from 13:22 to 13:25 UTC; the list does not say what time it covers. (b) Session 9's log entries do not record that the tests used made-up inputs; the plan requires it (STEP 4, line 196), but the record is silent, so the sentence says more than the log. (c) Correction 7 (e) records a comparison of correction 6's words with the rule as built on all 3,351 article files, the test articles among them (one differs, PMC3881817). That read article texts outside the builder, and the list leaves it out.
- Where it shows: docs/log.md lines 180 and 181; the plan, correction 7 (e) (line 288).
- Fix: "Up to <time> UTC on <date>:". Add "Step 6 read validation files only." Record the tests' made-up inputs in docs/log.md, or drop the sentence. Add "Comparing correction 6's words with the rule as built read the title lines of every EI article file, the test articles among them, and no answer."
- Severity: minor.

### 17. One use of session 8's test result is missing from the list
- Location: "EI's test split was read in session 8" (line 184): "This plan uses some of them:".
- What is wrong: Correction 4's first reason for not adding Qwen3-8B-Base is s8_ei's test macro-F1, 0.940, close to the ceiling of EI's labels. The list names three uses and not this one.
- Where it shows: the plan, correction 4 (line 274).
- Fix: Add "My decision not to try a larger base model (correction 4) rests partly on s8_ei's test macro-F1."
- Severity: minor.

### 18. The plan's SHA-256 in the preamble is out of date
- Location: preamble, outside the update's text (line 8): "the plan's SHA-256 when this draft was written".
- What is wrong: The plan changed after the draft: correction 7 (e) gained words at 13:23 UTC (docs/log.md line 181). Its SHA-256 is now 82dca6b37f83f8bf6d4937d99acc4b45a6979187ab2c649113069a517892a6f9. The recorded 581e54e4... could not be reproduced from the current file (taking out the 13:23 words gives neither value).
- Fix: Record the plan's SHA-256, and its commit, at filing.
- Severity: minor.

### 19. The status note understates what correction 7 (f)'s alternative would change
- Location: Status line (line 4): "The last section follows correction 7 (f) as written."
- What is wrong: If Johann chooses correction 7 (f)'s alternative (EI runs from Qwen3-4B-Base; PubMedQA and s7_b3 out of v0.5), more than the last section changes: "from s7_b3's folder" in the recipes, "PubMedQA's question goes to s7_b3", K5, the hand-off pair with s7_b3, s7_b3's calibration line, and the readmission pool's members.
- Where it shows: the plan, correction 7 (f) (line 289).
- Fix: "If I choose otherwise before filing, that choice is a further correction, and every section that names s7_b3 changes with it."
- Severity: minor.

### 20. "This registration's embargo" is unclear if the update is filed as a new registration
- Location: "Why this update" (line 20) and "Power and the bar" (line 154): "I end this registration's embargo the same day".
- What is wrong: If OSF takes the update only as a new registration, "this registration" names the new one, and https://osf.io/kaxmn/ stays embargoed. The plan says "The OSF embargo is ended early on the same day".
- Where it shows: the plan, THE RELEASE (line 155).
- Fix: "I end the embargo of https://osf.io/kaxmn/, and of this update if it is filed as a separate registration, the same day."
- Severity: minor.

### 21. With swap averaging kept, I-35's run spread uses single-pass runs
- Location: "The hand-off analyses", the learned rule (line 80): "the standard deviation across its runs of the probability each run gives".
- What is wrong: recipe_session9.py writes no swap-averaged file for each run, so when swap averaging is kept the calibrate stage passes each run's single-pass predictions, and I-35's run spread, and the top answer it is read at, come from those, while the specialist's own answers are swap-averaged. The update does not say so.
- Where it shows: scripts/pod_session9.sh lines 1160 to 1167; src/eval/recipe_session9.py (only `<system>_sa` is written).
- Fix: Say "(with swap averaging kept, each run's single-pass probabilities)", or write each run's swap-averaged validation file and use it.
- Severity: minor.

## Count by severity

- blocker: 0
- should fix: 8 (findings 1 to 8)
- minor: 13 (findings 9 to 21)

## Rules of the plan checked and found correct

Each was checked against the plan as corrected and against the code named.
- EI's license-clean records: the clean test, CC BY-SA left out, Europe PMC's errors (47 and 12), the clean counts (5,279 prompts in 1,431 articles; 654 in 179), the monitoring subset (200 in 113 articles, random.Random(20260927) over IDs sorted as text), the test records, and the word-statistics exception (src/tasks/ei_session9.py filter_clean and monitor_lines). Correction 7 (a)'s rerun on 3,348 articles, with no class changed.
- The recipes: the shared settings (from s7_b3 with --init, --keep last, --freeze-steps 0; rank 16, alpha 32, dropout 0.05; 1e-4 for the projection and head; cosine to 1e-6; clipping 1.0; weight decay 0.01; 32 records a step; --eval-every 80), base (2 epochs, 4,096 tokens, 5e-5, seeds 20260927 to 20260929), swap (exchange, same-text rule, 10,557 records, 1 epoch), long (6,144 tokens with 16 to spare; 99.03 and 98.68 percent against 98.91), lr1 (1e-4), lr2 (2e-4), two runs each with the first runs first (src/eval/recipe_session9.py recipe_settings and Decider; src/train/train_qwen_head.py defaults; scripts/pod_session9.sh train_run).
- The long pilot: peak at most 76 GiB with --headroom-gib 0, the higher training rate, NOT BUILT drops long.
- The recipe rule with corrections 2, 3 and 7 (b): the score (session 8's rule, 5 folds, 20 repeats, seed 20260927), m (the base runs' spread, at least 0.005), the two-run pass test, lr1 and lr2 as one change, the combination and its pass test, the winner, the third run's seed, s9_ei, the weight average (rank 48 and alpha 96; two runs rank 32 and alpha 64; kept unless above by more than m; the 1e-5 check), swap averaging (raw probabilities, the exchange, the margin, test inputs from the reading checks), the one-run candidate rule, and recipe_selection.json's contents.
- Calibration: the four families and their forms (temperature_intercepts' first intercept 0; vector's 5 free parameters; dirichlet_odir's 25 pairs chosen by inner 5-fold cross-validation from the seed), the tie within 1e-12 to the family listed first, the fit on all 654, the choice after the swap-averaging decision, the same family for K8's and K10's other forms, session 8's rule (an exact tie to temperature) for every other system, gemma4_31b_long on 1,237, BioGPT-Large on 150, and the ECE limit (the larger of 0.05 and the 95th percentile of 5,000 draws at 1,218, seed 20260927, from the clean validation predictions) (src/eval/calibrate_session9.py, calibration_family_cv.py, calibrate_session7.py, ece_noise_floor.py).
- The reading checks: the inputs (none 1,218; title 1,209 by corrections 6 and 7 (e); swap 1,218 at each length; the same-text rule), the systems and their calibrations, every statistic with 1,000 article resamples, the time-point subset, the readings and their exact texts (0.60 and 0.75; 0.05; 0.85 and 0.70), the swap caveats and correction 3's single pass, the references, and the stop above 0.85 with exit 3, the wait and the 3-hour rule (src/eval/reading_checks_session9.py; src/tasks/ei_session9.py title_of and swapped; the checks_s8 stage).
- Gemma on the longer input: the condition (recipe length 6,144), session 8's code unchanged, the 8,192-token limit, the calibration on 1,237, and its use in criterion 3, K1 and K7.
- The readmission pool, version 2: the three-character codes, the 20 chapters, the 20 indicators, the target encoding (weight 20, 5 seeded folds, the whole split for validation and test), the refits with session 8's trees and no search, lightgbm_tuned's refit and its 0.001 check, TabICLv2 (2.2.0, jingang/TabICL at 4dcd344e, defaults, random_state 20260927, licenses), the grid (19,448, 8,008 and 3,003 points), T by Newton's method on 1/T, the out-of-fold comparison on the same folds (5 by 20, seed), and the 0.002 rule (at least) (src/eval/trees_session9.py).
- BioGPT-Large-PubMedQA with correction 5: the input, the nine learned tokens, the 384-token cut, the opening words, the one-token answers renormalized, the checkpoint's arguments with the 10-minute fallback, and uncalibrated probabilities as primary (src/eval/biogpt_session9.py).
- The hand-off analyses: the pairs, the coverage grid, AURC per item, E-AURC, AUGRC, the four deferral curves, the registered rates, 1,000 resamples (EI by article, PubMedQA by item), and I-35 (L2, C 1.0, the four features, frozen on validation, compared on test, paired), with the room figures 31, 23 and 23 (src/eval/handoff_curves_session9.py; docs/log.md line 181).
- The cost rule with corrections 1 and 2: the $125 cap, the $115 stop (first $100 and $90), the $140 condition, its place after the long pilot, the order of what may be dropped, the maintenance check, and exit 4 (recipe_session9.py cost; the driver's cost_rule).
- Power and the bar: the four criteria with v0.5's specialist, the prior's shares (3,054, 2,454 and 4,538 of 10,046), the large model of criterion 3, 10,000 article resamples, the decision as H1 to H4, the release and its failure text, and every power figure (s 0.6975, D 1.6465, se 0.0256, pass line -0.0503, 0.0250 at no gain, 0.0718 for 80 percent, the four "worse than s8_ei" rows) (compare_session9.py release_bar; power.json).
- The comparisons: K1 (with session 8's eight EI metrics), K2, K3 (correction 2), K4, K5, K6, K7, K8, K9 (with the _s8rule calibration), K10 (correction 3), the sensitivity, zeros.json, the calibration slopes and intercepts, and the known contamination.
- The freeze and the test stage: what the freeze records, its refusals (osf_filed.txt with exit 5, the four stages finished or NOT BUILT, no v0.5 test file, checks_s8/ not counted), testpreds' refusal and log, its job order, the unpacking check, and apply's SHA-256 check (calibrate_session9.py freeze and apply; the driver's freeze and testpreds; trees_session9.py frozen_inputs).
- The order on EI's test split: prepare, the reading checks on s8_ei behind osf_filed.txt, the validation stages, the freeze, testpreds, then the analyses (the driver's all, which also refuses with exit 5 while osf_filed.txt is missing).
- Session 8's read and its uses: the freeze at 23:00:15 UTC and the test start at 01:13:53 UTC (docs/log.md lines 167 to 170), the hand-off figures of Table 16, the power's source, and the prior's macro-F1 0.1976 (Table 5).
- The license recheck reads only the test records' article IDs (src/tools/ei_license_recheck.py, article_prompts).
- Correction 7 (f)'s statement on PubMedQA, as the plan words it (only the cross-references I-09 and I-14 dropped).
- Style: first person throughout; no em or en dashes; none of the three words that rule 16 of AGENT_RULES.md bans; plain American English; every number traced to the plan, docs/fiorillo_v0_4.md, docs/log.md or power.json (the prior's shares, Gemma 4 31B-it's revision and vLLM 0.30.0 are in the v0.4 report and the log; 113 articles, 10,557 records, 99.03 and 98.68 percent, 12,501 prompts, 949, 38, 12 and 9 prompts, tabicl 2.2.0, 4dcd344e and 648f7d65 are in the log).
